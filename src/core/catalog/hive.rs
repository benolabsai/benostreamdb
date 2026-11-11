// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use ahash::AHashMap;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use faststr::FastStr;
use hive_metastore::{
    EnvironmentContext, FieldSchema, SerDeInfo, StorageDescriptor, Table, ThriftHiveMetastoreClient,
};
use std::net::ToSocketAddrs;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::common::{
    catalog_conflict_error, catalog_not_found_error, ensure_storage_table_initialized,
    extract_metadata_location_from_updates, load_table_metadata_from_location,
    persist_updates_to_storage, table_root_from_metadata_location,
    validate_requirements_against_metadata,
};
use super::{Catalog, TableMetadata};
use arrow::datatypes::SchemaRef;

#[derive(Clone)]
pub struct HiveMetastoreClient {
    client: Arc<ThriftHiveMetastoreClient>,
    #[allow(dead_code)]
    address: String,
}

impl HiveMetastoreClient {
    pub fn new(address: String) -> Result<Self> {
        let addr_str = address
            .trim_start_matches("thrift://")
            .trim_start_matches("http://");

        // Naive address parsing, assuming host:port
        let addr = addr_str
            .to_socket_addrs()
            .map_err(|e| anyhow!("Invalid Hive Metastore address '{}': {}", addr_str, e))?
            .next()
            .ok_or_else(|| anyhow!("Could not resolve Hive Metastore address: {}", addr_str))?;

        let client = hive_metastore::ThriftHiveMetastoreClientBuilder::new("hms")
            .make_codec(volo_thrift::codec::default::DefaultMakeCodec::buffered())
            .address(addr)
            .build();

        Ok(Self {
            client: Arc::new(client),
            address,
        })
    }

    async fn commit_table_internal(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
        requirements: Vec<serde_json::Value>,
    ) -> Result<()> {
        let ns = FastStr::new(namespace);
        let tn = FastStr::new(table_name);

        let existing_table = match self.client.get_table(ns.clone(), tn.clone()).await {
            Ok(volo_thrift::error::MaybeException::Ok(t)) => t,
            Ok(volo_thrift::error::MaybeException::Exception(e)) => {
                return Err(map_hive_exception("get_table", &format!("{e:?}")));
            }
            Err(e) => return Err(anyhow!("Failed to get Hive table for commit: {:?}", e)),
        };

        let existing_params = existing_table.parameters.clone().unwrap_or_default();
        let existing_meta_loc = existing_params
            .get(&FastStr::from_static_str("metadata_location"))
            .map(ToString::to_string)
            .or_else(|| {
                existing_table
                    .sd
                    .as_ref()
                    .and_then(|sd| sd.location.as_ref().map(ToString::to_string))
            })
            .unwrap_or_default();

        let existing_meta = if existing_meta_loc.is_empty() {
            None
        } else {
            Some(load_table_metadata_from_location(&existing_meta_loc).await?)
        };

        validate_requirements_against_metadata(&requirements, existing_meta.as_ref(), "hive")?;

        let table_root = table_root_from_metadata_location(&existing_meta_loc);
        let new_metadata_location = match extract_metadata_location_from_updates(&updates) {
            Some(loc) => Some(loc),
            None if !table_root.is_empty() => {
                Some(persist_updates_to_storage(&table_root, &updates).await?)
            }
            None => None,
        };

        let mut params = existing_params;
        params.insert(
            FastStr::from_static_str("table_type"),
            FastStr::from_static_str("ICEBERG"),
        );
        if let Some(ref new_loc) = new_metadata_location {
            if !existing_meta_loc.is_empty() && existing_meta_loc != *new_loc {
                params.insert(
                    FastStr::from_static_str("previous_metadata_location"),
                    FastStr::new(&existing_meta_loc),
                );
            }
            params.insert(
                FastStr::from_static_str("metadata_location"),
                FastStr::new(new_loc),
            );
        }

        let updated_table = Table {
            parameters: Some(params),
            ..existing_table
        };

        if !existing_meta_loc.is_empty() {
            let env_ctx = build_hive_commit_env_context(&existing_meta_loc);
            match self
                .client
                .alter_table_with_environment_context(ns, tn, updated_table, env_ctx)
                .await
            {
                Ok(volo_thrift::error::MaybeException::Ok(())) => Ok(()),
                Ok(volo_thrift::error::MaybeException::Exception(exc)) => {
                    Err(map_hive_alter_exception(&format!("{exc:?}")))
                }
                Err(e) => Err(anyhow!(
                    "Failed to alter Hive table with CAS context: {:?}",
                    e
                )),
            }
        } else {
            match self.client.alter_table(ns, tn, updated_table).await {
                Ok(volo_thrift::error::MaybeException::Ok(())) => Ok(()),
                Ok(volo_thrift::error::MaybeException::Exception(exc)) => {
                    Err(map_hive_alter_exception(&format!("{exc:?}")))
                }
                Err(e) => Err(anyhow!("Failed to alter Hive table: {:?}", e)),
            }
        }
    }
}

pub(crate) fn build_hive_commit_env_context(expected_meta_loc: &str) -> EnvironmentContext {
    let mut props = AHashMap::new();
    props.insert(
        FastStr::from_static_str("expected_parameter_key"),
        FastStr::from_static_str("metadata_location"),
    );
    props.insert(
        FastStr::from_static_str("expected_parameter_value"),
        FastStr::new(expected_meta_loc),
    );
    EnvironmentContext {
        properties: Some(props),
    }
}

pub(crate) fn map_hive_exception(operation: &str, detail: &str) -> anyhow::Error {
    let lower = detail.to_ascii_lowercase();
    if lower.contains("alreadyexists") || lower.contains("already exists") {
        catalog_conflict_error("hive", format!("Hive {operation} conflict: {detail}"))
    } else if lower.contains("nosuchobject")
        || lower.contains("not found")
        || lower.contains("does not exist")
    {
        catalog_not_found_error("hive", format!("Hive {operation} not found: {detail}"))
    } else {
        anyhow!("Hive Exception in {operation}: {detail}")
    }
}

pub(crate) fn map_hive_alter_exception(detail: &str) -> anyhow::Error {
    let lower = detail.to_ascii_lowercase();
    if lower.contains("expected_parameter")
        || lower.contains("expected parameter")
        || lower.contains("mismatch")
        || lower.contains("concurrent")
        || lower.contains("invalidoperation")
        || lower.contains("metaexception")
    {
        catalog_conflict_error("hive", format!("Hive alter_table CAS conflict: {detail}"))
    } else {
        map_hive_exception("alter_table", detail)
    }
}

#[async_trait]
impl Catalog for HiveMetastoreClient {
    async fn create_namespace(&self, namespace: &str) -> Result<()> {
        let database = hive_metastore::Database {
            name: Some(FastStr::new(namespace)),
            ..Default::default()
        };
        match self.client.create_database(database).await {
            Ok(volo_thrift::error::MaybeException::Ok(())) => Ok(()),
            Ok(volo_thrift::error::MaybeException::Exception(e)) => {
                let msg = format!("{e:?}");
                if msg.to_ascii_lowercase().contains("alreadyexists") {
                    Ok(())
                } else {
                    Err(map_hive_exception("create_database", &msg))
                }
            }
            Err(e) => Err(anyhow!("Failed to create Hive database: {:?}", e)),
        }
    }

    async fn namespace_exists(&self, namespace: &str) -> Result<bool> {
        match self.client.get_database(FastStr::new(namespace)).await {
            Ok(volo_thrift::error::MaybeException::Ok(_)) => Ok(true),
            Ok(volo_thrift::error::MaybeException::Exception(e)) => {
                let msg = format!("{e:?}");
                let lower = msg.to_ascii_lowercase();
                if lower.contains("nosuchobject") || lower.contains("not found") {
                    Ok(false)
                } else {
                    Err(anyhow!("Hive Exception in get_database: {msg}"))
                }
            }
            Err(e) => Err(anyhow!("Failed to check Hive database: {:?}", e)),
        }
    }

    async fn drop_namespace(&self, namespace: &str) -> Result<()> {
        let tables = self.list_tables(namespace).await?;
        if !tables.is_empty() {
            return Err(catalog_conflict_error(
                "hive",
                format!("Hive database '{namespace}' is not empty"),
            ));
        }
        match self
            .client
            .drop_database(FastStr::new(namespace), false, false)
            .await
        {
            Ok(volo_thrift::error::MaybeException::Ok(())) => Ok(()),
            Ok(volo_thrift::error::MaybeException::Exception(e)) => {
                let msg = format!("{e:?}");
                let lower = msg.to_ascii_lowercase();
                if lower.contains("invalidoperation") || lower.contains("not empty") {
                    Err(catalog_conflict_error("hive", msg))
                } else {
                    Err(map_hive_exception("drop_database", &msg))
                }
            }
            Err(e) => Err(anyhow!("Failed to drop Hive database: {:?}", e)),
        }
    }

    async fn create_table(
        &self,
        namespace: &str,
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        let clean_root = location.map(|loc| loc.trim_end_matches('/').to_string());
        let initial_metadata_location = if let Some(ref root) = clean_root {
            Some(ensure_storage_table_initialized(root, &schema).await?)
        } else {
            None
        };

        fn arrow_to_hive_type(dt: &arrow::datatypes::DataType) -> String {
            match dt {
                arrow::datatypes::DataType::Boolean => "boolean".to_string(),
                arrow::datatypes::DataType::Int8 | arrow::datatypes::DataType::UInt8 => {
                    "tinyint".to_string()
                }
                arrow::datatypes::DataType::Int16 | arrow::datatypes::DataType::UInt16 => {
                    "smallint".to_string()
                }
                arrow::datatypes::DataType::Int32 | arrow::datatypes::DataType::UInt32 => {
                    "int".to_string()
                }
                arrow::datatypes::DataType::Int64 | arrow::datatypes::DataType::UInt64 => {
                    "bigint".to_string()
                }
                arrow::datatypes::DataType::Float32 => "float".to_string(),
                arrow::datatypes::DataType::Float64 => "double".to_string(),
                arrow::datatypes::DataType::Utf8 | arrow::datatypes::DataType::LargeUtf8 => {
                    "string".to_string()
                }
                arrow::datatypes::DataType::Binary
                | arrow::datatypes::DataType::LargeBinary
                | arrow::datatypes::DataType::FixedSizeBinary(_) => "binary".to_string(),
                arrow::datatypes::DataType::Date32 | arrow::datatypes::DataType::Date64 => {
                    "date".to_string()
                }
                arrow::datatypes::DataType::Timestamp(_, _) => "timestamp".to_string(),
                _ => "string".to_string(),
            }
        }

        let columns: Vec<FieldSchema> = schema
            .fields()
            .iter()
            .map(|f| FieldSchema {
                name: Some(FastStr::new(f.name())),
                r#type: Some(FastStr::new(arrow_to_hive_type(f.data_type()))),
                comment: None,
            })
            .collect();

        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i32;

        let sd = StorageDescriptor {
            cols: Some(columns),
            location: clean_root.as_deref().map(FastStr::new),
            input_format: Some(FastStr::from("org.apache.hadoop.mapred.TextInputFormat")),
            output_format: Some(FastStr::from(
                "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
            )),
            serde_info: Some(SerDeInfo {
                name: None,
                serialization_lib: Some(FastStr::from(
                    "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
                )),
                parameters: None,
            }),
            ..Default::default()
        };

        let loc_faststr = initial_metadata_location
            .as_deref()
            .map(FastStr::new)
            .unwrap_or_else(|| FastStr::from_static_str(""));

        let mut params = AHashMap::new();
        params.insert(
            FastStr::from_static_str("table_type"),
            FastStr::from_static_str("ICEBERG"),
        );
        params.insert(FastStr::from_static_str("metadata_location"), loc_faststr);

        let table = Table {
            table_name: Some(FastStr::new(table_name)),
            db_name: Some(FastStr::new(namespace)),
            owner: Some(FastStr::from_static_str("benostream")),
            create_time: Some(now),
            last_access_time: Some(now),
            retention: Some(0),
            sd: Some(sd),
            partition_keys: Some(vec![]),
            parameters: Some(params),
            view_original_text: None,
            view_expanded_text: None,
            table_type: Some(FastStr::from_static_str("EXTERNAL_TABLE")),
            ..Default::default()
        };

        match self.client.create_table(table).await {
            Ok(volo_thrift::error::MaybeException::Ok(())) => Ok(()),
            Ok(volo_thrift::error::MaybeException::Exception(e)) => {
                Err(map_hive_exception("create_table", &format!("{e:?}")))
            }
            Err(e) => Err(anyhow!("Failed to create Hive table: {:?}", e)),
        }
    }

    async fn drop_table(&self, namespace: &str, table_name: &str) -> Result<()> {
        match self
            .client
            .drop_table(FastStr::new(namespace), FastStr::new(table_name), false)
            .await
        {
            Ok(volo_thrift::error::MaybeException::Ok(())) => Ok(()),
            Ok(volo_thrift::error::MaybeException::Exception(e)) => {
                let msg = format!("{e:?}");
                if msg.to_ascii_lowercase().contains("nosuchobject") {
                    Ok(())
                } else {
                    Err(map_hive_exception("drop_table", &msg))
                }
            }
            Err(e) => Err(anyhow!("Failed to drop Hive table: {:?}", e)),
        }
    }

    async fn list_tables(&self, namespace: &str) -> Result<Vec<String>> {
        match self.client.get_all_tables(FastStr::new(namespace)).await {
            Ok(volo_thrift::error::MaybeException::Ok(tables)) => {
                let mut names: Vec<String> = tables.into_iter().map(|s| s.to_string()).collect();
                names.sort();
                Ok(names)
            }
            Ok(volo_thrift::error::MaybeException::Exception(e)) => {
                Err(map_hive_exception("get_all_tables", &format!("{e:?}")))
            }
            Err(e) => Err(anyhow!("Failed to list Hive tables: {:?}", e)),
        }
    }

    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        let ns = FastStr::new(namespace);
        let tn = FastStr::new(table_name);

        let table = match self.client.get_table(ns, tn).await {
            Ok(volo_thrift::error::MaybeException::Ok(t)) => t,
            Ok(volo_thrift::error::MaybeException::Exception(e)) => {
                return Err(map_hive_exception("get_table", &format!("{e:?}")));
            }
            Err(e) => return Err(anyhow!("Failed to get Hive table: {:?}", e)),
        };

        let location = table
            .parameters
            .as_ref()
            .and_then(|p| p.get(&FastStr::from_static_str("metadata_location")))
            .map(ToString::to_string)
            .or_else(|| {
                table
                    .sd
                    .as_ref()
                    .and_then(|sd| sd.location.as_ref().map(ToString::to_string))
            })
            .ok_or_else(|| anyhow!("Table missing metadata_location"))?;

        load_table_metadata_from_location(&location).await
    }

    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        Err(anyhow!("Hive Metastore does not support branching."))
    }

    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool> {
        let ns = FastStr::new(namespace);
        let tn = FastStr::new(table_name);
        match self.client.get_table(ns, tn).await {
            Ok(volo_thrift::error::MaybeException::Ok(_)) => Ok(true),
            _ => Ok(false),
        }
    }

    async fn commit_table(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
    ) -> Result<()> {
        self.commit_table_internal(namespace, table_name, updates, Vec::new())
            .await
    }

    async fn commit_table_with_requirements(
        &self,
        namespace: &str,
        table_name: &str,
        updates: Vec<serde_json::Value>,
        requirements: Vec<serde_json::Value>,
    ) -> Result<()> {
        self.commit_table_internal(namespace, table_name, updates, requirements)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::BenoStreamError;

    #[test]
    fn test_hive_env_context_cas_properties() -> Result<()> {
        let ctx = build_hive_commit_env_context("s3://wh/db/tbl/metadata/v1.metadata.json");
        let props = ctx
            .properties
            .ok_or_else(|| anyhow!("expected EnvironmentContext properties"))?;
        assert_eq!(
            props
                .get(&FastStr::from_static_str("expected_parameter_key"))
                .map(FastStr::as_str),
            Some("metadata_location")
        );
        assert_eq!(
            props
                .get(&FastStr::from_static_str("expected_parameter_value"))
                .map(FastStr::as_str),
            Some("s3://wh/db/tbl/metadata/v1.metadata.json")
        );
        Ok(())
    }

    #[test]
    fn test_map_hive_alter_exception_maps_cas_mismatch_to_409() {
        let err = map_hive_alter_exception(
            "InvalidOperationException(message: \"The table has been modified concurrently: expected_parameter_value mismatch\")",
        );
        let beno = err.downcast_ref::<BenoStreamError>();
        assert!(matches!(
            beno,
            Some(BenoStreamError::CatalogRequest {
                status: Some(409),
                ..
            })
        ));
    }
}
