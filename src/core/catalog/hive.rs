// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::{Result, anyhow};
use std::net::ToSocketAddrs;
use async_trait::async_trait;
use std::time::{SystemTime, UNIX_EPOCH};
use hive_metastore::{ThriftHiveMetastoreClient, FieldSchema, SerDeInfo, StorageDescriptor, Table};
use std::sync::Arc;
use faststr::FastStr; 
use ahash::AHashMap;

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
        // In production, might want better DNS resolution handling
        let addr = addr_str
            .to_socket_addrs()
            .map_err(|e| anyhow!("Invalid Hive Metastore address '{}': {}", addr_str, e))?
            .next()
            .ok_or_else(|| anyhow!("Could not resolve Hive Metastore address: {}", addr_str))?;

        let client = hive_metastore::ThriftHiveMetastoreClientBuilder::new("hms")
            .address(addr)
            .build();

        Ok(Self {
            client: Arc::new(client),
            address,
        })
    }
}

#[async_trait]
impl Catalog for HiveMetastoreClient {
    async fn create_table(
        &self,
        namespace: &str, 
        table_name: &str,
        schema: SchemaRef,
        location: Option<&str>,
    ) -> Result<()> {
        // 1. Initialize table manifest if location is provided
        if let Some(loc) = location {
             crate::Table::create_async(loc.to_string(), schema.clone()).await?;
        }

        // 2. Extract columns from Arrow schema for Hive
        let columns: Vec<FieldSchema> = schema.fields().iter().map(|f| {
             FieldSchema {
                 name: Some(FastStr::new(f.name())),
                 r#type: Some(FastStr::new(f.data_type().to_string())),
                 comment: None,
             }
        }).collect();

        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i32;

        let sd = StorageDescriptor {
             cols: Some(columns),
             location: location.map(|s| FastStr::new(s)),
             input_format: Some(FastStr::from("org.apache.hadoop.mapred.TextInputFormat")), 
             output_format: Some(FastStr::from("org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat")),
             serde_info: Some(SerDeInfo {
                 name: None,
                 serialization_lib: Some(FastStr::from("org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe")),
                 parameters: None,
             }),
             ..Default::default()
        };
        
        // Explicitly format parameters to avoid lifetime issues
        // We clone the location string immediately into FastStr to satisfy static requirements if any
        let loc_faststr = if let Some(loc) = location {
            FastStr::new(loc)
        } else {
            FastStr::from_static_str("")
        };
        
        let mut params = AHashMap::new();
        params.insert(FastStr::from_static_str("table_type"), FastStr::from_static_str("ICEBERG"));
        params.insert(FastStr::from_static_str("metadata_location"), loc_faststr);

        let table = Table {
            table_name: Some(FastStr::new(table_name)),
            db_name: Some(FastStr::new(namespace)),
            owner: Some(FastStr::from_static_str("hyperstream")),
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
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow!("Failed to create Hive table: {:?}", e)),
        }
    }
    
    async fn load_table(&self, namespace: &str, table_name: &str) -> Result<TableMetadata> {
        let ns = FastStr::new(namespace);
        let tn = FastStr::new(table_name);
        
        // Handle MaybeException
        let table = match self.client.get_table(ns, tn).await {
            Ok(maybe_exc) => match maybe_exc {
                volo_thrift::error::MaybeException::Ok(t) => t,
                volo_thrift::error::MaybeException::Exception(e) => return Err(anyhow!("Hive Exception: {:?}", e)),
            },
             Err(e) => return Err(anyhow!("Failed to get Hive table: {:?}", e)),
        };

        // Access parameters 
        let location = table.parameters.as_ref()
            .and_then(|p| p.get(&FastStr::from_static_str("metadata_location")))
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("Table missing metadata_location"))?;

        Ok(TableMetadata::minimal(location))
    }
    
    async fn create_branch(&self, _branch_name: &str, _source_ref: Option<&str>) -> Result<()> {
        Err(anyhow!("Hive Metastore does not support branching."))
    }
    
    async fn table_exists(&self, namespace: &str, table_name: &str) -> Result<bool> {
        let ns = FastStr::new(namespace);
        let tn = FastStr::new(table_name);
        match self.client.get_table(ns, tn).await {
            Ok(maybe_exc) => match maybe_exc {
                volo_thrift::error::MaybeException::Ok(_) => Ok(true),
                _ => Ok(false),
            },
            Err(_) => Ok(false),
        }
    }

    async fn commit_table(&self, namespace: &str, table_name: &str, updates: Vec<serde_json::Value>) -> Result<()> {
        // For Hive, we need to alter the table to update the metadata_location parameter
        let ns = FastStr::new(namespace);
        let tn = FastStr::new(table_name);
        
        // First, get the current table
        let existing_table = match self.client.get_table(ns.clone(), tn.clone()).await {
            Ok(volo_thrift::error::MaybeException::Ok(t)) => t,
            Ok(volo_thrift::error::MaybeException::Exception(e)) => return Err(anyhow!("Hive Exception: {:?}", e)),
            Err(e) => return Err(anyhow!("Failed to get Hive table for commit: {:?}", e)),
        };
        
        // Extract new metadata location from updates
        let mut new_metadata_location: Option<String> = None;
        for update in &updates {
            if let Some(action) = update.get("action").and_then(|v| v.as_str()) {
                if action == "set-current-snapshot" || action == "add-snapshot" {
                    if let Some(snapshot) = update.get("snapshot") {
                        if let Some(manifest_list) = snapshot.get("manifest-list").and_then(|v| v.as_str()) {
                            if let Some(table_root) = manifest_list.rsplit_once("/_manifest/") {
                                new_metadata_location = Some(format!("{}/metadata/v{}.metadata.json", 
                                    table_root.0,
                                    snapshot.get("snapshot-id").and_then(|v| v.as_i64()).unwrap_or(1)
                                ));
                            }
                        }
                    }
                }
            }
        }
        
        // Update parameters map
        let mut params = existing_table.parameters.unwrap_or_default();
        if let Some(new_loc) = new_metadata_location {
            params.insert(FastStr::from_static_str("metadata_location"), FastStr::new(&new_loc));
        }
        
        // Clone existing table with updated parameters
        let updated_table = Table {
            parameters: Some(params),
            ..existing_table
        };
        
        // Call alter_table to update
        match self.client.alter_table(ns, tn, updated_table).await {
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow!("Failed to alter Hive table: {:?}", e)),
        }
    }
}
