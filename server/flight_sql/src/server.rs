use arrow_flight::sql::metadata::SqlInfoDataBuilder;
use arrow_flight::sql::server::FlightSqlService;
use arrow_flight::sql::ProstMessageExt;
use arrow_flight::sql::{
    ActionBeginTransactionRequest, ActionBeginTransactionResult,
    ActionClosePreparedStatementRequest, ActionCreatePreparedStatementRequest,
    ActionCreatePreparedStatementResult, ActionEndTransactionRequest, Any, Command,
    CommandGetCatalogs, CommandGetCrossReference, CommandGetDbSchemas, CommandGetExportedKeys,
    CommandGetImportedKeys, CommandGetPrimaryKeys, CommandGetSqlInfo, CommandGetTableTypes,
    CommandGetTables, CommandGetXdbcTypeInfo, CommandPreparedStatementQuery,
    CommandPreparedStatementUpdate, CommandStatementIngest, CommandStatementQuery,
    CommandStatementUpdate, SqlInfo, TicketStatementQuery,
};
use arrow_flight::{
    Action, FlightData, FlightDescriptor, FlightEndpoint, FlightInfo, HandshakeRequest,
    HandshakeResponse, IpcMessage, SchemaAsIpc, Ticket,
};
use futures::stream::BoxStream;
use futures::TryStreamExt;
use prost::Message;
use tonic::{Request, Response, Status, Streaming};

#[derive(Clone)]
pub struct BenoStreamFlightSqlService {
    pub session: benostreamdb::core::sql::session::BenoStreamSession,
    tx: std::sync::Arc<tokio::sync::Mutex<TransactionState>>,
    prepared: std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<Vec<u8>, String>>>,
}

/// Generate a unique prepared-statement handle.
fn new_statement_handle() -> Vec<u8> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    format!("stmt-{}", COUNTER.fetch_add(1, Ordering::Relaxed)).into_bytes()
}

/// Transaction state for the Flight SQL session.
///
/// BenoStreamDB commits are Iceberg snapshots: a transaction stages statements
/// and publishes them as a single snapshot on COMMIT (or discards them on
/// ROLLBACK). Autocommit (the default) executes each statement immediately.
struct TransactionState {
    autocommit: bool,
    staged: Vec<String>,
}

impl Default for TransactionState {
    fn default() -> Self {
        Self {
            autocommit: true,
            staged: Vec::new(),
        }
    }
}

/// Arrow schema for the Flight SQL `GetPrimaryKeys` result.
fn primary_keys_schema() -> std::sync::Arc<arrow::datatypes::Schema> {
    use arrow::datatypes::{DataType, Field, Schema};
    std::sync::Arc::new(Schema::new(vec![
        Field::new("catalog_name", DataType::Utf8, true),
        Field::new("db_schema_name", DataType::Utf8, true),
        Field::new("table_name", DataType::Utf8, false),
        Field::new("column_name", DataType::Utf8, false),
        Field::new("key_name", DataType::Utf8, true),
        Field::new("key_sequence", DataType::Int32, false),
    ]))
}

/// Parse the `SetAutocommit` action body. The ADBC driver sends a protobuf
/// message with a single `bool autocommit` field (tag 1, varint); fall back to
/// treating a non-empty body's first byte as the flag.
fn parse_autocommit_body(body: &[u8]) -> bool {
    if body.is_empty() {
        return true;
    }
    if body.len() >= 2 && body[0] == 0x08 {
        return body[1] != 0;
    }
    body[0] != 0
}

impl BenoStreamFlightSqlService {
    pub fn new(session: benostreamdb::core::sql::session::BenoStreamSession) -> Self {
        Self {
            session,
            tx: std::sync::Arc::new(tokio::sync::Mutex::new(TransactionState::default())),
            prepared: std::sync::Arc::new(
                tokio::sync::Mutex::new(std::collections::HashMap::new()),
            ),
        }
    }

    /// Look up the SQL for a prepared-statement handle.
    async fn prepared_sql(&self, handle: &[u8]) -> Result<String, Status> {
        self.prepared
            .lock()
            .await
            .get(handle)
            .cloned()
            .ok_or_else(|| Status::invalid_argument("unknown prepared statement handle"))
    }

    /// Execute a statement immediately and return its result batches.
    async fn execute_statement(
        &self,
        sql: &str,
    ) -> Result<Vec<arrow::record_batch::RecordBatch>, Status> {
        let df = self
            .session
            .sql_to_df(sql)
            .await
            .map_err(|e| Status::internal(format!("Error executing statement: {}", e)))?;
        df.collect()
            .await
            .map_err(|e| Status::internal(format!("Error collecting execution result: {}", e)))
    }
}

/// Custom Flight ticket type for a live subscription. A client sends a
/// `Command::Unknown(Any)` with this `type_url` and a JSON body; the server
/// streams the drained events back. This is the true-streaming counterpart of
/// the `subscribe_events` SQL table function.
const SUBSCRIBE_TYPE_URL: &str = "type.googleapis.com/benostreamdb.Subscribe";

/// JSON body of a subscription ticket.
#[derive(serde::Deserialize)]
struct SubscribeRequest {
    table: String,
    #[serde(default)]
    filter: String,
    #[serde(default = "default_max_events")]
    max_events: u64,
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
}

fn default_max_events() -> u64 {
    100
}
fn default_timeout_ms() -> u64 {
    1000
}

/// Arrow schema of the `subscribe_events` result (one row per event).
fn subscribe_events_schema() -> std::sync::Arc<arrow::datatypes::Schema> {
    use arrow::datatypes::{DataType, Field, Schema};
    std::sync::Arc::new(Schema::new(vec![
        Field::new("event_type", DataType::Utf8, false),
        Field::new("rows", DataType::Int64, false),
    ]))
}

#[tonic::async_trait]
impl FlightSqlService for BenoStreamFlightSqlService {
    type FlightService = BenoStreamFlightSqlService;

    async fn do_handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<BoxStream<'static, Result<HandshakeResponse, Status>>>, Status> {
        // BenoStreamDB does not require authentication; return a connection id
        // so clients can correlate requests.
        let response = HandshakeResponse {
            protocol_version: 0,
            payload: new_statement_handle().into(),
        };
        let stream = futures::stream::iter(vec![Ok(response)]);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_get_fallback(
        &self,
        _request: Request<Ticket>,
        message: Any,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        if message.type_url != SUBSCRIBE_TYPE_URL {
            return Err(Status::unimplemented(format!(
                "do_get_fallback not implemented for: {:?}",
                message.type_url
            )));
        }
        let req: SubscribeRequest = serde_json::from_slice(message.value.as_ref())
            .map_err(|e| Status::invalid_argument(format!("invalid subscribe ticket: {e}")))?;
        let esc = |s: &str| s.replace('\'', "''");
        let sql = format!(
            "SELECT * FROM subscribe_events('{}', '{}', {}, {})",
            esc(&req.table),
            esc(&req.filter),
            req.max_events,
            req.timeout_ms
        );
        let df = self
            .session
            .sql_to_df(&sql)
            .await
            .map_err(|e| Status::internal(format!("Error planning subscription: {e}")))?;
        let schema = df.schema().inner().clone();
        let batches = df
            .collect()
            .await
            .map_err(|e| Status::internal(format!("Error draining subscription: {e}")))?;
        let flight_data = arrow_flight::utils::batches_to_flight_data(&schema, batches)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(Box::pin(futures::stream::iter(
            flight_data.into_iter().map(Ok),
        ))))
    }

    async fn get_flight_info_fallback(
        &self,
        cmd: Command,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        if let Command::Unknown(any) = &cmd {
            if any.type_url == SUBSCRIBE_TYPE_URL {
                let options = datafusion::arrow::ipc::writer::IpcWriteOptions::default();
                let schema = subscribe_events_schema();
                let schema_as_ipc = SchemaAsIpc::new(schema.as_ref(), &options);
                let schema_bytes = IpcMessage::try_from(schema_as_ipc)
                    .map_err(|e| Status::internal(e.to_string()))?
                    .0;
                let ticket = Ticket::new(any.encode_to_vec());
                let endpoint = FlightEndpoint {
                    ticket: Some(ticket),
                    location: vec![],
                    ..Default::default()
                };
                return Ok(Response::new(FlightInfo {
                    schema: schema_bytes,
                    endpoint: vec![endpoint],
                    flight_descriptor: Some(request.into_inner()),
                    total_bytes: -1,
                    total_records: -1,
                    ordered: false,
                    app_metadata: vec![].into(),
                }));
            }
        }
        Err(Status::unimplemented(format!(
            "get_flight_info: invalid request: {}",
            cmd.type_url()
        )))
    }

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let sql = query.query.clone();

        // When autocommit is off, stage the statement and return an empty
        // result set; it runs as part of the transaction on COMMIT.
        {
            let mut tx = self.tx.lock().await;
            if !tx.autocommit {
                tx.staged.push(sql.clone());
                drop(tx);
                let options = datafusion::arrow::ipc::writer::IpcWriteOptions::default();
                let empty = datafusion::arrow::datatypes::Schema::empty();
                let schema_as_ipc = SchemaAsIpc::new(&empty, &options);
                let schema_bytes = IpcMessage::try_from(schema_as_ipc)
                    .map_err(|e| Status::internal(e.to_string()))?
                    .0;
                let flight_info = FlightInfo {
                    schema: schema_bytes,
                    endpoint: vec![],
                    flight_descriptor: Some(request.into_inner()),
                    total_bytes: -1,
                    total_records: -1,
                    ordered: false,
                    app_metadata: vec![].into(),
                };
                return Ok(Response::new(flight_info));
            }
        }

        // DDL statements have no result set. Execute them here and return a
        // FlightInfo with no endpoints: ADBC's ExecuteQuery cancels the
        // follow-up DoGet when the schema is empty, so the statement would
        // otherwise never run.
        if self.session.is_ddl(&sql).await.unwrap_or(false)
            || self.session.is_dml(&sql).await.unwrap_or(false)
        {
            self.execute_statement(&sql).await?;
            let options = datafusion::arrow::ipc::writer::IpcWriteOptions::default();
            let empty = datafusion::arrow::datatypes::Schema::empty();
            let schema_as_ipc = SchemaAsIpc::new(&empty, &options);
            let schema_bytes = IpcMessage::try_from(schema_as_ipc)
                .map_err(|e| Status::internal(e.to_string()))?
                .0;
            let flight_info = FlightInfo {
                schema: schema_bytes,
                endpoint: vec![],
                flight_descriptor: Some(request.into_inner()),
                total_bytes: -1,
                total_records: -1,
                ordered: false,
                app_metadata: vec![].into(),
            };
            return Ok(Response::new(flight_info));
        }

        let schema = self
            .session
            .get_schema(&sql)
            .await
            .map_err(|e| Status::internal(format!("Error planning query: {}", e)))?;

        let options = datafusion::arrow::ipc::writer::IpcWriteOptions::default();
        let schema_as_ipc = SchemaAsIpc::new(schema.as_ref(), &options);
        let schema_bytes = IpcMessage::try_from(schema_as_ipc)
            .map_err(|e| Status::internal(e.to_string()))?
            .0;

        let ticket_query = TicketStatementQuery {
            statement_handle: sql.into_bytes().into(),
        };
        let ticket = Ticket::new(ticket_query.as_any().encode_to_vec());

        let endpoint = FlightEndpoint {
            ticket: Some(ticket),
            location: vec![],
            ..Default::default()
        };

        let flight_info = FlightInfo {
            schema: schema_bytes,
            endpoint: vec![endpoint],
            flight_descriptor: Some(request.into_inner()),
            total_bytes: -1,
            total_records: -1,
            ordered: false,
            app_metadata: vec![].into(),
        };

        Ok(Response::new(flight_info))
    }

    async fn get_flight_info_prepared_statement(
        &self,
        cmd: CommandPreparedStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let sql = self.prepared_sql(&cmd.prepared_statement_handle).await?;

        // DDL statements have no result set. Execute them here and return a
        // FlightInfo with no endpoints: ADBC's ExecuteQuery cancels the
        // follow-up DoGet when the schema is empty, so the statement would
        // otherwise never run.
        if self.session.is_ddl(&sql).await.unwrap_or(false)
            || self.session.is_dml(&sql).await.unwrap_or(false)
        {
            self.execute_statement(&sql).await?;
            let options = datafusion::arrow::ipc::writer::IpcWriteOptions::default();
            let empty = datafusion::arrow::datatypes::Schema::empty();
            let schema_as_ipc = SchemaAsIpc::new(&empty, &options);
            let schema_bytes = IpcMessage::try_from(schema_as_ipc)
                .map_err(|e| Status::internal(e.to_string()))?
                .0;
            let flight_info = FlightInfo {
                schema: schema_bytes,
                endpoint: vec![],
                flight_descriptor: Some(request.into_inner()),
                total_bytes: -1,
                total_records: -1,
                ordered: false,
                app_metadata: vec![].into(),
            };
            return Ok(Response::new(flight_info));
        }

        let schema = self
            .session
            .get_schema(&sql)
            .await
            .map_err(|e| Status::internal(format!("Error planning query: {}", e)))?;

        let options = datafusion::arrow::ipc::writer::IpcWriteOptions::default();
        let schema_as_ipc = SchemaAsIpc::new(schema.as_ref(), &options);
        let schema_bytes = IpcMessage::try_from(schema_as_ipc)
            .map_err(|e| Status::internal(e.to_string()))?
            .0;

        let ticket_query = TicketStatementQuery {
            statement_handle: sql.into_bytes().into(),
        };
        let ticket = Ticket::new(ticket_query.as_any().encode_to_vec());
        let endpoint = FlightEndpoint {
            ticket: Some(ticket),
            location: vec![],
            ..Default::default()
        };
        let flight_info = FlightInfo {
            schema: schema_bytes,
            endpoint: vec![endpoint],
            flight_descriptor: Some(request.into_inner()),
            total_bytes: -1,
            total_records: -1,
            ordered: false,
            app_metadata: vec![].into(),
        };
        Ok(Response::new(flight_info))
    }

    async fn get_flight_info_catalogs(
        &self,
        query: CommandGetCatalogs,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let flight_descriptor = request.into_inner();
        let ticket = Ticket::new(query.as_any().encode_to_vec());
        let endpoint = FlightEndpoint::new().with_ticket(ticket);
        let flight_info = FlightInfo::new()
            .try_with_schema(&query.into_builder().schema())
            .map_err(|e| Status::internal(e.to_string()))?
            .with_endpoint(endpoint)
            .with_descriptor(flight_descriptor);
        Ok(Response::new(flight_info))
    }

    async fn get_flight_info_schemas(
        &self,
        query: CommandGetDbSchemas,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let flight_descriptor = request.into_inner();
        let ticket = Ticket::new(query.as_any().encode_to_vec());
        let endpoint = FlightEndpoint::new().with_ticket(ticket);
        let flight_info = FlightInfo::new()
            .try_with_schema(&query.into_builder().schema())
            .map_err(|e| Status::internal(e.to_string()))?
            .with_endpoint(endpoint)
            .with_descriptor(flight_descriptor);
        Ok(Response::new(flight_info))
    }

    async fn get_flight_info_tables(
        &self,
        query: CommandGetTables,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let flight_descriptor = request.into_inner();
        let ticket = Ticket::new(query.as_any().encode_to_vec());
        let endpoint = FlightEndpoint::new().with_ticket(ticket);
        let flight_info = FlightInfo::new()
            .try_with_schema(&query.into_builder().schema())
            .map_err(|e| Status::internal(e.to_string()))?
            .with_endpoint(endpoint)
            .with_descriptor(flight_descriptor);
        Ok(Response::new(flight_info))
    }

    async fn get_flight_info_table_types(
        &self,
        query: CommandGetTableTypes,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let flight_descriptor = request.into_inner();
        let ticket = Ticket::new(query.as_any().encode_to_vec());
        let endpoint = FlightEndpoint::new().with_ticket(ticket);
        let flight_info = FlightInfo::new()
            .try_with_schema(&query.into_builder().schema())
            .map_err(|e| Status::internal(e.to_string()))?
            .with_endpoint(endpoint)
            .with_descriptor(flight_descriptor);
        Ok(Response::new(flight_info))
    }

    async fn get_flight_info_sql_info(
        &self,
        query: CommandGetSqlInfo,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let flight_descriptor = request.into_inner();
        let ticket = Ticket::new(query.as_any().encode_to_vec());
        let endpoint = FlightEndpoint::new().with_ticket(ticket);

        let mut builder = SqlInfoDataBuilder::new();
        builder.append(
            SqlInfo::FlightSqlServerName,
            "BenoStreamDB Flight SQL Server",
        );
        builder.append(SqlInfo::FlightSqlServerVersion, "1");
        builder.append(SqlInfo::FlightSqlServerArrowVersion, "1.3");
        let sql_info_data = builder
            .build()
            .map_err(|e| Status::internal(e.to_string()))?;

        let flight_info = FlightInfo::new()
            .try_with_schema(query.into_builder(&sql_info_data).schema().as_ref())
            .map_err(|e| Status::internal(e.to_string()))?
            .with_endpoint(endpoint)
            .with_descriptor(flight_descriptor);
        Ok(Response::new(flight_info))
    }

    async fn get_flight_info_primary_keys(
        &self,
        query: CommandGetPrimaryKeys,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let flight_descriptor = request.into_inner();
        let ticket = Ticket::new(query.as_any().encode_to_vec());
        let endpoint = FlightEndpoint::new().with_ticket(ticket);
        let flight_info = FlightInfo::new()
            .try_with_schema(&primary_keys_schema())
            .map_err(|e| Status::internal(e.to_string()))?
            .with_endpoint(endpoint)
            .with_descriptor(flight_descriptor);
        Ok(Response::new(flight_info))
    }

    async fn get_flight_info_exported_keys(
        &self,
        _query: CommandGetExportedKeys,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn get_flight_info_imported_keys(
        &self,
        _query: CommandGetImportedKeys,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn get_flight_info_cross_reference(
        &self,
        _query: CommandGetCrossReference,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        let sql = String::from_utf8(ticket.statement_handle.to_vec())
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        let df = self
            .session
            .sql_to_df(&sql)
            .await
            .map_err(|e| Status::internal(format!("Error planning query: {}", e)))?;

        let schema = df.schema().inner().clone();

        let batches = df
            .collect()
            .await
            .map_err(|e| Status::internal(format!("Error collecting batches: {}", e)))?;

        let flight_data_stream = arrow_flight::utils::batches_to_flight_data(&schema, batches)
            .map_err(|e| Status::internal(e.to_string()))?;

        let output_stream = futures::stream::iter(flight_data_stream.into_iter().map(Ok));
        Ok(Response::new(Box::pin(output_stream)))
    }

    async fn do_get_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        let sql = self.prepared_sql(&query.prepared_statement_handle).await?;
        let df = self
            .session
            .sql_to_df(&sql)
            .await
            .map_err(|e| Status::internal(format!("Error planning query: {}", e)))?;
        let schema = df.schema().inner().clone();
        let batches = df
            .collect()
            .await
            .map_err(|e| Status::internal(format!("Error collecting batches: {}", e)))?;
        let flight_data_stream = arrow_flight::utils::batches_to_flight_data(&schema, batches)
            .map_err(|e| Status::internal(e.to_string()))?;
        let output_stream = futures::stream::iter(flight_data_stream.into_iter().map(Ok));
        Ok(Response::new(Box::pin(output_stream)))
    }

    async fn do_get_catalogs(
        &self,
        query: CommandGetCatalogs,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        let mut builder = query.into_builder();
        for catalog_name in self.session.get_ctx().catalog_names() {
            builder.append(catalog_name);
        }
        let schema = builder.schema();
        let batch = builder.build();
        let stream = arrow_flight::encode::FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(futures::stream::once(async { batch }))
            .map_err(|e| Status::internal(e.to_string()));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_get_schemas(
        &self,
        query: CommandGetDbSchemas,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        let mut builder = query.into_builder();
        for catalog_name in self.session.get_ctx().catalog_names() {
            if let Some(catalog) = self.session.get_ctx().catalog(&catalog_name) {
                for schema_name in catalog.schema_names() {
                    builder.append(catalog_name.clone(), schema_name);
                }
            }
        }
        let schema = builder.schema();
        let batch = builder.build();
        let stream = arrow_flight::encode::FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(futures::stream::once(async { batch }))
            .map_err(|e| Status::internal(e.to_string()));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_get_tables(
        &self,
        query: CommandGetTables,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        let mut builder = query.into_builder();
        let dummy_schema = datafusion::arrow::datatypes::Schema::empty();

        for catalog_name in self.session.get_ctx().catalog_names() {
            if let Some(catalog) = self.session.get_ctx().catalog(&catalog_name) {
                for schema_name in catalog.schema_names() {
                    if let Some(schema) = catalog.schema(&schema_name) {
                        for table_name in schema.table_names() {
                            let arrow_schema = match schema.table(&table_name).await {
                                Ok(Some(table)) => table.schema(),
                                _ => std::sync::Arc::new(dummy_schema.clone()),
                            };
                            builder
                                .append(
                                    catalog_name.clone(),
                                    schema_name.clone(),
                                    table_name,
                                    "TABLE",
                                    arrow_schema.as_ref(),
                                )
                                .map_err(|e| Status::internal(e.to_string()))?;
                        }
                    }
                }
            }
        }

        let schema = builder.schema();
        let batch = builder.build();
        let stream = arrow_flight::encode::FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(futures::stream::once(async { batch }))
            .map_err(|e| Status::internal(e.to_string()));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_get_table_types(
        &self,
        query: CommandGetTableTypes,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        let mut builder = query.into_builder();
        builder.append("TABLE");
        builder.append("VIEW");

        let schema = builder.schema();
        let batch = builder.build();
        let stream = arrow_flight::encode::FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(futures::stream::once(async { batch }))
            .map_err(|e| Status::internal(e.to_string()));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_get_sql_info(
        &self,
        query: CommandGetSqlInfo,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        let mut data_builder = SqlInfoDataBuilder::new();
        data_builder.append(
            SqlInfo::FlightSqlServerName,
            "BenoStreamDB Flight SQL Server",
        );
        data_builder.append(SqlInfo::FlightSqlServerVersion, "1");
        data_builder.append(SqlInfo::FlightSqlServerArrowVersion, "1.3");
        let sql_info_data = data_builder
            .build()
            .map_err(|e| Status::internal(e.to_string()))?;

        let builder = query.into_builder(&sql_info_data);
        let schema = builder.schema();
        let batch = builder.build();
        let stream = arrow_flight::encode::FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(futures::stream::once(async { batch }))
            .map_err(|e| Status::internal(e.to_string()));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_get_primary_keys(
        &self,
        query: CommandGetPrimaryKeys,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        let catalog = query.catalog.clone().unwrap_or_default();
        let db_schema = query.db_schema.clone().unwrap_or_default();
        let table = query.table.clone();
        let keys = self
            .session
            .get_primary_keys(&catalog, &db_schema, &table)
            .await;

        let schema = primary_keys_schema();
        let mut catalog_arr = Vec::new();
        let mut schema_arr = Vec::new();
        let mut table_arr = Vec::new();
        let mut column_arr = Vec::new();
        let mut key_name_arr = Vec::new();
        let mut key_seq_arr = Vec::new();
        for (i, col) in keys.iter().enumerate() {
            catalog_arr.push(Some(catalog.clone()));
            schema_arr.push(Some(db_schema.clone()));
            table_arr.push(Some(table.clone()));
            column_arr.push(Some(col.clone()));
            key_name_arr.push(Some("PRIMARY".to_string()));
            key_seq_arr.push(Some((i + 1) as i32));
        }
        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![
                std::sync::Arc::new(arrow::array::StringArray::from_iter(
                    catalog_arr.iter().map(|s| s.as_deref()),
                )),
                std::sync::Arc::new(arrow::array::StringArray::from_iter(
                    schema_arr.iter().map(|s| s.as_deref()),
                )),
                std::sync::Arc::new(arrow::array::StringArray::from_iter(
                    table_arr.iter().map(|s| s.as_deref()),
                )),
                std::sync::Arc::new(arrow::array::StringArray::from_iter(
                    column_arr.iter().map(|s| s.as_deref()),
                )),
                std::sync::Arc::new(arrow::array::StringArray::from_iter(
                    key_name_arr.iter().map(|s| s.as_deref()),
                )),
                std::sync::Arc::new(arrow::array::Int32Array::from(key_seq_arr)),
            ],
        )
        .map_err(|e| Status::internal(e.to_string()))?;

        let stream = arrow_flight::encode::FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(futures::stream::once(async { Ok(batch) }))
            .map_err(|e| Status::internal(e.to_string()));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_get_exported_keys(
        &self,
        _query: CommandGetExportedKeys,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn do_get_imported_keys(
        &self,
        _query: CommandGetImportedKeys,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn do_get_cross_reference(
        &self,
        _query: CommandGetCrossReference,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn do_put_statement_update(
        &self,
        query: CommandStatementUpdate,
        _request: Request<arrow_flight::sql::server::PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        let sql = query.query;
        {
            let mut tx = self.tx.lock().await;
            if !tx.autocommit {
                tx.staged.push(sql);
                return Ok(0);
            }
        }
        let batches = self.execute_statement(&sql).await?;
        Ok(batches.iter().map(|b| b.num_rows() as i64).sum())
    }

    async fn do_put_prepared_statement_query(
        &self,
        query: CommandPreparedStatementQuery,
        _request: Request<arrow_flight::sql::server::PeekableFlightDataStream>,
    ) -> Result<arrow_flight::sql::DoPutPreparedStatementResult, Status> {
        // Validate the handle; the result set is fetched via DoGet.
        self.prepared_sql(&query.prepared_statement_handle).await?;
        Ok(arrow_flight::sql::DoPutPreparedStatementResult {
            prepared_statement_handle: Some(query.prepared_statement_handle),
        })
    }

    async fn do_put_prepared_statement_update(
        &self,
        query: CommandPreparedStatementUpdate,
        _request: Request<arrow_flight::sql::server::PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        let sql = self.prepared_sql(&query.prepared_statement_handle).await?;
        let batches = self.execute_statement(&sql).await?;
        Ok(batches.iter().map(|b| b.num_rows() as i64).sum())
    }

    async fn do_action_create_prepared_statement(
        &self,
        query: ActionCreatePreparedStatementRequest,
        _request: Request<Action>,
    ) -> Result<ActionCreatePreparedStatementResult, Status> {
        let handle = new_statement_handle();
        self.prepared
            .lock()
            .await
            .insert(handle.clone(), query.query.clone());

        // Best-effort dataset schema so clients can bind result columns.
        let dataset_schema = match self.session.get_schema(&query.query).await {
            Ok(schema) => {
                let options = datafusion::arrow::ipc::writer::IpcWriteOptions::default();
                let schema_as_ipc = SchemaAsIpc::new(schema.as_ref(), &options);
                IpcMessage::try_from(schema_as_ipc)
                    .map(|m| m.0.to_vec())
                    .unwrap_or_default()
            }
            Err(_) => Vec::new(),
        };

        Ok(ActionCreatePreparedStatementResult {
            prepared_statement_handle: handle.into(),
            dataset_schema: dataset_schema.into(),
            parameter_schema: Vec::new().into(),
        })
    }

    async fn do_action_close_prepared_statement(
        &self,
        query: ActionClosePreparedStatementRequest,
        _request: Request<Action>,
    ) -> Result<(), Status> {
        self.prepared
            .lock()
            .await
            .remove(query.prepared_statement_handle.as_ref());
        Ok(())
    }

    async fn do_action_fallback(
        &self,
        request: Request<Action>,
    ) -> Result<
        Response<<Self as arrow_flight::flight_service_server::FlightService>::DoActionStream>,
        Status,
    > {
        let action_type = request.get_ref().r#type.clone();
        // The ADBC Flight SQL driver disables autocommit by sending a
        // `SetAutocommit` action. BenoStreamDB supports staged commits, so
        // honour it: enabling autocommit commits any staged statements.
        if action_type.to_ascii_lowercase().contains("autocommit") {
            let enabled = parse_autocommit_body(&request.get_ref().body);
            let mut tx = self.tx.lock().await;
            if enabled && !tx.autocommit {
                let staged = std::mem::take(&mut tx.staged);
                tx.autocommit = true;
                drop(tx);
                for sql in staged {
                    self.execute_statement(&sql).await?;
                }
            } else {
                tx.autocommit = enabled;
            }
            return Ok(Response::new(Box::pin(futures::stream::empty())));
        }
        Err(Status::invalid_argument(format!(
            "do_action: unsupported action type: {}",
            action_type
        )))
    }

    async fn do_action_begin_transaction(
        &self,
        _query: ActionBeginTransactionRequest,
        _request: Request<Action>,
    ) -> Result<ActionBeginTransactionResult, Status> {
        let mut tx = self.tx.lock().await;
        tx.autocommit = false;
        tx.staged.clear();
        Ok(ActionBeginTransactionResult {
            transaction_id: prost::bytes::Bytes::from_static(b"benostream-tx"),
        })
    }

    async fn do_action_end_transaction(
        &self,
        query: ActionEndTransactionRequest,
        _request: Request<Action>,
    ) -> Result<(), Status> {
        let commit = query.action == arrow_flight::sql::EndTransaction::Commit as i32;
        let mut tx = self.tx.lock().await;
        let staged = std::mem::take(&mut tx.staged);
        tx.autocommit = true;
        drop(tx);
        if commit {
            for sql in staged {
                self.execute_statement(&sql).await?;
            }
        }
        Ok(())
    }

    async fn get_flight_info_xdbc_type_info(
        &self,
        query: CommandGetXdbcTypeInfo,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let flight_descriptor = request.into_inner();
        let ticket = Ticket::new(query.as_any().encode_to_vec());
        let endpoint = FlightEndpoint::new().with_ticket(ticket);
        let infos = xdbc_type_info_data()?;
        let schema = query.into_builder(&infos).schema();
        let flight_info = FlightInfo::new()
            .try_with_schema(schema.as_ref())
            .map_err(|e| Status::internal(e.to_string()))?
            .with_endpoint(endpoint)
            .with_descriptor(flight_descriptor);
        Ok(Response::new(flight_info))
    }

    async fn do_get_xdbc_type_info(
        &self,
        query: CommandGetXdbcTypeInfo,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        let infos = xdbc_type_info_data()?;
        let builder = query.into_builder(&infos);
        let schema = builder.schema();
        let batch = builder
            .build()
            .map_err(|e| Status::internal(e.to_string()))?;
        let stream = arrow_flight::encode::FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(futures::stream::once(async { Ok(batch) }))
            .map_err(|e| Status::internal(e.to_string()));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_put_statement_ingest(
        &self,
        ticket: CommandStatementIngest,
        request: Request<arrow_flight::sql::server::PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        // Decode the incoming Arrow stream and append it to the target table.
        let mut stream = request.into_inner();
        let mut schema: Option<arrow::datatypes::SchemaRef> = None;
        let mut batches: Vec<arrow::record_batch::RecordBatch> = Vec::new();
        while let Some(data) = stream
            .try_next()
            .await
            .map_err(|e| Status::internal(e.to_string()))?
        {
            if schema.is_none() {
                let message = arrow::ipc::root_as_message(&data.data_header[..])
                    .map_err(|e| Status::internal(format!("invalid flight data header: {e}")))?;
                let fb = message
                    .header_as_schema()
                    .ok_or_else(|| Status::internal("flight data header is not a schema"))?;
                schema = Some(std::sync::Arc::new(arrow::ipc::convert::fb_to_schema(fb)));
            }
            let batch = arrow_flight::utils::flight_data_to_arrow_batch(
                &data,
                schema.clone().expect("schema set above"),
                &std::collections::HashMap::new(),
            )
            .map_err(|e| Status::internal(e.to_string()))?;
            batches.push(batch);
        }

        let rows: i64 = batches.iter().map(|b| b.num_rows() as i64).sum();
        if batches.is_empty() {
            return Ok(0);
        }

        // Stage the batches and insert them into the target table.
        let staging = format!("__ingest_{}", ticket.table);
        let ctx = self.session.get_ctx();
        let mem = datafusion::datasource::MemTable::try_new(
            schema.expect("schema set above"),
            vec![batches],
        )
        .map_err(|e| Status::internal(e.to_string()))?;
        ctx.register_table(&staging, std::sync::Arc::new(mem))
            .map_err(|e| Status::internal(e.to_string()))?;
        let sql = format!("INSERT INTO {} SELECT * FROM {}", ticket.table, staging);
        let result = self.execute_statement(&sql).await;
        let _ = ctx.deregister_table(&staging);
        result?;
        Ok(rows)
    }

    async fn register_sql_info(&self, _id: i32, _result: &arrow_flight::sql::SqlInfo) {}
}

/// Build the XDBC type-info table advertised by the server.
fn xdbc_type_info_data() -> Result<arrow_flight::sql::metadata::XdbcTypeInfoData, Status> {
    use arrow_flight::sql::metadata::{XdbcTypeInfo, XdbcTypeInfoDataBuilder};
    use arrow_flight::sql::{Nullable, Searchable, XdbcDataType};
    let mut builder = XdbcTypeInfoDataBuilder::new();
    for (name, dt) in [
        ("BOOLEAN", XdbcDataType::XdbcBit),
        ("INTEGER", XdbcDataType::XdbcInteger),
        ("BIGINT", XdbcDataType::XdbcBigint),
        ("REAL", XdbcDataType::XdbcFloat),
        ("DOUBLE", XdbcDataType::XdbcDouble),
        ("VARCHAR", XdbcDataType::XdbcVarchar),
        ("TIMESTAMP", XdbcDataType::XdbcTimestamp),
    ] {
        builder.append(XdbcTypeInfo {
            type_name: name.to_string(),
            data_type: dt,
            nullable: Nullable::NullabilityNullable,
            searchable: Searchable::Char,
            ..Default::default()
        });
    }
    builder.build().map_err(|e| Status::internal(e.to_string()))
}
