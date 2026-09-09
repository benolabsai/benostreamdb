use arrow_flight::sql::metadata::SqlInfoDataBuilder;
use arrow_flight::sql::server::FlightSqlService;
use arrow_flight::sql::ProstMessageExt;
use arrow_flight::sql::{
    ActionClosePreparedStatementRequest, ActionCreatePreparedStatementRequest,
    ActionCreatePreparedStatementResult, Any, CommandGetCatalogs, CommandGetCrossReference,
    CommandGetDbSchemas, CommandGetExportedKeys, CommandGetImportedKeys, CommandGetPrimaryKeys,
    CommandGetSqlInfo, CommandGetTableTypes, CommandGetTables, CommandPreparedStatementQuery,
    CommandPreparedStatementUpdate, CommandStatementQuery, CommandStatementUpdate, SqlInfo,
    TicketStatementQuery,
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
pub struct HyperStreamFlightSqlService {
    pub session: hyperstreamdb::core::sql::session::HyperStreamSession,
}

impl HyperStreamFlightSqlService {
    pub fn new(session: hyperstreamdb::core::sql::session::HyperStreamSession) -> Self {
        Self { session }
    }
}

#[tonic::async_trait]
impl FlightSqlService for HyperStreamFlightSqlService {
    type FlightService = HyperStreamFlightSqlService;

    async fn do_handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<BoxStream<'static, Result<HandshakeResponse, Status>>>, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn do_get_fallback(
        &self,
        _request: Request<Ticket>,
        message: Any,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        Err(Status::unimplemented(format!(
            "do_get_fallback not implemented for: {:?}",
            message.type_url
        )))
    }

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let sql = query.query.clone();

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
        _cmd: CommandPreparedStatementQuery,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("Not implemented"))
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
            "HyperStreamDB Flight SQL Server",
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
        _query: CommandGetPrimaryKeys,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("Not implemented"))
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
        _query: CommandPreparedStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        Err(Status::unimplemented("Not implemented"))
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
            "HyperStreamDB Flight SQL Server",
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
        _query: CommandGetPrimaryKeys,
        _request: Request<Ticket>,
    ) -> Result<Response<BoxStream<'static, Result<FlightData, Status>>>, Status> {
        Err(Status::unimplemented("Not implemented"))
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
        let df = self
            .session
            .sql_to_df(&sql)
            .await
            .map_err(|e| Status::internal(format!("Error executing statement: {}", e)))?;

        let batches = df
            .collect()
            .await
            .map_err(|e| Status::internal(format!("Error collecting execution result: {}", e)))?;

        let mut row_count = 0;
        for batch in batches {
            row_count += batch.num_rows() as i64;
        }

        Ok(row_count)
    }

    async fn do_put_prepared_statement_query(
        &self,
        _query: CommandPreparedStatementQuery,
        _request: Request<arrow_flight::sql::server::PeekableFlightDataStream>,
    ) -> Result<arrow_flight::sql::DoPutPreparedStatementResult, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn do_put_prepared_statement_update(
        &self,
        _query: CommandPreparedStatementUpdate,
        _request: Request<arrow_flight::sql::server::PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn do_action_create_prepared_statement(
        &self,
        _query: ActionCreatePreparedStatementRequest,
        _request: Request<Action>,
    ) -> Result<ActionCreatePreparedStatementResult, Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn do_action_close_prepared_statement(
        &self,
        _query: ActionClosePreparedStatementRequest,
        _request: Request<Action>,
    ) -> Result<(), Status> {
        Err(Status::unimplemented("Not implemented"))
    }

    async fn register_sql_info(&self, _id: i32, _result: &arrow_flight::sql::SqlInfo) {}
}
