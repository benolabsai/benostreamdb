use arrow_flight::sql::server::FlightSqlService;
use arrow_flight::sql::{
    CommandGetCatalogs, CommandGetDbSchemas, CommandGetSqlInfo, CommandGetTableTypes,
    CommandGetTables, CommandStatementQuery, TicketStatementQuery,
};
use arrow_flight::{FlightDescriptor, Ticket};
use futures::StreamExt;
use hyperstreamdb::core::sql::session::HyperStreamSession;
use hyperstreamdb_flight::HyperStreamFlightSqlService;
use tonic::Request;

#[tokio::test]
async fn test_flight_sql_info() {
    let session = HyperStreamSession::new(None);
    let service = HyperStreamFlightSqlService::new(session);

    let query = CommandGetSqlInfo { info: vec![] };
    let desc = FlightDescriptor::new_cmd(vec![]);
    let response = service
        .get_flight_info_sql_info(query.clone(), Request::new(desc))
        .await
        .expect("get_flight_info_sql_info should succeed");

    let flight_info = response.into_inner();
    assert!(
        !flight_info.endpoint.is_empty(),
        "endpoint should be present"
    );

    let ticket = query;
    let stream_res = service
        .do_get_sql_info(ticket, Request::new(Ticket::new(vec![])))
        .await
        .expect("do_get_sql_info should succeed");

    let mut stream = stream_res.into_inner();
    let first_batch = stream.next().await;
    assert!(
        first_batch.is_some(),
        "should stream at least one flight data message"
    );
}

#[tokio::test]
async fn test_flight_statement_query() {
    let session = HyperStreamSession::new(None);
    let service = HyperStreamFlightSqlService::new(session);

    let sql = "SELECT 42 AS answer, 'hyperstream' AS engine";
    let cmd = CommandStatementQuery {
        query: sql.to_string(),
        transaction_id: None,
    };
    let desc = FlightDescriptor::new_cmd(vec![]);

    let info_resp = service
        .get_flight_info_statement(cmd, Request::new(desc))
        .await
        .expect("get_flight_info_statement should succeed");

    let flight_info = info_resp.into_inner();
    assert_eq!(flight_info.endpoint.len(), 1);

    // Now execute via do_get_statement
    let ticket_query = TicketStatementQuery {
        statement_handle: sql.as_bytes().to_vec().into(),
    };
    let data_resp = service
        .do_get_statement(ticket_query, Request::new(Ticket::new(vec![])))
        .await
        .expect("do_get_statement should succeed");

    let mut stream = data_resp.into_inner();
    let mut batches = Vec::new();
    while let Some(item) = stream.next().await {
        let flight_data = item.expect("flight data item ok");
        batches.push(flight_data);
    }
    assert!(
        !batches.is_empty(),
        "expected flight data batches for SELECT query"
    );
}

#[tokio::test]
async fn test_flight_metadata_catalogs_schemas_tables() {
    let session = HyperStreamSession::new(None);
    let service = HyperStreamFlightSqlService::new(session);

    // Catalogs
    let cat_resp = service
        .get_flight_info_catalogs(
            CommandGetCatalogs {},
            Request::new(FlightDescriptor::new_cmd(vec![])),
        )
        .await
        .expect("catalogs info");
    assert!(!cat_resp.into_inner().endpoint.is_empty());

    // Schemas
    let schema_cmd = CommandGetDbSchemas {
        catalog: None,
        db_schema_filter_pattern: None,
    };
    let schema_resp = service
        .get_flight_info_schemas(schema_cmd, Request::new(FlightDescriptor::new_cmd(vec![])))
        .await
        .expect("schemas info");
    assert!(!schema_resp.into_inner().endpoint.is_empty());

    // Tables
    let tables_cmd = CommandGetTables {
        catalog: None,
        db_schema_filter_pattern: None,
        table_name_filter_pattern: None,
        table_types: vec![],
        include_schema: false,
    };
    let tables_resp = service
        .get_flight_info_tables(tables_cmd, Request::new(FlightDescriptor::new_cmd(vec![])))
        .await
        .expect("tables info");
    assert!(!tables_resp.into_inner().endpoint.is_empty());

    // Table Types
    let types_resp = service
        .get_flight_info_table_types(
            CommandGetTableTypes {},
            Request::new(FlightDescriptor::new_cmd(vec![])),
        )
        .await
        .expect("table types info");
    assert!(!types_resp.into_inner().endpoint.is_empty());
}
