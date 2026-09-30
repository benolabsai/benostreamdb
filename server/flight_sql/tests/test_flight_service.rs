use arrow_flight::sql::server::FlightSqlService;
use arrow_flight::sql::{
    CommandGetCatalogs, CommandGetDbSchemas, CommandGetSqlInfo, CommandGetTableTypes,
    CommandGetTables, CommandStatementQuery, TicketStatementQuery,
};
use arrow_flight::{FlightDescriptor, Ticket};
use benostreamdb::core::sql::session::BenoStreamSession;
use benostreamdb_flight::BenoStreamFlightSqlService;
use futures::StreamExt;
use tonic::Request;

#[tokio::test]
async fn test_flight_sql_info() {
    let session = BenoStreamSession::new(None);
    let service = BenoStreamFlightSqlService::new(session);

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
    let session = BenoStreamSession::new(None);
    let service = BenoStreamFlightSqlService::new(session);

    let sql = "SELECT 42 AS answer, 'benostream' AS engine";
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
    let session = BenoStreamSession::new(None);
    let service = BenoStreamFlightSqlService::new(session);

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

#[tokio::test]
async fn test_flight_concurrent_queries() {
    let session = BenoStreamSession::new(None);
    let service = BenoStreamFlightSqlService::new(session);

    let mut handles = Vec::new();
    for i in 0..8 {
        let s = service.clone();
        handles.push(tokio::spawn(async move {
            let sql = format!("SELECT {} AS query_id, sqrt({}) AS val", i, i * 100);
            let cmd = CommandStatementQuery {
                query: sql.clone(),
                transaction_id: None,
            };
            let desc = FlightDescriptor::new_cmd(vec![]);
            let info = s
                .get_flight_info_statement(cmd, Request::new(desc))
                .await
                .expect("get_flight_info_statement concurrent")
                .into_inner();
            assert!(!info.endpoint.is_empty());

            let ticket_query = TicketStatementQuery {
                statement_handle: sql.into_bytes().into(),
            };
            let resp = s
                .do_get_statement(ticket_query, Request::new(Ticket::new(vec![])))
                .await
                .expect("do_get_statement concurrent");
            let mut stream = resp.into_inner();
            let mut batch_count = 0;
            while let Some(batch) = stream.next().await {
                let _ = batch.expect("valid flight data");
                batch_count += 1;
            }
            assert!(batch_count > 0, "must stream back results");
        }));
    }

    for h in handles {
        h.await.expect("task completed without panic");
    }
}

#[tokio::test]
async fn test_flight_large_arrow_batches() {
    let session = BenoStreamSession::new(None);
    let service = BenoStreamFlightSqlService::new(session);

    // Generate large batch through DataFusion unnest range
    let sql = "SELECT unnest(range(0, 10000)) AS id, 'payload_chunk_data' AS marker";
    let cmd = CommandStatementQuery {
        query: sql.to_string(),
        transaction_id: None,
    };
    let desc = FlightDescriptor::new_cmd(vec![]);
    let info = service
        .get_flight_info_statement(cmd, Request::new(desc))
        .await
        .expect("flight info for large batch")
        .into_inner();
    assert!(!info.endpoint.is_empty());

    let ticket_query = TicketStatementQuery {
        statement_handle: sql.as_bytes().to_vec().into(),
    };
    let resp = service
        .do_get_statement(ticket_query, Request::new(Ticket::new(vec![])))
        .await
        .expect("do_get for large batch");
    let mut stream = resp.into_inner();

    let mut total_batches = 0;
    while let Some(batch) = stream.next().await {
        let _ = batch.expect("valid chunk");
        total_batches += 1;
    }
    assert!(total_batches > 0, "should produce chunked flight data");
}

#[tokio::test]
async fn test_flight_error_handling() {
    let session = BenoStreamSession::new(None);
    let service = BenoStreamFlightSqlService::new(session);

    // 1. Syntax Error
    let bad_sql = "SELECT NONEXISTENT SYNTAX FROM";
    let cmd = CommandStatementQuery {
        query: bad_sql.to_string(),
        transaction_id: None,
    };
    let res = service
        .get_flight_info_statement(cmd, Request::new(FlightDescriptor::new_cmd(vec![])))
        .await;
    assert!(res.is_err(), "syntax error must return Err Status");

    // 2. Missing Table
    let missing_table = "SELECT * FROM definitely_does_not_exist_xyz123";
    let cmd2 = CommandStatementQuery {
        query: missing_table.to_string(),
        transaction_id: None,
    };
    let res2 = service
        .get_flight_info_statement(cmd2, Request::new(FlightDescriptor::new_cmd(vec![])))
        .await;
    assert!(res2.is_err(), "missing table query must return Err Status");
}

#[tokio::test]
async fn test_flight_stream_cancellation() {
    let session = BenoStreamSession::new(None);
    let service = BenoStreamFlightSqlService::new(session);

    let sql = "SELECT unnest(range(0, 50000)) AS id";
    let ticket_query = TicketStatementQuery {
        statement_handle: sql.as_bytes().to_vec().into(),
    };
    let resp = service
        .do_get_statement(ticket_query, Request::new(Ticket::new(vec![])))
        .await
        .expect("do_get for cancellation");
    let mut stream = resp.into_inner();

    // Consume first batch, then drop the stream to simulate client-side cancellation
    let first = stream.next().await;
    assert!(first.is_some(), "first batch received");
    drop(stream);

    // Verify service remains unblocked and responsive for subsequent queries
    let follow_up = "SELECT 999 AS keepalive";
    let ticket_followup = TicketStatementQuery {
        statement_handle: follow_up.as_bytes().to_vec().into(),
    };
    let resp2 = service
        .do_get_statement(ticket_followup, Request::new(Ticket::new(vec![])))
        .await
        .expect("subsequent query succeeds");
    let mut stream2 = resp2.into_inner();
    assert!(
        stream2.next().await.is_some(),
        "service is healthy after stream cancel"
    );
}
