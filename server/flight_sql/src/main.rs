use arrow_flight::flight_service_server::FlightServiceServer;
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::transport::Server;

use benostreamdb_flight::BenoStreamFlightSqlService;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting BenoStreamDB Arrow Flight SQL Server...");

    let mut session = benostreamdb::core::sql::session::BenoStreamSession::new(None);

    // Base location used to derive `CREATE TABLE` URIs when the catalog does
    // not assign one.
    if let Ok(warehouse) = std::env::var("BSDB_WAREHOUSE") {
        session.set_warehouse(Some(warehouse));
    }

    // Optional external catalog (Nessie / REST / Glue / Hive / Unity / JDBC),
    // loaded from BENOSTREAM_CONFIG, ./benostream.toml, or
    // ~/.benostream/config.toml. Bound under BSDB_CATALOG_NAME (default: the
    // catalog type, e.g. "rest") so `CREATE TABLE <name>.<schema>.<table>`
    // mirrors into it.
    match benostreamdb::core::catalog::CatalogConfig::load_default() {
        Ok(cfg) => {
            let catalog_type = cfg.catalog_type;
            match benostreamdb::core::catalog::create_catalog_async(catalog_type, cfg.config).await
            {
                Ok(catalog) => {
                    let name = std::env::var("BSDB_CATALOG_NAME")
                        .unwrap_or_else(|_| format!("{:?}", catalog_type).to_lowercase());
                    session.register_catalog(&name, Arc::from(catalog)).await;
                    println!("Registered external catalog as '{}'", name);
                }
                Err(e) => eprintln!("Failed to create external catalog: {}", e),
            }
        }
        Err(e) => {
            println!(
                "No external catalog configured ({}); using local storage only",
                e
            );
        }
    }

    // Create our Flight SQL service
    let flight_sql_service = BenoStreamFlightSqlService::new(session);

    // In arrow-flight, FlightSqlService might be a trait.
    // We can try to use it directly with FlightServiceServer if it auto-implements FlightService
    let svc = FlightServiceServer::new(flight_sql_service);

    let addr: SocketAddr = "0.0.0.0:50051".parse()?;
    println!("Listening on grpc://{}", addr);

    Server::builder().add_service(svc).serve(addr).await?;

    Ok(())
}
