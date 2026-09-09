use arrow_flight::flight_service_server::FlightServiceServer;
use std::net::SocketAddr;
use tonic::transport::Server;

use hyperstreamdb_flight::HyperStreamFlightSqlService;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting HyperStreamDB Arrow Flight SQL Server...");

    let session = hyperstreamdb::core::sql::session::HyperStreamSession::new(None);

    // Create our Flight SQL service
    let flight_sql_service = HyperStreamFlightSqlService::new(session);

    // In arrow-flight, FlightSqlService might be a trait.
    // We can try to use it directly with FlightServiceServer if it auto-implements FlightService
    let svc = FlightServiceServer::new(flight_sql_service);

    let addr: SocketAddr = "0.0.0.0:50051".parse()?;
    println!("Listening on grpc://{}", addr);

    Server::builder().add_service(svc).serve(addr).await?;

    Ok(())
}
