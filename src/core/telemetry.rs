// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Telemetry and Observability Configuration

/// Initializes the metrics exporter if the `observability` feature is enabled.
/// Spawns an HTTP server on `0.0.0.0:9090` serving Prometheus metrics on `/metrics`.
#[cfg(feature = "observability")]
pub fn init_metrics_exporter() -> anyhow::Result<()> {
    use metrics_exporter_prometheus::PrometheusBuilder;
    use std::net::SocketAddr;

    let builder = PrometheusBuilder::new();
    let addr: SocketAddr = "0.0.0.0:9090".parse()?;

    // Install the global metrics recorder
    builder.with_http_listener(addr).install()?;

    tracing::info!("Metrics exporter started at http://{}", addr);
    Ok(())
}

#[cfg(not(feature = "observability"))]
pub fn init_metrics_exporter() -> anyhow::Result<()> {
    tracing::debug!("Metrics exporter is disabled. Enable the `observability` feature to start Prometheus metrics.");
    Ok(())
}
