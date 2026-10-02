// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Telemetry and Observability Configuration
//!
//! The engine records metrics through **two** systems:
//!
//! 1. the `prometheus` crate, in [`crate::telemetry::metrics`] (ingest, query,
//!    compaction, index build, back-pressure, read/delete phases); and
//! 2. the `metrics` facade, in the query planner, compaction, and manifest
//!    commit paths.
//!
//! [`render_metrics`] concatenates both so every host serves one `/metrics`
//! body. The `metrics`-facade side is only captured when the `observability`
//! feature is enabled and [`install_metrics_recorder`] has run.

/// Handle to the `metrics`-facade Prometheus recorder, installed once.
#[cfg(feature = "observability")]
static PROMETHEUS_HANDLE: once_cell::sync::OnceCell<metrics_exporter_prometheus::PrometheusHandle> =
    once_cell::sync::OnceCell::new();

/// Install the `metrics`-facade Prometheus recorder once.
///
/// Idempotent: a second call is a no-op. Hosts that serve their own `/metrics`
/// (the Flight SQL server, the search gateway) should call this at startup so
/// the facade metrics are captured, then serve [`render_metrics`].
#[cfg(feature = "observability")]
pub fn install_metrics_recorder() {
    if PROMETHEUS_HANDLE.get().is_some() {
        return;
    }
    match metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder() {
        Ok(handle) => {
            let _ = PROMETHEUS_HANDLE.set(handle);
        }
        Err(e) => {
            tracing::warn!("Failed to install metrics recorder: {e}");
        }
    }
}

/// No-op when the `observability` feature is disabled.
#[cfg(not(feature = "observability"))]
pub fn install_metrics_recorder() {}

/// Render every engine metric in Prometheus text format (version 0.0.4).
///
/// Concatenates the `prometheus`-crate registry
/// ([`crate::telemetry::metrics::gather_text`]) with the `metrics`-facade
/// recorder (when the `observability` feature is enabled and the recorder was
/// installed). This is the single body served by every host's `/metrics`.
pub fn render_metrics() -> String {
    let base = crate::telemetry::metrics::gather_text();
    #[cfg(feature = "observability")]
    {
        if let Some(handle) = PROMETHEUS_HANDLE.get() {
            return format!("{base}{}", handle.render());
        }
    }
    base
}

/// Resolve the metrics listener address from `BSDB_METRICS_BIND` (default
/// `127.0.0.1`) and `BSDB_METRICS_PORT` (default `9090`).
pub fn metrics_addr() -> String {
    let bind = std::env::var("BSDB_METRICS_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("BSDB_METRICS_PORT").unwrap_or_else(|_| "9090".to_string());
    format!("{bind}:{port}")
}

/// Initializes the metrics exporter if the `observability` feature is enabled.
///
/// Installs the `metrics`-facade recorder and spawns a small HTTP server on
/// `BSDB_METRICS_BIND:BSDB_METRICS_PORT` (default `127.0.0.1:9090`) serving
/// `/metrics`, `/health`, and `/readyz`. The server runs on a dedicated thread
/// so it does not require an ambient Tokio runtime.
#[cfg(feature = "observability")]
pub fn init_metrics_exporter() -> anyhow::Result<()> {
    install_metrics_recorder();
    // OTLP push is independent of the scrape endpoint: start it whenever
    // BSDB_OTLP_ENDPOINT is set, so pull (`/metrics`) and push can coexist.
    crate::telemetry::otlp::spawn_pusher();
    let bind = metrics_addr();
    std::thread::Builder::new()
        .name("bsdb-metrics".to_string())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    tracing::warn!("Failed to build metrics runtime: {e}");
                    return;
                }
            };
            rt.block_on(async move {
                use axum::{routing::get, Router};
                let app = Router::new()
                    .route("/metrics", get(|| async { render_metrics() }))
                    .route("/health", get(|| async { "ok" }))
                    .route("/readyz", get(|| async { "ok" }));
                match tokio::net::TcpListener::bind(&bind).await {
                    Ok(listener) => {
                        tracing::info!("Metrics exporter started at http://{bind}/metrics");
                        if let Err(e) = axum::serve(listener, app).await {
                            tracing::warn!("Metrics server stopped: {e}");
                        }
                    }
                    Err(e) => tracing::warn!("Failed to bind metrics listener {bind}: {e}"),
                }
            });
        })?;
    Ok(())
}

#[cfg(not(feature = "observability"))]
pub fn init_metrics_exporter() -> anyhow::Result<()> {
    // Even without the scrape endpoint, an OTLP endpoint can be pushed to.
    crate::telemetry::otlp::spawn_pusher();
    tracing::debug!("Metrics exporter is disabled. Enable the `observability` feature to start Prometheus metrics.");
    Ok(())
}
