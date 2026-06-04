use opentelemetry::global;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{Metadata, Subscriber};
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::{Layer, SubscriberExt};
use tracing_subscriber::{layer::Context, Registry};

struct ReloadableFilter {
    filter: Arc<RwLock<EnvFilter>>,
}

impl<S: Subscriber> Layer<S> for ReloadableFilter {
    fn enabled(&self, metadata: &Metadata<'_>, ctx: Context<'_, S>) -> bool {
        self.filter.read().enabled(metadata, ctx)
    }
}

static SHARED_FILTER: once_cell::sync::OnceCell<Arc<RwLock<EnvFilter>>> =
    once_cell::sync::OnceCell::new();
static INIT: std::sync::Once = std::sync::Once::new();

pub struct TelemetryGuard;

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        shutdown_telemetry();
    }
}

pub fn shutdown_telemetry() {
    opentelemetry::global::shutdown_tracer_provider();
}

static TRACER_PROVIDER: once_cell::sync::OnceCell<opentelemetry_sdk::trace::TracerProvider> =
    once_cell::sync::OnceCell::new();

pub fn flush_telemetry() {
    if let Some(provider) = TRACER_PROVIDER.get() {
        for p in provider.force_flush() {
            tracing::debug!("Telemetry flush result: {:?}", p);
        }
    }
}

pub fn init_tracing(service_name: &str) -> Result<TelemetryGuard, Box<dyn std::error::Error>> {
    let mut init_err: Option<Box<dyn std::error::Error>> = None;

    INIT.call_once(|| {
        if let Err(e) = do_init_tracing(service_name) {
            init_err = Some(e);
        }
    });

    if let Some(e) = init_err {
        return Err(e);
    }

    Ok(TelemetryGuard)
}

fn do_init_tracing(service_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    global::set_text_map_propagator(TraceContextPropagator::new());

    let base_filter = EnvFilter::from_default_env();
    let shared_filter = Arc::new(RwLock::new(base_filter));
    let _ = SHARED_FILTER.set(shared_filter.clone());

    let filter_layer = ReloadableFilter {
        filter: shared_filter,
    };

    // Check env var to enable Jaeger/OTLP
    let enable_jaeger =
        std::env::var("JAEGER_ENABLED").unwrap_or_else(|_| "false".to_string()) == "true";

    if enable_jaeger {
        let provider = opentelemetry_sdk::trace::TracerProvider::builder()
            .with_batch_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .build_span_exporter()?,
                opentelemetry_sdk::runtime::Tokio,
            )
            .with_config(opentelemetry_sdk::trace::config().with_resource(
                opentelemetry_sdk::Resource::new(vec![opentelemetry::KeyValue::new(
                    "service.name",
                    service_name.to_string(),
                )]),
            ))
            .build();

        let tracer = opentelemetry::trace::TracerProvider::tracer(&provider, "hyperstreamdb");
        let _ = TRACER_PROVIDER.set(provider.clone());
        opentelemetry::global::set_tracer_provider(provider);

        let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

        let subscriber = Registry::default().with(filter_layer).with(telemetry);

        let _ = tracing::subscriber::set_global_default(subscriber);
    } else {
        let subscriber = Registry::default().with(filter_layer).with(fmt::layer());

        let _ = tracing::subscriber::set_global_default(subscriber);
    }

    Ok(())
}

pub fn update_log_level(level: &str) -> Result<(), String> {
    if let Some(shared) = SHARED_FILTER.get() {
        let new_filter = EnvFilter::try_new(level).map_err(|e| e.to_string())?;
        let mut filter = shared.write();
        *filter = new_filter;
        Ok(())
    } else {
        std::env::set_var("RUST_LOG", level);
        Ok(())
    }
}
