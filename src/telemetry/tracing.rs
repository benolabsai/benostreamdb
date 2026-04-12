use opentelemetry::global;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use tracing_subscriber::layer::{SubscriberExt, Layer};
use tracing_subscriber::{Registry, layer::Context};
use tracing_subscriber::fmt;
use tracing_subscriber::filter::EnvFilter;
use tracing::{Subscriber, Metadata};
use std::sync::{Arc, RwLock};

struct ReloadableFilter {
    filter: Arc<RwLock<EnvFilter>>,
}

impl<S: Subscriber> Layer<S> for ReloadableFilter {
    fn enabled(&self, metadata: &Metadata<'_>, ctx: Context<'_, S>) -> bool {
        self.filter.read().unwrap().enabled(metadata, ctx)
    }

    fn on_event(&self, event: &tracing::Event<'_>, ctx: Context<'_, S>) {
        self.filter.read().unwrap().on_event(event, ctx);
    }
}

static SHARED_FILTER: once_cell::sync::OnceCell<Arc<RwLock<EnvFilter>>> = once_cell::sync::OnceCell::new();

pub fn init_tracing(service_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    global::set_text_map_propagator(TraceContextPropagator::new());

    let base_filter = EnvFilter::from_default_env();
    let shared_filter = Arc::new(RwLock::new(base_filter));
    let _ = SHARED_FILTER.set(shared_filter.clone());

    let filter_layer = ReloadableFilter {
        filter: shared_filter,
    };

    // Check env var to enable Jaeger/OTLP
    let enable_jaeger = std::env::var("JAEGER_ENABLED").unwrap_or_else(|_| "false".to_string()) == "true";

    if enable_jaeger {
        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
            )
            .with_trace_config(
                opentelemetry_sdk::trace::config().with_resource(
                    opentelemetry_sdk::Resource::new(vec![
                        opentelemetry::KeyValue::new("service.name", service_name.to_string()),
                    ]),
                ),
            )
            .install_batch(opentelemetry_sdk::runtime::Tokio)?;

        let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);
        
        let subscriber = Registry::default()
            .with(filter_layer)
            .with(telemetry);

        let _ = tracing::subscriber::set_global_default(subscriber);
    } else {
         let subscriber = Registry::default()
            .with(filter_layer)
            .with(fmt::layer());

         let _ = tracing::subscriber::set_global_default(subscriber);
    }
    
    Ok(())
}

pub fn update_log_level(level: &str) -> Result<(), String> {
    if let Some(shared) = SHARED_FILTER.get() {
        let new_filter = EnvFilter::try_new(level).map_err(|e| e.to_string())?;
        let mut filter = shared.write().unwrap();
        *filter = new_filter;
        Ok(())
    } else {
        std::env::set_var("RUST_LOG", level);
        Ok(())
    }
}
