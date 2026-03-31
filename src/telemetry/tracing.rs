// Copyright (c) 2026 Richard Albright. All rights reserved.

use opentelemetry::global;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::Registry;

pub fn init_tracing(service_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    global::set_text_map_propagator(TraceContextPropagator::new());

    // By default, only log to stdout
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
            .with(tracing_subscriber::EnvFilter::from_default_env())
            .with(telemetry);

        tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    } else {
         // Standard logging - use try_init to avoid panics on second call
         let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();
    }
    
    Ok(())
}
