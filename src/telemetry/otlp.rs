// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! OTLP/HTTP metrics push.
//!
//! The engine records metrics with the `prometheus` crate (and, behind the
//! `observability` feature, the `metrics` facade). Rather than re-instrumenting
//! the whole engine against the OpenTelemetry API, this module converts the
//! Prometheus registry to the OTLP/HTTP **JSON** encoding and POSTs it to a
//! collector. That keeps the push path dependency-light: it uses only
//! `reqwest` and `serde_json`, which the engine already depends on.
//!
//! Enable by setting `BSDB_OTLP_ENDPOINT` (e.g. `http://localhost:4318`); the
//! pusher appends `/v1/metrics`. Other knobs:
//!
//! - `BSDB_OTLP_INTERVAL_SECS` — push period, default 15.
//! - `BSDB_OTLP_HEADERS` — comma-separated `key=value` headers (e.g. auth).
//! - `BSDB_SERVICE_NAME` — `service.name` resource attribute, default
//!   `benostreamdb`.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

fn now_unix_nano() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// The configured OTLP endpoint, if any (`BSDB_OTLP_ENDPOINT`).
pub fn endpoint() -> Option<String> {
    std::env::var("BSDB_OTLP_ENDPOINT")
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
}

fn push_interval() -> Duration {
    let secs = std::env::var("BSDB_OTLP_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|s| *s > 0)
        .unwrap_or(15);
    Duration::from_secs(secs)
}

fn service_name() -> String {
    std::env::var("BSDB_SERVICE_NAME").unwrap_or_else(|_| "benostreamdb".to_string())
}

fn extra_headers() -> Vec<(String, String)> {
    std::env::var("BSDB_OTLP_HEADERS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|pair| {
                    let (k, v) = pair.split_once('=')?;
                    let k = k.trim();
                    let v = v.trim();
                    if k.is_empty() || v.is_empty() {
                        None
                    } else {
                        Some((k.to_string(), v.to_string()))
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn attributes(labels: &[prometheus::proto::LabelPair]) -> Vec<Value> {
    labels
        .iter()
        .map(|l| {
            json!({
                "key": l.get_name(),
                "value": { "stringValue": l.get_value() }
            })
        })
        .collect()
}

/// Convert a Prometheus histogram to an OTLP `histogram` data point.
///
/// Prometheus buckets are cumulative and include a `+Inf` bucket; OTLP
/// `bucketCounts` are per-bucket and `explicitBounds` excludes `+Inf`, so the
/// counts are differenced and the overflow bucket becomes the final entry.
fn histogram_datapoint(
    labels: &[prometheus::proto::LabelPair],
    hist: &prometheus::proto::Histogram,
    ts: u64,
) -> Value {
    let mut bounds: Vec<f64> = Vec::new();
    let mut counts: Vec<u64> = Vec::new();
    let mut prev = 0u64;
    let mut finite_total = 0u64;
    for b in hist.get_bucket() {
        let upper = b.get_upper_bound();
        if upper.is_finite() {
            let cum = b.get_cumulative_count();
            counts.push(cum.saturating_sub(prev));
            prev = cum;
            finite_total = cum;
            bounds.push(upper);
        }
    }
    // Overflow bucket: everything above the last finite bound.
    counts.push(hist.get_sample_count().saturating_sub(finite_total));
    json!({
        "attributes": attributes(labels),
        "timeUnixNano": ts.to_string(),
        "count": hist.get_sample_count().to_string(),
        "sum": hist.get_sample_sum(),
        "bucketCounts": counts.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
        "explicitBounds": bounds,
    })
}

fn scalar_datapoint(labels: &[prometheus::proto::LabelPair], value: f64, ts: u64) -> Value {
    json!({
        "attributes": attributes(labels),
        "timeUnixNano": ts.to_string(),
        "asDouble": value,
    })
}

fn family_to_metric(family: &prometheus::proto::MetricFamily, ts: u64) -> Option<Value> {
    use prometheus::proto::MetricType;
    let name = family.get_name();
    let description = family.get_help();
    let metrics = family.get_metric();

    let metric = match family.get_field_type() {
        MetricType::COUNTER => {
            let points: Vec<Value> = metrics
                .iter()
                .map(|m| scalar_datapoint(m.get_label(), m.get_counter().get_value(), ts))
                .collect();
            json!({
                "name": name,
                "description": description,
                "unit": "1",
                "sum": {
                    "dataPoints": points,
                    "aggregationTemporality": 2,
                    "isMonotonic": true
                }
            })
        }
        MetricType::GAUGE | MetricType::UNTYPED => {
            let points: Vec<Value> = metrics
                .iter()
                .map(|m| {
                    let v = if family.get_field_type() == MetricType::GAUGE {
                        m.get_gauge().get_value()
                    } else {
                        m.get_untyped().get_value()
                    };
                    scalar_datapoint(m.get_label(), v, ts)
                })
                .collect();
            json!({ "name": name, "description": description, "unit": "1", "gauge": { "dataPoints": points } })
        }
        MetricType::HISTOGRAM => {
            let points: Vec<Value> = metrics
                .iter()
                .map(|m| histogram_datapoint(m.get_label(), m.get_histogram(), ts))
                .collect();
            json!({
                "name": name,
                "description": description,
                "unit": "1",
                "histogram": {
                    "dataPoints": points,
                    "aggregationTemporality": 2
                }
            })
        }
        // SUMMARY has no direct OTLP mapping here; skip rather than misreport.
        MetricType::SUMMARY => return None,
    };
    Some(metric)
}

/// Build an OTLP/HTTP `ExportMetricsServiceRequest` JSON body from the
/// Prometheus default registry (`prometheus::gather()`).
pub fn build_export_request() -> Value {
    let ts = now_unix_nano();
    let metrics: Vec<Value> = prometheus::gather()
        .iter()
        .filter_map(|f| family_to_metric(f, ts))
        .collect();

    json!({
        "resourceMetrics": [{
            "resource": {
                "attributes": [
                    { "key": "service.name", "value": { "stringValue": service_name() } },
                    { "key": "service.version", "value": { "stringValue": env!("CARGO_PKG_VERSION") } }
                ]
            },
            "scopeMetrics": [{
                "scope": { "name": "benostreamdb", "version": env!("CARGO_PKG_VERSION") },
                "metrics": metrics
            }]
        }]
    })
}

/// Push one batch to `endpoint/v1/metrics`. Returns the HTTP status on success.
pub fn push_once(endpoint: &str, body: &Value) -> anyhow::Result<u16> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let url = format!("{endpoint}/v1/metrics");
    let mut req = client
        .post(&url)
        .header("content-type", "application/json")
        .json(body);
    for (k, v) in extra_headers() {
        req = req.header(k, v);
    }
    let resp = req.send()?;
    Ok(resp.status().as_u16())
}

/// Spawn the background OTLP pusher if `BSDB_OTLP_ENDPOINT` is set.
///
/// Runs on a dedicated thread using `reqwest::blocking`, so it needs no Tokio
/// runtime and does not interfere with the caller's runtime. Idempotent per
/// process: call once at startup.
pub fn spawn_pusher() {
    let Some(endpoint) = endpoint() else {
        return;
    };
    let interval = push_interval();
    let name = service_name();
    let spawned = std::thread::Builder::new()
        .name("bsdb-otlp-push".to_string())
        .spawn(move || {
            tracing::info!(
                "OTLP metrics push enabled → {endpoint}/v1/metrics (every {:?})",
                interval
            );
            loop {
                let body = build_export_request();
                match push_once(&endpoint, &body) {
                    Ok(code) if (200..300).contains(&code) => {
                        tracing::debug!("OTLP push ok ({code})");
                    }
                    Ok(code) => tracing::warn!("OTLP push returned HTTP {code}"),
                    Err(e) => tracing::warn!("OTLP push failed: {e}"),
                }
                std::thread::sleep(interval);
            }
        });
    match spawned {
        Ok(_) => tracing::info!("OTLP pusher started for service.name={name}"),
        Err(e) => tracing::warn!("failed to spawn OTLP pusher: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_request_has_expected_shape() {
        // Touch a registered metric so the registry is non-empty.
        crate::telemetry::metrics::INGEST_ROWS_TOTAL.inc();
        let body = build_export_request();
        let rm = body["resourceMetrics"].as_array().expect("resourceMetrics");
        assert_eq!(rm.len(), 1);
        let metrics = rm[0]["scopeMetrics"][0]["metrics"]
            .as_array()
            .expect("metrics");
        assert!(
            metrics
                .iter()
                .any(|m| m["name"] == "bsdb_ingest_rows_total"),
            "converted OTLP body is missing a registered counter: {body}"
        );
        // Counters must be cumulative + monotonic sums.
        let counter = metrics
            .iter()
            .find(|m| m["name"] == "bsdb_ingest_rows_total")
            .expect("counter");
        assert_eq!(counter["sum"]["aggregationTemporality"], 2);
        assert_eq!(counter["sum"]["isMonotonic"], true);
    }

    #[test]
    fn histogram_counts_are_differenced_and_include_overflow() {
        use prometheus::core::Collector;

        let hist = prometheus::Histogram::with_opts(
            prometheus::HistogramOpts::new("t_hist", "test").buckets(vec![1.0, 2.0]),
        )
        .unwrap();
        hist.observe(0.5);
        hist.observe(1.5);
        hist.observe(5.0);
        let families = hist.collect();
        let proto_hist = families[0].get_metric()[0].get_histogram();
        let dp = histogram_datapoint(&[], proto_hist, 0);
        // Buckets: <=1 -> 1, <=2 -> 1, +Inf -> 1. Per-bucket counts differenced.
        assert_eq!(dp["count"], "3");
        assert_eq!(dp["explicitBounds"], json!([1.0, 2.0]));
        assert_eq!(dp["bucketCounts"], json!(["1", "1", "1"]));
    }
}
