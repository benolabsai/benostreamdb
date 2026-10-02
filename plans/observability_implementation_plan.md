# Observability Implementation Plan

> **Status: implemented.** The core owns the registry
> ([`gather_text`](../src/telemetry/metrics.rs:313) +
> [`render_metrics`](../src/core/telemetry.rs:50)); the `observability` exporter
> is configurable (`BSDB_METRICS_BIND`/`BSDB_METRICS_PORT`, localhost default);
> Flight SQL serves `/metrics`, `/health`, `/readyz`; `bsdb-search` merges the
> core registry; and the JNI `gatherMetrics()` bridge is in place for the Spark
> and Trino connectors. The env-var prefix was unified to `BSDB_` /
> `BSDB_SEARCH_` / `BSDB_QDRANT_` (hard rename, no aliases).
>
> **OTLP push + Grafana are now implemented** (the follow-ups previously listed
> as out of scope): [`src/telemetry/otlp.rs`](../src/telemetry/otlp.rs) converts
> the Prometheus registry to OTLP/HTTP JSON and pushes it
> (`BSDB_OTLP_ENDPOINT`, `BSDB_OTLP_INTERVAL_SECS`, `BSDB_OTLP_HEADERS`,
> `BSDB_SERVICE_NAME`); [`deploy/observability/`](../deploy/observability/README.md)
> brings up a collector + Prometheus + a provisioned Grafana dashboard.

Implements the recommendations from the observability review. Goal: one metrics
source of truth in the core, exposed by every host — Flight SQL as the canonical
server-mode surface, `bsdb-search` merged, and a JNI bridge for the JVM
connectors.

## Design decisions

1. **The core owns the registry.** All engine metrics are recorded with the
   `prometheus` crate into the default registry in
   [`src/telemetry/metrics.rs`](../src/telemetry/metrics.rs:32). Do not move
   ownership into any server.
2. **One renderer.** Add `gather_text()` to the core and have every host call it.
3. **Fix the `observability` exporter.** It currently installs a
   `metrics-exporter-prometheus` recorder (the `metrics` facade), but the core
   records with the `prometheus` crate directly — so the exporter exposes
   **nothing**. Replace it with a small axum server that serves
   `gather_text()`.
4. **Flight SQL is the reference server-mode surface.** Add `/metrics`,
   `/health`, `/readyz`, tracing init, and a configurable bind/port defaulting
   to `127.0.0.1`.
5. **JVM connectors bridge, not listen.** Add a JNI `gatherMetrics()` and
   register into the host metrics system (Spark `Source` / Trino JMX).

## Work items

### 1. Core metrics renderer — `src/telemetry/metrics.rs`

Add a public accessor:

```rust
/// Render every registered metric in Prometheus text format (version 0.0.4).
pub fn gather_text() -> String {
    use prometheus::TextEncoder;
    let encoder = TextEncoder::new();
    let mut out = String::new();
    encoder
        .encode_utf8(&prometheus::gather(), &mut out)
        .unwrap_or_default();
    out
}
```

### 2. Configurable, working exporter — `src/core/telemetry.rs`

Replace the `metrics-exporter-prometheus` listener with an axum server serving
the core registry. Read `BSDB_METRICS_BIND` (default `127.0.0.1`) and
`BSDB_METRICS_PORT` (default `9090`). Keep the `observability` feature gate.
Spawn the server on a background task so `init_metrics_exporter()` returns.

- Verify whether anything records via the `metrics` facade; if so, keep that
  exporter too, otherwise drop the `metrics-exporter-prometheus` dependency.

### 3. Flight SQL observability — `server/flight_sql`

- `Cargo.toml`: add `axum`, `tracing`, `tracing-subscriber` (or reuse the core's
  `init_tracing`).
- `src/main.rs`:
  - call `benostreamdb::telemetry::tracing::init_tracing("flight_sql")` and hold
    the guard;
  - spawn an axum server on `BSDB_METRICS_BIND:BSDB_METRICS_PORT`
    (default `127.0.0.1:9090`) with:
    - `GET /metrics` → `benostreamdb::telemetry::metrics::gather_text()`
      (`text/plain; version=0.0.4`);
    - `GET /health` → `200 ok` (liveness);
    - `GET /readyz` → readiness: storage reachable / warehouse configured;
      `503` otherwise.
- Keep the gRPC listener on `50051`; the HTTP listener is separate.

### 4. Merge core registry into `bsdb-search` — `contrib/benostreamdb-search`

- `src/handlers/metrics.rs`: append
  `benostreamdb::telemetry::metrics::gather_text()` to
  `state.metrics.gather_text()`. Namespaces do not collide
  (`bsdb_search_*` vs `benostreamdb_*`).
- Add a test asserting a core metric name appears in the `/metrics` body.

### 5. JNI bridge — `src/core/ffi.rs`

Add two entry points returning the Prometheus text:

```rust
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_gatherMetrics(
    mut env: JNIEnv, _class: JClass,
) -> jstring {
    match env.new_string(crate::telemetry::metrics::gather_text()) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}
```

and the Trino equivalent
(`Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_gatherMetrics`).

Declare the native methods in the connectors:

- `spark-benostreamdb/.../jni/BenoStreamJNIBridge.scala`:
  `@native def gatherMetrics(): String`
- `trino-benostreamdb/.../BenoStreamDBJNIBridge.java`:
  `public static native String gatherMetrics();`

Follow-up (optional, larger): register the parsed metrics into Spark's
`MetricsSystem` / Trino's JMX so the host's existing scrape picks them up.

### 6. Verification

- `cargo build -p benostreamdb --features observability`
- `cargo build -p benostreamdb-flight`
- `cargo build -p benostreamdb-search`
- `cargo build --features java`
- `cargo test -p benostreamdb-flight` and the search metrics test.
- Manual: `curl localhost:9090/metrics` on Flight SQL; confirm
  `benostreamdb_*` and `bsdb_search_*` both appear on `bsdb-search`.

### 7. Documentation

- `docs/monitoring.md`: document the endpoints per deployment mode, the
  `BSDB_METRICS_BIND`/`BSDB_METRICS_PORT` knobs, and the full metric catalog
  (`benostreamdb_*` and `bsdb_search_*`) with what a rising value means.
- `docs/CONFIGURATION.md`: add the two metrics env vars.
- `docs/RESOURCE_LIMITS.md`: cross-reference the metrics endpoint.

## Out of scope

- Alert rules — separate follow-up (the dashboard is delivered).
- Per-tenant labels and per-query bytes-scanned — separate follow-up.
