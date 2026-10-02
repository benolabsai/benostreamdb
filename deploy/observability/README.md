# Observability stack

OTLP collector + Prometheus + Grafana for BenoStreamDB. Two ways to get metrics
in, both supported simultaneously:

1. **OTLP push** — the engine converts its Prometheus registry to OTLP/HTTP JSON
   and pushes every `BSDB_OTLP_INTERVAL_SECS` (default 15) to
   `BSDB_OTLP_ENDPOINT`. See `src/telemetry/otlp.rs`.
2. **Pull scrape** — every host serves `/metrics` (Prometheus text). Set
   `BSDB_METRICS_BIND=0.0.0.0` so Prometheus can reach it.

## Run

```bash
docker compose -f deploy/observability/docker-compose.observability.yml up -d
```

| Service | URL | Notes |
|---|---|---|
| Grafana | http://localhost:3000 | `admin` / `admin`; dashboard "BenoStreamDB" provisioned |
| Prometheus | http://localhost:9091 | scrapes the collector (`:8889`) and the host `:9090` |
| OTLP/HTTP | http://localhost:4318 | set `BSDB_OTLP_ENDPOINT` to this |
| OTLP/gRPC | localhost:4317 | |
| Collector metrics | http://localhost:8889/metrics | Prometheus-format re-export |

### Point the engine at it

```bash
export BSDB_OTLP_ENDPOINT=http://localhost:4318
export BSDB_SERVICE_NAME=benostreamdb        # resource service.name
export BSDB_OTLP_INTERVAL_SECS=15
# optional auth/multi-tenant headers:
# export BSDB_OTLP_HEADERS="authorization=Bearer <token>,x-tenant=acme"

# For pull scraping as well:
export BSDB_METRICS_BIND=0.0.0.0
export BSDB_METRICS_PORT=9090
```

The Flight SQL server, the search gateway, and the JVM connectors (via the JNI
metrics bridge) all expose the same registry, so one pipeline covers every host.

## Dashboard

`grafana/dashboards/benostreamdb.json` panels (all `bsdb_*` metrics):

- Ingest rows/sec, total rows, ingest RSS, back-pressure pauses/sec
- Vector search latency p50/p99 (`bsdb_search_latency_seconds`)
- Query latency p99 (`bsdb_query_latency_seconds`)
- Cache hit ratio (`bsdb_cache_hits_total` / `bsdb_cache_misses_total`)
- Commit and compaction latency p99 (`bsdb_commit_duration_seconds`,
  `bsdb_compaction_duration_seconds`)
- Active files/segments, manifest conflicts/sec
- I/O bytes/sec (read/write), index-build gate wait p99

## Env reference

| Variable | Default | Purpose |
|---|---|---|
| `BSDB_OTLP_ENDPOINT` | (unset) | enables OTLP push; e.g. `http://localhost:4318` |
| `BSDB_OTLP_INTERVAL_SECS` | `15` | push period |
| `BSDB_OTLP_HEADERS` | (unset) | comma-separated `key=value` headers |
| `BSDB_SERVICE_NAME` | `benostreamdb` | OTLP `service.name` |
| `BSDB_METRICS_BIND` | `127.0.0.1` | `/metrics` bind address |
| `BSDB_METRICS_PORT` | `9090` | `/metrics` port |

## Notes

- OTLP push is dependency-light: it reuses `reqwest`/`serde_json` and converts
  the existing `prometheus` registry, so no re-instrumentation was needed.
- Histogram buckets are converted from Prometheus' cumulative form to OTLP's
  per-bucket `bucketCounts` + `explicitBounds` (with the `+Inf` overflow bucket
  appended); counters export as cumulative, monotonic sums.
- Without an OTLP endpoint the pusher is a no-op; pull scraping is unaffected.
