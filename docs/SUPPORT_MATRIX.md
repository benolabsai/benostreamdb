# Supported Surface & Build Matrix

Status: living document. Owner: BenoStreamDB core.

This document is the authoritative map between **what we ship** (the release
workflow in [`.github/workflows/release.yml`](../.github/workflows/release.yml))
and **what we support**, so documentation, the feature matrix, and limitations
stay aligned with the actual build. It also marks the boundary between the
**supported core** and **experimental / deferred** surfaces.

---

## 1. Platforms and Python ABI

Wheels are ABI3 (`abi3`, built against CPython 3.10) unless noted, produced by
`PyO3/maturin-action`. The stable ABI is forward-compatible, so the same wheels
install on CPython **3.10 through 3.15**; 3.15 is supported but is not yet a CI
smoke-test target (CI runs 3.10–3.12, with ABI3 wheels built on 3.10):

| Platform | Architectures | Notes |
|---|---|---|
| Linux (manylinux, zig) | `x86_64`, `aarch64` | `manylinux: auto`; sdist also published |
| macOS | `x86_64`, `aarch64` | |
| Windows | `x86_64` | `aarch64` not shipped (no aarch64 Python interpreter on the runner) |

Artifacts per release: core wheel, `bsdb-search` gateway wheel, `bsdb-flight`
server wheel, `bsdb-mcp` server wheel, plus a source distribution. The release
gate runs the weekly soak (`soak.yml`) and a lock-down `soak-gate` job before
any tag is published. Writes are byte-reproducible for a given revision because
`SOURCE_DATE_EPOCH` is pinned in the release workflow.

---

## 2. Supported core

These surfaces are covered by CI and are the ones we stand behind:

- **Engine** — Iceberg-format tables (V2/V3 subset), WAL + MVCC commits, indexes
  (HNSW/IVF, TurboQuant variants, inverted/BM25, bitmap, CSR graph), compaction,
  vacuum, time travel, schema evolution, multi-writer OCC.
- **SQL** — the full DataFusion surface plus the engine's UDFs (vector distance,
  pgvector operators, JSON path, graph UDAFs) and the catalog DDL layer
  (`CREATE INDEX`, `ALTER TABLE`, `OPTIMIZE`, `VACUUM`, `ALTER … EXECUTE`).
- **Python** — `benostreamdb.Table` / `Session`, `FlightClient`, `SearchClient`
  (see [`docs/integrations/PYTHON.md`](integrations/PYTHON.md)).
- **Flight SQL** — trusted server-mode surface (see §4).
- **Untrusted connectors** — MCP server ([`contrib/benostreamdb-mcp`](../contrib/benostreamdb-mcp))
  and the OpenSearch/Qdrant-compatible search gateway
  ([`contrib/benostreamdb-search`](../contrib/benostreamdb-search)). Both enforce
  the catalog storage policy.

---

## 3. Experimental / deferred / optional

Not part of the supported-core guarantee:

| Surface | Where | Status |
|---|---|---|
| Spark connector (`spark-benostreamdb`) | subproject | Native (Iceberg-free) path; see [`plans/connector_native_udf_plan.md`](../plans/connector_native_udf_plan.md) |
| Trino connector (`trino-benostreamdb`) | subproject | Native path; typed set-returning table functions still open |
| Flink | — | **No bespoke connector.** Integrate via the Iceberg Flink connector + Flight SQL/JDBC |
| GPU backends (CUDA / ROCm / Metal / Intel) | feature-gated + `benostream-gpu-ann` | Batch-oriented acceleration only; not end-to-end resident GPU search ([`docs/GPU_SETUP_GUIDE.md`](GPU_SETUP_GUIDE.md)) |
| Broker event adapters (Kafka/NATS/RabbitMQ) | planned `contrib/benostreamdb-ingest` | Deferred — see [`docs/ROADMAP.md`](ROADMAP.md) (Theme 6) |
| LangChain / LlamaIndex adapters | planned | Deferred |
| Iceberg `variant` (v3), Delta/Hudi read formats, Flight streaming subscriptions | — | Deferred (see Roadmap themes 7–9) |

Feature gates for the core crate: `python`, `java`, plus the GPU backends. A
default build is CPU-only and Python/JVM-free.

---

## 4. Trust model (must match the deployment)

- **Trusted** (full catalog authority, **never** policy-enforced): the embedded
  core, Arrow Flight SQL, Spark, Trino, dbt. Flight SQL is for unsupported
  languages talking to the core and must **not** be exposed on an untrusted
  network.
- **Untrusted** (catalog-policy enforced): MCP, the OpenSearch/Qdrant-compatible
  Search REST gateway, and any external agent.
  - **Search gateway (server connector)** — schemas are provisioned on the
    backend and the gateway **never creates a schema**. With no policy it is
    read-only (reads allowed; **no** writable schema) and logs a startup warning
    (`BSDB_SEARCH_REQUIRE_POLICY=1` makes it refuse to start instead). With a
    policy, writes are allowed only to schemas listed `read-write` (there is no
    default `scratch` namespace) and reads are the read-only floor / policy
    allowlist. A `no_access_schemas` list denies **both** read and write. This is
    [`SchemaPermissions::for_server_connector`](../src/core/catalog/policy.rs).
  - **MCP (local agent)** — same lists, but a local agent keeps a writable
    `scratch` scratchpad by default.
  Enforcement is identical across MCP and the gateway APIs: the OpenSearch API
  (`_search`, `_count`, `_doc/{id}`, `_mapping`) and the Qdrant API
  (search/query/scroll/retrieve/count/recommend/discover plus every mutating
  route).

See [`docs/CATALOG_USAGE.md`](CATALOG_USAGE.md) and
[`SECURITY.md`](../SECURITY.md).

---

## 5. Known limitations

- **Iceberg deviations** are enumerated in
  [`docs/ICEBERG_COMPATIBILITY.md`](ICEBERG_COMPATIBILITY.md) and the pgvector
  notes in the [`README`](../README.md).
- **The search gateway has no `scratch` write namespace** and never creates a
  schema: writes require a policy grant naming the schema. For MCP (a local
  agent) `scratch` is a single shared schema name, not per-tenant — a scratch
  write grant cannot be used to write outside `scratch`, but there is no
  tenant-level isolation *within* `scratch`.
- **Catalog-policy concurrency**: read-modify-write policy updates use
  optimistic concurrency (CAS); a backend that does not support conditional
  PUTs falls back to last-writer-wins.
- **Benchmarks**: only the full-site Wikipedia Graph-RAG benchmark is stood
  behind; see [`docs/BENCHMARKING.md`](BENCHMARKING.md) for the reproducibility
  contract.

---

## 6. Keeping this current

Update this file in the same PR that changes:

- the release matrix in [`.github/workflows/release.yml`](../.github/workflows/release.yml),
- the set of cargo features, or
- the trusted/untrusted connector boundary.
