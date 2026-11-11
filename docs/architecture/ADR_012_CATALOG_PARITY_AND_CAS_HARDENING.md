# ADR-012: Catalog Parity, Compare-and-Swap Hardening, and the Cross-Catalog Conformance Contract

> **Status: implemented (`v0.16.0`).**
> Implemented across [`src/core/catalog/mod.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/mod.rs), [`common.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/common.rs), [`memory.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/memory.rs), [`rest.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/rest.rs), [`jdbc.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/jdbc.rs), [`glue.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/glue.rs), [`hive.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/hive.rs), [`unity.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/unity.rs), [`nessie.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/nessie.rs), and the Python bindings under [`src/python/catalog/`](https://github.com/benolabsai/benostreamdb/blob/main/src/python/catalog/mod.rs).
> Certified by [`tests/test_catalog_conformance.rs`](https://github.com/benolabsai/benostreamdb/blob/main/tests/test_catalog_conformance.rs) (13 tests: in-process memory, SQLite in-memory and file-backed CAS, optional PostgreSQL, an Axum mock Iceberg REST catalog with OAuth2, an AWS-SigV4 mock for `s3tables` and Glue REST, a Unity mock covering both the Unity Tables and Iceberg REST surfaces, and a Nessie REST v2 mock).
> Promoted from the completed Part II of `plans/iceberg_v3_and_catalog_parity_plan.md`. The Iceberg V3 spec-conformance work from the same plan remains open and stays a live plan.

---

## 1. Context & Operational Motivation

A BenoStreamDB table's identity lives in a catalog, not in its Parquet files. That is what lets a second writer discover the current `metadata_location`, and it is what makes *concurrent* writers safe: two processes can both stage Parquet, but only one may win the metadata pointer update. If a catalog adapter performs an unconditional overwrite, the loser's commit silently erases the winner's snapshot — accepted data disappears with no error.

Comparing the adapter surface against [`apache/iceberg-rust`](https://github.com/apache/iceberg-rust) (`rest`, `glue`, `hms`, `sql`, `s3tables`, `memory`) exposed two problems:

1. **Coverage gaps.** BenoStreamDB had no **AWS S3 Tables** catalog (Iceberg REST over SigV4) and no **in-process `memory`** catalog. The missing memory catalog was an operational cost, not just a feature gap: every multi-writer conformance test needed SQLite files or a live server, so concurrency semantics were under-tested by construction.

2. **Optimistic-locking bugs in adapters already shipped.** `jdbc.rs` emitted hardcoded `?` placeholders (invalid on PostgreSQL), stored `{location}/manifest.json` as the metadata pointer while `commit_table` wrote `metadata/v{snap_id}.metadata.json` (corrupting `location` after the first commit), and had no CAS predicate at all. `glue.rs` never passed `version_id` to `UpdateTable`. `hive.rs` never used `alter_table_with_environment_context`'s `expected_parameter` lock. `unity.rs` PATCHed `{"storage_location": ...}` against the Unity Tables API instead of committing through the Unity Iceberg REST surface. `nessie.rs` ignored explicit `set-metadata-location` updates, so metadata-only commits (`update_schema`, `delete`, `vacuum`) could not advance the pointer.

**The invariant at stake:** every catalog adapter must implement atomic compare-and-swap on `metadata_location` and must surface a lost race as a retryable conflict (`status: 409`), never as a silent overwrite. A "works in a single-writer test" adapter is not a catalog.

---

## 2. Decision

1. **Parity is a requirement, not a nice-to-have.** Ship the two missing catalogs (`s3tables`, `memory`) so the adapter surface matches `iceberg-rust`.
2. **Conflict is a first-class error.** Every adapter maps a lost CAS to `BenoStreamError::CatalogRequest { status: Some(409), .. }`. Callers already retry on 409 (`MAX_CATALOG_RETRIES = 8` in `commit_catalog_step_async`), so the retry loop becomes catalog-agnostic.
3. **Shared semantics live in `common.rs`.** `ensure_storage_table_initialized`, `extract_metadata_location_from_updates`, `load_table_metadata_from_location`, `persist_updates_to_storage`, `table_root_from_metadata_location`, `validate_requirements_against_metadata`, and `catalog_conflict_error` are shared by all adapters instead of reimplemented per backend — which is precisely how the `manifest.json` path divergence happened.
4. **One contract, many backends.** A single conformance suite runs an identical contract against every catalog, including mocked REST surfaces. A new adapter is not "done" until it passes the same suite.

---

## 3. What Shipped

### 3.1 `CatalogType::Memory` — in-process reference catalog — [`memory.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/memory.rs)

Backed by `Arc<parking_lot::RwLock<MemoryCatalogState>>`. Two clients naming the same `memory://<name>` URI share one process-wide state, so concurrent writers observe each other's commits and conflicts. Enforces full `commit_table_with_requirements` semantics (`assert-table-uuid`, `assert-ref-snapshot-id`, `assert-current-schema-id`, `assert-last-assigned-field-id`, `assert-default-spec-id`) and returns `BenoStreamError::CatalogRequest { status: Some(409), .. }` on conflict. Implements `create_namespace`, `create_table`, `drop_table`, `list_tables`, `load_table`, `table_exists`, `commit_table`, and `commit_table_with_requirements`.

The point is not the catalog itself; it is that multi-writer convergence and HTTP-409 semantics are now testable with zero infrastructure and zero network flakiness.

### 3.2 `RestCatalogAuth::AwsSigV4` and `CatalogType::S3Tables` — [`rest.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/rest.rs), [`catalog/mod.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/mod.rs)

`RestCatalogAuth::AwsSigV4 { region, service }` signs `GET`, `POST`, `DELETE`, and `HEAD` requests using credentials already resolved through `aws-config` (`aws-sigv4` was already present transitively via `aws-sdk-glue` and `object_store`, so this added no dependency). `/v1/config?warehouse=<arn_or_name>` discovery populates the canonical `prefix` override, which is what both S3 Tables and Glue's managed Iceberg REST endpoint require.

`CatalogType::S3Tables` (aliases `s3_tables`, `s3-tables`) accepts a table-bucket ARN as `warehouse`, derives the region from the ARN (or AWS defaults), allows an explicit `url` override, routes through `RestCatalogClient` with `service: "s3tables"`, and honors `BSDB_SSRF_GUARD=1`. Glue gains an optional `mode = "rest"` for its managed Iceberg REST endpoint (`service: "glue"`), complementing the native `aws-sdk-glue` client.

### 3.3 JDBC dialect, canonical paths, and CAS — [`jdbc.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/jdbc.rs)

- **Placeholders:** dialect detection from the URI (`postgres://` / `postgresql://` → `$1, $2, …`; `sqlite://` / `mysql://` → `?`).
- **Metadata path:** `create_table` stores `{table_location}/metadata/v0.metadata.json` after initializing metadata through `Table::create_async`; `load_table` strips `/metadata/v…` consistently (with a `/manifest.json` fallback for pre-existing rows).
- **CAS:** explicit `set-metadata-location` is checked first, falling back to snapshot derivation; SQL `TableRequirement`s are validated against the persisted `TableMetadata`; the update executes as

  ```sql
  UPDATE iceberg_tables
     SET previous_metadata_location = metadata_location,
         metadata_location = ?
   WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?
     AND (metadata_location = ? OR (? IS NULL AND metadata_location IS NULL))
  ```

  `rows_affected() == 0` returns `CatalogRequest { catalog: "jdbc", status: Some(409), reason: "CAS conflict" }`.
- **Coverage:** `drop_table` and `list_tables` implemented.

This last point is the one that matters most in production: before the fix, `rows_affected() == 0` was not checked at all, so a concurrent commit was overwritten without any signal.

### 3.4 Glue `version_id` conditional commit — [`glue.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/glue.rs)

`create_table` now sets `metadata_location` to the initial metadata JSON rather than the bare table location, and `load_table` resolves the table root from a `metadata/vN.metadata.json` pointer. Commits read `existing_table.version_id()`, requirements are validated before `UpdateTable`, and `expected_version_id` is passed so Glue rejects the stale write; `ConcurrentModificationException` maps to `409`. `delete_table` and paginated `get_tables` implemented.

### 3.5 Hive `expected_parameter` CAS — [`hive.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/hive.rs)

`commit_table` reads explicit `set-metadata-location` updates and uses `alter_table_with_environment_context` with `expected_parameter_key = "metadata_location"` and `expected_parameter_value = <prior>`, mapping the metastore's conflict exception to `409`. `get_all_tables` implemented.

### 3.6 Unity Iceberg REST mode — [`unity.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/unity.rs)

Databricks Unity Catalog is reached through `/api/2.1/unity-catalog/iceberg-rest/v1/...` — the standard Iceberg REST protocol with a Bearer PAT, which gets `requirements` CAS and `409 Conflict` semantics for free. The legacy Unity Tables endpoint is retained as a compatibility path with explicit conflict verification where the REST surface is unavailable. `drop_table` and `list_tables` implemented.

### 3.7 Nessie metadata-location commits — [`nessie.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/nessie.rs)

`commit_table_with_requirements` now resolves the pointer through `extract_metadata_location_from_updates` first, then falls back to `persist_updates_to_storage(&table_root, …)`, then to the existing pointer. That is what makes metadata-only commits (`update_schema`, `delete`, `vacuum`) advance the Nessie `ICEBERG_TABLE` content cleanly instead of requiring an `add-snapshot`. `expectedHash` OCC is preserved, `409 ReferenceConflictException` maps to `409`, and `getEntries` is implemented.

### 3.8 Python bindings — [`src/python/catalog/`](https://github.com/benolabsai/benostreamdb/blob/main/src/python/catalog/mod.rs)

`PyMemoryCatalog` and `PyS3TablesCatalog` are registered on the module. `create_catalog("memory", {"uri": "memory://dev_catalog", "warehouse": "file:///tmp/bsdb_wh"})` works with no configuration file, which is what makes catalog-touching notebook examples runnable without cloud credentials.

---

## 4. The Cross-Catalog Conformance Contract

[`tests/test_catalog_conformance.rs`](https://github.com/benolabsai/benostreamdb/blob/main/tests/test_catalog_conformance.rs) runs `run_catalog_contract(...)` against each backend. The contract is namespace create/exists/drop, table create/exists/load, `commit_table_with_requirements` with a stale requirement (must conflict), list, and drop, with the conflict asserted structurally via `is_catalog_conflict` rather than by string matching.

| Test | Backend | Gate |
|---|---|---|
| `memory_catalog_conformance` | `memory` | always on |
| `memory_catalog_concurrent_writers_409_convergence` | `memory` | always on |
| `jdbc_sqlite_memory_conformance` | `jdbc` SQLite `:memory:` | always on |
| `jdbc_sqlite_file_conformance_and_concurrent_cas` | `jdbc` SQLite file | always on |
| `jdbc_postgres_conformance_if_configured` | `jdbc` PostgreSQL | `DATABASE_URL` |
| `rest_catalog_mock_oauth2_and_conformance` | `rest` + OAuth2 | always on (Axum mock) |
| `s3tables_sigv4_rest_mock_conformance` | `s3tables` | always on (SigV4 mock) |
| `glue_rest_sigv4_mock_conformance` | `glue` REST mode | always on (SigV4 mock) |
| `unity_standard_and_iceberg_rest_conformance` | `unity` | always on (Unity mock) |
| `nessie_rest_v2_mock_conformance` | `nessie` | always on (Nessie v2 mock) |
| `nessie_catalog_conformance_if_configured` | `nessie` live | `NESSIE_URI` |
| `sql_ddl_external_catalog_lifecycle_memory_and_jdbc` | SQL DDL over `memory` / `jdbc` | always on |
| `search_gateway_precreated_storage_registration_is_idempotent` | search gateway | always on |

The mocks are per-surface and deliberately keep separate authorization predicates: the generic Iceberg REST mock accepts `Bearer mock-token-…`, the SigV4 mocks verify `x-amz-content-sha256` canonicalization, and the Unity mock accepts only a PAT. Sharing a load handler across the Unity Iceberg REST sub-app was an actual bug found and fixed here — it accepted the REST mock's bearer tokens and rejected PATs, so every Unity load returned 401.

CI wiring: the dead `--ignore=tests/integration/test_catalog.py` flag (no such file exists) was removed from [`.github/workflows/ci.yml`](https://github.com/benolabsai/benostreamdb/blob/main/.github/workflows/ci.yml), and [`docs/CATALOG_USAGE.md`](https://www.benostreamdb.org/guides/catalog_usage.html) and [`docs/SUPPORT_MATRIX.md`](https://www.benostreamdb.org/guides/support_matrix.html) now carry the accurate CAS guarantees plus `s3tables` and `memory` configuration examples.

---

## 5. Consequences and Trade-offs

- **`CatalogType` gained two variants.** `CatalogType` is a plain `pub enum` in a `pub mod` with no `#[non_exhaustive]`, so downstream exhaustive `match` arms over it will not compile against this version. That is the one genuinely breaking surface in this change set and is why the ADR ships in a minor release rather than a patch. Making the enum `#[non_exhaustive]` now would be the same break, so it was deliberately deferred rather than smuggled in.
- **Conflict handling is uniform, retry policy is not.** Adapters surface 409; whether to retry remains the caller's decision, bounded by `MAX_CATALOG_RETRIES = 8`. A catalog that returns 409 spuriously will now cost eight retries instead of appearing to succeed.
- **The memory catalog is process-local.** Two OS processes naming the same `memory://` URI do not share state. It is a test and single-process fixture, not a coordination primitive, and must never be presented as one.
- **SigV4 signing is verified against mocks, not live S3 Tables.** Canonical-request construction, header signing, and `/v1/config` prefix discovery are asserted by `s3tables_sigv4_rest_mock_conformance`; a live AWS round-trip remains a credential-gated path.
- **Shared `common.rs` is now load-bearing.** Path resolution, requirement validation, and conflict construction are single-implementation. That removes the divergence class that produced the `manifest.json` bug, at the cost of the shared helpers becoming a change-amplification point for every adapter.
