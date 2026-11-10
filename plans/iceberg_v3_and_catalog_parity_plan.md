# BenoStreamDB — Iceberg V3 Spec Conformance & Catalog Parity Plan

**Status (2026-11-10, target release: post-v0.15.2 / v0.16.x): `V3.1` docs honesty completed; `V3.2–V3.4` and `C1–C4` proposed.**  
**Companion plan:** [production_readiness_plan.md](production_readiness_plan.md) (WS3/WS4 exit gates).

---

## 0. Why This Work Matters (Architecture & Operational Context)

BenoStreamDB commits data using open Apache Iceberg metadata (`v{N}.metadata.json`, Avro `snap-*.avro` manifest lists, and Avro `*-m*.avro` data manifests) over Parquet segments and Puffin overlay sidecars. For engineers and platform teams choosing an indexed lakehouse engine, open-table portability is only real if external engines (Apache Spark, Trino, Snowflake, DuckDB, and `apache/iceberg-rust`) read and evolve BenoStreamDB tables without translation layers or data duplication.

Two audit gaps remain open (**Workstream V3.1** is done; Workstreams **V3.2–V3.4** and **C1–C4** below are not yet implemented):

1. **Iceberg V3 validation is asymmetric and partial:** [`test_bsdb_to_pyiceberg.py`](https://github.com/benolabsai/benostreamdb/blob/main/tests/integration/test_bsdb_to_pyiceberg.py#L1) validates against **PyIceberg 0.12.0**, which targets Iceberg V2. Meanwhile, BenoStreamDB writes V3 row-lineage metadata (`next-row-id`, `first-row-id`, `added-rows`), but its Deletion Vector (DV) path is incomplete on write and unwired on read, and `attach_manifest_metadata` hardcodes `format-version = "2"`.
2. **Catalog parity and multi-writer CAS hardening lag the REST client:** Compared against the [`apache/iceberg-rust`](https://github.com/apache/iceberg-rust) catalog surface (`rest`, `glue`, `hms`, `sql`, `s3tables`, `memory`), BenoStreamDB lacks **AWS S3 Tables** (`SigV4`-signed REST) and an **in-process `memory` catalog**, while several existing adapters (`jdbc.rs`, `glue.rs`, `hive.rs`, `unity.rs`, `nessie.rs`) contain path, placeholder, or missing optimistic-locking bugs under concurrent writers.

### Already Resolved Baseline (`v0.15.1` + Staged `M2`)

The following prerequisites are **already resolved in source** ahead of this plan:
- **V3 row-lineage strict OCC (`format_version = 3`):** Contiguous `_row_id` and `_last_updated_sequence_number` assignment, `next-row-id` propagation, and rejected Parquet cleanup on `ManifestVersionMismatch` are implemented and verified across 6 concurrent writers in [`test_m2_commit_state_machine.rs`](https://github.com/benolabsai/benostreamdb/blob/main/tests/test_m2_commit_state_machine.rs) (Pillar 7) and [`test_multi_writer_differential.rs`](https://github.com/benolabsai/benostreamdb/blob/main/tests/test_multi_writer_differential.rs).
- **V3 nanosecond schema types:** `timestamp_ns` and `timestamptz_ns` are mapped in [`src/core/manifest/types.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/manifest/types.rs#L1423-L1447) with nanosecond `ManifestValue` bounds decode.
- **REST catalog CAS & requirement validation:** `Catalog::commit_table_with_requirements`, `RestCatalogClient` HTTP 409 propagation, `commit_catalog_step_async` retry (`MAX_CATALOG_RETRIES = 8`), and server-side `TableRequirement` validation in [`src/bin/iceberg_rest.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/bin/iceberg_rest.rs#L547-L654) are shipped and certified by `pillar1_axum_rest_catalog_409_conflict_requirements_and_diverged_location`.

This plan specifies exact remaining code fixes, independent test oracles, and acceptance criteria for both tracks.

---

## Part I — Apache Iceberg V3 Conformance & Independent Oracle Strategy

### 1. Replacing PyIceberg with `apache/iceberg-rust` as the V3 Reader Oracle

PyIceberg validates V2 metadata, manifest lists, and Parquet schema field IDs well, and it should remain in Python integration tests for Python-ecosystem parity. However, relying solely on PyIceberg for V3 leaves row lineage (`first_row_id` propagation across manifests and snapshots), V3 Deletion Vectors (`deletion-vector-v1` inside Puffin blobs), and V3 type additions unverified by an independent parser.

Using **`apache/iceberg-rust`** (`iceberg` crate, alongside Java `iceberg-core` for Spark/Trino writes) resolves this cleanly:

| Verification Role | Engine / Tool | What It Validates | Why Not PyIceberg Alone? |
|---|---|---|---|
| **In-CI Native Rust Reader Oracle** | `apache/iceberg-rust` (`iceberg`) in an isolated conformance crate | Independent parsing of V2/V3 `TableMetadata`, Avro manifests, `first_row_id` row lineage, Puffin V3 Deletion Vectors (`deletion-vector-v1`), and arrow-rs scan outputs via `cargo test` without Python/JVM startup overhead | PyIceberg is V2-centric and requires Python venv wheel builds |
| **Bi-Directional Engine Oracle** | Spark 4.0 (`iceberg-spark-runtime`) & Trino Iceberg connector | External **writes**, `DELETE`/`MERGE`/`UPDATE`, schema/partition evolution, and V3 DV creation read back by BenoStreamDB (`spark → bsdb → spark` and `trino → bsdb → trino`) | `iceberg-rust` focuses primarily on catalog/table reading and Iceberg-native writes; JVM engines remain the reference writers for multi-engine mutations |
| **Python Ecosystem Interop** | `pyiceberg` (`tests/integration/*.py`) | Embedded Python user workflows (`pyiceberg.catalog.load_table` on V2 tables) | Retain for Python compatibility, do not treat as V3 certification |

#### Dependency Isolation Decision (Avoiding `arrow` / `object_store` Diamond Conflicts)

[`Cargo.toml`](https://github.com/benolabsai/benostreamdb/blob/main/Cargo.toml#L85-L119) pins `arrow = "57.2.0"`, `datafusion = "52.0.0"`, and `object_store = "0.12.4"`, audited under [`deny.toml`](https://github.com/benolabsai/benostreamdb/blob/main/deny.toml). Upstream `iceberg` crate releases often pin a different `arrow`, `parquet`, or `object_store` major/minor version. Pulling `iceberg` directly into the main workspace `Cargo.toml` would bloat production binaries (`benostreamdb`, `bsdb-search`, `benostreamdb-flight`), risk `cargo deny` license/advisory churn, and trigger unified-resolver conflicts.

**Selected architecture:** Place the `apache/iceberg-rust` conformance harness in an **excluded standalone Cargo package** (e.g., `tests/iceberg_rust_oracle/Cargo.toml` omitted from `[workspace].members`) communicating strictly through on-disk/object-store Iceberg table artifacts and shared temp directories. Both crates compile cleanly against their own `Cargo.lock` graphs while reading/writing byte-exact table trees.

---

### 2. Known Iceberg V3 Gaps in BenoStreamDB (Verified in Source)

Inspection of the current writer, reader, Puffin, and table maintenance paths reveals five concrete conformance gaps that must be closed before claiming V3 write parity:

1. **Manifest Avro header hardcodes V2 metadata ([`src/core/iceberg/writer.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/writer.rs#L54-L70)):**
   `attach_manifest_metadata` writes `writer.add_user_metadata("format-version".to_string(), "2")?` unconditionally, even when writing manifests for a `format_version = 3` table. External V3 readers inspect the Avro `format-version` key to select V3 manifest parsing rules.
2. **Manifest Avro `r2` schema omits V3 Deletion Vector fields ([`src/core/iceberg/writer.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/writer.rs#L965-L1007)):**
   [`src/core/iceberg/manifest.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/manifest.rs#L248-L278) already parses `referenced_data_file`, `content_offset`, and `content_size_in_bytes` on read, but `IcebergWriter::generate_manifest_schema` omits those three fields (`field-id` `143`, `144`, `145`) from the `r2` Avro schema.
3. **Manifest `content` / `file_format` encoding for V3 Deletion Vectors ([`src/core/iceberg/writer.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/writer.rs#L134-L160)):**
   - Per the Apache Iceberg V3 specification, Deletion Vectors are manifest delete entries with `content = 1` (`POSITION_DELETES`), `file_format = "PUFFIN"`, non-null `referenced_data_file` (`143`), `content_offset` (`144`), and `content_size_in_bytes` (`145`).
   - Currently, `writer.rs` maps `DeleteContent::DeletionVector` to `content_id = 3` (matching a custom check in [`manifest.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/manifest.rs#L483-L499)) and sets `delete_format = "AVRO"` for any path not ending in `.parquet` — mislabeling `.puffin` files as Avro.
4. **Puffin DV blob type and binary framing ([`src/core/puffin.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/puffin.rs#L678-L703)):**
   - `DELETION_VECTOR_TYPE` is defined as `"apache-datasketches-theta-v1"` instead of the Iceberg V3 spec identifier `"deletion-vector-v1"`.
   - `read_deletion_vector_from_puffin` and `read_deletion_vector_from_bytes` deserialize raw 32-bit `roaring::RoaringBitmap::deserialize_from` payloads, whereas Iceberg V3 defines the Puffin DV payload envelope as:
     * `4-byte big-endian length` (size in bytes of magic + bitmap, excluding length & CRC bytes),
     * `4-byte magic sequence` `[0xD1, 0xD3, 0x39, 0x64]` (big-endian `0xD1D33964`, equivalent to little-endian `1681511377` / `0x6439D3D1`),
     * **portable 64-bit roaring bitmap** (little-endian container serialization / `RoaringTreemap`),
     * `4-byte big-endian CRC-32` over magic + bitmap payload.
   - Neither function currently has a caller inside `HybridReader` (`src/core/reader/`), so **V3 Deletion Vectors are neither written nor applied on read**.
5. **Delete writer & staged writes on V3 ([`src/core/table/maintenance.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/maintenance.rs#L323-L345) & [`src/core/table/stage.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/stage.rs#L37-L73)):**
   - `Table::delete_async` hardcodes `IcebergDeleteWriter::new(self.uri.clone(), 2, self.store.clone())`, emitting V2 position-delete Parquet files on V3 tables instead of Puffin DVs.
   - `Table::write_mode()` returns `"buffered"` for `format_version >= 3` because distributed Spark/Trino task staging (`stage_data_files_async`) cannot reserve disjoint `_row_id` ranges without driver-side range allocation. If high-volume V3 distributed writes are required later, driver-coordinated `_row_id` block pre-reservation will be needed; until then, `"buffered"` remains the explicit documented contract.

---

### 3. Workstreams for Iceberg V3 Execution

#### Workstream V3.1 — Documentation Honesty Gate (**Done**, 2026-11-10)
- Updated [`docs/ICEBERG_COMPATIBILITY.md`](https://github.com/benolabsai/benostreamdb/blob/main/docs/ICEBERG_COMPATIBILITY.md#L45) and [`docs/ICEBERG_STORAGE.md`](https://github.com/benolabsai/benostreamdb/blob/main/docs/ICEBERG_STORAGE.md#L200) (plus [`docs/PRODUCTION_READINESS_ASSESSMENT.md`](https://github.com/benolabsai/benostreamdb/blob/main/docs/PRODUCTION_READINESS_ASSESSMENT.md#L21)) so Deletion Vectors are accurately listed as **Planned / Partial** (V2 position/equality deletes are used on write today; Puffin `deletion-vector-v1` write/read wiring and `apache/iceberg-rust` verification remain open under Workstreams V3.2–V3.4).

#### Workstream V3.2 — Manifest Schema, Metadata, & Spec-Compliant DV Codec
- Parameterize `IcebergWriter` (`attach_manifest_metadata`, `write_manifest_list`, `write_manifest_chunks`, `write_delete_manifest`) by `format_version: u8` (`2` or `3`):
  * Write `"format-version" -> format_version.to_string()` into Avro manifest metadata.
  * Add optional fields `143` (`referenced_data_file`), `144` (`content_offset`), and `145` (`content_size_in_bytes`) to `generate_manifest_schema` `r2`.
  * Emit `file_format = "PUFFIN"` for `DeleteContent::DeletionVector` and adopt standard Iceberg V3 `content = 1` + `file_format = "PUFFIN"` (retaining backward compatibility on read when `content == 3` with valid DV fields).
- Update [`src/core/puffin.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/puffin.rs#L678-L703):
  * Change `DELETION_VECTOR_TYPE` to `"deletion-vector-v1"`.
  * Implement spec-exact `write_deletion_vector_to_puffin` / `encode_deletion_vector_v1` / `decode_deletion_vector_v1` framing (`4B BE` length, `4B` magic `[0xD1, 0xD3, 0x39, 0x64]` [BE `0xD1D33964` / LE `1681511377`], little-endian 64-bit portable roaring bitmap, `4B BE` CRC-32 with CRC validation returning `BenoStreamError::DeletionVectorDeserialize`).
  * Ensure Puffin DV blobs share the Puffin footer property `referenced-data-file` / `cardinality` per spec, coexist cleanly with BenoStreamDB's compound index Puffin files, or reside in dedicated `del-dv-<uuid>.puffin` files written through `self.store`.

#### Workstream V3.3 — Read/Write/Compact Path Integration for V3 DVs
- Wire `read_deletion_vector_from_bytes` (with byte-range S3 read using `content_offset..content_offset + content_size_in_bytes` + CRC check) into `HybridReader` / `PositionDeleteReader` / [`src/core/manifest/manager/load.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/manifest/manager/load.rs#L238) so `DeleteContent::DeletionVector` masks deleted positions identically to Parquet position deletes and integrates into [`compact_bin`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/compaction.rs#L284).
- Wire `Table::delete_async` and `MergePlanner` so `format_version >= 3` writes V3 Puffin DVs (and merges prior DVs on the same `referenced_data_file` into a single Puffin blob per file, as required by Iceberg V3's *at-most-one DV per data file per snapshot* invariant).
- Wire `vacuum` and `remove_orphan_files_async` in [`src/core/manifest/manager/commit.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/manifest/manager/commit.rs#L504) so DV `.puffin` files referenced in `Manifest::delete_files` are retained and expired DVs are reaped safely.

#### Workstream V3.4 — `apache/iceberg-rust` & Multi-Engine Verification Matrix
- Add isolated crate `tests/iceberg_rust_oracle/` depending on `iceberg`:
  * **V2 suite:** BenoStreamDB writes/deletes/compacts/evolves schema+partition → `iceberg-rust` opens table, scans via Arrow, asserts exact row values, schema field IDs, and bounds.
  * **V3 suite:** BenoStreamDB creates `format_version = 3` table → appends multiple commits (checking `first_row_id` and `next_row_id`) → executes `delete_async` producing `deletion-vector-v1` Puffin blobs → `iceberg-rust` opens snapshot and reads exact surviving rows → BenoStreamDB compacts → `iceberg-rust` verifies clean state and preserved `_row_id` lineage.
- Add Java interop matrix (CI optional / nightly workflow alongside `ecosystem`): Spark 4.0 and Trino write V2 position deletes and V3 DVs → BenoStreamDB reads; BenoStreamDB mutates → Spark/Trino read.

---

## Part II — Catalog Parity & Hardening Plan (`apache/iceberg-rust` Alignment)

### 1. Catalog Parity Matrix

Comparing `CatalogType` in [`src/core/catalog/mod.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/mod.rs#L96-L248) against [`apache/iceberg-rust`](https://github.com/apache/iceberg-rust):

| `iceberg-rust` Crate | BenoStreamDB Counterpart | Protocol / Auth | Current State & Gaps |
|---|---|---|---|
| `iceberg-catalog-rest` | `CatalogType::Rest` ([`rest.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/rest.rs)) | Iceberg REST v1 + Bearer + OAuth2 | Strongest adapter: implements `commit_table_with_requirements` (409 conflict propagation), `drop_table`, `list_tables`. **Missing:** AWS SigV4 request signing, `/v1/config` warehouse-prefix auto-discovery, `namespace_exists` / `drop_namespace`. |
| `iceberg-catalog-s3tables` | *(None)* | Iceberg REST over AWS SigV4 (`s3tables` endpoint) | **Missing.** Can be implemented cleanly as `CatalogType::S3Tables` backed by `RestCatalogClient` with SigV4 signing against `https://s3tables.<region>.amazonaws.com/iceberg/v1` and ARN warehouse prefix, or dedicated `aws-sdk-s3tables`. |
| `iceberg-catalog-glue` | `CatalogType::Glue` ([`glue.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/glue.rs)) | AWS SDK `aws-sdk-glue` | **Hardening needed:** `create_table` sets `metadata_location` to the table root instead of `metadata/v{N}.metadata.json`; `commit_table` performs unconditional `update_table` without `version_id` optimistic locking (`ConcurrentModificationException`); `drop_table`/`list_tables` unimplemented. Also consider Glue Iceberg REST endpoint option via SigV4. |
| `iceberg-catalog-hms` | `CatalogType::Hive` ([`hive.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/hive.rs)) | Apache Hive Metastore Thrift | **Hardening needed:** `create_table` stores bare location as `metadata_location`; `commit_table` does not inspect explicit `set-metadata-location` updates and lacks HMS CAS / `expected_parameter` locking (`alter_table_with_environment_context`); `drop_table`/`list_tables` unimplemented. |
| `iceberg-catalog-sql` | `CatalogType::Jdbc` ([`jdbc.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/jdbc.rs)) | `sqlx::AnyPool` (SQLite / Postgres) | **Critical bugs:** hardcoded `?` SQL bind placeholders fail on Postgres (`$1`); `create_table` writes `{loc}/manifest.json` (`load_table` only strips `/manifest.json`, corrupting `location` after first `commit_table` which writes `metadata/v{snap_id}.metadata.json`); `commit_table` ignores `set-metadata-location` and lacks `WHERE metadata_location = ?` optimistic CAS check. |
| `iceberg-catalog-memory` | *(None)* | In-process `RwLock<HashMap<...>>` | **Missing.** Needed for zero-infrastructure deterministic unit tests, differential SQL tests, and multi-writer requirement CAS testing without external services or SQLite files. |
| *(N/A in `iceberg-rust` core)* | `CatalogType::Nessie` ([`nessie.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/nessie.rs)) | Nessie REST v2 | Preserves Git-like branch semantics (`expectedHash` OCC). Needs `set-metadata-location` handling plus `drop_table` / `list_tables`. |
| *(N/A in `iceberg-rust` core)* | `CatalogType::Unity` ([`unity.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/unity.rs)) | Databricks Unity Catalog REST | `commit_table` currently PATCHes `{"storage_location": ...}` via `/api/2.1/unity-catalog/tables/...` instead of Databricks Iceberg REST (`/api/2.1/unity-catalog/iceberg-rest/v1/...`) atomic commit semantics; no CAS check; `drop_table`/`list_tables` unimplemented. |

---

### 2. Catalog Implementation & Hardening Workstreams

#### Workstream C1 — AWS SigV4 REST Auth & `CatalogType::S3Tables`
1. Add `RestCatalogAuth::AwsSigV4 { region: String, service: String }` to [`src/core/catalog/rest.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/rest.rs#L17-L27).
   - Sign outbound requests (`GET`, `POST`, `DELETE`, `HEAD`) using AWS SDK credentials already resolved via `aws-config` (`aws-sigv4` is already in the transitive lockfile from `aws-sdk-glue` and `object_store`).
   - Support REST catalog config discovery (`GET /v1/config?warehouse=<arn_or_name>`) so S3 Tables and Glue Iceberg REST endpoints return canonical `prefix` overrides automatically.
2. Add `CatalogType::S3Tables` (`"s3tables"`) to [`src/core/catalog/mod.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/mod.rs#L97-L122):
   - Accept `"warehouse"` (table bucket ARN), `"region"` (derived from ARN or AWS defaults), and optional `"url"` override (`https://s3tables.<region>.amazonaws.com/iceberg`).
   - Route `CatalogType::S3Tables` through `RestCatalogClient` with `RestCatalogAuth::AwsSigV4 { region, service: "s3tables".to_string() }` and SSRF validation when `BSDB_SSRF_GUARD=1`.
3. Optionally allow `CatalogType::Glue` config `"mode" = "rest"` to talk to Glue's managed Iceberg REST endpoint (`service: "glue"`), complementing the existing native `aws-sdk-glue` client.

#### Workstream C2 — `CatalogType::Memory` (In-Process Reference Catalog)
1. Implement `src/core/catalog/memory.rs` (`MemoryCatalogClient`) behind `CatalogType::Memory` (`"memory"`):
   - Backed by `Arc<parking_lot::RwLock<MemoryCatalogState>>` (reusable across handles pointing to the same catalog URI or explicit builder).
   - Enforces full `commit_table_with_requirements` semantics (`assert-table-uuid`, `assert-ref-snapshot-id`, `assert-current-schema-id`, `assert-last-assigned-field-id`, `assert-default-spec-id`), returning `BenoStreamError::CatalogRequest { status: Some(409), .. }` on conflict so multi-writer catalog convergence tests exercise real HTTP-409 semantics without network flakiness.
   - Implements `create_namespace`, `create_table`, `drop_table`, `list_tables`, `load_table`, `table_exists`, `commit_table`, and `commit_table_with_requirements`.
   - Note: [`src/core/catalog/config.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/config.rs) and Python/SQL catalog entry points (`create_catalog("memory", ...)`) pick up `"memory"` cleanly.

#### Workstream C3 — Hardening Existing Catalog Adapters (Correctness & CAS)
Fix real bugs in all five non-REST catalogs so every adapter implements **atomic compare-and-swap** and **consistent `metadata_location` resolution**:

1. **Fix `JdbcCatalogClient` ([`src/core/catalog/jdbc.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/jdbc.rs#L76-L223)):**
   - **Dialect placeholders:** Detect backend from URI (`postgres://` / `postgresql://` vs `sqlite://` / `mysql://`) and emit `$1, $2, ...` for Postgres vs `?` for SQLite/MySQL (or refactor to backend-specific query helpers via `sqlx`), and verify wide `VARCHAR(1024)` columns for long S3 metadata URIs (currently `VARCHAR(255)` truncates long URIs).
   - **Canonical metadata path & `load_table`:** `create_table` should store `{table_location}/metadata/v0.metadata.json` (after initializing metadata via `Table::create_async`), and `load_table` should strip `/metadata/v...` (and fallback `/manifest.json`) consistently or read `TableMetadata` from storage to determine `location`.
   - **Explicit `set-metadata-location` & Optimistic CAS:** Read `current_meta`, check explicit `set-metadata-location` first (falling back to snapshot derivation), validate `requirements`, and execute:
     ```sql
     UPDATE iceberg_tables
        SET previous_metadata_location = metadata_location,
            metadata_location = ?
      WHERE catalog_name = ? AND table_namespace = ? AND table_name = ?
        AND (metadata_location = ? OR (? IS NULL AND metadata_location IS NULL))
     ```
     If `rows_affected() == 0`, return `BenoStreamError::CatalogRequest { catalog: "jdbc", status: Some(409), reason: "CAS conflict" }` so callers trigger conflict retry rather than silently overwriting a concurrent commit!
   - Implement `drop_table` and `list_tables`.

2. **Harden `GlueCatalogClient` ([`src/core/catalog/glue.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/glue.rs#L95-L301)):**
   - Fix `create_table` `metadata_location` parameter to point to the initial metadata JSON rather than bare table location, and verify `load_table` resolves table root correctly when `metadata_location` ends in `/metadata/vN.metadata.json`.
   - Pass Glue `expected_version_id` / `version_id` conditional lock (`UpdateTableInput`) when committing (and map `ConcurrentModificationException` to `CatalogRequest { status: Some(409), .. }`), plus verify `requirements` against the loaded `TableMetadata` before calling `UpdateTable`.
   - Implement `drop_table` (`delete_table`) and `list_tables` (`get_tables` paginated).

3. **Harden `HiveMetastoreClient` ([`src/core/catalog/hive.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/hive.rs#L146-L277)):**
   - Read explicit `set-metadata-location` updates in `commit_table`.
   - Use Hive Metastore's `alter_table_with_environment_context` with `expected_parameter_key = "metadata_location"` and `expected_parameter_value = <prior>` for atomic compare-and-swap (mapping conflict exceptions to status `409`).
   - Implement `drop_table` and `list_tables` (`get_all_tables`).

4. **Harden `UnityCatalogClient` ([`src/core/catalog/unity.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/unity.rs#L217-L280)):**
   - Add an Iceberg REST mode for Databricks Unity Catalog (`/api/2.1/unity-catalog/iceberg-rest/v1/...`, which uses the standard Iceberg REST protocol + Bearer token and supports `requirements` CAS + `409 Conflict`). Keep existing Unity tables endpoint compatibility with explicit conflict verification where REST Iceberg endpoints are unavailable.
   - Implement `drop_table` and `list_tables`.

5. **Harden `NessieClient` ([`src/core/catalog/nessie.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/catalog/nessie.rs#L175-L254)):**
   - Support explicit `set-metadata-location` updates alongside `add-snapshot` commits so metadata-only commits (`update_schema`, `delete`, `vacuum`) update the Nessie pointer cleanly.
   - Map Nessie HTTP `409 ReferenceConflictException` to `BenoStreamError::CatalogRequest { catalog: "nessie", status: Some(409), .. }` and implement `drop_table` and `list_tables` (`getEntries`).

#### Workstream C4 — Cross-Catalog Conformance Suite & CI Wiring
- Create `tests/test_catalog_conformance.rs` running an identical contract suite across:
  1. `CatalogType::Memory` (always-on in CI),
  2. `CatalogType::Jdbc` over SQLite `:memory:` / temp file (always-on in CI; plus optional Postgres if `DATABASE_URL` set),
  3. `CatalogType::Rest` against an embedded Axum mock Iceberg REST server (exercising OAuth2 refresh, `requirements` CAS 409 retry, `drop_table`, `list_tables`, SigV4 header canonicalization unit tests),
  4. `CatalogType::Nessie` gated on `NESSIE_URI` (`docker-compose-rustfs-nessie.yml`).
- Remove the dead `--ignore=tests/integration/test_catalog.py` line in [`.github/workflows/ci.yml`](https://github.com/benolabsai/benostreamdb/blob/main/.github/workflows/ci.yml#L255) (no such file exists on disk) and ensure the Rust catalog suite executes on every PR in CI.
- Update [`docs/CATALOG_USAGE.md`](https://github.com/benolabsai/benostreamdb/blob/main/docs/CATALOG_USAGE.md#L1-L20) and [`docs/SUPPORT_MATRIX.md`](https://github.com/benolabsai/benostreamdb/blob/main/docs/SUPPORT_MATRIX.md) with accurate CAS guarantees and `s3tables`/`memory` configuration examples.

---

## Acceptance & Sequencing

1. **Phase A (In-Process Catalog + JDBC Fixes + Conformance Harness):** Workstreams **C2** (`Memory`), **C3.1** (`Jdbc` Postgres/SQLite/CAS/metadata path fixes), and **C4** (`test_catalog_conformance.rs` + CI cleanup). (**V3.1** docs honesty is already completed.) Lowest blast radius; immediately catches catalog regressions on every commit.
2. **Phase B (SigV4 + `S3Tables` + Native CAS Hardening):** Workstreams **C1** (`AwsSigV4` + `S3Tables`), **C3.2–C3.5** (`Glue`, `Hive`, `Unity`, `Nessie` CAS + `drop_table`/`list_tables`).
3. **Phase C (Iceberg V3 Full Spec & Oracle):** Workstreams **V3.2**, **V3.3**, and **V3.4** (`deletion-vector-v1` codec + reader/writer wiring, `format-version = 3` Avro metadata + `143–145` fields, isolated `apache/iceberg-rust` reader oracle, and Spark 4.0/Trino bi-directional V3 round-trip verification).
