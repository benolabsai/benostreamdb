# BenoStreamDB — Iceberg V3 Spec Conformance Plan

**Status (2026-11-11, target release: post-`v0.16.0`): `V3.1` done; `V3.2`, `V3.3`, and `V3.4` open.**
**Catalog parity (`C1–C4`) is complete and promoted to [`ADR-012`](https://github.com/benolabsai/benostreamdb/blob/main/docs/architecture/ADR_012_CATALOG_PARITY_AND_CAS_HARDENING.md).** What remains in this plan is Part I only: spec-exact V3 Deletion Vectors and the independent `apache/iceberg-rust` reader oracle.
**Companion plan:** [production_readiness_plan.md](production_readiness_plan.md) (WS3/WS4 exit gates).

---

## 0. Why This Work Matters (Architecture & Operational Context)

BenoStreamDB commits data using open Apache Iceberg metadata (`v{N}.metadata.json`, Avro `snap-*.avro` manifest lists, and Avro `*-m*.avro` data manifests) over Parquet segments and Puffin overlay sidecars. For engineers and platform teams choosing an indexed lakehouse engine, open-table portability is only real if external engines (Apache Spark, Trino, Snowflake, DuckDB, and `apache/iceberg-rust`) read and evolve BenoStreamDB tables without translation layers or data duplication.

One audit gap remains open: **Iceberg V3 validation is asymmetric and partial.** [`test_bsdb_to_pyiceberg.py`](https://github.com/benolabsai/benostreamdb/blob/main/tests/integration/test_bsdb_to_pyiceberg.py#L1) validates against **PyIceberg 0.12.0**, which targets Iceberg V2. Meanwhile, BenoStreamDB writes V3 row-lineage metadata (`next-row-id`, `first-row-id`, `added-rows`), but its Deletion Vector (DV) path is incomplete on write and unwired on read, and `attach_manifest_metadata` hardcodes `format-version = "2"`.

The practical consequence: a `format_version = 3` table written by BenoStreamDB is not V3-conformant to an external reader that trusts the Avro `format-version` key, and deletes on such a table are emitted as V2 Parquet position deletes. Until V3.2–V3.4 land, the documented contract is "V3 row lineage, V2 deletes" — which is what [`docs/ICEBERG_COMPATIBILITY.md`](https://github.com/benolabsai/benostreamdb/blob/main/docs/ICEBERG_COMPATIBILITY.md) and [`docs/ICEBERG_STORAGE.md`](https://github.com/benolabsai/benostreamdb/blob/main/docs/ICEBERG_STORAGE.md) now state explicitly.

### Already Resolved Baseline (`v0.15.1` + Staged `M2`)

The following prerequisites are **already resolved in source** ahead of this plan:
- **V3 row-lineage strict OCC (`format_version = 3`):** Contiguous `_row_id` and `_last_updated_sequence_number` assignment, `next-row-id` propagation, and rejected Parquet cleanup on `ManifestVersionMismatch` are implemented and verified across 6 concurrent writers in [`test_m2_commit_state_machine.rs`](https://github.com/benolabsai/benostreamdb/blob/main/tests/test_m2_commit_state_machine.rs) (Pillar 7) and [`test_multi_writer_differential.rs`](https://github.com/benolabsai/benostreamdb/blob/main/tests/test_multi_writer_differential.rs).
- **V3 nanosecond schema types:** `timestamp_ns` and `timestamptz_ns` are mapped in [`src/core/manifest/types.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/manifest/types.rs) alongside `time`, with nanosecond `ManifestValue` bounds decode.
- **REST catalog CAS & requirement validation:** `Catalog::commit_table_with_requirements`, `RestCatalogClient` HTTP 409 propagation, `commit_catalog_step_async` retry (`MAX_CATALOG_RETRIES = 8`), and server-side `TableRequirement` validation in [`src/bin/iceberg_rest.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/bin/iceberg_rest.rs) are shipped and certified by `pillar1_axum_rest_catalog_409_conflict_requirements_and_diverged_location`.
- **Catalog parity and CAS hardening (C1–C4):** `memory` and `s3tables` catalogs, AWS SigV4, and atomic compare-and-swap in `jdbc` / `glue` / `hive` / `unity` / `nessie`. See [`ADR-012`](https://github.com/benolabsai/benostreamdb/blob/main/docs/architecture/ADR_012_CATALOG_PARITY_AND_CAS_HARDENING.md).

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

### 2. Known Iceberg V3 Gaps in BenoStreamDB (Verified in Source, Still Open)

Inspection of the current writer, reader, Puffin, and table maintenance paths reveals five concrete conformance gaps that must be closed before claiming V3 write parity:

1. **Manifest Avro header hardcodes V2 metadata ([`src/core/iceberg/writer.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/writer.rs#L54-L70)):**
   `attach_manifest_metadata` writes `writer.add_user_metadata("format-version".to_string(), "2")?` unconditionally, even when writing manifests for a `format_version = 3` table. External V3 readers inspect the Avro `format-version` key to select V3 manifest parsing rules.
2. **Manifest Avro `r2` schema omits V3 Deletion Vector fields ([`src/core/iceberg/writer.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/writer.rs#L965-L1007)):**
   [`src/core/iceberg/manifest.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/manifest.rs#L248-L278) already parses `referenced_data_file`, `content_offset`, and `content_size_in_bytes` on read, but `IcebergWriter::generate_manifest_schema` omits those three fields (`field-id` `143`, `144`, `145`) from the `r2` Avro schema.
3. **Manifest `content` / `file_format` encoding for V3 Deletion Vectors ([`src/core/iceberg/writer.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/writer.rs#L134-L160)):**
   - Per the Apache Iceberg V3 specification, Deletion Vectors are manifest delete entries with `content = 1` (`POSITION_DELETES`), `file_format = "PUFFIN"`, non-null `referenced_data_file` (`143`), `content_offset` (`144`), and `content_size_in_bytes` (`145`).
   - Currently, `writer.rs` maps `DeleteContent::DeletionVector` to `content_id = 3` (matching a custom check in [`manifest.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/manifest.rs#L483-L499)) and sets `delete_format = "AVRO"` for any path not ending in `.parquet` — mislabeling `.puffin` files as Avro.
4. **Puffin DV blob type and binary framing ([`src/core/puffin.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/puffin.rs#L762)):**
   - `DELETION_VECTOR_TYPE` is defined as `"apache-datasketches-theta-v1"` instead of the Iceberg V3 spec identifier `"deletion-vector-v1"`.
   - `read_deletion_vector_from_puffin` and `read_deletion_vector_from_bytes` deserialize raw 32-bit `roaring::RoaringBitmap::deserialize_from` payloads, whereas Iceberg V3 defines the Puffin DV payload envelope as:
     * `4-byte big-endian length` (size in bytes of magic + bitmap, excluding length & CRC bytes),
     * `4-byte magic sequence` `[0xD1, 0xD3, 0x39, 0x64]` (big-endian `0xD1D33964`, equivalent to little-endian `1681511377` / `0x6439D3D1`),
     * **portable 64-bit roaring bitmap** (little-endian container serialization / `RoaringTreemap`),
     * `4-byte big-endian CRC-32` over magic + bitmap payload.
   - Neither function currently has a caller inside `HybridReader` (`src/core/reader/`), so **V3 Deletion Vectors are neither written nor applied on read**.
5. **Delete writer & staged writes on V3 ([`src/core/table/maintenance.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/maintenance.rs) & [`src/core/table/stage.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/stage.rs)):**
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
- Update [`src/core/puffin.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/puffin.rs#L762):
  * Change `DELETION_VECTOR_TYPE` to `"deletion-vector-v1"`.
  * Implement spec-exact `write_deletion_vector_to_puffin` / `encode_deletion_vector_v1` / `decode_deletion_vector_v1` framing (`4B BE` length, `4B` magic `[0xD1, 0xD3, 0x39, 0x64]` [BE `0xD1D33964` / LE `1681511377`], little-endian 64-bit portable roaring bitmap, `4B BE` CRC-32 with CRC validation returning `BenoStreamError::DeletionVectorDeserialize`).
  * Ensure Puffin DV blobs share the Puffin footer property `referenced-data-file` / `cardinality` per spec, coexist cleanly with BenoStreamDB's compound index Puffin files, or reside in dedicated `del-dv-<uuid>.puffin` files written through `self.store`.

#### Workstream V3.3 — Read/Write/Compact Path Integration for V3 DVs
- Wire `read_deletion_vector_from_bytes` (with byte-range S3 read using `content_offset..content_offset + content_size_in_bytes` + CRC check) into `HybridReader` / `PositionDeleteReader` / [`src/core/manifest/manager/load.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/manifest/manager/load.rs) so `DeleteContent::DeletionVector` masks deleted positions identically to Parquet position deletes and integrates into [`compact_bin`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/compaction.rs).
- Wire `Table::delete_async` and `MergePlanner` so `format_version >= 3` writes V3 Puffin DVs (and merges prior DVs on the same `referenced_data_file` into a single Puffin blob per file, as required by Iceberg V3's *at-most-one DV per data file per snapshot* invariant).
- Wire `vacuum` and `remove_orphan_files_async` in [`src/core/manifest/manager/commit.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/manifest/manager/commit.rs) so DV `.puffin` files referenced in `Manifest::delete_files` are retained and expired DVs are reaped safely.

#### Workstream V3.4 — `apache/iceberg-rust` & Multi-Engine Verification Matrix
- Add isolated crate `tests/iceberg_rust_oracle/` depending on `iceberg`:
  * **V2 suite:** BenoStreamDB writes/deletes/compacts/evolves schema+partition → `iceberg-rust` opens table, scans via Arrow, asserts exact row values, schema field IDs, and bounds.
  * **V3 suite:** BenoStreamDB creates `format_version = 3` table → appends multiple commits (checking `first_row_id` and `next_row_id`) → executes `delete_async` producing `deletion-vector-v1` Puffin blobs → `iceberg-rust` opens snapshot and reads exact surviving rows → BenoStreamDB compacts → `iceberg-rust` verifies clean state and preserved `_row_id` lineage.
- Add Java interop matrix (CI optional / nightly workflow alongside `ecosystem`): Spark 4.0 and Trino write V2 position deletes and V3 DVs → BenoStreamDB reads; BenoStreamDB mutates → Spark/Trino read.

---

## Acceptance & Sequencing

Workstreams are strictly ordered because each depends on the prior one's artifacts:

1. **V3.2 (codec + manifest schema):** self-contained in `writer.rs` and `puffin.rs`, verifiable by unit tests that round-trip the DV envelope and assert Avro field ids `143–145` and the `format-version` user-metadata key.
2. **V3.3 (path integration):** depends on V3.2's encoder. Exit gate is that a delete on a `format_version = 3` table produces exactly one `deletion-vector-v1` Puffin blob per data file per snapshot, that `HybridReader` masks those positions identically to V2 Parquet position deletes, and that `vacuum` retains referenced DVs while reaping expired ones.
3. **V3.4 (independent oracle):** depends on V3.2 and V3.3 producing spec-valid artifacts. The isolated `iceberg-rust` crate is the certification gate; PyIceberg remains a V2 compatibility check only.

`format_version = 3` remains documented as **row lineage conformant / deletes V2** until V3.3 lands, and `Table::write_mode()` remains `"buffered"` on V3 as an explicit, documented limitation rather than an accident.
