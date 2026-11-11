# ADR-010: Iceberg V3 Writes Must Be the Spec, and a Foreign Reader Is the Acceptance Test

> **Status: accepted (`v0.16.0`).** Implemented across [`puffin.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/puffin.rs), [`iceberg/writer.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/writer.rs), [`iceberg/manifest.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/manifest.rs), [`iceberg/delete.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/iceberg/delete.rs), [`manifest/types.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/manifest/types.rs), [`metadata.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/metadata.rs), [`table/maintenance.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/table/maintenance.rs), and [`merge.rs`](https://github.com/benolabsai/benostreamdb/blob/main/src/core/merge.rs).
> Change history: see the `[Unreleased]` / `0.16.0` entries in [`CHANGELOG.md`](https://github.com/benolabsai/benostreamdb/blob/main/CHANGELOG.md). Catalog-side decisions from the same programme are in [`ADR-009`](https://github.com/benolabsai/benostreamdb/blob/main/docs/architecture/ADR_009_CATALOG_PARITY_AND_CAS_HARDENING.md).

---

## 1. Context

BenoStreamDB's portability claim rests on writing ordinary Apache Iceberg metadata: `v{N}.metadata.json`, Avro manifest lists, Avro data/delete manifests, Parquet segments, Puffin sidecars. Two forces made a decision necessary.

**Force 1 — a private dialect is a silent data-loss risk.** The engine had begun encoding V3 concepts in shapes only it understood: the manifest Avro metadata hardcoded `format-version = "2"`; the deletion-vector Puffin blob used a foreign type identifier (`apache-datasketches-theta-v1`) with a raw roaring payload; delete entries used a custom `content = 3` code; and the V3 manifest schema never emitted fields `143–145`. A foreign engine opening such a table has two choices and both are bad: reject it, or trust the metadata and return rows the writer deleted.

**Force 2 — the existing verification could not detect this class of bug.** PyIceberg 0.12 is V2-centric, so it validates the V2 surface and ignores V3 semantics. Worse, BenoStreamDB's own reader applies *every* deletion vector attached to a manifest entry and never consults `referenced_data_file`, so it is structurally incapable of noticing a vector that no external engine can match. "Our reader agrees with our writer" is not evidence.

## 2. Decision

**D1 — Write the published V3 spec, never a private encoding.** Deletion vectors use blob type `deletion-vector-v1` and the spec envelope (4-byte big-endian length, magic `D1 D3 39 64`, portable 64-bit roaring bitmap, 4-byte big-endian CRC-32). Manifest metadata carries the table's own `format-version`. Delete entries use `content = 1` (`POSITION_DELETES`) with `file_format = "PUFFIN"` and populated `referenced_data_file` / `content_offset` / `content_size_in_bytes`.

*Rejected:* keeping the custom `content = 3` encoding and documenting it. It is strictly less work, but it makes every V3 table engine-specific, which is the one thing an open table format exists to prevent. The legacy code is still **accepted on read** so tables written by earlier builds keep resolving.

**D2 — V2 is frozen, not migrated.** Every V3 behavior is gated on `format_version >= 3`. V2 tables keep Parquet position deletes and their manifests still declare `format-version = 2`.

*Rejected:* silently upgrading tables on write. It would change the delete representation under operators who never asked for it, and V2 remains the interchange format most clusters actually accept.

**D3 — A foreign reader is the acceptance criterion, and it must not share code with the engine.** Conformance means `apache/iceberg-rust`, Spark, or Trino opens the table and sees the exact row set. The Rust oracle is a separate workspace that depends on `iceberg` only; it communicates with the engine exclusively through on-disk artifacts.

*Rejected:* asserting against PyIceberg, or against our own reader. Both already passed while all three of the defects listed in §3 were live. Code sharing is what made them invisible.

**D4 — Reject rather than repair.** A deletion vector with a bad length, magic, or CRC is a typed error (`BenoStreamError::DeletionVectorDeserialize`), not a warning with a best-effort bitmap.

*Rejected:* tolerating malformed vectors. A silently-short delete list resurrects deleted rows; a failed query is the better failure.

## 3. Consequences

**Positive.** A `format-version = 3` table written by BenoStreamDB is readable by `iceberg-rust` 0.11 and by Spark/Trino, including after deletions. V3 lineage is publishable: each snapshot carries a row range so a reader can assign stable `_row_id` values. The engine's delete representation is now the interchange format rather than a local convention.

**Negative and accepted.**
- **CRC-32 is hand-rolled** (IEEE 802.3, ~15 lines) instead of adding a dependency, because `deny.toml` audits every crate in the graph.
- **Positions above `u32::MAX` are an error**, not a 64-bit position. The engine's row positions are `u32`; a table with more than 4 billion rows in one data file is out of scope, and the failure is loud.
- **`Table::write_mode()` stays `"buffered"` on V3.** Distributed Spark/Trino task staging cannot reserve disjoint `_row_id` ranges without driver-side allocation. Safety comes from [`ADR-008`](https://github.com/benolabsai/benostreamdb/blob/main/docs/architecture/ADR_008_PENDING_COMMIT_STATE_MACHINE.md)'s `PendingCommit` machine, not from a staged-write contract.
- **`_row_id` is not in `current_schema()`.** It is a metadata column by spec; lineage lives in `next-row-id` and the snapshot row range. Tooling that expects to `SELECT _row_id` will not find it there.
- **Runtime pinning is bounded by what Iceberg publishes.** `iceberg-spark-runtime` exists only for Spark minor series 3.4, 3.5, 4.0 and 4.1 — there is no 4.2 artifact, so Spark 4.2 has no supported pairing. The JVM matrix therefore pins Spark 3.5.2 (`3.5_2.12`, Java 17) and 4.0.x (`4.0_2.13`, Java 21) on Iceberg 1.11; 1.6 predates V3 vectors.
- **The JVM matrix runs weekly, not per-PR.** A Spark submit is a JVM start plus a ~200 MB Maven resolve. The trade is accepted because the format only moves when `src/core/iceberg/**` or `metadata.rs` changes, which the workflow's path filter watches.

**Risk this decision accepts.** A bug in the V3 codec is now a bug in a format other engines depend on; there is no longer a private encoding to fall back to. The mitigation is the three independent readers in §4, not the engine's own test suite.

## 4. How the Decision Is Enforced

| Gate | Enforces | Cadence |
|---|---|---|
| [`tests/test_iceberg_v3_conformance.rs`](https://github.com/benolabsai/benostreamdb/blob/main/tests/test_iceberg_v3_conformance.rs) | D1, D2 — manifest metadata/schema shape, parsed by a hand-written Avro container reader rather than `apache_avro`, so a wrong key name cannot pass by construction | every push |
| [`tests/iceberg_rust_oracle`](https://github.com/benolabsai/benostreamdb/blob/main/tests/iceberg_rust_oracle/README.md) | D3 — `iceberg-rust` opens the table, asserts format version, `next_row_id`, snapshot row range, and the exact scanned row set after a vector | every push |
| [`.github/workflows/iceberg-interop.yml`](https://github.com/benolabsai/benostreamdb/blob/main/.github/workflows/iceberg-interop.yml) | D3 — Spark 3.5/4.0 and Trino 482/484 read the table, including V3 lineage columns | weekly + on table-format changes |
| `puffin.rs` unit tests | D4 — round-trip plus rejection of bad length, magic, and CRC | every push |

This ADR should be revisited if: Iceberg publishes a Spark runtime for a new minor series (re-pin the matrix), `apache/iceberg-rust` starts matching vectors by a different key (re-check D1's path normalization), or the engine gains 64-bit row positions (D4's `u32` bound must be lifted).
