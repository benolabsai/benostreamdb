# iceberg-rust reader oracle

Verifies that a table written by BenoStreamDB is readable by the upstream
[`apache/iceberg-rust`](https://github.com/apache/iceberg-rust) crate — the
reference Rust implementation, which shares no code with this engine.

A reader built from the writer's own assumptions proves nothing. This crate
therefore links `iceberg` only; it never depends on `benostreamdb`. It opens the
table tree as a *static table* from its `metadata/v<N>.metadata.json` and asserts
what any external engine must be able to see.

## Why a separate crate and workspace root

`iceberg` 0.11 pins `arrow-*` 58.x and its own `object_store`. The engine pins
`arrow` 57.2 / `object_store` 0.12.4 and audits its graph with `deny.toml`.
Pulling `iceberg` into the main workspace would bloat the production binaries and
create a unified-resolver conflict, so this package declares its own
`[workspace]` (the same trick `fuzz/` uses for `libfuzzer-sys`). The two sides
communicate only through on-disk artifacts.

## Running it

1. Produce fixtures with the engine (this test is `#[ignore]`d):

   ```bash
   BSDB_FIXTURE_DIR=/tmp/bsdb_oracle_fixture \
     cargo test --test oracle_fixture -- --ignored --nocapture
   ```

   That writes a V2 table and a V3 table, each seeded with ids 1–5 and then
   deleting ids ≥ 3 (V2 via Parquet position deletes, V3 via a
   `deletion-vector-v1` Puffin blob).

2. Point the oracle at a table:

   ```bash
   BSDB_ORACLE_URI=file:///tmp/bsdb_oracle_fixture/v2 \
   BSDB_ORACLE_SUITE=v2 BSDB_ORACLE_EXPECT_IDS=1,2 \
     cargo test --manifest-path tests/iceberg_rust_oracle/Cargo.toml -- --nocapture
   ```

   ```bash
   BSDB_ORACLE_URI=file:///tmp/bsdb_oracle_fixture/v3 \
   BSDB_ORACLE_SUITE=v3 BSDB_ORACLE_EXPECT_IDS=1,2 \
     cargo test --manifest-path tests/iceberg_rust_oracle/Cargo.toml -- --nocapture
   ```

With no `BSDB_ORACLE_URI` set the test prints a `SKIPPED` line and passes, so an
unconfigured run never looks like a green conformance result.

## What it asserts

| Assertion | Why it matters |
|---|---|
| `metadata.format_version()` matches the suite | The reader infers V2/V3 from the metadata, not from our code path |
| V3: `metadata.next_row_id() > 0` | Row lineage must be published at the table level |
| V3: `current_snapshot().first_row_id().is_some()` | Each snapshot must record its row range, or no reader can assign stable `_row_id`s |
| Scanned `id` values equal `BSDB_ORACLE_EXPECT_IDS` | Proves manifests, bounds, and deletes (including V3 deletion vectors) are applied by a foreign engine |

`_row_id` and `_last_updated_sequence_number` are *metadata* columns and are
deliberately absent from `current_schema()`, so the oracle checks lineage through
`next_row_id` / the snapshot row range rather than by looking for schema fields.

## Bugs this harness caught

Added during the `v0.16.0` V3.4 workstream; each was invisible to
BenoStreamDB's own reader and to PyIceberg.

1. **Manifest `content` metadata used numbers.** We wrote `content = "0"` / `"1"`.
   `iceberg-rust` parses that key with a strict `FromStr` that accepts only
   `"data"` / `"deletes"` and rejected every delete manifest with
   `Invalid manifest content type: 1`. Fixed in
   `IcebergWriter::attach_manifest_metadata`.
2. **V3 snapshots carried no row range.** `first-row-id` / `added-rows` were only
   populated on the flush path, so a delete or compaction snapshot advertised no
   lineage. Now backfilled in `TableMetadata` normalization: a snapshot that adds
   no rows records `first-row-id` = next unassigned id with `added-rows = 0`.
3. **Deletion vectors pointed at a relative data-file path.** The manifest stores
   `file:///…/seg_x.parquet`, while `referenced_data_file` held `seg_x.parquet`.
   Iceberg matches a DV to its data file by comparing that string verbatim, so the
   DV matched nothing and the external reader returned the deleted rows. Fixed in
   `Table::write_deletion_vector_file_async` by normalizing to the absolute form.

Our own read path was blind to (3) because it applies every DV attached to a
manifest entry rather than matching on `referenced_data_file` — which is exactly
the class of bug an independent oracle exists to find.
