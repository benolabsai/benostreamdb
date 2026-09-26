# Iceberg Compatibility Matrix

This document records which Iceberg V2/V3 features BenoStreamDB writes and reads
conformantly, how that is validated, and the known deviations. It is the
"published compatibility matrix" deliverable of WS4 in
`plans/production_readiness_plan.md`.

## How this is validated

Independently of BenoStreamDB's own reader, tables are validated with
**PyIceberg** (an independent Apache Iceberg implementation) and against the
Iceberg spec's structural requirements.

| Layer | Test | What it proves |
|-------|------|----------------|
| Structural (Rust) | `tests/test_iceberg_conformance.rs` | The metadata JSON, manifest-list Avro, and data-manifest Avro carry the spec-required fields, field-ids, and types. |
| Independent reader (Python) | `tests/integration/test_bsdb_to_pyiceberg.py` | PyIceberg loads a BenoStreamDB table and reads its data (schema + values) correctly. |
| Reverse direction (Python) | `tests/integration/test_pyiceberg_compat.py` | BenoStreamDB reads a PyIceberg-written table. |

Run the Python interop tests with the project virtualenv:

```bash
.venv/bin/python -m pytest tests/integration/test_bsdb_to_pyiceberg.py -v
```

Validated against **pyiceberg 0.12.0** / **pyarrow 25.0.1**.

## Feature matrix

| Feature | V2/V3 | Status | Notes |
|---------|-------|--------|-------|
| Table metadata JSON (`v{N}.metadata.json`) | V2 | ✅ Conformant | All required fields present; validated by PyIceberg load. |
| Manifest list (`snap-*.avro`) | V2 | ✅ Conformant | All fields carry spec `field-id`s; `manifest_path` and `partitions.element-id` present. |
| Data manifest (`*-m*.avro`) | V2 | ✅ Conformant | `field-id`s on every field; maps annotated `logicalType: map`; arrays carry `element-id`. |
| Absolute artifact paths | V2 | ✅ Conformant | Snapshot `manifest-list`, manifest-list `manifest_path`, and data-manifest `file_path` are absolute URIs (required by external readers). |
| Snapshot log / metadata log | V2 | ✅ Conformant | `snapshot-log` populated; `metadata-log` present. |
| Sort orders | V2 | ✅ Conformant | Persisted in metadata JSON (`sort-orders`, `default-sort-order-id`). |
| Partition specs / evolution | V2 | ✅ Conformant | `partition-specs`, `default-spec-id`; evolution preserves the manifest list (see §2.4 in the plan). |
| Column statistics (NDV / bounds) | V2 | ✅ Conformant | `null_value_counts`, `lower_bounds`, `upper_bounds` written per field id. |
| Nanosecond timestamps | V2 | ✅ Conformant | `Timestamp(Nanosecond)` round-trips without precision loss. |
| Row lineage (`_row_id`, `_last_updated_sequence_number`) | V3 | ✅ Conformant | `next-row-id` in metadata; `first-row-id`/`added-rows` on the snapshot. |
| Equality / position deletes | V2 | ✅ Conformant | Delete files registered in manifest entries. |
| Deletion vectors | V3 | ⚠️ Extension | Written into Puffin blobs (BenoStream extension; not a base V3 reader feature). |
| Puffin index blobs | — | ⚠️ Extension | BenoStream overlay indexes (HNSW/TQ/PQ/BM25/bitmap) ride in Puffin; standard readers ignore them and fall back to full scans. |
| `_staging`, `_wal` directories | — | ⚠️ Extension | BenoStream durability artifacts; standard readers ignore them. |

Legend: ✅ conformant and machine-checked · ⚠️ documented deviation/extension.

## Known deviations from the Iceberg spec

1. **BenoStream overlay indexes (Puffin).** Index blobs are an extension. A
   standard Iceberg reader sees the data files and returns correct results via a
   full scan; only BenoStream uses the overlay indexes.
2. **`_staging/` and `_wal/` directories** live beside the table. They are
   ignored by standard readers and excluded from orphan cleanup.
3. **Deletion vectors** are supported as a BenoStream extension over Puffin,
   not as a base V3 reader feature.
4. **pgvector-style vector types** (`FixedSizeList<Float32>` etc.) are stored as
   standard Iceberg `list<float>` columns; vector distance operators are
   BenoStream SQL extensions (see the README's pgvector notes).

## Bugs found and fixed by this validation

Validating with an independent reader surfaced four conformance bugs that
BenoStreamDB's own round-trip tests did not catch:

1. **Relative `manifest-list` path** — external readers resolve it against the
   process CWD, not the table location. Now written absolute.
2. **Relative `manifest_path` in the manifest list** — same class; now absolute.
3. **Relative `file_path` in the data manifest** — now absolute on write and
   relativized back on read (so internal path handling is unchanged).
4. **Avro schemas missing Iceberg metadata** — added `field-id` on every field,
   `logicalType: map` on map arrays, and `element-id` on list arrays.

See `plans/production_readiness_plan.md` §2.5 for details.
