# Zero-copy reads with the Trino Iceberg connector

BenoStreamDB already stores its data as Parquet and already writes Iceberg-shaped
metadata. Making that metadata **spec-compliant** means an Iceberg reader can open
a BenoStreamDB table directory directly: no export, no rewrite, no second copy of
the data. Trino's Iceberg connector, PyIceberg, and any other Iceberg client read
the exact files the engine wrote — including vector columns.

## What "zero copy" requires

An Iceberg table is three layers. A reader must accept all three, so all three are
part of the contract:

| Layer | What readers need | Where we write it |
| --- | --- | --- |
| `metadata/vN.metadata.json` | `TableMetadataParser`-valid JSON | [`src/core/metadata.rs`](../../src/core/metadata.rs) |
| `_manifest/snap-*.avro` + `*-m*.avro` + `*-d*.avro` | Manifest list, plus `schema` / `partition-spec` / `partition-spec-id` / `format-version` / `content` Avro user metadata; one content type per manifest | [`src/core/iceberg/writer.rs`](../../src/core/iceberg/writer.rs) |
| `*.parquet` | `field_id` on every column, including nested elements | [`src/core/segment.rs`](../../src/core/segment.rs) |
| `del-pos-*.parquet` | `file_path` / `pos` with Iceberg's reserved ids, rows naming absolute data-file URIs | [`src/core/iceberg/delete.rs`](../../src/core/iceberg/delete.rs) |

### 1. Table metadata

Iceberg's parser is stricter than "looks like Iceberg". The following are
enforced at the metadata boundary (see `TableMetadata::normalize_iceberg_metadata`
and the manual `Serialize` impls in [`src/core/manifest/types.rs`](../../src/core/manifest/types.rs)):

- every schema is `{"type":"struct","schema-id":N,"fields":[...]}` — `schema-id`
  is kebab-case, and `type` is required (an alias on deserialize is not enough);
- `last-partition-id` is present (999 for unpartitioned tables, then the highest
  assigned field id);
- `sort-orders` is non-empty (Iceberg rejects `[]`; unsorted is
  `{"order-id":0,"fields":[]}`);
- `parent-snapshot-id` is **omitted** for root snapshots — Iceberg rejects an
  explicit JSON `null`;
- `last-sequence-number` is at least the largest snapshot sequence number;
- partition fields carry concrete `source-id` and `field-id` integers;
- nested types use Iceberg's grammar:
  `{"type":"list","element-id":301,"element":"double","element-required":false}`,
  `{"type":"map","key-id":...}` and `{"type":"struct","fields":[...]}`. Arrow-only
  spellings such as `int32`, `largeutf8`, or `timestamp(microsecond, none)` are
  normalised onto Iceberg names;
- v3-only snapshot fields (`first-row-id`, `added-rows`) are stripped from v2
  tables;
- every mutating operation publishes a snapshot, with the spec's
  `summary.operation`: `append` for writes, `delete` for deletes and truncate,
  `overwrite` for MERGE, and `replace` for compaction. A manifest commit on its
  own is invisible to external readers — the snapshot is what moves
  `current-snapshot-id`.

### 2. Manifest Avro metadata

`ManifestReader` reads the table schema and partition spec from Avro user
metadata, not from the Avro schema. We now attach:

```
schema            = {"type":"struct","schema-id":2,...}
partition-spec    = []                  # the fields array, not the spec wrapper
partition-spec-id = 0
format-version    = 2
content           = 0                   # 0 = data
```

Without these keys a reader can open the Avro container and still fail to build a
`ManifestEntry`.

### 3. Parquet field ids

Iceberg maps columns **by id**, not by name or position. Every data file is
written with `PARQUET:field_id`, which arrow-rs emits as the physical Parquet
field id. Nested ids mirror `Schema::from_arrow`, so the manifest schema and the
data file agree: a child of top-level column `id` is `id * 100 + position + 1`.
For a vector column at position 3, the column is `3` and the element is `301`.

If a Parquet file has no field ids, readers either fail
(`Parquet file does not have field-ids and the Iceberg table does not have
'schema.name-mapping.default'`) or mis-map columns. We do not rely on a name
mapping.

### 4. Delete files

Iceberg requires a manifest to hold a single content type, so deletes are
written to their own manifest, declared `content = 1`. A snapshot references
exactly one delete manifest: the current delete set, with older delete manifests
dropped rather than carried forward.

- **Position deletes are Parquet**, with the spec's reserved column ids:
  `file_path` = `2147483546`, `pos` = `2147483545`. The spec also permits Avro
  and ORC; Parquet is what the reference writers produce and has the broadest
  client support.
- **Delete rows name the full data-file URI** that the data manifest declares
  (`file:///…` or `s3://…`), as the spec's position-delete schema requires.
  Store-relative paths make readers silently skip the delete.
- **Entry status follows the spec.** A delete file first introduced by this
  snapshot is `ADDED` (status 1) with a `snapshot_id`; one carried over from the
  parent snapshot is `EXISTING` (0) with a null `snapshot_id`. The manifest-list
  entry's `added_data_files_count` / `existing_data_files_count` report the same
  split.
- **Equality deletes stay Avro**, which the spec also allows.
- The engine's own `_manifest/vN.json` remains the internal source of truth for
  the delete list, so the split does not change how BenoStreamDB applies them.

## Reading a BenoStreamDB table with Trino

Configure an Iceberg catalog as usual (`rest`, `hive_metastore`, `glue`, `jdbc`,
or `nessie`) and register the table directory:

```sql
CALL iceberg.system.register_table(
    schema_name    => 'docs',
    table_name     => 'rag_docs',
    table_location => 's3://warehouse/docs/rag_docs');

SELECT id, title FROM iceberg.docs.rag_docs;
```

`register_table` reads `<table_location>/metadata/*.metadata.json` and records
the pointer. The `location` inside the metadata must equal `table_location`,
which it does because the engine always writes the absolute table URI.

**Pass `metadata_file_name` once a table has ten or more metadata versions.**
Trino selects the newest file by lexicographic filename, so `v9.metadata.json`
sorts after `v10.metadata.json`. Read `metadata/version-hint.text` (the engine
maintains it monotonically) and name the file explicitly:

```sql
CALL iceberg.system.register_table(
    schema_name        => 'docs',
    table_name         => 'rag_docs',
    table_location     => 's3://warehouse/docs/rag_docs',
    metadata_file_name => 'v27.metadata.json');
```

`metadata_location => 's3://.../metadata/v27.metadata.json'` is equivalent.

The procedure is enabled by default
(`iceberg.register-table-procedure.enabled=true`).

## Reading the same table with PyIceberg

```python
from pyiceberg.table import StaticTable

table = StaticTable.from_metadata("/warehouse/docs/rag_docs/metadata/v2.metadata.json")
print(table.schema())
print(table.scan().to_arrow().to_pydict())
```

Verified output for a table with `id BIGINT, name VARCHAR, vec DOUBLE[]`:

```
{'id': [1, 2], 'name': ['alpha', 'beta'], 'vec': [[1.0, 0.0], [0.0, 1.0]]}
```

## Verification

- [`tests/integration_test_iceberg_zero_copy.rs`](../../tests/integration_test_iceberg_zero_copy.rs)
  asserts all three layers after a real commit: metadata keys, the Avro user
  metadata on the manifest, and the Parquet field ids (including `301` on the
  vector element).
- The metadata was additionally parsed with **iceberg-core 1.12.0**
  (`TableMetadataParser.fromJson`), the version Trino 484 embeds.
- The spec semantics are asserted directly, without relying on a particular
  client: one `content = 1` manifest per snapshot, `ADDED` vs `EXISTING` entry
  status, a null `snapshot_id` for `EXISTING`, matching manifest-list counters,
  and delete rows that reference declared data-file URIs.
- As one client-level check, a PyIceberg 0.12 scan after `DELETE id = 2`
  returned `{'id': [1, 3, 4, 5]}` — the delete was applied, not merely parsed.
- The connector suites remain green: Trino 482 and 484 (65/65 each, one
  pre-existing skip) and Spark 4.0 (13/13).

## Known limits

- **Equality deletes are Avro.** The spec permits this; Trino and Spark read
  Avro equality deletes. Clients that only implement position deletes (for
  example PyIceberg, in any format) cannot consume them, so such a table may
  need compaction before being read by those clients.
- **Index metadata is BenoStreamDB-specific.** `index_files`, `file_checksum`,
  and the per-column `indexes` array are stored as extra Avro fields / JSON keys.
  Iceberg readers ignore unknown fields, so they are inert for external reads;
  vector/graph acceleration remains a BenoStreamDB feature.
- **Data files are uncompressed** (`Compression::UNCOMPRESSED`). Valid, but larger
  than a typical Iceberg writer's output.
