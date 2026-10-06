# BenoStreamDB Trino Connector

Universal Vector & Metadata Streaming Connector for Trino.

The BenoStreamDB Trino connector enables distributed SQL analytics and vector search over BenoStreamDB tables using Trino. It implements the Trino SPI (Service Provider Interface) and delegates IO and indexing operations to the native Rust engine via JNI.

## Requirements

- Trino 468 (the connector is compiled against the Trino 468 SPI)
- A JDK **23+** to *compile* (the 468 SPI is Java 23 bytecode). The emitted
  bytecode still targets 17/21 and runs on the Trino image's Java 23.
- BenoStreamDB core native library (`libbenostreamdb.so`) on `java.library.path`
- `libstdc++.so.6` in the Trino image (Apache Arrow's `arrow-c-data` JNI lib
  needs it; the stock Trino image is a stripped RHEL UBI without a C++ runtime)

## Building

```bash
cd trino-benostreamdb
# JAVA_HOME must point at a JDK >= 23 (e.g. /usr/lib/jvm/java-25-openjdk-amd64)
JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64 mvn clean package -DskipTests
```

This generates the plugin ZIP archive in `target/`. The top-level
[`build-connectors.sh`](../build-connectors.sh) builds it with the right JDK and
flattens the ZIP for you.

## Installation & Configuration

1. **Extract Plugin**:
   ```bash
   mkdir -p /usr/lib/trino/plugin/benostream
   unzip target/trino-*-SNAPSHOT.zip -d /usr/lib/trino/plugin/benostream
   ```

2. **Configure Catalog**:
   Sample catalog configurations are provided in [`etc/catalog/`](etc/catalog/):
   ```properties
   connector.name=benostreamdb
   benostream.warehouse=s3://my-bucket/
   benostream.s3.endpoint=http://rustfs:9000
   benostream.s3.access-key=${TRINO_S3_ACCESS_KEY}
   benostream.s3.secret-key=${TRINO_S3_SECRET_KEY}
   ```

   > **Config-key note.** The connector factory reads `benostream.warehouse`
   > (default `s3://default`); the older `benostream.base-uri` key is ignored.
   > Schemas are subdirectories of the warehouse, so
   > `benostreamdb.default.events` resolves to `<warehouse>/default/events`.

3. **Local Docker Environment**:
   The [`etc/`](etc/) directory contains the complete Trino configuration (`config.properties`, `jvm.config`, `node.properties`, `catalog/`) for containerized testing.

## Running in the latest Trino container (one command)

The connector needs the plugin ZIP **and** `libbenostreamdb.so` on
`java.library.path`. The native lib must be built for the *Trino image's* glibc
(Debian, ~2.34) — a host `cargo build` links against the host glibc and fails
with `version 'GLIBC_2.4x' not found`. The helper script handles both:

```bash
# Builds the manylinux wheel (optional), stages the .so, and builds the image.
benchmarks/competitors/build_trino_connector.sh [--build-wheel]

# Then run it (the benchmark compose service does this automatically):
docker compose -f benchmarks/competitors/docker-compose.bench.yml up -d trino
```

The image is [`benchmarks/competitors/Dockerfile.trino`](../benchmarks/competitors/Dockerfile.trino):
it extends `trinodb/trino`, flattens the plugin ZIP into
`/usr/lib/trino/plugin/benostream` (Trino's loader does not scan the ZIP's
nested `trino-benostream-<version>/` directory), and installs the native lib to
`/usr/lib/trino/lib` (added to `java.library.path` in `jvm.config`).

## Packaging status

Running the connector in the stock `trinodb/trino:468` image surfaced a series of
packaging defects. All are now fixed:

1. **SPI version mismatch.** The connector was compiled against Trino SPI 435
   while the image runs 468. Trino 468 changed several `ConnectorMetadata`
   signatures (`getTableHandle` gained two `Optional<ConnectorTableVersion>`
   args; `beginCreateTable`/`finishInsert`/`finishMerge` gained args), so the
   old overrides were dead code and Trino reported
   `ConnectorMetadata getTableHandle() is not implemented`.
   **Fix:** target SPI 468 and update the overrides. The 468 SPI is Java 23
   bytecode, so the build now uses a JDK ≥ 23 (see `build-connectors.sh`).

2. **The plugin ZIP is not flat.** The `trino-plugin` Maven packaging emits the
   JARs under a `trino-benostream-<version>/` base directory. Trino's plugin
   loader only scans JARs *directly* in the plugin dir (it does not recurse), so
   the nested layout fails with
   `No service providers of type io.trino.spi.Plugin in the classpath`.
   **Fix:** `build-connectors.sh` flattens the ZIP, and `Dockerfile.trino`
   flattens either layout at image-build time.

3. **The native lib was built for the host glibc, not the target.** A host
   `cargo build` links against the host glibc (e.g. 2.43) and fails in the Trino
   image (glibc 2.34) with `version 'GLIBC_2.4x' not found`; a `zig`-linked
   build still fails with `undefined symbol: __isoc23_sscanf`; and the Python
   wheel's `.so` is a Python extension that fails with
   `undefined symbol: PyExc_RuntimeError` when loaded as a JNI lib.
   **Fix:** build the cdylib in a `manylinux_2_28` container and stage that
   `.so` (see `build_trino_connector.sh`). The `arrow/pyarrow` feature is gated
   behind the `python` feature so the JNI build does not link libpython.

4. **The transaction handle was not serializable.** Trino ships the
   `ConnectorTransactionHandle` to workers inside the serialized
   `TaskUpdateRequest`; the empty handle class failed with
   `No serializer found for class ...BenoStreamDBTransactionHandle`.
   **Fix:** make it an enum (`BenoStreamDBTransactionHandle.INSTANCE`).

5. **Arrow's JNI lib needs `libstdc++.so.6`.** `arrow-c-data`'s JNI library is
   extracted from the JAR and loaded at query time; the stripped RHEL UBI Trino
   image has no C++ runtime, so it failed with
   `libstdc++.so.6: cannot open shared object file`.
   **Fix:** `Dockerfile.trino` copies `libstdc++` from a UBI stage.

6. **Arrow 14's `MemoryUtil` needs deep reflection.** On Java 23 it fails with
   `Could not initialize class org.apache.arrow.memory.util.MemoryUtil`.
   **Fix:** `jvm.config` opens `java.base/java.nio` (and friends) to the
   unnamed module.

## Verified surface area

Exercised end-to-end through Trino 468 against a BenoStreamDB Iceberg table
(23/23 checks pass):

| Area | Statements |
|---|---|
| Metadata | `SHOW SCHEMAS`, `SHOW TABLES`, `SHOW COLUMNS`, `DESCRIBE`, `SHOW CREATE TABLE`, `information_schema.tables` |
| Read | `SELECT *`, `count(*)`, `ORDER BY`, `GROUP BY`, aggregates |
| Predicate pushdown | `=`, `<`, `IN`, `BETWEEN` (translated to engine index queries) |
| Plan | `EXPLAIN` |
| Write | `CREATE TABLE AS SELECT`, `INSERT … VALUES`, `INSERT … SELECT`, `DELETE` |
| DDL | `CREATE SCHEMA`, `DROP TABLE` |
| Row-level | `MERGE` — `UPDATE`, `DELETE`, `INSERT`, multi-clause and conditional forms |

### MERGE support

The connector implements Trino's full MERGE surface (the grammar is Trino's, not
Iceberg's): `WHEN MATCHED THEN UPDATE`, `WHEN MATCHED THEN DELETE`,
`WHEN NOT MATCHED THEN INSERT`, multiple `WHEN` clauses, and `AND` conditions.
It uses the `DELETE_ROW_AND_INSERT_ROW` paradigm with a hidden `_bsdb_row_id`
column (sourced from the primary key), the SPI `MergePage` helper, and
delete-before-insert so an `UPDATE` reuses the same key.

Verified end-to-end (values checked, not just statement success):

| Case | Result |
|---|---|
| `WHEN MATCHED THEN UPDATE SET value = s.value` | new value written |
| `WHEN MATCHED THEN DELETE` | row removed |
| `WHEN NOT MATCHED THEN INSERT (id, value) VALUES (…)` | row inserted with the source values |
| multi-clause (`UPDATE` + `INSERT`) | both applied |
| `WHEN MATCHED AND t.value < 5 THEN UPDATE` | skipped when the condition is false |

The hidden row-id column carries the primary key's own type, so numeric,
`VARCHAR`, and boolean keys all work (the delete predicate is quoted for
strings).

> **Implementation note.** Trino's planner compares `ColumnHandle`s across
> separate `getColumnHandles()` calls (e.g. `mergeCaseSetColumns.indexOf(...)` in
> `QueryPlanner.planMerge`), so every connector handle type implements
> `equals`/`hashCode`. Without it the planner silently falls back to the
> pre-update target row for `UPDATE` and to `NULL` for `INSERT`.

> **`DROP TABLE` note.** `SHOW TABLES` derives its listing from the objects
> actually present, so a dropped table disappears immediately even though a
> local filesystem keeps the now-empty directory behind.

## Querying

```sql
SELECT * FROM benostreamdb.default.events
WHERE severity = 'ERROR' AND timestamp > NOW() - INTERVAL '1' DAY;
```

### Predicate & Vector Pushdown
The connector translates Trino domain constraints directly into BenoStreamDB index queries (inverted scalar indexes and vector indexes), pruning data before decoding Arrow batches.
