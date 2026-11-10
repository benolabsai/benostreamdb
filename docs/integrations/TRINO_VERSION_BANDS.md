# Trino version bands and connector compatibility

Status: implemented and tested on Trino 482 and 484 (both bands green).

## Why this document exists

Trino publishes a release roughly every two weeks, offers **no LTS**, and makes
**no binary or source compatibility promise** for the connector SPI. A plugin
built against one release is expected to be rebuilt for the next. The practical
question is therefore not "does it work on every version", but "how many
distinct source shapes must we maintain, and which ones do real deployments
run".

This document records the answer from measurement, not from documentation.

## How the bands were measured

Boundaries were read directly from the published `trino-spi` jars:

```bash
for v in 468 470 472 474 476 478 480 482 484; do
  curl -sO https://repo1.maven.org/maven2/io/trino/trino-spi/$v/trino-spi-$v.jar
  echo "=== $v ==="
  javap -cp trino-spi-$v.jar io.trino.spi.connector.ConnectorMetadata | grep finishMerge
  javap -cp trino-spi-$v.jar io.trino.spi.connector.ConnectorPageSource | grep getNext
  javap -cp trino-spi-$v.jar io.trino.spi.connector.ConnectorSplitManager | grep getSplits
  javap -cp trino-spi-$v.jar io.trino.spi.connector.ConnectorPageSinkProvider | grep createMergeSink
done
```

## Measured result

| Band | Versions | `ConnectorPageSource` | `getSplits` filter arg | `finishMerge` returns | `createMergeSink` |
|------|----------|----------------------|------------------------|-----------------------|-------------------|
| A | <= 473 | `getNextPage` | `DynamicFilter` | `void` | 5 args |
| B | 474-477 | both (deprecation window) | `DynamicFilter` | `void` | 5 args |
| C | 478-481 | `getNextSourcePage` | `DynamicFilter` | `void` | 5 args |
| D | 482-483 | `getNextSourcePage` | `Set<ColumnHandle>` | `void` | 5 args |
| E | 484+ | `getNextSourcePage` | `Set<ColumnHandle>` | `Optional<ConnectorOutputMetadata>` | 6 args (`MemoryContext`) |

Two things stand out.

1. Boundaries cluster at **474, 478 and 482** — one every four releases, about
   every eight weeks. Trino follows a consistent add-default, deprecate,
   delete cadence with a four-release window: `getNextSourcePage` appears in
   474 and `getNextPage` is gone in 478.
2. **484 breaks the rhythm** because it bundles a return-type sweep
   (`void` -> `Optional`). A return type cannot be deprecation-windowed at all,
   so that change lands with no migration path for the old signature.

## Supported bands

We ship two artifacts, which covers essentially every real deployment:

| Band | Versions | Role | Coverage |
|------|----------|------|----------|
| `d` | 482-483 | current-minus | compile + full test suite |
| `e` | 484+ | current and forward | compile + full test suite |

Band `e` is the default. The 468 claim in older docs has been dropped: it is
band A, four breaking changes behind, and no longer supported.

### Build

```bash
# Band e (default), Trino 484
mvn package

# Band d, Trino 482
mvn package -Dtrino.version=482 -Dtrino.band=d

# Deprecated aliases kept for existing scripts
mvn package -P trino-482
mvn package -P trino-484
```

### CI matrix

```bash
for v in 482 484; do
  band=d; [ "$v" = 484 ] && band=e
  mvn -B test -Dtrino.version=$v -Dtrino.band=$band
done
```

## How the split is implemented

Only the members that genuinely differ are band-specific. Everything else stays
in one shared source tree so logic can never silently diverge.

| Differing member | Shared base | Band `d` shell | Band `e` shell |
|------------------|-------------|----------------|----------------|
| `finishMerge` | `AbstractBenoStreamDBMetadata` | returns `void` | returns `Optional<ConnectorOutputMetadata>` |
| `createMergeSink` | `AbstractBenoStreamDBPageSinkProvider` | 5 args | 6 args with `MemoryContext` |

Band shells live in `src/main/java-trino-d` and `src/main/java-trino-e`, and are
wired in by `build-helper-maven-plugin` using the `trino.band` property.

### Verified non-seams

These were checked and are identical in bands D and E, so they stay shared:

- `ConnectorSplitManager.getSplits(..., Set<ColumnHandle>, Constraint)`
- `ConnectorSplitSource.getNextBatch(int, DynamicFilterSnapshot)` — note the
  1-argument form used to exist; 468-era test code calling `getNextBatch(1000)`
  does not compile on either supported band
- `ConnectorPageSourceProvider.createPageSource` — both the 8 and 9 argument
  overloads are still declared, so either can be implemented
- `ConnectorSplit` — carries `isRemotelyAccessible`, `getAddresses`,
  `getAffinityKey`, `getSplitWeight`, `getRetainedSizeInBytes`. The
  `getSplitInfo()` accessor asserted by the old test no longer exists in either
  band, and that test was removed rather than pinned.
- `ConnectorSplitManager.getSplits` 5-argument form with `Constraint`
- `ConnectorPageSinkProvider.createPageSink` (both overloads)
- `Procedure` argument registration rules (see below)

## Runtime requirements

### JVM flags

Trino 484's `io.trino.simd.BlockEncodingSimdSupport` references
`jdk.incubator.vector.VectorShape`. That class exists on JDK 25 but only inside
the `jdk.incubator.vector` incubator module, which is **not resolved by
default**. Without the flag the engine dies during Guice provisioning with:

```
java.lang.NoClassDefFoundError: jdk/incubator/vector/VectorShape
	at io.trino.simd.BlockEncodingSimdSupport.detectSimd
```

Trino's own launcher adds the module. Any embedder (surefire, a test harness, a
custom launcher) must match:

```
--add-modules=jdk.incubator.vector
--add-opens=java.base/java.nio=ALL-UNNAMED   # Arrow Java memory access
```

### Java version

The SPI is compiled for Java 23, so build and run with JDK 23 or newer.
`java.release=23`. JDK 25 is what CI uses.

### Procedure arguments

Two independent validations apply to a `VARCHAR` procedure argument, and they
pull in opposite directions:

- `Procedure.Argument` validates a declared **default value** against
  `type.getJavaType()`, which is `Slice` for varchar. Defaults must be
  `Slices.utf8Slice(...)`.
- the CALL execution path supplies **`String`** values and rejects a method
  handle whose parameter type is `Slice` with
  `Argument 'TABLE' has invalid type io.airlift.slice.Slice (expected java.lang.String)`.

So procedure methods take `String`, and their registered defaults are `Slice`.
Both requirements are satisfied simultaneously; see `BenoStreamProcedures`.

### Test stack

JUnit is pinned to **6.1.3** via the `junit-bom`. `trino-testing` and
`trino-main` pull JUnit 6 transitively, so declaring JUnit 5 leaves
`junit-platform-engine` at 1.11.4 beside a JUnit 6 API that references
`org.junit.platform.engine.OutputDirectoryCreator`, and the forked JVM fails to
load. Surefire is 3.6.0; 3.2.5 predates Platform 6 and mis-detects it. The
`junit-vintage-engine` is present because `TrinoConnectorTest` is still JUnit 4,
and without it surefire silently runs zero tests from that class.

Jackson follows the Trino release's airbase pin (2.22.3 for 484). Since Jackson
2.20, `jackson-annotations` dropped the patch component, so the valid
coordinate is `2.22`, not `2.22.3`.

## Real-catalog join testing

`TestCrossCatalogJoin` runs inside a real `DistributedQueryRunner` with a second
first-party catalog and asserts three things:

- a scalar join between a BenoStreamDB table and a foreign catalog
- a `system.vector_search` result joined directly against a foreign catalog,
  with no intermediate materialisation
- a cross-catalog aggregate

### Why tpch and not PostgreSQL for the in-process tier

`trino-postgresql` is **not published to Maven Central past 476**; the JDBC
connectors now ship only inside the server tarball. Pinning
`trino-postgresql:476` against a 482/484 engine would mean testing a band A/B
plugin against a band D/E core, which cannot link. `trino-tpch`, by contrast, is
published at every release and is pinned to `${trino.version}`, so the test
always exercises a plugin built for exactly the engine under test.

A real PostgreSQL join test therefore belongs in a **server-distribution tier**:
download the Trino server tarball for the target version, use the bundling
`plugin/postgresql` directory via `DistributedQueryRunner.installPlugin(Path)`
or run the server and reach it over JDBC, and point it at a Postgres container.
That tier is not yet implemented.

### Iceberg zero-copy join

BenoStreamDB writes a custom `_manifest/v2.json` layout, not standard Iceberg
metadata. Trino's Iceberg connector reads standard metadata only, so it cannot
join our tables directly today. Zero-copy join with a real Iceberg catalog
requires one of:

1. an external-table overlay in the Rust engine (register a table whose storage
   is an existing Iceberg table), or
2. emitting standard Iceberg metadata so the same files are readable by both
   Trino's Iceberg connector and our engine.

Neither is implemented. Until then, zero-copy claims should be limited to
"reads the Parquet files in place" and must not be described as Iceberg catalog
interoperability.

## Scalar functions: how they are contributed (solved)

Spark registers seven SQL UDFs. Trino now exposes all seven, but **not** through
the connector, and the difference matters.

### The wrong path

`ConnectorMetadata` exposes `listFunctions`, `getFunctions` and
`getFunctionMetadata`, plus `Connector.getFunctionProvider()`. Implementing all
of them compiles and looks correct, but **Trino never calls `listFunctions`**
for a catalog created at runtime (instrumenting it produced a call count of 0),
so nothing is ever enumerated and every call fails with
`Function '<name>' not registered`. No amount of FunctionId or naming fiddling
fixes that, which is why the earlier investigation looked so erratic.

A red herring along the way: `cosine_distance` and `dot_product` *did* resolve.
They were not ours. Trino 484 ships both as built-ins, together with
`euclidean_distance` and `hamming_distance`. That was confirmed by the plugin
failing to install with
`Function already registered: cosine_distance(array(double),array(double)):double`.

### The correct path

Per Trino's developer guide, scalar functions are contributed by the **plugin**:

```java
@Override
public Set<Class<?>> getFunctions() {
    return Set.of(BenoStreamScalarFunctions.class);
}
```

The class is annotated with `@ScalarFunction`, `@Description` and `@SqlType`,
and the engine derives the signatures. Two non-obvious requirements:

- An `array(...)` argument arrives as a Trino **`Block`**, not a `List`.
  Declaring `List` fails at specialization with
  `java types do not match`.
- Plugin functions land in the global `system.builtin` namespace, so they
  **cannot shadow a built-in** and cannot be scoped to a catalog or schema.
  There is no per-schema overloading in Trino.

### Resulting parity map

| Spark UDF | Trino | Notes |
|-----------|-------|-------|
| `cosine_distance` | built-in | cannot be overridden |
| `dot_product` | built-in | cannot be overridden |
| `l2_distance` | ours | equivalent to built-in `euclidean_distance` |
| `vector_distance` | ours | metric dispatch; `dot` negated, as in Spark |
| `sparse_dot_product` | ours | merge-join over index arrays |
| `hybrid_score` | ours | alpha clamped to [0,1] |
| `reciprocal_rank_fusion` | ours | sums 1/(k+rank) over positive ranks |

`TestBenoStreamDBQueries` asserts that the five registered functions are visible
and evaluate correctly, and pins the semantics of the built-ins (including that
opposing vectors yield 2.0 for cosine, i.e. no cap at 1).

### Getting engine semantics on our data

Because a built-in cannot be overridden per catalog, engine-side scoring for our
tables comes from pushdown instead: `system.vector_search` and
`system.hybrid_search` return `distance` / `rrf_score` computed inside the Rust
session. A hand-written `cosine_distance(col, ...)` on a BenoStreamDB table is
evaluated by Trino. If matching Spark bit-for-bit matters for that case, the
path is a custom Trino type (`Plugin.getTypes()`) so the function has a distinct
signature rather than a shadowed one.

### Test-harness note

`add_index` registers an index and builds it on a **detached background task**.
A query issued immediately afterwards can observe no index, which made the
hybrid test flaky. Tests call `build_index` explicitly; the two-phase contract
should be documented for users as well.

## Refresh procedure

When a new Trino release appears:

1. Check `https://repo1.maven.org/maven2/io/trino/trino-spi/maven-metadata.xml`
   for `<release>`.
2. Diff the SPI seams with the `javap` loop above.
3. If nothing moved, add the version to the existing band and to the CI matrix.
4. If a seam moved, add a new band directory with the differing shell only.
5. Never claim a version is supported because it compiled once; every band in
   the matrix must run the full suite.
