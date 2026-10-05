# BenoStreamDB Trino Connector

Universal Vector & Metadata Streaming Connector for Trino.

The BenoStreamDB Trino connector enables distributed SQL analytics and vector search over BenoStreamDB tables using Trino. It implements the Trino SPI (Service Provider Interface) and delegates IO and indexing operations to the native Rust engine via JNI.

## Requirements

- Trino 435+
- Java 17+ (or Java 21)
- BenoStreamDB core native library (`libbenostreamdb.so`) on `java.library.path`

## Building

```bash
cd trino-benostreamdb
mvn clean install -DskipTests
```

This generates the plugin ZIP archive in `target/`.

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
   benostream.base-uri=s3://my-bucket/
   benostream.s3.endpoint=http://rustfs:9000
   benostream.s3.access-key=${TRINO_S3_ACCESS_KEY}
   benostream.s3.secret-key=${TRINO_S3_SECRET_KEY}
   ```

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

## Packaging status (not yet production-worthy)

Running the connector in the stock `trinodb/trino` image surfaced two packaging
defects that must be fixed before this is production-ready:

1. **The plugin ZIP is not flat.** The `trino-plugin` Maven packaging emits the
   JARs under a `trino-benostream-<version>/` base directory. Trino's plugin
   loader only scans JARs *directly* in the plugin dir (it does not recurse), so
   the nested layout fails with
   `No service providers of type io.trino.spi.Plugin in the classpath`.
   **Fix:** flatten the ZIP at build time (see `build-connectors.sh`) or set the
   assembly to `includeBaseDirectory=false`.

2. **The native lib is built for the host glibc, not the target.** The connector
   loads `libbenostreamdb.so` via JNI, but:
   - a host `cargo build` links against the host glibc (e.g. 2.43) and fails in
     the Trino image (glibc 2.34) with `version 'GLIBC_2.4x' not found`;
   - a `zig`-linked build still fails with `undefined symbol: __isoc23_sscanf`,
     because the C dependencies (`tikv-jemalloc-sys`, `aws-lc-sys`) are compiled
     against the host's glibc headers;
   - the Python wheel's `.so` is a Python extension and fails with
     `undefined symbol: PyExc_RuntimeError` when loaded as a JNI lib.

   **Fix:** build the cdylib in a `manylinux_2_28` container (glibc 2.28 headers)
   and ship that `.so` alongside the plugin. `build-connectors.sh` should do
   this instead of a host `cargo build`.

Until both are fixed, the connector can be exercised in the benchmark compose
stack via the workarounds in
[`benchmarks/competitors/Dockerfile.trino`](../benchmarks/competitors/Dockerfile.trino)
(flatten the ZIP, stage a manylinux-built `.so`), but the shipped artifacts are
not yet correct.

## Querying

```sql
SELECT * FROM benostreamdb.default.events
WHERE severity = 'ERROR' AND timestamp > NOW() - INTERVAL '1' DAY;
```

### Predicate & Vector Pushdown
The connector translates Trino domain constraints directly into BenoStreamDB index queries (inverted scalar indexes and vector indexes), pruning data before decoding Arrow batches.
