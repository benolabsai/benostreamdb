#!/bin/bash
set -e

# Configuration
MAVEN_VERSION="3.9.6"
MAVEN_DIR=".maven"
MVN_BIN="${MAVEN_DIR}/apache-maven-${MAVEN_VERSION}/bin/mvn"
JAVA_DIR=".java"
JDK_21_DIR="${JAVA_DIR}/jdk-21"

# Derive the connector version from the core engine's Cargo.toml (single source).
CORE_VERSION="$(grep -m1 '^version = "' Cargo.toml | sed -E 's/^version = "([^"]+)".*/\1/')"
echo "Core version: ${CORE_VERSION}"

# Parse arguments
#
# The JNI entry points in src/core/ffi.rs are behind the `java` Cargo feature.
# Building without it produces a library that exports ZERO Java_com_benostreamdb_*
# symbols, so every connector fails at its first native call. It is therefore
# not optional here.
CARGO_FEATURES="java"
ARTIFACT_SUFFIX=""

for arg in "$@"; do
    case $arg in
        --cuda)
            echo "Enabling CUDA support..."
            CARGO_FEATURES="java,cuda"
            ARTIFACT_SUFFIX="-cuda"
            # Check for nvcc
            if ! command -v nvcc &> /dev/null; then
                echo "WARNING: nvcc not found. CUDA build will likely fail."
            fi
            shift
            ;;
        --metal)
            echo "Apple Metal support is natively enabled on macOS builds (default on Apple Silicon)."
            if [[ "$(uname -s)" != "Darwin" ]]; then
                echo "WARNING: Apple Metal is only available on macOS / Apple Silicon. Building standard artifact."
            fi
            shift
            ;;
    esac
done

# Ensure Maven is available
if [ ! -f "$MVN_BIN" ]; then
    echo "Installing Maven ${MAVEN_VERSION}..."
    mkdir -p "$MAVEN_DIR"
    curl -L -o "${MAVEN_DIR}/maven.tar.gz" "https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz"
    tar -xzf "${MAVEN_DIR}/maven.tar.gz" -C "$MAVEN_DIR"
    rm "${MAVEN_DIR}/maven.tar.gz"
fi

# Ensure JDK 21 is available (since system only has JRE 21)
if [ ! -d "$JDK_21_DIR" ]; then
    echo "Installing portable JDK 21..."
    mkdir -p "$JAVA_DIR"
    curl -L -o "${JAVA_DIR}/jdk21.tar.gz" "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse?project=jdk"
    mkdir -p "$JDK_21_DIR"
    tar -xzf "${JAVA_DIR}/jdk21.tar.gz" -C "$JDK_21_DIR" --strip-components=1
    rm "${JAVA_DIR}/jdk21.tar.gz"
fi

# Function to build with specific Java version
build_with_java() {
    local java_home=$1
    local profile=$2
    local extra_args=$3
    local project_dir=$4

    echo "Building $project_dir with $profile using JAVA_HOME=$java_home"
    export JAVA_HOME=$java_home
    "$JAVA_HOME/bin/java" -version
    
    local release_version="17"
    if [[ "$profile" == *"java-21"* ]]; then
        release_version="21"
    fi
    
    echo "Running Maven for $project_dir with release $release_version"
    export MAVEN_OPTS="-Djava.release=$release_version -Dmaven.compiler.release=$release_version"
    # `maven.test.skip` (not just `skipTests`) also skips *compiling* test
    # sources: the Spark Java interop test references Scala objects and does not
    # compile under the packaging build's phase ordering.
    "$MVN_BIN" clean package -P"$profile" $extra_args -Dmaven.test.skip=true -Drevision="$CORE_VERSION" -f "$project_dir/pom.xml"
}

# Find Java homes
JAVA_17_HOME="${JAVA_17_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
JAVA_21_HOME="$(pwd)/${JDK_21_DIR}"

# The Trino 482+ SPI is compiled for Java 23 (class file version 67), so the
# connector must be *compiled* with a JDK >= 23. A JDK 21 compiler cannot read
# the SPI and fails with "class file has wrong version 67.0, should be 65.0".
# Prefer a system JDK 23+. The bytecode target stays 23 (the pom default), so no
# java.release override is passed for this module.
if [ -z "${JAVA_TRINO_HOME:-}" ]; then
    JAVA_TRINO_HOME=""
    for cand in /usr/lib/jvm/java-25-openjdk-amd64 /usr/lib/jvm/java-24-openjdk-amd64 /usr/lib/jvm/java-23-openjdk-amd64; do
        if [ -x "$cand/bin/javac" ]; then JAVA_TRINO_HOME="$cand"; break; fi
    done
fi
if [ -z "$JAVA_TRINO_HOME" ]; then
    echo "ERROR: the Trino connector's SPI requires JDK >= 23 to compile." >&2
    echo "       Install a JDK 23+ (e.g. /usr/lib/jvm/java-25-openjdk-amd64) and re-run." >&2
    exit 1
fi
echo "Trino connector will be compiled with: $JAVA_TRINO_HOME"

# Create output directory
mkdir -p connector-artifacts

# --- Build Native Library ---
echo "--- Building Native Core ---"
echo "Cargo features: $CARGO_FEATURES"
cargo build --release --features "$CARGO_FEATURES"
LIB_PATH="target/release/libbenostreamdb.so"
if [ ! -f "$LIB_PATH" ]; then
    # Fallback for macOS or potential naming
    LIB_PATH="target/release/libbenostreamdb.dylib"
fi

# Hard gate: shipping a JVM connector whose native library lacks the JNI
# exports is a silent, total failure at runtime (UnsatisfiedLinkError on the
# first metadata call), so verify the symbols before anything is packaged.
if ! nm -D --defined-only "$LIB_PATH" 2>/dev/null | grep -q 'Java_com_benostreamdb'; then
    echo "ERROR: $LIB_PATH exports no JNI symbols." >&2
    echo "       The 'java' Cargo feature must be enabled (CARGO_FEATURES='$CARGO_FEATURES')." >&2
    exit 1
fi
echo "JNI symbol check passed: $(nm -D --defined-only "$LIB_PATH" | grep -c 'Java_com_benostreamdb') exported entry points"

# Function to prepare resources
prepare_resources() {
    local target_dir=$1
    mkdir -p "$target_dir/src/main/resources"
    cp "$LIB_PATH" "$target_dir/src/main/resources/"
}

# --- Spark Connector Matrix ---
echo "--- Building Spark Connectors ---"
prepare_resources "spark-benostreamdb"
for java_version in "17" "21"; do
    java_home_var="JAVA_${java_version}_HOME"
    java_home="${!java_home_var}"
    
    for spark_version in "3.5" "4.0"; do
        build_with_java "$java_home" "spark-$spark_version,java-$java_version" "" "spark-benostreamdb"
        cp spark-benostreamdb/target/spark-benostream-*.jar "connector-artifacts/spark-benostream-spark-${spark_version}-java-${java_version}${ARTIFACT_SUFFIX}.jar"
    done
done

# --- Trino Connector Matrix ---
echo "--- Building Trino Connectors ---"
# NOTE: unlike Spark, the Trino connector does NOT bundle the native lib in the
# plugin JAR. Trino loads it from `java.library.path` (see Dockerfile.trino),
# and a host-built lib bundled in the JAR would shadow the manylinux one.
rm -f trino-benostreamdb/src/main/resources/libbenostreamdb.so
# Trino publishes no LTS and does not keep the connector SPI source- or
# binary-compatible between releases, so one artifact is built per supported
# band (see plans/trino_version_bands.md). Band `e` is current; band `d` is
# current-minus.
for trino_target in "482:d" "484:e"; do
    trino_version="${trino_target%%:*}"
    trino_band="${trino_target##*:}"
    echo "Building Trino connector for $trino_version (band $trino_band)"
    export JAVA_HOME="$JAVA_TRINO_HOME"
    "$MVN_BIN" clean package -Dmaven.test.skip=true -Drevision="$CORE_VERSION" \
        -Dtrino.version="$trino_version" -Dtrino.band="$trino_band" \
        -f trino-benostreamdb/pom.xml
    # For Trino, the main JAR is in target/ but the ZIP contains all deps.
    # The artifact version tracks the core engine version (`${revision}`).
    cp "trino-benostreamdb/target/trino-benostream-${CORE_VERSION}.jar" "connector-artifacts/trino-benostream-${trino_version}${ARTIFACT_SUFFIX}.jar"
    # Flatten the plugin ZIP. Trino's plugin loader only scans JARs *directly*
    # in the plugin dir (it does not recurse), but the `trino-plugin` packaging
    # nests them under `trino-benostream-<version>/`, which fails with
    # "No service providers of type io.trino.spi.Plugin in the classpath".
    flat_zip="$(pwd)/connector-artifacts/trino-benostream-${trino_version}${ARTIFACT_SUFFIX}.zip"
    tmp_zip="$(mktemp -d)"
    unzip -q "trino-benostreamdb/target/trino-benostream-${CORE_VERSION}.zip" -d "$tmp_zip"
    ( cd "$tmp_zip/trino-benostream-${CORE_VERSION}" && zip -q -r "$flat_zip" . )
    rm -rf "$tmp_zip"
done

echo "Build complete. Artifacts are in connector-artifacts/"
ls -lh connector-artifacts/
