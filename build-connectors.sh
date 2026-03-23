#!/bin/bash
set -e

# Configuration
MAVEN_VERSION="3.9.6"
MAVEN_DIR=".maven"
MVN_BIN="${MAVEN_DIR}/apache-maven-${MAVEN_VERSION}/bin/mvn"
JAVA_DIR=".java"
JDK_21_DIR="${JAVA_DIR}/jdk-21"

# Parse arguments
CARGO_FEATURES=""
ARTIFACT_SUFFIX=""

for arg in "$@"; do
    case $arg in
        --cuda)
            echo "Enabling CUDA support..."
            CARGO_FEATURES="cuda"
            ARTIFACT_SUFFIX="-cuda"
            # Check for nvcc
            if ! command -v nvcc &> /dev/null; then
                echo "WARNING: nvcc not found. CUDA build will likely fail."
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
    "$MVN_BIN" clean package -P"$profile" $extra_args -DskipTests -f "$project_dir/pom.xml"
}

# Find Java homes
JAVA_17_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
JAVA_21_HOME="$(pwd)/${JDK_21_DIR}"

# Create output directory
mkdir -p connector-artifacts

# --- Build Native Library ---
echo "--- Building Native Core ---"
if [ -n "$CARGO_FEATURES" ]; then
    cargo build --release --features "$CARGO_FEATURES"
else
    cargo build --release
fi
LIB_PATH="target/release/libhyperstreamdb.so"
if [ ! -f "$LIB_PATH" ]; then
    # Fallback for macOS or potential naming
    LIB_PATH="target/release/libhyperstreamdb.dylib"
fi

# Function to prepare resources
prepare_resources() {
    local target_dir=$1
    mkdir -p "$target_dir/src/main/resources"
    cp "$LIB_PATH" "$target_dir/src/main/resources/"
}

# --- Spark Connector Matrix ---
echo "--- Building Spark Connectors ---"
prepare_resources "spark-hyperstream"
for java_version in "17" "21"; do
    java_home_var="JAVA_${java_version}_HOME"
    java_home="${!java_home_var}"
    
    for spark_version in "3.5" "4.0"; do
        build_with_java "$java_home" "spark-$spark_version,java-$java_version" "" "spark-hyperstream"
        cp spark-hyperstream/target/spark-hyperstream-*.jar "connector-artifacts/spark-hyperstream-spark-${spark_version}-java-${java_version}${ARTIFACT_SUFFIX}.jar"
    done
done

# --- Trino Connector Matrix ---
echo "--- Building Trino Connectors ---"
prepare_resources "trino-hyperstream"
for java_version in "17" "21"; do
    java_home_var="JAVA_${java_version}_HOME"
    java_home="${!java_home_var}"
    
    build_with_java "$java_home" "java-$java_version" "" "trino-hyperstream"
    # For Trino, the main JAR is in target/ but the ZIP contains all deps
    cp trino-hyperstream/target/trino-hyperstream-0.1.0-SNAPSHOT.jar "connector-artifacts/trino-hyperstream-java-${java_version}${ARTIFACT_SUFFIX}.jar"
    cp trino-hyperstream/target/trino-hyperstream-0.1.0-SNAPSHOT.zip "connector-artifacts/trino-hyperstream-java-${java_version}${ARTIFACT_SUFFIX}.zip"
done

echo "Build complete. Artifacts are in connector-artifacts/"
ls -lh connector-artifacts/
