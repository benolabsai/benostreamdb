# ---------------------------------------------------------------------------
# HyperStreamDB Quickstart (All-in-One)
# Runs Search (ES 7.10 + Qdrant) and Flight SQL in a single container
#
# Usage:
#   docker build -t hyperstreamdb .
#   docker run -p 9200:9200 -p 6333:6333 -p 50051:50051 hyperstreamdb
#
# Ports:
#   9200  — Elasticsearch 7.10 compatible REST API
#   6333  — Qdrant compatible REST API
#   50051 — Arrow Flight SQL gRPC (ADBC/JDBC/ODBC)
# ---------------------------------------------------------------------------
FROM rust:1.93-slim AS builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    python3 \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy workspace manifests for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY hyperstreamdb-search/Cargo.toml hyperstreamdb-search/Cargo.toml
COPY hyperstreamdb-flight/Cargo.toml hyperstreamdb-flight/Cargo.toml

# Create stub sources for dependency layer caching and manifest validation
RUN mkdir -p src \
    && echo "pub fn stub() {}" > src/lib.rs \
    && echo "fn main() {}" > build.rs \
    && mkdir -p hyperstreamdb-search/src \
    && echo "fn main() {}" > hyperstreamdb-search/src/main.rs \
    && echo "pub fn stub() {}" > hyperstreamdb-search/src/lib.rs \
    && mkdir -p hyperstreamdb-flight/src \
    && echo "fn main() {}" > hyperstreamdb-flight/src/main.rs \
    && mkdir -p tests/bin benches \
    && echo "fn main() {}" > tests/test_connector_ffi.rs \
    && echo "fn main() {}" > tests/bin/generate_iceberg_manifests.rs \
    && echo "fn main() {}" > tests/bin/verify_iceberg_read_check.rs \
    && echo "fn main() {}" > benches/performance.rs \
    && echo "fn main() {}" > benches/bench_table.rs \
    && cargo build --release -p hyperstreamdb-search -p hyperstreamdb-flight \
    && rm -rf target/release/.fingerprint/hyperstreamdb* \
              target/release/deps/*hyperstreamdb* \
              target/release/deps/libhyperstreamdb* \
              target/release/build/hyperstreamdb* \
              target/release/hypersearch* \
              target/release/hyperstreamdb-flight*

# Copy actual source
COPY build.rs ./
COPY src ./src
COPY hyperstreamdb-search/src ./hyperstreamdb-search/src
COPY hyperstreamdb-flight/src ./hyperstreamdb-flight/src

# Build both binaries
RUN cargo build --release -p hyperstreamdb-search -p hyperstreamdb-flight


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y \
    libssl3t64 \
    ca-certificates \
    curl \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r hyperstream && useradd -r -g hyperstream -m hyperstream

WORKDIR /app

# Copy both service binaries
COPY --from=builder /app/target/release/hypersearch /usr/local/bin/
COPY --from=builder /app/target/release/hyperstreamdb-flight /usr/local/bin/

# Copy entrypoint
COPY docker/quickstart-entrypoint.sh /usr/local/bin/quickstart-entrypoint.sh

# Create default data directory
RUN mkdir -p /home/hyperstream/.hyperstreamdb/search \
    && chown -R hyperstream:hyperstream /home/hyperstream/.hyperstreamdb

# ES 7.10 API + Qdrant API + Flight SQL gRPC + Prometheus metrics
EXPOSE 9200 6333 50051 9090

# Health check against ES cluster health endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:9200/_cluster/health || exit 1

USER hyperstream

# Default environment
ENV HYPERSEARCH_BIND=0.0.0.0
ENV HYPERSEARCH_PORT=9200
ENV QDRANT_BIND=0.0.0.0
ENV QDRANT_PORT=6333
ENV HYPERSEARCH_AUTO_REFRESH_SECS=5
ENV RUST_LOG=info

ENTRYPOINT ["quickstart-entrypoint.sh"]
CMD []
