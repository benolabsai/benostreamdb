# ---------------------------------------------------------------------------
# BenoStreamDB Quickstart (All-in-One)
# Runs Search (ES 7.10 + Qdrant) and Flight SQL in a single container
#
# Usage:
#   docker build -t benostreamdb .
#   docker run -p 9200:9200 -p 6333:6333 -p 50051:50051 benostreamdb
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
COPY benostreamdb-search/Cargo.toml benostreamdb-search/Cargo.toml
COPY benostreamdb-flight/Cargo.toml benostreamdb-flight/Cargo.toml

# Create stub sources for dependency layer caching and manifest validation
RUN mkdir -p src \
    && echo "pub fn stub() {}" > src/lib.rs \
    && echo "fn main() {}" > build.rs \
    && mkdir -p benostreamdb-search/src \
    && echo "fn main() {}" > benostreamdb-search/src/main.rs \
    && echo "pub fn stub() {}" > benostreamdb-search/src/lib.rs \
    && mkdir -p benostreamdb-flight/src \
    && echo "fn main() {}" > benostreamdb-flight/src/main.rs \
    && mkdir -p tests/bin benches \
    && echo "fn main() {}" > tests/test_connector_ffi.rs \
    && echo "fn main() {}" > tests/bin/generate_iceberg_manifests.rs \
    && echo "fn main() {}" > tests/bin/verify_iceberg_read_check.rs \
    && echo "fn main() {}" > benches/performance.rs \
    && echo "fn main() {}" > benches/bench_table.rs \
    && cargo build --release -p benostreamdb-search -p benostreamdb-flight \
    && rm -rf target/release/.fingerprint/benostreamdb* \
              target/release/deps/*benostreamdb* \
              target/release/deps/libbenostreamdb* \
              target/release/build/benostreamdb* \
              target/release/bsdb-search* \
              target/release/benostreamdb-flight*

# Copy actual source
COPY build.rs ./
COPY src ./src
COPY benostreamdb-search/src ./benostreamdb-search/src
COPY benostreamdb-flight/src ./benostreamdb-flight/src

# Build both binaries
RUN cargo build --release -p benostreamdb-search -p benostreamdb-flight


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

RUN groupadd -r benostream && useradd -r -g benostream -m benostream

WORKDIR /app

# Copy both service binaries
COPY --from=builder /app/target/release/bsdb-search /usr/local/bin/
COPY --from=builder /app/target/release/benostreamdb-flight /usr/local/bin/

# Copy entrypoint
COPY docker/quickstart-entrypoint.sh /usr/local/bin/quickstart-entrypoint.sh

# Create default data directory
RUN mkdir -p /home/benostream/.benostreamdb/search \
    && chown -R benostream:benostream /home/benostream/.benostreamdb

# ES 7.10 API + Qdrant API + Flight SQL gRPC + Prometheus metrics
EXPOSE 9200 6333 50051 9090

# Health check against ES cluster health endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:9200/_cluster/health || exit 1

USER benostream

# Default environment
ENV BENOSEARCH_BIND=0.0.0.0
ENV BENOSEARCH_PORT=9200
ENV QDRANT_BIND=0.0.0.0
ENV QDRANT_PORT=6333
ENV BENOSEARCH_AUTO_REFRESH_SECS=5
ENV RUST_LOG=info

ENTRYPOINT ["quickstart-entrypoint.sh"]
CMD []
