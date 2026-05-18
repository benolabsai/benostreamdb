# ---------------------------------------------------------------------------
# Stage 1: Build
# ---------------------------------------------------------------------------
FROM rust:1.85-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy lockfile first for better caching
COPY Cargo.toml Cargo.lock ./
COPY rust-toolchain.toml ./

# Create a dummy src to cache dependency builds
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && cargo build --release \
    && rm -rf src

# Copy actual source
COPY src ./src

# Build server binaries
RUN cargo build --release --bin gateway --bin iceberg_rest

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r hyperstream && useradd -r -g hyperstream hyperstream

WORKDIR /app

# Copy binaries
COPY --from=builder /app/target/release/gateway /usr/local/bin/
COPY --from=builder /app/target/release/iceberg_rest /usr/local/bin/

# Expose ports
EXPOSE 3000 8181

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Switch to non-root user
USER hyperstream

# Default: run gateway
ENTRYPOINT ["gateway"]
CMD []
