#!/bin/bash
# ---------------------------------------------------------------------------
# HyperStreamDB Quickstart Entrypoint
# Launches both Search (ES 7.10 + Qdrant) and Flight SQL services
# ---------------------------------------------------------------------------
set -e

echo "============================================="
echo "  HyperStreamDB Quickstart"
echo "  ES 7.10 API:     http://0.0.0.0:9200"
echo "  Qdrant API:      http://0.0.0.0:6333"
echo "  Flight SQL gRPC: grpc://0.0.0.0:50051"
echo "============================================="

# Launch Flight SQL server in background
echo "[quickstart] Starting Flight SQL server on :50051..."
hyperstreamdb-flight &
FLIGHT_PID=$!

# Launch Search server in foreground (captures signals for graceful shutdown)
echo "[quickstart] Starting Search server on :9200 (ES) + :6333 (Qdrant)..."

# Trap signals to shut down both processes gracefully
cleanup() {
    echo "[quickstart] Shutting down..."
    kill $FLIGHT_PID 2>/dev/null || true
    wait $FLIGHT_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

hypersearch &
SEARCH_PID=$!

# Wait for either process to exit
wait -n $FLIGHT_PID $SEARCH_PID 2>/dev/null || true

# If one died, kill the other and exit
cleanup
