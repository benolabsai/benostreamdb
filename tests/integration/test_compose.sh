#!/usr/bin/env bash
set -ex

# ---------------------------------------------------------------------------
# Integration test for docker-compose.production.yml
# Starts the stack with a key, waits for health, runs an authenticated query,
# and tears it down.
# ---------------------------------------------------------------------------

export BSDB_API_KEY="test-key-123"
export BSDB_SEARCH_PORT="9200"
export BSDB_QDRANT_PORT="6333"
export FLIGHT_PORT="50051"
export AWS_ACCESS_KEY_ID="test"
export AWS_SECRET_ACCESS_KEY="test"

# Go to repository root
cd "$(dirname "$0")/../.."

echo "Starting compose stack..."
docker compose -f docker-compose.production.yml up -d --build

echo "Waiting for services to be healthy..."
# Wait up to 60s for all services to report healthy
timeout 60 bash -c '
while true; do
    healthy_count=$(docker compose -f docker-compose.production.yml ps --format json | grep -o "\"Health\":\"healthy\"" | wc -l)
    if [ "$healthy_count" -eq 2 ]; then
        echo "All services healthy!"
        break
    fi
    sleep 2
done
'

echo "Running authenticated query on search service..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $BSDB_API_KEY" http://localhost:$BSDB_SEARCH_PORT/_cluster/health)
if [ "$HTTP_STATUS" -ne 200 ]; then
    echo "Authenticated query failed with status $HTTP_STATUS"
    docker compose -f docker-compose.production.yml logs
    docker compose -f docker-compose.production.yml down
    exit 1
fi

echo "Running unauthenticated query on search service (should fail)..."
HTTP_STATUS_UNAUTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$BSDB_SEARCH_PORT/_cluster/health)
if [ "$HTTP_STATUS_UNAUTH" -ne 401 ]; then
    echo "Unauthenticated query returned $HTTP_STATUS_UNAUTH instead of 401"
    docker compose -f docker-compose.production.yml logs
    docker compose -f docker-compose.production.yml down
    exit 1
fi

echo "Tearing down..."
docker compose -f docker-compose.production.yml down -v

echo "Integration test passed."
