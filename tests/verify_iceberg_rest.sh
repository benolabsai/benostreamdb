#!/bin/bash
set -e

# 1. Seed data
echo "Seeding data..."
export HYPERSTREAM_STORAGE_URI="file:///tmp/hdb_test/default/sample_table"
cargo run --bin setup_test_data

# 2. Start REST server in background
echo "Starting REST server..."
export HYPERSTREAM_STORAGE_URI="file:///tmp/hdb_test"
cargo run --bin iceberg_rest > /tmp/iceberg_rest.log 2>&1 &
SERVER_PID=$!

# Cleanup on exit
trap "kill $SERVER_PID" EXIT

# Wait for server
echo "Waiting for server to start..."
for i in {1..10}; do
    if curl -s http://localhost:8181/v1/config > /dev/null; then
        echo "Server is up!"
        break
    fi
    sleep 2
done

# 3. Verify endpoints
echo "Verifying /v1/config..."
curl -s http://localhost:8181/v1/config | jq .

echo "Verifying /v1/hdb/namespaces..."
curl -s http://localhost:8181/v1/hdb/namespaces | jq .

echo "Verifying /v1/hdb/namespaces/default/tables..."
curl -s http://localhost:8181/v1/hdb/namespaces/default/tables | jq .

echo "Verifying /v1/hdb/namespaces/default/tables/sample_table..."
curl -s http://localhost:8181/v1/hdb/namespaces/default/tables/sample_table | jq .
