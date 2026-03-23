#!/bin/bash
set -e

# Ensure clean state
rm -rf /tmp/hyperstream_rest_test
mkdir -p /tmp/hyperstream_rest_test

# Start Server in background
export HYPERSTREAM_STORAGE_URI="file:///tmp/hyperstream_rest_test"
cargo run --bin iceberg_rest > /tmp/rest_server.log 2>&1 &
SERVER_PID=$!
echo "Server started with PID $SERVER_PID"

# Wait for server
sleep 5

# 1. Create Table
echo "Creating table..."
curl -v -X POST http://127.0.0.1:8181/v1/hdb/namespaces/default/tables \
  -H "Content-Type: application/json" \
  -d '{"name": "test_table", "schema": {"type": "struct", "fields": [{"id": 1, "name": "id", "type": "int", "required": true}, {"id": 2, "name": "data", "type": "string", "required": false}]}}'

# 2. Add Partition Spec
echo "Adding partition spec..."
curl -v -X POST http://127.0.0.1:8181/v1/hdb/namespaces/default/tables/test_table \
  -H "Content-Type: application/json" \
  -d '{"updates": [{"action": "add-partition-spec", "spec": {"spec-id": 1, "fields": [{"source-id": 1, "field-id": 1000, "name": "id_bucket", "transform": "bucket[16]"}]}}, {"action": "set-default-spec", "spec-id": 1}]}'

# 3. Set Properties
echo "Setting properties..."
curl -v -X POST http://127.0.0.1:8181/v1/hdb/namespaces/default/tables/test_table \
  -H "Content-Type: application/json" \
  -d '{"updates": [{"action": "set-properties", "updates": {"comment": "Rest Update Test", "write.format.default": "parquet"}}]}'

# 4. Verify Metadata
echo "Verifying metadata..."
RESPONSE=$(curl -s http://127.0.0.1:8181/v1/hdb/namespaces/default/tables/test_table)

# Check Default Spec ID
echo "$RESPONSE" | grep '"default-spec-id":1' || { echo "Failed: default-spec-id not updated"; exit 1; }

# Check Properties
echo "$RESPONSE" | grep '"comment":"Rest Update Test"' || { echo "Failed: Property comment not set"; exit 1; }

echo "Success!"

kill $SERVER_PID
