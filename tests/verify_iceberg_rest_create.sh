#!/bin/bash
set -e

# Compile the server
echo "Compiling iceberg_rest..."
cargo build --bin iceberg_rest

# Start the server in background
echo "Starting iceberg_rest server..."
target/debug/iceberg_rest > /tmp/iceberg_rest.log 2>&1 &
SERVER_PID=$!

# Ensure server is killed on exit
cleanup() {
    echo "Stopping server..."
    kill $SERVER_PID || true
}
trap cleanup EXIT

# Wait for server to start
sleep 2

# Create Table Payload
PAYLOAD='{
  "name": "test_table_01",
  "schema": {
    "type": "struct",
    "fields": [
      { "id": 1, "name": "id", "type": "int", "required": true },
      { "id": 2, "name": "data", "type": "string", "required": false }
    ]
  },
  "partition-spec": {
    "spec-id": 0,
    "fields": []
  },
  "properties": {
    "owner": "test_user"
  }
}'

echo "Sending CreateTable request..."
curl -v -X POST http://127.0.0.1:8181/v1/hdb/namespaces/default/tables \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"

echo ""
echo "Verifying table exists..."
# List tables
curl -s http://127.0.0.1:8181/v1/hdb/namespaces/default/tables | grep "test_table_01"

echo "Success!"
