#!/bin/bash
set -e

# Start the server in background (assume compiled)
echo "Starting iceberg_rest server..."
target/debug/iceberg_rest > /tmp/iceberg_rest_update.log 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "Stopping server..."
    kill $SERVER_PID || true
}
trap cleanup EXIT

sleep 2

# 1. Create Table
CREATE_PAYLOAD='{
  "name": "test_update_table",
  "schema": {
    "type": "struct",
    "fields": [
      { "id": 1, "name": "id", "type": "int", "required": true }
    ]
  }
}'
echo "Creating table..."
curl -s -X POST http://127.0.0.1:8181/v1/hdb/namespaces/default/tables \
  -H "Content-Type: application/json" \
  -d "$CREATE_PAYLOAD" > /dev/null

# 2. Update Table (Add Schema)
UPDATE_PAYLOAD='{
  "updates": [
    {
      "action": "add-schema",
      "schema": {
        "type": "struct",
        "schema-id": 1,
        "fields": [
          { "id": 1, "name": "id", "type": "int", "required": true },
          { "id": 2, "name": "new_col", "type": "string", "required": false }
        ]
      }
    },
    {
      "action": "set-current-schema",
      "schema-id": 1
    }
  ]
}'

echo "Updating table (schema evolution)..."
curl -v -X POST http://127.0.0.1:8181/v1/hdb/namespaces/default/tables/test_update_table \
  -H "Content-Type: application/json" \
  -d "$UPDATE_PAYLOAD" 2>&1 | tee /tmp/update_response.txt

RESPONSE=$(cat /tmp/update_response.txt)

# Verify "new_col" is in the response metadata > schemas
if echo "$RESPONSE" | grep -q "new_col"; then
    echo "Success: New column found in metadata!"
else
    echo "Failure: New column NOT found."
    exit 1
fi
