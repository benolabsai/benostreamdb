#!/bin/bash
set -e

DATA_DIR="/tmp/hdb_test_delete"
TABLE_DIR="$DATA_DIR/default/test_delete_table"
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

# 1. Compile generator and server
echo "Compiling..."
cargo build --bin generate_iceberg_manifests
cargo build --bin iceberg_rest

# 2. Generate Manifests
echo "Generating Iceberg Manifests..."
# Generate Manifests INSIDE the Table Directory
GEN_DIR="$TABLE_DIR/data"
mkdir -p "$GEN_DIR"
# Pass ABSOLUTE URI
ABS_GEN_URI="file://$GEN_DIR"
./target/debug/generate_iceberg_manifests "$GEN_DIR" "$ABS_GEN_URI"

# 3. Start Server
echo "Starting Iceberg REST Server..."
export HYPERSTREAM_STORAGE_URI="file://$DATA_DIR"
./target/debug/iceberg_rest > /tmp/iceberg_rest_delete.log 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "Stopping server..."
    kill $SERVER_PID || true
}
trap cleanup EXIT

sleep 3

# 4. Create Table
echo "Creating Table..."
CREATE_PAYLOAD='{
  "name": "test_delete_table",
  "schema": {
    "type": "struct",
    "fields": [
      { "id": 1, "name": "category", "type": "string", "required": false }
    ]
  },
  "partition-spec": {
    "spec-id": 0,
    "fields": [
      { "name": "category", "transform": "identity", "source-id": 1, "field-id": 1000 }
    ]
  }
}'

curl -s -X POST http://127.0.0.1:8181/v1/hdb/namespaces/default/tables \
  -H "Content-Type: application/json" \
  -d "$CREATE_PAYLOAD" | jq .

# 5. Add Snapshot via REST
# We point to the generated manifest list.
# Use RELATIVE path from Storage Root for manifest-list to avoid object_store confusion
# Storage Root is $DATA_DIR.
# Path to snap-1.avro is $TABLE_DIR/data/snap-1.avro
# RELATIVE from DATA_DIR: default/test_delete_table/data/snap-1.avro
RELATIVE_MANIFEST_LIST="default/test_delete_table/data/snap-1.avro"

echo "Adding Snapshot with Deletes..."
UPDATE_PAYLOAD=$(jq -n --arg ml "$RELATIVE_MANIFEST_LIST" '{
  "updates": [
    {
      "action": "add-snapshot",
      "snapshot": {
        "manifest-list": $ml,
        "summary": { "operation": "append" }
      }
    }
  ]
}')

curl -s -X POST http://127.0.0.1:8181/v1/hdb/namespaces/default/tables/test_delete_table \
  -H "Content-Type: application/json" \
  -d "$UPDATE_PAYLOAD" > /tmp/update_resp.json

cat /tmp/update_resp.json | jq .

# 6. Verify Internal State (v1.json)
# The server should have committed a new manifest v1.json containing the entries.
MANIFEST_FILE=$(ls "$TABLE_DIR/_manifest/v"*.json | sort -V | tail -n 1)
echo "Inspecting latest manifest: $MANIFEST_FILE"

# using jq to count delete_files in entries.
# entries is an array. specific entry should have delete_files array.
# We expect 1 data file, associated with 1 delete file.
DELETE_COUNT=$(cat "$MANIFEST_FILE" | jq '[.entries[] | .delete_files | length] | add')

echo "Total Delete Files Associated: $DELETE_COUNT"

if [ "$DELETE_COUNT" -gt 0 ]; then
    echo "SUCCESS: Delete files were correctly imported and associated!"
else
    echo "FAILURE: No delete files found in the internal manifest."
    cat "$MANIFEST_FILE"
    exit 1
fi

echo "Verifying Read content (Deletes Applied)..."
cargo build --bin verify_iceberg_read_check
./target/debug/verify_iceberg_read_check
