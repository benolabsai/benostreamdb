#!/bin/bash
set -e

# Configuration
API_URL="http://localhost:8181"
NAMESPACE="default"
TABLE="puffin_test"
STORAGE_DIR="/tmp/hyperstream_puffin_test"

# 1. Cleanup and Reset
rm -rf $STORAGE_DIR
mkdir -p $STORAGE_DIR

# 2. Create Table
echo "Creating table..."
curl -X POST "$API_URL/v1/main/namespaces/$NAMESPACE/tables" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "'$TABLE'",
    "schema": {
      "type": "struct",
      "fields": [
        {"id": 1, "name": "id", "required": true, "type": "int"},
        {"id": 2, "name": "val", "required": false, "type": "string"}
      ]
    }
  }'

# 3. Simulate Data Ingestion (Add Snapshot)
# We create a dummy parquet file and manifest
DATA_FILE="$STORAGE_DIR/data-1.parquet"
touch $DATA_FILE

echo "Adding snapshot..."
curl -X POST "$API_URL/v1/main/namespaces/$NAMESPACE/tables/$TABLE" \
  -H "Content-Type: application/json" \
  -d '{
    "updates": [
      {
        "action": "add-snapshot",
        "snapshot": {
          "snapshot-id": 1,
          "manifest-list": "file://'$DATA_FILE'"
        }
      }
    ]
  }'

# 4. Attach a Puffin Sidecar Index
# We'll use a dummy puffin file path
PUFFIN_FILE="$STORAGE_DIR/sidecar.puffin"
touch $PUFFIN_FILE

echo "Attaching Puffin Sidecar Index..."
curl -X POST "$API_URL/v1/main/namespaces/$NAMESPACE/tables/$TABLE" \
  -H "Content-Type: application/json" \
  -d '{
    "updates": [
      {
        "action": "add-sidecar-index",
        "file-path": "data-1.parquet",
        "index-file": {
          "file_path": "sidecar.puffin",
          "index_type": "vector",
          "column_name": "val",
          "blob_type": "hnsw-v1",
          "offset": 4,
          "length": 1024
        }
      }
    ]
  }'

# 5. Verify Metadata
echo "Verifying metadata..."
METADATA_URL="$API_URL/v1/main/namespaces/$NAMESPACE/tables/$TABLE"
RESPONSE=$(curl -s "$METADATA_URL")

if echo "$RESPONSE" | grep -q "puffin"; then
  echo "SUCCESS: Puffin metadata found in table response"
else
  echo "FAILURE: Puffin metadata missing"
  echo "$RESPONSE"
  exit 1
fi
