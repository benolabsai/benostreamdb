#!/bin/bash
set -e

DATA_DIR="/tmp/hdb_test_delete"
TABLE_DIR="$DATA_DIR/default/test_delete_table"
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

echo "Compiling Rust binaries..."
cargo build --bin generate_iceberg_manifests
cargo build --bin iceberg_rest

echo "Setting up Python Environment..."
if [ ! -d ".venv_iceberg" ]; then
    python3 -m venv .venv_iceberg
    source .venv_iceberg/bin/activate
    pip install maturin pandas pyarrow
else
    source .venv_iceberg/bin/activate
fi

echo "Building Python bindings..."
if ! command -v maturin &> /dev/null; then
    echo "Using system maturin..."
fi
if maturin develop; then
    echo "Maturin develop success."
else
    echo "Maturin develop failed. Trying pip install ."
    pip install .
fi

echo "Generating Iceberg Data..."
GEN_DIR="$TABLE_DIR/data"
mkdir -p "$GEN_DIR"
ABS_GEN_URI="file://$GEN_DIR"
./target/debug/generate_iceberg_manifests "$GEN_DIR" "$ABS_GEN_URI"

echo "Starting Iceberg REST Server..."
export HYPERSTREAM_STORAGE_URI="file://$DATA_DIR"
# Assuming iceberg_rest runs on 8181 by default as per other script
./target/debug/iceberg_rest > /tmp/iceberg_rest_py.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID" EXIT

sleep 3

# Manual Table Setup via REST (Simulating Client)
echo "Creating Table via REST..."
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
  -d "$CREATE_PAYLOAD" > /dev/null

RELATIVE_MANIFEST_LIST="default/test_delete_table/data/snap-1.avro"

echo "Adding Snapshot via REST..."
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
  -d "$UPDATE_PAYLOAD" > /dev/null

echo "Running Python Verification..."
python3 tests/test_iceberg_python.py
