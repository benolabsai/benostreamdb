#!/bin/bash
set -e

# Setup test table URI
export TEST_URI="file:///tmp/hdb_admin_test/default/admin_table"
export HYPERSTREAM_STORAGE_URI="$TEST_URI"

# Clean up any existing data
rm -rf /tmp/hdb_admin_test

echo "1. Generating test data using setup_test_data..."
cargo run --bin setup_test_data

echo "2. Testing Compaction..."
cargo run --bin hyperstreamdb-admin -- compact --uri "$TEST_URI" --target-file-size 10MB

echo "3. Testing Vacuum..."
cargo run --bin hyperstreamdb-admin -- vacuum --uri "$TEST_URI" --retain-versions 1

echo "E2E Admin CLI tests completed successfully!"
