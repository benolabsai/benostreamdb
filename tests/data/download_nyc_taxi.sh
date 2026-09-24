#!/bin/bash
# Download NYC Taxi dataset for testing
# Dataset: ~1.5B rows, ~200GB Parquet files

set -e

DATA_DIR="tests/data/nyc_taxi"
mkdir -p "$DATA_DIR"

echo "Downloading NYC Taxi dataset (Parquet format)..."
echo "This will download ~200GB of data. Press Ctrl+C to cancel."
sleep 3

# NYC Taxi data is available from AWS Open Data Registry
# https://registry.opendata.aws/nyc-tlc-trip-records-pds/

BASE_URL="https://d37ci6vzurychx.cloudfront.net/trip-data"

# Download 2023 data (only January for testing)
for month in "01"; do
    file="yellow_tripdata_2023-${month}.parquet"
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "Downloading $file..."
        wget -P "$DATA_DIR" "$BASE_URL/$file"
    else
        echo "$file already exists, skipping..."
    fi
done

echo "Download complete!"
echo "Files saved to: $DATA_DIR"
echo ""
echo "Dataset stats:"
du -sh "$DATA_DIR"
ls -lh "$DATA_DIR"
