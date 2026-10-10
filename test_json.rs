fn main() {
    let _: fn(&[arrow::record_batch::RecordBatch]) -> Result<Vec<serde_json::Map<String, serde_json::Value>>, arrow::error::ArrowError> = arrow_json::writer::record_batches_to_json_rows;
}
