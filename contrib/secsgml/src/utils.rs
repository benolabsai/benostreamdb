//! Tar location calculation utilities matching secsgml2.utils.

use crate::error::SecSgmlError;
use serde_json::Value;

/// Computes byte offsets for each document inside a TAR archive containing
/// `metadata.json` followed by the extracted documents in order.
///
/// Modifies the given `Value::Object` metadata in place, setting
/// `"secsgml_start_byte"` and `"secsgml_end_byte"` on each document.
pub fn calculate_documents_locations_in_tar(metadata: &mut Value) -> Result<(), SecSgmlError> {
    let mut placeholder_metadata = metadata.clone();

    let doc_key = "documents";
    let size_key = "secsgml_size_bytes";
    let start_key = "secsgml_start_byte";
    let end_key = "secsgml_end_byte";

    let docs_array = match placeholder_metadata
        .get_mut(doc_key)
        .and_then(Value::as_array_mut)
    {
        Some(arr) => arr,
        None => return Ok(()),
    };

    let _document_length = docs_array.len();

    for doc in docs_array.iter_mut() {
        if let Some(map) = doc.as_object_mut() {
            map.insert(
                start_key.to_string(),
                Value::String("9999999999".to_string()),
            );
            map.insert(end_key.to_string(), Value::String("9999999999".to_string()));
        }
    }

    let placeholder_json = serde_json::to_vec(&placeholder_metadata)?;
    let metadata_size = placeholder_json.len();

    let mut current_pos = 512 + metadata_size;
    current_pos += (512 - (current_pos % 512)) % 512;

    let target_docs = match metadata.get_mut(doc_key).and_then(Value::as_array_mut) {
        Some(arr) => arr,
        None => return Ok(()),
    };

    for doc in target_docs.iter_mut() {
        if let Some(map) = doc.as_object_mut() {
            let size_bytes = map.get(size_key).and_then(Value::as_u64).unwrap_or(0) as usize;
            let start_byte = current_pos + 512;
            let end_byte = start_byte + size_bytes;

            map.insert(
                start_key.to_string(),
                Value::String(format!("{:010}", start_byte)),
            );
            map.insert(
                end_key.to_string(),
                Value::String(format!("{:010}", end_byte)),
            );

            let file_total_size = 512 + size_bytes;
            let padded_size = file_total_size + (512 - (file_total_size % 512)) % 512;
            current_pos += padded_size;
        }
    }

    Ok(())
}
