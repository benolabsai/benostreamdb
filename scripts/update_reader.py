import re

with open("src/core/reader.rs", "r") as f:
    content = f.read()

# Line 266: if let Ok(bytes) = ret.bytes().await {
content = content.replace(
    "if let Ok(bytes) = ret.bytes().await {",
    "if let Ok(bytes) = ret.bytes().await {\n                             crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(bytes.len() as u64);"
)

# Line 283: Ok(bytes) => { (for Puffin file get_range)
content = content.replace(
    "Ok(bytes) => {\n                         match crate::core::puffin::read_deletion_vector_from_bytes(&bytes) {",
    "Ok(bytes) => {\n                         crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(bytes.len() as u64);\n                         match crate::core::puffin::read_deletion_vector_from_bytes(&bytes) {"
)

# Line 411: data_res mapping for bloom filter
content = content.replace(
    "let data_res = self.store.get_range(&pq_path, start..end).await;",
    "let data_res = self.store.get_range(&pq_path, start..end).await.map(|b| {\n                        crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);\n                        b\n                    });"
)

# Line 414: fallback get_range
content = content.replace(
    "Err(_) => self.store.get_range(&pq_path, start..file_size).await?,",
    "Err(_) => {\n                            let b = self.store.get_range(&pq_path, start..file_size).await?;\n                            crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);\n                            b\n                        },"
)

# Line 576: get_range for inverted index
content = content.replace(
    "self.store.get_range(&inv_path, (offset as u64)..(offset as u64 + length as u64)).await?",
    "{\n                                 let b = self.store.get_range(&inv_path, (offset as u64)..(offset as u64 + length as u64)).await?;\n                                 crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);\n                                 b\n                             }"
)

# Line 581: fallback get inverted index
content = content.replace(
    "Ok(res) => res.bytes().await?.to_vec(),",
    "Ok(res) => {\n                                     let b = res.bytes().await?;\n                                     crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);\n                                     b.to_vec()\n                                 },"
)

# Line 778: resp.bytes() for idx
content = content.replace(
    "let index_bytes = resp.bytes().await?;",
    "let index_bytes = resp.bytes().await?;\n                         crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(index_bytes.len() as u64);"
)

# Line 1328: fallback get inverted index bytes
content = content.replace(
    "Ok(res) => res.bytes().await?.to_vec(),",
    "Ok(res) => {\n                     let b = res.bytes().await?;\n                     crate::telemetry::metrics::IO_BYTES_READ_TOTAL.inc_by(b.len() as u64);\n                     b.to_vec()\n                 },"
)

with open("src/core/reader.rs", "w") as f:
    f.write(content)

print("Updated reader.rs")
