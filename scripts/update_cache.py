import re
import os

files_to_update = [
    "src/core/index/hnsw_ivf.rs",
    "src/core/manifest.rs",
    "src/core/reader.rs"
]

cache_names = {
    "HNSW_IVF_CACHE": "hnsw_ivf",
    "LATEST_VERSION_CACHE": "latest_version",
    "MANIFEST_CACHE": "manifest",
    "MANIFEST_LIST_CACHE": "manifest_list",
    "BLOOM_FILTER_CACHE": "bloom_filter",
    "INVERTED_INDEX_CACHE": "inverted_index",
    "BYTE_CACHE": "byte",
    "INDEX_CACHE": "index",
    "PARQUET_META_CACHE": "parquet_meta",
    "BLOCK_CACHE": "block",
    "HNSW_CACHE": "hnsw"
}

for file_path in files_to_update:
    with open(file_path, "r") as f:
        content = f.read()

    # We need to add `use crate::core::cache::CacheExt;` at the top of the files if not already there
    if "use crate::core::cache::CacheExt;" not in content:
        # Find the first use statement and add it
        content = re.sub(r'^(use [^;]+;)', r'use crate::core::cache::CacheExt;\n\1', content, count=1, flags=re.MULTILINE)

    for cache_var, cache_name in cache_names.items():
        # Match something like `CACHE_VAR.get(&key).await`
        # or `crate::core::cache::CACHE_VAR.get(&key).await`
        pattern = r'(' + cache_var + r')\.get\(([^)]+)\)'
        
        def replacer(match):
            return f'{match.group(1)}.get_with_metrics({match.group(2)}, "{cache_name}")'
            
        content = re.sub(pattern, replacer, content)

    with open(file_path, "w") as f:
        f.write(content)

print("Updated files.")
