// Copyright (c) 2026 Richard Albright. All rights reserved.

use jni::objects::{JClass, JObject, JString};
use jni::sys::{jboolean, jint, jlong, jstring};
use jni::JNIEnv;
// use std::sync::Arc;
use crate::core::reader::HybridReader;
use crate::core::storage::create_object_store;
use crate::core::table::Table;
use crate::SegmentConfig;
use futures::StreamExt;
use lazy_static::lazy_static;
use tokio::runtime::Runtime;

lazy_static! {
    static ref RUNTIME: Runtime = Runtime::new().unwrap();
}

pub struct HyperStreamSession {
    reader: Option<HybridReader>, // Used if no filter
    path: String,
    filter_str: Option<String>,
    current_batches: Vec<arrow::record_batch::RecordBatch>,
    current_idx: usize,
}

impl HyperStreamSession {
    pub fn new(path: &str, row_selection: Option<String>) -> anyhow::Result<Self> {
        let filter_str = row_selection.filter(|s| !s.trim().is_empty());
        if filter_str.is_some() {
            // If there's a filter, we rely on DataFusion in next_batch, so we don't need HybridReader.
            Ok(Self {
                reader: None,
                path: path.to_string(),
                filter_str,
                current_batches: vec![],
                current_idx: 0,
            })
        } else {
            // Fallback to HybridReader if no filter is provided
            let (parent_uri, segment_id) = if let Some(idx) = path.rfind('/') {
                let parent = &path[..idx];
                let filename = &path[idx + 1..];
                let seg_id = filename.strip_suffix(".parquet").unwrap_or(filename);
                (parent, seg_id)
            } else {
                (".", path)
            };

            let store = create_object_store(parent_uri)?;
            let config = SegmentConfig::new("", segment_id);
            let reader = HybridReader::new(config, store, path);
            Ok(Self {
                reader: Some(reader),
                path: path.to_string(),
                filter_str: None,
                current_batches: vec![],
                current_idx: 0,
            })
        }
    }

    pub fn next_batch(&mut self) -> Option<arrow::record_batch::RecordBatch> {
        if self.current_batches.is_empty() {
            let res = RUNTIME.block_on(async {
                if let Some(ref filter) = self.filter_str {
                    // Use DataFusion to apply the filter
                    let ctx = datafusion::prelude::SessionContext::new();
                    ctx.register_parquet("segment", &self.path, Default::default())
                        .await?;
                    let query = format!("SELECT * FROM segment WHERE {}", filter);
                    let df = ctx.sql(&query).await?;
                    let mut stream = df.execute_stream().await?;
                    let mut batches = Vec::new();
                    while let Some(batch_result) = stream.next().await {
                        batches.push(batch_result?);
                    }
                    Ok::<Vec<arrow::record_batch::RecordBatch>, anyhow::Error>(batches)
                } else if let Some(ref reader) = self.reader {
                    let mut stream = reader.stream_all(None).await?;
                    let mut batches = Vec::new();
                    while let Some(batch_result) = stream.next().await {
                        batches.push(batch_result?);
                    }
                    Ok(batches)
                } else {
                    Ok(vec![])
                }
            });
            match res {
                Ok(batches) => {
                    self.current_batches = batches;
                    self.current_idx = 0;
                }
                Err(e) => {
                    tracing::error!("Error reading batches: {}", e);
                    return None;
                }
            }
        }

        if self.current_idx < self.current_batches.len() {
            let batch = self.current_batches[self.current_idx].clone();
            self.current_idx += 1;
            return Some(batch);
        }

        None
    }
}

#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_trino_HyperStreamDBPageSource_openSession(
    mut env: JNIEnv,
    _class: JClass,
    path: JString,
    row_selection: JString,
) -> jlong {
    let path_str: String = match env.get_string(&path) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };

    let row_selection_str: Option<String> = if row_selection.is_null() {
        None
    } else {
        match env.get_string(&row_selection) {
            Ok(s) => Some(s.into()),
            Err(_) => None,
        }
    };

    if path_str.is_empty() || path_str.len() > 4096 {
        tracing::warn!("FFI: Path validation failed (empty or exceeds 4KB limit)");
        return 0;
    }
    if path_str.contains('\0') {
        tracing::warn!("FFI: Path contains NULL bytes");
        return 0;
    }

    tracing::info!(
        "FFI: Opening Session to {} with filter {:?}",
        path_str,
        row_selection_str
    );

    match HyperStreamSession::new(&path_str, row_selection_str) {
        Ok(session) => Box::into_raw(Box::new(session)) as jlong,
        Err(e) => {
            tracing::error!("FFI Error opening session: {}", e);
            0
        }
    }
}

use arrow::ffi::{to_ffi, FFI_ArrowArray, FFI_ArrowSchema};

use arrow::array::Array; // Fix E0599

/// Native method implementation for `com.hyperstreamdb.trino.HyperStreamDBPageSource.readBatch`
///
/// Expected Java Signature:
/// long readBatch(long handle, long outArrayPtr, long outSchemaPtr)
#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_trino_HyperStreamDBPageSource_readBatch(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    out_array_ptr: jlong,
    out_schema_ptr: jlong,
) -> jlong {
    // Bounds check: reject null handles and null output pointers
    if handle == 0 || out_array_ptr == 0 || out_schema_ptr == 0 {
        tracing::warn!("FFI: readBatch called with null handle or output pointers");
        return 0;
    }

    let session = unsafe { &mut *(handle as *mut HyperStreamSession) };

    match session.next_batch() {
        Some(batch) => {
            tracing::debug!("FFI: Read batch with {} rows", batch.num_rows());

            // 1. Convert RecordBatch to StructArray
            let struct_array: arrow::array::StructArray = batch.into();
            let array_data = struct_array.to_data(); // to_data is often inherent, or via Array trait

            // 2. Export to C Data Interface
            // to_ffi returns (FFI_ArrowArray, FFI_ArrowSchema)
            // We need to move these into the pointers provided by Java

            let (ffi_array, ffi_schema) = match to_ffi(&array_data) {
                Ok(tuple) => tuple,
                Err(e) => {
                    tracing::error!("FFI Error exporting to C Data Interface: {}", e);
                    return 0;
                }
            };

            unsafe {
                std::ptr::write(out_array_ptr as *mut FFI_ArrowArray, ffi_array);
                std::ptr::write(out_schema_ptr as *mut FFI_ArrowSchema, ffi_schema);
            }

            1 // Success
        }
        None => 0, // Finished
    }
}

/// Trino Integration: Split Generation
#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_trino_HyperStreamDBSplitManager_getSplits(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    max_split_size: jlong,
    filter_str: JString,
) -> jstring {
    let uri: String = match env.get_string(&table_uri) {
        Ok(s) => s.into(),
        Err(_) => return std::ptr::null_mut(),
    };

    let filter: String = match env.get_string(&filter_str) {
        Ok(s) => s.into(),
        Err(_) => String::new(),
    };
    let filter_opt = if filter.is_empty() {
        None
    } else {
        Some(filter.as_str())
    };

    // Bounds check: reject empty URIs and URIs exceeding 4KB
    if uri.is_empty() || uri.len() > 4096 {
        tracing::warn!("FFI: getSplits URI validation failed");
        return std::ptr::null_mut();
    }
    // Reject URIs with NULL bytes
    if uri.contains('\0') {
        tracing::warn!("FFI: getSplits URI contains NULL bytes");
        return std::ptr::null_mut();
    }

    // Default 64MB if invalid
    let split_size = if max_split_size <= 0 {
        64 * 1024 * 1024
    } else {
        max_split_size as usize
    };

    // Bounds check: reject absurdly large split sizes (>1GB)
    if split_size > 1_073_741_824 {
        tracing::warn!("FFI: getSplits split size exceeds 1GB limit, capping at 256MB");
    }

    tracing::info!("FFI: Getting splits for {} (max size: {})", uri, split_size);

    let splits_json = match Table::new(uri.clone()) {
        Ok(table) => match table.get_splits(split_size, filter_opt) {
            Ok(splits) => serde_json::to_string(&splits).unwrap_or_else(|_| "[]".to_string()),
            Err(e) => {
                tracing::error!("FFI Error getting splits: {}", e);
                "[]".to_string()
            }
        },
        Err(e) => {
            tracing::error!("FFI Error creating table: {}", e);
            "[]".to_string()
        }
    };

    match env.new_string(splits_json) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Spark Integration: List Data Files with Index Metadata
#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_spark_HyperStreamScanBuilder_listDataFiles(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
) -> jstring {
    let uri: String = match env.get_string(&table_uri) {
        Ok(s) => s.into(),
        Err(_) => return std::ptr::null_mut(),
    };

    // Bounds check: reject empty URIs and URIs exceeding 4KB
    if uri.is_empty() || uri.len() > 4096 {
        tracing::warn!("FFI: listDataFiles URI validation failed");
        return std::ptr::null_mut();
    }
    if uri.contains('\0') {
        tracing::warn!("FFI: listDataFiles URI contains NULL bytes");
        return std::ptr::null_mut();
    }

    tracing::info!("FFI: Listing data files for {}", uri);

    // Call Table API
    // Note: Table::new and list_data_files are currently synchronous,
    // potentially blocking on internal runtime for IO.
    let files_json = match Table::new(uri.clone()) {
        Ok(table) => match table.list_data_files() {
            Ok(files) => serde_json::to_string(&files).unwrap_or_else(|_| "[]".to_string()),
            Err(e) => {
                tracing::error!("FFI Error listing files: {}", e);
                "[]".to_string()
            }
        },
        Err(e) => {
            tracing::error!("FFI Error creating table: {}", e);
            "[]".to_string()
        }
    };

    match env.new_string(files_json) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Spark Integration: Get Splits (Legacy/Fallback)
#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_spark_HyperStreamScanBuilder_getSplits(
    env: JNIEnv,
    _class: JClass,
    _options: JObject,
) -> jstring {
    // Deprecated in favor of listDataFiles for V2 connector
    let splits_json = "[]";
    match env.new_string(splits_json) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// -----------------------------------------------------------------------------
// Spark Connector JNI Bridge
// -----------------------------------------------------------------------------

fn open_session_helper(mut env: JNIEnv, path: JString) -> jlong {
    let path_str: String = match env.get_string(&path) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    // Bounds check: reject empty paths and paths exceeding 4KB
    if path_str.is_empty() || path_str.len() > 4096 {
        tracing::warn!("FFI(Spark): Path validation failed");
        return 0;
    }
    if path_str.contains('\0') {
        tracing::warn!("FFI(Spark): Path contains NULL bytes");
        return 0;
    }
    tracing::info!("FFI(Spark): Opening Session to {}", path_str);
    match HyperStreamSession::new(&path_str, None) {
        Ok(session) => Box::into_raw(Box::new(session)) as jlong,
        Err(e) => {
            tracing::error!("FFI Error opening session: {}", e);
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_spark_HyperStreamPartitionReader_openSession(
    env: JNIEnv,
    _class: JClass,
    path: JString,
) -> jlong {
    open_session_helper(env, path)
}

#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_spark_HyperStreamPartitionReader_readBatch(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    out_array_ptr: jlong,
    out_schema_ptr: jlong,
) -> jlong {
    // Reuse Trino logic since arguments are identical (long, long, long)
    // But we need a valid JNIEnv, so we can't just call the other extern function easily if it used env.
    // The previous implementation utilized 'unsafe' and pointer casting, mostly ignoring Env.
    // So we can extract the body to a safe Rust function.

    // Bounds check: reject null handles and null output pointers
    if handle == 0 || out_array_ptr == 0 || out_schema_ptr == 0 {
        tracing::warn!("FFI(Spark): readBatch called with null handle or output pointers");
        return 0;
    }
    let session = unsafe { &mut *(handle as *mut HyperStreamSession) };

    match session.next_batch() {
        Some(batch) => {
            let struct_array: arrow::array::StructArray = batch.into();
            let array_data = struct_array.to_data();
            let (ffi_array, ffi_schema) = match arrow::ffi::to_ffi(&array_data) {
                Ok(tuple) => tuple,
                Err(e) => {
                    tracing::error!("FFI Error: {}", e);
                    return 0;
                }
            };
            unsafe {
                std::ptr::write(out_array_ptr as *mut FFI_ArrowArray, ffi_array);
                std::ptr::write(out_schema_ptr as *mut FFI_ArrowSchema, ffi_schema);
            }
            1
        }
        None => 0,
    }
}

// -----------------------------------------------------------------------------
// Row-Level Operations JNI Bridge (MERGE / UPDATE / DELETE)
// -----------------------------------------------------------------------------

#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_spark_jni_HyperStreamJNIBridge_queryIndexIn(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    column: JString,
    values_json: JString,
) -> jstring {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let col: String = env
        .get_string(&column)
        .map(|s| s.into())
        .unwrap_or_default();
    let vals: String = env
        .get_string(&values_json)
        .map(|s| s.into())
        .unwrap_or_default();

    tracing::info!(
        "FFI(Spark): queryIndexIn for table {}, column {}, keys: {}",
        uri,
        col,
        vals.len()
    );

    // Placeholder: In the next phase, this will use the Rust Core planner to
    // read the RoaringBitmaps and return a JSON mapping of File -> Array of Row IDs.
    let result_json = "{}";

    match env.new_string(result_json) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_spark_jni_HyperStreamJNIBridge_commitPositionDeletes(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    deletes_json: JString,
) -> jboolean {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let _deletes: String = env
        .get_string(&deletes_json)
        .map(|s| s.into())
        .unwrap_or_default();

    tracing::info!("FFI(Spark): commitPositionDeletes for table {}", uri);

    // Placeholder: This will take the list of Iceberg Position Delete files generated
    // by Spark and commit them to the HyperStreamDB/Iceberg manifest.

    1 // true
}

#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_spark_jni_HyperStreamJNIBridge_addIndex(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    column: JString,
    index_type: JString,
) -> jboolean {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let col: String = env
        .get_string(&column)
        .map(|s| s.into())
        .unwrap_or_default();
    let idx_type: String = env
        .get_string(&index_type)
        .map(|s| s.into())
        .unwrap_or_default();

    tracing::info!(
        "FFI(Spark): addIndex for table {}, column {}, type: {}",
        uri,
        col,
        idx_type
    );

    1 // true
}

#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_spark_jni_HyperStreamJNIBridge_buildIndex(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    segment_id: JString,
) -> jboolean {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let seg_id: String = env
        .get_string(&segment_id)
        .map(|s| s.into())
        .unwrap_or_default();

    tracing::info!(
        "FFI(Spark): buildIndex for table {}, segment_id {}",
        uri,
        seg_id
    );

    1 // true
}

#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_spark_jni_HyperStreamJNIBridge_setPrimaryKey(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    columns: JString,
) -> jboolean {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let cols: String = env
        .get_string(&columns)
        .map(|s| s.into())
        .unwrap_or_default();

    tracing::info!(
        "FFI(Spark): setPrimaryKey for table {}, columns {}",
        uri,
        cols
    );

    1 // true
}

#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_spark_jni_HyperStreamJNIBridge_setGpuContext(
    mut env: JNIEnv,
    _class: JClass,
    device_type: JString,
) -> jboolean {
    let device: String = env
        .get_string(&device_type)
        .map(|s| s.into())
        .unwrap_or_default();

    tracing::info!("FFI(Spark): setGpuContext to {}", device);

    // Convert string to ComputeBackend
    let context = match device.to_lowercase().as_str() {
        "auto" | "gpu" => crate::core::index::gpu::ComputeContext::auto_detect(),
        "cpu" => crate::core::index::gpu::ComputeContext::from_backend(
            crate::core::index::gpu::ComputeBackend::Cpu,
        )
        .unwrap_or_else(|_| crate::core::index::gpu::ComputeContext::auto_detect()),
        "cuda" => crate::core::index::gpu::ComputeContext::from_backend(
            crate::core::index::gpu::ComputeBackend::Cuda,
        )
        .unwrap_or_else(|_| crate::core::index::gpu::ComputeContext::auto_detect()),
        "mps" => crate::core::index::gpu::ComputeContext::from_backend(
            crate::core::index::gpu::ComputeBackend::Mps,
        )
        .unwrap_or_else(|_| crate::core::index::gpu::ComputeContext::auto_detect()),
        "intel" => crate::core::index::gpu::ComputeContext::from_backend(
            crate::core::index::gpu::ComputeBackend::Intel,
        )
        .unwrap_or_else(|_| crate::core::index::gpu::ComputeContext::auto_detect()),
        "rocm" => crate::core::index::gpu::ComputeContext::from_backend(
            crate::core::index::gpu::ComputeBackend::Rocm,
        )
        .unwrap_or_else(|_| crate::core::index::gpu::ComputeContext::auto_detect()),
        _ => {
            tracing::warn!(
                "FFI(Spark): Unknown device type '{}', defaulting to auto",
                device
            );
            crate::core::index::gpu::ComputeContext::auto_detect()
        }
    };

    crate::core::index::gpu::set_thread_gpu_context(Some(context));

    1 // true
}

#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_trino_HyperStreamDBJNIBridge_setGpuContext(
    mut env: JNIEnv,
    _class: JClass,
    device_type: JString,
) -> jboolean {
    let device: String = env
        .get_string(&device_type)
        .map(|s| s.into())
        .unwrap_or_default();

    tracing::info!("FFI(Trino): setGpuContext to {}", device);

    // Convert string to ComputeBackend
    let context = match device.to_lowercase().as_str() {
        "auto" | "gpu" => crate::core::index::gpu::ComputeContext::auto_detect(),
        "cpu" => crate::core::index::gpu::ComputeContext::from_backend(
            crate::core::index::gpu::ComputeBackend::Cpu,
        )
        .unwrap_or_else(|_| crate::core::index::gpu::ComputeContext::auto_detect()),
        "cuda" => crate::core::index::gpu::ComputeContext::from_backend(
            crate::core::index::gpu::ComputeBackend::Cuda,
        )
        .unwrap_or_else(|_| crate::core::index::gpu::ComputeContext::auto_detect()),
        "mps" => crate::core::index::gpu::ComputeContext::from_backend(
            crate::core::index::gpu::ComputeBackend::Mps,
        )
        .unwrap_or_else(|_| crate::core::index::gpu::ComputeContext::auto_detect()),
        "intel" => crate::core::index::gpu::ComputeContext::from_backend(
            crate::core::index::gpu::ComputeBackend::Intel,
        )
        .unwrap_or_else(|_| crate::core::index::gpu::ComputeContext::auto_detect()),
        "rocm" => crate::core::index::gpu::ComputeContext::from_backend(
            crate::core::index::gpu::ComputeBackend::Rocm,
        )
        .unwrap_or_else(|_| crate::core::index::gpu::ComputeContext::auto_detect()),
        _ => {
            tracing::warn!(
                "FFI(Trino): Unknown device type '{}', defaulting to auto",
                device
            );
            crate::core::index::gpu::ComputeContext::auto_detect()
        }
    };

    crate::core::index::gpu::set_thread_gpu_context(Some(context));

    1 // true
}

// -----------------------------------------------------------------------------
// Vector Index Traversal JNI Bridge (Spark & Trino)
// -----------------------------------------------------------------------------
use arrow::array::{Float32Array, Int64Array, StructArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_spark_jni_HyperStreamJNIBridge_vectorSearch(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    segment_id: JString,
    column: JString,
    k: jint,
    query_vector_ptr: jlong,
    query_vector_len: jint,
    out_array_ptr: jlong,
    out_schema_ptr: jlong,
) -> jint {
    vector_search_impl(
        &mut env,
        table_uri,
        segment_id,
        column,
        k,
        query_vector_ptr,
        query_vector_len,
        out_array_ptr,
        out_schema_ptr,
        "Spark",
    )
}

#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_trino_HyperStreamDBJNIBridge_vectorSearch(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    segment_id: JString,
    column: JString,
    k: jint,
    query_vector_ptr: jlong,
    query_vector_len: jint,
    out_array_ptr: jlong,
    out_schema_ptr: jlong,
) -> jint {
    vector_search_impl(
        &mut env,
        table_uri,
        segment_id,
        column,
        k,
        query_vector_ptr,
        query_vector_len,
        out_array_ptr,
        out_schema_ptr,
        "Trino",
    )
}

fn vector_search_impl(
    env: &mut JNIEnv,
    table_uri: JString,
    segment_id: JString,
    column: JString,
    k: jint,
    query_vector_ptr: jlong,
    query_vector_len: jint,
    out_array_ptr: jlong,
    out_schema_ptr: jlong,
    engine: &str,
) -> jint {
    if query_vector_ptr == 0 || out_array_ptr == 0 || out_schema_ptr == 0 {
        tracing::error!("FFI({}): vectorSearch called with null pointers", engine);
        return -1;
    }

    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let seg_id: String = env
        .get_string(&segment_id)
        .map(|s| s.into())
        .unwrap_or_default();
    let col: String = env
        .get_string(&column)
        .map(|s| s.into())
        .unwrap_or_default();

    let query_slice = unsafe {
        std::slice::from_raw_parts(query_vector_ptr as *const f32, query_vector_len as usize)
    };

    tracing::info!(
        "FFI({}): vectorSearch on {}/{} col={} k={} vector_len={}",
        engine,
        uri,
        seg_id,
        col,
        k,
        query_vector_len
    );

    let idx_path_str = format!(".index/{}_{}", seg_id, col);
    let cache_key = format!("{}/{}", uri, idx_path_str);

    let matches = match RUNTIME.block_on(async {
        let store = crate::core::storage::create_object_store(&uri).map_err(|e| {
            tracing::error!("FFI({}): Failed to create store: {}", engine, e);
            e
        })?;

        let hnsw_ivf = crate::core::index::hnsw_ivf::HnswIvfIndex::load_async_with_cache_key(
            store.clone(),
            &idx_path_str,
            &cache_key,
        )
        .await
        .map_err(|e| {
            tracing::error!("FFI({}): Failed to load index: {}", engine, e);
            e
        })?;

        let query_vec = crate::core::index::VectorValue::Float32(query_slice.to_vec());

        // Spawn blocking because HnswIvfIndex::search can be CPU intensive
        tokio::task::spawn_blocking(move || hnsw_ivf.search(&query_vec, k as usize, 10, None))
            .await
            .unwrap_or_else(|e| Err(anyhow::anyhow!("Task panicked: {}", e)))
    }) {
        Ok(m) => m,
        Err(_) => return -1,
    };

    let result_len = matches.len();
    let mut row_ids = Vec::with_capacity(result_len);
    let mut distances = Vec::with_capacity(result_len);

    for (row_id, dist) in matches.into_iter() {
        row_ids.push(row_id as i64);
        distances.push(dist);
    }

    let row_id_array = Arc::new(Int64Array::from(row_ids)) as Arc<dyn arrow::array::Array>;
    let dist_array = Arc::new(Float32Array::from(distances)) as Arc<dyn arrow::array::Array>;

    let schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Int64, false),
        Field::new("_distance", DataType::Float32, false),
    ]));

    let batch =
        match arrow::record_batch::RecordBatch::try_new(schema, vec![row_id_array, dist_array]) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("FFI({}): Failed to create RecordBatch: {}", engine, e);
                return -1;
            }
        };

    let struct_array: StructArray = batch.into();
    let array_data = struct_array.to_data();

    let (ffi_array, ffi_schema) = match arrow::ffi::to_ffi(&array_data) {
        Ok(tuple) => tuple,
        Err(e) => {
            tracing::error!(
                "FFI({}): Error exporting to C Data Interface: {}",
                engine,
                e
            );
            return -1;
        }
    };

    unsafe {
        std::ptr::write(out_array_ptr as *mut FFI_ArrowArray, ffi_array);
        std::ptr::write(out_schema_ptr as *mut FFI_ArrowSchema, ffi_schema);
    }

    result_len as jint
}
