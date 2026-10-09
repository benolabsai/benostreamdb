// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use jni::objects::{JClass, JObject, JString};
use jni::sys::{jboolean, jint, jlong, jstring};
use jni::JNIEnv;
// use std::sync::Arc;
use crate::core::manifest::IndexAlgorithm;
use crate::core::reader::HybridReader;
use crate::core::storage::create_object_store;
use crate::core::table::Table;
use crate::SegmentConfig;
use futures::StreamExt;
use std::sync::LazyLock;
use tokio::runtime::Runtime;

/// Shared Tokio runtime for the JNI entry points.
///
/// `Runtime::new()` has no infallible form and fails only if the OS cannot hand
/// the process a reactor/worker threads, which is unrecoverable for the JNI
/// bridge anyway. Mirrors `python::helpers::TOKIO_RUNTIME` (same justification,
/// and the same no-panic exemption).
#[allow(clippy::expect_used)]
static RUNTIME: LazyLock<Runtime> =
    LazyLock::new(|| Runtime::new().expect("Failed to create Tokio runtime for the JNI bridge"));

/// Initialize the native tracing subscriber when the JVM loads the library, so
/// `tracing::error!` from the JNI entry points reaches the Trino logs instead of
/// being silently dropped. Without this, a native failure surfaces only as a
/// generic `... failed` message on the Java side.
#[no_mangle]
pub extern "system" fn JNI_OnLoad(
    _vm: *mut jni::sys::JavaVM,
    _reserved: *mut std::ffi::c_void,
) -> jni::sys::jint {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .try_init();
    jni::sys::JNI_VERSION_1_8
}

pub struct BenoStreamSession {
    reader: Option<HybridReader>, // Used if no filter
    path: String,
    filter_str: Option<String>,
    current_batches: Vec<arrow::record_batch::RecordBatch>,
    current_idx: usize,
}

impl BenoStreamSession {
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
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBPageSource_openSession(
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

    match BenoStreamSession::new(&path_str, row_selection_str) {
        Ok(session) => Box::into_raw(Box::new(session)) as jlong,
        Err(e) => {
            tracing::error!("FFI Error opening session: {}", e);
            0
        }
    }
}

use arrow::ffi::{to_ffi, FFI_ArrowArray, FFI_ArrowSchema};

use arrow::array::Array; // Fix E0599

/// Native method implementation for `com.benostreamdb.trino.BenoStreamDBPageSource.readBatch`
///
/// Expected Java Signature:
/// long readBatch(long handle, long outArrayPtr, long outSchemaPtr)
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBPageSource_readBatch(
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

    let session = unsafe { &mut *(handle as *mut BenoStreamSession) };

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
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBSplitManager_getSplits(
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
pub extern "system" fn Java_com_benostreamdb_spark_BenoStreamScanBuilder_listDataFiles(
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
pub extern "system" fn Java_com_benostreamdb_spark_BenoStreamScanBuilder_getSplits(
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
    match BenoStreamSession::new(&path_str, None) {
        Ok(session) => Box::into_raw(Box::new(session)) as jlong,
        Err(e) => {
            tracing::error!("FFI Error opening session: {}", e);
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_BenoStreamPartitionReader_openSession(
    env: JNIEnv,
    _class: JClass,
    path: JString,
) -> jlong {
    open_session_helper(env, path)
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_BenoStreamPartitionReader_readBatch(
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
    let session = unsafe { &mut *(handle as *mut BenoStreamSession) };

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
// Index Lifecycle & Primary-Key JNI Bridge (Spark stored procedures).
//
// These back `CALL benostream.system.{add_index,drop_index,build_index,
// rebuild_index,set_primary_key,drop_primary_key,show_indexes,compact}` and
// route straight into the engine (no Iceberg Java). `queryIndexIn` /
// `commitPositionDeletes` were removed: the connector prunes with the SQL that
// `openQuery` already plans, and deletes commit as predicate removals, so those
// entry points had no callers and only returned sentinel stubs.
// -----------------------------------------------------------------------------

/// Split a comma-separated column list, trimming blanks.
fn split_columns(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect()
}

/// Spark: add an index on `column` using the named algorithm/category.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_addIndex(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    column: JString,
    index_category: JString,
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
        .get_string(&index_category)
        .map(|s| s.into())
        .unwrap_or_default();
    if uri.is_empty() || col.is_empty() {
        tracing::error!("FFI(Spark): addIndex requires a table and column");
        return 0;
    }
    let algorithm = IndexAlgorithm::from_name(&idx_type).unwrap_or_default();
    let col_log = col.clone();
    let res = RUNTIME.block_on(async {
        let table = Table::new_async(uri).await?;
        table.add_index(col, algorithm).await
    });
    match res {
        Ok(()) => {
            tracing::info!(
                "FFI(Spark): addIndex committed for '{}' ({})",
                col_log,
                idx_type
            );
            1
        }
        Err(e) => {
            tracing::error!("FFI(Spark): addIndex failed: {}", e);
            0
        }
    }
}

/// Spark: build/rebuild index files.
///
/// `segment_id` is the column to rebuild; `"all"`, `"null"`, or empty means
/// "fill in every missing index file" (the offline batch build).
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_buildIndex(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    segment_id: JString,
) -> jboolean {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let seg: String = env
        .get_string(&segment_id)
        .map(|s| s.into())
        .unwrap_or_default();
    if uri.is_empty() {
        tracing::error!("FFI(Spark): buildIndex requires a table");
        return 0;
    }
    let target = seg.trim().to_string();
    let all = target.is_empty()
        || target.eq_ignore_ascii_case("all")
        || target.eq_ignore_ascii_case("null");
    let res = RUNTIME.block_on(async {
        let table = Table::new_async(uri).await?;
        if all {
            table.recover_indexes_async().await.map(|_| ())
        } else {
            table.rebuild_index(target).await
        }
    });
    match res {
        Ok(()) => 1,
        Err(e) => {
            tracing::error!("FFI(Spark): buildIndex failed: {}", e);
            0
        }
    }
}

/// Spark: set the table primary key to the comma-separated `columns`.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_setPrimaryKey(
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
    let columns = split_columns(&cols);
    if uri.is_empty() || columns.is_empty() {
        tracing::error!("FFI(Spark): setPrimaryKey requires a table and at least one column");
        return 0;
    }
    let res = RUNTIME.block_on(async {
        let table = Table::new_async(uri).await?;
        table.set_primary_key_async(columns).await
    });
    match res {
        Ok(()) => 1,
        Err(e) => {
            tracing::error!("FFI(Spark): setPrimaryKey failed: {}", e);
            0
        }
    }
}

/// Spark: drop the comma-separated `columns` from the table primary key.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_dropPrimaryKey(
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
    let columns = split_columns(&cols);
    if uri.is_empty() || columns.is_empty() {
        tracing::error!("FFI(Spark): dropPrimaryKey requires a table and at least one column");
        return 0;
    }
    let res = RUNTIME.block_on(async {
        let table = Table::new_async(uri).await?;
        for c in columns {
            table.drop_primary_key(c).await?;
        }
        Ok::<(), anyhow::Error>(())
    });
    match res {
        Ok(()) => 1,
        Err(e) => {
            tracing::error!("FFI(Spark): dropPrimaryKey failed: {}", e);
            0
        }
    }
}

/// Spark: drop the index on `column`.
///
/// `index_category` is advisory only — the engine keys indexes by column, so a
/// single column can only have one index to drop.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_dropIndex(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    column: JString,
    index_category: JString,
) -> jboolean {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let col: String = env
        .get_string(&column)
        .map(|s| s.into())
        .unwrap_or_default();
    let _idx_type: String = env
        .get_string(&index_category)
        .map(|s| s.into())
        .unwrap_or_default();
    if uri.is_empty() || col.is_empty() {
        tracing::error!("FFI(Spark): dropIndex requires a table and column");
        return 0;
    }
    let res = RUNTIME.block_on(async {
        let table = Table::new_async(uri).await?;
        table.drop_index(col).await
    });
    match res {
        Ok(()) => 1,
        Err(e) => {
            tracing::error!("FFI(Spark): dropIndex failed: {}", e);
            0
        }
    }
}

/// Spark: compact the table's data files.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_compactTable(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
) -> jboolean {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    if uri.is_empty() {
        tracing::error!("FFI(Spark): compactTable requires a table");
        return 0;
    }
    let res = RUNTIME.block_on(async {
        let table = Table::new_async(uri).await?;
        table.rewrite_data_files_async(None).await
    });
    match res {
        Ok(()) => 1,
        Err(e) => {
            tracing::error!("FFI(Spark): compactTable failed: {}", e);
            0
        }
    }
}

/// Spark: list the table's indexes as a JSON array of index-file records.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_listIndexes(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
) -> jstring {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let json = if uri.is_empty() {
        "[]".to_string()
    } else {
        match RUNTIME.block_on(async {
            let table = Table::new_async(uri).await?;
            table.list_index_files().await
        }) {
            Ok(files) => serde_json::to_string(&files).unwrap_or_else(|_| "[]".to_string()),
            Err(e) => {
                tracing::error!("FFI(Spark): listIndexes failed: {}", e);
                "[]".to_string()
            }
        }
    };
    match env.new_string(json) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_setGpuContext(
    mut env: JNIEnv,
    _class: JClass,
    device_type: JString,
) -> jboolean {
    let device: String = env
        .get_string(&device_type)
        .map(|s| s.into())
        .unwrap_or_default();

    tracing::info!("FFI(Spark): setGpuContext to {}", device);

    // Shared device-string mapping (see `context_from_device_str`).
    let context = crate::core::index::gpu::context_from_device_str(&device);
    crate::core::index::gpu::set_thread_gpu_context(Some(context));

    1 // true
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_setGpuContext(
    mut env: JNIEnv,
    _class: JClass,
    device_type: JString,
) -> jboolean {
    let device: String = env
        .get_string(&device_type)
        .map(|s| s.into())
        .unwrap_or_default();

    tracing::info!("FFI(Trino): setGpuContext to {}", device);

    // Shared device-string mapping (see `context_from_device_str`).
    let context = crate::core::index::gpu::context_from_device_str(&device);
    crate::core::index::gpu::set_thread_gpu_context(Some(context));

    1 // true
}

/// Trino/Spark: install a comma-separated GPU device pool for multi-GPU
/// execution (e.g. `"cuda:0,cuda:1"`). Each engine worker thread is assigned
/// one device round-robin, so a single query spreads across GPUs. A list of 0
/// or 1 devices is a no-op (the process-wide `setGpuContext` still applies).
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_setGpuDevicePool(
    mut env: JNIEnv,
    _class: JClass,
    devices: JString,
) -> jboolean {
    let csv: String = env
        .get_string(&devices)
        .map(|s| s.into())
        .unwrap_or_default();
    crate::core::index::gpu::set_gpu_device_pool_from_str(&csv);
    1 // true
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_setGpuDevicePool(
    mut env: JNIEnv,
    _class: JClass,
    devices: JString,
) -> jboolean {
    let csv: String = env
        .get_string(&devices)
        .map(|s| s.into())
        .unwrap_or_default();
    crate::core::index::gpu::set_gpu_device_pool_from_str(&csv);
    1 // true
}

// -----------------------------------------------------------------------------
// Vector Index Traversal JNI Bridge (Spark & Trino)
// -----------------------------------------------------------------------------
use arrow::array::{Float32Array, Int64Array, StructArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_vectorSearch(
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
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_vectorSearch(
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
    // Trust boundary: `query_vector_len` is JNI-supplied and is used as a slice
    // length below. A non-positive value would cast to a huge `usize`, so reject
    // it, and reject an obviously-corrupt length (no real embedding exceeds the
    // cap) rather than trusting it blindly.
    const MAX_QUERY_DIMS: jint = 1 << 20;
    if query_vector_len <= 0 || query_vector_len > MAX_QUERY_DIMS {
        tracing::error!(
            "FFI({}): vectorSearch called with invalid vector_len={} (must be 1..={})",
            engine,
            query_vector_len,
            MAX_QUERY_DIMS
        );
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

    // SAFETY: `query_vector_ptr` is a JNI `jlong` holding a pointer to a
    // `jfloatArray`'s elements and `query_vector_len` is that array's length
    // (JNI `GetFloatArrayElements` contract) — the JVM owns the allocation and
    // keeps it valid for the duration of this call. The length was validated
    // positive and capped above; the data is copied into a `Vec` immediately, so
    // the borrow never outlives the JNI frame.
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
            true, // use_mmap: default to true for zero-copy
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

// -----------------------------------------------------------------------------
// Regional Drift Traversal JNI Bridge (Spark)
// -----------------------------------------------------------------------------
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_regionalDriftSearch(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    query: JString,
    seeds_json: JString,
    top_k: jint,
    hops: jint,
    n_depth: jint,
    k_followups: jint,
    mode_str: JString,
    out_array_ptr: jlong,
    out_schema_ptr: jlong,
) -> jint {
    if out_array_ptr == 0 || out_schema_ptr == 0 {
        tracing::error!("FFI(Spark): regionalDriftSearch called with null pointers");
        return -1;
    }

    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let query_str: String = env.get_string(&query).map(|s| s.into()).unwrap_or_default();
    let seeds_json_str: String = env
        .get_string(&seeds_json)
        .map(|s| s.into())
        .unwrap_or_default();
    let mode: String = env
        .get_string(&mode_str)
        .map(|s| s.into())
        .unwrap_or_default();

    let seeds_result: Result<Vec<String>, _> = serde_json::from_str(&seeds_json_str);
    let seeds_str = match seeds_result {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("FFI(Spark): failed to parse seeds_json: {}", e);
            return -1;
        }
    };
    let mut seeds = Vec::with_capacity(seeds_str.len());
    for s in seeds_str {
        if let Ok(v) = s.parse::<u64>() {
            seeds.push(v);
        }
    }

    let graph_mode = crate::core::sql::graph_udf::graph_view::parse_graph_mode(&mode);

    let res = RUNTIME.block_on(async {
        let table = match crate::core::table::Table::new_async(uri.clone()).await {
            Ok(t) => t,
            Err(e) => return Err(anyhow::anyhow!("Failed to open table: {}", e)),
        };

        table
            .regional_drift_with_mode(
                &query_str,
                &seeds,
                top_k as usize,
                hops as u32,
                n_depth as u32,
                k_followups as usize,
                graph_mode,
            )
            .await
    });

    let result = match res {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("FFI(Spark): regional_drift_with_mode error: {}", e);
            return -1;
        }
    };

    let result_len = result.all_discovered_nodes.len();
    let mut row_ids = Vec::with_capacity(result_len);
    for n in result.all_discovered_nodes {
        row_ids.push(n as i64);
    }

    let row_id_array = Arc::new(Int64Array::from(row_ids)) as Arc<dyn arrow::array::Array>;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "node_id",
        DataType::Int64,
        false,
    )]));

    let batch = match arrow::record_batch::RecordBatch::try_new(schema, vec![row_id_array]) {
        Ok(b) => b,
        Err(e) => {
            tracing::error!("FFI(Spark): Failed to create RecordBatch: {}", e);
            return -1;
        }
    };

    let struct_array: StructArray = batch.into();
    let array_data = struct_array.to_data();

    let (ffi_array, ffi_schema) = match to_ffi(&array_data) {
        Ok(tuple) => tuple,
        Err(e) => {
            tracing::error!("FFI(Spark): Error exporting to C Data Interface: {}", e);
            return -1;
        }
    };

    unsafe {
        std::ptr::write(out_array_ptr as *mut FFI_ArrowArray, ffi_array);
        std::ptr::write(out_schema_ptr as *mut FFI_ArrowSchema, ffi_schema);
    }

    result_len as jint
}

// -----------------------------------------------------------------------------
// Trino write / merge path
// -----------------------------------------------------------------------------

/// Import an Arrow batch from the C Data Interface.
///
/// # Safety
/// `array_ptr`/`schema_ptr` must point to valid, owned `FFI_ArrowArray` /
/// `FFI_ArrowSchema` structs exported by the caller; ownership transfers here.
unsafe fn import_batch(
    array_ptr: jlong,
    schema_ptr: jlong,
) -> anyhow::Result<arrow::record_batch::RecordBatch> {
    if array_ptr == 0 || schema_ptr == 0 {
        anyhow::bail!("null Arrow C Data Interface pointers");
    }
    let array = std::ptr::read(array_ptr as *const FFI_ArrowArray);
    let schema = std::ptr::read(schema_ptr as *const FFI_ArrowSchema);
    let data = arrow::ffi::from_ffi(array, &schema)?;
    let struct_array = StructArray::from(data);
    Ok(arrow::record_batch::RecordBatch::from(struct_array))
}

/// Trino: return the table's Arrow schema as JSON.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_getTableSchema(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
) -> jstring {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    if uri.is_empty() {
        return std::ptr::null_mut();
    }
    let json = match Table::new(uri) {
        Ok(table) => schema_to_json(&table.arrow_schema()),
        Err(e) => {
            tracing::error!("FFI(Trino): getTableSchema failed: {}", e);
            return std::ptr::null_mut();
        }
    };
    match env.new_string(json) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Trino: append an Arrow batch to the table and commit.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_appendBatch(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    in_array_ptr: jlong,
    in_schema_ptr: jlong,
) -> jboolean {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    if uri.is_empty() {
        return 0;
    }
    let batch = match unsafe { import_batch(in_array_ptr, in_schema_ptr) } {
        Ok(b) => b,
        Err(e) => {
            tracing::error!("FFI(Trino): appendBatch import failed: {}", e);
            return 0;
        }
    };
    let rows = batch.num_rows();
    // The JVM caller holds this batch while the engine buffers its own copy;
    // declare it so the ingest back-pressure ignores the caller's footprint.
    let caller_bytes = batch.get_array_memory_size() as u64;
    let res = RUNTIME.block_on(async {
        let table = Table::new_async(uri).await?;
        table.set_caller_reserved_bytes(caller_bytes);
        table.write_async(vec![batch]).await?;
        table.commit_async().await?;
        Ok::<(), anyhow::Error>(())
    });
    match res {
        Ok(()) => {
            tracing::info!("FFI(Trino): appended {} rows", rows);
            1
        }
        Err(e) => {
            tracing::error!("FFI(Trino): appendBatch failed: {}", e);
            0
        }
    }
}

/// Trino: merge (upsert) an Arrow batch on the given key columns.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_mergeRows(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    key_columns: JString,
    in_array_ptr: jlong,
    in_schema_ptr: jlong,
) -> jboolean {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let keys: String = env
        .get_string(&key_columns)
        .map(|s| s.into())
        .unwrap_or_default();
    if uri.is_empty() || keys.is_empty() {
        return 0;
    }
    let batch = match unsafe { import_batch(in_array_ptr, in_schema_ptr) } {
        Ok(b) => b,
        Err(e) => {
            tracing::error!("FFI(Trino): mergeRows import failed: {}", e);
            return 0;
        }
    };
    // The JVM caller holds this batch while the engine buffers its own copy;
    // declare it so the ingest back-pressure ignores the caller's footprint.
    let caller_bytes = batch.get_array_memory_size() as u64;
    let res = RUNTIME.block_on(async {
        let table = Table::new_async(uri).await?;
        table.set_caller_reserved_bytes(caller_bytes);
        // `Table::merge` drives its own runtime via `block_on`, so run it on a
        // blocking thread rather than inside the async context.
        tokio::task::spawn_blocking(move || {
            table.merge(
                vec![batch],
                &keys,
                crate::core::table::MergeMode::MergeOnRead,
            )
        })
        .await
        .map_err(|e| anyhow::anyhow!("merge task panicked: {e}"))?
    });
    match res {
        Ok(()) => 1,
        Err(e) => {
            tracing::error!("FFI(Trino): mergeRows failed: {}", e);
            0
        }
    }
}

/// Trino: delete rows matching a filter.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_deleteRows(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    filter: JString,
) -> jboolean {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let filter: String = env
        .get_string(&filter)
        .map(|s| s.into())
        .unwrap_or_default();
    if uri.is_empty() || filter.is_empty() {
        return 0;
    }
    let res = RUNTIME.block_on(async {
        let table = Table::new_async(uri).await?;
        table.delete_async(&filter).await
    });
    match res {
        Ok(()) => 1,
        Err(e) => {
            tracing::error!("FFI(Trino): deleteRows failed: {}", e);
            0
        }
    }
}

// The schema JSON helpers live in `crate::core::jni_util` so they can be fuzzed
// without the `java` feature (and therefore without a JVM).
use crate::core::jni_util::{schema_from_json, schema_to_json};

/// Trino: return the table's primary-key column names as a JSON array.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_getPrimaryKey(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
) -> jstring {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let json = if uri.is_empty() {
        "[]".to_string()
    } else {
        match Table::new(uri) {
            Ok(table) => {
                serde_json::to_string(&table.get_primary_key()).unwrap_or_else(|_| "[]".to_string())
            }
            Err(_) => "[]".to_string(),
        }
    };
    match env.new_string(json) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Trino: create a table from a `[{name, type, nullable}]` JSON schema.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_createTable(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    schema_json: JString,
) -> jboolean {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let json: String = env
        .get_string(&schema_json)
        .map(|s| s.into())
        .unwrap_or_default();
    if uri.is_empty() || json.is_empty() {
        return 0;
    }
    let schema = match schema_from_json(&json) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("FFI(Trino): createTable schema parse failed: {}", e);
            return 0;
        }
    };
    let res = RUNTIME.block_on(async {
        Table::create_async(uri, std::sync::Arc::new(schema))
            .await
            .map(|_| ())
    });
    match res {
        Ok(()) => 1,
        Err(e) => {
            tracing::error!("FFI(Trino): createTable failed: {}", e);
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Trino: SQL query pushdown + warehouse metadata listing
// ---------------------------------------------------------------------------

/// A materialized SQL query result, streamed batch-by-batch to the JNI caller.
///
/// Unlike [`BenoStreamSession`] (a low-level file-range reader), this runs the
/// query through the engine's DataFusion session, so the planner applies the
/// full pushdown surface — scalar/inverted indexes and vector search — before
/// any Parquet is decoded.
pub struct QuerySession {
    batches: Vec<arrow::record_batch::RecordBatch>,
    idx: usize,
}

impl QuerySession {
    fn next_batch(&mut self) -> Option<arrow::record_batch::RecordBatch> {
        if self.idx < self.batches.len() {
            let batch = self.batches[self.idx].clone();
            self.idx += 1;
            Some(batch)
        } else {
            None
        }
    }
}

/// Trino: run a SQL query against a table (registered as `t`) and return a
/// handle to the materialized result.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_openQuery(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    sql: JString,
) -> jlong {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    let query: String = env.get_string(&sql).map(|s| s.into()).unwrap_or_default();
    if uri.is_empty() || query.is_empty() {
        tracing::warn!("FFI(Trino): openQuery called with empty uri or sql");
        return 0;
    }
    let res = RUNTIME.block_on(async {
        let table = Table::new_async(uri).await?;
        table.sql(&query).await
    });
    match res {
        Ok(batches) => {
            tracing::info!("FFI(Trino): openQuery produced {} batch(es)", batches.len());
            Box::into_raw(Box::new(QuerySession { batches, idx: 0 })) as jlong
        }
        Err(e) => {
            tracing::error!("FFI(Trino): openQuery failed: {}", e);
            0
        }
    }
}

/// Trino: read the next batch from a query handle (C Data Interface export).
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_readQueryBatch(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    out_array_ptr: jlong,
    out_schema_ptr: jlong,
) -> jlong {
    if handle == 0 || out_array_ptr == 0 || out_schema_ptr == 0 {
        tracing::warn!("FFI(Trino): readQueryBatch called with null handle or output pointers");
        return 0;
    }
    let session = unsafe { &mut *(handle as *mut QuerySession) };
    match session.next_batch() {
        Some(batch) => {
            let struct_array: arrow::array::StructArray = batch.into();
            let array_data = struct_array.to_data();
            let (ffi_array, ffi_schema) = match to_ffi(&array_data) {
                Ok(tuple) => tuple,
                Err(e) => {
                    tracing::error!("FFI(Trino): readQueryBatch export failed: {}", e);
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

/// Trino: free a query handle.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_closeQuery(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            drop(Box::from_raw(handle as *mut QuerySession));
        }
    }
}

/// Immediate sub-directory names under `prefix` in the warehouse store.
fn list_subdirs(
    store: &std::sync::Arc<dyn object_store::ObjectStore>,
    prefix: &str,
) -> Vec<String> {
    // Derive sub-directory names from the objects actually present rather than
    // from `list_with_delimiter`'s common prefixes. A local filesystem keeps
    // empty directories after a DROP TABLE deletes every object, so the common
    // prefixes would make a dropped table keep appearing in SHOW TABLES.
    let prefix_path = object_store::path::Path::from(prefix);
    let prefix_depth = prefix_path.parts().count();
    let res = RUNTIME.block_on(async {
        let mut stream = store.list(Some(&prefix_path));
        let mut names = std::collections::BTreeSet::new();
        while let Some(item) = stream.next().await {
            let meta = item?;
            let subdir = meta
                .location
                .parts()
                .nth(prefix_depth)
                .map(|p| p.as_ref().to_string());
            if let Some(name) = subdir {
                // Skip hidden/placeholder entries such as the `.keep` marker that
                // CREATE SCHEMA writes so empty namespaces persist on object stores.
                if name.starts_with('.') {
                    continue;
                }
                names.insert(name);
            }
        }
        Ok::<_, object_store::Error>(names)
    });
    match res {
        Ok(names) => names.into_iter().collect(),
        Err(e) => {
            tracing::error!("FFI(Trino): list_subdirs({}) failed: {}", prefix, e);
            Vec::new()
        }
    }
}

/// Trino: list the schemas (top-level directories) under the warehouse.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_listSchemas(
    mut env: JNIEnv,
    _class: JClass,
    warehouse: JString,
) -> jstring {
    let wh: String = env
        .get_string(&warehouse)
        .map(|s| s.into())
        .unwrap_or_default();
    let names = match create_object_store(&wh) {
        Ok(store) => list_subdirs(&store, ""),
        Err(e) => {
            tracing::error!("FFI(Trino): listSchemas store failed: {}", e);
            Vec::new()
        }
    };
    let json = serde_json::to_string(&names).unwrap_or_else(|_| "[]".to_string());
    match env.new_string(json) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Trino: list the tables (sub-directories) under `<warehouse>/<schema>`.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_listTables(
    mut env: JNIEnv,
    _class: JClass,
    warehouse: JString,
    schema: JString,
) -> jstring {
    let wh: String = env
        .get_string(&warehouse)
        .map(|s| s.into())
        .unwrap_or_default();
    let sch: String = env
        .get_string(&schema)
        .map(|s| s.into())
        .unwrap_or_default();
    let names = match create_object_store(&wh) {
        Ok(store) => list_subdirs(&store, &sch),
        Err(e) => {
            tracing::error!("FFI(Trino): listTables store failed: {}", e);
            Vec::new()
        }
    };
    let json = serde_json::to_string(&names).unwrap_or_else(|_| "[]".to_string());
    match env.new_string(json) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Trino: create a schema (a top-level directory under the warehouse).
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_createSchema(
    mut env: JNIEnv,
    _class: JClass,
    warehouse: JString,
    schema: JString,
) -> jboolean {
    let wh: String = env
        .get_string(&warehouse)
        .map(|s| s.into())
        .unwrap_or_default();
    let sch: String = env
        .get_string(&schema)
        .map(|s| s.into())
        .unwrap_or_default();
    if wh.is_empty() || sch.is_empty() {
        return 0;
    }
    let res = (|| -> anyhow::Result<()> {
        let store = create_object_store(&wh)?;
        // Object stores have no real directories; a placeholder object makes the
        // schema appear as a common prefix in `list_with_delimiter`.
        let path = object_store::path::Path::from(format!("{sch}/.keep"));
        RUNTIME.block_on(async { store.put(&path, bytes::Bytes::new().into()).await })?;
        Ok(())
    })();
    match res {
        Ok(()) => 1,
        Err(e) => {
            tracing::error!("FFI(Trino): createSchema failed: {}", e);
            0
        }
    }
}

/// Trino: drop a table by deleting every object under its URI.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_dropTable(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
) -> jboolean {
    let uri: String = env
        .get_string(&table_uri)
        .map(|s| s.into())
        .unwrap_or_default();
    if uri.is_empty() {
        return 0;
    }
    let res = (|| -> anyhow::Result<()> {
        let store = create_object_store(&uri)?;
        let prefix = object_store::path::Path::from("");
        RUNTIME.block_on(async {
            let mut stream = store.list(Some(&prefix));
            while let Some(item) = stream.next().await {
                let meta = item?;
                store.delete(&meta.location).await?;
            }
            Ok::<(), anyhow::Error>(())
        })?;
        // Invalidate the manifest caches immediately: otherwise a subsequent
        // `getTableSchema`/`getTableHandle` still sees the deleted manifest
        // through `LATEST_VERSION_CACHE`/`MANIFEST_CACHE` (short TTLs), so a
        // `CREATE TABLE` right after `DROP TABLE` reports "already exists".
        let manager = crate::core::manifest::ManifestManager::new(store, "", &uri);
        RUNTIME.block_on(manager.invalidate_caches());
        Ok(())
    })();
    match res {
        Ok(()) => 1,
        Err(e) => {
            tracing::error!("FFI(Trino): dropTable failed: {}", e);
            0
        }
    }
}

/// Spark: render the engine's Prometheus metrics as text.
///
/// The connector registers this with Spark's metrics system so the host's
/// existing Prometheus/JMX sink picks up the engine's metrics.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_gatherMetrics(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    match env.new_string(crate::core::telemetry::render_metrics()) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Trino: render the engine's Prometheus metrics as text.
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_gatherMetrics(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    match env.new_string(crate::core::telemetry::render_metrics()) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// Spark: JNI mirrors of the Trino data/metadata surface.
//
// The Spark connector is native-backed (like the Trino connector) instead of
// wrapping Iceberg's Java `SparkTable`, so it needs the same JNI entry points
// under the `com.benostreamdb.spark.jni.BenoStreamJNIBridge` class name. JNI
// resolves a native method to a symbol derived from its *declaring class*, so
// these thin wrappers forward to the shared Trino implementations (identical
// bodies, no duplication).
// ---------------------------------------------------------------------------

/// Spark: free a partition-reader session handle (the per-file `BenoStreamSession`).
#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_BenoStreamPartitionReader_closeSession(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            drop(Box::from_raw(handle as *mut BenoStreamSession));
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_getTableSchema(
    env: JNIEnv,
    class: JClass,
    table_uri: JString,
) -> jstring {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_getTableSchema(env, class, table_uri)
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_appendBatch(
    env: JNIEnv,
    class: JClass,
    table_uri: JString,
    in_array_ptr: jlong,
    in_schema_ptr: jlong,
) -> jboolean {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_appendBatch(
        env,
        class,
        table_uri,
        in_array_ptr,
        in_schema_ptr,
    )
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_mergeRows(
    env: JNIEnv,
    class: JClass,
    table_uri: JString,
    key_columns: JString,
    in_array_ptr: jlong,
    in_schema_ptr: jlong,
) -> jboolean {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_mergeRows(
        env,
        class,
        table_uri,
        key_columns,
        in_array_ptr,
        in_schema_ptr,
    )
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_deleteRows(
    env: JNIEnv,
    class: JClass,
    table_uri: JString,
    filter: JString,
) -> jboolean {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_deleteRows(env, class, table_uri, filter)
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_getPrimaryKey(
    env: JNIEnv,
    class: JClass,
    table_uri: JString,
) -> jstring {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_getPrimaryKey(env, class, table_uri)
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_createTable(
    env: JNIEnv,
    class: JClass,
    table_uri: JString,
    schema_json: JString,
) -> jboolean {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_createTable(
        env,
        class,
        table_uri,
        schema_json,
    )
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_openQuery(
    env: JNIEnv,
    class: JClass,
    table_uri: JString,
    sql: JString,
) -> jlong {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_openQuery(env, class, table_uri, sql)
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_readQueryBatch(
    env: JNIEnv,
    class: JClass,
    handle: jlong,
    out_array_ptr: jlong,
    out_schema_ptr: jlong,
) -> jlong {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_readQueryBatch(
        env,
        class,
        handle,
        out_array_ptr,
        out_schema_ptr,
    )
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_closeQuery(
    env: JNIEnv,
    class: JClass,
    handle: jlong,
) {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_closeQuery(env, class, handle)
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_listSchemas(
    env: JNIEnv,
    class: JClass,
    warehouse: JString,
) -> jstring {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_listSchemas(env, class, warehouse)
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_listTables(
    env: JNIEnv,
    class: JClass,
    warehouse: JString,
    schema: JString,
) -> jstring {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_listTables(env, class, warehouse, schema)
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_createSchema(
    env: JNIEnv,
    class: JClass,
    warehouse: JString,
    schema: JString,
) -> jboolean {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_createSchema(env, class, warehouse, schema)
}

#[no_mangle]
pub extern "system" fn Java_com_benostreamdb_spark_jni_BenoStreamJNIBridge_dropTable(
    env: JNIEnv,
    class: JClass,
    table_uri: JString,
) -> jboolean {
    Java_com_benostreamdb_trino_BenoStreamDBJNIBridge_dropTable(env, class, table_uri)
}

#[cfg(test)]
mod list_subdirs_tests {
    use super::*;

    /// A dropped table must disappear from `SHOW TABLES` even though a local
    /// filesystem keeps the (now empty) directory behind.
    #[test]
    fn dropped_table_is_not_listed() {
        let dir = std::env::temp_dir().join(format!("bsdb_list_subdirs_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = create_object_store(dir.to_str().unwrap()).unwrap();

        RUNTIME.block_on(async {
            for path in ["default/t1/metadata/a.json", "default/t2/metadata/b.json"] {
                store
                    .put(
                        &object_store::path::Path::from(path),
                        bytes::Bytes::from_static(b"x").into(),
                    )
                    .await
                    .unwrap();
            }
        });
        assert_eq!(
            list_subdirs(&store, "default"),
            vec!["t1".to_string(), "t2".to_string()]
        );

        // Drop t1: delete every object, leaving the empty directory behind.
        RUNTIME.block_on(async {
            let mut stream = store.list(Some(&object_store::path::Path::from("default/t1")));
            while let Some(item) = stream.next().await {
                store.delete(&item.unwrap().location).await.unwrap();
            }
        });
        assert_eq!(
            list_subdirs(&store, "default"),
            vec!["t2".to_string()],
            "a dropped table must not be listed"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `DROP TABLE` must invalidate the manifest caches, otherwise a
    /// `CREATE TABLE` immediately after reports "already exists".
    #[test]
    fn drop_invalidates_manifest_cache() {
        let dir = std::env::temp_dir().join(format!("bsdb_drop_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let uri = dir.to_str().unwrap().to_string();
        let store = create_object_store(&uri).unwrap();

        // Write a manifest (v1) so the table "exists".
        let bytes = serde_json::to_vec(&crate::core::manifest::Manifest::default()).unwrap();
        RUNTIME.block_on(async {
            store
                .put(
                    &object_store::path::Path::from("_manifest/v1.json"),
                    bytes.into(),
                )
                .await
                .unwrap();
        });

        let manager = crate::core::manifest::ManifestManager::new(store.clone(), "", &uri);
        let (_, ver) = RUNTIME.block_on(manager.load_latest()).unwrap();
        assert_eq!(ver, 1, "table should be visible before the drop");

        // Delete every object (as dropTable does) and invalidate the caches.
        RUNTIME.block_on(async {
            let mut stream = store.list(Some(&object_store::path::Path::from("")));
            while let Some(item) = stream.next().await {
                store.delete(&item.unwrap().location).await.unwrap();
            }
        });
        RUNTIME.block_on(manager.invalidate_caches());

        let (_, ver) = RUNTIME.block_on(manager.load_latest()).unwrap();
        assert_eq!(
            ver, 0,
            "dropped table must not be visible through the cache"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The `.keep` placeholder written by CREATE SCHEMA must not surface as a
    /// table in SHOW TABLES, while the namespace that contains it stays listed.
    #[test]
    fn hidden_keep_marker_is_not_listed() {
        let dir = std::env::temp_dir().join(format!("bsdb_keep_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = create_object_store(dir.to_str().unwrap()).unwrap();

        RUNTIME.block_on(async {
            for path in ["default/.keep", "default/t1/metadata/a.json"] {
                store
                    .put(
                        &object_store::path::Path::from(path),
                        bytes::Bytes::from_static(b"x").into(),
                    )
                    .await
                    .unwrap();
            }
        });
        assert_eq!(
            list_subdirs(&store, ""),
            vec!["default".to_string()],
            "namespace holding only a .keep marker must still be listed"
        );
        assert_eq!(
            list_subdirs(&store, "default"),
            vec!["t1".to_string()],
            "the .keep placeholder must not be listed as a table"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
