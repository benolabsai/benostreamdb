use jni::JNIEnv;
use jni::objects::{JClass, JObject, JString};
use jni::sys::{jlong, jstring};
// use std::sync::Arc;
use tokio::runtime::Runtime;
use crate::core::reader::HybridReader;
use crate::SegmentConfig;
use crate::core::storage::create_object_store;
use crate::core::table::Table;
use futures::StreamExt;
use lazy_static::lazy_static;

lazy_static! {
    static ref RUNTIME: Runtime = Runtime::new().unwrap();
}

pub struct HyperStreamSession {
    reader: HybridReader,
    current_batches: Vec<arrow::record_batch::RecordBatch>,
    current_idx: usize,
}

impl HyperStreamSession {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        // Path expected: s3://bucket/path/to/segment_001.parquet or file:///path/to/file.parquet
        
        // 1. Extract Parent (Store Root) and Segment ID
        let (parent_uri, segment_id) = if let Some(idx) = path.rfind('/') {
            let parent = &path[..idx];
            let filename = &path[idx+1..];
            let seg_id = filename.strip_suffix(".parquet").unwrap_or(filename);
            (parent, seg_id)
        } else {
            // No slash? assume current dir? Unlikely for URI.
            (".", path)
        };

        // 2. Create Store pointing to parent directory
        let store = create_object_store(parent_uri)?;
        
        // 3. Config with just the ID
        let config = SegmentConfig::new("", segment_id); 
        
        let reader = HybridReader::new(config, store, path);
        Ok(Self {
            reader,
            current_batches: vec![],
            current_idx: 0,
        })
    }
    
    pub fn next_batch(&mut self) -> Option<arrow::record_batch::RecordBatch> {
        // Simple buffering logic: load everything once? 
        // Or implement async stream polling in sync context.
        if self.current_batches.is_empty() {
             // Load from Reader (blocking on global runtime)
             // In real impl, keep the stream open.
             let res = RUNTIME.block_on(async {
                 // FFI reads all columns by default
                 let mut stream = self.reader.stream_all(None).await?;
                 let mut batches = Vec::new();
                 while let Some(batch_result) = stream.next().await {
                     batches.push(batch_result?);
                 }
                 Ok::<Vec<arrow::record_batch::RecordBatch>, anyhow::Error>(batches)
             });
             match res {
                 Ok(batches) => {
                     self.current_batches = batches;
                     self.current_idx = 0;
                 },
                 Err(e) => {
                     eprintln!("Error reading batches: {}", e);
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

/// Native method implementation for `com.hyperstreamdb.trino.HyperStreamDBPageSource.openSession`
#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_trino_HyperStreamDBPageSource_openSession(
    mut env: JNIEnv,
    _class: JClass,
    path: JString,
) -> jlong {
    let path_str: String = match env.get_string(&path) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    
    println!("FFI: Opening Session to {}", path_str);
    
    match HyperStreamSession::new(&path_str) {
        Ok(session) => {
            Box::into_raw(Box::new(session)) as jlong
        },
        Err(e) => {
            eprintln!("FFI Error opening session: {}", e);
            0
        }
    }
}

use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, to_ffi};

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
    if handle == 0 { return 0; }
    
    let session = unsafe { &mut *(handle as *mut HyperStreamSession) };
    
    match session.next_batch() {
        Some(batch) => {
            println!("FFI: Read batch with {} rows", batch.num_rows());
            
            // 1. Convert RecordBatch to StructArray
            let struct_array: arrow::array::StructArray = batch.into();
            let array_data = struct_array.to_data(); // to_data is often inherent, or via Array trait

            // 2. Export to C Data Interface
            // to_ffi returns (FFI_ArrowArray, FFI_ArrowSchema)
            // We need to move these into the pointers provided by Java
            
            let (ffi_array, ffi_schema) = match to_ffi(&array_data) {
                Ok(tuple) => tuple,
                Err(e) => {
                    eprintln!("FFI Error exporting to C Data Interface: {}", e);
                    return 0;
                }
            };
            
            unsafe {
                std::ptr::write(out_array_ptr as *mut FFI_ArrowArray, ffi_array);
                std::ptr::write(out_schema_ptr as *mut FFI_ArrowSchema, ffi_schema);
            }

            1 // Success
        },
        None => 0 // Finished
    }
}

/// Trino Integration: Split Generation
#[no_mangle]
pub extern "system" fn Java_com_hyperstreamdb_trino_HyperStreamDBSplitManager_getSplits(
    mut env: JNIEnv,
    _class: JClass,
    table_uri: JString,
    max_split_size: jlong,
) -> jstring {
    let uri: String = match env.get_string(&table_uri) {
        Ok(s) => s.into(),
        Err(_) => return env.new_string("[]").unwrap().into_raw(),
    };
    
    // Default 64MB if invalid
    let split_size = if max_split_size <= 0 { 64 * 1024 * 1024 } else { max_split_size as usize };

    println!("FFI: Getting splits for {} (max size: {})", uri, split_size);

    let splits_json = match Table::new(uri.clone()) {
        Ok(table) => {
            match table.get_splits(split_size) {
                Ok(splits) => {
                    serde_json::to_string(&splits).unwrap_or_else(|_| "[]".to_string())
                },
                Err(e) => {
                    eprintln!("FFI Error getting splits: {}", e);
                    "[]".to_string()
                }
            }
        },
        Err(e) => {
             eprintln!("FFI Error creating table: {}", e);
             "[]".to_string()
        }
    };
    
    env.new_string(splits_json).unwrap().into_raw()
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
        Err(_) => return env.new_string("[]").unwrap().into_raw(),
    };

    println!("FFI: Listing data files for {}", uri);

    // Call Table API
    // Note: Table::new and list_data_files are currently synchronous, 
    // potentially blocking on internal runtime for IO.
    let files_json = match Table::new(uri.clone()) {
        Ok(table) => {
            match table.list_data_files() {
                Ok(files) => {
                    serde_json::to_string(&files).unwrap_or_else(|_| "[]".to_string())
                },
                Err(e) => {
                    eprintln!("FFI Error listing files: {}", e);
                    "[]".to_string()
                }
            }
        },
        Err(e) => {
             eprintln!("FFI Error creating table: {}", e);
             "[]".to_string()
        }
    };

    env.new_string(files_json).unwrap().into_raw()
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
    env.new_string(splits_json).unwrap().into_raw()
}

// -----------------------------------------------------------------------------
// Spark Connector JNI Bridge
// -----------------------------------------------------------------------------

fn open_session_helper(mut env: JNIEnv, path: JString) -> jlong {
    let path_str: String = match env.get_string(&path) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    println!("FFI(Spark): Opening Session to {}", path_str);
    match HyperStreamSession::new(&path_str) {
        Ok(session) => Box::into_raw(Box::new(session)) as jlong,
        Err(e) => {
            eprintln!("FFI Error opening session: {}", e);
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

    if handle == 0 { return 0; }
    let session = unsafe { &mut *(handle as *mut HyperStreamSession) };

    match session.next_batch() {
        Some(batch) => {
            let struct_array: arrow::array::StructArray = batch.into();
            let array_data = struct_array.to_data();
            let (ffi_array, ffi_schema) = match arrow::ffi::to_ffi(&array_data) {
                Ok(tuple) => tuple,
                Err(e) => { eprintln!("FFI Error: {}", e); return 0; }
            };
            unsafe {
                std::ptr::write(out_array_ptr as *mut FFI_ArrowArray, ffi_array);
                std::ptr::write(out_schema_ptr as *mut FFI_ArrowSchema, ffi_schema);
            }
            1
        },
        None => 0
    }
}
