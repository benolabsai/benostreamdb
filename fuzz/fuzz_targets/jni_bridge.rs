// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Fuzz the JNI bridge's untrusted-input parsing.
//!
//! The JNI entry points themselves need a `JNIEnv` (a JVM), but the logic they
//! run on Java-supplied strings is pure Rust: the `[{name, type, nullable}]`
//! schema JSON parsed by `createTable`, and the Arrow type-name mapping. The
//! GA requirement is that malformed input returns an error, never panics.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        // JNI `createTable`: parse the schema JSON.
        let _ = benostreamdb::core::jni_util::schema_from_json(s);
        // JNI `getTableSchema`/`createTable`: map an Arrow type name.
        let _ = benostreamdb::core::jni_util::arrow_type_from_str(s);
    }
});
