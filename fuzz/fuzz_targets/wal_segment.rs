// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Fuzz the WAL segment header validator, Arrow IPC reader, torn-EOF tail
//! recovery wrapper, and `benostream:tx_id` schema metadata parser.
//!
//! Arbitrary bytes must either recover a valid batch prefix or return `Err` —
//! never panic, abort, or trigger an unbounded allocation.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = benostreamdb::core::wal::fuzz_replay_wal_bytes(data);
});
