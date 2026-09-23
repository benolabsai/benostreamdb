// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Heap memory discipline for long-lived processes that rebuild indexes
//! in-process (the A4 ingest orchestrator, the compaction daemon).
//!
//! ## The problem
//! glibc gives each thread its own malloc arena and keeps freed memory in the
//! arena that released it. Heavy small-allocation churn — the HNSW/TQ builders
//! do millions of tiny allocations per segment — therefore ratchets RSS toward
//! the sum of every arena's high-water mark. Measured on the whole-site
//! Wikipedia node load: **82 GB RSS vs 18 GB live**. Capping arenas
//! (`M_ARENA_MAX=2`, see `tame_glibc_arenas` in `lib.rs`) bounds the *number* of
//! arenas but does not return the freed pages to the kernel.
//!
//! ## The fix
//! `malloc_trim(0)` walks the arenas and releases free pages back to the OS. We
//! call it at work-unit boundaries in the ingest loop, gated by a memory budget
//! so the (arena-walking) cost is only paid when RSS has actually grown.
//!
//! ## Allocator evaluation (A4)
//! - **jemalloc** (chosen): better fragmentation behaviour and returns memory to the OS
//!   naturally. With `tikv-jemallocator` without the `unprefixed` feature, it correctly
//!   only routes Rust allocations to jemalloc and avoids corrupting the Python runtime's
//!   own malloc state.
//! - **glibc + `malloc_trim`**: caused OS/driver deadlocks when the NVIDIA GPU driver
//!   was active concurrently with heap trimming. Removed.
//! - **mimalloc**: rejected — static-TLS failure under pyo3.
//! - **slab-allocating the HNSW/TQ builders**: the real fix (freed memory
//!   becomes reusable), but a large refactor of the index builders. Deferred.
//!
//! The demo's *external* strategy (a fresh process per chunk) resets the
//! allocator high-water mark entirely; `malloc_trim` is the in-process
//! equivalent for the library's `ingest_async` path.

/// Return freed heap pages to the OS.
pub fn trim_heap() -> bool {
    // No-op because jemalloc automatically returns memory to the OS
    // via background threads.
    false
}

/// Resident set size in bytes, if the platform exposes it.
#[cfg(target_os = "linux")]
pub fn rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
pub fn rss_bytes() -> Option<u64> {
    None
}

/// Default RSS budget for opportunistic trimming, in GiB.
pub const DEFAULT_MEMORY_BUDGET_GB: f64 = 8.0;

/// Resolve the RSS trim budget in bytes.
///
/// A positive `HDB_INGEST_MEMORY_BUDGET_GB` overrides `default_gb`.
pub fn memory_budget_bytes(default_gb: f64) -> u64 {
    let gb = std::env::var("HDB_INGEST_MEMORY_BUDGET_GB")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|g| *g > 0.0)
        .unwrap_or(default_gb);
    (gb * 1024.0 * 1024.0 * 1024.0) as u64
}

/// Trim the heap if RSS is over budget.
///
/// Convenience wrapper for one-shot trim sites (flush boundaries, the end of a
/// background index build) that don't keep a [`HeapTrimPolicy`]. Reading RSS is
/// a `/proc` read, so the under-budget path is cheap.
pub fn trim_if_over_budget(default_gb: f64) -> bool {
    if rss_bytes()
        .map(|r| r > memory_budget_bytes(default_gb))
        .unwrap_or(false)
    {
        trim_heap()
    } else {
        false
    }
}

/// Budget-gated heap trimming for a long-running loop.
///
/// Call [`HeapTrimPolicy::maybe_trim`] at work-unit boundaries. It only invokes
/// [`trim_heap`] when RSS exceeds the budget, so the trim cost is paid only when
/// memory has actually grown.
#[derive(Debug, Clone)]
pub struct HeapTrimPolicy {
    budget_bytes: u64,
    trims: u64,
    released_bytes: u64,
}

impl HeapTrimPolicy {
    /// Create a policy that trims whenever RSS exceeds `budget_bytes`.
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            trims: 0,
            released_bytes: 0,
        }
    }

    /// Resolve a policy from an explicit budget or the
    /// `HDB_INGEST_MEMORY_BUDGET_GB` environment variable. Returns `None` when
    /// neither is set (trimming disabled).
    pub fn from_budget_or_env(budget_bytes: Option<u64>) -> Option<Self> {
        if let Some(b) = budget_bytes {
            return Some(Self::new(b));
        }
        let gb: f64 = std::env::var("HDB_INGEST_MEMORY_BUDGET_GB")
            .ok()?
            .parse()
            .ok()?;
        if gb <= 0.0 {
            return None;
        }
        Some(Self::new((gb * 1024.0 * 1024.0 * 1024.0) as u64))
    }

    /// Trim if RSS exceeds the budget. Returns `true` when a trim ran.
    pub fn maybe_trim(&mut self) -> bool {
        let Some(rss) = rss_bytes() else {
            return false;
        };
        if rss <= self.budget_bytes {
            return false;
        }
        let trimmed = trim_heap();
        self.trims += 1;
        if let Some(after) = rss_bytes() {
            self.released_bytes += rss.saturating_sub(after);
        }
        trimmed
    }

    /// Number of trims performed.
    pub fn trims(&self) -> u64 {
        self.trims
    }

    /// Total bytes observed released across all trims.
    pub fn released_bytes(&self) -> u64 {
        self.released_bytes
    }

    /// The configured budget in bytes.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_heap_is_callable_and_idempotent() {
        // Must not panic on any platform. On glibc the return value is whether
        // anything was released (false is valid when the heap is already tight).
        let _ = trim_heap();
        let _ = trim_heap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn rss_bytes_reports_a_plausible_value() {
        let rss = rss_bytes().expect("VmRSS should be readable on linux");
        assert!(rss > 0, "rss should be positive, got {rss}");
    }

    #[test]
    fn policy_only_trims_over_budget() {
        // A huge budget means "never over" -> no trim, no counter movement.
        let mut q = HeapTrimPolicy::new(u64::MAX);
        assert!(!q.maybe_trim());
        assert_eq!(q.trims(), 0);
        assert_eq!(q.released_bytes(), 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn policy_trims_when_over_budget() {
        // Budget of 0 means "always over" -> a trim runs and is counted.
        let mut p = HeapTrimPolicy::new(0);
        p.maybe_trim();
        assert_eq!(p.trims(), 1);
    }

    #[test]
    fn policy_from_explicit_budget() {
        let p = HeapTrimPolicy::from_budget_or_env(Some(1024)).expect("explicit budget");
        assert_eq!(p.budget_bytes(), 1024);
    }
}
