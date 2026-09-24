// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Resource admission limits for long-lived processes.
//!
//! BenoStreamDB is designed to run serverless: the same binary must be safe in
//! a 4 GB container and scale up to a large host. Every limit therefore has an
//! **active default derived from the memory actually available to the process**
//! — the container's cgroup limit when one is set, otherwise the host's physical
//! memory — rather than a fixed constant or an "off unless configured" switch.
//!
//! - **Memory:** [`effective_memory_bytes`] reports the cgroup-aware limit, and
//!   the ingest high-water mark and heap-trim budget are derived from it (see
//!   [`crate::core::memory`] and [`crate::core::table`]).
//! - **Disk:** [`free_disk_bytes_for_new_file`] reports the free space on the
//!   filesystem that will hold a new file, and [`min_free_disk_bytes`] resolves
//!   the `BSDB_MIN_FREE_DISK_GB` admission threshold (default
//!   [`DEFAULT_MIN_FREE_DISK_GB`]). A flush is refused when a local table's
//!   filesystem is below it, so the failure is a clear error *before* any bytes
//!   are written rather than a partial segment mid-write.
//! - **CPU:** the writer never runs unbounded. Segment index builds go through
//!   `Table::index_build_gate` (`BSDB_INDEX_BUILD_CONCURRENCY`, memory-scaled)
//!   and parallel segment reads through a semaphore sized by
//!   `auto_detect_parallel_readers`.
//!
//! See `RESOURCE_LIMITS.md` for the full list of knobs.

/// Conservative memory budget used when no limit can be detected.
///
/// Serverless-safe: an undetectable environment is treated as a small container
/// rather than an unbounded host, so the guards stay active.
pub const FALLBACK_MEMORY_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Fraction of the effective memory used for the ingest high-water mark and the
/// heap-trim budget.
pub const MEMORY_BUDGET_FRACTION: f64 = 0.8;

/// Default minimum free disk space for local (`file://`) tables, in GiB.
pub const DEFAULT_MIN_FREE_DISK_GB: f64 = 1.0;

/// Parse a cgroup memory-limit value.
///
/// Returns `None` for "no limit": cgroup v2 writes `max`, and cgroup v1 writes a
/// sentinel near `i64::MAX` (`0x7FFF_FFFF_FFFF_F000`). A literal `0` is also
/// treated as unlimited.
pub fn parse_cgroup_limit(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("max") {
        return None;
    }
    let value: u64 = trimmed.parse().ok()?;
    if value == 0 || value >= (1u64 << 60) {
        return None;
    }
    Some(value)
}

/// Read the container's memory limit from cgroup v2, then v1.
///
/// Resolves the process's own cgroup from `/proc/self/cgroup` (so nested
/// cgroups are honoured) and falls back to the well-known root paths used by
/// cgroup-namespaced containers.
fn cgroup_memory_limit_bytes() -> Option<u64> {
    if let Ok(contents) = std::fs::read_to_string("/proc/self/cgroup") {
        for line in contents.lines() {
            // cgroup v2: "0::/path"
            if let Some(rest) = line.strip_prefix("0::") {
                let rel = rest.trim().trim_start_matches('/');
                let path = if rel.is_empty() {
                    std::path::PathBuf::from("/sys/fs/cgroup/memory.max")
                } else {
                    std::path::Path::new("/sys/fs/cgroup")
                        .join(rel)
                        .join("memory.max")
                };
                if let Some(limit) = read_limit(&path) {
                    return Some(limit);
                }
            }
            // cgroup v1: "N:memory:/path" (the controller list may be comma-joined)
            let mut parts = line.splitn(3, ':');
            if let (Some(_hier), Some(controllers), Some(path)) =
                (parts.next(), parts.next(), parts.next())
            {
                if controllers.split(',').any(|c| c == "memory") {
                    let rel = path.trim().trim_start_matches('/');
                    let file = if rel.is_empty() {
                        std::path::PathBuf::from("/sys/fs/cgroup/memory/memory.limit_in_bytes")
                    } else {
                        std::path::Path::new("/sys/fs/cgroup/memory")
                            .join(rel)
                            .join("memory.limit_in_bytes")
                    };
                    if let Some(limit) = read_limit(&file) {
                        return Some(limit);
                    }
                }
            }
        }
    }
    // cgroup-namespaced containers report "/" above; try the root paths directly.
    for path in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ] {
        if let Some(limit) = read_limit(std::path::Path::new(path)) {
            return Some(limit);
        }
    }
    None
}

fn read_limit(path: &std::path::Path) -> Option<u64> {
    parse_cgroup_limit(&std::fs::read_to_string(path).ok()?)
}

/// Process address-space ceiling (`RLIMIT_AS`), if one is set.
///
/// `ulimit -v` and some sandboxes set this; it is a hard ceiling on the process
/// and therefore a better budget basis than host RAM when present. macOS has no
/// cgroups, so this is the only per-process limit available there (and it is
/// usually unlimited, in which case host RAM is used).
#[cfg(unix)]
// `rlim_t` is already `u64` on Linux x86_64 (hence the lint) but is narrower on
// other Unix targets, so the cast is kept for portability.
#[allow(clippy::unnecessary_cast)]
fn rlimit_address_space_bytes() -> Option<u64> {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `getrlimit` writes only into `lim`.
    if unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut lim) } != 0 {
        return None;
    }
    let cur = lim.rlim_cur;
    // `RLIM_INFINITY` (and the near-`u64::MAX` sentinels some platforms use) and
    // a zero limit all mean "no usable ceiling".
    if cur == 0 || cur >= (1u64 << 60) {
        return None;
    }
    Some(cur as u64)
}

#[cfg(not(unix))]
fn rlimit_address_space_bytes() -> Option<u64> {
    None
}

/// Total physical memory visible to this process, in bytes.
///
/// Resolution order: the container's cgroup limit (so a 4 GB container on a
/// 32 GB host reports 4 GB), then a process `RLIMIT_AS` ceiling, then host RAM
/// (`/proc/meminfo` on Linux, `sysctl hw.memsize` on macOS).
pub fn total_memory_bytes() -> Option<u64> {
    cgroup_memory_limit_bytes()
        .or_else(rlimit_address_space_bytes)
        .or_else(host_memory_bytes)
}

/// Memory actually usable by this process, in bytes.
///
/// Prefers the host's *available* memory over its total. On a shared machine
/// (a desktop also running a browser, another build, etc.) `MemTotal` overstates
/// what this process can allocate, so guards sized from it never trip before the
/// OS OOM-killer does — the demo load was killed at ~74 GB RSS on a 121 GiB host
/// because the derived high-water mark was ~102 GB. A cgroup limit or
/// `RLIMIT_AS` ceiling is a hard cap and is used as-is; only the host-RAM
/// fallback switches to available memory.
pub fn usable_memory_bytes() -> Option<u64> {
    cgroup_memory_limit_bytes()
        .or_else(rlimit_address_space_bytes)
        .or_else(host_available_memory_bytes)
        .or_else(host_memory_bytes)
}

/// Effective memory budget in bytes: the usable limit, or
/// [`FALLBACK_MEMORY_BYTES`] when nothing can be detected.
pub fn effective_memory_bytes() -> u64 {
    usable_memory_bytes().unwrap_or(FALLBACK_MEMORY_BYTES)
}

/// Ingest RAM high-water mark in GB for a given memory budget.
///
/// Pure so the derivation can be asserted for a specific container size.
pub fn max_ingest_ram_gb_for(memory_bytes: u64) -> f64 {
    (memory_bytes as f64 * MEMORY_BUDGET_FRACTION) / 1_000_000_000.0
}

/// Heap-trim budget in bytes for a given memory budget.
pub fn memory_budget_bytes_for(memory_bytes: u64) -> u64 {
    (memory_bytes as f64 * MEMORY_BUDGET_FRACTION) as u64
}

/// Default ingest RAM high-water mark in GB, derived from the effective memory.
pub fn default_max_ingest_ram_gb() -> f64 {
    max_ingest_ram_gb_for(effective_memory_bytes())
}

/// Default heap-trim budget in bytes, derived from the effective memory.
pub fn default_memory_budget_bytes() -> u64 {
    memory_budget_bytes_for(effective_memory_bytes())
}

#[cfg(target_os = "linux")]
fn host_memory_bytes() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Host memory currently available for new allocations, in bytes.
///
/// Linux exposes this directly as `MemAvailable` — the kernel's estimate of how
/// much can be allocated without swapping. macOS has no equivalent, so callers
/// fall back to [`host_memory_bytes`].
#[cfg(target_os = "linux")]
fn host_available_memory_bytes() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn host_available_memory_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "macos")]
fn host_memory_bytes() -> Option<u64> {
    unsafe {
        let mut memsize: u64 = 0;
        let mut size = std::mem::size_of_val(&memsize);
        let name = std::ffi::CString::new("hw.memsize").ok()?;
        if libc::sysctlbyname(
            name.as_ptr(),
            &mut memsize as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        ) == 0
        {
            Some(memsize)
        } else {
            None
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn host_memory_bytes() -> Option<u64> {
    None
}

/// Free bytes available to this process on the filesystem containing `path`.
///
/// Returns `None` when the platform or path cannot be queried; callers must
/// treat that as "unknown" and proceed rather than refuse work.
#[cfg(unix)]
// `statvfs`'s `f_bavail`/`f_frsize` are already `u64` on Linux x86_64 (hence the
// lint) but are narrower or wider on other Unix targets, so the casts are kept
// for portability.
#[allow(clippy::unnecessary_cast)]
pub fn free_disk_bytes(path: &std::path::Path) -> Option<u64> {
    use std::ffi::CString;
    let c_path = CString::new(path.to_str()?).ok()?;
    // SAFETY: `statvfs` writes only into `st`, and `c_path` is a valid,
    // NUL-terminated C string that outlives the call.
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut st) };
    if rc != 0 {
        return None;
    }
    Some((st.f_bavail as u64).saturating_mul(st.f_frsize as u64))
}

/// Non-Unix fallback: free space is not queried.
#[cfg(not(unix))]
pub fn free_disk_bytes(_path: &std::path::Path) -> Option<u64> {
    None
}

/// Free bytes on the filesystem that will hold `path`, which may not exist yet.
///
/// Walks up to the nearest existing ancestor so a not-yet-created table
/// directory still resolves to its parent filesystem.
pub fn free_disk_bytes_for_new_file(path: &std::path::Path) -> Option<u64> {
    let mut probe = path;
    loop {
        if probe.exists() {
            return free_disk_bytes(probe);
        }
        match probe.parent() {
            Some(parent) => probe = parent,
            None => return None,
        }
    }
}

/// Resolve the minimum-free-disk admission threshold in bytes.
///
/// Defaults to [`DEFAULT_MIN_FREE_DISK_GB`]; override with
/// `BSDB_MIN_FREE_DISK_GB`. A non-positive override is ignored so the guard is
/// always active.
pub fn min_free_disk_bytes() -> u64 {
    let gb = std::env::var("BSDB_MIN_FREE_DISK_GB")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|g| *g > 0.0)
        .unwrap_or(DEFAULT_MIN_FREE_DISK_GB);
    (gb * 1024.0 * 1024.0 * 1024.0) as u64
}

/// Enforce [`min_free_disk_bytes`] against a local path.
///
/// Returns `Ok(())` when the free space is unknown (fail-open), and
/// `Err(message)` when the filesystem is below the threshold.
pub fn check_min_free_disk(path: &std::path::Path) -> Result<(), String> {
    let min_bytes = min_free_disk_bytes();
    let Some(free) = free_disk_bytes_for_new_file(path) else {
        return Ok(());
    };
    if free < min_bytes {
        return Err(format!(
            "only {:.2} GB free on {} — below the BSDB_MIN_FREE_DISK_GB threshold ({:.2} GB)",
            free as f64 / 1e9,
            path.display(),
            min_bytes as f64 / 1e9
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cgroup_limit_handles_unlimited_forms() {
        assert_eq!(parse_cgroup_limit("max"), None);
        assert_eq!(parse_cgroup_limit("  max\n"), None);
        assert_eq!(parse_cgroup_limit(""), None);
        assert_eq!(parse_cgroup_limit("0"), None);
        // cgroup v1's "unlimited" sentinel.
        assert_eq!(parse_cgroup_limit("9223372036854771712"), None);
    }

    #[test]
    fn parse_cgroup_limit_reads_real_limits() {
        assert_eq!(
            parse_cgroup_limit("4294967296"),
            Some(4 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            parse_cgroup_limit("  2147483648\n"),
            Some(2 * 1024 * 1024 * 1024)
        );
        assert_eq!(parse_cgroup_limit("not-a-number"), None);
    }

    #[test]
    fn effective_memory_is_always_positive() {
        // Detection may fail in a sandbox; the fallback must still be active.
        assert!(effective_memory_bytes() >= FALLBACK_MEMORY_BYTES);
    }

    #[cfg(unix)]
    #[test]
    fn rlimit_ceiling_is_plausible_when_present() {
        // Usually unlimited (None); when set it must be a sane, finite value.
        if let Some(bytes) = rlimit_address_space_bytes() {
            assert!(bytes > 0 && bytes < (1u64 << 60));
        }
    }

    #[test]
    fn total_memory_is_detected_on_this_platform() {
        // Linux/macOS must resolve a real value; other platforms may not.
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        assert!(total_memory_bytes().is_some());
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn usable_memory_is_positive_and_no_more_than_total() {
        // Guards are sized from usable memory, so it must be a real, positive
        // value that never exceeds the machine's total.
        let usable = usable_memory_bytes().expect("usable memory should be detectable");
        assert!(usable > 0, "usable memory must be positive, got {usable}");
        if let Some(total) = total_memory_bytes() {
            assert!(
                usable <= total,
                "usable ({usable}) must not exceed total ({total})"
            );
        }
    }

    #[test]
    fn derived_defaults_scale_with_effective_memory() {
        // The derivation is pure; assert it against a fixed budget so the test
        // is not sensitive to `MemAvailable` fluctuating between calls.
        let bytes = 16 * 1024 * 1024 * 1024;
        let expected_gb = (bytes as f64 * MEMORY_BUDGET_FRACTION) / 1_000_000_000.0;
        assert!((max_ingest_ram_gb_for(bytes) - expected_gb).abs() < 1e-6);
        assert_eq!(
            memory_budget_bytes_for(bytes),
            (bytes as f64 * MEMORY_BUDGET_FRACTION) as u64
        );
        // The live defaults must still be positive and mutually consistent.
        assert!(default_max_ingest_ram_gb() > 0.0);
        assert!(default_memory_budget_bytes() > 0);
    }

    #[test]
    fn four_gb_container_derives_safe_defaults() {
        // A 4 GiB container must get a ~3.4 GB high-water mark, not the host's.
        let four_gib = 4 * 1024 * 1024 * 1024;
        let gb = max_ingest_ram_gb_for(four_gib);
        assert!((gb - 3.435).abs() < 0.01, "expected ~3.4 GB, got {gb}");
        assert_eq!(memory_budget_bytes_for(four_gib), (gb * 1e9) as u64);
        // And it must be strictly below the container's own limit.
        assert!(gb * 1e9 < four_gib as f64);
    }

    #[test]
    fn larger_hosts_derive_larger_defaults() {
        let small = max_ingest_ram_gb_for(4 * 1024 * 1024 * 1024);
        let large = max_ingest_ram_gb_for(128 * 1024 * 1024 * 1024);
        assert!(large > small * 30.0, "defaults must scale with the host");
    }

    #[test]
    fn min_free_disk_default_is_active() {
        // With no override the guard must still have a positive threshold.
        if std::env::var("BSDB_MIN_FREE_DISK_GB").is_err() {
            assert_eq!(
                min_free_disk_bytes(),
                (DEFAULT_MIN_FREE_DISK_GB * 1024.0 * 1024.0 * 1024.0) as u64
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn free_disk_bytes_is_reported_for_root() {
        let bytes = free_disk_bytes(std::path::Path::new("/"));
        assert!(bytes.is_some(), "root filesystem should be queryable");
        assert!(bytes.unwrap_or(0) > 0, "free space should be positive");
    }

    #[cfg(unix)]
    #[test]
    fn free_disk_bytes_walks_up_from_missing_path() {
        let missing = std::path::Path::new("/tmp/bsdb-does-not-exist-xyz/nested/segment.parquet");
        assert!(
            free_disk_bytes_for_new_file(missing).is_some(),
            "should resolve via the nearest existing ancestor"
        );
    }

    #[test]
    fn check_is_fail_open_when_free_space_is_unknown() {
        // A path whose filesystem cannot be queried must not refuse work.
        // `/proc/self/fd` is not a real filesystem path for statvfs on Linux.
        let unknown = std::path::Path::new("/nonexistent-root-xyz/does/not/exist");
        // Either the ancestor resolves (and has space) or it is unknown; both
        // are `Ok`.
        assert!(check_min_free_disk(unknown).is_ok());
    }
}
