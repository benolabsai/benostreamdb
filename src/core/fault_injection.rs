// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Deterministic fault injection for crash-recovery testing (WS2).
//!
//! The production-readiness review's gate #1 is: *kill the process at every
//! commit/WAL/manifest/index boundary and prove recovery is deterministic.*
//! A real `SIGKILL` cannot be observed from inside the same process, so this
//! module models a crash as a **named boundary that aborts the operation** with
//! an error at exactly the point a `SIGKILL` would land. The caller then
//! re-opens the table and asserts the invariants (atomicity, durability, no
//! torn snapshots, WAL recovery).
//!
//! The design is deliberately zero-cost when disarmed: every instrumented site
//! calls [`check`], which is a single relaxed atomic load on the hot path.
//!
//! ```no_run
//! use benostreamdb::core::fault_injection::{arm, disarm, CrashPoint};
//! arm(CrashPoint::ManifestCommit);
//! // ... run a write/commit; it will fail at the manifest commit boundary ...
//! disarm();
//! ```

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// A named crash boundary in the write/commit/maintenance lifecycle.
///
/// The discriminants are stable and used as the atomic payload, so new points
/// must be appended (never renumbered) to keep the encoding unambiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CrashPoint {
    /// No injection armed.
    None = 0,
    /// After a batch is appended to the WAL, before it is fsynced.
    WalAppend = 1,
    /// After the WAL is fsynced, before the manifest commit.
    WalFlush = 2,
    /// After the manifest commit, before the WAL is truncated.
    WalTruncate = 3,
    /// After data files are staged, before they are uploaded to the store.
    DataUpload = 4,
    /// After index files are built, before they are uploaded.
    IndexUpload = 5,
    /// Immediately before the manifest commit (the atomic publish point).
    ManifestCommit = 6,
    /// Immediately after the manifest commit returns (visibility delay).
    ManifestVisible = 7,
    /// At the start of a compaction, before any new file is written.
    CompactionStart = 8,
    /// After compaction files are written, before the manifest swap.
    CompactionManifestSwap = 9,
    /// At the start of a vacuum, before any artifact is deleted.
    VacuumStart = 10,
    /// After the vacuum has decided what to delete, before deleting it.
    VacuumDelete = 11,
}

impl CrashPoint {
    /// Every injectable boundary, in lifecycle order. Used by the sweep driver.
    pub const ALL: &'static [CrashPoint] = &[
        CrashPoint::WalAppend,
        CrashPoint::WalFlush,
        CrashPoint::WalTruncate,
        CrashPoint::DataUpload,
        CrashPoint::IndexUpload,
        CrashPoint::ManifestCommit,
        CrashPoint::ManifestVisible,
        CrashPoint::CompactionStart,
        CrashPoint::CompactionManifestSwap,
        CrashPoint::VacuumStart,
        CrashPoint::VacuumDelete,
    ];

    /// Stable, human-readable name (also used in the injected error message).
    pub fn as_str(&self) -> &'static str {
        match self {
            CrashPoint::None => "none",
            CrashPoint::WalAppend => "wal_append",
            CrashPoint::WalFlush => "wal_flush",
            CrashPoint::WalTruncate => "wal_truncate",
            CrashPoint::DataUpload => "data_upload",
            CrashPoint::IndexUpload => "index_upload",
            CrashPoint::ManifestCommit => "manifest_commit",
            CrashPoint::ManifestVisible => "manifest_visible",
            CrashPoint::CompactionStart => "compaction_start",
            CrashPoint::CompactionManifestSwap => "compaction_manifest_swap",
            CrashPoint::VacuumStart => "vacuum_start",
            CrashPoint::VacuumDelete => "vacuum_delete",
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => CrashPoint::WalAppend,
            2 => CrashPoint::WalFlush,
            3 => CrashPoint::WalTruncate,
            4 => CrashPoint::DataUpload,
            5 => CrashPoint::IndexUpload,
            6 => CrashPoint::ManifestCommit,
            7 => CrashPoint::ManifestVisible,
            8 => CrashPoint::CompactionStart,
            9 => CrashPoint::CompactionManifestSwap,
            10 => CrashPoint::VacuumStart,
            11 => CrashPoint::VacuumDelete,
            _ => CrashPoint::None,
        }
    }
}

impl std::fmt::Display for CrashPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The armed crash point (`0` == disarmed). A single global is sufficient: the
/// sweep driver runs one injection at a time, and tests that need isolation use
/// `#[serial]`-style discipline (the sweep is a single test).
static ARMED: AtomicU8 = AtomicU8::new(0);
/// How many times an injection has actually fired (diagnostics / assertions).
static HITS: AtomicUsize = AtomicUsize::new(0);

/// Arm the injector to fire at `point`. Fires at most once, then auto-disarms.
pub fn arm(point: CrashPoint) {
    ARMED.store(point as u8, Ordering::SeqCst);
}

/// Disarm the injector.
pub fn disarm() {
    ARMED.store(0, Ordering::SeqCst);
}

/// The currently armed point, if any.
pub fn armed() -> Option<CrashPoint> {
    match ARMED.load(Ordering::SeqCst) {
        0 => None,
        v => Some(CrashPoint::from_u8(v)),
    }
}

/// Number of times an injection has fired since process start.
pub fn hits() -> usize {
    HITS.load(Ordering::SeqCst)
}

/// Reset the hit counter (test helper).
pub fn reset_hits() {
    HITS.store(0, Ordering::SeqCst);
}

/// The crash-injection check. Call at every boundary.
///
/// Returns `Ok(())` immediately when disarmed (one relaxed load). When armed for
/// `point`, it disarms and returns an error — modelling a process death at that
/// boundary: the operation aborts *before* the next durable step.
#[inline]
pub fn check(point: CrashPoint) -> anyhow::Result<()> {
    let armed = ARMED.load(Ordering::Relaxed);
    if armed == 0 {
        return Ok(());
    }
    if armed == point as u8
        && ARMED
            .compare_exchange(armed, 0, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    {
        HITS.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("injected crash at boundary '{}'", point.as_str());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disarmed_check_is_ok() {
        disarm();
        assert!(check(CrashPoint::ManifestCommit).is_ok());
        assert!(armed().is_none());
    }

    #[test]
    fn armed_check_fires_once_then_disarms() {
        reset_hits();
        arm(CrashPoint::WalFlush);
        assert!(check(CrashPoint::WalFlush).is_err());
        // Auto-disarmed: a second check at the same point is a no-op.
        assert!(check(CrashPoint::WalFlush).is_ok());
        assert!(armed().is_none());
        assert_eq!(hits(), 1);
        disarm();
    }

    #[test]
    fn armed_point_does_not_fire_at_other_points() {
        reset_hits();
        arm(CrashPoint::DataUpload);
        assert!(check(CrashPoint::ManifestCommit).is_ok());
        assert!(check(CrashPoint::DataUpload).is_err());
        disarm();
    }

    #[test]
    fn all_points_round_trip_through_u8() {
        for p in CrashPoint::ALL {
            assert_eq!(CrashPoint::from_u8(*p as u8), *p);
            assert_ne!(p.as_str(), "none");
        }
    }
}
