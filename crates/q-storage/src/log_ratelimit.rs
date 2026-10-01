//! A tiny lock-free "log at most once per interval" gate.
//!
//! Measured 2026-10-01 on a fresh syncing node: `[KALMAN] Rejecting invalid bandwidth: 0` at
//! ~20 lines/s and `Unknown gossipsub topic … dropping` every ~2 s, both at WARN, drowning the
//! lines that matter. Put one `static` per call site; `check()` says whether to emit now and
//! how many lines were swallowed since the last one.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct RateLimiter {
    last_secs: AtomicU64,
    suppressed: AtomicU64,
    interval_secs: u64,
}

impl RateLimiter {
    pub const fn new(interval_secs: u64) -> Self {
        Self { last_secs: AtomicU64::new(0), suppressed: AtomicU64::new(0), interval_secs }
    }

    /// `Some(n)` = log now, and `n` earlier lines were suppressed; `None` = stay quiet.
    pub fn check(&self) -> Option<u64> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        self.check_at(now)
    }

    pub fn check_at(&self, now_secs: u64) -> Option<u64> {
        let last = self.last_secs.load(Ordering::Relaxed);
        if last == 0 || now_secs.saturating_sub(last) >= self.interval_secs {
            // Only one racing caller wins the slot; the losers count as suppressed.
            if self
                .last_secs
                .compare_exchange(last, now_secs.max(1), Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Some(self.suppressed.swap(0, Ordering::Relaxed));
            }
        }
        self.suppressed.fetch_add(1, Ordering::Relaxed);
        None
    }
}

// Tests: crates/q-storage/tests/integrity_guards_tests.rs (the lib unit-test target does not
// compile at 1e36503e4, so in-module tests would never run).
