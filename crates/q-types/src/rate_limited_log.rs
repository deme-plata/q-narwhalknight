//! Rate-limited tracing helper for high-volume log sites.
//!
//! Generalizes the `LAST_LOG: AtomicU64` 10-second suppression pattern at
//! `crates/q-api-server/src/handlers.rs:9705` (SYNC GATE 503) so multiple
//! call sites can share the infrastructure without re-implementing it.
//!
//! Each [`RateLimitedLog`] keeps a map of subkey → (last_log_ts,
//! suppressed_count). On [`RateLimitedLog::check`], a `Some(suppressed)`
//! return means the caller SHOULD emit a log line (the count of
//! previously-suppressed entries since the last emit in that window);
//! `None` means suppress this call.
//!
//! ## Why this lives in q-types
//!
//! Both `q-network` and `q-storage` need to rate-limit logs, but their
//! dependency direction is `q-network → q-storage`. Placing the helper
//! in either crate creates a cycle. `q-types` is the common transitive
//! dependency.
//!
//! ## Usage
//!
//! ```ignore
//! use q_types::rate_limited_log::RateLimitedLog;
//! use q_types::rl_warn;
//!
//! static GOSSIP_FWD_FAIL: RateLimitedLog = RateLimitedLog::new(10);
//!
//! rl_warn!(GOSSIP_FWD_FAIL, &topic_str, "⚠️ Gossipsub forward FAILED topic={} reason={}", topic_str, reason);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Suppression window-keyed log limiter. `&'static` so callers can hold
/// a reference for the program's lifetime without arc-juggling.
pub struct RateLimitedLog {
    window_secs: u64,
    state: OnceLock<RwLock<HashMap<String, (AtomicU64, AtomicU64)>>>,
}

impl RateLimitedLog {
    /// Construct a new limiter with the given suppression window in seconds.
    /// `const fn` so callers can declare `static` instances.
    pub const fn new(window_secs: u64) -> Self {
        Self {
            window_secs,
            state: OnceLock::new(),
        }
    }

    /// Decide whether to emit a log for `subkey`.
    ///
    /// Returns `Some(suppressed_count)` when the caller should emit (the
    /// count is the number of entries suppressed since the previous emit
    /// for this subkey; 0 on the first call), `None` when the caller
    /// should NOT emit.
    pub fn check(&'static self, subkey: &str) -> Option<u64> {
        let map = self.state.get_or_init(|| RwLock::new(HashMap::new()));
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if let Ok(guard) = map.read() {
            if let Some((last, suppressed)) = guard.get(subkey) {
                let prev = last.load(Relaxed);
                if now >= prev.saturating_add(self.window_secs)
                    && last
                        .compare_exchange(prev, now, Relaxed, Relaxed)
                        .is_ok()
                {
                    return Some(suppressed.swap(0, Relaxed));
                }
                suppressed.fetch_add(1, Relaxed);
                return None;
            }
        }

        if let Ok(mut guard) = map.write() {
            guard
                .entry(subkey.to_string())
                .or_insert_with(|| (AtomicU64::new(now), AtomicU64::new(0)));
        }
        Some(0)
    }
}

#[macro_export]
macro_rules! rl_warn {
    ($limiter:expr, $subkey:expr, $($arg:tt)*) => {
        if let Some(__rl_suppressed) = $limiter.check($subkey) {
            if __rl_suppressed > 0 {
                tracing::warn!("{} (+{} suppressed)", format_args!($($arg)*), __rl_suppressed);
            } else {
                tracing::warn!($($arg)*);
            }
        }
    };
}

#[macro_export]
macro_rules! rl_error {
    ($limiter:expr, $subkey:expr, $($arg:tt)*) => {
        if let Some(__rl_suppressed) = $limiter.check($subkey) {
            if __rl_suppressed > 0 {
                tracing::error!("{} (+{} suppressed)", format_args!($($arg)*), __rl_suppressed);
            } else {
                tracing::error!($($arg)*);
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LIMITER: RateLimitedLog = RateLimitedLog::new(1);

    #[test]
    fn first_call_emits_with_zero_suppressed() {
        let r = TEST_LIMITER.check("first-call-unique-key");
        assert_eq!(r, Some(0), "first call must emit with 0 suppressed count");
    }

    #[test]
    fn second_immediate_call_suppressed() {
        let _ = TEST_LIMITER.check("second-immediate-key");
        let r = TEST_LIMITER.check("second-immediate-key");
        assert_eq!(r, None, "immediate second call within window must suppress");
    }

    #[test]
    fn distinct_subkeys_independent() {
        let r1 = TEST_LIMITER.check("subkey-a");
        let r2 = TEST_LIMITER.check("subkey-b");
        assert_eq!(r1, Some(0));
        assert_eq!(r2, Some(0));
    }

    #[test]
    fn window_expiry_emits_with_suppressed_count() {
        let _ = TEST_LIMITER.check("expiry-key");
        let _ = TEST_LIMITER.check("expiry-key");
        let _ = TEST_LIMITER.check("expiry-key");
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let r = TEST_LIMITER.check("expiry-key");
        assert!(matches!(r, Some(n) if n >= 2),
            "after window expiry, emit must report suppressed count, got {:?}", r);
    }
}
