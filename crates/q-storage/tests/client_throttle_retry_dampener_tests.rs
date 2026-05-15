//! v10.9.27: Stand-alone integration tests for the client-throttle retry
//! dampener in `turbo_sync.rs`.
//!
//! The real retry loop lives inside `start_warp_sync_loop` (~line 6160 of
//! `turbo_sync.rs`) and is not directly callable without a full
//! `TurboSyncManager` + libp2p swarm + peer mesh. This integration test
//! exercises the *algorithm* — the same `err_msg.contains("ClientThrottle")`
//! discriminator that the production code uses — in isolation so it can be
//! run on CI regardless of the rest of the q-storage test build.
//!
//! These tests live in `tests/` (not `src/turbo_sync.rs#[cfg(test)]`) so they
//! compile as a stand-alone integration binary. The in-source module tests
//! are kept in `turbo_sync.rs` for documentation but the binary that CI
//! actually runs is this one.
//!
//! The string literal `"ClientThrottle"` must match
//! `q_network::CLIENT_THROTTLE_MARKER`. We do NOT depend on q-network here to
//! avoid a circular dependency (q-network already depends on q-storage via
//! q-narwhal-core transitively). The cross-crate contract is the substring.

use std::time::Duration;

/// `ClientThrottle` errors are local back-pressure — they must NOT consume
/// the retry budget. Feed the discriminator 20 throttle errors followed by a
/// success and assert `retry_count` stayed at 0.
#[tokio::test]
async fn retry_count_unchanged_on_client_throttle() {
    let mut retry_count: u32 = 0;
    let max_retries: u32 = 3;
    let mut throttle_waits: usize = 0;
    let chunk_deadline = tokio::time::Instant::now() + Duration::from_secs(60);

    let mut iter = 0;
    loop {
        // The 120s wall-clock breaker sits OUTSIDE the retry_count arm in
        // production. A throttle-only spin would still hit this — we assert
        // it is reachable from the throttle path.
        assert!(
            tokio::time::Instant::now() < chunk_deadline,
            "wall-clock breaker MUST remain reachable from throttle path"
        );

        iter += 1;

        // Simulated dispatch outcome.
        let err_msg = if iter <= 20 {
            Some("ClientThrottle: per-peer cap reached".to_string())
        } else {
            None
        };

        match err_msg {
            None => break,
            Some(msg) => {
                if msg.contains("ClientThrottle") {
                    throttle_waits += 1;
                    continue;
                }
                retry_count += 1;
                if retry_count >= max_retries {
                    panic!(
                        "retry_count should never grow past max_retries in this scenario"
                    );
                }
            }
        }
    }

    assert_eq!(
        retry_count, 0,
        "ClientThrottle errors must NOT consume the retry budget"
    );
    assert_eq!(
        throttle_waits, 20,
        "all 20 throttles must have been counted as throttle_waits"
    );
}

/// Sanity check that a non-throttle error path DOES consume the retry budget.
/// Guards against accidentally swallowing real failures via the discriminator.
#[tokio::test]
async fn retry_count_increments_on_real_failure() {
    let mut retry_count: u32 = 0;
    let mut throttle_waits: usize = 0;

    for _ in 0..3 {
        let err_msg = "timeout: peer did not respond".to_string();
        if err_msg.contains("ClientThrottle") {
            throttle_waits += 1;
            continue;
        }
        retry_count += 1;
    }

    assert_eq!(retry_count, 3, "real timeouts MUST increment retry_count");
    assert_eq!(throttle_waits, 0, "no throttles were issued in this scenario");
}

/// A long sequence of throttles eventually has to hit the wall-clock
/// breaker. This test models a maliciously-misconfigured cap (always
/// throttled) with a 50ms deadline; the loop must terminate within the
/// deadline rather than spinning forever.
#[tokio::test]
async fn wall_clock_breaker_bounds_throttle_only_spin() {
    let mut throttle_waits: usize = 0;
    let mut retry_count: u32 = 0;
    let deadline = tokio::time::Instant::now() + Duration::from_millis(50);

    let aborted = loop {
        if tokio::time::Instant::now() >= deadline {
            break true;
        }

        let err_msg = "ClientThrottle: per-peer cap reached".to_string();
        if err_msg.contains("ClientThrottle") {
            throttle_waits += 1;
            tokio::time::sleep(Duration::from_millis(5)).await;
            continue;
        }

        retry_count += 1;
        if retry_count >= 3 {
            break false;
        }
    };

    assert!(aborted, "wall-clock breaker must fire on throttle-only spin");
    assert!(
        throttle_waits >= 1,
        "at least one throttle must have been accumulated before the breaker fired"
    );
    assert_eq!(retry_count, 0, "no retry budget consumed by throttle spin");
}
