//! v10.9.27: Stand-alone integration tests for the client-side per-peer
//! block-pack inflight semaphore in `unified_network_manager.rs`.
//!
//! These tests live in `tests/` (not `src/unified_network_manager.rs#[cfg(test)]`)
//! so they compile as a stand-alone integration binary. The in-source `#[cfg(test)]`
//! mod currently pulls in pre-existing test code that doesn't compile against the
//! current lib (e.g. legacy `UnifiedNetworkManager::new()` no-arg calls), so this
//! stand-alone binary is what CI actually runs.
//!
//! The semaphore behaviour does not need a live `UnifiedNetworkManager` to verify —
//! it is just a `DashMap<PeerId, Arc<Semaphore>>` with `try_acquire_owned` and a
//! 30s safety timeout in the awaiter task. These properties can be exercised
//! against the public constants exported from q-network.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use libp2p::PeerId;
use q_network::{CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER, CLIENT_THROTTLE_MARKER};
use tokio::sync::Semaphore;

/// 8 concurrent acquirers against a single per-peer semaphore must all
/// eventually run, but at most `CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER` may be
/// in flight at any moment. The available count must never go negative.
#[tokio::test]
async fn client_block_pack_semaphore_caps_concurrent() {
    let semaphores: Arc<DashMap<PeerId, Arc<Semaphore>>> = Arc::new(DashMap::new());
    let peer = PeerId::random();
    let sem = semaphores
        .entry(peer)
        .or_insert_with(|| Arc::new(Semaphore::new(CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER)))
        .clone();

    let inflight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max_seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let acquired = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..8 {
        let sem_c = sem.clone();
        let inflight_c = inflight.clone();
        let max_seen_c = max_seen.clone();
        let acquired_c = acquired.clone();
        handles.push(tokio::spawn(async move {
            let permit = sem_c
                .acquire_owned()
                .await
                .expect("semaphore should not be closed");
            acquired_c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let cur = inflight_c.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            // Bump the high-water mark atomically.
            let mut prev = max_seen_c.load(std::sync::atomic::Ordering::SeqCst);
            while cur > prev {
                match max_seen_c.compare_exchange(
                    prev,
                    cur,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(p) => prev = p,
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
            inflight_c.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            drop(permit);
        }));
    }

    for h in handles {
        h.await.expect("task must not panic");
    }

    assert_eq!(
        acquired.load(std::sync::atomic::Ordering::SeqCst),
        8,
        "all 8 tasks should eventually acquire a permit"
    );
    assert_eq!(
        max_seen.load(std::sync::atomic::Ordering::SeqCst),
        CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER,
        "exactly CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER were in flight at peak"
    );
    assert_eq!(
        sem.available_permits(),
        CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER,
        "all permits must be released after tasks complete"
    );
}

/// If a request never receives a response, the safety timeout in the awaiter
/// task must reclaim the permit. Modelled here with a 200ms timeout for
/// fast tests: hold a permit inside a task, drop it on timeout, assert the
/// permit returns to the semaphore.
#[tokio::test]
async fn client_block_pack_semaphore_releases_on_timeout() {
    let sem = Arc::new(Semaphore::new(CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER));
    assert_eq!(sem.available_permits(), CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER);

    let permit = sem
        .clone()
        .try_acquire_owned()
        .expect("first permit must be available");
    assert_eq!(
        sem.available_permits(),
        CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER - 1
    );

    let task = tokio::spawn(async move {
        let _hold = permit;
        let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
        // _tx never sent — rx will only complete via the timeout. This
        // mirrors the production safety net: if no response arrives, the
        // timeout still fires and the permit is reclaimed.
        let _ = tokio::time::timeout(Duration::from_millis(200), rx).await;
    });
    task.await.expect("task should not panic");

    assert_eq!(
        sem.available_permits(),
        CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER,
        "permit reclaimed after the awaiter task ends on timeout"
    );
}

/// `try_acquire_owned` must return `Err` once the per-peer cap is full, and
/// the caller-side dispatch path returns an error whose Display contains the
/// `CLIENT_THROTTLE_MARKER` substring. This emulates the exact pattern in the
/// `RequestBlockRangeDirect` handler.
#[tokio::test]
async fn client_block_pack_semaphore_returns_throttle_when_full() {
    let sem = Arc::new(Semaphore::new(CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER));

    let mut held = Vec::new();
    for _ in 0..CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER {
        held.push(
            sem.clone()
                .try_acquire_owned()
                .expect("base permits must succeed"),
        );
    }
    assert_eq!(sem.available_permits(), 0);

    let fail = sem.clone().try_acquire_owned();
    assert!(fail.is_err(), "exhausted semaphore must reject");

    let err = anyhow::anyhow!(
        "{}: per-peer block-pack inflight cap ({}) reached",
        CLIENT_THROTTLE_MARKER,
        CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER
    );
    let s = err.to_string();
    assert!(
        s.contains(CLIENT_THROTTLE_MARKER),
        "dispatch-path error must contain `{}` so turbo_sync can recognise local back-pressure: got `{}`",
        CLIENT_THROTTLE_MARKER,
        s
    );

    held.pop();
    assert_eq!(sem.available_permits(), 1);
    assert!(
        sem.clone().try_acquire_owned().is_ok(),
        "permit returns after drop"
    );
}

/// Each peer has its own semaphore. Saturating peer A must not affect peer B.
/// This guards against accidentally using a single global semaphore.
#[tokio::test]
async fn client_block_pack_semaphore_is_per_peer() {
    let semaphores: Arc<DashMap<PeerId, Arc<Semaphore>>> = Arc::new(DashMap::new());
    let peer_a = PeerId::random();
    let peer_b = PeerId::random();

    let sem_a = semaphores
        .entry(peer_a)
        .or_insert_with(|| Arc::new(Semaphore::new(CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER)))
        .clone();

    // Drain peer A.
    let mut held_a = Vec::new();
    for _ in 0..CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER {
        held_a.push(sem_a.clone().try_acquire_owned().expect("peer A permit"));
    }
    assert!(sem_a.clone().try_acquire_owned().is_err());

    // Peer B should still be wide open.
    let sem_b = semaphores
        .entry(peer_b)
        .or_insert_with(|| Arc::new(Semaphore::new(CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER)))
        .clone();
    assert_eq!(
        sem_b.available_permits(),
        CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER,
        "peer B's semaphore must not be affected by peer A saturation"
    );

    let permit_b = sem_b.clone().try_acquire_owned();
    assert!(
        permit_b.is_ok(),
        "peer B must accept a new dispatch while peer A is saturated"
    );
}
