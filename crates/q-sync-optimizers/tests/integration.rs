//! Cross-module integration tests for q-sync-optimizers.
//!
//! Each test exercises one item end-to-end in a way that mimics how
//! turbo_sync.rs will eventually call it. These tests live in `tests/` so
//! they only see the crate's *public* API — no private-item back doors.

use q_sync_optimizers::{
    BetaScoreRegistry, ChunkFloorEstimator, CubicRegistry, EwmvRtt, KalmanBdpEstimator,
    KalmanSnapshot, LittlesLawEstimator, MarkovRegistry, PeerFailFeatures, PeerFailPredictor,
    PeerState,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

// -----------------------------------------------------------------------------
// Item 1 — Kalman BDP estimator
// -----------------------------------------------------------------------------

#[test]
fn item1_kalman_bdp_known_value_and_fallback() {
    let est = KalmanBdpEstimator::new();
    // 100 Mbps × 100 ms → ~1221 KiB.
    let kb = est.chunk_size_kb(KalmanSnapshot::new(100.0, 100.0, 0.9));
    assert!((1220..=1221).contains(&kb), "got {kb}");

    // Low confidence → fallback 512 KiB.
    let fb = est.chunk_size_kb(KalmanSnapshot::new(100.0, 100.0, 0.1));
    assert_eq!(fb, 512);
}

#[test]
fn item1_kalman_floor_can_be_raised_by_item5() {
    // Couple Item 1 to Item 5: peer set of 16 with 8000 bits/block → 4 KiB floor.
    // Set floor higher to test the interaction.
    let mut est = KalmanBdpEstimator::new();
    let floor_est = ChunkFloorEstimator::new();
    let floor_kib = floor_est.floor_kib(16);
    est.set_floor_kb(floor_kib.max(128));
    // Tiny BDP would yield 64 KiB but we raised the floor to ≥128.
    let kb = est.chunk_size_kb(KalmanSnapshot::new(1.0, 1.0, 0.99));
    assert!(kb >= 128, "got {kb}");
}

// -----------------------------------------------------------------------------
// Item 2 — Bayesian peer reliability
// -----------------------------------------------------------------------------

#[test]
fn item2_thompson_sampling_picks_better_peer() {
    let mut reg = BetaScoreRegistry::<&'static str>::new();
    for _ in 0..50 {
        reg.record_success(&"good");
    }
    for _ in 0..50 {
        reg.record_failure(&"bad");
    }
    let mut rng = StdRng::seed_from_u64(101);
    let mut good_picks = 0;
    for _ in 0..200 {
        if reg.thompson_pick(&["good", "bad"], &mut rng) == Some(&"good") {
            good_picks += 1;
        }
    }
    assert!(good_picks > 180, "good was picked {good_picks}/200");
}

// -----------------------------------------------------------------------------
// Item 3 — CUBIC AIMD
// -----------------------------------------------------------------------------

#[test]
fn item3_cubic_grows_and_halves_correctly() {
    let mut reg = CubicRegistry::<u32>::new();
    let initial = reg.cwnd(&1);
    for _ in 0..10 {
        reg.on_success(&1);
    }
    assert_eq!(reg.cwnd(&1), initial + 10);
    reg.on_loss(&1);
    // (initial + 10) * 0.7 = 26 * 0.7 = 18.2 → 18
    assert_eq!(reg.cwnd(&1), 18);
}

// -----------------------------------------------------------------------------
// Item 4 — Little's law (combined with CUBIC)
// -----------------------------------------------------------------------------

#[test]
fn item4_littles_law_combined_with_cubic_takes_min() {
    let mut ll = LittlesLawEstimator::new();
    // 30 ms RTT, 1000 b/s target → L = 30.
    ll.record_rtt_ms(30.0);
    assert_eq!(ll.optimal_inflight(), Some(30));
    // CUBIC says 50 → min is 30.
    assert_eq!(ll.combined_with_cubic(50), 30);
    // CUBIC says 10 → min is 10.
    assert_eq!(ll.combined_with_cubic(10), 10);
}

// -----------------------------------------------------------------------------
// Item 5 — info-theoretic chunk floor
// -----------------------------------------------------------------------------

#[test]
fn item5_floor_matches_spec_example() {
    // 8 peers × 8000 bits/block = 24 K bits → 3 KiB.
    let est = ChunkFloorEstimator::new();
    assert_eq!(est.floor_kib(8), 3);
}

// -----------------------------------------------------------------------------
// Item 6 — Markov peer state
// -----------------------------------------------------------------------------

#[test]
fn item6_markov_progresses_fast_to_slow_to_stalled() {
    let mut reg = MarkovRegistry::<u32>::new();
    let peer = 7u32;
    assert_eq!(reg.state(&peer), PeerState::Fast);
    // Establish baseline.
    reg.record_rtt(&peer, 50.0);
    // Two consecutive high-RTT samples → Slow.
    reg.record_rtt(&peer, 100.0);
    reg.record_rtt(&peer, 200.0);
    assert_eq!(reg.state(&peer), PeerState::Slow);
    // Timeout → Stalled.
    reg.record_timeout(&peer);
    assert_eq!(reg.state(&peer), PeerState::Stalled);
    // Success → Fast.
    reg.record_success(&peer);
    assert_eq!(reg.state(&peer), PeerState::Fast);
}

#[test]
fn item6_markov_rank_orders_fast_first() {
    let mut reg = MarkovRegistry::<&'static str>::new();
    reg.record_timeout(&"slow"); // Fast → Slow
    let ranked = reg.rank_by_state(&["slow", "healthy"]);
    assert_eq!(ranked, vec![&"healthy", &"slow"]);
}

// -----------------------------------------------------------------------------
// Item 7 — EWMV RTT adaptive timeout
// -----------------------------------------------------------------------------

#[test]
fn item7_ewmv_timeout_grows_with_jitter() {
    let mut steady = EwmvRtt::new();
    let mut bursty = EwmvRtt::new();
    for _ in 0..50 {
        steady.record(20.0);
    }
    for i in 0..50 {
        let s = if i % 2 == 0 { 20.0 } else { 5_000.0 };
        bursty.record(s);
    }
    assert!(bursty.timeout_ms() >= steady.timeout_ms());
    // And we're always inside the [5s, 120s] safety window.
    let t = bursty.timeout_ms();
    assert!((5_000.0..=120_000.0).contains(&t), "timeout {t} out of range");
}

// -----------------------------------------------------------------------------
// Item 8 — Logistic regression for failing peer
// -----------------------------------------------------------------------------

#[test]
fn item8_predictor_separates_healthy_from_failing() {
    let pred = PeerFailPredictor::new();
    let healthy = PeerFailFeatures::default();
    let failing = PeerFailFeatures {
        last_3_rtts_ms: [50.0, 200.0, 800.0],
        last_3_response_sizes: [1_000.0, 500.0, 100.0],
        secs_since_reconnect: 3600.0,
        recent_failure_count: 30,
    };
    let p_h = pred.p_fail(&healthy);
    let p_f = pred.p_fail(&failing);
    assert!(p_h < 0.3, "healthy p_fail = {p_h}");
    assert!(p_f > 0.9, "failing p_fail = {p_f}");
    assert!(!pred.should_skip(&healthy));
    assert!(pred.should_skip(&failing));
}

// -----------------------------------------------------------------------------
// Cross-module: scheduler-shaped end-to-end smoke test
// -----------------------------------------------------------------------------

/// Mimic one full chunk-scheduling decision using all 8 building blocks.
#[test]
fn cross_module_full_decision_path() {
    // Setup peer registries.
    let mut betas = BetaScoreRegistry::<&'static str>::new();
    let mut cubics = CubicRegistry::<&'static str>::new();
    let mut markov = MarkovRegistry::<&'static str>::new();
    let mut littles = std::collections::HashMap::<&'static str, LittlesLawEstimator>::new();
    let mut rtts = std::collections::HashMap::<&'static str, EwmvRtt>::new();
    let pred = PeerFailPredictor::new();

    let peers = ["epsilon", "beta", "gamma"];

    // Seed history: epsilon is great, beta is mediocre, gamma is failing.
    for _ in 0..30 {
        betas.record_success(&"epsilon");
        cubics.on_success(&"epsilon");
        markov.record_rtt(&"epsilon", 20.0);
        rtts.entry("epsilon").or_default().record(20.0);
        littles.entry("epsilon").or_default().record_rtt_ms(20.0);
    }
    for _ in 0..15 {
        betas.record_success(&"beta");
        betas.record_failure(&"beta");
        cubics.on_success(&"beta");
        markov.record_rtt(&"beta", 80.0);
        rtts.entry("beta").or_default().record(80.0);
        littles.entry("beta").or_default().record_rtt_ms(80.0);
    }
    for _ in 0..30 {
        betas.record_failure(&"gamma");
    }
    markov.record_rtt(&"gamma", 100.0);
    markov.record_timeout(&"gamma");
    markov.record_timeout(&"gamma");

    // 1. Rank by Markov state.
    let ranked = markov.rank_by_state(&peers);
    assert_eq!(ranked.first(), Some(&&"epsilon"));

    // 2. Filter out "about to fail" via Item 8.
    let gamma_feats = PeerFailFeatures {
        last_3_rtts_ms: [100.0, 200.0, 500.0],
        last_3_response_sizes: [500.0, 200.0, 100.0],
        secs_since_reconnect: 7200.0,
        recent_failure_count: 30,
    };
    assert!(pred.should_skip(&gamma_feats));

    // 3. Thompson sample remaining (epsilon, beta).
    let candidates: Vec<&'static str> = ranked
        .into_iter()
        .copied()
        .filter(|p| *p != "gamma")
        .collect();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let picked = betas.thompson_pick(&candidates, &mut rng).copied();
    assert!(picked.is_some());

    // 4. CUBIC + Little's law cap for the picked peer.
    let p = picked.unwrap();
    let cwnd = cubics.cwnd(&p);
    let combined = littles
        .get(p)
        .map(|l| l.combined_with_cubic(cwnd))
        .unwrap_or(cwnd);
    assert!(combined >= 1);

    // 5. Adaptive timeout for the picked peer.
    if let Some(rtt) = rtts.get(p) {
        let t = rtt.timeout_ms();
        assert!(
            (5_000.0..=120_000.0).contains(&t),
            "adaptive timeout {t} out of safety range"
        );
    }
}
