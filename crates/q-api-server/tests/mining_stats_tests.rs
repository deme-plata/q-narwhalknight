//! Mining Statistics Tests
//!
//! v3.2.25-beta: Tests for mining stats tracking, miner identification, and reward calculations
//!
//! These tests verify:
//! - Multiple miners to same wallet are tracked separately
//! - Miner IDs are properly distinguished
//! - Hash rate aggregation works correctly
//! - SSE events contain proper miner identification
//!
//! Run with: cargo test --package q-api-server --test mining_stats_tests

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// MOCK STRUCTURES (mirrors production code)
// ============================================================================

/// Simulates MinerStats from instant_mining_rewards.rs
#[derive(Debug, Clone)]
pub struct MinerStats {
    pub address: String,
    pub worker_id: String,
    pub last_hashrate: f64,
    pub total_solutions: u64,
    pub last_seen: Instant,
    pub total_rewards_earned: u128,
}

/// Simulates MiningStatistics tracking
#[derive(Debug, Default)]
pub struct MiningStatistics {
    /// Key: "address:worker_id" for unique miner tracking
    miners: HashMap<String, MinerStats>,
    pub total_solutions_submitted: u64,
}

impl MiningStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update miner with default worker_id "direct" (OLD behavior - don't use!)
    pub fn update_miner(&mut self, address: String, hash_rate: f64) {
        self.update_miner_with_worker(address, hash_rate, "direct".to_string());
    }

    /// Update miner with specific worker_id (NEW behavior - v3.2.25-beta)
    pub fn update_miner_with_worker(&mut self, address: String, hash_rate: f64, worker_id: String) {
        let key = format!("{}:{}", address, worker_id);

        let stats = self.miners.entry(key).or_insert_with(|| MinerStats {
            address: address.clone(),
            worker_id: worker_id.clone(),
            last_hashrate: 0.0,
            total_solutions: 0,
            last_seen: Instant::now(),
            total_rewards_earned: 0,
        });

        stats.last_hashrate = hash_rate;
        stats.last_seen = Instant::now();
        stats.total_solutions += 1;
    }

    /// Get all miners (for iteration)
    pub fn get_all_miners(&self) -> Vec<&MinerStats> {
        self.miners.values().collect()
    }

    /// Get miners for a specific wallet address
    pub fn get_miners_for_address(&self, address: &str) -> Vec<&MinerStats> {
        self.miners
            .values()
            .filter(|m| m.address == address)
            .collect()
    }

    /// Get miner count for a wallet
    pub fn get_miner_count_for_address(&self, address: &str) -> usize {
        self.get_miners_for_address(address).len()
    }

    /// Get total hash rate for a wallet
    pub fn get_total_hashrate_for_address(&self, address: &str) -> f64 {
        self.get_miners_for_address(address)
            .iter()
            .map(|m| m.last_hashrate)
            .sum()
    }

    /// Get a specific miner by address and worker_id
    pub fn get_miner(&self, address: &str, worker_id: &str) -> Option<&MinerStats> {
        let key = format!("{}:{}", address, worker_id);
        self.miners.get(&key)
    }
}

// ============================================================================
// SINGLE MINER TESTS
// ============================================================================

mod single_miner_tests {
    use super::*;

    #[test]
    fn test_single_miner_registration() {
        let mut stats = MiningStatistics::new();
        let wallet = "qnk1234567890";

        stats.update_miner_with_worker(
            wallet.to_string(),
            100.0,
            "miner-001".to_string(),
        );

        let miners = stats.get_miners_for_address(wallet);
        assert_eq!(miners.len(), 1);
        assert_eq!(miners[0].worker_id, "miner-001");
        assert_eq!(miners[0].last_hashrate, 100.0);
    }

    #[test]
    fn test_single_miner_hashrate_update() {
        let mut stats = MiningStatistics::new();
        let wallet = "qnk1234567890";

        stats.update_miner_with_worker(wallet.to_string(), 100.0, "miner-001".to_string());
        stats.update_miner_with_worker(wallet.to_string(), 150.0, "miner-001".to_string());
        stats.update_miner_with_worker(wallet.to_string(), 200.0, "miner-001".to_string());

        let miner = stats.get_miner(wallet, "miner-001").unwrap();
        assert_eq!(miner.last_hashrate, 200.0);
        assert_eq!(miner.total_solutions, 3);
    }

    #[test]
    fn test_miner_with_default_worker_id() {
        let mut stats = MiningStatistics::new();
        let wallet = "qnk1234567890";

        // Using old method (should use "direct" as worker_id)
        stats.update_miner(wallet.to_string(), 100.0);

        let miner = stats.get_miner(wallet, "direct").unwrap();
        assert_eq!(miner.worker_id, "direct");
    }
}

// ============================================================================
// MULTIPLE MINERS SAME WALLET TESTS (The bug we fixed in v3.2.25-beta)
// ============================================================================

mod multi_miner_tests {
    use super::*;

    #[test]
    fn test_two_miners_same_wallet_tracked_separately() {
        let mut stats = MiningStatistics::new();
        let wallet = "qnk8207f268efae031bb1998cd0abe02a98bba69acb1d0ae0ed05ef6ceedc18f4f1";

        // Miner 1 submits
        stats.update_miner_with_worker(wallet.to_string(), 50.0, "miner-alpha".to_string());

        // Miner 2 submits
        stats.update_miner_with_worker(wallet.to_string(), 75.0, "miner-beta".to_string());

        // Should have 2 miners, not 1
        let count = stats.get_miner_count_for_address(wallet);
        assert_eq!(count, 2, "Two miners should be tracked separately, got {}", count);

        // Each miner should have their own stats
        let miner_alpha = stats.get_miner(wallet, "miner-alpha").unwrap();
        let miner_beta = stats.get_miner(wallet, "miner-beta").unwrap();

        assert_eq!(miner_alpha.last_hashrate, 50.0);
        assert_eq!(miner_beta.last_hashrate, 75.0);
    }

    #[test]
    fn test_five_miners_same_wallet() {
        let mut stats = MiningStatistics::new();
        let wallet = "qnk_test_wallet";

        for i in 1..=5 {
            stats.update_miner_with_worker(
                wallet.to_string(),
                10.0 * i as f64,
                format!("miner-{}", i),
            );
        }

        assert_eq!(stats.get_miner_count_for_address(wallet), 5);

        // Total hashrate should be 10 + 20 + 30 + 40 + 50 = 150
        let total = stats.get_total_hashrate_for_address(wallet);
        assert!((total - 150.0).abs() < 0.001, "Total hashrate should be 150, got {}", total);
    }

    #[test]
    fn test_multiple_wallets_multiple_miners() {
        let mut stats = MiningStatistics::new();

        // Wallet A: 2 miners
        stats.update_miner_with_worker("wallet_a".to_string(), 100.0, "a-miner-1".to_string());
        stats.update_miner_with_worker("wallet_a".to_string(), 200.0, "a-miner-2".to_string());

        // Wallet B: 3 miners
        stats.update_miner_with_worker("wallet_b".to_string(), 50.0, "b-miner-1".to_string());
        stats.update_miner_with_worker("wallet_b".to_string(), 60.0, "b-miner-2".to_string());
        stats.update_miner_with_worker("wallet_b".to_string(), 70.0, "b-miner-3".to_string());

        assert_eq!(stats.get_miner_count_for_address("wallet_a"), 2);
        assert_eq!(stats.get_miner_count_for_address("wallet_b"), 3);

        assert!((stats.get_total_hashrate_for_address("wallet_a") - 300.0).abs() < 0.001);
        assert!((stats.get_total_hashrate_for_address("wallet_b") - 180.0).abs() < 0.001);
    }

    #[test]
    fn test_old_behavior_would_merge_miners() {
        let mut stats = MiningStatistics::new();
        let wallet = "qnk_test_wallet";

        // Simulating OLD buggy behavior: both use "direct" worker_id
        stats.update_miner(wallet.to_string(), 50.0);
        stats.update_miner(wallet.to_string(), 75.0);

        // With old behavior, only 1 miner would be tracked (merged)
        let count = stats.get_miner_count_for_address(wallet);
        assert_eq!(count, 1, "Old behavior merges miners - got {} (expected 1)", count);

        // And hashrate would be overwritten (last value wins)
        let miner = stats.get_miner(wallet, "direct").unwrap();
        assert_eq!(miner.last_hashrate, 75.0, "Old behavior overwrites hashrate");
    }
}

// ============================================================================
// P2P MINER TRACKING TESTS
// ============================================================================

mod p2p_miner_tests {
    use super::*;

    #[test]
    fn test_p2p_miner_worker_id_format() {
        let mut stats = MiningStatistics::new();
        let wallet = "qnk_remote_miner";

        // P2P miners use "p2p:PEER_ID" format
        stats.update_miner_with_worker(
            wallet.to_string(),
            100.0,
            "p2p:12D3KooWQbKp6RYgZpC3dUCYou5LrVmd7pFa74rQj7rsK1sWUnfu".to_string(),
        );

        let miners = stats.get_miners_for_address(wallet);
        assert_eq!(miners.len(), 1);
        assert!(miners[0].worker_id.starts_with("p2p:"));
    }

    #[test]
    fn test_mixed_local_and_p2p_miners() {
        let mut stats = MiningStatistics::new();
        let wallet = "qnk_mixed_wallet";

        // Local miner
        stats.update_miner_with_worker(wallet.to_string(), 100.0, "local-miner-1".to_string());

        // P2P miner
        stats.update_miner_with_worker(
            wallet.to_string(),
            200.0,
            "p2p:12D3KooWABC123".to_string(),
        );

        let miners = stats.get_miners_for_address(wallet);
        assert_eq!(miners.len(), 2);

        let local_miners: Vec<_> = miners.iter().filter(|m| !m.worker_id.starts_with("p2p:")).collect();
        let p2p_miners: Vec<_> = miners.iter().filter(|m| m.worker_id.starts_with("p2p:")).collect();

        assert_eq!(local_miners.len(), 1);
        assert_eq!(p2p_miners.len(), 1);
    }
}

// ============================================================================
// CONCURRENT UPDATE TESTS
// ============================================================================

mod concurrent_tests {
    use super::*;
    use std::sync::Mutex;
    use std::thread;

    #[test]
    fn test_concurrent_miner_updates() {
        let stats = Arc::new(Mutex::new(MiningStatistics::new()));
        let wallet = "qnk_concurrent_test";
        let mut handles = vec![];

        for i in 0..10 {
            let stats_clone = Arc::clone(&stats);
            let wallet_clone = wallet.to_string();

            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let mut locked = stats_clone.lock().unwrap();
                    locked.update_miner_with_worker(
                        wallet_clone.clone(),
                        (i * 100 + j) as f64,
                        format!("miner-{}", i),
                    );
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        let locked = stats.lock().unwrap();
        let count = locked.get_miner_count_for_address(wallet);

        // Should have exactly 10 miners (one per thread)
        assert_eq!(count, 10, "Should have 10 distinct miners, got {}", count);
    }
}

// ============================================================================
// SSE EVENT TESTS (verify proper event formatting)
// ============================================================================

mod sse_event_tests {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct MiningStatsEvent {
        pub miner_address: String,
        pub total_rewards: f64,
        pub total_blocks_found: u64,
        pub current_balance: f64,
        pub avg_hash_rate: f64,
        pub miner_id: Option<String>,
        pub worker_id: Option<String>,
        pub timestamp: String,
    }

    fn create_sse_event(stats: &MinerStats) -> MiningStatsEvent {
        let miner_id = if stats.worker_id != "direct" && !stats.worker_id.starts_with("p2p:") {
            Some(stats.worker_id.clone())
        } else {
            None
        };

        MiningStatsEvent {
            miner_address: stats.address.clone(),
            total_rewards: 0.0,
            total_blocks_found: stats.total_solutions,
            current_balance: 0.0,
            avg_hash_rate: stats.last_hashrate,
            miner_id,
            worker_id: Some(stats.worker_id.clone()),
            timestamp: "2026-01-21T12:00:00Z".to_string(),
        }
    }

    #[test]
    fn test_sse_event_includes_miner_id() {
        let stats = MinerStats {
            address: "qnk_test".to_string(),
            worker_id: "my-custom-miner-id".to_string(),
            last_hashrate: 100.0,
            total_solutions: 5,
            last_seen: Instant::now(),
            total_rewards_earned: 0,
        };

        let event = create_sse_event(&stats);

        assert_eq!(event.worker_id, Some("my-custom-miner-id".to_string()));
        assert_eq!(event.miner_id, Some("my-custom-miner-id".to_string()));
    }

    #[test]
    fn test_sse_event_direct_miner_no_miner_id() {
        let stats = MinerStats {
            address: "qnk_test".to_string(),
            worker_id: "direct".to_string(),
            last_hashrate: 100.0,
            total_solutions: 5,
            last_seen: Instant::now(),
            total_rewards_earned: 0,
        };

        let event = create_sse_event(&stats);

        assert_eq!(event.worker_id, Some("direct".to_string()));
        assert_eq!(event.miner_id, None); // "direct" should not populate miner_id
    }

    #[test]
    fn test_sse_event_p2p_miner_no_miner_id() {
        let stats = MinerStats {
            address: "qnk_test".to_string(),
            worker_id: "p2p:12D3KooWABC".to_string(),
            last_hashrate: 100.0,
            total_solutions: 5,
            last_seen: Instant::now(),
            total_rewards_earned: 0,
        };

        let event = create_sse_event(&stats);

        assert_eq!(event.worker_id, Some("p2p:12D3KooWABC".to_string()));
        assert_eq!(event.miner_id, None); // P2P miners should not populate miner_id
    }

    #[test]
    fn test_multiple_sse_events_for_wallet() {
        let mut mining_stats = MiningStatistics::new();
        let wallet = "qnk_multi_miner";

        mining_stats.update_miner_with_worker(wallet.to_string(), 50.0, "miner-1".to_string());
        mining_stats.update_miner_with_worker(wallet.to_string(), 75.0, "miner-2".to_string());

        let all_miners = mining_stats.get_miners_for_address(wallet);
        let events: Vec<_> = all_miners.iter().map(|m| create_sse_event(m)).collect();

        // Should generate 2 events
        assert_eq!(events.len(), 2);

        // Each event should have unique miner_id
        let miner_ids: Vec<_> = events.iter()
            .filter_map(|e| e.miner_id.as_ref())
            .collect();
        assert_eq!(miner_ids.len(), 2);
        assert!(miner_ids.contains(&&"miner-1".to_string()));
        assert!(miner_ids.contains(&&"miner-2".to_string()));
    }
}

// ============================================================================
// REGRESSION TESTS (prevent re-introduction of bugs)
// ============================================================================

mod regression_tests {
    use super::*;

    /// Regression test for the multi-miner tracking bug fixed in v3.2.25-beta
    ///
    /// BUG: Multiple miners mining to the same wallet were all tracked as
    /// a single "direct" worker, causing UI to show only 1 miner with
    /// the hashrate of the last submission.
    ///
    /// FIX: Use miner_id from the mining submission as worker_id.
    #[test]
    fn test_regression_multi_miner_tracking_v3_2_25() {
        let mut stats = MiningStatistics::new();
        let wallet = "qnk8207f268efae031bb1998cd0abe02a98bba69acb1d0ae0ed05ef6ceedc18f4f1";

        // Simulate two miners submitting solutions
        // OLD BUG: Both would use "direct" worker_id
        // NEW FIX: Each uses their unique miner_id

        // Miner 1: miner_id = "abc123"
        stats.update_miner_with_worker(wallet.to_string(), 50.0, "abc123".to_string());

        // Miner 2: miner_id = "xyz789"
        stats.update_miner_with_worker(wallet.to_string(), 75.0, "xyz789".to_string());

        // EXPECTED: 2 miners tracked
        let count = stats.get_miner_count_for_address(wallet);
        assert_eq!(count, 2, "REGRESSION: Multi-miner tracking broken! Got {} miners (expected 2)", count);

        // EXPECTED: Both hashrates preserved
        let miner1 = stats.get_miner(wallet, "abc123").expect("REGRESSION: Miner 1 not found");
        let miner2 = stats.get_miner(wallet, "xyz789").expect("REGRESSION: Miner 2 not found");

        assert_eq!(miner1.last_hashrate, 50.0, "REGRESSION: Miner 1 hashrate overwritten");
        assert_eq!(miner2.last_hashrate, 75.0, "REGRESSION: Miner 2 hashrate overwritten");

        // EXPECTED: Total hashrate is sum of both
        let total = stats.get_total_hashrate_for_address(wallet);
        assert!((total - 125.0).abs() < 0.001, "REGRESSION: Total hashrate wrong ({} != 125)", total);
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_miner_lookup_performance() {
        let mut stats = MiningStatistics::new();

        // Add 1000 different miners
        for i in 0..1000 {
            stats.update_miner_with_worker(
                format!("wallet_{}", i % 100),
                100.0,
                format!("miner_{}", i),
            );
        }

        let iterations = 10000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = stats.get_miners_for_address("wallet_50");
        }

        let elapsed = start.elapsed();
        let per_lookup_us = elapsed.as_micros() / iterations as u128;

        println!("Miner lookup: {} us per lookup", per_lookup_us);

        // Should be under 100 microseconds
        assert!(per_lookup_us < 100, "Miner lookup too slow: {} us", per_lookup_us);
    }

    #[test]
    fn test_hashrate_aggregation_performance() {
        let mut stats = MiningStatistics::new();
        let wallet = "qnk_perf_test";

        // Add 100 miners to same wallet
        for i in 0..100 {
            stats.update_miner_with_worker(wallet.to_string(), i as f64, format!("miner_{}", i));
        }

        let iterations = 10000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = stats.get_total_hashrate_for_address(wallet);
        }

        let elapsed = start.elapsed();
        let per_agg_us = elapsed.as_micros() / iterations as u128;

        println!("Hashrate aggregation: {} us per call", per_agg_us);

        // Should be under 50 microseconds
        assert!(per_agg_us < 50, "Hashrate aggregation too slow: {} us", per_agg_us);
    }
}
