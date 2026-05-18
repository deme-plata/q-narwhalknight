//! Item 5 — Information-theoretic chunk-size floor.
//!
//! `min_chunk_size_bits = log₂(peers_in_set) * bits_per_block`.
//!
//! Intuition: with N peers in the candidate set, each chunk-dispatch decision
//! is `log₂(N)` bits of routing information. To amortize the routing overhead
//! we must send at least `log₂(N)` bits-worth of payload per decision.
//!
//! With 8 peers × ~8000 bits per block, floor = 24 K bits ≈ 3 KB of blocks.
//!
//! ## Output
//! Returned as KiB (rounded up) so it can directly raise the floor in
//! [`crate::kalman_bdp::KalmanBdpEstimator`].
//!
//! ## Prometheus
//! Expose `qnk_chunk_size_floor` gauge updated on every recompute.

use serde::{Deserialize, Serialize};

/// Default per-block size estimate (bits).
pub const DEFAULT_BITS_PER_BLOCK: f64 = 8_000.0;
/// Absolute KiB floor — never return less.
pub const ABS_MIN_FLOOR_KB: u32 = 1;

/// Computes information-theoretic chunk floors.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChunkFloorEstimator {
    pub bits_per_block: f64,
}

impl Default for ChunkFloorEstimator {
    fn default() -> Self {
        Self {
            bits_per_block: DEFAULT_BITS_PER_BLOCK,
        }
    }
}

impl ChunkFloorEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_bits_per_block(bits_per_block: f64) -> Self {
        Self {
            bits_per_block: bits_per_block.max(1.0),
        }
    }

    /// Compute the floor in **bits**.
    ///
    /// Special cases:
    /// - `peers_in_set == 0` → 0 bits (no decisions to amortize).
    /// - `peers_in_set == 1` → `bits_per_block` (log₂(1)=0 but we still need ≥1 block).
    pub fn floor_bits(&self, peers_in_set: u32) -> f64 {
        if peers_in_set == 0 {
            return 0.0;
        }
        if peers_in_set == 1 {
            // log₂(1) = 0; promote to "at least one block".
            return self.bits_per_block;
        }
        (peers_in_set as f64).log2() * self.bits_per_block
    }

    /// Compute the floor in **KiB** (round up so we don't undershoot).
    pub fn floor_kib(&self, peers_in_set: u32) -> u32 {
        let bits = self.floor_bits(peers_in_set);
        let kib = bits / 8.0 / 1024.0;
        let ceil = kib.ceil();
        let raw = if ceil < ABS_MIN_FLOOR_KB as f64 {
            ABS_MIN_FLOOR_KB
        } else if ceil > u32::MAX as f64 {
            u32::MAX
        } else {
            ceil as u32
        };
        raw.max(ABS_MIN_FLOOR_KB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_peers_with_default_bits() {
        // log2(8)*8000 = 3*8000 = 24000 bits = 3000 B = 2.93 KiB → 3 KiB.
        let est = ChunkFloorEstimator::new();
        assert_eq!(est.floor_kib(8), 3);
    }

    #[test]
    fn zero_peers_is_min_one_kib() {
        let est = ChunkFloorEstimator::new();
        assert_eq!(est.floor_kib(0), ABS_MIN_FLOOR_KB);
    }

    #[test]
    fn one_peer_still_at_least_one_block() {
        let est = ChunkFloorEstimator::new();
        // 1 block ≈ 8000 bits ≈ 1 KiB
        assert!(est.floor_kib(1) >= 1);
    }

    #[test]
    fn larger_peer_set_raises_floor() {
        let est = ChunkFloorEstimator::new();
        let f4 = est.floor_kib(4);
        let f16 = est.floor_kib(16);
        let f256 = est.floor_kib(256);
        assert!(f4 < f16);
        assert!(f16 < f256);
    }

    #[test]
    fn known_value_64_peers() {
        // log2(64) * 8000 = 6 * 8000 = 48000 bits = 6000 B = 5.86 KiB → 6
        let est = ChunkFloorEstimator::new();
        assert_eq!(est.floor_kib(64), 6);
    }

    #[test]
    fn custom_bits_per_block() {
        // log2(8) * 16000 = 48000 bits = 6 KiB.
        let est = ChunkFloorEstimator::with_bits_per_block(16_000.0);
        assert_eq!(est.floor_kib(8), 6);
    }
}
