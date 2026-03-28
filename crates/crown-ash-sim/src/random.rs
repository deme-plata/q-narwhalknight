//! Deterministic RNG derived from block hashes.
//!
//! Uses a custom hash-mixing construction: `state = mix(state || domain || "CROWN_ASH")`.
//! The WASM plugin uses the host's `plugin_sha3_256`, but this standalone implementation
//! guarantees identical sequences on all nodes for the native simulation.
//!
//! **No floating point** — all randomness produces integer/FixedPoint values.

use serde::{Deserialize, Serialize};

/// Number of mixing rounds per hash step.
const MIX_ROUNDS: usize = 16;

/// Domain separation suffix.
const SUFFIX: &[u8] = b"CROWN_ASH";

/// Deterministic pseudo-random number generator seeded from block hashes.
///
/// Each call to `next_u32` advances the internal state deterministically.
/// Two `DeterministicRng` instances constructed with the same `block_hash` and `domain`
/// will always produce the same sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicRng {
    state: [u8; 32],
}

impl DeterministicRng {
    /// Create a new RNG from a block hash and a domain-separation string.
    ///
    /// The domain string ensures that different subsystems (combat, events, AI)
    /// drawing from the same block hash produce independent sequences.
    pub fn new(block_hash: [u8; 32], domain: &str) -> Self {
        let mut buf = Vec::with_capacity(32 + domain.len() + SUFFIX.len());
        buf.extend_from_slice(&block_hash);
        buf.extend_from_slice(domain.as_bytes());
        buf.extend_from_slice(SUFFIX);

        let state = hash_bytes(&buf);
        Self { state }
    }

    /// Produce the next pseudo-random `u32` and advance the state.
    pub fn next_u32(&mut self) -> u32 {
        // Hash current state with counter suffix to produce output.
        let mut buf = [0u8; 32 + SUFFIX.len()];
        buf[..32].copy_from_slice(&self.state);
        buf[32..].copy_from_slice(SUFFIX);
        self.state = hash_bytes(&buf[..32 + SUFFIX.len()]);

        // Take the first 4 bytes as a little-endian u32.
        u32::from_le_bytes([self.state[0], self.state[1], self.state[2], self.state[3]])
    }

    /// Return a value in `[min, max]` (inclusive on both ends).
    ///
    /// Panics if `min > max`.
    pub fn range(&mut self, min: i64, max: i64) -> i64 {
        assert!(min <= max, "range: min ({}) must be <= max ({})", min, max);
        if min == max {
            return min;
        }
        let span = (max - min + 1) as u64;
        let r = self.next_u32() as u64;
        min + (r % span) as i64
    }

    /// Return `true` with probability `numerator / denominator`.
    ///
    /// `chance(1, 100)` is a 1% chance.  Panics if `denominator == 0`.
    pub fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        assert!(denominator > 0, "chance: denominator must be > 0");
        if numerator >= denominator {
            return true;
        }
        if numerator == 0 {
            return false;
        }
        let r = self.next_u32() % denominator;
        r < numerator
    }
}

// ---------------------------------------------------------------------------
// Custom hash-mixing function
// ---------------------------------------------------------------------------
// This is a deterministic, non-cryptographic mixing function that
// XORs, rotates, and diffuses bytes.  It produces 32 bytes of output
// from arbitrary-length input.  It does NOT need to be collision-resistant;
// it only needs to be deterministic and well-distributed.

fn hash_bytes(input: &[u8]) -> [u8; 32] {
    // Initialise state from a nothing-up-my-sleeve constant (first 32 bytes of pi).
    let mut h: [u8; 32] = [
        0x24, 0x3F, 0x6A, 0x88, 0x85, 0xA3, 0x08, 0xD3,
        0x13, 0x19, 0x8A, 0x2E, 0x03, 0x70, 0x73, 0x44,
        0xA4, 0x09, 0x38, 0x22, 0x29, 0x9F, 0x31, 0xD0,
        0x08, 0x2E, 0xFA, 0x98, 0xEC, 0x4E, 0x6C, 0x89,
    ];

    // Absorb input bytes.
    for (i, &byte) in input.iter().enumerate() {
        h[i % 32] ^= byte;
        // Diffuse after every full block.
        if i % 32 == 31 {
            mix_state(&mut h);
        }
    }

    // Final mixing rounds.
    for _ in 0..MIX_ROUNDS {
        mix_state(&mut h);
    }

    h
}

/// Single round of state mixing: ARX (add-rotate-xor) across 32 bytes.
fn mix_state(h: &mut [u8; 32]) {
    for i in 0..32 {
        let prev = h[(i + 31) % 32];
        let next = h[(i + 1) % 32];
        h[i] = h[i]
            .wrapping_add(prev)
            .rotate_left(3)
            ^ next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed() {
        let hash = [0xABu8; 32];
        let mut a = DeterministicRng::new(hash, "combat");
        let mut b = DeterministicRng::new(hash, "combat");
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_domains_differ() {
        let hash = [0x42u8; 32];
        let mut a = DeterministicRng::new(hash, "combat");
        let mut b = DeterministicRng::new(hash, "events");
        // Very unlikely to be equal across 10 draws.
        let same = (0..10).filter(|_| a.next_u32() == b.next_u32()).count();
        assert!(same < 5);
    }

    #[test]
    fn range_bounds() {
        let mut rng = DeterministicRng::new([0x01; 32], "test");
        for _ in 0..200 {
            let v = rng.range(5, 10);
            assert!(v >= 5 && v <= 10);
        }
    }

    #[test]
    fn chance_always_never() {
        let mut rng = DeterministicRng::new([0x02; 32], "test");
        assert!(rng.chance(100, 100)); // always
        assert!(!rng.chance(0, 100));  // never
    }
}
