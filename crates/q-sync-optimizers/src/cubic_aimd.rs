//! Item 3 — TCP-CUBIC-style AIMD (Additive Increase / Multiplicative Decrease)
//! for per-peer block-pack parallelism (`cwnd`).
//!
//! Replaces the fixed `CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER = 16` with a
//! per-peer dynamic window that grows on success and shrinks on failure.
//!
//! ## Update rule
//! On success: `cwnd += 1`   (additive increase)
//! On loss:    `cwnd *= 0.7` (multiplicative decrease, rounded down, floor 1)
//!
//! Floor: 1 (must always allow at least one in-flight request).
//! Ceiling: 64 (libp2p connection + server semaphore practical limit).
//!
//! This is *not* the full CUBIC root-function trajectory — that's overkill
//! for blockchain block requests. We implement the simpler AIMD that CUBIC
//! reduces to in steady state. The spec calls this "CUBIC AIMD".

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;

pub const CWND_FLOOR: u32 = 1;
pub const CWND_CEILING: u32 = 64;
pub const CWND_INITIAL: u32 = 16; // Match legacy CLIENT_INFLIGHT_BLOCK_PACK_PER_PEER.
pub const BETA: f64 = 0.7;        // Multiplicative-decrease factor.

/// Per-peer congestion window.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CubicWindow {
    pub cwnd: u32,
}

impl Default for CubicWindow {
    fn default() -> Self {
        Self { cwnd: CWND_INITIAL }
    }
}

impl CubicWindow {
    /// Start at [`CWND_INITIAL`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Start at a specific value (clamped to floor/ceiling).
    pub fn with_initial(cwnd: u32) -> Self {
        Self {
            cwnd: cwnd.clamp(CWND_FLOOR, CWND_CEILING),
        }
    }

    /// Current window size (number of in-flight requests allowed).
    pub fn cwnd(&self) -> u32 {
        self.cwnd
    }

    /// Additive increase: `cwnd += 1` (capped at ceiling).
    pub fn on_success(&mut self) {
        if self.cwnd < CWND_CEILING {
            self.cwnd += 1;
        }
    }

    /// Multiplicative decrease: `cwnd = max(floor, floor(cwnd * 0.7))`.
    pub fn on_loss(&mut self) {
        let new = (self.cwnd as f64 * BETA).floor() as u32;
        self.cwnd = new.max(CWND_FLOOR);
    }

    /// Reset to initial — e.g. on reconnect.
    pub fn reset(&mut self) {
        self.cwnd = CWND_INITIAL;
    }
}

/// Per-peer CUBIC registry.
#[derive(Debug, Clone, Default)]
pub struct CubicRegistry<K: Eq + Hash + Clone> {
    peers: HashMap<K, CubicWindow>,
}

impl<K: Eq + Hash + Clone> CubicRegistry<K> {
    pub fn new() -> Self {
        Self { peers: HashMap::new() }
    }

    pub fn cwnd(&self, peer: &K) -> u32 {
        self.peers.get(peer).map(|w| w.cwnd()).unwrap_or(CWND_INITIAL)
    }

    pub fn on_success(&mut self, peer: &K) {
        self.peers.entry(peer.clone()).or_default().on_success();
    }

    pub fn on_loss(&mut self, peer: &K) {
        self.peers.entry(peer.clone()).or_default().on_loss();
    }

    pub fn reset(&mut self, peer: &K) {
        if let Some(w) = self.peers.get_mut(peer) {
            w.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_window_is_16() {
        let w = CubicWindow::new();
        assert_eq!(w.cwnd(), 16);
    }

    #[test]
    fn additive_increase() {
        let mut w = CubicWindow::new();
        for _ in 0..5 {
            w.on_success();
        }
        assert_eq!(w.cwnd(), 21);
    }

    #[test]
    fn multiplicative_decrease() {
        let mut w = CubicWindow::with_initial(20);
        w.on_loss();
        // 20 * 0.7 = 14 (floor).
        assert_eq!(w.cwnd(), 14);
    }

    #[test]
    fn floor_is_one() {
        let mut w = CubicWindow::with_initial(2);
        w.on_loss(); // 1.4 → 1
        assert_eq!(w.cwnd(), 1);
        w.on_loss(); // 0.7 → 0, then clamped to 1
        assert_eq!(w.cwnd(), 1);
    }

    #[test]
    fn ceiling_is_64() {
        let mut w = CubicWindow::with_initial(63);
        w.on_success();
        assert_eq!(w.cwnd(), 64);
        w.on_success(); // should NOT exceed.
        assert_eq!(w.cwnd(), 64);
    }

    #[test]
    fn growth_then_loss_cycle() {
        let mut w = CubicWindow::with_initial(10);
        for _ in 0..70 {
            w.on_success();
        }
        // Capped at 64.
        assert_eq!(w.cwnd(), CWND_CEILING);
        w.on_loss();
        // 64 * 0.7 = 44.8 → 44
        assert_eq!(w.cwnd(), 44);
    }

    #[test]
    fn registry_independent_per_peer() {
        let mut reg = CubicRegistry::<u32>::new();
        for _ in 0..10 {
            reg.on_success(&1);
        }
        reg.on_loss(&2);
        assert_eq!(reg.cwnd(&1), 26);
        assert_eq!(reg.cwnd(&2), 11); // 16 → 11.2 → 11
    }

    #[test]
    fn unknown_peer_returns_initial() {
        let reg = CubicRegistry::<u32>::new();
        assert_eq!(reg.cwnd(&99), CWND_INITIAL);
    }

    #[test]
    fn reset_returns_to_initial() {
        let mut w = CubicWindow::with_initial(50);
        w.reset();
        assert_eq!(w.cwnd(), CWND_INITIAL);
    }

    #[test]
    fn with_initial_clamps() {
        assert_eq!(CubicWindow::with_initial(0).cwnd(), CWND_FLOOR);
        assert_eq!(CubicWindow::with_initial(1000).cwnd(), CWND_CEILING);
    }
}
