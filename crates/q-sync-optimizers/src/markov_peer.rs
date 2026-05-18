//! Item 6 — Markov chain peer-state model.
//!
//! Three states: {Fast, Slow, Stalled}.
//!
//! ## Transitions
//! - **Fast → Slow**: 2 consecutive RTTs > 1.5 × EWMA.
//! - **Slow → Stalled**: 1 timeout OR 3 consecutive Slow ticks.
//! - **Slow → Fast**: 2 consecutive RTTs < EWMA.
//! - **Stalled → Fast**: only after a successful chunk.
//!
//! ## EWMA
//! `mean_rtt = α × sample + (1 - α) × mean_rtt`, α = 0.2.
//!
//! ## Prometheus
//! Expose per-peer state as `qnk_peer_state{peer=...}` enum (0=Fast 1=Slow 2=Stalled).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;

/// EWMA smoothing factor for RTT (same as Little's law module by convention).
pub const RTT_ALPHA: f64 = 0.2;
/// "Slow" threshold multiplier on the EWMA.
pub const SLOW_THRESHOLD: f64 = 1.5;
/// Consecutive slow samples that trigger Stalled.
pub const SLOW_TICKS_TO_STALL: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerState {
    Fast = 0,
    Slow = 1,
    Stalled = 2,
}

impl Default for PeerState {
    fn default() -> Self {
        Self::Fast
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MarkovPeer {
    pub state: PeerState,
    pub mean_rtt_ms: Option<f64>,
    /// Consecutive samples > 1.5 × EWMA (used in Fast→Slow check).
    consecutive_above: u32,
    /// Consecutive samples < EWMA (used in Slow→Fast check).
    consecutive_below: u32,
    /// Consecutive ticks spent in Slow (used in Slow→Stalled check).
    slow_ticks: u32,
}

impl Default for MarkovPeer {
    fn default() -> Self {
        Self {
            state: PeerState::Fast,
            mean_rtt_ms: None,
            consecutive_above: 0,
            consecutive_below: 0,
            slow_ticks: 0,
        }
    }
}

impl MarkovPeer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn state(&self) -> PeerState {
        self.state
    }

    /// Record a normal (non-timeout) RTT sample. Returns the new state.
    pub fn record_rtt(&mut self, rtt_ms: f64) -> PeerState {
        if !rtt_ms.is_finite() || rtt_ms < 0.0 {
            return self.state;
        }

        // Compare against PRIOR mean before updating it.
        let prior = self.mean_rtt_ms;

        // Update EWMA with this sample.
        self.mean_rtt_ms = Some(match self.mean_rtt_ms {
            None => rtt_ms,
            Some(prev) => RTT_ALPHA * rtt_ms + (1.0 - RTT_ALPHA) * prev,
        });

        // Need a prior mean to make any classification.
        let mean = match prior {
            Some(m) if m > 0.0 => m,
            _ => return self.state,
        };

        let is_above = rtt_ms > SLOW_THRESHOLD * mean;
        let is_below = rtt_ms < mean;

        if is_above {
            self.consecutive_above = self.consecutive_above.saturating_add(1);
        } else {
            self.consecutive_above = 0;
        }
        if is_below {
            self.consecutive_below = self.consecutive_below.saturating_add(1);
        } else {
            self.consecutive_below = 0;
        }

        match self.state {
            PeerState::Fast => {
                if self.consecutive_above >= 2 {
                    self.state = PeerState::Slow;
                    self.slow_ticks = 1;
                }
            }
            PeerState::Slow => {
                // Tick counter: every record_rtt while still in Slow advances it.
                self.slow_ticks = self.slow_ticks.saturating_add(1);
                if self.consecutive_below >= 2 {
                    self.state = PeerState::Fast;
                    self.slow_ticks = 0;
                } else if self.slow_ticks >= SLOW_TICKS_TO_STALL {
                    self.state = PeerState::Stalled;
                }
            }
            PeerState::Stalled => {
                // Only `record_success` can leave Stalled.
            }
        }

        self.state
    }

    /// Record a timeout. From Slow we jump to Stalled; from Fast we go to Slow.
    pub fn record_timeout(&mut self) -> PeerState {
        match self.state {
            PeerState::Fast => {
                self.state = PeerState::Slow;
                self.slow_ticks = 1;
                self.consecutive_above = 0;
                self.consecutive_below = 0;
            }
            PeerState::Slow => {
                self.state = PeerState::Stalled;
            }
            PeerState::Stalled => {}
        }
        self.state
    }

    /// Record a successful chunk. Stalled → Fast; otherwise no state change
    /// (the [`record_rtt`] call that comes with the success drives Fast↔Slow).
    pub fn record_success(&mut self) -> PeerState {
        if self.state == PeerState::Stalled {
            self.state = PeerState::Fast;
            self.slow_ticks = 0;
            self.consecutive_above = 0;
            self.consecutive_below = 0;
        }
        self.state
    }

    /// Reset to defaults — call on peer reconnect.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Registry of per-peer Markov states.
#[derive(Debug, Clone, Default)]
pub struct MarkovRegistry<K: Eq + Hash + Clone> {
    peers: HashMap<K, MarkovPeer>,
}

impl<K: Eq + Hash + Clone> MarkovRegistry<K> {
    pub fn new() -> Self {
        Self { peers: HashMap::new() }
    }

    pub fn record_rtt(&mut self, peer: &K, rtt_ms: f64) -> PeerState {
        self.peers.entry(peer.clone()).or_default().record_rtt(rtt_ms)
    }

    pub fn record_timeout(&mut self, peer: &K) -> PeerState {
        self.peers.entry(peer.clone()).or_default().record_timeout()
    }

    pub fn record_success(&mut self, peer: &K) -> PeerState {
        self.peers.entry(peer.clone()).or_default().record_success()
    }

    pub fn state(&self, peer: &K) -> PeerState {
        self.peers.get(peer).map(|p| p.state).unwrap_or(PeerState::Fast)
    }

    pub fn reset(&mut self, peer: &K) {
        if let Some(p) = self.peers.get_mut(peer) {
            p.reset();
        }
    }

    /// Rank candidates by state (Fast first, then Slow, then Stalled), preserving
    /// input order within each bucket.
    pub fn rank_by_state<'a>(&self, candidates: &'a [K]) -> Vec<&'a K> {
        let mut fast = Vec::new();
        let mut slow = Vec::new();
        let mut stalled = Vec::new();
        for k in candidates {
            match self.state(k) {
                PeerState::Fast => fast.push(k),
                PeerState::Slow => slow.push(k),
                PeerState::Stalled => stalled.push(k),
            }
        }
        fast.extend(slow);
        fast.extend(stalled);
        fast
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_in_fast() {
        let p = MarkovPeer::new();
        assert_eq!(p.state(), PeerState::Fast);
    }

    #[test]
    fn fast_to_slow_after_two_high_rtts() {
        let mut p = MarkovPeer::new();
        // Establish baseline.
        p.record_rtt(50.0);
        // 50 * 1.5 = 75; 100 > 75 → above.
        let s1 = p.record_rtt(100.0);
        assert_eq!(s1, PeerState::Fast); // need TWO consecutive.
        // EWMA after 100 = 0.2*100 + 0.8*50 = 60; 1.5*60 = 90; 200 > 90 → above.
        let s2 = p.record_rtt(200.0);
        assert_eq!(s2, PeerState::Slow);
    }

    #[test]
    fn slow_to_stalled_on_timeout() {
        let mut p = MarkovPeer::new();
        p.record_rtt(50.0);
        p.record_rtt(100.0);
        p.record_rtt(200.0); // now Slow
        assert_eq!(p.state(), PeerState::Slow);
        let s = p.record_timeout();
        assert_eq!(s, PeerState::Stalled);
    }

    #[test]
    fn slow_to_stalled_after_three_slow_ticks() {
        let mut p = MarkovPeer::new();
        p.record_rtt(50.0);
        p.record_rtt(100.0);
        p.record_rtt(200.0); // tick 1 in Slow
        assert_eq!(p.state(), PeerState::Slow);
        // Two more ticks while still in Slow (not "below mean") triggers Stalled.
        p.record_rtt(200.0); // tick 2
        let s = p.record_rtt(200.0); // tick 3 → Stalled
        assert_eq!(s, PeerState::Stalled);
    }

    #[test]
    fn slow_to_fast_after_two_below_mean() {
        let mut p = MarkovPeer::new();
        p.record_rtt(50.0);
        p.record_rtt(100.0);
        p.record_rtt(200.0); // Slow
        assert_eq!(p.state(), PeerState::Slow);
        // Two consecutive below-mean RTTs.
        p.record_rtt(10.0);
        let s = p.record_rtt(10.0);
        assert_eq!(s, PeerState::Fast);
    }

    #[test]
    fn stalled_only_clears_on_success() {
        let mut p = MarkovPeer::new();
        p.record_rtt(50.0);
        p.record_timeout(); // Fast→Slow
        p.record_timeout(); // Slow→Stalled
        assert_eq!(p.state(), PeerState::Stalled);
        // Recording RTTs (even below mean) doesn't leave Stalled.
        p.record_rtt(1.0);
        p.record_rtt(1.0);
        assert_eq!(p.state(), PeerState::Stalled);
        // Success does.
        assert_eq!(p.record_success(), PeerState::Fast);
    }

    #[test]
    fn timeout_from_fast_moves_to_slow() {
        let mut p = MarkovPeer::new();
        assert_eq!(p.record_timeout(), PeerState::Slow);
    }

    #[test]
    fn reset_restores_fast() {
        let mut p = MarkovPeer::new();
        p.record_timeout();
        p.record_timeout();
        assert_eq!(p.state(), PeerState::Stalled);
        p.reset();
        assert_eq!(p.state(), PeerState::Fast);
    }

    #[test]
    fn rank_by_state_orders_fast_first() {
        let mut r = MarkovRegistry::<u32>::new();
        r.record_timeout(&1); // Slow
        r.record_timeout(&2);
        r.record_timeout(&2); // Stalled
        // 3 stays Fast.
        let ranked = r.rank_by_state(&[1u32, 2, 3]);
        assert_eq!(ranked, vec![&3, &1, &2]);
    }

    #[test]
    fn nan_rtt_ignored() {
        let mut p = MarkovPeer::new();
        p.record_rtt(50.0);
        let s = p.record_rtt(f64::NAN);
        assert_eq!(s, PeerState::Fast);
    }
}
