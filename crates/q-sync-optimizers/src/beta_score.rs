//! Item 2 — Bayesian peer reliability via Beta distribution + Thompson sampling.
//!
//! Each peer holds counters `α = successes + 1`, `β = failures + 1`
//! (Laplace smoothing). The posterior is `Beta(α, β)` with mean `α / (α + β)`.
//!
//! For chunk dispatch we pick the peer with the highest Thompson sample —
//! a sample drawn from each peer's posterior. This is provably optimal
//! exploration/exploitation balance for Bernoulli arms.
//!
//! Since `rand`'s default features don't include the Beta distribution, we
//! sample Beta(α,β) via the ratio of two Gammas:
//!   X ~ Gamma(α, 1), Y ~ Gamma(β, 1) ⇒ X/(X+Y) ~ Beta(α, β).
//!
//! For small integer α, β we just sum α Exponential(1) draws (Gamma(k,1) =
//! sum of k Exp(1)). For non-integer α (after many updates the counts are
//! integers, so this path is only for completeness) we fall back to mean.
//!
//! All math is O(α + β) per sample; α and β grow with traffic, so once a peer
//! has 1000s of successes we use the **mean** as the Thompson sample (the
//! posterior is so peaked sampling adds negligible noise). The cutover is at
//! `MAX_EXACT_SAMPLE` total updates.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;

/// Maximum (α + β) for which we draw an exact Gamma-based sample. Above this
/// we just use the posterior mean (variance ≈ p(1-p)/(α+β) is negligible).
pub const MAX_EXACT_SAMPLE: u32 = 200;

/// Per-peer Beta counters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BetaCounter {
    /// α = successes + 1 (Laplace).
    pub alpha: u32,
    /// β = failures + 1 (Laplace).
    pub beta: u32,
}

impl Default for BetaCounter {
    fn default() -> Self {
        Self { alpha: 1, beta: 1 }
    }
}

impl BetaCounter {
    /// Fresh counter (Beta(1,1) = uniform prior).
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one successful chunk.
    pub fn record_success(&mut self) {
        self.alpha = self.alpha.saturating_add(1);
    }

    /// Record one failed chunk (timeout, validation error, etc.).
    pub fn record_failure(&mut self) {
        self.beta = self.beta.saturating_add(1);
    }

    /// Posterior mean: `α / (α + β)`.
    pub fn mean(&self) -> f64 {
        let a = self.alpha as f64;
        let b = self.beta as f64;
        a / (a + b)
    }

    /// Draw a Thompson sample from `Beta(α, β)`.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let total = self.alpha.saturating_add(self.beta);
        if total >= MAX_EXACT_SAMPLE {
            return self.mean();
        }
        // Gamma(k,1) = sum of k independent Exp(1) draws.
        // Exp(1) = -ln(U) for U ~ Uniform(0,1).
        let x = gamma_integer_shape(self.alpha, rng);
        let y = gamma_integer_shape(self.beta, rng);
        let denom = x + y;
        if denom <= 0.0 {
            self.mean()
        } else {
            x / denom
        }
    }

    /// Reset to uniform prior — call on peer reconnect.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

fn gamma_integer_shape<R: Rng>(k: u32, rng: &mut R) -> f64 {
    let mut sum = 0.0;
    for _ in 0..k {
        // Avoid log(0).
        let u: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        sum += -u.ln();
    }
    sum
}

/// Registry of Beta scores keyed by peer ID.
#[derive(Debug, Clone, Default)]
pub struct BetaScoreRegistry<K: Eq + Hash + Clone> {
    peers: HashMap<K, BetaCounter>,
}

impl<K: Eq + Hash + Clone> BetaScoreRegistry<K> {
    pub fn new() -> Self {
        Self { peers: HashMap::new() }
    }

    pub fn record_success(&mut self, peer: &K) {
        self.peers.entry(peer.clone()).or_default().record_success();
    }

    pub fn record_failure(&mut self, peer: &K) {
        self.peers.entry(peer.clone()).or_default().record_failure();
    }

    /// Reset a peer's prior — call on reconnect.
    pub fn reset(&mut self, peer: &K) {
        if let Some(c) = self.peers.get_mut(peer) {
            c.reset();
        }
    }

    pub fn mean(&self, peer: &K) -> f64 {
        self.peers.get(peer).map(|c| c.mean()).unwrap_or(0.5)
    }

    pub fn counter(&self, peer: &K) -> BetaCounter {
        self.peers.get(peer).copied().unwrap_or_default()
    }

    /// Thompson-sample one of the candidates. Picks the peer whose drawn
    /// score is highest. Returns None if `candidates` is empty.
    pub fn thompson_pick<'a, R: Rng>(
        &mut self,
        candidates: &'a [K],
        rng: &mut R,
    ) -> Option<&'a K> {
        if candidates.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, peer) in candidates.iter().enumerate() {
            let counter = self.peers.entry(peer.clone()).or_default();
            let s = counter.sample(rng);
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }
        Some(&candidates[best_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn beta_2_1_mean_is_two_thirds() {
        // After 1 success, α=2 β=1, mean = 2/3.
        let mut c = BetaCounter::new();
        c.record_success();
        let diff = (c.mean() - 2.0 / 3.0).abs();
        assert!(diff < 1e-12, "mean was {}", c.mean());
    }

    #[test]
    fn default_is_uniform_prior() {
        let c = BetaCounter::new();
        assert_eq!(c.alpha, 1);
        assert_eq!(c.beta, 1);
        assert!((c.mean() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reset_returns_to_prior() {
        let mut c = BetaCounter::new();
        for _ in 0..10 {
            c.record_success();
        }
        assert!(c.mean() > 0.9);
        c.reset();
        assert_eq!(c.alpha, 1);
        assert_eq!(c.beta, 1);
    }

    #[test]
    fn sample_in_unit_interval() {
        let mut rng = StdRng::seed_from_u64(42);
        let c = BetaCounter::new();
        for _ in 0..1000 {
            let s = c.sample(&mut rng);
            assert!((0.0..=1.0).contains(&s), "sample {s} out of range");
        }
    }

    #[test]
    fn empirical_mean_matches_posterior_mean() {
        // Beta(5, 3) mean = 5/8 = 0.625. Empirically test.
        let mut rng = StdRng::seed_from_u64(7);
        let c = BetaCounter { alpha: 5, beta: 3 };
        let n = 20_000;
        let avg: f64 = (0..n).map(|_| c.sample(&mut rng)).sum::<f64>() / n as f64;
        let diff = (avg - 0.625).abs();
        assert!(diff < 0.01, "expected ≈0.625, got {avg}");
    }

    #[test]
    fn thompson_prefers_better_peer_on_average() {
        let mut reg = BetaScoreRegistry::<u32>::new();
        // Peer 0: good (90 succ, 10 fail). Peer 1: bad (10 succ, 90 fail).
        for _ in 0..90 {
            reg.record_success(&0);
        }
        for _ in 0..10 {
            reg.record_failure(&0);
        }
        for _ in 0..10 {
            reg.record_success(&1);
        }
        for _ in 0..90 {
            reg.record_failure(&1);
        }
        let candidates = vec![0u32, 1];
        let mut rng = StdRng::seed_from_u64(13);
        let n = 500;
        let mut picked_good = 0;
        for _ in 0..n {
            if reg.thompson_pick(&candidates, &mut rng) == Some(&0) {
                picked_good += 1;
            }
        }
        // Posterior of 0 is sharply > 1; with α+β=200 we use mean,
        // so we pick the good peer essentially every time.
        assert!(picked_good > n * 9 / 10, "good peer picked {picked_good}/{n}");
    }

    #[test]
    fn thompson_pick_none_for_empty_candidates() {
        let mut reg = BetaScoreRegistry::<u32>::new();
        let mut rng = StdRng::seed_from_u64(1);
        assert!(reg.thompson_pick(&[], &mut rng).is_none());
    }

    #[test]
    fn registry_reset_clears_peer() {
        let mut reg = BetaScoreRegistry::<u32>::new();
        for _ in 0..5 {
            reg.record_success(&7);
        }
        assert!(reg.mean(&7) > 0.5);
        reg.reset(&7);
        assert!((reg.mean(&7) - 0.5).abs() < 1e-12);
    }
}
