/// DNA encoding of swap history.
///
/// Each executed swap is encoded as a DNA sequence (ATGC).
/// The DNA chain grows with trade history — older, longer chains have
/// more mass and therefore more vote weight in the swarm consensus.
///
/// Encoding: 2 bits per base — A=00, T=01, G=10, C=11
/// Molecular weights (daltons): A=313, T=304, G=329, C=289

use serde::{Deserialize, Serialize};

/// Molecular weights in daltons (×10 for integer math).
const WEIGHT_A: u64 = 3130;
const WEIGHT_T: u64 = 3040;
const WEIGHT_G: u64 = 3290;
const WEIGHT_C: u64 = 2890;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaChain {
    /// The DNA sequence as a string of A/T/G/C characters.
    pub sequence: String,
    /// Cumulative molecular mass in daltons×10.
    pub mass_deci_daltons: u64,
    /// Number of swaps encoded in this chain.
    pub swap_count: u32,
}

impl DnaChain {
    pub fn new() -> Self {
        DnaChain {
            sequence: String::new(),
            mass_deci_daltons: 0,
            swap_count: 0,
        }
    }

    /// Append a swap to the DNA chain.
    /// Encodes price (f64→u64 at 1e8 precision) and amount (f64→u64 at 1e8 precision).
    pub fn record_swap(&mut self, price_display: f64, amount_display: f64, profit_pct: f64) {
        let price_bits = (price_display * 1e8) as u64;
        let amount_bits = (amount_display * 1e8) as u64;
        let profit_bits = ((profit_pct + 100.0) * 1e6) as u64; // offset to avoid negative

        let encoded = encode_u64(price_bits)
            + &encode_u64(amount_bits)
            + &encode_u64(profit_bits);

        for base in encoded.chars() {
            self.mass_deci_daltons += base_weight(base);
        }
        self.sequence.push_str(&encoded);
        self.swap_count += 1;

        // Keep last 1000 bases to bound memory (Lindy effect: trim oldest history)
        if self.sequence.len() > 1000 {
            let trim = self.sequence.len() - 1000;
            let trimmed = &self.sequence[trim..];
            // Recompute mass for retained portion
            self.mass_deci_daltons = trimmed.chars().map(base_weight).sum();
            self.sequence = trimmed.to_string();
        }
    }

    /// Mass in picograms (for consensus weight display).
    /// 1 pg ≈ 6.022×10^11 Da; we use a linear scale here for relative weighting.
    pub fn mass_picograms(&self) -> f64 {
        self.mass_deci_daltons as f64 / 1e6
    }

    /// Mutation: randomly flip a base (models strategy evolution).
    /// Low mutation rate (0.01%) — profitable chains mutate less.
    pub fn maybe_mutate(&mut self, profit_pct: f64, rng_seed: u64) {
        let base_rate = 0.0001_f64; // 0.01% per record
        // Selective pressure: failed trades mutate 10× faster
        let rate = if profit_pct < 0.0 { base_rate * 10.0 } else { base_rate };
        let roll = lcg_random(rng_seed) as f64 / u64::MAX as f64;
        if roll < rate && !self.sequence.is_empty() {
            let pos = (lcg_random(rng_seed ^ 0xDEAD) as usize) % self.sequence.len();
            let bases = ['A', 'T', 'G', 'C'];
            let new_base = bases[(lcg_random(rng_seed ^ 0xBEEF) as usize) % 4];
            // Rebuild sequence with mutation
            let mut chars: Vec<char> = self.sequence.chars().collect();
            let old = chars[pos];
            chars[pos] = new_base;
            self.mass_deci_daltons =
                self.mass_deci_daltons - base_weight(old) + base_weight(new_base);
            self.sequence = chars.iter().collect();
        }
    }

    /// Encode this chain's last 8 swaps as a display string (for UI).
    pub fn fingerprint(&self) -> String {
        if self.sequence.len() < 8 {
            self.sequence.clone()
        } else {
            self.sequence[self.sequence.len() - 8..].to_string()
        }
    }
}

/// Encode a u64 as a DNA string (32 bases, 2 bits per base).
fn encode_u64(value: u64) -> String {
    let mut result = String::with_capacity(32);
    for i in 0..32 {
        let bits = (value >> (62 - i * 2)) & 0x03;
        result.push(match bits {
            0 => 'A',
            1 => 'T',
            2 => 'G',
            3 => 'C',
            _ => unreachable!(),
        });
    }
    result
}

fn base_weight(base: char) -> u64 {
    match base {
        'A' => WEIGHT_A,
        'T' => WEIGHT_T,
        'G' => WEIGHT_G,
        'C' => WEIGHT_C,
        _ => 0,
    }
}

/// Simple LCG for deterministic "randomness" without a heavy dependency.
fn lcg_random(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_grows_with_swaps() {
        let mut chain = DnaChain::new();
        assert_eq!(chain.swap_count, 0);
        chain.record_swap(50000.0, 0.001, 0.5);
        assert_eq!(chain.swap_count, 1);
        assert!(!chain.sequence.is_empty());
        assert!(chain.mass_deci_daltons > 0);
    }

    #[test]
    fn heavier_chain_has_more_mass() {
        let mut a = DnaChain::new();
        let mut b = DnaChain::new();
        for _ in 0..10 { a.record_swap(50000.0, 0.001, 1.0); }
        for _ in 0..5  { b.record_swap(50000.0, 0.001, 1.0); }
        assert!(a.mass_deci_daltons > b.mass_deci_daltons);
    }
}
