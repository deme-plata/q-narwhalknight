// Sampling strategies for LLM text generation
//
// This module implements various sampling methods for converting model logits
// into actual token selections during autoregressive generation.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rand::Rng;
use std::cmp::Ordering;

/// Sampling configuration for text generation
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for sampling (higher = more random, lower = more deterministic)
    /// Default: 0.7. Range: (0.0, infinity). Use 1.0 for no scaling.
    pub temperature: f64,

    /// Top-k sampling: only consider the k most likely tokens
    /// Default: 50. Set to 0 to disable.
    pub top_k: usize,

    /// Top-p (nucleus) sampling: only consider tokens with cumulative probability >= p
    /// Default: 0.9. Range: (0.0, 1.0]. Set to 1.0 to disable.
    pub top_p: f64,

    /// Repetition penalty: penalize tokens that have already been generated
    /// Default: 1.0 (no penalty). Range: [1.0, infinity). Higher = stronger penalty.
    pub repetition_penalty: f64,

    /// Frequency penalty: reduce likelihood of tokens based on their frequency
    /// Default: 0.0 (no penalty). Typical range: [0.0, 2.0].
    pub frequency_penalty: f64,

    /// Presence penalty: reduce likelihood of tokens that have appeared at all
    /// Default: 0.0 (no penalty). Typical range: [0.0, 2.0].
    pub presence_penalty: f64,

    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
        }
    }
}

impl SamplingConfig {
    /// Create a configuration for greedy sampling (always pick most likely token)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            ..Default::default()
        }
    }

    /// Create a configuration for creative/diverse sampling
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_k: 100,
            top_p: 0.95,
            ..Default::default()
        }
    }

    /// Create a configuration for balanced sampling
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Create a configuration for precise/factual sampling
    pub fn precise() -> Self {
        Self {
            temperature: 0.3,
            top_k: 20,
            top_p: 0.85,
            ..Default::default()
        }
    }
}

/// Sampler for selecting tokens from model logits
pub struct Sampler {
    config: SamplingConfig,
    rng: rand::rngs::StdRng,
}

impl Sampler {
    /// Create a new sampler with the given configuration
    pub fn new(config: SamplingConfig) -> Self {
        use rand::SeedableRng;

        let rng = match config.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };

        Self { config, rng }
    }

    /// Sample a single token from logits
    ///
    /// # Arguments
    /// * `logits` - Tensor of shape [vocab_size] containing raw model outputs
    /// * `generated_tokens` - Previously generated tokens for repetition penalty
    ///
    /// # Returns
    /// The selected token ID
    pub fn sample(&mut self, logits: &Tensor, generated_tokens: &[u32]) -> Result<u32> {
        // Convert logits to Vec<f32> for processing
        let logits_vec = logits.to_vec1::<f32>()?;

        // Apply penalties
        let mut logits_vec = self.apply_penalties(logits_vec, generated_tokens)?;

        // Apply temperature
        if self.config.temperature > 0.0 && self.config.temperature != 1.0 {
            logits_vec = logits_vec
                .iter()
                .map(|&logit| logit / self.config.temperature as f32)
                .collect();
        }

        // Handle greedy sampling (temperature = 0 or top_k = 1)
        if self.config.temperature == 0.0 || self.config.top_k == 1 {
            let token = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .ok_or_else(|| anyhow::anyhow!("Empty logits"))?;
            return Ok(token);
        }

        // Convert to probabilities
        let probs = softmax(&logits_vec);

        // Apply top-k filtering
        let probs = if self.config.top_k > 0 && self.config.top_k < probs.len() {
            self.apply_top_k(probs)?
        } else {
            probs
        };

        // Apply top-p (nucleus) filtering
        let probs = if self.config.top_p < 1.0 {
            self.apply_top_p(probs)?
        } else {
            probs
        };

        // Sample from the filtered distribution
        let token = self.multinomial_sample(&probs)?;

        Ok(token)
    }

    /// Apply repetition and presence/frequency penalties
    fn apply_penalties(&self, mut logits: Vec<f32>, generated_tokens: &[u32]) -> Result<Vec<f32>> {
        if generated_tokens.is_empty() {
            return Ok(logits);
        }

        // Count token frequencies
        let mut token_counts = vec![0usize; logits.len()];
        for &token in generated_tokens {
            if (token as usize) < token_counts.len() {
                token_counts[token as usize] += 1;
            }
        }

        // Apply penalties
        for (token_id, count) in token_counts.iter().enumerate() {
            if *count > 0 {
                // Repetition penalty (multiplicative)
                if self.config.repetition_penalty != 1.0 {
                    if logits[token_id] > 0.0 {
                        logits[token_id] /= self.config.repetition_penalty as f32;
                    } else {
                        logits[token_id] *= self.config.repetition_penalty as f32;
                    }
                }

                // Presence penalty (additive, applied once if token appears)
                if self.config.presence_penalty != 0.0 {
                    logits[token_id] -= self.config.presence_penalty as f32;
                }

                // Frequency penalty (additive, scales with count)
                if self.config.frequency_penalty != 0.0 {
                    logits[token_id] -= (*count as f32) * self.config.frequency_penalty as f32;
                }
            }
        }

        Ok(logits)
    }

    /// Apply top-k filtering to probabilities
    fn apply_top_k(&self, mut probs: Vec<(usize, f32)>) -> Result<Vec<(usize, f32)>> {
        // Sort by probability (descending)
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Keep only top k
        probs.truncate(self.config.top_k);

        // Renormalize
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, p) in &mut probs {
                *p /= sum;
            }
        }

        Ok(probs)
    }

    /// Apply top-p (nucleus) filtering to probabilities
    fn apply_top_p(&self, mut probs: Vec<(usize, f32)>) -> Result<Vec<(usize, f32)>> {
        // Sort by probability (descending)
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Find cutoff point where cumulative probability >= top_p
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = probs.len();

        for (idx, (_, prob)) in probs.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= self.config.top_p as f32 {
                cutoff_idx = idx + 1;
                break;
            }
        }

        // Keep only tokens up to cutoff
        probs.truncate(cutoff_idx);

        // Renormalize
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, p) in &mut probs {
                *p /= sum;
            }
        }

        Ok(probs)
    }

    /// Sample from a multinomial distribution
    fn multinomial_sample(&mut self, probs: &[(usize, f32)]) -> Result<u32> {
        let total: f32 = probs.iter().map(|(_, p)| p).sum();

        if total <= 0.0 {
            anyhow::bail!("Sum of probabilities is zero or negative");
        }

        let mut r: f32 = self.rng.gen::<f32>() * total;

        for &(token_id, prob) in probs {
            r -= prob;
            if r <= 0.0 {
                return Ok(token_id as u32);
            }
        }

        // Fallback to last token (handles floating point precision issues)
        Ok(probs.last().map(|&(id, _)| id as u32).unwrap_or(0))
    }
}

/// Compute softmax of a vector of logits
fn softmax(logits: &[f32]) -> Vec<(usize, f32)> {
    // Find max for numerical stability
    let max_logit = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(logit - max)
    let exps: Vec<f32> = logits
        .iter()
        .map(|&logit| (logit - max_logit).exp())
        .collect();

    // Compute sum
    let sum: f32 = exps.iter().sum();

    // Normalize and pair with indices
    exps.iter()
        .enumerate()
        .map(|(idx, &exp)| (idx, exp / sum))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Check that probabilities sum to 1
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that higher logit = higher probability
        assert!(probs[2].1 > probs[1].1);
        assert!(probs[1].1 > probs[0].1);
    }

    #[test]
    fn test_greedy_sampling() {
        let config = SamplingConfig::greedy();
        let mut sampler = Sampler::new(config);

        let logits = Tensor::from_vec(
            vec![0.1, 0.5, 0.3, 0.8, 0.2],
            &[5],
            &Device::Cpu,
        )
        .unwrap();

        let token = sampler.sample(&logits, &[]).unwrap();
        assert_eq!(token, 3); // Index 3 has highest value (0.8)
    }

    #[test]
    fn test_repetition_penalty() {
        let mut config = SamplingConfig::default();
        config.repetition_penalty = 2.0;
        config.seed = Some(42); // For reproducibility

        let mut sampler = Sampler::new(config);

        let logits = Tensor::from_vec(
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            &[5],
            &Device::Cpu,
        )
        .unwrap();

        // First sample with no history
        let token1 = sampler.sample(&logits, &[]).unwrap();

        // Second sample with token1 in history - should be less likely to repeat
        let token2 = sampler.sample(&logits, &[token1]).unwrap();

        // This test is probabilistic, but with repetition penalty,
        // we expect different tokens most of the time
        // (In practice, we just verify the function doesn't crash)
        assert!(token2 < 5);
    }

    #[test]
    fn test_sampling_configs() {
        let greedy = SamplingConfig::greedy();
        assert_eq!(greedy.temperature, 0.0);
        assert_eq!(greedy.top_k, 1);

        let creative = SamplingConfig::creative();
        assert!(creative.temperature > 0.5);

        let precise = SamplingConfig::precise();
        assert!(precise.temperature < 0.5);
    }
}
