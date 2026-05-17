// Autoregressive text generation with stop token detection
//
// This module implements the generation loop for LLM inference,
// handling tokenization, sampling, and stop conditions.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::{Duration, Instant};

use crate::sampling::{Sampler, SamplingConfig};
use crate::tokenizer::GgufTokenizer;

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,

    /// Stop generation when any of these tokens are generated
    pub stop_tokens: Vec<u32>,

    /// Stop generation when any of these strings appear in the output
    pub stop_strings: Vec<String>,

    /// Sampling configuration
    pub sampling: SamplingConfig,

    /// Whether to echo the input prompt in the output
    pub echo_prompt: bool,

    /// Maximum time to spend generating (None = no limit)
    pub max_time: Option<Duration>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            stop_tokens: Vec::new(),
            stop_strings: Vec::new(),
            sampling: SamplingConfig::default(),
            echo_prompt: false,
            max_time: None,
        }
    }
}

impl GenerationConfig {
    /// Create a configuration with common stop tokens for chat models
    pub fn chat_defaults(tokenizer: &GgufTokenizer) -> Self {
        let mut stop_tokens = Vec::new();

        // Add EOS token
        if let Some(eos_id) = tokenizer.eos_token_id() {
            stop_tokens.push(eos_id);
        }

        Self {
            stop_tokens,
            stop_strings: vec![
                "</s>".to_string(),      // Mistral EOS
                "[/INST]".to_string(),   // Instruction end marker
                "Human:".to_string(),    // Common chat marker
                "User:".to_string(),     // Common chat marker
            ],
            ..Default::default()
        }
    }
}

/// Statistics collected during generation
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Number of tokens generated (excluding prompt)
    pub tokens_generated: usize,

    /// Total time spent generating
    pub generation_time: Duration,

    /// Tokens per second
    pub tokens_per_second: f64,

    /// Reason generation stopped
    pub stop_reason: StopReason,

    /// Prompt token count
    pub prompt_tokens: usize,
}

/// Reason why generation stopped
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Reached maximum token limit
    MaxTokens,

    /// Generated a stop token
    StopToken,

    /// Generated a stop string
    StopString,

    /// Reached time limit
    Timeout,

    /// End of sequence token
    EOS,
}

/// Autoregressive text generator
pub struct Generator {
    config: GenerationConfig,
    sampler: Sampler,
}

impl Generator {
    /// Create a new generator with the given configuration
    pub fn new(config: GenerationConfig) -> Self {
        let sampler = Sampler::new(config.sampling.clone());

        Self { config, sampler }
    }

    /// Generate text given a prompt and a model forward function
    ///
    /// # Arguments
    /// * `prompt_tokens` - Input token IDs
    /// * `tokenizer` - Tokenizer for decoding
    /// * `forward_fn` - Function that takes token history and returns next-token logits
    ///
    /// # Returns
    /// Tuple of (generated_text, statistics)
    ///
    /// # Example
    /// ```ignore
    /// let generator = Generator::new(GenerationConfig::default());
    /// let (text, stats) = generator.generate(
    ///     &prompt_tokens,
    ///     &tokenizer,
    ///     |tokens| {
    ///         // Run model inference
    ///         model.forward(tokens)
    ///     },
    /// )?;
    /// ```
    pub fn generate<F>(
        &mut self,
        prompt_tokens: &[u32],
        tokenizer: &GgufTokenizer,
        mut forward_fn: F,
    ) -> Result<(String, GenerationStats)>
    where
        F: FnMut(&[u32]) -> Result<Tensor>,
    {
        let start_time = Instant::now();

        let mut all_tokens = prompt_tokens.to_vec();
        let mut generated_tokens = Vec::new();
        let prompt_len = prompt_tokens.len();

        let mut stop_reason = StopReason::MaxTokens;

        // Generation loop
        for _ in 0..self.config.max_tokens {
            // Check timeout
            if let Some(max_time) = self.config.max_time {
                if start_time.elapsed() >= max_time {
                    stop_reason = StopReason::Timeout;
                    break;
                }
            }

            // Forward pass to get next-token logits
            let logits = forward_fn(&all_tokens)?;

            // Sample next token
            let next_token = self.sampler.sample(&logits, &generated_tokens)?;

            // Check for stop conditions
            if self.config.stop_tokens.contains(&next_token) {
                stop_reason = if Some(next_token) == tokenizer.eos_token_id() {
                    StopReason::EOS
                } else {
                    StopReason::StopToken
                };
                break;
            }

            // Add token to history
            all_tokens.push(next_token);
            generated_tokens.push(next_token);

            // Check for stop strings (decode incrementally)
            if !self.config.stop_strings.is_empty() {
                let current_text = tokenizer.decode(&generated_tokens, true)?;
                if self
                    .config
                    .stop_strings
                    .iter()
                    .any(|stop_str| current_text.contains(stop_str))
                {
                    stop_reason = StopReason::StopString;
                    break;
                }
            }
        }

        // Decode final output
        let output_tokens = if self.config.echo_prompt {
            &all_tokens
        } else {
            &generated_tokens
        };

        let output_text = tokenizer.decode(output_tokens, true)?;

        // Compute statistics
        let generation_time = start_time.elapsed();
        let tokens_per_second = if generation_time.as_secs_f64() > 0.0 {
            generated_tokens.len() as f64 / generation_time.as_secs_f64()
        } else {
            0.0
        };

        let stats = GenerationStats {
            tokens_generated: generated_tokens.len(),
            generation_time,
            tokens_per_second,
            stop_reason,
            prompt_tokens: prompt_len,
        };

        Ok((output_text, stats))
    }

    /// Generate text in streaming mode, calling a callback for each token
    ///
    /// # Arguments
    /// * `prompt_tokens` - Input token IDs
    /// * `tokenizer` - Tokenizer for decoding
    /// * `forward_fn` - Function that takes token history and returns next-token logits
    /// * `token_callback` - Called for each generated token with (token_id, decoded_text)
    ///
    /// # Returns
    /// Generation statistics
    pub fn generate_stream<F, C>(
        &mut self,
        prompt_tokens: &[u32],
        tokenizer: &GgufTokenizer,
        mut forward_fn: F,
        mut token_callback: C,
    ) -> Result<GenerationStats>
    where
        F: FnMut(&[u32]) -> Result<Tensor>,
        C: FnMut(u32, &str) -> Result<()>,
    {
        let start_time = Instant::now();

        let mut all_tokens = prompt_tokens.to_vec();
        let mut generated_tokens = Vec::new();
        let prompt_len = prompt_tokens.len();

        let mut stop_reason = StopReason::MaxTokens;

        // Generation loop
        for _ in 0..self.config.max_tokens {
            // Check timeout
            if let Some(max_time) = self.config.max_time {
                if start_time.elapsed() >= max_time {
                    stop_reason = StopReason::Timeout;
                    break;
                }
            }

            // Forward pass to get next-token logits
            let logits = forward_fn(&all_tokens)?;

            // Sample next token
            let next_token = self.sampler.sample(&logits, &generated_tokens)?;

            // Check for stop conditions
            if self.config.stop_tokens.contains(&next_token) {
                stop_reason = if Some(next_token) == tokenizer.eos_token_id() {
                    StopReason::EOS
                } else {
                    StopReason::StopToken
                };
                break;
            }

            // Add token to history
            all_tokens.push(next_token);
            generated_tokens.push(next_token);

            // Decode and emit token
            let token_text = tokenizer.decode(&[next_token], true)?;
            token_callback(next_token, &token_text)?;

            // Check for stop strings
            if !self.config.stop_strings.is_empty() {
                let current_text = tokenizer.decode(&generated_tokens, true)?;
                if self
                    .config
                    .stop_strings
                    .iter()
                    .any(|stop_str| current_text.contains(stop_str))
                {
                    stop_reason = StopReason::StopString;
                    break;
                }
            }
        }

        // Compute statistics
        let generation_time = start_time.elapsed();
        let tokens_per_second = if generation_time.as_secs_f64() > 0.0 {
            generated_tokens.len() as f64 / generation_time.as_secs_f64()
        } else {
            0.0
        };

        Ok(GenerationStats {
            tokens_generated: generated_tokens.len(),
            generation_time,
            tokens_per_second,
            stop_reason,
            prompt_tokens: prompt_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 512);
        assert!(!config.echo_prompt);
    }

    #[test]
    fn test_stop_reason() {
        assert_eq!(StopReason::EOS, StopReason::EOS);
        assert_ne!(StopReason::EOS, StopReason::MaxTokens);
    }

    #[test]
    fn test_generation_stats() {
        let stats = GenerationStats {
            tokens_generated: 100,
            generation_time: Duration::from_secs(10),
            tokens_per_second: 10.0,
            stop_reason: StopReason::MaxTokens,
            prompt_tokens: 50,
        };

        assert_eq!(stats.tokens_generated, 100);
        assert_eq!(stats.tokens_per_second, 10.0);
    }

    #[test]
    fn test_mock_generation() {
        let config = GenerationConfig {
            max_tokens: 5,
            ..Default::default()
        };

        let mut generator = Generator::new(config);

        // Mock tokenizer (in practice, load from GGUF)
        // This test demonstrates the API structure

        let prompt_tokens = vec![1, 2, 3];

        // Mock forward function that always returns the same logits
        let mock_forward = |_tokens: &[u32]| -> Result<Tensor> {
            // Return constant logits favoring token 42
            let logits = vec![0.1; 1000];
            Tensor::from_vec(logits, &[1000], &Device::Cpu).map_err(|e| anyhow::anyhow!("{}", e))
        };

        // In a real test, we'd need an actual tokenizer
        // For now, just verify the forward function works
        let result = mock_forward(&prompt_tokens);
        assert!(result.is_ok());
    }
}
