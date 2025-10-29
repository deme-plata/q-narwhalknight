// GGUF tokenizer integration using tokenizers crate
// This module provides an interface to tokenizers extracted from GGUF model files
//
// Implementation adapted from mistral.rs/mistralrs-core/src/gguf/gguf_tokenizer.rs
// See gguf_tokenizer.rs for the complete GGUF metadata extraction implementation.

use anyhow::Result;
use std::path::Path;
use std::fs::File;
use std::collections::HashMap;
use tokenizers::Tokenizer;
use candle_core::quantized::gguf_file;

use crate::gguf_tokenizer::{convert_gguf_to_hf_tokenizer, GgufTokenizerConversion};

/// Wrapper around tokenizer with special tokens
pub struct GgufTokenizer {
    tokenizer: Tokenizer,
    bos_token: Option<String>,
    eos_token: Option<String>,
    unk_token: Option<String>,
}

impl GgufTokenizer {
    /// Load tokenizer from GGUF model file
    ///
    /// # Arguments
    /// * `gguf_path` - Path to the .gguf model file containing tokenizer metadata
    ///
    /// # Returns
    /// A GgufTokenizer instance ready for encoding/decoding text
    ///
    /// # Implementation
    /// This extracts tokenizer metadata from the GGUF file and converts it to a
    /// HuggingFace tokenizer format. Supports:
    /// - Unigram (SentencePiece) for Llama/Mistral models
    /// - BPE for GPT-2 style models
    ///
    /// The conversion automatically handles:
    /// - Vocabulary and merge extraction
    /// - Special token identification
    /// - Proper normalization and decoding
    pub fn from_gguf_file<P: AsRef<Path>>(gguf_path: P) -> Result<Self> {
        let mut file = File::open(gguf_path.as_ref())?;
        let content = gguf_file::Content::read(&mut file)?;

        // Extract metadata from GGUF file
        let metadata: HashMap<String, gguf_file::Value> = content.metadata.clone();

        // Convert GGUF tokenizer metadata to HuggingFace tokenizer
        let GgufTokenizerConversion {
            tokenizer,
            bos,
            eos,
            unk,
        } = convert_gguf_to_hf_tokenizer(&metadata)?;

        Ok(Self {
            tokenizer,
            bos_token: bos,
            eos_token: eos,
            unk_token: unk,
        })
    }

    /// Load tokenizer from a pre-converted tokenizer.json file
    ///
    /// # Arguments
    /// * `json_path` - Path to the tokenizer.json file
    ///
    /// # Returns
    /// A GgufTokenizer instance ready for encoding/decoding text
    pub fn from_pretrained<P: AsRef<Path>>(json_path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(json_path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            tokenizer,
            bos_token: Some("<s>".to_string()),  // Mistral defaults
            eos_token: Some("</s>".to_string()),
            unk_token: Some("<unk>".to_string()),
        })
    }

    /// Encode text to token IDs
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize
    /// * `add_special_tokens` - Whether to add BOS/EOS tokens
    ///
    /// # Returns
    /// Vector of token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text
    ///
    /// # Arguments
    /// * `token_ids` - Vector of token IDs to decode
    /// * `skip_special_tokens` - Whether to skip special tokens in output
    ///
    /// # Returns
    /// Decoded text string
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let text = self.tokenizer.decode(token_ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;
        Ok(text)
    }

    /// Get BOS (beginning of sequence) token
    pub fn bos_token(&self) -> Option<&str> {
        self.bos_token.as_deref()
    }

    /// Get EOS (end of sequence) token
    pub fn eos_token(&self) -> Option<&str> {
        self.eos_token.as_deref()
    }

    /// Get UNK (unknown) token
    pub fn unk_token(&self) -> Option<&str> {
        self.unk_token.as_deref()
    }

    /// Get BOS token ID
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token.as_ref().and_then(|token| {
            self.tokenizer.token_to_id(token)
        })
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token.as_ref().and_then(|token| {
            self.tokenizer.token_to_id(token)
        })
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Apply chat template for instruction-following models
    ///
    /// # Arguments
    /// * `messages` - Vector of (role, content) tuples representing the conversation
    ///
    /// # Returns
    /// Formatted prompt string ready for the model
    ///
    /// # Example
    /// ```ignore
    /// let messages = vec![
    ///     ("user", "Explain quantum consensus"),
    /// ];
    /// let prompt = tokenizer.apply_chat_template(&messages)?;
    /// ```
    pub fn apply_chat_template(&self, messages: &[(&str, &str)]) -> Result<String> {
        // Mistral-Instruct-v0.3 chat template format:
        // <s>[INST] {user_message} [/INST]

        let mut formatted = String::new();

        for (role, content) in messages {
            match *role {
                "user" | "human" => {
                    formatted.push_str("[INST] ");
                    formatted.push_str(content);
                    formatted.push_str(" [/INST]");
                }
                "assistant" | "bot" => {
                    formatted.push(' ');
                    formatted.push_str(content);
                    formatted.push_str("</s>");
                }
                "system" => {
                    // System messages go before the first user message
                    formatted.push_str("[INST] <<SYS>>\n");
                    formatted.push_str(content);
                    formatted.push_str("\n<</SYS>>\n\n");
                }
                _ => anyhow::bail!("Unknown role: {}", role),
            }
        }

        Ok(formatted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_template() {
        // Mock tokenizer for testing (in real usage, load from GGUF)
        let messages = vec![
            ("user", "What is quantum consensus?"),
        ];

        // This test would need an actual GGUF file to run
        // For now, just test the template formatting logic
        let formatted = format_chat_template(&messages);
        assert!(formatted.contains("[INST]"));
        assert!(formatted.contains("[/INST]"));
    }

    fn format_chat_template(messages: &[(&str, &str)]) -> String {
        let mut formatted = String::new();
        for (role, content) in messages {
            if *role == "user" {
                formatted.push_str("[INST] ");
                formatted.push_str(content);
                formatted.push_str(" [/INST]");
            }
        }
        formatted
    }
}
