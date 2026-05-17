//! Chat Protector - TemporalShield protection for AI chat content
//!
//! Protects sensitive chat content (user prompts, AI reasoning) with
//! TemporalShield for HNDL attack resistance. Uses (3,5) threshold sharing.
//!
//! ## Security Properties
//! - User prompts reveal trading strategies, personal info
//! - AI reasoning may contain sensitive analysis
//! - Protected with information-theoretic OTP encryption
//! - Threshold (3,5) requires 3 of 5 trustees to decrypt

use q_storage::{ChatMessage, ProtectedChatMessage};
use q_temporal_shield::{TemporalShield, TemporalShieldConfig, TrusteePublicKey};
use tracing::{debug, error, info, warn};

/// Threshold parameters for chat protection
pub const CHAT_PROTECTION_THRESHOLD: usize = 3;
pub const CHAT_PROTECTION_TOTAL: usize = 5;

/// Chat Protector - wraps chat messages in TemporalEnvelope
pub struct ChatProtector {
    shield: TemporalShield,
    trustees: Vec<TrusteePublicKey>,
}

impl ChatProtector {
    /// Create a new chat protector with the given trustees
    pub fn new(trustees: Vec<TrusteePublicKey>) -> Result<Self, String> {
        if trustees.len() != CHAT_PROTECTION_TOTAL {
            return Err(format!(
                "Expected {} trustees, got {}",
                CHAT_PROTECTION_TOTAL,
                trustees.len()
            ));
        }

        let config = TemporalShieldConfig::custom(
            CHAT_PROTECTION_THRESHOLD,
            CHAT_PROTECTION_TOTAL,
            128, // 128-bit security
        )
        .map_err(|e| format!("Config error: {:?}", e))?;

        let shield = TemporalShield::new(config);

        Ok(Self { shield, trustees })
    }

    /// Protect a chat message
    ///
    /// Encrypts the content and reasoning fields, returns a ProtectedChatMessage.
    pub fn protect_message(&self, message: &ChatMessage) -> Result<ProtectedChatMessage, String> {
        // Compute content hash for search indexing
        let content_hash = *blake3::hash(message.content.as_bytes()).as_bytes();

        // Protect content
        let protected_content = if !message.content.is_empty() {
            let envelope = self.shield
                .protect(message.content.as_bytes(), &self.trustees)
                .map_err(|e| format!("Content protection failed: {:?}", e))?;
            envelope.to_bytes()
                .map_err(|e| format!("Envelope serialization failed: {:?}", e))?
        } else {
            Vec::new()
        };

        // Protect reasoning (if present)
        let protected_reasoning = if let Some(ref reasoning) = message.reasoning {
            if !reasoning.is_empty() {
                let envelope = self.shield
                    .protect(reasoning.as_bytes(), &self.trustees)
                    .map_err(|e| format!("Reasoning protection failed: {:?}", e))?;
                Some(
                    envelope.to_bytes()
                        .map_err(|e| format!("Reasoning envelope serialization failed: {:?}", e))?
                )
            } else {
                None
            }
        } else {
            None
        };

        Ok(ProtectedChatMessage::new(
            message.index,
            message.role.clone(),
            protected_content,
            protected_reasoning,
            message.generation_stats.clone(),
            content_hash,
        ))
    }

    /// Reconstruct a chat message from protected form
    ///
    /// Requires at least `CHAT_PROTECTION_THRESHOLD` decrypted shares.
    pub fn reconstruct_message(
        &self,
        protected: &ProtectedChatMessage,
        content_shares: &[(usize, Vec<u8>)],
        reasoning_shares: Option<&[(usize, Vec<u8>)]>,
    ) -> Result<ChatMessage, String> {
        // Reconstruct content
        let content = if !protected.protected_content.is_empty() && protected.is_protected {
            let envelope = q_temporal_shield::TemporalEnvelope::from_bytes(&protected.protected_content)
                .map_err(|e| format!("Failed to parse content envelope: {:?}", e))?;

            let bytes = self.shield
                .reconstruct(&envelope, content_shares)
                .map_err(|e| format!("Content reconstruction failed: {:?}", e))?;

            String::from_utf8(bytes)
                .map_err(|e| format!("Invalid UTF-8 content: {}", e))?
        } else {
            // Unprotected or empty
            String::from_utf8_lossy(&protected.protected_content).to_string()
        };

        // Reconstruct reasoning (if present)
        let reasoning = if let Some(ref protected_reasoning) = protected.protected_reasoning {
            if !protected_reasoning.is_empty() && protected.is_protected {
                let shares = reasoning_shares.ok_or("Reasoning shares required but not provided")?;

                let envelope = q_temporal_shield::TemporalEnvelope::from_bytes(protected_reasoning)
                    .map_err(|e| format!("Failed to parse reasoning envelope: {:?}", e))?;

                let bytes = self.shield
                    .reconstruct(&envelope, shares)
                    .map_err(|e| format!("Reasoning reconstruction failed: {:?}", e))?;

                Some(String::from_utf8(bytes)
                    .map_err(|e| format!("Invalid UTF-8 reasoning: {}", e))?)
            } else {
                Some(String::from_utf8_lossy(protected_reasoning).to_string())
            }
        } else {
            None
        };

        Ok(ChatMessage {
            index: protected.index,
            role: protected.role.clone(),
            content,
            timestamp: protected.timestamp,
            images: None,
            audio: None,
            reasoning,
            generation_stats: protected.generation_stats.clone(),
        })
    }
}

/// Check if a user wants protected chat (from settings or header)
pub fn should_protect_chat(enable_protection: bool) -> bool {
    // Could also check environment variable or user preferences
    enable_protection || std::env::var("Q_PROTECT_CHAT").is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use q_temporal_shield::TrusteePublicKey;

    fn generate_test_trustees(n: usize) -> Vec<TrusteePublicKey> {
        (0..n)
            .map(|i| {
                let keypair = TrusteePublicKey::generate(Some(format!("Test-{}", i))).unwrap();
                keypair.public_key
            })
            .collect()
    }

    #[test]
    fn test_chat_protector_creation() {
        let trustees = generate_test_trustees(5);
        let protector = ChatProtector::new(trustees).unwrap();
        assert!(protector.trustees.len() == 5);
    }

    #[test]
    fn test_wrong_trustee_count() {
        let trustees = generate_test_trustees(3);
        let result = ChatProtector::new(trustees);
        assert!(result.is_err());
    }
}
