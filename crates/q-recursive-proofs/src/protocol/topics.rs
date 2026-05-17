//! Gossipsub Topics for Recursive Proof Network
//!
//! Defines the P2P topics used for proof generation coordination.

/// Topic for epoch proof task announcements
pub const TOPIC_EPOCH_PROOF_TASK: &str = "/qnk/epoch-proof-task";

/// Topic for epoch proof submissions
pub const TOPIC_EPOCH_PROOFS: &str = "/qnk/epoch-proofs";

/// Topic for proof verification results
pub const TOPIC_PROOF_VERIFICATION: &str = "/qnk/proof-verification";

/// Topic for light client proof requests
pub const TOPIC_LIGHT_CLIENT_REQUEST: &str = "/qnk/light-client-request";

/// Topic for light client proof responses
pub const TOPIC_LIGHT_CLIENT_RESPONSE: &str = "/qnk/light-client-response";

/// DHT key prefix for epoch proofs
pub const DHT_EPOCH_PROOF_PREFIX: &str = "/qnk/proofs/epoch/";

/// DHT key for latest light client proof
pub const DHT_LIGHT_CLIENT_LATEST: &str = "/qnk/proofs/light-client/latest";

/// All recursive proof topics
pub struct RecursiveProofTopics;

impl RecursiveProofTopics {
    /// Get all topic strings
    pub fn all() -> Vec<&'static str> {
        vec![
            TOPIC_EPOCH_PROOF_TASK,
            TOPIC_EPOCH_PROOFS,
            TOPIC_PROOF_VERIFICATION,
            TOPIC_LIGHT_CLIENT_REQUEST,
            TOPIC_LIGHT_CLIENT_RESPONSE,
        ]
    }

    /// Get DHT key for epoch proof
    pub fn epoch_proof_key(epoch: u64) -> String {
        format!("{}{}", DHT_EPOCH_PROOF_PREFIX, epoch)
    }

    /// Get DHT key for light client proof
    pub fn light_client_key() -> &'static str {
        DHT_LIGHT_CLIENT_LATEST
    }

    /// Is this a recursive proof topic?
    pub fn is_recursive_proof_topic(topic: &str) -> bool {
        Self::all().contains(&topic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_list() {
        let topics = RecursiveProofTopics::all();
        assert_eq!(topics.len(), 5);
        assert!(topics.contains(&TOPIC_EPOCH_PROOFS));
    }

    #[test]
    fn test_epoch_key() {
        let key = RecursiveProofTopics::epoch_proof_key(42);
        assert_eq!(key, "/qnk/proofs/epoch/42");
    }
}
