use libp2p::gossipsub::{IdentTopic, Topic};
use serde::{Deserialize, Serialize};
use tracing::info;
use uuid::Uuid;
use chrono;

/// Gossipsub topics for distributed AI inference
pub const TOPIC_AI_INFERENCE_REQUEST: &str = "qnk/ai/inference-request/v1";
pub const TOPIC_AI_LAYER_OUTPUT: &str = "qnk/ai/layer-output/v1";
pub const TOPIC_AI_NODE_CAPABILITY: &str = "qnk/ai/node-capability/v1";
pub const TOPIC_AI_COORDINATOR: &str = "qnk/ai/coordinator/v1";
pub const TOPIC_AI_HEARTBEAT: &str = "qnk/ai/heartbeat/v1";

/// AI-specific Gossipsub topics manager
pub struct DistributedAITopics {
    pub inference_request: IdentTopic,
    pub layer_output: IdentTopic,
    pub node_capability: IdentTopic,
    pub coordinator: IdentTopic,
    pub heartbeat: IdentTopic,
}

impl DistributedAITopics {
    pub fn new() -> Self {
        info!("🤖 Initializing Distributed AI Gossipsub topics");

        Self {
            inference_request: IdentTopic::new(TOPIC_AI_INFERENCE_REQUEST),
            layer_output: IdentTopic::new(TOPIC_AI_LAYER_OUTPUT),
            node_capability: IdentTopic::new(TOPIC_AI_NODE_CAPABILITY),
            coordinator: IdentTopic::new(TOPIC_AI_COORDINATOR),
            heartbeat: IdentTopic::new(TOPIC_AI_HEARTBEAT),
        }
    }

    /// Get all AI topics for subscription
    pub fn all_topics(&self) -> Vec<IdentTopic> {
        vec![
            self.inference_request.clone(),
            self.layer_output.clone(),
            self.node_capability.clone(),
            self.coordinator.clone(),
            self.heartbeat.clone(),
        ]
    }

    /// Check if a topic hash matches any AI topic
    pub fn is_ai_topic(&self, topic: &libp2p::gossipsub::TopicHash) -> bool {
        let topic_str = topic.as_str();
        topic_str.starts_with("qnk/ai/")
    }
}

impl Default for DistributedAITopics {
    fn default() -> Self {
        Self::new()
    }
}

/// AI message envelope for Gossipsub with AEGIS-QL authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIGossipsubMessage {
    /// Protocol version for compatibility checking (v0.9.29+ FIX: Prevents binary incompatibility)
    /// Version 1: Initial protocol with all current features
    /// Increment when making breaking changes to message format
    #[serde(default = "default_protocol_version")]
    pub protocol_version: u32,

    pub message_id: String,
    pub timestamp: i64,
    pub sender_node_id: String,
    pub sender_peer_id: String,
    pub payload: AIMessagePayload,

    // AEGIS-QL post-quantum message authentication (Phase 1 enhancement)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)] // v0.9.14 FIX: Backwards compatibility - use None if field missing
    pub aegis_signature: Option<Vec<u8>>, // AEGIS-256 MAC for message integrity
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)] // v0.9.14 FIX: Backwards compatibility - use None if field missing
    pub sender_public_key: Option<Vec<u8>>, // Ed25519 public key for verification

    // Retry and reliability metadata
    #[serde(default)] // v0.9.14 FIX: Backwards compatibility - use 0 if field missing
    pub sequence_number: u64, // Monotonic sequence for deduplication
    #[serde(default)] // v0.9.14 FIX: Backwards compatibility - use 0 if field missing
    pub retry_count: u8, // Number of retries (for exponential backoff)
    #[serde(default)] // v0.9.14 FIX: Backwards compatibility - use Normal if field missing
    pub priority: MessagePriority, // Priority for gossipsub mesh routing
}

/// Default protocol version for backwards compatibility
/// v0.9.29+ nodes will use version 1, older nodes default to 0
fn default_protocol_version() -> u32 {
    0 // Old binaries without version field will deserialize as version 0
}

/// Current protocol version - increment when making breaking changes
pub const CURRENT_PROTOCOL_VERSION: u32 = 1;

/// Message priority for gossipsub routing optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    Low = 0,      // Heartbeats, capability announcements
    Normal = 1,   // Regular inference requests
    High = 2,     // Layer outputs, KV cache updates
    Critical = 3, // Coordinator election, error recovery
}

impl Default for MessagePriority {
    fn default() -> Self {
        MessagePriority::Normal
    }
}

impl AIGossipsubMessage {
    /// Create a new message with automatic sequence numbering
    /// v0.9.29+ FIX: Now includes protocol version for compatibility checking
    pub fn new(
        sender_node_id: String,
        sender_peer_id: String,
        payload: AIMessagePayload,
        sequence_number: u64,
    ) -> Self {
        use uuid::Uuid;

        let priority = match &payload {
            AIMessagePayload::CoordinatorElection { .. } => MessagePriority::Critical,
            AIMessagePayload::LayerOutput { .. } | AIMessagePayload::KVCacheUpdate { .. } => MessagePriority::High,
            AIMessagePayload::InferenceRequest { .. } | AIMessagePayload::InferenceResponse { .. } => MessagePriority::Normal,
            AIMessagePayload::Heartbeat { .. } | AIMessagePayload::NodeCapability { .. } => MessagePriority::Low,
            _ => MessagePriority::Normal,
        };

        Self {
            protocol_version: CURRENT_PROTOCOL_VERSION, // v0.9.29+ FIX: Set protocol version
            message_id: Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            sender_node_id,
            sender_peer_id,
            payload,
            aegis_signature: None,
            sender_public_key: None,
            sequence_number,
            retry_count: 0,
            priority,
        }
    }

    /// Increment retry count for exponential backoff
    pub fn increment_retry(&mut self) {
        self.retry_count = self.retry_count.saturating_add(1);
    }

    /// Calculate exponential backoff delay in milliseconds
    pub fn backoff_delay_ms(&self) -> u64 {
        // Exponential backoff: 100ms, 200ms, 400ms, 800ms, 1600ms (max 5 retries)
        let base_delay = 100;
        let max_retries = 5;
        if self.retry_count >= max_retries {
            return base_delay * (1 << (max_retries - 1)); // Cap at max delay
        }
        base_delay * (1 << self.retry_count)
    }

    /// Check if message should be retired (too many retries)
    pub fn should_retire(&self) -> bool {
        self.retry_count >= 5
    }

    /// Verify message authenticity (placeholder for AEGIS-QL verification)
    pub fn verify_signature(&self) -> bool {
        // TODO: Implement AEGIS-QL signature verification when q-aegis-ql crate is available
        // For now, allow unsigned messages for backwards compatibility
        if self.aegis_signature.is_none() {
            return true; // Allow unsigned messages
        }

        // Verify signature with AEGIS-256 MAC
        // This will be implemented once q-aegis-ql compilation is fixed
        true
    }
}

/// Distributed AI message payload types
/// v0.9.29+ FIX: Added explicit discriminants for binary stability
/// IMPORTANT: Never reorder variants or change discriminants - this will break compatibility!
/// To add new message types, append to the end with the next sequential number
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(u8)] // Force explicit u8 discriminants for binary stability
pub enum AIMessagePayload {
    InferenceRequest {
        request_id: String,
        prompt: String,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        model: String,
    } = 0,
    InferenceResponse {
        request_id: String,
        generated_text: String,
        tokens_generated: usize,
        latency_ms: u64,
        nodes_participated: Vec<String>,
    } = 1,
    LayerOutput {
        request_id: String,
        layer_index: usize,
        compressed_data: Vec<u8>,
        shape: Vec<usize>,
    } = 2,
    NodeCapability {
        node_id: String,
        peer_id: String,
        capability: NodeCapability,
        available_layers: usize,
    } = 3,
    CoordinatorElection {
        node_id: String,
        score: u64,
        uptime_secs: u64,
        inference_count: u64,
    } = 4,
    Heartbeat {
        node_id: String,
        active_requests: usize,
        layers_assigned: Option<(usize, usize)>, // (start, end)
    } = 5,
    LayerAssignment {
        request_id: String,
        assignments: std::collections::HashMap<String, (usize, usize)>, // node_id -> (start_layer, end_layer)
    } = 6,
    KVCacheUpdate {
        request_id: String,
        layer_index: usize,
        cache_data: Vec<u8>,
        sequence_length: usize,
    } = 7,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeCapability {
    CPU { cores: usize, ram_gb: usize },
    CUDA { vram_gb: usize, compute_capability: String },
    Metal { vram_gb: usize },
}

impl NodeCapability {
    pub fn score(&self) -> u64 {
        match self {
            NodeCapability::CPU { cores, ram_gb } => (*cores as u64) * 10 + (*ram_gb as u64),
            NodeCapability::CUDA { vram_gb, .. } => (*vram_gb as u64) * 1000,
            NodeCapability::Metal { vram_gb } => (*vram_gb as u64) * 800,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_topics_creation() {
        let topics = DistributedAITopics::new();
        assert_eq!(topics.inference_request.as_str(), TOPIC_AI_INFERENCE_REQUEST);
        assert_eq!(topics.all_topics().len(), 5);
    }

    #[test]
    fn test_is_ai_topic() {
        use libp2p::gossipsub::Topic;

        let topics = DistributedAITopics::new();
        let ai_topic = IdentTopic::new("qnk/ai/test/v1");
        let other_topic = IdentTopic::new("qnk/dex/orders/v1");

        assert!(topics.is_ai_topic(&ai_topic.hash()));
        assert!(!topics.is_ai_topic(&other_topic.hash()));
    }

    #[test]
    fn test_node_capability_scoring() {
        let cpu = NodeCapability::CPU { cores: 8, ram_gb: 16 };
        let cuda = NodeCapability::CUDA {
            vram_gb: 12,
            compute_capability: "8.0".to_string(),
        };

        assert!(cuda.score() > cpu.score());
        assert_eq!(cpu.score(), 96); // 8*10 + 16
        assert_eq!(cuda.score(), 12000); // 12*1000
    }
}
