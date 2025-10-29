use libp2p::gossipsub::{IdentTopic, Topic};
use serde::{Deserialize, Serialize};
use tracing::info;

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

/// AI message envelope for Gossipsub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIGossipsubMessage {
    pub message_id: String,
    pub timestamp: i64,
    pub sender_node_id: String,
    pub sender_peer_id: String,
    pub payload: AIMessagePayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AIMessagePayload {
    InferenceRequest {
        request_id: String,
        prompt: String,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        model: String,
    },
    InferenceResponse {
        request_id: String,
        generated_text: String,
        tokens_generated: usize,
        latency_ms: u64,
        nodes_participated: Vec<String>,
    },
    LayerOutput {
        request_id: String,
        layer_index: usize,
        compressed_data: Vec<u8>,
        shape: Vec<usize>,
    },
    NodeCapability {
        node_id: String,
        peer_id: String,
        capability: NodeCapability,
        available_layers: usize,
    },
    CoordinatorElection {
        node_id: String,
        score: u64,
        uptime_secs: u64,
        inference_count: u64,
    },
    Heartbeat {
        node_id: String,
        active_requests: usize,
        layers_assigned: Option<(usize, usize)>, // (start, end)
    },
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
