//! Gossipsub Message Handler for Distributed AI Inference
//!
//! This module handles incoming AI inference messages from the Gossipsub network.
//! It routes messages to appropriate handlers based on message type.

use crate::types::{AIMessage, InferenceRequest, InferenceResponse, TensorData};
use crate::rpc_worker::{RpcWorkerInfo, WorkerStatus};
use anyhow::{anyhow, Result};
use libp2p::gossipsub::Message as GossipsubMessage;
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

/// Handler for AI Gossipsub messages
pub struct AIGossipsubHandler {
    /// Node ID for this handler
    node_id: String,

    /// Channel to send inference requests to the coordinator
    inference_tx: Option<mpsc::Sender<InferenceRequest>>,

    /// Channel to send layer outputs to the next node
    layer_output_tx: Option<mpsc::Sender<TensorData>>,

    /// Active inference requests being tracked
    active_requests: Arc<RwLock<HashMap<String, InferenceRequestState>>>,

    /// v5.1.0: Registry of available RPC workers discovered via gossipsub
    /// Key: peer_id, Value: RpcWorkerInfo
    rpc_workers: Arc<RwLock<HashMap<String, RpcWorkerInfo>>>,

    /// Statistics for monitoring
    stats: Arc<RwLock<AIHandlerStats>>,
}

/// State of an active inference request
#[derive(Debug, Clone)]
pub struct InferenceRequestState {
    pub request_id: String,
    pub prompt: String,
    pub status: RequestStatus,
    pub nodes_participated: Vec<String>,
    pub started_at: std::time::Instant,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RequestStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
}

/// Statistics for the AI handler
#[derive(Debug, Clone, Default)]
pub struct AIHandlerStats {
    pub total_requests_received: u64,
    pub total_requests_processed: u64,
    pub total_layer_outputs_received: u64,
    pub total_capability_announcements: u64,
    pub total_heartbeats_received: u64,
    pub average_latency_ms: u64,
}

impl AIGossipsubHandler {
    /// Create a new AI Gossipsub handler
    pub fn new(node_id: String) -> Self {
        info!("🤖 Initializing AI Gossipsub handler for node: {}", node_id);

        Self {
            node_id,
            inference_tx: None,
            layer_output_tx: None,
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            rpc_workers: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(AIHandlerStats::default())),
        }
    }

    /// Set the inference request channel
    pub fn set_inference_channel(&mut self, tx: mpsc::Sender<InferenceRequest>) {
        self.inference_tx = Some(tx);
    }

    /// Set the layer output channel
    pub fn set_layer_output_channel(&mut self, tx: mpsc::Sender<TensorData>) {
        self.layer_output_tx = Some(tx);
    }

    /// Handle an incoming Gossipsub message
    pub async fn handle_message(&self, topic: &str, message: &GossipsubMessage) -> Result<()> {
        debug!("📨 Received AI message on topic: {}", topic);

        // Parse the message as AIMessage
        let ai_message: AIMessage = match serde_json::from_slice(&message.data) {
            Ok(msg) => msg,
            Err(e) => {
                warn!("⚠️ Failed to parse AI message: {}", e);
                return Err(anyhow!("Invalid AI message format: {}", e));
            }
        };

        // Route to appropriate handler based on message type
        match ai_message {
            AIMessage::InferenceRequest(req) => {
                self.handle_inference_request(req).await
            }
            AIMessage::InferenceResponse(resp) => {
                self.handle_inference_response(resp).await
            }
            AIMessage::LayerOutput(tensor) => {
                self.handle_layer_output(tensor).await
            }
            AIMessage::NodeCapability(assignment) => {
                self.handle_node_capability(assignment).await
            }
            AIMessage::CoordinatorElection { node_id, score } => {
                self.handle_coordinator_election(node_id, score).await
            }
            AIMessage::Heartbeat { node_id, timestamp } => {
                self.handle_heartbeat(node_id, timestamp).await
            }
            AIMessage::RpcWorkerAvailable(info) => {
                self.handle_rpc_worker_available(info).await
            }
            AIMessage::RpcWorkerStopped { peer_id } => {
                self.handle_rpc_worker_stopped(peer_id).await
            }
        }
    }

    /// Handle an inference request
    async fn handle_inference_request(&self, request: InferenceRequest) -> Result<()> {
        info!("🎯 Received inference request: {} (prompt: {}...)",
              request.request_id,
              request.prompt.chars().take(50).collect::<String>());

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_requests_received += 1;
        }

        // Store request state
        {
            let mut active = self.active_requests.write().await;
            active.insert(
                request.request_id.clone(),
                InferenceRequestState {
                    request_id: request.request_id.clone(),
                    prompt: request.prompt.clone(),
                    status: RequestStatus::Pending,
                    nodes_participated: vec![],
                    started_at: std::time::Instant::now(),
                },
            );
        }

        // Forward to inference coordinator if channel is set
        if let Some(tx) = &self.inference_tx {
            tx.send(request).await
                .map_err(|e| anyhow!("Failed to send inference request: {}", e))?;
            info!("✅ Forwarded inference request to coordinator");
        } else {
            warn!("⚠️ No inference channel configured - request not forwarded");
        }

        Ok(())
    }

    /// Handle an inference response
    async fn handle_inference_response(&self, response: InferenceResponse) -> Result<()> {
        info!("✅ Received inference response: {} ({} tokens, {}ms latency)",
              response.request_id,
              response.tokens_generated,
              response.latency_ms);

        // Update request state
        {
            let mut active = self.active_requests.write().await;
            if let Some(state) = active.get_mut(&response.request_id) {
                state.status = RequestStatus::Completed;
                state.nodes_participated = response.nodes_participated.clone();

                // Update statistics
                let latency = state.started_at.elapsed().as_millis() as u64;
                let mut stats = self.stats.write().await;
                stats.total_requests_processed += 1;
                stats.average_latency_ms =
                    (stats.average_latency_ms * (stats.total_requests_processed - 1) + latency)
                    / stats.total_requests_processed;
            }
        }

        info!("📊 Response details: {}", response.generated_text.chars().take(100).collect::<String>());
        Ok(())
    }

    /// Handle a layer output from another node
    async fn handle_layer_output(&self, tensor: TensorData) -> Result<()> {
        debug!("📦 Received layer output: request={}, layer={}, size={}",
               tensor.request_id, tensor.layer_index, tensor.data.len());

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_layer_outputs_received += 1;
        }

        // Forward to layer processing if channel is set
        if let Some(tx) = &self.layer_output_tx {
            tx.send(tensor).await
                .map_err(|e| anyhow!("Failed to send layer output: {}", e))?;
        }

        Ok(())
    }

    /// Handle a node capability announcement
    async fn handle_node_capability(&self, assignment: crate::types::LayerAssignment) -> Result<()> {
        info!("💪 Node capability announced: {} (layers {}-{}, capability score: {})",
              assignment.node_id,
              assignment.layer_start,
              assignment.layer_end,
              assignment.device_capability.score());

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_capability_announcements += 1;
        }

        // TODO: Update capability registry for coordinator
        Ok(())
    }

    /// Handle a coordinator election message
    async fn handle_coordinator_election(&self, node_id: String, score: u64) -> Result<()> {
        info!("🗳️ Coordinator election from node {} with score {}", node_id, score);

        // TODO: Implement coordinator election logic
        // For now, just log the election attempt

        Ok(())
    }

    /// Handle a heartbeat from an active node
    async fn handle_heartbeat(&self, node_id: String, timestamp: i64) -> Result<()> {
        debug!("💓 Heartbeat from node {} at timestamp {}", node_id, timestamp);

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_heartbeats_received += 1;
        }

        // TODO: Update node liveness tracking
        Ok(())
    }

    /// v5.1.0: Handle RPC worker availability announcement
    async fn handle_rpc_worker_available(&self, info: RpcWorkerInfo) -> Result<()> {
        info!("🖥️ RPC worker available: {}:{} (peer: {}, {}GB mem, status: {:?})",
              info.host, info.port, info.peer_id, info.available_memory_gb, info.status);

        {
            let mut stats = self.stats.write().await;
            stats.total_capability_announcements += 1;
        }

        // Add/update worker in registry
        self.rpc_workers.write().await.insert(info.peer_id.clone(), info);

        Ok(())
    }

    /// v5.1.0: Handle RPC worker stopped notification
    async fn handle_rpc_worker_stopped(&self, peer_id: String) -> Result<()> {
        info!("🛑 RPC worker stopped: peer {}", peer_id);

        // Remove from registry
        self.rpc_workers.write().await.remove(&peer_id);

        Ok(())
    }

    /// v5.1.0: Get all known RPC workers (ready or busy)
    pub async fn get_rpc_workers(&self) -> Vec<RpcWorkerInfo> {
        self.rpc_workers.read().await.values().cloned().collect()
    }

    /// v5.1.0: Get ready RPC workers for building --rpc argument
    pub async fn get_ready_rpc_workers(&self) -> Vec<RpcWorkerInfo> {
        self.rpc_workers.read().await
            .values()
            .filter(|w| matches!(w.status, WorkerStatus::Ready | WorkerStatus::Busy))
            .cloned()
            .collect()
    }

    /// v5.1.0: Build the `--rpc` argument string for llama.cpp model loading
    /// Returns "host1:port1,host2:port2" or None if no workers available
    pub async fn build_rpc_arg(&self) -> Option<String> {
        let workers = self.get_ready_rpc_workers().await;
        if workers.is_empty() {
            None
        } else {
            Some(workers.iter()
                .map(|w| format!("{}:{}", w.host, w.port))
                .collect::<Vec<_>>()
                .join(","))
        }
    }

    /// v5.1.0: Get shared reference to RPC worker registry
    pub fn rpc_workers_ref(&self) -> Arc<RwLock<HashMap<String, RpcWorkerInfo>>> {
        Arc::clone(&self.rpc_workers)
    }

    /// Get current handler statistics
    pub async fn get_stats(&self) -> AIHandlerStats {
        self.stats.read().await.clone()
    }

    /// Get active request count
    pub async fn active_request_count(&self) -> usize {
        self.active_requests.read().await.len()
    }

    /// Get a specific request state
    pub async fn get_request_state(&self, request_id: &str) -> Option<InferenceRequestState> {
        self.active_requests.read().await.get(request_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DeviceCapability, LayerAssignment};

    #[tokio::test]
    async fn test_handler_creation() {
        let handler = AIGossipsubHandler::new("test-node".to_string());
        assert_eq!(handler.node_id, "test-node");
        assert_eq!(handler.active_request_count().await, 0);
    }

    #[tokio::test]
    async fn test_inference_request_handling() {
        let handler = AIGossipsubHandler::new("test-node".to_string());

        let request = InferenceRequest {
            request_id: "test-req-123".to_string(),
            prompt: "What is the meaning of life?".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            model: "mistral-7b-instruct-v0.3".to_string(),
            timestamp: None,
            deterministic_seed: None,
            required_model_hash: None,
            max_price_per_token: None,
        };

        // Handle request (without coordinator channel)
        let result = handler.handle_inference_request(request).await;
        assert!(result.is_ok());

        // Check that request was stored
        assert_eq!(handler.active_request_count().await, 1);

        let state = handler.get_request_state("test-req-123").await;
        assert!(state.is_some());
        assert_eq!(state.unwrap().status, RequestStatus::Pending);
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        let handler = AIGossipsubHandler::new("test-node".to_string());

        // Simulate receiving messages
        let request = InferenceRequest {
            request_id: "req-1".to_string(),
            prompt: "test".to_string(),
            max_tokens: None,
            temperature: None,
            model: "test-model".to_string(),
            timestamp: None,
            deterministic_seed: None,
            required_model_hash: None,
            max_price_per_token: None,
        };

        handler.handle_inference_request(request).await.unwrap();

        let stats = handler.get_stats().await;
        assert_eq!(stats.total_requests_received, 1);
    }
}
