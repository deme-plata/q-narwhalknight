// Loopix Network Components
// Provides network management for anonymous messaging

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Network node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopixNode {
    pub node_id: String,
    pub address: String,
    pub public_key: [u8; 32],
}

/// Configuration for Loopix network node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopixNodeConfig {
    pub node_id: String,
    pub listen_address: String,
    pub max_connections: usize,
}

/// Events emitted by the Loopix network
#[derive(Debug, Clone)]
pub enum LoopixEvent {
    NodeConnected(String),
    NodeDisconnected(String),
    MessageReceived(Vec<u8>),
}

/// Commands for controlling the Loopix network
#[derive(Debug, Clone)]
pub enum LoopixCommand {
    Connect(String),
    Disconnect(String),
    SendMessage(Vec<u8>),
}

/// Network statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatistics {
    pub connected_nodes: usize,
    pub messages_sent: u64,
    pub messages_received: u64,
}

/// Network topology manager
pub struct LoopixNetwork {
    pub nodes: HashMap<String, LoopixNode>,
}

impl LoopixNetwork {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: LoopixNode) {
        self.nodes.insert(node.node_id.clone(), node);
    }

    pub fn get_node(&self, node_id: &str) -> Option<&LoopixNode> {
        self.nodes.get(node_id)
    }

    pub fn remove_node(&mut self, node_id: &str) -> Option<LoopixNode> {
        self.nodes.remove(node_id)
    }
}