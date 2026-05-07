// Hop-limited message routing types.
// Extracted from nova-chat src/network/routing.rs.
// libp2p::PeerId and Multiaddr replaced with String throughout — wire-compatible,
// serde-serializable, no libp2p version dependency.
//
// NOTE: RouterConfig.route_timeout defaults to 30s for direct TCP. Callers running
// over Tor + Dandelion++ should use 180s to accommodate stem phase + Tor RTT.

use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteMessage {
    pub message_id: String,
    /// Sender's peer ID (wallet address or libp2p PeerId as string)
    pub source: String,
    /// Recipient's peer ID
    pub destination: String,
    pub payload: Vec<u8>,
    pub hop_count: u32,
    pub max_hops: u32,
    pub timestamp: u64,
    pub message_type: RouteMessageType,
}

impl RouteMessage {
    pub fn new_direct(source: impl Into<String>, destination: impl Into<String>, payload: Vec<u8>) -> Self {
        Self {
            message_id: uuid::Uuid::new_v4().to_string(),
            source: source.into(),
            destination: destination.into(),
            payload,
            hop_count: 0,
            max_hops: RouterConfig::default().max_hops,
            timestamp: now_secs(),
            message_type: RouteMessageType::DirectMessage,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteMessageType {
    DirectMessage,
    GroupMessage { group_id: String },
    Broadcast,
    RoutingControl,
}

/// Routing decision returned by the router for each incoming message.
#[derive(Debug)]
pub enum RouteAction {
    /// Deliver to local application.
    Deliver(RouteMessage),
    /// Forward to next_hop (peer ID as String).
    Forward { message: RouteMessage, next_hop: String },
    /// Drop with reason.
    Drop(String),
}

#[derive(Debug, Clone)]
pub struct RouterConfig {
    pub max_hops: u32,
    /// How long to keep seen-message IDs to prevent loops.
    pub cache_timeout: Duration,
    /// How long to consider a routing table entry valid.
    /// Use 180s when running over Tor + Dandelion++.
    pub route_timeout: Duration,
    pub cluster_size_limit: usize,
    pub enable_super_nodes: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            max_hops: 8,
            cache_timeout: Duration::from_secs(300),
            route_timeout: Duration::from_secs(30),
            cluster_size_limit: 100,
            enable_super_nodes: true,
        }
    }
}

impl RouterConfig {
    /// Tor-safe config with extended timeouts for Dandelion++ stem phase.
    pub fn tor_safe() -> Self {
        Self {
            route_timeout: Duration::from_secs(180),
            cache_timeout: Duration::from_secs(600),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub routing_table_size: usize,
    pub direct_connections: usize,
    pub local_cluster_size: usize,
    pub super_nodes_count: usize,
    pub pending_routes: usize,
    pub cached_messages: usize,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
