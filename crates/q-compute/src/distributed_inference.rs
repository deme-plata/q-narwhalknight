//! Distributed AI Inference — Issue #005
//!
//! Routes inference requests across compute tunnel peers for distributed
//! model serving. Nodes advertise their model capabilities and pricing;
//! the router picks the best peer based on latency, load, and model match.
//!
//! ## Design
//!
//! - `DistributedInferenceRouter` tracks available inference peers
//! - `InferenceRoute` describes how to reach a specific model on a peer
//! - Peers announce capabilities via gossipsub (ComputePeerInfo)
//! - Revenue is split: node earns routing fee, peer earns inference fee
//!
//! ## Pricing
//!
//! Each peer sets its own price per 1K tokens. The router adds a routing
//! markup (default 10%) and the total is charged to the requestor.

#![allow(dead_code)]

use crate::inference_pool::ModelTier;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use tracing::{debug, info, warn};

/// Default routing markup percentage (10%)
const DEFAULT_ROUTING_MARKUP_PCT: u64 = 10;

/// Peer TTL in seconds
const PEER_INFERENCE_TTL_SECS: u64 = 120;

/// Maximum concurrent remote inference requests
const MAX_REMOTE_INFLIGHT: usize = 32;

// ═══════════════════════════════════════════════════════════════════
// Peer Inference Capabilities
// ═══════════════════════════════════════════════════════════════════

/// Capabilities advertised by a peer for distributed inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInferenceInfo {
    pub peer_id: String,
    pub models: Vec<PeerModelInfo>,
    pub load: f64,
    pub latency_ms: u32,
    pub price_per_1k_tokens: u64,
    pub max_concurrent: u32,
    pub active_requests: u32,
    pub last_seen_ms: u64,
    pub total_jobs: u64,
    pub avg_tps: f64,
}

/// Info about a specific model on a peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerModelInfo {
    pub name: String,
    pub tier: ModelTier,
    pub loaded: bool,
    pub context_size: u32,
}

// ═══════════════════════════════════════════════════════════════════
// Inference Route
// ═══════════════════════════════════════════════════════════════════

/// A route to a specific inference peer + model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRoute {
    pub peer_id: String,
    pub model_name: String,
    pub model_tier: ModelTier,
    pub total_price_per_1k: u64,
    pub estimated_latency_ms: u32,
    pub estimated_tps: f64,
}

// ═══════════════════════════════════════════════════════════════════
// Remote Inference Request / Response
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteInferenceRequest {
    pub request_id: String,
    pub from_peer: String,
    pub wallet: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub model: String,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteInferenceResponse {
    pub request_id: String,
    pub from_peer: String,
    pub text: String,
    pub tokens_generated: u32,
    pub tokens_per_second: f64,
    pub revenue_micro_qug: u64,
    pub success: bool,
    pub error: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════
// Router Statistics
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributedInferenceStats {
    pub total_requests_routed: u64,
    pub total_requests_completed: u64,
    pub total_requests_failed: u64,
    pub routing_revenue_micro_qug: u64,
    pub known_peers: u32,
    pub active_peers: u32,
    pub avg_latency_ms: f64,
}

// ═══════════════════════════════════════════════════════════════════
// Distributed Inference Router
// ═══════════════════════════════════════════════════════════════════

/// Routes inference requests to the best available peer.
pub struct DistributedInferenceRouter {
    peers: Arc<RwLock<HashMap<String, PeerInferenceInfo>>>,
    routing_markup_pct: u64,
    total_routed: Arc<AtomicU64>,
    total_completed: Arc<AtomicU64>,
    total_failed: Arc<AtomicU64>,
    routing_revenue: Arc<AtomicU64>,
    cumulative_latency_ms: Arc<AtomicU64>,
    inflight: Arc<AtomicU64>,
    our_peer_id: String,
}

impl DistributedInferenceRouter {
    pub fn new(our_peer_id: String) -> Self {
        let markup = std::env::var("INFERENCE_ROUTING_MARKUP_PCT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_ROUTING_MARKUP_PCT);

        info!(
            "🌐 [DIST INFERENCE] Router initialized: peer={}, markup={}%",
            our_peer_id, markup
        );

        Self {
            peers: Arc::new(RwLock::new(HashMap::new())),
            routing_markup_pct: markup,
            total_routed: Arc::new(AtomicU64::new(0)),
            total_completed: Arc::new(AtomicU64::new(0)),
            total_failed: Arc::new(AtomicU64::new(0)),
            routing_revenue: Arc::new(AtomicU64::new(0)),
            cumulative_latency_ms: Arc::new(AtomicU64::new(0)),
            inflight: Arc::new(AtomicU64::new(0)),
            our_peer_id,
        }
    }

    /// Update or insert peer inference capabilities.
    pub fn update_peer(&self, info: PeerInferenceInfo) {
        let peer_id = info.peer_id.clone();
        let model_count = info.models.len();
        let mut peers = self.peers.write();
        peers.insert(peer_id.clone(), info);
        debug!("🌐 [DIST INFERENCE] Peer updated: {} ({} models)", peer_id, model_count);
    }

    pub fn remove_peer(&self, peer_id: &str) {
        let mut peers = self.peers.write();
        if peers.remove(peer_id).is_some() {
            info!("🌐 [DIST INFERENCE] Peer removed: {}", peer_id);
        }
    }

    /// Evict peers that haven't been seen within the TTL.
    pub fn evict_stale_peers(&self) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let ttl_ms = PEER_INFERENCE_TTL_SECS * 1000;
        let mut peers = self.peers.write();
        let before = peers.len();
        peers.retain(|_, info| now_ms.saturating_sub(info.last_seen_ms) < ttl_ms);
        let evicted = before - peers.len();
        if evicted > 0 {
            info!("🌐 [DIST INFERENCE] Evicted {} stale peers ({} remaining)", evicted, peers.len());
        }
    }

    /// Find the best route for a model inference request.
    ///
    /// Selection: model must be loaded, peer must have capacity,
    /// lowest composite score = load * 100 + latency_ms.
    pub fn find_route(&self, model_name: &str) -> Option<InferenceRoute> {
        let peers = self.peers.read();
        let mut best: Option<(f64, InferenceRoute)> = None;

        for info in peers.values() {
            if info.active_requests >= info.max_concurrent {
                continue;
            }
            let model = match info.models.iter().find(|m| m.name == model_name && m.loaded) {
                Some(m) => m,
                None => continue,
            };

            let score = info.load * 100.0 + info.latency_ms as f64;
            let peer_price = info.price_per_1k_tokens;
            let markup = peer_price * self.routing_markup_pct / 100;

            let route = InferenceRoute {
                peer_id: info.peer_id.clone(),
                model_name: model.name.clone(),
                model_tier: model.tier,
                total_price_per_1k: peer_price + markup,
                estimated_latency_ms: info.latency_ms,
                estimated_tps: info.avg_tps,
            };

            match best {
                None => best = Some((score, route)),
                Some((best_score, _)) if score < best_score => best = Some((score, route)),
                _ => {}
            }
        }

        best.map(|(_, route)| route)
    }

    /// Find all routes for a model, sorted by score (best first).
    pub fn find_all_routes(&self, model_name: &str) -> Vec<InferenceRoute> {
        let peers = self.peers.read();
        let mut routes: Vec<(f64, InferenceRoute)> = Vec::new();

        for info in peers.values() {
            if info.active_requests >= info.max_concurrent {
                continue;
            }
            let model = match info.models.iter().find(|m| m.name == model_name && m.loaded) {
                Some(m) => m,
                None => continue,
            };

            let score = info.load * 100.0 + info.latency_ms as f64;
            let peer_price = info.price_per_1k_tokens;
            let markup = peer_price * self.routing_markup_pct / 100;

            routes.push((score, InferenceRoute {
                peer_id: info.peer_id.clone(),
                model_name: model.name.clone(),
                model_tier: model.tier,
                total_price_per_1k: peer_price + markup,
                estimated_latency_ms: info.latency_ms,
                estimated_tps: info.avg_tps,
            }));
        }

        routes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        routes.into_iter().map(|(_, r)| r).collect()
    }

    pub fn record_completion(&self, latency_ms: u64, routing_revenue: u64) {
        self.total_completed.fetch_add(1, Ordering::Relaxed);
        self.cumulative_latency_ms.fetch_add(latency_ms, Ordering::Relaxed);
        self.routing_revenue.fetch_add(routing_revenue, Ordering::Relaxed);
        self.inflight.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn record_failure(&self) {
        self.total_failed.fetch_add(1, Ordering::Relaxed);
        self.inflight.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn record_routed(&self) {
        self.total_routed.fetch_add(1, Ordering::Relaxed);
        self.inflight.fetch_add(1, Ordering::Relaxed);
    }

    pub fn can_route(&self) -> bool {
        (self.inflight.load(Ordering::Relaxed) as usize) < MAX_REMOTE_INFLIGHT
    }

    pub fn peer_count(&self) -> usize {
        self.peers.read().len()
    }

    /// List all loaded models across all peers.
    pub fn available_models(&self) -> Vec<String> {
        let peers = self.peers.read();
        let mut models: Vec<String> = peers
            .values()
            .flat_map(|p| p.models.iter().filter(|m| m.loaded).map(|m| m.name.clone()))
            .collect();
        models.sort();
        models.dedup();
        models
    }

    pub fn stats(&self) -> DistributedInferenceStats {
        let peers = self.peers.read();
        let total_completed = self.total_completed.load(Ordering::Relaxed);
        let cumulative_latency = self.cumulative_latency_ms.load(Ordering::Relaxed);

        let avg_latency = if total_completed > 0 {
            cumulative_latency as f64 / total_completed as f64
        } else {
            0.0
        };

        let active_peers = peers.values().filter(|p| p.models.iter().any(|m| m.loaded)).count();

        DistributedInferenceStats {
            total_requests_routed: self.total_routed.load(Ordering::Relaxed),
            total_requests_completed: total_completed,
            total_requests_failed: self.total_failed.load(Ordering::Relaxed),
            routing_revenue_micro_qug: self.routing_revenue.load(Ordering::Relaxed),
            known_peers: peers.len() as u32,
            active_peers: active_peers as u32,
            avg_latency_ms: avg_latency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_peer(peer_id: &str, model: &str, load: f64, latency_ms: u32) -> PeerInferenceInfo {
        PeerInferenceInfo {
            peer_id: peer_id.to_string(),
            models: vec![PeerModelInfo {
                name: model.to_string(),
                tier: ModelTier::Medium,
                loaded: true,
                context_size: 8192,
            }],
            load,
            latency_ms,
            price_per_1k_tokens: 50,
            max_concurrent: 4,
            active_requests: 0,
            last_seen_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            total_jobs: 100,
            avg_tps: 25.0,
        }
    }

    #[test]
    fn test_router_creation() {
        let router = DistributedInferenceRouter::new("test-peer".to_string());
        assert_eq!(router.peer_count(), 0);
        assert!(router.can_route());
    }

    #[test]
    fn test_add_and_remove_peer() {
        let router = DistributedInferenceRouter::new("router-1".to_string());
        router.update_peer(make_peer("peer-A", "llama-8b", 0.3, 50));
        assert_eq!(router.peer_count(), 1);
        router.remove_peer("peer-A");
        assert_eq!(router.peer_count(), 0);
    }

    #[test]
    fn test_find_route_single_peer() {
        let router = DistributedInferenceRouter::new("router-1".to_string());
        router.update_peer(make_peer("peer-A", "llama-8b", 0.2, 30));
        let route = router.find_route("llama-8b");
        assert!(route.is_some());
        let route = route.unwrap();
        assert_eq!(route.peer_id, "peer-A");
        // Price = 50 + 10% markup = 55
        assert_eq!(route.total_price_per_1k, 55);
    }

    #[test]
    fn test_find_route_no_matching_model() {
        let router = DistributedInferenceRouter::new("router-1".to_string());
        router.update_peer(make_peer("peer-A", "llama-8b", 0.2, 30));
        assert!(router.find_route("gpt-4").is_none());
    }

    #[test]
    fn test_find_route_picks_best_peer() {
        let router = DistributedInferenceRouter::new("router-1".to_string());
        // Score A = 0.1*100 + 200 = 210
        router.update_peer(make_peer("peer-A", "llama-8b", 0.1, 200));
        // Score B = 0.3*100 + 20 = 50 (best)
        router.update_peer(make_peer("peer-B", "llama-8b", 0.3, 20));
        // Score C = 0.9*100 + 10 = 100
        router.update_peer(make_peer("peer-C", "llama-8b", 0.9, 10));
        let route = router.find_route("llama-8b").unwrap();
        assert_eq!(route.peer_id, "peer-B");
    }

    #[test]
    fn test_find_route_skips_overloaded() {
        let router = DistributedInferenceRouter::new("router-1".to_string());
        let mut overloaded = make_peer("peer-A", "llama-8b", 0.1, 10);
        overloaded.active_requests = overloaded.max_concurrent;
        router.update_peer(overloaded);
        router.update_peer(make_peer("peer-B", "llama-8b", 0.5, 50));
        let route = router.find_route("llama-8b").unwrap();
        assert_eq!(route.peer_id, "peer-B");
    }

    #[test]
    fn test_find_all_routes_sorted() {
        let router = DistributedInferenceRouter::new("router-1".to_string());
        router.update_peer(make_peer("peer-A", "llama-8b", 0.5, 100));
        router.update_peer(make_peer("peer-B", "llama-8b", 0.1, 20));
        router.update_peer(make_peer("peer-C", "llama-8b", 0.3, 50));
        let routes = router.find_all_routes("llama-8b");
        assert_eq!(routes.len(), 3);
        assert_eq!(routes[0].peer_id, "peer-B");
        assert_eq!(routes[1].peer_id, "peer-C");
        assert_eq!(routes[2].peer_id, "peer-A");
    }

    #[test]
    fn test_available_models() {
        let router = DistributedInferenceRouter::new("router-1".to_string());
        router.update_peer(make_peer("peer-A", "llama-8b", 0.2, 30));
        router.update_peer(make_peer("peer-B", "glm-9b", 0.3, 40));
        let mut peer_c = make_peer("peer-C", "llama-8b", 0.1, 20);
        peer_c.models.push(PeerModelInfo {
            name: "mistral-7b".to_string(),
            tier: ModelTier::Medium,
            loaded: true,
            context_size: 4096,
        });
        router.update_peer(peer_c);
        let models = router.available_models();
        assert_eq!(models, vec!["glm-9b", "llama-8b", "mistral-7b"]);
    }

    #[test]
    fn test_record_completion_updates_stats() {
        let router = DistributedInferenceRouter::new("router-1".to_string());
        router.record_routed();
        router.record_completion(50, 100);
        let stats = router.stats();
        assert_eq!(stats.total_requests_routed, 1);
        assert_eq!(stats.total_requests_completed, 1);
        assert_eq!(stats.routing_revenue_micro_qug, 100);
        assert_eq!(stats.avg_latency_ms, 50.0);
    }

    #[test]
    fn test_record_failure_updates_stats() {
        let router = DistributedInferenceRouter::new("router-1".to_string());
        router.record_routed();
        router.record_failure();
        let stats = router.stats();
        assert_eq!(stats.total_requests_failed, 1);
    }

    #[test]
    fn test_evict_stale_peers() {
        let router = DistributedInferenceRouter::new("router-1".to_string());
        let mut stale = make_peer("stale-peer", "llama-8b", 0.1, 20);
        stale.last_seen_ms = 0;
        router.update_peer(stale);
        router.update_peer(make_peer("fresh-peer", "llama-8b", 0.2, 30));
        assert_eq!(router.peer_count(), 2);
        router.evict_stale_peers();
        assert_eq!(router.peer_count(), 1);
    }

    #[test]
    fn test_stats_empty_router() {
        let router = DistributedInferenceRouter::new("empty".to_string());
        let stats = router.stats();
        assert_eq!(stats.total_requests_routed, 0);
        assert_eq!(stats.known_peers, 0);
        assert_eq!(stats.avg_latency_ms, 0.0);
    }

    #[test]
    fn test_can_route_inflight_limit() {
        let router = DistributedInferenceRouter::new("router-1".to_string());
        assert!(router.can_route());
        for _ in 0..MAX_REMOTE_INFLIGHT {
            router.record_routed();
        }
        assert!(!router.can_route());
        router.record_completion(10, 5);
        assert!(router.can_route());
    }

    #[test]
    fn test_unloaded_model_not_routable() {
        let router = DistributedInferenceRouter::new("router-1".to_string());
        let mut peer = make_peer("peer-A", "llama-8b", 0.1, 20);
        peer.models[0].loaded = false;
        router.update_peer(peer);
        assert!(router.find_route("llama-8b").is_none());
    }
}
