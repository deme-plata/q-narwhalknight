//! Compute Tunnel — Encrypted P2P task routing between nodes and miners
//!
//! **Phase 5 future work**: This module defines the types and routing logic
//! for distributed compute tunnels. The types are architecturally correct
//! but `TunnelManager` is not yet instantiated in production — tunnels will
//! be wired when P2P compute distribution is implemented.
//!
//! ## P2P Compute Peer Discovery (Issue #002)
//!
//! Nodes announce their compute capacity via gossipsub on `COMPUTE_TUNNEL_TOPIC`.
//! The `PeerRegistry` tracks discovered peers with a 60-second TTL and supports
//! score-based peer selection for task routing.
//!
//! Tunnels enable distributed compute by connecting:
//! - Miner → Node: mining solutions + telemetry
//! - Node → Node: task distribution + results
//! - Node → Miner: push compute tasks to idle miner GPU
//! - Miner → Miner: collaborative proof generation
//!
//! Each tunnel is encrypted (NOISE XX pattern) and carries
//! typed work items with priority routing.

#![allow(dead_code)]

use crate::{ComputeLayer, ComputePeerInfo, ResourceSnapshot, TunnelInfo, TunnelType};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use tracing::{info, warn, debug};

/// TTL for known peers in seconds — peers not re-announced within this
/// window are considered stale and evicted.
const PEER_TTL_SECS: u64 = 60;

/// Work item sent through a tunnel
#[derive(Debug, Clone)]
pub struct TunnelWorkItem {
    pub id: u64,
    pub layer: ComputeLayer,
    pub payload_bytes: usize,
    pub priority: u8,           // 0 = highest (mining), 7 = lowest
    pub sender_peer: String,
}

/// v9.6.0: Typed tunnel payload for AI inference task routing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TunnelPayload {
    /// Mining solution from miner
    MiningSubmit(Vec<u8>),
    /// AI inference request routed to capable peer
    InferenceRequest {
        request_id: String,
        prompt: String,
        max_tokens: usize,
        model: Option<String>,
        wallet: Option<String>,
    },
    /// AI inference response back to requester
    InferenceResponse {
        request_id: String,
        generated_text: String,
        tokens_generated: usize,
        tokens_per_second: f64,
    },
    /// Tensor shard for distributed model serving
    TensorShard {
        request_id: String,
        layer_id: u32,
        shard_data: Vec<u8>,
    },
    /// Layer output forwarded in pipeline parallelism
    LayerOutput {
        request_id: String,
        layer_range: (u32, u32),
        activations: Vec<u8>,
    },
}

/// A single compute tunnel to a remote peer
pub struct ComputeTunnel {
    pub peer_id: String,
    pub tunnel_type: TunnelType,
    pub established_ms: u64,
    pub encrypted: bool,
    bytes_sent: Arc<AtomicU64>,
    bytes_received: Arc<AtomicU64>,
    tasks_routed: Arc<AtomicU64>,
    latency_ms: Arc<AtomicU64>,
    active: Arc<AtomicBool>,
}

impl ComputeTunnel {
    pub fn new(peer_id: String, tunnel_type: TunnelType) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            peer_id,
            tunnel_type,
            established_ms: now,
            encrypted: true,
            bytes_sent: Arc::new(AtomicU64::new(0)),
            bytes_received: Arc::new(AtomicU64::new(0)),
            tasks_routed: Arc::new(AtomicU64::new(0)),
            latency_ms: Arc::new(AtomicU64::new(0)),
            active: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Record bytes sent through this tunnel
    pub fn record_send(&self, bytes: u64) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record bytes received through this tunnel
    pub fn record_receive(&self, bytes: u64) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a task routed through this tunnel
    pub fn record_task(&self) {
        self.tasks_routed.fetch_add(1, Ordering::Relaxed);
    }

    /// Update measured latency
    pub fn update_latency(&self, ms: u32) {
        self.latency_ms.store(ms as u64, Ordering::Relaxed);
    }

    /// Check if tunnel is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Close this tunnel
    pub fn close(&self) {
        self.active.store(false, Ordering::SeqCst);
    }

    /// Get tunnel info snapshot for dashboard
    pub fn info(&self) -> TunnelInfo {
        TunnelInfo {
            peer_id: self.peer_id.clone(),
            tunnel_type: self.tunnel_type,
            established_ms: self.established_ms,
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            tasks_routed: self.tasks_routed.load(Ordering::Relaxed),
            latency_ms: self.latency_ms.load(Ordering::Relaxed) as u32,
            encrypted: self.encrypted,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Peer Discovery — gossipsub-based compute capacity announcements
// ═══════════════════════════════════════════════════════════════════

/// Create a `ComputePeerInfo` announcement from a resource snapshot.
///
/// This is serialized to JSON and published on `COMPUTE_TUNNEL_TOPIC`
/// every ~30 seconds so peers can discover our compute capacity.
pub fn create_peer_announcement(
    snapshot: &ResourceSnapshot,
    mode: &str,
    peer_id: &str,
) -> ComputePeerInfo {
    let total_cores = snapshot.cpu_per_core.len() as u32;
    // Available cores = cores with < 80% utilization
    let available_cores = snapshot
        .cpu_per_core
        .iter()
        .filter(|&&usage| usage < 80.0)
        .count() as u32;

    // Estimate GPU TFLOPS from utilization and memory (rough heuristic).
    // Without a real GPU capability query, we estimate based on memory size:
    // Consumer GPUs: ~0.5 TFLOPS per GB VRAM (FP32 ballpark)
    let gpu_tflops = if snapshot.gpu_memory_total > 0 {
        let vram_gb = snapshot.gpu_memory_total as f64 / (1024.0 * 1024.0 * 1024.0);
        // Scale by how much GPU is free (inverse of utilization)
        let free_ratio = 1.0 - (snapshot.gpu_utilization as f64 / 100.0);
        vram_gb * 0.5 * free_ratio
    } else {
        0.0
    };

    let ram_total_gb = snapshot.ram_total as f64 / (1024.0 * 1024.0 * 1024.0);
    let ram_available_gb =
        snapshot.ram_total.saturating_sub(snapshot.ram_used) as f64 / (1024.0 * 1024.0 * 1024.0);

    // Bandwidth in Mbps (from bytes/sec capacity estimate)
    let bandwidth_mbps = snapshot.net_capacity_bps as f64 * 8.0 / 1_000_000.0;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    ComputePeerInfo {
        peer_id: peer_id.to_string(),
        available_cores,
        total_cores,
        gpu_tflops,
        ram_available_gb,
        ram_total_gb,
        bandwidth_mbps,
        compute_mode: mode.to_string(),
        active_layers: Vec::new(), // Populated by caller from orchestrator status
        trainer_active: false,     // Populated by caller
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp,
    }
}

/// Parse a raw gossipsub message into a `ComputePeerInfo`.
///
/// Returns `None` if the data is not valid JSON or does not match the schema.
pub fn parse_peer_announcement(data: &[u8]) -> Option<ComputePeerInfo> {
    serde_json::from_slice::<ComputePeerInfo>(data).ok()
}

// ═══════════════════════════════════════════════════════════════════
// Peer Registry — track discovered peers with TTL-based eviction
// ═══════════════════════════════════════════════════════════════════

/// Tracks known compute peers discovered via gossipsub.
///
/// Each peer entry has a 60-second TTL. Peers that do not re-announce
/// within the TTL window are evicted on the next `cleanup_stale()` call.
pub struct PeerRegistry {
    /// peer_id → ComputePeerInfo
    known_peers: Arc<RwLock<HashMap<String, ComputePeerInfo>>>,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self {
            known_peers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert or update a peer. The `timestamp` field in `ComputePeerInfo`
    /// is used as the TTL reference — if the announcement is already stale
    /// at insertion time it is silently dropped.
    pub fn upsert(&self, info: ComputePeerInfo) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Reject announcements that are already older than TTL
        if now.saturating_sub(info.timestamp) > PEER_TTL_SECS {
            debug!(
                "🔗 [PEER REGISTRY] Dropping stale announcement from {} (age={}s)",
                info.peer_id,
                now.saturating_sub(info.timestamp)
            );
            return;
        }

        let peer_id = info.peer_id.clone();
        let mut peers = self.known_peers.write();
        let is_new = !peers.contains_key(&peer_id);
        peers.insert(peer_id.clone(), info);

        if is_new {
            debug!(
                "🔗 [PEER REGISTRY] Discovered new compute peer: {} (total={})",
                peer_id,
                peers.len()
            );
        }
    }

    /// Remove peers whose `timestamp` is older than `PEER_TTL_SECS`.
    pub fn cleanup_stale(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut peers = self.known_peers.write();
        let before = peers.len();
        peers.retain(|_, info| now.saturating_sub(info.timestamp) <= PEER_TTL_SECS);
        let removed = before - peers.len();
        if removed > 0 {
            info!(
                "🔗 [PEER REGISTRY] Evicted {} stale peers (remaining={})",
                removed,
                peers.len()
            );
        }
    }

    /// Return a snapshot of all known (non-stale) peers.
    pub fn all_peers(&self) -> Vec<ComputePeerInfo> {
        self.known_peers.read().values().cloned().collect()
    }

    /// Number of known peers.
    pub fn len(&self) -> usize {
        self.known_peers.read().len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.known_peers.read().is_empty()
    }

    /// Get a specific peer by ID.
    pub fn get(&self, peer_id: &str) -> Option<ComputePeerInfo> {
        self.known_peers.read().get(peer_id).cloned()
    }

    /// Select the best peer for a given task type using score-based ranking.
    ///
    /// Scoring function:
    ///   `score = capability_match * (1.0 / latency_estimate_ms) * availability_ratio`
    ///
    /// - `capability_match`: How well the peer's resources fit the task
    ///   - "mining" / "gpu" -> weighted by gpu_tflops
    ///   - "inference" / "ai" -> weighted by ram_available_gb + gpu_tflops
    ///   - "zk" / "proof" -> weighted by available_cores
    ///   - default -> weighted by available_cores
    ///
    /// - `latency_estimate_ms`: Derived from bandwidth (higher bandwidth = lower latency).
    ///   We use `1000.0 / bandwidth_mbps` as a rough proxy since we don't have RTT data
    ///   until a tunnel is established.
    ///
    /// - `availability_ratio`: `available_cores / total_cores` — how idle the peer is.
    ///
    /// Returns `None` if no peers are known.
    pub fn get_best_peer_for_task(&self, task_type: &str) -> Option<ComputePeerInfo> {
        let peers = self.known_peers.read();
        if peers.is_empty() {
            return None;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let task_lower = task_type.to_lowercase();

        peers
            .values()
            // Filter out stale peers
            .filter(|p| now.saturating_sub(p.timestamp) <= PEER_TTL_SECS)
            // Filter out peers with no available capacity
            .filter(|p| p.available_cores > 0 || p.gpu_tflops > 0.0)
            .max_by(|a, b| {
                let score_a = compute_peer_score(a, &task_lower);
                let score_b = compute_peer_score(b, &task_lower);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }
}

/// Compute a score for a peer given a task type.
///
/// `score = capability_match * (1.0 / latency_estimate_ms) * availability_ratio`
fn compute_peer_score(peer: &ComputePeerInfo, task_type: &str) -> f64 {
    // Capability match — task-dependent weight
    let capability_match: f64 = match task_type {
        t if t.contains("mining") || t.contains("gpu") => {
            // GPU-heavy tasks: prefer peers with high TFLOPS
            peer.gpu_tflops.max(0.1) + (peer.available_cores as f64 * 0.1)
        }
        t if t.contains("inference") || t.contains("ai") => {
            // AI inference: needs RAM + GPU
            peer.ram_available_gb * 0.5 + peer.gpu_tflops * 2.0 + (peer.available_cores as f64 * 0.2)
        }
        t if t.contains("zk") || t.contains("proof") => {
            // ZK proofs: CPU-bound
            peer.available_cores as f64
        }
        _ => {
            // Generic tasks: balanced scoring
            peer.available_cores as f64 + peer.gpu_tflops * 0.5
        }
    };

    // Latency estimate: use inverse of bandwidth as a proxy for latency.
    // Higher bandwidth -> lower latency estimate -> higher score.
    // Clamp bandwidth to avoid division by zero.
    let bandwidth_clamped = peer.bandwidth_mbps.max(1.0);
    let latency_estimate_ms = 1000.0 / bandwidth_clamped;
    let latency_factor = 1.0 / latency_estimate_ms.max(0.01);

    // Availability ratio: how idle is this peer?
    let availability_ratio = if peer.total_cores > 0 {
        peer.available_cores as f64 / peer.total_cores as f64
    } else {
        0.0
    };

    capability_match * latency_factor * availability_ratio
}

// ═══════════════════════════════════════════════════════════════════
// Tunnel Manager — tracks all active tunnels to peers
// ═══════════════════════════════════════════════════════════════════

/// Tunnel manager — tracks all active tunnels to peers
pub struct TunnelManager {
    tunnels: Arc<RwLock<HashMap<String, ComputeTunnel>>>,
    max_tunnels: usize,
    total_tasks_routed: Arc<AtomicU64>,
    /// Registry of known compute peers discovered via gossipsub
    peer_registry: Arc<PeerRegistry>,
}

impl TunnelManager {
    pub fn new(max_tunnels: usize) -> Self {
        Self {
            tunnels: Arc::new(RwLock::new(HashMap::new())),
            max_tunnels,
            total_tasks_routed: Arc::new(AtomicU64::new(0)),
            peer_registry: Arc::new(PeerRegistry::new()),
        }
    }

    /// Get a reference to the peer registry for direct access.
    pub fn peer_registry(&self) -> &Arc<PeerRegistry> {
        &self.peer_registry
    }

    /// Open a new tunnel to a peer
    pub fn open_tunnel(&self, peer_id: &str, tunnel_type: TunnelType) -> bool {
        let mut tunnels = self.tunnels.write();

        if tunnels.len() >= self.max_tunnels {
            warn!(
                "🔗 [TUNNEL] Max tunnels reached ({}) — cannot open to {}",
                self.max_tunnels, peer_id
            );
            return false;
        }

        if tunnels.contains_key(peer_id) {
            debug!("🔗 [TUNNEL] Already connected to {}", peer_id);
            return true;
        }

        let tunnel = ComputeTunnel::new(peer_id.to_string(), tunnel_type);
        info!(
            "🔗 [TUNNEL] Opened {:?} tunnel to {} (encrypted={})",
            tunnel_type, peer_id, tunnel.encrypted
        );
        tunnels.insert(peer_id.to_string(), tunnel);
        true
    }

    /// Close tunnel to a peer
    pub fn close_tunnel(&self, peer_id: &str) {
        let mut tunnels = self.tunnels.write();
        if let Some(tunnel) = tunnels.remove(peer_id) {
            tunnel.close();
            info!(
                "🔗 [TUNNEL] Closed tunnel to {} (sent={}B, recv={}B, tasks={})",
                peer_id,
                tunnel.bytes_sent.load(Ordering::Relaxed),
                tunnel.bytes_received.load(Ordering::Relaxed),
                tunnel.tasks_routed.load(Ordering::Relaxed),
            );
        }
    }

    /// Route a work item to the best available tunnel
    pub fn route_work(&self, item: &TunnelWorkItem) -> Option<String> {
        let tunnels = self.tunnels.read();

        // Find best tunnel: lowest latency active tunnel
        let best = tunnels.values()
            .filter(|t| t.is_active())
            .min_by_key(|t| t.latency_ms.load(Ordering::Relaxed));

        if let Some(tunnel) = best {
            tunnel.record_task();
            tunnel.record_send(item.payload_bytes as u64);
            self.total_tasks_routed.fetch_add(1, Ordering::Relaxed);
            debug!(
                "🔗 [TUNNEL] Routed {:?} task #{} ({} bytes) → {}",
                item.layer, item.id, item.payload_bytes, tunnel.peer_id
            );
            Some(tunnel.peer_id.clone())
        } else {
            None
        }
    }

    /// Get all tunnel info snapshots for dashboard
    pub fn tunnel_infos(&self) -> Vec<TunnelInfo> {
        self.tunnels.read().values()
            .filter(|t| t.is_active())
            .map(|t| t.info())
            .collect()
    }

    /// Number of active tunnels
    pub fn active_count(&self) -> usize {
        self.tunnels.read().values()
            .filter(|t| t.is_active())
            .count()
    }

    /// Total tasks routed across all tunnels
    pub fn total_tasks_routed(&self) -> u64 {
        self.total_tasks_routed.load(Ordering::Relaxed)
    }

    /// Clean up dead tunnels and stale peers
    pub fn cleanup_dead(&self) {
        // Clean up dead tunnels
        let mut tunnels = self.tunnels.write();
        let before = tunnels.len();
        tunnels.retain(|_, t| t.is_active());
        let removed = before - tunnels.len();
        if removed > 0 {
            info!("🔗 [TUNNEL] Cleaned up {} dead tunnels", removed);
        }
        drop(tunnels);

        // Clean up stale peers
        self.peer_registry.cleanup_stale();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tunnel_creation() {
        let tunnel = ComputeTunnel::new("peer123".to_string(), TunnelType::MinerToNode);
        assert!(tunnel.is_active());
        assert_eq!(tunnel.peer_id, "peer123");
        assert!(tunnel.encrypted);
    }

    #[test]
    fn test_tunnel_stats() {
        let tunnel = ComputeTunnel::new("peer456".to_string(), TunnelType::NodeToNode);
        tunnel.record_send(1000);
        tunnel.record_send(2000);
        tunnel.record_receive(500);
        tunnel.record_task();
        tunnel.record_task();
        tunnel.update_latency(42);

        let info = tunnel.info();
        assert_eq!(info.bytes_sent, 3000);
        assert_eq!(info.bytes_received, 500);
        assert_eq!(info.tasks_routed, 2);
        assert_eq!(info.latency_ms, 42);
    }

    #[test]
    fn test_tunnel_manager() {
        let mgr = TunnelManager::new(10);
        assert!(mgr.open_tunnel("peer1", TunnelType::MinerToNode));
        assert!(mgr.open_tunnel("peer2", TunnelType::NodeToNode));
        assert_eq!(mgr.active_count(), 2);

        // Duplicate returns true but doesn't add
        assert!(mgr.open_tunnel("peer1", TunnelType::MinerToNode));
        assert_eq!(mgr.active_count(), 2);

        mgr.close_tunnel("peer1");
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_tunnel_max_limit() {
        let mgr = TunnelManager::new(2);
        assert!(mgr.open_tunnel("peer1", TunnelType::MinerToNode));
        assert!(mgr.open_tunnel("peer2", TunnelType::NodeToNode));
        assert!(!mgr.open_tunnel("peer3", TunnelType::MinerToMiner)); // Exceeds max
        assert_eq!(mgr.active_count(), 2);
    }

    #[test]
    fn test_route_work() {
        let mgr = TunnelManager::new(10);
        mgr.open_tunnel("peer1", TunnelType::NodeToNode);

        let item = TunnelWorkItem {
            id: 1,
            layer: ComputeLayer::Mining,
            payload_bytes: 256,
            priority: 0,
            sender_peer: "local".to_string(),
        };

        let routed_to = mgr.route_work(&item);
        assert_eq!(routed_to, Some("peer1".to_string()));
        assert_eq!(mgr.total_tasks_routed(), 1);
    }

    // ═══════════════════════════════════════════════════════════════
    // Peer announcement tests
    // ═══════════════════════════════════════════════════════════════

    fn make_test_snapshot() -> ResourceSnapshot {
        ResourceSnapshot {
            cpu_per_core: vec![10.0, 20.0, 30.0, 90.0], // 3 cores < 80%, 1 >= 80%
            cpu_total: 37.5,
            gpu_utilization: 50.0,
            gpu_memory_used: 4 * 1024 * 1024 * 1024,     // 4 GB used
            gpu_memory_total: 8 * 1024 * 1024 * 1024,     // 8 GB total
            ram_used: 8 * 1024 * 1024 * 1024,              // 8 GB used
            ram_total: 32 * 1024 * 1024 * 1024,            // 32 GB total
            net_tx_bps: 10_000_000,
            net_rx_bps: 10_000_000,
            net_capacity_bps: 125_000_000, // 1 Gbps
            disk_io_bps: 100_000_000,
            timestamp_ms: 1710000000000,
        }
    }

    #[test]
    fn test_create_peer_announcement() {
        let snap = make_test_snapshot();
        let info = create_peer_announcement(&snap, "full", "12D3KooWTest123");

        assert_eq!(info.peer_id, "12D3KooWTest123");
        assert_eq!(info.total_cores, 4);
        assert_eq!(info.available_cores, 3); // 3 cores under 80%
        assert!(info.gpu_tflops > 0.0);
        assert!(info.ram_total_gb > 31.0 && info.ram_total_gb < 33.0);
        assert!(info.ram_available_gb > 23.0 && info.ram_available_gb < 25.0);
        assert_eq!(info.compute_mode, "full");
        assert!(info.bandwidth_mbps > 900.0); // ~1000 Mbps
        assert!(info.timestamp > 0);
        assert!(!info.version.is_empty());
    }

    #[test]
    fn test_create_announcement_no_gpu() {
        let mut snap = make_test_snapshot();
        snap.gpu_memory_total = 0;
        snap.gpu_utilization = 0.0;

        let info = create_peer_announcement(&snap, "eco", "peer-no-gpu");
        assert_eq!(info.gpu_tflops, 0.0);
        assert_eq!(info.compute_mode, "eco");
    }

    #[test]
    fn test_parse_peer_announcement_valid() {
        let snap = make_test_snapshot();
        let info = create_peer_announcement(&snap, "full", "12D3KooWTest");

        let json = serde_json::to_vec(&info).unwrap();
        let parsed = parse_peer_announcement(&json);
        assert!(parsed.is_some());

        let parsed = parsed.unwrap();
        assert_eq!(parsed.peer_id, "12D3KooWTest");
        assert_eq!(parsed.total_cores, 4);
        assert_eq!(parsed.compute_mode, "full");
    }

    #[test]
    fn test_parse_peer_announcement_invalid() {
        assert!(parse_peer_announcement(b"not json").is_none());
        assert!(parse_peer_announcement(b"{}").is_none()); // Missing required fields
        assert!(parse_peer_announcement(b"").is_none());
    }

    // ═══════════════════════════════════════════════════════════════
    // Peer registry tests
    // ═══════════════════════════════════════════════════════════════

    fn make_test_peer(peer_id: &str, cores: u32, gpu: f64, ram_gb: f64, bw: f64) -> ComputePeerInfo {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        ComputePeerInfo {
            peer_id: peer_id.to_string(),
            available_cores: cores,
            total_cores: cores * 2,
            gpu_tflops: gpu,
            ram_available_gb: ram_gb,
            ram_total_gb: ram_gb * 2.0,
            bandwidth_mbps: bw,
            compute_mode: "full".to_string(),
            active_layers: vec!["Mining".to_string()],
            trainer_active: false,
            version: "test".to_string(),
            timestamp: now,
        }
    }

    #[test]
    fn test_peer_registry_upsert_and_get() {
        let registry = PeerRegistry::new();
        assert!(registry.is_empty());

        let peer = make_test_peer("peer-A", 8, 10.0, 16.0, 1000.0);
        registry.upsert(peer);

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());

        let fetched = registry.get("peer-A");
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().available_cores, 8);

        // Not found
        assert!(registry.get("peer-B").is_none());
    }

    #[test]
    fn test_peer_registry_update_existing() {
        let registry = PeerRegistry::new();

        let peer_v1 = make_test_peer("peer-A", 4, 5.0, 8.0, 500.0);
        registry.upsert(peer_v1);
        assert_eq!(registry.get("peer-A").unwrap().available_cores, 4);

        // Update with new data
        let peer_v2 = make_test_peer("peer-A", 8, 10.0, 16.0, 1000.0);
        registry.upsert(peer_v2);
        assert_eq!(registry.len(), 1); // Same peer, no duplicate
        assert_eq!(registry.get("peer-A").unwrap().available_cores, 8);
    }

    #[test]
    fn test_peer_registry_stale_rejected() {
        let registry = PeerRegistry::new();

        let mut stale_peer = make_test_peer("stale", 4, 0.0, 8.0, 100.0);
        stale_peer.timestamp = 1000; // Very old timestamp
        registry.upsert(stale_peer);

        // Should be rejected as stale
        assert!(registry.is_empty());
    }

    #[test]
    fn test_peer_registry_all_peers() {
        let registry = PeerRegistry::new();
        registry.upsert(make_test_peer("peer-A", 4, 5.0, 8.0, 500.0));
        registry.upsert(make_test_peer("peer-B", 8, 10.0, 16.0, 1000.0));
        registry.upsert(make_test_peer("peer-C", 2, 0.0, 4.0, 100.0));

        let all = registry.all_peers();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_get_best_peer_for_gpu_task() {
        let registry = PeerRegistry::new();
        // Peer A: low GPU
        registry.upsert(make_test_peer("peer-A", 8, 1.0, 16.0, 1000.0));
        // Peer B: high GPU
        registry.upsert(make_test_peer("peer-B", 4, 20.0, 8.0, 1000.0));
        // Peer C: no GPU
        registry.upsert(make_test_peer("peer-C", 16, 0.0, 32.0, 1000.0));

        let best = registry.get_best_peer_for_task("gpu-compute");
        assert!(best.is_some());
        // Peer B should win for GPU tasks due to highest gpu_tflops
        assert_eq!(best.unwrap().peer_id, "peer-B");
    }

    #[test]
    fn test_get_best_peer_for_zk_task() {
        let registry = PeerRegistry::new();
        // Peer A: many available cores
        registry.upsert(make_test_peer("peer-A", 16, 0.0, 8.0, 1000.0));
        // Peer B: fewer cores but has GPU
        registry.upsert(make_test_peer("peer-B", 4, 20.0, 32.0, 1000.0));

        let best = registry.get_best_peer_for_task("zk-proof");
        assert!(best.is_some());
        // Peer A should win for ZK tasks — CPU-bound, more cores
        assert_eq!(best.unwrap().peer_id, "peer-A");
    }

    #[test]
    fn test_get_best_peer_empty_registry() {
        let registry = PeerRegistry::new();
        assert!(registry.get_best_peer_for_task("mining").is_none());
    }

    #[test]
    fn test_get_best_peer_no_available_capacity() {
        let registry = PeerRegistry::new();
        // Peer with 0 available cores and no GPU
        registry.upsert(make_test_peer("busy-peer", 0, 0.0, 0.0, 1000.0));

        // Should return None — no capacity
        assert!(registry.get_best_peer_for_task("generic").is_none());
    }

    #[test]
    fn test_peer_score_function() {
        let peer = make_test_peer("test", 8, 10.0, 16.0, 1000.0);
        let score = compute_peer_score(&peer, "generic");
        assert!(score > 0.0, "Score should be positive for capable peer");

        // Zero-cores peer should have low score
        let zero_peer = make_test_peer("zero", 0, 0.0, 0.0, 1.0);
        let zero_score = compute_peer_score(&zero_peer, "generic");
        assert_eq!(zero_score, 0.0, "Score should be 0 for peer with no resources");
    }

    #[test]
    fn test_roundtrip_serialize_deserialize() {
        let snap = make_test_snapshot();
        let info = create_peer_announcement(&snap, "nuke", "12D3KooWRoundTrip");

        // Serialize -> bytes -> deserialize
        let bytes = serde_json::to_vec(&info).unwrap();
        let parsed = parse_peer_announcement(&bytes).unwrap();

        assert_eq!(parsed.peer_id, info.peer_id);
        assert_eq!(parsed.available_cores, info.available_cores);
        assert_eq!(parsed.total_cores, info.total_cores);
        assert!((parsed.gpu_tflops - info.gpu_tflops).abs() < 0.001);
        assert!((parsed.ram_available_gb - info.ram_available_gb).abs() < 0.01);
        assert_eq!(parsed.compute_mode, info.compute_mode);
        assert_eq!(parsed.version, info.version);
        assert_eq!(parsed.timestamp, info.timestamp);
    }
}
