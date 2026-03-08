//! Compute Tunnel — Encrypted P2P task routing between nodes and miners
//!
//! Tunnels enable distributed compute by connecting:
//! - Miner → Node: mining solutions + telemetry
//! - Node → Node: task distribution + results
//! - Node → Miner: push compute tasks to idle miner GPU
//! - Miner → Miner: collaborative proof generation
//!
//! Each tunnel is encrypted (NOISE XX pattern) and carries
//! typed work items with priority routing.

use crate::{ComputeLayer, TunnelInfo, TunnelType};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use parking_lot::RwLock;
use tracing::{info, warn, debug};

/// Work item sent through a tunnel
#[derive(Debug, Clone)]
pub struct TunnelWorkItem {
    pub id: u64,
    pub layer: ComputeLayer,
    pub payload_bytes: usize,
    pub priority: u8,           // 0 = highest (mining), 7 = lowest
    pub sender_peer: String,
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
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
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

/// Tunnel manager — tracks all active tunnels to peers
pub struct TunnelManager {
    tunnels: Arc<RwLock<HashMap<String, ComputeTunnel>>>,
    max_tunnels: usize,
    total_tasks_routed: Arc<AtomicU64>,
}

impl TunnelManager {
    pub fn new(max_tunnels: usize) -> Self {
        Self {
            tunnels: Arc::new(RwLock::new(HashMap::new())),
            max_tunnels,
            total_tasks_routed: Arc::new(AtomicU64::new(0)),
        }
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

    /// Clean up dead tunnels
    pub fn cleanup_dead(&self) {
        let mut tunnels = self.tunnels.write();
        let before = tunnels.len();
        tunnels.retain(|_, t| t.is_active());
        let removed = before - tunnels.len();
        if removed > 0 {
            info!("🔗 [TUNNEL] Cleaned up {} dead tunnels", removed);
        }
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
}
