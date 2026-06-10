//! Compute Orchestrator — Adaptive 8-layer priority scheduler
//!
//! Monitors resource utilization every 100ms and assigns work
//! to idle CPU/GPU/RAM. Mining (Layer 0) always has priority.
//! Lower layers fill the gaps.

use crate::{ComputeLayer, ComputeMode, ComputePeerInfo, ComputeStatus, LayerStats, AtomicU64Ser};
use crate::core_enforcer::CoreEnforcer;
use crate::resource_monitor::ResourceMonitor;
use crate::trainer::Trainer;
use crate::os_tuner::OsTuner;
use crate::inference_pool::InferenceWorkerPool;
use crate::tunnel::{PeerRegistry, TunnelManager, create_peer_announcement, parse_peer_announcement};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use parking_lot::RwLock;
use tracing::{info, debug, trace, warn};

/// Concrete core range assigned to a layer (start..end)
#[derive(Debug, Clone, PartialEq)]
pub struct CoreRange {
    pub start: usize,
    pub end: usize, // exclusive
}

impl CoreRange {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Number of cores in this range
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to core_affinity CoreId set
    pub fn to_core_ids(&self) -> Vec<core_affinity::CoreId> {
        (self.start..self.end)
            .map(|id| core_affinity::CoreId { id })
            .collect()
    }

    /// #013: Convert to a comma-separated CPU list for cgroup cpuset (e.g. "0,1,2,3")
    pub fn to_cpuset_str(&self) -> String {
        (self.start..self.end)
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// #013: Convert to a Vec of core indices
    pub fn to_core_vec(&self) -> Vec<usize> {
        (self.start..self.end).collect()
    }
}

/// Core assignment for each layer
#[derive(Debug, Clone)]
struct LayerAssignment {
    layer: ComputeLayer,
    /// Advisory core budget (count, not specific cores — OS handles actual scheduling)
    core_budget: usize,
    /// v9.5.1: Concrete core range for enforcement (#013)
    core_range: Option<CoreRange>,
    active: bool,
    tasks_completed: Arc<AtomicU64>,
    tasks_pending: Arc<AtomicU64>,
    revenue_micro_qug: Arc<AtomicU64>,
    /// Tick counter for feedback loop — last tick where tasks_completed increased
    last_active_tick: u64,
    /// Snapshot of tasks_completed at last activity check
    prev_tasks_completed: u64,
}

/// The Compute Orchestrator — brain of Starship Endgame
pub struct Orchestrator {
    mode: Arc<RwLock<ComputeMode>>,
    monitor: Arc<ResourceMonitor>,
    trainer: Arc<Trainer>,
    assignments: Arc<RwLock<HashMap<ComputeLayer, LayerAssignment>>>,
    total_cores: usize,
    mining_cores: Arc<AtomicU64>,      // Cores reserved for mining
    running: Arc<AtomicBool>,
    /// v9.6.0: AI inference worker pool (runs inference on idle cores)
    inference_pool: Arc<InferenceWorkerPool>,
    /// v9.5.0: Tunnel manager + peer registry for P2P compute (Issue #002)
    tunnel_manager: Arc<TunnelManager>,
    /// Local peer ID (set after P2P identity is available)
    local_peer_id: Arc<RwLock<String>>,
    /// v9.7.0: Real CPU affinity enforcement via sched_setaffinity (#013)
    core_enforcer: Arc<RwLock<CoreEnforcer>>,
}

impl Orchestrator {
    /// Create a new orchestrator
    pub fn new(mode: ComputeMode) -> Self {
        let total_cores = num_cpus::get();
        let monitor = Arc::new(ResourceMonitor::new());
        let trainer = Arc::new(Trainer::new());

        // Default: mining gets 75% of cores, rest shared
        let mining_cores = (total_cores * 3 / 4).max(1);

        info!(
            "🚀 [STARSHIP] Compute Orchestrator initialized — {} cores, mode={:?}",
            total_cores, mode
        );

        let mut assignments = HashMap::new();
        for layer in ComputeLayer::all() {
            assignments.insert(*layer, LayerAssignment {
                layer: *layer,
                core_budget: 0,
                core_range: None,
                active: *layer == ComputeLayer::Mining,
                tasks_completed: Arc::new(AtomicU64::new(0)),
                tasks_pending: Arc::new(AtomicU64::new(0)),
                revenue_micro_qug: Arc::new(AtomicU64::new(0)),
                last_active_tick: 0,
                prev_tasks_completed: 0,
            });
        }

        // #034: Wire inference pool revenue callback to orchestrator assignments
        // Clone the HashMap — the Arc<AtomicU64> values inside are shared, so
        // writes from the callback are visible in the orchestrator's copy.
        let mut pool = InferenceWorkerPool::new();
        let callback_assignments: HashMap<ComputeLayer, LayerAssignment> = assignments.clone();
        pool.set_orchestrator_callback(move |layer, revenue| {
            if let Some(assignment) = callback_assignments.get(&layer) {
                assignment.tasks_completed.fetch_add(1, Ordering::Relaxed);
                assignment.revenue_micro_qug.fetch_add(revenue, Ordering::Relaxed);
            }
        });

        let inference_pool = Arc::new(pool);

        Self {
            mode: Arc::new(RwLock::new(mode)),
            monitor,
            trainer,
            assignments: Arc::new(RwLock::new(assignments)),
            total_cores,
            mining_cores: Arc::new(AtomicU64::new(mining_cores as u64)),
            running: Arc::new(AtomicBool::new(false)),
            inference_pool,
            tunnel_manager: Arc::new(TunnelManager::new(64)), // Max 64 simultaneous tunnels
            local_peer_id: Arc::new(RwLock::new(String::new())),
            core_enforcer: Arc::new(RwLock::new(CoreEnforcer::new())),
        }
    }

    /// Set the local peer ID (called once the P2P identity is available).
    pub fn set_local_peer_id(&self, peer_id: &str) {
        *self.local_peer_id.write() = peer_id.to_string();
        info!("🚀 [STARSHIP] Local peer ID set to {}", peer_id);
    }

    /// Get current compute mode
    pub fn mode(&self) -> ComputeMode {
        *self.mode.read()
    }

    /// Set compute mode
    pub fn set_mode(&self, mode: ComputeMode) {
        info!("🚀 [STARSHIP] Compute mode changed to {:?}", mode);
        *self.mode.write() = mode;
    }

    /// Get resource monitor
    pub fn monitor(&self) -> &Arc<ResourceMonitor> {
        &self.monitor
    }

    /// Get trainer
    pub fn trainer(&self) -> &Arc<Trainer> {
        &self.trainer
    }

    /// Get inference worker pool
    pub fn inference_pool(&self) -> &Arc<InferenceWorkerPool> {
        &self.inference_pool
    }

    /// Get the peer registry for direct access (delegates to tunnel manager).
    pub fn peer_registry(&self) -> &Arc<PeerRegistry> {
        self.tunnel_manager.peer_registry()
    }

    /// Get the tunnel manager for P2P compute tunnel lifecycle.
    pub fn tunnel_manager(&self) -> &Arc<TunnelManager> {
        &self.tunnel_manager
    }

    /// Record a completed task for a layer
    pub fn record_task(&self, layer: ComputeLayer, revenue_micro_qug: u64) {
        let assignments = self.assignments.read();
        if let Some(assignment) = assignments.get(&layer) {
            assignment.tasks_completed.fetch_add(1, Ordering::Relaxed);
            assignment.revenue_micro_qug.fetch_add(revenue_micro_qug, Ordering::Relaxed);
        }
    }

    /// v9.5.1: Record inference revenue specifically (convenience for external callers)
    pub fn record_inference_revenue(&self, tokens: u64, price_per_token_micro_qug: u64) {
        let revenue = tokens.saturating_mul(price_per_token_micro_qug);
        self.record_task(ComputeLayer::AiInference, revenue);
    }

    /// v9.5.1: Get the concrete core range assigned to a layer (#013)
    pub fn get_layer_core_range(&self, layer: &ComputeLayer) -> Option<CoreRange> {
        let assignments = self.assignments.read();
        assignments.get(layer).and_then(|a| a.core_range.clone())
    }

    /// v9.7.0: Get the core enforcer for direct access (e.g. worker threads
    /// that need to pin themselves to their layer's assigned cores).
    pub fn core_enforcer(&self) -> &Arc<RwLock<CoreEnforcer>> {
        &self.core_enforcer
    }

    /// v9.7.0: Pin the calling thread to the specific cores for a layer (#013).
    ///
    /// Uses `libc::sched_setaffinity` on Linux to set affinity to the FULL
    /// set of cores (not just one). On non-Linux platforms, logs a warning
    /// and degrades gracefully.
    ///
    /// Returns true if affinity was successfully enforced, false otherwise.
    pub fn enforce_affinity_for_layer(layer: &ComputeLayer, core_range: &CoreRange) -> bool {
        use crate::core_enforcer::AffinityResult;

        if core_range.is_empty() {
            return false;
        }

        let cores: Vec<usize> = (core_range.start..core_range.end).collect();

        // Use a temporary enforcer for static calls (thread-local enforcement).
        // For tracked enforcement, callers should use core_enforcer() directly.
        let mut enforcer = CoreEnforcer::new();
        matches!(
            enforcer.enforce_layer_affinity(*layer, &cores),
            AffinityResult::Enforced { .. }
        )
    }

    /// v9.7.0: Release affinity for a layer — resets the calling thread to
    /// be schedulable on all cores (#013).
    ///
    /// Returns true if affinity was successfully released.
    pub fn release_affinity_for_layer(layer: &ComputeLayer) -> bool {
        use crate::core_enforcer::AffinityResult;

        let mut enforcer = CoreEnforcer::new();
        matches!(
            enforcer.release_affinity(*layer),
            AffinityResult::Released
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // Gossipsub compute tunnel integration (Issue #002)
    // ═══════════════════════════════════════════════════════════════

    /// Serialize a `ComputePeerInfo` announcement for publishing on the
    /// compute tunnel gossipsub topic.
    ///
    /// The announcement includes the current resource snapshot, compute mode,
    /// active layers, and trainer status. The caller should publish the returned
    /// bytes on `COMPUTE_TUNNEL_TOPIC`.
    pub fn get_peer_announcement(&self) -> Vec<u8> {
        let snapshot = self.monitor.snapshot();
        let mode = self.mode();
        let peer_id = self.local_peer_id.read().clone();

        let mut info = create_peer_announcement(&snapshot, &mode.to_string(), &peer_id);

        // Populate active layers from current assignments
        let assignments = self.assignments.read();
        info.active_layers = assignments
            .iter()
            .filter(|(_, a)| a.active && a.core_budget > 0)
            .map(|(layer, _)| layer.name().to_string())
            .collect();

        // Populate trainer status
        info.trainer_active = !self.trainer.active_cheats().is_empty();

        serde_json::to_vec(&info).unwrap_or_else(|e| {
            warn!("🚀 [STARSHIP] Failed to serialize peer announcement: {}", e);
            Vec::new()
        })
    }

    /// Process a received gossipsub message from `COMPUTE_TUNNEL_TOPIC`.
    ///
    /// Parses the announcement and stores it in the peer registry with
    /// a 60-second TTL. Stale or self-announcements are silently dropped.
    pub fn process_peer_announcement(&self, data: &[u8]) {
        let info = match parse_peer_announcement(data) {
            Some(i) => i,
            None => {
                debug!("🚀 [STARSHIP] Ignoring unparseable compute peer announcement ({} bytes)", data.len());
                return;
            }
        };

        // Skip our own announcements
        let local_id = self.local_peer_id.read().clone();
        if !local_id.is_empty() && info.peer_id == local_id {
            return;
        }

        debug!(
            "🚀 [STARSHIP] Received compute announcement from {} — cores={}/{}, gpu={:.1}TF, ram={:.1}/{:.1}GB, mode={}",
            info.peer_id, info.available_cores, info.total_cores,
            info.gpu_tflops, info.ram_available_gb, info.ram_total_gb,
            info.compute_mode
        );

        self.tunnel_manager.peer_registry().upsert(info);
    }

    /// Get full compute status for dashboard
    pub fn status(&self) -> ComputeStatus {
        let resources = self.monitor.snapshot();
        let assignments = self.assignments.read();

        let mut layers = Vec::new();
        for layer in ComputeLayer::all() {
            if let Some(a) = assignments.get(layer) {
                layers.push((layer.name().to_string(), LayerStats {
                    cores_assigned: a.core_budget as u32,
                    tasks_completed: AtomicU64Ser(a.tasks_completed.load(Ordering::Relaxed)),
                    tasks_pending: a.tasks_pending.load(Ordering::Relaxed) as u32,
                    revenue_micro_qug: a.revenue_micro_qug.load(Ordering::Relaxed),
                    active_since_ms: if a.active { 1 } else { 0 },
                }));
            }
        }

        let total_revenue: u64 = assignments.values()
            .map(|a| a.revenue_micro_qug.load(Ordering::Relaxed))
            .sum();

        let trainer = self.trainer.clone();
        let cheats = trainer.active_cheats();

        // v9.6.0: Get AI inference stats if pool is active
        let ai_inference = if self.inference_pool.has_engine() {
            Some(self.inference_pool.stats())
        } else {
            None
        };

        // Include discovered compute peers from the registry
        let cluster_peers = self.tunnel_manager.peer_registry().all_peers();

        ComputeStatus {
            mode: self.mode(),
            resources,
            layers,
            tunnels: self.tunnel_manager.tunnel_infos(),
            cluster_peers,
            trainer_active: !cheats.is_empty(),
            trainer_cheats: cheats,
            performance_boost_pct: trainer.estimated_boost_pct(),
            total_revenue_micro_qug: total_revenue,
            ai_inference,
        }
    }

    /// Start the orchestrator background loop
    pub fn spawn(&self) {
        let running = self.running.clone();
        running.store(true, Ordering::SeqCst);

        // 1. Start resource monitor
        self.monitor.spawn();

        // 2. Apply OS-level tuning
        let mode = self.mode();
        if mode == ComputeMode::Full || mode == ComputeMode::Nuke {
            OsTuner::apply_all();
        }

        // 3. Activate trainer if NUKE mode
        if mode == ComputeMode::Nuke {
            self.trainer.activate_all();
        }

        // 4. Start the adaptive scheduler loop
        let monitor = self.monitor.clone();
        let assignments = self.assignments.clone();
        let mining_cores = self.mining_cores.clone();
        let mode_arc = self.mode.clone();
        let total_cores = self.total_cores;
        let _trainer = self.trainer.clone();
        let inference_pool = self.inference_pool.clone();
        let tunnel_manager = self.tunnel_manager.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            info!("🚀 [STARSHIP] Adaptive scheduler started — {} total cores", total_cores);
            let mut tick: u64 = 0;

            loop {
                interval.tick().await;
                if !running.load(Ordering::Relaxed) { break; }
                tick += 1;

                let snap = monitor.snapshot();
                let mode = *mode_arc.read();

                if mode == ComputeMode::MiningOnly {
                    continue;
                }

                let idle_cpu = 100.0 - snap.cpu_total;

                let mut assignments = assignments.write();

                // Mining always gets its reserved cores (pinned to 0..N for cache locality)
                let mining_reserved = mining_cores.load(Ordering::Relaxed) as usize;
                if let Some(mining) = assignments.get_mut(&ComputeLayer::Mining) {
                    mining.core_budget = mining_reserved;
                    mining.core_range = Some(CoreRange::new(0, mining_reserved));
                    mining.active = true;
                }

                // #039: Feedback loop — detect idle layers and reclaim their budgets
                let mut idle_layers: Vec<ComputeLayer> = Vec::new();
                for layer in ComputeLayer::all() {
                    if *layer == ComputeLayer::Mining { continue; }
                    if let Some(a) = assignments.get_mut(layer) {
                        let current_completed = a.tasks_completed.load(Ordering::Relaxed);
                        if current_completed > a.prev_tasks_completed {
                            a.last_active_tick = tick;
                            a.prev_tasks_completed = current_completed;
                        }
                        // If no new tasks completed in 10 ticks (10s), mark idle
                        if a.active && a.core_budget > 0 && tick > a.last_active_tick + 10 {
                            idle_layers.push(*layer);
                        }
                    }
                }
                // Reclaim cores from idle layers
                for layer in &idle_layers {
                    if let Some(a) = assignments.get_mut(layer) {
                        trace!(
                            "🚀 [STARSHIP] Reclaiming {} cores from {} (idle for {}s)",
                            a.core_budget, layer.name(), tick - a.last_active_tick
                        );
                        a.core_budget = 0;
                        a.core_range = None;
                        a.active = false;
                    }
                }

                // #032: Weighted distribution of spare cores
                if idle_cpu > 20.0 && mode != ComputeMode::MiningOnly {
                    let spare_count = total_cores.saturating_sub(mining_reserved);

                    if spare_count > 0 {
                        let layers_to_fill: Vec<ComputeLayer> = match mode {
                            ComputeMode::Eco => vec![ComputeLayer::AiInference],
                            ComputeMode::Full => vec![
                                ComputeLayer::AiInference,
                                ComputeLayer::ZkProofGen,
                                ComputeLayer::BridgeVerify,
                            ],
                            ComputeMode::Nuke => vec![
                                ComputeLayer::AiInference,
                                ComputeLayer::ZkProofGen,
                                ComputeLayer::BridgeVerify,
                                ComputeLayer::IpfsPin,
                                ComputeLayer::VdfCompute,
                                ComputeLayer::RenderFarm,
                                ComputeLayer::IdleCrypto,
                            ],
                            ComputeMode::MiningOnly => vec![],
                        };

                        // Filter out idle layers (they lost their budget this tick)
                        let active_layers: Vec<&ComputeLayer> = layers_to_fill.iter()
                            .filter(|l| !idle_layers.contains(l))
                            .collect();

                        let total_weight: u32 = active_layers.iter().map(|l| l.weight()).sum();

                        if total_weight > 0 {
                            let mut allocated = 0;
                            // Core ranges start after mining's reserved range
                            let mut range_cursor = mining_reserved;
                            for (i, layer) in active_layers.iter().enumerate() {
                                let budget = if i == active_layers.len() - 1 {
                                    // Last layer gets remainder to avoid rounding loss
                                    spare_count - allocated
                                } else {
                                    (spare_count as u64 * layer.weight() as u64 / total_weight as u64) as usize
                                };
                                if let Some(assignment) = assignments.get_mut(layer) {
                                    assignment.core_budget = budget;
                                    // v9.5.1: Compute concrete core range (#013)
                                    assignment.core_range = if budget > 0 {
                                        let range = CoreRange::new(range_cursor, range_cursor + budget);
                                        range_cursor += budget;
                                        Some(range)
                                    } else {
                                        None
                                    };
                                    assignment.active = budget > 0;
                                    if budget > 0 {
                                        assignment.last_active_tick = assignment.last_active_tick.max(tick.saturating_sub(5));
                                    }
                                }
                                allocated += budget;
                            }
                        }

                        trace!(
                            "🚀 [STARSHIP] Core assignment: mining={}, spare={} across {} layers (weighted), idle_cpu={:.1}%",
                            mining_reserved, spare_count, active_layers.len(), idle_cpu
                        );
                    }
                }

                // Sync inference pool with AiInference layer core budget + range (#013, #014)
                if let Some(ai_assignment) = assignments.get(&ComputeLayer::AiInference) {
                    let cores: Vec<usize> = if let Some(ref range) = ai_assignment.core_range {
                        (range.start..range.end).collect()
                    } else {
                        vec![]
                    };
                    inference_pool.update_cores(cores);
                }

                // If mining is struggling (CPU > 90%), reclaim all non-mining cores
                if snap.cpu_total > 90.0 {
                    for layer in ComputeLayer::all() {
                        if *layer != ComputeLayer::Mining {
                            if let Some(assignment) = assignments.get_mut(layer) {
                                if assignment.active && assignment.core_budget > 0 {
                                    debug!(
                                        "🚀 [STARSHIP] Reclaiming {} cores from {} for mining (CPU={:.1}%)",
                                        assignment.core_budget, layer.name(), snap.cpu_total
                                    );
                                    assignment.core_budget = 0;
                                    assignment.core_range = None;
                                    assignment.active = false;
                                }
                            }
                        }
                    }
                    inference_pool.update_cores(vec![]);
                }

                // Periodic tunnel + peer registry cleanup (every 30 ticks = 30s)
                if tick % 30 == 0 {
                    tunnel_manager.cleanup_dead();
                }
            }
            info!("🚀 [STARSHIP] Adaptive scheduler stopped");
        });
    }

    /// Stop the orchestrator
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.monitor.stop();
        info!("🚀 [STARSHIP] Orchestrator stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let orch = Orchestrator::new(ComputeMode::Full);
        assert_eq!(orch.mode(), ComputeMode::Full);
        assert!(orch.total_cores > 0);
    }

    #[test]
    fn test_mode_change() {
        let orch = Orchestrator::new(ComputeMode::Eco);
        assert_eq!(orch.mode(), ComputeMode::Eco);
        orch.set_mode(ComputeMode::Nuke);
        assert_eq!(orch.mode(), ComputeMode::Nuke);
    }

    #[test]
    fn test_record_task() {
        let orch = Orchestrator::new(ComputeMode::Full);
        orch.record_task(ComputeLayer::Mining, 1000);
        orch.record_task(ComputeLayer::Mining, 2000);
        let status = orch.status();
        let mining = status.layers.iter().find(|(name, _)| name == "Mining").unwrap();
        assert_eq!(mining.1.tasks_completed.0, 2);
        assert_eq!(mining.1.revenue_micro_qug, 3000);
    }

    #[test]
    fn test_compute_mode_parse() {
        assert_eq!("nuke".parse::<ComputeMode>().unwrap(), ComputeMode::Nuke);
        assert_eq!("eco".parse::<ComputeMode>().unwrap(), ComputeMode::Eco);
        assert_eq!("full".parse::<ComputeMode>().unwrap(), ComputeMode::Full);
        assert_eq!("mining-only".parse::<ComputeMode>().unwrap(), ComputeMode::MiningOnly);
        assert_eq!("yolo".parse::<ComputeMode>().unwrap(), ComputeMode::Nuke);
        assert!("invalid".parse::<ComputeMode>().is_err());
    }

    #[test]
    fn test_get_peer_announcement() {
        let orch = Orchestrator::new(ComputeMode::Full);
        orch.set_local_peer_id("12D3KooWTestOrch");

        let bytes = orch.get_peer_announcement();
        assert!(!bytes.is_empty(), "Peer announcement should not be empty");

        // Should be valid JSON
        let parsed: ComputePeerInfo = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.peer_id, "12D3KooWTestOrch");
        assert_eq!(parsed.compute_mode, "full");
        // total_cores may be 0 in test environments where ResourceMonitor
        // snapshot has no per-core data yet (not spawned)
        assert!(parsed.total_cores <= num_cpus::get() as u32 + 1);
    }

    #[test]
    fn test_process_peer_announcement() {
        let orch = Orchestrator::new(ComputeMode::Full);
        orch.set_local_peer_id("local-peer");

        // Create a foreign peer announcement
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let foreign_peer = ComputePeerInfo {
            peer_id: "remote-peer-123".to_string(),
            available_cores: 16,
            total_cores: 32,
            gpu_tflops: 15.0,
            ram_available_gb: 48.0,
            ram_total_gb: 64.0,
            bandwidth_mbps: 10000.0,
            compute_mode: "nuke".to_string(),
            active_layers: vec!["Mining".to_string(), "AI Inference".to_string()],
            trainer_active: true,
            version: "9.7.0".to_string(),
            timestamp: now,
        };

        let data = serde_json::to_vec(&foreign_peer).unwrap();
        orch.process_peer_announcement(&data);

        // Should be stored in registry
        assert_eq!(orch.peer_registry().len(), 1);
        let fetched = orch.peer_registry().get("remote-peer-123").unwrap();
        assert_eq!(fetched.available_cores, 16);
        assert_eq!(fetched.gpu_tflops, 15.0);
    }

    #[test]
    fn test_process_own_announcement_ignored() {
        let orch = Orchestrator::new(ComputeMode::Full);
        orch.set_local_peer_id("my-peer-id");

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let self_announcement = ComputePeerInfo {
            peer_id: "my-peer-id".to_string(),
            available_cores: 8,
            total_cores: 8,
            gpu_tflops: 0.0,
            ram_available_gb: 16.0,
            ram_total_gb: 32.0,
            bandwidth_mbps: 1000.0,
            compute_mode: "full".to_string(),
            active_layers: vec![],
            trainer_active: false,
            version: "test".to_string(),
            timestamp: now,
        };

        let data = serde_json::to_vec(&self_announcement).unwrap();
        orch.process_peer_announcement(&data);

        // Should NOT be stored (it's our own)
        assert_eq!(orch.peer_registry().len(), 0);
    }

    #[test]
    fn test_process_invalid_announcement() {
        let orch = Orchestrator::new(ComputeMode::Full);
        orch.process_peer_announcement(b"not json");
        orch.process_peer_announcement(b"");
        orch.process_peer_announcement(b"{}");

        // None should be stored
        assert_eq!(orch.peer_registry().len(), 0);
    }

    #[test]
    fn test_core_range() {
        let range = CoreRange::new(4, 8);
        assert_eq!(range.len(), 4);
        assert!(!range.is_empty());
        let ids = range.to_core_ids();
        assert_eq!(ids.len(), 4);
        assert_eq!(ids[0].id, 4);
        assert_eq!(ids[3].id, 7);

        let empty = CoreRange::new(0, 0);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_core_range_cpuset_str() {
        let range = CoreRange::new(4, 8);
        assert_eq!(range.to_cpuset_str(), "4,5,6,7");

        let single = CoreRange::new(0, 1);
        assert_eq!(single.to_cpuset_str(), "0");

        let empty = CoreRange::new(0, 0);
        assert_eq!(empty.to_cpuset_str(), "");
    }

    #[test]
    fn test_core_range_to_vec() {
        let range = CoreRange::new(2, 6);
        assert_eq!(range.to_core_vec(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_core_enforcer_accessible() {
        let orch = Orchestrator::new(ComputeMode::Full);
        let enforcer = orch.core_enforcer();
        let guard = enforcer.read();
        assert!(guard.total_cores() > 0);
    }

    #[test]
    fn test_release_affinity_graceful() {
        // Release on a layer that was never pinned — should not crash
        let result = Orchestrator::release_affinity_for_layer(&ComputeLayer::AiInference);
        let _ = result; // May succeed or fail depending on platform
    }

    #[test]
    fn test_record_inference_revenue() {
        let orch = Orchestrator::new(ComputeMode::Full);

        // Record 1000 tokens at 5 micro-QUG each = 5000 revenue
        orch.record_inference_revenue(1000, 5);

        let status = orch.status();
        let ai_layer = status.layers.iter().find(|(name, _)| name == "AI Inference").unwrap();
        assert_eq!(ai_layer.1.tasks_completed.0, 1);
        assert_eq!(ai_layer.1.revenue_micro_qug, 5000);
    }

    #[test]
    fn test_get_layer_core_range() {
        let orch = Orchestrator::new(ComputeMode::Full);

        // Initially, non-mining layers have no range
        assert!(orch.get_layer_core_range(&ComputeLayer::AiInference).is_none());

        // Mining should have a range (set in constructor)
        // Note: ranges only get set after spawn() scheduler runs, but
        // the constructor sets mining_cores so let's verify the concept
        let assignments = orch.assignments.read();
        let mining = assignments.get(&ComputeLayer::Mining).unwrap();
        // The range is set by the scheduler, not the constructor, so it's None initially
        // This is correct — the scheduler loop sets it on first tick
        drop(assignments);
    }

    #[test]
    fn test_enforce_affinity_graceful() {
        // Even on systems with limited cores, this should not crash
        let range = CoreRange::new(0, 1);
        let result = Orchestrator::enforce_affinity_for_layer(&ComputeLayer::Mining, &range);
        // We don't assert true — it may fail in CI/containers — just assert no panic
        let _ = result;
    }

    #[test]
    fn test_status_includes_peers() {
        let orch = Orchestrator::new(ComputeMode::Full);
        orch.set_local_peer_id("local");

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Add two remote peers
        for i in 0..2 {
            let peer = ComputePeerInfo {
                peer_id: format!("remote-{}", i),
                available_cores: 4,
                total_cores: 8,
                gpu_tflops: 5.0,
                ram_available_gb: 8.0,
                ram_total_gb: 16.0,
                bandwidth_mbps: 1000.0,
                compute_mode: "full".to_string(),
                active_layers: vec![],
                trainer_active: false,
                version: "test".to_string(),
                timestamp: now,
            };
            let data = serde_json::to_vec(&peer).unwrap();
            orch.process_peer_announcement(&data);
        }

        let status = orch.status();
        assert_eq!(status.cluster_peers.len(), 2);
    }
}
