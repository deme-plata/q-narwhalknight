//! Compute Orchestrator — Adaptive 8-layer priority scheduler
//!
//! Monitors resource utilization every 100ms and assigns work
//! to idle CPU/GPU/RAM. Mining (Layer 0) always has priority.
//! Lower layers fill the gaps.

use crate::{ComputeLayer, ComputeMode, ComputeStatus, LayerStats, AtomicU64Ser};
use crate::resource_monitor::ResourceMonitor;
use crate::trainer::Trainer;
use crate::os_tuner::OsTuner;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use parking_lot::RwLock;
use tracing::{info, debug, trace};

/// Core assignment for each layer
#[derive(Debug, Clone)]
struct LayerAssignment {
    layer: ComputeLayer,
    cores: Vec<usize>,       // Which CPU cores are assigned
    active: bool,
    tasks_completed: Arc<AtomicU64>,
    tasks_pending: Arc<AtomicU64>,
    revenue_micro_qug: Arc<AtomicU64>,
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
                cores: Vec::new(),
                active: *layer == ComputeLayer::Mining, // Only mining active by default
                tasks_completed: Arc::new(AtomicU64::new(0)),
                tasks_pending: Arc::new(AtomicU64::new(0)),
                revenue_micro_qug: Arc::new(AtomicU64::new(0)),
            });
        }

        Self {
            mode: Arc::new(RwLock::new(mode)),
            monitor,
            trainer,
            assignments: Arc::new(RwLock::new(assignments)),
            total_cores,
            mining_cores: Arc::new(AtomicU64::new(mining_cores as u64)),
            running: Arc::new(AtomicBool::new(false)),
        }
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

    /// Record a completed task for a layer
    pub fn record_task(&self, layer: ComputeLayer, revenue_micro_qug: u64) {
        let assignments = self.assignments.read();
        if let Some(assignment) = assignments.get(&layer) {
            assignment.tasks_completed.fetch_add(1, Ordering::Relaxed);
            assignment.revenue_micro_qug.fetch_add(revenue_micro_qug, Ordering::Relaxed);
        }
    }

    /// Get full compute status for dashboard
    pub fn status(&self) -> ComputeStatus {
        let resources = self.monitor.snapshot();
        let assignments = self.assignments.read();

        let mut layers = Vec::new();
        for layer in ComputeLayer::all() {
            if let Some(a) = assignments.get(layer) {
                layers.push((layer.name().to_string(), LayerStats {
                    cores_assigned: a.cores.len() as u32,
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

        ComputeStatus {
            mode: self.mode(),
            resources,
            layers,
            tunnels: Vec::new(), // Populated by tunnel module
            cluster_peers: Vec::new(), // Populated from gossipsub
            trainer_active: !cheats.is_empty(),
            trainer_cheats: cheats,
            performance_boost_pct: trainer.estimated_boost_pct(),
            total_revenue_micro_qug: total_revenue,
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

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            info!("🚀 [STARSHIP] Adaptive scheduler started — {} total cores", total_cores);

            loop {
                interval.tick().await;
                if !running.load(Ordering::Relaxed) { break; }

                let snap = monitor.snapshot();
                let mode = *mode_arc.read();

                if mode == ComputeMode::MiningOnly {
                    continue; // Don't touch anything in mining-only mode
                }

                let idle_cpu = 100.0 - snap.cpu_total;
                let _idle_ram_pct = if snap.ram_total > 0 {
                    ((snap.ram_total - snap.ram_used) as f64 / snap.ram_total as f64 * 100.0) as f32
                } else {
                    0.0
                };

                // Adaptive core assignment based on idle resources
                let mut assignments = assignments.write();

                // Mining always gets its reserved cores
                let mining_reserved = mining_cores.load(Ordering::Relaxed) as usize;
                if let Some(mining) = assignments.get_mut(&ComputeLayer::Mining) {
                    mining.cores = (0..mining_reserved).collect();
                    mining.active = true;
                }

                // If CPU is > 20% idle and mode allows, assign to lower layers
                if idle_cpu > 20.0 && mode != ComputeMode::MiningOnly {
                    let spare_cores: Vec<usize> = (mining_reserved..total_cores).collect();
                    let spare_count = spare_cores.len();

                    if spare_count > 0 {
                        // Distribute spare cores across layers by priority
                        let layers_to_fill = match mode {
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

                        let cores_per_layer = spare_count / layers_to_fill.len().max(1);
                        let mut core_idx = mining_reserved;

                        for layer in &layers_to_fill {
                            if let Some(assignment) = assignments.get_mut(layer) {
                                let end = (core_idx + cores_per_layer).min(total_cores);
                                assignment.cores = (core_idx..end).collect();
                                assignment.active = true;
                                core_idx = end;
                            }
                        }

                        trace!(
                            "🚀 [STARSHIP] Core assignment: mining={}, spare={} across {} layers, idle_cpu={:.1}%",
                            mining_reserved, spare_count, layers_to_fill.len(), idle_cpu
                        );
                    }
                }

                // If mining is struggling (CPU > 90%), reclaim cores from lower layers
                if snap.cpu_total > 90.0 {
                    for layer in ComputeLayer::all() {
                        if *layer != ComputeLayer::Mining {
                            if let Some(assignment) = assignments.get_mut(layer) {
                                if assignment.active && !assignment.cores.is_empty() {
                                    debug!(
                                        "🚀 [STARSHIP] Reclaiming {} cores from {} for mining (CPU={:.1}%)",
                                        assignment.cores.len(), layer.name(), snap.cpu_total
                                    );
                                    assignment.cores.clear();
                                    assignment.active = false;
                                }
                            }
                        }
                    }
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
}
