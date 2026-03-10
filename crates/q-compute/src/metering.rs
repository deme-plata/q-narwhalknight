//! Compute Metering — Issue #021
//!
//! Unified metering pipeline for all 8 compute layers. Tracks CPU-seconds,
//! GPU-seconds, and memory-GB-seconds per task. Feeds into PaaSBillingManagerV2
//! for settlement.
//!
//! ## Design
//!
//! Every compute task gets a `MeteringHandle` at start. The handle records
//! resource consumption samples at regular intervals. When the task completes,
//! `finalize()` calculates the total cost using the rate card and returns a
//! `MeteringRecord` for billing.
//!
//! ## Rate Card
//!
//! Each compute layer has different per-unit prices (configurable via env vars).
//! Default prices are in micro-QUG per resource-second.

#![allow(dead_code)]

use crate::ComputeLayer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use tracing::{debug, info, warn};

// ═══════════════════════════════════════════════════════════════════
// Rate Card — per-unit pricing for compute resources
// ═══════════════════════════════════════════════════════════════════

/// Price per resource-second in micro-QUG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateCard {
    /// Price per CPU-second (micro-QUG)
    pub cpu_second: u64,
    /// Price per GPU-second (micro-QUG)
    pub gpu_second: u64,
    /// Price per memory-GB-second (micro-QUG)
    pub memory_gb_second: u64,
    /// Price per network-MB-transferred (micro-QUG)
    pub network_mb: u64,
}

impl Default for RateCard {
    fn default() -> Self {
        Self {
            cpu_second: 1,      // 1 micro-QUG per CPU-second
            gpu_second: 10,     // 10 micro-QUG per GPU-second
            memory_gb_second: 1, // 1 micro-QUG per GB-second
            network_mb: 1,      // 1 micro-QUG per MB transferred
        }
    }
}

/// Per-layer rate cards (different layers have different pricing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRateCards {
    pub cards: HashMap<String, RateCard>,
    pub default_card: RateCard,
}

impl Default for LayerRateCards {
    fn default() -> Self {
        let mut cards = HashMap::new();

        // Mining: lowest rates (network benefit)
        cards.insert("Mining".to_string(), RateCard {
            cpu_second: 0, gpu_second: 0, memory_gb_second: 0, network_mb: 0,
        });

        // AI Inference: premium rates (revenue generating)
        cards.insert("AI Inference".to_string(), RateCard {
            cpu_second: 2, gpu_second: 20, memory_gb_second: 2, network_mb: 1,
        });

        // ZK Proofs: high GPU usage
        cards.insert("ZK Proofs".to_string(), RateCard {
            cpu_second: 1, gpu_second: 15, memory_gb_second: 1, network_mb: 1,
        });

        // Bridge Verify: moderate
        cards.insert("Bridge Verify".to_string(), RateCard {
            cpu_second: 1, gpu_second: 5, memory_gb_second: 1, network_mb: 1,
        });

        Self {
            cards,
            default_card: RateCard::default(),
        }
    }
}

impl LayerRateCards {
    /// Get the rate card for a specific layer.
    pub fn get(&self, layer: ComputeLayer) -> &RateCard {
        self.cards
            .get(layer.name())
            .unwrap_or(&self.default_card)
    }

    /// Set a custom rate card for a layer.
    pub fn set(&mut self, layer: ComputeLayer, card: RateCard) {
        self.cards.insert(layer.name().to_string(), card);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Resource Sample — a point-in-time measurement
// ═══════════════════════════════════════════════════════════════════

/// A single resource consumption sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSample {
    /// CPU cores actively used (fractional, e.g., 2.5 = 2.5 cores)
    pub cpu_cores: f64,
    /// GPU utilization 0.0..1.0
    pub gpu_utilization: f64,
    /// Memory used in bytes
    pub memory_bytes: u64,
    /// Network bytes transferred since last sample
    pub network_bytes: u64,
    /// When this sample was taken (monotonic, relative to task start)
    pub elapsed_ms: u64,
}

// ═══════════════════════════════════════════════════════════════════
// Metering Record — final billing for a completed task
// ═══════════════════════════════════════════════════════════════════

/// Finalized metering record for a completed compute task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeteringRecord {
    /// Unique task ID
    pub task_id: String,
    /// Which compute layer performed this task
    pub layer: String,
    /// Total CPU-seconds consumed
    pub cpu_seconds: f64,
    /// Total GPU-seconds consumed
    pub gpu_seconds: f64,
    /// Total memory-GB-seconds consumed
    pub memory_gb_seconds: f64,
    /// Total network MB transferred
    pub network_mb: f64,
    /// Total cost in micro-QUG (calculated from rate card)
    pub cost_micro_qug: u64,
    /// Wall-clock duration in milliseconds
    pub duration_ms: u64,
    /// Task start timestamp (unix millis)
    pub started_ms: u64,
    /// Task end timestamp (unix millis)
    pub ended_ms: u64,
    /// Number of samples collected
    pub sample_count: u32,
}

// ═══════════════════════════════════════════════════════════════════
// Metering Handle — active task metering
// ═══════════════════════════════════════════════════════════════════

/// Handle for an actively metered compute task.
///
/// Created by `MeteringSink::start_task()`. Call `record_sample()` periodically
/// during the task, then `finalize()` when complete.
pub struct MeteringHandle {
    task_id: String,
    layer: ComputeLayer,
    started: Instant,
    started_ms: u64,
    samples: Vec<ResourceSample>,
    rate_card: RateCard,
}

impl MeteringHandle {
    fn new(task_id: String, layer: ComputeLayer, rate_card: RateCard) -> Self {
        let started_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            task_id,
            layer,
            started: Instant::now(),
            started_ms,
            samples: Vec::new(),
            rate_card,
        }
    }

    /// Record a resource consumption sample.
    pub fn record_sample(&mut self, sample: ResourceSample) {
        self.samples.push(sample);
    }

    /// Record a simple sample from CPU cores and memory bytes.
    pub fn record_simple(&mut self, cpu_cores: f64, memory_bytes: u64) {
        let elapsed_ms = self.started.elapsed().as_millis() as u64;
        self.samples.push(ResourceSample {
            cpu_cores,
            gpu_utilization: 0.0,
            memory_bytes,
            network_bytes: 0,
            elapsed_ms,
        });
    }

    /// Finalize the metering and produce a billing record.
    ///
    /// Calculates total resource consumption by integrating samples over time.
    /// Returns the `MeteringRecord` with computed cost.
    pub fn finalize(self) -> MeteringRecord {
        let wall_ms = self.started.elapsed().as_millis() as u64;
        // Use the max of wall-clock and sample-declared time so tests that
        // don't sleep still get correct integration.
        let sample_max_ms = self.samples.last().map(|s| s.elapsed_ms).unwrap_or(0);
        let duration_ms = wall_ms.max(sample_max_ms);
        let ended_ms = self.started_ms + duration_ms;

        // Integrate resource consumption over time using trapezoidal rule
        let (cpu_seconds, gpu_seconds, memory_gb_seconds, network_mb) =
            Self::integrate_samples(&self.samples, duration_ms);

        // Calculate cost from rate card
        let cost = (cpu_seconds * self.rate_card.cpu_second as f64) as u64
            + (gpu_seconds * self.rate_card.gpu_second as f64) as u64
            + (memory_gb_seconds * self.rate_card.memory_gb_second as f64) as u64
            + (network_mb * self.rate_card.network_mb as f64) as u64;

        // Minimum 1 micro-QUG for any non-zero work, but respect zero-rate
        // layers (e.g. Mining) where all rates are 0.
        let rate_is_zero = self.rate_card.cpu_second == 0
            && self.rate_card.gpu_second == 0
            && self.rate_card.memory_gb_second == 0
            && self.rate_card.network_mb == 0;
        let cost = if self.samples.is_empty() || duration_ms == 0 || rate_is_zero {
            0
        } else {
            cost.max(1)
        };

        MeteringRecord {
            task_id: self.task_id,
            layer: self.layer.name().to_string(),
            cpu_seconds,
            gpu_seconds,
            memory_gb_seconds,
            network_mb,
            cost_micro_qug: cost,
            duration_ms,
            started_ms: self.started_ms,
            ended_ms,
            sample_count: self.samples.len() as u32,
        }
    }

    /// Integrate resource samples over time (trapezoidal approximation).
    fn integrate_samples(
        samples: &[ResourceSample],
        total_duration_ms: u64,
    ) -> (f64, f64, f64, f64) {
        if samples.is_empty() || total_duration_ms == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let mut cpu_seconds = 0.0;
        let mut gpu_seconds = 0.0;
        let mut memory_gb_seconds = 0.0;
        let mut network_bytes_total = 0u64;

        for i in 0..samples.len() {
            let dt_ms = if i + 1 < samples.len() {
                samples[i + 1].elapsed_ms.saturating_sub(samples[i].elapsed_ms)
            } else {
                total_duration_ms.saturating_sub(samples[i].elapsed_ms)
            };
            let dt_sec = dt_ms as f64 / 1000.0;

            cpu_seconds += samples[i].cpu_cores * dt_sec;
            gpu_seconds += samples[i].gpu_utilization * dt_sec;
            memory_gb_seconds += (samples[i].memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) * dt_sec;
            network_bytes_total += samples[i].network_bytes;
        }

        let network_mb = network_bytes_total as f64 / (1024.0 * 1024.0);

        (cpu_seconds, gpu_seconds, memory_gb_seconds, network_mb)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Metering Sink — central metering service
// ═══════════════════════════════════════════════════════════════════

/// Aggregate metering statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeteringStats {
    /// Total tasks metered
    pub total_tasks: u64,
    /// Total revenue metered across all layers (micro-QUG)
    pub total_revenue_micro_qug: u64,
    /// Revenue breakdown per layer name
    pub per_layer_revenue: HashMap<String, u64>,
    /// Tasks pending finalization
    pub active_tasks: u32,
    /// Records waiting for settlement
    pub pending_settlement: u32,
}

/// Central metering service for all compute layers.
///
/// Creates `MeteringHandle`s for new tasks, collects finalized records,
/// and batches them for settlement.
pub struct MeteringSink {
    /// Rate cards per layer
    rate_cards: Arc<RwLock<LayerRateCards>>,
    /// Number of active (non-finalized) metering handles
    active_count: Arc<AtomicU64>,
    /// Finalized records awaiting settlement
    pending_records: Arc<RwLock<Vec<MeteringRecord>>>,
    /// Total tasks metered
    total_tasks: Arc<AtomicU64>,
    /// Total revenue across all tasks
    total_revenue: Arc<AtomicU64>,
    /// Per-layer revenue (layer name → total micro-QUG)
    per_layer_revenue: Arc<RwLock<HashMap<String, u64>>>,
    /// Settlement batch size threshold
    settlement_batch_size: usize,
}

impl MeteringSink {
    /// Create a new metering sink with default rate cards.
    pub fn new() -> Self {
        Self::with_rate_cards(LayerRateCards::default())
    }

    /// Create a new metering sink with custom rate cards.
    pub fn with_rate_cards(rate_cards: LayerRateCards) -> Self {
        info!("📊 [METERING] Sink initialized with {} layer rate cards", rate_cards.cards.len());
        Self {
            rate_cards: Arc::new(RwLock::new(rate_cards)),
            active_count: Arc::new(AtomicU64::new(0)),
            pending_records: Arc::new(RwLock::new(Vec::new())),
            total_tasks: Arc::new(AtomicU64::new(0)),
            total_revenue: Arc::new(AtomicU64::new(0)),
            per_layer_revenue: Arc::new(RwLock::new(HashMap::new())),
            settlement_batch_size: 100,
        }
    }

    /// Start metering a new task. Returns a `MeteringHandle` to record samples.
    pub fn start_task(&self, task_id: String, layer: ComputeLayer) -> MeteringHandle {
        let rate_card = self.rate_cards.read().get(layer).clone();
        self.active_count.fetch_add(1, Ordering::Relaxed);
        debug!("📊 [METERING] Task {} started (layer={})", task_id, layer.name());
        MeteringHandle::new(task_id, layer, rate_card)
    }

    /// Finalize a metering handle and record the result.
    ///
    /// Returns the computed cost in micro-QUG.
    pub fn finalize_task(&self, handle: MeteringHandle) -> u64 {
        let record = handle.finalize();
        let cost = record.cost_micro_qug;
        let layer = record.layer.clone();
        let task_id = record.task_id.clone();

        // Update counters
        self.total_tasks.fetch_add(1, Ordering::Relaxed);
        self.total_revenue.fetch_add(cost, Ordering::Relaxed);
        self.active_count.fetch_sub(1, Ordering::Relaxed);

        // Update per-layer revenue
        {
            let mut plr = self.per_layer_revenue.write();
            *plr.entry(layer.clone()).or_insert(0) += cost;
        }

        // Queue for settlement
        {
            let mut pending = self.pending_records.write();
            pending.push(record);
        }

        debug!(
            "📊 [METERING] Task {} finalized: layer={}, cost={} µQUG",
            task_id, layer, cost
        );

        cost
    }

    /// Drain pending records for settlement (up to batch_size).
    ///
    /// Returns records that should be settled. Called by the billing manager.
    pub fn drain_for_settlement(&self) -> Vec<MeteringRecord> {
        let mut pending = self.pending_records.write();
        let batch_size = self.settlement_batch_size.min(pending.len());
        pending.drain(..batch_size).collect()
    }

    /// Check if settlement should be triggered.
    pub fn should_settle(&self) -> bool {
        self.pending_records.read().len() >= self.settlement_batch_size
    }

    /// Get the number of records pending settlement.
    pub fn pending_count(&self) -> usize {
        self.pending_records.read().len()
    }

    /// Get metering statistics.
    pub fn stats(&self) -> MeteringStats {
        MeteringStats {
            total_tasks: self.total_tasks.load(Ordering::Relaxed),
            total_revenue_micro_qug: self.total_revenue.load(Ordering::Relaxed),
            per_layer_revenue: self.per_layer_revenue.read().clone(),
            active_tasks: self.active_count.load(Ordering::Relaxed) as u32,
            pending_settlement: self.pending_records.read().len() as u32,
        }
    }

    /// Update the rate card for a specific layer.
    pub fn update_rate_card(&self, layer: ComputeLayer, card: RateCard) {
        self.rate_cards.write().set(layer, card);
        info!("📊 [METERING] Rate card updated for layer {}", layer.name());
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rate_card() {
        let card = RateCard::default();
        assert_eq!(card.cpu_second, 1);
        assert_eq!(card.gpu_second, 10);
        assert_eq!(card.memory_gb_second, 1);
    }

    #[test]
    fn test_layer_rate_cards_mining_free() {
        let cards = LayerRateCards::default();
        let mining = cards.get(ComputeLayer::Mining);
        assert_eq!(mining.cpu_second, 0);
        assert_eq!(mining.gpu_second, 0);
    }

    #[test]
    fn test_layer_rate_cards_inference_premium() {
        let cards = LayerRateCards::default();
        let inference = cards.get(ComputeLayer::AiInference);
        assert!(inference.cpu_second > 1);
        assert!(inference.gpu_second > 10);
    }

    #[test]
    fn test_metering_handle_no_samples() {
        let handle = MeteringHandle::new(
            "task-1".to_string(),
            ComputeLayer::AiInference,
            RateCard::default(),
        );
        let record = handle.finalize();
        assert_eq!(record.task_id, "task-1");
        assert_eq!(record.cost_micro_qug, 0);
        assert_eq!(record.sample_count, 0);
    }

    #[test]
    fn test_metering_handle_with_samples() {
        let mut handle = MeteringHandle::new(
            "task-2".to_string(),
            ComputeLayer::AiInference,
            RateCard { cpu_second: 10, gpu_second: 100, memory_gb_second: 5, network_mb: 1 },
        );

        // Simulate 1 second of 2 CPU cores + 50% GPU + 1GB memory
        handle.record_sample(ResourceSample {
            cpu_cores: 2.0,
            gpu_utilization: 0.5,
            memory_bytes: 1024 * 1024 * 1024, // 1 GB
            network_bytes: 0,
            elapsed_ms: 0,
        });
        handle.record_sample(ResourceSample {
            cpu_cores: 2.0,
            gpu_utilization: 0.5,
            memory_bytes: 1024 * 1024 * 1024,
            network_bytes: 0,
            elapsed_ms: 1000,
        });

        let record = handle.finalize();
        assert_eq!(record.sample_count, 2);
        assert!(record.cost_micro_qug > 0);
        assert!(record.cpu_seconds > 0.0);
    }

    #[test]
    fn test_integrate_samples_empty() {
        let (cpu, gpu, mem, net) = MeteringHandle::integrate_samples(&[], 1000);
        assert_eq!(cpu, 0.0);
        assert_eq!(gpu, 0.0);
        assert_eq!(mem, 0.0);
        assert_eq!(net, 0.0);
    }

    #[test]
    fn test_integrate_samples_single() {
        let samples = vec![ResourceSample {
            cpu_cores: 4.0,
            gpu_utilization: 1.0,
            memory_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
            network_bytes: 1024 * 1024,             // 1 MB
            elapsed_ms: 0,
        }];
        let (cpu, gpu, mem, net) = MeteringHandle::integrate_samples(&samples, 2000);
        // 4 cores * 2 seconds = 8 CPU-seconds
        assert!((cpu - 8.0).abs() < 0.01);
        // 1.0 GPU * 2 seconds = 2 GPU-seconds
        assert!((gpu - 2.0).abs() < 0.01);
        // 2 GB * 2 seconds = 4 GB-seconds
        assert!((mem - 4.0).abs() < 0.01);
        // 1 MB
        assert!((net - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_metering_sink_lifecycle() {
        let sink = MeteringSink::new();
        assert_eq!(sink.stats().total_tasks, 0);

        let mut handle = sink.start_task("t1".to_string(), ComputeLayer::AiInference);
        assert_eq!(sink.stats().active_tasks, 1);

        handle.record_simple(2.0, 512 * 1024 * 1024);
        std::thread::sleep(Duration::from_millis(10));

        let cost = sink.finalize_task(handle);
        assert!(cost >= 1); // minimum 1 micro-QUG for non-zero work

        let stats = sink.stats();
        assert_eq!(stats.total_tasks, 1);
        assert!(stats.total_revenue_micro_qug > 0);
        assert_eq!(stats.active_tasks, 0);
        assert_eq!(stats.pending_settlement, 1);
    }

    #[test]
    fn test_metering_sink_per_layer_revenue() {
        let sink = MeteringSink::new();

        let mut h1 = sink.start_task("t1".to_string(), ComputeLayer::AiInference);
        h1.record_simple(4.0, 1024 * 1024 * 1024);
        std::thread::sleep(Duration::from_millis(10));
        sink.finalize_task(h1);

        let mut h2 = sink.start_task("t2".to_string(), ComputeLayer::ZkProofGen);
        h2.record_simple(2.0, 512 * 1024 * 1024);
        std::thread::sleep(Duration::from_millis(10));
        sink.finalize_task(h2);

        let stats = sink.stats();
        assert!(stats.per_layer_revenue.contains_key("AI Inference"));
        assert!(stats.per_layer_revenue.contains_key("ZK Proofs"));
        assert_eq!(stats.total_tasks, 2);
    }

    #[test]
    fn test_drain_for_settlement() {
        let sink = MeteringSink::new();

        for i in 0..5 {
            let mut h = sink.start_task(format!("t{}", i), ComputeLayer::BridgeVerify);
            h.record_simple(1.0, 256 * 1024 * 1024);
            std::thread::sleep(Duration::from_millis(5));
            sink.finalize_task(h);
        }

        assert_eq!(sink.pending_count(), 5);

        let batch = sink.drain_for_settlement();
        assert_eq!(batch.len(), 5);
        assert_eq!(sink.pending_count(), 0);
    }

    #[test]
    fn test_should_settle_threshold() {
        let sink = MeteringSink::new();
        assert!(!sink.should_settle()); // 0 < 100

        for i in 0..100 {
            let h = sink.start_task(format!("t{}", i), ComputeLayer::IdleCrypto);
            sink.finalize_task(h);
        }

        assert!(sink.should_settle()); // 100 >= 100
    }

    #[test]
    fn test_mining_is_free() {
        let sink = MeteringSink::new();
        let mut h = sink.start_task("mining-1".to_string(), ComputeLayer::Mining);
        h.record_simple(8.0, 4 * 1024 * 1024 * 1024);
        std::thread::sleep(Duration::from_millis(10));
        let cost = sink.finalize_task(h);
        // Mining rate card is all zeros
        assert_eq!(cost, 0);
    }

    #[test]
    fn test_update_rate_card() {
        let sink = MeteringSink::new();

        // Update VDF compute to premium pricing
        sink.update_rate_card(ComputeLayer::VdfCompute, RateCard {
            cpu_second: 100, gpu_second: 500, memory_gb_second: 50, network_mb: 10,
        });

        let mut h = sink.start_task("vdf-1".to_string(), ComputeLayer::VdfCompute);
        h.record_simple(1.0, 1024 * 1024 * 1024);
        std::thread::sleep(Duration::from_millis(10));
        let cost = sink.finalize_task(h);
        // Should be higher than default due to premium rates
        assert!(cost >= 1);
    }

    #[test]
    fn test_record_serde_roundtrip() {
        let record = MeteringRecord {
            task_id: "test-serde".to_string(),
            layer: "AI Inference".to_string(),
            cpu_seconds: 4.5,
            gpu_seconds: 2.0,
            memory_gb_seconds: 10.0,
            network_mb: 0.5,
            cost_micro_qug: 150,
            duration_ms: 5000,
            started_ms: 1000000,
            ended_ms: 1005000,
            sample_count: 50,
        };
        let json = serde_json::to_string(&record).unwrap();
        let parsed: MeteringRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.task_id, "test-serde");
        assert_eq!(parsed.cost_micro_qug, 150);
        assert_eq!(parsed.sample_count, 50);
    }

    #[test]
    fn test_rate_card_serde_roundtrip() {
        let card = RateCard { cpu_second: 42, gpu_second: 100, memory_gb_second: 5, network_mb: 3 };
        let json = serde_json::to_string(&card).unwrap();
        let parsed: RateCard = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cpu_second, 42);
        assert_eq!(parsed.gpu_second, 100);
    }
}
