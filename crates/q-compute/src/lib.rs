//! # q-compute: Starship Endgame Revolution
//!
//! 100% compute utilization orchestrator for QNK nodes.
//! Every cycle counts. Not a single electron wasted.
//!
//! ## Architecture
//!
//! 8-layer priority compute scheduler:
//! - Layer 0: Mining (always wins)
//! - Layer 1: AI Inference
//! - Layer 2: ZK Proof Generation
//! - Layer 3: Bridge Verification
//! - Layer 4: IPFS Pinning
//! - Layer 5: VDF Computation
//! - Layer 6: Render Farm
//! - Layer 7: Idle Crypto

pub mod orchestrator;
pub mod gpu_miner; // Issue #003: GPU mining acceleration via wgpu compute pipelines
pub mod core_enforcer; // v9.7.0: Real CPU affinity enforcement via sched_setaffinity (#013)
pub mod trainer;
pub mod tunnel;
pub mod os_tuner;
pub mod resource_monitor;
pub mod inference_pool; // v9.6.0: AI inference worker pool for idle cores
pub mod distributed_inference; // v9.8.0: Issue #005 — distributed AI inference routing
pub mod gpu_scheduler; // Issue #023: Multi-GPU scheduling and layer assignment
pub mod compute_reputation; // Issue #022: Node reputation for compute task assignment
pub mod metering; // Issue #021: Compute billing & metering — unified resource tracking
pub mod model_catalog; // Issue #025: Model catalog & hot-swap — dynamic AI model management
pub mod tensor_parallel; // Issue #018: Cross-node tensor parallelism for distributed model inference
pub mod job_wal; // Issue #026: Compute job queue persistence (WAL)
pub mod bridge_verification; // Issue #016: Bridge safety — multi-peer attestation & 2-of-3 quorum
pub mod marketplace; // Issue #017: Proof-of-Useful-Work marketplace — replace idle crypto with revenue
pub mod marketplace_p2p; // Issue #027: Compute marketplace P2P protocol — gossipsub routing, order book, settlement
pub mod tunnel_rekey; // Issue #024: Tunnel key rotation — forward secrecy via periodic rekey
pub mod zk_proof_farm; // Issue #006: ZK proof farm — background proof generation
pub mod grover_backend; // Issue #015: Quantum Grover mining — amplitude-amplified nonce search
#[cfg(feature = "metrics")]
pub mod metrics; // Prometheus metrics export (feature-gated)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Gossipsub topic for compute tunnel peer announcements.
/// Nodes publish their compute capacity on this topic so peers can
/// discover available resources for distributed task routing.
pub const COMPUTE_TUNNEL_TOPIC: &str = "/qnk/compute-tunnel";

/// Compute mode — how aggressive should we be?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeMode {
    /// Mining only — traditional behavior, no extra compute
    MiningOnly,
    /// Eco — fill idle cycles gently, respect thermal limits
    Eco,
    /// Full — maximize every core, GPU, and byte of RAM
    Full,
    /// NUKE — trainer mode, all cheats active, overclock everything
    Nuke,
}

impl Default for ComputeMode {
    fn default() -> Self {
        ComputeMode::Full
    }
}

impl std::fmt::Display for ComputeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeMode::MiningOnly => write!(f, "mining-only"),
            ComputeMode::Eco => write!(f, "eco"),
            ComputeMode::Full => write!(f, "full"),
            ComputeMode::Nuke => write!(f, "nuke"),
        }
    }
}

impl std::str::FromStr for ComputeMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mining-only" | "mining_only" | "miningonly" => Ok(ComputeMode::MiningOnly),
            "eco" | "quiet" | "gentle" => Ok(ComputeMode::Eco),
            "full" | "max" | "all" => Ok(ComputeMode::Full),
            "nuke" | "yolo" | "extreme" | "trainer" => Ok(ComputeMode::Nuke),
            _ => Err(format!("Unknown compute mode: '{}'. Use: mining-only, eco, full, nuke", s)),
        }
    }
}

/// Priority layers for compute task scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ComputeLayer {
    Mining = 0,
    AiInference = 1,
    ZkProofGen = 2,
    BridgeVerify = 3,
    IpfsPin = 4,
    VdfCompute = 5,
    RenderFarm = 6,
    IdleCrypto = 7,
}

impl ComputeLayer {
    pub fn all() -> &'static [ComputeLayer] {
        &[
            ComputeLayer::Mining,
            ComputeLayer::AiInference,
            ComputeLayer::ZkProofGen,
            ComputeLayer::BridgeVerify,
            ComputeLayer::IpfsPin,
            ComputeLayer::VdfCompute,
            ComputeLayer::RenderFarm,
            ComputeLayer::IdleCrypto,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            ComputeLayer::Mining => "Mining",
            ComputeLayer::AiInference => "AI Inference",
            ComputeLayer::ZkProofGen => "ZK Proofs",
            ComputeLayer::BridgeVerify => "Bridge Verify",
            ComputeLayer::IpfsPin => "IPFS Pinning",
            ComputeLayer::VdfCompute => "VDF Compute",
            ComputeLayer::RenderFarm => "Render Farm",
            ComputeLayer::IdleCrypto => "Idle Crypto",
        }
    }

    /// Priority weight for core allocation. Higher = more cores when distributing.
    /// Mining is handled separately (reserved cores), so its weight here is irrelevant.
    pub fn weight(&self) -> u32 {
        match self {
            ComputeLayer::Mining => 100,       // Reserved separately
            ComputeLayer::AiInference => 50,   // Revenue-generating
            ComputeLayer::ZkProofGen => 20,
            ComputeLayer::BridgeVerify => 15,
            ComputeLayer::IpfsPin => 5,
            ComputeLayer::VdfCompute => 5,
            ComputeLayer::RenderFarm => 3,
            ComputeLayer::IdleCrypto => 1,
        }
    }
}

/// Snapshot of resource utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// CPU utilization 0-100 per core
    pub cpu_per_core: Vec<f32>,
    /// Overall CPU utilization 0-100
    pub cpu_total: f32,
    /// GPU utilization 0-100 (0 if no GPU)
    pub gpu_utilization: f32,
    /// GPU memory used in bytes
    pub gpu_memory_used: u64,
    /// GPU memory total in bytes
    pub gpu_memory_total: u64,
    /// GPU temperature in degrees Celsius (0.0 if unavailable)
    pub gpu_temperature: f32,
    /// GPU device name (empty string if unavailable)
    pub gpu_name: String,
    /// RAM used in bytes
    pub ram_used: u64,
    /// RAM total in bytes
    pub ram_total: u64,
    /// Network TX bytes/sec
    pub net_tx_bps: u64,
    /// Network RX bytes/sec
    pub net_rx_bps: u64,
    /// Network total capacity bytes/sec (estimated)
    pub net_capacity_bps: u64,
    /// Disk I/O bytes/sec
    pub disk_io_bps: u64,
    /// Timestamp (unix millis)
    pub timestamp_ms: u64,
}

/// Individual GPU device info for multi-GPU nodes (Issue #023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    /// GPU index (0, 1, 2, ...)
    pub id: u32,
    /// Device name (e.g., "NVIDIA GeForce RTX 4090")
    pub name: String,
    /// Total VRAM in MB
    pub vram_total_mb: u64,
    /// Used VRAM in MB
    pub vram_used_mb: u64,
    /// Utilization 0-100%
    pub utilization: f32,
    /// Temperature in Celsius
    pub temperature: f32,
    /// Assigned compute layer (if any)
    pub assigned_layer: Option<ComputeLayer>,
}

/// Maps compute layers to GPU device IDs (Issue #023)
pub type GpuAssignment = HashMap<ComputeLayer, Vec<u32>>;

/// Per-layer compute stats
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerStats {
    /// Cores assigned to this layer
    pub cores_assigned: u32,
    /// Tasks completed
    pub tasks_completed: AtomicU64Ser,
    /// Tasks pending
    pub tasks_pending: u32,
    /// Revenue earned (in micro-QUG)
    pub revenue_micro_qug: u64,
    /// Active since (unix millis, 0 = inactive)
    pub active_since_ms: u64,
}

/// Serializable wrapper for AtomicU64
#[derive(Debug, Clone, Default)]
pub struct AtomicU64Ser(pub u64);

impl Serialize for AtomicU64Ser {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for AtomicU64Ser {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        u64::deserialize(deserializer).map(AtomicU64Ser)
    }
}

/// Compute cluster peer info (received via gossipsub on COMPUTE_TUNNEL_TOPIC).
///
/// Each node periodically publishes its compute capabilities so that peers
/// can discover resources for distributed task routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputePeerInfo {
    /// Peer ID (libp2p PeerId as string)
    pub peer_id: String,
    /// Number of CPU cores currently available for compute tasks
    pub available_cores: u32,
    /// Total CPU cores on this machine
    pub total_cores: u32,
    /// GPU compute capability in TFLOPS (0.0 if no GPU)
    pub gpu_tflops: f64,
    /// RAM currently available in GB
    pub ram_available_gb: f64,
    /// Total RAM in GB
    pub ram_total_gb: f64,
    /// Network bandwidth in Mbps (estimated)
    pub bandwidth_mbps: f64,
    /// Current compute mode as string (e.g. "full", "eco", "nuke")
    pub compute_mode: String,
    /// Names of currently active compute layers
    pub active_layers: Vec<String>,
    /// Whether the trainer (cheat engine) is active
    pub trainer_active: bool,
    /// Software version string
    pub version: String,
    /// Unix timestamp in seconds when this announcement was created
    pub timestamp: u64,
}

/// Tunnel connection between two compute peers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelInfo {
    pub peer_id: String,
    pub tunnel_type: TunnelType,
    pub established_ms: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub tasks_routed: u64,
    pub latency_ms: u32,
    pub encrypted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TunnelType {
    /// Miner → Node: mining solutions + telemetry
    MinerToNode,
    /// Node → Node: task distribution + results
    NodeToNode,
    /// Node → Miner: push compute tasks to idle miner GPU
    NodeToMiner,
    /// Miner → Miner: collaborative proof generation
    MinerToMiner,
}

/// Full compute status for dashboard/API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeStatus {
    pub mode: ComputeMode,
    pub resources: ResourceSnapshot,
    pub layers: Vec<(String, LayerStats)>,
    pub tunnels: Vec<TunnelInfo>,
    pub cluster_peers: Vec<ComputePeerInfo>,
    pub trainer_active: bool,
    pub trainer_cheats: Vec<String>,
    pub performance_boost_pct: f32,
    pub total_revenue_micro_qug: u64,
    /// v9.6.0: AI inference statistics (populated when inference pool is active)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai_inference: Option<inference_pool::AIInferenceStats>,
}
