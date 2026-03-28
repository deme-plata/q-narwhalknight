//! Miner Link Protocol - WebSocket relay between wallet and personal miner
//!
//! Defines the bidirectional message protocol used by:
//! - The miner binary (connects as role=miner)
//! - The API server relay hub (bridges messages)
//! - The browser wallet (connects as role=wallet)

use serde::{Deserialize, Serialize};

/// GPU device info sent from miner to wallet via MinerLink
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub vendor: String,
    pub compute_units: u32,
    pub memory_mb: u64,
    pub max_clock_mhz: u32,
    pub api: String,
}

/// All messages exchanged over the miner link WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MinerLinkMessage {
    /// Miner → Server: Register this miner for a wallet
    Register {
        wallet: String,
        miner_id: String,
        miner_name: Option<String>,
    },

    /// Miner → Wallet: Periodic stats update (every 1s)
    Stats {
        miner_id: String,
        hashrate: f64,
        total_hashes: u64,
        solutions: u64,
        blocks_found: u64,
        uptime_secs: u64,
        threads_active: u32,
        threads_total: u32,
        cpu_vendor: String,
        has_avx2: bool,
        has_avx512: bool,
        intensity: u8,
        is_mining: bool,
        current_block_height: u64,
        temperature_estimate: Option<f64>,
        // v10.2.1: GPU miner fields
        #[serde(default)]
        gpu_active: bool,
        #[serde(default)]
        gpu_hashrate: f64,
        #[serde(default)]
        gpu_devices: Vec<GpuDeviceInfo>,
    },

    /// Miner → Wallet: A mining solution was found
    SolutionFound {
        miner_id: String,
        block_height: u64,
        nonce: u64,
        hash_preview: String,
    },

    /// Wallet → Miner: Send a control command
    Command {
        command_id: String,
        action: MinerCommand,
    },

    /// Miner → Wallet: Acknowledge a command
    Ack {
        command_id: String,
        success: bool,
        message: String,
    },

    /// Server → Client: Link established confirmation
    LinkEstablished {
        peer_type: String,
        connected_miners: u32,
        connected_wallets: u32,
    },

    /// Wallet → Server: Auth token for wallet identity
    Auth {
        token: String,
    },

    /// Keepalive
    Ping,
    Pong,
}

/// Commands that can be sent from wallet to miner
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum MinerCommand {
    Pause,
    Resume,
    SetThreads { count: u32 },
    SetIntensity { level: u8 },
    GetDetailedStats,
}
