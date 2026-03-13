pub mod pool_client;
pub mod stratum;
pub mod tor_proxy;

pub use pool_client::PoolClient;
pub use stratum::{StratumClient, StratumMessage, StratumMethod};
pub use tor_proxy::TorProxyManager;

use crate::{WorkUnit, Solution, MiningEvent};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, debug};
use uuid::Uuid;
use rand::Rng;

/// Parse Stratum mining.notify message into WorkUnit
fn parse_mining_work(message: &StratumMessage) -> Result<WorkUnit> {
    // Parse Stratum parameters into Q-NarwhalKnight work format
    let params = message.params.as_array()
        .ok_or_else(|| anyhow::anyhow!("Invalid mining.notify params"))?;
    
    if params.len() < 8 {
        return Err(anyhow::anyhow!("Insufficient mining.notify parameters"));
    }
    
    let job_id = params[0].as_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid job_id"))?
        .to_string();
    
    let previous_hash_str = params[1].as_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid previous_hash"))?;
    
    let merkle_root_str = params[2].as_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid merkle_root"))?;
    
    // Convert hex strings to bytes
    let previous_hash = hex_to_bytes32(previous_hash_str)?;
    let merkle_root = hex_to_bytes32(merkle_root_str)?;
    
    // Default difficulty target (will be updated by mining.set_difficulty)
    let difficulty_target = [0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                           0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                           0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                           0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
    
    Ok(WorkUnit {
        job_id,
        previous_hash,
        merkle_root,
        timestamp: chrono::Utc::now().timestamp() as u64,
        difficulty_target,
        nonce_range: (0, u64::MAX),
        extra_data: Vec::new(),
    })
}

fn hex_to_bytes32(hex_str: &str) -> Result<[u8; 32]> {
    let hex_clean = hex_str.trim_start_matches("0x");
    let bytes = hex::decode(hex_clean)?;
    
    if bytes.len() != 32 {
        return Err(anyhow::anyhow!("Invalid hash length: expected 32 bytes, got {}", bytes.len()));
    }
    
    let mut result = [0u8; 32];
    result.copy_from_slice(&bytes);
    Ok(result)
}

/// Pool discovery and selection
pub struct PoolDiscovery {
    known_pools: Vec<PoolInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolInfo {
    pub name: String,
    pub url: String,
    pub location: String,
    pub fee_percentage: f64,
    pub min_payout: f64,
    pub supports_tor: bool,
    pub stratum_version: String,
    pub last_block_time: Option<chrono::DateTime<chrono::Utc>>,
    pub active_miners: u32,
    pub pool_hash_rate: f64,
}

impl PoolDiscovery {
    pub fn new() -> Self {
        Self {
            known_pools: vec![
                PoolInfo {
                    name: "Quillon Official Pool".to_string(),
                    url: "stratum+tor://pool.qnarwhal.onion:4444".to_string(),
                    location: "Anonymous".to_string(),
                    fee_percentage: 1.0,
                    min_payout: 0.01,
                    supports_tor: true,
                    stratum_version: "2.0".to_string(),
                    last_block_time: None,
                    active_miners: 1250,
                    pool_hash_rate: 1.2e12, // 1.2 TH/s
                },
                PoolInfo {
                    name: "Quantum Miners United".to_string(),
                    url: "stratum+tor://qmu.onion:3333".to_string(),
                    location: "Distributed".to_string(),
                    fee_percentage: 0.5,
                    min_payout: 0.005,
                    supports_tor: true,
                    stratum_version: "2.0".to_string(),
                    last_block_time: None,
                    active_miners: 890,
                    pool_hash_rate: 8.5e11, // 850 GH/s
                },
                PoolInfo {
                    name: "Anonymous Hash Collective".to_string(),
                    url: "stratum+tor://ahc.onion:5555".to_string(),
                    location: "Global".to_string(),
                    fee_percentage: 0.75,
                    min_payout: 0.001,
                    supports_tor: true,
                    stratum_version: "2.0".to_string(),
                    last_block_time: None,
                    active_miners: 2100,
                    pool_hash_rate: 2.1e12, // 2.1 TH/s
                },
            ],
        }
    }
    
    pub async fn discover_pools(&mut self) -> Result<Vec<PoolInfo>> {
        info!("🔍 Discovering Quillon mining pools...");
        
        // In production, this would query the network for active pools
        // For now, return known pools with updated stats
        
        for pool in &mut self.known_pools {
            // Simulate pool stats update
            let mut rng = rand::thread_rng();
            let minutes_ago = (rng.gen::<u64>() % 15) as i64;
            pool.last_block_time = Some(chrono::Utc::now() - chrono::Duration::minutes(minutes_ago));
            pool.active_miners = (pool.active_miners as i32 + (rng.gen::<i32>() % 20 - 10)).max(0) as u32;
            pool.pool_hash_rate *= 0.95 + (rng.gen::<f64>() * 0.1); // ±5% variance
        }
        
        info!("✅ Found {} active mining pools", self.known_pools.len());
        Ok(self.known_pools.clone())
    }
    
    pub fn get_recommended_pool(&self) -> Option<&PoolInfo> {
        // Recommend pool with best efficiency (low fee, high hash rate)
        self.known_pools.iter()
            .filter(|pool| pool.supports_tor)
            .min_by(|a, b| {
                let a_score = a.fee_percentage / (a.pool_hash_rate / 1e12);
                let b_score = b.fee_percentage / (b.pool_hash_rate / 1e12);
                a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}