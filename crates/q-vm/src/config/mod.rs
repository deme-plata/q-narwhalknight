//! Configuration management for the VM

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::RwLock;

// VM Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmConfig {
    pub narwhal_bullshark: NarwhalBullsharkConfig,
}

// Narwhal-Bullshark specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarwhalBullsharkConfig {
    pub batch_size: usize,
    pub max_parallel_executions: usize,
    pub max_mempool_size: usize,
    pub block_production_interval_ms: u64,
    pub transaction_timeout_ms: u64,
    pub enable_metrics: bool,
    pub metrics_interval_seconds: u64,
}

impl Default for VmConfig {
    fn default() -> Self {
        Self {
            narwhal_bullshark: NarwhalBullsharkConfig::default(),
        }
    }
}

impl Default for NarwhalBullsharkConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            max_parallel_executions: 4,
            max_mempool_size: 100000,
            block_production_interval_ms: 1000,
            transaction_timeout_ms: 5000,
            enable_metrics: true,
            metrics_interval_seconds: 10,
        }
    }
}

// Singleton configuration
lazy_static::lazy_static! {
    static ref CONFIG: RwLock<VmConfig> = RwLock::new(VmConfig::default());
}

// Load configuration from file
pub fn load_config(path: impl AsRef<Path>) -> Result<(), String> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(format!("Configuration file not found: {}", path.display()));
    }

    let content =
        fs::read_to_string(path).map_err(|e| format!("Failed to read config file: {}", e))?;

    let config: VmConfig =
        toml::from_str(&content).map_err(|e| format!("Failed to parse config file: {}", e))?;

    // Update global config
    let mut global_config = CONFIG.write().unwrap();
    *global_config = config;

    Ok(())
}

// Get configuration
pub fn get_config() -> VmConfig {
    CONFIG.read().unwrap().clone()
}

// Get Narwhal-Bullshark configuration
pub fn get_narwhal_bullshark_config() -> NarwhalBullsharkConfig {
    CONFIG.read().unwrap().narwhal_bullshark.clone()
}

// Update batch size
pub fn update_batch_size(batch_size: usize) {
    let mut config = CONFIG.write().unwrap();
    config.narwhal_bullshark.batch_size = batch_size;
}
