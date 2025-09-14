//! DAGKnight blockchain with distributed AI capabilities
use std::sync::Arc;
use tokio::signal;
use tracing::info;

mod api;
mod cache;
mod consensus;
mod contracts;
mod error;
mod fault_tolerance;
mod models;
mod network;
mod state;
mod vm;
mod config;

use crate::cache::{ModelCache, CacheProvider};
use crate::fault_tolerance::{RecoveryManager, RecoverySettings};
use crate::models::ModelRegistry;
use crate::state::StateDB;
use crate::vm::ai::executor::AIExecutor;
use crate::vm::cache::ContractCache;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("Starting DAGKnight with distributed AI capabilities");
    
    // Initialize core components
    let registry = Arc::new(ModelRegistry::new());
    let cache = Arc::new(ModelCache::new(
        CacheProvider::Layered,
        50000,
        Some("redis://localhost:6379".to_string()),
    ));
    
    let recovery_settings = RecoverySettings {
        enable_replication: true,
        replication_factor: 2,
        max_retries: 3,
        retry_delay_ms: 500,
        task_timeout_secs: 120,
    };
    
    let _recovery = Arc::new(RecoveryManager::new(recovery_settings));
    let _state_db = Arc::new(StateDB::new());
    
    // Initialize model registry with defaults
    registry.initialize_defaults().await;
    
    // Initialize contract cache for VM
    let contract_cache = Arc::new(ContractCache::new());
    
    // Initialize AI executor with contract cache instead of model cache
    let _ai_executor = AIExecutor::new(
        contract_cache,
    ).await.unwrap();
    
    // Start cache maintenance
    cache.start_cleanup_task();
    
    info!("System initialized and ready");
    
    // Wait for shutdown signal
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Shutdown signal received, stopping DAGKnight");
        },
        Err(err) => {
            eprintln!("Unable to listen for shutdown signal: {}", err);
        },
    }
    
    Ok(())
}