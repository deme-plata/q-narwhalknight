//! DAGKnight blockchain with distributed AI capabilities and integrated consensus
use std::sync::Arc;
use tokio::signal;
use tracing::info;

mod api;
mod cache;
mod config;
mod consensus;
mod contracts;
mod dag;
mod dag_integration; // Add DAG integration
mod error;
mod fault_tolerance;
mod models;
mod network;
mod state;
mod vm;

use crate::cache::{CacheProvider, ModelCache};
use crate::dag_integration::VMIntegratedDAG;
use crate::fault_tolerance::{RecoveryManager, RecoverySettings};
use crate::models::ModelRegistry;
use crate::state::StateDB;
use crate::vm::ai::executor::AIExecutor;
use crate::vm::cache::ContractCache;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("Starting DAGKnight with integrated consensus and distributed AI capabilities");

    // Initialize integrated DAG-VM system
    let node_id = [1u8; 32]; // TODO: Get from config or generate
    let f = 1; // Byzantine fault tolerance (2f+1 = total nodes)
    let state_db_path = "./vm_state";

    let integrated_dag = Arc::new(VMIntegratedDAG::new(node_id, f, state_db_path).await?);

    // Start the integrated system
    integrated_dag.start().await?;

    // Initialize additional AI components
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

    // Initialize model registry with defaults
    registry.initialize_defaults().await;

    // Initialize contract cache for VM
    let contract_cache = Arc::new(ContractCache::new());

    // Initialize AI executor with contract cache
    let _ai_executor = AIExecutor::new(contract_cache).await.unwrap();

    // Start cache maintenance
    cache.start_cleanup_task();

    // Display system status
    let status = integrated_dag.get_integrated_status().await?;
    info!("Integrated DAG-VM System Status:");
    info!("  Current Round: {}", status.consensus_status.current_round);
    info!(
        "  VM Metrics: {} contracts deployed, {} calls executed",
        status.vm_metrics.total_contracts_deployed, status.vm_metrics.total_contract_calls
    );

    info!("System initialized and ready - DAG consensus with VM execution active");

    // Wait for shutdown signal
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Shutdown signal received, stopping DAGKnight");
        }
        Err(err) => {
            eprintln!("Unable to listen for shutdown signal: {}", err);
        }
    }

    Ok(())
}
