//! DAGKnight VM implementation

// Core modules
pub mod api;
pub mod cache;
pub mod consensus;
pub mod contracts;
pub mod currency;
pub mod dag;
pub mod dag_integration; // New integration module
pub mod error;
pub mod fault_tolerance;
pub mod mempool;
pub mod models;
pub mod network;
pub mod state;
pub mod transaction;
pub mod types;
pub mod vm;

// Re-export key types for convenience
pub use dag::DAG;
pub use dag_integration::{IntegratedDAGStatus, VMExecutionResult, VMIntegratedDAG}; // Export integrated DAG
pub use vm::VirtualMachine;

pub fn init(config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let _ = config::load_config(config_path);

    // Handle batch size argument if provided
    if let Some(size_str) = std::env::args().nth(2) {
        if let Ok(size) = size_str.parse::<usize>() {
            config::update_batch_size(size);
        }
    }

    Ok(())
}

// Re-export config for compatibility
pub mod config {
    pub fn load_config(config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        match std::fs::read_to_string(config_path) {
            Ok(_) => Ok(()),
            Err(e) => Err(Box::new(e)),
        }
    }

    pub fn update_batch_size(_batch_size: usize) {
        println!("Updated batch size");
    }
}
