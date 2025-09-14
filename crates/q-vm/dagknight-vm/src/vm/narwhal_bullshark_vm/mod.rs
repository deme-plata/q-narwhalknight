use std::sync::Arc;
use std::time::Instant;
use crate::vm::{VirtualMachine, VmError};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use serde_big_array::BigArray;

// Define Transaction type locally if not available elsewhere
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: String,
    pub data: Vec<u8>,
}

pub type NodeId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartContractTx {
    pub address: u64,
    pub function: String,
    pub arguments: Vec<u8>,
    pub sender: u64,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub nonce: u64,
    pub value: u64,
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
}

pub struct NarwhalBullsharkVm {
    _node_id: NodeId,
    _peers: Vec<NodeId>,
    _vm: Arc<VirtualMachine>,
    // Add explicit TPS tracking fields
    tx_count: Arc<RwLock<u64>>,
    start_time: Arc<RwLock<Instant>>,
    // Other fields would go here in a real implementation
}

impl NarwhalBullsharkVm {
    pub fn new(node_id: NodeId, peers: Vec<NodeId>, vm: Arc<VirtualMachine>) -> Self {
        Self {
            _node_id: node_id,
            _peers: peers,
            _vm: vm,
            // Initialize TPS tracking
            tx_count: Arc::new(RwLock::new(0)),
            start_time: Arc::new(RwLock::new(Instant::now())),
        }
    }
    
    pub async fn start(&self) -> Result<(), VmError> {
        println!("Starting NarwhalBullshark VM...");
        // Reset TPS counter when starting
        self.reset_tps_counter().await?;
        Ok(())
    }
    
    pub async fn stop(&self) -> Result<(), VmError> {
        println!("Stopping NarwhalBullshark VM...");
        Ok(())
    }
    
    pub async fn submit_transaction(&self, _tx: SmartContractTx) -> Result<[u8; 32], VmError> {
        // Increment transaction count for TPS calculation
        {
            let mut count = self.tx_count.write().await;
            *count += 1;
        }
        
        // Return a dummy hash for testing
        Ok([0; 32])
    }
    
    pub async fn get_tps(&self) -> f64 {
        // Get transaction count and elapsed time
        let count = *self.tx_count.read().await;
        let start = *self.start_time.read().await;
        let elapsed = start.elapsed().as_secs_f64();
        
        // Debug output to verify values
        println!("DEBUG - TX count: {}, elapsed: {:.2}s", count, elapsed);
        
        // Avoid division by zero
        if elapsed <= 0.001 {  // Using a small threshold instead of exact zero
            return 0.0;
        }
        
        // Calculate and return actual TPS
        count as f64 / elapsed
    }
    
    // New method to provide detailed TPS metrics for debugging
    pub async fn get_detailed_tps(&self) -> (u64, f64, f64) {
        let count = *self.tx_count.read().await;
        let start = *self.start_time.read().await;
        let elapsed = start.elapsed().as_secs_f64();
        
        let tps = if elapsed <= 0.0 { 0.0 } else { count as f64 / elapsed };
        
        (count, elapsed, tps)
    }
    
    // Reset TPS counter for fresh measurements
    pub async fn reset_tps_counter(&self) -> Result<(), VmError> {
        {
            let mut count = self.tx_count.write().await;
            *count = 0;
        }
        {
            let mut start = self.start_time.write().await;
            *start = Instant::now();
        }
        println!("TPS counter reset successfully");
        Ok(())
    }
}

// Config functions for the benchmarking tool
pub mod config {
    pub fn load_config(config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("Loading config from {}", config_path);
        Ok(())
    }
    
    pub fn update_batch_size(batch_size: usize) {
        println!("Updating batch size to {}", batch_size);
    }
}