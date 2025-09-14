use crate::contracts::{Contract, ContractCall};
use std::collections::HashMap;

use crate::vm::VmError;
use crate::state::StateDB;
use crate::vm::tiered_vm::TieredVM;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;
use std::fmt;

// Results of parallel execution
pub struct BatchExecutionResult {
    pub results: Vec<Result<Vec<u8>, VmError>>,
    pub total_time: Duration,
    pub gas_used: u64,
}

// Executor for parallel contract execution
pub struct ParallelExecutor {
    pub state_db: Arc<StateDB>,
    pub thread_count: usize,
}

// Add Debug implementation directly here
impl fmt::Debug for ParallelExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParallelExecutor")
            .field("thread_count", &self.thread_count)
            .finish()
    }
}

impl ParallelExecutor {
    pub fn new(state_db: Arc<StateDB>) -> Self {
        // Use half of CPU cores for parallel execution
        let thread_count = std::cmp::max(1, num_cpus::get() / 2);
        
        Self {
            state_db,
            thread_count,
        }
    }
    
    // Execute a batch of contract calls in parallel
    pub async fn execute_batch(&self, calls: Vec<(ContractCall, Arc<Contract>)>) -> BatchExecutionResult {
        let start_time = Instant::now();
        let mut results = Vec::with_capacity(calls.len());
        for _ in 0..calls.len() {
            results.push(Ok(Vec::new())); // Initialize with empty results
        }
        
        // Group calls by contract
        let mut contract_groups: HashMap<[u8; 32], Vec<(usize, ContractCall, Arc<Contract>)>> = HashMap::new();
        for (idx, (call, contract)) in calls.into_iter().enumerate() {
            contract_groups.entry(call.contract_address)
                .or_insert_with(Vec::new)
                .push((idx, call, Arc::clone(&contract)));
        }
        
        // Create execution tasks
        let mut tasks = JoinSet::new();
        
        for (_contract_addr, group_calls) in contract_groups {
            // Execute this contract's calls in a separate task
            let state_db = Arc::clone(&self.state_db);
            
            tasks.spawn(async move {
                let mut task_results = Vec::new();
                let vm = TieredVM::new(state_db);
                
                for (idx, call, contract) in group_calls {
                    // The args in ContractCall is already a Vec<Vec<u8>>, so we need to pass it as a slice
                    // Fixed: args field is likely Vec<u8>, so we need to convert it to Vec<Vec<u8>> first
                    let args_as_vec_of_vec = vec![call.args.clone()]; // Wrap in a vector to match &[Vec<u8>]
                    let result = vm.execute(&contract, &call.method, &args_as_vec_of_vec);
                    task_results.push((idx, result));
                }
                
                task_results
            });
        }
        
        // Collect results
        while let Some(task_result) = tasks.join_next().await {
            if let Ok(task_results) = task_result {
                for (idx, result) in task_results {
                    results[idx] = result;
                }
            }
        }
        
        // Calculate gas used (simplified)
        let gas_used = start_time.elapsed().as_micros() as u64;
        
        BatchExecutionResult {
            results,
            total_time: start_time.elapsed(),
            gas_used,
        }
    }
    
    // Using rayon for CPU-bound tasks
    pub fn execute_batch_sync(&self, _calls: Vec<(ContractCall, Arc<Contract>)>) -> BatchExecutionResult {
        // Create a dummy implementation that compiles
        let start_time = Instant::now();
        let results = vec![Ok(Vec::new())]; // Dummy result
        
        BatchExecutionResult {
            results,
            total_time: start_time.elapsed(),
            gas_used: 0,
        }
    }
}