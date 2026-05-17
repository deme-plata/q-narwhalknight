//! Fault tolerance for distributed computation
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use futures::future::{self, Future};
use std::pin::Pin;
use tracing::{info, warn, error, debug, instrument};
use thiserror::Error;

/// Recovery error
#[derive(Debug, Error)]
pub enum RecoveryError {
    #[error("Task failed: {0}")]
    TaskFailed(String),
    
    #[error("All tasks failed")]
    AllTasksFailed,
    
    #[error("Timeout: {0}")]
    Timeout(String),
}

type Result<T> = std::result::Result<T, RecoveryError>;

/// Recovery manager for fault-tolerant computations
pub struct RecoveryManager {
    /// Node reliability ratings
    node_reliability: Arc<RwLock<HashMap<String, f64>>>,
    /// Failed nodes
    failed_nodes: Arc<RwLock<HashSet<String>>>,
    /// Recovery settings
    settings: Arc<RecoverySettings>,
}

/// Recovery settings
#[derive(Debug, Clone)]
pub struct RecoverySettings {
    /// Enable task replication
    pub enable_replication: bool,
    /// Replication factor (how many duplicate tasks to run)
    pub replication_factor: usize,
    /// Max retry attempts
    pub max_retries: usize,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Task timeout in seconds
    pub task_timeout_secs: u64,
}

impl Default for RecoverySettings {
    fn default() -> Self {
        Self {
            enable_replication: false,
            replication_factor: 1,
            max_retries: 3,
            retry_delay_ms: 500,
            task_timeout_secs: 60,
        }
    }
}

/// Node status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is healthy
    Healthy,
    /// Node is partially degraded
    Degraded,
    /// Node is unhealthy
    Unhealthy,
    /// Node is offline
    Offline,
}

impl RecoveryManager {
    /// Create a new recovery manager
    pub fn new(settings: RecoverySettings) -> Self {
        Self {
            node_reliability: Arc::new(RwLock::new(HashMap::new())),
            failed_nodes: Arc::new(RwLock::new(HashSet::new())),
            settings: Arc::new(settings),
        }
    }
    
    /// Execute tasks with recovery
    #[instrument(skip(self, tasks), fields(task_count = %tasks.len()))]
    pub async fn execute_with_recovery<T, E, F>(
        &self,
        tasks: Vec<F>,
    ) -> Result<Vec<T>>
    where
        T: Send + 'static,
        E: std::error::Error + Send + Sync + 'static,
        F: Future<Output = std::result::Result<T, E>> + Send + 'static,
    {
        let task_count = tasks.len();
        info!("Executing {} tasks with recovery", task_count);
        
        if task_count == 0 {
            return Ok(vec![]);
        }
        
        let settings = self.settings.clone();
        
        // Wrap each task with timeout and tracking
        let wrapped_tasks: Vec<_> = tasks.into_iter()
            .enumerate()
            .map(|(idx, task)| {
                let settings = settings.clone();
                async move {
                    let task_id = format!("task_{}", idx);
                    debug!("Starting {}", task_id);
                    
                    let start_time = Instant::now();
                    let result = tokio::time::timeout(
                        Duration::from_secs(settings.task_timeout_secs),
                        task
                    ).await;
                    
                    match result {
                        Ok(Ok(value)) => {
                            let elapsed = start_time.elapsed();
                            debug!("{} completed successfully in {:?}", task_id, elapsed);
                            Ok(value)
                        },
                        Ok(Err(e)) => {
                            error!("{} failed with error: {}", task_id, e);
                            Err(RecoveryError::TaskFailed(e.to_string()))
                        },
                        Err(_) => {
                            error!("{} timed out after {} seconds", task_id, settings.task_timeout_secs);
                            Err(RecoveryError::Timeout(task_id))
                        }
                    }
                }
            })
            .collect();
            
        // Execute all tasks with recovery
        let mut all_results = vec![];
        let mut any_succeeded = false;
        
        for (retry, results) in self.retry_execution(wrapped_tasks).await.into_iter().enumerate() {
            debug!("Retry {}: Received {} results", retry, results.len());
            
            for result in results {
                match result {
                    Ok(value) => {
                        all_results.push(Ok(value));
                        any_succeeded = true;
                    },
                    Err(e) => {
                        all_results.push(Err(e));
                    }
                }
            }
            
            // If we got successful results for all tasks, we're done
            if all_results.len() >= task_count && any_succeeded {
                break;
            }
        }
        
        // Return combined results (filter for success)
        let successful_results: Vec<_> = all_results.into_iter()
            .filter_map(|r| r.ok())
            .collect();
            
        if successful_results.is_empty() {
            error!("All tasks failed after retries");
            Err(RecoveryError::AllTasksFailed)
        } else {
            Ok(successful_results)
        }
    }
    
    /// Execute tasks with retries
    async fn retry_execution<T, F>(
        &self,
        tasks: Vec<F>,
    ) -> Vec<Vec<Result<T>>>
    where
        T: Send + 'static,
        F: Future<Output = Result<T>> + Send + 'static,
    {
        let max_retries = self.settings.max_retries;
        let mut results = Vec::with_capacity(max_retries + 1);
        let mut remaining_tasks = tasks;
        
        // First attempt
        let first_results = self.execute_batch(&remaining_tasks).await;
        results.push(first_results.clone());
        
        // Filter failed tasks for retry
        let mut failed_indices: Vec<usize> = first_results.iter()
            .enumerate()
            .filter_map(|(i, r)| if r.is_err() { Some(i) } else { None })
            .collect();
            
        // No failures, early return
        if failed_indices.is_empty() {
            return results;
        }
        
        // Retry failed tasks
        for retry in 0..max_retries {
            if failed_indices.is_empty() {
                break;
            }
            
            // Delay before retry
            tokio::time::sleep(Duration::from_millis(self.settings.retry_delay_ms)).await;
            
            info!("Retry {}/{}: Retrying {} failed tasks", 
                  retry + 1, max_retries, failed_indices.len());
                  
            // Extract failed tasks for retry
            let retry_tasks: Vec<_> = failed_indices.iter()
                .map(|&i| Box::pin(remaining_tasks[i].clone()) as Pin<Box<dyn Future<Output = Result<T>> + Send>>)
                .collect();
                
            // Execute retry batch
            let retry_results = self.execute_batch(&retry_tasks).await;
            results.push(retry_results.clone());
            
            // Update failed indices for next retry
            failed_indices = retry_results.iter()
                .enumerate()
                .filter_map(|(i, r)| if r.is_err() { Some(failed_indices[i]) } else { None })
                .collect();
        }
        
        results
    }
    
    /// Execute a batch of tasks
    async fn execute_batch<T, F>(
        &self,
        tasks: &[F],
    ) -> Vec<Result<T>>
    where
        T: Send + 'static,
        F: Future<Output = Result<T>> + Send + 'static,
    {
        let futures: Vec<_> = tasks.iter()
            .map(|task| task.clone())
            .collect();
            
        future::join_all(futures).await
    }
    
    /// Record node success
    pub async fn record_node_success(&self, node_id: &str) {
        let mut reliability = self.node_reliability.write().await;
        let current = reliability.get(node_id).copied().unwrap_or(0.5);
        
        // Increase reliability (with ceiling)
        let new_reliability = f64::min(1.0, current + 0.1);
        reliability.insert(node_id.to_string(), new_reliability);
        
        // Remove from failed nodes if present
        let mut failed = self.failed_nodes.write().await;
        failed.remove(node_id);
    }
    
    /// Record node failure
    pub async fn record_node_failure(&self, node_id: &str) {
        // Update reliability
        let mut reliability = self.node_reliability.write().await;
        let current = reliability.get(node_id).copied().unwrap_or(0.5);
        
        // Decrease reliability (with floor)
        let new_reliability = f64::max(0.0, current - 0.2);
        reliability.insert(node_id.to_string(), new_reliability);
        
        // Add to failed nodes if reliability drops too low
        if new_reliability < 0.3 {
            let mut failed = self.failed_nodes.write().await;
            failed.insert(node_id.to_string());
            
            warn!("Node {} marked as failed (reliability: {})", node_id, new_reliability);
        }
    }
    
    /// Check if a node is failed
    pub async fn is_node_failed(&self, node_id: &str) -> bool {
        let failed = self.failed_nodes.read().await;
        failed.contains(node_id)
    }
    
    /// Get node status
    pub async fn get_node_status(&self, node_id: &str) -> NodeStatus {
        let reliability = self.node_reliability.read().await;
        let failed = self.failed_nodes.read().await;
        
        if failed.contains(node_id) {
            return NodeStatus::Offline;
        }
        
        match reliability.get(node_id).copied().unwrap_or(0.5) {
            r if r >= 0.8 => NodeStatus::Healthy,
            r if r >= 0.5 => NodeStatus::Degraded,
            _ => NodeStatus::Unhealthy,
        }
    }
    
    /// Reset node status
    pub async fn reset_node(&self, node_id: &str) {
        let mut reliability = self.node_reliability.write().await;
        reliability.insert(node_id.to_string(), 0.5);
        
        let mut failed = self.failed_nodes.write().await;
        failed.remove(node_id);
        
        info!("Reset status for node {}", node_id);
    }
    
    /// Get healthiest nodes
    pub async fn get_healthiest_nodes(&self, count: usize) -> Vec<String> {
        let reliability = self.node_reliability.read().await;
        let failed = self.failed_nodes.read().await;
        
        let mut nodes: Vec<_> = reliability.iter()
            .filter(|(node_id, _)| !failed.contains(*node_id))
            .collect();
            
        // Sort by reliability (highest first)
        nodes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take requested count
        nodes.iter()
            .take(count)
            .map(|(node_id, _)| (*node_id).clone())
            .collect()
    }
}

impl Default for RecoveryManager {
    fn default() -> Self {
        Self::new(RecoverySettings::default())
    }
}
