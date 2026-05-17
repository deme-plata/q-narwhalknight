//! AI model execution engine
use crate::contracts::{AIModelCall, ShardingCapability};
use crate::state::ResourceUsage;
use crate::cache::ModelCache;
use crate::models::ModelRegistry;
use crate::fault_tolerance::RecoveryManager;
use ollama_rs::Ollama;
use ollama_rs::generation::completion::{request::GenerationRequest, GenerationContext};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::error::Error;
use tracing::{info, warn, error, debug, instrument};

/// Error types for AI execution
#[derive(Debug, thiserror::Error)]
pub enum AIExecutionError {
    #[error("Ollama error: {0}")]
    OllamaError(String),
    
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Computation timeout")]
    Timeout,
    
    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),
    
    #[error("Shard allocation failed: {0}")]
    ShardAllocationFailed(String),
    
    #[error("Node failure: {0}")]
    NodeFailure(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

type Result<T> = std::result::Result<T, AIExecutionError>;

/// AI execution engine
pub struct AIExecutor {
    /// Ollama client
    ollama: Ollama,
    /// GPU availability
    gpu_enabled: bool,
    /// Model cache
    cache: Arc<ModelCache>,
    /// Model registry
    registry: Arc<ModelRegistry>,
    /// Recovery manager
    recovery: Arc<RecoveryManager>,
    /// Performance metrics per model
    performance_metrics: Arc<Mutex<HashMap<String, Vec<ExecutionMetrics>>>>,
}

/// Execution metrics for learning
struct ExecutionMetrics {
    model: String,
    shard_count: u64,
    input_size: usize,
    execution_time: Duration,
    memory_used: u64,
    success: bool,
}

impl AIExecutor {
    /// Create a new AI executor
    pub async fn new(
        cache: Arc<ModelCache>,
        registry: Arc<ModelRegistry>,
        recovery: Arc<RecoveryManager>,
    ) -> Result<Self> {
        let ollama = Ollama::default();
        
        // Check if Ollama is running
        match ollama.list_local_models().await {
            Ok(_) => info!("Connected to Ollama service"),
            Err(e) => {
                error!("Failed to connect to Ollama: {}", e);
                return Err(AIExecutionError::OllamaError(e.to_string()));
            }
        }
        
        // Try to check for GPU
        let gpu_enabled = match ollama.generate("nous-hermes:1b", "Are you using GPU?").await {
            Ok(_) => {
                info!("GPU acceleration available");
                true
            },
            Err(e) => {
                warn!("GPU acceleration unavailable: {}", e);
                false
            }
        };
        
        Ok(Self {
            ollama,
            gpu_enabled,
            cache,
            registry,
            recovery,
            performance_metrics: Arc::new(Mutex::new(HashMap::new())),
        })
    }
    
    /// Execute an AI model
    #[instrument(skip(self, model_call), fields(model = %model_call.model))]
    pub async fn execute(
        &self,
        model_call: &AIModelCall,
        contract_address: [u8; 32],
    ) -> Result<(Vec<u8>, ResourceUsage)> {
        // Check model registry
        let model_info = self.registry.get_model(&model_call.model).await
            .ok_or_else(|| AIExecutionError::ModelNotFound(model_call.model.clone()))?;
        
        // Check cache based on policy
        if let Some(cached_result) = match model_call.cache_policy {
            crate::contracts::CachePolicy::NoCache => None,
            crate::contracts::CachePolicy::UseCache(ttl) => {
                self.cache.get(&model_call.model, &model_call.input, ttl).await
            },
            crate::contracts::CachePolicy::ForceRefresh => None,
        } {
            info!("Cache hit for model {}", model_call.model);
            return Ok((cached_result, ResourceUsage::minimal()));
        }
        
        info!("Cache miss for model {}", model_call.model);
        
        // Check if model supports sharding
        let sharding_strategy = match model_info.sharding_capability {
            ShardingCapability::None => {
                if model_call.shard_count > 1 {
                    warn!("Model {} doesn't support sharding but shard_count is {}", 
                          model_call.model, model_call.shard_count);
                }
                ShardingStrategy::None
            },
            ShardingCapability::Horizontal => ShardingStrategy::Horizontal,
            ShardingCapability::Vertical => ShardingStrategy::Vertical,
            ShardingCapability::Full => {
                // Choose best strategy based on historical performance
                self.determine_best_strategy(&model_call.model, model_call.input.len()).await
            }
        };
        
        // Prepare execution
        let start_time = Instant::now();
        let result: Result<Vec<u8>>;
        let nodes_used: u64;
        
        // Execute based on sharding strategy
        match sharding_strategy {
            ShardingStrategy::None => {
                debug!("Executing model {} without sharding", model_call.model);
                result = self.execute_single(model_call).await;
                nodes_used = 1;
            },
            ShardingStrategy::Horizontal => {
                let shard_count = self.calculate_optimal_shard_count(
                    &model_call.model, 
                    model_call.input.len()
                ).await;
                
                debug!("Executing model {} with horizontal sharding (shards: {})", 
                       model_call.model, shard_count);
                       
                result = self.execute_horizontal_sharded(model_call, shard_count).await;
                nodes_used = shard_count;
            },
            ShardingStrategy::Vertical => {
                debug!("Executing model {} with vertical sharding", model_call.model);
                result = self.execute_vertical_sharded(model_call).await;
                nodes_used = model_call.shard_count;
            }
        }
        
        let execution_time = start_time.elapsed();
        
        // Record metrics for future optimization
        self.record_execution_metrics(
            &model_call.model,
            nodes_used,
            model_call.input.len(),
            execution_time,
            result.is_ok(),
        ).await;
        
        // Calculate resource usage
        let usage = ResourceUsage {
            cpu_time: execution_time.as_millis() as u64,
            memory_used: estimate_memory_usage(&model_call.model, model_call.input.len()),
            gpu_time: if self.gpu_enabled { 
                Some(execution_time.as_millis() as u64) 
            } else { 
                None 
            },
        };
        
        // Get the result or handle error
        let output = match result {
            Ok(output) => {
                // Update cache if successful
                if let crate::contracts::CachePolicy::UseCache(ttl) = model_call.cache_policy {
                    self.cache.set(&model_call.model, &model_call.input, &output, ttl).await;
                }
                output
            },
            Err(e) => return Err(e),
        };
        
        Ok((output, usage))
    }
    
    /// Execute model on a single node
    async fn execute_single(&self, model_call: &AIModelCall) -> Result<Vec<u8>> {
        let input_str = String::from_utf8_lossy(&model_call.input).to_string();
        
        // Create the generation request
        let req = GenerationRequest::new(model_call.model.clone(), input_str);
        
        // Execute with timeout protection
        let result = tokio::time::timeout(
            Duration::from_secs(120), // 2 minute timeout
            self.ollama.generate(req)
        ).await;
        
        match result {
            Ok(Ok(response)) => {
                Ok(response.response.into_bytes())
            },
            Ok(Err(e)) => {
                error!("Ollama execution error: {}", e);
                Err(AIExecutionError::OllamaError(e.to_string()))
            },
            Err(_) => {
                error!("Execution timed out");
                Err(AIExecutionError::Timeout)
            }
        }
    }
    
    /// Execute model with horizontal sharding (input splitting)
    async fn execute_horizontal_sharded(
        &self, 
        model_call: &AIModelCall,
        shard_count: u64
    ) -> Result<Vec<u8>> {
        // Split input
        let input_chunks = split_input(&model_call.input, shard_count as usize);
        
        // Prepare tasks
        let mut tasks = Vec::with_capacity(input_chunks.len());
        
        for chunk in input_chunks {
            let ollama = self.ollama.clone();
            let model = model_call.model.clone();
            
            let task = tokio::spawn(async move {
                let input_str = String::from_utf8_lossy(&chunk).to_string();
                let req = GenerationRequest::new(model, input_str);
                
                match ollama.generate(req).await {
                    Ok(response) => Ok(response.response.into_bytes()),
                    Err(e) => Err(AIExecutionError::OllamaError(e.to_string())),
                }
            });
            
            tasks.push(task);
        }
        
        // Set up fault tolerance
        let recovery = self.recovery.clone();
        
        // Wait for results with recovery
        let results = self.recovery.execute_with_recovery(tasks).await?;
        
        // Merge results
        let merged = merge_outputs(results)?;
        
        Ok(merged)
    }
    
    /// Execute model with vertical sharding (model splitting)
    async fn execute_vertical_sharded(&self, model_call: &AIModelCall) -> Result<Vec<u8>> {
        // For this example, we'll simulate vertical sharding
        // In a real implementation, this would distribute model layers across nodes
        
        warn!("Vertical sharding is simulated in this implementation");
        
        // Fallback to single execution for this demo
        self.execute_single(model_call).await
    }
    
    /// Determine best sharding strategy based on historical performance
    async fn determine_best_strategy(&self, model: &str, input_size: usize) -> ShardingStrategy {
        let metrics = self.performance_metrics.lock().await;
        
        // If we don't have enough data, default to horizontal
        if !metrics.contains_key(model) {
            return ShardingStrategy::Horizontal;
        }
        
        // Find similar workloads
        let model_metrics = metrics.get(model).unwrap();
        let similar_workloads: Vec<_> = model_metrics.iter()
            .filter(|m| (m.input_size as f64 * 0.8..=m.input_size as f64 * 1.2).contains(&(input_size as f64)))
            .collect();
            
        if similar_workloads.is_empty() {
            return ShardingStrategy::Horizontal;
        }
        
        // Count successes for each strategy
        let horizontal_success = similar_workloads.iter()
            .filter(|m| m.shard_count > 1 && m.success)
            .count();
            
        let vertical_success = similar_workloads.iter()
            .filter(|m| m.shard_count > 1 && m.success)
            .count();
            
        // Choose the more successful strategy
        if horizontal_success >= vertical_success {
            ShardingStrategy::Horizontal
        } else {
            ShardingStrategy::Vertical
        }
    }
    
    /// Calculate optimal shard count based on historical performance
    async fn calculate_optimal_shard_count(&self, model: &str, input_size: usize) -> u64 {
        let metrics = self.performance_metrics.lock().await;
        
        if !metrics.contains_key(model) {
            return 4; // Default to 4 shards if no data
        }
        
        let model_metrics = metrics.get(model).unwrap();
        
        // Find metrics for similar workloads
        let similar_workloads: Vec<_> = model_metrics.iter()
            .filter(|m| (m.input_size as f64 * 0.8..=m.input_size as f64 * 1.2).contains(&(input_size as f64)))
            .filter(|m| m.success) // Only consider successful executions
            .collect();
            
        if similar_workloads.is_empty() {
            return 4; // Default if no similar workloads
        }
        
        // Group by shard count and calculate average execution time
        let mut shard_performance: HashMap<u64, (Duration, usize)> = HashMap::new();
        
        for metric in similar_workloads {
            let entry = shard_performance.entry(metric.shard_count).or_insert((Duration::from_secs(0), 0));
            entry.0 += metric.execution_time;
            entry.1 += 1;
        }
        
        // Find shard count with lowest average execution time
        shard_performance.iter()
            .map(|(shard_count, (total_time, count))| {
                let avg_time = total_time.div_f64(*count as f64);
                (*shard_count, avg_time)
            })
            .min_by_key(|(_, time)| time.as_millis() as u64)
            .map(|(shard_count, _)| shard_count)
            .unwrap_or(4) // Default if comparison fails
    }
    
    /// Record metrics for future optimization
    async fn record_execution_metrics(
        &self,
        model: &str,
        shard_count: u64,
        input_size: usize,
        execution_time: Duration,
        success: bool,
    ) {
        let metric = ExecutionMetrics {
            model: model.to_string(),
            shard_count,
            input_size,
            execution_time,
            memory_used: estimate_memory_usage(model, input_size),
            success,
        };
        
        let mut metrics = self.performance_metrics.lock().await;
        
        // Add to metrics history
        metrics.entry(model.to_string())
            .or_insert_with(Vec::new)
            .push(metric);
            
        // Keep only the last 100 metrics per model
        if let Some(model_metrics) = metrics.get_mut(model) {
            if model_metrics.len() > 100 {
                model_metrics.sort_by_key(|m| m.execution_time);
                model_metrics.truncate(100);
            }
        }
    }
}

/// Sharding strategy
enum ShardingStrategy {
    /// No sharding
    None,
    /// Horizontal sharding (input splitting)
    Horizontal,
    /// Vertical sharding (model splitting)
    Vertical,
}

/// Split input data into chunks
fn split_input(input: &[u8], chunk_count: usize) -> Vec<Vec<u8>> {
    if chunk_count <= 1 {
        return vec![input.to_vec()];
    }
    
    let chunk_size = (input.len() / chunk_count) + 1;
    let mut chunks = Vec::with_capacity(chunk_count);
    
    for i in 0..chunk_count {
        let start = i * chunk_size;
        if start >= input.len() {
            break;
        }
        
        let end = (start + chunk_size).min(input.len());
        chunks.push(input[start..end].to_vec());
    }
    
    chunks
}

/// Merge outputs from multiple chunks
fn merge_outputs(outputs: Vec<Result<Vec<u8>>>) -> Result<Vec<u8>> {
    // Process any errors
    for output in &outputs {
        if let Err(e) = output {
            return Err(e.clone());
        }
    }
    
    // Combine successful outputs
    let mut result = Vec::new();
    for output in outputs {
        if let Ok(data) = output {
            result.extend_from_slice(&data);
        }
    }
    
    Ok(result)
}

/// Estimate memory usage based on model and input size
fn estimate_memory_usage(model: &str, input_size: usize) -> u64 {
    // This is a simplified estimation - a real implementation would have more sophisticated logic
    let base_memory = match model.split(':').next().unwrap_or("unknown") {
        "llama2" => 4000,
        "deepseek" => 6000,
        "mistral" => 8000,
        "phi" => 3000,
        _ => 5000, // Default for unknown models
    };
    
    // Scale with input size (simplified)
    base_memory + (input_size as u64 / 100)
}
