#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log function
log() {
  echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
  echo -e "${RED}[ERROR] $1${NC}"
  exit 1
}

success() {
  echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warning() {
  echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check for dependencies
check_dependencies() {
  log "Checking dependencies..."
  
  local deps=("cargo" "git" "jq" "curl" "docker")
  local missing=()
  
  for dep in "${deps[@]}"; do
    if ! command -v "$dep" &> /dev/null; then
      missing+=("$dep")
    fi
  done
  
  if [ ${#missing[@]} -ne 0 ]; then
    error "Missing dependencies: ${missing[*]}"
  fi
  
  # Check if Ollama is installed
  if ! command -v ollama &> /dev/null; then
    warning "Ollama not found. Will attempt to install."
    install_ollama
  fi

  success "All dependencies found."
}

# Install Ollama if not present
install_ollama() {
  log "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  
  if [ $? -ne 0 ]; then
    error "Failed to install Ollama."
  fi
  
  success "Ollama installed successfully."
}

# Create project structure
setup_project_structure() {
  local project_root="$1"
  
  log "Setting up project structure at $project_root..."
  
  mkdir -p "$project_root/src/vm/ai"
  mkdir -p "$project_root/src/contracts"
  mkdir -p "$project_root/src/network/p2p"
  mkdir -p "$project_root/src/state"
  mkdir -p "$project_root/src/consensus"
  mkdir -p "$project_root/src/cache"
  mkdir -p "$project_root/src/models"
  mkdir -p "$project_root/src/fault_tolerance"
  mkdir -p "$project_root/config"
  
  success "Project structure created."
}

# Update Cargo.toml
update_cargo_toml() {
  local project_root="$1"
  local cargo_file="$project_root/Cargo.toml"
  
  log "Updating Cargo.toml with new dependencies..."
  
  # Create backup of existing Cargo.toml if it exists
  if [ -f "$cargo_file" ]; then
    cp "$cargo_file" "${cargo_file}.bak"
  else
    # Create a new one if it doesn't exist
    cat > "$cargo_file" <<EOF
[package]
name = "dagknight"
version = "0.1.0"
edition = "2021"
description = "DAGKnight blockchain with distributed AI capabilities"
authors = ["DAGKnight Team"]

[dependencies]
EOF
  fi
  
  # Append new dependencies
  cat >> "$cargo_file" <<EOF
# AI Integration
ollama-rs = "0.1.5"
async-trait = "0.1.68"
reqwest = { version = "0.11", features = ["json"] }
tokio-tungstenite = "0.20.0"

# Caching
redis = { version = "0.23.0", features = ["tokio-comp"] }
lru = "0.10.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3.3"

# Concurrency and async
tokio = { version = "1.28", features = ["full"] }
futures = "0.3"

# Cryptographic
sha2 = "0.10.6"
ed25519-dalek = "1.0.1"
rand = "0.8.5"

# Metrics and monitoring
prometheus = "0.13.3"
tracing = "0.1.37"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Testing
criterion = { version = "0.4", optional = true }
mockall = { version = "0.11.3", optional = true }

[features]
default = ["ai", "cache"]
ai = []
cache = []
dynamic-allocation = []
fault-tolerance = []
metrics = []
testing = ["criterion", "mockall"]

[[bin]]
name = "dagknight"
path = "src/main.rs"
EOF
  
  success "Cargo.toml updated with new dependencies."
}

# Create AI contract types
create_contract_types() {
  local project_root="$1"
  local file="$project_root/src/contracts/mod.rs"
  
  log "Creating AI contract types..."
  
  cat > "$file" <<EOF
//! DAGKnight contract types and transaction definitions
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// AI model execution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIModelCall {
    /// The model identifier to execute
    pub model: String,
    /// Serialized input data
    pub input: Vec<u8>,
    /// Maximum gas limit for execution
    pub gas_limit: u64,
    /// Number of nodes to distribute computation across
    pub shard_count: u64,
    /// Optional caching preferences
    pub cache_policy: CachePolicy,
}

/// Cache policy for AI model executions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CachePolicy {
    /// Do not use cache
    NoCache,
    /// Use cache if available with specified TTL in seconds
    UseCache(u64),
    /// Force cache refresh
    ForceRefresh,
}

/// Transaction types supported by DAGKnight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionType {
    /// Standard value transfer
    Transfer {
        recipient: [u8; 32],
        amount: u64,
    },
    /// Smart contract execution
    ContractExecution {
        contract: [u8; 32],
        method: String,
        args: Vec<u8>,
    },
    /// AI model execution
    AIModelExecution(AIModelCall),
    /// Model registration
    RegisterModel(ModelRegistration),
}

/// Model registration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistration {
    /// Model identifier
    pub model_id: String,
    /// Model description
    pub description: String,
    /// Model version
    pub version: String,
    /// Required memory in MB
    pub memory_required: u64,
    /// Sharding capability
    pub sharding_capability: ShardingCapability,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Model sharding capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingCapability {
    /// Cannot be sharded
    None,
    /// Can be sharded horizontally (different inputs)
    Horizontal,
    /// Can be sharded vertically (model layers)
    Vertical,
    /// Can be sharded both ways
    Full,
}

/// Resource requirements for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum CPU cores
    pub min_cpu_cores: u32,
    /// Minimum memory in MB
    pub min_memory_mb: u64,
    /// GPU memory requirement in MB
    pub gpu_memory_mb: Option<u64>,
    /// Disk space requirement in MB
    pub disk_space_mb: u64,
    /// Average execution time per input token in ms
    pub avg_exec_time_per_token_ms: f64,
}

/// Transaction with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Transaction hash
    pub hash: [u8; 32],
    /// Transaction type
    pub tx_type: TransactionType,
    /// Sender's public key
    pub sender: [u8; 32],
    /// Nonce
    pub nonce: u64,
    /// Gas price
    pub gas_price: u64,
    /// Gas limit
    pub gas_limit: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Signature
    pub signature: [u8; 64],
}
EOF
  
  success "AI contract types created."
}

# Create VM for AI execution
create_vm_ai_executor() {
  local project_root="$1"
  local file="$project_root/src/vm/ai/executor.rs"
  
  log "Creating AI executor..."
  
  cat > "$file" <<EOF
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
EOF
  
  success "AI executor created."
}

# Create model registry
create_model_registry() {
  local project_root="$1"
  local file="$project_root/src/models/mod.rs"
  
  log "Creating model registry..."
  
  cat > "$file" <<EOF
//! Model registry for DAGKnight
use crate::contracts::{ModelRegistration, ShardingCapability, ResourceRequirements};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error};

/// Model registry
pub struct ModelRegistry {
    models: RwLock<HashMap<String, ModelInfo>>,
}

/// Extended model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model registration info
    pub registration: ModelRegistration,
    /// Popularity score
    pub popularity: f64,
    /// Performance metrics
    pub performance: ModelPerformance,
    /// Quality metrics
    pub quality: ModelQuality,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Average tokens per second
    pub avg_tokens_per_second: f64,
    /// Average RAM usage in MB
    pub avg_ram_usage_mb: u64,
    /// Average GPU usage in MB
    pub avg_gpu_usage_mb: Option<u64>,
}

/// Model quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelQuality {
    /// Average quality score (0-100)
    pub quality_score: f64,
    /// Number of ratings
    pub num_ratings: u64,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
        }
    }
    
    /// Register a new model
    pub async fn register_model(&self, registration: ModelRegistration) -> bool {
        let mut models = self.models.write().await;
        
        if models.contains_key(&registration.model_id) {
            warn!("Model {} already registered", registration.model_id);
            return false;
        }
        
        let model_info = ModelInfo {
            registration: registration.clone(),
            popularity: 0.0,
            performance: ModelPerformance {
                avg_tokens_per_second: 0.0,
                avg_ram_usage_mb: registration.resource_requirements.min_memory_mb,
                avg_gpu_usage_mb: registration.resource_requirements.gpu_memory_mb,
            },
            quality: ModelQuality {
                quality_score: 0.0,
                num_ratings: 0,
            },
        };
        
        models.insert(registration.model_id.clone(), model_info);
        info!("Registered model: {}", registration.model_id);
        
        true
    }
    
    /// Get model information
    pub async fn get_model(&self, model_id: &str) -> Option<ModelInfo> {
        let models = self.models.read().await;
        models.get(model_id).cloned()
    }
    
    /// Update model performance
    pub async fn update_performance(
        &self,
        model_id: &str,
        tokens_per_second: f64,
        ram_usage_mb: u64,
        gpu_usage_mb: Option<u64>,
    ) -> bool {
        let mut models = self.models.write().await;
        
        if let Some(model) = models.get_mut(model_id) {
            // Update with exponential moving average
            let alpha = 0.1; // Weight for new observations
            
            model.performance.avg_tokens_per_second = 
                (1.0 - alpha) * model.performance.avg_tokens_per_second + alpha * tokens_per_second;
                
            model.performance.avg_ram_usage_mb = 
                ((1.0 - alpha) * model.performance.avg_ram_usage_mb as f64 + alpha * ram_usage_mb as f64) as u64;
                
            if let Some(gpu_usage) = gpu_usage_mb {
                model.performance.avg_gpu_usage_mb = Some(
                    ((1.0 - alpha) * model.performance.avg_gpu_usage_mb.unwrap_or(0) as f64 + 
                     alpha * gpu_usage as f64) as u64
                );
            }
            
            // Increase popularity
            model.popularity += 0.1;
            return true;
        }
        
        warn!("Model {} not found for performance update", model_id);
        false
    }
    
    /// Update model quality
    pub async fn update_quality(&self, model_id: &str, quality_score: f64) -> bool {
        let mut models = self.models.write().await;
        
        if let Some(model) = models.get_mut(model_id) {
            // Update with weighted average
            let current_score = model.quality.quality_score;
            let num_ratings = model.quality.num_ratings;
            
            model.quality.quality_score = 
                (current_score * num_ratings as f64 + quality_score) / (num_ratings as f64 + 1.0);
                
            model.quality.num_ratings += 1;
            
            return true;
        }
        
        warn!("Model {} not found for quality update", model_id);
        false
    }
    
    /// List all available models
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }
    
    /// Find models that meet resource constraints
    pub async fn find_models_by_resources(
        &self,
        max_memory: u64,
        gpu_required: bool,
    ) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        
        models.values()
            .filter(|m| {
                m.registration.resource_requirements.min_memory_mb <= max_memory &&
                (!gpu_required || m.registration.resource_requirements.gpu_memory_mb.is_some())
            })
            .cloned()
            .collect()
    }
    
    /// Find models by sharding capability
    pub async fn find_models_by_sharding(
        &self,
        capability: ShardingCapability,
    ) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        
        models.values()
            .filter(|m| {
                match (capability, &m.registration.sharding_capability) {
                    (ShardingCapability::None, _) => true,
                    (ShardingCapability::Horizontal, ShardingCapability::Horizontal | ShardingCapability::Full) => true,
                    (ShardingCapability::Vertical, ShardingCapability::Vertical | ShardingCapability::Full) => true,
                    (ShardingCapability::Full, ShardingCapability::Full) => true,
                    _ => false,
                }
            })
            .cloned()
            .collect()
    }
    
    /// Initialize registry with default models
    pub async fn initialize_defaults(&self) {
        // Register some default models
        let models = [
            ModelRegistration {
                model_id: "llama2:7b".to_string(),
                description: "Meta's Llama 2 7B parameter model".to_string(),
                version: "2.0".to_string(),
                memory_required: 16000,
                sharding_capability: ShardingCapability::Horizontal,
                resource_requirements: ResourceRequirements {
                    min_cpu_cores: 4,
                    min_memory_mb: 16000,
                    gpu_memory_mb: Some(8000),
                    disk_space_mb: 14000,
                    avg_exec_time_per_token_ms: 15.0,
                },
            },
            ModelRegistration {
                model_id: "deepseek-r1:1.5b".to_string(),
                description: "DeepSeek R1 1.5B parameter model".to_string(),
                version: "1.0".to_string(),
                memory_required: 3000,
                sharding_capability: ShardingCapability::Full,
                resource_requirements: ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_mb: 4000,
                    gpu_memory_mb: Some(3000),
                    disk_space_mb: 3000,
                    avg_exec_time_per_token_ms: 5.0,
                },
            },
            ModelRegistration {
                model_id: "mistral:7b".to_string(),
                description: "Mistral 7B parameter model".to_string(),
                version: "1.0".to_string(),
                memory_required: 16000,
                sharding_capability: ShardingCapability::Vertical,
                resource_requirements: ResourceRequirements {
                    min_cpu_cores: 4,
                    min_memory_mb: 16000,
                    gpu_memory_mb: Some(7000),
                    disk_space_mb: 13500,
                    avg_exec_time_per_token_ms: 12.0,
                },
            },
            ModelRegistration {
                model_id: "phi-2:3b".to_string(),
                description: "Microsoft's Phi-2 3B parameter model".to_string(),
                version: "2.0".to_string(),
                memory_required: 6000,
                sharding_capability: ShardingCapability::Horizontal,
                resource_requirements: ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_mb: 8000,
                    gpu_memory_mb: Some(4000),
                    disk_space_mb: 6000,
                    avg_exec_time_per_token_ms: 8.0,
                },
            },
        ];
        
        for model in models {
            self.register_model(model).await;
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
EOF
  
  success "Model registry created."
}

# Create caching layer
create_caching_layer() {
  local project_root="$1"
  local file="$project_root/src/cache/mod.rs"
  
  log "Creating caching layer..."
  
  cat > "$file" <<EOF
//! Caching layer for AI model outputs
use redis::{Client, AsyncCommands};
use lru::LruCache;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::num::NonZeroUsize;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::{info, warn, error, debug};
use std::time::{Duration, Instant};

/// Cache provider type
pub enum CacheProvider {
    /// In-memory LRU cache
    Memory,
    /// Redis cache
    Redis,
    /// Combined (layered) cache
    Layered,
}

/// Model cache for storing AI outputs
pub struct ModelCache {
    /// In-memory cache
    memory_cache: Arc<Mutex<LruCache<u64, CacheEntry>>>,
    /// Redis client if available
    redis_client: Option<Client>,
    /// Cache provider
    provider: CacheProvider,
    /// Cache hit statistics
    stats: Arc<Mutex<CacheStats>>,
}

/// Cache entry with expiration
struct CacheEntry {
    /// Output data
    data: Vec<u8>,
    /// Expiration timestamp
    expires_at: Instant,
}

/// Cache statistics
#[derive(Debug, Default)]
struct CacheStats {
    /// Number of gets
    gets: u64,
    /// Number of sets
    sets: u64,
    /// Number of memory hits
    memory_hits: u64,
    /// Number of redis hits
    redis_hits: u64,
    /// Number of misses
    misses: u64,
}

impl ModelCache {
    /// Create a new model cache
    pub fn new(provider: CacheProvider, memory_size: usize, redis_url: Option<String>) -> Self {
        // Create memory cache
        let memory_size = NonZeroUsize::new(memory_size).unwrap_or(NonZeroUsize::new(10000).unwrap());
        let memory_cache = Arc::new(Mutex::new(LruCache::new(memory_size)));
        
        // Create Redis client if URL provided
        let redis_client = if let Some(url) = redis_url {
            match Client::open(url) {
                Ok(client) => {
                    info!("Redis cache connected");
                    Some(client)
                },
                Err(e) => {
                    error!("Failed to connect to Redis: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Warn if Redis requested but not available
        if matches!(provider, CacheProvider::Redis | CacheProvider::Layered) && redis_client.is_none() {
            warn!("Redis cache requested but not available, falling back to memory cache");
        }
        
        Self {
            memory_cache,
            redis_client,
            provider,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }
    
    /// Get cached result
    pub async fn get(&self, model: &str, input: &[u8], ttl: u64) -> Option<Vec<u8>> {
        let key = Self::generate_key(model, input);
        
        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.gets += 1;
        }
        
        // Try memory cache first
        if let Some(entry) = self.check_memory_cache(key).await {
            if entry.expires_at > Instant::now() {
                // Update stats
                {
                    let mut stats = self.stats.lock().await;
                    stats.memory_hits += 1;
                }
                return Some(entry.data);
            }
        }
        
        // If Redis is available and enabled, try it next
        if matches!(self.provider, CacheProvider::Redis | CacheProvider::Layered) {
            if let Some(client) = &self.redis_client {
                if let Some(data) = self.check_redis_cache(client, model, input).await {
                    // Also update memory cache for next time
                    self.update_memory_cache(key, &data, ttl).await;
                    
                    // Update stats
                    {
                        let mut stats = self.stats.lock().await;
                        stats.redis_hits += 1;
                    }
                    
                    return Some(data);
                }
            }
        }
        
        // Update stats for miss
        {
            let mut stats = self.stats.lock().await;
            stats.misses += 1;
        }
        
        None
    }
    
    /// Check memory cache
    async fn check_memory_cache(&self, key: u64) -> Option<CacheEntry> {
        let mut cache = self.memory_cache.lock().await;
        cache.get(&key).cloned()
    }
    
    /// Check Redis cache
    async fn check_redis_cache(&self, client: &Client, model: &str, input: &[u8]) -> Option<Vec<u8>> {
        let key = format!("dagknight:model:{}:{}", model, hex::encode(Self::hash_bytes(input)));
        
        match client.get_async_connection().await {
            Ok(mut conn) => {
                match conn.get::<_, Option<Vec<u8>>>(&key).await {
                    Ok(Some(data)) => Some(data),
                    Ok(None) => None,
                    Err(e) => {
                        error!("Redis error while getting key {}: {}", key, e);
                        None
                    }
                }
            },
            Err(e) => {
                error!("Failed to get Redis connection: {}", e);
                None
            }
        }
    }
    
    /// Set value in cache
    pub async fn set(&self, model: &str, input: &[u8], output: &[u8], ttl: u64) {
        let key = Self::generate_key(model, input);
        
        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.sets += 1;
        }
        
        // Update memory cache
        self.update_memory_cache(key, output, ttl).await;
        
        // Update Redis if available
        if matches!(self.provider, CacheProvider::Redis | CacheProvider::Layered) {
            if let Some(client) = &self.redis_client {
                self.update_redis_cache(client, model, input, output, ttl).await;
            }
        }
    }
    
    /// Update memory cache
    async fn update_memory_cache(&self, key: u64, data: &[u8], ttl: u64) {
        let entry = CacheEntry {
            data: data.to_vec(),
            expires_at: Instant::now() + Duration::from_secs(ttl),
        };
        
        let mut cache = self.memory_cache.lock().await;
        cache.put(key, entry);
    }
    
    /// Update Redis cache
    async fn update_redis_cache(&self, client: &Client, model: &str, input: &[u8], output: &[u8], ttl: u64) {
        let key = format!("dagknight:model:{}:{}", model, hex::encode(Self::hash_bytes(input)));
        
        match client.get_async_connection().await {
            Ok(mut conn) => {
                let _: Result<(), redis::RedisError> = conn.set_ex(&key, output, ttl as usize).await;
            },
            Err(e) => {
                error!("Failed to get Redis connection for set: {}", e);
            }
        }
    }
    
    /// Generate cache key
    fn generate_key(model: &str, input: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        model.hash(&mut hasher);
        input.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Hash bytes
    fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher.finalize().into()
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.stats.lock().await.clone()
    }
    
    /// Start periodic cleanup
    pub fn start_cleanup_task(&self) {
        let memory_cache = self.memory_cache.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                let now = Instant::now();
                let mut cache = memory_cache.lock().await;
                
                // Remove expired entries
                cache.retain(|_, entry| entry.expires_at > now);
                
                debug!("Cache cleanup completed, size: {}", cache.len());
            }
        });
    }
}
EOF
  
  success "Caching layer created."
}

# Create fault tolerance module
create_fault_tolerance() {
  local project_root="$1"
  local file="$project_root/src/fault_tolerance/mod.rs"
  
  log "Creating fault tolerance module..."
  
  cat > "$file" <<EOF
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
EOF
  
  success "Fault tolerance module created."
}

# Create script for resource usage tracking
create_resource_tracking() {
  local project_root="$1"
  local file="$project_root/src/state/mod.rs"
  
  log "Creating resource tracking module..."
  
  cat > "$file" <<EOF
//! State management for DAGKnight
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};

/// Resource usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU time in milliseconds
    pub cpu_time: u64,
    /// Memory used in bytes
    pub memory_used: u64,
    /// GPU time in milliseconds (if used)
    pub gpu_time: Option<u64>,
}

impl ResourceUsage {
    /// Create a minimal resource usage entry for cached results
    pub fn minimal() -> Self {
        Self {
            cpu_time: 1,
            memory_used: 1024,
            gpu_time: None,
        }
    }
    
    /// Calculate resource cost based on usage
    pub fn calculate_cost(&self) -> u64 {
        // Base cost from CPU
        let cpu_cost = self.cpu_time * 1; // 1 token per ms of CPU time
        
        // Memory cost
        let memory_cost = (self.memory_used / (1024 * 1024)) * 10; // 10 tokens per MB
        
        // GPU cost if used
        let gpu_cost = self.gpu_time.map(|time| time * 5).unwrap_or(0); // 5 tokens per ms of GPU time
        
        cpu_cost + memory_cost + gpu_cost
    }
}

/// Resource ledger for tracking contributions
pub struct ResourceLedger {
    /// Tracks resource contributions per node
    contributions: RwLock<HashMap<[u8; 32], ResourcePool>>,
}

/// Resource pool for a single node
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourcePool {
    /// Total CPU time provided (ms)
    pub total_cpu: u64,
    /// Total memory provided (MB)
    pub total_memory: u64,
    /// Total GPU time provided (ms)
    pub total_gpu: u64,
    /// Pending rewards (tokens)
    pub pending_rewards: u64,
}

impl ResourceLedger {
    /// Create a new resource ledger
    pub fn new() -> Self {
        Self {
            contributions: RwLock::new(HashMap::new()),
        }
    }
    
    /// Update resource usage for a node
    pub async fn update_resources(&self, node: [u8; 32], usage: &ResourceUsage) {
        let mut ledger = self.contributions.write().await;
        let pool = ledger.entry(node).or_insert(ResourcePool::default());
        
        pool.total_cpu += usage.cpu_time;
        pool.total_memory += usage.memory_used / (1024 * 1024); // Convert to MB
        pool.total_gpu += usage.gpu_time.unwrap_or(0);
        
        // Calculate rewards
        let reward = usage.calculate_cost();
        pool.pending_rewards += reward;
    }
    
    /// Get resource pool for a node
    pub async fn get_resource_pool(&self, node: &[u8; 32]) -> Option<ResourcePool> {
        let ledger = self.contributions.read().await;
        ledger.get(node).cloned()
    }
    
    /// Get all resource pools
    pub async fn get_all_resource_pools(&self) -> HashMap<[u8; 32],
    ResourcePool> {
        let ledger = self.contributions.read().await;
        ledger.clone()
    }
    
    /// Claim rewards for a node
    pub async fn claim_rewards(&self, node: &[u8; 32]) -> u64 {
        let mut ledger = self.contributions.write().await;
        
        if let Some(pool) = ledger.get_mut(node) {
            let rewards = pool.pending_rewards;
            pool.pending_rewards = 0;
            rewards
        } else {
            0
        }
    }
    
    /// Get total resource usage across all nodes
    pub async fn get_total_resource_usage(&self) -> ResourcePool {
        let ledger = self.contributions.read().await;
        
        let mut total = ResourcePool::default();
        for pool in ledger.values() {
            total.total_cpu += pool.total_cpu;
            total.total_memory += pool.total_memory;
            total.total_gpu += pool.total_gpu;
            total.pending_rewards += pool.pending_rewards;
        }
        
        total
    }
}

/// State database for DAGKnight
pub struct StateDB {
    /// Resource ledger
    pub resource_ledger: ResourceLedger,
    // Other state components would go here
}

impl StateDB {
    /// Create a new state database
    pub fn new() -> Self {
        Self {
            resource_ledger: ResourceLedger::new(),
        }
    }
    
    /// Update resource ledger
    pub async fn update_resource_ledger(
        &self,
        node: [u8; 32],
        usage: &ResourceUsage
    ) {
        self.resource_ledger.update_resources(node, usage).await;
    }
}

impl Default for StateDB {
    fn default() -> Self {
        Self::new()
    }
}
EOF
  
  success "Resource tracking module created."
}

# Create the network message types
create_network_messages() {
  local project_root="$1"
  local file="$project_root/src/network/p2p.rs"
  
  log "Creating network message types..."
  
  cat > "$file" <<EOF
//! P2P network messages for DAGKnight
use crate::state::ResourceUsage;
use serde::{Serialize, Deserialize};
use std::time::Duration;

/// Network message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMessage {
    /// Transaction message
    Transaction {
        /// Transaction data
        data: Vec<u8>,
        /// Transaction hash
        hash: [u8; 32],
        /// Timestamp
        timestamp: u64,
    },
    
    /// Block message
    Block {
        /// Block data
        data: Vec<u8>,
        /// Block hash
        hash: [u8; 32],
        /// Block height
        height: u64,
        /// Timestamp
        timestamp: u64,
    },
    
    /// Consensus message
    Consensus {
        /// Consensus type
        consensus_type: ConsensusType,
        /// Message data
        data: Vec<u8>,
        /// Timestamp
        timestamp: u64,
    },
    
    /// Compute task message
    ComputeTask {
        /// Contract address
        contract: [u8; 32],
        /// Model identifier
        model: String,
        /// Input data
        input: Vec<u8>,
        /// Timestamp
        timestamp: u64,
    },
    
    /// Compute result message
    ComputeResult {
        /// Contract address
        contract: [u8; 32],
        /// Output data
        output: Vec<u8>,
        /// Cryptographic proof
        proof: [u8; 64],
        /// Resource usage
        resources: ResourceUsage,
        /// Timestamp
        timestamp: u64,
    },
    
    /// Node status message
    NodeStatus {
        /// Node ID
        node_id: [u8; 32],
        /// Status
        status: NodeStatus,
        /// Available resources
        available_resources: AvailableResources,
        /// Timestamp
        timestamp: u64,
    },
    
    /// Model registry message
    ModelRegistry {
        /// Action
        action: ModelRegistryAction,
        /// Data
        data: Vec<u8>,
        /// Timestamp
        timestamp: u64,
    },
}

/// Consensus message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusType {
    /// Prepare message
    Prepare,
    /// Commit message
    Commit,
    /// View change
    ViewChange,
    /// New view
    NewView,
    /// Validation result
    ValidationResult,
}

/// Node status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is online
    Online,
    /// Node is offline
    Offline,
    /// Node is syncing
    Syncing,
    /// Node is ready for compute
    ReadyForCompute,
    /// Node is busy
    Busy,
}

/// Available resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableResources {
    /// Available CPU cores
    pub cpu_cores: u32,
    /// Available memory in MB
    pub memory_mb: u64,
    /// Available GPU memory in MB
    pub gpu_memory_mb: Option<u64>,
    /// Available disk space in MB
    pub disk_space_mb: u64,
    /// Available network bandwidth in Mbps
    pub network_bandwidth_mbps: u64,
    /// Node latency in ms
    pub latency_ms: u64,
}

/// Model registry actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelRegistryAction {
    /// Register a new model
    Register,
    /// Update model information
    Update,
    /// Remove a model
    Remove,
    /// Query for model
    Query,
    /// Report model performance
    ReportPerformance,
}

/// P2P message handler
pub struct P2PMessageHandler {
    // Message handlers would go here
}

impl P2PMessageHandler {
    /// Create a new message handler
    pub fn new() -> Self {
        Self {}
    }
    
    /// Handle network message
    pub async fn handle_message(&self, message: NetworkMessage) -> Result<(), MessageError> {
        match message {
            NetworkMessage::ComputeTask { contract, model, input, timestamp } => {
                // Handle compute task
                self.handle_compute_task(contract, model, input, timestamp).await
            },
            NetworkMessage::ComputeResult { contract, output, proof, resources, timestamp } => {
                // Handle compute result
                self.handle_compute_result(contract, output, proof, resources, timestamp).await
            },
            // Other message types would be handled here
            _ => {
                // Placeholder for other message types
                Ok(())
            }
        }
    }
    
    /// Handle compute task message
    async fn handle_compute_task(
        &self,
        contract: [u8; 32],
        model: String,
        input: Vec<u8>,
        timestamp: u64,
    ) -> Result<(), MessageError> {
        // This would integrate with the AI executor
        Ok(())
    }
    
    /// Handle compute result message
    async fn handle_compute_result(
        &self,
        contract: [u8; 32],
        output: Vec<u8>,
        proof: [u8; 64],
        resources: ResourceUsage,
        timestamp: u64,
    ) -> Result<(), MessageError> {
        // This would verify and process the result
        Ok(())
    }
}

/// Message error
#[derive(Debug, thiserror::Error)]
pub enum MessageError {
    #[error("Invalid message: {0}")]
    InvalidMessage(String),
    
    #[error("Processing error: {0}")]
    ProcessingError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
}
EOF
  
  success "Network message types created."
}

# Create main script that integrates all components
create_integration_script() {
  local project_root="$1"
  local file="$project_root/src/main.rs"
  
  log "Creating main integration file..."
  
  cat > "$file" <<EOF
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

use crate::cache::{ModelCache, CacheProvider};
use crate::fault_tolerance::{RecoveryManager, RecoverySettings};
use crate::models::ModelRegistry;
use crate::state::StateDB;
use crate::vm::ai::executor::AIExecutor;

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
    
    let recovery = Arc::new(RecoveryManager::new(recovery_settings));
    let state_db = Arc::new(StateDB::new());
    
    // Initialize model registry with defaults
    registry.initialize_defaults().await;
    
    // Initialize AI executor
    let ai_executor = AIExecutor::new(
        cache.clone(),
        registry.clone(),
        recovery.clone(),
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
EOF
  
  success "Main integration file created."
}

# Create config file for the distributed AI components
create_config_file() {
  local project_root="$1"
  local file="$project_root/config/ai_config.json"
  
  log "Creating configuration file..."
  
  cat > "$file" <<EOF
{
  "ai": {
    "enabled": true,
    "max_model_size_gb": 10,
    "default_shard_count": 4,
    "max_shard_count": 100,
    "ollama": {
      "endpoint": "http://localhost:11434",
      "timeout_seconds": 180
    }
  },
  "cache": {
    "provider": "layered",
    "memory": {
      "max_entries": 50000,
      "default_ttl_seconds": 3600
    },
    "redis": {
      "url": "redis://localhost:6379",
      "default_ttl_seconds": 86400
    }
  },
  "fault_tolerance": {
    "enable_replication": true,
    "replication_factor": 2,
    "max_retries": 3,
    "retry_delay_ms": 500,
    "task_timeout_secs": 120
  },
  "models": [
    {
      "model_id": "llama2:7b",
      "description": "Meta's Llama 2 7B parameter model",
      "version": "2.0",
      "memory_required": 16000,
      "sharding_capability": "horizontal",
      "resource_requirements": {
        "min_cpu_cores": 4,
        "min_memory_mb": 16000,
        "gpu_memory_mb": 8000,
        "disk_space_mb": 14000,
        "avg_exec_time_per_token_ms": 15.0
      }
    },
    {
      "model_id": "deepseek-r1:1.5b",
      "description": "DeepSeek R1 1.5B parameter model",
      "version": "1.0",
      "memory_required": 3000,
      "sharding_capability": "full",
      "resource_requirements": {
        "min_cpu_cores": 2,
        "min_memory_mb": 4000,
        "gpu_memory_mb": 3000,
        "disk_space_mb": 3000,
        "avg_exec_time_per_token_ms": 5.0
      }
    },
    {
      "model_id": "mistral:7b",
      "description": "Mistral 7B parameter model",
      "version": "1.0",
      "memory_required": 16000,
      "sharding_capability": "vertical",
      "resource_requirements": {
        "min_cpu_cores": 4,
        "min_memory_mb": 16000,
        "gpu_memory_mb": 7000,
        "disk_space_mb": 13500,
        "avg_exec_time_per_token_ms": 12.0
      }
    },
    {
      "model_id": "phi-2:3b",
      "description": "Microsoft's Phi-2 3B parameter model",
      "version": "2.0",
      "memory_required": 6000,
      "sharding_capability": "horizontal",
      "resource_requirements": {
        "min_cpu_cores": 2,
        "min_memory_mb": 8000,
        "gpu_memory_mb": 4000,
        "disk_space_mb": 6000,
        "avg_exec_time_per_token_ms": 8.0
      }
    }
  ]
}
EOF
  
  success "Configuration file created."
}

# Create master bash script that implements everything
create_master_script() {
  local project_root="$1"
  local script_file="$project_root/setup_dagknight_ai.sh"
  
  log "Creating master setup script..."
  
  cat > "$script_file" <<EOF
#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log function
log() {
  echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
  echo -e "${RED}[ERROR] $1${NC}"
  exit 1
}

success() {
  echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warning() {
  echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Project directory
PROJECT_ROOT="$PWD"

# Header
log "===================================================================="
log "  DAGKnight Distributed AI Integration Setup"
log "===================================================================="

# Check dependencies
log "Checking dependencies..."
deps=("cargo" "git" "docker" "curl")
missing=()

for dep in "\${deps[@]}"; do
  if ! command -v "\$dep" &> /dev/null; then
    missing+=("\$dep")
  fi
done

if [ \${#missing[@]} -ne 0 ]; then
  error "Missing dependencies: \${missing[*]}"
fi

success "All base dependencies found."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
  warning "Ollama not found. Installing..."
  curl -fsSL https://ollama.com/install.sh | sh
  if [ \$? -ne 0 ]; then
    error "Failed to install Ollama."
  fi
fi

success "Ollama is installed."

# Check if Redis is available
if ! docker ps | grep -q redis; then
  warning "Redis not found. Starting Redis container..."
  docker run --name dagknight-redis -p 6379:6379 -d redis:alpine
  if [ \$? -ne 0 ]; then
    warning "Failed to start Redis container. Cache will fall back to memory-only."
  else
    success "Redis container started."
  fi
else
  success "Redis is available."
fi

# Setup project structure
log "Setting up project structure..."
mkdir -p "\$PROJECT_ROOT/src/vm/ai"
mkdir -p "\$PROJECT_ROOT/src/contracts"
mkdir -p "\$PROJECT_ROOT/src/network/p2p"
mkdir -p "\$PROJECT_ROOT/src/state"
mkdir -p "\$PROJECT_ROOT/src/consensus"
mkdir -p "\$PROJECT_ROOT/src/cache"
mkdir -p "\$PROJECT_ROOT/src/models"
mkdir -p "\$PROJECT_ROOT/src/fault_tolerance"
mkdir -p "\$PROJECT_ROOT/config"

# Create Cargo.toml
log "Creating Cargo.toml..."
cat > "\$PROJECT_ROOT/Cargo.toml" <<EOL
[package]
name = "dagknight"
version = "0.1.0"
edition = "2021"
description = "DAGKnight blockchain with distributed AI capabilities"
authors = ["DAGKnight Team"]

[dependencies]
# AI Integration
ollama-rs = "0.1.5"
async-trait = "0.1.68"
reqwest = { version = "0.11", features = ["json"] }
tokio-tungstenite = "0.20.0"

# Caching
redis = { version = "0.23.0", features = ["tokio-comp"] }
lru = "0.10.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3.3"

# Concurrency and async
tokio = { version = "1.28", features = ["full"] }
futures = "0.3"

# Cryptographic
sha2 = "0.10.6"
ed25519-dalek = "1.0.1"
rand = "0.8.5"
hex = "0.4.3"
thiserror = "1.0"

# Metrics and monitoring
prometheus = "0.13.3"
tracing = "0.1.37"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Testing
criterion = { version = "0.4", optional = true }
mockall = { version = "0.11.3", optional = true }

[features]
default = ["ai", "cache"]
ai = []
cache = []
dynamic-allocation = []
fault-tolerance = []
metrics = []
testing = ["criterion", "mockall"]

[[bin]]
name = "dagknight"
path = "src/main.rs"
EOL

# Create all source files
log "Creating source files..."

# Create contract types
cat > "\$PROJECT_ROOT/src/contracts/mod.rs" <<EOL
$(cat "$project_root/src/contracts/mod.rs")
EOL

# Create VM for AI execution
cat > "\$PROJECT_ROOT/src/vm/ai/executor.rs" <<EOL
$(cat "$project_root/src/vm/ai/executor.rs")
EOL

# Create VM module
mkdir -p "\$PROJECT_ROOT/src/vm/ai"
cat > "\$PROJECT_ROOT/src/vm/mod.rs" <<EOL
//! Virtual Machine for DAGKnight
pub mod ai;
EOL

# Create model registry
cat > "\$PROJECT_ROOT/src/models/mod.rs" <<EOL
$(cat "$project_root/src/models/mod.rs")
EOL

# Create caching layer
cat > "\$PROJECT_ROOT/src/cache/mod.rs" <<EOL
$(cat "$project_root/src/cache/mod.rs")
EOL

# Create fault tolerance module
cat > "\$PROJECT_ROOT/src/fault_tolerance/mod.rs" <<EOL
$(cat "$project_root/src/fault_tolerance/mod.rs")
EOL

# Create resource tracking
cat > "\$PROJECT_ROOT/src/state/mod.rs" <<EOL
$(cat "$project_root/src/state/mod.rs")
EOL

# Create network message types
cat > "\$PROJECT_ROOT/src/network/p2p.rs" <<EOL
$(cat "$project_root/src/network/p2p.rs")
EOL

# Create network module
cat > "\$PROJECT_ROOT/src/network/mod.rs" <<EOL
//! Network module for DAGKnight
pub mod p2p;
EOL

# Create error module
cat > "\$PROJECT_ROOT/src/error.rs" <<EOL
//! Error types for DAGKnight
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Network error: {0}")]
    Network(String),
    
    #[error("Execution error: {0}")]
    Execution(String),
    
    #[error("AI model error: {0}")]
    AIModel(String),
    
    #[error("Resource allocation error: {0}")]
    ResourceAllocation(String),
    
    #[error("Consensus error: {0}")]
    Consensus(String),
    
    #[error("Cache error: {0}")]
    Cache(String),
    
    #[error("Other error: {0}")]
    Other(String),
}
EOL

# Create main integration file
cat > "\$PROJECT_ROOT/src/main.rs" <<EOL
$(cat "$project_root/src/main.rs")
EOL

# Create config file for the distributed AI components
cat > "\$PROJECT_ROOT/config/ai_config.json" <<EOL
$(cat "$project_root/config/ai_config.json")
EOL

# Pull default models with Ollama
log "Pulling default models with Ollama..."

models=("llama2:7b" "phi:1.5b")

for model in "\${models[@]}"; do
  log "Pulling model \$model (this may take a while)..."
  if ! ollama pull "\$model"; then
    warning "Failed to pull model \$model. It will be downloaded on first use."
  else
    success "Model \$model pulled successfully."
  fi
done

# Final steps
log "Building project..."
cargo build

success "================================================="
success "  DAGKnight Distributed AI Integration Setup Complete!"
success "================================================="
log "To run your DAGKnight node: cargo run"
log "API documentation will be available at: http://localhost:8545/docs"

# Make script executable
chmod +x "\$script_file"

EOF
  
  chmod +x "$script_file"
  success "Master setup script created."
}

# Main function
main() {
  log "Starting DAGKnight distributed AI setup script..."
  
  # Get project directory
  read -p "Enter project directory path (default: ./dagknight): " project_root
  project_root=${project_root:-"./dagknight"}
  
  # Create project directory if it doesn't exist
  mkdir -p "$project_root"
  
  # Check dependencies
  check_dependencies
  
  # Setup project structure
  setup_project_structure "$project_root"
  
  # Update Cargo.toml
  update_cargo_toml "$project_root"
  
  # Create contract types
  create_contract_types "$project_root"
  
  # Create VM for AI execution
  create_vm_ai_executor "$project_root"
  
  # Create model registry
  create_model_registry "$project_root"
  
  # Create caching layer
  create_caching_layer "$project_root"
  
  # Create fault tolerance module
  create_fault_tolerance "$project_root"
  
  # Create resource tracking
  create_resource_tracking "$project_root"
  
  # Create network message types
  create_network_messages "$project_root"
  
  # Create main integration script
  create_integration_script "$project_root"
  
  # Create config file
  create_config_file "$project_root"
  
  # Create master script
  create_master_script "$project_root"
  
  success "==============================================="
  success "Setup completed successfully!"
  success "Your DAGKnight AI integration is ready to run."
  success "==============================================="
  log "Run the following command to complete the setup:"
  log "cd $project_root && ./setup_dagknight_ai.sh"
}

# Run the main function
main
