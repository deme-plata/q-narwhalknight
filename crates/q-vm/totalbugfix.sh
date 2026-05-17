#!/bin/bash
set -e

echo "Fixing DAGKnight VM compilation errors..."

# 1. Fix imports in src/vm/ai/executor.rs
echo "Fixing imports in src/vm/ai/executor.rs..."
if [ -f src/vm/ai/executor.rs ]; then
  sed -i 's/use crate::cache::ModelCache;/use crate::vm::cache::ModelCache;/' src/vm/ai/executor.rs
  sed -i 's/use crate::models::ModelRegistry;/use crate::models::ModelRegistry;/' src/vm/ai/executor.rs
  sed -i 's/use crate::fault_tolerance::RecoveryManager;/use crate::fault_tolerance::RecoveryManager;/' src/vm/ai/executor.rs
  sed -i 's/, GenerationContext//g' src/vm/ai/executor.rs
  sed -i '/use std::error::Error;/d' src/vm/ai/executor.rs
fi

# 2. Fix crate reference in src/consensus/pbft.rs
echo "Fixing imports in src/consensus/pbft.rs..."
sed -i 's/use dagknight_vm::transaction::Transaction;/use crate::transaction::Transaction;/' src/consensus/pbft.rs

# 3. Add missing types to src/contracts/mod.rs
echo "Adding missing types to src/contracts/mod.rs..."
cat >> src/contracts/mod.rs << 'EOL'

/// AI model call data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIModelCall {
    /// Model identifier
    pub model: String,
    /// Input data
    pub input: Vec<u8>,
    /// Cache policy
    pub cache_policy: CachePolicy,
    /// Number of shards to use
    pub shard_count: u64,
}

/// Cache policy for model calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CachePolicy {
    /// Do not use cache
    NoCache,
    /// Use cache with TTL in seconds
    UseCache(u64),
    /// Force refresh cached value
    ForceRefresh,
}

/// Model registration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistration {
    /// Model identifier
    pub model_id: String,
    /// Model description
    pub description: String,
    /// Version string
    pub version: String,
    /// Memory required in MB
    pub memory_required: u64,
    /// Sharding capability
    pub sharding_capability: ShardingCapability,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Sharding capability for models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardingCapability {
    /// No sharding support
    None,
    /// Horizontal sharding (data parallel)
    Horizontal,
    /// Vertical sharding (model parallel)
    Vertical,
    /// Full sharding support
    Full,
}

/// Resource requirements for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum CPU cores
    pub min_cpu_cores: u32,
    /// Minimum memory in MB
    pub min_memory_mb: u64,
    /// Required GPU memory in MB
    pub gpu_memory_mb: Option<u64>,
    /// Required disk space in MB
    pub disk_space_mb: u64,
    /// Average execution time per token in ms
    pub avg_exec_time_per_token_ms: f64,
}
EOL

# 4. Fix scope issue with blockchain in src/consensus/pbft.rs
echo "Fixing blockchain scope issue in src/consensus/pbft.rs..."
sed -i 's/let _blockchain = self.blockchain.write().await;/let mut blockchain = self.blockchain.write().await;/' src/consensus/pbft.rs

# 5. Fix ollama.generate() call in src/vm/ai/executor.rs
echo "Fixing ollama.generate() call in src/vm/ai/executor.rs..."
if [ -f src/vm/ai/executor.rs ]; then
  sed -i 's/ollama.generate("nous-hermes:1b", "Are you using GPU?").await/ollama.generate(GenerationRequest::new("nous-hermes:1b", "Are you using GPU?")).await/' src/vm/ai/executor.rs
fi

# 6. Fix NetworkMessage pattern matching in src/network/p2p.rs
echo "Fixing pattern matching in src/network/p2p.rs..."
sed -i 's/NetworkMessage::Transaction { data, hash, timestamp }/NetworkMessage::Transaction { data, _hash, _timestamp }/' src/network/p2p.rs
sed -i 's/NetworkMessage::Block { data, hash, height, timestamp }/NetworkMessage::Block { data, _hash, _height, _timestamp }/' src/network/p2p.rs
sed -i 's/NetworkMessage::Consensus { consensus_type, data, timestamp }/NetworkMessage::Consensus { _consensus_type, data, _timestamp }/' src/network/p2p.rs
sed -i 's/NetworkMessage::ComputeTask { contract, model, input, timestamp }/NetworkMessage::ComputeTask { _contract, model, input, _timestamp }/' src/network/p2p.rs
sed -i 's/NetworkMessage::ComputeResult { contract, output, proof, resources, timestamp }/NetworkMessage::ComputeResult { _contract, output, _proof, _resources, _timestamp }/' src/network/p2p.rs
sed -i 's/NetworkMessage::NodeStatus { node_id, status, available_resources, timestamp }/NetworkMessage::NodeStatus { _node_id, _status, _available_resources, _timestamp }/' src/network/p2p.rs
sed -i 's/NetworkMessage::ModelRegistry { action, data, timestamp }/NetworkMessage::ModelRegistry { _action, data, _timestamp }/' src/network/p2p.rs

# 7. Add Clone derive for AIExecutionError in src/vm/ai/executor.rs
echo "Adding Clone derive for AIExecutionError..."
if [ -f src/vm/ai/executor.rs ]; then
  sed -i '/#\[derive(Debug, thiserror::Error)\]/c\#[derive(Debug, Clone, thiserror::Error)]' src/vm/ai/executor.rs
fi

# 8. Create missing module files
echo "Creating missing module files..."
mkdir -p src/api src/error src/vm/ai

# Create api module
cat > src/api/mod.rs << 'EOL'
//! API module for DAGKnight VM
EOL

# Create error module
cat > src/error/mod.rs << 'EOL'
//! Error definitions for DAGKnight VM
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DagknightError {
    #[error("VM error: {0}")]
    VmError(#[from] crate::vm::VmError),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("Consensus error: {0}")]
    ConsensusError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}
EOL

# Create vm/ai module
cat > src/vm/ai/mod.rs << 'EOL'
//! AI execution module
pub mod executor;
EOL

# Update vm/mod.rs to include ai module
if ! grep -q "pub mod ai;" src/vm/mod.rs; then
  sed -i '/\/\/ Submodules of the VM/a pub mod ai;' src/vm/mod.rs
fi

# Fix Clone trait requirement in fault_tolerance/mod.rs
echo "Fixing Clone trait requirement in src/fault_tolerance/mod.rs..."
sed -i 's/let mut final_results = results.clone();/let mut final_results = results;/' src/fault_tolerance/mod.rs

echo "All fixes applied. Running cargo check to verify..."
cargo check
