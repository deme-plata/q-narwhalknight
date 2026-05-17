use anyhow::Result;
use serde::{Deserialize, Serialize};
/// State management for DAGKnight
use std::sync::Arc;
use tokio::sync::RwLock;

/// Storage backend trait for persistent state
#[async_trait::async_trait]
pub trait StateStorage: Send + Sync {
    async fn save_state(&self, state: &VmState) -> Result<()>;
    async fn load_state(&self) -> Result<Option<VmState>>;
    async fn save_checkpoint(&self, height: u64, state: &VmState) -> Result<()>;
    async fn load_checkpoint(&self, height: u64) -> Result<Option<VmState>>;
    async fn get_state_root(&self) -> Result<[u8; 32]>;
}

/// VM state with enhanced persistence and integrity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmState {
    pub contracts: std::collections::HashMap<u64, Vec<u8>>,
    pub storage: std::collections::HashMap<u64, std::collections::HashMap<Vec<u8>, Vec<u8>>>,
    pub balances: std::collections::HashMap<u64, u64>,
    pub nonces: std::collections::HashMap<u64, u64>,
    pub state_root: [u8; 32],
    pub block_height: u64,
    pub last_update: std::time::SystemTime,
}

impl Default for VmState {
    fn default() -> Self {
        Self {
            contracts: std::collections::HashMap::new(),
            storage: std::collections::HashMap::new(),
            balances: std::collections::HashMap::new(),
            nonces: std::collections::HashMap::new(),
            state_root: [0u8; 32],
            block_height: 0,
            last_update: std::time::SystemTime::now(),
        }
    }
}

impl VmState {
    /// Calculate the state root hash
    pub fn calculate_state_root(&self) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};

        let serialized = bincode::serialize(self).unwrap_or_default();
        let mut hasher = Sha3_256::new();
        hasher.update(&serialized);
        hasher.finalize().into()
    }

    /// Update state root after modifications
    pub fn update_state_root(&mut self) {
        self.state_root = self.calculate_state_root();
        self.last_update = std::time::SystemTime::now();
    }
}

pub struct StateDB {
    pub state: Arc<RwLock<VmState>>,
    pub resource_ledger: Option<Box<dyn std::any::Any + Send + Sync>>,
    pub storage: Option<Arc<dyn StateStorage>>,
    pub auto_persist: bool,
    pub checkpoint_interval: u64, // Checkpoint every N blocks
}

impl std::fmt::Debug for StateDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateDB")
            .field("auto_persist", &self.auto_persist)
            .field("checkpoint_interval", &self.checkpoint_interval)
            .field("has_storage", &self.storage.is_some())
            .field("has_resource_ledger", &self.resource_ledger.is_some())
            .finish()
    }
}

impl StateDB {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(VmState::default())),
            resource_ledger: None,
            storage: None,
            auto_persist: false,
            checkpoint_interval: 100, // Default: checkpoint every 100 blocks
        }
    }

    pub fn with_state(state: Arc<RwLock<VmState>>) -> Self {
        Self {
            state,
            resource_ledger: None,
            storage: None,
            auto_persist: false,
            checkpoint_interval: 100,
        }
    }

    pub fn with_storage(storage: Arc<dyn StateStorage>) -> Self {
        Self {
            state: Arc::new(RwLock::new(VmState::default())),
            resource_ledger: None,
            storage: Some(storage),
            auto_persist: true,
            checkpoint_interval: 100,
        }
    }

    // Add a new method for testing purposes
    pub fn new_in_memory() -> Self {
        Self::new()
    }

    /// Load state from persistent storage
    pub async fn load_from_storage(&self) -> Result<()> {
        if let Some(storage) = &self.storage {
            if let Some(loaded_state) = storage.load_state().await? {
                let mut state = self.state.write().await;
                *state = loaded_state;
                tracing::info!(
                    "📂 State loaded from persistent storage at height {}",
                    state.block_height
                );
            }
        }
        Ok(())
    }

    /// Save state to persistent storage
    pub async fn save_to_storage(&self) -> Result<()> {
        if let Some(storage) = &self.storage {
            let state = self.state.read().await;
            storage.save_state(&*state).await?;
            tracing::debug!(
                "💾 State persisted to storage at height {}",
                state.block_height
            );
        }
        Ok(())
    }

    /// Create a checkpoint at the specified block height
    pub async fn checkpoint(&self, height: u64) -> Result<()> {
        if let Some(storage) = &self.storage {
            let mut state = self.state.write().await;
            state.block_height = height;
            state.update_state_root();

            storage.save_checkpoint(height, &*state).await?;
            tracing::info!(
                "📸 Checkpoint created at height {} with state root {}",
                height,
                hex::encode(state.state_root)
            );
        }
        Ok(())
    }

    /// Load state from a specific checkpoint
    pub async fn load_checkpoint(&self, height: u64) -> Result<bool> {
        if let Some(storage) = &self.storage {
            if let Some(checkpoint_state) = storage.load_checkpoint(height).await? {
                let mut state = self.state.write().await;
                *state = checkpoint_state;
                tracing::info!("🔄 State restored from checkpoint at height {}", height);
                return Ok(true);
            }
        }
        Ok(false)
    }
}

// Resource usage struct for AI execution
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub compute_units: u64,
    pub memory_bytes: u64,
    pub storage_bytes: u64,
    pub cpu_time: u64,
    pub memory_used: u64,
    pub gpu_time: u64,
}

use crate::vm::{ContractState, StateAccess, VmError};

#[async_trait::async_trait]
impl StateAccess for StateDB {
    async fn get_contract(&self, address: u64) -> Result<Option<Vec<u8>>, VmError> {
        let state = self.state.read().await;
        Ok(state.contracts.get(&address).cloned())
    }

    async fn get_storage(&self, address: u64, key: &[u8]) -> Result<Option<Vec<u8>>, VmError> {
        let state = self.state.read().await;
        Ok(state
            .storage
            .get(&address)
            .and_then(|storage| storage.get(key))
            .cloned())
    }

    async fn set_storage(&self, address: u64, key: Vec<u8>, value: Vec<u8>) -> Result<(), VmError> {
        {
            let mut state = self.state.write().await;
            state
                .storage
                .entry(address)
                .or_insert_with(std::collections::HashMap::new)
                .insert(key, value);
            state.update_state_root();
        }

        // Auto-persist if enabled
        if self.auto_persist {
            if let Err(e) = self.save_to_storage().await {
                tracing::warn!("Failed to auto-persist state: {}", e);
            }
        }

        Ok(())
    }

    async fn get_balance(&self, address: u64) -> Result<u64, VmError> {
        let state = self.state.read().await;
        Ok(state.balances.get(&address).copied().unwrap_or(0))
    }

    async fn set_balance(&self, address: u64, amount: u64) -> Result<(), VmError> {
        {
            let mut state = self.state.write().await;
            state.balances.insert(address, amount);
            state.update_state_root();
        }

        // Auto-persist if enabled
        if self.auto_persist {
            if let Err(e) = self.save_to_storage().await {
                tracing::warn!("Failed to auto-persist state: {}", e);
            }
        }

        Ok(())
    }

    async fn get_nonce(&self, address: u64) -> Result<u64, VmError> {
        let state = self.state.read().await;
        Ok(state.nonces.get(&address).copied().unwrap_or(0))
    }

    async fn get_contract_state(&self, address: u64) -> Result<Option<ContractState>, VmError> {
        let state = self.state.read().await;
        if let Some(contract_code) = state.contracts.get(&address) {
            Ok(Some(ContractState {
                code: contract_code.clone(),
                storage: state.storage.get(&address).cloned().unwrap_or_default(),
                balance: state.balances.get(&address).copied().unwrap_or(0),
                nonce: state.nonces.get(&address).copied().unwrap_or(0),
            }))
        } else {
            Ok(None)
        }
    }
}

/// In-memory implementation of StateStorage for testing
pub struct InMemoryStateStorage {
    current_state: RwLock<Option<VmState>>,
    checkpoints: RwLock<std::collections::HashMap<u64, VmState>>,
}

impl InMemoryStateStorage {
    pub fn new() -> Self {
        Self {
            current_state: RwLock::new(None),
            checkpoints: RwLock::new(std::collections::HashMap::new()),
        }
    }
}

#[async_trait::async_trait]
impl StateStorage for InMemoryStateStorage {
    async fn save_state(&self, state: &VmState) -> Result<()> {
        let mut current = self.current_state.write().await;
        *current = Some(state.clone());
        Ok(())
    }

    async fn load_state(&self) -> Result<Option<VmState>> {
        let current = self.current_state.read().await;
        Ok(current.clone())
    }

    async fn save_checkpoint(&self, height: u64, state: &VmState) -> Result<()> {
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.insert(height, state.clone());
        Ok(())
    }

    async fn load_checkpoint(&self, height: u64) -> Result<Option<VmState>> {
        let checkpoints = self.checkpoints.read().await;
        Ok(checkpoints.get(&height).cloned())
    }

    async fn get_state_root(&self) -> Result<[u8; 32]> {
        let current = self.current_state.read().await;
        Ok(current.as_ref().map(|s| s.state_root).unwrap_or_default())
    }
}
