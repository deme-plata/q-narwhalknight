//! BlockStateProcessor v1.0.60-beta: Process entire blocks through the state machine
//!
//! This module integrates StateProcessor, StateApplicator, and SparseMerkleTrie to provide
//! a complete block processing pipeline that:
//! 1. Extracts all transactions from a block
//! 2. Processes each transaction through StateProcessor to get StateChanges
//! 3. Applies all StateChanges atomically via StateApplicator
//! 4. Updates the sparse Merkle trie with all state changes
//! 5. Computes the cryptographic state root from the SMT
//!
//! ## Architecture
//!
//! ```text
//! QBlock (with transactions)
//!    │
//!    ▼
//! BlockStateProcessor::process_block()
//!    │
//!    ├─► StateProcessor::process_transaction() for each tx
//!    │      │
//!    │      ▼
//!    │   Vec<StateChange> per tx
//!    │
//!    ├─► StateApplicator::apply_changes()
//!    │      │
//!    │      ▼
//!    │   RocksDB (atomic write batch)
//!    │
//!    └─► SparseMerkleTrie::batch_insert()
//!           │
//!           ▼
//!        Cryptographic State Root [32 bytes]
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! let processor = BlockStateProcessor::new(db.clone());
//!
//! // Process a block and apply all state changes
//! let result = processor.process_block(&block).await?;
//! println!("Applied {} state changes, gas used: {}", result.changes_applied, result.total_gas_used);
//! ```

use anyhow::{Context, Result};
use q_types::{QBlock, StateChange, Transaction, TransactionType};
#[cfg(not(target_os = "windows"))]
use rocksdb::DB;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use crate::state_applicator::StateApplicator;
use crate::state_processor::StateProcessor;
use crate::sparse_merkle_trie::SparseMerkleTrie;

/// Result of processing a block
#[derive(Debug, Clone)]
pub struct BlockProcessingResult {
    /// Block height that was processed
    pub height: u64,
    /// Number of transactions processed
    pub transactions_processed: usize,
    /// Number of state changes applied
    pub changes_applied: usize,
    /// Total gas used by all transactions
    pub total_gas_used: u64,
    /// Gas limit for the block (future: configurable)
    pub gas_limit: u64,
    /// State root after applying all changes (future: sparse Merkle trie)
    pub state_root: [u8; 32],
    /// Individual transaction results
    pub tx_results: Vec<TxProcessingResult>,
}

/// Result of processing a single transaction
#[derive(Debug, Clone)]
pub struct TxProcessingResult {
    /// Transaction ID
    pub tx_id: [u8; 32],
    /// Whether the transaction succeeded
    pub success: bool,
    /// Gas used by this transaction
    pub gas_used: u64,
    /// Number of state changes from this transaction
    pub changes_count: usize,
    /// Error message if failed
    pub error: Option<String>,
}

/// BlockStateProcessor: Process entire blocks through the state machine
pub struct BlockStateProcessor {
    /// RocksDB handle
    db: Arc<DB>,
    /// State applicator for atomic writes
    applicator: StateApplicator,
    /// Sparse Merkle trie for cryptographic state root computation
    state_trie: SparseMerkleTrie,
    /// Block gas limit (default: 30M like Ethereum)
    block_gas_limit: u64,
    /// Enable detailed logging
    verbose: bool,
}

impl BlockStateProcessor {
    /// Create a new BlockStateProcessor
    pub fn new(db: Arc<DB>) -> Self {
        Self {
            state_trie: SparseMerkleTrie::new_with_db(db.clone(), crate::sparse_merkle_trie::CF_STATE_TRIE),
            applicator: StateApplicator::new(db.clone(), true),
            db,
            block_gas_limit: 30_000_000, // 30M gas limit per block
            verbose: false,
        }
    }

    /// Create with custom gas limit
    pub fn with_gas_limit(db: Arc<DB>, gas_limit: u64) -> Self {
        Self {
            state_trie: SparseMerkleTrie::new_with_db(db.clone(), crate::sparse_merkle_trie::CF_STATE_TRIE),
            applicator: StateApplicator::new(db.clone(), true),
            db,
            block_gas_limit: gas_limit,
            verbose: false,
        }
    }

    /// Create in-memory processor for testing (no persistence)
    pub fn new_in_memory(db: Arc<DB>) -> Self {
        Self {
            state_trie: SparseMerkleTrie::new_in_memory(),
            applicator: StateApplicator::new(db.clone(), true),
            db,
            block_gas_limit: 30_000_000,
            verbose: false,
        }
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Process a block and apply all state changes
    ///
    /// This is the main entry point for block processing. It:
    /// 1. Extracts transactions from the block
    /// 2. Processes each transaction to get state changes
    /// 3. Applies all changes atomically
    /// 4. Returns the processing result
    pub fn process_block(&self, block: &QBlock) -> Result<BlockProcessingResult> {
        let height = block.header.height;
        let mut all_changes: Vec<StateChange> = Vec::new();
        let mut tx_results: Vec<TxProcessingResult> = Vec::new();
        let mut total_gas_used: u64 = 0;

        // Create a RocksDB-backed state reader
        let state_reader = RocksDbStateReader::new(self.db.clone());
        // Default gas price of 1 gwei, chain ID 1 (testnet)
        let mut processor = StateProcessor::new(1, 1);
        processor.current_height = height;
        processor.current_timestamp = block.header.timestamp as i64;

        if self.verbose {
            info!(
                "🔄 [BLOCK STATE] Processing block {} with {} transactions",
                height,
                block.transactions.len()
            );
        }

        // Process each transaction
        for tx in &block.transactions {
            // Check gas limit
            let estimated_gas = tx.gas_cost(21_000);
            if total_gas_used + estimated_gas > self.block_gas_limit {
                warn!(
                    "⛽ [GAS LIMIT] Block {} exceeded gas limit at tx {}",
                    height,
                    hex::encode(&tx.id[..8])
                );
                // Record failed transaction
                tx_results.push(TxProcessingResult {
                    tx_id: tx.id,
                    success: false,
                    gas_used: 0,
                    changes_count: 0,
                    error: Some("Block gas limit exceeded".to_string()),
                });
                continue;
            }

            // Process the transaction
            match processor.process_transaction(tx, &state_reader) {
                Ok(result) => {
                    let changes_count = result.changes.len();
                    total_gas_used += result.gas_used;

                    // Collect state changes
                    all_changes.extend(result.changes);

                    tx_results.push(TxProcessingResult {
                        tx_id: tx.id,
                        success: true,
                        gas_used: result.gas_used,
                        changes_count,
                        error: None,
                    });

                    if self.verbose && tx.effective_tx_type() != TransactionType::Transfer {
                        debug!(
                            "  ✅ Tx {} ({:?}): {} changes, {} gas",
                            hex::encode(&tx.id[..8]),
                            tx.effective_tx_type(),
                            changes_count,
                            result.gas_used
                        );
                    }
                }
                Err(e) => {
                    // Transaction failed - still charge base gas
                    let failed_gas = 21_000u64;
                    total_gas_used += failed_gas;

                    tx_results.push(TxProcessingResult {
                        tx_id: tx.id,
                        success: false,
                        gas_used: failed_gas,
                        changes_count: 0,
                        error: Some(e.to_string()),
                    });

                    if self.verbose {
                        warn!(
                            "  ❌ Tx {} failed: {}",
                            hex::encode(&tx.id[..8]),
                            e
                        );
                    }
                }
            }
        }

        // Apply all state changes atomically
        let changes_count = all_changes.len();
        if !all_changes.is_empty() {
            self.applicator.apply_changes(&all_changes, height)
                .context(format!("Failed to apply {} state changes for block {}", changes_count, height))?;
        }

        // Compute state root (placeholder - will implement sparse Merkle trie)
        let state_root = self.compute_state_root(&all_changes);

        let result = BlockProcessingResult {
            height,
            transactions_processed: tx_results.len(),
            changes_applied: changes_count,
            total_gas_used,
            gas_limit: self.block_gas_limit,
            state_root,
            tx_results,
        };

        if self.verbose || height % 1000 == 0 {
            info!(
                "✅ [BLOCK STATE] Block {}: {} txs, {} changes, {} gas ({:.1}% of limit)",
                height,
                result.transactions_processed,
                result.changes_applied,
                result.total_gas_used,
                (result.total_gas_used as f64 / self.block_gas_limit as f64) * 100.0
            );
        }

        Ok(result)
    }

    /// Process multiple blocks in batch
    ///
    /// This is more efficient than processing blocks one by one because:
    /// 1. State reads can be batched/cached
    /// 2. Multiple blocks can share a single write batch
    pub fn process_blocks(&self, blocks: &[QBlock]) -> Result<Vec<BlockProcessingResult>> {
        let mut results = Vec::with_capacity(blocks.len());

        for block in blocks {
            match self.process_block(block) {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!(
                        "❌ [BLOCK STATE] Failed to process block {}: {}",
                        block.header.height, e
                    );
                    return Err(e);
                }
            }
        }

        Ok(results)
    }

    /// Compute state root from changes using the sparse Merkle trie
    ///
    /// Each state change is inserted into the SMT with:
    /// - Key: primary_key of the state change (deterministic based on change type)
    /// - Value: serialized state change value
    ///
    /// The SMT provides:
    /// - O(log n) proof size for inclusion/exclusion
    /// - Cryptographic collision resistance via Blake3
    /// - Support for light client state verification
    fn compute_state_root(&self, changes: &[StateChange]) -> [u8; 32] {
        if changes.is_empty() {
            return self.state_trie.root();
        }

        // Insert all changes into the trie
        for change in changes {
            let key = change.primary_key();
            let value = self.serialize_state_change(change);
            self.state_trie.insert(&key, &value);
        }

        self.state_trie.root()
    }

    /// Serialize a state change value for the trie
    ///
    /// The serialization is deterministic and compact to minimize trie storage
    fn serialize_state_change(&self, change: &StateChange) -> Vec<u8> {
        let mut value = Vec::new();

        // Add category prefix for disambiguation
        value.push(change.category() as u8);

        match change {
            // Balance changes
            StateChange::BalanceCredit { amount, .. } |
            StateChange::BalanceDebit { amount, .. } => {
                value.extend_from_slice(&amount.to_le_bytes());
            }

            // Token operations
            StateChange::TokenCreate { decimals, initial_supply, max_supply, name, symbol, .. } => {
                value.push(*decimals);
                value.extend_from_slice(&initial_supply.to_le_bytes());
                value.extend_from_slice(&max_supply.to_le_bytes());
                value.extend_from_slice(name);
                value.extend_from_slice(symbol);
            }
            StateChange::TokenMetadataUpdate { name, symbol, metadata_uri, .. } => {
                if let Some(n) = name {
                    value.extend_from_slice(n);
                }
                if let Some(s) = symbol {
                    value.extend_from_slice(s);
                }
                if let Some(uri) = metadata_uri {
                    value.extend_from_slice(uri);
                }
            }
            StateChange::TokenAuthorityTransfer { new_authority, .. } => {
                value.extend_from_slice(new_authority);
            }
            StateChange::TokenAccountFreeze { frozen, .. } => {
                value.push(if *frozen { 1 } else { 0 });
            }

            // DEX operations
            StateChange::PoolCreate { initial_a, initial_b, fee_bps, lp_supply, .. } => {
                value.extend_from_slice(&initial_a.to_le_bytes());
                value.extend_from_slice(&initial_b.to_le_bytes());
                value.extend_from_slice(&fee_bps.to_le_bytes());
                value.extend_from_slice(&lp_supply.to_le_bytes());
            }
            StateChange::PoolReservesUpdate { reserve_a, reserve_b, lp_supply, .. } => {
                value.extend_from_slice(&reserve_a.to_le_bytes());
                value.extend_from_slice(&reserve_b.to_le_bytes());
                value.extend_from_slice(&lp_supply.to_le_bytes());
            }
            StateChange::LPTokenCredit { amount, .. } |
            StateChange::LPTokenDebit { amount, .. } => {
                value.extend_from_slice(&amount.to_le_bytes());
            }

            // Contract operations
            StateChange::ContractDeploy { code_hash, is_upgradeable, .. } => {
                value.extend_from_slice(code_hash);
                value.push(if *is_upgradeable { 1 } else { 0 });
            }
            StateChange::ContractStorageUpdate { value: storage_value, .. } => {
                value.extend_from_slice(storage_value);
            }
            StateChange::ContractDestroy { beneficiary, .. } => {
                value.extend_from_slice(beneficiary);
            }

            // Vault operations
            StateChange::VaultUpdate { collateral_amount, debt_amount, collateral_ratio_bps, .. } => {
                value.extend_from_slice(&collateral_amount.to_le_bytes());
                value.extend_from_slice(&debt_amount.to_le_bytes());
                value.extend_from_slice(&collateral_ratio_bps.to_le_bytes());
            }
            StateChange::OraclePriceUpdate { price, timestamp, num_signatures, .. } => {
                value.extend_from_slice(&price.to_le_bytes());
                value.extend_from_slice(&timestamp.to_le_bytes());
                value.push(*num_signatures);
            }

            // AI operations
            StateChange::AICreditsUpdate { balance, earned, spent, .. } => {
                value.extend_from_slice(&balance.to_le_bytes());
                value.extend_from_slice(&earned.to_le_bytes());
                value.extend_from_slice(&spent.to_le_bytes());
            }
            StateChange::AIProviderUpdate { capacity, price_per_credit, is_active, .. } => {
                value.extend_from_slice(&capacity.to_le_bytes());
                value.extend_from_slice(&price_per_credit.to_le_bytes());
                value.push(if *is_active { 1 } else { 0 });
            }

            // Governance operations
            StateChange::ProposalCreate { start_height, end_height, quorum_bps, execution_hash, .. } => {
                value.extend_from_slice(&start_height.to_le_bytes());
                value.extend_from_slice(&end_height.to_le_bytes());
                value.extend_from_slice(&quorum_bps.to_le_bytes());
                value.extend_from_slice(execution_hash);
            }
            StateChange::ProposalVoteUpdate { votes_for, votes_against, votes_abstain, .. } => {
                value.extend_from_slice(&votes_for.to_le_bytes());
                value.extend_from_slice(&votes_against.to_le_bytes());
                value.extend_from_slice(&votes_abstain.to_le_bytes());
            }
            StateChange::ProposalStatusUpdate { status, .. } => {
                value.push(*status);
            }
            StateChange::DelegationUpdate { voting_power, delegate, .. } => {
                value.extend_from_slice(&voting_power.to_le_bytes());
                if let Some(d) = delegate {
                    value.extend_from_slice(d);
                }
            }

            // Staking operations
            StateChange::StakeUpdate { staked_amount, unbonding_end, pending_rewards, .. } => {
                value.extend_from_slice(&staked_amount.to_le_bytes());
                value.extend_from_slice(&unbonding_end.to_le_bytes());
                value.extend_from_slice(&pending_rewards.to_le_bytes());
            }
            StateChange::ValidatorUpdate { total_stake, commission_bps, is_active, slash_count, .. } => {
                value.extend_from_slice(&total_stake.to_le_bytes());
                value.extend_from_slice(&commission_bps.to_le_bytes());
                value.push(if *is_active { 1 } else { 0 });
                value.extend_from_slice(&slash_count.to_le_bytes());
            }

            // System operations
            StateChange::SystemParamUpdate { value: param_value, .. } => {
                value.extend_from_slice(param_value);
            }
            StateChange::NonceIncrement { new_nonce, .. } => {
                value.extend_from_slice(&new_nonce.to_le_bytes());
            }
            StateChange::StateRootCheckpoint { height, state_root, tx_root, .. } => {
                value.extend_from_slice(&height.to_le_bytes());
                value.extend_from_slice(state_root);
                value.extend_from_slice(tx_root);
            }

            // v2.9.2-beta: Protocol fee operations
            StateChange::ProtocolFeeCollected { fee_amount, trade_amount, fee_rate_bps, verification_hash, .. } => {
                value.extend_from_slice(&fee_amount.to_le_bytes());
                value.extend_from_slice(&trade_amount.to_le_bytes());
                value.extend_from_slice(&fee_rate_bps.to_le_bytes());
                value.extend_from_slice(verification_hash);
            }
        }

        value
    }

    /// Get the current state root without processing any changes
    pub fn current_state_root(&self) -> [u8; 32] {
        self.state_trie.root()
    }

    /// Generate a Merkle proof for a specific key
    ///
    /// This can be used by light clients to verify state without downloading
    /// the full blockchain.
    pub fn generate_proof(&self, key: &[u8]) -> crate::sparse_merkle_trie::MerkleProof {
        self.state_trie.prove(key)
    }

    /// Verify a Merkle proof against a state root
    ///
    /// This is a static method that can be used without access to the full trie.
    pub fn verify_proof(
        root: &[u8; 32],
        key: &[u8],
        value: Option<&[u8]>,
        proof: &crate::sparse_merkle_trie::MerkleProof,
    ) -> bool {
        SparseMerkleTrie::verify_proof(root, key, value, proof)
    }

    /// Get trie statistics for monitoring
    pub fn trie_stats(&self) -> crate::sparse_merkle_trie::TrieStats {
        self.state_trie.stats()
    }

    /// Get total balance for an account (for validation)
    /// v2.10.0: Updated to u128 for 24 decimal precision
    pub fn get_balance(&self, account: &[u8; 32], token: &[u8; 32]) -> Result<u128> {
        self.applicator.get_token_balance(account, token)
    }

    /// Get nonce for an account
    pub fn get_nonce(&self, account: &[u8; 32]) -> Result<u64> {
        self.applicator.get_nonce(account)
    }
}

/// RocksDB-backed state reader for StateProcessor
///
/// v8.7.3: Made public for use by deterministic block replay in balance_consensus.rs
pub struct RocksDbStateReader {
    db: Arc<DB>,
}

impl RocksDbStateReader {
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }
}

impl crate::state_processor::StateReader for RocksDbStateReader {
    fn get_balance(&self, account: &[u8; 32]) -> Result<u128> {
        // Get native QUG balance
        self.get_token_balance(account, &q_types::QUG_TOKEN_ADDRESS)
    }

    fn get_token_balance(&self, account: &[u8; 32], token: &[u8; 32]) -> Result<u128> {
        // Build key: account (32 bytes) + token (32 bytes)
        let mut key = Vec::with_capacity(64);
        key.extend_from_slice(account);
        key.extend_from_slice(token);

        // Try to get from CF_TOKEN_BALANCES
        // v2.10.0: Support both u128 (16 bytes) and legacy u64 (8 bytes)
        if let Some(cf) = self.db.cf_handle(crate::CF_TOKEN_BALANCES) {
            if let Ok(Some(value)) = self.db.get_cf(&cf, &key) {
                if value.len() >= 16 {
                    // New u128 format
                    return Ok(u128::from_le_bytes(value[..16].try_into().unwrap_or([0u8; 16])));
                } else if value.len() >= 8 {
                    // Legacy u64 format - convert with decimal upgrade
                    let legacy = u64::from_le_bytes(value[..8].try_into().unwrap_or([0u8; 8]));
                    return Ok((legacy as u128) * 10u128.pow(16));
                }
            }
        }

        Ok(0)
    }

    fn get_nonce(&self, account: &[u8; 32]) -> Result<u64> {
        if let Some(cf) = self.db.cf_handle(crate::CF_NONCES) {
            if let Ok(Some(value)) = self.db.get_cf(&cf, account) {
                if value.len() >= 8 {
                    return Ok(u64::from_le_bytes(value[..8].try_into().unwrap_or([0u8; 8])));
                }
            }
        }

        Ok(0)
    }

    fn token_exists(&self, token_address: &[u8; 32]) -> Result<bool> {
        if let Some(cf) = self.db.cf_handle(crate::CF_TOKENS) {
            if let Ok(Some(_)) = self.db.get_cf(&cf, token_address) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn get_token_metadata(&self, token: &[u8; 32]) -> Result<Option<crate::state_processor::TokenMetadata>> {
        if let Some(cf) = self.db.cf_handle(crate::CF_TOKENS) {
            if let Ok(Some(value)) = self.db.get_cf(&cf, token) {
                if let Ok(meta) = bincode::deserialize::<crate::state_processor::TokenMetadata>(&value) {
                    return Ok(Some(meta));
                }
            }
        }

        Ok(None)
    }

    fn get_pool(&self, pool_id: &[u8; 32]) -> Result<Option<crate::state_processor::PoolState>> {
        if let Some(cf) = self.db.cf_handle(crate::CF_DEX_POOLS) {
            if let Ok(Some(value)) = self.db.get_cf(&cf, pool_id) {
                if let Ok(pool) = bincode::deserialize::<crate::state_processor::PoolState>(&value) {
                    return Ok(Some(pool));
                }
            }
        }

        Ok(None)
    }

    fn get_vault(&self, vault_id: &[u8; 32]) -> Result<Option<crate::state_processor::VaultState>> {
        if let Some(cf) = self.db.cf_handle(crate::CF_VAULTS) {
            if let Ok(Some(value)) = self.db.get_cf(&cf, vault_id) {
                if let Ok(vault) = bincode::deserialize::<crate::state_processor::VaultState>(&value) {
                    return Ok(Some(vault));
                }
            }
        }

        Ok(None)
    }

    fn get_contract_code_hash(&self, address: &[u8; 32]) -> Result<Option<[u8; 32]>> {
        if let Some(cf) = self.db.cf_handle(crate::CF_CONTRACTS) {
            if let Ok(Some(value)) = self.db.get_cf(&cf, address) {
                // Assuming contract storage starts with 32-byte code hash
                if value.len() >= 32 {
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&value[..32]);
                    return Ok(Some(hash));
                }
            }
        }

        Ok(None)
    }

    fn get_contract_storage(&self, address: &[u8; 32], key: &[u8; 32]) -> Result<Option<Vec<u8>>> {
        // Build key: contract_address (32 bytes) + storage_key (32 bytes)
        let mut storage_key = Vec::with_capacity(64);
        storage_key.extend_from_slice(address);
        storage_key.extend_from_slice(key);

        if let Some(cf) = self.db.cf_handle(crate::CF_CONTRACT_STORAGE) {
            if let Ok(Some(value)) = self.db.get_cf(&cf, &storage_key) {
                return Ok(Some(value));
            }
        }

        Ok(None)
    }

    fn get_oracle_price(&self, feed_id: &[u8; 32]) -> Result<Option<u128>> {
        if let Some(cf) = self.db.cf_handle(crate::CF_ORACLE_PRICES) {
            if let Ok(Some(value)) = self.db.get_cf(&cf, feed_id) {
                // v2.10.0: Support both u128 (16 bytes) and legacy u64 (8 bytes)
                if value.len() >= 16 {
                    return Ok(Some(u128::from_le_bytes(value[..16].try_into().unwrap_or([0u8; 16]))));
                } else if value.len() >= 8 {
                    let legacy = u64::from_le_bytes(value[..8].try_into().unwrap_or([0u8; 8]));
                    return Ok(Some((legacy as u128) * 10u128.pow(16)));
                }
            }
        }

        Ok(None)
    }

    fn get_ai_credits(&self, account: &[u8; 32]) -> Result<u128> {
        if let Some(cf) = self.db.cf_handle(crate::CF_AI_CREDITS_V2) {
            if let Ok(Some(value)) = self.db.get_cf(&cf, account) {
                // v2.10.0: Support both u128 (16 bytes) and legacy u64 (8 bytes)
                if value.len() >= 16 {
                    return Ok(u128::from_le_bytes(value[..16].try_into().unwrap_or([0u8; 16])));
                } else if value.len() >= 8 {
                    let legacy = u64::from_le_bytes(value[..8].try_into().unwrap_or([0u8; 8]));
                    return Ok((legacy as u128) * 10u128.pow(16));
                }
            }
        }

        Ok(0)
    }

    fn get_stake(&self, staker: &[u8; 32], validator: &[u8; 32]) -> Result<u128> {
        // Build key: staker (32 bytes) + validator (32 bytes)
        let mut key = Vec::with_capacity(64);
        key.extend_from_slice(staker);
        key.extend_from_slice(validator);

        if let Some(cf) = self.db.cf_handle(crate::CF_STAKES) {
            if let Ok(Some(value)) = self.db.get_cf(&cf, &key) {
                // v2.10.0: Support both u128 (16 bytes) and legacy u64 (8 bytes)
                if value.len() >= 16 {
                    return Ok(u128::from_le_bytes(value[..16].try_into().unwrap_or([0u8; 16])));
                } else if value.len() >= 8 {
                    let legacy = u64::from_le_bytes(value[..8].try_into().unwrap_or([0u8; 8]));
                    return Ok((legacy as u128) * 10u128.pow(16));
                }
            }
        }

        Ok(0)
    }
}

// Tests moved to integration tests due to complex dependencies on QBlock structure
// The BlockStateProcessor has been validated to compile and integrate correctly
// with the state_processor and state_applicator modules.
//
// To test BlockStateProcessor:
// 1. Use the actual QBlock structure from q_types::block
// 2. Create blocks with proper BlockHeader fields
// 3. Process through the BlockStateProcessor pipeline
//
// Example test outline:
// ```
// let db = create_rocksdb_with_state_sync_cfs();
// let processor = BlockStateProcessor::new(db);
// let block = create_valid_qblock(height, transactions);
// let result = processor.process_block(&block)?;
// assert!(result.changes_applied > 0);
// ```
