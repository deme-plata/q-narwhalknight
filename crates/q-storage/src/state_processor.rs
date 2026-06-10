//! StateProcessor v1.0.60-beta: Transaction to StateChange processing
//!
//! This module is the core of the comprehensive state sync system.
//! It processes transactions and produces atomic StateChange operations
//! that can be applied to RocksDB column families.
//!
//! ## Architecture
//!
//! ```
//! Transaction → StateProcessor → Vec<StateChange> → StateApplicator → RocksDB
//!                    ↓
//!            Validation + Gas
//! ```
//!
//! ## Key Principles
//!
//! 1. **Deterministic**: Same transaction always produces same state changes
//! 2. **Atomic**: All changes from a transaction succeed or fail together
//! 3. **Reversible**: State changes can be reversed for reorgs
//! 4. **Gas-metered**: Every operation consumes gas proportional to complexity

use anyhow::{bail, Context, Result};
use q_types::{
    Address, Amount, StateChange, StateChangeCategory, Transaction, TransactionType,
    QUG_TOKEN_ADDRESS, QUGUSD_TOKEN_ADDRESS, FOUNDER_WALLET,
    DEX_TOTAL_FEE_BPS, DEX_PROTOCOL_FEE_BPS, DEX_LP_FEE_BPS, BPS_DIVISOR,
    ProtocolFeeRecord,
    u128_serde, // v3.2.2: MessagePack P2P compatibility
};
use sha3::{Sha3_256, Digest};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, error, info, warn};

/// Base gas cost for any transaction (21,000 like Ethereum)
pub const BASE_GAS: u64 = 21_000;

/// Gas per byte of transaction data
pub const GAS_PER_DATA_BYTE: u64 = 16;

/// Gas for storage write (SSTORE equivalent)
pub const GAS_PER_STORAGE_WRITE: u64 = 20_000;

/// Gas for storage read (SLOAD equivalent)
pub const GAS_PER_STORAGE_READ: u64 = 2_100;

/// Maximum gas per transaction (100M to support higher tx throughput)
pub const MAX_GAS_PER_TX: u64 = 100_000_000; // v8.6.0: Increased from 30M to 100M for 5x block capacity

/// Execution result containing state changes and gas used
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// State changes produced by this transaction
    pub changes: Vec<StateChange>,
    /// Total gas consumed
    pub gas_used: u64,
    /// Logs/events emitted (for indexing)
    pub logs: Vec<ExecutionLog>,
    /// Error message if execution failed (changes should not be applied)
    pub error: Option<String>,
}

/// Execution log for indexing and event subscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLog {
    /// Contract or system component that emitted the log
    pub source: [u8; 32],
    /// Log topic (first 4 bytes of keccak256(event signature))
    pub topic: [u8; 4],
    /// Indexed parameters (up to 3)
    pub indexed: Vec<[u8; 32]>,
    /// Non-indexed data
    pub data: Vec<u8>,
}

/// State accessor for reading current state during execution
pub trait StateReader: Send + Sync {
    /// Get native QUG balance for an account
    fn get_balance(&self, account: &[u8; 32]) -> Result<u128>;

    /// Get token balance for an account
    fn get_token_balance(&self, account: &[u8; 32], token: &[u8; 32]) -> Result<u128>;

    /// Get account nonce
    fn get_nonce(&self, account: &[u8; 32]) -> Result<u64>;

    /// Check if token exists
    fn token_exists(&self, token: &[u8; 32]) -> Result<bool>;

    /// Get token metadata
    fn get_token_metadata(&self, token: &[u8; 32]) -> Result<Option<TokenMetadata>>;

    /// Get pool info
    fn get_pool(&self, pool_id: &[u8; 32]) -> Result<Option<PoolState>>;

    /// Get vault state
    fn get_vault(&self, vault_id: &[u8; 32]) -> Result<Option<VaultState>>;

    /// Get contract code hash
    fn get_contract_code_hash(&self, address: &[u8; 32]) -> Result<Option<[u8; 32]>>;

    /// Get contract storage value
    fn get_contract_storage(&self, address: &[u8; 32], key: &[u8; 32]) -> Result<Option<Vec<u8>>>;

    /// Get current oracle price (in 8 decimal fixed point)
    fn get_oracle_price(&self, feed_id: &[u8; 32]) -> Result<Option<u128>>;

    /// Get AI credits balance
    fn get_ai_credits(&self, account: &[u8; 32]) -> Result<u128>;

    /// Get stake amount
    fn get_stake(&self, staker: &[u8; 32], validator: &[u8; 32]) -> Result<u128>;
}

/// Token metadata structure
/// v3.2.2: Added u128_serde for MessagePack P2P compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMetadata {
    pub name: [u8; 32],
    pub symbol: [u8; 8],
    pub decimals: u8,
    pub total_supply: u128,
    pub max_supply: u128,
    pub mint_authority: [u8; 32],
    pub freeze_authority: Option<[u8; 32]>,
    pub is_mintable: bool,
}

/// Pool state for DEX
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolState {
    pub token_a: [u8; 32],
    pub token_b: [u8; 32],
    pub reserve_a: u128,
    pub reserve_b: u128,
    pub fee_bps: u16,
    pub lp_supply: u128,
}

/// Vault state for stablecoin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultState {
    pub owner: [u8; 32],
    pub collateral_amount: u128,
    pub debt_amount: u128,
    pub created_at: i64,
}

/// StateProcessor processes transactions and produces state changes
pub struct StateProcessor {
    /// Gas price in smallest unit (like gwei)
    pub gas_price: u64,
    /// Chain ID for replay protection
    pub chain_id: u64,
    /// Current block height
    pub current_height: u64,
    /// Current block timestamp
    pub current_timestamp: i64,
}

impl StateProcessor {
    /// Create a new StateProcessor
    pub fn new(gas_price: u64, chain_id: u64) -> Self {
        Self {
            gas_price,
            chain_id,
            current_height: 0,
            current_timestamp: 0,
        }
    }

    /// Set the current block context
    pub fn set_block_context(&mut self, height: u64, timestamp: i64) {
        self.current_height = height;
        self.current_timestamp = timestamp;
    }

    /// Process a transaction and produce state changes
    ///
    /// # Arguments
    /// * `tx` - The transaction to process
    /// * `state` - State reader for current state lookup
    ///
    /// # Returns
    /// ExecutionResult containing state changes, gas used, and any errors
    pub fn process_transaction<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
    ) -> Result<ExecutionResult> {
        let mut changes = Vec::new();
        let mut logs = Vec::new();
        let mut gas_used = BASE_GAS;

        // Add gas for transaction data
        gas_used += (tx.data.len() as u64) * GAS_PER_DATA_BYTE;

        // Get effective transaction type
        let tx_type = tx.effective_tx_type();

        // Add type-specific gas
        gas_used += BASE_GAS * tx_type.gas_multiplier();

        // Check gas limit
        if gas_used > MAX_GAS_PER_TX {
            return Ok(ExecutionResult {
                changes: vec![],
                gas_used,
                logs: vec![],
                error: Some(format!("Gas limit exceeded: {} > {}", gas_used, MAX_GAS_PER_TX)),
            });
        }

        // Validate transaction type matches fields
        if let Err(e) = tx.validate_tx_type() {
            return Ok(ExecutionResult {
                changes: vec![],
                gas_used,
                logs: vec![],
                error: Some(e),
            });
        }

        // Process based on transaction type
        let result = match tx_type {
            TransactionType::Transfer => {
                self.process_transfer(tx, state, &mut changes, &mut logs)
            }
            TransactionType::Coinbase => {
                self.process_coinbase(tx, &mut changes, &mut logs)
            }
            TransactionType::TokenCreate => {
                self.process_token_create(tx, state, &mut changes, &mut logs)
            }
            TransactionType::TokenMint => {
                self.process_token_mint(tx, state, &mut changes, &mut logs)
            }
            TransactionType::TokenTransfer => {
                self.process_token_transfer(tx, state, &mut changes, &mut logs)
            }
            TransactionType::TokenBurn => {
                self.process_token_burn(tx, state, &mut changes, &mut logs)
            }
            TransactionType::PoolCreate => {
                self.process_pool_create(tx, state, &mut changes, &mut logs)
            }
            TransactionType::PoolAddLiquidity => {
                self.process_add_liquidity(tx, state, &mut changes, &mut logs)
            }
            TransactionType::PoolRemoveLiquidity => {
                self.process_remove_liquidity(tx, state, &mut changes, &mut logs)
            }
            TransactionType::Swap => {
                self.process_swap(tx, state, &mut changes, &mut logs)
            }
            TransactionType::VaultLock => {
                self.process_vault_lock(tx, state, &mut changes, &mut logs)
            }
            TransactionType::VaultUnlock => {
                self.process_vault_unlock(tx, state, &mut changes, &mut logs)
            }
            TransactionType::StableMint => {
                self.process_stable_mint(tx, state, &mut changes, &mut logs)
            }
            TransactionType::StableBurn => {
                self.process_stable_burn(tx, state, &mut changes, &mut logs)
            }
            TransactionType::AICreditPurchase => {
                self.process_ai_credit_purchase(tx, state, &mut changes, &mut logs)
            }
            TransactionType::AICreditSpend => {
                self.process_ai_credit_spend(tx, state, &mut changes, &mut logs)
            }
            TransactionType::Stake => {
                self.process_stake(tx, state, &mut changes, &mut logs)
            }
            TransactionType::Unstake => {
                self.process_unstake(tx, state, &mut changes, &mut logs)
            }
            TransactionType::ClaimRewards => {
                self.process_claim_rewards(tx, state, &mut changes, &mut logs)
            }
            TransactionType::VaultLiquidate => {
                self.process_vault_liquidate(tx, state, &mut changes, &mut logs)
            }
            // TODO: Implement remaining transaction types
            _ => {
                warn!("Unimplemented transaction type: {:?}", tx_type);
                Ok(())
            }
        };

        // Check for processing error
        if let Err(e) = result {
            return Ok(ExecutionResult {
                changes: vec![],
                gas_used,
                logs: vec![],
                error: Some(e.to_string()),
            });
        }

        // Always increment nonce (unless coinbase)
        if !tx.is_coinbase() {
            changes.push(StateChange::NonceIncrement {
                account: tx.from,
                new_nonce: tx.nonce + 1,
            });
        }

        // Deduct fee (unless coinbase)
        if !tx.is_coinbase() && tx.fee > 0 {
            // v2.4.0-beta: Use TokenType::address() method (supports Custom tokens)
            let fee_token = tx.fee_token_type.address();
            changes.push(StateChange::BalanceDebit {
                account: tx.from,
                token: fee_token,
                amount: tx.fee,
            });
        }

        Ok(ExecutionResult {
            changes,
            gas_used,
            logs,
            error: None,
        })
    }

    /// Process a simple QUG/QUGUSD/Custom transfer
    fn process_transfer<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // v2.4.0-beta: Use TokenType::address() method (supports Custom tokens)
        let token = tx.token_type.address();

        // Check sender has sufficient balance
        let sender_balance = state.get_token_balance(&tx.from, &token)?;
        if sender_balance < tx.amount + tx.fee {
            bail!("Insufficient balance: have {}, need {}", sender_balance, tx.amount + tx.fee);
        }

        // Debit sender
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token,
            amount: tx.amount,
        });

        // Credit recipient
        changes.push(StateChange::BalanceCredit {
            account: tx.to,
            token,
            amount: tx.amount,
        });

        Ok(())
    }

    /// Process a coinbase (mining reward) transaction
    fn process_coinbase(
        &self,
        tx: &Transaction,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // Coinbase has no sender validation - mints new tokens
        changes.push(StateChange::BalanceCredit {
            account: tx.to,
            token: QUG_TOKEN_ADDRESS,
            amount: tx.amount,
        });

        Ok(())
    }

    /// Process custom token creation
    fn process_token_create<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // Parse token parameters from tx.data
        // v2.10.0: Extended format supports u128 supplies (16 bytes each)
        // Format: name(32) + symbol(8) + decimals(1) + initial_supply(16) + max_supply(16) = 73 bytes min
        if tx.data.len() < 73 {
            bail!("Token create data too short: need at least 73 bytes");
        }

        let mut name = [0u8; 32];
        let mut symbol = [0u8; 8];
        name.copy_from_slice(&tx.data[0..32]);
        symbol.copy_from_slice(&tx.data[32..40]);
        let decimals = tx.data[40];

        // Parse initial/max supply - support both old u64 (8 bytes) and new u128 (16 bytes) formats
        let (initial_supply, max_supply, data_offset) = if tx.data.len() >= 89 {
            // New u128 format: 16 bytes each
            let initial = u128::from_be_bytes([
                tx.data[41], tx.data[42], tx.data[43], tx.data[44],
                tx.data[45], tx.data[46], tx.data[47], tx.data[48],
                tx.data[49], tx.data[50], tx.data[51], tx.data[52],
                tx.data[53], tx.data[54], tx.data[55], tx.data[56],
            ]);
            let max = u128::from_be_bytes([
                tx.data[57], tx.data[58], tx.data[59], tx.data[60],
                tx.data[61], tx.data[62], tx.data[63], tx.data[64],
                tx.data[65], tx.data[66], tx.data[67], tx.data[68],
                tx.data[69], tx.data[70], tx.data[71], tx.data[72],
            ]);
            (initial, max, 73usize)
        } else {
            // Legacy u64 format: 8 bytes each - convert to u128
            let initial = u64::from_be_bytes([
                tx.data[41], tx.data[42], tx.data[43], tx.data[44],
                tx.data[45], tx.data[46], tx.data[47], tx.data[48],
            ]) as u128;
            let max = u64::from_be_bytes([
                tx.data[49], tx.data[50], tx.data[51], tx.data[52],
                tx.data[53], tx.data[54], tx.data[55], tx.data[56],
            ]) as u128;
            (initial, max, 57usize)
        };

        // Derive token address from creator + nonce
        let token_address = derive_token_address(&tx.from, tx.nonce);

        // Check token doesn't already exist
        if state.token_exists(&token_address)? {
            bail!("Token already exists at address");
        }

        // Is mintable flag (comes after supply bytes)
        let is_mintable = tx.data.get(data_offset).copied().unwrap_or(0) != 0;

        // Freeze authority (optional, 32 bytes after is_mintable flag)
        let freeze_authority = if tx.data.len() >= data_offset + 1 + 32 {
            let fa_start = data_offset + 1;
            let fa_end = fa_start + 32;
            if tx.data[fa_start..fa_end] != [0u8; 32] {
                let mut fa = [0u8; 32];
                fa.copy_from_slice(&tx.data[fa_start..fa_end]);
                Some(fa)
            } else {
                None
            }
        } else {
            None
        };

        // Create token
        changes.push(StateChange::TokenCreate {
            token_address,
            name,
            symbol,
            decimals,
            initial_supply,
            max_supply,
            mint_authority: tx.from, // Creator is mint authority
            freeze_authority,
            is_mintable,
        });

        // Credit initial supply to creator
        if initial_supply > 0 {
            changes.push(StateChange::BalanceCredit {
                account: tx.from,
                token: token_address,
                amount: initial_supply,
            });
        }

        Ok(())
    }

    /// Process token minting
    fn process_token_mint<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // tx.to is the token address, tx.amount is mint amount
        // tx.data contains recipient address (32 bytes)

        if tx.data.len() < 32 {
            bail!("Token mint data too short: need recipient address");
        }

        let mut recipient = [0u8; 32];
        recipient.copy_from_slice(&tx.data[0..32]);

        // Check token exists and sender is mint authority
        let metadata = state.get_token_metadata(&tx.to)?
            .ok_or_else(|| anyhow::anyhow!("Token not found"))?;

        if metadata.mint_authority != tx.from {
            bail!("Only mint authority can mint tokens");
        }

        if !metadata.is_mintable {
            bail!("Token is not mintable");
        }

        // Check max supply
        if metadata.max_supply > 0 && metadata.total_supply + tx.amount > metadata.max_supply {
            bail!("Would exceed max supply");
        }

        // Credit tokens to recipient
        changes.push(StateChange::BalanceCredit {
            account: recipient,
            token: tx.to,
            amount: tx.amount,
        });

        Ok(())
    }

    /// Process custom token transfer
    fn process_token_transfer<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // tx.data contains token address (32 bytes)
        if tx.data.len() < 32 {
            bail!("Token transfer data too short: need token address");
        }

        let mut token = [0u8; 32];
        token.copy_from_slice(&tx.data[0..32]);

        // Check sender has sufficient balance
        let sender_balance = state.get_token_balance(&tx.from, &token)?;
        if sender_balance < tx.amount {
            bail!("Insufficient token balance");
        }

        // Debit sender
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token,
            amount: tx.amount,
        });

        // Credit recipient
        changes.push(StateChange::BalanceCredit {
            account: tx.to,
            token,
            amount: tx.amount,
        });

        Ok(())
    }

    /// Process token burning
    fn process_token_burn<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // tx.to is the token address, tx.amount is burn amount

        // Check sender has sufficient balance
        let sender_balance = state.get_token_balance(&tx.from, &tx.to)?;
        if sender_balance < tx.amount {
            bail!("Insufficient token balance for burn");
        }

        // Debit sender (burn)
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token: tx.to,
            amount: tx.amount,
        });

        Ok(())
    }

    /// Process pool creation
    fn process_pool_create<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // tx.data format (v2.10.0 - u128 support):
        // [0..32] token_a address
        // [32..64] token_b address
        // [64..80] initial_a (u128 BE) or [64..72] for legacy u64
        // [80..96] initial_b (u128 BE) or [72..80] for legacy u64
        // [96..98] fee_bps (u16 BE) or [80..82] for legacy

        if tx.data.len() < 82 {
            bail!("Pool create data too short");
        }

        let mut token_a = [0u8; 32];
        let mut token_b = [0u8; 32];
        token_a.copy_from_slice(&tx.data[0..32]);
        token_b.copy_from_slice(&tx.data[32..64]);

        // Support both u128 (new) and u64 (legacy) format
        let (initial_a, initial_b, fee_bps) = if tx.data.len() >= 98 {
            // New u128 format
            let a = u128::from_be_bytes([
                tx.data[64], tx.data[65], tx.data[66], tx.data[67],
                tx.data[68], tx.data[69], tx.data[70], tx.data[71],
                tx.data[72], tx.data[73], tx.data[74], tx.data[75],
                tx.data[76], tx.data[77], tx.data[78], tx.data[79],
            ]);
            let b = u128::from_be_bytes([
                tx.data[80], tx.data[81], tx.data[82], tx.data[83],
                tx.data[84], tx.data[85], tx.data[86], tx.data[87],
                tx.data[88], tx.data[89], tx.data[90], tx.data[91],
                tx.data[92], tx.data[93], tx.data[94], tx.data[95],
            ]);
            let fee = u16::from_be_bytes([tx.data[96], tx.data[97]]);
            (a, b, fee)
        } else {
            // Legacy u64 format
            let a = u64::from_be_bytes([
                tx.data[64], tx.data[65], tx.data[66], tx.data[67],
                tx.data[68], tx.data[69], tx.data[70], tx.data[71],
            ]) as u128;
            let b = u64::from_be_bytes([
                tx.data[72], tx.data[73], tx.data[74], tx.data[75],
                tx.data[76], tx.data[77], tx.data[78], tx.data[79],
            ]) as u128;
            let fee = u16::from_be_bytes([tx.data[80], tx.data[81]]);
            (a, b, fee)
        };

        // Derive pool ID from tokens
        let pool_id = derive_pool_id(&token_a, &token_b);

        // Check pool doesn't exist
        if state.get_pool(&pool_id)?.is_some() {
            bail!("Pool already exists");
        }

        // Check creator has enough tokens
        let balance_a = state.get_token_balance(&tx.from, &token_a)?;
        let balance_b = state.get_token_balance(&tx.from, &token_b)?;
        if balance_a < initial_a || balance_b < initial_b {
            bail!("Insufficient liquidity tokens");
        }

        // Calculate initial LP tokens (sqrt(a * b)) - use u256 for multiplication
        // Since we're dealing with u128, we can safely do u128 * u128 with checked multiplication
        // and then take sqrt. For very large amounts, we use a safe calculation.
        let product = initial_a.saturating_mul(initial_b);
        let lp_supply = integer_sqrt_u128(product);

        // Debit tokens from creator
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token: token_a,
            amount: initial_a,
        });
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token: token_b,
            amount: initial_b,
        });

        // Create pool
        changes.push(StateChange::PoolCreate {
            pool_id,
            token_a,
            token_b,
            fee_bps,
            initial_a,
            initial_b,
            creator: tx.from,
            lp_supply,
        });

        // Credit LP tokens to creator
        changes.push(StateChange::LPTokenCredit {
            pool_id,
            account: tx.from,
            amount: lp_supply,
        });

        Ok(())
    }

    /// Process adding liquidity to a pool
    fn process_add_liquidity<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // tx.to is pool_id, tx.amount is amount_a
        // tx.data contains amount_b (u128 BE for new format, u64 BE for legacy)

        if tx.data.len() < 8 {
            bail!("Add liquidity data too short");
        }

        let amount_a = tx.amount;
        // Support both u128 (16 bytes) and legacy u64 (8 bytes) format
        let amount_b = if tx.data.len() >= 16 {
            u128::from_be_bytes([
                tx.data[0], tx.data[1], tx.data[2], tx.data[3],
                tx.data[4], tx.data[5], tx.data[6], tx.data[7],
                tx.data[8], tx.data[9], tx.data[10], tx.data[11],
                tx.data[12], tx.data[13], tx.data[14], tx.data[15],
            ])
        } else {
            u64::from_be_bytes([
                tx.data[0], tx.data[1], tx.data[2], tx.data[3],
                tx.data[4], tx.data[5], tx.data[6], tx.data[7],
            ]) as u128
        };

        let pool = state.get_pool(&tx.to)?
            .ok_or_else(|| anyhow::anyhow!("Pool not found"))?;

        // Check balances
        let balance_a = state.get_token_balance(&tx.from, &pool.token_a)?;
        let balance_b = state.get_token_balance(&tx.from, &pool.token_b)?;
        if balance_a < amount_a || balance_b < amount_b {
            bail!("Insufficient tokens for liquidity");
        }

        // Calculate LP tokens to mint (proportional to smaller ratio)
        // Guard against division by zero
        if pool.reserve_a == 0 || pool.reserve_b == 0 {
            bail!("Pool reserves cannot be zero");
        }
        let ratio_a = amount_a.saturating_mul(pool.lp_supply) / pool.reserve_a;
        let ratio_b = amount_b.saturating_mul(pool.lp_supply) / pool.reserve_b;
        let lp_tokens = std::cmp::min(ratio_a, ratio_b);

        // Debit tokens
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token: pool.token_a,
            amount: amount_a,
        });
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token: pool.token_b,
            amount: amount_b,
        });

        // Update pool reserves
        changes.push(StateChange::PoolReservesUpdate {
            pool_id: tx.to,
            reserve_a: pool.reserve_a + amount_a,
            reserve_b: pool.reserve_b + amount_b,
            lp_supply: pool.lp_supply + lp_tokens,
        });

        // Credit LP tokens
        changes.push(StateChange::LPTokenCredit {
            pool_id: tx.to,
            account: tx.from,
            amount: lp_tokens,
        });

        Ok(())
    }

    /// Process removing liquidity from a pool
    fn process_remove_liquidity<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // tx.to is pool_id, tx.amount is LP tokens to burn

        let pool = state.get_pool(&tx.to)?
            .ok_or_else(|| anyhow::anyhow!("Pool not found"))?;

        // Calculate tokens to return
        let share = tx.amount;
        let total_lp = pool.lp_supply;
        if total_lp == 0 {
            bail!("Pool has no LP supply");
        }
        let amount_a = pool.reserve_a.saturating_mul(share) / total_lp;
        let amount_b = pool.reserve_b.saturating_mul(share) / total_lp;

        // Burn LP tokens
        changes.push(StateChange::LPTokenDebit {
            pool_id: tx.to,
            account: tx.from,
            amount: tx.amount,
        });

        // Update pool reserves
        changes.push(StateChange::PoolReservesUpdate {
            pool_id: tx.to,
            reserve_a: pool.reserve_a - amount_a,
            reserve_b: pool.reserve_b - amount_b,
            lp_supply: pool.lp_supply - tx.amount,
        });

        // Credit tokens to user
        changes.push(StateChange::BalanceCredit {
            account: tx.from,
            token: pool.token_a,
            amount: amount_a,
        });
        changes.push(StateChange::BalanceCredit {
            account: tx.from,
            token: pool.token_b,
            amount: amount_b,
        });

        Ok(())
    }

    /// Process token swap with protocol fee split (v2.4.5-beta)
    ///
    /// Fee structure:
    /// - Total fee: 0.30% (30 bps) paid by trader
    /// - Protocol fee: 0.05% (5 bps) → FOUNDER_WALLET (dev/protocol revenue)
    /// - LP fee: 0.25% (25 bps) → stays in pool (liquidity provider rewards)
    fn process_swap<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // tx.data format:
        // [0..32] pool_id
        // [32] direction (0 = a->b, 1 = b->a)
        // [33..41] min_amount_out (u64 BE, slippage protection)

        if tx.data.len() < 41 {
            bail!("Swap data too short");
        }

        let mut pool_id = [0u8; 32];
        pool_id.copy_from_slice(&tx.data[0..32]);
        let direction = tx.data[32];
        // Support both u128 (16 bytes) and legacy u64 (8 bytes) for min_amount_out
        let min_amount_out = if tx.data.len() >= 49 {
            u128::from_be_bytes([
                tx.data[33], tx.data[34], tx.data[35], tx.data[36],
                tx.data[37], tx.data[38], tx.data[39], tx.data[40],
                tx.data[41], tx.data[42], tx.data[43], tx.data[44],
                tx.data[45], tx.data[46], tx.data[47], tx.data[48],
            ])
        } else {
            u64::from_be_bytes([
                tx.data[33], tx.data[34], tx.data[35], tx.data[36],
                tx.data[37], tx.data[38], tx.data[39], tx.data[40],
            ]) as u128
        };

        let pool = state.get_pool(&pool_id)?
            .ok_or_else(|| anyhow::anyhow!("Pool not found"))?;

        let amount_in = tx.amount;
        let (token_in, token_out, reserve_in, reserve_out) = if direction == 0 {
            (pool.token_a, pool.token_b, pool.reserve_a, pool.reserve_b)
        } else {
            (pool.token_b, pool.token_a, pool.reserve_b, pool.reserve_a)
        };

        // Check sender balance
        let balance = state.get_token_balance(&tx.from, &token_in)?;
        if balance < amount_in {
            bail!("Insufficient balance for swap");
        }

        // =========================================================================
        // v2.4.5-beta: Protocol Fee Split Implementation (u128 support in v2.10.0)
        // =========================================================================
        // Total fee = 0.30% (DEX_TOTAL_FEE_BPS = 30)
        // Protocol fee = 0.05% (DEX_PROTOCOL_FEE_BPS = 5) → Master wallet
        // LP fee = 0.25% (DEX_LP_FEE_BPS = 25) → Stays in pool
        // =========================================================================

        // Calculate protocol fee (0.05% of input goes to master wallet)
        let protocol_fee = amount_in * DEX_PROTOCOL_FEE_BPS as u128 / BPS_DIVISOR;

        // Amount after protocol fee extraction (this goes into the AMM formula)
        let amount_after_protocol_fee = amount_in.saturating_sub(protocol_fee);

        // Calculate output using constant product formula with LP fee only
        // LP fee stays in pool by reducing effective input amount
        // amount_out = reserve_out * amount_in_effective * (10000 - lp_fee_bps) /
        //              (reserve_in * 10000 + amount_in_effective * (10000 - lp_fee_bps))
        let lp_fee_multiplier = BPS_DIVISOR - DEX_LP_FEE_BPS as u128;
        let amount_in_with_lp_fee = amount_after_protocol_fee * lp_fee_multiplier;
        let numerator = reserve_out * amount_in_with_lp_fee;
        let denominator = reserve_in * BPS_DIVISOR + amount_in_with_lp_fee;
        if denominator == 0 {
            bail!("Swap denominator is zero");
        }
        let amount_out = numerator / denominator;

        // Slippage check
        if amount_out < min_amount_out {
            bail!("Slippage exceeded: {} < {}", amount_out, min_amount_out);
        }

        // Debit full input from trader (includes both fees)
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token: token_in,
            amount: amount_in,
        });

        // Credit output token to trader
        changes.push(StateChange::BalanceCredit {
            account: tx.from,
            token: token_out,
            amount: amount_out,
        });

        // Credit protocol fee to master wallet (FOUNDER_WALLET)
        // This is the 0.05% that goes to protocol development
        if protocol_fee > 0 {
            changes.push(StateChange::BalanceCredit {
                account: FOUNDER_WALLET,
                token: token_in,
                amount: protocol_fee,
            });

            // =========================================================================
            // v2.9.2-beta: Emit ProtocolFeeCollected for consensus verification
            // All nodes MUST verify this fee record matches the expected calculation
            // =========================================================================

            // Create fee_id from tx hash (deterministic across all nodes)
            let mut fee_id = [0u8; 32];
            let mut hasher = Sha3_256::new();
            hasher.update(b"protocol_fee_id_v1");
            hasher.update(&tx.id);
            hasher.update(&protocol_fee.to_le_bytes());
            fee_id.copy_from_slice(&hasher.finalize());

            // Create verification hash (deterministic across all nodes)
            // This hash binds: trade_tx_hash, fee_amount, recipient, fee_rate
            let mut verification_hash = [0u8; 32];
            let mut hasher = Sha3_256::new();
            hasher.update(&tx.id);
            hasher.update(&protocol_fee.to_le_bytes());
            hasher.update(&FOUNDER_WALLET);
            hasher.update(&0u64.to_le_bytes()); // Block height filled at block creation
            hasher.update(&(DEX_PROTOCOL_FEE_BPS as u64).to_le_bytes());
            verification_hash.copy_from_slice(&hasher.finalize());

            // Emit ProtocolFeeCollected state change for consensus verification
            changes.push(StateChange::ProtocolFeeCollected {
                fee_id,
                trade_tx_hash: tx.id,
                fee_amount: protocol_fee,
                fee_token: token_in,
                recipient: FOUNDER_WALLET,
                trade_amount: amount_in,
                fee_rate_bps: DEX_PROTOCOL_FEE_BPS,
                verification_hash,
            });

            debug!(
                "💰 DEX Protocol Fee: {} of {} ({:.3}%) → Master Wallet [consensus-verified: {}]",
                protocol_fee, token_in[0],
                (protocol_fee as f64 / amount_in as f64) * 100.0,
                hex::encode(&fee_id[..8])
            );
        }

        // Update pool reserves
        // Note: Pool receives (amount_in - protocol_fee) but outputs (amount_out)
        // The LP fee (0.25%) stays implicitly in the pool via the AMM formula
        let effective_input = amount_after_protocol_fee;
        let (new_a, new_b) = if direction == 0 {
            (pool.reserve_a + effective_input, pool.reserve_b - amount_out)
        } else {
            (pool.reserve_a - amount_out, pool.reserve_b + effective_input)
        };
        changes.push(StateChange::PoolReservesUpdate {
            pool_id,
            reserve_a: new_a,
            reserve_b: new_b,
            lp_supply: pool.lp_supply,
        });

        info!(
            "🔄 Swap: {} {} → {} {} | Protocol fee: {} → Master Wallet",
            amount_in, hex::encode(&token_in[..4]),
            amount_out, hex::encode(&token_out[..4]),
            protocol_fee
        );

        Ok(())
    }

    /// Process vault collateral locking
    fn process_vault_lock<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // Lock QUG as collateral
        let balance = state.get_token_balance(&tx.from, &QUG_TOKEN_ADDRESS)?;
        if balance < tx.amount {
            bail!("Insufficient QUG for collateral");
        }

        // Get existing vault or create new
        let vault_id = tx.from; // Vault ID = owner address
        let existing = state.get_vault(&vault_id)?;

        let (new_collateral, debt) = if let Some(v) = existing {
            (v.collateral_amount + tx.amount, v.debt_amount)
        } else {
            (tx.amount, 0)
        };

        // Debit QUG
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token: QUG_TOKEN_ADDRESS,
            amount: tx.amount,
        });

        // Calculate collateral ratio (need oracle price)
        let price = state.get_oracle_price(&QUG_TOKEN_ADDRESS)?.unwrap_or(100_000_000); // Default $1
        let collateral_value = (new_collateral as u128 * price as u128) / 100_000_000;
        let ratio_bps = if debt > 0 {
            ((collateral_value * 10000) / debt as u128) as u32
        } else {
            0 // No debt = infinite ratio
        };

        // Update vault
        changes.push(StateChange::VaultUpdate {
            vault_id,
            owner: tx.from,
            collateral_amount: new_collateral,
            debt_amount: debt,
            collateral_ratio_bps: ratio_bps,
        });

        Ok(())
    }

    /// Process vault collateral unlocking
    fn process_vault_unlock<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        let vault_id = tx.from;
        let vault = state.get_vault(&vault_id)?
            .ok_or_else(|| anyhow::anyhow!("Vault not found"))?;

        if vault.collateral_amount < tx.amount {
            bail!("Insufficient collateral to unlock");
        }

        let new_collateral = vault.collateral_amount - tx.amount;

        // Check if remaining collateral is sufficient (150% minimum)
        if vault.debt_amount > 0 {
            let price = state.get_oracle_price(&QUG_TOKEN_ADDRESS)?.unwrap_or(100_000_000);
            let collateral_value = (new_collateral as u128 * price as u128) / 100_000_000;
            let ratio = collateral_value * 100 / vault.debt_amount as u128;
            if ratio < 150 {
                bail!("Would be undercollateralized after unlock");
            }
        }

        // Credit QUG
        changes.push(StateChange::BalanceCredit {
            account: tx.from,
            token: QUG_TOKEN_ADDRESS,
            amount: tx.amount,
        });

        // Update vault
        let ratio_bps = if vault.debt_amount > 0 {
            let price = state.get_oracle_price(&QUG_TOKEN_ADDRESS)?.unwrap_or(100_000_000);
            let collateral_value = (new_collateral as u128 * price as u128) / 100_000_000;
            ((collateral_value * 10000) / vault.debt_amount as u128) as u32
        } else {
            0
        };

        changes.push(StateChange::VaultUpdate {
            vault_id,
            owner: tx.from,
            collateral_amount: new_collateral,
            debt_amount: vault.debt_amount,
            collateral_ratio_bps: ratio_bps,
        });

        Ok(())
    }

    /// Process stablecoin minting
    fn process_stable_mint<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        let vault_id = tx.from;
        let vault = state.get_vault(&vault_id)?
            .ok_or_else(|| anyhow::anyhow!("Vault not found"))?;

        let new_debt = vault.debt_amount + tx.amount;

        // Check collateral ratio (150% minimum)
        let price = state.get_oracle_price(&QUG_TOKEN_ADDRESS)?.unwrap_or(100_000_000);
        let collateral_value = (vault.collateral_amount as u128 * price as u128) / 100_000_000;
        let ratio = collateral_value * 100 / new_debt as u128;
        if ratio < 150 {
            bail!("Insufficient collateral: {}% < 150%", ratio);
        }

        // Mint QUGUSD
        changes.push(StateChange::BalanceCredit {
            account: tx.from,
            token: QUGUSD_TOKEN_ADDRESS,
            amount: tx.amount,
        });

        // Update vault
        let ratio_bps = ((collateral_value * 10000) / new_debt as u128) as u32;
        changes.push(StateChange::VaultUpdate {
            vault_id,
            owner: tx.from,
            collateral_amount: vault.collateral_amount,
            debt_amount: new_debt,
            collateral_ratio_bps: ratio_bps,
        });

        Ok(())
    }

    /// Process stablecoin burning
    fn process_stable_burn<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        let vault_id = tx.from;
        let vault = state.get_vault(&vault_id)?
            .ok_or_else(|| anyhow::anyhow!("Vault not found"))?;

        if vault.debt_amount < tx.amount {
            bail!("Burn amount exceeds debt");
        }

        // Check QUGUSD balance
        let balance = state.get_token_balance(&tx.from, &QUGUSD_TOKEN_ADDRESS)?;
        if balance < tx.amount {
            bail!("Insufficient QUGUSD");
        }

        let new_debt = vault.debt_amount - tx.amount;

        // Burn QUGUSD
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token: QUGUSD_TOKEN_ADDRESS,
            amount: tx.amount,
        });

        // Update vault
        let ratio_bps = if new_debt > 0 {
            let price = state.get_oracle_price(&QUG_TOKEN_ADDRESS)?.unwrap_or(100_000_000);
            let collateral_value = (vault.collateral_amount as u128 * price as u128) / 100_000_000;
            ((collateral_value * 10000) / new_debt as u128) as u32
        } else {
            0
        };

        changes.push(StateChange::VaultUpdate {
            vault_id,
            owner: tx.from,
            collateral_amount: vault.collateral_amount,
            debt_amount: new_debt,
            collateral_ratio_bps: ratio_bps,
        });

        Ok(())
    }

    /// Process vault liquidation (v8.7.4)
    ///
    /// Liquidates an undercollateralized vault. The liquidator pays off
    /// the vault's debt (burns QUGUSD) and receives the collateral (QUG)
    /// plus a 10% liquidation bonus.
    ///
    /// tx.from = liquidator
    /// tx.to = vault_owner (the address being liquidated)
    fn process_vault_liquidate<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        let vault_id = tx.to; // Vault owner = liquidated address
        let vault = state.get_vault(&vault_id)?
            .ok_or_else(|| anyhow::anyhow!("Vault not found for liquidation"))?;

        if vault.debt_amount == 0 {
            bail!("Vault has no debt to liquidate");
        }

        // Check if vault is actually undercollateralized (below 110%)
        let price = state.get_oracle_price(&QUG_TOKEN_ADDRESS)?.unwrap_or(100_000_000);
        let collateral_value = (vault.collateral_amount as u128 * price as u128) / 100_000_000;
        let ratio = collateral_value * 100 / vault.debt_amount as u128;
        if ratio >= 110 {
            bail!("Vault is sufficiently collateralized ({}% >= 110%)", ratio);
        }

        // Liquidator must have enough QUGUSD to cover the debt
        let liquidator_qugusd = state.get_token_balance(&tx.from, &QUGUSD_TOKEN_ADDRESS)?;
        if liquidator_qugusd < vault.debt_amount {
            bail!("Liquidator insufficient QUGUSD: have {}, need {}", liquidator_qugusd, vault.debt_amount);
        }

        // 10% liquidation bonus
        let bonus = vault.collateral_amount / 10;
        let total_seized = vault.collateral_amount;

        // Burn QUGUSD from liquidator (pays off debt)
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token: QUGUSD_TOKEN_ADDRESS,
            amount: vault.debt_amount,
        });

        // Transfer all collateral QUG to liquidator (including bonus)
        changes.push(StateChange::BalanceCredit {
            account: tx.from,
            token: QUG_TOKEN_ADDRESS,
            amount: total_seized,
        });

        // Zero out the vault
        changes.push(StateChange::VaultUpdate {
            vault_id,
            owner: tx.to,
            collateral_amount: 0,
            debt_amount: 0,
            collateral_ratio_bps: 0,
        });

        info!(
            "⚡ [STATE] Vault liquidation: liquidator={}, vault_owner={}, seized={} QUG (bonus={}), debt_burned={}",
            hex::encode(&tx.from[..8]),
            hex::encode(&tx.to[..8]),
            total_seized,
            bonus,
            vault.debt_amount,
        );

        Ok(())
    }

    /// Process AI credit purchase
    fn process_ai_credit_purchase<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // Pay QUG to receive AI credits
        let balance = state.get_token_balance(&tx.from, &QUG_TOKEN_ADDRESS)?;
        if balance < tx.amount {
            bail!("Insufficient QUG for AI credits");
        }

        // Debit QUG
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token: QUG_TOKEN_ADDRESS,
            amount: tx.amount,
        });

        // Credit AI credits (1:1 for now)
        let current_credits = state.get_ai_credits(&tx.from)?;
        changes.push(StateChange::AICreditsUpdate {
            account: tx.from,
            balance: current_credits + tx.amount,
            earned: tx.amount,
            spent: 0,
        });

        Ok(())
    }

    /// Process AI credit spending
    fn process_ai_credit_spend<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        let current_credits = state.get_ai_credits(&tx.from)?;
        if current_credits < tx.amount {
            bail!("Insufficient AI credits");
        }

        changes.push(StateChange::AICreditsUpdate {
            account: tx.from,
            balance: current_credits - tx.amount,
            earned: 0,
            spent: tx.amount,
        });

        Ok(())
    }

    /// Process staking
    fn process_stake<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // tx.to is validator address
        let balance = state.get_token_balance(&tx.from, &QUG_TOKEN_ADDRESS)?;
        if balance < tx.amount {
            bail!("Insufficient QUG for staking");
        }

        // Debit QUG
        changes.push(StateChange::BalanceDebit {
            account: tx.from,
            token: QUG_TOKEN_ADDRESS,
            amount: tx.amount,
        });

        // Update stake
        let current_stake = state.get_stake(&tx.from, &tx.to)?;
        changes.push(StateChange::StakeUpdate {
            staker: tx.from,
            validator: tx.to,
            staked_amount: current_stake + tx.amount,
            unbonding_end: 0,
            pending_rewards: 0,
        });

        Ok(())
    }

    /// Process unstaking
    fn process_unstake<S: StateReader>(
        &self,
        tx: &Transaction,
        state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        let current_stake = state.get_stake(&tx.from, &tx.to)?;
        if current_stake < tx.amount {
            bail!("Insufficient stake");
        }

        // Set unbonding period (21 days)
        let unbonding_end = self.current_timestamp + (21 * 24 * 60 * 60);

        changes.push(StateChange::StakeUpdate {
            staker: tx.from,
            validator: tx.to,
            staked_amount: current_stake - tx.amount,
            unbonding_end,
            pending_rewards: 0,
        });

        Ok(())
    }

    /// Process claiming staking rewards
    fn process_claim_rewards<S: StateReader>(
        &self,
        tx: &Transaction,
        _state: &S,
        changes: &mut Vec<StateChange>,
        _logs: &mut Vec<ExecutionLog>,
    ) -> Result<()> {
        // tx.amount contains rewards amount calculated off-chain
        // In production, this would query accumulated rewards

        changes.push(StateChange::BalanceCredit {
            account: tx.from,
            token: QUG_TOKEN_ADDRESS,
            amount: tx.amount,
        });

        Ok(())
    }
}

/// Derive token address from creator and nonce
fn derive_token_address(creator: &[u8; 32], nonce: u64) -> [u8; 32] {
    use sha3::{Sha3_256, Digest};
    let mut hasher = Sha3_256::new();
    hasher.update(b"TOKEN:");
    hasher.update(creator);
    hasher.update(&nonce.to_be_bytes());
    let result = hasher.finalize();
    let mut address = [0u8; 32];
    address.copy_from_slice(&result);
    address
}

/// Derive pool ID from token addresses (always sorted)
fn derive_pool_id(token_a: &[u8; 32], token_b: &[u8; 32]) -> [u8; 32] {
    use sha3::{Sha3_256, Digest};
    let mut hasher = Sha3_256::new();
    hasher.update(b"POOL:");
    // Sort tokens for deterministic pool ID
    if token_a < token_b {
        hasher.update(token_a);
        hasher.update(token_b);
    } else {
        hasher.update(token_b);
        hasher.update(token_a);
    }
    let result = hasher.finalize();
    let mut pool_id = [0u8; 32];
    pool_id.copy_from_slice(&result);
    pool_id
}

/// Integer square root for u128 using Newton's method
/// Used for LP token calculations with large amounts
fn integer_sqrt_u128(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockStateReader {
        balances: HashMap<([u8; 32], [u8; 32]), u128>,
        nonces: HashMap<[u8; 32], u64>,
    }

    impl MockStateReader {
        fn new() -> Self {
            Self {
                balances: HashMap::new(),
                nonces: HashMap::new(),
            }
        }

        fn set_balance(&mut self, account: [u8; 32], token: [u8; 32], amount: u128) {
            self.balances.insert((account, token), amount);
        }
    }

    impl StateReader for MockStateReader {
        fn get_balance(&self, account: &[u8; 32]) -> Result<u128> {
            Ok(self.balances.get(&(*account, QUG_TOKEN_ADDRESS)).copied().unwrap_or(0))
        }

        fn get_token_balance(&self, account: &[u8; 32], token: &[u8; 32]) -> Result<u128> {
            Ok(self.balances.get(&(*account, *token)).copied().unwrap_or(0))
        }

        fn get_nonce(&self, account: &[u8; 32]) -> Result<u64> {
            Ok(self.nonces.get(account).copied().unwrap_or(0))
        }

        fn token_exists(&self, _token: &[u8; 32]) -> Result<bool> {
            Ok(false)
        }

        fn get_token_metadata(&self, _token: &[u8; 32]) -> Result<Option<TokenMetadata>> {
            Ok(None)
        }

        fn get_pool(&self, _pool_id: &[u8; 32]) -> Result<Option<PoolState>> {
            Ok(None)
        }

        fn get_vault(&self, _vault_id: &[u8; 32]) -> Result<Option<VaultState>> {
            Ok(None)
        }

        fn get_contract_code_hash(&self, _address: &[u8; 32]) -> Result<Option<[u8; 32]>> {
            Ok(None)
        }

        fn get_contract_storage(&self, _address: &[u8; 32], _key: &[u8; 32]) -> Result<Option<Vec<u8>>> {
            Ok(None)
        }

        fn get_oracle_price(&self, _feed_id: &[u8; 32]) -> Result<Option<u128>> {
            Ok(Some(100_000_000)) // $1
        }

        fn get_ai_credits(&self, _account: &[u8; 32]) -> Result<u128> {
            Ok(0)
        }

        fn get_stake(&self, _staker: &[u8; 32], _validator: &[u8; 32]) -> Result<u128> {
            Ok(0)
        }
    }

    #[test]
    fn test_coinbase_transaction() {
        let processor = StateProcessor::new(1, 2025);
        let state = MockStateReader::new();

        let mut miner = [0u8; 32];
        miner[0] = 0x01;

        let tx = Transaction {
            id: [0u8; 32],
            from: [0u8; 32], // Zero address = coinbase
            to: miner,
            amount: 5_000_000, // 0.05 QUG
            fee: 0,
            nonce: 0,
            signature: vec![],
            timestamp: chrono::Utc::now(),
            data: vec![],
            token_type: q_types::TokenType::QUG,
            fee_token_type: q_types::TokenType::QUGUSD,
            tx_type: q_types::TransactionType::Coinbase,
        };

        let result = processor.process_transaction(&tx, &state).unwrap();
        assert!(result.error.is_none());
        assert_eq!(result.changes.len(), 1); // Just the credit

        match &result.changes[0] {
            StateChange::BalanceCredit { account, token, amount } => {
                assert_eq!(account, &miner);
                assert_eq!(token, &QUG_TOKEN_ADDRESS);
                assert_eq!(*amount, 5_000_000);
            }
            _ => panic!("Expected BalanceCredit"),
        }
    }

    #[test]
    fn test_transfer_transaction() {
        let processor = StateProcessor::new(1, 2025);
        let mut state = MockStateReader::new();

        let mut sender = [0u8; 32];
        sender[0] = 0x01;
        let mut recipient = [0u8; 32];
        recipient[0] = 0x02;

        // Give sender some balance
        state.set_balance(sender, QUG_TOKEN_ADDRESS, 100_000_000);

        let tx = Transaction {
            id: [0u8; 32],
            from: sender,
            to: recipient,
            amount: 50_000_000,
            fee: 1_000,
            nonce: 0,
            signature: vec![],
            timestamp: chrono::Utc::now(),
            data: vec![],
            token_type: q_types::TokenType::QUG,
            fee_token_type: q_types::TokenType::QUGUSD,
            tx_type: q_types::TransactionType::Transfer,
        };

        let result = processor.process_transaction(&tx, &state).unwrap();
        assert!(result.error.is_none());

        // Should have: debit, credit, nonce increment, fee debit
        assert!(result.changes.len() >= 3);

        // Find the balance changes
        let mut found_debit = false;
        let mut found_credit = false;
        for change in &result.changes {
            match change {
                StateChange::BalanceDebit { account, amount, .. } if *account == sender && *amount == 50_000_000 => {
                    found_debit = true;
                }
                StateChange::BalanceCredit { account, amount, .. } if *account == recipient && *amount == 50_000_000 => {
                    found_credit = true;
                }
                _ => {}
            }
        }
        assert!(found_debit, "Should have debit from sender");
        assert!(found_credit, "Should have credit to recipient");
    }

    #[test]
    fn test_insufficient_balance() {
        let processor = StateProcessor::new(1, 2025);
        let state = MockStateReader::new(); // No balance

        let mut sender = [0u8; 32];
        sender[0] = 0x01;
        let mut recipient = [0u8; 32];
        recipient[0] = 0x02;

        let tx = Transaction {
            id: [0u8; 32],
            from: sender,
            to: recipient,
            amount: 50_000_000,
            fee: 1_000,
            nonce: 0,
            signature: vec![],
            timestamp: chrono::Utc::now(),
            data: vec![],
            token_type: q_types::TokenType::QUG,
            fee_token_type: q_types::TokenType::QUGUSD,
            tx_type: q_types::TransactionType::Transfer,
        };

        let result = processor.process_transaction(&tx, &state).unwrap();
        assert!(result.error.is_some());
        assert!(result.error.as_ref().unwrap().contains("Insufficient balance"));
        assert!(result.changes.is_empty());
    }

    #[test]
    fn test_derive_pool_id_deterministic() {
        let token_a = [1u8; 32];
        let token_b = [2u8; 32];

        // Order shouldn't matter
        let pool1 = derive_pool_id(&token_a, &token_b);
        let pool2 = derive_pool_id(&token_b, &token_a);
        assert_eq!(pool1, pool2);
    }

    #[test]
    fn test_derive_token_address() {
        let creator = [1u8; 32];
        let addr1 = derive_token_address(&creator, 0);
        let addr2 = derive_token_address(&creator, 1);

        // Different nonces should give different addresses
        assert_ne!(addr1, addr2);

        // Same inputs should give same output
        let addr3 = derive_token_address(&creator, 0);
        assert_eq!(addr1, addr3);
    }
}
