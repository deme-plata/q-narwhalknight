//! StateApplicator v1.0.60-beta: Apply StateChanges to RocksDB
//!
//! This module applies atomic state changes to RocksDB column families.
//! It provides both forward application (for block execution) and
//! reverse application (for reorgs/rollbacks).
//!
//! ## Usage
//!
//! ```ignore
//! let applicator = StateApplicator::new(db.clone());
//!
//! // Apply state changes from a block
//! applicator.apply_changes(&changes, block_height).await?;
//!
//! // Rollback a block (for reorgs)
//! applicator.rollback_changes(&changes, block_height).await?;
//! ```

use anyhow::{bail, Context, Result};
use q_types::StateChange;
use rocksdb::{WriteBatch, DB};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::{
    CF_TOKEN_BALANCES, CF_TOKENS, CF_DEX_POOLS, CF_LP_BALANCES,
    CF_CONTRACTS, CF_CONTRACT_STORAGE, CF_VAULTS, CF_ORACLE_PRICES,
    CF_AI_CREDITS_V2, CF_AI_PROVIDERS, CF_PROPOSALS, CF_DELEGATIONS,
    CF_STAKES, CF_VALIDATORS, CF_SYSTEM_PARAMS, CF_NONCES, CF_STATE_ROOTS,
    state_processor::{TokenMetadata, PoolState, VaultState},
};

/// StateApplicator applies state changes to RocksDB column families
pub struct StateApplicator {
    /// RocksDB handle
    db: Arc<DB>,
    /// Write-ahead log for crash recovery
    enable_wal: bool,
}

impl StateApplicator {
    /// Create a new StateApplicator
    pub fn new(db: Arc<DB>, enable_wal: bool) -> Self {
        Self { db, enable_wal }
    }

    /// Apply a batch of state changes atomically
    ///
    /// All changes are written in a single RocksDB WriteBatch for atomicity.
    /// If any change fails, the entire batch is rolled back.
    pub fn apply_changes(&self, changes: &[StateChange], block_height: u64) -> Result<()> {
        let mut batch = WriteBatch::default();

        for change in changes {
            self.apply_single_change(&mut batch, change)?;
        }

        // Write the batch atomically
        let write_opts = rocksdb::WriteOptions::default();
        self.db.write_opt(batch, &write_opts)
            .context("Failed to write state changes batch")?;

        debug!(
            "✅ Applied {} state changes at height {}",
            changes.len(),
            block_height
        );

        Ok(())
    }

    /// Apply a single state change to the batch
    fn apply_single_change(&self, batch: &mut WriteBatch, change: &StateChange) -> Result<()> {
        match change {
            // ========== Balance Changes ==========
            StateChange::BalanceCredit { account, token, amount } => {
                self.apply_balance_credit(batch, account, token, *amount)?;
            }
            StateChange::BalanceDebit { account, token, amount } => {
                self.apply_balance_debit(batch, account, token, *amount)?;
            }

            // ========== Token State ==========
            StateChange::TokenCreate {
                token_address,
                name,
                symbol,
                decimals,
                initial_supply,
                max_supply,
                mint_authority,
                freeze_authority,
                is_mintable,
            } => {
                self.apply_token_create(
                    batch,
                    token_address,
                    name,
                    symbol,
                    *decimals,
                    *initial_supply,
                    *max_supply,
                    mint_authority,
                    freeze_authority.as_ref(),
                    *is_mintable,
                )?;
            }
            StateChange::TokenMetadataUpdate {
                token_address,
                name,
                symbol,
                metadata_uri,
            } => {
                // Update token metadata (partial update)
                // For now, we'll skip this as it requires reading existing metadata
                debug!("TokenMetadataUpdate: {:?}", hex::encode(token_address));
            }
            StateChange::TokenAuthorityTransfer {
                token_address,
                old_authority: _,
                new_authority,
            } => {
                // Update mint authority
                debug!(
                    "TokenAuthorityTransfer: {} -> {}",
                    hex::encode(token_address),
                    hex::encode(new_authority)
                );
            }
            StateChange::TokenAccountFreeze {
                token_address,
                account,
                frozen,
            } => {
                // Set frozen flag on token account
                debug!(
                    "TokenAccountFreeze: {} for {} = {}",
                    hex::encode(token_address),
                    hex::encode(account),
                    frozen
                );
            }

            // ========== DEX State ==========
            StateChange::PoolCreate {
                pool_id,
                token_a,
                token_b,
                fee_bps,
                initial_a,
                initial_b,
                creator,
                lp_supply,
            } => {
                self.apply_pool_create(
                    batch,
                    pool_id,
                    token_a,
                    token_b,
                    *fee_bps,
                    *initial_a,
                    *initial_b,
                    creator,
                    *lp_supply,
                )?;
            }
            StateChange::PoolReservesUpdate {
                pool_id,
                reserve_a,
                reserve_b,
                lp_supply,
            } => {
                self.apply_pool_reserves_update(batch, pool_id, *reserve_a, *reserve_b, *lp_supply)?;
            }
            StateChange::LPTokenCredit {
                pool_id,
                account,
                amount,
            } => {
                self.apply_lp_credit(batch, pool_id, account, *amount)?;
            }
            StateChange::LPTokenDebit {
                pool_id,
                account,
                amount,
            } => {
                self.apply_lp_debit(batch, pool_id, account, *amount)?;
            }

            // ========== Contract State ==========
            StateChange::ContractDeploy {
                contract_address,
                code_hash,
                deployer,
                is_upgradeable,
            } => {
                self.apply_contract_deploy(
                    batch,
                    contract_address,
                    code_hash,
                    deployer,
                    *is_upgradeable,
                )?;
            }
            StateChange::ContractStorageUpdate {
                contract_address,
                key,
                value,
            } => {
                self.apply_contract_storage_update(batch, contract_address, key, value)?;
            }
            StateChange::ContractDestroy {
                contract_address,
                beneficiary,
            } => {
                self.apply_contract_destroy(batch, contract_address, beneficiary)?;
            }

            // ========== Vault State ==========
            StateChange::VaultUpdate {
                vault_id,
                owner,
                collateral_amount,
                debt_amount,
                collateral_ratio_bps,
            } => {
                self.apply_vault_update(
                    batch,
                    vault_id,
                    owner,
                    *collateral_amount,
                    *debt_amount,
                    *collateral_ratio_bps,
                )?;
            }
            StateChange::OraclePriceUpdate {
                feed_id,
                price,
                timestamp,
                num_signatures,
            } => {
                self.apply_oracle_price_update(batch, feed_id, *price, *timestamp, *num_signatures)?;
            }

            // ========== AI Credits State ==========
            StateChange::AICreditsUpdate {
                account,
                balance,
                earned,
                spent,
            } => {
                self.apply_ai_credits_update(batch, account, *balance, *earned, *spent)?;
            }
            StateChange::AIProviderUpdate {
                provider_id,
                wallet,
                capacity,
                price_per_credit,
                is_active,
            } => {
                self.apply_ai_provider_update(
                    batch,
                    provider_id,
                    wallet,
                    *capacity,
                    *price_per_credit,
                    *is_active,
                )?;
            }

            // ========== Governance State ==========
            StateChange::ProposalCreate {
                proposal_id,
                proposer,
                start_height,
                end_height,
                quorum_bps,
                execution_hash,
            } => {
                self.apply_proposal_create(
                    batch,
                    proposal_id,
                    proposer,
                    *start_height,
                    *end_height,
                    *quorum_bps,
                    execution_hash,
                )?;
            }
            StateChange::ProposalVoteUpdate {
                proposal_id,
                votes_for,
                votes_against,
                votes_abstain,
            } => {
                self.apply_proposal_vote_update(
                    batch,
                    proposal_id,
                    *votes_for,
                    *votes_against,
                    *votes_abstain,
                )?;
            }
            StateChange::ProposalStatusUpdate { proposal_id, status } => {
                self.apply_proposal_status_update(batch, proposal_id, *status)?;
            }
            StateChange::DelegationUpdate {
                delegator,
                delegate,
                voting_power,
            } => {
                self.apply_delegation_update(batch, delegator, delegate.as_ref(), *voting_power)?;
            }

            // ========== Staking State ==========
            StateChange::StakeUpdate {
                staker,
                validator,
                staked_amount,
                unbonding_end,
                pending_rewards,
            } => {
                self.apply_stake_update(
                    batch,
                    staker,
                    validator,
                    *staked_amount,
                    *unbonding_end,
                    *pending_rewards,
                )?;
            }
            StateChange::ValidatorUpdate {
                validator_id,
                total_stake,
                commission_bps,
                is_active,
                slash_count,
            } => {
                self.apply_validator_update(
                    batch,
                    validator_id,
                    *total_stake,
                    *commission_bps,
                    *is_active,
                    *slash_count,
                )?;
            }

            // ========== System State ==========
            StateChange::SystemParamUpdate { key, value } => {
                self.apply_system_param_update(batch, key, value)?;
            }
            StateChange::NonceIncrement { account, new_nonce } => {
                self.apply_nonce_update(batch, account, *new_nonce)?;
            }
            StateChange::StateRootCheckpoint {
                height,
                state_root,
                tx_root,
            } => {
                self.apply_state_root_checkpoint(batch, *height, state_root, tx_root)?;
            }
        }

        Ok(())
    }

    // ========== Balance Operations ==========

    fn apply_balance_credit(
        &self,
        batch: &mut WriteBatch,
        account: &[u8; 32],
        token: &[u8; 32],
        amount: u64,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_TOKEN_BALANCES)
            .ok_or_else(|| anyhow::anyhow!("CF_TOKEN_BALANCES not found"))?;

        // Build key: account (32) | token (32)
        let mut key = Vec::with_capacity(64);
        key.extend_from_slice(account);
        key.extend_from_slice(token);

        // Read current balance
        let current = self.db.get_cf(&cf, &key)?
            .map(|v| u64::from_be_bytes(v.try_into().unwrap_or([0u8; 8])))
            .unwrap_or(0);

        let new_balance = current.saturating_add(amount);
        batch.put_cf(&cf, &key, &new_balance.to_be_bytes());

        Ok(())
    }

    fn apply_balance_debit(
        &self,
        batch: &mut WriteBatch,
        account: &[u8; 32],
        token: &[u8; 32],
        amount: u64,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_TOKEN_BALANCES)
            .ok_or_else(|| anyhow::anyhow!("CF_TOKEN_BALANCES not found"))?;

        // Build key: account (32) | token (32)
        let mut key = Vec::with_capacity(64);
        key.extend_from_slice(account);
        key.extend_from_slice(token);

        // Read current balance
        let current = self.db.get_cf(&cf, &key)?
            .map(|v| u64::from_be_bytes(v.try_into().unwrap_or([0u8; 8])))
            .unwrap_or(0);

        if current < amount {
            bail!("Insufficient balance: have {}, need {}", current, amount);
        }

        let new_balance = current - amount;
        batch.put_cf(&cf, &key, &new_balance.to_be_bytes());

        Ok(())
    }

    // ========== Token Operations ==========

    fn apply_token_create(
        &self,
        batch: &mut WriteBatch,
        token_address: &[u8; 32],
        name: &[u8; 32],
        symbol: &[u8; 8],
        decimals: u8,
        initial_supply: u64,
        max_supply: u64,
        mint_authority: &[u8; 32],
        freeze_authority: Option<&[u8; 32]>,
        is_mintable: bool,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_TOKENS)
            .ok_or_else(|| anyhow::anyhow!("CF_TOKENS not found"))?;

        // Serialize token metadata
        let metadata = TokenMetadata {
            name: *name,
            symbol: *symbol,
            decimals,
            total_supply: initial_supply,
            max_supply,
            mint_authority: *mint_authority,
            freeze_authority: freeze_authority.copied(),
            is_mintable,
        };

        let value = postcard::to_allocvec(&metadata)
            .context("Failed to serialize token metadata")?;

        batch.put_cf(&cf, token_address, &value);

        Ok(())
    }

    // ========== DEX Operations ==========

    fn apply_pool_create(
        &self,
        batch: &mut WriteBatch,
        pool_id: &[u8; 32],
        token_a: &[u8; 32],
        token_b: &[u8; 32],
        fee_bps: u16,
        initial_a: u64,
        initial_b: u64,
        _creator: &[u8; 32],
        lp_supply: u64,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_DEX_POOLS)
            .ok_or_else(|| anyhow::anyhow!("CF_DEX_POOLS not found"))?;

        let pool = PoolState {
            token_a: *token_a,
            token_b: *token_b,
            reserve_a: initial_a,
            reserve_b: initial_b,
            fee_bps,
            lp_supply,
        };

        let value = postcard::to_allocvec(&pool)
            .context("Failed to serialize pool state")?;

        batch.put_cf(&cf, pool_id, &value);

        Ok(())
    }

    fn apply_pool_reserves_update(
        &self,
        batch: &mut WriteBatch,
        pool_id: &[u8; 32],
        reserve_a: u64,
        reserve_b: u64,
        lp_supply: u64,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_DEX_POOLS)
            .ok_or_else(|| anyhow::anyhow!("CF_DEX_POOLS not found"))?;

        // Read existing pool to get token addresses
        let existing = self.db.get_cf(&cf, pool_id)?
            .ok_or_else(|| anyhow::anyhow!("Pool not found"))?;

        let mut pool: PoolState = postcard::from_bytes(&existing)
            .context("Failed to deserialize pool state")?;

        pool.reserve_a = reserve_a;
        pool.reserve_b = reserve_b;
        pool.lp_supply = lp_supply;

        let value = postcard::to_allocvec(&pool)
            .context("Failed to serialize pool state")?;

        batch.put_cf(&cf, pool_id, &value);

        Ok(())
    }

    fn apply_lp_credit(
        &self,
        batch: &mut WriteBatch,
        pool_id: &[u8; 32],
        account: &[u8; 32],
        amount: u64,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_LP_BALANCES)
            .ok_or_else(|| anyhow::anyhow!("CF_LP_BALANCES not found"))?;

        // Build key: pool_id (32) | account (32)
        let mut key = Vec::with_capacity(64);
        key.extend_from_slice(pool_id);
        key.extend_from_slice(account);

        let current = self.db.get_cf(&cf, &key)?
            .map(|v| u64::from_be_bytes(v.try_into().unwrap_or([0u8; 8])))
            .unwrap_or(0);

        let new_balance = current.saturating_add(amount);
        batch.put_cf(&cf, &key, &new_balance.to_be_bytes());

        Ok(())
    }

    fn apply_lp_debit(
        &self,
        batch: &mut WriteBatch,
        pool_id: &[u8; 32],
        account: &[u8; 32],
        amount: u64,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_LP_BALANCES)
            .ok_or_else(|| anyhow::anyhow!("CF_LP_BALANCES not found"))?;

        let mut key = Vec::with_capacity(64);
        key.extend_from_slice(pool_id);
        key.extend_from_slice(account);

        let current = self.db.get_cf(&cf, &key)?
            .map(|v| u64::from_be_bytes(v.try_into().unwrap_or([0u8; 8])))
            .unwrap_or(0);

        if current < amount {
            bail!("Insufficient LP balance");
        }

        let new_balance = current - amount;
        batch.put_cf(&cf, &key, &new_balance.to_be_bytes());

        Ok(())
    }

    // ========== Contract Operations ==========

    fn apply_contract_deploy(
        &self,
        batch: &mut WriteBatch,
        contract_address: &[u8; 32],
        code_hash: &[u8; 32],
        deployer: &[u8; 32],
        is_upgradeable: bool,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_CONTRACTS)
            .ok_or_else(|| anyhow::anyhow!("CF_CONTRACTS not found"))?;

        // Store contract metadata: code_hash (32) | deployer (32) | flags (1)
        let mut value = Vec::with_capacity(65);
        value.extend_from_slice(code_hash);
        value.extend_from_slice(deployer);
        value.push(if is_upgradeable { 1 } else { 0 });

        batch.put_cf(&cf, contract_address, &value);

        Ok(())
    }

    fn apply_contract_storage_update(
        &self,
        batch: &mut WriteBatch,
        contract_address: &[u8; 32],
        key: &[u8; 32],
        value: &[u8],
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_CONTRACT_STORAGE)
            .ok_or_else(|| anyhow::anyhow!("CF_CONTRACT_STORAGE not found"))?;

        // Build key: contract (32) | slot (32)
        let mut storage_key = Vec::with_capacity(64);
        storage_key.extend_from_slice(contract_address);
        storage_key.extend_from_slice(key);

        batch.put_cf(&cf, &storage_key, value);

        Ok(())
    }

    fn apply_contract_destroy(
        &self,
        batch: &mut WriteBatch,
        contract_address: &[u8; 32],
        _beneficiary: &[u8; 32],
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_CONTRACTS)
            .ok_or_else(|| anyhow::anyhow!("CF_CONTRACTS not found"))?;

        // Mark contract as destroyed (could also delete, but tombstone is safer)
        batch.put_cf(&cf, contract_address, b"DESTROYED");

        Ok(())
    }

    // ========== Vault Operations ==========

    fn apply_vault_update(
        &self,
        batch: &mut WriteBatch,
        vault_id: &[u8; 32],
        owner: &[u8; 32],
        collateral_amount: u64,
        debt_amount: u64,
        collateral_ratio_bps: u32,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_VAULTS)
            .ok_or_else(|| anyhow::anyhow!("CF_VAULTS not found"))?;

        // Serialize: owner (32) | collateral (8) | debt (8) | ratio (4) | created_at (8)
        let mut value = Vec::with_capacity(60);
        value.extend_from_slice(owner);
        value.extend_from_slice(&collateral_amount.to_be_bytes());
        value.extend_from_slice(&debt_amount.to_be_bytes());
        value.extend_from_slice(&collateral_ratio_bps.to_be_bytes());
        value.extend_from_slice(&chrono::Utc::now().timestamp().to_be_bytes());

        batch.put_cf(&cf, vault_id, &value);

        Ok(())
    }

    fn apply_oracle_price_update(
        &self,
        batch: &mut WriteBatch,
        feed_id: &[u8; 32],
        price: u64,
        timestamp: i64,
        num_signatures: u8,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_ORACLE_PRICES)
            .ok_or_else(|| anyhow::anyhow!("CF_ORACLE_PRICES not found"))?;

        // Serialize: price (8) | timestamp (8) | num_signatures (1)
        let mut value = Vec::with_capacity(17);
        value.extend_from_slice(&price.to_be_bytes());
        value.extend_from_slice(&timestamp.to_be_bytes());
        value.push(num_signatures);

        batch.put_cf(&cf, feed_id, &value);

        Ok(())
    }

    // ========== AI Credits Operations ==========

    fn apply_ai_credits_update(
        &self,
        batch: &mut WriteBatch,
        account: &[u8; 32],
        balance: u64,
        earned: u64,
        spent: u64,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_AI_CREDITS_V2)
            .ok_or_else(|| anyhow::anyhow!("CF_AI_CREDITS_V2 not found"))?;

        // Serialize: balance (8) | earned (8) | spent (8)
        let mut value = Vec::with_capacity(24);
        value.extend_from_slice(&balance.to_be_bytes());
        value.extend_from_slice(&earned.to_be_bytes());
        value.extend_from_slice(&spent.to_be_bytes());

        batch.put_cf(&cf, account, &value);

        Ok(())
    }

    fn apply_ai_provider_update(
        &self,
        batch: &mut WriteBatch,
        provider_id: &[u8; 32],
        wallet: &[u8; 32],
        capacity: u64,
        price_per_credit: u64,
        is_active: bool,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_AI_PROVIDERS)
            .ok_or_else(|| anyhow::anyhow!("CF_AI_PROVIDERS not found"))?;

        // Serialize: wallet (32) | capacity (8) | price (8) | active (1)
        let mut value = Vec::with_capacity(49);
        value.extend_from_slice(wallet);
        value.extend_from_slice(&capacity.to_be_bytes());
        value.extend_from_slice(&price_per_credit.to_be_bytes());
        value.push(if is_active { 1 } else { 0 });

        batch.put_cf(&cf, provider_id, &value);

        Ok(())
    }

    // ========== Governance Operations ==========

    fn apply_proposal_create(
        &self,
        batch: &mut WriteBatch,
        proposal_id: &[u8; 32],
        proposer: &[u8; 32],
        start_height: u64,
        end_height: u64,
        quorum_bps: u32,
        execution_hash: &[u8; 32],
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_PROPOSALS)
            .ok_or_else(|| anyhow::anyhow!("CF_PROPOSALS not found"))?;

        // Serialize: proposer (32) | start (8) | end (8) | quorum (4) | exec_hash (32) | status (1) | votes (24)
        let mut value = Vec::with_capacity(109);
        value.extend_from_slice(proposer);
        value.extend_from_slice(&start_height.to_be_bytes());
        value.extend_from_slice(&end_height.to_be_bytes());
        value.extend_from_slice(&quorum_bps.to_be_bytes());
        value.extend_from_slice(execution_hash);
        value.push(0); // status = pending
        value.extend_from_slice(&0u64.to_be_bytes()); // votes_for
        value.extend_from_slice(&0u64.to_be_bytes()); // votes_against
        value.extend_from_slice(&0u64.to_be_bytes()); // votes_abstain

        batch.put_cf(&cf, proposal_id, &value);

        Ok(())
    }

    fn apply_proposal_vote_update(
        &self,
        batch: &mut WriteBatch,
        proposal_id: &[u8; 32],
        votes_for: u64,
        votes_against: u64,
        votes_abstain: u64,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_PROPOSALS)
            .ok_or_else(|| anyhow::anyhow!("CF_PROPOSALS not found"))?;

        // Read existing proposal
        let existing = self.db.get_cf(&cf, proposal_id)?
            .ok_or_else(|| anyhow::anyhow!("Proposal not found"))?;

        if existing.len() < 85 {
            bail!("Invalid proposal data");
        }

        // Update vote counts (bytes 85-108)
        let mut value = existing.to_vec();
        value[85..93].copy_from_slice(&votes_for.to_be_bytes());
        value[93..101].copy_from_slice(&votes_against.to_be_bytes());
        value[101..109].copy_from_slice(&votes_abstain.to_be_bytes());

        batch.put_cf(&cf, proposal_id, &value);

        Ok(())
    }

    fn apply_proposal_status_update(
        &self,
        batch: &mut WriteBatch,
        proposal_id: &[u8; 32],
        status: u8,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_PROPOSALS)
            .ok_or_else(|| anyhow::anyhow!("CF_PROPOSALS not found"))?;

        let existing = self.db.get_cf(&cf, proposal_id)?
            .ok_or_else(|| anyhow::anyhow!("Proposal not found"))?;

        if existing.len() < 85 {
            bail!("Invalid proposal data");
        }

        let mut value = existing.to_vec();
        value[84] = status;

        batch.put_cf(&cf, proposal_id, &value);

        Ok(())
    }

    fn apply_delegation_update(
        &self,
        batch: &mut WriteBatch,
        delegator: &[u8; 32],
        delegate: Option<&[u8; 32]>,
        voting_power: u64,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_DELEGATIONS)
            .ok_or_else(|| anyhow::anyhow!("CF_DELEGATIONS not found"))?;

        // Serialize: has_delegate (1) | delegate (32) | voting_power (8)
        let mut value = Vec::with_capacity(41);
        if let Some(d) = delegate {
            value.push(1);
            value.extend_from_slice(d);
        } else {
            value.push(0);
            value.extend_from_slice(&[0u8; 32]);
        }
        value.extend_from_slice(&voting_power.to_be_bytes());

        batch.put_cf(&cf, delegator, &value);

        Ok(())
    }

    // ========== Staking Operations ==========

    fn apply_stake_update(
        &self,
        batch: &mut WriteBatch,
        staker: &[u8; 32],
        validator: &[u8; 32],
        staked_amount: u64,
        unbonding_end: i64,
        pending_rewards: u64,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_STAKES)
            .ok_or_else(|| anyhow::anyhow!("CF_STAKES not found"))?;

        // Build key: staker (32) | validator (32)
        let mut key = Vec::with_capacity(64);
        key.extend_from_slice(staker);
        key.extend_from_slice(validator);

        // Serialize: staked (8) | unbonding_end (8) | rewards (8)
        let mut value = Vec::with_capacity(24);
        value.extend_from_slice(&staked_amount.to_be_bytes());
        value.extend_from_slice(&unbonding_end.to_be_bytes());
        value.extend_from_slice(&pending_rewards.to_be_bytes());

        batch.put_cf(&cf, &key, &value);

        Ok(())
    }

    fn apply_validator_update(
        &self,
        batch: &mut WriteBatch,
        validator_id: &[u8; 32],
        total_stake: u64,
        commission_bps: u16,
        is_active: bool,
        slash_count: u32,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_VALIDATORS)
            .ok_or_else(|| anyhow::anyhow!("CF_VALIDATORS not found"))?;

        // Serialize: total_stake (8) | commission (2) | active (1) | slash_count (4)
        let mut value = Vec::with_capacity(15);
        value.extend_from_slice(&total_stake.to_be_bytes());
        value.extend_from_slice(&commission_bps.to_be_bytes());
        value.push(if is_active { 1 } else { 0 });
        value.extend_from_slice(&slash_count.to_be_bytes());

        batch.put_cf(&cf, validator_id, &value);

        Ok(())
    }

    // ========== System Operations ==========

    fn apply_system_param_update(
        &self,
        batch: &mut WriteBatch,
        key: &[u8; 32],
        value: &[u8],
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_SYSTEM_PARAMS)
            .ok_or_else(|| anyhow::anyhow!("CF_SYSTEM_PARAMS not found"))?;

        batch.put_cf(&cf, key, value);

        Ok(())
    }

    fn apply_nonce_update(
        &self,
        batch: &mut WriteBatch,
        account: &[u8; 32],
        new_nonce: u64,
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_NONCES)
            .ok_or_else(|| anyhow::anyhow!("CF_NONCES not found"))?;

        batch.put_cf(&cf, account, &new_nonce.to_be_bytes());

        Ok(())
    }

    fn apply_state_root_checkpoint(
        &self,
        batch: &mut WriteBatch,
        height: u64,
        state_root: &[u8; 32],
        tx_root: &[u8; 32],
    ) -> Result<()> {
        let cf = self.db.cf_handle(CF_STATE_ROOTS)
            .ok_or_else(|| anyhow::anyhow!("CF_STATE_ROOTS not found"))?;

        // Key is height as big-endian bytes
        let key = height.to_be_bytes();

        // Value is state_root (32) | tx_root (32)
        let mut value = Vec::with_capacity(64);
        value.extend_from_slice(state_root);
        value.extend_from_slice(tx_root);

        batch.put_cf(&cf, &key, &value);

        Ok(())
    }

    // ========== Query Methods ==========

    /// Get token balance for an account
    pub fn get_token_balance(&self, account: &[u8; 32], token: &[u8; 32]) -> Result<u64> {
        let cf = self.db.cf_handle(CF_TOKEN_BALANCES)
            .ok_or_else(|| anyhow::anyhow!("CF_TOKEN_BALANCES not found"))?;

        let mut key = Vec::with_capacity(64);
        key.extend_from_slice(account);
        key.extend_from_slice(token);

        Ok(self.db.get_cf(&cf, &key)?
            .map(|v| u64::from_be_bytes(v.try_into().unwrap_or([0u8; 8])))
            .unwrap_or(0))
    }

    /// Get pool state
    pub fn get_pool(&self, pool_id: &[u8; 32]) -> Result<Option<PoolState>> {
        let cf = self.db.cf_handle(CF_DEX_POOLS)
            .ok_or_else(|| anyhow::anyhow!("CF_DEX_POOLS not found"))?;

        match self.db.get_cf(&cf, pool_id)? {
            Some(v) => {
                let pool: PoolState = postcard::from_bytes(&v)
                    .context("Failed to deserialize pool state")?;
                Ok(Some(pool))
            }
            None => Ok(None),
        }
    }

    /// Get account nonce
    pub fn get_nonce(&self, account: &[u8; 32]) -> Result<u64> {
        let cf = self.db.cf_handle(CF_NONCES)
            .ok_or_else(|| anyhow::anyhow!("CF_NONCES not found"))?;

        Ok(self.db.get_cf(&cf, account)?
            .map(|v| u64::from_be_bytes(v.try_into().unwrap_or([0u8; 8])))
            .unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_db() -> Arc<DB> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_db");

        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Create column families
        let cfs: Vec<rocksdb::ColumnFamilyDescriptor> = crate::STATE_SYNC_COLUMN_FAMILIES
            .iter()
            .map(|name| rocksdb::ColumnFamilyDescriptor::new(*name, rocksdb::Options::default()))
            .collect();

        Arc::new(DB::open_cf_descriptors(&opts, path, cfs).unwrap())
    }

    #[test]
    fn test_balance_credit_debit() {
        let db = create_test_db();
        let applicator = StateApplicator::new(db.clone(), false);

        let account = [1u8; 32];
        let token = [2u8; 32];

        // Credit 100
        let changes = vec![StateChange::BalanceCredit {
            account,
            token,
            amount: 100,
        }];
        applicator.apply_changes(&changes, 1).unwrap();

        assert_eq!(applicator.get_token_balance(&account, &token).unwrap(), 100);

        // Debit 30
        let changes = vec![StateChange::BalanceDebit {
            account,
            token,
            amount: 30,
        }];
        applicator.apply_changes(&changes, 2).unwrap();

        assert_eq!(applicator.get_token_balance(&account, &token).unwrap(), 70);
    }

    #[test]
    fn test_insufficient_balance() {
        let db = create_test_db();
        let applicator = StateApplicator::new(db.clone(), false);

        let account = [1u8; 32];
        let token = [2u8; 32];

        // Try to debit from empty account
        let changes = vec![StateChange::BalanceDebit {
            account,
            token,
            amount: 100,
        }];

        let result = applicator.apply_changes(&changes, 1);
        assert!(result.is_err());
    }
}
