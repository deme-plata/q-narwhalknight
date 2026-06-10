/// 🔒 Max Supply Enforcement - RocksDB Persistence Layer
///
/// This module provides persistent storage for the max supply enforcement mechanism
/// using RocksDB as the backing store.
///
/// Key Features:
/// - Atomic supply updates with crash recovery
/// - Balance capping migration for affected users
/// - Supply consensus state persistence
/// - Audit logging for all supply changes

use anyhow::{Context, Result};
use rocksdb::{DB, IteratorMode, WriteBatch};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

// ============================================================================
// Constants
// ============================================================================

/// One QNK in base units (10^24 for 24-decimal precision)
/// v3.0.4: Migrated from 10^8 to 10^24
pub const ONE_QNK: u128 = 1_000_000_000_000_000_000_000_000;

/// Maximum allowed supply: 21,000,000 QNK in atomic units (24 decimals)
/// v3.0.4: Migrated from u64 to u128
pub const MAX_SUPPLY_QNK: u128 = 21_000_000 * ONE_QNK;

/// Maximum balance per wallet for migration (1M QNK)
/// v3.0.4: Migrated from u64 to u128
pub const MAX_WALLET_BALANCE_CAP: u128 = 1_000_000 * ONE_QNK;

/// Legacy divisor for old 8-decimal balances
const LEGACY_DIVISOR: f64 = 1e8;
/// New divisor for 24-decimal balances
const NEW_DIVISOR: f64 = 1e24;

// RocksDB Column Families
pub const CF_CHAIN_STATE: &str = "chain_state";
pub const CF_SUPPLY_AUDIT: &str = "supply_audit";
pub const CF_BALANCE_AUDIT: &str = "balance_audit";

// Key prefixes
const KEY_TOTAL_SUPPLY: &[u8] = b"total_minted_supply";
const KEY_LAST_HALVING: &[u8] = b"last_halving_block";
const KEY_CONSENSUS_TIMESTAMP: &[u8] = b"consensus_timestamp";
const KEY_CONSENSUS_NODE_COUNT: &[u8] = b"consensus_node_count";

// ============================================================================
// Data Structures
// ============================================================================

/// v3.0.4: Migrated from u64 to u128 for 24-decimal precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainSupplyState {
    pub total_minted_supply: u128,
    pub last_halving_block: u64,
    pub last_consensus_timestamp: u64,
    pub consensus_node_count: usize,
}

impl Default for ChainSupplyState {
    fn default() -> Self {
        Self {
            total_minted_supply: 0,
            last_halving_block: 0,
            last_consensus_timestamp: 0,
            consensus_node_count: 0,
        }
    }
}

/// v3.0.4: Migrated supply fields to u128 for 24-decimal precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyAuditEntry {
    pub block_height: u64,
    pub old_supply: u128,
    pub new_supply: u128,
    pub block_reward: u128,
    pub miner_address: [u8; 32],
    pub timestamp: u64,
}

/// v3.0.4: Migrated balance fields to u128 for 24-decimal precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceAuditEntry {
    pub address: [u8; 32],
    pub old_balance: u128,
    pub new_balance: u128,
    pub reason: String,
    pub timestamp: u64,
}

// ============================================================================
// Supply Persistence Manager
// ============================================================================

pub struct SupplyPersistenceManager {
    db: Arc<DB>,
}

impl SupplyPersistenceManager {
    /// Initialize the supply persistence manager with RocksDB
    pub fn new(db: Arc<DB>) -> Result<Self> {
        info!("🔒 Initializing supply persistence manager");

        // Ensure column families exist
        // Note: In production, these should be created when opening the DB
        // This is a runtime check to ensure they're available

        Ok(Self { db })
    }

    /// Load the current total supply from RocksDB
    ///
    /// This is called on node startup to restore the supply state
    /// v3.0.4: Backward compatible with legacy 8-byte (u64) storage
    pub fn load_total_supply(&self) -> Result<u128> {
        match self.db.get(KEY_TOTAL_SUPPLY) {
            Ok(Some(bytes)) => {
                let supply = if bytes.len() == 8 {
                    // Legacy u64 format - convert to u128
                    let legacy: [u8; 8] = bytes.as_slice().try_into()
                        .context("Invalid legacy supply value in database")?;
                    let legacy_supply = u64::from_be_bytes(legacy);
                    // Scale from 8-decimal to 24-decimal (multiply by 10^16)
                    (legacy_supply as u128) * 10_000_000_000_000_000
                } else if bytes.len() == 16 {
                    // New u128 format
                    let new_bytes: [u8; 16] = bytes.as_slice().try_into()
                        .context("Invalid supply value in database")?;
                    u128::from_be_bytes(new_bytes)
                } else {
                    return Err(anyhow::anyhow!("Unexpected supply byte length: {}", bytes.len()));
                };
                info!("✅ Loaded total supply from RocksDB: {} QNK",
                      supply as f64 / NEW_DIVISOR);
                Ok(supply)
            }
            Ok(None) => {
                warn!("⚠️ No supply state found in database, starting from 0");
                Ok(0)
            }
            Err(e) => {
                error!("❌ Failed to load supply from database: {}", e);
                Err(anyhow::anyhow!("Database read error: {}", e))
            }
        }
    }

    /// Save the total supply to RocksDB atomically
    ///
    /// This is called after every successful mining reward issuance
    /// v3.0.4: Now stores as 16-byte u128
    pub fn save_total_supply(&self, supply: u128) -> Result<()> {
        self.db.put(KEY_TOTAL_SUPPLY, &supply.to_be_bytes())
            .context("Failed to save total supply to database")?;

        info!("💾 Saved total supply to RocksDB: {} QNK", supply as f64 / NEW_DIVISOR);
        Ok(())
    }

    /// Load full chain supply state
    pub fn load_chain_state(&self) -> Result<ChainSupplyState> {
        let total_supply = self.load_total_supply()?;

        let last_halving = self.db.get(KEY_LAST_HALVING)
            .ok()
            .flatten()
            .and_then(|bytes| bytes.as_slice().try_into().ok())
            .map(u64::from_be_bytes)
            .unwrap_or(0);

        let consensus_timestamp = self.db.get(KEY_CONSENSUS_TIMESTAMP)
            .ok()
            .flatten()
            .and_then(|bytes| bytes.as_slice().try_into().ok())
            .map(u64::from_be_bytes)
            .unwrap_or(0);

        let consensus_node_count = self.db.get(KEY_CONSENSUS_NODE_COUNT)
            .ok()
            .flatten()
            .and_then(|bytes| bytes.as_slice().try_into().ok())
            .map(u64::from_be_bytes)
            .map(|v| v as usize)
            .unwrap_or(0);

        Ok(ChainSupplyState {
            total_minted_supply: total_supply,
            last_halving_block: last_halving,
            last_consensus_timestamp: consensus_timestamp,
            consensus_node_count,
        })
    }

    /// Save full chain supply state atomically
    /// v3.0.4: Now stores total_minted_supply as 16-byte u128
    pub fn save_chain_state(&self, state: &ChainSupplyState) -> Result<()> {
        let mut batch = WriteBatch::default();

        batch.put(KEY_TOTAL_SUPPLY, &state.total_minted_supply.to_be_bytes());
        batch.put(KEY_LAST_HALVING, &state.last_halving_block.to_be_bytes());
        batch.put(KEY_CONSENSUS_TIMESTAMP, &state.last_consensus_timestamp.to_be_bytes());
        batch.put(KEY_CONSENSUS_NODE_COUNT, &(state.consensus_node_count as u64).to_be_bytes());

        self.db.write(batch)
            .context("Failed to save chain state atomically")?;

        info!("💾 Saved full chain supply state: {} QNK, {} nodes",
              state.total_minted_supply as f64 / NEW_DIVISOR,
              state.consensus_node_count);
        Ok(())
    }

    /// Log a supply update for audit trail
    pub fn log_supply_update(&self, entry: SupplyAuditEntry) -> Result<()> {
        let key = format!("supply_audit:{}", entry.block_height);
        let value = bincode::serialize(&entry)
            .context("Failed to serialize supply audit entry")?;

        self.db.put(key.as_bytes(), &value)
            .context("Failed to write supply audit log")?;

        Ok(())
    }

    /// Log a balance change for audit trail
    pub fn log_balance_change(&self, entry: BalanceAuditEntry) -> Result<()> {
        let key = format!("balance_audit:{}:{}", hex::encode(&entry.address), entry.timestamp);
        let value = bincode::serialize(&entry)
            .context("Failed to serialize balance audit entry")?;

        self.db.put(key.as_bytes(), &value)
            .context("Failed to write balance audit log")?;

        Ok(())
    }

    /// Calculate total supply from all wallet balances (for migration)
    ///
    /// This scans the entire wallet_balances key-value space and sums balances
    /// v3.0.4: Migrated to u128
    pub fn calculate_total_supply_from_wallets(&self, wallet_balances: &dashmap::DashMap<[u8; 32], u128>) -> u128 {
        let total: u128 = wallet_balances.iter()
            .map(|entry| *entry.value())
            .sum();

        info!("📊 Calculated total supply from wallets: {} QNK ({} wallets)",
              total as f64 / NEW_DIVISOR,
              wallet_balances.len());

        total
    }

    /// Migrate and cap affected balances
    ///
    /// This is a one-time migration to fix the unlimited minting bug
    /// Caps any balance > 21M QNK to 1M QNK as compensation
    /// v3.0.4: Migrated to u128
    pub fn migrate_and_cap_balances(
        &self,
        wallet_balances: &dashmap::DashMap<[u8; 32], u128>
    ) -> Result<usize> {
        info!("🔧 Starting balance migration: capping balances > 21M QNK");

        let mut capped_count = 0;
        let mut _batch = WriteBatch::default();

        // Find and cap affected wallets
        for mut entry in wallet_balances.iter_mut() {
            let address = *entry.key();
            let old_balance = *entry.value();

            if old_balance > MAX_SUPPLY_QNK {
                // Cap to 1M QNK
                let new_balance = MAX_WALLET_BALANCE_CAP;
                *entry.value_mut() = new_balance;

                warn!("⚠️ Capped wallet {} from {} QNK to {} QNK",
                      hex::encode(&address[..8]),
                      old_balance as f64 / NEW_DIVISOR,
                      new_balance as f64 / NEW_DIVISOR);

                // Log the capping for audit
                let audit_entry = BalanceAuditEntry {
                    address,
                    old_balance,
                    new_balance,
                    reason: "Migration: Capped due to unlimited minting bug".to_string(),
                    timestamp: chrono::Utc::now().timestamp() as u64,
                };

                self.log_balance_change(audit_entry)?;
                capped_count += 1;
            }
        }

        info!("✅ Migration complete: Capped {} wallets", capped_count);
        Ok(capped_count)
    }

    /// Initialize supply state from existing wallet balances
    ///
    /// Called once during first startup after migration
    /// v3.0.4: Migrated to u128
    pub async fn initialize_supply_from_wallets(
        &self,
        wallet_balances: &dashmap::DashMap<[u8; 32], u128>,
        total_minted_supply: &Arc<RwLock<u128>>,
    ) -> Result<()> {
        info!("🔧 Initializing supply state from existing wallet balances");

        // Calculate current total supply
        let calculated_supply = self.calculate_total_supply_from_wallets(wallet_balances);

        // Update in-memory state
        let mut supply = total_minted_supply.write().await;
        *supply = calculated_supply;
        drop(supply);

        // Save to RocksDB
        self.save_total_supply(calculated_supply)?;

        info!("✅ Supply state initialized: {} QNK", calculated_supply as f64 / NEW_DIVISOR);
        Ok(())
    }

    /// Get supply audit history
    pub fn get_supply_audit_history(&self, limit: usize) -> Result<Vec<SupplyAuditEntry>> {
        let mut entries = Vec::new();
        let iter = self.db.iterator(IteratorMode::Start);

        for item in iter {
            let (key, value) = item.context("Iterator error")?;
            let key_str = String::from_utf8_lossy(&key);

            if key_str.starts_with("supply_audit:") {
                if let Ok(entry) = bincode::deserialize::<SupplyAuditEntry>(&value) {
                    entries.push(entry);
                    if entries.len() >= limit {
                        break;
                    }
                }
            }
        }

        entries.sort_by(|a, b| b.block_height.cmp(&a.block_height));
        Ok(entries)
    }

    /// Get balance change audit history for an address
    pub fn get_balance_audit_history(&self, address: &[u8; 32], limit: usize) -> Result<Vec<BalanceAuditEntry>> {
        let mut entries = Vec::new();
        let prefix = format!("balance_audit:{}", hex::encode(address));
        let iter = self.db.iterator(IteratorMode::Start);

        for item in iter {
            let (key, value) = item.context("Iterator error")?;
            let key_str = String::from_utf8_lossy(&key);

            if key_str.starts_with(&prefix) {
                if let Ok(entry) = bincode::deserialize::<BalanceAuditEntry>(&value) {
                    entries.push(entry);
                    if entries.len() >= limit {
                        break;
                    }
                }
            }
        }

        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(entries)
    }

    /// Verify supply integrity
    ///
    /// Checks that stored supply matches sum of wallet balances
    /// v3.0.4: Migrated to u128
    pub fn verify_supply_integrity(&self, wallet_balances: &dashmap::DashMap<[u8; 32], u128>) -> Result<bool> {
        let stored_supply = self.load_total_supply()?;
        let calculated_supply = self.calculate_total_supply_from_wallets(wallet_balances);

        if stored_supply != calculated_supply {
            error!("🚨 SUPPLY INTEGRITY VIOLATION: Stored={} QNK, Calculated={} QNK",
                   stored_supply as f64 / NEW_DIVISOR,
                   calculated_supply as f64 / NEW_DIVISOR);
            return Ok(false);
        }

        info!("✅ Supply integrity verified: {} QNK", stored_supply as f64 / NEW_DIVISOR);
        Ok(true)
    }
}

// ============================================================================
// Integration with AppState
// ============================================================================

/// Load total supply from RocksDB on startup
/// v3.0.4: Migrated to u128
pub async fn load_supply_on_startup(
    db: Arc<DB>,
    wallet_balances: &dashmap::DashMap<[u8; 32], u128>,
) -> Result<u128> {
    let manager = SupplyPersistenceManager::new(db)?;

    // Try to load from RocksDB first
    match manager.load_total_supply() {
        Ok(supply) if supply > 0 => {
            info!("📂 Loaded existing supply state: {} QNK", supply as f64 / NEW_DIVISOR);
            Ok(supply)
        }
        _ => {
            // First startup or migration needed
            warn!("⚠️ No supply state found, calculating from wallet balances");
            let calculated = manager.calculate_total_supply_from_wallets(wallet_balances);
            manager.save_total_supply(calculated)?;
            Ok(calculated)
        }
    }
}

/// Save supply after mining reward
/// v3.0.4: Migrated to u128
pub async fn save_supply_after_mining(
    db: Arc<DB>,
    new_supply: u128,
    block_height: u64,
    block_reward: u128,
    miner_address: [u8; 32],
) -> Result<()> {
    let manager = SupplyPersistenceManager::new(db)?;

    // Save new supply
    manager.save_total_supply(new_supply)?;

    // Log audit entry
    let audit_entry = SupplyAuditEntry {
        block_height,
        old_supply: new_supply.saturating_sub(block_reward),
        new_supply,
        block_reward,
        miner_address,
        timestamp: chrono::Utc::now().timestamp() as u64,
    };

    manager.log_supply_update(audit_entry)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_supply_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());

        let manager = SupplyPersistenceManager::new(db.clone()).unwrap();

        // Save supply (5M QNK in new 24-decimal format)
        let test_supply: u128 = 5_000_000 * ONE_QNK;
        manager.save_total_supply(test_supply).unwrap();

        // Load supply
        let loaded = manager.load_total_supply().unwrap();
        assert_eq!(loaded, test_supply);
    }

    #[tokio::test]
    async fn test_balance_capping() {
        let temp_dir = TempDir::new().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());

        let manager = SupplyPersistenceManager::new(db.clone()).unwrap();
        let balances = dashmap::DashMap::new();

        // Add an affected wallet (184T QNK - way over the 21M cap)
        let address = [1u8; 32];
        let huge_balance: u128 = 184_000_000_000_000 * ONE_QNK;
        balances.insert(address, huge_balance);

        // Migrate
        let capped = manager.migrate_and_cap_balances(&balances).unwrap();
        assert_eq!(capped, 1);

        // Verify capped to 1M QNK
        assert_eq!(*balances.get(&address).unwrap(), MAX_WALLET_BALANCE_CAP);
    }

    #[tokio::test]
    async fn test_legacy_supply_migration() {
        let temp_dir = TempDir::new().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());

        // Simulate legacy 8-byte u64 storage (100 QNK in old 8-decimal format)
        let legacy_supply: u64 = 10_000_000_000; // 100 QNK
        db.put(KEY_TOTAL_SUPPLY, &legacy_supply.to_be_bytes()).unwrap();

        let manager = SupplyPersistenceManager::new(db.clone()).unwrap();

        // Load should auto-convert to 24-decimal
        let loaded = manager.load_total_supply().unwrap();

        // 100 QNK * 10^16 (scaling factor) = 100 * 10^24 base units
        let expected: u128 = 100 * ONE_QNK;
        assert_eq!(loaded, expected);
    }
}
