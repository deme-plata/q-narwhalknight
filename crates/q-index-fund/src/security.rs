//! Security module for QNK-INDEX
//!
//! Provides post-quantum signature verification, input validation,
//! and reentrancy protection.

use crate::types::IndexError;
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{DetachedSignature as DetachedSignatureTrait, PublicKey};
use sha3::{Digest, Sha3_256};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, warn};

/// Reentrancy guard
pub struct ReentrancyGuard {
    locked: AtomicBool,
}

impl ReentrancyGuard {
    pub fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
        }
    }

    /// Try to acquire the lock
    pub fn try_lock(&self) -> Option<ReentrancyLock<'_>> {
        if self.locked.compare_exchange(
            false, true,
            Ordering::SeqCst, Ordering::SeqCst
        ).is_ok() {
            Some(ReentrancyLock { guard: self })
        } else {
            warn!("Reentrancy detected!");
            None
        }
    }
}

impl Default for ReentrancyGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII lock that releases on drop
pub struct ReentrancyLock<'a> {
    guard: &'a ReentrancyGuard,
}

impl Drop for ReentrancyLock<'_> {
    fn drop(&mut self) {
        self.guard.locked.store(false, Ordering::SeqCst);
    }
}

/// Verify a Dilithium5 signature
pub fn verify_dilithium_signature(
    public_key: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<bool, IndexError> {
    // Validate key length
    if public_key.len() != dilithium5::public_key_bytes() {
        return Err(IndexError::InvalidInput(
            format!("Invalid public key length: {} (expected {})",
                public_key.len(), dilithium5::public_key_bytes())
        ));
    }

    // Validate signature length
    if signature.len() != dilithium5::signature_bytes() {
        return Err(IndexError::InvalidInput(
            format!("Invalid signature length: {} (expected {})",
                signature.len(), dilithium5::signature_bytes())
        ));
    }

    // Parse public key
    let pk = match dilithium5::PublicKey::from_bytes(public_key) {
        Ok(pk) => pk,
        Err(_) => return Ok(false),
    };

    // Parse signature
    let sig = match dilithium5::DetachedSignature::from_bytes(signature) {
        Ok(s) => s,
        Err(_) => return Err(IndexError::InvalidInput("Invalid signature format".into())),
    };

    // Verify signature using the correct API
    match dilithium5::verify_detached_signature(&sig, message, &pk) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Hash a message using SHA3-256
pub fn hash_sha3(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

/// Hash a message using BLAKE3
pub fn hash_blake3(data: &[u8]) -> [u8; 32] {
    let hash = blake3::hash(data);
    *hash.as_bytes()
}

/// Validate token address format
pub fn validate_address(address: &[u8; 32]) -> Result<(), IndexError> {
    // Check not all zeros
    if address.iter().all(|&b| b == 0) {
        return Err(IndexError::InvalidInput("Address cannot be all zeros".into()));
    }

    Ok(())
}

/// Validate index name
pub fn validate_name(name: &str) -> Result<(), IndexError> {
    if name.is_empty() {
        return Err(IndexError::InvalidInput("Name cannot be empty".into()));
    }

    if name.len() > 64 {
        return Err(IndexError::InvalidInput("Name too long (max 64 chars)".into()));
    }

    // Check for allowed characters (alphanumeric, space, dash, underscore)
    if !name.chars().all(|c| c.is_alphanumeric() || c == ' ' || c == '-' || c == '_') {
        return Err(IndexError::InvalidInput(
            "Name contains invalid characters".into()
        ));
    }

    Ok(())
}

/// Validate symbol
pub fn validate_symbol(symbol: &str) -> Result<(), IndexError> {
    if symbol.is_empty() {
        return Err(IndexError::InvalidInput("Symbol cannot be empty".into()));
    }

    if symbol.len() > 10 {
        return Err(IndexError::InvalidInput("Symbol too long (max 10 chars)".into()));
    }

    // Symbol must be uppercase alphanumeric
    if !symbol.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        return Err(IndexError::InvalidInput(
            "Symbol must be uppercase alphanumeric".into()
        ));
    }

    Ok(())
}

/// Validate weights array
pub fn validate_weights(weights: &[u16]) -> Result<(), IndexError> {
    if weights.is_empty() {
        return Err(IndexError::InvalidInput("Weights array cannot be empty".into()));
    }

    if weights.len() > 50 {
        return Err(IndexError::InvalidInput("Too many weights (max 50)".into()));
    }

    let total: u32 = weights.iter().map(|w| *w as u32).sum();
    if total != 10000 {
        return Err(IndexError::WeightsNot100Percent);
    }

    // Check no single weight > 50%
    for w in weights {
        if *w > 5000 {
            return Err(IndexError::InvalidWeight);
        }
    }

    Ok(())
}

/// Validate amount is within safe bounds
pub fn validate_amount(amount: u64, min: u64, max: u64) -> Result<(), IndexError> {
    if amount < min {
        return Err(IndexError::InsufficientBalance);
    }

    if amount > max {
        return Err(IndexError::InvalidInput(
            format!("Amount {} exceeds maximum {}", amount, max)
        ));
    }

    Ok(())
}

/// Check for duplicate addresses in a list
pub fn check_no_duplicates(addresses: &[[u8; 32]]) -> Result<(), IndexError> {
    let mut seen = HashSet::new();

    for addr in addresses {
        if !seen.insert(addr) {
            return Err(IndexError::InvalidInput("Duplicate address found".into()));
        }
    }

    Ok(())
}

/// Rate limiter for operations
pub struct RateLimiter {
    /// Operations per address: address -> (last_block, count)
    operations: dashmap::DashMap<[u8; 32], (u64, u32)>,

    /// Window size in blocks
    window_blocks: u64,

    /// Max operations per window
    max_ops_per_window: u32,
}

impl RateLimiter {
    pub fn new(window_blocks: u64, max_ops: u32) -> Self {
        Self {
            operations: dashmap::DashMap::new(),
            window_blocks,
            max_ops_per_window: max_ops,
        }
    }

    /// Check and record an operation
    pub fn check(&self, address: [u8; 32], current_block: u64) -> Result<(), IndexError> {
        let mut entry = self.operations.entry(address).or_insert((current_block, 0));

        let (last_block, count) = *entry;

        // Reset if outside window
        if current_block > last_block + self.window_blocks {
            *entry = (current_block, 1);
            return Ok(());
        }

        // Check limit
        if count >= self.max_ops_per_window {
            return Err(IndexError::RateLimited);
        }

        // Increment
        entry.1 += 1;

        Ok(())
    }

    /// Clear old entries
    pub fn cleanup(&self, current_block: u64) {
        let cutoff = current_block.saturating_sub(self.window_blocks * 2);
        self.operations.retain(|_, (block, _)| *block > cutoff);
    }
}

/// Audit log entry
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub block: u64,
    pub operation: String,
    pub actor: [u8; 32],
    pub target_index: Option<[u8; 32]>,
    pub details: String,
    pub success: bool,
}

/// Audit logger
pub struct AuditLog {
    entries: std::sync::RwLock<Vec<AuditEntry>>,
    max_entries: usize,
}

impl AuditLog {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: std::sync::RwLock::new(Vec::new()),
            max_entries,
        }
    }

    /// Log an operation
    pub fn log(&self, entry: AuditEntry) {
        let mut entries = self.entries.write().unwrap();

        debug!(
            "AUDIT: {} by {} on {:?}: {} ({})",
            entry.operation,
            hex::encode(&entry.actor[..8]),
            entry.target_index.map(|i| hex::encode(&i[..8])),
            entry.details,
            if entry.success { "SUCCESS" } else { "FAILED" }
        );

        entries.push(entry);

        // Trim if too many
        if entries.len() > self.max_entries {
            entries.remove(0);
        }
    }

    /// Get recent entries
    pub fn get_recent(&self, count: usize) -> Vec<AuditEntry> {
        let entries = self.entries.read().unwrap();
        entries.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }

    /// Get entries for a specific actor
    pub fn get_by_actor(&self, actor: [u8; 32]) -> Vec<AuditEntry> {
        let entries = self.entries.read().unwrap();
        entries.iter()
            .filter(|e| e.actor == actor)
            .cloned()
            .collect()
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new(10000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_name() {
        assert!(validate_name("QNK Top 10").is_ok());
        assert!(validate_name("Test-Index_123").is_ok());
        assert!(validate_name("").is_err());
        assert!(validate_name("a".repeat(65).as_str()).is_err());
        assert!(validate_name("Invalid@Name").is_err());
    }

    #[test]
    fn test_validate_symbol() {
        assert!(validate_symbol("QNK10").is_ok());
        assert!(validate_symbol("TEST").is_ok());
        assert!(validate_symbol("").is_err());
        assert!(validate_symbol("toolongsymbol").is_err());
        assert!(validate_symbol("lowercase").is_err());
    }

    #[test]
    fn test_validate_weights() {
        assert!(validate_weights(&[5000, 3000, 2000]).is_ok());
        assert!(validate_weights(&[3333, 3333, 3334]).is_ok());
        assert!(validate_weights(&[]).is_err());
        assert!(validate_weights(&[5000, 5000]).is_err()); // 100% but single > 50%
        assert!(validate_weights(&[5000, 3000, 1000]).is_err()); // Only 90%
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(100, 5);
        let addr = [1u8; 32];

        // First 5 should succeed
        for i in 0..5 {
            assert!(limiter.check(addr, 1000 + i).is_ok());
        }

        // 6th should fail
        assert!(limiter.check(addr, 1004).is_err());

        // After window, should work again
        assert!(limiter.check(addr, 1200).is_ok());
    }

    #[test]
    fn test_reentrancy_guard() {
        let guard = ReentrancyGuard::new();

        // First lock should succeed
        let lock1 = guard.try_lock();
        assert!(lock1.is_some());

        // Second lock should fail
        let lock2 = guard.try_lock();
        assert!(lock2.is_none());

        // After dropping, should work again
        drop(lock1);
        let lock3 = guard.try_lock();
        assert!(lock3.is_some());
    }

    #[test]
    fn test_hash_functions() {
        let data = b"test message";

        let sha3 = hash_sha3(data);
        assert_eq!(sha3.len(), 32);

        let blake = hash_blake3(data);
        assert_eq!(blake.len(), 32);

        // Different hashes should produce different outputs
        assert_ne!(sha3, blake);
    }
}
