//! AEGIS-QL Access Control System
//!
//! Provides cryptographic access control for centralized operations
//! that must be restricted to specific wallet addresses (e.g., founder wallet).
//!
//! Security model:
//! - Wallet addresses are public keys
//! - Operations require valid AEGIS-QL signatures
//! - Even with open source code, only holders of private keys can sign
//! - Post-quantum secure against quantum computer attacks

use super::{AegisError, AegisQL, PublicKey, SecretKey, Signature};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Access control level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    /// Founder - full control over all operations
    Founder,
    /// Admin - can perform administrative operations
    Admin,
    /// Operator - can perform routine operations
    Operator,
    /// User - standard user with no privileged access
    User,
}

/// Access control list entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AclEntry {
    /// Wallet address (32-byte public key hash)
    pub wallet_address: [u8; 32],
    /// AEGIS-QL public key for signature verification
    pub public_key: PublicKey,
    /// Access level
    pub access_level: AccessLevel,
    /// Optional description
    pub description: String,
}

/// AEGIS-QL access control system
pub struct AegisAccessControl {
    /// Access control list
    acl: Vec<AclEntry>,
    /// Founder wallet (immutable after initialization)
    founder_wallet: [u8; 32],
    /// AEGIS-QL crypto system
    aegis: AegisQL,
}

impl AegisAccessControl {
    /// Create a new access control system with founder wallet
    pub fn new(founder_wallet: [u8; 32], founder_public_key: PublicKey) -> Self {
        let mut acl = Vec::new();

        // Add founder to ACL
        acl.push(AclEntry {
            wallet_address: founder_wallet,
            public_key: founder_public_key,
            access_level: AccessLevel::Founder,
            description: "Founder wallet - full system control".to_string(),
        });

        Self {
            acl,
            founder_wallet,
            aegis: AegisQL::new(),
        }
    }

    /// Verify a wallet has the required access level for an operation
    pub fn verify_access(
        &self,
        wallet_address: &[u8; 32],
        signature: &Signature,
        message: &[u8],
        required_level: AccessLevel,
    ) -> Result<bool, AegisError> {
        // Find wallet in ACL
        let entry = self.acl.iter()
            .find(|e| &e.wallet_address == wallet_address)
            .ok_or(AegisError::Unauthorized)?;

        // Check if wallet has sufficient access level
        if !Self::has_access(entry.access_level, required_level) {
            return Err(AegisError::Unauthorized);
        }

        // Verify AEGIS-QL signature
        self.aegis.verify(message, signature, &entry.public_key)
    }

    /// Check if one access level includes another
    fn has_access(actual: AccessLevel, required: AccessLevel) -> bool {
        match required {
            AccessLevel::User => true, // All levels include User
            AccessLevel::Operator => matches!(actual, AccessLevel::Operator | AccessLevel::Admin | AccessLevel::Founder),
            AccessLevel::Admin => matches!(actual, AccessLevel::Admin | AccessLevel::Founder),
            AccessLevel::Founder => actual == AccessLevel::Founder,
        }
    }

    /// Add a new wallet to the ACL (founder-only operation)
    pub fn add_wallet(
        &mut self,
        requester_wallet: &[u8; 32],
        requester_signature: &Signature,
        new_wallet: [u8; 32],
        new_public_key: PublicKey,
        access_level: AccessLevel,
        description: String,
    ) -> Result<(), AegisError> {
        // Verify requester is founder
        let message = format!("ADD_WALLET:{:?}:{:?}", new_wallet, access_level);
        self.verify_access(
            requester_wallet,
            requester_signature,
            message.as_bytes(),
            AccessLevel::Founder,
        )?;

        // Add new wallet
        self.acl.push(AclEntry {
            wallet_address: new_wallet,
            public_key: new_public_key,
            access_level,
            description,
        });

        Ok(())
    }

    /// Remove a wallet from the ACL (founder-only operation)
    pub fn remove_wallet(
        &mut self,
        requester_wallet: &[u8; 32],
        requester_signature: &Signature,
        target_wallet: &[u8; 32],
    ) -> Result<(), AegisError> {
        // Cannot remove founder wallet
        if target_wallet == &self.founder_wallet {
            return Err(AegisError::Unauthorized);
        }

        // Verify requester is founder
        let message = format!("REMOVE_WALLET:{:?}", target_wallet);
        self.verify_access(
            requester_wallet,
            requester_signature,
            message.as_bytes(),
            AccessLevel::Founder,
        )?;

        // Remove wallet
        self.acl.retain(|e| &e.wallet_address != target_wallet);

        Ok(())
    }

    /// Get all wallets with a specific access level
    pub fn get_wallets_by_level(&self, level: AccessLevel) -> Vec<&AclEntry> {
        self.acl.iter()
            .filter(|e| e.access_level == level)
            .collect()
    }

    /// Check if a wallet is the founder
    pub fn is_founder(&self, wallet: &[u8; 32]) -> bool {
        wallet == &self.founder_wallet
    }

    /// Get founder wallet address
    pub fn get_founder_wallet(&self) -> [u8; 32] {
        self.founder_wallet
    }

    /// Get ACL size
    pub fn acl_size(&self) -> usize {
        self.acl.len()
    }
}

/// Helper function to create a message for bank operations
pub fn create_bank_operation_message(
    operation: &str,
    wallet: &[u8; 32],
    amount: u128,
    timestamp: i64,
) -> Vec<u8> {
    format!(
        "BANK_OP:{}:{}:{}:{}",
        operation,
        hex::encode(wallet),
        amount,
        timestamp
    ).into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_control_founder() {
        let mut aegis = AegisQL::new();
        let (founder_pk, founder_sk) = aegis.generate_keypair().unwrap();
        let founder_wallet = [1u8; 32];

        let acl = AegisAccessControl::new(founder_wallet, founder_pk.clone());

        assert!(acl.is_founder(&founder_wallet));
        assert_eq!(acl.acl_size(), 1);
    }

    #[test]
    fn test_access_levels() {
        assert!(AegisAccessControl::has_access(AccessLevel::Founder, AccessLevel::Admin));
        assert!(AegisAccessControl::has_access(AccessLevel::Founder, AccessLevel::Operator));
        assert!(AegisAccessControl::has_access(AccessLevel::Admin, AccessLevel::Operator));
        assert!(!AegisAccessControl::has_access(AccessLevel::Operator, AccessLevel::Admin));
        assert!(!AegisAccessControl::has_access(AccessLevel::User, AccessLevel::Founder));
    }
}
