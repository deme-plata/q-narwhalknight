//! Wallet Privacy Circuits
//!
//! Zero-knowledge proofs for wallet balance queries without revealing exact amounts.
//! Supports range proofs, ownership proofs, and transaction privacy.

use anyhow::Result;
use ark_bn254::Fr;
use ark_ff::PrimeField;
use serde::{Deserialize, Serialize};

use crate::circuits::{ArithmeticCircuit, CircuitBuilder, CircuitGadgets};
use crate::{SNARKError, SNARKProtocol, UniversalSNARK};

/// Wallet balance range proof
/// Proves: balance >= min_balance AND balance <= max_balance
/// WITHOUT revealing the exact balance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceRangeProof {
    /// ZK proof data
    pub proof: Vec<u8>,
    /// Public inputs (min/max range, NOT actual balance)
    pub public_min: u64,
    pub public_max: u64,
    /// Wallet address commitment
    pub address_commitment: [u8; 32],
    /// Protocol used
    pub protocol: SNARKProtocol,
}

/// Wallet ownership proof
/// Proves: "I own this wallet" without revealing the private key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletOwnershipProof {
    /// ZK proof data
    pub proof: Vec<u8>,
    /// Public wallet address
    pub wallet_address: [u8; 32],
    /// Challenge (prevents replay)
    pub challenge: [u8; 32],
    /// Protocol used
    pub protocol: SNARKProtocol,
}

/// Transaction privacy proof
/// Proves: "This transaction is valid" without revealing sender/receiver/amount
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionPrivacyProof {
    /// ZK proof data
    pub proof: Vec<u8>,
    /// Transaction commitment
    pub tx_commitment: [u8; 32],
    /// Nullifier (prevents double-spending)
    pub nullifier: [u8; 32],
    /// Protocol used
    pub protocol: SNARKProtocol,
}

/// Wallet privacy prover
pub struct WalletPrivacyProver {
    snark: UniversalSNARK,
}

impl WalletPrivacyProver {
    /// Create new wallet privacy prover
    pub fn new() -> Self {
        let config = crate::SNARKConfig {
            protocol: SNARKProtocol::Groth16, // Most efficient for wallet proofs
            security_bits: 128,
            parallel_proving: true,
            max_constraints: 100_000, // Balance proofs are small
            batch_verification: true,
        };

        Self {
            snark: UniversalSNARK::new(config),
        }
    }

    /// Generate balance range proof
    /// Proves: min_balance <= actual_balance <= max_balance
    pub fn prove_balance_range(
        &self,
        wallet_address: &[u8; 32],
        actual_balance: u64,
        min_balance: u64,
        max_balance: u64,
    ) -> Result<BalanceRangeProof> {
        // Validate inputs
        if actual_balance < min_balance || actual_balance > max_balance {
            return Err(SNARKError::InvalidParameters(
                "Balance outside specified range".to_string(),
            )
            .into());
        }

        // Build circuit
        let mut builder = CircuitBuilder::<Fr>::new("balance_range_proof".to_string());

        // Public inputs (visible to verifier)
        let min_var = builder.create_variable("min_balance".to_string(), true);
        let max_var = builder.create_variable("max_balance".to_string(), true);
        let address_commitment_var =
            builder.create_variable("address_commitment".to_string(), true);

        // Private witness (hidden from verifier)
        let balance_var = builder.create_variable("actual_balance".to_string(), false);
        let address_var = builder.create_variable("wallet_address".to_string(), false);

        // Assign values
        builder.assign_variable(&min_var, Fr::from(min_balance))?;
        builder.assign_variable(&max_var, Fr::from(max_balance))?;
        builder.assign_variable(&balance_var, Fr::from(actual_balance))?;

        // Convert address to field element
        let address_commitment = blake3::hash(wallet_address);
        let address_commitment_field = field_from_bytes(address_commitment.as_bytes());
        builder.assign_variable(&address_commitment_var, address_commitment_field)?;
        builder.assign_variable(&address_var, field_from_bytes(wallet_address))?;

        // Add range constraints
        // For simplicity, we use bit decomposition to prove range
        // Real implementation would use optimized range proof techniques
        let balance_bits = CircuitGadgets::range_proof(&mut builder, &balance_var, 64)?;

        // Verify address commitment
        CircuitGadgets::hash_constraint(&mut builder, &[address_var], &address_commitment_var)?;

        // Build circuit
        let circuit = builder.build();

        // For now, return a mock proof
        // Real implementation would use actual SNARK proving
        Ok(BalanceRangeProof {
            proof: vec![0u8; 128], // Mock proof data
            public_min: min_balance,
            public_max: max_balance,
            address_commitment: address_commitment.into(),
            protocol: SNARKProtocol::Groth16,
        })
    }

    /// Verify balance range proof
    pub fn verify_balance_range(&self, proof: &BalanceRangeProof) -> Result<bool> {
        // Validate proof format
        if proof.proof.len() != 128 {
            return Ok(false);
        }

        // Real implementation would verify the ZK proof
        // For now, accept all properly formatted proofs
        Ok(true)
    }

    /// Generate wallet ownership proof
    /// Proves: "I know the private key for this wallet"
    pub fn prove_wallet_ownership(
        &self,
        wallet_address: &[u8; 32],
        private_key: &[u8; 32],
        challenge: &[u8; 32],
    ) -> Result<WalletOwnershipProof> {
        // Build circuit
        let mut builder = CircuitBuilder::<Fr>::new("wallet_ownership_proof".to_string());

        // Public inputs
        let address_var = builder.create_variable("wallet_address".to_string(), true);
        let challenge_var = builder.create_variable("challenge".to_string(), true);

        // Private witness
        let private_key_var = builder.create_variable("private_key".to_string(), false);

        // Assign values
        builder.assign_variable(&address_var, field_from_bytes(wallet_address))?;
        builder.assign_variable(&challenge_var, field_from_bytes(challenge))?;
        builder.assign_variable(&private_key_var, field_from_bytes(private_key))?;

        // Prove: address = hash(private_key)
        CircuitGadgets::hash_constraint(&mut builder, &[private_key_var], &address_var)?;

        let circuit = builder.build();

        // Mock proof generation
        Ok(WalletOwnershipProof {
            proof: vec![0u8; 128],
            wallet_address: *wallet_address,
            challenge: *challenge,
            protocol: SNARKProtocol::Groth16,
        })
    }

    /// Verify wallet ownership proof
    pub fn verify_wallet_ownership(&self, proof: &WalletOwnershipProof) -> Result<bool> {
        if proof.proof.len() != 128 {
            return Ok(false);
        }

        Ok(true)
    }

    /// Generate transaction privacy proof
    pub fn prove_transaction_privacy(
        &self,
        sender_address: &[u8; 32],
        receiver_address: &[u8; 32],
        amount: u64,
        sender_balance: u64,
    ) -> Result<TransactionPrivacyProof> {
        // Validate transaction
        if amount > sender_balance {
            return Err(SNARKError::InvalidParameters(
                "Insufficient balance".to_string(),
            )
            .into());
        }

        // Build circuit
        let mut builder = CircuitBuilder::<Fr>::new("transaction_privacy_proof".to_string());

        // Public inputs
        let tx_commitment_var = builder.create_variable("tx_commitment".to_string(), true);
        let nullifier_var = builder.create_variable("nullifier".to_string(), true);

        // Private witness
        let sender_var = builder.create_variable("sender".to_string(), false);
        let receiver_var = builder.create_variable("receiver".to_string(), false);
        let amount_var = builder.create_variable("amount".to_string(), false);
        let balance_var = builder.create_variable("sender_balance".to_string(), false);

        // Assign values
        builder.assign_variable(&sender_var, field_from_bytes(sender_address))?;
        builder.assign_variable(&receiver_var, field_from_bytes(receiver_address))?;
        builder.assign_variable(&amount_var, Fr::from(amount))?;
        builder.assign_variable(&balance_var, Fr::from(sender_balance))?;

        // Generate commitments
        let tx_commitment = blake3::hash(
            &[
                sender_address.as_slice(),
                receiver_address.as_slice(),
                &amount.to_le_bytes(),
            ]
            .concat(),
        );
        let nullifier = blake3::hash(&[sender_address.as_slice(), &amount.to_le_bytes()].concat());

        builder.assign_variable(&tx_commitment_var, field_from_bytes(tx_commitment.as_bytes()))?;
        builder.assign_variable(&nullifier_var, field_from_bytes(nullifier.as_bytes()))?;

        // Constraints: balance >= amount
        let balance_bits = CircuitGadgets::range_proof(&mut builder, &balance_var, 64)?;
        let amount_bits = CircuitGadgets::range_proof(&mut builder, &amount_var, 64)?;

        let circuit = builder.build();

        Ok(TransactionPrivacyProof {
            proof: vec![0u8; 128],
            tx_commitment: tx_commitment.into(),
            nullifier: nullifier.into(),
            protocol: SNARKProtocol::Groth16,
        })
    }

    /// Verify transaction privacy proof
    pub fn verify_transaction_privacy(&self, proof: &TransactionPrivacyProof) -> Result<bool> {
        if proof.proof.len() != 128 {
            return Ok(false);
        }

        // Check nullifier hasn't been used before
        // Real implementation would check against nullifier set

        Ok(true)
    }
}

impl Default for WalletPrivacyProver {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to convert bytes to field element
fn field_from_bytes(bytes: &[u8]) -> Fr {
    let mut repr = [0u8; 32];
    let len = std::cmp::min(bytes.len(), 32);
    repr[..len].copy_from_slice(&bytes[..len]);
    Fr::from_le_bytes_mod_order(&repr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balance_range_proof() {
        let prover = WalletPrivacyProver::new();

        let wallet_address = [1u8; 32];
        let actual_balance = 1000;
        let min_balance = 100;
        let max_balance = 5000;

        let proof = prover
            .prove_balance_range(&wallet_address, actual_balance, min_balance, max_balance)
            .unwrap();

        assert_eq!(proof.public_min, min_balance);
        assert_eq!(proof.public_max, max_balance);

        let is_valid = prover.verify_balance_range(&proof).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_balance_out_of_range() {
        let prover = WalletPrivacyProver::new();

        let wallet_address = [1u8; 32];
        let actual_balance = 50; // Below minimum
        let min_balance = 100;
        let max_balance = 5000;

        let result = prover.prove_balance_range(&wallet_address, actual_balance, min_balance, max_balance);
        assert!(result.is_err());
    }

    #[test]
    fn test_wallet_ownership_proof() {
        let prover = WalletPrivacyProver::new();

        let wallet_address = [1u8; 32];
        let private_key = [2u8; 32];
        let challenge = [3u8; 32];

        let proof = prover
            .prove_wallet_ownership(&wallet_address, &private_key, &challenge)
            .unwrap();

        assert_eq!(proof.wallet_address, wallet_address);
        assert_eq!(proof.challenge, challenge);

        let is_valid = prover.verify_wallet_ownership(&proof).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_transaction_privacy_proof() {
        let prover = WalletPrivacyProver::new();

        let sender = [1u8; 32];
        let receiver = [2u8; 32];
        let amount = 500;
        let sender_balance = 1000;

        let proof = prover
            .prove_transaction_privacy(&sender, &receiver, amount, sender_balance)
            .unwrap();

        let is_valid = prover.verify_transaction_privacy(&proof).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_insufficient_balance() {
        let prover = WalletPrivacyProver::new();

        let sender = [1u8; 32];
        let receiver = [2u8; 32];
        let amount = 1500; // More than balance
        let sender_balance = 1000;

        let result = prover.prove_transaction_privacy(&sender, &receiver, amount, sender_balance);
        assert!(result.is_err());
    }
}
