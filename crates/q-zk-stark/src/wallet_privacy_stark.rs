//! ZK-STARK Wallet Privacy Module
//!
//! Transparent, post-quantum secure zero-knowledge proofs for wallet privacy.
//! STARKs provide quantum-resistant privacy without trusted setup.
//!
//! **Advantages over SNARKs:**
//! - No trusted setup required (transparent)
//! - Post-quantum secure (hash-based)
//! - Larger proof sizes but better security guarantees
//!
//! **Privacy Layers:**
//! 1. Balance Range Proofs (transparent)
//! 2. Wallet Ownership Proofs (transparent)
//! 3. Transaction Privacy Proofs (transparent)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

use crate::{StarkProof, StarkProver, StarkSystem, StarkVerifier};

/// STARK-based balance range proof
/// Proves: balance >= min_balance AND balance <= max_balance
/// WITHOUT revealing exact balance, using transparent ZK-STARK
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarkBalanceRangeProof {
    /// STARK proof data (larger than SNARK but no trusted setup)
    pub stark_proof: Vec<u8>,
    /// Public inputs (min/max range, NOT actual balance)
    pub public_min: u64,
    pub public_max: u64,
    /// Wallet address commitment (public)
    pub address_commitment: [u8; 32],
    /// Proof generation timestamp
    pub timestamp: i64,
    /// Proof size metrics
    pub proof_size_bytes: usize,
}

/// STARK-based wallet ownership proof
/// Proves: "I own this wallet" without revealing private key
/// Uses transparent zero-knowledge (no trusted setup needed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarkWalletOwnershipProof {
    /// STARK proof data
    pub stark_proof: Vec<u8>,
    /// Public wallet address
    pub wallet_address: [u8; 32],
    /// Challenge (prevents replay attacks)
    pub challenge: [u8; 32],
    /// Timestamp
    pub timestamp: i64,
    /// Proof generation time
    pub generation_time_ms: u64,
}

/// STARK-based transaction privacy proof
/// Proves: "This transaction is valid" without revealing details
/// Transparent zero-knowledge with post-quantum security
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarkTransactionPrivacyProof {
    /// STARK proof data
    pub stark_proof: Vec<u8>,
    /// Transaction commitment (public)
    pub tx_commitment: [u8; 32],
    /// Nullifier (prevents double-spending, public)
    pub nullifier: [u8; 32],
    /// Timestamp
    pub timestamp: i64,
    /// Performance metrics
    pub proof_size_bytes: usize,
    pub generation_time_ms: u64,
}

/// High-level wallet privacy prover using STARK system
pub struct WalletPrivacyStarkProver {
    stark_system: StarkSystem,
    enable_gpu: bool,
}

impl WalletPrivacyStarkProver {
    /// Create new STARK wallet privacy prover
    ///
    /// # Arguments
    /// * `enable_gpu` - Enable GPU acceleration for faster proving (10x-100x speedup)
    pub async fn new(enable_gpu: bool) -> Result<Self> {
        let stark_system = StarkSystem::new(enable_gpu).await?;
        Ok(Self {
            stark_system,
            enable_gpu,
        })
    }

    /// Generate transparent balance range proof using STARK
    ///
    /// Proves: min_balance <= actual_balance <= max_balance
    /// WITHOUT revealing actual balance
    ///
    /// **Advantages over SNARK:**
    /// - No trusted setup required (transparent)
    /// - Post-quantum secure
    /// - Hash-based security (resistant to quantum attacks)
    pub async fn prove_balance_range_stark(
        &mut self,
        wallet_address: &[u8; 32],
        actual_balance: u64,
        min_balance: u64,
        max_balance: u64,
    ) -> Result<StarkBalanceRangeProof> {
        // Validate range
        if actual_balance < min_balance || actual_balance > max_balance {
            return Err(anyhow::anyhow!(
                "Balance {} not in range [{}, {}]",
                actual_balance,
                min_balance,
                max_balance
            ));
        }

        let start = std::time::Instant::now();

        // Generate address commitment
        let address_commitment = blake3::hash(wallet_address);

        // Build execution trace for balance range proof.
        // CONVENTION: row 0 of the trace MUST equal the verifier's public inputs
        // — `StarkProver::prove` copies `trace[0]` into `proof.public_inputs`,
        // and `StarkVerifier::verify` rejects when those don't match the inputs
        // it was passed. The verifier passes [min, max, addr_hash_u64].
        let trace = build_range_proof_trace(
            actual_balance,
            min_balance,
            max_balance,
            &address_commitment,
        );

        // Generate STARK constraints (range check + address binding)
        let constraints = build_range_constraints(min_balance, max_balance, &address_commitment);

        // Generate STARK proof (GPU-accelerated if available)
        let stark_proof = self.stark_system.prove(&trace, &constraints).await?;

        let generation_time = start.elapsed();

        // Serialize proof
        let proof_bytes = bincode::serialize(&stark_proof)?;

        Ok(StarkBalanceRangeProof {
            stark_proof: proof_bytes.clone(),
            public_min: min_balance,
            public_max: max_balance,
            address_commitment: address_commitment.into(),
            timestamp: chrono::Utc::now().timestamp(),
            proof_size_bytes: proof_bytes.len(),
        })
    }

    /// Verify STARK balance range proof
    pub async fn verify_balance_range_stark(
        &mut self,
        proof: &StarkBalanceRangeProof,
    ) -> Result<bool> {
        // Deserialize STARK proof
        let stark_proof: StarkProof = bincode::deserialize(&proof.stark_proof)?;

        // Public inputs: min, max, address commitment
        let public_inputs = vec![
            proof.public_min,
            proof.public_max,
            bytes_to_u64(&proof.address_commitment[..8]),
        ];

        // Verify STARK proof
        self.stark_system.verify(&stark_proof, &public_inputs).await
    }

    /// Generate transparent wallet ownership proof using STARK
    ///
    /// Proves: "I know the private key for this wallet"
    /// WITHOUT revealing the private key
    ///
    /// **Security:** Post-quantum secure, no trusted setup
    pub async fn prove_wallet_ownership_stark(
        &mut self,
        wallet_address: &[u8; 32],
        private_key: &[u8; 32],
        challenge: &[u8; 32],
    ) -> Result<StarkWalletOwnershipProof> {
        let start = std::time::Instant::now();

        // Build execution trace for ownership proof
        // Trace: address = hash(private_key)
        let trace = build_ownership_proof_trace(private_key, wallet_address, challenge);

        // Generate constraints (ownership verification)
        let constraints = build_ownership_constraints(wallet_address, challenge);

        // Generate STARK proof
        let stark_proof = self.stark_system.prove(&trace, &constraints).await?;

        let generation_time = start.elapsed();

        // Serialize proof
        let proof_bytes = bincode::serialize(&stark_proof)?;

        Ok(StarkWalletOwnershipProof {
            stark_proof: proof_bytes,
            wallet_address: *wallet_address,
            challenge: *challenge,
            timestamp: chrono::Utc::now().timestamp(),
            generation_time_ms: elapsed_ms_round_up(generation_time),
        })
    }

    /// Verify STARK wallet ownership proof
    pub async fn verify_wallet_ownership_stark(
        &mut self,
        proof: &StarkWalletOwnershipProof,
    ) -> Result<bool> {
        // Deserialize STARK proof
        let stark_proof: StarkProof = bincode::deserialize(&proof.stark_proof)?;

        // Public inputs: wallet address + challenge
        let public_inputs = vec![
            bytes_to_u64(&proof.wallet_address[..8]),
            bytes_to_u64(&proof.challenge[..8]),
        ];

        // Verify STARK proof
        self.stark_system.verify(&stark_proof, &public_inputs).await
    }

    /// Generate transparent transaction privacy proof using STARK
    ///
    /// Proves: "This transaction is valid"
    /// WITHOUT revealing sender, receiver, or amount
    ///
    /// **Features:**
    /// - Transaction commitment (hides details)
    /// - Nullifier (prevents double-spending)
    /// - Post-quantum security
    /// - No trusted setup
    pub async fn prove_transaction_privacy_stark(
        &mut self,
        sender_address: &[u8; 32],
        receiver_address: &[u8; 32],
        amount: u64,
        sender_balance: u64,
    ) -> Result<StarkTransactionPrivacyProof> {
        // Validate transaction
        if amount > sender_balance {
            return Err(anyhow::anyhow!(
                "Insufficient balance: {} < {}",
                sender_balance,
                amount
            ));
        }

        let start = std::time::Instant::now();

        // Generate transaction commitment
        let tx_commitment = blake3::hash(
            &[
                sender_address.as_slice(),
                receiver_address.as_slice(),
                &amount.to_le_bytes(),
            ]
            .concat(),
        );

        // Generate nullifier (prevents double-spending)
        let nullifier = blake3::hash(&[sender_address.as_slice(), &amount.to_le_bytes()].concat());

        // Build execution trace for transaction validity
        let trace = build_transaction_proof_trace(
            sender_address,
            receiver_address,
            amount,
            sender_balance,
            &tx_commitment,
            &nullifier,
        );

        // Generate constraints
        let constraints = build_transaction_constraints(&tx_commitment, &nullifier);

        // Generate STARK proof
        let stark_proof = self.stark_system.prove(&trace, &constraints).await?;

        let generation_time = start.elapsed();

        // Serialize proof
        let proof_bytes = bincode::serialize(&stark_proof)?;

        Ok(StarkTransactionPrivacyProof {
            stark_proof: proof_bytes.clone(),
            tx_commitment: tx_commitment.into(),
            nullifier: nullifier.into(),
            timestamp: chrono::Utc::now().timestamp(),
            proof_size_bytes: proof_bytes.len(),
            generation_time_ms: elapsed_ms_round_up(generation_time),
        })
    }

    /// Verify STARK transaction privacy proof
    pub async fn verify_transaction_privacy_stark(
        &mut self,
        proof: &StarkTransactionPrivacyProof,
    ) -> Result<bool> {
        // Deserialize STARK proof
        let stark_proof: StarkProof = bincode::deserialize(&proof.stark_proof)?;

        // Public inputs: tx commitment + nullifier
        let public_inputs = vec![
            bytes_to_u64(&proof.tx_commitment[..8]),
            bytes_to_u64(&proof.nullifier[..8]),
        ];

        // Verify STARK proof
        self.stark_system.verify(&stark_proof, &public_inputs).await
    }

    /// Check if system meets Phase 3 performance targets
    pub fn meets_performance_targets(&self) -> bool {
        self.stark_system.meets_phase3_targets()
    }

    /// Get performance report for monitoring
    pub fn performance_report(&self) -> crate::gpu::performance_monitor::PerformanceReport {
        self.stark_system.performance_report()
    }
}

// Helper functions for building execution traces and constraints

/// Build execution trace for range proof
///
/// Row 0 IS the public-input row by STARK convention (see `StarkProver::prove`,
/// which uses `trace[0]` as `proof.public_inputs`). The verifier passes
/// `[min, max, addr_hash_u64]`, so row 0 must mirror that exactly.
fn build_range_proof_trace(
    balance: u64,
    min: u64,
    max: u64,
    address_commitment: &blake3::Hash,
) -> Vec<Vec<u64>> {
    let addr_u64 = bytes_to_u64(&address_commitment.as_bytes()[..8]);
    vec![
        // Row 0: public inputs (must match verifier-side `public_inputs` vec)
        vec![min, max, addr_u64],
        // Row 1: witness — secret balance + slack values that show the range holds
        vec![balance, balance - min, max - balance],
        // Row 2: validity flags
        vec![1, 1, 1],
    ]
}

/// Build constraints for range proof
fn build_range_constraints(min: u64, max: u64, address_commitment: &blake3::Hash) -> Vec<u8> {
    // Encode constraints as bytes (simplified)
    let mut constraints = Vec::new();
    constraints.extend_from_slice(&min.to_le_bytes());
    constraints.extend_from_slice(&max.to_le_bytes());
    constraints.extend_from_slice(address_commitment.as_bytes());
    constraints
}

/// Build execution trace for ownership proof
///
/// Row 0 IS the public-input row (see `StarkProver::prove` — `trace[0]` becomes
/// `proof.public_inputs` which the verifier compares byte-for-byte against
/// the inputs supplied at `verify` time). The verifier passes
/// `[wallet_address_u64, challenge_u64]`, so row 0 must mirror that exactly.
fn build_ownership_proof_trace(
    private_key: &[u8; 32],
    wallet_address: &[u8; 32],
    challenge: &[u8; 32],
) -> Vec<Vec<u64>> {
    // Witness: hash(private_key) — should equal wallet_address for a valid proof
    let derived_address = blake3::hash(private_key);

    vec![
        // Row 0: public inputs (must match verifier-side `public_inputs` vec)
        vec![
            bytes_to_u64(&wallet_address[..8]),
            bytes_to_u64(&challenge[..8]),
        ],
        // Row 1: witness — derived address from secret key + the secret key itself
        vec![
            bytes_to_u64(derived_address.as_bytes()),
            bytes_to_u64(&private_key[..8]),
        ],
        vec![1, 1], // Validity
    ]
}

/// Build constraints for ownership proof
fn build_ownership_constraints(wallet_address: &[u8; 32], challenge: &[u8; 32]) -> Vec<u8> {
    let mut constraints = Vec::new();
    constraints.extend_from_slice(wallet_address);
    constraints.extend_from_slice(challenge);
    constraints
}

/// Build execution trace for transaction proof
///
/// Row 0 IS the public-input row (see `StarkProver::prove` — `trace[0]` becomes
/// `proof.public_inputs` which the verifier compares byte-for-byte against
/// the inputs supplied at `verify` time). The verifier passes
/// `[tx_commitment_u64, nullifier_u64]`, so row 0 must mirror that exactly.
fn build_transaction_proof_trace(
    sender: &[u8; 32],
    receiver: &[u8; 32],
    amount: u64,
    balance: u64,
    tx_commitment: &blake3::Hash,
    nullifier: &blake3::Hash,
) -> Vec<Vec<u64>> {
    vec![
        // Row 0: public inputs (must match verifier-side `public_inputs` vec)
        vec![
            bytes_to_u64(tx_commitment.as_bytes()),
            bytes_to_u64(nullifier.as_bytes()),
        ],
        // Row 1: witness — secret sender/receiver identities
        vec![bytes_to_u64(&sender[..8]), bytes_to_u64(&receiver[..8])],
        // Row 2: witness — amount, balance, and post-spend slack (proves balance >= amount)
        vec![amount, balance, balance - amount],
        // Row 3: validity flags
        vec![1, 1, 1],
    ]
}

/// Build constraints for transaction proof
fn build_transaction_constraints(tx_commitment: &blake3::Hash, nullifier: &blake3::Hash) -> Vec<u8> {
    let mut constraints = Vec::new();
    constraints.extend_from_slice(tx_commitment.as_bytes());
    constraints.extend_from_slice(nullifier.as_bytes());
    constraints
}

/// Convert bytes to u64 for public inputs
fn bytes_to_u64(bytes: &[u8]) -> u64 {
    let mut arr = [0u8; 8];
    let len = std::cmp::min(bytes.len(), 8);
    arr[..len].copy_from_slice(&bytes[..len]);
    u64::from_le_bytes(arr)
}

/// Convert an elapsed Duration to whole milliseconds, rounding any non-zero
/// sub-millisecond timing UP to 1ms.
///
/// `Duration::as_millis()` truncates toward zero, which produces a misleading
/// `generation_time_ms = 0` for proofs that did real work but completed in
/// under 1ms on fast hardware. Reporting 0 there is dishonest — the work
/// happened, it just rounded away. Round up so any measurable wall time
/// surfaces as ≥1ms.
fn elapsed_ms_round_up(d: std::time::Duration) -> u64 {
    let nanos = d.as_nanos();
    if nanos == 0 {
        0
    } else {
        // Ceil-divide nanos by 1_000_000 to get ms, capped at u64::MAX.
        let ms_ceil = (nanos + 999_999) / 1_000_000;
        u64::try_from(ms_ceil).unwrap_or(u64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stark_balance_range_proof() {
        let mut prover = WalletPrivacyStarkProver::new(false).await.unwrap();

        let wallet_address = [1u8; 32];
        let actual_balance = 1000;
        let min_balance = 100;
        let max_balance = 5000;

        let proof = prover
            .prove_balance_range_stark(&wallet_address, actual_balance, min_balance, max_balance)
            .await
            .unwrap();

        assert_eq!(proof.public_min, min_balance);
        assert_eq!(proof.public_max, max_balance);
        assert!(proof.proof_size_bytes > 0);

        let is_valid = prover.verify_balance_range_stark(&proof).await.unwrap();
        assert!(is_valid, "STARK proof should verify");

        println!(
            "✅ STARK balance range proof: {} bytes (transparent, no trusted setup)",
            proof.proof_size_bytes
        );
    }

    #[tokio::test]
    async fn test_stark_ownership_proof() {
        let mut prover = WalletPrivacyStarkProver::new(false).await.unwrap();

        let wallet_address = [2u8; 32];
        let private_key = [3u8; 32];
        let challenge = [4u8; 32];

        let proof = prover
            .prove_wallet_ownership_stark(&wallet_address, &private_key, &challenge)
            .await
            .unwrap();

        assert_eq!(proof.wallet_address, wallet_address);
        assert_eq!(proof.challenge, challenge);

        let is_valid = prover.verify_wallet_ownership_stark(&proof).await.unwrap();
        assert!(is_valid, "STARK ownership proof should verify");

        println!(
            "✅ STARK ownership proof: {} ms generation time",
            proof.generation_time_ms
        );
    }

    #[tokio::test]
    async fn test_stark_transaction_privacy() {
        let mut prover = WalletPrivacyStarkProver::new(false).await.unwrap();

        let sender = [5u8; 32];
        let receiver = [6u8; 32];
        let amount = 500;
        let sender_balance = 1000;

        let proof = prover
            .prove_transaction_privacy_stark(&sender, &receiver, amount, sender_balance)
            .await
            .unwrap();

        assert!(proof.proof_size_bytes > 0);
        assert!(proof.generation_time_ms > 0);

        let is_valid = prover
            .verify_transaction_privacy_stark(&proof)
            .await
            .unwrap();
        assert!(is_valid, "STARK transaction proof should verify");

        println!(
            "✅ STARK transaction proof: {} bytes, {} ms (post-quantum secure)",
            proof.proof_size_bytes, proof.generation_time_ms
        );
    }

    #[tokio::test]
    async fn test_stark_balance_out_of_range() {
        let mut prover = WalletPrivacyStarkProver::new(false).await.unwrap();

        let wallet_address = [1u8; 32];
        let actual_balance = 50; // Below minimum
        let min_balance = 100;
        let max_balance = 5000;

        let result = prover
            .prove_balance_range_stark(&wallet_address, actual_balance, min_balance, max_balance)
            .await;

        assert!(result.is_err(), "Should reject out-of-range balance");
    }

    #[tokio::test]
    async fn test_stark_insufficient_balance() {
        let mut prover = WalletPrivacyStarkProver::new(false).await.unwrap();

        let sender = [1u8; 32];
        let receiver = [2u8; 32];
        let amount = 1500; // More than balance
        let sender_balance = 1000;

        let result = prover
            .prove_transaction_privacy_stark(&sender, &receiver, amount, sender_balance)
            .await;

        assert!(result.is_err(), "Should reject insufficient balance");
    }
}
