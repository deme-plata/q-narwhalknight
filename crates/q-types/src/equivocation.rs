/// Equivocation Detection Types for Phase 3 Consensus Security
///
/// This module defines the types needed to detect and prove Byzantine
/// equivocation (double-signing) behavior in validators.
///
/// Equivocation occurs when a validator:
/// 1. Signs two different blocks at the same height (double-signing)
/// 2. Votes for two different blocks in the same consensus round (double-voting)
///
/// Both are provable Byzantine faults that result in slashing.

use serde::{Deserialize, Serialize};

/// Unique identifier for a validator (hash of public key)
pub type ValidatorId = [u8; 32];

/// Block hash (32 bytes)
pub type BlockHash = [u8; 32];

/// Ed25519 signature (64 bytes) - stored as Vec for serde compatibility
pub type Signature = Vec<u8>;

/// Cryptographic proof of equivocation (double-signing)
///
/// This proof demonstrates that a validator signed two different
/// blocks at the same height - an unambiguous Byzantine fault.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EquivocationProof {
    /// The validator who equivocated
    pub validator: ValidatorId,

    /// Public key of the validator (for signature verification)
    pub public_key: [u8; 32],

    /// First block hash that was signed
    pub block_a: BlockHash,

    /// Second (conflicting) block hash that was signed
    pub block_b: BlockHash,

    /// Height at which both blocks were produced
    pub height: u64,

    /// Signature on block_a
    pub signature_a: Signature,

    /// Signature on block_b
    pub signature_b: Signature,

    /// Timestamp when equivocation was detected
    pub detected_at: u64,

    /// Block height when equivocation was detected
    pub detected_at_height: u64,
}

impl EquivocationProof {
    /// Create new equivocation proof
    pub fn new(
        validator: ValidatorId,
        public_key: [u8; 32],
        block_a: BlockHash,
        block_b: BlockHash,
        height: u64,
        signature_a: Signature,
        signature_b: Signature,
        detected_at: u64,
        detected_at_height: u64,
    ) -> Self {
        Self {
            validator,
            public_key,
            block_a,
            block_b,
            height,
            signature_a,
            signature_b,
            detected_at,
            detected_at_height,
        }
    }

    /// Verify that this is a valid equivocation proof
    ///
    /// Checks:
    /// 1. block_a != block_b (actually different blocks)
    /// 2. Both signatures are valid for the respective blocks
    /// 3. Both signatures are from the same validator
    pub fn verify(&self) -> Result<(), EquivocationError> {
        // Check blocks are different
        if self.block_a == self.block_b {
            return Err(EquivocationError::SameBlock);
        }

        // Verify signature_a
        if !self.verify_signature(&self.block_a, &self.signature_a) {
            return Err(EquivocationError::InvalidSignatureA);
        }

        // Verify signature_b
        if !self.verify_signature(&self.block_b, &self.signature_b) {
            return Err(EquivocationError::InvalidSignatureB);
        }

        Ok(())
    }

    /// Verify an Ed25519 signature
    fn verify_signature(&self, block_hash: &BlockHash, signature: &Signature) -> bool {
        use ed25519_dalek::{Signature as Ed25519Sig, Verifier, VerifyingKey};

        let Ok(verifying_key) = VerifyingKey::from_bytes(&self.public_key) else {
            return false;
        };

        // Convert Vec<u8> to [u8; 64]
        if signature.len() != 64 {
            return false;
        }
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(signature);

        // ed25519_dalek 2.x: from_bytes returns Signature directly
        let sig = Ed25519Sig::from_bytes(&sig_bytes);

        verifying_key.verify(block_hash, &sig).is_ok()
    }

    /// Get the hash of this proof (for deduplication)
    pub fn hash(&self) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};

        let mut hasher = Sha3_256::new();
        hasher.update(&self.validator);
        hasher.update(&self.block_a);
        hasher.update(&self.block_b);
        hasher.update(&self.height.to_le_bytes());
        hasher.finalize().into()
    }
}

/// Error types for equivocation verification
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EquivocationError {
    /// The two blocks are actually the same
    SameBlock,
    /// Signature A is invalid
    InvalidSignatureA,
    /// Signature B is invalid
    InvalidSignatureB,
    /// Validator public key is invalid
    InvalidPublicKey,
    /// Height mismatch
    HeightMismatch,
}

impl std::fmt::Display for EquivocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SameBlock => write!(f, "Blocks are identical, not equivocation"),
            Self::InvalidSignatureA => write!(f, "First signature is invalid"),
            Self::InvalidSignatureB => write!(f, "Second signature is invalid"),
            Self::InvalidPublicKey => write!(f, "Validator public key is invalid"),
            Self::HeightMismatch => write!(f, "Block heights do not match"),
        }
    }
}

impl std::error::Error for EquivocationError {}

/// Double-vote proof (within consensus rounds)
///
/// Similar to equivocation but for consensus votes rather than block production.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DoubleVoteProof {
    /// The validator who double-voted
    pub validator: ValidatorId,

    /// Public key of the validator
    pub public_key: [u8; 32],

    /// Consensus round number
    pub round: u64,

    /// First vertex/block voted for
    pub vote_a: BlockHash,

    /// Second (conflicting) vertex/block voted for
    pub vote_b: BlockHash,

    /// Signature on vote_a
    pub signature_a: Signature,

    /// Signature on vote_b
    pub signature_b: Signature,

    /// Timestamp when detected
    pub detected_at: u64,
}

impl DoubleVoteProof {
    /// Verify this is a valid double-vote proof
    pub fn verify(&self) -> Result<(), EquivocationError> {
        if self.vote_a == self.vote_b {
            return Err(EquivocationError::SameBlock);
        }

        // Signature verification would go here
        // (simplified for now - full implementation would verify vote messages)

        Ok(())
    }

    /// Get the hash of this proof
    pub fn hash(&self) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};

        let mut hasher = Sha3_256::new();
        hasher.update(&self.validator);
        hasher.update(&self.round.to_le_bytes());
        hasher.update(&self.vote_a);
        hasher.update(&self.vote_b);
        hasher.finalize().into()
    }
}

/// Slashing evidence that can be submitted on-chain
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SlashingEvidence {
    /// Double-signing proof (block production)
    Equivocation(EquivocationProof),
    /// Double-vote proof (consensus voting)
    DoubleVote(DoubleVoteProof),
}

impl SlashingEvidence {
    /// Get the validator being accused
    pub fn validator(&self) -> ValidatorId {
        match self {
            Self::Equivocation(proof) => proof.validator,
            Self::DoubleVote(proof) => proof.validator,
        }
    }

    /// Verify the evidence is valid
    pub fn verify(&self) -> Result<(), EquivocationError> {
        match self {
            Self::Equivocation(proof) => proof.verify(),
            Self::DoubleVote(proof) => proof.verify(),
        }
    }

    /// Get the hash of this evidence
    pub fn hash(&self) -> [u8; 32] {
        match self {
            Self::Equivocation(proof) => proof.hash(),
            Self::DoubleVote(proof) => proof.hash(),
        }
    }

    /// Get the severity of this evidence (for slashing calculation)
    pub fn severity(&self) -> SlashingSeverity {
        match self {
            Self::Equivocation(_) => SlashingSeverity::Severe,
            Self::DoubleVote(_) => SlashingSeverity::Major,
        }
    }
}

/// Severity level for slashing
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SlashingSeverity {
    /// Minor infraction (1% slash)
    Minor,
    /// Major infraction (10% slash)
    Major,
    /// Severe infraction (100% slash + removal)
    Severe,
}

impl SlashingSeverity {
    /// Get the slash percentage
    pub fn slash_percent(&self) -> u64 {
        match self {
            Self::Minor => 1,
            Self::Major => 10,
            Self::Severe => 100,
        }
    }
}

/// Slashing transaction to be included in a block
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlashingTransaction {
    /// Evidence proving Byzantine behavior
    pub evidence: SlashingEvidence,

    /// Reporter who submitted the evidence (gets a bounty)
    pub reporter: [u8; 32],

    /// Amount to slash from validator
    pub slash_amount: u64,

    /// Bounty amount for reporter (percentage of slash)
    pub bounty_amount: u64,

    /// Block height when slashing transaction was created
    pub created_at_height: u64,

    /// Signature of the reporter (optional, for bounty claim)
    pub reporter_signature: Option<Signature>,
}

impl SlashingTransaction {
    /// Create new slashing transaction
    pub fn new(
        evidence: SlashingEvidence,
        reporter: [u8; 32],
        validator_stake: u64,
        current_height: u64,
    ) -> Self {
        let severity = evidence.severity();
        let slash_amount = (validator_stake * severity.slash_percent()) / 100;

        // Reporter bounty: 10% of slashed amount
        let bounty_amount = slash_amount / 10;

        Self {
            evidence,
            reporter,
            slash_amount,
            bounty_amount,
            created_at_height: current_height,
            reporter_signature: None,
        }
    }

    /// Verify the slashing transaction
    pub fn verify(&self) -> Result<(), EquivocationError> {
        self.evidence.verify()
    }

    /// Get the validator being slashed
    pub fn validator(&self) -> ValidatorId {
        self.evidence.validator()
    }

    /// Get transaction hash
    pub fn hash(&self) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};

        let mut hasher = Sha3_256::new();
        hasher.update(&self.evidence.hash());
        hasher.update(&self.reporter);
        hasher.update(&self.slash_amount.to_le_bytes());
        hasher.update(&self.created_at_height.to_le_bytes());
        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    /// Deterministic signing key for tests (no rand / OsRng dependency).
    fn signing_key_from_index(i: u32) -> SigningKey {
        let mut seed = [0u8; 32];
        seed[0..4].copy_from_slice(&i.to_le_bytes());
        SigningKey::from_bytes(&seed)
    }

    fn create_test_equivocation() -> EquivocationProof {
        // Create a signing key
        let signing_key = signing_key_from_index(0);
        let public_key = signing_key.verifying_key();

        // Create two different block hashes
        let block_a = [1u8; 32];
        let block_b = [2u8; 32];

        // Sign both blocks
        use ed25519_dalek::Signer;
        let sig_a = signing_key.sign(&block_a);
        let sig_b = signing_key.sign(&block_b);

        // Create validator ID from public key hash
        use sha3::{Digest, Sha3_256};
        let validator: [u8; 32] = {
            let mut hasher = Sha3_256::new();
            hasher.update(public_key.as_bytes());
            hasher.finalize().into()
        };

        EquivocationProof::new(
            validator,
            public_key.to_bytes(),
            block_a,
            block_b,
            100, // height
            sig_a.to_bytes().to_vec(),
            sig_b.to_bytes().to_vec(),
            12345,
            101,
        )
    }

    #[test]
    fn test_equivocation_proof_verification() {
        let proof = create_test_equivocation();
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_same_block_rejected() {
        let mut proof = create_test_equivocation();
        proof.block_b = proof.block_a; // Make blocks the same
        assert_eq!(proof.verify(), Err(EquivocationError::SameBlock));
    }

    #[test]
    fn test_slashing_severity() {
        assert_eq!(SlashingSeverity::Minor.slash_percent(), 1);
        assert_eq!(SlashingSeverity::Major.slash_percent(), 10);
        assert_eq!(SlashingSeverity::Severe.slash_percent(), 100);
    }
}
