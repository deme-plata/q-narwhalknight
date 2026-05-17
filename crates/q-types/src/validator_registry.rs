/// Validator Registry with Dilithium5 Post-Quantum Signatures
/// v3.7.4: NIST Level 5 quantum-resistant validator registration
///
/// Each validator registers with a Dilithium5 public key (2,592 bytes).
/// Since registration is a one-time operation stored off-chain in the
/// validator registry, the large key size is acceptable.
///
/// Security: NIST Level 5 (AES-256 equivalent post-quantum security)
/// Public key: 2,592 bytes (stored once per validator)
/// Signature: 4,627 bytes (used for registration proof)

use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey as PQPublicKey, SignedMessage};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

/// Helper module for serializing Option<[u8; 64]>
mod option_sig_64 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &Option<[u8; 64]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(arr) => {
                let vec: Vec<u8> = arr.to_vec();
                vec.serialize(serializer)
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<[u8; 64]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<Vec<u8>> = Option::deserialize(deserializer)?;
        match opt {
            Some(vec) if vec.len() == 64 => {
                let mut arr = [0u8; 64];
                arr.copy_from_slice(&vec);
                Ok(Some(arr))
            }
            Some(_) => Err(serde::de::Error::custom("Expected 64-byte signature")),
            None => Ok(None),
        }
    }
}

/// Dilithium5 public key size (2,592 bytes)
pub const DILITHIUM5_PUBLIC_KEY_BYTES: usize = 2592;

/// Dilithium5 signature size (4,627 bytes)
pub const DILITHIUM5_SIGNATURE_BYTES: usize = 4627;

/// Ed25519 public key size (32 bytes) - fallback for hybrid mode
pub const ED25519_PUBLIC_KEY_BYTES: usize = 32;

/// Validator registration information with post-quantum keys
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    /// Unique validator ID (SHA3-256 hash of Dilithium5 public key)
    pub validator_id: [u8; 32],

    /// Dilithium5 post-quantum public key (NIST Level 5)
    /// This is the primary signing key for consensus operations
    #[serde(with = "serde_bytes")]
    pub dilithium5_pubkey: Vec<u8>,

    /// Ed25519 classical public key (fallback/hybrid mode)
    /// Used for backward compatibility during transition period
    pub ed25519_pubkey: [u8; 32],

    /// Human-readable validator name
    pub name: String,

    /// Validator's declared stake (in base units)
    #[serde(with = "crate::u128_serde")]
    pub stake: u128,

    /// Registration block height
    pub registration_height: u64,

    /// Registration timestamp (Unix seconds)
    pub registered_at: u64,

    /// Whether validator uses hybrid mode (both Ed25519 + Dilithium5)
    pub hybrid_mode: bool,

    /// Validator status
    pub status: ValidatorStatus,

    /// Contact endpoint (libp2p multiaddr or onion address)
    pub endpoint: Option<String>,
}

/// Validator status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidatorStatus {
    /// Pending activation (needs minimum stake)
    Pending,
    /// Active and participating in consensus
    Active,
    /// Temporarily offline (grace period)
    Inactive,
    /// Slashed due to misbehavior
    Slashed,
    /// Voluntarily exited
    Exited,
}

/// Validator registration request (signed with Dilithium5)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorRegistration {
    /// Validator information
    pub info: ValidatorInfo,

    /// Dilithium5 signature over the registration data
    /// Signs: SHA3-256(validator_id || stake || registration_height || name)
    #[serde(with = "serde_bytes")]
    pub dilithium5_signature: Vec<u8>,

    /// Optional Ed25519 signature (for hybrid mode)
    #[serde(with = "option_sig_64")]
    pub ed25519_signature: Option<[u8; 64]>,
}

impl ValidatorInfo {
    /// Create a new validator from Dilithium5 keypair
    pub fn new(
        dilithium5_pubkey: &[u8],
        ed25519_pubkey: [u8; 32],
        name: String,
        stake: u128,
        registration_height: u64,
        hybrid_mode: bool,
        endpoint: Option<String>,
    ) -> Self {
        // Compute validator ID as SHA3-256(dilithium5_pubkey)
        let validator_id = Self::compute_validator_id(dilithium5_pubkey);

        let registered_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            validator_id,
            dilithium5_pubkey: dilithium5_pubkey.to_vec(),
            ed25519_pubkey,
            name,
            stake,
            registration_height,
            registered_at,
            hybrid_mode,
            status: ValidatorStatus::Pending,
            endpoint,
        }
    }

    /// Compute validator ID from Dilithium5 public key
    pub fn compute_validator_id(dilithium5_pubkey: &[u8]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(b"qnk_validator_id_v1");
        hasher.update(dilithium5_pubkey);
        hasher.finalize().into()
    }

    /// Verify the validator ID matches the public key
    pub fn verify_validator_id(&self) -> bool {
        let expected_id = Self::compute_validator_id(&self.dilithium5_pubkey);
        self.validator_id == expected_id
    }

    /// Get registration data for signing
    fn get_registration_data(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.validator_id);
        data.extend_from_slice(&self.stake.to_le_bytes());
        data.extend_from_slice(&self.registration_height.to_le_bytes());
        data.extend_from_slice(self.name.as_bytes());
        data
    }

    /// Sign registration with Dilithium5 secret key
    pub fn sign_registration(&self, secret_key: &dilithium5::SecretKey) -> Vec<u8> {
        let data = self.get_registration_data();
        dilithium5::sign(&data, secret_key).as_bytes().to_vec()
    }
}

impl ValidatorRegistration {
    /// Create a signed registration
    pub fn new(
        info: ValidatorInfo,
        dilithium5_secret: &dilithium5::SecretKey,
        ed25519_signature: Option<[u8; 64]>,
    ) -> Self {
        let dilithium5_signature = info.sign_registration(dilithium5_secret);

        Self {
            info,
            dilithium5_signature,
            ed25519_signature,
        }
    }

    /// Verify the registration signature
    pub fn verify(&self) -> Result<(), String> {
        // 1. Verify validator ID
        if !self.info.verify_validator_id() {
            return Err("Validator ID does not match public key".to_string());
        }

        // 2. Verify Dilithium5 public key size
        if self.info.dilithium5_pubkey.len() != DILITHIUM5_PUBLIC_KEY_BYTES {
            return Err(format!(
                "Invalid Dilithium5 public key size: expected {}, got {}",
                DILITHIUM5_PUBLIC_KEY_BYTES,
                self.info.dilithium5_pubkey.len()
            ));
        }

        // 3. Parse public key
        let pubkey = dilithium5::PublicKey::from_bytes(&self.info.dilithium5_pubkey)
            .map_err(|_| "Invalid Dilithium5 public key format")?;

        // 4. Parse signed message
        let signed_msg = SignedMessage::from_bytes(&self.dilithium5_signature)
            .map_err(|_| "Invalid Dilithium5 signature format")?;

        // 5. Verify signature and recover message
        let recovered_data = dilithium5::open(&signed_msg, &pubkey)
            .map_err(|_| "Dilithium5 signature verification failed")?;

        // 6. Verify recovered data matches expected
        let expected_data = self.info.get_registration_data();
        if recovered_data.as_slice() != expected_data.as_slice() {
            return Err("Registration data mismatch".to_string());
        }

        // 7. If hybrid mode, verify Ed25519 signature is present
        if self.info.hybrid_mode && self.ed25519_signature.is_none() {
            return Err("Hybrid mode requires Ed25519 signature".to_string());
        }

        Ok(())
    }

    /// Get compact identifier for logging
    pub fn short_id(&self) -> String {
        hex::encode(&self.info.validator_id[..8])
    }
}

/// Validator registry for managing registered validators
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidatorRegistry {
    /// Registered validators indexed by validator_id
    validators: std::collections::HashMap<[u8; 32], ValidatorInfo>,
    /// Total stake across all active validators
    #[serde(with = "crate::u128_serde")]
    total_stake: u128,
}

impl ValidatorRegistry {
    /// Create empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new validator
    pub fn register(&mut self, registration: ValidatorRegistration) -> Result<(), String> {
        // Verify the registration signature
        registration.verify()?;

        let validator_id = registration.info.validator_id;

        // Check if already registered
        if self.validators.contains_key(&validator_id) {
            return Err(format!(
                "Validator {} already registered",
                hex::encode(&validator_id[..8])
            ));
        }

        // Add to total stake
        self.total_stake = self.total_stake.saturating_add(registration.info.stake);

        // Store validator info
        self.validators.insert(validator_id, registration.info);

        Ok(())
    }

    /// Get validator by ID
    pub fn get(&self, validator_id: &[u8; 32]) -> Option<&ValidatorInfo> {
        self.validators.get(validator_id)
    }

    /// Get all active validators
    pub fn get_active_validators(&self) -> Vec<&ValidatorInfo> {
        self.validators
            .values()
            .filter(|v| v.status == ValidatorStatus::Active)
            .collect()
    }

    /// Update validator status
    pub fn update_status(
        &mut self,
        validator_id: &[u8; 32],
        status: ValidatorStatus,
    ) -> Result<(), String> {
        let validator = self
            .validators
            .get_mut(validator_id)
            .ok_or_else(|| "Validator not found".to_string())?;

        // Handle stake changes on status transitions
        match (validator.status, status) {
            (ValidatorStatus::Active, ValidatorStatus::Slashed) => {
                // Slashed validators lose stake from total
                self.total_stake = self.total_stake.saturating_sub(validator.stake);
            }
            (ValidatorStatus::Active, ValidatorStatus::Exited) => {
                // Exited validators remove stake from total
                self.total_stake = self.total_stake.saturating_sub(validator.stake);
            }
            (ValidatorStatus::Pending, ValidatorStatus::Active) => {
                // Newly active - stake already counted
            }
            _ => {}
        }

        validator.status = status;
        Ok(())
    }

    /// Get total active stake
    pub fn total_active_stake(&self) -> u128 {
        self.total_stake
    }

    /// Number of registered validators
    pub fn validator_count(&self) -> usize {
        self.validators.len()
    }

    /// Get all registered validators (all statuses)
    pub fn get_all_validators(&self) -> Vec<&ValidatorInfo> {
        self.validators.values().collect()
    }

    /// Check if a peer ID is a registered validator
    /// Matches against endpoint field containing the peer ID
    pub fn is_registered_peer(&self, peer_id: &str) -> bool {
        self.validators.values().any(|v| {
            v.endpoint.as_ref().map_or(false, |ep| ep.contains(peer_id))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_registration() {
        // Generate Dilithium5 keypair
        let (pubkey, secret_key) = dilithium5::keypair();
        let ed25519_pubkey = [0u8; 32]; // Placeholder

        // Create validator info
        let info = ValidatorInfo::new(
            pubkey.as_bytes(),
            ed25519_pubkey,
            "Test Validator".to_string(),
            1_000_000_000_000_000_000, // 1 QUG stake
            100,                       // Registration height
            false,                     // Not hybrid mode
            Some("/ip4/127.0.0.1/tcp/9001".to_string()),
        );

        // Verify validator ID
        assert!(info.verify_validator_id());

        // Create signed registration
        let registration = ValidatorRegistration::new(info, &secret_key, None);

        // Verify registration
        assert!(registration.verify().is_ok());
    }

    #[test]
    fn test_validator_registry() {
        let (pubkey, secret_key) = dilithium5::keypair();
        let ed25519_pubkey = [0u8; 32];

        let info = ValidatorInfo::new(
            pubkey.as_bytes(),
            ed25519_pubkey,
            "Validator 1".to_string(),
            1_000_000_000_000_000_000,
            100,
            false,
            None,
        );

        let registration = ValidatorRegistration::new(info, &secret_key, None);

        let mut registry = ValidatorRegistry::new();
        assert!(registry.register(registration).is_ok());
        assert_eq!(registry.validator_count(), 1);

        // Cannot register same validator twice
        let info2 = ValidatorInfo::new(
            pubkey.as_bytes(),
            ed25519_pubkey,
            "Validator 1 Duplicate".to_string(),
            1_000_000_000_000_000_000,
            101,
            false,
            None,
        );
        let registration2 = ValidatorRegistration::new(info2, &secret_key, None);
        assert!(registry.register(registration2).is_err());
    }
}
