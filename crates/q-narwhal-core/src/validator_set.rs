/// Validator Set Management for Byzantine Fault Tolerance
///
/// This module implements the validator registry and Byzantine quorum calculations
/// required for Narwhal's reliable broadcast and DAG-Knight consensus.
///
/// Byzantine Quorum: 2f+1 where f = (n-1)/3
/// - For n=4: f=1, quorum=3 (tolerates 1 Byzantine node)
/// - For n=7: f=2, quorum=5 (tolerates 2 Byzantine nodes)
/// - For n=10: f=3, quorum=7 (tolerates 3 Byzantine nodes)
/// - For n=100: f=33, quorum=67 (tolerates 33 Byzantine nodes)

use anyhow::{anyhow, Result};
use q_types::{NodeId, PublicKey};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Validator information
#[derive(Clone, Debug)]
pub struct ValidatorInfo {
    /// Node ID (hash of public key)
    pub node_id: NodeId,
    /// Ed25519 public key for signature verification (32 bytes)
    pub public_key: PublicKey,
    /// Validator stake (for Phase 1+ weighted voting)
    pub stake: u64,
    /// Whether this validator is currently active
    pub active: bool,
}

// Custom serialization for ValidatorInfo (PublicKey doesn't implement Serialize)
impl Serialize for ValidatorInfo {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("ValidatorInfo", 4)?;
        state.serialize_field("node_id", &self.node_id)?;
        state.serialize_field("public_key", self.public_key.as_bytes())?;
        state.serialize_field("stake", &self.stake)?;
        state.serialize_field("active", &self.active)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ValidatorInfo {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            NodeId,
            PublicKey,
            Stake,
            Active,
        }

        struct ValidatorInfoVisitor;

        impl<'de> Visitor<'de> for ValidatorInfoVisitor {
            type Value = ValidatorInfo;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct ValidatorInfo")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ValidatorInfo, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut node_id = None;
                let mut public_key_bytes: Option<Vec<u8>> = None;
                let mut stake = None;
                let mut active = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::NodeId => {
                            if node_id.is_some() {
                                return Err(de::Error::duplicate_field("node_id"));
                            }
                            node_id = Some(map.next_value()?);
                        }
                        Field::PublicKey => {
                            if public_key_bytes.is_some() {
                                return Err(de::Error::duplicate_field("public_key"));
                            }
                            public_key_bytes = Some(map.next_value()?);
                        }
                        Field::Stake => {
                            if stake.is_some() {
                                return Err(de::Error::duplicate_field("stake"));
                            }
                            stake = Some(map.next_value()?);
                        }
                        Field::Active => {
                            if active.is_some() {
                                return Err(de::Error::duplicate_field("active"));
                            }
                            active = Some(map.next_value()?);
                        }
                    }
                }

                let node_id = node_id.ok_or_else(|| de::Error::missing_field("node_id"))?;
                let public_key_bytes = public_key_bytes.ok_or_else(|| de::Error::missing_field("public_key"))?;
                let stake = stake.ok_or_else(|| de::Error::missing_field("stake"))?;
                let active = active.ok_or_else(|| de::Error::missing_field("active"))?;

                // Deserialize PublicKey from bytes
                let public_key = PublicKey::from_bytes(&public_key_bytes.as_slice().try_into().map_err(|_| {
                    de::Error::custom("public_key must be exactly 32 bytes")
                })?)
                .map_err(de::Error::custom)?;

                Ok(ValidatorInfo {
                    node_id,
                    public_key,
                    stake,
                    active,
                })
            }
        }

        const FIELDS: &[&str] = &["node_id", "public_key", "stake", "active"];
        deserializer.deserialize_struct("ValidatorInfo", FIELDS, ValidatorInfoVisitor)
    }
}

/// Validator set with Byzantine quorum management
#[derive(Clone, Debug)]
pub struct ValidatorSet {
    /// Validators indexed by NodeId
    validators: HashMap<NodeId, ValidatorInfo>,
    /// Total number of validators
    n: usize,
    /// Maximum number of Byzantine faults tolerated: f = (n-1)/3
    f: usize,
    /// Byzantine quorum threshold: 2f+1
    quorum_threshold: usize,
}

impl ValidatorSet {
    /// Create new validator set from a list of validators
    ///
    /// # Byzantine Fault Tolerance
    /// - n validators total
    /// - f = (n-1)/3 Byzantine faults tolerated
    /// - Quorum = 2f+1 signatures required for certificate
    ///
    /// # Safety Requirements
    /// - Requires at least n=4 validators (tolerates f=1 Byzantine node)
    /// - For production: n≥7 recommended (tolerates f=2)
    pub fn new(validators: Vec<ValidatorInfo>) -> Result<Self> {
        let n = validators.len();

        // Safety: Require minimum 4 validators for meaningful BFT
        if n < 4 {
            return Err(anyhow!(
                "Validator set too small: {} validators (minimum 4 required for BFT)",
                n
            ));
        }

        // Calculate Byzantine parameters
        let f = (n - 1) / 3;
        let quorum_threshold = 2 * f + 1;

        // Build validator map
        let mut validator_map = HashMap::new();
        for validator in validators {
            if validator_map.insert(validator.node_id, validator.clone()).is_some() {
                return Err(anyhow!(
                    "Duplicate validator: {}",
                    hex::encode(validator.node_id)
                ));
            }
        }

        Ok(Self {
            validators: validator_map,
            n,
            f,
            quorum_threshold,
        })
    }

    /// Get total number of validators
    pub fn total_validators(&self) -> usize {
        self.n
    }

    /// Get maximum Byzantine faults tolerated (f)
    pub fn max_byzantine_faults(&self) -> usize {
        self.f
    }

    /// Get Byzantine quorum threshold (2f+1)
    ///
    /// This is the minimum number of signatures required to form a valid certificate.
    /// With 2f+1 signatures, we guarantee at least f+1 honest nodes participated,
    /// which ensures Byzantine agreement safety.
    pub fn quorum_threshold(&self) -> usize {
        self.quorum_threshold
    }

    /// Check if a node is in the validator set
    pub fn contains(&self, node_id: &NodeId) -> bool {
        self.validators.contains_key(node_id)
    }

    /// Get validator information
    pub fn get_validator(&self, node_id: &NodeId) -> Option<&ValidatorInfo> {
        self.validators.get(node_id)
    }

    /// Get public key for signature verification
    pub fn get_public_key(&self, node_id: &NodeId) -> Option<&PublicKey> {
        self.validators.get(node_id).map(|v| &v.public_key)
    }

    /// Check if validator is active
    pub fn is_active(&self, node_id: &NodeId) -> bool {
        self.validators
            .get(node_id)
            .map(|v| v.active)
            .unwrap_or(false)
    }

    /// Get all validator node IDs
    pub fn validator_ids(&self) -> Vec<NodeId> {
        self.validators.keys().copied().collect()
    }

    /// Get all active validators
    pub fn active_validators(&self) -> Vec<&ValidatorInfo> {
        self.validators
            .values()
            .filter(|v| v.active)
            .collect()
    }

    /// Verify that a set of signatures meets the Byzantine quorum
    ///
    /// Returns true if |signatures| >= 2f+1
    pub fn has_quorum(&self, signature_count: usize) -> bool {
        signature_count >= self.quorum_threshold
    }

    /// Verify that all signers are valid validators
    pub fn verify_signers(&self, signer_ids: &[NodeId]) -> Result<()> {
        for node_id in signer_ids {
            if !self.contains(node_id) {
                return Err(anyhow!(
                    "Invalid signer: {} not in validator set",
                    hex::encode(node_id)
                ));
            }
        }
        Ok(())
    }

    /// Calculate stake-weighted quorum (for Phase 1+)
    ///
    /// In Phase 0, all validators have equal weight (stake=1)
    /// In Phase 1+, validators may have different stakes
    pub fn stake_weighted_quorum(&self, total_stake: u64) -> u64 {
        // Byzantine quorum requires 2f+1 of total stake
        // For uniform stake: (2f+1)/n * total_stake
        // For weighted: 2/3 * total_stake (standard BFT threshold)
        (total_stake * 2) / 3 + 1
    }

    /// Get total stake of all validators
    pub fn total_stake(&self) -> u64 {
        self.validators.values().map(|v| v.stake).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::{rngs::OsRng, TryRngCore as _};  // TryRngCore for rand 0.9

    fn create_test_validator(stake: u64) -> ValidatorInfo {
        // ed25519-dalek v2.x: SigningKey::from_bytes or ::from_keypair_bytes
        let mut secret_bytes = [0u8; 32];
        OsRng.try_fill_bytes(&mut secret_bytes).unwrap();
        let signing_key = SigningKey::from_bytes(&secret_bytes);
        let public_key = signing_key.verifying_key();

        // NodeId = hash of public key
        let node_id = {
            use sha3::{Digest, Sha3_256};
            let mut hasher = Sha3_256::new();
            hasher.update(public_key.as_bytes());
            hasher.finalize().into()
        };

        ValidatorInfo {
            node_id,
            public_key,
            stake,
            active: true,
        }
    }

    #[test]
    fn test_validator_set_creation() {
        let validators = vec![
            create_test_validator(1),
            create_test_validator(1),
            create_test_validator(1),
            create_test_validator(1),
        ];

        let validator_set = ValidatorSet::new(validators).unwrap();

        assert_eq!(validator_set.total_validators(), 4);
        assert_eq!(validator_set.max_byzantine_faults(), 1); // f = (4-1)/3 = 1
        assert_eq!(validator_set.quorum_threshold(), 3); // 2f+1 = 3
    }

    #[test]
    fn test_byzantine_thresholds() {
        // Test various validator set sizes
        let test_cases = vec![
            (4, 1, 3),   // n=4: f=1, quorum=3
            (7, 2, 5),   // n=7: f=2, quorum=5
            (10, 3, 7),  // n=10: f=3, quorum=7
            (100, 33, 67), // n=100: f=33, quorum=67
        ];

        for (n, expected_f, expected_quorum) in test_cases {
            let validators: Vec<_> = (0..n).map(|_| create_test_validator(1)).collect();
            let validator_set = ValidatorSet::new(validators).unwrap();

            assert_eq!(
                validator_set.max_byzantine_faults(),
                expected_f,
                "Failed for n={}",
                n
            );
            assert_eq!(
                validator_set.quorum_threshold(),
                expected_quorum,
                "Failed for n={}",
                n
            );
        }
    }

    #[test]
    fn test_quorum_verification() {
        let validators = vec![
            create_test_validator(1),
            create_test_validator(1),
            create_test_validator(1),
            create_test_validator(1),
            create_test_validator(1),
            create_test_validator(1),
            create_test_validator(1),
        ];

        let validator_set = ValidatorSet::new(validators).unwrap();
        // n=7: f=2, quorum=5

        assert!(!validator_set.has_quorum(4)); // Not enough
        assert!(validator_set.has_quorum(5)); // Exactly quorum
        assert!(validator_set.has_quorum(6)); // More than quorum
        assert!(validator_set.has_quorum(7)); // All validators
    }

    #[test]
    fn test_minimum_validators() {
        // Too few validators should fail
        let validators = vec![
            create_test_validator(1),
            create_test_validator(1),
            create_test_validator(1),
        ];

        let result = ValidatorSet::new(validators);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("minimum 4 required"));
    }

    #[test]
    fn test_validator_lookup() {
        let validator1 = create_test_validator(1);
        let validator2 = create_test_validator(1);
        let validator3 = create_test_validator(1);
        let validator4 = create_test_validator(1);

        let node_id1 = validator1.node_id;
        let public_key1 = validator1.public_key;

        let validators = vec![validator1, validator2, validator3, validator4];
        let validator_set = ValidatorSet::new(validators).unwrap();

        // Test contains
        assert!(validator_set.contains(&node_id1));

        // Test get_public_key
        let retrieved_key = validator_set.get_public_key(&node_id1).unwrap();
        assert_eq!(retrieved_key.as_bytes(), public_key1.as_bytes());

        // Test non-existent validator
        let fake_id = [0u8; 32];
        assert!(!validator_set.contains(&fake_id));
        assert!(validator_set.get_public_key(&fake_id).is_none());
    }

    #[test]
    fn test_signer_verification() {
        let validator1 = create_test_validator(1);
        let validator2 = create_test_validator(1);
        let validator3 = create_test_validator(1);
        let validator4 = create_test_validator(1);

        let node_id1 = validator1.node_id;
        let node_id2 = validator2.node_id;

        let validators = vec![validator1, validator2, validator3, validator4];
        let validator_set = ValidatorSet::new(validators).unwrap();

        // Valid signers
        let valid_signers = vec![node_id1, node_id2];
        assert!(validator_set.verify_signers(&valid_signers).is_ok());

        // Invalid signer (not in set)
        let invalid_signers = vec![node_id1, [0u8; 32]];
        assert!(validator_set.verify_signers(&invalid_signers).is_err());
    }

    #[test]
    fn test_stake_weighted_quorum() {
        let validators = vec![
            create_test_validator(100),
            create_test_validator(200),
            create_test_validator(300),
            create_test_validator(400),
        ];

        let validator_set = ValidatorSet::new(validators).unwrap();

        let total_stake = validator_set.total_stake();
        assert_eq!(total_stake, 1000);

        // Stake-weighted quorum = 2/3 of total stake + 1
        let stake_quorum = validator_set.stake_weighted_quorum(total_stake);
        assert_eq!(stake_quorum, 667); // (1000 * 2)/3 + 1 = 667
    }

    #[test]
    fn test_duplicate_validator_rejection() {
        let validator1 = create_test_validator(1);
        let validator2 = create_test_validator(1);
        let validator3 = create_test_validator(1);

        // Create duplicate of validator1
        let validator1_duplicate = ValidatorInfo {
            node_id: validator1.node_id, // Same node_id
            public_key: validator1.public_key,
            stake: 1,
            active: true,
        };

        let validators = vec![validator1, validator2, validator3, validator1_duplicate];
        let result = ValidatorSet::new(validators);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Duplicate validator"));
    }
}
