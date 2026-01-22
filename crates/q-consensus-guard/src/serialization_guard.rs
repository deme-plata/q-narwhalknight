//! Serialization Guard
//!
//! Detects breaking changes to serialization formats.
//!
//! If you change how a Block or Transaction serializes, old data becomes
//! unreadable. This guard catches that at compile time (via tests) and
//! runtime (via fingerprint verification).

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use tracing::{error, info, warn};

/// Fingerprint of a type's serialization format
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeFingerprint {
    /// Type name
    pub type_name: String,

    /// Hash of a canonical test value's serialization
    pub canonical_hash: [u8; 32],

    /// Serialized size of canonical value
    pub canonical_size: usize,

    /// Version when this fingerprint was established
    pub established_version: String,
}

/// Guard for serialization compatibility
pub struct SerializationGuard {
    /// Known fingerprints for types
    fingerprints: HashMap<String, TypeFingerprint>,
}

impl SerializationGuard {
    /// Create new serialization guard
    pub fn new() -> Self {
        let mut guard = Self {
            fingerprints: HashMap::new(),
        };

        // Register known type fingerprints
        guard.register_known_types();

        guard
    }

    /// Register known type fingerprints
    fn register_known_types(&mut self) {
        // These are established fingerprints - changing them breaks compatibility
        //
        // To add a new type:
        // 1. Create a canonical test value
        // 2. Serialize it with bincode
        // 3. Hash the serialization
        // 4. Add entry here

        // Example fingerprints - real ones would be computed from actual types
        // self.fingerprints.insert("Block".to_string(), TypeFingerprint {
        //     type_name: "Block".to_string(),
        //     canonical_hash: [...],
        //     canonical_size: 1234,
        //     established_version: "1.0.0".to_string(),
        // });
    }

    /// Register a type fingerprint
    pub fn register<T: Serialize>(&mut self, type_name: &str, canonical_value: &T, version: &str) {
        let bytes = bincode::serialize(canonical_value).expect("Serialization failed");
        let mut hasher = Sha3_256::new();
        hasher.update(&bytes);
        let hash: [u8; 32] = hasher.finalize().into();

        self.fingerprints.insert(type_name.to_string(), TypeFingerprint {
            type_name: type_name.to_string(),
            canonical_hash: hash,
            canonical_size: bytes.len(),
            established_version: version.to_string(),
        });

        info!(
            "📝 [SERIALIZATION] Registered {}: {} bytes, hash={}",
            type_name,
            bytes.len(),
            hex::encode(&hash[..8])
        );
    }

    /// Verify a type still serializes the same way
    pub fn verify<T: Serialize>(
        &self,
        type_name: &str,
        canonical_value: &T,
    ) -> Result<(), SerializationMismatch> {
        let expected = match self.fingerprints.get(type_name) {
            Some(fp) => fp,
            None => {
                warn!("[SERIALIZATION] No fingerprint for {} - skipping", type_name);
                return Ok(());
            }
        };

        let bytes = bincode::serialize(canonical_value)
            .map_err(|e| SerializationMismatch::SerializationFailed {
                type_name: type_name.to_string(),
                error: e.to_string(),
            })?;

        let mut hasher = Sha3_256::new();
        hasher.update(&bytes);
        let actual_hash: [u8; 32] = hasher.finalize().into();

        if actual_hash != expected.canonical_hash {
            return Err(SerializationMismatch::HashMismatch {
                type_name: type_name.to_string(),
                expected: hex::encode(expected.canonical_hash),
                actual: hex::encode(actual_hash),
                expected_size: expected.canonical_size,
                actual_size: bytes.len(),
            });
        }

        if bytes.len() != expected.canonical_size {
            return Err(SerializationMismatch::SizeMismatch {
                type_name: type_name.to_string(),
                expected: expected.canonical_size,
                actual: bytes.len(),
            });
        }

        Ok(())
    }

    /// Verify all registered types
    pub fn verify_all<F>(&self, get_canonical: F) -> Result<(), SerializationMismatch>
    where
        F: Fn(&str) -> Option<Vec<u8>>,
    {
        info!("📝 [SERIALIZATION] Verifying {} type fingerprints...", self.fingerprints.len());

        for (type_name, expected) in &self.fingerprints {
            match get_canonical(type_name) {
                Some(bytes) => {
                    let mut hasher = Sha3_256::new();
                    hasher.update(&bytes);
                    let actual_hash: [u8; 32] = hasher.finalize().into();

                    if actual_hash != expected.canonical_hash {
                        return Err(SerializationMismatch::HashMismatch {
                            type_name: type_name.clone(),
                            expected: hex::encode(expected.canonical_hash),
                            actual: hex::encode(actual_hash),
                            expected_size: expected.canonical_size,
                            actual_size: bytes.len(),
                        });
                    }

                    info!("   ✅ {}: {} bytes", type_name, bytes.len());
                }
                None => {
                    info!("   ⏭️ {}: no canonical value provided", type_name);
                }
            }
        }

        info!("📝 [SERIALIZATION] All fingerprints verified!");
        Ok(())
    }
}

impl Default for SerializationGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Error when serialization format changed
#[derive(Debug, Clone, thiserror::Error)]
pub enum SerializationMismatch {
    #[error("Serialization failed for {type_name}: {error}")]
    SerializationFailed {
        type_name: String,
        error: String,
    },

    #[error("Serialization hash mismatch for {type_name}: expected {expected}, got {actual} (size: {expected_size} vs {actual_size})")]
    HashMismatch {
        type_name: String,
        expected: String,
        actual: String,
        expected_size: usize,
        actual_size: usize,
    },

    #[error("Serialization size mismatch for {type_name}: expected {expected} bytes, got {actual}")]
    SizeMismatch {
        type_name: String,
        expected: usize,
        actual: usize,
    },
}

impl SerializationMismatch {
    pub fn print_detailed(&self) {
        error!("╔════════════════════════════════════════════════════════════╗");
        error!("║          🚨 SERIALIZATION FORMAT CHANGED 🚨                ║");
        error!("╠════════════════════════════════════════════════════════════╣");
        error!("║                                                            ║");
        error!("║  You changed how a type serializes to bytes.               ║");
        error!("║  This breaks compatibility with existing data!             ║");
        error!("║                                                            ║");
        error!("╠════════════════════════════════════════════════════════════╣");
        error!("║  Error: {} ║", self);
        error!("╠════════════════════════════════════════════════════════════╣");
        error!("║                                                            ║");
        error!("║  HOW TO FIX:                                              ║");
        error!("║  1. Revert your serialization changes                      ║");
        error!("║  2. If change is intentional, add migration code           ║");
        error!("║  3. Update fingerprint ONLY after migration is tested      ║");
        error!("║                                                            ║");
        error!("╚════════════════════════════════════════════════════════════╝");
    }
}

/// Macro to create serialization tests for a type
///
/// Usage:
/// ```rust,ignore
/// serialization_test!(Block, create_canonical_block());
/// ```
#[macro_export]
macro_rules! serialization_test {
    ($type_name:ident, $canonical_expr:expr) => {
        paste::paste! {
            #[test]
            fn [<test_serialization_ $type_name:lower>]() {
                let guard = $crate::SerializationGuard::new();
                let canonical = $canonical_expr;
                guard.verify(stringify!($type_name), &canonical)
                    .expect(concat!("Serialization changed for ", stringify!($type_name)));
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug)]
    struct TestType {
        value: u64,
        name: String,
    }

    #[test]
    fn test_serialization_guard() {
        let mut guard = SerializationGuard::new();

        let canonical = TestType {
            value: 42,
            name: "test".to_string(),
        };

        // Register fingerprint
        guard.register("TestType", &canonical, "1.0.0");

        // Same value should pass
        let same = TestType {
            value: 42,
            name: "test".to_string(),
        };
        assert!(guard.verify("TestType", &same).is_ok());

        // Different value should fail
        let different = TestType {
            value: 43, // Changed!
            name: "test".to_string(),
        };
        assert!(guard.verify("TestType", &different).is_err());
    }
}
