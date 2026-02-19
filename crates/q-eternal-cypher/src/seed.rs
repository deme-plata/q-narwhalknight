//! Crystal seed: deterministic key derivation from a master secret.
//!
//! A [`CrystalSeed`] is a 512-bit (64-byte) master secret from which all
//! domain-specific keys are derived using Blake3 in KDF mode.  The seed
//! can optionally absorb additional quantum entropy to strengthen the
//! key material beyond what classical hardware RNG provides.
//!
//! ## Derivation Scheme
//!
//! ```text
//! CrystalSeed (512 bits)
//!   |
//!   +-- derive_key("validator/block-signing")  --> ProvenanceKey
//!   +-- derive_key("validator/vrf")            --> ProvenanceKey
//!   +-- derive_key("wallet/spending")          --> ProvenanceKey
//!   +-- derive_key("wallet/viewing")           --> ProvenanceKey
//!   ...
//! ```
//!
//! Each derivation produces a 32-byte output via
//! `blake3::derive_key(domain, seed)`.  The domain string acts as a
//! context separator, ensuring that keys derived for different purposes
//! are cryptographically independent even though they share the same
//! master seed.

use crate::provenance::{KeyMaterial, ProvenanceKey};
use serde::{Deserialize, Serialize};

/// Size of the master seed in bytes (512 bits).
pub const SEED_SIZE: usize = 64;

/// A 512-bit master seed for deterministic key derivation.
///
/// The seed is generated from the operating system's cryptographically
/// secure random number generator and can be strengthened by mixing in
/// additional entropy (e.g., from a quantum random number generator).
///
/// ## Security
///
/// - The seed MUST be stored securely (encrypted at rest, zeroized on drop).
/// - The seed MUST NOT be transmitted over the network.
/// - The seed SHOULD be backed up using a mnemonic or similar scheme.
///
/// ## Zeroization
///
/// The `Drop` implementation overwrites the seed bytes with zeros to
/// prevent residual secret material from lingering in memory.
#[derive(Clone, Serialize, Deserialize)]
pub struct CrystalSeed {
    /// The raw 512-bit seed material.
    #[serde(with = "seed_bytes")]
    seed: [u8; SEED_SIZE],
}

/// Custom serde module for fixed-size byte arrays, since serde's default
/// for `[u8; 64]` serializes as a sequence of integers which is verbose.
mod seed_bytes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 64], s: S) -> Result<S::Ok, S::Error> {
        let hex_str = hex::encode(bytes);
        hex_str.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 64], D::Error> {
        let hex_str = String::deserialize(d)?;
        let bytes = hex::decode(&hex_str).map_err(serde::de::Error::custom)?;
        let arr: [u8; 64] = bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected 64 bytes"))?;
        Ok(arr)
    }

    // hex is a transitive dependency via q-crypto-advanced
    mod hex {
        pub fn encode(bytes: &[u8]) -> String {
            bytes.iter().map(|b| format!("{:02x}", b)).collect()
        }

        pub fn decode(s: &str) -> Result<Vec<u8>, String> {
            if s.len() % 2 != 0 {
                return Err("odd-length hex string".into());
            }
            (0..s.len())
                .step_by(2)
                .map(|i| {
                    u8::from_str_radix(&s[i..i + 2], 16)
                        .map_err(|e| format!("invalid hex: {}", e))
                })
                .collect()
        }
    }
}

impl std::fmt::Debug for CrystalSeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the actual seed material in debug output
        write!(
            f,
            "CrystalSeed(fingerprint={:08x}...)",
            u32::from_be_bytes(blake3::hash(&self.seed).as_bytes()[..4].try_into().unwrap())
        )
    }
}

impl Drop for CrystalSeed {
    fn drop(&mut self) {
        // Overwrite with zeros to prevent secret material from lingering.
        // This is a best-effort measure; the compiler may optimize it away
        // in some cases.  For production, use `zeroize` crate.
        for byte in self.seed.iter_mut() {
            unsafe {
                std::ptr::write_volatile(byte, 0u8);
            }
        }
    }
}

impl CrystalSeed {
    /// Generate a new seed from the operating system's CSPRNG.
    ///
    /// # Panics
    ///
    /// Panics if the OS RNG fails (which indicates a catastrophic system
    /// failure and should not be silently ignored).
    pub fn generate() -> Self {
        let mut seed = [0u8; SEED_SIZE];
        rand::fill(&mut seed);
        Self { seed }
    }

    /// Create a seed from raw bytes.
    ///
    /// Returns `None` if the slice is not exactly [`SEED_SIZE`] bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != SEED_SIZE {
            return None;
        }
        let mut seed = [0u8; SEED_SIZE];
        seed.copy_from_slice(bytes);
        Some(Self { seed })
    }

    /// Mix additional entropy into the seed.
    ///
    /// The new seed is computed as:
    /// ```text
    /// new_seed = Blake3(old_seed || entropy || "q-eternal-cypher/quantum-mix")
    /// ```
    /// extended to 512 bits via Blake3 XOF mode.
    ///
    /// This is useful for incorporating entropy from a quantum random
    /// number generator (QRNG) or other hardware entropy source.
    pub fn add_quantum_entropy(&mut self, entropy: &[u8]) {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.seed);
        hasher.update(entropy);
        hasher.update(b"q-eternal-cypher/quantum-mix");
        let mut output = hasher.finalize_xof();
        output.fill(&mut self.seed);
    }

    /// Derive a [`ProvenanceKey`] for the given domain.
    ///
    /// The domain string should be a human-readable path that identifies
    /// the purpose of the key, e.g., `"validator/block-signing"` or
    /// `"wallet/spending"`.
    ///
    /// The derived key is always an Ed25519 key in Phase 0.  Callers can
    /// later upgrade it to post-quantum algorithms using
    /// [`ProvenanceKey::upgrade_from`].
    ///
    /// # Arguments
    ///
    /// * `domain` - A unique context string for key separation.
    ///
    /// # Returns
    ///
    /// A [`ProvenanceKey`] with `birth_height = 0` and no transition history.
    pub fn derive_key(&self, domain: &str) -> ProvenanceKey {
        // Blake3 KDF: derive_key(context, input_key_material) -> 32 bytes
        let derived = blake3::derive_key(domain, &self.seed);
        // Wrap as Ed25519 key material (the 32 bytes serve as the secret
        // from which an Ed25519 keypair would be generated; here we store
        // them as the "public key" placeholder -- actual keypair generation
        // is the caller's responsibility using ed25519-dalek).
        let material = KeyMaterial::Ed25519(derived.to_vec());
        ProvenanceKey::new(material, 0)
    }

    /// Derive raw 32-byte key material for a given domain without wrapping
    /// it in a [`ProvenanceKey`].
    ///
    /// This is useful when the caller needs the raw bytes for a non-signing
    /// purpose (e.g., symmetric encryption keys, nonce seeds).
    pub fn derive_raw(&self, domain: &str) -> [u8; 32] {
        blake3::derive_key(domain, &self.seed)
    }

    /// Return the Blake3 fingerprint of the seed (safe to log/display).
    pub fn fingerprint(&self) -> [u8; 32] {
        blake3::hash(&self.seed).into()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase::CryptoPhase;

    #[test]
    fn test_generate_produces_nonzero_seed() {
        let seed = CrystalSeed::generate();
        // Extremely unlikely to be all zeros
        assert!(seed.seed.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_from_bytes_correct_length() {
        let bytes = [0x42u8; SEED_SIZE];
        let seed = CrystalSeed::from_bytes(&bytes);
        assert!(seed.is_some());
        assert_eq!(seed.unwrap().seed, bytes);
    }

    #[test]
    fn test_from_bytes_wrong_length() {
        assert!(CrystalSeed::from_bytes(&[0u8; 32]).is_none());
        assert!(CrystalSeed::from_bytes(&[0u8; 63]).is_none());
        assert!(CrystalSeed::from_bytes(&[0u8; 65]).is_none());
        assert!(CrystalSeed::from_bytes(&[]).is_none());
    }

    #[test]
    fn test_derive_key_deterministic() {
        let seed = CrystalSeed::from_bytes(&[0xAB; SEED_SIZE]).unwrap();
        let key1 = seed.derive_key("test/signing");
        let key2 = seed.derive_key("test/signing");

        assert_eq!(key1.current.as_bytes(), key2.current.as_bytes());
    }

    #[test]
    fn test_derive_key_domain_separation() {
        let seed = CrystalSeed::from_bytes(&[0xAB; SEED_SIZE]).unwrap();
        let key_a = seed.derive_key("domain/a");
        let key_b = seed.derive_key("domain/b");

        // Different domains must produce different keys
        assert_ne!(key_a.current.as_bytes(), key_b.current.as_bytes());
    }

    #[test]
    fn test_derive_key_produces_genesis_phase() {
        let seed = CrystalSeed::generate();
        let key = seed.derive_key("test");
        assert_eq!(key.algorithm_phase, CryptoPhase::Phase0_Genesis);
        assert_eq!(key.birth_height, 0);
    }

    #[test]
    fn test_derive_raw() {
        let seed = CrystalSeed::from_bytes(&[0xCD; SEED_SIZE]).unwrap();
        let raw = seed.derive_raw("symmetric/aes-key");
        assert_eq!(raw.len(), 32);

        // Should be deterministic
        let raw2 = seed.derive_raw("symmetric/aes-key");
        assert_eq!(raw, raw2);
    }

    #[test]
    fn test_add_quantum_entropy_changes_seed() {
        let original_bytes = [0x11; SEED_SIZE];
        let mut seed = CrystalSeed::from_bytes(&original_bytes).unwrap();
        let fp_before = seed.fingerprint();

        seed.add_quantum_entropy(b"quantum noise from QRNG device");
        let fp_after = seed.fingerprint();

        assert_ne!(fp_before, fp_after);
    }

    #[test]
    fn test_add_quantum_entropy_deterministic() {
        let bytes = [0x22; SEED_SIZE];
        let mut seed1 = CrystalSeed::from_bytes(&bytes).unwrap();
        let mut seed2 = CrystalSeed::from_bytes(&bytes).unwrap();

        let entropy = b"deterministic test entropy";
        seed1.add_quantum_entropy(entropy);
        seed2.add_quantum_entropy(entropy);

        assert_eq!(seed1.fingerprint(), seed2.fingerprint());
    }

    #[test]
    fn test_fingerprint_does_not_leak_seed() {
        let seed = CrystalSeed::from_bytes(&[0xFF; SEED_SIZE]).unwrap();
        let fp = seed.fingerprint();
        // The fingerprint should not contain the raw seed bytes
        assert_ne!(&fp[..], &seed.seed[..32]);
    }

    #[test]
    fn test_debug_does_not_leak_seed() {
        let seed = CrystalSeed::from_bytes(&[0xAA; SEED_SIZE]).unwrap();
        let debug = format!("{:?}", seed);
        // Debug output should not contain the hex representation of the seed
        assert!(!debug.contains("aaaa"));
        assert!(debug.contains("CrystalSeed(fingerprint="));
    }

    #[test]
    fn test_serde_roundtrip() {
        let seed = CrystalSeed::from_bytes(&[0x55; SEED_SIZE]).unwrap();
        let json = serde_json::to_string(&seed).unwrap();
        let recovered: CrystalSeed = serde_json::from_str(&json).unwrap();
        assert_eq!(seed.seed, recovered.seed);
    }

    #[test]
    fn test_derive_multiple_keys_from_same_seed() {
        let seed = CrystalSeed::generate();
        let keys: Vec<ProvenanceKey> = (0..10)
            .map(|i| seed.derive_key(&format!("key/{}", i)))
            .collect();

        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(
                    keys[i].current.as_bytes(),
                    keys[j].current.as_bytes(),
                    "keys {} and {} collided",
                    i,
                    j
                );
            }
        }
    }
}
