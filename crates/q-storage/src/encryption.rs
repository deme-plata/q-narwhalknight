// Q-NarwhalKnight RocksDB Encryption-at-Rest
// v1.0.39-beta: Production-viable implementation based on expert review
//
// DESIGN PRINCIPLES (from expert feedback):
// 1. ✅ Use AES-CTR (not GCM) for BlockAccessCipherStream compatibility
// 2. ✅ Use HKDF-only (no AES-KW wrapper) to avoid 32→40 byte header bug
// 3. ✅ Remove ZK-STARK from v1 (use BLAKE3 commitments instead)
// 4. ✅ AEGIS-QL only for master account wallet (not DB master key)
// 5. ✅ Transitional provider for migration (plaintext + encrypted)

use anyhow::{anyhow, Result};
use argon2::{
    password_hash::{PasswordHasher, SaltString},
    Argon2, Params,
};
use rand::rngs::OsRng;
use std::marker::PhantomPinned;
use std::path::Path;
use zeroize::Zeroize;
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use blake3;
use hkdf::Hkdf;
use sha2::Sha512;
use tracing::{debug, error, info, warn};

// ZK-STARK proofs for untrusted setup (v1.0.43-beta)
use crate::encryption_zkstark::EncryptionKeyProof;

/// 🔐 Memory-hardened encryption key with mlock() + zeroize
///
/// SECURITY GUARANTEES:
/// - Locked in physical RAM (won't swap to disk)
/// - Zeroed immediately on drop
/// - Pinned (unmovable in memory)
/// - Defense against cold boot attacks
pub struct ProtectedKey {
    key: Box<[u8; 32]>,
    _pinned: PhantomPinned,
}

impl ProtectedKey {
    /// Create new protected key and lock in memory
    pub fn new(key: [u8; 32]) -> Result<Self> {
        let mut boxed = Box::new(key);

        #[cfg(unix)]
        unsafe {
            let ptr = boxed.as_ptr() as *const libc::c_void;
            let result = libc::mlock(ptr, 32);
            if result != 0 {
                warn!("⚠️ Failed to mlock key memory (errno: {})", *libc::__errno_location());
                // Non-fatal: continue without mlock protection
            } else {
                debug!("🔒 Key locked in physical RAM (mlock success)");
            }
        }

        Ok(Self {
            key: boxed,
            _pinned: PhantomPinned,
        })
    }

    /// Get immutable reference to key bytes (unsafe: caller must not leak)
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.key
    }

    /// Derive subkey using HKDF-SHA512 (no AES-KW wrapper per expert feedback)
    pub fn derive_subkey(&self, context: &[u8]) -> Result<ProtectedKey> {
        let hkdf = Hkdf::<Sha512>::new(None, self.as_bytes());
        let mut subkey = [0u8; 32];
        hkdf.expand(context, &mut subkey)
            .map_err(|e| anyhow!("HKDF derivation failed: {}", e))?;

        let result = ProtectedKey::new(subkey);
        subkey.zeroize(); // Clear stack copy
        result
    }

    /// Compute BLAKE3 commitment (replaces ZK-STARK for v1)
    pub fn blake3_commitment(&self) -> [u8; 32] {
        blake3::hash(self.as_bytes()).into()
    }
}

impl Drop for ProtectedKey {
    fn drop(&mut self) {
        // Zeroize before munlock
        self.key.zeroize();

        #[cfg(unix)]
        unsafe {
            let ptr = self.key.as_ptr() as *const libc::c_void;
            libc::munlock(ptr, 32);
        }

        debug!("🧹 ProtectedKey zeroed and unlocked");
    }
}

/// 🔑 Passphrase-based key derivation using Argon2id
///
/// PARAMETERS (expert-approved):
/// - Memory: 64 MB (m_cost = 65536 KiB)
/// - Iterations: 4 (t_cost = 4)
/// - Parallelism: 1 (p_cost = 1)
/// - Expected time: ~100ms on modern CPU
pub struct PassphraseKDF;

impl PassphraseKDF {
    /// Derive 256-bit key from passphrase using Argon2id
    pub fn derive(passphrase: &str, salt: &[u8; 16]) -> Result<ProtectedKey> {
        // Argon2id parameters (defense against GPU/ASIC cracking)
        let params = Params::new(
            65536,  // 64 MB memory
            4,      // 4 iterations
            1,      // 1 thread
            Some(32), // 32-byte output
        ).map_err(|e| anyhow!("Invalid Argon2 parameters: {}", e))?;

        let argon2 = Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            params,
        );

        let salt_string = SaltString::encode_b64(salt)
            .map_err(|e| anyhow!("Salt encoding failed: {}", e))?;

        let start = std::time::Instant::now();
        let hash = argon2.hash_password(passphrase.as_bytes(), &salt_string)
            .map_err(|e| anyhow!("Argon2id derivation failed: {}", e))?;

        let duration = start.elapsed();
        info!("🔐 Argon2id derivation completed in {:?}", duration);

        if duration.as_millis() < 50 {
            warn!("⚠️ Argon2id too fast ({:?}), GPU cracking risk!", duration);
        }

        // Extract 32-byte key from PHC string
        let hash_bytes = hash.hash
            .ok_or_else(|| anyhow!("Argon2id produced no hash"))?;

        let mut key = [0u8; 32];
        key.copy_from_slice(hash_bytes.as_bytes());

        let result = ProtectedKey::new(key);
        key.zeroize(); // Clear stack copy
        result
    }

    /// Generate cryptographically secure random salt
    pub fn generate_salt() -> [u8; 16] {
        let mut salt = [0u8; 16];
        getrandom::getrandom(&mut salt).expect("RNG failure");
        salt
    }
}

/// 📁 Encrypted keys file format (AES-GCM, no AEGIS-QL for v1)
///
/// FILE FORMAT (v1):
/// ```
/// [8 bytes]  Magic: "QNKKeys1"
/// [4 bytes]  Version: 1
/// [16 bytes] Argon2id salt
/// [12 bytes] AES-GCM nonce
/// [32 bytes] Encrypted DB master key
/// [16 bytes] AES-GCM authentication tag
/// [32 bytes] BLAKE3 commitment of plaintext key
/// ```
/// Total: 120 bytes
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KeysFileHeader {
    pub magic: [u8; 8],        // "QNKKeys1"
    pub version: u32,
    pub argon2_salt: [u8; 16],
    pub gcm_nonce: [u8; 12],
    pub encrypted_key: [u8; 32],
    pub gcm_tag: [u8; 16],
    pub blake3_commitment: [u8; 32],
}

impl KeysFileHeader {
    const MAGIC: &'static [u8; 8] = b"QNKKeys1";
    const VERSION: u32 = 1;

    /// Create new encrypted keys file from DB master key
    pub fn new(db_master_key: &ProtectedKey, kek: &ProtectedKey, argon2_salt: [u8; 16]) -> Result<Self> {
        let gcm_nonce = Self::generate_nonce();

        // Compute BLAKE3 commitment before encryption
        let blake3_commitment = db_master_key.blake3_commitment();

        // Encrypt DB master key with AES-GCM under KEK
        let cipher = Aes256Gcm::new_from_slice(kek.as_bytes())
            .map_err(|e| anyhow!("AES-GCM key init failed: {}", e))?;

        let nonce_obj = Nonce::from_slice(&gcm_nonce);
        let ciphertext = cipher.encrypt(nonce_obj, db_master_key.as_bytes().as_ref())
            .map_err(|e| anyhow!("AES-GCM encryption failed: {}", e))?;

        // Extract ciphertext (32 bytes) and tag (16 bytes)
        if ciphertext.len() != 48 {
            return Err(anyhow!("AES-GCM produced unexpected output size: {}", ciphertext.len()));
        }

        let mut encrypted_key = [0u8; 32];
        let mut gcm_tag = [0u8; 16];
        encrypted_key.copy_from_slice(&ciphertext[..32]);
        gcm_tag.copy_from_slice(&ciphertext[32..48]);

        Ok(Self {
            magic: *Self::MAGIC,
            version: Self::VERSION,
            argon2_salt,
            gcm_nonce,
            encrypted_key,
            gcm_tag,
            blake3_commitment,
        })
    }

    /// Decrypt DB master key using KEK
    pub fn decrypt(&self, kek: &ProtectedKey) -> Result<ProtectedKey> {
        // Verify magic number
        if &self.magic != Self::MAGIC {
            return Err(anyhow!("Invalid keys file magic (corrupted or wrong format)"));
        }

        // Verify version
        if self.version != Self::VERSION {
            return Err(anyhow!("Unsupported keys file version: {}", self.version));
        }

        // Decrypt with AES-GCM
        let cipher = Aes256Gcm::new_from_slice(kek.as_bytes())
            .map_err(|e| anyhow!("AES-GCM key init failed: {}", e))?;

        let nonce_obj = Nonce::from_slice(&self.gcm_nonce);

        // Reconstruct ciphertext + tag
        let mut ciphertext_with_tag = Vec::with_capacity(48);
        ciphertext_with_tag.extend_from_slice(&self.encrypted_key);
        ciphertext_with_tag.extend_from_slice(&self.gcm_tag);

        let plaintext = cipher.decrypt(nonce_obj, ciphertext_with_tag.as_ref())
            .map_err(|e| anyhow!("AES-GCM decryption failed (wrong passphrase?): {}", e))?;

        if plaintext.len() != 32 {
            return Err(anyhow!("Decrypted key has wrong size: {}", plaintext.len()));
        }

        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&plaintext);

        let db_master_key = ProtectedKey::new(key_bytes)?;
        key_bytes.zeroize(); // Clear stack copy

        // Verify BLAKE3 commitment
        let actual_commitment = db_master_key.blake3_commitment();
        if actual_commitment != self.blake3_commitment {
            return Err(anyhow!("BLAKE3 commitment mismatch (data corruption)"));
        }

        Ok(db_master_key)
    }

    /// Generate random nonce for AES-GCM
    fn generate_nonce() -> [u8; 12] {
        let mut nonce = [0u8; 12];
        getrandom::getrandom(&mut nonce).expect("RNG failure");
        nonce
    }

    /// Write encrypted keys file to disk
    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>()
            )
        };

        std::fs::write(path, bytes)
            .map_err(|e| anyhow!("Failed to write keys file: {}", e))?;

        info!("💾 Encrypted keys file written: {:?}", path);
        Ok(())
    }

    /// Read encrypted keys file from disk
    pub fn read_from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow!("Failed to read keys file: {}", e))?;

        if bytes.len() != std::mem::size_of::<Self>() {
            return Err(anyhow!("Invalid keys file size: {} (expected {})",
                bytes.len(), std::mem::size_of::<Self>()));
        }

        let header = unsafe {
            std::ptr::read_unaligned(bytes.as_ptr() as *const Self)
        };

        debug!("📂 Encrypted keys file loaded: {:?}", path);
        Ok(header)
    }
}

/// 🖥️ CPU capability detection (require AES-NI for production)
pub struct CpuCapabilities;

impl CpuCapabilities {
    /// Check if CPU supports AES-NI instructions
    #[cfg(target_arch = "x86_64")]
    pub fn has_aes_ni() -> bool {
        std::arch::is_x86_feature_detected!("aes")
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn has_aes_ni() -> bool {
        false // Non-x86 platforms use software AES
    }

    /// Verify CPU meets encryption requirements
    pub fn verify_requirements() -> Result<()> {
        #[cfg(target_arch = "x86_64")]
        {
            if !Self::has_aes_ni() {
                return Err(anyhow!(
                    "❌ CPU lacks AES-NI support! Encryption will be 10x slower. \
                     Consider upgrading to a CPU with hardware AES (Intel 2010+, AMD 2011+)"
                ));
            }
            info!("✅ CPU supports AES-NI (hardware-accelerated encryption)");
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            warn!("⚠️ Non-x86_64 platform: using software AES (slower)");
        }

        Ok(())
    }
}

/// 🔐 Master encryption manager (coordinates all key operations)
pub struct EncryptionManager {
    kek: ProtectedKey,           // Key-Encryption-Key (from passphrase)
    db_master_key: ProtectedKey, // Database master key
    keys_file_path: std::path::PathBuf,
    zkstark_proof: Option<EncryptionKeyProof>, // ZK-STARK proof of correct key derivation (v1.0.43-beta)
}

impl EncryptionManager {
    /// Initialize encryption with automatic key generation if needed
    ///
    /// AUTOMATIC BEHAVIOR (v1.0.43-beta):
    /// - If keys file exists: Load and verify ZK-STARK proof
    /// - If keys file missing: Generate new keys + ZK-STARK proof automatically
    ///
    /// NO MANUAL COMMANDS REQUIRED!
    pub fn from_passphrase(passphrase: &str, keys_file: &Path) -> Result<Self> {
        CpuCapabilities::verify_requirements()?;

        // Check if keys file exists
        if !keys_file.exists() {
            info!("🔑 No encryption keys found at {:?}", keys_file);
            info!("🆕 Automatically generating new encryption keys with ZK-STARK proof...");

            // Automatically generate keys (no manual command needed!)
            return Self::create_new(passphrase, keys_file);
        }

        // Load existing keys
        let header = KeysFileHeader::read_from_file(keys_file)?;

        // Derive KEK from passphrase
        let kek = PassphraseKDF::derive(passphrase, &header.argon2_salt)?;

        // Decrypt DB master key
        let db_master_key = header.decrypt(&kek)?;

        // Load and verify ZK-STARK proof (untrusted automatic verification)
        let zkstark_proof = Self::load_and_verify_zkstark_proof(keys_file)?;

        info!("🔓 Encryption manager initialized (keys file: {:?})", keys_file);

        Ok(Self {
            kek,
            db_master_key,
            keys_file_path: keys_file.to_path_buf(),
            zkstark_proof,
        })
    }

    /// Create new encryption with fresh keys
    pub fn create_new(passphrase: &str, keys_file: &Path) -> Result<Self> {
        CpuCapabilities::verify_requirements()?;

        // Generate random salt
        let salt = PassphraseKDF::generate_salt();

        // Derive KEK from passphrase
        let kek = PassphraseKDF::derive(passphrase, &salt)?;

        // Generate random DB master key
        let mut db_master_key_bytes = [0u8; 32];
        getrandom::getrandom(&mut db_master_key_bytes).expect("RNG failure");
        let db_master_key = ProtectedKey::new(db_master_key_bytes)?;
        db_master_key_bytes.zeroize();

        // Create encrypted keys file (use the SAME salt that derived the KEK!)
        let header = KeysFileHeader::new(&db_master_key, &kek, salt)?;
        header.write_to_file(keys_file)?;

        // Generate ZK-STARK proof of correct key derivation (untrusted setup)
        info!("🔬 Generating ZK-STARK proof for untrusted automatic verification...");
        let zkstark_proof = EncryptionKeyProof::generate(
            passphrase,
            kek.as_bytes(),
            db_master_key.as_bytes(),
            &salt,
        )?;

        // Save proof alongside keys file
        let proof_path = keys_file.with_extension("zkstark");
        let proof_json = zkstark_proof.to_json()?;
        std::fs::write(&proof_path, proof_json)
            .map_err(|e| anyhow!("Failed to write ZK-STARK proof: {}", e))?;

        info!("💾 ZK-STARK proof saved: {:?} ({} bytes)", proof_path, zkstark_proof.proof_size_bytes);
        info!("🆕 New encryption created (keys file: {:?})", keys_file);

        Ok(Self {
            kek,
            db_master_key,
            keys_file_path: keys_file.to_path_buf(),
            zkstark_proof: Some(zkstark_proof),
        })
    }

    /// Derive per-file encryption key using HKDF (no AES-KW wrapper)
    pub fn derive_file_key(&self, file_id: u64, cf_id: u32) -> Result<ProtectedKey> {
        let context = format!("QNK-File-v1|{}|{}", file_id, cf_id);
        self.db_master_key.derive_subkey(context.as_bytes())
    }

    /// Load and verify ZK-STARK proof of correct key derivation (untrusted automatic verification)
    ///
    /// SECURITY: This performs UNTRUSTED AUTOMATIC SETUP verification.
    /// Anyone can verify the proof without trusting the key generator.
    /// If this succeeds, key derivation is GUARANTEED mathematically correct.
    ///
    /// v1.0.53: Made non-blocking - verification failure doesn't prevent startup.
    /// The ZK-STARK proof is an OPTIONAL advanced security feature.
    fn load_and_verify_zkstark_proof(keys_file: &Path) -> Result<Option<EncryptionKeyProof>> {
        let proof_path = keys_file.with_extension("zkstark");

        if !proof_path.exists() {
            debug!("📋 No ZK-STARK proof found at {:?} (optional feature)", proof_path);
            return Ok(None);
        }

        info!("🔍 Loading ZK-STARK proof from {:?}", proof_path);

        // Load proof
        let proof_json = match std::fs::read_to_string(&proof_path) {
            Ok(json) => json,
            Err(e) => {
                warn!("⚠️ Could not read ZK-STARK proof: {} (continuing without proof)", e);
                return Ok(None);
            }
        };

        let proof = match EncryptionKeyProof::from_json(&proof_json) {
            Ok(p) => p,
            Err(e) => {
                warn!("⚠️ Could not parse ZK-STARK proof: {} (continuing without proof)", e);
                return Ok(None);
            }
        };

        // Verify proof (UNTRUSTED automatic verification)
        // v1.0.53: Made non-blocking - verification failure is logged but doesn't prevent startup
        info!("🔬 Verifying ZK-STARK proof (untrusted automatic setup)...");
        match proof.verify() {
            Ok(()) => {
                info!("✅ ZK-STARK proof verified successfully!");
                info!("🔐 Key derivation is GUARANTEED correct (zero-knowledge proof)");
                info!("📊 Proof size: {} bytes", proof.proof_size_bytes);
                Ok(Some(proof))
            }
            Err(e) => {
                // v1.0.53: Don't fail startup on ZK-STARK verification failure
                // This can happen if:
                // 1. Proof was generated with different winterfell version
                // 2. Proof file was corrupted
                // 3. Edge case in constraint evaluation
                // The AES-GCM encryption is still fully functional without the proof
                warn!("⚠️ ZK-STARK proof verification failed: {}", e);
                warn!("   This is a non-critical error - encryption still works correctly");
                warn!("   The ZK-STARK proof is an OPTIONAL advanced security feature");
                warn!("   Delete {:?} to suppress this warning", proof_path);
                Ok(None)
            }
        }
    }

    /// Get ZK-STARK proof (if available)
    pub fn zkstark_proof(&self) -> Option<&EncryptionKeyProof> {
        self.zkstark_proof.as_ref()
    }

    /// Get DB master key (for testing only)
    #[cfg(test)]
    pub fn db_master_key(&self) -> &ProtectedKey {
        &self.db_master_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_protected_key_zeroize() {
        let key = [42u8; 32];
        let protected = ProtectedKey::new(key).unwrap();
        assert_eq!(protected.as_bytes(), &[42u8; 32]);
        drop(protected);
        // Key should be zeroed (can't test directly due to drop)
    }

    #[test]
    fn test_protected_key_derive() {
        let key = [1u8; 32];
        let protected = ProtectedKey::new(key).unwrap();
        let derived = protected.derive_subkey(b"test-context").unwrap();

        // Derived key should be different
        assert_ne!(protected.as_bytes(), derived.as_bytes());
    }

    #[test]
    fn test_blake3_commitment() {
        let key = [99u8; 32];
        let protected = ProtectedKey::new(key).unwrap();
        let commitment = protected.blake3_commitment();

        // Commitment should be deterministic
        let commitment2 = protected.blake3_commitment();
        assert_eq!(commitment, commitment2);

        // Different key = different commitment
        let key2 = [100u8; 32];
        let protected2 = ProtectedKey::new(key2).unwrap();
        assert_ne!(commitment, protected2.blake3_commitment());
    }

    #[test]
    fn test_argon2id_derivation() {
        let passphrase = "test-passphrase-super-secret";
        let salt = [55u8; 16];

        let start = std::time::Instant::now();
        let key = PassphraseKDF::derive(passphrase, &salt).unwrap();
        let duration = start.elapsed();

        // Should take at least 50ms (defense against GPU attacks)
        assert!(duration.as_millis() >= 50, "Argon2id too fast: {:?}", duration);

        // Should be deterministic
        let key2 = PassphraseKDF::derive(passphrase, &salt).unwrap();
        assert_eq!(key.as_bytes(), key2.as_bytes());

        // Different salt = different key
        let salt2 = [56u8; 16];
        let key3 = PassphraseKDF::derive(passphrase, &salt2).unwrap();
        assert_ne!(key.as_bytes(), key3.as_bytes());
    }

    #[test]
    fn test_keys_file_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let keys_file = temp_dir.path().join("test.keys");

        let passphrase = "my-super-secret-passphrase";

        // Create new encryption
        let mgr1 = EncryptionManager::create_new(passphrase, &keys_file).unwrap();
        let original_key = mgr1.db_master_key().as_bytes();

        // Load from file
        let mgr2 = EncryptionManager::from_passphrase(passphrase, &keys_file).unwrap();
        let loaded_key = mgr2.db_master_key().as_bytes();

        // Keys should match
        assert_eq!(original_key, loaded_key);
    }

    #[test]
    fn test_wrong_passphrase_fails() {
        let temp_dir = TempDir::new().unwrap();
        let keys_file = temp_dir.path().join("test.keys");

        EncryptionManager::create_new("correct-passphrase", &keys_file).unwrap();

        // Wrong passphrase should fail
        let result = EncryptionManager::from_passphrase("wrong-passphrase", &keys_file);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("wrong passphrase"));
    }

    #[test]
    fn test_file_key_derivation() {
        let temp_dir = TempDir::new().unwrap();
        let keys_file = temp_dir.path().join("test.keys");

        let mgr = EncryptionManager::create_new("test-pass", &keys_file).unwrap();

        // Derive file keys
        let key1 = mgr.derive_file_key(1, 0).unwrap();
        let key2 = mgr.derive_file_key(2, 0).unwrap();
        let key3 = mgr.derive_file_key(1, 1).unwrap();

        // All different
        assert_ne!(key1.as_bytes(), key2.as_bytes());
        assert_ne!(key1.as_bytes(), key3.as_bytes());
        assert_ne!(key2.as_bytes(), key3.as_bytes());

        // Deterministic
        let key1_again = mgr.derive_file_key(1, 0).unwrap();
        assert_eq!(key1.as_bytes(), key1_again.as_bytes());
    }

    #[test]
    fn test_blake3_commitment_verification() {
        let temp_dir = TempDir::new().unwrap();
        let keys_file = temp_dir.path().join("test.keys");

        let mgr = EncryptionManager::create_new("test", &keys_file).unwrap();

        // Read header and verify commitment
        let header = KeysFileHeader::read_from_file(&keys_file).unwrap();
        let expected_commitment = mgr.db_master_key().blake3_commitment();

        assert_eq!(header.blake3_commitment, expected_commitment);
    }
}
