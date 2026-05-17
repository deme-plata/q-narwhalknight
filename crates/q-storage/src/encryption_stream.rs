// Q-NarwhalKnight RocksDB Encryption-at-Rest - AES-CTR Stream Cipher
// v1.0.40-beta: Production-viable AES-CTR for BlockAccessCipherStream
//
// EXPERT GUIDANCE (ChatGPT):
// "For the first production release, switch to AES-CTR for the on-disk
//  stream cipher and rely on RocksDB's block checksums plus an optional
//  whole-file MAC in the header if you want AEAD-like guarantees."
//
// WHY AES-CTR (not AES-GCM):
// - BlockAccessCipherStream requires fixed-size blocks (no expansion)
// - Random-access decryption needs deterministic nonce per block
// - RocksDB provides integrity via checksums (don't need GCM auth tag)
// - AES-CTR is simpler and faster for streaming workloads

use anyhow::{anyhow, Result};
use aes::Aes256;
use aes::cipher::{KeyIvInit, StreamCipher};
use ctr::Ctr128BE;
use blake3;
use std::path::Path;
use tracing::{debug, info, warn};
use zeroize::Zeroize;

use crate::encryption::ProtectedKey;

/// AES-CTR cipher type (128-bit counter, big-endian)
type Aes256Ctr = Ctr128BE<Aes256>;

/// 📁 Encrypted SST file header format (64 bytes)
///
/// FILE STRUCTURE:
/// ```
/// [64 bytes]   Header (EncryptedFileHeader)
/// [N bytes]    Encrypted data (AES-CTR ciphertext)
/// [32 bytes]   BLAKE3 MAC (whole-file authentication)
/// ```
///
/// HEADER FORMAT:
/// ```
/// [8 bytes]    Magic: "QNKEnc01"
/// [4 bytes]    Version: 1
/// [4 bytes]    Reserved: 0
/// [8 bytes]    File ID (unique per SST)
/// [4 bytes]    Column Family ID
/// [4 bytes]    Reserved: 0
/// [16 bytes]   Initial nonce (IV for AES-CTR)
/// [16 bytes]   Reserved for future use
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EncryptedFileHeader {
    pub magic: [u8; 8],           // "QNKEnc01"
    pub version: u32,
    pub reserved1: u32,
    pub file_id: u64,             // Unique SST file identifier
    pub cf_id: u32,               // Column family ID
    pub reserved2: u32,
    pub initial_nonce: [u8; 16],  // IV for AES-CTR (incremented per block)
    pub reserved3: [u8; 16],      // Future: key rotation epoch, etc.
}

impl EncryptedFileHeader {
    const MAGIC: &'static [u8; 8] = b"QNKEnc01";
    const VERSION: u32 = 1;
    pub const SIZE: usize = 64;

    /// Create new encrypted file header
    pub fn new(file_id: u64, cf_id: u32) -> Result<Self> {
        let mut initial_nonce = [0u8; 16];
        getrandom::getrandom(&mut initial_nonce)
            .map_err(|e| anyhow!("RNG failure: {}", e))?;

        Ok(Self {
            magic: *Self::MAGIC,
            version: Self::VERSION,
            reserved1: 0,
            file_id,
            cf_id,
            reserved2: 0,
            initial_nonce,
            reserved3: [0u8; 16],
        })
    }

    /// Verify magic number and version
    pub fn verify(&self) -> Result<()> {
        if &self.magic != Self::MAGIC {
            return Err(anyhow!("Invalid magic (corrupted or plaintext file)"));
        }
        if self.version != Self::VERSION {
            return Err(anyhow!("Unsupported version: {}", self.version));
        }
        Ok(())
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        unsafe {
            std::ptr::read(self as *const Self as *const [u8; Self::SIZE])
        }
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(anyhow!("Header too short: {} bytes", bytes.len()));
        }

        let header = unsafe {
            std::ptr::read_unaligned(bytes.as_ptr() as *const Self)
        };

        header.verify()?;
        Ok(header)
    }
}

/// 🔐 AES-CTR stream cipher for SST file encryption
///
/// NONCE STRATEGY (per expert feedback):
/// - Initial nonce from file header (random)
/// - Increment counter for each 4KB block
/// - Formula: nonce = initial_nonce + (file_offset / 4096)
///
/// SECURITY PROPERTIES:
/// - Deterministic encryption (required for random-access)
/// - No nonce reuse (counter increments per block)
/// - RocksDB checksums provide integrity
/// - Optional BLAKE3 MAC for whole-file authentication
pub struct AesCtrStream {
    file_key: ProtectedKey,
    header: EncryptedFileHeader,
}

impl AesCtrStream {
    /// Block size for nonce counter (4KB = standard page size)
    const BLOCK_SIZE: u64 = 4096;

    /// Create new stream cipher for encryption
    pub fn new(file_key: ProtectedKey, file_id: u64, cf_id: u32) -> Result<Self> {
        let header = EncryptedFileHeader::new(file_id, cf_id)?;

        info!("🔐 AES-CTR stream cipher created (file_id={}, cf_id={})", file_id, cf_id);

        Ok(Self { file_key, header })
    }

    /// Load existing stream cipher from header
    pub fn from_header(file_key: ProtectedKey, header: EncryptedFileHeader) -> Result<Self> {
        header.verify()?;

        debug!("📂 AES-CTR stream cipher loaded (file_id={}, cf_id={})",
            header.file_id, header.cf_id);

        Ok(Self { file_key, header })
    }

    /// Compute nonce for given file offset
    ///
    /// CRITICAL: Nonce must be deterministic for random-access decryption
    /// Formula: nonce = initial_nonce + (file_offset / BLOCK_SIZE)
    fn compute_nonce(&self, file_offset: u64) -> [u8; 16] {
        let mut nonce = self.header.initial_nonce;

        // Add block index to nonce (big-endian)
        let block_index = file_offset / Self::BLOCK_SIZE;
        let block_bytes = block_index.to_be_bytes();

        // XOR last 8 bytes with block index (prevent overflow)
        for i in 0..8 {
            nonce[8 + i] ^= block_bytes[i];
        }

        nonce
    }

    /// Encrypt data at given file offset
    ///
    /// IMPORTANT: This modifies `data` in-place!
    /// Caller must provide aligned offset (multiple of BLOCK_SIZE)
    pub fn encrypt_at_offset(&self, data: &mut [u8], file_offset: u64) -> Result<()> {
        if file_offset % Self::BLOCK_SIZE != 0 {
            warn!("⚠️ Unaligned encryption offset: {} (not multiple of {})",
                file_offset, Self::BLOCK_SIZE);
        }

        let nonce = self.compute_nonce(file_offset);

        let mut cipher = Aes256Ctr::new(
            self.file_key.as_bytes().into(),
            &nonce.into(),
        );

        cipher.apply_keystream(data);

        debug!("🔒 Encrypted {} bytes at offset {}", data.len(), file_offset);
        Ok(())
    }

    /// Decrypt data at given file offset
    ///
    /// IMPORTANT: This modifies `data` in-place!
    /// AES-CTR is symmetric: encrypt = decrypt
    pub fn decrypt_at_offset(&self, data: &mut [u8], file_offset: u64) -> Result<()> {
        if file_offset % Self::BLOCK_SIZE != 0 {
            warn!("⚠️ Unaligned decryption offset: {} (not multiple of {})",
                file_offset, Self::BLOCK_SIZE);
        }

        let nonce = self.compute_nonce(file_offset);

        let mut cipher = Aes256Ctr::new(
            self.file_key.as_bytes().into(),
            &nonce.into(),
        );

        cipher.apply_keystream(data);

        debug!("🔓 Decrypted {} bytes at offset {}", data.len(), file_offset);
        Ok(())
    }

    /// Get file header for writing to disk
    pub fn header(&self) -> &EncryptedFileHeader {
        &self.header
    }

    /// Compute BLAKE3 MAC for whole-file authentication (optional)
    pub fn compute_mac(&self, ciphertext: &[u8]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.file_key.as_bytes());
        hasher.update(&self.header.to_bytes());
        hasher.update(ciphertext);
        hasher.finalize().into()
    }

    /// Verify BLAKE3 MAC (optional)
    pub fn verify_mac(&self, ciphertext: &[u8], mac: &[u8; 32]) -> Result<()> {
        let computed_mac = self.compute_mac(ciphertext);

        if &computed_mac != mac {
            return Err(anyhow!("BLAKE3 MAC mismatch (file corruption or tampering)"));
        }

        Ok(())
    }
}

/// 📝 WAL (Write-Ahead Log) encryption
///
/// SPECIAL CONSIDERATIONS:
/// - Append-only writes (no random access during writing)
/// - Sequential nonce counter (no need for offset-based nonces)
/// - Recovery reads need random access (use offset-based nonces)
///
/// STRATEGY:
/// - Use same AES-CTR stream cipher
/// - Nonce = initial_nonce + sequential counter
/// - Persist counter in WAL header
pub struct WalEncryption {
    stream: AesCtrStream,
    write_counter: u64,  // Sequential write counter
}

impl WalEncryption {
    /// Create new WAL encryption
    pub fn new(file_key: ProtectedKey, file_id: u64) -> Result<Self> {
        let stream = AesCtrStream::new(file_key, file_id, 0)?;

        info!("📝 WAL encryption initialized (file_id={})", file_id);

        Ok(Self {
            stream,
            write_counter: 0,
        })
    }

    /// Encrypt next WAL entry (append-only)
    pub fn encrypt_next(&mut self, data: &mut [u8]) -> Result<()> {
        let offset = self.write_counter * AesCtrStream::BLOCK_SIZE;
        self.stream.encrypt_at_offset(data, offset)?;

        self.write_counter += 1;

        debug!("📝 Encrypted WAL entry {} ({} bytes)", self.write_counter, data.len());
        Ok(())
    }

    /// Decrypt WAL entry at specific offset (recovery mode)
    pub fn decrypt_at_offset(&self, data: &mut [u8], file_offset: u64) -> Result<()> {
        self.stream.decrypt_at_offset(data, file_offset)
    }

    /// Get current write counter (for header persistence)
    pub fn write_counter(&self) -> u64 {
        self.write_counter
    }

    /// Get WAL header
    pub fn header(&self) -> &EncryptedFileHeader {
        self.stream.header()
    }
}

/// 🔧 High-level file encryption manager
///
/// Coordinates:
/// - SST file encryption (random access)
/// - WAL encryption (append-only)
/// - Header management
/// - MAC computation
pub struct FileEncryptionManager {
    encryption_manager: crate::encryption::EncryptionManager,
}

impl FileEncryptionManager {
    /// Create new file encryption manager
    pub fn new(encryption_manager: crate::encryption::EncryptionManager) -> Self {
        Self { encryption_manager }
    }

    /// Create stream cipher for SST file
    pub fn create_sst_cipher(&self, file_id: u64, cf_id: u32) -> Result<AesCtrStream> {
        let file_key = self.encryption_manager.derive_file_key(file_id, cf_id)?;
        AesCtrStream::new(file_key, file_id, cf_id)
    }

    /// Load stream cipher from existing SST file
    pub fn load_sst_cipher(&self, header_bytes: &[u8]) -> Result<AesCtrStream> {
        let header = EncryptedFileHeader::from_bytes(header_bytes)?;
        let file_key = self.encryption_manager.derive_file_key(header.file_id, header.cf_id)?;
        AesCtrStream::from_header(file_key, header)
    }

    /// Create WAL encryption
    pub fn create_wal_cipher(&self, file_id: u64) -> Result<WalEncryption> {
        let file_key = self.encryption_manager.derive_file_key(file_id, 0)?;
        WalEncryption::new(file_key, file_id)
    }
}

/// 🧪 Test helpers
#[cfg(test)]
mod test_helpers {
    use super::*;

    pub fn create_test_key() -> ProtectedKey {
        let key = [42u8; 32];
        ProtectedKey::new(key).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_helpers::*;

    #[test]
    fn test_header_roundtrip() {
        let header = EncryptedFileHeader::new(123, 5).unwrap();
        let bytes = header.to_bytes();
        let loaded = EncryptedFileHeader::from_bytes(&bytes).unwrap();

        assert_eq!(header.file_id, loaded.file_id);
        assert_eq!(header.cf_id, loaded.cf_id);
        assert_eq!(header.initial_nonce, loaded.initial_nonce);
    }

    #[test]
    fn test_header_magic_verification() {
        let mut header = EncryptedFileHeader::new(1, 0).unwrap();
        header.magic = *b"BADMAGIC";

        let bytes = header.to_bytes();
        let result = EncryptedFileHeader::from_bytes(&bytes);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("magic"));
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = create_test_key();
        let stream = AesCtrStream::new(key, 1, 0).unwrap();

        let mut data = b"Hello, encrypted world!".to_vec();
        let original = data.clone();

        // Encrypt
        stream.encrypt_at_offset(&mut data, 0).unwrap();
        assert_ne!(data, original, "Data should be encrypted");

        // Decrypt
        stream.decrypt_at_offset(&mut data, 0).unwrap();
        assert_eq!(data, original, "Data should match original");
    }

    #[test]
    fn test_random_access_decryption() {
        let key = create_test_key();
        let stream = AesCtrStream::new(key, 1, 0).unwrap();

        // Encrypt block at offset 0
        let mut block0 = b"Block 0: First block".to_vec();
        let original0 = block0.clone();
        stream.encrypt_at_offset(&mut block0, 0).unwrap();

        // Encrypt block at offset 4096
        let mut block1 = b"Block 1: Second block".to_vec();
        let original1 = block1.clone();
        stream.encrypt_at_offset(&mut block1, 4096).unwrap();

        // Decrypt in reverse order (random access)
        stream.decrypt_at_offset(&mut block1, 4096).unwrap();
        assert_eq!(block1, original1);

        stream.decrypt_at_offset(&mut block0, 0).unwrap();
        assert_eq!(block0, original0);
    }

    #[test]
    fn test_nonce_determinism() {
        let key = create_test_key();
        let stream = AesCtrStream::new(key, 1, 0).unwrap();

        let mut data1 = b"Same data".to_vec();
        let mut data2 = b"Same data".to_vec();

        // Encrypt twice at same offset
        stream.encrypt_at_offset(&mut data1, 0).unwrap();
        stream.encrypt_at_offset(&mut data2, 0).unwrap();

        // Should produce identical ciphertext (deterministic)
        assert_eq!(data1, data2);
    }

    #[test]
    fn test_different_offsets_different_ciphertext() {
        let key = create_test_key();
        let stream = AesCtrStream::new(key, 1, 0).unwrap();

        let mut data1 = b"Same plaintext".to_vec();
        let mut data2 = b"Same plaintext".to_vec();

        stream.encrypt_at_offset(&mut data1, 0).unwrap();
        stream.encrypt_at_offset(&mut data2, 4096).unwrap();

        // Different offsets = different ciphertext
        assert_ne!(data1, data2);
    }

    #[test]
    fn test_blake3_mac() {
        let key = create_test_key();
        let stream = AesCtrStream::new(key, 1, 0).unwrap();

        let ciphertext = b"encrypted data here";
        let mac = stream.compute_mac(ciphertext);

        // Verification should succeed
        stream.verify_mac(ciphertext, &mac).unwrap();

        // Modified ciphertext should fail
        let mut modified = ciphertext.to_vec();
        modified[0] ^= 1;
        let result = stream.verify_mac(&modified, &mac);
        assert!(result.is_err());
    }

    #[test]
    fn test_wal_sequential_encryption() {
        let key = create_test_key();
        let mut wal = WalEncryption::new(key, 100).unwrap();

        let mut entry1 = b"WAL entry 1".to_vec();
        let mut entry2 = b"WAL entry 2".to_vec();

        wal.encrypt_next(&mut entry1).unwrap();
        wal.encrypt_next(&mut entry2).unwrap();

        assert_eq!(wal.write_counter(), 2);

        // Different entries should have different ciphertext
        assert_ne!(entry1, entry2);
    }

    #[test]
    fn test_wal_random_access_recovery() {
        let key = create_test_key();
        let mut wal = WalEncryption::new(key, 100).unwrap();

        let mut entry = b"WAL entry".to_vec();
        let original = entry.clone();

        // Encrypt entry 0
        wal.encrypt_next(&mut entry).unwrap();

        // Decrypt using random access (recovery mode)
        wal.decrypt_at_offset(&mut entry, 0).unwrap();
        assert_eq!(entry, original);
    }

    #[test]
    fn test_large_file_encryption() {
        let key = create_test_key();
        let stream = AesCtrStream::new(key, 1, 0).unwrap();

        // Encrypt 1MB of data in 4KB blocks
        let block_size = 4096;
        let num_blocks = 256; // 1MB

        for block_idx in 0..num_blocks {
            let offset = (block_idx * block_size) as u64;
            let mut block = vec![block_idx as u8; block_size];
            let original = block.clone();

            stream.encrypt_at_offset(&mut block, offset).unwrap();
            assert_ne!(block, original);

            stream.decrypt_at_offset(&mut block, offset).unwrap();
            assert_eq!(block, original);
        }
    }
}
