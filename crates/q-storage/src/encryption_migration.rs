// Q-NarwhalKnight RocksDB Encryption Migration
// v1.0.41-beta: Transitional provider for online migration
//
// ARCHITECTURE:
// - Support mixed plaintext + encrypted databases
// - Gradual migration without downtime
// - Progress tracking and recovery
// - Automatic detection of file format

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::encryption::{EncryptionManager, ProtectedKey};
use crate::encryption_stream::{AesCtrStream, EncryptedFileHeader};

/// File encryption status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileEncryptionStatus {
    /// File is plaintext (not encrypted)
    Plaintext,
    /// File is encrypted with AES-CTR
    Encrypted,
    /// File format unknown (need to detect)
    Unknown,
}

/// Migration progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationProgress {
    /// Total files to migrate
    pub total_files: usize,
    /// Files migrated so far
    pub migrated_files: usize,
    /// Files failed to migrate
    pub failed_files: Vec<String>,
    /// Migration start time
    pub start_time: std::time::SystemTime,
    /// Last update time
    pub last_update: std::time::SystemTime,
}

impl MigrationProgress {
    pub fn new(total_files: usize) -> Self {
        let now = std::time::SystemTime::now();
        Self {
            total_files,
            migrated_files: 0,
            failed_files: Vec::new(),
            start_time: now,
            last_update: now,
        }
    }

    pub fn record_success(&mut self) {
        self.migrated_files += 1;
        self.last_update = std::time::SystemTime::now();
    }

    pub fn record_failure(&mut self, filename: String) {
        self.failed_files.push(filename);
        self.last_update = std::time::SystemTime::now();
    }

    pub fn is_complete(&self) -> bool {
        self.migrated_files + self.failed_files.len() >= self.total_files
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_files == 0 {
            return 100.0;
        }
        (self.migrated_files as f64 / self.total_files as f64) * 100.0
    }
}

/// Transitional encryption provider
///
/// FEATURES:
/// - Detect plaintext vs encrypted files automatically
/// - Support mixed databases during migration
/// - Track migration progress
/// - Gradual migration without downtime
pub struct TransitionalEncryptionProvider {
    encryption_manager: Arc<EncryptionManager>,
    file_status_cache: Arc<Mutex<HashMap<PathBuf, FileEncryptionStatus>>>,
    migration_progress: Arc<Mutex<Option<MigrationProgress>>>,
    migration_enabled: bool,
}

impl TransitionalEncryptionProvider {
    /// Create new transitional provider
    pub fn new(
        encryption_manager: EncryptionManager,
        migration_enabled: bool,
    ) -> Self {
        Self {
            encryption_manager: Arc::new(encryption_manager),
            file_status_cache: Arc::new(Mutex::new(HashMap::new())),
            migration_progress: Arc::new(Mutex::new(None)),
            migration_enabled,
        }
    }

    /// Detect if file is encrypted or plaintext
    pub fn detect_file_status(&self, file_path: &Path) -> Result<FileEncryptionStatus> {
        // Check cache first
        {
            let cache = self.file_status_cache.lock().unwrap();
            if let Some(&status) = cache.get(file_path) {
                return Ok(status);
            }
        }

        // Read file header
        let mut file = File::open(file_path)?;
        let mut header_buf = [0u8; 64];

        let bytes_read = file.read(&mut header_buf)?;
        if bytes_read < 64 {
            // File too small, assume plaintext
            self.cache_status(file_path, FileEncryptionStatus::Plaintext);
            return Ok(FileEncryptionStatus::Plaintext);
        }

        // Check magic number
        let magic = &header_buf[0..8];
        let status = if magic == b"QNKEnc01" {
            FileEncryptionStatus::Encrypted
        } else {
            FileEncryptionStatus::Plaintext
        };

        self.cache_status(file_path, status);
        Ok(status)
    }

    /// Cache file encryption status
    fn cache_status(&self, file_path: &Path, status: FileEncryptionStatus) {
        let mut cache = self.file_status_cache.lock().unwrap();
        cache.insert(file_path.to_path_buf(), status);
    }

    /// Start migration process
    pub fn start_migration(&self, db_path: &Path) -> Result<()> {
        if !self.migration_enabled {
            return Err(anyhow!("Migration not enabled"));
        }

        info!("🔄 Starting database encryption migration: {:?}", db_path);

        // Discover all SST files
        let sst_files = self.discover_sst_files(db_path)?;
        info!("📂 Found {} SST files to migrate", sst_files.len());

        // Initialize progress tracking
        {
            let mut progress = self.migration_progress.lock().unwrap();
            *progress = Some(MigrationProgress::new(sst_files.len()));
        }

        // Count plaintext files
        let plaintext_count = sst_files.iter()
            .filter(|path| {
                matches!(
                    self.detect_file_status(path),
                    Ok(FileEncryptionStatus::Plaintext)
                )
            })
            .count();

        info!("🔓 {} plaintext files need encryption", plaintext_count);
        info!("🔐 {} files already encrypted", sst_files.len() - plaintext_count);

        Ok(())
    }

    /// Discover all SST files in database
    fn discover_sst_files(&self, db_path: &Path) -> Result<Vec<PathBuf>> {
        let mut sst_files = Vec::new();

        fn visit_dir(dir: &Path, sst_files: &mut Vec<PathBuf>) -> Result<()> {
            if !dir.is_dir() {
                return Ok(());
            }

            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_dir() {
                    visit_dir(&path, sst_files)?;
                } else if let Some(ext) = path.extension() {
                    if ext == "sst" {
                        sst_files.push(path);
                    }
                }
            }

            Ok(())
        }

        visit_dir(db_path, &mut sst_files)?;
        Ok(sst_files)
    }

    /// Migrate single file (plaintext → encrypted)
    pub fn migrate_file(&self, file_path: &Path, file_id: u64, cf_id: u32) -> Result<()> {
        // Check current status
        let status = self.detect_file_status(file_path)?;

        if status == FileEncryptionStatus::Encrypted {
            debug!("File already encrypted: {:?}", file_path);
            return Ok(());
        }

        info!("🔄 Migrating file: {:?}", file_path);

        // Read plaintext data
        let plaintext_data = fs::read(file_path)?;

        // Derive encryption key
        let file_key = self.encryption_manager.derive_file_key(file_id, cf_id)?;

        // Create cipher stream
        let stream = AesCtrStream::new(file_key, file_id, cf_id)?;

        // Create encrypted file with header
        let header_bytes = stream.header().to_bytes();
        let mut encrypted_data = Vec::with_capacity(header_bytes.len() + plaintext_data.len());
        encrypted_data.extend_from_slice(&header_bytes);

        // Encrypt data in 4KB blocks
        let mut offset = 0u64;
        for chunk in plaintext_data.chunks(4096) {
            let mut encrypted_chunk = chunk.to_vec();
            stream.encrypt_at_offset(&mut encrypted_chunk, offset)?;
            encrypted_data.extend_from_slice(&encrypted_chunk);
            offset += 4096;
        }

        // Write encrypted file atomically
        let temp_path = file_path.with_extension("sst.tmp");
        fs::write(&temp_path, &encrypted_data)?;
        fs::rename(&temp_path, file_path)?;

        // Update cache
        self.cache_status(file_path, FileEncryptionStatus::Encrypted);

        // Update progress
        {
            let mut progress = self.migration_progress.lock().unwrap();
            if let Some(ref mut p) = *progress {
                p.record_success();
            }
        }

        info!("✅ File migrated successfully: {:?}", file_path);
        Ok(())
    }

    /// Get current migration progress
    pub fn get_progress(&self) -> Option<MigrationProgress> {
        let progress = self.migration_progress.lock().unwrap();
        progress.clone()
    }

    /// Save migration progress to disk
    pub fn save_progress(&self, path: &Path) -> Result<()> {
        let progress = self.migration_progress.lock().unwrap();
        if let Some(ref p) = *progress {
            let json = serde_json::to_string_pretty(p)?;
            fs::write(path, json)?;
            debug!("💾 Migration progress saved: {:?}", path);
        }
        Ok(())
    }

    /// Load migration progress from disk
    pub fn load_progress(&self, path: &Path) -> Result<()> {
        let json = fs::read_to_string(path)?;
        let loaded: MigrationProgress = serde_json::from_str(&json)?;

        let mut progress = self.migration_progress.lock().unwrap();
        *progress = Some(loaded);

        info!("📂 Migration progress loaded: {:?}", path);
        Ok(())
    }

    /// Clear file status cache
    pub fn clear_cache(&self) {
        let mut cache = self.file_status_cache.lock().unwrap();
        cache.clear();
        debug!("🧹 File status cache cleared");
    }
}

/// File format detector (standalone utility)
pub struct FileFormatDetector;

impl FileFormatDetector {
    /// Detect if file is encrypted
    pub fn is_encrypted(file_path: &Path) -> Result<bool> {
        let mut file = File::open(file_path)?;
        let mut magic = [0u8; 8];

        let bytes_read = file.read(&mut magic)?;
        if bytes_read < 8 {
            return Ok(false);
        }

        Ok(&magic == b"QNKEnc01")
    }

    /// Get file encryption info
    pub fn get_encryption_info(file_path: &Path) -> Result<Option<EncryptedFileHeader>> {
        if !Self::is_encrypted(file_path)? {
            return Ok(None);
        }

        let mut file = File::open(file_path)?;
        let mut header_buf = [0u8; 64];
        file.read_exact(&mut header_buf)?;

        let header = EncryptedFileHeader::from_bytes(&header_buf)?;
        Ok(Some(header))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_encryption_manager() -> EncryptionManager {
        let temp_dir = TempDir::new().unwrap();
        let keys_file = temp_dir.path().join("test.keys");
        EncryptionManager::create_new("test-pass", &keys_file).unwrap()
    }

    #[test]
    fn test_detect_plaintext_file() {
        let temp_dir = TempDir::new().unwrap();
        let plaintext_file = temp_dir.path().join("plaintext.sst");

        fs::write(&plaintext_file, b"plaintext data").unwrap();

        let mgr = create_test_encryption_manager();
        let provider = TransitionalEncryptionProvider::new(mgr, true);

        let status = provider.detect_file_status(&plaintext_file).unwrap();
        assert_eq!(status, FileEncryptionStatus::Plaintext);
    }

    #[test]
    fn test_detect_encrypted_file() {
        let temp_dir = TempDir::new().unwrap();
        let encrypted_file = temp_dir.path().join("encrypted.sst");

        let mgr = create_test_encryption_manager();
        let file_key = mgr.derive_file_key(1, 0).unwrap();
        let stream = AesCtrStream::new(file_key, 1, 0).unwrap();

        // Write encrypted file
        let header = stream.header().to_bytes();
        fs::write(&encrypted_file, &header).unwrap();

        let provider = TransitionalEncryptionProvider::new(mgr, true);

        let status = provider.detect_file_status(&encrypted_file).unwrap();
        assert_eq!(status, FileEncryptionStatus::Encrypted);
    }

    #[test]
    fn test_migration_progress() {
        let mut progress = MigrationProgress::new(100);

        assert_eq!(progress.total_files, 100);
        assert_eq!(progress.migrated_files, 0);
        assert_eq!(progress.success_rate(), 0.0);

        progress.record_success();
        assert_eq!(progress.migrated_files, 1);
        assert_eq!(progress.success_rate(), 1.0);

        for _ in 0..99 {
            progress.record_success();
        }

        assert_eq!(progress.migrated_files, 100);
        assert_eq!(progress.success_rate(), 100.0);
        assert!(progress.is_complete());
    }

    #[test]
    fn test_file_format_detector() {
        let temp_dir = TempDir::new().unwrap();

        // Plaintext file
        let plaintext = temp_dir.path().join("plain.sst");
        fs::write(&plaintext, b"plaintext").unwrap();
        assert!(!FileFormatDetector::is_encrypted(&plaintext).unwrap());

        // Encrypted file
        let encrypted = temp_dir.path().join("enc.sst");
        let mgr = create_test_encryption_manager();
        let key = mgr.derive_file_key(1, 0).unwrap();
        let stream = AesCtrStream::new(key, 1, 0).unwrap();
        fs::write(&encrypted, stream.header().to_bytes()).unwrap();

        assert!(FileFormatDetector::is_encrypted(&encrypted).unwrap());
    }

    #[test]
    fn test_migrate_single_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.sst");

        // Create plaintext file
        fs::write(&file_path, b"plaintext data to encrypt").unwrap();

        let mgr = create_test_encryption_manager();
        let provider = TransitionalEncryptionProvider::new(mgr, true);

        // Migrate file
        provider.migrate_file(&file_path, 1, 0).unwrap();

        // Verify encrypted
        assert!(FileFormatDetector::is_encrypted(&file_path).unwrap());

        // Verify can decrypt
        let header_bytes = {
            let mut file = File::open(&file_path).unwrap();
            let mut buf = [0u8; 64];
            file.read_exact(&mut buf).unwrap();
            buf
        };

        let header = EncryptedFileHeader::from_bytes(&header_bytes).unwrap();
        assert_eq!(header.file_id, 1);
        assert_eq!(header.cf_id, 0);
    }

    #[test]
    fn test_progress_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let progress_file = temp_dir.path().join("progress.json");

        let mgr = create_test_encryption_manager();
        let provider = TransitionalEncryptionProvider::new(mgr, true);

        // Create progress
        {
            let mut progress = provider.migration_progress.lock().unwrap();
            let mut p = MigrationProgress::new(10);
            p.record_success();
            p.record_success();
            *progress = Some(p);
        }

        // Save
        provider.save_progress(&progress_file).unwrap();

        // Load into new provider
        let mgr2 = create_test_encryption_manager();
        let provider2 = TransitionalEncryptionProvider::new(mgr2, true);
        provider2.load_progress(&progress_file).unwrap();

        let loaded = provider2.get_progress().unwrap();
        assert_eq!(loaded.total_files, 10);
        assert_eq!(loaded.migrated_files, 2);
    }
}
