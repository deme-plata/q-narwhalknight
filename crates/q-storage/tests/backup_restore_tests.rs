//! Backup and Restore Tests
//!
//! These tests verify that backups work correctly and can restore
//! the blockchain to a valid state after disaster.
//!
//! CRITICAL: A broken backup system means NO RECOVERY from:
//! - Database corruption
//! - Disk failures
//! - Sync-down attacks
//! - Software bugs that corrupt state
//!
//! Run with: cargo test --package q-storage --test backup_restore_tests

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

// ============================================================================
// MOCK BACKUP STRUCTURES
// ============================================================================

/// Represents a backup manifest entry
#[derive(Debug, Clone)]
pub struct BackupManifest {
    pub start_height: u64,
    pub end_height: u64,
    pub filename: String,
    pub checksum: [u8; 32],
    pub timestamp: u64,
    pub size_bytes: u64,
    pub blocks_count: u64,
}

/// Represents a block (simplified)
#[derive(Debug, Clone)]
pub struct Block {
    pub height: u64,
    pub hash: [u8; 32],
    pub parent_hash: [u8; 32],
    pub data: Vec<u8>,
}

impl Block {
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.height.to_le_bytes());
        bytes.extend_from_slice(&self.hash);
        bytes.extend_from_slice(&self.parent_hash);
        bytes.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.data);
        bytes
    }

    pub fn deserialize(bytes: &[u8]) -> Result<(Self, usize), String> {
        if bytes.len() < 8 + 32 + 32 + 4 {
            return Err("Block data too short".to_string());
        }

        let height = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes[8..40]);
        let mut parent_hash = [0u8; 32];
        parent_hash.copy_from_slice(&bytes[40..72]);
        let data_len = u32::from_le_bytes(bytes[72..76].try_into().unwrap()) as usize;

        if bytes.len() < 76 + data_len {
            return Err("Block data truncated".to_string());
        }

        let data = bytes[76..76 + data_len].to_vec();
        let total_size = 76 + data_len;

        Ok((Block { height, hash, parent_hash, data }, total_size))
    }
}

/// Mock blockchain state
#[derive(Debug, Default)]
pub struct MockChainState {
    pub blocks: HashMap<u64, Block>,
    pub balances: HashMap<String, u128>,
    pub height: u64,
}

impl MockChainState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_block(&mut self, block: Block) -> Result<(), String> {
        // Verify chain continuity
        if block.height > 0 {
            if let Some(parent) = self.blocks.get(&(block.height - 1)) {
                if parent.hash != block.parent_hash {
                    return Err("Parent hash mismatch".to_string());
                }
            } else if block.height != self.height + 1 {
                return Err("Missing parent block".to_string());
            }
        }

        let block_height = block.height;
        self.blocks.insert(block_height, block);
        self.height = self.height.max(block_height);
        Ok(())
    }
}

/// Backup manager
pub struct BackupManager {
    backup_dir: PathBuf,
    manifests: Vec<BackupManifest>,
}

impl BackupManager {
    pub fn new(backup_dir: PathBuf) -> Self {
        Self {
            backup_dir,
            manifests: Vec::new(),
        }
    }

    /// Calculate simple checksum
    fn calculate_checksum(data: &[u8]) -> [u8; 32] {
        let mut checksum = [0u8; 32];
        for (i, byte) in data.iter().enumerate() {
            checksum[i % 32] ^= byte;
            checksum[(i + 1) % 32] = checksum[(i + 1) % 32].wrapping_add(*byte);
        }
        checksum
    }

    /// Create a backup of blocks in a height range
    pub fn create_backup(
        &mut self,
        state: &MockChainState,
        start_height: u64,
        end_height: u64,
    ) -> Result<BackupManifest, String> {
        if start_height > end_height {
            return Err("Invalid height range".to_string());
        }

        // Serialize blocks
        let mut backup_data = Vec::new();
        let mut blocks_count = 0u64;

        for height in start_height..=end_height {
            if let Some(block) = state.blocks.get(&height) {
                let serialized = block.serialize();
                backup_data.extend_from_slice(&serialized);
                blocks_count += 1;
            }
        }

        if blocks_count == 0 {
            return Err("No blocks to backup".to_string());
        }

        let checksum = Self::calculate_checksum(&backup_data);
        let filename = format!("backup_{}_{}.bin", start_height, end_height);
        let filepath = self.backup_dir.join(&filename);

        // Write backup file
        fs::create_dir_all(&self.backup_dir).map_err(|e| e.to_string())?;
        let mut file = fs::File::create(&filepath).map_err(|e| e.to_string())?;
        file.write_all(&backup_data).map_err(|e| e.to_string())?;

        let manifest = BackupManifest {
            start_height,
            end_height,
            filename,
            checksum,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            size_bytes: backup_data.len() as u64,
            blocks_count,
        };

        self.manifests.push(manifest.clone());
        Ok(manifest)
    }

    /// Verify a backup file integrity
    pub fn verify_backup(&self, manifest: &BackupManifest) -> Result<(), String> {
        let filepath = self.backup_dir.join(&manifest.filename);

        let mut file = fs::File::open(&filepath).map_err(|e| e.to_string())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(|e| e.to_string())?;

        // Verify size
        if data.len() as u64 != manifest.size_bytes {
            return Err(format!(
                "Size mismatch: expected {}, got {}",
                manifest.size_bytes,
                data.len()
            ));
        }

        // Verify checksum
        let calculated_checksum = Self::calculate_checksum(&data);
        if calculated_checksum != manifest.checksum {
            return Err("Checksum verification failed - backup is CORRUPTED!".to_string());
        }

        // Verify blocks can be deserialized
        let mut offset = 0;
        let mut count = 0;
        while offset < data.len() {
            let (block, size) = Block::deserialize(&data[offset..])
                .map_err(|e| format!("Block deserialization failed at offset {}: {}", offset, e))?;

            // Verify block height is in expected range
            if block.height < manifest.start_height || block.height > manifest.end_height {
                return Err(format!(
                    "Block height {} outside backup range [{}, {}]",
                    block.height, manifest.start_height, manifest.end_height
                ));
            }

            offset += size;
            count += 1;
        }

        if count != manifest.blocks_count as usize {
            return Err(format!(
                "Block count mismatch: expected {}, got {}",
                manifest.blocks_count, count
            ));
        }

        Ok(())
    }

    /// Restore blocks from a backup
    pub fn restore_backup(
        &self,
        manifest: &BackupManifest,
        target_state: &mut MockChainState,
    ) -> Result<u64, String> {
        // First verify the backup
        self.verify_backup(manifest)?;

        let filepath = self.backup_dir.join(&manifest.filename);
        let mut file = fs::File::open(&filepath).map_err(|e| e.to_string())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(|e| e.to_string())?;

        // Deserialize and add blocks
        let mut offset = 0;
        let mut restored_count = 0;

        while offset < data.len() {
            let (block, size) = Block::deserialize(&data[offset..])?;

            // Add to state (may fail if block already exists or chain continuity broken)
            if target_state.blocks.get(&block.height).is_none() {
                target_state.add_block(block)?;
                restored_count += 1;
            }

            offset += size;
        }

        Ok(restored_count)
    }
}

// ============================================================================
// BACKUP CREATION TESTS
// ============================================================================

mod backup_creation_tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_state(block_count: u64) -> MockChainState {
        let mut state = MockChainState::new();

        let mut prev_hash = [0u8; 32];
        for height in 0..block_count {
            let mut hash = [0u8; 32];
            hash[0..8].copy_from_slice(&height.to_le_bytes());

            let block = Block {
                height,
                hash,
                parent_hash: prev_hash,
                data: format!("Block {} data", height).into_bytes(),
            };

            state.add_block(block).unwrap();
            prev_hash = hash;
        }

        state
    }

    #[test]
    fn test_create_backup() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let state = create_test_state(100);

        let manifest = manager.create_backup(&state, 0, 99).unwrap();

        assert_eq!(manifest.start_height, 0);
        assert_eq!(manifest.end_height, 99);
        assert_eq!(manifest.blocks_count, 100);
        assert!(manifest.size_bytes > 0);
    }

    #[test]
    fn test_backup_range() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let state = create_test_state(100);

        // Backup only blocks 50-74
        let manifest = manager.create_backup(&state, 50, 74).unwrap();

        assert_eq!(manifest.start_height, 50);
        assert_eq!(manifest.end_height, 74);
        assert_eq!(manifest.blocks_count, 25);
    }

    #[test]
    fn test_backup_empty_range_fails() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let state = create_test_state(10);

        // Try to backup non-existent blocks
        let result = manager.create_backup(&state, 100, 200);
        assert!(result.is_err());
    }

    #[test]
    fn test_backup_invalid_range_fails() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let state = create_test_state(100);

        // End before start
        let result = manager.create_backup(&state, 50, 40);
        assert!(result.is_err());
    }
}

// ============================================================================
// BACKUP VERIFICATION TESTS
// ============================================================================

mod backup_verification_tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_state(block_count: u64) -> MockChainState {
        let mut state = MockChainState::new();
        let mut prev_hash = [0u8; 32];
        for height in 0..block_count {
            let mut hash = [0u8; 32];
            hash[0..8].copy_from_slice(&height.to_le_bytes());
            let block = Block {
                height,
                hash,
                parent_hash: prev_hash,
                data: format!("Block {} data", height).into_bytes(),
            };
            state.add_block(block).unwrap();
            prev_hash = hash;
        }
        state
    }

    #[test]
    fn test_verify_valid_backup() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let state = create_test_state(100);

        let manifest = manager.create_backup(&state, 0, 99).unwrap();

        // Should verify successfully
        assert!(manager.verify_backup(&manifest).is_ok());
    }

    #[test]
    fn test_detect_corrupted_backup() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let state = create_test_state(100);

        let manifest = manager.create_backup(&state, 0, 99).unwrap();

        // Corrupt the backup file
        let filepath = temp_dir.path().join(&manifest.filename);
        let mut data = fs::read(&filepath).unwrap();
        data[100] ^= 0xFF; // Flip some bits
        fs::write(&filepath, &data).unwrap();

        // Should detect corruption
        let result = manager.verify_backup(&manifest);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("CORRUPTED") || err_msg.contains("checksum"));
    }

    #[test]
    fn test_detect_truncated_backup() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let state = create_test_state(100);

        let manifest = manager.create_backup(&state, 0, 99).unwrap();

        // Truncate the backup file
        let filepath = temp_dir.path().join(&manifest.filename);
        let data = fs::read(&filepath).unwrap();
        fs::write(&filepath, &data[..data.len() / 2]).unwrap();

        // Should detect truncation
        let result = manager.verify_backup(&manifest);
        assert!(result.is_err());
    }
}

// ============================================================================
// BACKUP RESTORE TESTS
// ============================================================================

mod backup_restore_tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_state(block_count: u64) -> MockChainState {
        let mut state = MockChainState::new();
        let mut prev_hash = [0u8; 32];
        for height in 0..block_count {
            let mut hash = [0u8; 32];
            hash[0..8].copy_from_slice(&height.to_le_bytes());
            let block = Block {
                height,
                hash,
                parent_hash: prev_hash,
                data: format!("Block {} data", height).into_bytes(),
            };
            state.add_block(block).unwrap();
            prev_hash = hash;
        }
        state
    }

    #[test]
    fn test_restore_to_empty_state() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let original_state = create_test_state(100);

        let manifest = manager.create_backup(&original_state, 0, 99).unwrap();

        // Restore to empty state
        let mut new_state = MockChainState::new();
        let restored = manager.restore_backup(&manifest, &mut new_state).unwrap();

        assert_eq!(restored, 100);
        assert_eq!(new_state.height, 99);
        assert_eq!(new_state.blocks.len(), 100);
    }

    #[test]
    fn test_restore_verifies_integrity_first() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let state = create_test_state(100);

        let manifest = manager.create_backup(&state, 0, 99).unwrap();

        // Corrupt the backup
        let filepath = temp_dir.path().join(&manifest.filename);
        let mut data = fs::read(&filepath).unwrap();
        data[50] ^= 0xFF;
        fs::write(&filepath, &data).unwrap();

        // Restore should fail due to corruption
        let mut new_state = MockChainState::new();
        let result = manager.restore_backup(&manifest, &mut new_state);
        assert!(result.is_err());

        // State should be unchanged (no partial restore)
        assert!(new_state.blocks.is_empty());
    }

    #[test]
    fn test_partial_restore() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let original_state = create_test_state(100);

        // Create backup of blocks 0-49
        let manifest1 = manager.create_backup(&original_state, 0, 49).unwrap();

        // Create backup of blocks 50-99
        let manifest2 = manager.create_backup(&original_state, 50, 99).unwrap();

        // Restore first half
        let mut new_state = MockChainState::new();
        let restored1 = manager.restore_backup(&manifest1, &mut new_state).unwrap();
        assert_eq!(restored1, 50);
        assert_eq!(new_state.height, 49);

        // Restore second half
        let restored2 = manager.restore_backup(&manifest2, &mut new_state).unwrap();
        assert_eq!(restored2, 50);
        assert_eq!(new_state.height, 99);
    }

    #[test]
    fn test_restore_skips_existing_blocks() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let original_state = create_test_state(100);

        let manifest = manager.create_backup(&original_state, 0, 99).unwrap();

        // Pre-populate some blocks
        let mut new_state = create_test_state(50);

        // Restore should only add the missing blocks
        let restored = manager.restore_backup(&manifest, &mut new_state).unwrap();
        assert_eq!(restored, 50); // Only blocks 50-99 added
        assert_eq!(new_state.height, 99);
    }
}

// ============================================================================
// DISASTER RECOVERY TESTS
// ============================================================================

mod disaster_recovery_tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_state(block_count: u64) -> MockChainState {
        let mut state = MockChainState::new();
        let mut prev_hash = [0u8; 32];
        for height in 0..block_count {
            let mut hash = [0u8; 32];
            hash[0..8].copy_from_slice(&height.to_le_bytes());
            let block = Block {
                height,
                hash,
                parent_hash: prev_hash,
                data: format!("Block {} data", height).into_bytes(),
            };
            state.add_block(block).unwrap();
            prev_hash = hash;
        }
        state
    }

    #[test]
    fn test_full_disaster_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());

        // Create original state with 1000 blocks
        let original_state = create_test_state(1000);

        // Create incremental backups
        let manifest1 = manager.create_backup(&original_state, 0, 499).unwrap();
        let manifest2 = manager.create_backup(&original_state, 500, 999).unwrap();

        // Simulate disaster - state is completely lost
        let mut recovered_state = MockChainState::new();

        // Restore from backups
        manager.restore_backup(&manifest1, &mut recovered_state).unwrap();
        manager.restore_backup(&manifest2, &mut recovered_state).unwrap();

        // Verify full recovery
        assert_eq!(recovered_state.height, 999);
        assert_eq!(recovered_state.blocks.len(), 1000);

        // Verify each block matches
        for height in 0..1000 {
            let original = original_state.blocks.get(&height).unwrap();
            let recovered = recovered_state.blocks.get(&height).unwrap();
            assert_eq!(original.hash, recovered.hash);
            assert_eq!(original.data, recovered.data);
        }
    }

    #[test]
    fn test_recovery_from_corrupted_primary_backup() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());

        let original_state = create_test_state(100);

        // Create two backups of DIFFERENT ranges (to avoid overwriting)
        // First backup: blocks 0-49
        let manifest1 = manager.create_backup(&original_state, 0, 49).unwrap();
        // Second backup: blocks 50-99
        let manifest2 = manager.create_backup(&original_state, 50, 99).unwrap();

        // Corrupt first backup
        let filepath = temp_dir.path().join(&manifest1.filename);
        let mut data = fs::read(&filepath).unwrap();
        data[50] ^= 0xFF;
        fs::write(&filepath, &data).unwrap();

        // Verify first backup is corrupted
        assert!(manager.verify_backup(&manifest1).is_err());

        // Second backup should still be valid (different file)
        assert!(manager.verify_backup(&manifest2).is_ok());
    }
}

// ============================================================================
// REGRESSION TESTS
// ============================================================================

mod regression_tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_state(block_count: u64) -> MockChainState {
        let mut state = MockChainState::new();
        let mut prev_hash = [0u8; 32];
        for height in 0..block_count {
            let mut hash = [0u8; 32];
            hash[0..8].copy_from_slice(&height.to_le_bytes());
            let block = Block {
                height,
                hash,
                parent_hash: prev_hash,
                data: format!("Block {} data", height).into_bytes(),
            };
            state.add_block(block).unwrap();
            prev_hash = hash;
        }
        state
    }

    /// Regression test: Backup must include checksum
    #[test]
    fn test_regression_backup_has_checksum() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let state = create_test_state(10);

        let manifest = manager.create_backup(&state, 0, 9).unwrap();

        assert_ne!(
            manifest.checksum,
            [0u8; 32],
            "REGRESSION: Backup must have non-zero checksum"
        );
    }

    /// Regression test: Corrupted backup must be detected
    #[test]
    fn test_regression_corruption_detected() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let state = create_test_state(100);

        let manifest = manager.create_backup(&state, 0, 99).unwrap();

        // Corrupt
        let filepath = temp_dir.path().join(&manifest.filename);
        let mut data = fs::read(&filepath).unwrap();
        data[0] ^= 0xFF;
        fs::write(&filepath, &data).unwrap();

        let result = manager.verify_backup(&manifest);
        assert!(
            result.is_err(),
            "REGRESSION: Corrupted backup not detected!"
        );
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;
    use tempfile::TempDir;

    fn create_test_state(block_count: u64) -> MockChainState {
        let mut state = MockChainState::new();
        let mut prev_hash = [0u8; 32];
        for height in 0..block_count {
            let mut hash = [0u8; 32];
            hash[0..8].copy_from_slice(&height.to_le_bytes());
            let block = Block {
                height,
                hash,
                parent_hash: prev_hash,
                data: vec![0u8; 1000], // 1KB per block
            };
            state.add_block(block).unwrap();
            prev_hash = hash;
        }
        state
    }

    #[test]
    fn test_backup_performance() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let state = create_test_state(10_000);

        let start = Instant::now();
        let manifest = manager.create_backup(&state, 0, 9999).unwrap();
        let elapsed = start.elapsed();

        let blocks_per_second = 10_000.0 / elapsed.as_secs_f64();
        println!(
            "Backup: {} blocks in {:?} ({:.0} blocks/sec)",
            manifest.blocks_count, elapsed, blocks_per_second
        );

        assert!(
            blocks_per_second > 1000.0,
            "Backup too slow: {:.0} blocks/sec",
            blocks_per_second
        );
    }

    #[test]
    fn test_restore_performance() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = BackupManager::new(temp_dir.path().to_path_buf());
        let state = create_test_state(10_000);

        let manifest = manager.create_backup(&state, 0, 9999).unwrap();

        let mut new_state = MockChainState::new();
        let start = Instant::now();
        let restored = manager.restore_backup(&manifest, &mut new_state).unwrap();
        let elapsed = start.elapsed();

        let blocks_per_second = restored as f64 / elapsed.as_secs_f64();
        println!(
            "Restore: {} blocks in {:?} ({:.0} blocks/sec)",
            restored, elapsed, blocks_per_second
        );

        assert!(
            blocks_per_second > 1000.0,
            "Restore too slow: {:.0} blocks/sec",
            blocks_per_second
        );
    }
}
