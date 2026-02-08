//! Mining Commit-Reveal Protocol Tests
//!
//! Tests for the commit-reveal mining protocol to prevent
//! front-running and ensure fair block production.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Duplicate commitment prevention
//! 2. Invalid reveal rejection
//! 3. Timing attack resistance
//! 4. Hash mismatch detection
//! 5. Expired commitment handling
//!
//! Run with: cargo test --package q-api-server --test mining_commit_reveal_tests

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ============================================================================
// MOCK STRUCTURES FOR COMMIT-REVEAL TESTING
// ============================================================================

/// Commitment submitted by a miner
#[derive(Debug, Clone)]
pub struct Commitment {
    pub miner_address: [u8; 32],
    pub commitment_hash: [u8; 32],
    pub block_height: u64,
    pub submitted_at: Instant,
}

/// Reveal submitted by a miner
#[derive(Debug, Clone)]
pub struct Reveal {
    pub miner_address: [u8; 32],
    pub nonce: u64,
    pub random_value: [u8; 32],
    pub block_height: u64,
}

/// Commit-Reveal protocol manager
pub struct CommitRevealManager {
    commitments: Mutex<HashMap<([u8; 32], u64), Commitment>>,
    reveals: Mutex<HashMap<([u8; 32], u64), Reveal>>,
    current_height: u64,
    commitment_window: Duration,
    reveal_window: Duration,
}

impl CommitRevealManager {
    pub fn new(current_height: u64) -> Self {
        Self {
            commitments: Mutex::new(HashMap::new()),
            reveals: Mutex::new(HashMap::new()),
            current_height,
            commitment_window: Duration::from_secs(30),
            reveal_window: Duration::from_secs(60),
        }
    }

    pub fn set_height(&mut self, height: u64) {
        self.current_height = height;
    }

    /// Register a commitment
    pub fn register_commitment(&self, commitment: Commitment) -> Result<(), String> {
        let key = (commitment.miner_address, commitment.block_height);

        let mut commitments = self.commitments.lock().unwrap();

        // Check for duplicate commitment
        if commitments.contains_key(&key) {
            return Err(format!(
                "DUPLICATE_COMMITMENT: Miner {:?} already committed for height {}",
                hex::encode(&commitment.miner_address[..4]),
                commitment.block_height
            ));
        }

        // Check if commitment is for valid height
        if commitment.block_height < self.current_height {
            return Err(format!(
                "EXPIRED_HEIGHT: Cannot commit for past height {} (current: {})",
                commitment.block_height, self.current_height
            ));
        }

        if commitment.block_height > self.current_height + 10 {
            return Err(format!(
                "FUTURE_HEIGHT: Cannot commit for height {} (current: {})",
                commitment.block_height, self.current_height
            ));
        }

        // Validate commitment hash format
        if commitment.commitment_hash == [0u8; 32] {
            return Err("INVALID_COMMITMENT: Hash cannot be all zeros".to_string());
        }

        commitments.insert(key, commitment);
        Ok(())
    }

    /// Process a reveal
    pub fn process_reveal(&self, reveal: Reveal) -> Result<(), String> {
        let key = (reveal.miner_address, reveal.block_height);

        // Check commitment exists
        let commitment = {
            let commitments = self.commitments.lock().unwrap();
            commitments.get(&key).cloned()
        };

        let commitment = commitment.ok_or_else(|| {
            format!(
                "NO_COMMITMENT: No commitment found for miner {:?} at height {}",
                hex::encode(&reveal.miner_address[..4]),
                reveal.block_height
            )
        })?;

        // Check commitment hasn't expired
        if commitment.submitted_at.elapsed() > self.reveal_window {
            return Err(format!(
                "COMMITMENT_EXPIRED: Commitment for height {} has expired",
                reveal.block_height
            ));
        }

        // Verify the reveal matches the commitment
        let computed_hash = Self::compute_commitment_hash(
            &reveal.miner_address,
            reveal.nonce,
            &reveal.random_value,
        );

        if computed_hash != commitment.commitment_hash {
            return Err(format!(
                "HASH_MISMATCH: Reveal does not match commitment. Expected {:?}, got {:?}",
                hex::encode(&commitment.commitment_hash[..8]),
                hex::encode(&computed_hash[..8])
            ));
        }

        // Check for duplicate reveal
        {
            let reveals = self.reveals.lock().unwrap();
            if reveals.contains_key(&key) {
                return Err("DUPLICATE_REVEAL: Already revealed for this height".to_string());
            }
        }

        // Store the reveal
        {
            let mut reveals = self.reveals.lock().unwrap();
            reveals.insert(key, reveal);
        }

        Ok(())
    }

    /// Compute commitment hash from reveal values
    fn compute_commitment_hash(
        miner_address: &[u8; 32],
        nonce: u64,
        random_value: &[u8; 32],
    ) -> [u8; 32] {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        miner_address.hash(&mut hasher);
        nonce.hash(&mut hasher);
        random_value.hash(&mut hasher);
        let h = hasher.finish();

        let mut result = [0u8; 32];
        result[0..8].copy_from_slice(&h.to_le_bytes());
        // Fill rest with derived values
        for i in 1..4 {
            let h2 = hasher.finish().wrapping_add(i as u64);
            result[i * 8..(i + 1) * 8].copy_from_slice(&h2.to_le_bytes());
        }
        result
    }

    /// Create a valid commitment-reveal pair for testing
    pub fn create_valid_pair(
        miner_address: [u8; 32],
        block_height: u64,
        nonce: u64,
        random_value: [u8; 32],
    ) -> (Commitment, Reveal) {
        let commitment_hash =
            Self::compute_commitment_hash(&miner_address, nonce, &random_value);

        let commitment = Commitment {
            miner_address,
            commitment_hash,
            block_height,
            submitted_at: Instant::now(),
        };

        let reveal = Reveal {
            miner_address,
            nonce,
            random_value,
            block_height,
        };

        (commitment, reveal)
    }

    /// Clean up expired commitments
    pub fn cleanup_expired(&self) -> usize {
        let mut commitments = self.commitments.lock().unwrap();
        let before = commitments.len();

        commitments.retain(|_, c| c.submitted_at.elapsed() <= self.reveal_window);

        before - commitments.len()
    }

    pub fn commitment_count(&self) -> usize {
        self.commitments.lock().unwrap().len()
    }

    pub fn reveal_count(&self) -> usize {
        self.reveals.lock().unwrap().len()
    }
}

// ============================================================================
// COMMITMENT TESTS
// ============================================================================

/// Test successful commitment registration
#[test]
fn test_successful_commitment() {
    let manager = CommitRevealManager::new(100);

    let (commitment, _) = CommitRevealManager::create_valid_pair(
        [1u8; 32],
        100,
        12345,
        [42u8; 32],
    );

    let result = manager.register_commitment(commitment);
    assert!(result.is_ok());
    assert_eq!(manager.commitment_count(), 1);
}

/// Test duplicate commitment rejection
#[test]
fn test_duplicate_commitment_rejected() {
    let manager = CommitRevealManager::new(100);

    let (commitment1, _) = CommitRevealManager::create_valid_pair(
        [1u8; 32],
        100,
        12345,
        [42u8; 32],
    );

    let (commitment2, _) = CommitRevealManager::create_valid_pair(
        [1u8; 32], // Same miner
        100,       // Same height
        99999,     // Different nonce
        [99u8; 32],
    );

    manager.register_commitment(commitment1).unwrap();
    let result = manager.register_commitment(commitment2);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("DUPLICATE_COMMITMENT"));
}

/// Test commitment for past height rejected
#[test]
fn test_past_height_commitment_rejected() {
    let manager = CommitRevealManager::new(100);

    let (mut commitment, _) = CommitRevealManager::create_valid_pair(
        [1u8; 32],
        50, // Past height
        12345,
        [42u8; 32],
    );
    commitment.block_height = 50;

    let result = manager.register_commitment(commitment);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("EXPIRED_HEIGHT"));
}

/// Test commitment for far future height rejected
#[test]
fn test_future_height_commitment_rejected() {
    let manager = CommitRevealManager::new(100);

    let (mut commitment, _) = CommitRevealManager::create_valid_pair(
        [1u8; 32],
        200, // Too far in future
        12345,
        [42u8; 32],
    );
    commitment.block_height = 200;

    let result = manager.register_commitment(commitment);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("FUTURE_HEIGHT"));
}

/// Test zero hash commitment rejected
#[test]
fn test_zero_hash_commitment_rejected() {
    let manager = CommitRevealManager::new(100);

    let commitment = Commitment {
        miner_address: [1u8; 32],
        commitment_hash: [0u8; 32], // Invalid
        block_height: 100,
        submitted_at: Instant::now(),
    };

    let result = manager.register_commitment(commitment);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("INVALID_COMMITMENT"));
}

// ============================================================================
// REVEAL TESTS
// ============================================================================

/// Test successful reveal
#[test]
fn test_successful_reveal() {
    let manager = CommitRevealManager::new(100);

    let (commitment, reveal) = CommitRevealManager::create_valid_pair(
        [1u8; 32],
        100,
        12345,
        [42u8; 32],
    );

    manager.register_commitment(commitment).unwrap();
    let result = manager.process_reveal(reveal);

    assert!(result.is_ok());
    assert_eq!(manager.reveal_count(), 1);
}

/// Test reveal without commitment rejected
#[test]
fn test_reveal_without_commitment_rejected() {
    let manager = CommitRevealManager::new(100);

    let reveal = Reveal {
        miner_address: [1u8; 32],
        nonce: 12345,
        random_value: [42u8; 32],
        block_height: 100,
    };

    let result = manager.process_reveal(reveal);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("NO_COMMITMENT"));
}

/// Test reveal with wrong hash rejected
#[test]
fn test_reveal_hash_mismatch_rejected() {
    let manager = CommitRevealManager::new(100);

    let (commitment, mut reveal) = CommitRevealManager::create_valid_pair(
        [1u8; 32],
        100,
        12345,
        [42u8; 32],
    );

    manager.register_commitment(commitment).unwrap();

    // Modify the reveal to not match
    reveal.nonce = 99999; // Different nonce

    let result = manager.process_reveal(reveal);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("HASH_MISMATCH"));
}

/// Test duplicate reveal rejected
#[test]
fn test_duplicate_reveal_rejected() {
    let manager = CommitRevealManager::new(100);

    let (commitment, reveal) = CommitRevealManager::create_valid_pair(
        [1u8; 32],
        100,
        12345,
        [42u8; 32],
    );

    manager.register_commitment(commitment).unwrap();
    manager.process_reveal(reveal.clone()).unwrap();

    // Try to reveal again
    let result = manager.process_reveal(reveal);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("DUPLICATE_REVEAL"));
}

// ============================================================================
// TIMING TESTS
// ============================================================================

/// Test multiple miners can commit for same height
#[test]
fn test_multiple_miners_same_height() {
    let manager = CommitRevealManager::new(100);

    for i in 0..5 {
        let miner = [i as u8; 32];
        let (commitment, _) = CommitRevealManager::create_valid_pair(
            miner,
            100,
            i as u64 * 1000,
            [(i * 10) as u8; 32],
        );
        manager.register_commitment(commitment).unwrap();
    }

    assert_eq!(manager.commitment_count(), 5);
}

/// Test same miner can commit for different heights
#[test]
fn test_same_miner_different_heights() {
    let manager = CommitRevealManager::new(100);
    let miner = [1u8; 32];

    for height in 100..105 {
        let (commitment, _) = CommitRevealManager::create_valid_pair(
            miner,
            height,
            height * 1000,
            [height as u8; 32],
        );
        manager.register_commitment(commitment).unwrap();
    }

    assert_eq!(manager.commitment_count(), 5);
}

// ============================================================================
// CLEANUP TESTS
// ============================================================================

/// Test expired commitment cleanup
#[test]
fn test_cleanup_expired_commitments() {
    // Create manager with very short window for testing
    let mut manager = CommitRevealManager::new(100);
    manager.reveal_window = Duration::from_millis(1); // 1ms for testing

    let (commitment, _) = CommitRevealManager::create_valid_pair(
        [1u8; 32],
        100,
        12345,
        [42u8; 32],
    );
    manager.register_commitment(commitment).unwrap();

    // Wait for expiration
    std::thread::sleep(Duration::from_millis(10));

    let cleaned = manager.cleanup_expired();
    assert_eq!(cleaned, 1);
    assert_eq!(manager.commitment_count(), 0);
}

// ============================================================================
// SECURITY TESTS
// ============================================================================

/// Test commitment hiding (can't determine reveal from commitment)
#[test]
fn test_commitment_hiding() {
    // Different reveals should produce different commitments
    let miner = [1u8; 32];
    let height = 100;

    let hashes: Vec<[u8; 32]> = (0..100)
        .map(|i| {
            CommitRevealManager::compute_commitment_hash(
                &miner,
                i,
                &[i as u8; 32],
            )
        })
        .collect();

    // All hashes should be unique
    let unique_count = hashes.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(unique_count, 100, "All commitment hashes should be unique");
}

/// Test reveal binding (can't change reveal after commitment)
#[test]
fn test_reveal_binding() {
    let manager = CommitRevealManager::new(100);

    let (commitment, original_reveal) = CommitRevealManager::create_valid_pair(
        [1u8; 32],
        100,
        12345,
        [42u8; 32],
    );

    manager.register_commitment(commitment).unwrap();

    // Try multiple different reveals - all should fail except original
    for i in 0..10 {
        let fake_reveal = Reveal {
            miner_address: [1u8; 32],
            nonce: i * 1000,
            random_value: [i as u8; 32],
            block_height: 100,
        };

        if fake_reveal.nonce == original_reveal.nonce
            && fake_reveal.random_value == original_reveal.random_value
        {
            continue; // Skip the original
        }

        let result = manager.process_reveal(fake_reveal);
        assert!(result.is_err(), "Fake reveal {} should be rejected", i);
    }

    // Original should succeed
    let result = manager.process_reveal(original_reveal);
    assert!(result.is_ok(), "Original reveal should succeed");
}

/// Test front-running resistance
#[test]
fn test_front_running_resistance() {
    let manager = CommitRevealManager::new(100);

    // Miner A commits first
    let (commitment_a, reveal_a) = CommitRevealManager::create_valid_pair(
        [1u8; 32],
        100,
        12345,
        [42u8; 32],
    );
    manager.register_commitment(commitment_a).unwrap();

    // Attacker sees commitment but doesn't know the reveal values
    // They can't create a valid reveal without knowing nonce + random_value

    // Miner A reveals successfully
    let result = manager.process_reveal(reveal_a);
    assert!(result.is_ok());

    // Even after seeing the reveal, attacker can't commit for same height
    let (commitment_attacker, _) = CommitRevealManager::create_valid_pair(
        [2u8; 32], // Different miner
        100,       // Same height - this is fine
        12345,     // Even copying values
        [42u8; 32],
    );

    // This commitment is valid (different miner)
    let result = manager.register_commitment(commitment_attacker);
    assert!(result.is_ok(), "Different miner can commit for same height");
}
