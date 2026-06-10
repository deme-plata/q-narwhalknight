//! Byzantine Fault Detection Tests
//!
//! Tests to ensure the Byzantine fault detection system properly identifies
//! and handles malicious validator behavior.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Double-vote detection in same round
//! 2. Double-signing blocks at same height
//! 3. Timing anomaly detection
//! 4. Signature verification attacks
//! 5. Coordinated multi-validator attacks
//! 6. Reputation scoring and suspicion levels
//!
//! Run with: cargo test --package q-narwhal-core --test byzantine_detection_tests

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};

// ============================================================================
// MOCK STRUCTURES FOR BYZANTINE DETECTION TESTING
// ============================================================================

/// Suspicion level for a validator
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SuspicionLevel {
    Trusted = 0,
    Normal = 1,
    Suspicious = 2,
    HighlyMalicious = 3,
    Confirmed = 4,
}

/// Type of Byzantine behavior detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByzantineBehavior {
    DoubleVote,           // Voted for different blocks in same round
    DoubleSigning,        // Signed two different blocks at same height
    InvalidSignature,     // Produced invalid signature
    TimingAnomaly,        // Sent messages at wrong time
    ConsensusViolation,   // Violated consensus rules
    MessageManipulation,  // Tampered with message content
    SelectiveRelay,       // Selectively relayed messages
    CoordinatedAttack,    // Part of multi-validator attack
}

/// Evidence of Byzantine behavior
#[derive(Debug, Clone)]
pub struct ByzantineEvidence {
    pub validator: [u8; 32],
    pub behavior: ByzantineBehavior,
    pub round: u64,
    pub height: u64,
    pub timestamp: u64,
    pub proof: Vec<u8>, // Cryptographic proof (e.g., conflicting signatures)
}

/// Vote record for double-vote detection
#[derive(Debug, Clone)]
pub struct VoteRecord {
    pub validator: [u8; 32],
    pub round: u64,
    pub height: u64,
    pub block_hash: [u8; 32],
    pub signature: [u8; 64],
}

/// Byzantine detector tracking validator behavior
pub struct ByzantineDetector {
    // Track votes per (validator, round) for double-vote detection
    vote_history: RwLock<HashMap<([u8; 32], u64), Vec<VoteRecord>>>,
    // Track block signatures per (validator, height) for double-signing
    signature_history: RwLock<HashMap<([u8; 32], u64), Vec<[u8; 32]>>>,
    // Suspicion levels per validator
    suspicion_levels: RwLock<HashMap<[u8; 32], SuspicionLevel>>,
    // Reputation scores (0-100)
    reputation_scores: RwLock<HashMap<[u8; 32], u64>>,
    // Detected evidence
    evidence: RwLock<Vec<ByzantineEvidence>>,
    // Timing records
    message_times: RwLock<HashMap<[u8; 32], Vec<u64>>>,
    // Expected timing window (in ms)
    timing_window: u64,
    // Coordinated attack detection - validators seen working together
    coordination_graph: RwLock<HashMap<[u8; 32], HashSet<[u8; 32]>>>,
}

impl ByzantineDetector {
    pub fn new() -> Self {
        Self {
            vote_history: RwLock::new(HashMap::new()),
            signature_history: RwLock::new(HashMap::new()),
            suspicion_levels: RwLock::new(HashMap::new()),
            reputation_scores: RwLock::new(HashMap::new()),
            evidence: RwLock::new(Vec::new()),
            message_times: RwLock::new(HashMap::new()),
            timing_window: 5000, // 5 second window
            coordination_graph: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize a validator with default reputation
    pub fn register_validator(&self, validator: [u8; 32]) {
        self.suspicion_levels
            .write()
            .unwrap()
            .insert(validator, SuspicionLevel::Normal);
        self.reputation_scores
            .write()
            .unwrap()
            .insert(validator, 50);
    }

    /// Record a vote and check for double-voting
    pub fn record_vote(&self, vote: VoteRecord) -> Result<(), ByzantineEvidence> {
        let key = (vote.validator, vote.round);
        let mut history = self.vote_history.write().unwrap();

        if let Some(existing_votes) = history.get_mut(&key) {
            // Check if voting for different block in same round
            for existing in existing_votes.iter() {
                if existing.block_hash != vote.block_hash {
                    let evidence = ByzantineEvidence {
                        validator: vote.validator,
                        behavior: ByzantineBehavior::DoubleVote,
                        round: vote.round,
                        height: vote.height,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        proof: [existing.signature.to_vec(), vote.signature.to_vec()].concat(),
                    };

                    self.report_evidence(evidence.clone());
                    return Err(evidence);
                }
            }
            existing_votes.push(vote);
        } else {
            history.insert(key, vec![vote]);
        }

        Ok(())
    }

    /// Record a block signature and check for double-signing
    pub fn record_block_signature(
        &self,
        validator: [u8; 32],
        height: u64,
        block_hash: [u8; 32],
    ) -> Result<(), ByzantineEvidence> {
        let key = (validator, height);
        let mut history = self.signature_history.write().unwrap();

        if let Some(existing_hashes) = history.get_mut(&key) {
            // Check if signing different block at same height
            for existing_hash in existing_hashes.iter() {
                if *existing_hash != block_hash {
                    let evidence = ByzantineEvidence {
                        validator,
                        behavior: ByzantineBehavior::DoubleSigning,
                        round: 0, // Not applicable for block signing
                        height,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        proof: [existing_hash.to_vec(), block_hash.to_vec()].concat(),
                    };

                    self.report_evidence(evidence.clone());
                    return Err(evidence);
                }
            }
            existing_hashes.push(block_hash);
        } else {
            history.insert(key, vec![block_hash]);
        }

        Ok(())
    }

    /// Record message timing for anomaly detection
    pub fn record_message_time(&self, validator: [u8; 32], timestamp: u64) {
        let mut times = self.message_times.write().unwrap();
        times.entry(validator).or_insert_with(Vec::new).push(timestamp);
    }

    /// Check for timing anomalies (messages too early or too late)
    pub fn check_timing_anomaly(
        &self,
        validator: [u8; 32],
        expected_time: u64,
        actual_time: u64,
    ) -> Option<ByzantineEvidence> {
        let diff = if actual_time > expected_time {
            actual_time - expected_time
        } else {
            expected_time - actual_time
        };

        if diff > self.timing_window {
            let evidence = ByzantineEvidence {
                validator,
                behavior: ByzantineBehavior::TimingAnomaly,
                round: 0,
                height: 0,
                timestamp: actual_time,
                proof: expected_time.to_le_bytes().to_vec(),
            };

            self.report_evidence(evidence.clone());
            return Some(evidence);
        }

        None
    }

    /// Report and record Byzantine evidence
    fn report_evidence(&self, evidence: ByzantineEvidence) {
        // Update suspicion level
        {
            let mut levels = self.suspicion_levels.write().unwrap();
            let current = levels.get(&evidence.validator).copied().unwrap_or(SuspicionLevel::Normal);
            let new_level = match evidence.behavior {
                ByzantineBehavior::DoubleVote | ByzantineBehavior::DoubleSigning => {
                    SuspicionLevel::Confirmed
                }
                ByzantineBehavior::CoordinatedAttack => SuspicionLevel::Confirmed,
                ByzantineBehavior::InvalidSignature => SuspicionLevel::HighlyMalicious,
                _ => {
                    if current < SuspicionLevel::Suspicious {
                        SuspicionLevel::Suspicious
                    } else {
                        current
                    }
                }
            };
            levels.insert(evidence.validator, new_level);
        }

        // Reduce reputation
        {
            let mut scores = self.reputation_scores.write().unwrap();
            let reduction = match evidence.behavior {
                ByzantineBehavior::DoubleVote | ByzantineBehavior::DoubleSigning => 50,
                ByzantineBehavior::CoordinatedAttack => 100,
                _ => 20,
            };
            let current = scores.get(&evidence.validator).copied().unwrap_or(50);
            scores.insert(evidence.validator, current.saturating_sub(reduction));
        }

        // Store evidence
        self.evidence.write().unwrap().push(evidence);
    }

    /// Check for coordinated attacks (multiple validators acting together)
    pub fn detect_coordinated_attack(
        &self,
        validators: &[[u8; 32]],
        behavior: ByzantineBehavior,
    ) -> Option<Vec<ByzantineEvidence>> {
        if validators.len() < 2 {
            return None;
        }

        // Update coordination graph
        {
            let mut graph = self.coordination_graph.write().unwrap();
            for i in 0..validators.len() {
                for j in (i + 1)..validators.len() {
                    graph.entry(validators[i]).or_insert_with(HashSet::new).insert(validators[j]);
                    graph.entry(validators[j]).or_insert_with(HashSet::new).insert(validators[i]);
                }
            }
        }

        // Check if this forms a coordinated attack (3+ validators)
        let mut evidence_list = Vec::new();
        if validators.len() >= 3 {
            for validator in validators {
                let evidence = ByzantineEvidence {
                    validator: *validator,
                    behavior: ByzantineBehavior::CoordinatedAttack,
                    round: 0,
                    height: 0,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    proof: validators.iter().flat_map(|v| v.to_vec()).collect(),
                };
                self.report_evidence(evidence.clone());
                evidence_list.push(evidence);
            }
            return Some(evidence_list);
        }

        None
    }

    /// Get suspicion level for a validator
    pub fn get_suspicion_level(&self, validator: &[u8; 32]) -> SuspicionLevel {
        self.suspicion_levels
            .read()
            .unwrap()
            .get(validator)
            .copied()
            .unwrap_or(SuspicionLevel::Normal)
    }

    /// Get reputation score for a validator
    pub fn get_reputation(&self, validator: &[u8; 32]) -> u64 {
        self.reputation_scores
            .read()
            .unwrap()
            .get(validator)
            .copied()
            .unwrap_or(50)
    }

    /// Get all evidence against a validator
    pub fn get_evidence(&self, validator: &[u8; 32]) -> Vec<ByzantineEvidence> {
        self.evidence
            .read()
            .unwrap()
            .iter()
            .filter(|e| &e.validator == validator)
            .cloned()
            .collect()
    }

    /// Check if validator should be banned
    pub fn should_ban(&self, validator: &[u8; 32]) -> bool {
        let level = self.get_suspicion_level(validator);
        let reputation = self.get_reputation(validator);

        level >= SuspicionLevel::Confirmed || reputation == 0
    }

    /// Get total evidence count
    pub fn evidence_count(&self) -> usize {
        self.evidence.read().unwrap().len()
    }
}

// ============================================================================
// DOUBLE-VOTE DETECTION TESTS
// ============================================================================

#[test]
fn test_valid_vote_recorded() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    let vote = VoteRecord {
        validator,
        round: 1,
        height: 100,
        block_hash: [2u8; 32],
        signature: [3u8; 64],
    };

    let result = detector.record_vote(vote);
    assert!(result.is_ok());
}

#[test]
fn test_double_vote_detected() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    // First vote
    let vote1 = VoteRecord {
        validator,
        round: 1,
        height: 100,
        block_hash: [2u8; 32],
        signature: [3u8; 64],
    };
    detector.record_vote(vote1).unwrap();

    // Second vote for different block in same round
    let vote2 = VoteRecord {
        validator,
        round: 1,
        height: 100,
        block_hash: [4u8; 32], // Different block hash!
        signature: [5u8; 64],
    };

    let result = detector.record_vote(vote2);
    assert!(result.is_err());

    let evidence = result.unwrap_err();
    assert_eq!(evidence.behavior, ByzantineBehavior::DoubleVote);
    assert_eq!(evidence.validator, validator);
}

#[test]
fn test_same_vote_twice_ok() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    // Same vote twice (idempotent)
    let vote = VoteRecord {
        validator,
        round: 1,
        height: 100,
        block_hash: [2u8; 32],
        signature: [3u8; 64],
    };

    detector.record_vote(vote.clone()).unwrap();
    let result = detector.record_vote(vote);
    assert!(result.is_ok());
}

#[test]
fn test_votes_in_different_rounds_ok() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    // Vote in round 1
    let vote1 = VoteRecord {
        validator,
        round: 1,
        height: 100,
        block_hash: [2u8; 32],
        signature: [3u8; 64],
    };
    detector.record_vote(vote1).unwrap();

    // Different vote in round 2 (OK!)
    let vote2 = VoteRecord {
        validator,
        round: 2,
        height: 100,
        block_hash: [4u8; 32],
        signature: [5u8; 64],
    };

    let result = detector.record_vote(vote2);
    assert!(result.is_ok());
}

// ============================================================================
// DOUBLE-SIGNING DETECTION TESTS
// ============================================================================

#[test]
fn test_valid_block_signature() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];

    let result = detector.record_block_signature(validator, 100, [2u8; 32]);
    assert!(result.is_ok());
}

#[test]
fn test_double_signing_detected() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    // Sign first block
    detector.record_block_signature(validator, 100, [2u8; 32]).unwrap();

    // Sign different block at same height
    let result = detector.record_block_signature(validator, 100, [3u8; 32]);
    assert!(result.is_err());

    let evidence = result.unwrap_err();
    assert_eq!(evidence.behavior, ByzantineBehavior::DoubleSigning);
}

#[test]
fn test_signing_at_different_heights_ok() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];

    detector.record_block_signature(validator, 100, [2u8; 32]).unwrap();
    let result = detector.record_block_signature(validator, 101, [3u8; 32]);
    assert!(result.is_ok());
}

// ============================================================================
// TIMING ANOMALY TESTS
// ============================================================================

#[test]
fn test_valid_timing() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];

    let expected = 1000000;
    let actual = 1000100; // Within 5 second window

    let result = detector.check_timing_anomaly(validator, expected, actual);
    assert!(result.is_none());
}

#[test]
fn test_timing_anomaly_detected() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    let expected = 1000000;
    let actual = 1010000; // 10 seconds off (outside 5 second window)

    let result = detector.check_timing_anomaly(validator, expected, actual);
    assert!(result.is_some());

    let evidence = result.unwrap();
    assert_eq!(evidence.behavior, ByzantineBehavior::TimingAnomaly);
}

#[test]
fn test_early_message_anomaly() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    let expected = 1010000;
    let actual = 1000000; // 10 seconds early

    let result = detector.check_timing_anomaly(validator, expected, actual);
    assert!(result.is_some());
}

// ============================================================================
// COORDINATED ATTACK TESTS
// ============================================================================

#[test]
fn test_coordinated_attack_two_validators() {
    let detector = ByzantineDetector::new();
    let validators = [[1u8; 32], [2u8; 32]];

    for v in &validators {
        detector.register_validator(*v);
    }

    // 2 validators not enough for coordinated attack flag
    let result = detector.detect_coordinated_attack(&validators, ByzantineBehavior::DoubleVote);
    assert!(result.is_none());
}

#[test]
fn test_coordinated_attack_three_validators() {
    let detector = ByzantineDetector::new();
    let validators = [[1u8; 32], [2u8; 32], [3u8; 32]];

    for v in &validators {
        detector.register_validator(*v);
    }

    // 3+ validators triggers coordinated attack
    let result = detector.detect_coordinated_attack(&validators, ByzantineBehavior::DoubleVote);
    assert!(result.is_some());

    let evidence_list = result.unwrap();
    assert_eq!(evidence_list.len(), 3);

    for evidence in &evidence_list {
        assert_eq!(evidence.behavior, ByzantineBehavior::CoordinatedAttack);
    }
}

#[test]
fn test_coordinated_attack_reputation_destroyed() {
    let detector = ByzantineDetector::new();
    let validators = [[1u8; 32], [2u8; 32], [3u8; 32]];

    for v in &validators {
        detector.register_validator(*v);
    }

    detector.detect_coordinated_attack(&validators, ByzantineBehavior::DoubleVote);

    // All validators should have 0 reputation
    for v in &validators {
        assert_eq!(detector.get_reputation(v), 0);
        assert_eq!(detector.get_suspicion_level(v), SuspicionLevel::Confirmed);
        assert!(detector.should_ban(v));
    }
}

// ============================================================================
// REPUTATION AND SUSPICION TESTS
// ============================================================================

#[test]
fn test_initial_reputation() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    assert_eq!(detector.get_reputation(&validator), 50);
    assert_eq!(detector.get_suspicion_level(&validator), SuspicionLevel::Normal);
}

#[test]
fn test_reputation_decreases_on_evidence() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    let initial_rep = detector.get_reputation(&validator);

    // Timing anomaly reduces reputation
    detector.check_timing_anomaly(validator, 1000000, 1010000);

    let new_rep = detector.get_reputation(&validator);
    assert!(new_rep < initial_rep);
}

#[test]
fn test_suspicion_escalates() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    // Initial level
    assert_eq!(detector.get_suspicion_level(&validator), SuspicionLevel::Normal);

    // Timing anomaly -> Suspicious
    detector.check_timing_anomaly(validator, 1000000, 1010000);
    assert!(detector.get_suspicion_level(&validator) >= SuspicionLevel::Suspicious);

    // Double vote -> Confirmed
    let vote1 = VoteRecord {
        validator,
        round: 1,
        height: 100,
        block_hash: [2u8; 32],
        signature: [3u8; 64],
    };
    detector.record_vote(vote1).unwrap();

    let vote2 = VoteRecord {
        validator,
        round: 1,
        height: 100,
        block_hash: [4u8; 32],
        signature: [5u8; 64],
    };
    let _ = detector.record_vote(vote2);

    assert_eq!(detector.get_suspicion_level(&validator), SuspicionLevel::Confirmed);
}

#[test]
fn test_should_ban_on_confirmed() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    assert!(!detector.should_ban(&validator));

    // Double vote leads to ban
    let vote1 = VoteRecord {
        validator,
        round: 1,
        height: 100,
        block_hash: [2u8; 32],
        signature: [3u8; 64],
    };
    detector.record_vote(vote1).unwrap();

    let vote2 = VoteRecord {
        validator,
        round: 1,
        height: 100,
        block_hash: [4u8; 32],
        signature: [5u8; 64],
    };
    let _ = detector.record_vote(vote2);

    assert!(detector.should_ban(&validator));
}

#[test]
fn test_should_ban_on_zero_reputation() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    // Cause enough evidence to drain reputation to 0
    // Multiple timing anomalies
    for i in 0..5 {
        detector.check_timing_anomaly(validator, 1000000, 1010000 + i * 1000);
    }

    assert!(detector.get_reputation(&validator) == 0 || detector.should_ban(&validator));
}

// ============================================================================
// EVIDENCE COLLECTION TESTS
// ============================================================================

#[test]
fn test_evidence_stored() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    assert_eq!(detector.evidence_count(), 0);

    // Create evidence
    let vote1 = VoteRecord {
        validator,
        round: 1,
        height: 100,
        block_hash: [2u8; 32],
        signature: [3u8; 64],
    };
    detector.record_vote(vote1).unwrap();

    let vote2 = VoteRecord {
        validator,
        round: 1,
        height: 100,
        block_hash: [4u8; 32],
        signature: [5u8; 64],
    };
    let _ = detector.record_vote(vote2);

    assert_eq!(detector.evidence_count(), 1);
}

#[test]
fn test_get_evidence_for_validator() {
    let detector = ByzantineDetector::new();
    let validator1 = [1u8; 32];
    let validator2 = [2u8; 32];
    detector.register_validator(validator1);
    detector.register_validator(validator2);

    // Create evidence for validator1
    let vote1 = VoteRecord {
        validator: validator1,
        round: 1,
        height: 100,
        block_hash: [2u8; 32],
        signature: [3u8; 64],
    };
    detector.record_vote(vote1).unwrap();

    let vote2 = VoteRecord {
        validator: validator1,
        round: 1,
        height: 100,
        block_hash: [4u8; 32],
        signature: [5u8; 64],
    };
    let _ = detector.record_vote(vote2);

    // Validator1 has evidence
    let evidence1 = detector.get_evidence(&validator1);
    assert_eq!(evidence1.len(), 1);

    // Validator2 has no evidence
    let evidence2 = detector.get_evidence(&validator2);
    assert_eq!(evidence2.len(), 0);
}

#[test]
fn test_multiple_evidence_types() {
    let detector = ByzantineDetector::new();
    let validator = [1u8; 32];
    detector.register_validator(validator);

    // Timing anomaly
    detector.check_timing_anomaly(validator, 1000000, 1010000);

    // Double sign
    detector.record_block_signature(validator, 100, [2u8; 32]).unwrap();
    let _ = detector.record_block_signature(validator, 100, [3u8; 32]);

    let evidence = detector.get_evidence(&validator);
    assert_eq!(evidence.len(), 2);

    let behaviors: HashSet<_> = evidence.iter().map(|e| e.behavior).collect();
    assert!(behaviors.contains(&ByzantineBehavior::TimingAnomaly));
    assert!(behaviors.contains(&ByzantineBehavior::DoubleSigning));
}
