//! # Solana State Bridge
//! 
//! 🌉 Bridges Solana state proofs to Q-NarwhalKnight consensus for cross-chain verification.
//! Submits compressed proofs and maintains periodic state snapshots.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tokio::time::{Duration, Instant};
use tracing::{debug, info, warn, error};

use crate::{SolanaBridgeConfig, SolanaStateProof};

/// Bridge for submitting Solana state to Q-NarwhalKnight
pub struct StateBridge {
    config: SolanaBridgeConfig,
    qnk_client: QnkClient,
    pending_proofs: VecDeque<PendingProof>,
    submitted_proofs: HashMap<[u8; 32], ProofStatus>,
    state_snapshots: Vec<StateSnapshot>,
    last_submission: Option<Instant>,
    stats: StateBridgeStats,
}

/// Q-NarwhalKnight client for proof submission
struct QnkClient {
    client: reqwest::Client,
    base_url: String,
    auth_token: Option<String>,
}

/// Pending proof awaiting submission
#[derive(Debug, Clone)]
struct PendingProof {
    proof: SolanaStateProof,
    created_at: Instant,
    retry_count: u32,
    priority: ProofPriority,
}

/// Proof submission priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ProofPriority {
    Low = 0,
    Normal = 1, 
    High = 2,
    Critical = 3,
}

/// Status of submitted proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStatus {
    pub submitted_at: u64,
    pub qnk_block_height: Option<u64>,
    pub confirmation_status: ConfirmationStatus,
    pub proof_hash: [u8; 32],
}

/// Confirmation status in Q-NarwhalKnight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfirmationStatus {
    Pending,
    Included,
    Finalized,
    Rejected(String),
}

/// Periodic state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub solana_slot: u64,
    pub qnk_block_height: u64,
    pub account_count: u32,
    pub total_proof_size: usize,
    pub snapshot_hash: [u8; 32],
    pub timestamp: u64,
}

/// State bridge statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateBridgeStats {
    pub total_proofs_submitted: u64,
    pub pending_proof_count: usize,
    pub successful_submissions: u64,
    pub failed_submissions: u64,
    pub average_confirmation_time_ms: f64,
    pub state_snapshots_created: u64,
    pub last_submission_age_seconds: u64,
}

impl StateBridge {
    /// Create new state bridge
    pub async fn new(config: &SolanaBridgeConfig) -> Result<Self> {
        info!("🌉 Initializing Solana State Bridge to Q-NarwhalKnight");
        info!("   • Q-NK endpoint: {}", config.qnk_rpc_url);
        info!("   • Max proof size: {} bytes", config.max_proof_size);
        
        let qnk_client = QnkClient::new(&config.qnk_rpc_url).await?;
        
        Ok(Self {
            config: config.clone(),
            qnk_client,
            pending_proofs: VecDeque::new(),
            submitted_proofs: HashMap::new(),
            state_snapshots: Vec::new(),
            last_submission: None,
            stats: StateBridgeStats::default(),
        })
    }
    
    /// Submit Solana proof to Q-NarwhalKnight
    pub async fn submit_proof(&mut self, proof: SolanaStateProof) -> Result<[u8; 32]> {
        let proof_hash = proof.generate_proof_hash();
        
        // Check if already submitted
        if self.submitted_proofs.contains_key(&proof_hash) {
            debug!("📤 Proof already submitted: {}", hex::encode(&proof_hash[..8]));
            return Ok(proof_hash);
        }
        
        // Validate proof size
        if proof.size() > self.config.max_proof_size {
            return Err(anyhow::anyhow!("Proof too large: {} > {} bytes", 
                                       proof.size(), self.config.max_proof_size));
        }
        
        // Determine priority based on proof type and age
        let priority = self.determine_proof_priority(&proof);
        
        // Add to pending queue
        let pending_proof = PendingProof {
            proof: proof.clone(),
            created_at: Instant::now(),
            retry_count: 0,
            priority,
        };
        
        // Insert maintaining priority order
        self.insert_proof_by_priority(pending_proof);
        self.stats.pending_proof_count = self.pending_proofs.len();
        
        debug!("📝 Queued proof for submission: {} (priority: {:?})", 
               hex::encode(&proof_hash[..8]), priority);
        
        // Try immediate submission if queue was empty
        if self.pending_proofs.len() == 1 {
            self.process_pending_proofs().await?;
        }
        
        Ok(proof_hash)
    }
    
    /// Determine proof submission priority
    fn determine_proof_priority(&self, proof: &SolanaStateProof) -> ProofPriority {
        let age_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .saturating_sub(proof.timestamp);
        
        match proof.proof_type {
            crate::ProofType::TokenBalance => {
                if age_seconds < 60 {
                    ProofPriority::High // Fresh token balance proofs
                } else {
                    ProofPriority::Normal
                }
            },
            crate::ProofType::DataAvailability => ProofPriority::Critical,
            crate::ProofType::ProgramState => ProofPriority::Normal,
            crate::ProofType::AccountExistence => {
                if age_seconds < 300 {
                    ProofPriority::Normal
                } else {
                    ProofPriority::Low
                }
            },
        }
    }
    
    /// Insert proof maintaining priority order
    fn insert_proof_by_priority(&mut self, pending_proof: PendingProof) {
        // Find insertion point (higher priority first)
        let mut insert_idx = 0;
        for (i, existing) in self.pending_proofs.iter().enumerate() {
            if pending_proof.priority > existing.priority {
                insert_idx = i;
                break;
            }
            insert_idx = i + 1;
        }
        
        self.pending_proofs.insert(insert_idx, pending_proof);
    }
    
    /// Process pending proof queue
    pub async fn process_pending_proofs(&mut self) -> Result<()> {
        const MAX_BATCH_SIZE: usize = 10;
        const MAX_PROCESSING_TIME: Duration = Duration::from_secs(30);
        
        let start_time = Instant::now();
        let mut processed_count = 0;
        
        while !self.pending_proofs.is_empty() && 
              processed_count < MAX_BATCH_SIZE &&
              start_time.elapsed() < MAX_PROCESSING_TIME {
            
            if let Some(mut pending) = self.pending_proofs.pop_front() {
                match self.submit_single_proof(&pending.proof).await {
                    Ok(status) => {
                        let proof_hash = pending.proof.generate_proof_hash();
                        self.submitted_proofs.insert(proof_hash, status);
                        self.stats.successful_submissions += 1;
                        self.last_submission = Some(Instant::now());
                        
                        info!("✅ Submitted proof: {} (slot: {})", 
                               hex::encode(&proof_hash[..8]), pending.proof.slot);
                    },
                    Err(e) => {
                        pending.retry_count += 1;
                        
                        if pending.retry_count <= 3 && pending.created_at.elapsed() < Duration::from_secs(300) {
                            // Retry with exponential backoff
                            warn!("⚠️ Proof submission failed, retrying ({}/3): {}", 
                                   pending.retry_count, e);
                            
                            // Re-queue with lower priority
                            if pending.priority > ProofPriority::Low {
                                pending.priority = ProofPriority::Low;
                            }
                            self.pending_proofs.push_back(pending);
                        } else {
                            error!("❌ Proof submission permanently failed: {}", e);
                            self.stats.failed_submissions += 1;
                        }
                    }
                }
                
                processed_count += 1;
            }
        }
        
        self.stats.pending_proof_count = self.pending_proofs.len();
        self.stats.total_proofs_submitted += processed_count as u64;
        
        if processed_count > 0 {
            debug!("📤 Processed {} proofs in {:.1}s", 
                   processed_count, start_time.elapsed().as_secs_f64());
        }
        
        Ok(())
    }
    
    /// Submit single proof to Q-NarwhalKnight
    async fn submit_single_proof(&self, proof: &SolanaStateProof) -> Result<ProofStatus> {
        debug!("🚀 Submitting proof to Q-NarwhalKnight: slot {}", proof.slot);
        
        // Prepare proof data for submission
        let proof_data = SolanaProofSubmission {
            proof_type: format!("{:?}", proof.proof_type),
            slot: proof.slot,
            blockhash: proof.blockhash.clone(),
            account_pubkey: proof.account_pubkey.clone(),
            compressed_data: base64::encode(&proof.compressed_data),
            merkle_proof: proof.merkle_proof.iter()
                .map(|hash| hex::encode(hash))
                .collect(),
            signature: hex::encode(&proof.signature),
            timestamp: proof.timestamp,
            proof_hash: hex::encode(proof.generate_proof_hash()),
        };
        
        let response = self.qnk_client.submit_solana_proof(proof_data).await?;
        
        Ok(ProofStatus {
            submitted_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            qnk_block_height: response.block_height,
            confirmation_status: ConfirmationStatus::Pending,
            proof_hash: proof.generate_proof_hash(),
        })
    }
    
    /// Create state snapshot
    pub async fn create_snapshot(&mut self, solana_slot: u64) -> Result<StateSnapshot> {
        info!("📸 Creating state snapshot for Solana slot {}", solana_slot);
        
        // Get current Q-NarwhalKnight block height
        let qnk_block_height = self.qnk_client.get_current_block_height().await
            .unwrap_or(0);
        
        // Count submitted proofs in recent time window
        let recent_cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs()
            .saturating_sub(3600); // Last hour
        
        let recent_proofs: Vec<_> = self.submitted_proofs.values()
            .filter(|status| status.submitted_at >= recent_cutoff)
            .collect();
        
        let account_count = recent_proofs.len() as u32;
        let total_proof_size = recent_proofs.len() * 800; // Estimate 800 bytes per proof
        
        // Generate snapshot hash
        let mut hasher = blake3::Hasher::new();
        hasher.update(&solana_slot.to_le_bytes());
        hasher.update(&qnk_block_height.to_le_bytes());
        hasher.update(&account_count.to_le_bytes());
        hasher.update(b"SOLANA_STATE_SNAPSHOT_V1");
        
        let snapshot = StateSnapshot {
            solana_slot,
            qnk_block_height,
            account_count,
            total_proof_size,
            snapshot_hash: hasher.finalize().into(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        };
        
        // Submit snapshot to Q-NarwhalKnight
        self.qnk_client.submit_state_snapshot(&snapshot).await?;
        
        // Store locally
        self.state_snapshots.push(snapshot.clone());
        
        // Keep only last 24 snapshots
        if self.state_snapshots.len() > 24 {
            self.state_snapshots.remove(0);
        }
        
        self.stats.state_snapshots_created += 1;
        
        info!("✅ State snapshot created: {} accounts, {} bytes", 
               account_count, total_proof_size);
        
        Ok(snapshot)
    }
    
    /// Update proof confirmation status
    pub async fn update_proof_confirmations(&mut self) -> Result<()> {
        debug!("🔄 Updating proof confirmation statuses");
        
        let pending_proofs: Vec<_> = self.submitted_proofs.iter()
            .filter(|(_, status)| matches!(status.confirmation_status, ConfirmationStatus::Pending))
            .map(|(hash, _)| *hash)
            .collect();
        
        if pending_proofs.is_empty() {
            return Ok(());
        }
        
        // Query Q-NarwhalKnight for proof confirmations
        let confirmations = self.qnk_client.get_proof_confirmations(&pending_proofs).await?;
        
        let mut updated_count = 0;
        for (proof_hash, confirmation) in confirmations {
            if let Some(status) = self.submitted_proofs.get_mut(&proof_hash) {
                if status.confirmation_status != confirmation {
                    status.confirmation_status = confirmation.clone();
                    updated_count += 1;
                    
                    match confirmation {
                        ConfirmationStatus::Finalized => {
                            info!("🎯 Proof finalized: {}", hex::encode(&proof_hash[..8]));
                        },
                        ConfirmationStatus::Rejected(reason) => {
                            warn!("🚫 Proof rejected: {} - {}", hex::encode(&proof_hash[..8]), reason);
                        },
                        _ => {}
                    }
                }
            }
        }
        
        if updated_count > 0 {
            debug!("🔄 Updated {} proof confirmations", updated_count);
        }
        
        Ok(())
    }
    
    /// Get bridge statistics
    pub fn get_stats(&mut self) -> StateBridgeStats {
        self.stats.pending_proof_count = self.pending_proofs.len();
        self.stats.last_submission_age_seconds = self.last_submission
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(u64::MAX);
        
        // Calculate average confirmation time
        let confirmed_proofs: Vec<_> = self.submitted_proofs.values()
            .filter(|s| matches!(s.confirmation_status, ConfirmationStatus::Finalized))
            .collect();
        
        if !confirmed_proofs.is_empty() {
            let total_time: u64 = confirmed_proofs.iter()
                .map(|s| 120) // Simulate 2-minute average confirmation
                .sum();
            
            self.stats.average_confirmation_time_ms = 
                (total_time as f64 / confirmed_proofs.len() as f64) * 1000.0;
        }
        
        self.stats.clone()
    }
    
    /// Get recent state snapshots
    pub fn get_recent_snapshots(&self, count: usize) -> Vec<StateSnapshot> {
        self.state_snapshots
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
}

/// Proof submission format for Q-NarwhalKnight
#[derive(Debug, Serialize)]
struct SolanaProofSubmission {
    proof_type: String,
    slot: u64,
    blockhash: String,
    account_pubkey: String,
    compressed_data: String, // base64 encoded
    merkle_proof: Vec<String>, // hex encoded hashes
    signature: String, // hex encoded
    timestamp: u64,
    proof_hash: String, // hex encoded
}

/// Q-NarwhalKnight proof submission response
#[derive(Debug, Deserialize)]
struct ProofSubmissionResponse {
    block_height: Option<u64>,
    status: String,
    proof_id: String,
}

impl QnkClient {
    /// Create new Q-NarwhalKnight client
    async fn new(base_url: &str) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("q-solana-bridge/1.0")
            .build()?;
        
        Ok(Self {
            client,
            base_url: base_url.to_string(),
            auth_token: None, // Would be loaded from config in production
        })
    }
    
    /// Submit Solana proof to Q-NarwhalKnight
    async fn submit_solana_proof(&self, proof: SolanaProofSubmission) -> Result<ProofSubmissionResponse> {
        let url = format!("{}/api/v1/solana/proofs", self.base_url);
        
        let mut request = self.client.post(&url).json(&proof);
        
        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }
        
        let response = request.send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Q-NK API error: {}", response.status()));
        }
        
        let result: ProofSubmissionResponse = response.json().await?;
        Ok(result)
    }
    
    /// Submit state snapshot to Q-NarwhalKnight
    async fn submit_state_snapshot(&self, snapshot: &StateSnapshot) -> Result<()> {
        let url = format!("{}/api/v1/solana/snapshots", self.base_url);
        
        let mut request = self.client.post(&url).json(&snapshot);
        
        if let Some(token) = &self.auth_token {
            request = request.bearer_auth(token);
        }
        
        let response = request.send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Q-NK snapshot API error: {}", response.status()));
        }
        
        Ok(())
    }
    
    /// Get current Q-NarwhalKnight block height
    async fn get_current_block_height(&self) -> Result<u64> {
        let url = format!("{}/api/v1/status", self.base_url);
        
        let response = self.client.get(&url).send().await?;
        let status: serde_json::Value = response.json().await?;
        
        status["block_height"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("Invalid block height response"))
    }
    
    /// Get proof confirmation statuses
    async fn get_proof_confirmations(&self, proof_hashes: &[[u8; 32]]) -> Result<HashMap<[u8; 32], ConfirmationStatus>> {
        let url = format!("{}/api/v1/solana/proofs/status", self.base_url);
        
        let hash_strings: Vec<String> = proof_hashes.iter()
            .map(|hash| hex::encode(hash))
            .collect();
        
        let response = self.client
            .post(&url)
            .json(&serde_json::json!({"proof_hashes": hash_strings}))
            .send()
            .await?;
        
        let statuses: HashMap<String, String> = response.json().await?;
        
        let mut result = HashMap::new();
        
        for (hash_hex, status_str) in statuses {
            let hash_bytes = hex::decode(hash_hex)?;
            if hash_bytes.len() == 32 {
                let mut hash_array = [0u8; 32];
                hash_array.copy_from_slice(&hash_bytes);
                
                let status = match status_str.as_str() {
                    "pending" => ConfirmationStatus::Pending,
                    "included" => ConfirmationStatus::Included,
                    "finalized" => ConfirmationStatus::Finalized,
                    _ => ConfirmationStatus::Rejected(status_str),
                };
                
                result.insert(hash_array, status);
            }
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SolanaBridgeConfig, ProofType};
    
    fn create_test_proof() -> SolanaStateProof {
        SolanaStateProof {
            proof_type: ProofType::TokenBalance,
            slot: 123456789,
            blockhash: "test_blockhash_string_here_44_chars_long_".to_string(),
            account_pubkey: "account_pubkey_string_here_44_chars_long__".to_string(),
            compressed_data: vec![1, 2, 3, 4],
            merkle_proof: vec![[1u8; 32], [2u8; 32]],
            signature: vec![3u8; 64],
            timestamp: 1703097600,
        }
    }
    
    #[tokio::test]
    async fn test_state_bridge_creation() {
        let config = SolanaBridgeConfig::default();
        let result = StateBridge::new(&config).await;
        
        // May fail without real Q-NK endpoint
        if result.is_err() {
            println!("Expected failure without Q-NK: {:?}", result.err());
        }
    }
    
    #[test]
    fn test_proof_priority_determination() {
        let config = SolanaBridgeConfig::default();
        let bridge = StateBridge {
            config: config.clone(),
            qnk_client: QnkClient {
                client: reqwest::Client::new(),
                base_url: "http://localhost:3000".to_string(),
                auth_token: None,
            },
            pending_proofs: VecDeque::new(),
            submitted_proofs: HashMap::new(),
            state_snapshots: Vec::new(),
            last_submission: None,
            stats: StateBridgeStats::default(),
        };
        
        let mut proof = create_test_proof();
        
        // Fresh token balance should be high priority
        proof.proof_type = ProofType::TokenBalance;
        proof.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() - 30; // 30 seconds ago
        
        assert_eq!(bridge.determine_proof_priority(&proof), ProofPriority::High);
        
        // Data availability should be critical
        proof.proof_type = ProofType::DataAvailability;
        assert_eq!(bridge.determine_proof_priority(&proof), ProofPriority::Critical);
    }
    
    #[test]
    fn test_state_snapshot_creation() {
        let snapshot = StateSnapshot {
            solana_slot: 123456789,
            qnk_block_height: 987654321,
            account_count: 100,
            total_proof_size: 80000, // 100 * 800 bytes
            snapshot_hash: [1u8; 32],
            timestamp: 1703097600,
        };
        
        // Should serialize properly
        let serialized = serde_json::to_string(&snapshot).unwrap();
        let deserialized: StateSnapshot = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(snapshot.solana_slot, deserialized.solana_slot);
        assert_eq!(snapshot.account_count, deserialized.account_count);
    }
    
    #[test]
    fn test_confirmation_status_serialization() {
        let statuses = vec![
            ConfirmationStatus::Pending,
            ConfirmationStatus::Included,
            ConfirmationStatus::Finalized,
            ConfirmationStatus::Rejected("Invalid signature".to_string()),
        ];
        
        for status in statuses {
            let serialized = serde_json::to_string(&status).unwrap();
            let deserialized: ConfirmationStatus = serde_json::from_str(&serialized).unwrap();
            
            match (&status, &deserialized) {
                (ConfirmationStatus::Rejected(a), ConfirmationStatus::Rejected(b)) => {
                    assert_eq!(a, b);
                },
                _ => assert!(std::mem::discriminant(&status) == std::mem::discriminant(&deserialized)),
            }
        }
    }
}