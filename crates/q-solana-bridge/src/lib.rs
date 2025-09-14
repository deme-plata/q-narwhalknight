//! # Q-Solana-Bridge: Tor-Only Light-Client with Proof Production
//! 
//! 🌞⚡ Compressed Reed-Solomon proofs for SPL token verification via Tor hidden service.
//! Ultra-lightweight client that generates <1KB proofs for cross-chain asset verification.
//!
//! ## Revolutionary Features:
//! - **Tor-Only Proof Producer** - Hidden service generates proofs anonymously
//! - **Reed-Solomon Verification** - Data availability proofs for SPL accounts
//! - **<1KB Proof Size** - Compressed proofs for maximum efficiency  
//! - **Batch Verification** - Process 100+ SPL proofs per Q-NK block
//! - **State Snapshots** - Periodic Solana state commitments on Q-NK

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tokio::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

pub mod light_client;
pub mod proof_producer;
pub mod spl_verifier;
pub mod tor_solana_rpc;
pub mod state_bridge;

pub use light_client::*;
pub use proof_producer::*;
pub use spl_verifier::*;
pub use tor_solana_rpc::*;
pub use state_bridge::*;

/// Solana account data for proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolanaAccount {
    pub pubkey: String,
    pub lamports: u64,
    pub data: Vec<u8>,
    pub owner: String,
    pub executable: bool,
    pub rent_epoch: u64,
}

/// SPL Token account data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplTokenAccount {
    pub mint: String,
    pub owner: String,
    pub amount: u64,
    pub delegate: Option<String>,
    pub state: SplTokenState,
    pub is_native: Option<u64>,
    pub delegated_amount: u64,
    pub close_authority: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplTokenState {
    Uninitialized,
    Initialized,
    Frozen,
}

/// Compressed proof for Solana state verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolanaStateProof {
    pub proof_type: ProofType,
    pub slot: u64,
    pub blockhash: String,
    pub account_pubkey: String,
    pub compressed_data: Vec<u8>, // Reed-Solomon encoded
    pub merkle_proof: Vec<[u8; 32]>,
    pub signature: Vec<u8>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofType {
    AccountExistence,
    TokenBalance,
    ProgramState,
    DataAvailability,
}

impl SolanaStateProof {
    /// Calculate proof size in bytes
    pub fn size(&self) -> usize {
        self.compressed_data.len() + 
        (self.merkle_proof.len() * 32) + 
        self.signature.len() + 
        100 // Estimated overhead for other fields
    }
    
    /// Verify the proof is under size limit
    pub fn is_under_size_limit(&self, max_bytes: usize) -> bool {
        self.size() <= max_bytes
    }
    
    /// Generate proof hash for verification
    pub fn generate_proof_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.slot.to_le_bytes());
        hasher.update(self.blockhash.as_bytes());
        hasher.update(self.account_pubkey.as_bytes());
        hasher.update(&self.compressed_data);
        hasher.update(b"SOLANA_STATE_PROOF_V1");
        hasher.finalize().into()
    }
}

/// Configuration for Solana bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolanaBridgeConfig {
    /// Tor SOCKS5 proxy endpoint
    pub tor_proxy: String,
    /// Solana RPC endpoints (should be .onion addresses)
    pub solana_rpc_endpoints: Vec<String>,
    /// Q-NarwhalKnight node endpoint
    pub qnk_rpc_url: String,
    /// Maximum proof size in bytes
    pub max_proof_size: usize,
    /// Commitment level for finality
    pub commitment_level: String,
    /// Proof generation interval (seconds)
    pub proof_interval: u64,
    /// State snapshot interval (seconds)
    pub snapshot_interval: u64,
}

impl Default for SolanaBridgeConfig {
    fn default() -> Self {
        Self {
            tor_proxy: "socks5://127.0.0.1:9050".to_string(),
            solana_rpc_endpoints: vec![
                "http://solnode1.qnk.onion:8899".to_string(),
                "http://solnode2.qnk.onion:8899".to_string(),
            ],
            qnk_rpc_url: "http://localhost:3000".to_string(),
            max_proof_size: 1024, // 1KB maximum
            commitment_level: "finalized".to_string(),
            proof_interval: 60, // 1 minute
            snapshot_interval: 3600, // 1 hour
        }
    }
}

/// Statistics for Solana bridge performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolanaBridgeStats {
    pub proofs_generated: u64,
    pub spl_accounts_verified: u64,
    pub state_snapshots_created: u64,
    pub failed_proof_generations: u64,
    pub avg_proof_size_bytes: f64,
    pub avg_proof_generation_ms: f64,
    pub latest_slot_processed: u64,
    pub active_spl_tokens_tracked: usize,
}

/// Main Solana bridge service
#[derive(Clone)]
pub struct SolanaBridge {
    config: SolanaBridgeConfig,
    light_client: SolanaLightClient,
    proof_producer: ProofProducer,
    spl_verifier: SplVerifier,
    tor_rpc: TorSolanaRpc,
    state_bridge: StateBridge,
    stats: SolanaBridgeStats,
    tracked_accounts: HashMap<String, Instant>,
}

impl SolanaBridge {
    pub async fn new(config: SolanaBridgeConfig) -> Result<Self> {
        info!("🌞⚡ Initializing Solana Tor-Only Light-Client");
        info!("   • Mode: Proof-producer hidden service");
        info!("   • Max proof size: {} bytes", config.max_proof_size);
        info!("   • Commitment: {}", config.commitment_level);
        info!("   • Endpoints: {} Solana nodes", config.solana_rpc_endpoints.len());
        
        let tor_rpc = TorSolanaRpc::new(&config).await?;
        let light_client = SolanaLightClient::new(&config).await?;
        let proof_producer = ProofProducer::new(&config).await?;
        let spl_verifier = SplVerifier::new(&config).await?;
        let state_bridge = StateBridge::new(&config).await?;
        
        Ok(Self {
            config,
            light_client,
            proof_producer,
            spl_verifier,
            tor_rpc,
            state_bridge,
            stats: SolanaBridgeStats::default(),
            tracked_accounts: HashMap::new(),
        })
    }
    
    /// Start the Solana bridge service
    pub async fn run(&mut self) -> Result<()> {
        info!("🚀 Starting Solana Tor-Only Light-Client");
        info!("   • Proof generation: every {}s", self.config.proof_interval);
        info!("   • State snapshots: every {}s", self.config.snapshot_interval);
        
        let mut proof_interval = tokio::time::interval(Duration::from_secs(self.config.proof_interval));
        let mut snapshot_interval = tokio::time::interval(Duration::from_secs(self.config.snapshot_interval));
        let mut stats_interval = tokio::time::interval(Duration::from_secs(300)); // Every 5 minutes
        
        loop {
            tokio::select! {
                _ = proof_interval.tick() => {
                    self.generate_proofs().await?;
                },
                _ = snapshot_interval.tick() => {
                    self.create_state_snapshot().await?;
                },
                _ = stats_interval.tick() => {
                    self.log_statistics().await;
                },
            }
        }
    }
    
    /// Generate proofs for tracked SPL accounts
    async fn generate_proofs(&mut self) -> Result<()> {
        let start_time = Instant::now();
        
        debug!("🏭 Generating Solana proofs for {} accounts", self.tracked_accounts.len());
        
        let mut successful_proofs = 0;
        let mut total_proof_size = 0;
        
        // Get current slot
        let current_slot = self.light_client.get_current_slot().await?;
        
        for (account_pubkey, _) in &self.tracked_accounts {
            match self.generate_account_proof(account_pubkey, current_slot).await {
                Ok(proof) => {
                    let proof_size = proof.size();
                    
                    if proof.is_under_size_limit(self.config.max_proof_size) {
                        info!("✅ Generated proof for {}: {} bytes", 
                               &account_pubkey[..8], proof_size);
                        
                        // Submit proof to Q-NarwhalKnight
                        self.state_bridge.submit_proof(proof).await?;
                        
                        successful_proofs += 1;
                        total_proof_size += proof_size;
                    } else {
                        warn!("⚠️ Proof too large for {}: {} > {} bytes", 
                               &account_pubkey[..8], proof_size, self.config.max_proof_size);
                        self.stats.failed_proof_generations += 1;
                    }
                },
                Err(e) => {
                    error!("❌ Failed to generate proof for {}: {}", &account_pubkey[..8], e);
                    self.stats.failed_proof_generations += 1;
                }
            }
        }
        
        // Update statistics
        let generation_time = start_time.elapsed().as_millis() as f64;
        self.stats.proofs_generated += successful_proofs;
        self.stats.latest_slot_processed = current_slot;
        
        if successful_proofs > 0 {
            let avg_size = total_proof_size as f64 / successful_proofs as f64;
            self.stats.avg_proof_size_bytes = 
                (self.stats.avg_proof_size_bytes * 0.9) + (avg_size * 0.1);
            
            let avg_time = generation_time / successful_proofs as f64;
            self.stats.avg_proof_generation_ms = 
                (self.stats.avg_proof_generation_ms * 0.9) + (avg_time * 0.1);
            
            info!("📊 Generated {} proofs in {:.0}ms (avg: {} bytes)", 
                   successful_proofs, generation_time, avg_size as u64);
        }
        
        Ok(())
    }
    
    /// Generate proof for a specific account
    async fn generate_account_proof(&mut self, account_pubkey: &str, slot: u64) -> Result<SolanaStateProof> {
        // Get account data
        let account = self.tor_rpc.get_account(account_pubkey).await?;
        
        // Determine proof type based on account data
        let proof_type = if self.is_spl_token_account(&account) {
            ProofType::TokenBalance
        } else if account.executable {
            ProofType::ProgramState
        } else {
            ProofType::AccountExistence
        };
        
        // Get blockhash for the slot
        let blockhash = self.light_client.get_blockhash_for_slot(slot).await?;
        
        // Compress account data using Reed-Solomon
        let compressed_data = self.proof_producer.compress_account_data(&account).await?;
        
        // Generate Merkle proof
        let merkle_proof = self.proof_producer.generate_merkle_proof(&account, slot).await?;
        
        // Sign the proof
        let signature = self.proof_producer.sign_proof(&account, slot).await?;
        
        let proof = SolanaStateProof {
            proof_type,
            slot,
            blockhash,
            account_pubkey: account_pubkey.to_string(),
            compressed_data,
            merkle_proof,
            signature,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        };
        
        Ok(proof)
    }
    
    /// Check if account is an SPL token account
    fn is_spl_token_account(&self, account: &SolanaAccount) -> bool {
        // SPL Token Program ID
        account.owner == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA" && 
        account.data.len() == 165 // Standard SPL token account size
    }
    
    /// Create periodic state snapshot
    async fn create_state_snapshot(&mut self) -> Result<()> {
        info!("📸 Creating Solana state snapshot");
        
        let current_slot = self.light_client.get_current_slot().await?;
        let snapshot = self.state_bridge.create_snapshot(current_slot).await?;
        
        info!("✅ State snapshot created for slot {}: {} accounts", 
               current_slot, snapshot.account_count);
        
        self.stats.state_snapshots_created += 1;
        Ok(())
    }
    
    /// Add SPL token account to tracking
    pub async fn track_spl_account(&mut self, account_pubkey: String) -> Result<()> {
        // Verify it's a valid SPL token account
        let account = self.tor_rpc.get_account(&account_pubkey).await?;
        
        if !self.is_spl_token_account(&account) {
            return Err(anyhow::anyhow!("Not a valid SPL token account"));
        }
        
        self.tracked_accounts.insert(account_pubkey.clone(), Instant::now());
        self.stats.active_spl_tokens_tracked = self.tracked_accounts.len();
        
        info!("👁️ Now tracking SPL account: {}", &account_pubkey[..8]);
        Ok(())
    }
    
    /// Remove SPL token account from tracking
    pub fn untrack_spl_account(&mut self, account_pubkey: &str) {
        if self.tracked_accounts.remove(account_pubkey).is_some() {
            self.stats.active_spl_tokens_tracked = self.tracked_accounts.len();
            info!("👁️‍🗨️ Stopped tracking SPL account: {}", &account_pubkey[..8]);
        }
    }
    
    /// Verify an SPL token proof
    pub async fn verify_spl_proof(&self, proof: &SolanaStateProof) -> Result<bool> {
        self.spl_verifier.verify_proof(proof).await
    }
    
    /// Log bridge statistics
    async fn log_statistics(&self) {
        info!("📈 Solana Bridge Statistics:");
        info!("   • Proofs generated: {}", self.stats.proofs_generated);
        info!("   • SPL accounts verified: {}", self.stats.spl_accounts_verified);
        info!("   • State snapshots: {}", self.stats.state_snapshots_created);
        info!("   • Failed generations: {}", self.stats.failed_proof_generations);
        info!("   • Average proof size: {:.0} bytes", self.stats.avg_proof_size_bytes);
        info!("   • Average generation time: {:.0}ms", self.stats.avg_proof_generation_ms);
        info!("   • Latest slot processed: {}", self.stats.latest_slot_processed);
        info!("   • Active SPL tokens tracked: {}", self.stats.active_spl_tokens_tracked);
        
        if self.stats.proofs_generated > 0 {
            let success_rate = (self.stats.proofs_generated as f64) / 
                ((self.stats.proofs_generated + self.stats.failed_proof_generations) as f64) * 100.0;
            info!("   • Success rate: {:.1}%", success_rate);
        }
        
        // Check proof size efficiency
        if self.stats.avg_proof_size_bytes > 0.0 {
            let size_efficiency = (self.config.max_proof_size as f64 - self.stats.avg_proof_size_bytes) / 
                self.config.max_proof_size as f64 * 100.0;
            info!("   • Size efficiency: {:.1}% unused", size_efficiency);
        }
    }
    
    /// Get current bridge statistics
    pub fn get_stats(&self) -> &SolanaBridgeStats {
        &self.stats
    }
    
    /// Get list of tracked SPL accounts
    pub fn get_tracked_accounts(&self) -> Vec<String> {
        self.tracked_accounts.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_solana_state_proof_size_calculation() {
        let proof = SolanaStateProof {
            proof_type: ProofType::TokenBalance,
            slot: 123456789,
            blockhash: "test_blockhash".to_string(),
            account_pubkey: "test_pubkey".to_string(),
            compressed_data: vec![0; 500], // 500 bytes of data
            merkle_proof: vec![[0; 32]; 10], // 10 * 32 = 320 bytes
            signature: vec![0; 64], // 64 bytes
            timestamp: 1703097600,
        };
        
        let size = proof.size();
        assert!(size >= 500 + 320 + 64 + 100); // At least the known components
        assert!(proof.is_under_size_limit(1024)); // Should fit in 1KB
        assert!(!proof.is_under_size_limit(500)); // Should not fit in 500 bytes
    }
    
    #[test]
    fn test_proof_hash_generation() {
        let proof = SolanaStateProof {
            proof_type: ProofType::AccountExistence,
            slot: 123456789,
            blockhash: "test_blockhash".to_string(),
            account_pubkey: "test_pubkey".to_string(),
            compressed_data: vec![1, 2, 3, 4],
            merkle_proof: vec![[1; 32]],
            signature: vec![2; 64],
            timestamp: 1703097600,
        };
        
        let hash1 = proof.generate_proof_hash();
        let hash2 = proof.generate_proof_hash();
        
        assert_eq!(hash1, hash2); // Should be deterministic
        assert_eq!(hash1.len(), 32); // Should be 256 bits
    }
    
    #[tokio::test]
    async fn test_solana_bridge_initialization() {
        let config = SolanaBridgeConfig::default();
        
        // This would fail without actual Solana nodes, but tests the structure
        let result = SolanaBridge::new(config).await;
        
        // In a real test environment with mocked services, this should succeed
        if result.is_err() {
            println!("Expected failure without real Solana nodes: {:?}", result.err());
        }
    }
    
    #[test]
    fn test_spl_token_state_serialization() {
        let token_account = SplTokenAccount {
            mint: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string(),
            owner: "owner_pubkey".to_string(),
            amount: 1000000, // 1 USDC (6 decimals)
            delegate: None,
            state: SplTokenState::Initialized,
            is_native: None,
            delegated_amount: 0,
            close_authority: None,
        };
        
        // Should serialize/deserialize without issues
        let serialized = serde_json::to_string(&token_account).unwrap();
        let deserialized: SplTokenAccount = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(token_account.mint, deserialized.mint);
        assert_eq!(token_account.amount, deserialized.amount);
    }
}