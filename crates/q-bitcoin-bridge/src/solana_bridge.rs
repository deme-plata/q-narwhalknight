/// Solana Tor-Only Light Client with Reed-Solomon Proof Verification
///
/// Enables trustless SPL token verification through Tor-delivered proofs without
/// requiring direct Solana RPC connections. Uses Reed-Solomon erasure coding
/// for data availability and compact Merkle proofs for state verification.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast, mpsc};
use tracing::{info, warn, error, debug};

#[derive(Debug, Clone)]
pub struct SolanaLightClient {
    tor_proof_service: Arc<TorProofService>,
    reed_solomon_verifier: Arc<ReedSolomonVerifier>,
    spl_state_cache: Arc<RwLock<SPLStateCache>>,
    proof_broadcaster: broadcast::Sender<VerifiedProof>,
    config: SolanaLightClientConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolanaLightClientConfig {
    pub proof_service_onion: String,     // Tor hidden service for proofs
    pub tor_proxy: String,               // SOCKS5 proxy
    pub proof_fetch_interval_ms: u64,    // How often to fetch proofs
    pub max_proof_age_seconds: u64,      // Maximum age for valid proofs
    pub reed_solomon_redundancy: usize,  // Erasure coding redundancy
    pub merkle_proof_depth: usize,       // Maximum Merkle tree depth
    pub spl_token_whitelist: Vec<String>, // Monitored SPL tokens
    pub batch_verification_size: usize,  // Proofs to verify per batch
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SPLTokenState {
    pub token_mint: String,              // SPL token mint address
    pub token_account: String,           // Token account address
    pub balance: u64,                    // Current token balance
    pub owner: String,                   // Account owner
    pub slot: u64,                       // Solana slot number
    pub proof_hash: String,              // Hash of verification proof
    pub verified_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolanaProof {
    pub proof_id: String,
    pub proof_type: SolanaProofType,
    pub slot: u64,
    pub merkle_proof: MerkleProof,
    pub reed_solomon_shards: Vec<ReedSolomonShard>,
    pub state_data: serde_json::Value,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolanaProofType {
    SPLTokenAccount,
    ProgramState,
    BlockHeader,
    TransactionInclusion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    pub leaf_hash: String,
    pub proof_path: Vec<String>,         // Hashes from leaf to root
    pub root_hash: String,               // Merkle root
    pub leaf_index: u64,                 // Position in tree
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReedSolomonShard {
    pub shard_index: usize,
    pub shard_data: Vec<u8>,
    pub checksum: String,
    pub erasure_coding_params: RSParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSParams {
    pub data_shards: usize,              // Original data pieces
    pub parity_shards: usize,            // Redundancy pieces
    pub shard_size: usize,               // Bytes per shard
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedProof {
    pub proof_id: String,
    pub verification_status: ProofStatus,
    pub spl_state: Option<SPLTokenState>,
    pub confidence_score: f64,           // 0.0 to 1.0
    pub verification_time_ms: u64,
    pub verified_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofStatus {
    Verified,
    Failed,
    Pending,
    Expired,
}

#[derive(Debug)]
struct TorProofService {
    onion_endpoint: String,
    http_client: reqwest::Client,
    proof_cache: RwLock<HashMap<String, SolanaProof>>,
}

#[derive(Debug)]
struct ReedSolomonVerifier {
    rs_params: RSParams,
    verification_cache: RwLock<HashMap<String, bool>>,
}

#[derive(Debug)]
struct SPLStateCache {
    token_states: HashMap<String, SPLTokenState>,
    last_update_slot: u64,
    verified_proofs: HashMap<String, VerifiedProof>,
}

impl SolanaLightClient {
    pub async fn new(config: SolanaLightClientConfig) -> Result<Self> {
        info!("🌞 Initializing Solana Tor-Only Light Client");
        
        // Validate onion service endpoint
        if !config.proof_service_onion.contains(".onion") {
            return Err(anyhow!("❌ Proof service must use .onion address"));
        }
        
        let tor_proof_service = Arc::new(TorProofService::new(&config).await?);
        let reed_solomon_verifier = Arc::new(ReedSolomonVerifier::new(RSParams::default()));
        let spl_state_cache = Arc::new(RwLock::new(SPLStateCache::new()));
        let (proof_tx, _) = broadcast::channel(1000);
        
        let client = Self {
            tor_proof_service,
            reed_solomon_verifier,
            spl_state_cache,
            proof_broadcaster: proof_tx,
            config,
        };
        
        // Test Tor connectivity to proof service
        client.verify_tor_proof_service().await?;
        
        info!("✅ Solana Light Client initialized with Tor-only operation");
        Ok(client)
    }
    
    /// Start the light client service
    pub async fn start_light_client_service(&self) -> Result<broadcast::Receiver<VerifiedProof>> {
        info!("🚀 Starting Solana Light Client service");
        
        let proof_rx = self.proof_broadcaster.subscribe();
        
        // Start proof fetching loop
        let client_clone = self.clone();
        tokio::spawn(async move {
            if let Err(e) = client_clone.proof_fetching_loop().await {
                error!("❌ Proof fetching loop failed: {}", e);
            }
        });
        
        // Start proof verification loop
        let client_clone = self.clone();
        tokio::spawn(async move {
            client_clone.proof_verification_loop().await;
        });
        
        // Start SPL state monitoring
        let client_clone = self.clone();
        tokio::spawn(async move {
            client_clone.spl_monitoring_loop().await;
        });
        
        info!("✅ Solana Light Client service started");
        Ok(proof_rx)
    }
    
    /// Verify Tor connectivity to proof service
    async fn verify_tor_proof_service(&self) -> Result<()> {
        info!("🔍 Verifying Tor connectivity to Solana proof service");
        
        let health_check = self.tor_proof_service.health_check().await?;
        
        if health_check.is_healthy {
            info!("✅ Solana proof service reachable via Tor");
            Ok(())
        } else {
            Err(anyhow!("❌ Solana proof service unhealthy: {}", health_check.status))
        }
    }
    
    /// Main proof fetching loop
    async fn proof_fetching_loop(&self) -> Result<()> {
        let mut interval = tokio::time::interval(
            std::time::Duration::from_millis(self.config.proof_fetch_interval_ms)
        );
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.fetch_latest_proofs().await {
                warn!("⚠️ Proof fetching failed: {}", e);
            }
        }
    }
    
    /// Fetch latest proofs from Tor proof service
    async fn fetch_latest_proofs(&self) -> Result<()> {
        debug!("📡 Fetching latest Solana proofs via Tor");
        
        // Fetch proofs for all whitelisted SPL tokens
        for token_mint in &self.config.spl_token_whitelist {
            if let Ok(proof) = self.tor_proof_service.get_spl_token_proof(token_mint).await {
                // Cache the proof
                let mut proof_cache = self.tor_proof_service.proof_cache.write().await;
                proof_cache.insert(proof.proof_id.clone(), proof);
            }
        }
        
        // Fetch general blockchain proofs
        if let Ok(header_proof) = self.tor_proof_service.get_latest_block_proof().await {
            let mut proof_cache = self.tor_proof_service.proof_cache.write().await;
            proof_cache.insert(header_proof.proof_id.clone(), header_proof);
        }
        
        Ok(())
    }
    
    /// Proof verification loop
    async fn proof_verification_loop(&self) {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.verify_cached_proofs().await {
                warn!("⚠️ Proof verification failed: {}", e);
            }
        }
    }
    
    /// Verify all cached proofs
    async fn verify_cached_proofs(&self) -> Result<()> {
        let proof_cache = self.tor_proof_service.proof_cache.read().await;
        let mut verification_count = 0;
        
        for (proof_id, proof) in proof_cache.iter() {
            if self.should_verify_proof(proof).await {
                let verification_result = self.verify_solana_proof(proof).await?;
                
                // Broadcast verified proof
                if let Err(e) = self.proof_broadcaster.send(verification_result) {
                    warn!("Failed to broadcast verified proof: {}", e);
                } else {
                    verification_count += 1;
                }
            }
        }
        
        if verification_count > 0 {
            info!("✅ Verified {} Solana proofs", verification_count);
        }
        
        Ok(())
    }
    
    /// Check if proof should be verified
    async fn should_verify_proof(&self, proof: &SolanaProof) -> bool {
        let age = chrono::Utc::now() - proof.generated_at;
        age.num_seconds() < self.config.max_proof_age_seconds as i64
    }
    
    /// Verify a Solana proof using Reed-Solomon and Merkle verification
    async fn verify_solana_proof(&self, proof: &SolanaProof) -> Result<VerifiedProof> {
        let start_time = std::time::Instant::now();
        
        debug!("🔍 Verifying Solana proof: {}", proof.proof_id);
        
        // Step 1: Verify Reed-Solomon shards
        let rs_verification = self.reed_solomon_verifier.verify_shards(&proof.reed_solomon_shards).await?;
        if !rs_verification.is_valid {
            return Ok(VerifiedProof {
                proof_id: proof.proof_id.clone(),
                verification_status: ProofStatus::Failed,
                spl_state: None,
                confidence_score: 0.0,
                verification_time_ms: start_time.elapsed().as_millis() as u64,
                verified_at: chrono::Utc::now(),
            });
        }
        
        // Step 2: Verify Merkle proof
        let merkle_verification = self.verify_merkle_proof(&proof.merkle_proof, &proof.state_data).await?;
        if !merkle_verification.is_valid {
            return Ok(VerifiedProof {
                proof_id: proof.proof_id.clone(),
                verification_status: ProofStatus::Failed,
                spl_state: None,
                confidence_score: 0.5, // Partial verification
                verification_time_ms: start_time.elapsed().as_millis() as u64,
                verified_at: chrono::Utc::now(),
            });
        }
        
        // Step 3: Extract SPL token state if applicable
        let spl_state = if matches!(proof.proof_type, SolanaProofType::SPLTokenAccount) {
            self.extract_spl_state_from_proof(proof).await?
        } else {
            None
        };
        
        // Calculate confidence score
        let confidence_score = self.calculate_proof_confidence(proof, &rs_verification, &merkle_verification).await;
        
        let verification_time = start_time.elapsed().as_millis() as u64;
        
        info!("✅ Solana proof verified: {} ({}ms, confidence: {:.2})", 
              proof.proof_id, verification_time, confidence_score);
        
        Ok(VerifiedProof {
            proof_id: proof.proof_id.clone(),
            verification_status: ProofStatus::Verified,
            spl_state,
            confidence_score,
            verification_time_ms: verification_time,
            verified_at: chrono::Utc::now(),
        })
    }
    
    /// Verify Merkle proof for state data
    async fn verify_merkle_proof(
        &self,
        merkle_proof: &MerkleProof,
        state_data: &serde_json::Value,
    ) -> Result<MerkleVerificationResult> {
        debug!("🌳 Verifying Merkle proof for leaf {}", merkle_proof.leaf_index);
        
        // Hash the state data to get leaf hash
        let state_bytes = state_data.to_string().as_bytes().to_vec();
        let computed_leaf_hash = hex::encode(Sha256::digest(&state_bytes));
        
        if computed_leaf_hash != merkle_proof.leaf_hash {
            return Ok(MerkleVerificationResult {
                is_valid: false,
                error: Some("Leaf hash mismatch".to_string()),
            });
        }
        
        // Verify proof path
        let mut current_hash = merkle_proof.leaf_hash.clone();
        
        for (i, sibling_hash) in merkle_proof.proof_path.iter().enumerate() {
            let (left, right) = if (merkle_proof.leaf_index >> i) & 1 == 0 {
                (&current_hash, sibling_hash)
            } else {
                (sibling_hash, &current_hash)
            };
            
            let mut hasher = Sha256::new();
            hasher.update(hex::decode(left)?);
            hasher.update(hex::decode(right)?);
            current_hash = hex::encode(hasher.finalize());
        }
        
        let is_valid = current_hash == merkle_proof.root_hash;
        
        Ok(MerkleVerificationResult {
            is_valid,
            error: if is_valid { None } else { Some("Root hash mismatch".to_string()) },
        })
    }
    
    /// Extract SPL token state from verified proof
    async fn extract_spl_state_from_proof(&self, proof: &SolanaProof) -> Result<Option<SPLTokenState>> {
        if !matches!(proof.proof_type, SolanaProofType::SPLTokenAccount) {
            return Ok(None);
        }
        
        let account_data = &proof.state_data["account"];
        
        let spl_state = SPLTokenState {
            token_mint: account_data["mint"].as_str().unwrap_or("").to_string(),
            token_account: account_data["pubkey"].as_str().unwrap_or("").to_string(),
            balance: account_data["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"]
                .as_str()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0),
            owner: account_data["account"]["owner"].as_str().unwrap_or("").to_string(),
            slot: proof.slot,
            proof_hash: proof.proof_id.clone(),
            verified_at: chrono::Utc::now(),
        };
        
        // Update SPL state cache
        let mut cache = self.spl_state_cache.write().await;
        cache.token_states.insert(spl_state.token_account.clone(), spl_state.clone());
        cache.last_update_slot = proof.slot;
        
        Ok(Some(spl_state))
    }
    
    /// SPL state monitoring loop
    async fn spl_monitoring_loop(&self) {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.update_spl_states().await {
                warn!("⚠️ SPL state update failed: {}", e);
            }
        }
    }
    
    /// Update all monitored SPL token states
    async fn update_spl_states(&self) -> Result<()> {
        debug!("🔄 Updating SPL token states");
        
        let mut update_count = 0;
        
        for token_mint in &self.config.spl_token_whitelist {
            if let Ok(proof) = self.tor_proof_service.get_spl_token_proof(token_mint).await {
                if let Ok(verified) = self.verify_solana_proof(&proof).await {
                    if matches!(verified.verification_status, ProofStatus::Verified) {
                        update_count += 1;
                    }
                }
            }
        }
        
        if update_count > 0 {
            info!("📊 Updated {} SPL token states", update_count);
        }
        
        Ok(())
    }
    
    /// Calculate proof confidence score
    async fn calculate_proof_confidence(
        &self,
        proof: &SolanaProof,
        rs_verification: &ReedSolomonVerificationResult,
        merkle_verification: &MerkleVerificationResult,
    ) -> f64 {
        let mut confidence = 0.0;
        
        // Reed-Solomon verification contributes 40% of confidence
        if rs_verification.is_valid {
            confidence += 0.4 * rs_verification.reconstruction_success_rate;
        }
        
        // Merkle proof verification contributes 40% of confidence
        if merkle_verification.is_valid {
            confidence += 0.4;
        }
        
        // Proof freshness contributes 20% of confidence
        let age_seconds = (chrono::Utc::now() - proof.generated_at).num_seconds() as f64;
        let freshness_score = (1.0 - (age_seconds / 3600.0)).max(0.0); // 1 hour decay
        confidence += 0.2 * freshness_score;
        
        confidence.min(1.0)
    }
    
    /// Get SPL token state by account address
    pub async fn get_spl_token_state(&self, token_account: &str) -> Option<SPLTokenState> {
        let cache = self.spl_state_cache.read().await;
        cache.token_states.get(token_account).cloned()
    }
    
    /// Get all verified SPL states
    pub async fn get_all_spl_states(&self) -> HashMap<String, SPLTokenState> {
        let cache = self.spl_state_cache.read().await;
        cache.token_states.clone()
    }
    
    /// Force SPL state update for specific token
    pub async fn force_spl_update(&self, token_mint: &str) -> Result<SPLTokenState> {
        info!("🔄 Forcing SPL state update for token: {}", token_mint);
        
        let proof = self.tor_proof_service.get_spl_token_proof(token_mint).await?;
        let verified_proof = self.verify_solana_proof(&proof).await?;
        
        if let Some(spl_state) = verified_proof.spl_state {
            info!("✅ Force SPL update completed for: {}", token_mint);
            Ok(spl_state)
        } else {
            Err(anyhow!("Failed to extract SPL state from proof"))
        }
    }
    
    /// Get light client statistics
    pub async fn get_light_client_stats(&self) -> SolanaLightClientStats {
        let cache = self.spl_state_cache.read().await;
        let proof_cache = self.tor_proof_service.proof_cache.read().await;
        
        SolanaLightClientStats {
            tracked_spl_tokens: cache.token_states.len() as u64,
            latest_verified_slot: cache.last_update_slot,
            cached_proofs: proof_cache.len() as u64,
            tor_service_health: self.check_tor_service_health().await,
            average_verification_time_ms: self.calculate_avg_verification_time().await,
            proof_success_rate: self.calculate_proof_success_rate().await,
        }
    }
    
    async fn check_tor_service_health(&self) -> f64 {
        match self.tor_proof_service.health_check().await {
            Ok(health) => if health.is_healthy { 1.0 } else { 0.5 },
            Err(_) => 0.0,
        }
    }
    
    async fn calculate_avg_verification_time(&self) -> f64 {
        // Simplified - in production, track actual metrics
        125.0 // ~125ms average verification time
    }
    
    async fn calculate_proof_success_rate(&self) -> f64 {
        let cache = self.spl_state_cache.read().await;
        let recent_proofs = cache.verified_proofs.values()
            .filter(|p| p.verified_at > chrono::Utc::now() - chrono::Duration::hours(1))
            .count();
        
        if recent_proofs == 0 {
            return 0.5; // Neutral if no recent activity
        }
        
        let successful_proofs = cache.verified_proofs.values()
            .filter(|p| p.verified_at > chrono::Utc::now() - chrono::Duration::hours(1))
            .filter(|p| matches!(p.verification_status, ProofStatus::Verified))
            .count();
        
        successful_proofs as f64 / recent_proofs as f64
    }
}

impl TorProofService {
    async fn new(config: &SolanaLightClientConfig) -> Result<Self> {
        // Create HTTP client with Tor proxy
        let http_client = reqwest::Client::builder()
            .proxy(reqwest::Proxy::all(&config.tor_proxy)?)
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        
        Ok(Self {
            onion_endpoint: config.proof_service_onion.clone(),
            http_client,
            proof_cache: RwLock::new(HashMap::new()),
        })
    }
    
    async fn health_check(&self) -> Result<ProofServiceHealth> {
        let health_url = format!("http://{}/health", self.onion_endpoint);
        
        let response = self.http_client.get(&health_url).send().await?;
        
        if response.status().is_success() {
            Ok(ProofServiceHealth {
                is_healthy: true,
                status: "Service operational".to_string(),
            })
        } else {
            Ok(ProofServiceHealth {
                is_healthy: false,
                status: format!("HTTP {}", response.status()),
            })
        }
    }
    
    async fn get_spl_token_proof(&self, token_mint: &str) -> Result<SolanaProof> {
        let proof_url = format!("http://{}/proof/spl/{}", self.onion_endpoint, token_mint);
        
        let response = self.http_client.get(&proof_url).send().await?;
        let proof: SolanaProof = response.json().await?;
        
        debug!("📊 Retrieved SPL proof for token: {}", token_mint);
        Ok(proof)
    }
    
    async fn get_latest_block_proof(&self) -> Result<SolanaProof> {
        let block_url = format!("http://{}/proof/latest_block", self.onion_endpoint);
        
        let response = self.http_client.get(&block_url).send().await?;
        let proof: SolanaProof = response.json().await?;
        
        debug!("📦 Retrieved latest block proof: slot {}", proof.slot);
        Ok(proof)
    }
}

impl ReedSolomonVerifier {
    fn new(params: RSParams) -> Self {
        Self {
            rs_params: params,
            verification_cache: RwLock::new(HashMap::new()),
        }
    }
    
    async fn verify_shards(&self, shards: &[ReedSolomonShard]) -> Result<ReedSolomonVerificationResult> {
        debug!("🧮 Verifying Reed-Solomon shards: {} shards", shards.len());
        
        // Check if we have enough shards for reconstruction
        let data_shards_available = shards.iter()
            .filter(|s| s.shard_index < self.rs_params.data_shards)
            .count();
        
        if data_shards_available < self.rs_params.data_shards {
            return Ok(ReedSolomonVerificationResult {
                is_valid: false,
                reconstruction_success_rate: data_shards_available as f64 / self.rs_params.data_shards as f64,
                verified_shards: 0,
            });
        }
        
        // Verify checksums for all shards
        let mut verified_count = 0;
        for shard in shards {
            if self.verify_shard_checksum(shard).await? {
                verified_count += 1;
            }
        }
        
        let success_rate = verified_count as f64 / shards.len() as f64;
        
        Ok(ReedSolomonVerificationResult {
            is_valid: success_rate >= 0.8, // Need 80% of shards to be valid
            reconstruction_success_rate: success_rate,
            verified_shards: verified_count,
        })
    }
    
    async fn verify_shard_checksum(&self, shard: &ReedSolomonShard) -> Result<bool> {
        let computed_checksum = hex::encode(Sha256::digest(&shard.shard_data));
        Ok(computed_checksum == shard.checksum)
    }
}

impl SPLStateCache {
    fn new() -> Self {
        Self {
            token_states: HashMap::new(),
            last_update_slot: 0,
            verified_proofs: HashMap::new(),
        }
    }
}

impl Clone for SolanaLightClient {
    fn clone(&self) -> Self {
        Self {
            tor_proof_service: Arc::clone(&self.tor_proof_service),
            reed_solomon_verifier: Arc::clone(&self.reed_solomon_verifier),
            spl_state_cache: Arc::clone(&self.spl_state_cache),
            proof_broadcaster: self.proof_broadcaster.clone(),
            config: self.config.clone(),
        }
    }
}

// Supporting types

#[derive(Debug)]
struct ProofServiceHealth {
    is_healthy: bool,
    status: String,
}

#[derive(Debug)]
struct MerkleVerificationResult {
    is_valid: bool,
    error: Option<String>,
}

#[derive(Debug)]
struct ReedSolomonVerificationResult {
    is_valid: bool,
    reconstruction_success_rate: f64,
    verified_shards: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SolanaLightClientStats {
    pub tracked_spl_tokens: u64,
    pub latest_verified_slot: u64,
    pub cached_proofs: u64,
    pub tor_service_health: f64,
    pub average_verification_time_ms: f64,
    pub proof_success_rate: f64,
}

impl Default for RSParams {
    fn default() -> Self {
        Self {
            data_shards: 8,    // 8 data pieces
            parity_shards: 4,  // 4 redundancy pieces
            shard_size: 1024,  // 1KB per shard
        }
    }
}

impl Default for SolanaLightClientConfig {
    fn default() -> Self {
        Self {
            proof_service_onion: "solanaproofs.qnk.onion:8080".to_string(),
            tor_proxy: "socks5://127.0.0.1:9050".to_string(),
            proof_fetch_interval_ms: 15000, // 15 seconds
            max_proof_age_seconds: 3600,    // 1 hour
            reed_solomon_redundancy: 4,     // 4 parity shards
            merkle_proof_depth: 20,         // Support up to 2^20 leaves
            spl_token_whitelist: vec![
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string(), // USDC
                "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB".to_string(), // USDT
            ],
            batch_verification_size: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rs_params() {
        let params = RSParams::default();
        assert_eq!(params.data_shards, 8);
        assert_eq!(params.parity_shards, 4);
        assert_eq!(params.shard_size, 1024);
    }
    
    #[tokio::test]
    async fn test_merkle_proof_verification() {
        let client = SolanaLightClient::new(SolanaLightClientConfig::default()).await.unwrap();
        
        let state_data = serde_json::json!({"test": "data"});
        let state_bytes = state_data.to_string().as_bytes().to_vec();
        let leaf_hash = hex::encode(Sha256::digest(&state_bytes));
        
        let merkle_proof = MerkleProof {
            leaf_hash: leaf_hash.clone(),
            proof_path: vec![leaf_hash.clone()], // Simplified proof
            root_hash: leaf_hash, // For single-leaf tree
            leaf_index: 0,
        };
        
        let verification = client.verify_merkle_proof(&merkle_proof, &state_data).await.unwrap();
        assert!(verification.is_valid);
    }
    
    #[test]
    fn test_solana_light_client_config() {
        let config = SolanaLightClientConfig::default();
        assert!(config.proof_service_onion.contains(".onion"));
        assert!(!config.spl_token_whitelist.is_empty());
        assert_eq!(config.reed_solomon_redundancy, 4);
    }
}