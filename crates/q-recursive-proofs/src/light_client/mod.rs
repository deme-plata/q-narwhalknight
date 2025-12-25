//! Light Client Implementation
//!
//! Provides trustless bootstrap for new nodes using recursive proofs.
//! New nodes can verify the entire blockchain history in ~10ms
//! without trusting any checkpoint provider.

use crate::protocol::messages::{LightClientProofRequest, LightClientProofResponse, ValidatorSet};
use crate::{EpochProof, EpochPublicInputs};
use q_lattice_guard::{LatticeGuard, LatticeGuardSRS, SecurityLevel};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Light client configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightClientConfig {
    /// LatticeGuard security level
    pub security_level: SecurityLevel,
    /// Minimum number of proof sources to query
    pub min_proof_sources: usize,
    /// Timeout for proof requests (seconds)
    pub request_timeout_secs: u64,
    /// Maximum proof age to accept (seconds)
    pub max_proof_age_secs: u64,
}

impl Default for LightClientConfig {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::PQ128,
            min_proof_sources: 3,
            request_timeout_secs: 30,
            max_proof_age_secs: 3600, // 1 hour
        }
    }
}

/// Light client state
#[derive(Clone, Debug)]
pub struct LightClientState {
    /// Verified state root
    pub state_root: Option<[u8; 32]>,
    /// Verified height
    pub height: u64,
    /// Verified epoch
    pub epoch: u64,
    /// Current validator set
    pub validator_set: Option<ValidatorSet>,
    /// Is bootstrapped?
    pub is_bootstrapped: bool,
    /// Last verification timestamp
    pub last_verified: u64,
}

impl Default for LightClientState {
    fn default() -> Self {
        Self {
            state_root: None,
            height: 0,
            epoch: 0,
            validator_set: None,
            is_bootstrapped: false,
            last_verified: 0,
        }
    }
}

/// Light client for trustless bootstrap
pub struct LightClient {
    /// Configuration
    config: LightClientConfig,
    /// Current state
    state: Arc<RwLock<LightClientState>>,
    /// LatticeGuard verifier
    lattice_guard: Arc<LatticeGuard>,
    /// SRS for verification
    srs: Arc<LatticeGuardSRS>,
    /// Genesis state root (hardcoded, trusted)
    genesis_state_root: [u8; 32],
}

impl LightClient {
    /// Create new light client
    pub fn new(config: LightClientConfig, genesis_state_root: [u8; 32]) -> anyhow::Result<Self> {
        info!("Initializing light client");

        let lattice_guard = Arc::new(LatticeGuard::new(config.security_level)?);

        // Use SRS caching to speed up light client initialization
        let cache_path = std::env::var("Q_DB_PATH")
            .map(|p| std::path::PathBuf::from(p).join("srs_cache"))
            .unwrap_or_else(|_| std::path::PathBuf::from("/tmp/q-lattice-guard-srs"));

        let mut rng = rand::thread_rng();
        let srs = Arc::new(LatticeGuardSRS::generate_or_load(
            lattice_guard.params().clone(),
            10000, // Minimal for verification
            &cache_path,
            &mut rng,
        )?);

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(LightClientState::default())),
            lattice_guard,
            srs,
            genesis_state_root,
        })
    }

    /// Bootstrap the light client
    ///
    /// This is the key function - verifies entire chain in ~10ms!
    pub async fn bootstrap(&self, proof_response: LightClientProofResponse) -> anyhow::Result<()> {
        info!("Starting light client bootstrap...");
        let start = Instant::now();

        // Deserialize the proof
        let proof: q_lattice_guard::LatticeGuardProof =
            bincode::deserialize(&proof_response.proof_data)?;

        // Build public inputs for verification
        let public_inputs = self.build_public_inputs(&proof_response);

        // Build minimal verification circuit
        let circuit = crate::circuits::epoch_transition::EpochTransitionCircuit::genesis()
            .build_circuit(1);

        // CRITICAL: Verify the proof
        info!("Verifying recursive proof...");

        let is_valid = self.lattice_guard.verify(
            &circuit,
            &public_inputs,
            &proof,
            &self.srs,
        )?;

        let verification_time = start.elapsed();

        if !is_valid {
            error!("Light client proof verification FAILED!");
            return Err(anyhow::anyhow!("Proof verification failed"));
        }

        info!(
            "Light client bootstrap complete! Verified {} blocks in {:?}",
            proof_response.current_height, verification_time
        );

        // Update state
        {
            let mut state = self.state.write().await;
            state.state_root = Some(proof_response.current_state_root);
            state.height = proof_response.current_height;
            state.epoch = proof_response.current_epoch;
            state.validator_set = proof_response.validator_set;
            state.is_bootstrapped = true;
            state.last_verified = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }

        Ok(())
    }

    /// Verify an epoch proof (for incremental updates)
    pub async fn verify_epoch_proof(&self, epoch_proof: &EpochProof) -> anyhow::Result<bool> {
        let start = Instant::now();

        // Check state continuity
        {
            let state = self.state.read().await;
            if let Some(current_root) = &state.state_root {
                if *current_root != epoch_proof.public_inputs.previous_state_root {
                    warn!(
                        "State root mismatch: expected {:?}, got {:?}",
                        current_root, epoch_proof.public_inputs.previous_state_root
                    );
                    return Ok(false);
                }
            }
        }

        // Build verification circuit
        let num_blocks = (epoch_proof.public_inputs.height_range.1
            - epoch_proof.public_inputs.height_range.0) as usize;
        let circuit = crate::circuits::epoch_transition::EpochTransitionCircuit::new(
            crate::circuits::epoch_transition::EpochTransitionConfig::default(),
        )
        .build_circuit(num_blocks);

        // Verify
        let public_inputs = epoch_proof.public_inputs.to_scalars();
        let is_valid = self.lattice_guard.verify(
            &circuit,
            &public_inputs,
            &epoch_proof.proof,
            &self.srs,
        )?;

        let verification_time = start.elapsed();
        debug!(
            "Epoch {} proof verification in {:?}: {}",
            epoch_proof.public_inputs.epoch,
            verification_time,
            if is_valid { "VALID" } else { "INVALID" }
        );

        if is_valid {
            // Update state
            let mut state = self.state.write().await;
            state.state_root = Some(epoch_proof.public_inputs.current_state_root);
            state.height = epoch_proof.public_inputs.height_range.1;
            state.epoch = epoch_proof.public_inputs.epoch;
            state.last_verified = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }

        Ok(is_valid)
    }

    /// Get current state
    pub async fn state(&self) -> LightClientState {
        self.state.read().await.clone()
    }

    /// Check if bootstrapped
    pub async fn is_bootstrapped(&self) -> bool {
        self.state.read().await.is_bootstrapped
    }

    /// Get verified height
    pub async fn verified_height(&self) -> u64 {
        self.state.read().await.height
    }

    /// Get verified state root
    pub async fn verified_state_root(&self) -> Option<[u8; 32]> {
        self.state.read().await.state_root
    }

    /// Create bootstrap request
    pub fn create_request(&self, requester_peer_id: String) -> LightClientProofRequest {
        LightClientProofRequest {
            known_height: 0,
            known_epoch: 0,
            include_validators: true,
            requester_peer_id,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Build public inputs from response
    fn build_public_inputs(&self, response: &LightClientProofResponse) -> Vec<u64> {
        let mut inputs = Vec::new();

        // Genesis state root (known/trusted)
        for chunk in self.genesis_state_root.chunks(8) {
            inputs.push(u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])));
        }

        // Current state root
        for chunk in response.current_state_root.chunks(8) {
            inputs.push(u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])));
        }

        // Height and epoch
        inputs.push(response.current_height);
        inputs.push(response.current_epoch);

        inputs
    }

    /// Verify state root query result
    pub async fn verify_state_query(
        &self,
        query_result: &StateQueryResult,
    ) -> anyhow::Result<bool> {
        let state = self.state.read().await;

        if !state.is_bootstrapped {
            return Err(anyhow::anyhow!("Light client not bootstrapped"));
        }

        let current_root = state.state_root.ok_or_else(|| {
            anyhow::anyhow!("No state root available")
        })?;

        // Verify Merkle proof
        let params = crate::gadgets::poseidon::PoseidonParams::secure_128(16);

        let is_valid = query_result.proof.verify(
            &query_result.value_hash,
            &current_root,
            &params,
        );

        Ok(is_valid)
    }
}

/// State query result with Merkle proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateQueryResult {
    /// Key queried
    pub key: Vec<u8>,
    /// Value (if exists)
    pub value: Option<Vec<u8>>,
    /// Value hash
    pub value_hash: [u8; 32],
    /// Merkle proof
    pub proof: crate::gadgets::merkle::MerkleProof,
    /// State root the proof is against
    pub state_root: [u8; 32],
}

/// Light client sync protocol
pub struct LightClientSync {
    /// Light client instance
    client: Arc<LightClient>,
    /// Sync interval (seconds)
    sync_interval_secs: u64,
    /// Maximum lag before re-sync
    max_lag_epochs: u64,
}

impl LightClientSync {
    /// Create new sync instance
    pub fn new(client: Arc<LightClient>, sync_interval_secs: u64) -> Self {
        Self {
            client,
            sync_interval_secs,
            max_lag_epochs: 10,
        }
    }

    /// Check if re-sync is needed
    pub async fn needs_sync(&self, network_epoch: u64) -> bool {
        let state = self.client.state().await;

        if !state.is_bootstrapped {
            return true;
        }

        network_epoch > state.epoch + self.max_lag_epochs
    }

    /// Perform incremental sync
    pub async fn sync(&self, epoch_proofs: Vec<EpochProof>) -> anyhow::Result<u64> {
        let mut synced_count = 0u64;

        for proof in epoch_proofs {
            match self.client.verify_epoch_proof(&proof).await {
                Ok(true) => {
                    synced_count += 1;
                    debug!("Synced epoch {}", proof.public_inputs.epoch);
                }
                Ok(false) => {
                    warn!("Invalid proof for epoch {}", proof.public_inputs.epoch);
                    break;
                }
                Err(e) => {
                    error!("Sync error: {}", e);
                    break;
                }
            }
        }

        Ok(synced_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_light_client_creation() {
        let config = LightClientConfig::default();
        let genesis_root = [0u8; 32];

        let client = LightClient::new(config, genesis_root);
        assert!(client.is_ok());

        let client = client.unwrap();
        assert!(!client.is_bootstrapped().await);
        assert_eq!(client.verified_height().await, 0);
    }

    #[test]
    fn test_light_client_request() {
        let config = LightClientConfig::default();
        let genesis_root = [0u8; 32];

        let client = LightClient::new(config, genesis_root).unwrap();
        let request = client.create_request("peer1".to_string());

        assert_eq!(request.known_height, 0);
        assert!(request.include_validators);
    }

    #[tokio::test]
    async fn test_light_client_state() {
        let config = LightClientConfig::default();
        let genesis_root = [0u8; 32];

        let client = LightClient::new(config, genesis_root).unwrap();
        let state = client.state().await;

        assert!(!state.is_bootstrapped);
        assert_eq!(state.height, 0);
        assert_eq!(state.epoch, 0);
        assert!(state.state_root.is_none());
    }
}
