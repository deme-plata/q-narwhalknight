//! Recursive Proofs API
//!
//! Provides HTTP endpoints for:
//! - Light client bootstrap (trustless ~10ms verification)
//! - Epoch proof submission and verification
//! - Prover node management
//! - Proof status queries

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use q_recursive_proofs::{
    light_client::{LightClient, LightClientConfig, LightClientSync},
    protocol::{
        messages::{
            EpochProofSubmission, EpochProofTask,
            LightClientProofRequest, LightClientProofResponse, RewardParams,
        },
        prover_node::{ProverMetrics, ProverNode, ProverNodeConfig},
    },
    EpochProof,
};
use q_storage::StorageEngine;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use crate::AppState;

/// Prover initialization status
#[derive(Clone, Debug)]
pub enum ProverStatus {
    /// Not enabled
    Disabled,
    /// Initializing in background
    Initializing,
    /// Ready for use
    Ready,
    /// Initialization failed
    Failed(String),
}

/// Recursive proofs service state
pub struct RecursiveProofsService {
    /// Light client for trustless bootstrap
    pub light_client: Arc<LightClient>,
    /// Light client sync manager
    pub sync: Arc<LightClientSync>,
    /// Prover node (optional - populated asynchronously)
    /// Uses RwLock to allow async initialization without blocking startup
    pub prover_node: Arc<RwLock<Option<Arc<ProverNode>>>>,
    /// Prover initialization status
    pub prover_status: Arc<RwLock<ProverStatus>>,
    /// Storage engine for proof persistence
    pub storage: Arc<StorageEngine>,
    /// Current epoch
    pub current_epoch: Arc<RwLock<u64>>,
    /// Genesis state root
    pub genesis_state_root: [u8; 32],
    /// Reward parameters
    pub reward_params: RewardParams,
    /// Node peer ID
    pub peer_id: String,
}

impl RecursiveProofsService {
    /// Column family for recursive proofs (using manifest CF)
    const CF_PROOFS: &'static str = "manifest";

    /// Create new recursive proofs service
    ///
    /// If `enable_prover` is true, the prover node will be initialized
    /// asynchronously in the background to avoid blocking server startup.
    /// Use `get_prover_status()` to check initialization progress.
    pub fn new(
        storage: Arc<StorageEngine>,
        genesis_state_root: [u8; 32],
        peer_id: String,
        enable_prover: bool,
    ) -> anyhow::Result<Self> {
        info!("Initializing recursive proofs service");

        // Create light client
        let light_client_config = LightClientConfig::default();
        let light_client = Arc::new(LightClient::new(light_client_config, genesis_state_root)?);

        // Create sync manager
        let sync = Arc::new(LightClientSync::new(Arc::clone(&light_client), 60));

        // Create placeholders for async prover initialization
        let prover_node: Arc<RwLock<Option<Arc<ProverNode>>>> = Arc::new(RwLock::new(None));
        let prover_status = Arc::new(RwLock::new(if enable_prover {
            ProverStatus::Initializing
        } else {
            ProverStatus::Disabled
        }));

        // Spawn async prover initialization in background (non-blocking!)
        if enable_prover {
            info!("🚀 Prover node initialization started in background...");
            info!("   Server will start immediately while SRS generates");

            let prover_node_clone = Arc::clone(&prover_node);
            let prover_status_clone = Arc::clone(&prover_status);
            let peer_id_clone = peer_id.clone();

            tokio::spawn(async move {
                Self::initialize_prover_async(
                    prover_node_clone,
                    prover_status_clone,
                    peer_id_clone,
                ).await;
            });
        }

        Ok(Self {
            light_client,
            sync,
            prover_node,
            prover_status,
            storage,
            current_epoch: Arc::new(RwLock::new(0)),
            genesis_state_root,
            reward_params: RewardParams::default(),
            peer_id,
        })
    }

    /// Initialize prover node asynchronously
    ///
    /// This runs in a background task and populates the prover_node field
    /// when SRS generation is complete.
    async fn initialize_prover_async(
        prover_node: Arc<RwLock<Option<Arc<ProverNode>>>>,
        prover_status: Arc<RwLock<ProverStatus>>,
        peer_id: String,
    ) {
        info!("📦 Background prover initialization starting...");
        let start = std::time::Instant::now();

        // Use tokio::task::spawn_blocking for CPU-intensive SRS generation
        let result = tokio::task::spawn_blocking(move || {
            let prover_config = ProverNodeConfig {
                peer_id,
                ..Default::default()
            };
            ProverNode::new(prover_config)
        }).await;

        match result {
            Ok(Ok(node)) => {
                let duration = start.elapsed();
                info!("✅ Prover node initialized in {:?}", duration);
                info!("   Light mode SRS ready (~200K constraints)");

                // Update the shared state
                {
                    let mut prover = prover_node.write().await;
                    *prover = Some(Arc::new(node));
                }
                {
                    let mut status = prover_status.write().await;
                    *status = ProverStatus::Ready;
                }
            }
            Ok(Err(e)) => {
                error!("❌ Prover initialization failed: {}", e);
                let mut status = prover_status.write().await;
                *status = ProverStatus::Failed(e.to_string());
            }
            Err(e) => {
                error!("❌ Prover initialization task panicked: {}", e);
                let mut status = prover_status.write().await;
                *status = ProverStatus::Failed(format!("Task panic: {}", e));
            }
        }
    }

    /// Get prover initialization status
    pub async fn get_prover_status(&self) -> ProverStatus {
        self.prover_status.read().await.clone()
    }

    /// Check if prover is ready
    pub async fn is_prover_ready(&self) -> bool {
        matches!(*self.prover_status.read().await, ProverStatus::Ready)
    }

    /// Get prover node (if initialized)
    pub async fn get_prover(&self) -> Option<Arc<ProverNode>> {
        self.prover_node.read().await.clone()
    }

    /// Update current epoch
    pub async fn set_epoch(&self, epoch: u64) {
        let mut current = self.current_epoch.write().await;
        *current = epoch;
    }

    /// Get current epoch
    pub async fn get_epoch(&self) -> u64 {
        *self.current_epoch.read().await
    }

    /// Store epoch proof in database
    pub async fn store_epoch_proof(&self, epoch: u64, proof: &EpochProof) -> anyhow::Result<()> {
        let key = format!("recursive_proof:epoch:{}", epoch);
        let value = bincode::serialize(proof)?;
        let kv = self.storage.get_kv();
        kv.put(Self::CF_PROOFS, key.as_bytes(), &value).await?;
        info!("Stored epoch proof for epoch {}", epoch);
        Ok(())
    }

    /// Load epoch proof from database
    pub async fn load_epoch_proof(&self, epoch: u64) -> anyhow::Result<Option<EpochProof>> {
        let key = format!("recursive_proof:epoch:{}", epoch);
        let kv = self.storage.get_kv();
        match kv.get(Self::CF_PROOFS, key.as_bytes()).await? {
            Some(data) => {
                let proof: EpochProof = bincode::deserialize(&data)?;
                Ok(Some(proof))
            }
            None => Ok(None),
        }
    }

    /// Get accumulated proof for light client bootstrap
    pub async fn get_light_client_proof(&self) -> anyhow::Result<Option<LightClientProofResponse>> {
        let current_epoch = self.get_epoch().await;

        // Load the latest epoch proof
        if let Some(proof) = self.load_epoch_proof(current_epoch).await? {
            // Get current state from storage
            let current_height = self.storage.get_highest_contiguous_block().await.unwrap_or(0);
            let kv = self.storage.get_kv();
            let current_state_root = kv.get(Self::CF_PROOFS, b"state_root").await?
                .map(|v| {
                    let mut root = [0u8; 32];
                    root.copy_from_slice(&v[..32.min(v.len())]);
                    root
                })
                .unwrap_or([0u8; 32]);

            let response = LightClientProofResponse {
                proof_data: bincode::serialize(&proof.proof)?,
                current_state_root,
                current_height,
                current_epoch,
                validator_set: None,
                validator_set_proof: None,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                responder_peer_id: self.peer_id.clone(),
            };

            Ok(Some(response))
        } else {
            Ok(None)
        }
    }
}

// ============================================================================
// API Request/Response Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct LightClientBootstrapRequest {
    pub known_height: Option<u64>,
    pub include_validators: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LightClientBootstrapResponse {
    pub success: bool,
    pub verified_height: u64,
    pub verified_epoch: u64,
    pub state_root: Option<String>,
    pub verification_time_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EpochProofStatusResponse {
    pub current_epoch: u64,
    pub latest_proven_epoch: u64,
    pub proof_status: String,
    pub blocks_in_epoch: u64,
    pub estimated_next_proof_secs: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitEpochProofRequest {
    pub submission: EpochProofSubmission,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitEpochProofResponse {
    pub accepted: bool,
    pub is_valid: bool,
    pub verification_time_ms: u64,
    pub reward: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProverNodeStatusResponse {
    pub enabled: bool,
    /// Prover status: "disabled", "initializing", "ready", "failed"
    pub status: String,
    pub metrics: Option<ProverMetricsJson>,
    pub active_tasks: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProverMetricsJson {
    pub proofs_generated: u64,
    pub proofs_accepted: u64,
    pub rewards_earned: u64,
    pub avg_proving_time_ms: f64,
    pub active_tasks: usize,
}

impl From<ProverMetrics> for ProverMetricsJson {
    fn from(m: ProverMetrics) -> Self {
        Self {
            proofs_generated: m.proofs_generated,
            proofs_accepted: m.proofs_accepted,
            rewards_earned: m.rewards_earned,
            avg_proving_time_ms: m.avg_proving_time_ms,
            active_tasks: m.active_tasks,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LightClientStateResponse {
    pub is_bootstrapped: bool,
    pub verified_height: u64,
    pub verified_epoch: u64,
    pub state_root: Option<String>,
    pub last_verified: u64,
}

// ============================================================================
// API Handlers - Use AppState and extract recursive_proofs_service
// ============================================================================

/// GET /api/v1/recursive-proofs/light-client/state
pub async fn get_light_client_state(
    State(app_state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let service = match &app_state.recursive_proofs_service {
        Some(s) => s,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "success": false,
            "error": "Recursive proofs service not enabled"
        }))).into_response(),
    };

    let lc_state = service.light_client.state().await;

    Json(LightClientStateResponse {
        is_bootstrapped: lc_state.is_bootstrapped,
        verified_height: lc_state.height,
        verified_epoch: lc_state.epoch,
        state_root: lc_state.state_root.map(|r| hex::encode(r)),
        last_verified: lc_state.last_verified,
    }).into_response()
}

/// POST /api/v1/recursive-proofs/light-client/bootstrap
pub async fn request_bootstrap(
    State(app_state): State<Arc<AppState>>,
    Json(_request): Json<LightClientBootstrapRequest>,
) -> impl IntoResponse {
    let service = match &app_state.recursive_proofs_service {
        Some(s) => s,
        None => return Json(LightClientBootstrapResponse {
            success: false,
            verified_height: 0,
            verified_epoch: 0,
            state_root: None,
            verification_time_ms: 0,
            error: Some("Recursive proofs service not enabled".to_string()),
        }).into_response(),
    };

    let start = std::time::Instant::now();

    if service.light_client.is_bootstrapped().await {
        return Json(LightClientBootstrapResponse {
            success: true,
            verified_height: service.light_client.verified_height().await,
            verified_epoch: service.light_client.state().await.epoch,
            state_root: service.light_client.verified_state_root().await.map(|r| hex::encode(r)),
            verification_time_ms: 0,
            error: Some("Already bootstrapped".to_string()),
        }).into_response();
    }

    match service.get_light_client_proof().await {
        Ok(Some(proof_response)) => {
            match service.light_client.bootstrap(proof_response).await {
                Ok(()) => {
                    let verification_time = start.elapsed();
                    let lc_state = service.light_client.state().await;
                    info!("Light client bootstrap successful! Verified {} blocks in {:?}", lc_state.height, verification_time);
                    Json(LightClientBootstrapResponse {
                        success: true,
                        verified_height: lc_state.height,
                        verified_epoch: lc_state.epoch,
                        state_root: lc_state.state_root.map(|r| hex::encode(r)),
                        verification_time_ms: verification_time.as_millis() as u64,
                        error: None,
                    }).into_response()
                }
                Err(e) => {
                    error!("Light client bootstrap failed: {}", e);
                    Json(LightClientBootstrapResponse {
                        success: false,
                        verified_height: 0,
                        verified_epoch: 0,
                        state_root: None,
                        verification_time_ms: start.elapsed().as_millis() as u64,
                        error: Some(format!("Bootstrap failed: {}", e)),
                    }).into_response()
                }
            }
        }
        Ok(None) => Json(LightClientBootstrapResponse {
            success: false,
            verified_height: 0,
            verified_epoch: 0,
            state_root: None,
            verification_time_ms: 0,
            error: Some("No bootstrap proof available yet".to_string()),
        }).into_response(),
        Err(e) => Json(LightClientBootstrapResponse {
            success: false,
            verified_height: 0,
            verified_epoch: 0,
            state_root: None,
            verification_time_ms: 0,
            error: Some(format!("Failed to get proof: {}", e)),
        }).into_response(),
    }
}

/// GET /api/v1/recursive-proofs/light-client/proof
pub async fn get_bootstrap_proof(
    State(app_state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let service = match &app_state.recursive_proofs_service {
        Some(s) => s,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "success": false,
            "error": "Recursive proofs service not enabled"
        }))),
    };

    match service.get_light_client_proof().await {
        Ok(Some(proof)) => (StatusCode::OK, Json(serde_json::json!({
            "success": true,
            "proof": proof,
        }))),
        Ok(None) => (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "success": false,
            "error": "No proof available yet"
        }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "success": false,
            "error": format!("Failed to get proof: {}", e)
        }))),
    }
}

/// GET /api/v1/recursive-proofs/epochs/:epoch
pub async fn get_epoch_proof(
    State(app_state): State<Arc<AppState>>,
    Path(epoch): Path<u64>,
) -> impl IntoResponse {
    let service = match &app_state.recursive_proofs_service {
        Some(s) => s,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "success": false,
            "error": "Recursive proofs service not enabled"
        }))),
    };

    match service.load_epoch_proof(epoch).await {
        Ok(Some(proof)) => (StatusCode::OK, Json(serde_json::json!({
            "success": true,
            "epoch": epoch,
            "public_inputs": {
                "previous_state_root": hex::encode(proof.public_inputs.previous_state_root),
                "current_state_root": hex::encode(proof.public_inputs.current_state_root),
                "epoch": proof.public_inputs.epoch,
                "height_range": proof.public_inputs.height_range,
                "validator_set_hash": hex::encode(proof.public_inputs.validator_set_hash),
                "signature_count": proof.public_inputs.signature_count,
            },
            "metadata": {
                "version": proof.metadata.version,
                "prover_peer_id": proof.metadata.prover_peer_id,
                "proving_time_ms": proof.metadata.proving_time_ms,
                "created_at": proof.metadata.created_at,
            }
        }))),
        Ok(None) => (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "success": false,
            "error": format!("No proof found for epoch {}", epoch)
        }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "success": false,
            "error": format!("Failed to load proof: {}", e)
        }))),
    }
}

/// GET /api/v1/recursive-proofs/status
pub async fn get_proof_status(
    State(app_state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let service = match &app_state.recursive_proofs_service {
        Some(s) => s,
        None => return Json(EpochProofStatusResponse {
            current_epoch: 0,
            latest_proven_epoch: 0,
            proof_status: "service_disabled".to_string(),
            blocks_in_epoch: 0,
            estimated_next_proof_secs: None,
        }).into_response(),
    };

    let current_epoch = service.get_epoch().await;

    // Find latest proven epoch
    let mut latest_proven = 0u64;
    for epoch in (0..=current_epoch).rev() {
        if service.load_epoch_proof(epoch).await.ok().flatten().is_some() {
            latest_proven = epoch;
            break;
        }
    }

    let status = if latest_proven == current_epoch {
        "up_to_date"
    } else if service.is_prover_ready().await {
        "proving_in_progress"
    } else {
        "awaiting_proof"
    };

    Json(EpochProofStatusResponse {
        current_epoch,
        latest_proven_epoch: latest_proven,
        proof_status: status.to_string(),
        blocks_in_epoch: 1000,
        estimated_next_proof_secs: if status == "up_to_date" { None } else { Some(30) },
    }).into_response()
}

/// POST /api/v1/recursive-proofs/submit
pub async fn submit_epoch_proof(
    State(app_state): State<Arc<AppState>>,
    Json(request): Json<SubmitEpochProofRequest>,
) -> impl IntoResponse {
    let service = match &app_state.recursive_proofs_service {
        Some(s) => s,
        None => return Json(SubmitEpochProofResponse {
            accepted: false,
            is_valid: false,
            verification_time_ms: 0,
            reward: None,
            error: Some("Recursive proofs service not enabled".to_string()),
        }),
    };

    let start = std::time::Instant::now();
    let submission = request.submission;
    info!("Received epoch proof submission for epoch {}", submission.epoch);

    let is_valid = if let Some(prover) = service.get_prover().await {
        match prover.verify_proof(&submission).await {
            Ok(valid) => valid,
            Err(e) => {
                error!("Proof verification error: {}", e);
                return Json(SubmitEpochProofResponse {
                    accepted: false,
                    is_valid: false,
                    verification_time_ms: start.elapsed().as_millis() as u64,
                    reward: None,
                    error: Some(format!("Verification error: {}", e)),
                });
            }
        }
    } else {
        warn!("No prover node available for verification (status: {:?})", service.get_prover_status().await);
        false
    };

    let verification_time = start.elapsed();

    if is_valid {
        let task = EpochProofTask {
            epoch: submission.epoch,
            deadline: submission.created_at + 60,
            ..Default::default()
        };
        let reward = service.reward_params.calculate_reward(&submission, &task);

        if let Ok((proof, public_inputs)) = submission.to_proof() {
            let epoch_proof = EpochProof {
                proof,
                public_inputs,
                metadata: q_recursive_proofs::EpochProofMetadata {
                    version: submission.protocol_version,
                    prover_peer_id: Some(submission.prover_peer_id.clone()),
                    proving_time_ms: submission.proving_time_ms,
                    hardware_info: submission.hardware_info.as_ref().map(|h| format!("{:?}", h)),
                    created_at: submission.created_at,
                },
            };

            if let Err(e) = service.store_epoch_proof(submission.epoch, &epoch_proof).await {
                error!("Failed to store epoch proof: {}", e);
            }
        }

        info!("Epoch proof for epoch {} accepted! Reward: {} QNK", submission.epoch, reward);
        Json(SubmitEpochProofResponse {
            accepted: true,
            is_valid: true,
            verification_time_ms: verification_time.as_millis() as u64,
            reward: Some(reward),
            error: None,
        })
    } else {
        warn!("Epoch proof for epoch {} rejected - invalid", submission.epoch);
        Json(SubmitEpochProofResponse {
            accepted: false,
            is_valid: false,
            verification_time_ms: verification_time.as_millis() as u64,
            reward: None,
            error: Some("Proof verification failed".to_string()),
        })
    }
}

/// GET /api/v1/recursive-proofs/prover/status
pub async fn get_prover_status(
    State(app_state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let service = match &app_state.recursive_proofs_service {
        Some(s) => s,
        None => return Json(ProverNodeStatusResponse {
            enabled: false,
            status: "disabled".to_string(),
            metrics: None,
            active_tasks: 0,
        }),
    };

    let prover_status = service.get_prover_status().await;
    let status_str = match &prover_status {
        ProverStatus::Disabled => "disabled",
        ProverStatus::Initializing => "initializing",
        ProverStatus::Ready => "ready",
        ProverStatus::Failed(_) => "failed",
    };

    match service.get_prover().await {
        Some(prover) => {
            let metrics = prover.metrics().await;
            let active_tasks = metrics.active_tasks;
            Json(ProverNodeStatusResponse {
                enabled: true,
                status: status_str.to_string(),
                metrics: Some(metrics.into()),
                active_tasks,
            })
        }
        None => Json(ProverNodeStatusResponse {
            enabled: matches!(prover_status, ProverStatus::Initializing),
            status: status_str.to_string(),
            metrics: None,
            active_tasks: 0,
        }),
    }
}

/// POST /api/v1/recursive-proofs/prover/task
pub async fn submit_proving_task(
    State(app_state): State<Arc<AppState>>,
    Json(task): Json<EpochProofTask>,
) -> impl IntoResponse {
    let service = match &app_state.recursive_proofs_service {
        Some(s) => s,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "success": false,
            "error": "Recursive proofs service not enabled"
        }))),
    };

    info!("Received proving task for epoch {}", task.epoch);

    let prover_status = service.get_prover_status().await;
    if let Some(prover) = service.get_prover().await {
        match prover.handle_task(task.clone()).await {
            Ok(()) => (StatusCode::ACCEPTED, Json(serde_json::json!({
                "success": true,
                "message": format!("Proving task for epoch {} accepted", task.epoch)
            }))),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "success": false,
                "error": format!("Failed to start proving task: {}", e)
            }))),
        }
    } else {
        let status_msg = match prover_status {
            ProverStatus::Initializing => "Prover node still initializing, please try again later",
            ProverStatus::Failed(ref e) => &format!("Prover initialization failed: {}", e),
            _ => "Prover node not enabled on this node",
        };
        (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "success": false,
            "error": status_msg
        })))
    }
}

// ============================================================================
// P2P Integration
// ============================================================================

/// Handle incoming P2P messages for recursive proofs
pub struct RecursiveProofsP2PHandler {
    service: Arc<RecursiveProofsService>,
}

impl RecursiveProofsP2PHandler {
    pub fn new(service: Arc<RecursiveProofsService>) -> Self {
        Self { service }
    }

    /// Handle incoming epoch proof task from gossipsub
    pub async fn handle_epoch_task(&self, data: &[u8]) -> anyhow::Result<()> {
        let task: EpochProofTask = bincode::deserialize(data)?;
        info!("P2P: Received epoch proof task for epoch {}", task.epoch);

        if let Some(prover) = self.service.get_prover().await {
            prover.handle_task(task).await?;
        } else {
            info!("P2P: Prover not ready, ignoring task (status: {:?})", self.service.get_prover_status().await);
        }
        Ok(())
    }

    /// Handle incoming epoch proof submission from gossipsub
    pub async fn handle_epoch_proof(&self, data: &[u8]) -> anyhow::Result<()> {
        let submission: EpochProofSubmission = bincode::deserialize(data)?;
        info!("P2P: Received epoch proof for epoch {} from {}", submission.epoch, submission.prover_peer_id);

        if let Some(prover) = self.service.get_prover().await {
            if prover.verify_proof(&submission).await? {
                if let Ok((proof, public_inputs)) = submission.to_proof() {
                    let epoch_proof = EpochProof {
                        proof,
                        public_inputs,
                        metadata: q_recursive_proofs::EpochProofMetadata {
                            version: submission.protocol_version,
                            prover_peer_id: Some(submission.prover_peer_id),
                            proving_time_ms: submission.proving_time_ms,
                            hardware_info: None,
                            created_at: submission.created_at,
                        },
                    };
                    self.service.store_epoch_proof(submission.epoch, &epoch_proof).await?;
                }
            }
        } else {
            info!("P2P: Prover not ready, cannot verify proof (status: {:?})", self.service.get_prover_status().await);
        }
        Ok(())
    }

    /// Handle light client proof request from P2P
    pub async fn handle_light_client_request(&self, data: &[u8]) -> anyhow::Result<Vec<u8>> {
        let _request: LightClientProofRequest = bincode::deserialize(data)?;
        info!("P2P: Received light client proof request");

        match self.service.get_light_client_proof().await? {
            Some(response) => Ok(bincode::serialize(&response)?),
            None => Err(anyhow::anyhow!("No proof available")),
        }
    }
}
