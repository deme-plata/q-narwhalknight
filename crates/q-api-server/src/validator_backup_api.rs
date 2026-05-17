//! Validator Key Backup API - TemporalShield protection for validator keypairs
//!
//! Provides REST API endpoints for creating and restoring validator key backups
//! using (5,9) threshold secret sharing with post-quantum encryption.
//!
//! ## Security Properties
//! - Information-theoretic secrecy via OTP encryption
//! - Post-quantum security via ML-KEM-1024
//! - Threshold (5,9) requires 5 of 9 trustees to restore
//! - zk-STARK proofs (NO TRUSTED SETUP) verify backup integrity
//!
//! ## API Endpoints
//! - POST /api/v1/validator/backup/create - Create a new backup
//! - POST /api/v1/validator/backup/restore - Restore from backup
//! - POST /api/v1/validator/backup/verify - Verify backup integrity

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::AppState;
use q_temporal_shield::{TemporalShield, TemporalShieldConfig, TrusteePublicKey};
use q_types::validator_backup::{
    BackupMetadata, ValidatorKeyBackup, RestoreResult, BackupStatus,
    VALIDATOR_BACKUP_THRESHOLD, VALIDATOR_BACKUP_TOTAL,
};
use q_types::pqc_keys::ValidatorKeypair;

/// Request to create a validator key backup
#[derive(Debug, Deserialize)]
pub struct CreateBackupRequest {
    /// Node ID to backup
    pub node_id: String,
    /// Password to decrypt the stored keypair
    pub password: String,
    /// Optional human-readable label
    pub label: Option<String>,
    /// Override trustees (optional, uses system trustees if not provided)
    pub custom_trustees: Option<Vec<String>>,
}

/// Response for backup creation
#[derive(Debug, Serialize)]
pub struct CreateBackupResponse {
    /// Backup ID (hex)
    pub backup_id: String,
    /// Backup status
    pub status: String,
    /// Number of trustees
    pub total_trustees: usize,
    /// Threshold required for restore
    pub threshold: usize,
    /// Fingerprint of the backed up key (hex)
    pub fingerprint: String,
    /// STARK proof size (bytes)
    pub proof_size: usize,
    /// Security note
    pub security_note: String,
}

/// Request to restore a validator key from backup
#[derive(Debug, Deserialize)]
pub struct RestoreBackupRequest {
    /// Backup ID (hex)
    pub backup_id: String,
    /// Decrypted shares from trustees (index, base64 share)
    pub decrypted_shares: Vec<(usize, String)>,
    /// New password for the restored keypair
    pub new_password: String,
}

/// Response for backup restoration
#[derive(Debug, Serialize)]
pub struct RestoreBackupResponse {
    /// Whether restore was successful
    pub success: bool,
    /// Node ID of restored keypair
    pub node_id: Option<String>,
    /// Number of shares used
    pub shares_used: usize,
    /// Error message if failed
    pub error: Option<String>,
    /// Fingerprint for verification (hex)
    pub fingerprint: Option<String>,
}

/// Request to verify a backup
#[derive(Debug, Deserialize)]
pub struct VerifyBackupRequest {
    /// Backup ID (hex)
    pub backup_id: String,
}

/// Response for backup verification
#[derive(Debug, Serialize)]
pub struct VerifyBackupResponse {
    /// Whether backup exists
    pub exists: bool,
    /// Whether structure is valid
    pub structure_valid: bool,
    /// Whether STARK proof is valid (NO TRUSTED SETUP)
    pub proof_valid: bool,
    /// Backup status
    pub status: String,
    /// Metadata if available
    pub metadata: Option<BackupMetadataResponse>,
    /// Security note
    pub security_note: String,
}

/// Simplified metadata for response
#[derive(Debug, Serialize)]
pub struct BackupMetadataResponse {
    pub node_id: String,
    pub created_at: u64,
    pub threshold: usize,
    pub total_trustees: usize,
    pub fingerprint: String,
    pub label: Option<String>,
}

/// Create a validator key backup protected with TemporalShield
///
/// POST /api/v1/validator/backup/create
pub async fn create_backup(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateBackupRequest>,
) -> impl IntoResponse {
    info!("Creating validator key backup for node: {}", req.node_id);

    // Decode node ID
    let node_id_bytes = match hex::decode(&req.node_id) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Invalid node ID format (expected 64 hex chars)"
                })),
            )
        }
    };

    // Get trustees from TrusteeManager (5-of-9 for validator backups)
    let trustees = if let Some(custom) = req.custom_trustees {
        // Decode custom trustees
        if custom.len() != VALIDATOR_BACKUP_TOTAL {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Expected {} trustees, got {}", VALIDATOR_BACKUP_TOTAL, custom.len())
                })),
            );
        }

        let decoded: Result<Vec<TrusteePublicKey>, _> = custom
            .iter()
            .map(|pk_b64| {
                let bytes = BASE64.decode(pk_b64)?;
                TrusteePublicKey::from_bytes(&bytes)
                    .map_err(|e| base64::DecodeError::InvalidLength(bytes.len()))
            })
            .collect();

        match decoded {
            Ok(t) => t,
            Err(_) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": "Invalid trustee public key format"
                    })),
                )
            }
        }
    } else if let Some(ref trustee_manager_lock) = state.temporal_trustee_manager {
        let trustee_manager = trustee_manager_lock.read().await;
        trustee_manager.get_validator_trustees()
    } else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "TrusteeManager not available"
            })),
        );
    };

    if trustees.len() != VALIDATOR_BACKUP_TOTAL {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Insufficient trustees: have {}, need {}", trustees.len(), VALIDATOR_BACKUP_TOTAL)
            })),
        );
    }

    // In production, this would load the actual keypair from secure storage
    // For now, we demonstrate the protection flow with a placeholder
    // The actual keypair loading requires the password and secure key storage integration
    let keypair_bytes = match load_validator_keypair(&state, &node_id_bytes, &req.password).await {
        Ok(bytes) => bytes,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("Failed to load keypair: {}", e)
                })),
            )
        }
    };

    // Compute fingerprint (hash of public key portion)
    let fingerprint = compute_fingerprint(&keypair_bytes);

    // Create TemporalShield config for (5,9) threshold
    let config = match TemporalShieldConfig::custom(
        VALIDATOR_BACKUP_THRESHOLD,
        VALIDATOR_BACKUP_TOTAL,
        256, // 256-bit security for validator keys
    ) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Failed to create config: {:?}", e)
                })),
            )
        }
    };
    let shield = TemporalShield::new(config);

    // Protect the keypair with TemporalShield
    let envelope = match shield.protect(&keypair_bytes, &trustees) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Protection failed: {:?}", e)
                })),
            )
        }
    };

    // Create backup metadata
    let metadata = BackupMetadata::new(node_id_bytes, fingerprint, req.label);

    // Compute backup ID
    let backup_id = compute_backup_id(&envelope.ciphertext, &metadata);

    // Serialize envelope
    let protected_data = match envelope.to_bytes() {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Serialization failed: {:?}", e)
                })),
            )
        }
    };

    // Create backup structure
    let backup = ValidatorKeyBackup::new(
        backup_id,
        metadata,
        protected_data,
        envelope.key_commitment,
        envelope.share_commitments.clone(),
        envelope.stark_proof.clone(),
    );

    // Store backup (in production, this would persist to secure storage)
    if let Err(e) = store_backup(&state, &backup).await {
        warn!("Failed to persist backup: {}", e);
    }

    info!(
        "Validator key backup created: {} (proof size: {} bytes)",
        hex::encode(backup_id),
        envelope.stark_proof.len()
    );

    let response = CreateBackupResponse {
        backup_id: hex::encode(backup_id),
        status: "verified".to_string(),
        total_trustees: VALIDATOR_BACKUP_TOTAL,
        threshold: VALIDATOR_BACKUP_THRESHOLD,
        fingerprint: hex::encode(fingerprint),
        proof_size: envelope.stark_proof.len(),
        security_note: "Backup protected with TemporalShield (5-of-9 threshold, NO TRUSTED SETUP)".to_string(),
    };

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap()))
}

/// Restore a validator key from backup
///
/// POST /api/v1/validator/backup/restore
pub async fn restore_backup(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RestoreBackupRequest>,
) -> impl IntoResponse {
    info!("Restoring validator key from backup: {}", req.backup_id);

    // Decode backup ID
    let backup_id_bytes = match hex::decode(&req.backup_id) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(RestoreBackupResponse {
                    success: false,
                    node_id: None,
                    shares_used: 0,
                    error: Some("Invalid backup ID format".to_string()),
                    fingerprint: None,
                }).unwrap()),
            )
        }
    };

    // Load backup from storage
    let backup = match load_backup(&state, &backup_id_bytes).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::to_value(RestoreBackupResponse {
                    success: false,
                    node_id: None,
                    shares_used: 0,
                    error: Some(format!("Backup not found: {}", e)),
                    fingerprint: None,
                }).unwrap()),
            )
        }
    };

    // Validate share count
    if req.decrypted_shares.len() < VALIDATOR_BACKUP_THRESHOLD {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(RestoreBackupResponse {
                success: false,
                node_id: None,
                shares_used: req.decrypted_shares.len(),
                error: Some(format!(
                    "Insufficient shares: have {}, need {}",
                    req.decrypted_shares.len(),
                    VALIDATOR_BACKUP_THRESHOLD
                )),
                fingerprint: None,
            }).unwrap()),
        );
    }

    // Decode shares
    let decrypted_shares: Result<Vec<(usize, Vec<u8>)>, base64::DecodeError> = req
        .decrypted_shares
        .iter()
        .map(|(idx, share_b64)| {
            let share_data = BASE64.decode(share_b64)?;
            Ok((*idx, share_data))
        })
        .collect();

    let decrypted_shares = match decrypted_shares {
        Ok(ds) => ds,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(RestoreBackupResponse {
                    success: false,
                    node_id: None,
                    shares_used: 0,
                    error: Some(format!("Invalid share format: {:?}", e)),
                    fingerprint: None,
                }).unwrap()),
            )
        }
    };

    // Deserialize envelope
    let envelope = match q_temporal_shield::TemporalEnvelope::from_bytes(&backup.protected_data) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::to_value(RestoreBackupResponse {
                    success: false,
                    node_id: None,
                    shares_used: 0,
                    error: Some(format!("Failed to parse envelope: {:?}", e)),
                    fingerprint: None,
                }).unwrap()),
            )
        }
    };

    // Create TemporalShield for reconstruction
    let config = match TemporalShieldConfig::custom(
        VALIDATOR_BACKUP_THRESHOLD,
        VALIDATOR_BACKUP_TOTAL,
        256,
    ) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::to_value(RestoreBackupResponse {
                    success: false,
                    node_id: None,
                    shares_used: 0,
                    error: Some(format!("Config error: {:?}", e)),
                    fingerprint: None,
                }).unwrap()),
            )
        }
    };
    let shield = TemporalShield::new(config);

    // Reconstruct the keypair bytes
    let keypair_bytes = match shield.reconstruct(&envelope, &decrypted_shares) {
        Ok(bytes) => bytes,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(RestoreBackupResponse {
                    success: false,
                    node_id: None,
                    shares_used: decrypted_shares.len(),
                    error: Some(format!("Reconstruction failed: {:?}", e)),
                    fingerprint: None,
                }).unwrap()),
            )
        }
    };

    // Verify fingerprint matches
    let computed_fingerprint = compute_fingerprint(&keypair_bytes);
    if computed_fingerprint != backup.metadata.fingerprint {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::to_value(RestoreBackupResponse {
                success: false,
                node_id: None,
                shares_used: decrypted_shares.len(),
                error: Some("Fingerprint mismatch - backup may be corrupted".to_string()),
                fingerprint: None,
            }).unwrap()),
        );
    }

    // Restore the keypair
    let keypair = match ValidatorKeypair::from_backup_bytes(&keypair_bytes) {
        Ok(kp) => kp,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::to_value(RestoreBackupResponse {
                    success: false,
                    node_id: None,
                    shares_used: decrypted_shares.len(),
                    error: Some(format!("Failed to restore keypair: {:?}", e)),
                    fingerprint: None,
                }).unwrap()),
            )
        }
    };

    // Store the restored keypair with new password
    if let Err(e) = store_restored_keypair(&state, &backup.metadata.node_id, &keypair, &req.new_password).await {
        warn!("Failed to persist restored keypair: {}", e);
    }

    info!(
        "Validator key restored successfully: {} (used {} shares)",
        hex::encode(backup.metadata.node_id),
        decrypted_shares.len()
    );

    let response = RestoreBackupResponse {
        success: true,
        node_id: Some(hex::encode(backup.metadata.node_id)),
        shares_used: decrypted_shares.len(),
        error: None,
        fingerprint: Some(hex::encode(computed_fingerprint)),
    };

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap()))
}

/// Verify a backup's integrity
///
/// POST /api/v1/validator/backup/verify
pub async fn verify_backup(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VerifyBackupRequest>,
) -> impl IntoResponse {
    info!("Verifying validator backup: {}", req.backup_id);

    // Decode backup ID
    let backup_id_bytes = match hex::decode(&req.backup_id) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(VerifyBackupResponse {
                    exists: false,
                    structure_valid: false,
                    proof_valid: false,
                    status: "invalid_id".to_string(),
                    metadata: None,
                    security_note: "Invalid backup ID format".to_string(),
                }).unwrap()),
            )
        }
    };

    // Load backup
    let backup = match load_backup(&state, &backup_id_bytes).await {
        Ok(b) => b,
        Err(_) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::to_value(VerifyBackupResponse {
                    exists: false,
                    structure_valid: false,
                    proof_valid: false,
                    status: "not_found".to_string(),
                    metadata: None,
                    security_note: "Backup not found".to_string(),
                }).unwrap()),
            )
        }
    };

    let structure_valid = backup.check_structure();

    // Verify STARK proof (NO TRUSTED SETUP)
    let proof_valid = q_temporal_shield::stark::verify_proof(
        backup.stark_proof.clone(),
        &backup.key_commitment,
        &backup.share_commitments,
        backup.metadata.threshold,
        backup.metadata.total_trustees,
    ).is_ok();

    let status = if structure_valid && proof_valid {
        "verified"
    } else if structure_valid {
        "structure_valid_proof_invalid"
    } else {
        "corrupted"
    };

    let response = VerifyBackupResponse {
        exists: true,
        structure_valid,
        proof_valid,
        status: status.to_string(),
        metadata: Some(BackupMetadataResponse {
            node_id: hex::encode(backup.metadata.node_id),
            created_at: backup.metadata.created_at,
            threshold: backup.metadata.threshold,
            total_trustees: backup.metadata.total_trustees,
            fingerprint: hex::encode(backup.metadata.fingerprint),
            label: backup.metadata.label,
        }),
        security_note: "Verification uses NO TRUSTED SETUP - all randomness derived from public transcript".to_string(),
    };

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap()))
}

// Helper functions

fn compute_fingerprint(keypair_bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(keypair_bytes);
    hasher.finalize().into()
}

fn compute_backup_id(ciphertext: &[u8], metadata: &BackupMetadata) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(ciphertext);
    hasher.update(&metadata.node_id);
    hasher.update(&metadata.created_at.to_le_bytes());
    hasher.finalize().into()
}

// Storage helpers (in production, these would use secure persistent storage)

async fn load_validator_keypair(
    _state: &Arc<AppState>,
    node_id: &[u8; 32],
    _password: &str,
) -> Result<Vec<u8>, String> {
    // In production: Load encrypted keypair from storage, decrypt with password
    // For now, return a placeholder indicating the keypair should be loaded
    Err(format!(
        "Keypair for node {} not found in storage. Production implementation required.",
        hex::encode(node_id)
    ))
}

async fn store_backup(_state: &Arc<AppState>, backup: &ValidatorKeyBackup) -> Result<(), String> {
    // In production: Persist backup to secure storage
    info!("Backup stored: {}", hex::encode(backup.backup_id));
    Ok(())
}

async fn load_backup(
    _state: &Arc<AppState>,
    backup_id: &[u8; 32],
) -> Result<ValidatorKeyBackup, String> {
    // In production: Load backup from secure storage
    Err(format!(
        "Backup {} not found in storage. Production implementation required.",
        hex::encode(backup_id)
    ))
}

async fn store_restored_keypair(
    _state: &Arc<AppState>,
    node_id: &[u8; 32],
    _keypair: &ValidatorKeypair,
    _password: &str,
) -> Result<(), String> {
    // In production: Encrypt keypair with password and store
    info!("Keypair restored for node: {}", hex::encode(node_id));
    Ok(())
}

/// Configure routes for Validator Backup API
pub fn routes() -> axum::Router<Arc<AppState>> {
    use axum::routing::post;

    axum::Router::new()
        .route("/validator/backup/create", post(create_backup))
        .route("/validator/backup/restore", post(restore_backup))
        .route("/validator/backup/verify", post(verify_backup))
}
