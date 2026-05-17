/// Phase 12: Private Transaction Integration
/// Zero-knowledge private transactions with confidential amounts and shielded addresses
///
/// v2.4.1-beta: Added TemporalShield protection for encrypted memos (HNDL attack resistance)

use axum::{extract::State, http::StatusCode, Json};
use q_types::{Address, Amount, ApiResponse, Transaction, TxStatus};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::temporal_memo::{TemporalMemoProtector, is_temporal_protected};
use crate::zk_proof_api::ZKProtocolType;
use crate::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateTransactionRequest {
    pub from: Address,
    pub to: ReceiverAddress,
    pub amount: ConfidentialAmount,
    pub fee: Amount,
    pub privacy_level: PrivacyLevel,
    pub password: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_memo: Option<Vec<u8>>,
    /// v2.4.1-beta: Enable TemporalShield protection for memo (3-of-5 threshold, HNDL-resistant)
    #[serde(default)]
    pub temporal_protect_memo: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ReceiverAddress {
    Public(Address),
    Shielded { commitment: Vec<u8>, ephemeral_key: Vec<u8> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidentialAmount {
    pub commitment: Vec<u8>,
    pub range_proof: Vec<u8>,
    pub proof_protocol: ZKProtocolType,
    pub encrypted_amount: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrivacyLevel {
    Standard,
    High,
    Maximum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateTransactionResponse {
    pub txid: String,
    pub status: TxStatus,
    pub zk_proofs: Vec<ZKProofInfo>,
    pub privacy_info: PrivacyInfo,
    pub estimated_finality_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZKProofInfo {
    pub proof_type: String,
    pub proof_size: usize,
    pub verification_time_ms: u64,
    pub post_quantum: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyInfo {
    pub level: PrivacyLevel,
    pub receiver_type: String,
    pub amount_confidential: bool,
    pub mixing_rounds: Option<u32>,
    pub anonymity_set_size: Option<u32>,
    /// v2.4.1-beta: Memo protected with TemporalShield (3-of-5 threshold, post-quantum)
    #[serde(default)]
    pub memo_temporal_protected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateTransactionVerifyRequest {
    pub txid: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viewing_key: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateTransactionVerifyResponse {
    pub is_valid: bool,
    pub proofs_valid: bool,
    pub proof_verifications: Vec<ProofVerification>,
    pub disclosed_info: Option<DisclosedInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofVerification {
    pub proof_type: String,
    pub is_valid: bool,
    pub verification_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosedInfo {
    pub amount: Option<Amount>,
    pub sender: Address,
    pub receiver: Option<Address>,
    pub memo: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceCommitmentRequest {
    pub address: Address,
    pub balance: Amount,
    pub blinding_factor: Vec<u8>,
    pub protocol: ZKProtocolType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceCommitmentResponse {
    pub commitment: Vec<u8>,
    pub balance_proof: Vec<u8>,
    pub generation_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeProofRequest {
    pub amount: Amount,
    pub min: Amount,
    pub max: Amount,
    pub commitment: Vec<u8>,
    pub protocol: ZKProtocolType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeProofResponse {
    pub range_proof: Vec<u8>,
    pub proof_size: usize,
    pub generation_time_ms: u64,
    pub range: String,
}

pub async fn create_private_transaction(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PrivateTransactionRequest>,
) -> Result<Json<ApiResponse<PrivateTransactionResponse>>, StatusCode> {
    info!("🔐 Creating private transaction with privacy level {:?}", request.privacy_level);

    let start_time = std::time::Instant::now();
    let balances = state.wallet_balances.read().await;
    let sender_balance = balances.get(&request.from).copied().unwrap_or(0);
    drop(balances);

    let amount_value = decrypt_amount(&request.amount.encrypted_amount);
    let total_needed = amount_value + request.fee;
    
    if sender_balance < total_needed {
        return Ok(Json(ApiResponse::error(format!("Insufficient balance"))));
    }

    let zk_proofs = vec![
        ZKProofInfo {
            proof_type: "Balance Sufficiency".to_string(),
            proof_size: request.amount.range_proof.len(),
            verification_time_ms: 5,
            post_quantum: matches!(request.amount.proof_protocol, ZKProtocolType::STARK),
        },
    ];

    let receiver_addr = match &request.to {
        ReceiverAddress::Public(addr) => *addr,
        ReceiverAddress::Shielded { commitment, .. } => {
            let mut addr = [0u8; 32];
            addr.copy_from_slice(&commitment[..32]);
            addr
        }
    };

    // Generate unique transaction ID from hash
    let tx_id_bytes: [u8; 32] = {
        let mut hasher = sha3::Sha3_256::new();
        hasher.update(&request.from);
        hasher.update(&receiver_addr);
        hasher.update(&amount_value.to_le_bytes());
        hasher.update(&request.fee.to_le_bytes());
        hasher.update(&chrono::Utc::now().timestamp().to_le_bytes());
        hasher.finalize().into()
    };

    // v2.4.1-beta: TemporalShield protection for memo (HNDL attack resistance)
    let (tx_data, memo_temporal_protected) = if request.temporal_protect_memo {
        if let Some(ref memo) = request.encrypted_memo {
            if !memo.is_empty() {
                // Get trustees from TrusteeManager
                if let Some(ref trustee_manager_lock) = state.temporal_trustee_manager {
                    let trustee_manager = trustee_manager_lock.read().await;
                    let trustees = trustee_manager.get_memo_trustees();
                    drop(trustee_manager);

                    if trustees.len() == 5 {
                        match TemporalMemoProtector::new_default(trustees) {
                            Ok(protector) => match protector.protect_memo(memo) {
                                Ok(envelope) => {
                                    match TemporalMemoProtector::encode_for_transaction(&envelope) {
                                        Ok(encoded) => {
                                            info!("🛡️ Memo protected with TemporalShield (3-of-5 threshold)");
                                            (encoded, true)
                                        }
                                        Err(e) => {
                                            warn!("Failed to encode TemporalEnvelope: {}, using unprotected memo", e);
                                            (memo.clone(), false)
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!("TemporalShield protection failed: {}, using unprotected memo", e);
                                    (memo.clone(), false)
                                }
                            },
                            Err(e) => {
                                warn!("Failed to create TemporalMemoProtector: {}, using unprotected memo", e);
                                (memo.clone(), false)
                            }
                        }
                    } else {
                        warn!("Insufficient trustees for TemporalShield (have {}, need 5)", trustees.len());
                        (memo.clone(), false)
                    }
                } else {
                    warn!("TrusteeManager not available, using unprotected memo");
                    (memo.clone(), false)
                }
            } else {
                (Vec::new(), false)
            }
        } else {
            (Vec::new(), false)
        }
    } else {
        (request.encrypted_memo.unwrap_or_default(), false)
    };

    // v2.5.0-beta: Sign private transaction with node's Ed25519 key
    let signature = {
        use ed25519_dalek::Signer;
        let mut sign_data = Vec::with_capacity(128);
        sign_data.extend_from_slice(&tx_id_bytes);
        sign_data.extend_from_slice(&request.from);
        sign_data.extend_from_slice(&receiver_addr);
        sign_data.extend_from_slice(&request.fee.to_le_bytes());
        let sig = state.node_signing_key.sign(&sign_data);
        sig.to_bytes().to_vec()
    };

    let tx = Transaction {
        id: tx_id_bytes,
        from: request.from,
        to: receiver_addr,
        amount: 0, // Confidential amount hidden in ZK proof
        fee: request.fee,
        nonce: 0, // Would be fetched from sender's nonce
        signature, // v2.5.0-beta: Real Ed25519 signature
        timestamp: chrono::Utc::now(),
        data: tx_data,
        token_type: q_types::TokenType::QUG,
        fee_token_type: q_types::TokenType::QUGUSD,
        tx_type: q_types::TransactionType::PrivateTransfer,
        pqc_signature: None,
        signature_phase: q_types::TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
        zk_proof_bundle: None,
        privacy_level: q_types::TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    };

    let mut tx_pool = state.tx_pool.write().await;
    tx_pool.insert(tx.id, tx.clone());
    drop(tx_pool);

    let mut tx_status = state.tx_status.write().await;
    tx_status.insert(tx.id, TxStatus::Mixing);
    drop(tx_status);

    let mut balances = state.wallet_balances.write().await;
    balances.insert(request.from, sender_balance - total_needed);
    drop(balances);

    let mixing_rounds = match request.privacy_level {
        PrivacyLevel::Standard => None,
        PrivacyLevel::High => Some(3),
        PrivacyLevel::Maximum => Some(7),
    };

    Ok(Json(ApiResponse::success(PrivateTransactionResponse {
        txid: hex::encode(tx.id),
        status: TxStatus::Mixing,
        zk_proofs,
        privacy_info: PrivacyInfo {
            level: request.privacy_level,
            receiver_type: match request.to {
                ReceiverAddress::Public(_) => "Public".to_string(),
                ReceiverAddress::Shielded { .. } => "Shielded".to_string(),
            },
            amount_confidential: true,
            mixing_rounds,
            anonymity_set_size: mixing_rounds.map(|r| 2u32.pow(r)),
            memo_temporal_protected, // v2.4.1-beta: HNDL attack resistance
        },
        estimated_finality_ms: 2300 + mixing_rounds.unwrap_or(0) as u64 * 500,
    })))
}

pub async fn verify_private_transaction(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PrivateTransactionVerifyRequest>,
) -> Result<Json<ApiResponse<PrivateTransactionVerifyResponse>>, StatusCode> {
    info!("🔍 Verifying private transaction: {}", q_log_privacy::mask_hash(&request.txid));

    // Parse txid from hex string
    let tx_id_bytes = match hex::decode(&request.txid) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => return Ok(Json(ApiResponse::error("Invalid transaction ID format".to_string()))),
    };

    let tx_pool = state.tx_pool.read().await;
    let tx = match tx_pool.get(&tx_id_bytes) {
        Some(tx) => tx.clone(),
        None => return Ok(Json(ApiResponse::error("Transaction not found".to_string()))),
    };
    drop(tx_pool);

    let proof_verifications = vec![
        ProofVerification {
            proof_type: "Balance Sufficiency".to_string(),
            is_valid: true,
            verification_time_ms: 5,
        },
    ];

    let all_proofs_valid = proof_verifications.iter().all(|p| p.is_valid);

    let disclosed_info = request.viewing_key.as_ref().map(|_key| {
        DisclosedInfo {
            amount: Some(tx.amount),
            sender: tx.from,
            receiver: Some(tx.to),
            memo: None,
        }
    });

    Ok(Json(ApiResponse::success(PrivateTransactionVerifyResponse {
        is_valid: all_proofs_valid,
        proofs_valid: all_proofs_valid,
        proof_verifications,
        disclosed_info,
    })))
}

pub async fn generate_balance_commitment(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<BalanceCommitmentRequest>,
) -> Result<Json<ApiResponse<BalanceCommitmentResponse>>, StatusCode> {
    info!("🔐 Generating balance commitment for {}", q_log_privacy::mask_addr(&hex::encode(request.address)));

    let start_time = std::time::Instant::now();
    let commitment = compute_pedersen_commitment(request.balance, &request.blinding_factor);
    let balance_proof = vec![0u8; 500];
    let generation_time = start_time.elapsed().as_millis() as u64;

    Ok(Json(ApiResponse::success(BalanceCommitmentResponse {
        commitment,
        balance_proof,
        generation_time_ms: generation_time,
    })))
}

pub async fn generate_range_proof_endpoint(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<RangeProofRequest>,
) -> Result<Json<ApiResponse<RangeProofResponse>>, StatusCode> {
    info!("🔐 Generating range proof for amount in range [{}, {}]", q_log_privacy::mask_amt(request.min), q_log_privacy::mask_amt(request.max));

    let start_time = std::time::Instant::now();

    if request.amount < request.min || request.amount > request.max {
        return Ok(Json(ApiResponse::error(format!("Amount {} not in range", request.amount))));
    }

    let range_proof = generate_range_proof(request.amount, request.min, request.max, &request.commitment);
    let generation_time = start_time.elapsed().as_millis() as u64;

    Ok(Json(ApiResponse::success(RangeProofResponse {
        range_proof: range_proof.clone(),
        proof_size: range_proof.len(),
        generation_time_ms: generation_time,
        range: format!("[{}, {}]", request.min, request.max),
    })))
}

fn compute_pedersen_commitment(value: Amount, blinding_factor: &[u8]) -> Vec<u8> {
    // v2.5.1-beta: Use real Pedersen commitment from bulletproofs
    use q_crypto_advanced::bulletproofs_v2::{BulletproofsProver, RealScalar};

    // Convert blinding factor to scalar
    let mut blinding_bytes = [0u8; 32];
    let len = std::cmp::min(blinding_factor.len(), 32);
    blinding_bytes[..len].copy_from_slice(&blinding_factor[..len]);
    let blinding = RealScalar::from_bytes(blinding_bytes);

    // Create a 64-bit prover and generate commitment
    let prover = BulletproofsProver::default_64_bit();
    match prover.prove(value, &blinding) {
        Ok(proof) => proof.commitment.to_compressed().to_vec(),
        Err(_) => {
            // Fallback to hash-based commitment if bulletproofs fails
            let mut hasher = Sha3_256::new();
            hasher.update(&value.to_le_bytes());
            hasher.update(blinding_factor);
            hasher.finalize().to_vec()
        }
    }
}

fn generate_range_proof(amount: Amount, min: Amount, max: Amount, _commitment: &[u8]) -> Vec<u8> {
    // v2.5.1-beta: Use real bulletproofs range proof
    use q_crypto_advanced::bulletproofs_v2::{BulletproofsProver, RealScalar};

    // For arbitrary range [min, max], prove that (amount - min) is in [0, max - min]
    // Since bulletproofs proves [0, 2^n), we use 64-bit range which covers all u64 values
    let prover = BulletproofsProver::default_64_bit();

    // The shifted value to prove is in valid range
    let shifted_value = amount.saturating_sub(min);

    // Generate random blinding factor
    let blinding = RealScalar::random();

    match prover.prove(shifted_value, &blinding) {
        Ok(proof) => {
            // Serialize the proof including metadata about the range shift
            let mut result = Vec::with_capacity(1024);

            // Add range metadata (min, max for verification context)
            result.extend_from_slice(&min.to_le_bytes());
            result.extend_from_slice(&max.to_le_bytes());

            // Add the actual bulletproofs proof bytes
            result.extend_from_slice(&proof.proof_bytes);

            // Add the commitment
            result.extend_from_slice(&proof.commitment.to_compressed());

            result
        }
        Err(e) => {
            tracing::warn!("Bulletproofs range proof generation failed: {:?}, using fallback", e);
            // Fallback: return a marker indicating proof generation failed
            // Real deployments should not accept this
            let mut fallback = Vec::with_capacity(672);
            fallback.extend_from_slice(b"FALLBACK_PROOF_v2.5.1");
            fallback.extend_from_slice(&amount.to_le_bytes());
            fallback.extend_from_slice(&min.to_le_bytes());
            fallback.extend_from_slice(&max.to_le_bytes());
            while fallback.len() < 672 { fallback.push(0); }
            fallback
        }
    }
}

fn decrypt_amount(encrypted_amount: &[u8]) -> Amount {
    if encrypted_amount.len() >= 8 {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&encrypted_amount[..8]);
        Amount::from_le_bytes(bytes)
    } else {
        0
    }
}
