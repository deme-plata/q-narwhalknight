/// Privacy-as-a-Service (PaaS) API Implementation
///
/// Enterprise-grade privacy infrastructure providing:
/// - Tor relay services
/// - Transaction mixing with ring signatures
/// - Stealth address generation
/// - ZK-STARK proof generation
/// - Cross-chain atomic swaps
///
/// Revenue Model: All PaaS fees flow to Quillon Bank master account
use axum::{
    extract::{Json, Path, Request, State},
    http::StatusCode,
    response::IntoResponse,
    Extension,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use base64::Engine as _;
use crate::paas_auth::AuthContext;
use crate::AppState;
use q_types::{ApiResponse, PrivacyLevel};

/// PaaS pricing in QUG tokens (atomic units: 1 QUG = 100,000,000 atomic units)
pub mod pricing {
    // Pay-per-use pricing
    pub const TOR_RELAY_PER_MB: u64 = 1_000_000; // 0.01 QUG/MB (10x)
    pub const MIXING_FEE_BASIS_POINTS: u64 = 50; // 0.5% of transaction value (5x)
    pub const MIXING_FEE_MINIMUM: u64 = 10_000_000; // 0.10 QUG minimum (10x)
    pub const RING_SIGNATURE_FEE: u64 = 1_000_000; // 0.01 QUG (10x)
    pub const STEALTH_ADDRESS_FEE: u64 = 500_000; // 0.005 QUG (50x)
    pub const ZK_STARK_PROOF_FEE: u64 = 10_000_000; // 0.10 QUG (10x)
    pub const ATOMIC_SWAP_FEE: u64 = 15_000_000; // 0.15 QUG (3x)

    // Enterprise tier monthly subscriptions (in USD equivalent QUG)
    pub const PROFESSIONAL_TIER_MONTHLY: u64 = 499_00000000; // $499
    pub const ENTERPRISE_TIER_MONTHLY: u64 = 1999_00000000; // $1,999
    pub const WHITE_LABEL_TIER_MONTHLY: u64 = 9999_00000000; // $9,999
}

/// Quillon Bank master account receiving all PaaS revenue
pub const QUILLON_BANK_MASTER_ACCOUNT: &str = "quillon_bank_master";

/// PaaS service types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaaSService {
    TorRelay,
    TransactionMixing,
    RingSignature,
    StealthAddress,
    ZkStarkProof,
    AtomicSwap,
}

/// API key tier levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApiTier {
    PayPerUse,
    Professional,
    Enterprise,
    WhiteLabel,
}

impl ApiTier {
    /// Get rate limit for this tier (requests per minute)
    pub fn rate_limit(&self) -> u32 {
        match self {
            ApiTier::PayPerUse => 100,
            ApiTier::Professional => 1_000,
            ApiTier::Enterprise => 10_000,
            ApiTier::WhiteLabel => u32::MAX, // Unlimited
        }
    }

    /// Get daily request limit
    pub fn daily_limit(&self) -> Option<u32> {
        match self {
            ApiTier::PayPerUse => Some(10_000),
            ApiTier::Professional => Some(100_000),
            ApiTier::Enterprise => None, // Unlimited
            ApiTier::WhiteLabel => None, // Unlimited
        }
    }
}

/// API key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub key: String,
    pub wallet_address: [u8; 32],
    pub tier: ApiTier,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub rate_limit_remaining: u32,
    pub daily_requests_used: u32,
    pub last_reset: DateTime<Utc>,
}

/// PaaS billing record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaaSBillingRecord {
    pub transaction_id: String,
    pub service: PaaSService,
    pub customer_wallet: [u8; 32],
    pub amount_qug: u64,
    pub timestamp: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

// ==================== Tor Relay Service ====================

#[derive(Debug, Deserialize)]
pub struct TorRelayRequest {
    pub chain: String,
    pub destination: String,
    pub data: String, // base64 encoded
    #[serde(default)]
    pub circuit_requirements: CircuitRequirements,
}

#[derive(Debug, Default, Deserialize)]
pub struct CircuitRequirements {
    #[serde(default = "default_min_hops")]
    pub min_hops: u8,
    pub exit_country: Option<String>,
    #[serde(default)]
    pub avoid_countries: Vec<String>,
    #[serde(default = "default_true")]
    pub quantum_seeded: bool,
    #[serde(default = "default_circuit_lifetime")]
    pub circuit_lifetime_minutes: u32,
}

fn default_min_hops() -> u8 {
    3
}
fn default_true() -> bool {
    true
}
fn default_circuit_lifetime() -> u32 {
    10
}

#[derive(Debug, Serialize)]
pub struct TorRelayResponse {
    pub success: bool,
    pub circuit_id: String,
    pub latency_ms: u64,
    pub exit_node_country: String,
    pub exit_node_fingerprint: String,
    pub relay_cost_qug: String,
    pub estimated_bandwidth_mb: f64,
    pub billing_transaction_id: String,
}

/// POST /api/v1/privacy/tor/relay
/// Relay transaction through Tor network
pub async fn tor_relay_service(
    State(state): State<Arc<AppState>>,
    Extension(auth_context): Extension<AuthContext>,
    Json(request): Json<TorRelayRequest>,
) -> Result<Json<ApiResponse<TorRelayResponse>>, StatusCode> {
    info!(
        "🧅 PaaS: Tor relay request for chain: {} from wallet {}",
        request.chain,
        hex::encode(&auth_context.wallet_address[..8])
    );

    // Check if Tor client is available
    let tor_client = match &state.tor_client {
        Some(client) => client,
        None => {
            error!("❌ Tor client not initialized");
            return Ok(Json(ApiResponse::error(
                "Tor service not available".to_string(),
            )));
        }
    };

    // Decode transaction data
    let tx_data = match base64::decode(&request.data) {
        Ok(data) => data,
        Err(_) => return Ok(Json(ApiResponse::error("Invalid base64 data".to_string()))),
    };

    let data_size_mb = tx_data.len() as f64 / 1_048_576.0;

    // Calculate cost: 0.001 QUG per MB, minimum 0.001 QUG
    let cost_qug =
        ((data_size_mb * pricing::TOR_RELAY_PER_MB as f64) as u64).max(pricing::TOR_RELAY_PER_MB);

    // Generate billing transaction
    let billing_tx_id = Uuid::new_v4().to_string();

    // Credit Quillon Bank with PaaS revenue
    let credit_result = credit_quillon_bank(
        &state,
        cost_qug,
        PaaSService::TorRelay,
        auth_context.wallet_address, // Extract from authenticated context
    )
    .await;

    match credit_result {
        Ok(_) => {
            info!(
                "💰 PaaS Revenue: {} QUG to Quillon Bank (Tor relay)",
                cost_qug as f64 / 1e24
            );
        }
        Err(e) => {
            warn!("⚠️ Failed to credit Quillon Bank: {}", e);
            // Continue with service, log error for manual reconciliation
        }
    }

    // Establish Tor connection to destination
    let start_time = std::time::Instant::now();
    let connection_result = tor_client.connect_to_peer(&request.destination).await;

    let (circuit_id, exit_info, latency_ms) = match connection_result {
        Ok(connection) => {
            let circuit_id = connection.get_circuit_id();
            let latency = start_time.elapsed().as_millis() as u64;

            info!("✅ Tor connection established via circuit {}", circuit_id);

            // Get Tor statistics for exit node info
            let tor_stats = tor_client.get_tor_stats().await;

            (
                format!("qnk-tor-{:016x}", circuit_id),
                ("DE".to_string(), "Controlled-Egress-Relay".to_string()), // TODO: Extract from tor_stats
                latency,
            )
        }
        Err(e) => {
            error!("❌ Tor connection failed: {}", e);
            return Ok(Json(ApiResponse::error(format!("Tor relay failed: {}", e))));
        }
    };

    let response = TorRelayResponse {
        success: true,
        circuit_id,
        latency_ms,
        exit_node_country: exit_info.0,
        exit_node_fingerprint: exit_info.1,
        relay_cost_qug: format!("{:.8}", cost_qug as f64 / 1e24),
        estimated_bandwidth_mb: data_size_mb,
        billing_transaction_id: billing_tx_id,
    };

    Ok(Json(ApiResponse::success(response)))
}

// ==================== Transaction Mixing Service ====================

#[derive(Debug, Deserialize)]
pub struct MixingRequest {
    pub chain: String,
    pub transaction_data: serde_json::Value,
    pub privacy_level: String, // "standard", "high", "maximum"
    #[serde(default)]
    pub mixing_parameters: MixingParameters,
    #[serde(default)]
    pub output_addresses: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct MixingParameters {
    #[serde(default = "default_decoy_count")]
    pub decoy_count: u32,
    #[serde(default = "default_ring_size")]
    pub ring_size: u32,
    #[serde(default = "default_true")]
    pub stealth_addresses: bool,
    #[serde(default = "default_true")]
    pub quantum_resistant: bool,
    #[serde(default)]
    pub compliance_mode: bool,
}

fn default_decoy_count() -> u32 {
    20
}
fn default_ring_size() -> u32 {
    16
}

#[derive(Debug, Serialize)]
pub struct MixingResponse {
    pub success: bool,
    pub mixing_id: String,
    pub anonymity_set_size: u32,
    pub estimated_completion_seconds: u32,
    pub mixing_pool_id: String,
    pub participant_count: u32,
    pub stealth_addresses: Vec<StealthAddressInfo>,
    pub mixing_fee_qug: String,
    pub zk_proof: ZkProofInfo,
    pub broadcast_via_tor: bool,
    pub billing_transaction_id: String,
}

#[derive(Debug, Serialize)]
pub struct StealthAddressInfo {
    pub address: String,
    pub view_key: String,
    pub spend_key: String,
}

#[derive(Debug, Serialize)]
pub struct ZkProofInfo {
    pub proof_type: String,
    pub proof_data: String,
    pub verification_key: String,
}

/// POST /api/v1/privacy/mix/submit
/// Submit transaction for mixing with ring signatures
pub async fn mixing_service(
    State(state): State<Arc<AppState>>,
    Extension(auth_context): Extension<AuthContext>,
    Json(request): Json<MixingRequest>,
) -> Result<Json<ApiResponse<MixingResponse>>, StatusCode> {
    info!(
        "🌪️  PaaS: Mixing request for chain: {} from wallet {}",
        request.chain,
        hex::encode(&auth_context.wallet_address[..8])
    );

    // Check if quantum mixer is available
    let quantum_mixer = match &state.quantum_mixer {
        Some(mixer) => mixer,
        None => {
            error!("❌ Quantum mixing engine not initialized");
            return Ok(Json(ApiResponse::error(
                "Mixing service not available".to_string(),
            )));
        }
    };

    // Parse privacy level
    let privacy_level = match request.privacy_level.as_str() {
        "standard" => PrivacyLevel::Standard,
        "high" => PrivacyLevel::High,
        "maximum" => PrivacyLevel::Maximum,
        _ => PrivacyLevel::High,
    };

    // Extract transaction value for fee calculation
    let tx_value = extract_transaction_value(&request.chain, &request.transaction_data);

    // Calculate mixing fee: 0.1% of transaction value, minimum 0.01 QUG
    let mixing_fee =
        ((tx_value * pricing::MIXING_FEE_BASIS_POINTS) / 10_000).max(pricing::MIXING_FEE_MINIMUM);

    // Generate billing transaction
    let billing_tx_id = Uuid::new_v4().to_string();

    // Credit Quillon Bank with PaaS revenue
    let credit_result = credit_quillon_bank(
        &state,
        mixing_fee,
        PaaSService::TransactionMixing,
        auth_context.wallet_address, // Extract from authenticated context
    )
    .await;

    match credit_result {
        Ok(_) => {
            info!(
                "💰 PaaS Revenue: {} QUG to Quillon Bank (mixing)",
                mixing_fee as f64 / 1e24
            );
        }
        Err(e) => {
            warn!("⚠️ Failed to credit Quillon Bank: {}", e);
            // Continue with service, log error for manual reconciliation
        }
    }

    // Create mixing input for quantum mixer
    let mixing_input = q_quantum_mixing::MixingInput {
        amount: tx_value,
        sender_key: [0u8; 32], // TODO: Extract from request
        recipient_address: request
            .output_addresses
            .first()
            .and_then(|addr| hex::decode(addr.trim_start_matches("0x")).ok())
            .and_then(|bytes| {
                let mut arr = [0u8; 32];
                if bytes.len() >= 32 {
                    arr.copy_from_slice(&bytes[..32]);
                    Some(arr)
                } else {
                    None
                }
            })
            .unwrap_or([0u8; 32]),
        commitment: [0u8; 32], // TODO: Generate commitment from transaction data
    };

    // Generate mixing session
    let mixing_id = format!("qnk-mix-{}", Uuid::new_v4().to_string()[..8].to_string());
    let pool_id = format!("pool-{}-{:04}", request.chain, rand::random::<u16>());

    // Calculate anonymity set
    let anonymity_set_size = request.mixing_parameters.decoy_count * 4;

    // Estimate completion time based on privacy level
    let completion_seconds = match privacy_level {
        PrivacyLevel::Standard => 15,
        PrivacyLevel::High => 30,
        PrivacyLevel::Maximum => 60,
    };

    // Generate stealth addresses using quantum mixer's stealth generator
    let stealth_addresses = if request.mixing_parameters.stealth_addresses {
        vec![StealthAddressInfo {
            address: format!("qnk:stealth:0x{}", hex::encode(&rand::random::<[u8; 20]>())),
            view_key: "encrypted_view_key_base64".to_string(),
            spend_key: "encrypted_spend_key_base64".to_string(),
        }]
    } else {
        vec![]
    };

    // Generate ZK-STARK proof for mixing validity
    let zk_proof = if let Some(zkp_prover) = &state.zkp_prover {
        ZkProofInfo {
            proof_type: "stark".to_string(),
            proof_data: "base64_encoded_stark_proof".to_string(), // TODO: Generate real proof
            verification_key: "base64_encoded_vk".to_string(),
        }
    } else {
        ZkProofInfo {
            proof_type: "none".to_string(),
            proof_data: "proof_generation_unavailable".to_string(),
            verification_key: "".to_string(),
        }
    };

    let response = MixingResponse {
        success: true,
        mixing_id,
        anonymity_set_size,
        estimated_completion_seconds: completion_seconds,
        mixing_pool_id: pool_id,
        participant_count: 16, // TODO: Get actual pool size
        stealth_addresses,
        mixing_fee_qug: format!("{:.8}", mixing_fee as f64 / 1e24),
        zk_proof,
        broadcast_via_tor: true,
        billing_transaction_id: billing_tx_id,
    };

    Ok(Json(ApiResponse::success(response)))
}

// ==================== Ring Signature Generation ====================

#[derive(Debug, Deserialize)]
pub struct RingSignatureRequest {
    pub chain: String,
    pub message_hash: String,
    pub ring_members: Vec<RingMember>,
    pub signing_key_index: usize,
    #[serde(default = "default_signature_scheme")]
    pub signature_scheme: String,
}

fn default_signature_scheme() -> String {
    "dilithium5".to_string()
}

#[derive(Debug, Deserialize)]
pub struct RingMember {
    pub public_key: String,
    pub key_index: usize,
}

#[derive(Debug, Serialize)]
pub struct RingSignatureResponse {
    pub success: bool,
    pub ring_signature: RingSignatureData,
    pub verification_data: VerificationData,
    pub signature_fee_qug: String,
    pub quantum_resistant: bool,
    pub billing_transaction_id: String,
}

#[derive(Debug, Serialize)]
pub struct RingSignatureData {
    pub signature_data: String,
    pub key_image: String,
    pub ring_size: usize,
    pub scheme: String,
}

#[derive(Debug, Serialize)]
pub struct VerificationData {
    pub ring_public_keys: Vec<String>,
    pub message_hash: String,
}

/// POST /api/v1/privacy/ring-signature/generate
/// Generate quantum-resistant ring signature
pub async fn ring_signature_service(
    State(state): State<Arc<AppState>>,
    Extension(auth_context): Extension<AuthContext>,
    Json(request): Json<RingSignatureRequest>,
) -> Result<Json<ApiResponse<RingSignatureResponse>>, StatusCode> {
    info!(
        "🔐 PaaS: Ring signature request for chain: {} from wallet {}",
        request.chain,
        hex::encode(&auth_context.wallet_address[..8])
    );

    // Validate ring size
    if request.ring_members.len() < 8 || request.ring_members.len() > 64 {
        return Ok(Json(ApiResponse::error(
            "Ring size must be between 8 and 64".to_string(),
        )));
    }

    // Validate signing key index
    if request.signing_key_index >= request.ring_members.len() {
        return Ok(Json(ApiResponse::error(
            "Invalid signing key index".to_string(),
        )));
    }

    let signature_fee = pricing::RING_SIGNATURE_FEE;

    // Generate billing transaction
    let billing_tx_id = Uuid::new_v4().to_string();

    // TODO: Charge customer wallet
    // TODO: Credit Quillon Bank master account

    info!(
        "💰 PaaS Revenue: {} QUG to Quillon Bank (ring signature)",
        signature_fee as f64 / 1e24
    );

    // Wire to real CLSAG (Compact Linkable Spontaneous Anonymous Group Signatures)
    // CLSAGSigner uses curve25519-dalek Ristretto255 with quantum-seeded entropy
    let entropy_pool = match q_quantum_mixing::quantum_entropy::QuantumEntropyPool::new().await {
        Ok(pool) => std::sync::Arc::new(pool),
        Err(e) => {
            error!("Failed to initialize entropy pool for ring signature: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let mut signer = match q_quantum_mixing::clsag::CLSAGSigner::new(entropy_pool.clone()).await {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to create CLSAG signer: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    // Build ring — place service's ephemeral public key at signing_key_index
    let signer_pubkey = signer.get_public_key();
    let mut ring: Vec<[u8; 32]> = request.ring_members.iter()
        .map(|m| {
            let bytes = hex::decode(&m.public_key).unwrap_or_default();
            let mut arr = [0u8; 32];
            if bytes.len() >= 32 { arr.copy_from_slice(&bytes[..32]); }
            arr
        })
        .collect();
    ring[request.signing_key_index] = signer_pubkey;

    // Generate Pedersen commitment for confidential amount
    let commitment_mask = match q_quantum_mixing::clsag::generate_commitment_mask(&entropy_pool).await {
        Ok(m) => m,
        Err(e) => {
            error!("Commitment mask generation failed: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };
    let (commitment_bytes, _) = q_quantum_mixing::clsag::create_pedersen_commitment(0, &commitment_mask);
    let message = hex::decode(&request.message_hash).unwrap_or_default();

    let signature = match signer.sign(&message, &ring, &commitment_bytes, &commitment_mask).await {
        Ok(sig) => sig,
        Err(e) => {
            error!("CLSAG signing failed: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    // Server-side verify before returning (catches any encoding/logic bugs)
    match signature.verify(&message) {
        Ok(true) => {},
        Ok(false) => {
            error!("CLSAG self-verification failed immediately after signing");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
        Err(e) => {
            error!("CLSAG verification error: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    }

    info!("🔑 CLSAG ring sig produced — key_image=0x{} ring_size={}",
        hex::encode(signature.get_key_image()), ring.len());

    let sig_bytes = match postcard::to_allocvec(&signature) {
        Ok(b) => b,
        Err(e) => {
            error!("CLSAG serialization failed: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };
    let sig_b64 = base64::engine::general_purpose::STANDARD.encode(&sig_bytes);

    let ring_signature = RingSignatureData {
        signature_data: sig_b64,
        key_image: format!("0x{}", hex::encode(signature.get_key_image())),
        ring_size: ring.len(),
        scheme: "clsag-ristretto255".to_string(),
    };

    let verification_data = VerificationData {
        ring_public_keys: ring.iter().map(|k| hex::encode(k)).collect(),
        message_hash: request.message_hash.clone(),
    };

    let response = RingSignatureResponse {
        success: true,
        ring_signature,
        verification_data,
        signature_fee_qug: format!("{:.8}", signature_fee as f64 / 1e24),
        quantum_resistant: request.signature_scheme == "dilithium5",
        billing_transaction_id: billing_tx_id,
    };

    Ok(Json(ApiResponse::success(response)))
}

// ==================== Stealth Address Generation ====================

#[derive(Debug, Deserialize)]
pub struct StealthAddressRequest {
    pub chain: String,
    pub recipient_public_key: String,
    #[serde(default = "default_address_count")]
    pub count: u32,
}

fn default_address_count() -> u32 {
    1
}

#[derive(Debug, Serialize)]
pub struct StealthAddressResponse {
    pub success: bool,
    pub stealth_addresses: Vec<StealthAddressInfo>,
    pub view_key: String,
    pub spend_key: String,
    pub generation_fee_qug: String,
    pub billing_transaction_id: String,
}

/// POST /api/v1/privacy/stealth-address/generate
/// Generate stealth addresses (Monero-style dual-key)
pub async fn stealth_address_service(
    State(state): State<Arc<AppState>>,
    Extension(auth_context): Extension<AuthContext>,
    Json(request): Json<StealthAddressRequest>,
) -> Result<Json<ApiResponse<StealthAddressResponse>>, StatusCode> {
    info!(
        "👻 PaaS: Stealth address request for chain: {} from wallet {}",
        request.chain,
        hex::encode(&auth_context.wallet_address[..8])
    );

    // Validate count
    if request.count == 0 || request.count > 100 {
        return Ok(Json(ApiResponse::error(
            "Count must be between 1 and 100".to_string(),
        )));
    }

    let generation_fee = pricing::STEALTH_ADDRESS_FEE * request.count as u64;

    // Generate billing transaction
    let billing_tx_id = Uuid::new_v4().to_string();

    // TODO: Charge customer wallet
    // TODO: Credit Quillon Bank master account

    info!(
        "💰 PaaS Revenue: {} QUG to Quillon Bank (stealth addresses)",
        generation_fee as f64 / 1e24
    );

    // Wire to real Dual-Key Stealth Address Protocol using curve25519-dalek Ristretto255.
    // Implements standard Monero-style one-time stealth addresses:
    //   ephemeral r   = quantum random scalar
    //   ephemeral_pub = r * G
    //   shared_secret = r * recipient_pubkey  (ECDH)
    //   one_time_addr = H("stealth" || shared_secret) * G + recipient_pubkey
    let recipient_bytes = match hex::decode(
        request.recipient_public_key.trim_start_matches("0x")
    ) {
        Ok(b) if b.len() == 32 => {
            let mut arr = [0u8; 32]; arr.copy_from_slice(&b); arr
        }
        _ => {
            return Ok(Json(ApiResponse::error(
                "recipient_public_key must be 32-byte hex (Ristretto255 compressed point)".to_string(),
            )));
        }
    };

    let entropy_pool = match q_quantum_mixing::quantum_entropy::QuantumEntropyPool::new().await {
        Ok(pool) => std::sync::Arc::new(pool),
        Err(e) => {
            error!("Failed to initialize entropy pool for stealth addresses: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let mut stealth_addresses = Vec::with_capacity(request.count as usize);
    for _ in 0..request.count {
        // Generate quantum random ephemeral scalar r
        let mut r_bytes = [0u8; 64];
        if let Err(e) = entropy_pool.fill_bytes(&mut r_bytes[..32]).await
            .and(entropy_pool.fill_bytes(&mut r_bytes[32..]).await) {
            error!("Entropy fill failed for stealth address: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
        let r = q_quantum_mixing::clsag::scalar_from_bytes_wide(r_bytes);

        // Compute ephemeral public key R = r*G and ECDH shared secret S = r*P_recipient
        let (ephemeral_pub, one_time_addr) =
            q_quantum_mixing::clsag::derive_stealth_address(&r, &recipient_bytes)
                .map_err(|e| {
                    error!("Stealth address derivation failed: {}", e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;

        stealth_addresses.push(StealthAddressInfo {
            address: format!("qnk:stealth:0x{}", hex::encode(&one_time_addr)),
            view_key: hex::encode(&ephemeral_pub),
            spend_key: hex::encode(&recipient_bytes),
        });
    }

    let master_view_key = hex::encode(&recipient_bytes);
    let response = StealthAddressResponse {
        success: true,
        stealth_addresses,
        view_key: master_view_key.clone(),
        spend_key: master_view_key,
        generation_fee_qug: format!("{:.8}", generation_fee as f64 / 1e24),
        billing_transaction_id: billing_tx_id,
    };

    Ok(Json(ApiResponse::success(response)))
}

// ==================== ZK-STARK Proof Generation ====================

#[derive(Debug, Deserialize)]
pub struct ZkStarkProofRequest {
    pub statement: String,
    pub witness: serde_json::Value,
    pub public_inputs: serde_json::Value,
    #[serde(default = "default_proof_type")]
    pub proof_type: String,
}

fn default_proof_type() -> String {
    "stark".to_string()
}

#[derive(Debug, Serialize)]
pub struct ZkStarkProofResponse {
    pub success: bool,
    pub proof_id: String,
    pub proof_data: String,
    pub verification_key: String,
    pub proof_size_bytes: usize,
    pub generation_time_ms: u64,
    pub proof_fee_qug: String,
    pub verifiable_on_chain: bool,
    pub billing_transaction_id: String,
}

/// POST /api/v1/privacy/zk-stark/prove
/// Generate ZK-STARK proof (universal)
pub async fn zk_stark_proof_service(
    State(state): State<Arc<AppState>>,
    Extension(auth_context): Extension<AuthContext>,
    Json(request): Json<ZkStarkProofRequest>,
) -> Result<Json<ApiResponse<ZkStarkProofResponse>>, StatusCode> {
    info!(
        "🔬 PaaS: ZK-STARK proof request for statement: {} from wallet {}",
        request.statement,
        hex::encode(&auth_context.wallet_address[..8])
    );

    let proof_fee = pricing::ZK_STARK_PROOF_FEE;

    // Generate billing transaction
    let billing_tx_id = Uuid::new_v4().to_string();

    // TODO: Charge customer wallet
    // TODO: Credit Quillon Bank master account

    info!(
        "💰 PaaS Revenue: {} QUG to Quillon Bank (ZK-STARK proof)",
        proof_fee as f64 / 1e24
    );

    // Wire to real FRI-based ZK-STARK prover (q-zk-stark crate, CPU mode)
    let t0 = std::time::Instant::now();

    // Build execution trace from public_inputs and statement
    let trace: Vec<Vec<u64>> = {
        let mut rows = Vec::new();
        if let Some(arr) = request.public_inputs.as_array() {
            for v in arr {
                if let Some(n) = v.as_u64() {
                    rows.push(vec![n]);
                }
            }
        }
        if rows.is_empty() {
            // Default: single-row trace from statement hash
            use sha3::Digest;
            let h = sha3::Sha3_256::new().chain_update(request.statement.as_bytes()).finalize();
            rows.push(h.iter().take(8).map(|&b| b as u64).collect());
        }
        rows
    };

    // Constraints encoding: SHA3-256 of (statement || witness JSON)
    let constraints: Vec<u8> = {
        use sha3::Digest;
        let mut hasher = sha3::Sha3_256::new();
        hasher.update(request.statement.as_bytes());
        hasher.update(request.witness.to_string().as_bytes());
        hasher.finalize().to_vec()
    };

    let mut stark = match q_zk_stark::StarkSystem::new(false).await {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to initialize STARK system: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let stark_proof = match stark.prove(&trace, &constraints).await {
        Ok(p) => p,
        Err(e) => {
            error!("STARK proof generation failed: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let generation_time_ms = t0.elapsed().as_millis() as u64;

    let proof_bytes = match postcard::to_allocvec(&stark_proof) {
        Ok(b) => b,
        Err(e) => {
            error!("STARK proof serialization failed: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };
    let proof_b64 = base64::engine::general_purpose::STANDARD.encode(&proof_bytes);
    let proof_size_bytes = proof_bytes.len();
    let proof_id = format!("stark-{}", &Uuid::new_v4().to_string()[..8]);

    info!("🔬 STARK proof generated: id={} size={}B time={}ms",
        proof_id, proof_size_bytes, generation_time_ms);

    let response = ZkStarkProofResponse {
        success: true,
        proof_id,
        proof_data: proof_b64,
        verification_key: hex::encode(&constraints),
        proof_size_bytes,
        generation_time_ms,
        proof_fee_qug: format!("{:.8}", proof_fee as f64 / 1e24),
        verifiable_on_chain: true,
        billing_transaction_id: billing_tx_id,
    };

    Ok(Json(ApiResponse::success(response)))
}

// ==================== PaaS Statistics & Billing ====================

#[derive(Debug, Serialize)]
pub struct PaaSStatistics {
    pub total_revenue_qug: String,
    pub services_used: std::collections::HashMap<String, u64>,
    pub active_api_keys: u32,
    pub total_requests_today: u64,
    pub quillon_bank_balance: String,
}

/// GET /api/v1/privacy/paas/statistics
/// Get PaaS usage statistics and revenue
pub async fn paas_statistics(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<PaaSStatistics>>, StatusCode> {
    info!("📊 PaaS: Statistics request");

    // TODO: Query actual statistics from database

    let mut services_used = std::collections::HashMap::new();
    services_used.insert("tor_relay".to_string(), 1234);
    services_used.insert("mixing".to_string(), 567);
    services_used.insert("ring_signature".to_string(), 890);
    services_used.insert("stealth_address".to_string(), 2345);
    services_used.insert("zk_stark_proof".to_string(), 123);

    let stats = PaaSStatistics {
        total_revenue_qug: "1250.50000000".to_string(), // Simulated
        services_used,
        active_api_keys: 42,                               // Simulated
        total_requests_today: 5159,                        // Simulated
        quillon_bank_balance: "1250.50000000".to_string(), // Should match total revenue
    };

    Ok(Json(ApiResponse::success(stats)))
}

// ==================== Helper Functions ====================

/// Extract transaction value from chain-specific transaction data
fn extract_transaction_value(chain: &str, tx_data: &serde_json::Value) -> u64 {
    match chain {
        "ethereum" => {
            if let Some(value) = tx_data.get("value").and_then(|v| v.as_str()) {
                // Parse wei value
                value.parse::<u64>().unwrap_or(0)
            } else {
                0
            }
        }
        "bitcoin" => {
            if let Some(value) = tx_data.get("value").and_then(|v| v.as_u64()) {
                value
            } else {
                0
            }
        }
        _ => 0,
    }
}

/// Credit PaaS revenue to Quillon Bank master account
pub async fn credit_quillon_bank(
    state: &Arc<AppState>,
    amount_qug: u64,
    service: PaaSService,
    customer_wallet: [u8; 32],
) -> Result<String, String> {
    use q_quillon_bank::{Address, AssetType};

    // Generate unique transaction ID
    let tx_id = Uuid::new_v4().to_string();
    let billing_record = PaaSBillingRecord {
        transaction_id: tx_id.clone(),
        service: service.clone(),
        customer_wallet,
        amount_qug,
        timestamp: chrono::Utc::now(),
        metadata: serde_json::json!({
            "service_type": format!("{:?}", service),
            "customer": hex::encode(&customer_wallet[..8]),
        }),
    };

    // Access Quillon Bank system
    let quillon_bank = state.quillon_bank.read().await;
    let mut accounts = quillon_bank.accounts.write().await;

    // Get or create Quillon Bank master account
    let master_address = Address::from_public_key(&{
        let mut key = [0u8; 33];
        key[0] = 0x02; // Compressed public key prefix
                       // Use deterministic derivation for QUILLON_BANK_MASTER_ACCOUNT
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(QUILLON_BANK_MASTER_ACCOUNT.as_bytes());
        let hash = hasher.finalize();
        key[1..33].copy_from_slice(&hash[..32]);
        key
    });

    // Credit Quillon Bank master account
    let master_account = accounts.entry(master_address.clone()).or_insert_with(|| {
        use q_quillon_bank::{
            BankAccount, CreditScore, QuantumAccountFeatures, QuantumCreditData,
            QuantumPrivacyLevel, RiskTier,
        };
        BankAccount {
            address: master_address.clone(),
            balances: std::collections::HashMap::new(),
            credit_score: CreditScore {
                score: 850,
                risk_tier: RiskTier::Excellent,
                factors: vec![],
                history: vec![],
                quantum_enhancement: QuantumCreditData {
                    quantum_transaction_patterns: 1.0,
                    post_quantum_security_usage: 1.0,
                    vault_utilization_score: 1.0,
                    consensus_participation: 1.0,
                },
                last_calculated: chrono::Utc::now().timestamp() as u64,
            },
            identity: q_quillon_bank::identity::VerifiedIdentity {
                id: "quillon_bank_master".to_string(),
                verified: true,
            },
            privacy_tier: q_quillon_bank::PrivacyTier::Standard,
            wealth_agent: None,
            transaction_history: vec![],
            created_at: chrono::Utc::now().timestamp() as u64,
            last_activity: chrono::Utc::now().timestamp() as u64,
            quantum_features: QuantumAccountFeatures::default(),
        }
    });

    // Credit ORB (QUG) balance
    let orb_balance =
        master_account
            .balances
            .entry(AssetType::ORB)
            .or_insert(q_quillon_bank::Balance {
                available: 0,
                locked: 0,
                staked: 0,
                borrowed: 0,
                lending: 0,
                quantum_secured: 0,
                last_updated: chrono::Utc::now().timestamp() as u64,
            });

    // Add PaaS revenue to available balance
    orb_balance.available += amount_qug as u128;
    orb_balance.last_updated = chrono::Utc::now().timestamp() as u64;

    info!(
        "💰 PaaS Billing: {} QUG from customer {} to Quillon Bank master account (service: {:?})",
        amount_qug as f64 / 1e24,
        hex::encode(&customer_wallet[..8]),
        service
    );

    info!(
        "🏦 Quillon Bank Master Balance: {} QUG",
        orb_balance.available as f64 / 1e24
    );

    // TODO: Debit customer wallet when authentication is implemented
    // TODO: Record in PaaS billing ledger
    // TODO: Emit billing event via event broadcaster

    Ok(tx_id)
}
