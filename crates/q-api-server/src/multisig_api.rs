//! Multisig HTTP API — wires the previously-dormant `q-multisig` crate into
//! the live server so the "Organization & Multi-Sig" dashboard modal can
//! actually deploy a working CEO + associates wallet instead of stopping at
//! a disabled "Deploy on-chain" button.
//!
//! v10.11.103 (2026-08-18): Viktor: "finish the multi sig feature ... i
//! never actually make a working ceo multi sig wallet for associates."
//!
//! Flow:
//!   1. Every associate (via their OWN wallet session) calls `POST
//!      /register-key` once — a Hybrid-scheme signed call. The server
//!      already verifies Ed25519+Dilithium5 correspondence to the caller's
//!      address before this handler runs (`AuthenticatedWallet` extractor);
//!      we just persist the Dilithium5 public key bytes carried in the
//!      X-Wallet-Auth header so a CEO can later reference this person by
//!      their ordinary qnk address.
//!   2. The CEO calls `POST /create` with the org name, threshold, and each
//!      member's declared role/limits. Any member who hasn't registered yet
//!      is reported back by name so the CEO knows who to nudge.
//!   3. Any member calls `POST /propose` to ask the treasury to move funds;
//!      other members call `POST /sign` with their own Ed25519+Dilithium5
//!      signature over the proposal's payload hash; once the threshold of
//!      independently-verified signatures is met, anyone calls `POST
//!      /execute` to actually move the QUG.
//!
//! HONEST SCOPE for v0 (matches the frontend's own "draft, not enforced"
//! banner): per-member spending limits are stored and readable (so a
//! member can see "you can spend up to X/day") but NOT enforced at the
//! transaction-validation layer — a member who holds their own key can
//! still sign an ordinary transfer that ignores these numbers. Only
//! `MultisigAction::Transfer` is executable; `MintToken`/`SetDefaultThreshold`/
//! `RotateMember` proposals can be created and signed (the crypto plumbing
//! doesn't care), but `/execute` rejects them until their effects are
//! implemented here.

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use ed25519_dalek::{Signature as Ed25519Signature, VerifyingKey};
use pqcrypto_traits::sign::{DetachedSignature as _, PublicKey as _};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use q_multisig::wallet::{Address as MsAddress, HybridPublicKey, Member};
use q_multisig::{MultisigAction, MultisigProposal, MultisigWallet, OrgPolicy, OrgPolicyMember, ProposalStatus};

use crate::wallet_auth::AuthenticatedWallet;
use crate::AppState;

const QUG_DECIMALS_DIVISOR: f64 = 1e24;

fn parse_qnk_address(s: &str) -> Result<[u8; 32], String> {
    let hex_part = s.strip_prefix("qnk").unwrap_or(s);
    let bytes = hex::decode(hex_part).map_err(|_| format!("'{}' is not valid hex", s))?;
    if bytes.len() != 32 {
        return Err(format!("address must be 32 bytes, got {}", bytes.len()));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn addr_string(addr: &[u8; 32]) -> String {
    format!("qnk{}", hex::encode(addr))
}

fn store() -> Arc<q_multisig::MultisigStore> {
    q_multisig::storage::global()
}

fn err(msg: impl Into<String>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "success": false, "error": msg.into() }))
}

fn ok(data: serde_json::Value) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "success": true, "data": data }))
}

fn member_pubkey_for(addr: &[u8; 32]) -> Option<HybridPublicKey> {
    store().get_member_key(addr).map(|k| k.pubkey)
}

/// Resolve an ordinary qnk address to its `member_address()` inside a given
/// wallet, i.e. the value `wallet.find_member` actually indexes by. Returns
/// None if the address never registered a key, or isn't a member of this
/// wallet.
fn resolve_member_in_wallet(wallet: &MultisigWallet, ordinary_addr: &[u8; 32]) -> Option<MsAddress> {
    let pubkey = member_pubkey_for(ordinary_addr)?;
    let candidate = pubkey.member_address();
    wallet.find_member(&candidate).map(|_| candidate)
}

// ---------------------------------------------------------------------------
// POST /register-key
// ---------------------------------------------------------------------------

/// Path this endpoint's challenge is bound to. Not derived from the actual
/// request URI (unlike the shared `AuthenticatedWallet` extractor) — this
/// handler does its own narrow, self-contained verification (see module
/// doc), so the bound string just needs to match what the frontend signs.
const REGISTER_KEY_CHALLENGE_PATH: &str = "/api/v1/multisig/register-key";

#[derive(Deserialize)]
pub struct RegisterKeyRequest {
    pub address: String,
    pub timestamp: i64,
    pub ed25519_signature: String,   // hex, 64 bytes, over the challenge below
    pub dilithium5_signature: String, // hex, detached, over the SAME challenge
    pub dilithium5_public_key: String, // hex, ~2592 bytes
}

/// Registers a member's hybrid pubkey bundle so a CEO can later add them to
/// a multisig org by their ordinary qnk address. Deliberately does NOT go
/// through the shared `AuthenticatedWallet` extractor's Hybrid scheme: that
/// scheme verifies Dilithium5 via the embedded-SignedMessage form, which no
/// client in this codebase actually produces (the frontend's only
/// Dilithium5 primitive, `dilithium5Sign`, is detached — the same
/// convention already shipped for hybrid-signed sends). Verifying here
/// directly, with the SAME detached convention, avoids inventing a third
/// incompatible wire format and avoids touching the widely-shared
/// `wallet_auth.rs` extractor for the sake of one endpoint.
///
/// Proves two things cryptographically before persisting anything: (1) the
/// caller controls the Ed25519 key backing their ordinary wallet address
/// (signs a fresh, timestamped, path-bound challenge — same construction
/// `AuthenticatedWallet` uses), and (2) the caller controls the Dilithium5
/// secret key matching the public key they're registering (signs the same
/// challenge bytes, detached-verified against the submitted public key).
pub async fn register_key(Json(request): Json<RegisterKeyRequest>) -> Json<serde_json::Value> {
    let address = match parse_qnk_address(&request.address) {
        Ok(a) => a,
        Err(e) => return err(e),
    };

    let now = chrono::Utc::now().timestamp();
    if (now - request.timestamp).abs() > 300 {
        return err("Timestamp expired — must be within 5 minutes of server time");
    }

    // Same challenge construction as AuthenticatedWallet:
    // SHA3-256(address(32) || timestamp_LE(8) || path_utf8)
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    hasher.update(address);
    hasher.update(request.timestamp.to_le_bytes());
    hasher.update(REGISTER_KEY_CHALLENGE_PATH.as_bytes());
    let challenge = hasher.finalize();

    let ed25519_sig_bytes = match hex::decode(&request.ed25519_signature) {
        Ok(b) if b.len() == 64 => b,
        _ => return err("ed25519_signature must be 64 bytes hex"),
    };
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&ed25519_sig_bytes);
    let ed25519_sig = Ed25519Signature::from_bytes(&sig_arr);
    let ed25519_pk = match VerifyingKey::from_bytes(&address) {
        Ok(k) => k,
        Err(_) => return err("Address is not a valid Ed25519 public key"),
    };
    if let Err(e) = ed25519_pk.verify_strict(&challenge, &ed25519_sig) {
        tracing::warn!("🚫 [MULTISIG] register-key Ed25519 verify failed for {}: {}", request.address, e);
        return err("Ed25519 signature verification failed");
    }

    let dilithium5_pk_bytes = match hex::decode(&request.dilithium5_public_key) {
        Ok(b) => b,
        Err(_) => return err("dilithium5_public_key is not valid hex"),
    };
    let dilithium5_sig_bytes = match hex::decode(&request.dilithium5_signature) {
        Ok(b) => b,
        Err(_) => return err("dilithium5_signature is not valid hex"),
    };
    let dilithium5_pk = match pqcrypto_dilithium::dilithium5::PublicKey::from_bytes(&dilithium5_pk_bytes) {
        Ok(k) => k,
        Err(_) => return err("Invalid Dilithium5 public key bytes"),
    };
    let dilithium5_sig =
        match pqcrypto_dilithium::dilithium5::DetachedSignature::from_bytes(&dilithium5_sig_bytes) {
            Ok(s) => s,
            Err(_) => return err("Invalid Dilithium5 signature bytes"),
        };
    if pqcrypto_dilithium::dilithium5::verify_detached_signature(&dilithium5_sig, &challenge, &dilithium5_pk)
        .is_err()
    {
        tracing::warn!("🚫 [MULTISIG] register-key Dilithium5 verify failed for {}", request.address);
        return err("Dilithium5 signature verification failed");
    }

    let pubkey = HybridPublicKey {
        ed25519: ed25519_pk,
        dilithium5: dilithium5_pk_bytes,
    };
    let member_addr = pubkey.member_address();
    store().register_member_key(address, pubkey);
    store().clone().spawn_persist();

    tracing::info!(
        "🤝 [MULTISIG] {} registered a multisig-capable key (member_addr={})",
        &request.address[..16.min(request.address.len())],
        hex::encode(&member_addr[..8]),
    );

    ok(serde_json::json!({
        "address": addr_string(&address),
        "member_address": hex::encode(member_addr),
        "message": "Key registered. You can now be added to a multisig organization.",
    }))
}

#[derive(Deserialize)]
pub struct RegistrationStatusQuery {
    addresses: String, // comma-separated qnk addresses
}

/// GET /registration-status?addresses=qnk...,qnk...
/// Lets a CEO check which invited associates have registered yet, before
/// attempting to deploy — avoids a guess-and-fail create call.
pub async fn registration_status(
    axum::extract::Query(q): axum::extract::Query<RegistrationStatusQuery>,
) -> Json<serde_json::Value> {
    let s = store();
    let results: Vec<serde_json::Value> = q
        .addresses
        .split(',')
        .filter(|a| !a.trim().is_empty())
        .map(|a| {
            let trimmed = a.trim();
            match parse_qnk_address(trimmed) {
                Ok(addr) => serde_json::json!({
                    "address": trimmed,
                    "registered": s.is_registered(&addr),
                }),
                Err(e) => serde_json::json!({ "address": trimmed, "registered": false, "error": e }),
            }
        })
        .collect();
    ok(serde_json::json!({ "statuses": results }))
}

// ---------------------------------------------------------------------------
// POST /create
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct CreateOrgMember {
    pub address: String,
    pub name: String,
    pub role: String,
    #[serde(default)]
    pub per_tx_limit: u64,
    #[serde(default)]
    pub daily_limit: u64,
    #[serde(default)]
    pub approval_threshold: u64,
    #[serde(default)]
    pub approvals_required: u64,
}

#[derive(Deserialize)]
pub struct CreateOrgRequest {
    pub org_name: String,
    pub threshold: u8,
    pub members: Vec<CreateOrgMember>,
}

#[derive(Serialize)]
struct CreateOrgResult {
    success: bool,
    treasury_address: Option<String>,
    error: Option<String>,
    pending_registration: Vec<String>,
}

pub async fn create_org(
    auth: AuthenticatedWallet,
    State(_state): State<Arc<AppState>>,
    Json(request): Json<CreateOrgRequest>,
) -> Json<serde_json::Value> {
    if request.members.len() < 2 {
        return err("A multisig organization needs at least 2 members (the CEO counts as one).");
    }
    if request.org_name.trim().is_empty() {
        return err("Organization name cannot be empty.");
    }

    let caller_included = request.members.iter().any(|m| {
        parse_qnk_address(&m.address)
            .map(|a| a == auth.address)
            .unwrap_or(false)
    });
    if !caller_included {
        return err("The caller's own address must be included in the member list.");
    }

    let s = store();
    let mut members: Vec<Member> = Vec::new();
    let mut pending: Vec<String> = Vec::new();
    let mut policy_members: Vec<OrgPolicyMember> = Vec::new();

    for m in &request.members {
        let addr = match parse_qnk_address(&m.address) {
            Ok(a) => a,
            Err(e) => return err(format!("Invalid address '{}': {}", m.address, e)),
        };
        match s.get_member_key(&addr) {
            Some(reg) => {
                members.push(Member {
                    label: m.name.clone(),
                    pubkey: reg.pubkey,
                });
            }
            None => pending.push(format!("{} ({})", m.name, m.address)),
        }
        policy_members.push(OrgPolicyMember {
            address_hex: hex::encode(addr),
            name: m.name.clone(),
            role: m.role.clone(),
            per_tx_limit: m.per_tx_limit,
            daily_limit: m.daily_limit,
            approval_threshold: m.approval_threshold,
            approvals_required: m.approvals_required,
        });
    }

    if !pending.is_empty() {
        return Json(serde_json::to_value(CreateOrgResult {
            success: false,
            treasury_address: None,
            error: Some(format!(
                "{} member(s) haven't registered their multisig key yet — they need to open \
                 their own wallet and hit \"Join organization\" once before you can deploy.",
                pending.len()
            )),
            pending_registration: pending,
        })
        .unwrap());
    }

    let wallet = match MultisigWallet::new(members, request.threshold, request.org_name.clone()) {
        Ok(w) => w,
        Err(e) => return err(format!("Could not create wallet: {}", e)),
    };

    let policy = OrgPolicy {
        org_name: request.org_name.clone(),
        ceo_address_hex: hex::encode(auth.address),
        members: policy_members,
        updated_at_unix: chrono::Utc::now().timestamp(),
    };

    s.set_org_policy(wallet.address, policy);
    s.insert_wallet(wallet.clone());
    s.clone().spawn_persist();

    tracing::info!(
        "🏛️ [MULTISIG] Org '{}' deployed: treasury={} threshold={}/{} ceo={}",
        request.org_name,
        wallet.address_string(),
        request.threshold,
        wallet.members.len(),
        &addr_string(&auth.address)[..16],
    );

    Json(serde_json::to_value(CreateOrgResult {
        success: true,
        treasury_address: Some(wallet.address_string()),
        error: None,
        pending_registration: vec![],
    })
    .unwrap())
}

// ---------------------------------------------------------------------------
// GET /wallet/:addr  and  GET /mine
// ---------------------------------------------------------------------------

fn wallet_summary(wallet: &MultisigWallet) -> serde_json::Value {
    let policy = store().get_org_policy(&wallet.address);
    serde_json::json!({
        "treasury_address": wallet.address_string(),
        "label": wallet.label,
        "threshold": wallet.default_threshold,
        "member_count": wallet.members.len(),
        "members": wallet.members.iter().map(|m| serde_json::json!({
            "label": m.label,
            "member_address": hex::encode(m.pubkey.member_address()),
        })).collect::<Vec<_>>(),
        "created_at_unix": wallet.created_at_unix,
        "policy": policy.map(|p| serde_json::json!({
            "org_name": p.org_name,
            "ceo_address": format!("qnk{}", p.ceo_address_hex),
            "updated_at_unix": p.updated_at_unix,
            "members": p.members.iter().map(|m| serde_json::json!({
                "address": format!("qnk{}", m.address_hex),
                "name": m.name,
                "role": m.role,
                "per_tx_limit": m.per_tx_limit,
                "daily_limit": m.daily_limit,
                "approval_threshold": m.approval_threshold,
                "approvals_required": m.approvals_required,
            })).collect::<Vec<_>>(),
        })),
    })
}

pub async fn get_wallet(Path(addr_str): Path<String>) -> Json<serde_json::Value> {
    let addr = match parse_qnk_address(&addr_str) {
        Ok(a) => a,
        Err(e) => return err(e),
    };
    match store().get_wallet(&addr) {
        Some(w) => ok(wallet_summary(&w)),
        None => err("No multisig wallet at that address"),
    }
}

/// Also returns, per wallet, the caller's OWN declared role/limits and a
/// treasury balance snapshot — everything the "you're now a member"
/// celebration screen needs in one call.
pub async fn my_orgs(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let s = store();
    let mut results = Vec::new();
    for wallet in s.list_wallets() {
        if resolve_member_in_wallet(&wallet, &auth.address).is_none() {
            continue;
        }
        let mut summary = wallet_summary(&wallet);
        let balance = {
            let balances = state.wallet_balances.read().await;
            balances.get(&wallet.address).copied().unwrap_or(0)
        };
        summary["treasury_balance_qug"] = serde_json::json!(balance as f64 / QUG_DECIMALS_DIVISOR);
        if let Some(policy) = store().get_org_policy(&wallet.address) {
            let caller_hex = hex::encode(auth.address);
            if let Some(me) = policy.members.iter().find(|m| m.address_hex == caller_hex) {
                summary["my_role"] = serde_json::json!(me.role);
                summary["my_per_tx_limit"] = serde_json::json!(me.per_tx_limit);
                summary["my_daily_limit"] = serde_json::json!(me.daily_limit);
            }
        }
        results.push(summary);
    }
    ok(serde_json::json!({ "organizations": results }))
}

// ---------------------------------------------------------------------------
// POST /propose
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ProposeRequest {
    pub wallet_addr: String,
    pub token: String, // "QUG" or a qnk token address
    pub recipient: String,
    pub amount_raw: String, // u128 as decimal string, 24-dec raw units
    #[serde(default)]
    pub memo: Option<String>,
    #[serde(default)]
    pub required_override: Option<u8>,
}

pub async fn propose(
    auth: AuthenticatedWallet,
    Json(request): Json<ProposeRequest>,
) -> Json<serde_json::Value> {
    let wallet_addr = match parse_qnk_address(&request.wallet_addr) {
        Ok(a) => a,
        Err(e) => return err(e),
    };
    let s = store();
    let wallet = match s.get_wallet(&wallet_addr) {
        Some(w) => w,
        None => return err("No multisig wallet at that address"),
    };
    if resolve_member_in_wallet(&wallet, &auth.address).is_none() {
        return err("You are not a registered member of this organization");
    }
    let recipient = match parse_qnk_address(&request.recipient) {
        Ok(a) => a,
        Err(e) => return err(format!("Invalid recipient: {}", e)),
    };
    let amount_raw: u128 = match request.amount_raw.parse() {
        Ok(a) if a > 0 => a,
        _ => return err("amount_raw must be a positive integer (24-decimal raw units)"),
    };

    let action = MultisigAction::Transfer {
        token: request.token,
        recipient,
        amount_raw,
        memo: request.memo,
    };
    let proposal = MultisigProposal::new(
        wallet_addr,
        action,
        request.required_override,
        wallet.default_threshold,
        wallet.members.len(),
    );
    let payload_hash = proposal.payload_hash();
    let proposal_id = proposal.id;
    s.insert_proposal(proposal);
    s.clone().spawn_persist();

    tracing::info!(
        "📝 [MULTISIG] Proposal {} created on wallet {} by {}",
        proposal_id,
        addr_string(&wallet_addr),
        &addr_string(&auth.address)[..16],
    );

    ok(serde_json::json!({
        "proposal_id": proposal_id.to_string(),
        "payload_hash": hex::encode(payload_hash),
        "required_signatures": wallet.effective_threshold(request.required_override),
        "message": "Proposal created. Each required signer must sign payload_hash and POST /sign.",
    }))
}

/// serde's default enum/array serialization would emit `Address` ([u8;32])
/// as a raw JSON array of 32 numbers — turn it into the qnk-string shape
/// the frontend (and every other endpoint in this API) actually uses.
fn action_to_json(action: &MultisigAction) -> serde_json::Value {
    match action {
        MultisigAction::Transfer { token, recipient, amount_raw, memo } => serde_json::json!({
            "type": "Transfer",
            "token": token,
            "recipient": addr_string(recipient),
            "amount_raw": amount_raw.to_string(),
            "memo": memo,
        }),
        MultisigAction::MintToken { symbol, name, decimals, initial_supply_raw, initial_holders } => serde_json::json!({
            "type": "MintToken",
            "symbol": symbol,
            "name": name,
            "decimals": decimals,
            "initial_supply_raw": initial_supply_raw.to_string(),
            "initial_holders": initial_holders.iter().map(|(a, amt)| serde_json::json!({"address": addr_string(a), "amount_raw": amt.to_string()})).collect::<Vec<_>>(),
        }),
        MultisigAction::SetDefaultThreshold { new_threshold } => serde_json::json!({
            "type": "SetDefaultThreshold",
            "new_threshold": new_threshold,
        }),
        MultisigAction::RotateMember { remove_addr, add_member_label, add_member_ed25519, add_member_dilithium5 } => serde_json::json!({
            "type": "RotateMember",
            "remove_addr": remove_addr.map(|a| addr_string(&a)),
            "add_member_label": add_member_label,
            "add_member_ed25519": add_member_ed25519.map(hex::encode),
            "add_member_dilithium5": add_member_dilithium5.as_ref().map(hex::encode),
        }),
    }
}

pub async fn list_proposals(Path(addr_str): Path<String>) -> Json<serde_json::Value> {
    let addr = match parse_qnk_address(&addr_str) {
        Ok(a) => a,
        Err(e) => return err(e),
    };
    let s = store();
    let proposals = s.proposals_for_wallet(&addr);
    let out: Vec<serde_json::Value> = proposals
        .iter()
        .map(|p| {
            serde_json::json!({
                "proposal_id": p.id.to_string(),
                "action": action_to_json(&p.action),
                "required": p.required,
                "signatures_collected": p.signatures.len(),
                "status": p.status,
                "created_at_unix": p.created_at_unix,
                "payload_hash": hex::encode(p.payload_hash()),
            })
        })
        .collect();
    ok(serde_json::json!({ "proposals": out }))
}

// ---------------------------------------------------------------------------
// POST /sign
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct SignRequest {
    pub proposal_id: String,
    pub ed25519_sig: String,        // hex, 64 bytes
    pub dilithium5_signature: String, // hex, detached, ~4627 bytes
}

pub async fn sign_proposal(
    auth: AuthenticatedWallet,
    Json(request): Json<SignRequest>,
) -> Json<serde_json::Value> {
    let proposal_id = match request.proposal_id.parse() {
        Ok(id) => id,
        Err(_) => return err("Invalid proposal_id (must be a UUID)"),
    };
    let s = store();
    let proposal = match s.get_proposal(&proposal_id) {
        Some(p) => p,
        None => return err("Proposal not found"),
    };
    let wallet = match s.get_wallet(&proposal.wallet_addr) {
        Some(w) => w,
        None => return err("Proposal references a wallet that no longer exists"),
    };
    let member_addr = match resolve_member_in_wallet(&wallet, &auth.address) {
        Some(m) => m,
        None => return err("You are not a registered member of this organization"),
    };

    let ed25519_sig_bytes = match hex::decode(&request.ed25519_sig) {
        Ok(b) if b.len() == 64 => b,
        _ => return err("ed25519_sig must be 64 bytes hex"),
    };
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&ed25519_sig_bytes);
    let ed25519_sig = Ed25519Signature::from_bytes(&sig_arr);

    let dilithium5_signature = match hex::decode(&request.dilithium5_signature) {
        Ok(b) => b,
        Err(_) => return err("dilithium5_signature is not valid hex"),
    };

    let contrib = q_multisig::proposal::SignatureContribution {
        member_addr,
        ed25519_sig,
        dilithium5_signature,
        at_unix: chrono::Utc::now().timestamp(),
    };

    if let Err(e) = q_multisig::verify::verify_member_signature(&wallet, &proposal, &contrib) {
        tracing::warn!(
            "🚫 [MULTISIG] Rejected signature on proposal {} from {}: {}",
            proposal_id,
            &addr_string(&auth.address)[..16],
            e
        );
        return err(format!("Signature verification failed: {}", e));
    }

    let updated = s.update_proposal(&proposal_id, |p| {
        p.add_signature(contrib);
        if p.has_threshold_count() {
            p.status = ProposalStatus::Threshold;
        }
    });
    s.clone().spawn_persist();

    let updated = match updated {
        Some(p) => p,
        None => return err("Proposal disappeared mid-update"),
    };

    tracing::info!(
        "✍️ [MULTISIG] {} signed proposal {} ({}/{})",
        &addr_string(&auth.address)[..16],
        proposal_id,
        updated.signatures.len(),
        updated.required,
    );

    ok(serde_json::json!({
        "proposal_id": proposal_id.to_string(),
        "signatures_collected": updated.signatures.len(),
        "required": updated.required,
        "status": updated.status,
        "ready_to_execute": matches!(updated.status, ProposalStatus::Threshold),
    }))
}

// ---------------------------------------------------------------------------
// POST /execute
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ExecuteRequest {
    pub proposal_id: String,
}

pub async fn execute_proposal(
    auth: AuthenticatedWallet,
    State(state): State<Arc<AppState>>,
    Json(request): Json<ExecuteRequest>,
) -> Json<serde_json::Value> {
    let proposal_id = match request.proposal_id.parse() {
        Ok(id) => id,
        Err(_) => return err("Invalid proposal_id (must be a UUID)"),
    };
    let s = store();
    let proposal = match s.get_proposal(&proposal_id) {
        Some(p) => p,
        None => return err("Proposal not found"),
    };
    if matches!(proposal.status, ProposalStatus::Executed) {
        return err("Proposal already executed");
    }
    if matches!(proposal.status, ProposalStatus::Cancelled) {
        return err("Proposal was cancelled");
    }
    let wallet = match s.get_wallet(&proposal.wallet_addr) {
        Some(w) => w,
        None => return err("Proposal references a wallet that no longer exists"),
    };
    if resolve_member_in_wallet(&wallet, &auth.address).is_none() {
        return err("You are not a registered member of this organization");
    }
    if let Err(e) = q_multisig::verify::verify_proposal(&wallet, &proposal) {
        return err(format!("Proposal does not have enough valid signatures: {}", e));
    }

    let (token, recipient, amount_raw, memo) = match &proposal.action {
        MultisigAction::Transfer {
            token,
            recipient,
            amount_raw,
            memo,
        } => (token.clone(), *recipient, *amount_raw, memo.clone()),
        other => {
            return err(format!(
                "This proposal type ({:?}) is signed and threshold-met, but executing it isn't \
                 implemented yet — only Transfer proposals move funds today.",
                other
            ))
        }
    };

    if token.to_uppercase() != "QUG" {
        return err("v0 only supports QUG transfers from the treasury; custom-token treasury transfers aren't wired up yet");
    }

    // Move funds — same wallet_balances map every other QUG transfer uses.
    // No protocol fee is charged on internal treasury moves in v0.
    {
        let mut balances = state.wallet_balances.write().await;
        let treasury_balance = balances.get(&wallet.address).copied().unwrap_or(0);
        if treasury_balance < amount_raw {
            return err(format!(
                "Treasury has {:.8} QUG, proposal asks for {:.8} QUG",
                treasury_balance as f64 / QUG_DECIMALS_DIVISOR,
                amount_raw as f64 / QUG_DECIMALS_DIVISOR
            ));
        }
        balances.insert(wallet.address, treasury_balance - amount_raw);
        let recipient_balance = balances.get(&recipient).copied().unwrap_or(0);
        balances.insert(recipient, recipient_balance + amount_raw);
    }

    s.update_proposal(&proposal_id, |p| {
        p.status = ProposalStatus::Executed;
    });
    s.clone().spawn_persist();

    tracing::warn!(
        "💸 [MULTISIG EXECUTE] {} → {} : {:.8} QUG (proposal {}, memo={:?}, triggered by {})",
        wallet.address_string(),
        addr_string(&recipient),
        amount_raw as f64 / QUG_DECIMALS_DIVISOR,
        proposal_id,
        memo,
        &addr_string(&auth.address)[..16],
    );

    ok(serde_json::json!({
        "proposal_id": proposal_id.to_string(),
        "executed": true,
        "amount_qug": amount_raw as f64 / QUG_DECIMALS_DIVISOR,
        "recipient": addr_string(&recipient),
    }))
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn create_multisig_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/register-key", post(register_key))
        .route("/registration-status", get(registration_status))
        .route("/create", post(create_org))
        .route("/wallet/:addr", get(get_wallet))
        .route("/mine", get(my_orgs))
        .route("/propose", post(propose))
        .route("/proposals/:addr", get(list_proposals))
        .route("/sign", post(sign_proposal))
        .route("/execute", post(execute_proposal))
}
