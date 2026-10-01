//! Verify a peer's balance snapshot against the chain's own headers before a fresh node
//! adopts it — the SIGIL principle ("roots in the header make a snapshot verifiable instead
//! of trusted") applied to Quillon's existing `balance_root_v1` commitment.
//!
//! Why (measured 2026-10-01 on two fresh rig followers): the one-time bootstrap import adopts
//! the wallet balances of whichever peer answers first, unchecked —
//!   * follower A, at block 562, adopted Epsilon's balances "at height 23,339,713";
//!   * follower B imported 6,211 wallets from follower A (a fresh node holding copied,
//!     unverified balances, labelled "height 21,001"), then 10 s later a SECOND "one-time"
//!     import from Epsilon — two concurrent state-sync responses both saw "not done yet".
//!
//! What a header gives us: from height 17,742,000 the producer writes `balance_root_v1` into
//! `BlockHeader.state_root`, and the root of the state after block H sits in header H+1
//! (measured: live root at 23,342,547 == header 23,342,548). A header is trusted only if its
//! `producer_public_key` is the pinned producer id and its Ed25519 signature verifies — so it
//! does not matter which server the header came from.
//!
//! Mode (`Q_BOOTSTRAP_VERIFY`): `shadow` (default) logs the verdict and imports anyway;
//! `enforce` imports only a VERIFIED snapshot; `off` skips the check.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing::{error, info, warn};

/// First height whose header carries a balance root (block_producer.rs BALANCE_ROOT_PRODUCER_START).
pub const FIRST_ROOTED_HEADER: u64 = 17_742_000;

/// The producer pin baked into the binary (`<ed25519 id hex>:<dilithium hex>`), same file
/// main.rs registers for PQC-004.
const DEFAULT_PRODUCER_PIN: &str = include_str!("producer_dilithium_pin.txt");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Off,
    Shadow,
    Enforce,
}

pub fn mode() -> Mode {
    match std::env::var("Q_BOOTSTRAP_VERIFY").unwrap_or_default().to_ascii_lowercase().as_str() {
        "off" | "0" => Mode::Off,
        "enforce" => Mode::Enforce,
        _ => Mode::Shadow,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The snapshot's root equals the state_root of a producer-signed header.
    Verified { header_height: u64 },
    /// Authentic headers were checked and none commits to this snapshot.
    Mismatch { checked: Vec<u64> },
    /// Could not be checked (below the first rooted header, no reachable/authentic header...).
    Unverifiable(String),
}

impl Verdict {
    pub fn is_verified(&self) -> bool {
        matches!(self, Verdict::Verified { .. })
    }
}

/// Parse a snapshot the same way the importer does (hex address -> 32 bytes, decimal amount)
/// and hash it with the node's own `balance_root_v1`.
pub fn snapshot_root(wallet_balances: &HashMap<String, String>) -> [u8; 32] {
    q_storage::balance_root::balance_root_v1(wallet_balances.iter().filter_map(|(addr_hex, amount)| {
        let bytes = hex::decode(addr_hex).ok()?;
        if bytes.len() != 32 {
            return None;
        }
        let mut addr = [0u8; 32];
        addr.copy_from_slice(&bytes);
        Some((addr, amount.parse::<u128>().ok()?))
    }))
}

/// The pinned producer's Ed25519 id: `Q_PRODUCER_DILITHIUM_PUBKEY_HEX` wins, else the baked pin.
pub fn pinned_producer_ids() -> Vec<[u8; 32]> {
    let pins = std::env::var("Q_PRODUCER_DILITHIUM_PUBKEY_HEX")
        .unwrap_or_else(|_| DEFAULT_PRODUCER_PIN.trim().to_string());
    parse_pin_ids(&pins)
}

pub fn parse_pin_ids(pins: &str) -> Vec<[u8; 32]> {
    pins.split(',')
        .filter_map(|entry| {
            let id_hex = entry.trim().split(':').next()?;
            let id = hex::decode(id_hex.trim()).ok()?;
            if id.len() != 32 {
                return None;
            }
            let mut out = [0u8; 32];
            out.copy_from_slice(&id);
            Some(out)
        })
        .collect()
}

/// `Ok(state_root)` iff the block was signed by a pinned producer.
pub fn authentic_state_root(block: &q_types::QBlock, pinned: &[[u8; 32]]) -> Result<[u8; 32], String> {
    let producer = block
        .header
        .producer_public_key
        .ok_or_else(|| "header has no producer public key".to_string())?;
    if !pinned.iter().any(|p| *p == producer) {
        return Err(format!("producer {}… is not the pinned producer", hex::encode(&producer[..8])));
    }
    block.verify_producer_signature()?;
    Ok(block.header.state_root)
}

/// Where to fetch headers from. Any server will do — authenticity comes from the signature.
fn header_sources() -> Vec<String> {
    match std::env::var("Q_BOOTSTRAP_VERIFY_URLS") {
        Ok(v) if !v.trim().is_empty() => v.split(',').map(|s| s.trim().trim_end_matches('/').to_string()).collect(),
        _ => vec!["https://quillon.xyz".to_string(), "http://89.149.241.126:8080".to_string()],
    }
}

async fn fetch_block(client: &reqwest::Client, base: &str, height: u64) -> Option<q_types::QBlock> {
    #[derive(serde::Deserialize)]
    struct Resp {
        success: bool,
        data: Option<q_types::QBlock>,
    }
    let url = format!("{}/api/v1/blocks/{}", base, height);
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let r: Resp = resp.json().await.ok()?;
    if r.success { r.data } else { None }
}

/// Check a snapshot labelled `labelled_height` (the serving peer's height) against the
/// producer-signed headers just after it. Order: H+1 (the exact commitment), then H+2 and H
/// (the label and the balances can be one block apart when the tip moves mid-snapshot), H+3.
pub async fn verify_snapshot(wallet_balances: &HashMap<String, String>, labelled_height: u64) -> Verdict {
    if labelled_height.saturating_add(1) < FIRST_ROOTED_HEADER {
        return Verdict::Unverifiable(format!(
            "labelled height {} is below {} — headers there carry no balance root",
            labelled_height, FIRST_ROOTED_HEADER
        ));
    }
    let pinned = pinned_producer_ids();
    if pinned.is_empty() {
        return Verdict::Unverifiable("no pinned producer id".into());
    }
    let root = snapshot_root(wallet_balances);
    let client = match reqwest::Client::builder().timeout(Duration::from_secs(10)).build() {
        Ok(c) => c,
        Err(e) => return Verdict::Unverifiable(format!("http client: {}", e)),
    };
    let candidates = [labelled_height + 1, labelled_height + 2, labelled_height, labelled_height + 3];
    let mut checked = Vec::new();
    let mut last_problem = String::from("no header source answered");
    for h in candidates {
        for base in header_sources() {
            let Some(block) = fetch_block(&client, &base, h).await else { continue };
            if block.header.height != h {
                last_problem = format!("{} answered height {} for {}", base, block.header.height, h);
                continue;
            }
            match authentic_state_root(&block, &pinned) {
                Ok(state_root) => {
                    checked.push(h);
                    if state_root == root {
                        return Verdict::Verified { header_height: h };
                    }
                    break; // authentic header for h, root differs: try the next height
                }
                Err(e) => last_problem = format!("header {} from {}: {}", h, base, e),
            }
        }
    }
    if checked.is_empty() {
        Verdict::Unverifiable(last_problem)
    } else {
        Verdict::Mismatch { checked }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VerdictRecord {
    pub mode: String,
    pub labelled_height: u64,
    pub wallets: usize,
    pub snapshot_root: String,
    pub verdict: String,
    pub imported: bool,
    pub unix_time: u64,
}

static LAST: Lazy<Mutex<Option<VerdictRecord>>> = Lazy::new(|| Mutex::new(None));

/// The most recent bootstrap-import verdict on this node (None = no import attempted).
pub fn last_verdict() -> Option<VerdictRecord> {
    LAST.lock().clone()
}

/// Equal-height balance-hash comparisons against peers (state-sync divergence check).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct DivergenceStats {
    pub matches: u64,
    pub mismatches: u64,
    /// Peer answered at a different height — not comparable, not evidence of anything.
    pub skipped_unequal_height: u64,
    pub last_mismatch: Option<DivergenceEvent>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DivergenceEvent {
    pub height: u64,
    pub peer_height: u64,
    pub our_hash: String,
    pub peer_hash: String,
    pub unix_time: u64,
}

static DIVERGENCE: Lazy<Mutex<DivergenceStats>> = Lazy::new(|| Mutex::new(DivergenceStats::default()));

pub fn note_divergence_skipped() {
    DIVERGENCE.lock().skipped_unequal_height += 1;
}

pub fn note_divergence_check(matched: bool, height: u64, peer_height: u64, our_hash: &str, peer_hash: &str) {
    let mut d = DIVERGENCE.lock();
    if matched {
        d.matches += 1;
    } else {
        d.mismatches += 1;
        d.last_mismatch = Some(DivergenceEvent {
            height,
            peer_height,
            our_hash: our_hash.to_string(),
            peer_hash: peer_hash.to_string(),
            unix_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        });
    }
}

pub fn divergence_stats() -> DivergenceStats {
    DIVERGENCE.lock().clone()
}

/// `GET /api/v1/integrity/bootstrap-verdict` (alias `/api/v1/integrity/status`): how this
/// node's balances were bootstrapped (`data: null` = no one-time import, e.g. the genesis
/// node) and the equal-height divergence record. Observability only — nothing is halted.
pub async fn bootstrap_verdict_handler() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "success": true,
        "mode": format!("{:?}", mode()).to_lowercase(),
        "data": last_verdict(),
        "divergence": divergence_stats(),
        "compare_rule": "balance hashes are compared only at EQUAL height; a mismatch is evidence only if it persists across several equal-height samples",
    }))
}

/// Called by the state-sync merge right before the one-time wallet import.
///
/// 1. Claims the import atomically (false -> true on `done`), so two concurrent state-sync
///    responses can no longer both run the "one-time" import.
/// 2. Verifies the snapshot (unless mode = off) and records the verdict.
/// 3. Returns whether to import. In `enforce` mode an unverified snapshot is refused and the
///    claim released, so a later, verifiable snapshot can still be imported.
pub async fn claim_and_verify_import(
    done: &AtomicBool,
    wallet_balances: &HashMap<String, String>,
    labelled_height: u64,
) -> bool {
    if done.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        info!("🔒 [BOOTSTRAP VERIFY] one-time wallet import already claimed by another state-sync response — skipping");
        return false;
    }
    let m = mode();
    let verdict = if m == Mode::Off {
        Verdict::Unverifiable("verification off (Q_BOOTSTRAP_VERIFY=off)".into())
    } else {
        verify_snapshot(wallet_balances, labelled_height).await
    };
    let import = match m {
        Mode::Enforce => verdict.is_verified(),
        _ => true,
    };
    let root_hex = hex::encode(snapshot_root(wallet_balances));
    match &verdict {
        Verdict::Verified { header_height } => info!(
            "✅ [BOOTSTRAP VERIFY] snapshot of {} wallets (labelled h={}) VERIFIED: root {}… == producer-signed header {}",
            wallet_balances.len(), labelled_height, &root_hex[..16], header_height
        ),
        Verdict::Mismatch { checked } => error!(
            "🚨 [BOOTSTRAP VERIFY] snapshot of {} wallets (labelled h={}) MISMATCH: root {}… is committed by none of the signed headers {:?} — mode {:?}, import={}",
            wallet_balances.len(), labelled_height, &root_hex[..16], checked, m, import
        ),
        Verdict::Unverifiable(why) => warn!(
            "⚠️ [BOOTSTRAP VERIFY] snapshot of {} wallets (labelled h={}) UNVERIFIABLE: {} — mode {:?}, import={}",
            wallet_balances.len(), labelled_height, why, m, import
        ),
    }
    *LAST.lock() = Some(VerdictRecord {
        mode: format!("{:?}", m).to_lowercase(),
        labelled_height,
        wallets: wallet_balances.len(),
        snapshot_root: root_hex,
        verdict: format!("{:?}", verdict),
        imported: import,
        unix_time: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    });
    if !import {
        done.store(false, Ordering::SeqCst); // release: a verifiable snapshot may still come
    }
    import
}

// Tests: crates/q-api-server/tests/bootstrap_verify_tests.rs
