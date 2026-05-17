//! VDF Lane Mining — Genus-2 Jacobian sequential computation for CPU-fair mining.
//!
//! This module implements the VDF mining lane as a self-contained thread that can
//! be spawned alongside BLAKE3 mining threads. It:
//!   1. Fetches challenges from the server (reuses the shared challenge cache)
//!   2. Computes the VDF (T sequential doublings on genus-2 Jacobian)
//!   3. Generates a Wesolowski proof
//!   4. Submits via the centralized solution submitter channel
//!
//! The VDF thread is inherently single-core — extra threads don't help.

use anyhow::Result;
use q_vdf::genus2_cantor::{
    CurveParams, JacElement, evaluate_vdf, generate_proof, verify_proof,
};
use sha3::{Digest, Sha3_256};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tracing::{debug, error, info, warn};

/// Check if VDF lane is active on the server by inspecting the challenge response.
pub fn check_vdf_active(challenge_json: &serde_json::Value) -> bool {
    challenge_json["data"]["vdf_lane_active"].as_bool().unwrap_or(false)
}

/// Get VDF target iterations from the challenge response.
pub fn get_vdf_iterations(challenge_json: &serde_json::Value) -> u64 {
    challenge_json["data"]["vdf_target_iterations"].as_u64().unwrap_or(4300)
}

/// VDF mining thread — runs on a single dedicated CPU core.
///
/// Arguments:
/// - `is_running`: shared flag to stop mining
/// - `wallet`: miner wallet address string
/// - `server_url`: API server URL
/// - `vdf_proofs_counter`: atomic counter for TUI stats
/// - `solution_tx`: channel to submit solutions to the centralized submitter
/// - `new_block_signal`: incremented by the SSE listener on every new block arrival.
///   Checked before and after VDF computation to avoid wasted work on stale challenges.
///   Using this instead of an HTTP re-check saves ~10-50ms latency per cycle.
pub fn vdf_mining_thread(
    is_running: Arc<AtomicBool>,
    wallet: String,
    server_url: String,
    vdf_proofs_counter: Arc<AtomicU64>,
    solution_tx: tokio::sync::mpsc::UnboundedSender<crate::solution_submitter::SolutionMessage>,
    tokio_handle: tokio::runtime::Handle,
    new_block_signal: Arc<AtomicU64>,
) {
    info!("🧮 VDF mining thread started (Genus-2 Jacobian, pq128 curve)");
    info!("   This thread uses exactly 1 CPU core — sequential computation");

    let curve = CurveParams::pq128();
    let mut nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new());

    let mut consecutive_errors = 0u32;

    while is_running.load(Ordering::Relaxed) {
        // ── Fetch challenge ──────────────────────────────────────
        let challenge_url = format!("{}/api/v1/mining/challenge", server_url);
        let resp = match client.get(&challenge_url).send() {
            Ok(r) => match r.text() {
                Ok(t) => t,
                Err(e) => {
                    warn!("🧮 VDF: challenge read error: {}", e);
                    backoff_sleep(&mut consecutive_errors, &is_running);
                    continue;
                }
            },
            Err(e) => {
                warn!("🧮 VDF: challenge fetch error: {}", e);
                backoff_sleep(&mut consecutive_errors, &is_running);
                continue;
            }
        };

        let resp_json: serde_json::Value = match serde_json::from_str(&resp) {
            Ok(j) => j,
            Err(_) => {
                warn!("🧮 VDF: invalid JSON from challenge endpoint");
                backoff_sleep(&mut consecutive_errors, &is_running);
                continue;
            }
        };

        // Check VDF lane is active
        if !check_vdf_active(&resp_json) {
            debug!("🧮 VDF lane not active, waiting 30s...");
            for _ in 0..30 {
                if !is_running.load(Ordering::Relaxed) { return; }
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            continue;
        }

        consecutive_errors = 0;
        let data = &resp_json["data"];
        let challenge_hash = match data["challenge_hash"].as_str() {
            Some(h) => h.to_string(),
            None => { continue; }
        };
        let difficulty_target = data["difficulty_target"].as_str().unwrap_or("ff").to_string();
        let block_height = data["block_height"].as_u64().unwrap_or(0);
        let vdf_iters_full = get_vdf_iterations(&resp_json);

        // VDF-001: operator cap to prevent runaway iteration counts from malicious challenges
        let vdf_iters_cap: u64 = std::env::var("Q_VDF_ITERATIONS_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(u64::MAX);
        let vdf_iters = vdf_iters_full.min(vdf_iters_cap);
        if vdf_iters < vdf_iters_full {
            debug!("[VDF] iterations capped: requested={} effective={} cap={}",
                vdf_iters_full, vdf_iters, vdf_iters_cap);
        }

        // Snapshot the new-block signal before starting computation.
        // If it changes by the time we finish, a new block arrived mid-compute.
        let block_signal_before = new_block_signal.load(Ordering::Relaxed);

        // ── Compute VDF ──────────────────────────────────────────
        nonce += 1;
        let challenge_bytes = match hex::decode(&challenge_hash) {
            Ok(b) => b,
            Err(_) => { continue; }
        };

        let mut hash_input = [0u8; 40];
        if challenge_bytes.len() >= 32 {
            hash_input[..32].copy_from_slice(&challenge_bytes[..32]);
        }
        hash_input[32..].copy_from_slice(&nonce.to_le_bytes());
        let seed = blake3::hash(&hash_input);

        let g = match JacElement::from_seed(seed.as_bytes(), &curve) {
            Ok(g) => g,
            Err(e) => {
                warn!("🧮 VDF: seed-to-element failed: {}", e);
                continue;
            }
        };

        let eval_start = std::time::Instant::now();
        let y = match evaluate_vdf(&g, vdf_iters, &curve, 0) {
            Ok(y) => y,
            Err(e) => {
                warn!("🧮 VDF: evaluation failed: {}", e);
                continue;
            }
        };
        let eval_time = eval_start.elapsed();

        // Check if a new block arrived during computation using the shared signal.
        // Avoids an HTTP round-trip (10-50ms) — this is a nanosecond atomic load.
        if new_block_signal.load(Ordering::Relaxed) != block_signal_before {
            debug!("🧮 VDF: new block arrived during eval (H:{}), restarting", block_height);
            continue;
        }

        // ── Generate proof ───────────────────────────────────────
        let proof_start = std::time::Instant::now();
        let proof = match generate_proof(&g, &y, vdf_iters, &curve) {
            Ok(p) => p,
            Err(e) => {
                warn!("🧮 VDF: proof generation failed: {}", e);
                continue;
            }
        };
        let proof_time = proof_start.elapsed();

        // Local verification
        match verify_proof(&g, &y, &proof, vdf_iters, &curve) {
            Ok(true) => {},
            Ok(false) => {
                error!("🧮 VDF: LOCAL VERIFICATION FAILED — skipping (bug?)");
                continue;
            }
            Err(e) => {
                error!("🧮 VDF: verification error: {}", e);
                continue;
            }
        }

        // ── Serialize and submit ─────────────────────────────────
        let fb = curve.field_bytes();
        let vdf_output_bytes = y.to_bytes(fb);
        let vdf_proof_bytes = proof.to_bytes(fb);
        let vdf_output_hex = hex::encode(&vdf_output_bytes);
        let vdf_proof_hex = hex::encode(&vdf_proof_bytes);

        let mut sha3 = Sha3_256::new();
        sha3.update(&vdf_output_bytes);
        let hash_hex = hex::encode(sha3.finalize());

        // Submit via the centralized solution submitter
        let solution = serde_json::json!({
            "miner_address": wallet,
            "nonce": nonce,
            "hash": hash_hex,
            "difficulty_target": difficulty_target,
            "challenge_hash": challenge_hash,
            "vdf_output": vdf_output_hex,
            "vdf_proof": vdf_proof_hex,
            "vdf_iterations_count": vdf_iters,
            "miner_id": format!("vdf-{}", &wallet[3..11.min(wallet.len())]),
            "worker_name": "VDF CPU Lane",
            "miner_version": env!("CARGO_PKG_VERSION"),
            "hash_rate": 1.0 / (eval_time.as_secs_f64() + proof_time.as_secs_f64()),
        });

        // Submit via HTTP (same as BLAKE3 submissions)
        let submit_url = format!("{}/api/v1/mining/submit", server_url);
        match client.post(&submit_url)
            .header("Content-Type", "application/json")
            .body(serde_json::to_string(&solution).unwrap_or_default())
            .send()
        {
            Ok(resp) => {
                let status = resp.status();
                vdf_proofs_counter.fetch_add(1, Ordering::Relaxed);
                if status.is_success() {
                    info!("🧮 VDF proof #{} accepted | H:{} | eval:{:.1}s proof:{:.1}s",
                        vdf_proofs_counter.load(Ordering::Relaxed),
                        block_height, eval_time.as_secs_f64(), proof_time.as_secs_f64());
                } else {
                    debug!("🧮 VDF proof #{} status:{} | H:{}",
                        vdf_proofs_counter.load(Ordering::Relaxed),
                        status, block_height);
                }
            }
            Err(e) => {
                warn!("🧮 VDF: submit error: {}", e);
            }
        }

    }

    info!("🧮 VDF mining thread stopped");
}

fn backoff_sleep(errors: &mut u32, is_running: &Arc<AtomicBool>) {
    *errors += 1;
    let delay = std::cmp::min(*errors * 2, 30);
    for _ in 0..delay {
        if !is_running.load(Ordering::Relaxed) { return; }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
