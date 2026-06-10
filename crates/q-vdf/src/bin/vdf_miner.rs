//! Q-NarwhalKnight VDF CPU Miner — Genus-2 Jacobian sequential mining
//!
//! Mines the VDF lane by computing sequential doublings on a genus-2
//! hyperelliptic curve Jacobian and submitting Wesolowski proofs.
//!
//! Usage:
//!   ./vdf-miner --server https://quillon.xyz --wallet qnk<your_address>
//!   ./vdf-miner --server http://localhost:8080 --wallet qnk<your_address>
//!
//! The miner:
//!   1. Fetches a mining challenge from the server
//!   2. Computes the VDF (T sequential doublings, ~2 seconds)
//!   3. Generates a Wesolowski proof (~3 seconds)
//!   4. Submits the proof to the server
//!   5. Repeats with the next challenge
//!
//! This is CPU-only mining. GPUs cannot speed it up — each doubling
//! depends on the previous one (inherently sequential).

use anyhow::Result;
use q_vdf::genus2_cantor::{
    CurveParams, JacElement, evaluate_vdf, generate_proof, verify_proof,
};
use sha3::{Digest, Sha3_256};
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let server = args.iter()
        .position(|a| a == "--server")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "https://quillon.xyz".to_string());

    let wallet = args.iter()
        .position(|a| a == "--wallet")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("ERROR: --wallet <address> is required");
            eprintln!("Usage: ./vdf-miner --server https://quillon.xyz --wallet qnk<your_address>");
            std::process::exit(1);
        });

    if !wallet.starts_with("qnk") || wallet.len() != 67 {
        eprintln!("ERROR: Invalid wallet address. Must start with 'qnk' and be 67 characters.");
        std::process::exit(1);
    }

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║      Q-NARWHALKNIGHT VDF CPU MINER (v10.3.4)            ║");
    println!("║      Curve: y² = x⁵ + x² - 1 (pq128, 256-bit)         ║");
    println!("║      Proof: Wesolowski O(log T) verification            ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Server: {:<49}║", &server);
    println!("║  Wallet: {}...{}  ║", &wallet[..10], &wallet[wallet.len()-6..]);
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let curve = CurveParams::pq128();
    let mut nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let mut total_proofs = 0u64;
    let mut total_accepted = 0u64;
    let mut total_rejected = 0u64;
    let start_time = Instant::now();

    loop {
        // ── Fetch challenge ──────────────────────────────────────────
        let challenge_url = format!("{}/api/v1/mining/challenge", server);
        let resp = match ureq::get(&challenge_url).call() {
            Ok(r) => r.into_string()?,
            Err(e) => {
                eprintln!("⚠️  Challenge fetch failed: {}. Retrying in 5s...", e);
                std::thread::sleep(Duration::from_secs(5));
                continue;
            }
        };
        let resp: serde_json::Value = serde_json::from_str(&resp)?;
        let data = &resp["data"];

        let challenge_hash = match data["challenge_hash"].as_str() {
            Some(h) => h.to_string(),
            None => {
                eprintln!("⚠️  Server returned no challenge (syncing?). Retrying in 10s...");
                std::thread::sleep(Duration::from_secs(10));
                continue;
            }
        };
        let difficulty_target = data["difficulty_target"].as_str().unwrap().to_string();
        let block_height = data["block_height"].as_u64().unwrap_or(0);
        let vdf_iters = data["vdf_target_iterations"].as_u64().unwrap_or(4300);
        let vdf_active = data["vdf_lane_active"].as_bool().unwrap_or(false);

        if !vdf_active {
            eprintln!("⚠️  VDF lane not active (height {}). Waiting 30s...", block_height);
            std::thread::sleep(Duration::from_secs(30));
            continue;
        }

        // ── Compute VDF ──────────────────────────────────────────────
        nonce += 1;
        let challenge_bytes = hex::decode(&challenge_hash)?;
        let mut hash_input = [0u8; 40];
        hash_input[..32].copy_from_slice(&challenge_bytes);
        hash_input[32..].copy_from_slice(&nonce.to_le_bytes());
        let seed = blake3::hash(&hash_input);

        let g = JacElement::from_seed(seed.as_bytes(), &curve)?;

        let eval_start = Instant::now();
        let y = evaluate_vdf(&g, vdf_iters, &curve, 0)?;
        let eval_time = eval_start.elapsed();

        // ── Generate proof ───────────────────────────────────────────
        let proof_start = Instant::now();
        let proof = generate_proof(&g, &y, vdf_iters, &curve)?;
        let proof_time = proof_start.elapsed();

        // ── Quick local verify ───────────────────────────────────────
        let valid = verify_proof(&g, &y, &proof, vdf_iters, &curve)?;
        if !valid {
            eprintln!("⚠️  Local verification FAILED — skipping submission (bug?)");
            continue;
        }

        // ── Serialize and submit ─────────────────────────────────────
        let fb = curve.field_bytes();
        let vdf_output_hex = hex::encode(y.to_bytes(fb));
        let vdf_proof_hex = hex::encode(proof.to_bytes(fb));

        let mut sha3 = Sha3_256::new();
        sha3.update(&y.to_bytes(fb));
        let hash_hex = hex::encode(sha3.finalize());

        let body = serde_json::json!({
            "miner_address": wallet,
            "nonce": nonce,
            "hash": hash_hex,
            "difficulty_target": difficulty_target,
            "challenge_hash": challenge_hash,
            "vdf_output": vdf_output_hex,
            "vdf_proof": vdf_proof_hex,
            "vdf_iterations_count": vdf_iters,
            "miner_id": format!("vdf-cpu-{}", &wallet[3..11]),
            "worker_name": "VDF CPU Miner",
            "miner_version": "10.3.4",
            "hash_rate": 1.0 / (eval_time.as_secs_f64() + proof_time.as_secs_f64()),
        });

        let submit_url = format!("{}/api/v1/mining/submit", server);
        let body_str = serde_json::to_string(&body)?;
        let submit_result = ureq::post(&submit_url)
            .set("Content-Type", "application/json")
            .send_string(&body_str);

        total_proofs += 1;
        let total_time = eval_time + proof_time;
        let uptime = start_time.elapsed().as_secs();
        let rate = if uptime > 0 { total_proofs as f64 / uptime as f64 * 3600.0 } else { 0.0 };

        match submit_result {
            Ok(resp) => {
                let resp_text = resp.into_string().unwrap_or_default();
                let resp_json: serde_json::Value = serde_json::from_str(&resp_text).unwrap_or_default();
                let accepted = resp_json["data"]["accepted"].as_bool().unwrap_or(false);
                if accepted {
                    total_accepted += 1;
                    let reward = resp_json["data"]["reward_qnk"].as_f64().unwrap_or(0.0);
                    println!("✅ #{} | H:{} | eval:{:.1}s proof:{:.1}s | reward:{:.6} QUG | {}/{} accepted | {:.0}/hr",
                        total_proofs, block_height,
                        eval_time.as_secs_f64(), proof_time.as_secs_f64(),
                        reward, total_accepted, total_proofs, rate);
                } else {
                    total_rejected += 1;
                    let msg = resp_json["data"]["message"].as_str()
                        .or_else(|| resp_json["error"].as_str())
                        .unwrap_or("unknown");
                    println!("❌ #{} | H:{} | {:.1}s | {} | {}/{} accepted",
                        total_proofs, block_height, total_time.as_secs_f64(),
                        msg, total_accepted, total_proofs);
                }
            }
            Err(e) => {
                total_rejected += 1;
                eprintln!("❌ #{} | H:{} | Submit error: {} | {}/{} accepted",
                    total_proofs, block_height, e, total_accepted, total_proofs);
            }
        }

        // Brief pause to avoid hammering server between blocks
        std::thread::sleep(Duration::from_millis(100));
    }
}
