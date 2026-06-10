//! VDF Test Miner — submits a single Genus-2 VDF proof to a server for testing.
//!
//! Usage:
//!   cargo run --release --package q-vdf --bin vdf_test_miner -- --server http://HOST:PORT
//!
//! This fetches a mining challenge, computes the VDF, generates a Wesolowski proof,
//! and submits it. Used to verify the server's PATH A verification works correctly.

use anyhow::Result;
use num_bigint::BigUint;
use q_vdf::genus2_cantor::{
    CurveParams, JacElement, WesolowskiProof, evaluate_vdf, generate_proof, verify_proof,
};
use sha3::{Digest, Sha3_256};
use std::time::Instant;

fn main() -> Result<()> {
    let server = std::env::args()
        .skip_while(|a| a != "--server")
        .nth(1)
        .unwrap_or_else(|| "http://5.79.79.158:8086".to_string());

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║         GENUS-2 VDF TEST MINER (v10.3.4)                ║");
    println!("║         Curve: y² = x⁵ + x² - 1 (pq128)               ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("Server: {}", server);

    // Step 1: Fetch challenge
    println!("\n[1/5] Fetching mining challenge...");
    let challenge_url = format!("{}/api/v1/mining/challenge", server);
    let resp_body = ureq::get(&challenge_url).call()?.into_string()?;
    let resp: serde_json::Value = serde_json::from_str(&resp_body)?;

    let data = &resp["data"];
    let challenge_hash = data["challenge_hash"].as_str().unwrap();
    let difficulty_target = data["difficulty_target"].as_str().unwrap();
    let block_height = data["block_height"].as_u64().unwrap();
    let vdf_lane_active = data["vdf_lane_active"].as_bool().unwrap_or(false);
    let vdf_target_iters = data["vdf_target_iterations"].as_u64().unwrap_or(4300);

    println!("  Block height:    {}", block_height);
    println!("  Challenge hash:  {}...", &challenge_hash[..16]);
    println!("  Difficulty:      {}...", &difficulty_target[..16]);
    println!("  VDF lane active: {}", vdf_lane_active);
    println!("  VDF iterations:  {}", vdf_target_iters);

    if !vdf_lane_active {
        println!("\n⚠️  VDF lane is NOT active on this server.");
        println!("    Set Q_GENUS2_VDF_ACTIVATION_HEIGHT=1 in the server env.");
        println!("    Continuing anyway (server will reject)...");
    }

    // Step 2: Derive seed and generator
    println!("\n[2/5] Deriving VDF generator from challenge...");
    let challenge_bytes = hex::decode(challenge_hash)?;
    let nonce: u64 = 42; // Test nonce

    let mut hash_input = [0u8; 40];
    hash_input[..32].copy_from_slice(&challenge_bytes);
    hash_input[32..].copy_from_slice(&nonce.to_le_bytes());
    let seed = blake3::hash(&hash_input);

    let curve = CurveParams::pq128();
    let g = JacElement::from_seed(seed.as_bytes(), &curve)?;
    println!("  Generator degree: {}", g.degree);
    println!("  Generator valid:  {}", g.validate(&curve));

    // Step 3: Evaluate VDF
    println!("\n[3/5] Evaluating VDF ({} sequential doublings)...", vdf_target_iters);
    let eval_start = Instant::now();
    let y = evaluate_vdf(&g, vdf_target_iters, &curve, 1000)?;
    let eval_time = eval_start.elapsed();
    println!("  VDF output valid: {}", y.validate(&curve));
    println!("  Evaluation time:  {:.3}s", eval_time.as_secs_f64());

    // Step 4: Generate Wesolowski proof
    println!("\n[4/5] Generating Wesolowski proof...");
    let proof_start = Instant::now();
    let proof = generate_proof(&g, &y, vdf_target_iters, &curve)?;
    let proof_time = proof_start.elapsed();
    println!("  Proof time:   {:.3}s", proof_time.as_secs_f64());

    // Verify locally before submitting
    let local_valid = verify_proof(&g, &y, &proof, vdf_target_iters, &curve)?;
    println!("  Local verify: {} (should be true)", local_valid);

    if !local_valid {
        anyhow::bail!("Local verification failed! Bug in VDF code.");
    }

    // Serialize
    let fb = curve.field_bytes();
    let vdf_output_bytes = y.to_bytes(fb);
    let vdf_proof_bytes = proof.to_bytes(fb);
    let vdf_output_hex = hex::encode(&vdf_output_bytes);
    let vdf_proof_hex = hex::encode(&vdf_proof_bytes);

    // Compute hash = SHA3-256(vdf_output_bytes) for the submission
    let mut sha3 = Sha3_256::new();
    sha3.update(&vdf_output_bytes);
    let hash_bytes = sha3.finalize();
    let hash_hex = hex::encode(&hash_bytes);

    println!("  VDF output: {} bytes ({}...)", vdf_output_bytes.len(), &vdf_output_hex[..20]);
    println!("  VDF proof:  {} bytes ({}...)", vdf_proof_bytes.len(), &vdf_proof_hex[..20]);
    println!("  Hash:       {}...", &hash_hex[..16]);

    // Step 5: Submit to server
    println!("\n[5/5] Submitting VDF solution to server...");
    let submit_url = format!("{}/api/v1/mining/submit", server);
    let miner_address = "qnk0000000000000000000000000000000000000000000000000000000000test01";

    let body = serde_json::json!({
        "miner_address": miner_address,
        "nonce": nonce,
        "hash": hash_hex,
        "difficulty_target": difficulty_target,
        "challenge_hash": challenge_hash,
        "vdf_output": vdf_output_hex,
        "vdf_proof": vdf_proof_hex,
        "vdf_iterations_count": vdf_target_iters,
        "miner_id": "vdf-test-miner-v1",
        "worker_name": "VDF Test Miner",
        "miner_version": "10.3.4",
        "hash_rate": 0.5,
    });

    let body_str = serde_json::to_string(&body)?;
    let submit_body = ureq::post(&submit_url)
        .set("Content-Type", "application/json")
        .send_string(&body_str)?
        .into_string()?;
    let submit_resp: serde_json::Value = serde_json::from_str(&submit_body)?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                    SUBMISSION RESULT                      ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Response: {}", serde_json::to_string_pretty(&submit_resp)?);
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Eval time:  {:.3}s", eval_time.as_secs_f64());
    println!("║  Proof time: {:.3}s", proof_time.as_secs_f64());
    println!("║  Total:      {:.3}s", (eval_time + proof_time).as_secs_f64());
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}
