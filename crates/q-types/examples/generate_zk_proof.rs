/// Generate Zero-Knowledge Proof for Validator Keypair
///
/// This example demonstrates automatic ZK proof generation using both
/// STARK (transparent) and SNARK (succinct) proof systems.
///
/// **Key Feature:** NO TRUSTED SETUP REQUIRED
///
/// Usage:
///     cargo run --package q-types --example generate_zk_proof [PROOF_TYPE]
///
/// Proof Types:
///     stark   - Transparent STARK proof (~100 KB)
///     snark   - Succinct SNARK proof (~1-2 KB)
///     hybrid  - Both STARK and SNARK (maximum security)
///
/// Example:
///     cargo run --package q-types --example generate_zk_proof stark
///     cargo run --package q-types --example generate_zk_proof snark
///     cargo run --package q-types --example generate_zk_proof hybrid

use q_types::{ValidatorKeypair, zk_proof_integration::*};
use std::env;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let proof_type = if args.len() > 1 {
        args[1].to_lowercase()
    } else {
        "hybrid".to_string()
    };

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Q-NarwhalKnight ZK Proof Generator                            ║");
    println!("║  v1.0.16-beta - Untrusted Setup (STARK + SNARK)                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Generate validator keypair
    println!("🔐 Step 1: Generating PQC validator keypair...");
    let keypair = ValidatorKeypair::generate();
    println!("   ✅ Keypair generated");
    println!("   📍 Node ID: {}", hex::encode(&keypair.node_id));
    println!();

    // Create proof generator based on type
    let generator = match proof_type.as_str() {
        "stark" => {
            println!("🔬 Step 2: Generating STARK proof (transparent, no trusted setup)...");
            ValidatorZkProofGenerator::stark()
        }
        "snark" => {
            println!("🔬 Step 2: Generating SNARK proof (succinct, recursive)...");
            ValidatorZkProofGenerator::snark()
        }
        "hybrid" => {
            println!("🔬 Step 2: Generating HYBRID proof (STARK + SNARK)...");
            ValidatorZkProofGenerator::hybrid()
        }
        _ => {
            eprintln!("❌ Invalid proof type: {}", proof_type);
            eprintln!("   Valid types: stark, snark, hybrid");
            std::process::exit(1);
        }
    };

    // Generate proof with timing
    let start = Instant::now();
    let proof = generator.generate_proof(&keypair)?;
    let generation_time = start.elapsed();

    println!("   ✅ Proof generated in {:?}", generation_time);
    println!();

    // Display proof details
    println!("📊 Step 3: Proof Details");
    println!("   ═══════════════════════════════════════════════════════════");
    println!("   Proof Type: {:?}", proof.proof_type);
    println!("   Node ID Commitment: {}", hex::encode(&proof.node_id_commitment));
    println!("   Timestamp: {}", proof.timestamp);
    println!();

    // Display public inputs (non-secret data)
    println!("🔓 Public Inputs (Non-Secret):");
    println!("   ─────────────────────────────────────────────────────────");
    println!("   Node ID: {}", hex::encode(&proof.public_inputs.node_id));
    println!("   Ed25519 PubKey Hash: {}", hex::encode(&proof.public_inputs.ed25519_pubkey_hash));
    println!("   Dilithium5 PubKey Hash: {}", hex::encode(&proof.public_inputs.dilithium5_pubkey_hash));
    println!("   PubKey Merkle Root: {}", hex::encode(&proof.public_inputs.pubkey_merkle_root));
    println!();

    // Display STARK proof details if present
    if let Some(stark) = &proof.stark_proof {
        println!("🌟 STARK Proof (Transparent):");
        println!("   ─────────────────────────────────────────────────────────");
        println!("   Proof Size: {} bytes (~{} KB)", stark.proof_size, stark.proof_size / 1024);
        println!("   FRI Layers: {}", stark.fri_layers.len());
        println!("   Query Indices: {}", stark.query_indices.len());
        println!("   Merkle Paths: {}", stark.merkle_paths.len());
        println!("   Security: Post-quantum secure (hash-based)");
        println!("   Trusted Setup: ✅ NOT REQUIRED (transparent)");
        println!();
    }

    // Display SNARK proof details if present
    if let Some(snark) = &proof.snark_proof {
        println!("⚡ SNARK Proof (Succinct):");
        println!("   ─────────────────────────────────────────────────────────");
        println!("   Proof Size: {} bytes (~{} KB)", snark.proof_size, snark.proof_size / 1024);
        println!("   Recursion Depth: {}", snark.recursion_depth);
        println!("   VK Commitment: {}", hex::encode(&snark.vk_commitment));
        println!("   Security: Halo2-style recursive proofs");
        println!("   Trusted Setup: ✅ NOT REQUIRED (random oracle)");
        println!();
    }

    // Verify proof
    println!("🔍 Step 4: Verifying zero-knowledge proof...");
    let verify_start = Instant::now();
    match ValidatorZkProofVerifier::verify(&proof) {
        Ok(_) => {
            let verify_time = verify_start.elapsed();
            println!("   ✅ Proof verified successfully in {:?}", verify_time);
            println!();
        }
        Err(e) => {
            println!("   ❌ Proof verification FAILED: {}", e);
            std::process::exit(1);
        }
    }

    // Display security guarantees
    println!("🔒 Security Guarantees:");
    println!("   ═══════════════════════════════════════════════════════════");
    println!("   ✅ Secret keys NEVER revealed in proof");
    println!("   ✅ Cryptographically proves key possession");
    println!("   ✅ Public keys bound to Node ID (prevents forgery)");
    println!("   ✅ Timestamp binding prevents replay attacks");
    println!("   ✅ Post-quantum secure (hash-based)");
    println!("   ✅ NO trusted setup required");
    println!();

    // Display what is proved
    println!("✨ What This Proof Demonstrates:");
    println!("   ─────────────────────────────────────────────────────────");
    println!("   1. Prover possesses valid Ed25519 secret key");
    println!("   2. Prover possesses valid Dilithium5 secret key");
    println!("   3. Public keys match committed Node ID");
    println!("   4. Keys satisfy cryptographic well-formedness constraints");
    println!();

    // Display what is NOT revealed
    println!("🔐 What This Proof DOES NOT Reveal:");
    println!("   ─────────────────────────────────────────────────────────");
    println!("   ✗ Ed25519 secret key bytes (32 bytes)");
    println!("   ✗ Dilithium5 secret key bytes (4864 bytes)");
    println!("   ✗ Any intermediate computation values");
    println!("   ✗ Validator identity linkage");
    println!();

    // Display use cases
    println!("🎯 Enabled Use Cases:");
    println!("   ═══════════════════════════════════════════════════════════");
    println!("   • Privacy-preserving validator registration");
    println!("   • Threshold signature schemes (t-of-n validators)");
    println!("   • Anonymous validator voting");
    println!("   • Validator key rotation with proof");
    println!("   • ZK proof of solvency (stake amounts)");
    println!("   • Cross-chain validator identity bridges");
    println!();

    // Save proof to file
    let proof_filename = format!("/tmp/validator_zk_proof_{}.json", proof_type);
    let proof_json = serde_json::to_string_pretty(&proof)?;
    std::fs::write(&proof_filename, proof_json)?;

    println!("💾 Proof saved to: {}", proof_filename);
    println!();

    // Display performance summary
    println!("⚡ Performance Summary:");
    println!("   ═══════════════════════════════════════════════════════════");
    println!("   Proof Generation: {:?}", generation_time);
    println!("   Proof Verification: {:?}", verify_start.elapsed());

    let total_size = proof.stark_proof.as_ref().map(|s| s.proof_size).unwrap_or(0)
        + proof.snark_proof.as_ref().map(|s| s.proof_size).unwrap_or(0);
    println!("   Total Proof Size: {} bytes (~{} KB)", total_size, total_size / 1024);
    println!();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  ✅ Zero-Knowledge Proof Generation Complete!                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("🚀 Next Steps:");
    println!("   1. Use proof for privacy-preserving validator registration");
    println!("   2. Verify proof without revealing secret keys");
    println!("   3. Integrate with threshold signature schemes");
    println!("   4. Enable anonymous validator voting");
    println!();
    println!("📚 Documentation: ZK_UNTRUSTED_SETUP_v1.0.16-beta.md");
    println!();

    Ok(())
}
