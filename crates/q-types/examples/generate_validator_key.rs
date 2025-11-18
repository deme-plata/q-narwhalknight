//! Generate a validator keypair for PQC block signing
//!
//! Usage:
//!   cargo run --package q-types --example generate_validator_key [OUTPUT_PATH]
//!
//! Example:
//!   cargo run --package q-types --example generate_validator_key /tmp/validator.json

use q_types::pqc_keys::ValidatorKeypair;
use pqcrypto_traits::sign::PublicKey;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let output_path = if args.len() > 1 {
        &args[1]
    } else {
        "/tmp/validator.json"
    };

    println!("🔐 Generating validator keypair...\n");

    let keypair = ValidatorKeypair::generate();

    println!("Generated keypair:");
    println!("  Node ID: {}", hex::encode(&keypair.node_id));
    println!("  Ed25519 public key: {} bytes", keypair.ed25519_verifying.to_bytes().len());
    println!("  Dilithium5 public key: {} bytes", keypair.dilithium5_public.as_bytes().len());
    println!("  Preferred phase: {:?}\n", keypair.preferred_phase);

    keypair.save_to_file(output_path)
        .expect("Failed to save keypair");

    println!("✅ Keypair saved to: {}\n", output_path);
    println!("To use this key with q-api-server:");
    println!("  q-api-server --validator-key {}", output_path);
}
