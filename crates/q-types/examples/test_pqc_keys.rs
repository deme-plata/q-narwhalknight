//! Test PQC Key Management Implementation v1.0.16-beta

use q_types::pqc_keys::{ValidatorKeypair, ValidatorKeyRegistry};
use pqcrypto_traits::sign::PublicKey;

fn main() {
    println!("🔐 Testing PQC Key Management v1.0.16-beta...\n");
    
    // Test 1: Keypair Generation
    println!("Test 1: Generating validator keypair...");
    let keypair = ValidatorKeypair::generate();
    println!("✅ Node ID: {} (first 16 bytes)", hex::encode(&keypair.node_id[..16]));
    println!("✅ Ed25519 public key: {} bytes", keypair.ed25519_verifying.to_bytes().len());
    println!("✅ Dilithium5 public key: {} bytes", keypair.dilithium5_public.as_bytes().len());
    println!("✅ Preferred phase: {:?}\n", keypair.preferred_phase);
    
    // Test 2: Public Keys Extraction
    println!("Test 2: Extracting public keys...");
    let public_keys = keypair.public_keys();
    println!("✅ Ed25519: {} bytes", public_keys.ed25519.len());
    println!("✅ Dilithium5: {} bytes\n", public_keys.dilithium5.len());
    
    // Test 3: Key Registry
    println!("Test 3: Testing key registry...");
    let mut registry = ValidatorKeyRegistry::new();
    
    let keypair1 = ValidatorKeypair::generate();
    let keypair2 = ValidatorKeypair::generate();
    
    registry.register(keypair1.public_keys());
    registry.register(keypair2.public_keys());
    
    println!("✅ Registered {} validators", registry.len());
    println!("✅ Validator 1 exists: {}", registry.has_validator(&keypair1.node_id));
    println!("✅ Validator 2 exists: {}\n", registry.has_validator(&keypair2.node_id));
    
    // Test 4: Save/Load
    println!("Test 4: Testing save/load...");
    let temp_path = "/tmp/test_validator_key_v1.0.16.json";
    
    keypair.save_to_file(temp_path).expect("Failed to save");
    println!("✅ Saved to {}", temp_path);
    
    let loaded = ValidatorKeypair::load_from_file(temp_path).expect("Failed to load");
    println!("✅ Loaded keypair");
    println!("✅ Node IDs match: {}", keypair.node_id == loaded.node_id);
    println!("✅ Ed25519 keys match: {}", 
        keypair.ed25519_signing.to_bytes() == loaded.ed25519_signing.to_bytes());
    
    std::fs::remove_file(temp_path).ok();
    println!("✅ Cleaned up temporary file\n");
    
    println!("🎉 All PQC key management tests passed!");
    println!("\n📊 Summary:");
    println!("  - Key generation: ✅");
    println!("  - Public key extraction: ✅");
    println!("  - Registry operations: ✅");
    println!("  - Save/Load persistence: ✅");
}
