/// Generate Validator Keypair with Encrypted Storage (v1.0.16-beta)
///
/// This tool generates a new validator keypair for PQC signing and saves it
/// with AES-256-GCM encryption + Argon2 password derivation.
///
/// Usage:
///     cargo run --package q-types --example generate_encrypted_validator_key <output_path>
///
/// Example:
///     cargo run --package q-types --example generate_encrypted_validator_key /etc/q-narwhalknight/validator_encrypted.json
///
/// Security:
/// - Uses AES-256-GCM for authenticated encryption
/// - Derives encryption key from password using Argon2id
/// - Zeroizes sensitive data after use
/// - Stores Argon2 hash for password verification
///
/// The generated file will be encrypted and require a password to load.

use q_types::ValidatorKeypair;
use pqcrypto_traits::sign::PublicKey;
use std::env;
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <output_path>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} /etc/q-narwhalknight/validator_encrypted.json", args[0]);
        std::process::exit(1);
    }

    let output_path = &args[1];

    println!("🔐 ════════════════════════════════════════════════════════");
    println!("🔐 Q-NarwhalKnight Encrypted Validator Key Generator");
    println!("🔐 v1.0.16-beta - AES-256-GCM + Argon2");
    println!("🔐 ════════════════════════════════════════════════════════");
    println!();

    // Prompt for password
    print!("🔑 Enter password for key encryption: ");
    io::stdout().flush()?;
    let password = rpassword::read_password()?;

    if password.len() < 12 {
        eprintln!("❌ Password must be at least 12 characters long");
        std::process::exit(1);
    }

    print!("🔑 Confirm password: ");
    io::stdout().flush()?;
    let password_confirm = rpassword::read_password()?;

    if password != password_confirm {
        eprintln!("❌ Passwords do not match");
        std::process::exit(1);
    }

    println!();
    println!("🔐 Generating validator keypair...");

    // Generate keypair
    let keypair = ValidatorKeypair::generate();

    println!();
    println!("✅ Generated keypair:");
    println!("   Node ID: {}", hex::encode(&keypair.node_id));
    println!("   Ed25519 public key: {} bytes", keypair.ed25519_verifying.as_bytes().len());
    println!("   Dilithium5 public key: {} bytes", keypair.dilithium5_public.as_bytes().len());
    println!("   Preferred phase: {:?}", keypair.preferred_phase);
    println!();

    println!("🔐 Encrypting with AES-256-GCM + Argon2...");

    // Save with encryption
    keypair.save_encrypted(output_path, &password)?;

    println!();
    println!("✅ ════════════════════════════════════════════════════════");
    println!("✅ Encrypted keypair saved to: {}", output_path);
    println!("✅ ════════════════════════════════════════════════════════");
    println!();
    println!("🔐 Security Features:");
    println!("   • AES-256-GCM authenticated encryption");
    println!("   • Argon2id password-based key derivation");
    println!("   • Automatic key zeroization after use");
    println!("   • Password verification on load");
    println!();
    println!("⚠️  IMPORTANT SECURITY NOTES:");
    println!("   1. Store this password securely (password manager recommended)");
    println!("   2. Create encrypted backups of this file");
    println!("   3. Set file permissions: chmod 600 {}", output_path);
    println!("   4. Never commit this file to version control");
    println!("   5. If you lose the password, you cannot recover the key!");
    println!();
    println!("📖 To use this key with q-api-server:");
    println!("   export Q_VALIDATOR_PASSWORD=\"your-password\"");
    println!("   q-api-server --validator-key-encrypted {}", output_path);
    println!();
    println!("🔍 File contents are ENCRYPTED - plaintext keys are NOT visible");
    println!();

    Ok(())
}
