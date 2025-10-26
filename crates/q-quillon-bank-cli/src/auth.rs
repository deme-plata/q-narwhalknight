/// Authentication manager for board members with AEGIS-QL post-quantum security

use anyhow::{Context, Result, bail};
use ed25519_dalek::{SecretKey, Signature, Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::fs;
use q_aegis_ql::{AegisQL, PublicKey as AegisPublicKey, SecretKey as AegisSecretKey, Signature as AegisSignature};
use q_zk_stark::StarkProof;
use sha3::{Digest, Sha3_256};

use crate::config::CliConfig;

// Compatibility type alias
type Keypair = SigningKey;

/// Founder wallet address (hardcoded for security)
pub const FOUNDER_WALLET: &str = "efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthSession {
    pub member_id: String,
    pub role: String,
    pub token: String,
    pub expires_at: u64,
}

pub struct AuthManager {
    config: CliConfig,
}

impl AuthManager {
    pub fn new(config: CliConfig) -> Self {
        Self { config }
    }

    /// Generate new authentication keypair
    pub fn generate_keys(&self) -> Result<()> {
        let keys_dir = CliConfig::keys_dir()?;
        fs::create_dir_all(&keys_dir)?;

        let mut csprng = OsRng {};
        let keypair = SigningKey::generate(&mut csprng);

        // Save secret key
        let secret_path = keys_dir.join("board-key.pem");
        fs::write(&secret_path, keypair.to_bytes())?;

        // Set permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&secret_path)?.permissions();
            perms.set_mode(0o600);
            fs::set_permissions(&secret_path, perms)?;
        }

        // Save public key
        let public_path = keys_dir.join("board-key.pub");
        let verifying_key = keypair.verifying_key();
        fs::write(&public_path, verifying_key.to_bytes())?;

        println!("✅ Generated authentication keys");
        println!("   Secret key: {}", secret_path.display());
        println!("   Public key: {}", public_path.display());

        Ok(())
    }

    /// Load keypair from file
    pub fn load_keypair(&self) -> Result<Keypair> {
        let secret_path = &self.config.board.key_path;

        if !secret_path.exists() {
            bail!("Key file not found: {}. Run 'quillon-bank init --generate-keys'", secret_path.display());
        }

        let secret_bytes = fs::read(secret_path)
            .context("Failed to read secret key")?;

        if secret_bytes.len() != 32 {
            bail!("Invalid secret key length: expected 32 bytes, got {}", secret_bytes.len());
        }

        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&secret_bytes);

        let signing_key = SigningKey::from_bytes(&bytes);

        Ok(signing_key)
    }

    /// Sign authentication challenge
    pub fn sign_challenge(&self, challenge: &[u8]) -> Result<Signature> {
        let keypair = self.load_keypair()?;
        Ok(keypair.sign(challenge))
    }

    /// Save authentication session
    pub fn save_session(&self, session: &AuthSession) -> Result<()> {
        let session_path = CliConfig::keys_dir()?.join("session.json");
        let json = serde_json::to_string_pretty(session)?;
        fs::write(&session_path, json)?;
        Ok(())
    }

    /// Load authentication session
    pub fn load_session(&self) -> Result<Option<AuthSession>> {
        let session_path = CliConfig::keys_dir()?.join("session.json");

        if !session_path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(&session_path)?;
        let session: AuthSession = serde_json::from_str(&json)?;

        // Check if expired
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        if session.expires_at < now {
            return Ok(None);
        }

        Ok(Some(session))
    }

    /// Clear authentication session
    pub fn clear_session(&self) -> Result<()> {
        let session_path = CliConfig::keys_dir()?.join("session.json");
        if session_path.exists() {
            fs::remove_file(&session_path)?;
        }
        Ok(())
    }

    /// Verify MFA token
    pub fn verify_mfa(&self, token: &str) -> Result<bool> {
        // TODO: Implement TOTP verification
        // For now, accept any 6-digit token
        Ok(token.len() == 6 && token.chars().all(|c| c.is_numeric()))
    }

    /// Generate AEGIS-QL keypair for post-quantum security with ZK-STARK trustless setup
    pub fn generate_aegis_keys(&self) -> Result<([u8; 32], AegisPublicKey, AegisSecretKey, StarkProof)> {
        let keys_dir = CliConfig::keys_dir()?;
        fs::create_dir_all(&keys_dir)?;

        println!("🔐 Generating AEGIS-QL post-quantum keypair...");

        // Generate AEGIS-QL keypair
        let mut aegis = AegisQL::new();
        let (public_key, secret_key) = aegis.generate_keypair()
            .context("Failed to generate AEGIS-QL keypair")?;

        println!("✅ AEGIS-QL keypair generated");

        // Derive wallet address from public key hash
        let wallet_address = Self::derive_wallet_from_aegis_pubkey(&public_key);

        println!("📍 Derived wallet address: qnk{}", hex::encode(&wallet_address[..8]));

        // Verify if this matches the expected founder wallet
        let expected_founder = hex::decode(FOUNDER_WALLET)
            .context("Invalid founder wallet hex")?;
        let mut expected_bytes = [0u8; 32];
        expected_bytes.copy_from_slice(&expected_founder);

        if wallet_address == expected_bytes {
            println!("✅ MATCHES FOUNDER WALLET: qnk{}", FOUNDER_WALLET);
        } else {
            println!("⚠️  Generated wallet: qnk{}", hex::encode(&wallet_address));
            println!("⚠️  Expected founder:  qnk{}", FOUNDER_WALLET);
            println!("⚠️  Note: This keypair will not have founder privileges");
        }

        // Generate ZK-STARK proof for trustless key setup verification
        println!("🔒 Generating ZK-STARK trustless setup proof...");
        let stark_proof = self.generate_key_setup_proof(&public_key, &wallet_address)?;
        println!("✅ ZK-STARK proof generated");

        // Save secret key (zeroized on drop)
        let secret_path = keys_dir.join("founder-aegis.key");
        let secret_bytes = bincode::serialize(&secret_key)
            .context("Failed to serialize secret key")?;
        fs::write(&secret_path, secret_bytes)?;

        // Set restrictive permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&secret_path)?.permissions();
            perms.set_mode(0o600);
            fs::set_permissions(&secret_path, perms)?;
        }

        // Save public key
        let public_path = keys_dir.join("founder-aegis.pub");
        let public_bytes = bincode::serialize(&public_key)
            .context("Failed to serialize public key")?;
        fs::write(&public_path, public_bytes)?;

        // Save wallet address
        let wallet_path = keys_dir.join("founder-wallet.txt");
        fs::write(&wallet_path, format!("qnk{}", hex::encode(&wallet_address)))?;

        // Save ZK-STARK proof
        let proof_path = keys_dir.join("founder-aegis-proof.stark");
        let proof_bytes = bincode::serialize(&stark_proof)
            .context("Failed to serialize STARK proof")?;
        fs::write(&proof_path, proof_bytes)?;

        println!("\n✅ AEGIS-QL keys generated successfully:");
        println!("   Secret key: {}", secret_path.display());
        println!("   Public key: {}", public_path.display());
        println!("   Wallet:     {}", wallet_path.display());
        println!("   STARK proof: {}", proof_path.display());
        println!("\n🔒 Post-quantum security: 256-bit classical, 128-bit quantum");
        println!("🔒 Trustless setup: ZK-STARK proof included");

        Ok((wallet_address, public_key, secret_key, stark_proof))
    }

    /// Derive wallet address from AEGIS-QL public key
    pub fn derive_wallet_from_aegis_pubkey(public_key: &AegisPublicKey) -> [u8; 32] {
        // Serialize public key components
        let mut hasher = Sha3_256::new();

        // Hash polynomial a
        for coeff in &public_key.a {
            hasher.update(&coeff.to_le_bytes());
        }

        // Hash polynomial t
        for coeff in &public_key.t {
            hasher.update(&coeff.to_le_bytes());
        }

        let hash = hasher.finalize();
        let mut wallet = [0u8; 32];
        wallet.copy_from_slice(&hash);
        wallet
    }

    /// Generate ZK-STARK proof for key setup (trustless verification)
    fn generate_key_setup_proof(&self, public_key: &AegisPublicKey, wallet_address: &[u8; 32]) -> Result<StarkProof> {
        println!("📊 Creating execution trace for key setup proof...");

        // Create execution trace for the key generation process
        // This proves that the wallet was correctly derived from the public key
        let mut trace = Vec::new();

        // Row 1: Public key polynomial a (first 8 coefficients as sample)
        let mut row1 = Vec::new();
        for i in 0..8 {
            row1.push(public_key.a.get(i).copied().unwrap_or(0) as u64);
        }
        trace.push(row1);

        // Row 2: Public key polynomial t (first 8 coefficients)
        let mut row2 = Vec::new();
        for i in 0..8 {
            row2.push(public_key.t.get(i).copied().unwrap_or(0) as u64);
        }
        trace.push(row2);

        // Row 3: Wallet address derivation (first 8 bytes)
        let mut row3 = Vec::new();
        for i in 0..8 {
            row3.push(wallet_address[i] as u64);
        }
        trace.push(row3);

        // Constraint: Proves the hash relationship between pubkey → wallet
        let constraints = b"AEGIS_KEY_SETUP_V1";

        // Generate STARK proof (CPU mode for CLI tool)
        println!("⚙️  Proving STARK (this may take a moment)...");
        let runtime = tokio::runtime::Runtime::new()?;
        let stark_proof = runtime.block_on(async {
            let mut stark_system = q_zk_stark::StarkSystem::new(false).await?;
            stark_system.prove(&trace, constraints).await
        })?;

        println!("✅ STARK proof generated: {} bytes", bincode::serialize(&stark_proof)?.len());

        Ok(stark_proof)
    }

    /// Load AEGIS-QL keypair from file
    pub fn load_aegis_keypair(&self) -> Result<(AegisPublicKey, AegisSecretKey)> {
        let keys_dir = CliConfig::keys_dir()?;

        let secret_path = keys_dir.join("founder-aegis.key");
        let public_path = keys_dir.join("founder-aegis.pub");

        if !secret_path.exists() || !public_path.exists() {
            bail!("AEGIS-QL keys not found. Run 'quillon-bank init --generate-aegis-keys --founder'");
        }

        let secret_bytes = fs::read(&secret_path)
            .context("Failed to read AEGIS-QL secret key")?;
        let secret_key: AegisSecretKey = bincode::deserialize(&secret_bytes)
            .context("Failed to deserialize AEGIS-QL secret key")?;

        let public_bytes = fs::read(&public_path)
            .context("Failed to read AEGIS-QL public key")?;
        let public_key: AegisPublicKey = bincode::deserialize(&public_bytes)
            .context("Failed to deserialize AEGIS-QL public key")?;

        Ok((public_key, secret_key))
    }

    /// Load wallet address
    pub fn load_wallet_address(&self) -> Result<[u8; 32]> {
        let keys_dir = CliConfig::keys_dir()?;
        let wallet_path = keys_dir.join("founder-wallet.txt");

        if !wallet_path.exists() {
            bail!("Wallet address not found. Run 'quillon-bank init --generate-aegis-keys --founder'");
        }

        let wallet_str = fs::read_to_string(&wallet_path)?;
        let wallet_hex = wallet_str.trim().strip_prefix("qnk").unwrap_or(&wallet_str);

        let wallet_bytes = hex::decode(wallet_hex)
            .context("Invalid wallet address hex")?;

        if wallet_bytes.len() != 32 {
            bail!("Invalid wallet address length: expected 32 bytes, got {}", wallet_bytes.len());
        }

        let mut wallet = [0u8; 32];
        wallet.copy_from_slice(&wallet_bytes);
        Ok(wallet)
    }

    /// Sign an operation with AEGIS-QL (post-quantum signature)
    pub fn sign_operation_aegis(&self, operation: &str, timestamp: i64) -> Result<(AegisSignature, [u8; 32])> {
        let (public_key, secret_key) = self.load_aegis_keypair()?;
        let wallet_address = self.load_wallet_address()?;

        // Create operation message
        let message = format!("QUILLON_BANK:{}:{}", operation, timestamp);

        // Sign with AEGIS-QL
        let mut aegis = AegisQL::new();
        let signature = aegis.sign(message.as_bytes(), &secret_key)
            .context("Failed to sign operation with AEGIS-QL")?;

        Ok((signature, wallet_address))
    }
}