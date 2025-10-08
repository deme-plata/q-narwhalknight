/// Authentication manager for board members

use anyhow::{Context, Result, bail};
use ed25519_dalek::{SecretKey, Signature, Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::fs;

use crate::config::CliConfig;

// Compatibility type alias
type Keypair = SigningKey;

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
}