use anyhow::{anyhow, Result};
use bip39::Mnemonic;
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use sha3::{Digest, Sha3_256};

/// Wallet holds the Ed25519 keypair and derived address.
pub struct Wallet {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    address: String,
    address_bytes: [u8; 32],
}

impl Wallet {
    /// Create a new wallet with a random 24-word mnemonic.
    /// Returns (wallet, mnemonic_phrase).
    pub fn create() -> Result<(Self, String)> {
        use rand::RngCore;
        let mut entropy = [0u8; 32]; // 256 bits = 24-word mnemonic
        rand::thread_rng().fill_bytes(&mut entropy);
        let mnemonic = Mnemonic::from_entropy_in(bip39::Language::English, &entropy)
            .map_err(|e| anyhow!("Failed to generate mnemonic: {}", e))?;
        let phrase = mnemonic.to_string();
        let wallet = Self::from_mnemonic(&phrase)?;
        Ok((wallet, phrase))
    }

    /// Import a wallet from an existing mnemonic phrase.
    /// Key derivation: SHA3-256(mnemonic_string) → Ed25519 private key
    /// This matches the server and web frontend derivation exactly.
    pub fn from_mnemonic(phrase: &str) -> Result<Self> {
        // Validate mnemonic (12 or 24 words, valid BIP39)
        let _mnemonic = Mnemonic::parse_in(bip39::Language::English, phrase)
            .map_err(|e| anyhow!("Invalid mnemonic: {}", e))?;

        // Derive Ed25519 key from mnemonic STRING via SHA3-256
        // MUST match server: SHA3-256(mnemonic_text) → private key bytes
        let mut hasher = Sha3_256::new();
        hasher.update(phrase.as_bytes());
        let seed: [u8; 32] = hasher.finalize().into();

        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();

        let pubkey_bytes = verifying_key.to_bytes();
        let address = format!("qnk{}", hex::encode(pubkey_bytes));
        let address_bytes = pubkey_bytes;

        Ok(Self {
            signing_key,
            verifying_key,
            address,
            address_bytes,
        })
    }

    /// Get the wallet address (qnk + hex pubkey).
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Get the raw 32-byte address (public key bytes).
    pub fn address_bytes(&self) -> &[u8; 32] {
        &self.address_bytes
    }

    /// Get the public key bytes.
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }

    /// Sign arbitrary data with Ed25519.
    pub fn sign(&self, data: &[u8]) -> Vec<u8> {
        let sig = self.signing_key.sign(data);
        sig.to_bytes().to_vec()
    }

    /// Generate the X-Wallet-Auth header value for an API request.
    ///
    /// Challenge = SHA3-256(address_bytes || timestamp_le_8 || path_utf8)
    /// Signature = Ed25519(challenge)
    pub fn auth_header(&self, path: &str) -> String {
        let timestamp = chrono::Utc::now().timestamp();

        // Build challenge: address_bytes || timestamp_le || path_utf8
        let ts_bytes = (timestamp as u64).to_le_bytes();
        let path_bytes = path.as_bytes();

        let mut challenge_input =
            Vec::with_capacity(32 + 8 + path_bytes.len());
        challenge_input.extend_from_slice(&self.address_bytes);
        challenge_input.extend_from_slice(&ts_bytes);
        challenge_input.extend_from_slice(path_bytes);

        let mut hasher = Sha3_256::new();
        hasher.update(&challenge_input);
        let challenge: [u8; 32] = hasher.finalize().into();

        // Sign the challenge
        let signature = self.sign(&challenge);

        // Build JSON header
        serde_json::json!({
            "address": self.address,
            "timestamp": timestamp,
            "scheme": "Ed25519",
            "signature": hex::encode(&signature),
        })
        .to_string()
    }

    /// Sign a transaction for submission.
    /// Returns the hex-encoded Ed25519 signature of the transaction hash.
    pub fn sign_transaction(&self, tx_hash: &[u8]) -> String {
        let sig = self.sign(tx_hash);
        hex::encode(&sig)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_wallet() {
        let (wallet, mnemonic) = Wallet::create().unwrap();
        let words: Vec<&str> = mnemonic.split_whitespace().collect();
        assert!(words.len() == 12 || words.len() == 24);
        assert!(wallet.address().starts_with("qnk"));
        assert_eq!(wallet.address().len(), 3 + 64); // "qnk" + 64 hex chars
    }

    #[test]
    fn test_import_wallet_deterministic() {
        let (_, mnemonic) = Wallet::create().unwrap();
        let w1 = Wallet::from_mnemonic(&mnemonic).unwrap();
        let w2 = Wallet::from_mnemonic(&mnemonic).unwrap();
        assert_eq!(w1.address(), w2.address());
    }

    #[test]
    fn test_invalid_mnemonic() {
        let result = Wallet::from_mnemonic("not a valid mnemonic phrase");
        assert!(result.is_err());
    }

    #[test]
    fn test_auth_header_format() {
        let (wallet, _) = Wallet::create().unwrap();
        let header = wallet.auth_header("/api/v1/wallet/balance");
        let parsed: serde_json::Value = serde_json::from_str(&header).unwrap();
        assert_eq!(parsed["scheme"], "Ed25519");
        assert!(parsed["address"].as_str().unwrap().starts_with("qnk"));
        assert!(parsed["signature"].as_str().unwrap().len() == 128); // 64 bytes hex
        assert!(parsed["timestamp"].as_i64().unwrap() > 0);
    }
}
