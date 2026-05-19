//! Ed25519 X-Wallet-Auth signer for the Quillon DEX swap API.
//!
//! Matches `gui/quantum-wallet/src/services/walletAuth.ts:170-237` and the server-side
//! verifier in `crates/q-api-server/src/wallet_auth.rs:209-237`.
//!
//! Derivation:
//!   private_key = SHA3-256(seed_string_utf8_bytes)         // 32 bytes
//!   public_key  = Ed25519::derive(private_key)             // 32 bytes
//!   address     = "qnk" + hex(public_key)
//!
//! X-Wallet-Auth challenge:
//!   challenge = SHA3-256(address_bytes || timestamp_i64_le || path_utf8)
//!   signature = Ed25519::sign(private_key, challenge)      // 64 bytes
//!
//! Header JSON:
//!   {"address":"qnk<64hex>","timestamp":<unix_seconds>,"scheme":"Ed25519","signature":"<128hex>"}

use anyhow::{anyhow, Context, Result};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use serde::Serialize;
use sha3::{Digest, Sha3_256};

#[derive(Clone)]
pub struct AgentWallet {
    sk: SigningKey,
    address_hex: [u8; 32],
}

#[derive(Serialize)]
struct AuthHeader<'a> {
    address: &'a str,
    timestamp: i64,
    scheme: &'static str,
    signature: String,
}

impl AgentWallet {
    /// Build a wallet from a seed string (the same string used in walletAuth.ts mnemonic path).
    pub fn from_seed_string(seed: &str) -> Self {
        let mut h = Sha3_256::new();
        h.update(seed.as_bytes());
        let priv_bytes: [u8; 32] = h.finalize().into();
        let sk = SigningKey::from_bytes(&priv_bytes);
        let pk: VerifyingKey = sk.verifying_key();
        AgentWallet {
            sk,
            address_hex: pk.to_bytes(),
        }
    }

    /// Build a wallet from one of three sources in order:
    ///   1. env var `TRADING_SEED`            — literal seed string
    ///   2. env var `TRADING_SEED_FILE`       — path to a file containing the seed
    ///   3. file `~/.claude/quillon-agent-seed` (default)
    pub fn from_env_or_default() -> Result<Self> {
        if let Ok(v) = std::env::var("TRADING_SEED") {
            return Ok(Self::from_seed_string(v.trim()));
        }
        let path = match std::env::var("TRADING_SEED_FILE") {
            Ok(p) => p,
            Err(_) => {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
                format!("{home}/.claude/quillon-agent-seed")
            }
        };
        let seed = std::fs::read_to_string(&path)
            .with_context(|| format!("read seed file {path}"))?;
        let seed = seed.trim();
        if seed.is_empty() {
            return Err(anyhow!("seed file {path} is empty"));
        }
        Ok(Self::from_seed_string(seed))
    }

    pub fn address(&self) -> String {
        format!("qnk{}", hex::encode(self.address_hex))
    }

    pub fn pubkey_bytes(&self) -> [u8; 32] {
        self.address_hex
    }

    /// Produce the JSON value for the `X-Wallet-Auth` header for a request to `path`.
    pub fn sign_request(&self, path: &str) -> Result<String> {
        let timestamp = chrono::Utc::now().timestamp();
        let mut h = Sha3_256::new();
        h.update(self.address_hex);
        h.update(timestamp.to_le_bytes());
        h.update(path.as_bytes());
        let challenge: [u8; 32] = h.finalize().into();
        let sig = self.sk.sign(&challenge);
        let address = self.address();
        let header = AuthHeader {
            address: &address,
            timestamp,
            scheme: "Ed25519",
            signature: hex::encode(sig.to_bytes()),
        };
        Ok(serde_json::to_string(&header)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: known agent wallet seed derives to known address.
    #[test]
    fn derive_matches_known_address() {
        let seed = "9c83a476b9c1ba558429058ffb2297dfa0cb0284f96c48c661fab6f93cd1ee41";
        let w = AgentWallet::from_seed_string(seed);
        assert_eq!(
            w.address(),
            "qnk7154929a6aa0c118791373ea21004aca6e494e6e031c36f780cd5acedf031ccb"
        );
    }
}
