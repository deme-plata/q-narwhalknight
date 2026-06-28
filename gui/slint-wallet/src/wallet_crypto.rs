//! Password-based mnemonic encryption — BYTE-IDENTICAL to the quillon.xyz web
//! wallet (`services/walletAuth.ts::encryptPrivateKey`), so a wallet protected
//! on the website can be unlocked here and vice-versa.
//!
//! Scheme: PBKDF2-HMAC-SHA256(password, 16-byte salt, 100_000 iters) -> 32-byte key;
//! AES-256-GCM(12-byte IV) over the mnemonic UTF-8 bytes (WebCrypto appends the
//! 16-byte tag to the ciphertext, which is exactly what the `aes-gcm` crate does);
//! serialized as JSON `{ "salt":[u8], "iv":[u8], "data":[u8] }`.

use aes_gcm::{aead::{Aead, KeyInit}, Aes256Gcm, Nonce};
use anyhow::{anyhow, Result};
use pbkdf2::pbkdf2_hmac;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::Sha256;

const PBKDF2_ITERS: u32 = 100_000; // MUST match walletAuth.ts

#[derive(Serialize, Deserialize)]
struct EncBlob {
    salt: Vec<u8>,
    iv: Vec<u8>,
    data: Vec<u8>,
}

fn derive_key(password: &str, salt: &[u8]) -> [u8; 32] {
    let mut key = [0u8; 32];
    pbkdf2_hmac::<Sha256>(password.as_bytes(), salt, PBKDF2_ITERS, &mut key);
    key
}

/// Encrypt a mnemonic with a password. Output JSON is interchangeable with the website.
pub fn encrypt_mnemonic(mnemonic: &str, password: &str) -> Result<String> {
    let mut salt = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut salt);
    let mut iv = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut iv);

    let key = derive_key(password, &salt);
    let cipher = Aes256Gcm::new_from_slice(&key).map_err(|e| anyhow!("key init: {}", e))?;
    let data = cipher
        .encrypt(Nonce::from_slice(&iv), mnemonic.as_bytes())
        .map_err(|e| anyhow!("encrypt: {}", e))?;

    Ok(serde_json::to_string(&EncBlob { salt: salt.to_vec(), iv: iv.to_vec(), data })?)
}

/// Decrypt a website- or slint-encrypted mnemonic blob with the password.
pub fn decrypt_mnemonic(blob_json: &str, password: &str) -> Result<String> {
    let blob: EncBlob = serde_json::from_str(blob_json)
        .map_err(|_| anyhow!("Not an encrypted-mnemonic blob"))?;
    let key = derive_key(password, &blob.salt);
    let cipher = Aes256Gcm::new_from_slice(&key).map_err(|e| anyhow!("key init: {}", e))?;
    let plain = cipher
        .decrypt(Nonce::from_slice(&blob.iv), blob.data.as_ref())
        .map_err(|_| anyhow!("Incorrect password or corrupted data"))?;
    Ok(String::from_utf8(plain)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let m = "test test test test test test test test test test test junk";
        let blob = encrypt_mnemonic(m, "hunter2").unwrap();
        assert_eq!(decrypt_mnemonic(&blob, "hunter2").unwrap(), m);
        assert!(decrypt_mnemonic(&blob, "wrong").is_err());
    }
}
