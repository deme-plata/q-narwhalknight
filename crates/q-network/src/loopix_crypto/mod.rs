/// Loopix Cryptographic Components
///
/// Provides cryptographic primitives for the Loopix mix network:
/// - Fixed-size cell padding for traffic analysis resistance
/// - Secure random number generation
/// - Message encryption/decryption utilities

use ring::rand::{SecureRandom, SystemRandom};
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use anyhow::Result;

pub mod padding;

pub use padding::{CELL_SIZE, pad_to_cell, unpad_from_cell};

/// Secure key type for encryption
pub type SecureKey = [u8; 32];

/// Generate a secure random key
pub fn keygen() -> SecureKey {
    let rng = SystemRandom::new();
    let mut key = [0u8; 32];
    rng.fill(&mut key).expect("Failed to generate random key");
    key
}

/// Encrypt data with layered encryption for mix network
pub fn encrypt_layered(data: Vec<u8>, keys: &[SecureKey]) -> Result<Vec<u8>> {
    let mut encrypted = data;
    
    for key in keys.iter().rev() {
        encrypted = encrypt_single_layer(encrypted, key)?;
    }
    
    Ok(encrypted)
}

/// Decrypt single layer of encryption
pub fn decrypt_single_layer(data: Vec<u8>, key: &SecureKey) -> Result<Vec<u8>> {
    if data.len() < 12 {
        return Err(anyhow::anyhow!("Data too short for decryption"));
    }

    let nonce_bytes = &data[0..12];
    let ciphertext = &data[12..];

    let unbound_key = UnboundKey::new(&AES_256_GCM, key)
        .map_err(|_| anyhow::anyhow!("Invalid key"))?;
    let key = LessSafeKey::new(unbound_key);

    let nonce = Nonce::try_assume_unique_for_key(nonce_bytes)
        .map_err(|_| anyhow::anyhow!("Invalid nonce"))?;

    let mut mutable_ciphertext = ciphertext.to_vec();
    let plaintext = key.open_in_place(nonce, Aad::empty(), &mut mutable_ciphertext)
        .map_err(|_| anyhow::anyhow!("Decryption failed"))?;

    Ok(plaintext.to_vec())
}

/// Decrypt one onion layer (alias for backward compatibility)
pub fn decrypt_one_onion_layer(data: Vec<u8>, key: &SecureKey) -> Result<Vec<u8>> {
    decrypt_single_layer(data, key)
}

/// Encrypt single layer
fn encrypt_single_layer(data: Vec<u8>, key: &SecureKey) -> Result<Vec<u8>> {
    let rng = SystemRandom::new();
    let mut nonce_bytes = [0u8; 12];
    rng.fill(&mut nonce_bytes)
        .map_err(|_| anyhow::anyhow!("Failed to generate nonce"))?;
    
    let unbound_key = UnboundKey::new(&AES_256_GCM, key)
        .map_err(|_| anyhow::anyhow!("Invalid key"))?;
    let key = LessSafeKey::new(unbound_key);
    
    let nonce = Nonce::try_assume_unique_for_key(&nonce_bytes)
        .map_err(|_| anyhow::anyhow!("Invalid nonce"))?;
    
    let mut mutable_data = data;
    key.seal_in_place_append_tag(nonce, Aad::empty(), &mut mutable_data)
        .map_err(|_| anyhow::anyhow!("Encryption failed"))?;
    
    let mut result = nonce_bytes.to_vec();
    result.extend_from_slice(&mutable_data);
    
    Ok(result)
}