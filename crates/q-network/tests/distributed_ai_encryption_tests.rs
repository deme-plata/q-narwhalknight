//! Distributed AI Encryption Tests
//!
//! Tests for encryption/decryption of AI prompts and responses
//! in the distributed AI inference system.
//!
//! CRITICAL SCENARIOS TESTED:
//! 1. Prompt encryption roundtrip
//! 2. Ciphertext integrity verification
//! 3. Key derivation security
//! 4. Wrong key rejection
//! 5. Tampered ciphertext detection
//! 6. Empty input handling
//!
//! Run with: cargo test --package q-network --test distributed_ai_encryption_tests

use std::collections::HashMap;
use std::sync::Mutex;

// ============================================================================
// MOCK ENCRYPTION STRUCTURES
// ============================================================================

/// Encryption key for AI prompt protection
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct EncryptionKey {
    key_bytes: [u8; 32],
    key_id: [u8; 16],
}

impl EncryptionKey {
    /// Derive a key from a secret and context
    pub fn derive(secret: &[u8], context: &[u8]) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        secret.hash(&mut hasher);
        context.hash(&mut hasher);
        let h1 = hasher.finish();

        // Derive more bytes
        h1.hash(&mut hasher);
        let h2 = hasher.finish();
        h2.hash(&mut hasher);
        let h3 = hasher.finish();
        h3.hash(&mut hasher);
        let h4 = hasher.finish();

        let mut key_bytes = [0u8; 32];
        key_bytes[0..8].copy_from_slice(&h1.to_le_bytes());
        key_bytes[8..16].copy_from_slice(&h2.to_le_bytes());
        key_bytes[16..24].copy_from_slice(&h3.to_le_bytes());
        key_bytes[24..32].copy_from_slice(&h4.to_le_bytes());

        let mut key_id = [0u8; 16];
        key_id[0..8].copy_from_slice(&h1.to_le_bytes());
        key_id[8..16].copy_from_slice(&h2.to_le_bytes());

        Self { key_bytes, key_id }
    }

    pub fn key_id(&self) -> &[u8; 16] {
        &self.key_id
    }
}

/// Encrypted prompt data
#[derive(Clone, Debug)]
pub struct EncryptedPrompt {
    key_id: [u8; 16],
    nonce: [u8; 12],
    ciphertext: Vec<u8>,
    auth_tag: [u8; 16],
}

impl EncryptedPrompt {
    pub fn ciphertext_len(&self) -> usize {
        self.ciphertext.len()
    }

    pub fn key_id(&self) -> &[u8; 16] {
        &self.key_id
    }
}

/// AI prompt encryptor with AES-GCM-like mock
pub struct PromptEncryptor {
    keys: Mutex<HashMap<[u8; 16], EncryptionKey>>,
    nonce_counter: Mutex<u64>,
}

impl PromptEncryptor {
    pub fn new() -> Self {
        Self {
            keys: Mutex::new(HashMap::new()),
            nonce_counter: Mutex::new(0),
        }
    }

    /// Register an encryption key
    pub fn register_key(&self, key: EncryptionKey) {
        let mut keys = self.keys.lock().unwrap();
        keys.insert(*key.key_id(), key);
    }

    /// Get a unique nonce
    fn get_nonce(&self) -> [u8; 12] {
        let mut counter = self.nonce_counter.lock().unwrap();
        *counter += 1;
        let mut nonce = [0u8; 12];
        nonce[0..8].copy_from_slice(&counter.to_le_bytes());
        nonce
    }

    /// Compute authentication tag (simplified MAC)
    fn compute_auth_tag(key: &[u8; 32], nonce: &[u8; 12], ciphertext: &[u8]) -> [u8; 16] {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        nonce.hash(&mut hasher);
        ciphertext.hash(&mut hasher);
        let h1 = hasher.finish();
        h1.hash(&mut hasher);
        let h2 = hasher.finish();

        let mut tag = [0u8; 16];
        tag[0..8].copy_from_slice(&h1.to_le_bytes());
        tag[8..16].copy_from_slice(&h2.to_le_bytes());
        tag
    }

    /// Encrypt a prompt
    pub fn encrypt_prompt(&self, key: &EncryptionKey, prompt: &str) -> Result<EncryptedPrompt, String> {
        if prompt.is_empty() {
            return Err("EMPTY_PROMPT: Cannot encrypt empty prompt".to_string());
        }

        let nonce = self.get_nonce();
        let plaintext = prompt.as_bytes();

        // XOR-based encryption (simplified - real impl would use AES-GCM)
        let mut ciphertext = Vec::with_capacity(plaintext.len());
        for (i, byte) in plaintext.iter().enumerate() {
            let key_byte = key.key_bytes[i % 32];
            let nonce_byte = nonce[i % 12];
            ciphertext.push(byte ^ key_byte ^ nonce_byte);
        }

        let auth_tag = Self::compute_auth_tag(&key.key_bytes, &nonce, &ciphertext);

        Ok(EncryptedPrompt {
            key_id: *key.key_id(),
            nonce,
            ciphertext,
            auth_tag,
        })
    }

    /// Decrypt a prompt
    pub fn decrypt_prompt(&self, encrypted: &EncryptedPrompt) -> Result<String, String> {
        let keys = self.keys.lock().unwrap();
        let key = keys
            .get(&encrypted.key_id)
            .ok_or_else(|| "KEY_NOT_FOUND: No key registered for this key_id".to_string())?;

        // Verify authentication tag
        let expected_tag =
            Self::compute_auth_tag(&key.key_bytes, &encrypted.nonce, &encrypted.ciphertext);

        if expected_tag != encrypted.auth_tag {
            return Err("AUTH_TAG_MISMATCH: Ciphertext integrity check failed".to_string());
        }

        // Decrypt (reverse the XOR)
        let mut plaintext = Vec::with_capacity(encrypted.ciphertext.len());
        for (i, byte) in encrypted.ciphertext.iter().enumerate() {
            let key_byte = key.key_bytes[i % 32];
            let nonce_byte = encrypted.nonce[i % 12];
            plaintext.push(byte ^ key_byte ^ nonce_byte);
        }

        String::from_utf8(plaintext).map_err(|e| format!("INVALID_UTF8: {}", e))
    }

    /// Decrypt with a specific key (for testing wrong key scenarios)
    pub fn decrypt_with_key(
        &self,
        encrypted: &EncryptedPrompt,
        key: &EncryptionKey,
    ) -> Result<String, String> {
        // Verify authentication tag
        let expected_tag =
            Self::compute_auth_tag(&key.key_bytes, &encrypted.nonce, &encrypted.ciphertext);

        if expected_tag != encrypted.auth_tag {
            return Err("AUTH_TAG_MISMATCH: Ciphertext integrity check failed".to_string());
        }

        // Decrypt
        let mut plaintext = Vec::with_capacity(encrypted.ciphertext.len());
        for (i, byte) in encrypted.ciphertext.iter().enumerate() {
            let key_byte = key.key_bytes[i % 32];
            let nonce_byte = encrypted.nonce[i % 12];
            plaintext.push(byte ^ key_byte ^ nonce_byte);
        }

        String::from_utf8(plaintext).map_err(|e| format!("INVALID_UTF8: {}", e))
    }
}

/// AI Response encryption (responses from workers)
#[derive(Clone, Debug)]
pub struct EncryptedResponse {
    session_id: [u8; 16],
    nonce: [u8; 12],
    ciphertext: Vec<u8>,
    auth_tag: [u8; 16],
}

impl EncryptedResponse {
    pub fn ciphertext_len(&self) -> usize {
        self.ciphertext.len()
    }
}

/// Session-based encryption for AI responses
pub struct SessionEncryptor {
    sessions: Mutex<HashMap<[u8; 16], EncryptionKey>>,
    nonce_counter: Mutex<u64>,
}

impl SessionEncryptor {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            nonce_counter: Mutex::new(0),
        }
    }

    /// Create a new encryption session
    pub fn create_session(&self, session_id: [u8; 16], shared_secret: &[u8]) -> EncryptionKey {
        let key = EncryptionKey::derive(shared_secret, &session_id);
        let mut sessions = self.sessions.lock().unwrap();
        sessions.insert(session_id, key.clone());
        key
    }

    fn get_nonce(&self) -> [u8; 12] {
        let mut counter = self.nonce_counter.lock().unwrap();
        *counter += 1;
        let mut nonce = [0u8; 12];
        nonce[0..8].copy_from_slice(&counter.to_le_bytes());
        nonce
    }

    /// Encrypt an AI response
    pub fn encrypt_response(
        &self,
        session_id: &[u8; 16],
        response: &str,
    ) -> Result<EncryptedResponse, String> {
        let sessions = self.sessions.lock().unwrap();
        let key = sessions
            .get(session_id)
            .ok_or_else(|| "SESSION_NOT_FOUND: No session for this ID".to_string())?;

        if response.is_empty() {
            return Err("EMPTY_RESPONSE: Cannot encrypt empty response".to_string());
        }

        let nonce = self.get_nonce();
        let plaintext = response.as_bytes();

        // Encrypt
        let mut ciphertext = Vec::with_capacity(plaintext.len());
        for (i, byte) in plaintext.iter().enumerate() {
            let key_byte = key.key_bytes[i % 32];
            let nonce_byte = nonce[i % 12];
            ciphertext.push(byte ^ key_byte ^ nonce_byte);
        }

        let auth_tag = PromptEncryptor::compute_auth_tag(&key.key_bytes, &nonce, &ciphertext);

        Ok(EncryptedResponse {
            session_id: *session_id,
            nonce,
            ciphertext,
            auth_tag,
        })
    }

    /// Decrypt an AI response
    pub fn decrypt_response(&self, encrypted: &EncryptedResponse) -> Result<String, String> {
        let sessions = self.sessions.lock().unwrap();
        let key = sessions
            .get(&encrypted.session_id)
            .ok_or_else(|| "SESSION_NOT_FOUND: No session for this ID".to_string())?;

        // Verify authentication tag
        let expected_tag =
            PromptEncryptor::compute_auth_tag(&key.key_bytes, &encrypted.nonce, &encrypted.ciphertext);

        if expected_tag != encrypted.auth_tag {
            return Err("AUTH_TAG_MISMATCH: Response integrity check failed".to_string());
        }

        // Decrypt
        let mut plaintext = Vec::with_capacity(encrypted.ciphertext.len());
        for (i, byte) in encrypted.ciphertext.iter().enumerate() {
            let key_byte = key.key_bytes[i % 32];
            let nonce_byte = encrypted.nonce[i % 12];
            plaintext.push(byte ^ key_byte ^ nonce_byte);
        }

        String::from_utf8(plaintext).map_err(|e| format!("INVALID_UTF8: {}", e))
    }

    pub fn session_count(&self) -> usize {
        self.sessions.lock().unwrap().len()
    }
}

// ============================================================================
// PROMPT ENCRYPTION TESTS
// ============================================================================

/// Test basic encryption/decryption roundtrip
#[test]
fn test_encrypt_decrypt_roundtrip() {
    let encryptor = PromptEncryptor::new();
    let key = EncryptionKey::derive(b"secret_key", b"context");
    encryptor.register_key(key.clone());

    let original = "What is the capital of France?";
    let encrypted = encryptor.encrypt_prompt(&key, original).unwrap();
    let decrypted = encryptor.decrypt_prompt(&encrypted).unwrap();

    assert_eq!(original, decrypted);
}

/// Test encryption produces different ciphertext for same plaintext (due to nonce)
#[test]
fn test_encryption_randomness() {
    let encryptor = PromptEncryptor::new();
    let key = EncryptionKey::derive(b"secret_key", b"context");

    let prompt = "Test prompt";
    let encrypted1 = encryptor.encrypt_prompt(&key, prompt).unwrap();
    let encrypted2 = encryptor.encrypt_prompt(&key, prompt).unwrap();

    // Ciphertexts should be different due to different nonces
    assert_ne!(encrypted1.nonce, encrypted2.nonce);
    assert_ne!(encrypted1.ciphertext, encrypted2.ciphertext);
}

/// Test wrong key fails decryption
#[test]
fn test_wrong_key_fails_decryption() {
    let encryptor = PromptEncryptor::new();
    let key1 = EncryptionKey::derive(b"key1", b"context");
    let key2 = EncryptionKey::derive(b"key2", b"context");

    let prompt = "Secret prompt";
    let encrypted = encryptor.encrypt_prompt(&key1, prompt).unwrap();

    // Try to decrypt with wrong key
    let result = encryptor.decrypt_with_key(&encrypted, &key2);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("AUTH_TAG_MISMATCH"));
}

/// Test tampered ciphertext is detected
#[test]
fn test_tampered_ciphertext_detected() {
    let encryptor = PromptEncryptor::new();
    let key = EncryptionKey::derive(b"secret_key", b"context");
    encryptor.register_key(key.clone());

    let prompt = "Original prompt";
    let mut encrypted = encryptor.encrypt_prompt(&key, prompt).unwrap();

    // Tamper with ciphertext
    if !encrypted.ciphertext.is_empty() {
        encrypted.ciphertext[0] ^= 0xFF;
    }

    let result = encryptor.decrypt_prompt(&encrypted);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("AUTH_TAG_MISMATCH"));
}

/// Test empty prompt rejection
#[test]
fn test_empty_prompt_rejected() {
    let encryptor = PromptEncryptor::new();
    let key = EncryptionKey::derive(b"secret_key", b"context");

    let result = encryptor.encrypt_prompt(&key, "");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("EMPTY_PROMPT"));
}

/// Test key not found error
#[test]
fn test_key_not_found_error() {
    let encryptor = PromptEncryptor::new();
    let key = EncryptionKey::derive(b"secret_key", b"context");
    // Note: key NOT registered

    let encrypted = encryptor.encrypt_prompt(&key, "Test").unwrap();
    let result = encryptor.decrypt_prompt(&encrypted);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("KEY_NOT_FOUND"));
}

// ============================================================================
// KEY DERIVATION TESTS
// ============================================================================

/// Test key derivation produces different keys for different secrets
#[test]
fn test_key_derivation_different_secrets() {
    let key1 = EncryptionKey::derive(b"secret1", b"context");
    let key2 = EncryptionKey::derive(b"secret2", b"context");

    assert_ne!(key1.key_bytes, key2.key_bytes);
    assert_ne!(key1.key_id, key2.key_id);
}

/// Test key derivation produces different keys for different contexts
#[test]
fn test_key_derivation_different_contexts() {
    let key1 = EncryptionKey::derive(b"secret", b"context1");
    let key2 = EncryptionKey::derive(b"secret", b"context2");

    assert_ne!(key1.key_bytes, key2.key_bytes);
    assert_ne!(key1.key_id, key2.key_id);
}

/// Test key derivation is deterministic
#[test]
fn test_key_derivation_deterministic() {
    let key1 = EncryptionKey::derive(b"secret", b"context");
    let key2 = EncryptionKey::derive(b"secret", b"context");

    assert_eq!(key1.key_bytes, key2.key_bytes);
    assert_eq!(key1.key_id, key2.key_id);
}

// ============================================================================
// SESSION ENCRYPTION TESTS
// ============================================================================

/// Test session-based response encryption
#[test]
fn test_session_response_encryption() {
    let session_enc = SessionEncryptor::new();
    let session_id = [1u8; 16];
    session_enc.create_session(session_id, b"shared_secret");

    let response = "The capital of France is Paris.";
    let encrypted = session_enc.encrypt_response(&session_id, response).unwrap();
    let decrypted = session_enc.decrypt_response(&encrypted).unwrap();

    assert_eq!(response, decrypted);
}

/// Test session not found error
#[test]
fn test_session_not_found() {
    let session_enc = SessionEncryptor::new();
    let session_id = [1u8; 16];
    // Note: session NOT created

    let result = session_enc.encrypt_response(&session_id, "Test response");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("SESSION_NOT_FOUND"));
}

/// Test empty response rejection
#[test]
fn test_empty_response_rejected() {
    let session_enc = SessionEncryptor::new();
    let session_id = [1u8; 16];
    session_enc.create_session(session_id, b"secret");

    let result = session_enc.encrypt_response(&session_id, "");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("EMPTY_RESPONSE"));
}

/// Test multiple sessions are independent
#[test]
fn test_multiple_sessions_independent() {
    let session_enc = SessionEncryptor::new();

    let session1 = [1u8; 16];
    let session2 = [2u8; 16];

    session_enc.create_session(session1, b"secret1");
    session_enc.create_session(session2, b"secret2");

    assert_eq!(session_enc.session_count(), 2);

    // Encrypt with session1
    let encrypted1 = session_enc
        .encrypt_response(&session1, "Response 1")
        .unwrap();

    // Encrypt with session2
    let encrypted2 = session_enc
        .encrypt_response(&session2, "Response 2")
        .unwrap();

    // Each session can only decrypt its own responses
    assert_eq!(
        session_enc.decrypt_response(&encrypted1).unwrap(),
        "Response 1"
    );
    assert_eq!(
        session_enc.decrypt_response(&encrypted2).unwrap(),
        "Response 2"
    );
}

/// Test large prompt encryption
#[test]
fn test_large_prompt_encryption() {
    let encryptor = PromptEncryptor::new();
    let key = EncryptionKey::derive(b"secret_key", b"context");
    encryptor.register_key(key.clone());

    // Create a large prompt (10KB)
    let large_prompt = "A".repeat(10 * 1024);

    let encrypted = encryptor.encrypt_prompt(&key, &large_prompt).unwrap();
    let decrypted = encryptor.decrypt_prompt(&encrypted).unwrap();

    assert_eq!(large_prompt, decrypted);
    assert_eq!(encrypted.ciphertext_len(), 10 * 1024);
}

/// Test Unicode prompt encryption
#[test]
fn test_unicode_prompt_encryption() {
    let encryptor = PromptEncryptor::new();
    let key = EncryptionKey::derive(b"secret_key", b"context");
    encryptor.register_key(key.clone());

    let unicode_prompt = "你好世界! Привет мир! 🌍🚀";

    let encrypted = encryptor.encrypt_prompt(&key, unicode_prompt).unwrap();
    let decrypted = encryptor.decrypt_prompt(&encrypted).unwrap();

    assert_eq!(unicode_prompt, decrypted);
}

/// Test tampered auth tag is detected
#[test]
fn test_tampered_auth_tag_detected() {
    let encryptor = PromptEncryptor::new();
    let key = EncryptionKey::derive(b"secret_key", b"context");
    encryptor.register_key(key.clone());

    let prompt = "Test prompt";
    let mut encrypted = encryptor.encrypt_prompt(&key, prompt).unwrap();

    // Tamper with auth tag
    encrypted.auth_tag[0] ^= 0xFF;

    let result = encryptor.decrypt_prompt(&encrypted);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("AUTH_TAG_MISMATCH"));
}
