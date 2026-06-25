//! 🔐 Q3 production wiring: a post-quantum secure channel over a Tor circuit stream.
//!
//! This is what turns the circuit-level KEM (see `circuit_manager`) into something that
//! actually runs over the wire on a real connection:
//!
//!   1. **Handshake** — the initiator sends its Dilithium5-authenticated handshake that
//!      carries a hybrid X25519+Kyber-1024 KEM public key; the responder verifies the
//!      signature (which covers the KEM key, so a MITM can't swap it), encapsulates, and
//!      sends the ciphertext back. Both sides derive the same 32-byte circuit key.
//!   2. **Channel** — [`PqAead`] encrypts/authenticates every subsequent payload with
//!      ChaCha20-Poly1305 under that key. So circuit traffic is confidential + integrity-
//!      protected by a key that stays secret unless BOTH X25519 *and* Kyber-1024 are broken.
//!
//! Generic over any `AsyncRead + AsyncWrite` stream so it works on a real Tor `TcpStream`
//! (via `QTorClient::connect_to_peer_pq`) and is unit-testable over `tokio::io::duplex`.
//!
//! Backward-compatible: nothing uses this unless a caller opts into `connect_to_peer_pq`
//! and the remote runs the matching accept side. The legacy `connect_to_peer` path and its
//! callers are untouched.

use anyhow::{anyhow, Result};
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::circuit_manager::{CircuitAuthHandshake, CircuitManager};
use crate::quantum_resistant::PQKeyPair;

/// Max handshake frame (the Dilithium5 pubkey ~2.6KB + sig ~4.6KB + Kyber-1024 ~1.6KB ≈ 9KB).
const MAX_FRAME: usize = 64 * 1024;
/// Reject handshakes whose timestamp is older than this (replay window).
const HANDSHAKE_MAX_AGE_SECS: u64 = 120;

async fn write_frame<S: AsyncWrite + Unpin>(stream: &mut S, data: &[u8]) -> Result<()> {
    if data.len() > MAX_FRAME {
        return Err(anyhow!("handshake frame too large: {} bytes", data.len()));
    }
    stream.write_all(&(data.len() as u32).to_be_bytes()).await?;
    stream.write_all(data).await?;
    stream.flush().await?;
    Ok(())
}

async fn read_frame<S: AsyncRead + Unpin>(stream: &mut S) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_FRAME {
        return Err(anyhow!("peer handshake frame too large: {len} bytes"));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    Ok(buf)
}

/// Initiator side. `handshake`/`ephemeral` come from
/// [`CircuitManager::create_auth_handshake_kem`] (built before any `.await` so the manager
/// lock isn't held across IO). Returns the agreed 32-byte circuit key.
pub async fn client_handshake<S>(
    stream: &mut S,
    handshake: &CircuitAuthHandshake,
    ephemeral: &PQKeyPair,
) -> Result<[u8; 32]>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let encoded = bincode::serialize(handshake)
        .map_err(|e| anyhow!("encode handshake: {e}"))?;
    write_frame(stream, &encoded).await?;
    let ciphertext = read_frame(stream).await?;
    CircuitManager::complete_kem(ephemeral, &ciphertext)
}

/// Responder side. Reads + authenticates the peer's handshake, encapsulates to its KEM key,
/// returns `(peer_auth_fingerprint, circuit_key)`. `circuit_id` MUST match the value the
/// initiator used (it's part of the signed challenge).
pub async fn server_handshake<S>(stream: &mut S, circuit_id: u64) -> Result<([u8; 32], [u8; 32])>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let encoded = read_frame(stream).await?;
    let handshake: CircuitAuthHandshake =
        bincode::deserialize(&encoded).map_err(|e| anyhow!("decode handshake: {e}"))?;
    // Dilithium5 verification — also authenticates the embedded KEM public key.
    let peer_fingerprint =
        CircuitManager::verify_auth_handshake(&handshake, circuit_id, HANDSHAKE_MAX_AGE_SECS)?;
    let (ciphertext, circuit_key) = CircuitManager::respond_kem(&handshake)?;
    write_frame(stream, &ciphertext).await?;
    Ok((peer_fingerprint, circuit_key))
}

/// ChaCha20-Poly1305 AEAD over the agreed circuit key.
///
/// Nonce = `[0u8;4] || counter.to_be_bytes()`. The caller MUST supply a strictly increasing
/// counter per direction and MUST NOT reuse a `(key, counter)` pair (nonce reuse breaks
/// confidentiality). Pair this with separate send/recv counters per peer.
pub struct PqAead {
    cipher: ChaCha20Poly1305,
}

impl PqAead {
    pub fn new(key: &[u8; 32]) -> Self {
        // new_from_slice only fails on wrong length; 32 is correct by construction.
        let cipher = ChaCha20Poly1305::new_from_slice(key)
            .expect("ChaCha20Poly1305 key is always 32 bytes");
        Self { cipher }
    }

    fn nonce(counter: u64) -> [u8; 12] {
        let mut n = [0u8; 12];
        n[4..].copy_from_slice(&counter.to_be_bytes());
        n
    }

    /// Encrypt + authenticate `plaintext`. Output is ciphertext‖tag.
    pub fn seal(&self, counter: u64, plaintext: &[u8]) -> Result<Vec<u8>> {
        let nonce = Self::nonce(counter);
        self.cipher
            .encrypt(Nonce::from_slice(&nonce), plaintext)
            .map_err(|_| anyhow!("AEAD seal failed"))
    }

    /// Verify + decrypt. Fails if the ciphertext was tampered with or the counter is wrong.
    pub fn open(&self, counter: u64, ciphertext: &[u8]) -> Result<Vec<u8>> {
        let nonce = Self::nonce(counter);
        self.cipher
            .decrypt(Nonce::from_slice(&nonce), ciphertext)
            .map_err(|_| anyhow!("AEAD open failed (tampered or wrong key/counter)"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pq_secure_channel_handshake_and_aead_roundtrip() {
        let cm = CircuitManager::new_test("127.0.0.1:9050".parse().unwrap());
        let circuit_id: u64 = 0xC0FFEE;
        let (handshake, ephemeral) = cm.create_auth_handshake_kem(circuit_id).unwrap();

        let (mut client_io, mut server_io) = tokio::io::duplex(16 * 1024);

        let server = tokio::spawn(async move { server_handshake(&mut server_io, circuit_id).await });
        let client_key = client_handshake(&mut client_io, &handshake, &ephemeral)
            .await
            .unwrap();
        let (_peer_fp, server_key) = server.await.unwrap().unwrap();

        // Both ends derived the same, real (non-zero) key.
        assert_eq!(client_key, server_key, "circuit keys must match");
        assert_ne!(client_key, [0u8; 32], "circuit key must not be the zero placeholder");

        // AEAD round-trip with the agreed key.
        let sender = PqAead::new(&client_key);
        let receiver = PqAead::new(&server_key);
        let msg = b"quillon block gossip payload";
        let sealed = sender.seal(1, msg).unwrap();
        assert_ne!(&sealed[..], &msg[..], "payload must be encrypted");
        assert_eq!(receiver.open(1, &sealed).unwrap(), msg);

        // Tamper detection.
        let mut bad = sealed.clone();
        bad[0] ^= 0xFF;
        assert!(receiver.open(1, &bad).is_err(), "tampered ciphertext must fail");
        // Wrong counter must fail.
        assert!(receiver.open(2, &sealed).is_err(), "wrong counter must fail");
    }

    #[tokio::test]
    async fn server_rejects_wrong_circuit_id() {
        let cm = CircuitManager::new_test("127.0.0.1:9050".parse().unwrap());
        let (handshake, ephemeral) = cm.create_auth_handshake_kem(1).unwrap();
        let (mut client_io, mut server_io) = tokio::io::duplex(16 * 1024);
        // Server expects a DIFFERENT circuit_id → signature challenge mismatch.
        let server = tokio::spawn(async move { server_handshake(&mut server_io, 999).await });
        // Client write may succeed; the handshake result on the server must be an error.
        let _ = client_handshake(&mut client_io, &handshake, &ephemeral).await;
        assert!(server.await.unwrap().is_err(), "server must reject mismatched circuit_id");
    }
}
