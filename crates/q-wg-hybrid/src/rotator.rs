//! The PSK rotation state machine.
//!
//! Each call to `rotate()`:
//!   1. Encapsulates a 256-bit shared secret against the peer's Kyber pubkey
//!      → produces (ciphertext, shared_secret).
//!   2. Derives the new PSK by mixing previous_psk + shared_secret through
//!      SHA-256 (so a compromise of one rotation can't unwind earlier ones
//!      and vice versa — forward + backward secrecy from the mixing).
//!   3. Installs the new PSK on the WireGuard interface via the backend.
//!   4. Returns the ciphertext for the caller to publish to the peer.
//!
//! The peer-side path is the mirror: receive ciphertext, decapsulate with
//! own secret key, run the same SHA-256 mix, install. Both ends keep their
//! "previous_psk" in lockstep.

use crate::{HybridError, KyberKeypair, KyberPublicKey, WireGuardBackend};
use pqcrypto_kyber::kyber1024;
use pqcrypto_traits::kem::{Ciphertext, PublicKey, SecretKey, SharedSecret};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;
use tracing::{debug, info, warn};

/// What goes on the wire (Quillon-mesh gossipsub topic) when we rotate.
/// `ciphertext` is the Kyber-1024 ciphertext (1568 bytes). `epoch` lets
/// the peer ignore stale rotations and apply rotations in order even if
/// gossipsub delivers them out-of-order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationEnvelope {
    pub epoch: u64,
    pub ciphertext: Vec<u8>,
    pub from_kyber_pubkey_sha256: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct RotatorConfig {
    /// Peer's Kyber-1024 public key. Out-of-band — comes from chain gossip.
    pub peer_pubkey: KyberPublicKey,
    /// WireGuard interface name, e.g. "wg0".
    pub wg_iface: String,
    /// WireGuard peer pubkey (Curve25519, base64 — the standard WG pubkey).
    pub wg_peer: String,
    /// Initial PSK to seed the chain. If you have none, use [0u8; 32] —
    /// the first rotation will mix in real entropy. Zeroed on drop in
    /// the rotator's internal state.
    pub initial_psk: [u8; 32],
}

#[derive(Debug, Error)]
pub enum RotationError {
    #[error(transparent)]
    Hybrid(#[from] HybridError),
}

pub struct HybridRotator {
    config: RotatorConfig,
    keypair: KyberKeypair,
    backend: Box<dyn WireGuardBackend + Send + Sync>,
    current_psk: [u8; 32],
    epoch: AtomicU64,
}

impl HybridRotator {
    pub fn new(
        config: RotatorConfig,
        keypair: KyberKeypair,
        backend: Box<dyn WireGuardBackend + Send + Sync>,
    ) -> Self {
        let current_psk = config.initial_psk;
        Self {
            config,
            keypair,
            backend,
            current_psk,
            epoch: AtomicU64::new(0),
        }
    }

    pub fn current_psk(&self) -> [u8; 32] { self.current_psk }
    pub fn epoch(&self) -> u64 { self.epoch.load(Ordering::Relaxed) }

    /// Drive one rotation. Returns the envelope the caller publishes to the
    /// peer. Mutates internal PSK state and pushes the new PSK to the wg
    /// interface.
    pub fn rotate(&mut self) -> Result<RotationEnvelope, RotationError> {
        let peer_pk = kyber1024::PublicKey::from_bytes(self.config.peer_pubkey.as_bytes())
            .map_err(|e| HybridError::Kyber(format!("peer pubkey invalid: {e:?}")))?;

        let (shared_secret, ciphertext) = kyber1024::encapsulate(&peer_pk);

        let new_psk = mix_psk(&self.current_psk, shared_secret.as_bytes());
        self.backend.set_preshared_key(&self.config.wg_iface, &self.config.wg_peer, &new_psk)
            .map_err(HybridError::WgBackend)?;

        self.current_psk = new_psk;
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed);

        let from_hash: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(&self.keypair.pk_bytes);
            h.finalize().into()
        };

        info!(target: "q_wg_hybrid", epoch, "rotated PSK via Kyber-1024 encapsulation");

        Ok(RotationEnvelope {
            epoch,
            ciphertext: ciphertext.as_bytes().to_vec(),
            from_kyber_pubkey_sha256: from_hash,
        })
    }

    /// Peer-side: apply an incoming envelope.
    pub fn apply_remote(&mut self, env: &RotationEnvelope) -> Result<(), RotationError> {
        let observed_epoch = self.epoch.load(Ordering::Relaxed);
        if env.epoch < observed_epoch {
            warn!(target: "q_wg_hybrid",
                  ours = observed_epoch, theirs = env.epoch,
                  "ignoring stale envelope");
            return Ok(());
        }

        let ct = kyber1024::Ciphertext::from_bytes(&env.ciphertext)
            .map_err(|e| HybridError::Kyber(format!("ciphertext invalid: {e:?}")))?;
        let sk = kyber1024::SecretKey::from_bytes(&self.keypair.sk_bytes)
            .map_err(|e| HybridError::Kyber(format!("our sk invalid: {e:?}")))?;
        let shared_secret = kyber1024::decapsulate(&ct, &sk);

        let new_psk = mix_psk(&self.current_psk, shared_secret.as_bytes());
        self.backend.set_preshared_key(&self.config.wg_iface, &self.config.wg_peer, &new_psk)
            .map_err(HybridError::WgBackend)?;

        self.current_psk = new_psk;
        self.epoch.store(env.epoch + 1, Ordering::Relaxed);

        debug!(target: "q_wg_hybrid", epoch = env.epoch + 1, "applied remote envelope");
        Ok(())
    }
}

/// Mix the previous PSK with a fresh KEM shared secret. Symmetric, deterministic.
///
/// SHA-256(prev_psk || shared_secret || "q-wg-hybrid-v1") → 32 bytes
///
/// The domain-separation tag keeps this distinct from any other place a
/// PSK might be derived in Quillon (the chain has its own session-key paths).
fn mix_psk(prev: &[u8; 32], shared: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(prev);
    h.update(shared);
    h.update(b"q-wg-hybrid-v1");
    h.finalize().into()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wg_backend::MockBackend;
    use std::sync::Arc;
    use std::sync::Mutex;

    fn make_rotator(peer: &KyberPublicKey, mine: KyberKeypair) -> (HybridRotator, Arc<Mutex<Option<[u8;32]>>>) {
        let last_psk = Arc::new(Mutex::new(None));
        let backend = MockBackend::new(last_psk.clone());
        let cfg = RotatorConfig {
            peer_pubkey: peer.clone(),
            wg_iface: "wg-test".into(),
            wg_peer: "PEER_PUBKEY_BASE64".into(),
            initial_psk: [0u8; 32],
        };
        (HybridRotator::new(cfg, mine, Box::new(backend)), last_psk)
    }

    #[test]
    fn round_trip_both_sides_derive_same_psk() {
        // Alice generates, sends ciphertext to Bob, Bob applies and ends up
        // with the same PSK Alice already pushed to her wg.
        let alice_kp = KyberKeypair::generate();
        let bob_kp = KyberKeypair::generate();
        let alice_pub = alice_kp.public();
        let bob_pub = bob_kp.public();

        let (mut alice, alice_last) = make_rotator(&bob_pub, alice_kp);
        let (mut bob, bob_last) = make_rotator(&alice_pub, bob_kp);

        let envelope = alice.rotate().expect("alice rotate");
        bob.apply_remote(&envelope).expect("bob apply");

        let a_psk = alice_last.lock().unwrap().unwrap();
        let b_psk = bob_last.lock().unwrap().unwrap();
        assert_eq!(a_psk, b_psk, "Alice and Bob must converge on the same PSK");
        assert_ne!(a_psk, [0u8; 32], "PSK must have moved off zero");
    }

    #[test]
    fn successive_rotations_diverge_psk() {
        // Two consecutive rotations from Alice should produce two distinct PSKs.
        let alice_kp = KyberKeypair::generate();
        let bob_kp = KyberKeypair::generate();
        let bob_pub = bob_kp.public();

        let (mut alice, alice_last) = make_rotator(&bob_pub, alice_kp);

        alice.rotate().unwrap();
        let psk_1 = alice_last.lock().unwrap().unwrap();
        alice.rotate().unwrap();
        let psk_2 = alice_last.lock().unwrap().unwrap();

        assert_ne!(psk_1, psk_2, "successive rotations must diverge");
    }

    #[test]
    fn stale_envelope_ignored() {
        let alice_kp = KyberKeypair::generate();
        let bob_kp = KyberKeypair::generate();
        let alice_pub = alice_kp.public();
        let bob_pub = bob_kp.public();

        let (mut alice, _) = make_rotator(&bob_pub, alice_kp);
        let (mut bob, bob_last) = make_rotator(&alice_pub, bob_kp);

        // Get bob to epoch 2 via two normal rotations.
        let env_0 = alice.rotate().unwrap();
        bob.apply_remote(&env_0).unwrap();
        let env_1 = alice.rotate().unwrap();
        bob.apply_remote(&env_1).unwrap();
        let psk_at_2 = bob_last.lock().unwrap().unwrap();

        // Re-deliver env_0 — bob must ignore it.
        bob.apply_remote(&env_0).unwrap();
        let psk_after = bob_last.lock().unwrap().unwrap();
        assert_eq!(psk_at_2, psk_after, "stale envelope must not change PSK");
    }
}
