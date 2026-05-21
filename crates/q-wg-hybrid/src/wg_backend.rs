//! Pluggable WireGuard configuration backend.
//!
//! The library is unit-tested via `MockBackend` (records the last PSK set
//! by the rotator). Production uses `LinuxWgCli` which shells out to the
//! `wg` CLI. Other backends (netlink, BoringTun control socket) can be
//! added by implementing `WireGuardBackend`.

use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WgBackendError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("wg set returned non-zero: status={status} stderr={stderr}")]
    WgCliFailed { status: i32, stderr: String },
    #[error("backend not yet implemented for this platform")]
    Unimplemented,
}

pub trait WireGuardBackend {
    /// Install a new pre-shared key on the given interface for the given
    /// peer. The PSK is 32 bytes of raw entropy — the caller has already
    /// done the Kyber mixing.
    fn set_preshared_key(
        &self,
        iface: &str,
        peer_pubkey_base64: &str,
        psk: &[u8; 32],
    ) -> Result<(), WgBackendError>;
}

/// Backend used by unit tests. Stores the last PSK in a shared Arc<Mutex<_>>
/// so tests can introspect what got installed.
pub struct MockBackend {
    pub last_psk: Arc<Mutex<Option<[u8; 32]>>>,
}

impl MockBackend {
    pub fn new(last_psk: Arc<Mutex<Option<[u8; 32]>>>) -> Self {
        Self { last_psk }
    }
}

impl WireGuardBackend for MockBackend {
    fn set_preshared_key(&self, _iface: &str, _peer: &str, psk: &[u8; 32]) -> Result<(), WgBackendError> {
        *self.last_psk.lock().unwrap() = Some(*psk);
        Ok(())
    }
}

/// Production Linux backend. Calls:
///   echo <base64(psk)> | wg set <iface> peer <peer_pubkey> preshared-key /dev/stdin
///
/// `wg set` is atomic for the PSK field — it does not drop the tunnel; the
/// next handshake (or the existing session keys rolled at the next ChaCha20
/// nonce window) starts using the new PSK. WireGuard's existing nonce-
/// monotonicity protections cover the brief overlap.
pub struct LinuxWgCli;

impl WireGuardBackend for LinuxWgCli {
    fn set_preshared_key(
        &self,
        iface: &str,
        peer_pubkey_base64: &str,
        psk: &[u8; 32],
    ) -> Result<(), WgBackendError> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let psk_b64 = base64_encode(psk);

        let mut child = Command::new("wg")
            .args([
                "set", iface,
                "peer", peer_pubkey_base64,
                "preshared-key", "/dev/stdin",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(psk_b64.as_bytes())?;
            stdin.write_all(b"\n")?;
        }

        let out = child.wait_with_output()?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            return Err(WgBackendError::WgCliFailed {
                status: out.status.code().unwrap_or(-1),
                stderr,
            });
        }
        Ok(())
    }
}

/// Tiny base64 encoder — avoids pulling another dep just for one call.
/// Standard RFC 4648 alphabet, no line wrapping.
fn base64_encode(input: &[u8; 32]) -> String {
    const ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(44);
    let mut i = 0;
    while i + 3 <= input.len() {
        let b = ((input[i] as u32) << 16) | ((input[i+1] as u32) << 8) | (input[i+2] as u32);
        out.push(ALPHA[((b >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((b >> 12) & 0x3F) as usize] as char);
        out.push(ALPHA[((b >>  6) & 0x3F) as usize] as char);
        out.push(ALPHA[((b      ) & 0x3F) as usize] as char);
        i += 3;
    }
    // 32 bytes = 30 + 2 → one padded group (2 input bytes → 3 chars + '=')
    if i < input.len() {
        let remaining = input.len() - i;
        let b = match remaining {
            2 => ((input[i] as u32) << 16) | ((input[i+1] as u32) << 8),
            1 => (input[i] as u32) << 16,
            _ => unreachable!(),
        };
        out.push(ALPHA[((b >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((b >> 12) & 0x3F) as usize] as char);
        if remaining == 2 {
            out.push(ALPHA[((b >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        out.push('=');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trip_known_vector() {
        // 32 zero bytes → "AAAA..." (44 chars after padding)
        let zeros = [0u8; 32];
        assert_eq!(base64_encode(&zeros), "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=");
    }

    #[test]
    fn base64_alternating_pattern() {
        let mut b = [0u8; 32];
        for i in 0..32 { b[i] = i as u8; }
        let out = base64_encode(&b);
        assert_eq!(out.len(), 44, "32 bytes encode to 44 chars with padding");
        assert!(out.ends_with('='), "32 bytes (mod 3 = 2) requires one '=' pad");
    }

    #[test]
    fn mock_backend_records_psk() {
        let last = Arc::new(Mutex::new(None));
        let b = MockBackend::new(last.clone());
        let psk = [0xAAu8; 32];
        b.set_preshared_key("wg0", "peer", &psk).unwrap();
        assert_eq!(last.lock().unwrap().unwrap(), psk);
    }
}
