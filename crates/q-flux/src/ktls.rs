//! kTLS (Kernel TLS) offload for q-flux — Issue #017.
//!
//! When enabled, symmetric TLS encryption/decryption is offloaded to the
//! Linux kernel (4.13+), allowing zero-copy `sendfile()` for static files
//! and reducing context switches for all TLS traffic.
//!
//! Only AES-128-GCM and AES-256-GCM are supported (the most common TLS 1.3 ciphers).
//! Falls back gracefully to userspace TLS if activation fails.
//!
//! Requires: Linux kernel with `CONFIG_TLS=y` and the `tls` kernel module loaded.

#![allow(dead_code)] // kTLS functions are compiled but wired in incrementally

#[allow(unused_imports)]
use std::os::unix::io::AsRawFd;

/// Linux kTLS constants from `linux/tls.h`.
const SOL_TLS: libc::c_int = 282;
const TLS_TX: libc::c_int = 1;
#[allow(dead_code)]
const TLS_RX: libc::c_int = 2;

/// TLS 1.3 version constant.
const TLS_1_3_VERSION: u16 = 0x0304;
/// TLS 1.2 version constant.
const TLS_1_2_VERSION: u16 = 0x0303;

/// kTLS cipher type IDs (from `linux/tls.h`).
const TLS_CIPHER_AES_GCM_128: u16 = 51;
const TLS_CIPHER_AES_GCM_256: u16 = 52;

/// AES-128-GCM kTLS crypto info structure.
/// Matches `struct tls12_crypto_info_aes_gcm_128` from `linux/tls.h`.
#[repr(C)]
#[derive(Clone)]
struct TlsCryptoInfoAesGcm128 {
    /// TLS version (0x0303 for TLS 1.2, 0x0304 for TLS 1.3).
    version: u16,
    /// Cipher type (TLS_CIPHER_AES_GCM_128 = 51).
    cipher_type: u16,
    /// Initialization vector (4 bytes explicit nonce for TLS 1.2, 8 bytes for TLS 1.3).
    iv: [u8; 8],
    /// Encryption key (16 bytes for AES-128).
    key: [u8; 16],
    /// Salt (4 bytes implicit nonce).
    salt: [u8; 4],
    /// Record sequence number.
    rec_seq: [u8; 8],
}

/// AES-256-GCM kTLS crypto info structure.
/// Matches `struct tls12_crypto_info_aes_gcm_256` from `linux/tls.h`.
#[repr(C)]
#[derive(Clone)]
struct TlsCryptoInfoAesGcm256 {
    version: u16,
    cipher_type: u16,
    iv: [u8; 8],
    key: [u8; 32],
    salt: [u8; 4],
    rec_seq: [u8; 8],
}

/// Result of attempting kTLS activation on a socket.
#[derive(Debug)]
pub enum KtlsResult {
    /// kTLS TX successfully activated.
    Activated,
    /// kTLS not supported for this cipher suite.
    UnsupportedCipher(String),
    /// kTLS activation failed (kernel error).
    Failed(String),
    /// kTLS disabled in configuration.
    Disabled,
}

impl std::fmt::Display for KtlsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KtlsResult::Activated => write!(f, "kTLS TX activated"),
            KtlsResult::UnsupportedCipher(c) => write!(f, "kTLS unsupported cipher: {}", c),
            KtlsResult::Failed(e) => write!(f, "kTLS activation failed: {}", e),
            KtlsResult::Disabled => write!(f, "kTLS disabled"),
        }
    }
}

/// Attempt to activate kTLS TX offload on a connected TLS socket.
///
/// After a successful TLS handshake, this extracts the negotiated cipher keys
/// and installs them into the kernel via `setsockopt(SOL_TLS, TLS_TX, ...)`.
/// Once activated, the kernel handles TLS record encryption in the socket layer,
/// enabling zero-copy `sendfile()` for static files.
///
/// # Arguments
/// * `fd` - Raw file descriptor of the TCP socket (after TLS handshake).
/// * `tls_version` - Negotiated TLS version (0x0303 for 1.2, 0x0304 for 1.3).
/// * `cipher_name` - Negotiated cipher suite name (e.g., "TLS13_AES_128_GCM_SHA256").
/// * `key` - Symmetric encryption key bytes.
/// * `iv` - Initialization vector / nonce bytes.
/// * `seq` - Record sequence number (8 bytes).
///
/// # Returns
/// `KtlsResult` indicating success or the reason for failure.
///
/// # Safety
/// Uses `setsockopt` FFI. The fd must be a valid TCP socket with a completed TLS handshake.
pub fn activate_ktls_tx(
    fd: std::os::unix::io::RawFd,
    tls_version: u16,
    cipher_name: &str,
    key: &[u8],
    iv: &[u8],
    seq: &[u8],
) -> KtlsResult {
    // Determine cipher type and validate key/IV sizes
    let cipher_lower = cipher_name.to_lowercase();

    if cipher_lower.contains("aes_128_gcm") || cipher_lower.contains("aes128gcm") {
        if key.len() != 16 {
            return KtlsResult::Failed(format!("AES-128-GCM key must be 16 bytes, got {}", key.len()));
        }
        activate_aes_gcm_128(fd, tls_version, key, iv, seq)
    } else if cipher_lower.contains("aes_256_gcm") || cipher_lower.contains("aes256gcm") {
        if key.len() != 32 {
            return KtlsResult::Failed(format!("AES-256-GCM key must be 32 bytes, got {}", key.len()));
        }
        activate_aes_gcm_256(fd, tls_version, key, iv, seq)
    } else {
        KtlsResult::UnsupportedCipher(cipher_name.to_string())
    }
}

fn activate_aes_gcm_128(
    fd: std::os::unix::io::RawFd,
    tls_version: u16,
    key: &[u8],
    iv: &[u8],
    seq: &[u8],
) -> KtlsResult {
    let mut info = TlsCryptoInfoAesGcm128 {
        version: tls_version,
        cipher_type: TLS_CIPHER_AES_GCM_128,
        iv: [0u8; 8],
        key: [0u8; 16],
        salt: [0u8; 4],
        rec_seq: [0u8; 8],
    };

    // Key
    info.key.copy_from_slice(key);

    // IV: first 4 bytes are the salt (implicit nonce), rest is the explicit IV
    if iv.len() >= 4 {
        info.salt.copy_from_slice(&iv[..4]);
    }
    if iv.len() > 4 {
        let copy_len = (iv.len() - 4).min(8);
        info.iv[..copy_len].copy_from_slice(&iv[4..4 + copy_len]);
    }

    // Record sequence number
    if !seq.is_empty() {
        let copy_len = seq.len().min(8);
        info.rec_seq[..copy_len].copy_from_slice(&seq[..copy_len]);
    }

    let result = unsafe {
        libc::setsockopt(
            fd,
            SOL_TLS,
            TLS_TX,
            &info as *const _ as *const libc::c_void,
            std::mem::size_of::<TlsCryptoInfoAesGcm128>() as libc::socklen_t,
        )
    };

    if result == 0 {
        KtlsResult::Activated
    } else {
        let err = std::io::Error::last_os_error();
        KtlsResult::Failed(format!("setsockopt(SOL_TLS, TLS_TX): {}", err))
    }
}

fn activate_aes_gcm_256(
    fd: std::os::unix::io::RawFd,
    tls_version: u16,
    key: &[u8],
    iv: &[u8],
    seq: &[u8],
) -> KtlsResult {
    let mut info = TlsCryptoInfoAesGcm256 {
        version: tls_version,
        cipher_type: TLS_CIPHER_AES_GCM_256,
        iv: [0u8; 8],
        key: [0u8; 32],
        salt: [0u8; 4],
        rec_seq: [0u8; 8],
    };

    info.key.copy_from_slice(key);

    if iv.len() >= 4 {
        info.salt.copy_from_slice(&iv[..4]);
    }
    if iv.len() > 4 {
        let copy_len = (iv.len() - 4).min(8);
        info.iv[..copy_len].copy_from_slice(&iv[4..4 + copy_len]);
    }

    if !seq.is_empty() {
        let copy_len = seq.len().min(8);
        info.rec_seq[..copy_len].copy_from_slice(&seq[..copy_len]);
    }

    let result = unsafe {
        libc::setsockopt(
            fd,
            SOL_TLS,
            TLS_TX,
            &info as *const _ as *const libc::c_void,
            std::mem::size_of::<TlsCryptoInfoAesGcm256>() as libc::socklen_t,
        )
    };

    if result == 0 {
        KtlsResult::Activated
    } else {
        let err = std::io::Error::last_os_error();
        KtlsResult::Failed(format!("setsockopt(SOL_TLS, TLS_TX): {}", err))
    }
}

/// Check if the kernel TLS module is loaded and functional.
/// This is a lightweight check suitable for startup diagnostics.
pub fn is_ktls_available() -> bool {
    // Try to load the tls kernel module (idempotent if already loaded)
    let _ = std::process::Command::new("modprobe")
        .arg("tls")
        .output();

    crate::acceptor::probe_ktls().available
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ktls_result_display() {
        assert_eq!(format!("{}", KtlsResult::Activated), "kTLS TX activated");
        assert_eq!(
            format!("{}", KtlsResult::UnsupportedCipher("CHACHA20".into())),
            "kTLS unsupported cipher: CHACHA20"
        );
        assert!(format!("{}", KtlsResult::Failed("test".into())).contains("kTLS activation failed"));
        assert_eq!(format!("{}", KtlsResult::Disabled), "kTLS disabled");
    }

    #[test]
    fn test_activate_wrong_key_size() {
        let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
        if fd < 0 { return; } // Skip if socket creation fails in CI

        let result = activate_ktls_tx(fd, TLS_1_3_VERSION, "TLS13_AES_128_GCM_SHA256", &[0u8; 8], &[], &[]);
        unsafe { libc::close(fd); }

        match result {
            KtlsResult::Failed(msg) => assert!(msg.contains("16 bytes")),
            _ => panic!("Expected Failed for wrong key size"),
        }
    }

    #[test]
    fn test_activate_wrong_key_size_256() {
        let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
        if fd < 0 { return; }

        let result = activate_ktls_tx(fd, TLS_1_3_VERSION, "TLS13_AES_256_GCM_SHA384", &[0u8; 16], &[], &[]);
        unsafe { libc::close(fd); }

        match result {
            KtlsResult::Failed(msg) => assert!(msg.contains("32 bytes")),
            _ => panic!("Expected Failed for wrong key size"),
        }
    }

    #[test]
    fn test_unsupported_cipher() {
        let result = activate_ktls_tx(0, TLS_1_3_VERSION, "TLS13_CHACHA20_POLY1305_SHA256", &[], &[], &[]);
        match result {
            KtlsResult::UnsupportedCipher(c) => assert!(c.contains("CHACHA20")),
            _ => panic!("Expected UnsupportedCipher"),
        }
    }

    #[test]
    fn test_tls_version_constants() {
        assert_eq!(TLS_1_2_VERSION, 0x0303);
        assert_eq!(TLS_1_3_VERSION, 0x0304);
    }

    #[test]
    fn test_cipher_type_constants() {
        assert_eq!(TLS_CIPHER_AES_GCM_128, 51);
        assert_eq!(TLS_CIPHER_AES_GCM_256, 52);
    }

    #[test]
    fn test_crypto_info_struct_sizes() {
        // Verify struct sizes match kernel expectations
        // AES-128-GCM: 2 + 2 + 8 + 16 + 4 + 8 = 40 bytes
        assert_eq!(std::mem::size_of::<TlsCryptoInfoAesGcm128>(), 40);
        // AES-256-GCM: 2 + 2 + 8 + 32 + 4 + 8 = 56 bytes
        assert_eq!(std::mem::size_of::<TlsCryptoInfoAesGcm256>(), 56);
    }
}
