//! Safe Rust wrapper for SQIsign post-quantum isogeny-based signatures.
//!
//! SQIsign is a post-quantum digital signature scheme based on supersingular
//! isogenies, offering the smallest signature+key sizes among all PQ candidates.
//!
//! This crate wraps the official C reference implementation via FFI.
//!
//! ## NIST Level 1 sizes
//! - Public key: 65 bytes
//! - Secret key: 353 bytes
//! - Signature: 148 bytes (detached)
//!
//! ## Example
//! ```
//! use q_sqisign::KeyPair;
//!
//! let kp = KeyPair::generate().expect("keygen");
//! let msg = b"post-quantum hello";
//! let sig = kp.sign(msg).expect("sign");
//! assert!(q_sqisign::verify(kp.public_key(), msg, &sig).expect("verify"));
//! ```

pub mod error;

pub use error::SqiSignError;
use q_sqisign_sys as ffi;
use zeroize::Zeroize;

/// Public key size in bytes (Level 1).
pub const PUBLIC_KEY_BYTES: usize = ffi::CRYPTO_PUBLICKEYBYTES;

/// Secret key size in bytes (Level 1).
pub const SECRET_KEY_BYTES: usize = ffi::CRYPTO_SECRETKEYBYTES;

/// Detached signature size in bytes (Level 1).
pub const SIGNATURE_BYTES: usize = ffi::CRYPTO_BYTES;

/// A SQIsign secret key. Zeroized on drop.
struct SecretKey {
    bytes: [u8; SECRET_KEY_BYTES],
}

impl Zeroize for SecretKey {
    fn zeroize(&mut self) {
        self.bytes.zeroize();
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        self.zeroize();
    }
}

/// A SQIsign key pair (public key + secret key).
///
/// The secret key is zeroized when the key pair is dropped.
pub struct KeyPair {
    pk: [u8; PUBLIC_KEY_BYTES],
    sk: SecretKey,
}

impl KeyPair {
    /// Generate a fresh SQIsign key pair using the C reference implementation.
    pub fn generate() -> Result<Self, SqiSignError> {
        let mut pk = [0u8; PUBLIC_KEY_BYTES];
        let mut sk = SecretKey {
            bytes: [0u8; SECRET_KEY_BYTES],
        };

        let ret = unsafe { ffi::crypto_sign_keypair(pk.as_mut_ptr(), sk.bytes.as_mut_ptr()) };

        if ret != 0 {
            return Err(SqiSignError::KeyGenFailed(ret));
        }

        Ok(KeyPair { pk, sk })
    }

    /// Returns the public key bytes.
    pub fn public_key(&self) -> &[u8; PUBLIC_KEY_BYTES] {
        &self.pk
    }

    /// Sign a message, producing a detached signature.
    ///
    /// The returned `Vec<u8>` contains exactly `SIGNATURE_BYTES` (148) bytes.
    pub fn sign(&self, message: &[u8]) -> Result<Vec<u8>, SqiSignError> {
        // crypto_sign produces sig || message
        let mut sm = vec![0u8; SIGNATURE_BYTES + message.len()];
        let mut smlen: std::os::raw::c_ulonglong = 0;

        let ret = unsafe {
            ffi::crypto_sign(
                sm.as_mut_ptr(),
                &mut smlen,
                message.as_ptr(),
                message.len() as std::os::raw::c_ulonglong,
                self.sk.bytes.as_ptr(),
            )
        };

        if ret != 0 {
            return Err(SqiSignError::SigningFailed(ret));
        }

        // Extract only the signature prefix (first SIGNATURE_BYTES bytes)
        sm.truncate(SIGNATURE_BYTES);
        Ok(sm)
    }
}

/// Verify a detached SQIsign signature.
///
/// Returns `Ok(true)` if the signature is valid, `Ok(false)` if the public key
/// or signature have wrong sizes, or `Err` if the C library encounters an internal error.
pub fn verify(
    pk: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<bool, SqiSignError> {
    if pk.len() != PUBLIC_KEY_BYTES {
        return Err(SqiSignError::InvalidPublicKeySize {
            expected: PUBLIC_KEY_BYTES,
            got: pk.len(),
        });
    }
    if signature.len() != SIGNATURE_BYTES {
        return Err(SqiSignError::InvalidSignatureSize {
            expected: SIGNATURE_BYTES,
            got: signature.len(),
        });
    }

    let ret = unsafe {
        ffi::sqisign_verify(
            message.as_ptr(),
            message.len() as std::os::raw::c_ulonglong,
            signature.as_ptr(),
            signature.len() as std::os::raw::c_ulonglong,
            pk.as_ptr(),
        )
    };

    Ok(ret == 0)
}

/// Verify a signed message (signature || message) and extract the original message.
///
/// This uses the NIST `crypto_sign_open` API. On success, returns the original message.
pub fn open(pk: &[u8], signed_message: &[u8]) -> Result<Vec<u8>, SqiSignError> {
    if pk.len() != PUBLIC_KEY_BYTES {
        return Err(SqiSignError::InvalidPublicKeySize {
            expected: PUBLIC_KEY_BYTES,
            got: pk.len(),
        });
    }

    let mut m_out = vec![0u8; signed_message.len()];
    let mut mlen: std::os::raw::c_ulonglong = 0;

    let ret = unsafe {
        ffi::crypto_sign_open(
            m_out.as_mut_ptr(),
            &mut mlen,
            signed_message.as_ptr(),
            signed_message.len() as std::os::raw::c_ulonglong,
            pk.as_ptr(),
        )
    };

    if ret != 0 {
        return Err(SqiSignError::VerificationFailed);
    }

    m_out.truncate(mlen as usize);
    Ok(m_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keygen() {
        let kp = KeyPair::generate().expect("keygen should succeed");
        assert!(
            kp.public_key().iter().any(|&b| b != 0),
            "public key should not be all zeros"
        );
    }

    #[test]
    fn test_sign_verify_roundtrip() {
        let kp = KeyPair::generate().expect("keygen");
        let msg = b"SQIsign roundtrip test";

        let sig = kp.sign(msg).expect("signing");
        assert_eq!(sig.len(), SIGNATURE_BYTES);

        let valid = verify(kp.public_key(), msg, &sig).expect("verify");
        assert!(valid, "signature should be valid");
    }

    #[test]
    fn test_wrong_message_rejected() {
        let kp = KeyPair::generate().expect("keygen");
        let msg = b"original message";
        let wrong_msg = b"tampered message";

        let sig = kp.sign(msg).expect("signing");

        let valid = verify(kp.public_key(), wrong_msg, &sig).expect("verify");
        assert!(!valid, "wrong message should fail verification");
    }

    #[test]
    fn test_wrong_key_rejected() {
        let kp1 = KeyPair::generate().expect("keygen");
        let kp2 = KeyPair::generate().expect("keygen");
        let msg = b"key mismatch test";

        let sig = kp1.sign(msg).expect("signing");

        let valid = verify(kp2.public_key(), msg, &sig).expect("verify");
        assert!(!valid, "wrong key should fail verification");
    }

    #[test]
    fn test_corrupted_signature_rejected() {
        let kp = KeyPair::generate().expect("keygen");
        let msg = b"corruption test";

        let mut sig = kp.sign(msg).expect("signing");
        // Flip a bit in the signature
        sig[0] ^= 0xFF;

        let valid = verify(kp.public_key(), msg, &sig).expect("verify");
        assert!(!valid, "corrupted signature should fail verification");
    }

    #[test]
    fn test_open_roundtrip() {
        let kp = KeyPair::generate().expect("keygen");
        let msg = b"open API test message";

        // Sign produces sig || msg
        let mut sm = vec![0u8; SIGNATURE_BYTES + msg.len()];
        let mut smlen: std::os::raw::c_ulonglong = 0;
        let ret = unsafe {
            ffi::crypto_sign(
                sm.as_mut_ptr(),
                &mut smlen,
                msg.as_ptr(),
                msg.len() as std::os::raw::c_ulonglong,
                kp.sk.bytes.as_ptr(),
            )
        };
        assert_eq!(ret, 0);
        sm.truncate(smlen as usize);

        let recovered = open(kp.public_key(), &sm).expect("open");
        assert_eq!(&recovered, msg);
    }

    #[test]
    fn test_signature_size_constants() {
        assert_eq!(PUBLIC_KEY_BYTES, 65);
        assert_eq!(SECRET_KEY_BYTES, 353);
        assert_eq!(SIGNATURE_BYTES, 148);
    }

    #[test]
    fn test_invalid_pk_size_error() {
        let short_pk = [0u8; 32];
        let msg = b"test";
        let sig = [0u8; SIGNATURE_BYTES];

        match verify(&short_pk, msg, &sig) {
            Err(SqiSignError::InvalidPublicKeySize { expected: 65, got: 32 }) => {}
            other => panic!("expected InvalidPublicKeySize, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_sig_size_error() {
        let pk = [0u8; PUBLIC_KEY_BYTES];
        let msg = b"test";
        let short_sig = [0u8; 32];

        match verify(&pk, msg, &short_sig) {
            Err(SqiSignError::InvalidSignatureSize {
                expected: 148,
                got: 32,
            }) => {}
            other => panic!("expected InvalidSignatureSize, got {:?}", other),
        }
    }

    #[test]
    fn test_multiple_signatures_independent() {
        let kp = KeyPair::generate().expect("keygen");
        let msg1 = b"message one";
        let msg2 = b"message two";

        let sig1 = kp.sign(msg1).expect("sign msg1");
        let sig2 = kp.sign(msg2).expect("sign msg2");

        // Each signature should only verify against its own message
        assert!(verify(kp.public_key(), msg1, &sig1).unwrap());
        assert!(verify(kp.public_key(), msg2, &sig2).unwrap());
        assert!(!verify(kp.public_key(), msg1, &sig2).unwrap());
        assert!(!verify(kp.public_key(), msg2, &sig1).unwrap());
    }

    #[test]
    fn test_empty_message() {
        let kp = KeyPair::generate().expect("keygen");
        let msg = b"";

        let sig = kp.sign(msg).expect("signing empty message");
        assert!(verify(kp.public_key(), msg, &sig).unwrap());
    }
}
