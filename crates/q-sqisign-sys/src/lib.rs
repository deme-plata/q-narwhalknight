//! Raw FFI bindings to the SQIsign post-quantum signature scheme.
//!
//! This crate provides unsafe C bindings to the official SQIsign NIST Round 2
//! reference implementation. Use `q-sqisign` for a safe Rust wrapper.
//!
//! ## Security Level 1 (NIST Level I — 128-bit classical security)
//! - Public key: 65 bytes
//! - Secret key: 353 bytes
//! - Signature: 148 bytes

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::{c_int, c_uchar, c_ulonglong};

// ============================================================================
// SQIsign Level 1 (NIST Level I) constants
// ============================================================================

/// Public key size in bytes (Level 1)
pub const CRYPTO_PUBLICKEYBYTES: usize = 65;

/// Secret key size in bytes (Level 1)
pub const CRYPTO_SECRETKEYBYTES: usize = 353;

/// Signature size in bytes (Level 1)
pub const CRYPTO_BYTES: usize = 148;

// ============================================================================
// SQIsign Level 1 (NIST Level I) FFI functions
//
// The C library namespaces all symbols as sqisign_lvl1_ref_*
// ============================================================================

extern "C" {
    /// Generate a SQIsign key pair (Level 1).
    ///
    /// # Safety
    /// - `pk` must point to at least `CRYPTO_PUBLICKEYBYTES` (65) bytes
    /// - `sk` must point to at least `CRYPTO_SECRETKEYBYTES` (353) bytes
    ///
    /// Returns 0 on success.
    #[link_name = "sqisign_lvl1_ref_crypto_sign_keypair"]
    pub fn crypto_sign_keypair(pk: *mut c_uchar, sk: *mut c_uchar) -> c_int;

    /// Sign a message (Level 1). Produces signature || message.
    ///
    /// # Safety
    /// - `sm` must point to at least `mlen + CRYPTO_BYTES` bytes
    /// - `smlen` must be a valid pointer
    /// - `m` must point to `mlen` bytes
    /// - `sk` must point to `CRYPTO_SECRETKEYBYTES` bytes
    ///
    /// Returns 0 on success.
    #[link_name = "sqisign_lvl1_ref_crypto_sign"]
    pub fn crypto_sign(
        sm: *mut c_uchar,
        smlen: *mut c_ulonglong,
        m: *const c_uchar,
        mlen: c_ulonglong,
        sk: *const c_uchar,
    ) -> c_int;

    /// Verify and open a signed message (Level 1).
    ///
    /// If verification succeeds, the original message is written to `m`.
    ///
    /// # Safety
    /// - `m` must point to at least `smlen` bytes
    /// - `mlen` must be a valid pointer
    /// - `sm` must point to `smlen` bytes
    /// - `pk` must point to `CRYPTO_PUBLICKEYBYTES` bytes
    ///
    /// Returns 0 on success (valid signature), non-zero on failure.
    #[link_name = "sqisign_lvl1_ref_crypto_sign_open"]
    pub fn crypto_sign_open(
        m: *mut c_uchar,
        mlen: *mut c_ulonglong,
        sm: *const c_uchar,
        smlen: c_ulonglong,
        pk: *const c_uchar,
    ) -> c_int;
}

// ============================================================================
// Direct (non-NIST) API — includes detached verify
// ============================================================================

extern "C" {
    /// Generate a SQIsign key pair (direct API, Level 1).
    #[link_name = "sqisign_lvl1_ref_sqisign_keypair"]
    pub fn sqisign_keypair(pk: *mut c_uchar, sk: *mut c_uchar) -> c_int;

    /// Sign a message (direct API, Level 1). Produces signature || message.
    #[link_name = "sqisign_lvl1_ref_sqisign_sign"]
    pub fn sqisign_sign(
        sm: *mut c_uchar,
        smlen: *mut c_ulonglong,
        m: *const c_uchar,
        mlen: c_ulonglong,
        sk: *const c_uchar,
    ) -> c_int;

    /// Open a signed message (direct API, Level 1).
    #[link_name = "sqisign_lvl1_ref_sqisign_open"]
    pub fn sqisign_open(
        m: *mut c_uchar,
        mlen: *mut c_ulonglong,
        sm: *const c_uchar,
        smlen: c_ulonglong,
        pk: *const c_uchar,
    ) -> c_int;

    /// Verify a detached signature (Level 1).
    ///
    /// # Safety
    /// - `m` must point to `mlen` bytes (the original message)
    /// - `sig` must point to `siglen` bytes (the signature, without message)
    /// - `pk` must point to `CRYPTO_PUBLICKEYBYTES` bytes
    ///
    /// Returns 0 if the signature is valid, 1 otherwise.
    #[link_name = "sqisign_lvl1_ref_sqisign_verify"]
    pub fn sqisign_verify(
        m: *const c_uchar,
        mlen: c_ulonglong,
        sig: *const c_uchar,
        siglen: c_ulonglong,
        pk: *const c_uchar,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_roundtrip() {
        unsafe {
            let mut pk = [0u8; CRYPTO_PUBLICKEYBYTES];
            let mut sk = [0u8; CRYPTO_SECRETKEYBYTES];

            let ret = crypto_sign_keypair(pk.as_mut_ptr(), sk.as_mut_ptr());
            assert_eq!(ret, 0, "keypair generation failed");

            // Public key should not be all zeros
            assert!(pk.iter().any(|&b| b != 0), "public key is all zeros");
        }
    }

    #[test]
    fn test_sign_and_open() {
        unsafe {
            let mut pk = [0u8; CRYPTO_PUBLICKEYBYTES];
            let mut sk = [0u8; CRYPTO_SECRETKEYBYTES];

            let ret = crypto_sign_keypair(pk.as_mut_ptr(), sk.as_mut_ptr());
            assert_eq!(ret, 0);

            let message = b"Hello, SQIsign!";
            let mlen = message.len() as c_ulonglong;

            // Sign: output is signature || message
            let mut sm = vec![0u8; CRYPTO_BYTES + message.len()];
            let mut smlen: c_ulonglong = 0;

            let ret = crypto_sign(
                sm.as_mut_ptr(),
                &mut smlen,
                message.as_ptr(),
                mlen,
                sk.as_ptr(),
            );
            assert_eq!(ret, 0, "signing failed");
            assert_eq!(smlen as usize, CRYPTO_BYTES + message.len());

            // Open: verify and recover message
            let mut m_out = vec![0u8; smlen as usize];
            let mut m_out_len: c_ulonglong = 0;

            let ret = crypto_sign_open(
                m_out.as_mut_ptr(),
                &mut m_out_len,
                sm.as_ptr(),
                smlen,
                pk.as_ptr(),
            );
            assert_eq!(ret, 0, "verification failed");
            assert_eq!(&m_out[..m_out_len as usize], message);
        }
    }

    #[test]
    fn test_detached_verify() {
        unsafe {
            let mut pk = [0u8; CRYPTO_PUBLICKEYBYTES];
            let mut sk = [0u8; CRYPTO_SECRETKEYBYTES];

            let ret = crypto_sign_keypair(pk.as_mut_ptr(), sk.as_mut_ptr());
            assert_eq!(ret, 0);

            let message = b"Detached verification test";
            let mlen = message.len() as c_ulonglong;

            // Sign (produces sig || message)
            let mut sm = vec![0u8; CRYPTO_BYTES + message.len()];
            let mut smlen: c_ulonglong = 0;

            let ret = crypto_sign(
                sm.as_mut_ptr(),
                &mut smlen,
                message.as_ptr(),
                mlen,
                sk.as_ptr(),
            );
            assert_eq!(ret, 0);

            // Extract just the signature (first CRYPTO_BYTES bytes)
            let sig = &sm[..CRYPTO_BYTES];

            // Detached verify
            let ret = sqisign_verify(
                message.as_ptr(),
                mlen,
                sig.as_ptr(),
                CRYPTO_BYTES as c_ulonglong,
                pk.as_ptr(),
            );
            assert_eq!(ret, 0, "detached verification failed");
        }
    }

    #[test]
    fn test_invalid_signature_rejected() {
        unsafe {
            let mut pk = [0u8; CRYPTO_PUBLICKEYBYTES];
            let mut sk = [0u8; CRYPTO_SECRETKEYBYTES];

            let ret = crypto_sign_keypair(pk.as_mut_ptr(), sk.as_mut_ptr());
            assert_eq!(ret, 0);

            let message = b"Test message";

            // Create a forged signature (all zeros)
            let mut forged_sm = vec![0u8; CRYPTO_BYTES + message.len()];
            forged_sm[CRYPTO_BYTES..].copy_from_slice(message);
            let smlen = forged_sm.len() as c_ulonglong;

            let mut m_out = vec![0u8; forged_sm.len()];
            let mut m_out_len: c_ulonglong = 0;

            let ret = crypto_sign_open(
                m_out.as_mut_ptr(),
                &mut m_out_len,
                forged_sm.as_ptr(),
                smlen,
                pk.as_ptr(),
            );
            assert_ne!(ret, 0, "forged signature should be rejected");
        }
    }
}
