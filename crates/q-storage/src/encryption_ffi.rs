// Q-NarwhalKnight RocksDB Encryption FFI
// v1.0.40-beta: FFI bindings for C++ EncryptionProvider
//
// ARCHITECTURE:
// - Expose Rust encryption functions to C++
// - Manage memory safety across FFI boundary
// - Provide C-compatible error codes

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::slice;
use anyhow::Result;
use tracing::{debug, error, warn};

use crate::encryption::{EncryptionManager, ProtectedKey};
use crate::encryption_stream::{AesCtrStream, EncryptedFileHeader};

/// Error codes for C++ interop
const FFI_SUCCESS: c_int = 0;
const FFI_ERROR_NULL_PTR: c_int = -1;
const FFI_ERROR_INVALID_ARG: c_int = -2;
const FFI_ERROR_ENCRYPTION_FAILED: c_int = -3;
const FFI_ERROR_DECRYPTION_FAILED: c_int = -4;
const FFI_ERROR_KEY_DERIVATION_FAILED: c_int = -5;

/// Opaque handle to AesCtrStream for C++
///
/// SAFETY: C++ must call qnk_destroy_cipher_stream() to free
struct CipherStreamHandle {
    stream: AesCtrStream,
}

/// Create AES-CTR cipher stream
///
/// PARAMETERS:
/// - file_id: Unique SST file identifier
/// - cf_id: Column family ID
/// - file_key: 32-byte encryption key
/// - key_len: Must be 32
/// - header: Optional existing header (null for new files)
/// - header_len: Header size (0 for new files, 64 for existing)
///
/// RETURNS: Opaque pointer to cipher stream (null on error)
///
/// SAFETY: Caller must call qnk_destroy_cipher_stream() to free
#[no_mangle]
pub extern "C" fn qnk_create_cipher_stream(
    file_id: u64,
    cf_id: u32,
    file_key: *const u8,
    key_len: usize,
    header: *const u8,
    header_len: usize,
) -> *mut CipherStreamHandle {

    // Validate inputs
    if file_key.is_null() || key_len != 32 {
        error!("FFI: Invalid file_key (null or wrong size)");
        return ptr::null_mut();
    }

    // Convert file_key to ProtectedKey
    let key_bytes = unsafe {
        let slice = slice::from_raw_parts(file_key, key_len);
        let mut arr = [0u8; 32];
        arr.copy_from_slice(slice);
        arr
    };

    let protected_key = match ProtectedKey::new(key_bytes) {
        Ok(key) => key,
        Err(e) => {
            error!("FFI: Failed to create ProtectedKey: {}", e);
            return ptr::null_mut();
        }
    };

    // Create cipher stream
    let result = if header.is_null() || header_len == 0 {
        // New file (encryption mode)
        AesCtrStream::new(protected_key, file_id, cf_id)
    } else {
        // Existing file (decryption mode)
        if header_len < 64 {
            error!("FFI: Header too small: {} bytes", header_len);
            return ptr::null_mut();
        }

        let header_bytes = unsafe {
            slice::from_raw_parts(header, header_len)
        };

        let file_header = match EncryptedFileHeader::from_bytes(header_bytes) {
            Ok(h) => h,
            Err(e) => {
                error!("FFI: Invalid file header: {}", e);
                return ptr::null_mut();
            }
        };

        AesCtrStream::from_header(protected_key, file_header)
    };

    match result {
        Ok(stream) => {
            let handle = Box::new(CipherStreamHandle { stream });
            Box::into_raw(handle)
        }
        Err(e) => {
            error!("FFI: Failed to create cipher stream: {}", e);
            ptr::null_mut()
        }
    }
}

/// Destroy cipher stream and free memory
///
/// SAFETY: stream must be a valid pointer from qnk_create_cipher_stream()
#[no_mangle]
pub extern "C" fn qnk_destroy_cipher_stream(stream: *mut CipherStreamHandle) {
    if stream.is_null() {
        return;
    }

    unsafe {
        let _ = Box::from_raw(stream);
        // Drop will clean up ProtectedKey (zeroize + munlock)
    }

    debug!("FFI: Cipher stream destroyed");
}

/// Encrypt data at given offset (in-place)
///
/// PARAMETERS:
/// - stream: Cipher stream handle
/// - data: Data buffer (will be modified in-place)
/// - len: Data length
/// - offset: File offset (must be multiple of 4096)
///
/// RETURNS: 0 on success, negative on error
///
/// SAFETY: data must be valid for len bytes
#[no_mangle]
pub extern "C" fn qnk_encrypt_at_offset(
    stream: *mut CipherStreamHandle,
    data: *mut u8,
    len: usize,
    offset: u64,
) -> c_int {

    if stream.is_null() || data.is_null() {
        return FFI_ERROR_NULL_PTR;
    }

    let handle = unsafe { &mut *stream };
    let data_slice = unsafe { slice::from_raw_parts_mut(data, len) };

    match handle.stream.encrypt_at_offset(data_slice, offset) {
        Ok(()) => FFI_SUCCESS,
        Err(e) => {
            error!("FFI: Encryption failed: {}", e);
            FFI_ERROR_ENCRYPTION_FAILED
        }
    }
}

/// Decrypt data at given offset (in-place)
///
/// PARAMETERS:
/// - stream: Cipher stream handle
/// - data: Data buffer (will be modified in-place)
/// - len: Data length
/// - offset: File offset (must be multiple of 4096)
///
/// RETURNS: 0 on success, negative on error
///
/// SAFETY: data must be valid for len bytes
#[no_mangle]
pub extern "C" fn qnk_decrypt_at_offset(
    stream: *mut CipherStreamHandle,
    data: *mut u8,
    len: usize,
    offset: u64,
) -> c_int {

    if stream.is_null() || data.is_null() {
        return FFI_ERROR_NULL_PTR;
    }

    let handle = unsafe { &mut *stream };
    let data_slice = unsafe { slice::from_raw_parts_mut(data, len) };

    match handle.stream.decrypt_at_offset(data_slice, offset) {
        Ok(()) => FFI_SUCCESS,
        Err(e) => {
            error!("FFI: Decryption failed: {}", e);
            FFI_ERROR_DECRYPTION_FAILED
        }
    }
}

/// Get file header from cipher stream
///
/// PARAMETERS:
/// - stream: Cipher stream handle
/// - header_out: Buffer to write header (must be at least 64 bytes)
/// - header_size: Size of header_out buffer
///
/// RETURNS: 0 on success, negative on error
///
/// SAFETY: header_out must be valid for header_size bytes
#[no_mangle]
pub extern "C" fn qnk_get_file_header(
    stream: *mut CipherStreamHandle,
    header_out: *mut u8,
    header_size: usize,
) -> c_int {

    if stream.is_null() || header_out.is_null() {
        return FFI_ERROR_NULL_PTR;
    }

    if header_size < 64 {
        error!("FFI: Header buffer too small: {} bytes", header_size);
        return FFI_ERROR_INVALID_ARG;
    }

    let handle = unsafe { &*stream };
    let header_bytes = handle.stream.header().to_bytes();

    unsafe {
        ptr::copy_nonoverlapping(header_bytes.as_ptr(), header_out, 64);
    }

    FFI_SUCCESS
}

/// Derive file encryption key
///
/// PARAMETERS:
/// - keys_file_path: Path to keys file (C string)
/// - passphrase: Encryption passphrase (C string)
/// - file_id: Unique file identifier
/// - cf_id: Column family ID
/// - key_out: Buffer to write key (must be at least 32 bytes)
/// - key_size: Size of key_out buffer
///
/// RETURNS: 0 on success, negative on error
///
/// SAFETY: All pointers must be valid C strings/buffers
#[no_mangle]
pub extern "C" fn qnk_derive_file_key(
    keys_file_path: *const c_char,
    passphrase: *const c_char,
    file_id: u64,
    cf_id: u32,
    key_out: *mut u8,
    key_size: usize,
) -> c_int {

    if keys_file_path.is_null() || passphrase.is_null() || key_out.is_null() {
        return FFI_ERROR_NULL_PTR;
    }

    if key_size < 32 {
        error!("FFI: Key buffer too small: {} bytes", key_size);
        return FFI_ERROR_INVALID_ARG;
    }

    // Convert C strings to Rust
    let keys_file = unsafe {
        match CStr::from_ptr(keys_file_path).to_str() {
            Ok(s) => s,
            Err(e) => {
                error!("FFI: Invalid keys_file_path UTF-8: {}", e);
                return FFI_ERROR_INVALID_ARG;
            }
        }
    };

    let passphrase_str = unsafe {
        match CStr::from_ptr(passphrase).to_str() {
            Ok(s) => s,
            Err(e) => {
                error!("FFI: Invalid passphrase UTF-8: {}", e);
                return FFI_ERROR_INVALID_ARG;
            }
        }
    };

    // Load encryption manager
    let mgr = match EncryptionManager::from_passphrase(passphrase_str, keys_file.as_ref()) {
        Ok(m) => m,
        Err(e) => {
            error!("FFI: Failed to load encryption manager: {}", e);
            return FFI_ERROR_KEY_DERIVATION_FAILED;
        }
    };

    // Derive file key
    let file_key = match mgr.derive_file_key(file_id, cf_id) {
        Ok(key) => key,
        Err(e) => {
            error!("FFI: Failed to derive file key: {}", e);
            return FFI_ERROR_KEY_DERIVATION_FAILED;
        }
    };

    // Copy key to output buffer
    unsafe {
        ptr::copy_nonoverlapping(file_key.as_bytes().as_ptr(), key_out, 32);
    }

    debug!("FFI: Derived file key (file_id={}, cf_id={})", file_id, cf_id);

    FFI_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_ffi_create_destroy_cipher_stream() {
        let key = [42u8; 32];

        let stream = qnk_create_cipher_stream(
            1,      // file_id
            0,      // cf_id
            key.as_ptr(),
            key.len(),
            ptr::null(),  // No header (new file)
            0,
        );

        assert!(!stream.is_null(), "Stream creation failed");

        qnk_destroy_cipher_stream(stream);
    }

    #[test]
    fn test_ffi_encrypt_decrypt_roundtrip() {
        let key = [42u8; 32];

        let stream = qnk_create_cipher_stream(
            1, 0,
            key.as_ptr(), key.len(),
            ptr::null(), 0,
        );
        assert!(!stream.is_null());

        let mut data = b"Hello, FFI encryption!".to_vec();
        let original = data.clone();

        // Encrypt
        let result = qnk_encrypt_at_offset(stream, data.as_mut_ptr(), data.len(), 0);
        assert_eq!(result, FFI_SUCCESS);
        assert_ne!(data, original);

        // Decrypt
        let result = qnk_decrypt_at_offset(stream, data.as_mut_ptr(), data.len(), 0);
        assert_eq!(result, FFI_SUCCESS);
        assert_eq!(data, original);

        qnk_destroy_cipher_stream(stream);
    }

    #[test]
    fn test_ffi_get_header() {
        let key = [42u8; 32];

        let stream = qnk_create_cipher_stream(
            123, 5,
            key.as_ptr(), key.len(),
            ptr::null(), 0,
        );
        assert!(!stream.is_null());

        let mut header = [0u8; 64];
        let result = qnk_get_file_header(stream, header.as_mut_ptr(), header.len());
        assert_eq!(result, FFI_SUCCESS);

        // Verify magic number
        assert_eq!(&header[0..8], b"QNKEnc01");

        qnk_destroy_cipher_stream(stream);
    }

    #[test]
    fn test_ffi_derive_file_key() {
        let temp_dir = TempDir::new().unwrap();
        let keys_file = temp_dir.path().join("test.keys");

        // Create encryption manager
        let mgr = EncryptionManager::create_new("test-passphrase", &keys_file).unwrap();
        drop(mgr);

        // Convert to C strings
        let keys_path = CString::new(keys_file.to_str().unwrap()).unwrap();
        let pass = CString::new("test-passphrase").unwrap();

        let mut key = [0u8; 32];
        let result = qnk_derive_file_key(
            keys_path.as_ptr(),
            pass.as_ptr(),
            1, 0,
            key.as_mut_ptr(), key.len(),
        );

        assert_eq!(result, FFI_SUCCESS);
        assert_ne!(key, [0u8; 32], "Key should not be all zeros");
    }

    #[test]
    fn test_ffi_null_pointer_handling() {
        // Test null stream
        let result = qnk_encrypt_at_offset(ptr::null_mut(), ptr::null_mut(), 0, 0);
        assert_eq!(result, FFI_ERROR_NULL_PTR);

        // Test null key
        let stream = qnk_create_cipher_stream(1, 0, ptr::null(), 32, ptr::null(), 0);
        assert!(stream.is_null());
    }
}
