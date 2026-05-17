//! v9.1.0: PoW Relay Stamps — lightweight anti-spam proof-of-work on P2P messages.
//!
//! Each gossipsub message is stamped with a 2-byte nonce that, when appended to the
//! message data and hashed with BLAKE3, produces a hash with >= `DIFFICULTY` leading
//! zero bits.  Cost: ~10µs per legitimate message (256 avg BLAKE3 iterations for 8-bit
//! difficulty).  Spammers flooding millions of messages pay O(n) compute while nodes
//! verify in O(1).
//!
//! The stamp is prepended to the message payload: `[stamp_hi, stamp_lo, ...data]`.
//! Old nodes that don't understand stamps will see 2 extra bytes at the start of
//! the payload and silently fail deserialization (graceful degradation).  The stamp
//! feature is gated behind the `pow-stamp-v1` handshake capability.

/// Number of leading zero bits required in the BLAKE3 hash.
/// 8 bits = 256 average iterations ≈ 10µs on modern CPUs.
const DIFFICULTY: u32 = 8;

/// Stamp a message by finding a 2-byte nonce such that
/// `BLAKE3(data || nonce)` has `DIFFICULTY` leading zero bits.
///
/// Returns the 2-byte stamp.  If no valid stamp is found within 65536 iterations
/// (statistically near-impossible at 8-bit difficulty), returns `[0, 0]`.
#[inline]
pub fn stamp_message(data: &[u8]) -> [u8; 2] {
    // Pre-allocate a buffer: data || nonce (2 bytes)
    let mut buf = Vec::with_capacity(data.len() + 2);
    buf.extend_from_slice(data);
    buf.push(0);
    buf.push(0);

    let data_len = data.len();

    for nonce in 0u16..=u16::MAX {
        let bytes = nonce.to_le_bytes();
        buf[data_len] = bytes[0];
        buf[data_len + 1] = bytes[1];

        let hash = blake3::hash(&buf);
        if leading_zero_bits(hash.as_bytes()) >= DIFFICULTY {
            return bytes;
        }
    }

    // Fallback (should never happen with 8-bit difficulty)
    [0, 0]
}

/// Verify that a stamp is valid for the given data.
///
/// Returns `true` if `BLAKE3(data || stamp)` has `DIFFICULTY` leading zero bits.
#[inline]
pub fn verify_stamp(data: &[u8], stamp: &[u8; 2]) -> bool {
    let mut buf = Vec::with_capacity(data.len() + 2);
    buf.extend_from_slice(data);
    buf.extend_from_slice(stamp);

    let hash = blake3::hash(&buf);
    leading_zero_bits(hash.as_bytes()) >= DIFFICULTY
}

/// Prepend a PoW stamp to message data.
/// Returns a new Vec with `[stamp_hi, stamp_lo, ...data]`.
pub fn stamp_and_prepend(data: &[u8]) -> Vec<u8> {
    let stamp = stamp_message(data);
    let mut out = Vec::with_capacity(2 + data.len());
    out.extend_from_slice(&stamp);
    out.extend_from_slice(data);
    out
}

/// Strip and verify a PoW stamp from received message data.
/// Returns `Some(payload)` if the stamp is valid, `None` if invalid or too short.
pub fn verify_and_strip(stamped_data: &[u8]) -> Option<&[u8]> {
    if stamped_data.len() < 3 {
        return None; // Too short to contain stamp + any data
    }
    let stamp: [u8; 2] = [stamped_data[0], stamped_data[1]];
    let payload = &stamped_data[2..];

    if verify_stamp(payload, &stamp) {
        Some(payload)
    } else {
        None
    }
}

/// Count leading zero bits in a byte slice.
#[inline]
fn leading_zero_bits(bytes: &[u8]) -> u32 {
    let mut count = 0u32;
    for &byte in bytes {
        if byte == 0 {
            count += 8;
        } else {
            count += byte.leading_zeros();
            break;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stamp_verify_roundtrip() {
        let data = b"Hello, quantum world!";
        let stamp = stamp_message(data);
        assert!(verify_stamp(data, &stamp), "Stamp should verify for original data");
    }

    #[test]
    fn test_stamp_reject_tampered() {
        let data = b"Original message";
        let stamp = stamp_message(data);
        let tampered = b"Tampered message";
        assert!(!verify_stamp(tampered, &stamp), "Stamp should reject tampered data");
    }

    #[test]
    fn test_stamp_and_strip_roundtrip() {
        let data = b"Block data with VDF proof";
        let stamped = stamp_and_prepend(data);
        assert_eq!(stamped.len(), data.len() + 2);

        let payload = verify_and_strip(&stamped).expect("Should verify successfully");
        assert_eq!(payload, data);
    }

    #[test]
    fn test_strip_rejects_short() {
        assert!(verify_and_strip(&[0, 1]).is_none());
        assert!(verify_and_strip(&[]).is_none());
    }

    #[test]
    fn test_leading_zero_bits() {
        assert_eq!(leading_zero_bits(&[0x00, 0x00, 0xFF]), 16);
        assert_eq!(leading_zero_bits(&[0x00, 0x80, 0xFF]), 8);
        assert_eq!(leading_zero_bits(&[0x01, 0xFF]), 7);
        assert_eq!(leading_zero_bits(&[0xFF]), 0);
        assert_eq!(leading_zero_bits(&[0x00]), 8);
    }

    #[test]
    fn test_stamp_performance() {
        // Verify that stamping takes a reasonable amount of time
        let data = b"Performance test data for BLAKE3 PoW stamp";
        let start = std::time::Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = stamp_message(data);
        }
        let elapsed = start.elapsed();
        let per_stamp = elapsed / iterations;
        // Should be well under 1ms (typically ~10µs for 8-bit difficulty)
        assert!(per_stamp.as_millis() < 10, "Stamping too slow: {:?} per stamp", per_stamp);
    }
}
