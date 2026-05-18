//! Adapter boundary for canonical epoch public-input scalar encoding.
//!
//! The ADR table contains eight 32-bit limbs for each 32-byte root/hash plus four
//! scalar metadata fields. To keep the accepted 28-scalar boundary, this adapter
//! treats `signature_count` as the final scalar and deliberately omits
//! `epoch_end_timestamp` from the canonical vector.

use crate::EpochPublicInputs;
use anyhow::{anyhow, bail, Result};
use q_lattice_guard::Scalar;

/// Canonical number of scalar public inputs accepted by the IVC adapter.
pub const EPOCH_PUBLIC_INPUTS_LEN: usize = 28;

const HASH_LIMBS: usize = 8;
const PREVIOUS_STATE_ROOT_RANGE: std::ops::Range<usize> = 0..8;
const CURRENT_STATE_ROOT_RANGE: std::ops::Range<usize> = 8..16;
const EPOCH_INDEX: usize = 16;
const HEIGHT_RANGE_START_INDEX: usize = 17;
const HEIGHT_RANGE_END_INDEX: usize = 18;
const VALIDATOR_SET_HASH_RANGE: std::ops::Range<usize> = 19..27;
const SIGNATURE_COUNT_INDEX: usize = 27;

/// Encode epoch public inputs in the canonical ADR order.
///
/// The fixed 28-scalar encoding is:
/// - 0..8: previous state root, little-endian `u32` limbs
/// - 8..16: current state root, little-endian `u32` limbs
/// - 16: epoch
/// - 17: starting block height
/// - 18: ending block height
/// - 19..27: validator-set hash, little-endian `u32` limbs
/// - 27: signature count
///
/// `epoch_end_timestamp` is intentionally not encoded because including both it
/// and `signature_count` would require 29 scalars.
pub fn encode_public_inputs(inputs: &EpochPublicInputs) -> Vec<Scalar> {
    let mut scalars = Vec::with_capacity(EPOCH_PUBLIC_INPUTS_LEN);

    encode_limbs(&mut scalars, &inputs.previous_state_root);
    encode_limbs(&mut scalars, &inputs.current_state_root);

    scalars.push(inputs.epoch);
    scalars.push(inputs.height_range.0);
    scalars.push(inputs.height_range.1);

    encode_limbs(&mut scalars, &inputs.validator_set_hash);
    scalars.push(inputs.signature_count as Scalar);

    debug_assert_eq!(scalars.len(), EPOCH_PUBLIC_INPUTS_LEN);
    scalars
}

/// Decode canonical ADR-order public input scalars into epoch public inputs.
///
/// This rejects non-canonical vector lengths and root/hash limbs that cannot be
/// represented as `u32` little-endian limbs. Since the 28-scalar encoding omits
/// `epoch_end_timestamp`, decoded values use `0` for that field.
pub fn decode_public_inputs(scalars: &[Scalar]) -> Result<EpochPublicInputs> {
    if scalars.len() != EPOCH_PUBLIC_INPUTS_LEN {
        bail!(
            "expected {EPOCH_PUBLIC_INPUTS_LEN} epoch public input scalars, got {}",
            scalars.len()
        );
    }

    let previous_state_root =
        decode_limbs(&scalars[PREVIOUS_STATE_ROOT_RANGE], "previous_state_root")?;
    let current_state_root =
        decode_limbs(&scalars[CURRENT_STATE_ROOT_RANGE], "current_state_root")?;
    let validator_set_hash =
        decode_limbs(&scalars[VALIDATOR_SET_HASH_RANGE], "validator_set_hash")?;

    let signature_count = scalars[SIGNATURE_COUNT_INDEX];
    if signature_count > u32::MAX as Scalar {
        bail!("signature_count exceeds u32::MAX: {signature_count}");
    }

    Ok(EpochPublicInputs {
        previous_state_root,
        current_state_root,
        epoch: scalars[EPOCH_INDEX],
        height_range: (
            scalars[HEIGHT_RANGE_START_INDEX],
            scalars[HEIGHT_RANGE_END_INDEX],
        ),
        validator_set_hash,
        signature_count: signature_count as u32,
        epoch_end_timestamp: 0,
    })
}

fn encode_limbs(scalars: &mut Vec<Scalar>, bytes: &[u8; 32]) {
    for chunk in bytes.chunks_exact(4) {
        let limb = u32::from_le_bytes(chunk.try_into().expect("chunks_exact yields 4 bytes"));
        scalars.push(limb as Scalar);
    }
}

fn decode_limbs(scalars: &[Scalar], field_name: &str) -> Result<[u8; 32]> {
    if scalars.len() != HASH_LIMBS {
        return Err(anyhow!(
            "{field_name} expected {HASH_LIMBS} limbs, got {}",
            scalars.len()
        ));
    }

    let mut bytes = [0u8; 32];
    for (index, scalar) in scalars.iter().copied().enumerate() {
        if scalar > u32::MAX as Scalar {
            bail!("{field_name} limb {index} exceeds u32::MAX: {scalar}");
        }

        bytes[index * 4..(index + 1) * 4].copy_from_slice(&(scalar as u32).to_le_bytes());
    }

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arbitrary_inputs() -> EpochPublicInputs {
        EpochPublicInputs {
            previous_state_root: [
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
                0x1c, 0x1d, 0x1e, 0x1f,
            ],
            current_state_root: [
                0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d,
                0x1e, 0x0f, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44,
                0x33, 0x22, 0x11, 0x00,
            ],
            epoch: 42,
            height_range: (1_000, 2_000),
            validator_set_hash: [
                0xaa, 0xbb, 0xcc, 0xdd, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0,
                0xb0, 0xc0, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98,
                0x76, 0x54, 0x32, 0x10,
            ],
            signature_count: 67,
            epoch_end_timestamp: 0,
        }
    }

    #[test]
    fn round_trips_arbitrary_encodable_epoch_public_inputs() {
        let inputs = arbitrary_inputs();
        let scalars = encode_public_inputs(&inputs);

        assert_eq!(scalars.len(), EPOCH_PUBLIC_INPUTS_LEN);
        assert_eq!(scalars[0], 0x0302_0100);
        assert_eq!(scalars[8], 0xc3d2_e1f0);
        assert_eq!(scalars[16], inputs.epoch);
        assert_eq!(scalars[17], inputs.height_range.0);
        assert_eq!(scalars[18], inputs.height_range.1);
        assert_eq!(scalars[19], 0xddcc_bbaa);
        assert_eq!(scalars[27], inputs.signature_count as Scalar);

        let decoded = decode_public_inputs(&scalars).expect("valid adapter scalars decode");
        assert_eq!(decoded, inputs);
    }

    #[test]
    fn reject_length_not_equal_to_28() {
        let mut scalars = encode_public_inputs(&arbitrary_inputs());

        scalars.pop();
        let short_error = decode_public_inputs(&scalars).unwrap_err().to_string();
        assert!(short_error.contains("expected 28 epoch public input scalars, got 27"));

        scalars.push(67);
        scalars.push(0);
        let long_error = decode_public_inputs(&scalars).unwrap_err().to_string();
        assert!(long_error.contains("expected 28 epoch public input scalars, got 29"));
    }

    #[test]
    fn reject_overflowing_root_or_hash_limb_values() {
        for (index, field_name) in [
            (0, "previous_state_root"),
            (8, "current_state_root"),
            (19, "validator_set_hash"),
        ] {
            let mut scalars = encode_public_inputs(&arbitrary_inputs());
            scalars[index] = u32::MAX as Scalar + 1;

            let error = decode_public_inputs(&scalars).unwrap_err().to_string();
            assert!(
                error.contains(field_name),
                "error {error:?} should name {field_name}"
            );
            assert!(
                error.contains("exceeds u32::MAX"),
                "error {error:?} should describe limb overflow"
            );
        }
    }
}
