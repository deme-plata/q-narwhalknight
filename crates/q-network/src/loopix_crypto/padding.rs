/// Padding utilities for Loopix protocol
/// Implements fixed-size cell padding for traffic analysis resistance

use ring::rand::{SecureRandom, SystemRandom};

/// Fixed cell size for all Loopix messages (512 bytes)
pub const CELL_SIZE: usize = 512;

/// Pad payload to fixed cell size with secure random padding
pub fn pad_to_cell(payload: &[u8]) -> Result<Vec<u8>, anyhow::Error> {
    if payload.len() > CELL_SIZE - 2 {
        return Err(anyhow::anyhow!("Payload too large for cell"));
    }

    let mut cell = vec![0u8; CELL_SIZE];

    // Write length prefix (2 bytes)
    let len = payload.len() as u16;
    cell[0..2].copy_from_slice(&len.to_be_bytes());

    // Write payload
    cell[2..2 + payload.len()].copy_from_slice(payload);

    // Fill remainder with cryptographically secure random padding
    let rng = SystemRandom::new();
    rng.fill(&mut cell[2 + payload.len()..])
        .map_err(|_| anyhow::anyhow!("Failed to generate random padding"))?;

    Ok(cell)
}

/// Extract payload from padded cell
pub fn unpad_from_cell(cell: &[u8]) -> Result<Vec<u8>, anyhow::Error> {
    if cell.len() != CELL_SIZE {
        return Err(anyhow::anyhow!("Invalid cell size"));
    }

    // Read length prefix
    let payload_len = u16::from_be_bytes([cell[0], cell[1]]) as usize;

    if payload_len > CELL_SIZE - 2 {
        return Err(anyhow::anyhow!("Invalid payload length"));
    }

    // Extract payload
    let payload = cell[2..2 + payload_len].to_vec();

    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padding_and_unpadding() {
        let payload = b"test message for padding".to_vec();

        // Pad to cell
        let cell = pad_to_cell(&payload).unwrap();
        assert_eq!(cell.len(), CELL_SIZE);

        // Unpad from cell
        let extracted = unpad_from_cell(&cell).unwrap();
        assert_eq!(extracted, payload);
    }

    #[test]
    fn test_empty_payload() {
        let payload = vec![];

        let cell = pad_to_cell(&payload).unwrap();
        let extracted = unpad_from_cell(&cell).unwrap();
        assert_eq!(extracted, payload);
    }

    #[test]
    fn test_payload_too_large() {
        let payload = vec![0u8; CELL_SIZE]; // Too large

        let result = pad_to_cell(&payload);
        assert!(result.is_err());
    }
}