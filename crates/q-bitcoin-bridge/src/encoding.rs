use crate::NodeAdvertisement;
/// Direct encoding/decoding for Q-Knight advertisements in Bitcoin OP_RETURN data
///
/// This module handles the serialization and deserialization of node advertisements
/// for embedding in Bitcoin transactions using OP_RETURN outputs.
use anyhow::{anyhow, Result};
use serde_json;

/// Maximum size for OP_RETURN data (Bitcoin consensus rule)
pub const MAX_OP_RETURN_SIZE: usize = 80;

/// Q-Knight protocol identifier for Bitcoin-embedded data
pub const PROTOCOL_MAGIC: &[u8] = b"QKNT"; // Q-Knight identifier

/// Encode a node advertisement for Bitcoin OP_RETURN embedding
pub async fn encode_direct(advertisement: &NodeAdvertisement) -> Result<Vec<u8>> {
    // Serialize advertisement to compact JSON
    let json_data = serde_json::to_vec(advertisement)?;

    // Create protocol envelope
    let mut encoded_data = Vec::new();
    encoded_data.extend_from_slice(PROTOCOL_MAGIC);
    encoded_data.push(0x01); // Version byte
    encoded_data.extend_from_slice(&json_data);

    // Check size constraint
    if encoded_data.len() > MAX_OP_RETURN_SIZE {
        return Err(anyhow!(
            "Advertisement too large: {} bytes (max: {})",
            encoded_data.len(),
            MAX_OP_RETURN_SIZE
        ));
    }

    Ok(encoded_data)
}

/// Decode a node advertisement from Bitcoin OP_RETURN data
pub async fn decode_direct(data: &[u8]) -> Result<NodeAdvertisement> {
    // Check minimum size
    if data.len() < 5 {
        return Err(anyhow!("Data too short for Q-Knight advertisement"));
    }

    // Verify protocol magic
    if &data[0..4] != PROTOCOL_MAGIC {
        return Err(anyhow!("Invalid protocol magic"));
    }

    // Check version
    let version = data[4];
    if version != 0x01 {
        return Err(anyhow!("Unsupported protocol version: {}", version));
    }

    // Extract and deserialize JSON data
    let json_data = &data[5..];
    let advertisement: NodeAdvertisement = serde_json::from_slice(json_data)?;

    Ok(advertisement)
}

/// Encode advertisement with compression for larger data
pub async fn encode_compressed(advertisement: &NodeAdvertisement) -> Result<Vec<u8>> {
    use std::io::Write;

    // Serialize to JSON
    let json_data = serde_json::to_vec(advertisement)?;

    // Compress using LZ4
    let compressed = lz4_flex::compress(&json_data);

    // Create protocol envelope
    let mut encoded_data = Vec::new();
    encoded_data.extend_from_slice(PROTOCOL_MAGIC);
    encoded_data.push(0x02); // Compressed version
    encoded_data.extend_from_slice(&compressed);

    // Check size constraint
    if encoded_data.len() > MAX_OP_RETURN_SIZE {
        return Err(anyhow!(
            "Compressed advertisement still too large: {} bytes",
            encoded_data.len()
        ));
    }

    Ok(encoded_data)
}

/// Decode compressed advertisement
pub async fn decode_compressed(data: &[u8]) -> Result<NodeAdvertisement> {
    // Check minimum size
    if data.len() < 5 {
        return Err(anyhow!("Data too short for compressed advertisement"));
    }

    // Verify protocol magic
    if &data[0..4] != PROTOCOL_MAGIC {
        return Err(anyhow!("Invalid protocol magic"));
    }

    // Check version
    let version = data[4];
    if version != 0x02 {
        return Err(anyhow!("Not a compressed advertisement"));
    }

    // Extract and decompress data
    let compressed_data = &data[5..];
    let json_data = lz4_flex::decompress(compressed_data, 1024)?; // Max 1KB decompressed

    // Deserialize JSON
    let advertisement: NodeAdvertisement = serde_json::from_slice(&json_data)?;

    Ok(advertisement)
}

/// Create a compact advertisement for maximum Bitcoin efficiency
pub fn create_compact_advertisement(
    node_id: &[u8; 32],
    onion_address: &str,
    port: u16,
    capabilities: &[String],
) -> NodeAdvertisement {
    use chrono::Utc;

    // Use abbreviated capability codes
    let compact_capabilities = capabilities
        .iter()
        .map(|cap| match cap.as_str() {
            "dag-consensus" => "DAG".to_string(),
            "quantum-ready" => "QR".to_string(),
            "validator" => "VAL".to_string(),
            "relay" => "REL".to_string(),
            _ => cap.clone(),
        })
        .collect();

    NodeAdvertisement {
        node_id: *node_id,
        onion_address: onion_address.to_string(),
        port,
        protocol_version: "qk/0.1".to_string(), // Abbreviated
        capabilities: compact_capabilities,
        signature: vec![], // Will be added later
        timestamp: Utc::now(),
        expires_at: Utc::now() + chrono::Duration::hours(1),
    }
}

/// Batch encode multiple advertisements for efficiency
pub async fn encode_batch(advertisements: &[NodeAdvertisement]) -> Result<Vec<Vec<u8>>> {
    let mut encoded_ads = Vec::new();

    for ad in advertisements {
        // Try compressed encoding first
        if let Ok(compressed) = encode_compressed(ad).await {
            encoded_ads.push(compressed);
        } else {
            // Fall back to direct encoding
            let direct = encode_direct(ad).await?;
            encoded_ads.push(direct);
        }
    }

    Ok(encoded_ads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[tokio::test]
    async fn test_direct_encoding() {
        let advertisement = NodeAdvertisement {
            node_id: [42u8; 32],
            onion_address: "test123.onion".to_string(),
            port: 8333,
            protocol_version: "qk/0.1".to_string(),
            capabilities: vec!["DAG".to_string(), "QR".to_string()],
            signature: vec![],
            timestamp: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::hours(1),
        };

        let encoded = encode_direct(&advertisement).await.unwrap();
        assert!(encoded.len() <= MAX_OP_RETURN_SIZE);

        let decoded = decode_direct(&encoded).await.unwrap();
        assert_eq!(advertisement.node_id, decoded.node_id);
        assert_eq!(advertisement.onion_address, decoded.onion_address);
        assert_eq!(advertisement.port, decoded.port);
    }

    #[tokio::test]
    async fn test_compressed_encoding() {
        let advertisement = NodeAdvertisement {
            node_id: [42u8; 32],
            onion_address: "verylongonionaddressfortesting.onion".to_string(),
            port: 8333,
            protocol_version: "q-knight/0.1.0".to_string(),
            capabilities: vec![
                "dag-consensus".to_string(),
                "quantum-ready".to_string(),
                "validator".to_string(),
            ],
            signature: vec![0u8; 64], // Dummy signature
            timestamp: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::hours(1),
        };

        let compressed = encode_compressed(&advertisement).await.unwrap();
        let direct = encode_direct(&advertisement).await;

        // Compressed should be smaller for larger data
        if direct.is_ok() {
            assert!(compressed.len() <= direct.unwrap().len());
        }

        let decoded = decode_compressed(&compressed).await.unwrap();
        assert_eq!(advertisement.node_id, decoded.node_id);
        assert_eq!(advertisement.onion_address, decoded.onion_address);
    }

    #[test]
    fn test_compact_advertisement() {
        let node_id = [1u8; 32];
        let ad = create_compact_advertisement(
            &node_id,
            "test.onion",
            8333,
            &["dag-consensus".to_string(), "quantum-ready".to_string()],
        );

        assert_eq!(ad.node_id, node_id);
        assert_eq!(ad.capabilities, vec!["DAG".to_string(), "QR".to_string()]);
        assert_eq!(ad.protocol_version, "qk/0.1");
    }

    #[tokio::test]
    async fn test_batch_encoding() {
        let advertisements = vec![
            create_compact_advertisement(
                &[1u8; 32],
                "node1.onion",
                8333,
                &["dag-consensus".to_string()],
            ),
            create_compact_advertisement(
                &[2u8; 32],
                "node2.onion",
                8333,
                &["validator".to_string()],
            ),
        ];

        let encoded_batch = encode_batch(&advertisements).await.unwrap();
        assert_eq!(encoded_batch.len(), 2);

        for encoded in &encoded_batch {
            assert!(encoded.len() <= MAX_OP_RETURN_SIZE);
        }
    }

    #[tokio::test]
    async fn test_invalid_data_handling() {
        // Test invalid magic
        let invalid_magic = b"FAKE\x01test";
        assert!(decode_direct(invalid_magic).await.is_err());

        // Test invalid version
        let invalid_version = b"QKNT\xFF{}";
        assert!(decode_direct(invalid_version).await.is_err());

        // Test too short data
        let too_short = b"QK";
        assert!(decode_direct(too_short).await.is_err());
    }
}
