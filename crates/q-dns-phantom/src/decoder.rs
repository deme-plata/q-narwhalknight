use crate::{DNSPhantomMessage, crypto::SteganoEncryption};
use anyhow::{anyhow, Result};
use tracing::{debug, warn, error};
use std::collections::HashMap;

/// Message decoder for encrypted DNS steganographic payloads
pub struct MessageDecoder {
    encryptor: Option<SteganoEncryption>,
}

impl MessageDecoder {
    /// Create new decoder without encryption (legacy mode)
    pub fn new() -> Self {
        Self {
            encryptor: None,
        }
    }

    /// Create decoder with encryption for secure steganographic decoding
    pub fn new_with_encryption(node_id: &str) -> Result<Self> {
        let encryptor = SteganoEncryption::new(node_id)?;
        Ok(Self {
            encryptor: Some(encryptor),
        })
    }

    /// Decode DNS TXT records back into original messages
    pub async fn decode_from_txt_records(&self, txt_records: Vec<(String, String)>) -> Result<Vec<DNSPhantomMessage>> {
        let mut decoded_messages = Vec::new();

        // Group TXT records by message (based on sequence numbers and timestamps)
        let grouped_records = self.group_related_records(txt_records)?;

        for record_group in grouped_records {
            match self.decode_single_message(record_group).await {
                Ok(message) => decoded_messages.push(message),
                Err(e) => {
                    warn!("Failed to decode message group: {}", e);
                    continue;
                }
            }
        }

        debug!("🔓 DNS-Phantom: Successfully decoded {} messages from TXT records", decoded_messages.len());
        Ok(decoded_messages)
    }

    /// Group related TXT records that belong to the same steganographic message
    fn group_related_records(&self, txt_records: Vec<(String, String)>) -> Result<Vec<Vec<(String, String)>>> {
        let mut message_groups: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for (domain, txt_data) in txt_records {
            // Parse the TXT record to extract metadata
            if let Some(message_id) = self.extract_message_id(&domain, &txt_data)? {
                // Skip cover traffic (marked with p=cover)
                if txt_data.contains("p=cover") {
                    debug!("🎭 DNS-Phantom: Skipping cover traffic record: {}", domain);
                    continue;
                }

                message_groups.entry(message_id)
                    .or_insert_with(Vec::new)
                    .push((domain, txt_data));
            }
        }

        // Convert to sorted groups (by sequence number)
        let mut result = Vec::new();
        for (_, mut group) in message_groups {
            // Sort by sequence number extracted from domain
            group.sort_by(|a, b| {
                let seq_a = self.extract_sequence_number(&a.0).unwrap_or(0);
                let seq_b = self.extract_sequence_number(&b.0).unwrap_or(0);
                seq_a.cmp(&seq_b)
            });
            result.push(group);
        }

        Ok(result)
    }

    /// Extract message ID from domain and TXT data for grouping
    fn extract_message_id(&self, domain: &str, txt_data: &str) -> Result<Option<String>> {
        // Parse TXT record format: "v=qnk1 p=phantom enc=aes256 t=1234567890 seq=01 d=..."
        let mut timestamp = None;
        let mut is_phantom = false;

        for part in txt_data.split_whitespace() {
            if part.starts_with("t=") {
                timestamp = part[2..].parse::<u64>().ok();
            } else if part == "p=phantom" {
                is_phantom = true;
            }
        }

        if !is_phantom {
            return Ok(None); // Not a phantom message
        }

        // Group messages by timestamp (within 60-second window)
        if let Some(ts) = timestamp {
            let group_id = format!("msg_{}", ts / 60); // 60-second grouping window
            Ok(Some(group_id))
        } else {
            Ok(None)
        }
    }

    /// Extract sequence number from domain name
    fn extract_sequence_number(&self, domain: &str) -> Option<u8> {
        // Domain format: "verification_01.s.quillon.xyz"
        if let Some(parts) = domain.split('.').next() {
            if let Some(seq_part) = parts.split('_').last() {
                return u8::from_str_radix(seq_part, 16).ok();
            }
        }
        None
    }

    /// Decode a single message from grouped TXT records
    async fn decode_single_message(&self, record_group: Vec<(String, String)>) -> Result<DNSPhantomMessage> {
        debug!("🔍 DNS-Phantom: Decoding message from {} TXT records", record_group.len());

        // Extract and concatenate all data parts
        let mut data_chunks = Vec::new();

        for (_domain, txt_data) in record_group {
            if let Some(data_part) = self.extract_data_from_txt(&txt_data)? {
                data_chunks.push(data_part);
            }
        }

        if data_chunks.is_empty() {
            return Err(anyhow!("No data chunks found in TXT records"));
        }

        // Concatenate all chunks
        let mut combined_data = Vec::new();
        for chunk in data_chunks {
            combined_data.extend_from_slice(&chunk);
        }

        // Decrypt if encryption is enabled
        let decrypted_data = match &self.encryptor {
            Some(encryptor) => {
                debug!("🔓 DNS-Phantom: Decrypting {} bytes of steganographic data", combined_data.len());
                encryptor.decrypt_payload(&combined_data)?
            }
            None => {
                debug!("⚠️  DNS-Phantom: No decryption - processing plaintext data");
                combined_data
            }
        };

        // Decompress the data
        let decompressed = lz4_flex::decompress_size_prepended(&decrypted_data)
            .map_err(|e| anyhow!("Decompression failed: {}", e))?;

        // Deserialize back to message
        let message: DNSPhantomMessage = bincode::deserialize(&decompressed)
            .map_err(|e| anyhow!("Deserialization failed: {}", e))?;

        debug!("✅ DNS-Phantom: Successfully decoded steganographic message");
        Ok(message)
    }

    /// Extract Base32-encoded data from TXT record
    fn extract_data_from_txt(&self, txt_data: &str) -> Result<Option<Vec<u8>>> {
        // Parse format: "v=qnk1 p=phantom enc=aes256 t=1234567890 seq=01 d=DATA_HERE"
        for part in txt_data.split_whitespace() {
            if part.starts_with("d=") {
                let encoded_data = &part[2..];
                return match base32::decode(base32::Alphabet::Crockford, encoded_data) {
                    Some(decoded) => Ok(Some(decoded)),
                    None => Err(anyhow!("Base32 decoding failed for: {}", encoded_data)),
                };
            }
        }
        Ok(None)
    }

    /// Validate that a TXT record looks like legitimate steganographic data
    pub fn validate_txt_record(&self, domain: &str, txt_data: &str) -> bool {
        // Check if it matches our TXT record format
        if !txt_data.contains("v=qnk1") {
            return false;
        }

        // Check domain format (should be *.s.quillon.xyz for steganographic records)
        if !domain.contains(".s.quillon.xyz") {
            return false;
        }

        // Check for required fields
        txt_data.contains("p=phantom") &&
        txt_data.contains("t=") &&
        txt_data.contains("seq=") &&
        txt_data.contains("d=")
    }

    /// Test decryption capability with a known message
    pub async fn test_decryption(&self, test_message: &DNSPhantomMessage) -> Result<()> {
        // This would encode and then decode to test round-trip
        // For now, just verify the encryptor works
        if let Some(encryptor) = &self.encryptor {
            let test_data = bincode::serialize(test_message)?;
            encryptor.test_encryption_round_trip(&test_data)?;
            debug!("✅ DNS-Phantom: Decryption test passed");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DNSPhantomMessage, encoding::MessageEncoder};

    #[tokio::test]
    async fn test_encode_decode_round_trip() {
        let node_id = "test-node-decoder";

        // Create test message
        let test_message = DNSPhantomMessage::NodeAdvertisement {
            node_id: node_id.to_string(),
            capabilities: vec!["consensus".to_string(), "steganographic".to_string()],
            timestamp: chrono::Utc::now().timestamp() as u64,
        };

        // Encode with encryption
        let encoder = MessageEncoder::new_with_encryption(
            crate::EncodingMethod::TXTRecordSteganography,
            node_id
        ).unwrap();

        let encoded_queries = encoder.encode_to_dns_queries(&test_message).await.unwrap();

        // Convert to TXT records format
        let txt_records: Vec<(String, String)> = encoded_queries
            .into_iter()
            .map(|(domain, data)| (domain, String::from_utf8_lossy(&data).to_string()))
            .collect();

        // Decode with same encryption
        let decoder = MessageDecoder::new_with_encryption(node_id).unwrap();
        let decoded_messages = decoder.decode_from_txt_records(txt_records).await.unwrap();

        assert!(!decoded_messages.is_empty());
        // Note: More specific assertions would depend on DNSPhantomMessage structure
    }
}