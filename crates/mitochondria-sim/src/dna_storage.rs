/// DNA-based blockchain storage system
///
/// This module implements the core concept of encoding blockchain data
/// into DNA sequences for biological storage and replication.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::{DropletNode, DNABlockchain, DNASynthesisEvent};

/// DNA storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DNAStorageConfig {
    pub bases_per_byte: usize,      // DNA bases per byte of data
    pub error_correction_level: f64, // Error correction redundancy
    pub synthesis_speed_bps: f64,    // Base pairs synthesized per second
    pub degradation_rate: f64,       // DNA degradation per time unit
}

impl Default for DNAStorageConfig {
    fn default() -> Self {
        Self {
            bases_per_byte: 4,          // 2 bits per base (A=00, T=01, G=10, C=11)
            error_correction_level: 0.3, // 30% redundancy
            synthesis_speed_bps: 100.0,  // 100 bp/s synthesis
            degradation_rate: 0.001,     // 0.1% degradation per time unit
        }
    }
}

/// Encode arbitrary data into DNA sequence
pub fn encode_data_to_dna(data: &[u8]) -> String {
    let bases = ['A', 'T', 'G', 'C'];
    let mut dna_sequence = String::new();
    
    for byte in data {
        // Each byte becomes 4 DNA bases (2 bits per base)
        for shift in (0..8).step_by(2).rev() {
            let two_bits = (byte >> shift) & 0b11;
            dna_sequence.push(bases[two_bits as usize]);
        }
    }
    
    dna_sequence
}

/// Decode DNA sequence back to binary data
pub fn decode_dna_to_data(dna_sequence: &str) -> Result<Vec<u8>> {
    if dna_sequence.len() % 4 != 0 {
        return Err(anyhow::anyhow!("Invalid DNA sequence length for decoding"));
    }
    
    let mut data = Vec::new();
    let chars: Vec<char> = dna_sequence.chars().collect();
    
    for chunk in chars.chunks(4) {
        let mut byte = 0u8;
        
        for (i, &base) in chunk.iter().enumerate() {
            let bits = match base {
                'A' => 0b00,
                'T' => 0b01,
                'G' => 0b10,
                'C' => 0b11,
                _ => return Err(anyhow::anyhow!("Invalid DNA base: {}", base)),
            };
            byte |= bits << (6 - i * 2);
        }
        
        data.push(byte);
    }
    
    Ok(data)
}

/// Calculate total DNA mass across the network
pub fn calculate_total_dna_mass(droplets: &HashMap<String, DropletNode>) -> f64 {
    droplets
        .values()
        .map(|droplet| droplet.dna_data.total_mass_picograms)
        .sum()
}

/// Find droplet with the heaviest DNA chain (consensus leader)
pub fn find_heaviest_droplet(droplets: &HashMap<String, DropletNode>) -> String {
    droplets
        .iter()
        .max_by(|(_, a), (_, b)| {
            a.dna_data.total_mass_picograms
                .partial_cmp(&b.dna_data.total_mass_picograms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(id, _)| id.clone())
        .unwrap_or_else(|| "no_droplets".to_string())
}

/// Simulate DNA replication for blockchain backup
pub async fn replicate_dna_chain(
    source: &DropletNode, 
    target: &mut DropletNode,
    config: &DNAStorageConfig
) -> Result<()> {
    if target.energy_level < 0.6 {
        warn!("🔋 Insufficient energy for DNA replication in {}", target.droplet_id);
        return Ok(());
    }

    let replication_energy_cost = source.dna_data.total_mass_picograms * 0.1;
    
    // Copy synthesis history (simplified replication)
    for event in &source.dna_data.synthesis_history {
        let replicated_event = DNASynthesisEvent {
            block_height: event.block_height,
            sequence_added: event.sequence_added.clone(),
            synthesis_time_ms: (event.synthesis_time_ms as f64 / config.synthesis_speed_bps * 100.0) as u64,
            energy_cost: event.energy_cost * 0.8, // Replication is cheaper
            synthesized_at: Utc::now(),
        };
        
        target.dna_data.synthesis_history.push(replicated_event);
    }

    // Update target droplet state
    target.energy_level -= replication_energy_cost;
    target.dna_data.chain_length = source.dna_data.chain_length;
    target.dna_data.latest_block_hash = source.dna_data.latest_block_hash.clone();
    target.dna_data.total_mass_picograms += source.dna_data.total_mass_picograms * 0.9;
    target.size_nanoliters += replication_energy_cost * 3.0;

    info!("🧬 DNA chain replicated from {} to {}", 
          source.droplet_id, target.droplet_id);
    
    Ok(())
}

/// Simulate DNA degradation over time
pub async fn degrade_dna_storage(
    droplet: &mut DropletNode,
    config: &DNAStorageConfig,
    dt: f64
) -> Result<()> {
    let degradation_factor = 1.0 - (config.degradation_rate * dt);
    
    // Reduce DNA mass due to degradation
    droplet.dna_data.total_mass_picograms *= degradation_factor;
    
    // Occasionally lose synthesis history entries
    if degradation_factor < 0.995 && !droplet.dna_data.synthesis_history.is_empty() {
        let remove_index = droplet.dna_data.synthesis_history.len() - 1;
        droplet.dna_data.synthesis_history.remove(remove_index);
        droplet.dna_data.chain_length = droplet.dna_data.chain_length.saturating_sub(1);
        
        debug!("🧪 DNA degradation removed block from {}", droplet.droplet_id);
    }
    
    Ok(())
}

/// Verify DNA chain integrity using error correction
pub fn verify_dna_integrity(dna_chain: &DNABlockchain) -> f64 {
    if dna_chain.synthesis_history.is_empty() {
        return 0.0;
    }
    
    // Calculate integrity based on chain consistency
    let expected_mass = dna_chain.synthesis_history
        .iter()
        .map(|event| event.energy_cost * 2.0)
        .sum::<f64>();
    
    let mass_integrity = (dna_chain.total_mass_picograms / expected_mass).min(1.0);
    let length_integrity = (dna_chain.chain_length as f64 / dna_chain.synthesis_history.len() as f64).min(1.0);
    
    (mass_integrity + length_integrity) / 2.0
}

/// Encode blockchain transaction into DNA
pub fn encode_transaction_to_dna(
    from: &str,
    to: &str, 
    amount: u64,
    timestamp: DateTime<Utc>
) -> String {
    // Create transaction data
    let transaction_data = format!("{}->{}:{}", from, to, amount);
    let timestamp_data = timestamp.timestamp_millis().to_be_bytes();
    
    let mut combined_data = transaction_data.into_bytes();
    combined_data.extend_from_slice(&timestamp_data);
    
    encode_data_to_dna(&combined_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dna_encoding_decoding() {
        let original_data = b"Hello, DNA blockchain!";
        let dna_sequence = encode_data_to_dna(original_data);
        let decoded_data = decode_dna_to_data(&dna_sequence).unwrap();
        
        assert_eq!(original_data, &decoded_data[..]);
    }

    #[test]
    fn test_transaction_encoding() {
        let dna_sequence = encode_transaction_to_dna(
            "droplet_0001",
            "droplet_0002", 
            100,
            Utc::now()
        );
        
        assert!(!dna_sequence.is_empty());
        assert!(dna_sequence.chars().all(|c| matches!(c, 'A' | 'T' | 'G' | 'C')));
    }

    #[test]
    fn test_dna_integrity_verification() {
        let mut dna_chain = DNABlockchain {
            chain_length: 2,
            genesis_hash: "genesis".to_string(),
            latest_block_hash: "latest".to_string(),
            total_mass_picograms: 4.0,
            synthesis_history: vec![
                DNASynthesisEvent {
                    block_height: 0,
                    sequence_added: "ATGC".to_string(),
                    synthesis_time_ms: 100,
                    energy_cost: 1.0,
                    synthesized_at: Utc::now(),
                },
                DNASynthesisEvent {
                    block_height: 1,
                    sequence_added: "GCTA".to_string(),
                    synthesis_time_ms: 150,
                    energy_cost: 1.0,
                    synthesized_at: Utc::now(),
                }
            ],
        };
        
        let integrity = verify_dna_integrity(&dna_chain);
        assert!(integrity > 0.9); // Should be high integrity
    }
}