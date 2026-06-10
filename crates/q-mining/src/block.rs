/// Quantum-Enhanced PoW Block Structure
/// 
/// This module implements the QuantumPoWBlock - the core data structure for 
/// quantum-enhanced proof-of-work mining in Q-NarwhalKnight.

use q_types::*;
use q_dag_knight::QuantumVDFProof;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use anyhow::Result;

/// Quantum-enhanced Proof-of-Work block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPoWBlock {
    /// Block header with core mining data
    pub header: BlockHeader,
    
    /// Mining-specific data
    pub mining_data: MiningData,
    
    /// Quantum enhancements
    pub quantum_data: QuantumData,
    
    /// Transactions included in block
    pub transactions: Vec<Transaction>,
    
    /// Block signature (Dilithium5)
    pub signature: Vec<u8>,
}

/// Block header containing essential mining information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockHeader {
    /// Hash of previous block
    pub parent_hash: [u8; 32],
    
    /// Block height in PoW chain
    pub height: u64,
    
    /// Block timestamp (Unix timestamp)
    pub timestamp: u64,
    
    /// Merkle root of transactions
    pub tx_merkle_root: [u8; 32],
    
    /// Mining difficulty target
    pub difficulty: u32,
    
    /// Mining nonce
    pub nonce: u64,
    
    /// Miner's address/identity
    pub miner_address: [u8; 20],
}

/// Mining-specific data and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningData {
    /// Mining algorithm used
    pub algorithm: MiningAlgorithm,
    
    /// Time spent mining (milliseconds)
    pub mining_time_ms: u64,
    
    /// Hash rate achieved during mining
    pub hash_rate: f64,
    
    /// Mining reward amount
    pub reward_amount: u64,
    
    /// Extra nonce for extended nonce space
    pub extra_nonce: u64,
}

/// Quantum enhancement data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumData {
    /// Quantum seed from VDF system
    pub quantum_seed: Option<[u8; 32]>,
    
    /// VDF proof for timing assurance
    pub vdf_proof: Option<QuantumVDFProof>,
    
    /// Quantum entropy quality (0.0-1.0)
    pub entropy_quality: f64,
    
    /// Quantum enhancement level used
    pub enhancement_level: f64,
    
    /// Quantum randomness injection points
    pub injection_points: Vec<u64>,
}

/// Mining algorithm specification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MiningAlgorithm {
    /// SHA-3-256 with quantum enhancements
    QuantumSHA3 { enhancement_level: f64 },
    
    /// Classical SHA-3-256
    ClassicalSHA3,
    
    /// Future: Memory-hard quantum algorithm
    QuantumArgon2 { memory_size: u32 },
}

/// Mining template provided to miners
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningTemplate {
    /// Previous block hash
    pub parent_hash: [u8; 32],
    
    /// Target block height
    pub height: u64,
    
    /// Current mining difficulty
    pub difficulty: u32,
    
    /// Transactions to include
    pub transactions: Vec<Transaction>,
    
    /// Quantum seed for enhancement
    pub quantum_seed: Option<[u8; 32]>,
    
    /// Target block time
    pub target_time: Duration,
    
    /// Mining reward
    pub reward_amount: u64,
    
    /// Template expiration time
    pub expires_at: u64,
}

/// Difficulty target for mining
#[derive(Debug, Clone, Copy)]
pub struct DifficultyTarget {
    /// Difficulty as compact representation
    pub compact: u32,
    
    /// Target hash (256-bit)
    pub target: [u8; 32],
}

impl QuantumPoWBlock {
    /// Create new quantum PoW block from template
    pub fn new(template: MiningTemplate, miner_address: [u8; 20]) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        let tx_merkle_root = Self::calculate_merkle_root(&template.transactions);
        
        Self {
            header: BlockHeader {
                parent_hash: template.parent_hash,
                height: template.height,
                timestamp,
                tx_merkle_root,
                difficulty: template.difficulty,
                nonce: 0,
                miner_address,
            },
            mining_data: MiningData {
                algorithm: MiningAlgorithm::QuantumSHA3 { enhancement_level: 0.7 },
                mining_time_ms: 0,
                hash_rate: 0.0,
                reward_amount: template.reward_amount,
                extra_nonce: 0,
            },
            quantum_data: QuantumData {
                quantum_seed: template.quantum_seed,
                vdf_proof: None,
                entropy_quality: 0.0,
                enhancement_level: 0.7,
                injection_points: Vec::new(),
            },
            transactions: template.transactions,
            signature: Vec::new(),
        }
    }
    
    /// Calculate SHA-3 hash of the block
    pub fn hash(&self) -> [u8; 32] {
        let serialized = self.serialize_for_hash();
        let hash = Sha3_256::digest(&serialized);
        hash.into()
    }
    
    /// Calculate hash of block header only
    pub fn header_hash(&self) -> [u8; 32] {
        let header_data = bincode::serialize(&self.header).unwrap();
        let hash = Sha3_256::digest(&header_data);
        hash.into()
    }
    
    /// Serialize block for hashing (excluding signature)
    fn serialize_for_hash(&self) -> Vec<u8> {
        let mut data = Vec::new();
        
        // Serialize header
        data.extend_from_slice(&bincode::serialize(&self.header).unwrap());
        
        // Serialize mining data
        data.extend_from_slice(&bincode::serialize(&self.mining_data).unwrap());
        
        // Serialize quantum data
        data.extend_from_slice(&bincode::serialize(&self.quantum_data).unwrap());
        
        // Serialize transaction data
        data.extend_from_slice(&bincode::serialize(&self.transactions).unwrap());
        
        data
    }
    
    /// Check if block meets difficulty target
    pub fn meets_difficulty(&self, target: &DifficultyTarget) -> bool {
        let hash = self.hash();
        self.hash_meets_target(&hash, &target.target)
    }
    
    /// Check if hash meets target
    fn hash_meets_target(&self, hash: &[u8; 32], target: &[u8; 32]) -> bool {
        for i in 0..32 {
            match hash[i].cmp(&target[i]) {
                std::cmp::Ordering::Less => return true,
                std::cmp::Ordering::Greater => return false,
                std::cmp::Ordering::Equal => continue,
            }
        }
        false // Hashes are equal, which meets target
    }
    
    /// Calculate Merkle root of transactions
    fn calculate_merkle_root(transactions: &[Transaction]) -> [u8; 32] {
        if transactions.is_empty() {
            return [0u8; 32];
        }
        
        let mut hashes: Vec<[u8; 32]> = transactions
            .iter()
            .map(|tx| {
                let tx_data = bincode::serialize(tx).unwrap();
                Sha3_256::digest(&tx_data).into()
            })
            .collect();
        
        // Build Merkle tree
        while hashes.len() > 1 {
            let mut next_level = Vec::new();
            
            for chunk in hashes.chunks(2) {
                let mut hasher = Sha3_256::new();
                hasher.update(&chunk[0]);
                
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    // Odd number of hashes - duplicate last hash
                    hasher.update(&chunk[0]);
                }
                
                next_level.push(hasher.finalize().into());
            }
            
            hashes = next_level;
        }
        
        hashes[0]
    }
    
    /// Add quantum enhancement data after VDF computation
    pub fn add_quantum_enhancement(&mut self, 
                                  vdf_proof: QuantumVDFProof, 
                                  entropy_quality: f64,
                                  injection_points: Vec<u64>) {
        self.quantum_data.vdf_proof = Some(vdf_proof);
        self.quantum_data.entropy_quality = entropy_quality;
        self.quantum_data.injection_points = injection_points;
    }
    
    /// Update mining statistics after successful mining
    pub fn finalize_mining(&mut self, mining_time: Duration, hash_rate: f64) {
        self.mining_data.mining_time_ms = mining_time.as_millis() as u64;
        self.mining_data.hash_rate = hash_rate;
    }
    
    /// Sign block with miner's private key (Dilithium5 post-quantum)
    ///
    /// # v2.4.7-beta: NIST PQC standard Dilithium5
    /// - Secret key: 4,864 bytes
    /// - Signed message: 4,627 bytes + message length
    pub fn sign(&mut self, private_key: &[u8]) -> Result<()> {
        use pqcrypto_dilithium::dilithium5;
        use pqcrypto_traits::sign::{SecretKey, SignedMessage};

        let block_hash = self.hash();

        // Parse Dilithium5 secret key
        let sk = dilithium5::SecretKey::from_bytes(private_key)
            .map_err(|_| anyhow::anyhow!("Invalid Dilithium5 secret key (expected 4,864 bytes)"))?;

        // Sign the block hash with Dilithium5
        let signed_message = dilithium5::sign(&block_hash, &sk);
        self.signature = signed_message.as_bytes().to_vec();

        tracing::debug!(
            "🔐 Block {} signed with Dilithium5 ({} bytes)",
            hex::encode(&block_hash[..8]),
            self.signature.len()
        );
        Ok(())
    }

    /// Verify block signature using Dilithium5 post-quantum verification
    ///
    /// # v2.4.7-beta: Proper cryptographic verification
    /// - Public key: 2,592 bytes
    /// - Verifies both signature validity and message integrity
    pub fn verify_signature(&self, public_key: &[u8]) -> bool {
        use pqcrypto_dilithium::dilithium5;
        use pqcrypto_traits::sign::{PublicKey, SignedMessage};

        if self.signature.is_empty() {
            return false;
        }

        // Parse Dilithium5 public key
        let pk = match dilithium5::PublicKey::from_bytes(public_key) {
            Ok(pk) => pk,
            Err(_) => {
                tracing::warn!("Invalid Dilithium5 public key");
                return false;
            }
        };

        // Parse signed message
        let signed_msg = match dilithium5::SignedMessage::from_bytes(&self.signature) {
            Ok(sm) => sm,
            Err(_) => {
                tracing::warn!("Invalid Dilithium5 signed message format");
                return false;
            }
        };

        // Verify signature and get original message
        let expected_hash = self.hash();
        match dilithium5::open(&signed_msg, &pk) {
            Ok(verified_message) => {
                if verified_message == expected_hash {
                    tracing::debug!("✅ Block signature verified (Dilithium5)");
                    true
                } else {
                    tracing::warn!("❌ Block hash mismatch after signature verification");
                    false
                }
            }
            Err(_) => {
                tracing::warn!("❌ Dilithium5 signature verification failed");
                false
            }
        }
    }
    
    /// Get block size in bytes
    pub fn size(&self) -> usize {
        bincode::serialize(self).unwrap().len()
    }
    
    /// Validate block structure and data
    pub fn validate(&self) -> Result<()> {
        // Check timestamp is reasonable (not too far in future)
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        if self.header.timestamp > now + 7200 { // 2 hours in future
            return Err(anyhow::anyhow!("Block timestamp too far in future"));
        }
        
        // Check difficulty is reasonable
        if self.header.difficulty == 0 {
            return Err(anyhow::anyhow!("Invalid difficulty (cannot be zero)"));
        }
        
        // Check height is sequential
        if self.header.height == 0 && self.header.parent_hash != [0u8; 32] {
            return Err(anyhow::anyhow!("Genesis block must have zero parent hash"));
        }
        
        // Validate Merkle root
        let calculated_root = Self::calculate_merkle_root(&self.transactions);
        if calculated_root != self.header.tx_merkle_root {
            return Err(anyhow::anyhow!("Invalid transaction Merkle root"));
        }
        
        // Check quantum enhancement levels are valid
        if self.quantum_data.entropy_quality < 0.0 || self.quantum_data.entropy_quality > 1.0 {
            return Err(anyhow::anyhow!("Invalid entropy quality"));
        }
        
        if self.quantum_data.enhancement_level < 0.0 || self.quantum_data.enhancement_level > 1.0 {
            return Err(anyhow::anyhow!("Invalid quantum enhancement level"));
        }
        
        Ok(())
    }
}

impl DifficultyTarget {
    /// Create difficulty target from compact representation
    pub fn from_compact(compact: u32) -> Self {
        let mut target = [0u8; 32];
        
        // Extract exponent and mantissa from compact format
        let exponent = (compact >> 24) & 0xFF;
        let mantissa = compact & 0x00FFFFFF;
        
        // Calculate target from compact representation
        if exponent <= 3 {
            let shift = 8 * (3 - exponent);
            let value = mantissa >> shift;
            target[31] = (value & 0xFF) as u8;
            target[30] = ((value >> 8) & 0xFF) as u8;
            target[29] = ((value >> 16) & 0xFF) as u8;
        } else {
            let byte_index = 32 - exponent as usize;
            if byte_index < 29 {
                target[byte_index + 2] = (mantissa & 0xFF) as u8;
                target[byte_index + 1] = ((mantissa >> 8) & 0xFF) as u8;
                target[byte_index] = ((mantissa >> 16) & 0xFF) as u8;
            }
        }
        
        Self { compact, target }
    }
    
    /// Create target from difficulty (leading zero bits)
    pub fn from_difficulty_bits(bits: u32) -> Self {
        let mut target = [0xFFu8; 32];
        
        let bytes_to_zero = bits / 8;
        let remaining_bits = bits % 8;
        
        // Zero out full bytes
        for i in 0..bytes_to_zero.min(32) as usize {
            target[i] = 0;
        }
        
        // Handle remaining bits
        if bytes_to_zero < 32 && remaining_bits > 0 {
            target[bytes_to_zero as usize] >>= remaining_bits;
        }
        
        Self {
            compact: bits,
            target,
        }
    }
}

impl MiningTemplate {
    /// Create mining template for next block
    pub fn create_next(parent_block: &QuantumPoWBlock,
                      transactions: Vec<Transaction>,
                      difficulty: u32,
                      quantum_seed: Option<[u8; 32]>,
                      reward_amount: u64) -> Self {
        let expires_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() + 300; // 5 minute expiration
            
        Self {
            parent_hash: parent_block.hash(),
            height: parent_block.header.height + 1,
            difficulty,
            transactions,
            quantum_seed,
            target_time: Duration::from_secs(30),
            reward_amount,
            expires_at,
        }
    }
    
    /// Check if template has expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now > self.expires_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use q_types::Transaction;
    
    fn create_test_template() -> MiningTemplate {
        MiningTemplate {
            parent_hash: [1u8; 32],
            height: 1,
            difficulty: 4,
            transactions: vec![],
            quantum_seed: Some([42u8; 32]),
            target_time: Duration::from_secs(30),
            reward_amount: 2_000_000_000,
            expires_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + 300,
        }
    }
    
    #[test]
    fn test_block_creation() {
        let template = create_test_template();
        let miner_address = [5u8; 20];
        
        let block = QuantumPoWBlock::new(template, miner_address);
        
        assert_eq!(block.header.height, 1);
        assert_eq!(block.header.miner_address, miner_address);
        assert_eq!(block.quantum_data.quantum_seed, Some([42u8; 32]));
    }
    
    #[test]
    fn test_block_hashing() {
        let template = create_test_template();
        let miner_address = [5u8; 20];
        
        let block1 = QuantumPoWBlock::new(template.clone(), miner_address);
        let block2 = QuantumPoWBlock::new(template, miner_address);
        
        // Same template should produce different blocks due to timestamp
        // But let's test with same timestamp
        let mut block3 = block1.clone();
        block3.header.nonce = 12345;
        
        assert_eq!(block1.hash(), block2.hash()); // Same data, same hash
        assert_ne!(block1.hash(), block3.hash()); // Different nonce, different hash
    }
    
    #[test]
    fn test_difficulty_target() {
        let target = DifficultyTarget::from_difficulty_bits(8);
        
        // With 8 leading zero bits, first byte should be 0
        assert_eq!(target.target[0], 0);
        
        // Test compact format
        let target2 = DifficultyTarget::from_compact(0x1d00ffff);
        assert!(target2.target[0] < 0x10); // Should have leading zeros
    }
    
    #[test]
    fn test_merkle_root_calculation() {
        // Test with empty transactions
        let root = QuantumPoWBlock::calculate_merkle_root(&[]);
        assert_eq!(root, [0u8; 32]);
        
        // Test with single transaction
        let tx = Transaction {
            id: [1u8; 32],
            inputs: vec![],
            outputs: vec![],
            signature: vec![],
            timestamp: chrono::Utc::now(),
        };
        
        let root = QuantumPoWBlock::calculate_merkle_root(&[tx.clone()]);
        assert_ne!(root, [0u8; 32]);
        
        // Test with multiple transactions
        let tx2 = Transaction {
            id: [2u8; 32],
            inputs: vec![],
            outputs: vec![],
            signature: vec![],
            timestamp: chrono::Utc::now(),
        };
        
        let root2 = QuantumPoWBlock::calculate_merkle_root(&[tx.clone(), tx2]);
        assert_ne!(root, root2); // Different transactions, different root
    }
    
    #[test]
    fn test_block_validation() {
        let template = create_test_template();
        let miner_address = [5u8; 20];
        
        let block = QuantumPoWBlock::new(template, miner_address);
        
        // Valid block should pass validation
        assert!(block.validate().is_ok());
        
        // Test with invalid quantum data
        let mut invalid_block = block.clone();
        invalid_block.quantum_data.entropy_quality = 1.5; // Invalid > 1.0
        assert!(invalid_block.validate().is_err());
        
        // Test with zero difficulty
        let mut invalid_block2 = block;
        invalid_block2.header.difficulty = 0;
        assert!(invalid_block2.validate().is_err());
    }
    
    #[test]
    fn test_template_expiration() {
        let mut template = create_test_template();
        
        // Fresh template should not be expired
        assert!(!template.is_expired());
        
        // Set expiration in the past
        template.expires_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() - 100;
            
        assert!(template.is_expired());
    }
}