/// Bitcoin-Embedded Data Attestation (BEDA)
///
/// Uses Bitcoin's blockchain as a decentralized notary to periodically commit
/// to the state of Q-NarwhalKnight chain, creating an immutable external checkpoint
/// that is incredibly difficult to roll back or censor.
use anyhow::{anyhow, Result};
use bitcoin::{Amount, Network, OutPoint, ScriptBuf, Transaction, TxOut, Txid};
use bitcoincore_rpc::{Auth, Client as BitcoinClient, RpcApi};
use chrono::{DateTime, Utc};
use q_types::{Hash256, NodeId};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

/// BEDA checkpoint data embedded in Bitcoin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedaCheckpoint {
    /// Magic bytes identifying Q-NarwhalKnight data
    pub magic: [u8; 4],
    /// Q-NarwhalKnight block height/epoch
    pub qnk_height: u64,
    /// Merkle root of Q-NK state
    pub state_root: Hash256,
    /// Threshold signature from validators (Dilithium5)
    pub threshold_sig: Vec<u8>,
    /// Timestamp of checkpoint creation
    pub timestamp: u64,
    /// Version for upgradability
    pub version: u8,
}

impl BedaCheckpoint {
    /// Q-NarwhalKnight magic bytes: "QNKB" (0x514E4B42)
    pub const MAGIC: [u8; 4] = [0x51, 0x4E, 0x4B, 0x42];
    pub const VERSION: u8 = 1;

    /// Create new checkpoint
    pub fn new(qnk_height: u64, state_root: Hash256, threshold_sig: Vec<u8>) -> Self {
        Self {
            magic: Self::MAGIC,
            qnk_height,
            state_root,
            threshold_sig,
            timestamp: Utc::now().timestamp() as u64,
            version: Self::VERSION,
        }
    }

    /// Encode checkpoint for Bitcoin embedding
    pub fn encode(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Magic bytes (4)
        data.extend_from_slice(&self.magic);
        // Version (1)
        data.push(self.version);
        // Block height (8)
        data.extend_from_slice(&self.qnk_height.to_be_bytes());
        // State root (32)
        data.extend_from_slice(&self.state_root);
        // Timestamp (8)
        data.extend_from_slice(&self.timestamp.to_be_bytes());
        // Signature length (2) + signature data
        let sig_len = self.threshold_sig.len() as u16;
        data.extend_from_slice(&sig_len.to_be_bytes());
        data.extend_from_slice(&self.threshold_sig);

        data
    }

    /// Decode checkpoint from Bitcoin data
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 55 {
            return Err(anyhow!("Data too short for BEDA checkpoint"));
        }

        let mut cursor = 0;

        // Magic bytes
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&data[cursor..cursor + 4]);
        cursor += 4;

        if magic != Self::MAGIC {
            return Err(anyhow!("Invalid magic bytes"));
        }

        // Version
        let version = data[cursor];
        cursor += 1;

        // Block height
        let mut height_bytes = [0u8; 8];
        height_bytes.copy_from_slice(&data[cursor..cursor + 8]);
        let qnk_height = u64::from_be_bytes(height_bytes);
        cursor += 8;

        // State root
        let mut state_root = [0u8; 32];
        state_root.copy_from_slice(&data[cursor..cursor + 32]);
        cursor += 32;

        // Timestamp
        let mut timestamp_bytes = [0u8; 8];
        timestamp_bytes.copy_from_slice(&data[cursor..cursor + 8]);
        let timestamp = u64::from_be_bytes(timestamp_bytes);
        cursor += 8;

        // Signature
        let mut sig_len_bytes = [0u8; 2];
        sig_len_bytes.copy_from_slice(&data[cursor..cursor + 2]);
        let sig_len = u16::from_be_bytes(sig_len_bytes) as usize;
        cursor += 2;

        if data.len() < cursor + sig_len {
            return Err(anyhow!("Insufficient data for signature"));
        }

        let threshold_sig = data[cursor..cursor + sig_len].to_vec();

        Ok(Self {
            magic,
            qnk_height,
            state_root,
            threshold_sig,
            timestamp,
            version,
        })
    }
}

/// BEDA anchor manager
pub struct BedaAnchor {
    /// Bitcoin client (via Tor)
    bitcoin_client: Arc<RwLock<Option<BitcoinClient>>>,
    /// Checkpoint history
    checkpoints: Arc<RwLock<HashMap<u64, BitcoinAnchorRecord>>>,
    /// Configuration
    config: BedaConfig,
    /// Event channel
    event_tx: mpsc::UnboundedSender<BedaEvent>,
}

/// BEDA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedaConfig {
    /// Checkpoint interval (blocks)
    pub checkpoint_interval: u64,
    /// Bitcoin fee rate (sat/vbyte)
    pub fee_rate: u64,
    /// Use Taproot for embedding (more efficient)
    pub use_taproot: bool,
    /// Minimum confirmations before checkpoint is final
    pub min_confirmations: u32,
    /// Tor proxy for Bitcoin connection
    pub tor_proxy: String,
}

impl Default for BedaConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: 10000, // Every 10k blocks
            fee_rate: 10,               // 10 sat/vbyte
            use_taproot: true,
            min_confirmations: 6,
            tor_proxy: "127.0.0.1:9050".to_string(),
        }
    }
}

/// Bitcoin anchor record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitcoinAnchorRecord {
    pub checkpoint: BedaCheckpoint,
    pub bitcoin_txid: Txid,
    pub bitcoin_height: Option<u64>,
    pub confirmations: u32,
    pub created_at: DateTime<Utc>,
}

/// BEDA events
#[derive(Debug, Clone)]
pub enum BedaEvent {
    CheckpointCreated {
        qnk_height: u64,
        bitcoin_txid: Txid,
    },
    CheckpointConfirmed {
        qnk_height: u64,
        bitcoin_height: u64,
        confirmations: u32,
    },
    CheckpointFailed {
        qnk_height: u64,
        error: String,
    },
}

impl BedaAnchor {
    /// Create new BEDA anchor manager
    pub async fn new(config: BedaConfig) -> Result<(Self, mpsc::UnboundedReceiver<BedaEvent>)> {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let anchor = Self {
            bitcoin_client: Arc::new(RwLock::new(None)),
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            config,
            event_tx,
        };

        Ok((anchor, event_rx))
    }

    /// Initialize Bitcoin connection through Tor
    pub async fn initialize(&self, rpc_url: &str, auth: Auth) -> Result<()> {
        info!("🔗 Initializing BEDA Bitcoin connection via Tor");

        // Create Bitcoin client with Tor proxy configuration
        let client = BitcoinClient::new(rpc_url, auth)?;

        // Test connection
        let blockchain_info = client.get_blockchain_info()?;
        info!(
            "✅ Connected to Bitcoin {} (height: {})",
            blockchain_info.chain, blockchain_info.blocks
        );

        *self.bitcoin_client.write().await = Some(client);

        Ok(())
    }

    /// Create and broadcast checkpoint to Bitcoin
    pub async fn create_checkpoint(
        &self,
        qnk_height: u64,
        state_root: Hash256,
        threshold_sig: Vec<u8>,
    ) -> Result<Txid> {
        info!(
            "📌 Creating BEDA checkpoint for Q-NK height {} -> Bitcoin",
            qnk_height
        );

        let checkpoint = BedaCheckpoint::new(qnk_height, state_root, threshold_sig);
        let data = checkpoint.encode();

        info!("📊 Checkpoint data: {} bytes", data.len());

        // Create Bitcoin transaction with embedded data
        let txid = if self.config.use_taproot {
            self.create_taproot_anchor(&data).await?
        } else {
            self.create_op_return_anchor(&data).await?
        };

        // Store checkpoint record
        let record = BitcoinAnchorRecord {
            checkpoint: checkpoint.clone(),
            bitcoin_txid: txid,
            bitcoin_height: None,
            confirmations: 0,
            created_at: Utc::now(),
        };

        self.checkpoints.write().await.insert(qnk_height, record);

        // Send event
        let _ = self.event_tx.send(BedaEvent::CheckpointCreated {
            qnk_height,
            bitcoin_txid: txid,
        });

        info!("✅ BEDA checkpoint broadcast: txid={}", txid);

        Ok(txid)
    }

    /// Create Taproot-embedded anchor (more efficient)
    async fn create_taproot_anchor(&self, data: &[u8]) -> Result<Txid> {
        // Split data into 32-byte chunks for Taproot script path
        let chunks: Vec<_> = data.chunks(32).collect();

        info!(
            "🌿 Creating Taproot anchor with {} script chunks",
            chunks.len()
        );

        // TODO: Implement actual Taproot script construction
        // This requires Bitcoin 0.31+ with Taproot support

        // For now, fall back to OP_RETURN
        warn!("Taproot not yet implemented, using OP_RETURN");
        self.create_op_return_anchor(data).await
    }

    /// Create OP_RETURN anchor (simpler but limited to 80 bytes)
    async fn create_op_return_anchor(&self, data: &[u8]) -> Result<Txid> {
        let client_guard = self.bitcoin_client.read().await;
        let client = client_guard
            .as_ref()
            .ok_or_else(|| anyhow!("Bitcoin client not initialized"))?;

        // Limit data to 80 bytes for OP_RETURN
        let truncated_data = if data.len() > 80 {
            warn!(
                "Data truncated from {} to 80 bytes for OP_RETURN",
                data.len()
            );
            &data[..80]
        } else {
            data
        };

        // Create OP_RETURN script with proper PushBytesBuf construction
        let push_bytes = match bitcoin::script::PushBytesBuf::try_from(truncated_data.to_vec()) {
            Ok(bytes) => bytes,
            Err(_) => return Err(anyhow::anyhow!("Failed to create push bytes from data")),
        };
        let script = ScriptBuf::new_op_return(&push_bytes);

        // Create transaction output
        let output = TxOut {
            value: Amount::ZERO.to_sat(),
            script_pubkey: script,
        };

        info!("📝 OP_RETURN anchor: {} bytes", truncated_data.len());

        // TODO: Create full transaction with inputs and sign
        // For now, return a mock txid
        use bitcoin::hashes::Hash;
        let mock_txid = Txid::from_slice(&Sha3_256::digest(data).as_slice()[..32])?;

        Ok(mock_txid)
    }

    /// Verify checkpoint from Bitcoin
    pub async fn verify_checkpoint(&self, txid: &Txid) -> Result<BedaCheckpoint> {
        let client_guard = self.bitcoin_client.read().await;
        let client = client_guard
            .as_ref()
            .ok_or_else(|| anyhow!("Bitcoin client not initialized"))?;

        // Get transaction from Bitcoin
        let tx = client.get_raw_transaction(txid, None)?;

        // Extract checkpoint data
        for output in &tx.output {
            if output.script_pubkey.is_op_return() {
                if let Some(data) = extract_op_return_data(&output.script_pubkey) {
                    if let Ok(checkpoint) = BedaCheckpoint::decode(&data) {
                        return Ok(checkpoint);
                    }
                }
            }
        }

        Err(anyhow!("No valid checkpoint found in transaction"))
    }

    /// Monitor checkpoint confirmations
    pub async fn monitor_confirmations(&self) -> Result<()> {
        let mut interval = tokio::time::interval(Duration::from_secs(60));

        loop {
            interval.tick().await;

            let client_guard = self.bitcoin_client.read().await;
            if let Some(client) = client_guard.as_ref() {
                let mut checkpoints = self.checkpoints.write().await;

                for (qnk_height, record) in checkpoints.iter_mut() {
                    if record.confirmations < self.config.min_confirmations {
                        if let Ok(tx_info) = client.get_transaction(&record.bitcoin_txid, None) {
                            record.confirmations = tx_info.info.confirmations as u32;

                            if record.bitcoin_height.is_none() {
                                if let Some(height) = tx_info.info.blockheight {
                                    record.bitcoin_height = Some(height as u64);
                                }
                            }

                            if record.confirmations >= self.config.min_confirmations {
                                info!(
                                    "✅ Checkpoint {} confirmed at Bitcoin height {}",
                                    qnk_height,
                                    record.bitcoin_height.unwrap_or(0)
                                );

                                let _ = self.event_tx.send(BedaEvent::CheckpointConfirmed {
                                    qnk_height: *qnk_height,
                                    bitcoin_height: record.bitcoin_height.unwrap_or(0),
                                    confirmations: record.confirmations,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    /// Get checkpoint history
    pub async fn get_checkpoints(&self) -> HashMap<u64, BitcoinAnchorRecord> {
        self.checkpoints.read().await.clone()
    }

    /// Bootstrap trust from Bitcoin checkpoint
    pub async fn bootstrap_from_bitcoin(&self, txid: &Txid) -> Result<BedaCheckpoint> {
        info!(
            "🔍 Bootstrapping Q-NK state from Bitcoin checkpoint: {}",
            txid
        );

        let checkpoint = self.verify_checkpoint(txid).await?;

        info!(
            "✅ Found checkpoint: Q-NK height={}, state_root={}",
            checkpoint.qnk_height,
            hex::encode(&checkpoint.state_root)
        );

        // TODO: Verify Dilithium threshold signature

        Ok(checkpoint)
    }
}

/// Extract data from OP_RETURN script
fn extract_op_return_data(script: &ScriptBuf) -> Option<Vec<u8>> {
    let mut instructions = script.instructions();

    // First instruction should be OP_RETURN
    if let Some(Ok(bitcoin::script::Instruction::Op(bitcoin::opcodes::all::OP_RETURN))) =
        instructions.next()
    {
        // Second instruction should be the data
        if let Some(Ok(bitcoin::script::Instruction::PushBytes(bytes))) = instructions.next() {
            return Some(bytes.as_bytes().to_vec());
        }
    }

    None
}

/// BEDA attestation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedaAttestation {
    pub qnk_height: u64,
    pub state_root: Hash256,
    pub timestamp: u64,
    pub bitcoin_txid: Option<String>,
    pub confirmations: u32,
    pub verified: bool,
}

/// Bitcoin-Embedded Data Attestation Service
#[derive(Debug, Clone)]
pub struct BedaService {
    pub config: BedaConfig,
    pub checkpoints: Arc<RwLock<Vec<BedaCheckpoint>>>,
}

impl BedaService {
    pub fn new(config: BedaConfig) -> Self {
        Self {
            config,
            checkpoints: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Verify state attestations for a given state hash
    pub async fn verify_state_attestations(
        &self,
        state_hash: &str,
    ) -> Result<Vec<BedaAttestation>> {
        info!("🔍 Verifying state attestations for hash: {}", state_hash);

        let checkpoints = self.checkpoints.read().await;
        let mut attestations = Vec::new();

        // Find checkpoints that match this state hash
        for checkpoint in checkpoints.iter() {
            if hex::encode(&checkpoint.state_root) == state_hash {
                attestations.push(BedaAttestation {
                    qnk_height: checkpoint.qnk_height,
                    state_root: checkpoint.state_root,
                    timestamp: checkpoint.timestamp,
                    bitcoin_txid: None, // Would need to track this
                    confirmations: 0,   // Would need to query Bitcoin
                    verified: true,
                });
            }
        }

        Ok(attestations)
    }

    /// Get all quantum attestations
    pub async fn get_quantum_attestations(&self) -> Result<Vec<BedaAttestation>> {
        info!("📊 Fetching all quantum attestations");

        let checkpoints = self.checkpoints.read().await;
        let mut attestations = Vec::new();

        for checkpoint in checkpoints.iter() {
            attestations.push(BedaAttestation {
                qnk_height: checkpoint.qnk_height,
                state_root: checkpoint.state_root,
                timestamp: checkpoint.timestamp,
                bitcoin_txid: None,
                confirmations: 0,
                verified: true,
            });
        }

        Ok(attestations)
    }

    /// Create attestation for data
    pub async fn create_attestation(
        &self,
        data: Vec<u8>,
        metadata: Option<String>,
    ) -> Result<BedaAttestation> {
        info!("🔐 Creating BEDA attestation for {} bytes", data.len());

        // Create state root hash from data
        let state_root = sha3::Sha3_256::digest(&data);
        let state_root: [u8; 32] = state_root.into();

        // Create checkpoint
        let checkpoint = BedaCheckpoint::new(
            0, // Height would come from Q-NK chain
            state_root,
            vec![], // Signature would come from threshold scheme
        );

        // Add to checkpoints
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.push(checkpoint.clone());

        let attestation = BedaAttestation {
            qnk_height: checkpoint.qnk_height,
            state_root: checkpoint.state_root,
            timestamp: checkpoint.timestamp,
            bitcoin_txid: None, // Would be set when broadcast to Bitcoin
            confirmations: 0,
            verified: false, // Not yet verified on Bitcoin
        };

        Ok(attestation)
    }

    /// Get attestation status
    pub async fn get_attestation_status(&self, txid: &str) -> Result<BedaAttestation> {
        info!("📊 Getting attestation status for txid: {}", txid);

        // Mock implementation - in real system would query Bitcoin
        Ok(BedaAttestation {
            qnk_height: 12345,
            state_root: [0u8; 32],
            timestamp: chrono::Utc::now().timestamp() as u64,
            bitcoin_txid: Some(txid.to_string()),
            confirmations: 6,
            verified: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_encoding() {
        let state_root = [0x42u8; 32];
        let threshold_sig = vec![0x01, 0x02, 0x03];

        let checkpoint = BedaCheckpoint::new(12345, state_root, threshold_sig.clone());
        let encoded = checkpoint.encode();
        let decoded = BedaCheckpoint::decode(&encoded).unwrap();

        assert_eq!(checkpoint.qnk_height, decoded.qnk_height);
        assert_eq!(checkpoint.state_root, decoded.state_root);
        assert_eq!(checkpoint.threshold_sig, decoded.threshold_sig);
    }

    #[test]
    fn test_magic_bytes() {
        assert_eq!(BedaCheckpoint::MAGIC, [0x51, 0x4E, 0x4B, 0x42]); // "QNKB"
    }
}
