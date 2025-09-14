// Q-NarwhalKnight Bitcoin ZMQ Real-time Monitor
// Receives instant notifications for new blocks and transactions

use std::sync::Arc;
use tokio::sync::broadcast;
use zmq::{Context, Socket, SocketType};
use serde::{Deserialize, Serialize};
use sha256::digest;
use tracing::{info, warn, error, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitcoinBlockNotification {
    pub hash: String,
    pub height: u64,
    pub timestamp: u64,
    pub raw_block: Vec<u8>,
    pub merkle_root: String,
    pub previous_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitcoinTxNotification {
    pub txid: String,
    pub raw_tx: Vec<u8>,
    pub size: usize,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QNKBlockstamp {
    pub qnk_block_hash: String,
    pub btc_block_hash: String,
    pub btc_height: u64,
    pub timestamp: u64,
    pub merkle_root: String,
    pub created_at: u64,
}

pub struct BitcoinZMQMonitor {
    context: Arc<Context>,
    block_sender: broadcast::Sender<BitcoinBlockNotification>,
    tx_sender: broadcast::Sender<BitcoinTxNotification>,
    blockstamp_sender: broadcast::Sender<QNKBlockstamp>,
    bitcoin_rpc_url: String,
    zmq_block_url: String,
    zmq_tx_url: String,
}

impl BitcoinZMQMonitor {
    pub fn new() -> anyhow::Result<Self> {
        let context = Arc::new(Context::new());
        let (block_sender, _) = broadcast::channel(1000);
        let (tx_sender, _) = broadcast::channel(10000);
        let (blockstamp_sender, _) = broadcast::channel(1000);
        
        Ok(Self {
            context,
            block_sender,
            tx_sender,
            blockstamp_sender,
            bitcoin_rpc_url: "http://localhost:8332".to_string(),
            zmq_block_url: "tcp://localhost:28332".to_string(),
            zmq_tx_url: "tcp://localhost:28333".to_string(),
        })
    }
    
    pub fn block_receiver(&self) -> broadcast::Receiver<BitcoinBlockNotification> {
        self.block_sender.subscribe()
    }
    
    pub fn tx_receiver(&self) -> broadcast::Receiver<BitcoinTxNotification> {
        self.tx_sender.subscribe()
    }
    
    pub fn blockstamp_receiver(&self) -> broadcast::Receiver<QNKBlockstamp> {
        self.blockstamp_sender.subscribe()
    }
    
    pub async fn start_monitoring(&self) -> anyhow::Result<()> {
        info!("🔗 Starting Bitcoin ZMQ Monitor for Q-NarwhalKnight");
        info!("📡 Block notifications: {}", self.zmq_block_url);
        info!("💰 Transaction notifications: {}", self.zmq_tx_url);
        
        // Start block monitoring task
        let block_monitor = self.start_block_monitor().await?;
        
        // Start transaction monitoring task  
        let tx_monitor = self.start_tx_monitor().await?;
        
        // Wait for both monitors
        tokio::select! {
            result = block_monitor => {
                error!("Block monitor stopped: {:?}", result);
            }
            result = tx_monitor => {
                error!("Transaction monitor stopped: {:?}", result);
            }
        }
        
        Ok(())
    }
    
    async fn start_block_monitor(&self) -> anyhow::Result<()> {
        let context = self.context.clone();
        let sender = self.block_sender.clone();
        let blockstamp_sender = self.blockstamp_sender.clone();
        let block_url = self.zmq_block_url.clone();
        
        tokio::task::spawn_blocking(move || {
            info!("🔵 Starting Bitcoin block monitor on {}", block_url);
            
            let socket = context.socket(SocketType::SUB).unwrap();
            socket.connect(&block_url).unwrap();
            socket.set_subscribe(b"rawblock").unwrap();
            
            let mut block_count = 0u64;
            
            loop {
                let msg = match socket.recv_multipart(0) {
                    Ok(msg) => msg,
                    Err(e) => {
                        error!("ZMQ block receive error: {}", e);
                        std::thread::sleep(std::time::Duration::from_secs(1));
                        continue;
                    }
                };
                
                if msg.len() >= 2 {
                    let topic = String::from_utf8_lossy(&msg[0]);
                    let raw_block = &msg[1];
                    
                    if topic == "rawblock" {
                        block_count += 1;
                        let block_hash = digest(raw_block);
                        let timestamp = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs();
                        
                        info!("📦 New Bitcoin block #{}: {}", block_count, &block_hash[..16]);
                        
                        // Parse block header for detailed information
                        let (height, merkle_root, previous_hash) = 
                            Self::parse_block_header(raw_block).unwrap_or((0, "unknown".to_string(), "unknown".to_string()));
                        
                        let notification = BitcoinBlockNotification {
                            hash: block_hash.clone(),
                            height,
                            timestamp,
                            raw_block: raw_block.to_vec(),
                            merkle_root: merkle_root.clone(),
                            previous_hash,
                        };
                        
                        // Send block notification
                        if let Err(_) = sender.send(notification) {
                            warn!("No block subscribers active");
                        }
                        
                        // Create Q-NarwhalKnight blockstamp
                        let qnk_block_hash = format!("qnk_block_{}", block_count);
                        let blockstamp = QNKBlockstamp {
                            qnk_block_hash,
                            btc_block_hash: block_hash,
                            btc_height: height,
                            timestamp,
                            merkle_root,
                            created_at: timestamp,
                        };
                        
                        info!("🔗 Created QNK blockstamp: BTC #{} → QNK #{}", height, block_count);
                        
                        if let Err(_) = blockstamp_sender.send(blockstamp) {
                            warn!("No blockstamp subscribers active");
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    async fn start_tx_monitor(&self) -> anyhow::Result<()> {
        let context = self.context.clone();
        let sender = self.tx_sender.clone();
        let tx_url = self.zmq_tx_url.clone();
        
        tokio::task::spawn_blocking(move || {
            info!("💰 Starting Bitcoin transaction monitor on {}", tx_url);
            
            let socket = context.socket(SocketType::SUB).unwrap();
            socket.connect(&tx_url).unwrap();
            socket.set_subscribe(b"rawtx").unwrap();
            
            let mut tx_count = 0u64;
            
            loop {
                let msg = match socket.recv_multipart(0) {
                    Ok(msg) => msg,
                    Err(e) => {
                        error!("ZMQ transaction receive error: {}", e);
                        std::thread::sleep(std::time::Duration::from_secs(1));
                        continue;
                    }
                };
                
                if msg.len() >= 2 {
                    let topic = String::from_utf8_lossy(&msg[0]);
                    let raw_tx = &msg[1];
                    
                    if topic == "rawtx" {
                        tx_count += 1;
                        let txid = digest(raw_tx);
                        let timestamp = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs();
                        
                        debug!("💳 New Bitcoin transaction #{}: {}", tx_count, &txid[..16]);
                        
                        let notification = BitcoinTxNotification {
                            txid,
                            raw_tx: raw_tx.to_vec(),
                            size: raw_tx.len(),
                            timestamp,
                        };
                        
                        if let Err(_) = sender.send(notification) {
                            // Don't log for transactions as there might be many without subscribers
                        }
                        
                        // Log every 100 transactions to avoid spam
                        if tx_count % 100 == 0 {
                            info!("💰 Processed {} Bitcoin transactions", tx_count);
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    fn parse_block_header(raw_block: &[u8]) -> Option<(u64, String, String)> {
        // Simple block header parsing (first 80 bytes)
        if raw_block.len() < 80 {
            return None;
        }
        
        // Extract previous block hash (bytes 4-35, reversed)
        let mut prev_hash = raw_block[4..36].to_vec();
        prev_hash.reverse();
        let previous_hash = hex::encode(prev_hash);
        
        // Extract merkle root (bytes 36-67, reversed)
        let mut merkle = raw_block[36..68].to_vec();
        merkle.reverse();
        let merkle_root = hex::encode(merkle);
        
        // Height would need to be determined via RPC call
        // For now, use a placeholder
        let height = 0; // This should be fetched via RPC
        
        Some((height, merkle_root, previous_hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_zmq_monitor_creation() {
        let monitor = BitcoinZMQMonitor::new().unwrap();
        assert!(!monitor.zmq_block_url.is_empty());
        assert!(!monitor.zmq_tx_url.is_empty());
    }
}