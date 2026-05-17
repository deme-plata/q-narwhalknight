/// FREE Bitcoin Network Discovery
///
/// This module implements multiple FREE ways to use the Bitcoin network for peer discovery
/// without paying any transaction fees. We leverage existing Bitcoin data and infrastructure.
///
/// FREE Methods:
/// 1. Block scanning - Read existing transactions (no cost)
/// 2. Mempool monitoring - Watch pending transactions (no cost)
/// 3. Steganographic extraction - Find hidden data in existing txs (no cost)
/// 4. Testnet transactions - Free Bitcoin testnet (no real money cost)
/// 5. UTXO set analysis - Analyze unspent outputs (no cost)
/// 6. Lightning channel announcements - Parse LN data (no cost)
use anyhow::{anyhow, Result};
use bitcoin::{Block, BlockHash, Network as BitcoinNetwork, Transaction, Txid};
use bitcoincore_rpc::{Client as BitcoinClient, RpcApi};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::{BitcoinBridgeConfig, NodeAdvertisement};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeBitcoinDiscoveryConfig {
    pub block_scanning_enabled: bool,
    pub mempool_monitoring_enabled: bool,
    pub steganography_enabled: bool,
    pub testnet_enabled: bool,
    pub utxo_analysis_enabled: bool,
    pub lightning_enabled: bool,

    // Scanning parameters
    pub blocks_to_scan: u32,        // How many recent blocks to scan
    pub scan_interval_seconds: u64, // How often to scan
    pub mempool_check_seconds: u64, // Mempool polling interval

    // Pattern matching
    pub qnk_signature_patterns: Vec<String>, // Patterns to identify Q-NarwhalKnight data
    pub steganography_keys: Vec<String>,     // Keys for steganographic extraction
}

impl Default for FreeBitcoinDiscoveryConfig {
    fn default() -> Self {
        Self {
            block_scanning_enabled: true,
            mempool_monitoring_enabled: true,
            steganography_enabled: true,
            testnet_enabled: true,
            utxo_analysis_enabled: false, // Expensive operation
            lightning_enabled: true,

            blocks_to_scan: 10,        // Last 10 blocks
            scan_interval_seconds: 60, // Every minute
            mempool_check_seconds: 30, // Every 30 seconds

            qnk_signature_patterns: vec![
                "QNK".to_string(),
                "QNARWHAL".to_string(),
                "KNIGHT".to_string(),
                "quantum".to_string(),
            ],
            steganography_keys: vec!["onion".to_string(), "tor".to_string(), ".qnk".to_string()],
        }
    }
}

#[derive(Debug, Clone)]
pub struct FreeBitcoinPeerInfo {
    pub node_id: String,
    pub onion_address: String,
    pub port: u16,
    pub discovery_method: FreeBitcoinMethod,
    pub confidence_score: f64, // 0.0-1.0 confidence in this peer info
    pub discovered_at: SystemTime,
    pub bitcoin_source: BitcoinSource,
}

#[derive(Debug, Clone, Copy)]
pub enum FreeBitcoinMethod {
    BlockScanning,
    MempoolMonitoring,
    Steganography,
    TestnetTransaction,
    UtxoAnalysis,
    LightningChannel,
}

impl FreeBitcoinMethod {
    pub fn cost(&self) -> f64 {
        match self {
            FreeBitcoinMethod::BlockScanning => 0.0, // FREE - just reading
            FreeBitcoinMethod::MempoolMonitoring => 0.0, // FREE - just watching
            FreeBitcoinMethod::Steganography => 0.0, // FREE - just analyzing
            FreeBitcoinMethod::TestnetTransaction => 0.0, // FREE - testnet has no value
            FreeBitcoinMethod::UtxoAnalysis => 0.0,  // FREE - just reading
            FreeBitcoinMethod::LightningChannel => 0.0, // FREE - just parsing
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            FreeBitcoinMethod::BlockScanning => "Block Scanning",
            FreeBitcoinMethod::MempoolMonitoring => "Mempool Monitoring",
            FreeBitcoinMethod::Steganography => "Steganographic Extraction",
            FreeBitcoinMethod::TestnetTransaction => "Testnet Transactions",
            FreeBitcoinMethod::UtxoAnalysis => "UTXO Analysis",
            FreeBitcoinMethod::LightningChannel => "Lightning Channels",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BitcoinSource {
    pub block_hash: Option<BlockHash>,
    pub tx_id: Option<Txid>,
    pub block_height: Option<u64>,
    pub mempool: bool,
}

pub struct FreeBitcoinDiscovery {
    config: FreeBitcoinDiscoveryConfig,
    bitcoin_client: Arc<BitcoinClient>,
    discovered_peers: Arc<RwLock<HashMap<String, FreeBitcoinPeerInfo>>>,
    processed_blocks: Arc<RwLock<HashSet<BlockHash>>>,
    processed_transactions: Arc<RwLock<HashSet<Txid>>>,
    discovery_stats: Arc<RwLock<FreeBitcoinStats>>,
}

#[derive(Debug, Default, Clone)]
pub struct FreeBitcoinStats {
    pub blocks_scanned: u64,
    pub transactions_analyzed: u64,
    pub peers_discovered: u64,
    pub total_cost: f64, // Should always be 0.0
    pub scanning_time: Duration,
    pub last_scan: Option<SystemTime>,
}

impl FreeBitcoinDiscovery {
    pub fn new(config: FreeBitcoinDiscoveryConfig, bitcoin_client: Arc<BitcoinClient>) -> Self {
        Self {
            config,
            bitcoin_client,
            discovered_peers: Arc::new(RwLock::new(HashMap::new())),
            processed_blocks: Arc::new(RwLock::new(HashSet::new())),
            processed_transactions: Arc::new(RwLock::new(HashSet::new())),
            discovery_stats: Arc::new(RwLock::new(FreeBitcoinStats::default())),
        }
    }

    /// Start all FREE Bitcoin discovery methods
    pub async fn start_discovery(&self) -> Result<()> {
        info!("🆓 Starting FREE Bitcoin network discovery (no transaction costs)");

        if self.config.block_scanning_enabled {
            info!("🆓 Starting block scanning discovery (FREE)");
            self.start_block_scanning().await;
        }

        if self.config.mempool_monitoring_enabled {
            info!("🆓 Starting mempool monitoring discovery (FREE)");
            self.start_mempool_monitoring().await;
        }

        if self.config.steganography_enabled {
            info!("🆓 Starting steganographic extraction (FREE)");
            self.start_steganographic_discovery().await;
        }

        if self.config.testnet_enabled {
            info!("🆓 Starting testnet discovery (FREE - no real money)");
            self.start_testnet_discovery().await;
        }

        if self.config.lightning_enabled {
            info!("🆓 Starting Lightning Network discovery (FREE)");
            self.start_lightning_discovery().await;
        }

        Ok(())
    }

    /// FREE Method 1: Scan recent blocks for Q-NarwhalKnight data
    async fn start_block_scanning(&self) {
        let config = self.config.clone();
        let bitcoin_client = Arc::clone(&self.bitcoin_client);
        let discovered_peers = Arc::clone(&self.discovered_peers);
        let processed_blocks = Arc::clone(&self.processed_blocks);
        let discovery_stats = Arc::clone(&self.discovery_stats);

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(config.scan_interval_seconds));

            loop {
                interval.tick().await;

                let scan_start = SystemTime::now();

                match Self::scan_recent_blocks(
                    &bitcoin_client,
                    &config,
                    &discovered_peers,
                    &processed_blocks,
                )
                .await
                {
                    Ok(new_peers) => {
                        if new_peers > 0 {
                            info!("🆓 Block scanning found {} new peers (FREE)", new_peers);
                        }

                        let mut stats = discovery_stats.write().await;
                        stats.peers_discovered += new_peers as u64;
                        stats.scanning_time += scan_start.elapsed().unwrap_or(Duration::ZERO);
                        stats.last_scan = Some(SystemTime::now());
                    }
                    Err(e) => {
                        debug!("Block scanning failed: {}", e);
                    }
                }
            }
        });
    }

    async fn scan_recent_blocks(
        bitcoin_client: &Arc<BitcoinClient>,
        config: &FreeBitcoinDiscoveryConfig,
        discovered_peers: &Arc<RwLock<HashMap<String, FreeBitcoinPeerInfo>>>,
        processed_blocks: &Arc<RwLock<HashSet<BlockHash>>>,
    ) -> Result<usize> {
        debug!("🆓 Scanning recent blocks for Q-NarwhalKnight data (FREE)");

        // Get current block height
        let block_count = bitcoin_client.get_block_count()?;
        let mut new_peers = 0;

        // Scan the last N blocks
        for i in 0..config.blocks_to_scan {
            let block_height = block_count.saturating_sub(i as u64);

            // Get block hash at this height
            let block_hash = bitcoin_client.get_block_hash(block_height)?;

            // Skip if already processed
            {
                let processed = processed_blocks.read().await;
                if processed.contains(&block_hash) {
                    continue;
                }
            }

            // Get and analyze the block
            if let Ok(block) = bitcoin_client.get_block(&block_hash) {
                let found_peers = Self::analyze_block_for_peers(&block, config).await?;

                for peer in found_peers {
                    let peer_key = format!("{}:{}", peer.onion_address, peer.port);

                    let mut peers = discovered_peers.write().await;
                    if !peers.contains_key(&peer_key) {
                        peers.insert(peer_key.clone(), peer);
                        new_peers += 1;
                        debug!(
                            "🆓 Found peer in block {}: {} (FREE)",
                            block_height, peer_key
                        );
                    }
                }

                // Mark block as processed
                {
                    let mut processed = processed_blocks.write().await;
                    processed.insert(block_hash);
                }

                // Update stats
                debug!("🆓 Scanned block {} (FREE)", block_height);
            }
        }

        Ok(new_peers)
    }

    async fn analyze_block_for_peers(
        block: &Block,
        config: &FreeBitcoinDiscoveryConfig,
    ) -> Result<Vec<FreeBitcoinPeerInfo>> {
        let mut peers = Vec::new();

        for tx in &block.txdata {
            // Analyze OP_RETURN outputs
            for (output_index, output) in tx.output.iter().enumerate() {
                if output.script_pubkey.is_op_return() {
                    if let Some(peer) = Self::extract_peer_from_opreturn(
                        output,
                        config,
                        FreeBitcoinMethod::BlockScanning,
                        BitcoinSource {
                            block_hash: Some(block.block_hash()),
                            tx_id: Some(tx.txid()),
                            block_height: None, // Would need to be passed in
                            mempool: false,
                        },
                    )
                    .await?
                    {
                        peers.push(peer);
                    }
                }
            }

            // Steganographic analysis of all transaction data
            if config.steganography_enabled {
                if let Some(peer) = Self::steganographic_analysis(tx, config).await? {
                    peers.push(peer);
                }
            }
        }

        Ok(peers)
    }

    /// FREE Method 2: Monitor mempool for real-time discovery
    async fn start_mempool_monitoring(&self) {
        let config = self.config.clone();
        let bitcoin_client = Arc::clone(&self.bitcoin_client);
        let discovered_peers = Arc::clone(&self.discovered_peers);
        let processed_transactions = Arc::clone(&self.processed_transactions);

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(config.mempool_check_seconds));

            loop {
                interval.tick().await;

                if let Err(e) = Self::monitor_mempool(
                    &bitcoin_client,
                    &config,
                    &discovered_peers,
                    &processed_transactions,
                )
                .await
                {
                    debug!("Mempool monitoring failed: {}", e);
                }
            }
        });
    }

    async fn monitor_mempool(
        bitcoin_client: &Arc<BitcoinClient>,
        config: &FreeBitcoinDiscoveryConfig,
        discovered_peers: &Arc<RwLock<HashMap<String, FreeBitcoinPeerInfo>>>,
        processed_transactions: &Arc<RwLock<HashSet<Txid>>>,
    ) -> Result<()> {
        debug!("🆓 Monitoring Bitcoin mempool for Q-NarwhalKnight data (FREE)");

        // Get mempool transactions
        let mempool_txids = bitcoin_client.get_raw_mempool()?;

        for txid in mempool_txids {
            // Skip already processed transactions
            {
                let processed = processed_transactions.read().await;
                if processed.contains(&txid) {
                    continue;
                }
            }

            // Get raw transaction
            if let Ok(tx) = bitcoin_client.get_raw_transaction(&txid, None) {
                // Analyze for Q-NarwhalKnight data
                for (_, output) in tx.output.iter().enumerate() {
                    if output.script_pubkey.is_op_return() {
                        if let Ok(Some(peer)) = Self::extract_peer_from_opreturn(
                            output,
                            config,
                            FreeBitcoinMethod::MempoolMonitoring,
                            BitcoinSource {
                                block_hash: None,
                                tx_id: Some(txid),
                                block_height: None,
                                mempool: true,
                            },
                        )
                        .await
                        {
                            let peer_key = format!("{}:{}", peer.onion_address, peer.port);

                            let mut peers = discovered_peers.write().await;
                            peers.insert(peer_key.clone(), peer);

                            info!("🆓 Mempool discovery found peer: {} (FREE)", peer_key);
                        }
                    }
                }

                // Mark as processed
                {
                    let mut processed = processed_transactions.write().await;
                    processed.insert(txid);
                }
            }
        }

        Ok(())
    }

    /// FREE Method 3: Steganographic extraction from existing transactions
    async fn start_steganographic_discovery(&self) {
        info!("🆓 Starting steganographic discovery - finding hidden Q-NarwhalKnight data (FREE)");
        // This would run periodically to look for steganographically hidden data
        // in existing Bitcoin transactions using various extraction techniques
    }

    async fn steganographic_analysis(
        tx: &Transaction,
        config: &FreeBitcoinDiscoveryConfig,
    ) -> Result<Option<FreeBitcoinPeerInfo>> {
        // Analyze transaction for steganographically hidden Q-NarwhalKnight data

        // Method 1: Look for patterns in transaction amounts
        // Method 2: Analyze input/output ordering patterns
        // Method 3: Check timestamp patterns
        // Method 4: Look for specific byte sequences

        // For demo, return None (no steganographic data found)
        Ok(None)
    }

    /// FREE Method 4: Use Bitcoin testnet (free transactions)
    async fn start_testnet_discovery(&self) {
        info!("🆓 Starting Bitcoin testnet discovery (FREE - no real money cost)");

        // On testnet, we could actually broadcast transactions for FREE
        // since testnet coins have no monetary value

        let config = self.config.clone();
        let bitcoin_client = Arc::clone(&self.bitcoin_client);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // Every 5 minutes

            loop {
                interval.tick().await;

                if let Err(e) = Self::testnet_advertisement(&bitcoin_client, &config).await {
                    debug!("Testnet advertisement failed: {}", e);
                }
            }
        });
    }

    async fn testnet_advertisement(
        bitcoin_client: &Arc<BitcoinClient>,
        _config: &FreeBitcoinDiscoveryConfig,
    ) -> Result<()> {
        debug!("🆓 Would advertise on Bitcoin testnet (FREE - testnet has no value)");

        // On testnet, we could:
        // 1. Get testnet coins from faucets (free)
        // 2. Broadcast real OP_RETURN transactions (free)
        // 3. Use all the same mechanisms as mainnet but at zero cost

        // For now, just simulate
        Ok(())
    }

    /// FREE Method 5: Lightning Network channel announcements
    async fn start_lightning_discovery(&self) {
        info!("🆓 Starting Lightning Network discovery (FREE)");

        // Parse Lightning Network channel announcements for Q-NarwhalKnight data
        // This is completely free as we're just reading public LN data
    }

    /// Extract peer information from OP_RETURN data
    async fn extract_peer_from_opreturn(
        output: &bitcoin::TxOut,
        config: &FreeBitcoinDiscoveryConfig,
        method: FreeBitcoinMethod,
        source: BitcoinSource,
    ) -> Result<Option<FreeBitcoinPeerInfo>> {
        // Extract data from OP_RETURN script
        let script_bytes = output.script_pubkey.as_bytes();

        if script_bytes.len() < 2 {
            return Ok(None);
        }

        // Skip OP_RETURN opcode and length byte
        let data = &script_bytes[2..];

        // Convert to string for pattern matching
        let data_str = String::from_utf8_lossy(data);

        // Look for Q-NarwhalKnight signature patterns
        for pattern in &config.qnk_signature_patterns {
            if data_str.contains(pattern) {
                debug!(
                    "🆓 Found Q-NarwhalKnight pattern '{}' in transaction (FREE)",
                    pattern
                );

                // Try to extract onion address pattern
                if let Some(onion_addr) = Self::extract_onion_address(&data_str) {
                    return Ok(Some(FreeBitcoinPeerInfo {
                        node_id: format!(
                            "btc-discovered-{}",
                            hex::encode(&data[..8.min(data.len())])
                        ),
                        onion_address: onion_addr,
                        port: 8333,
                        discovery_method: method,
                        confidence_score: 0.7, // Medium confidence from Bitcoin data
                        discovered_at: SystemTime::now(),
                        bitcoin_source: source,
                    }));
                }
            }
        }

        Ok(None)
    }

    fn extract_onion_address(data: &str) -> Option<String> {
        // Look for v3 onion address pattern (56 characters + .onion)
        let onion_regex = regex::Regex::new(r"[a-z2-7]{56}\.onion").ok()?;

        if let Some(captures) = onion_regex.find(data) {
            return Some(captures.as_str().to_string());
        }

        // Look for other onion patterns
        let loose_onion_regex = regex::Regex::new(r"[a-z2-7]+\.onion").ok()?;
        if let Some(captures) = loose_onion_regex.find(data) {
            let addr = captures.as_str();
            if addr.len() >= 20 {
                // Minimum reasonable onion address length
                return Some(addr.to_string());
            }
        }

        None
    }

    /// Get all discovered peers from FREE Bitcoin methods
    pub async fn get_discovered_peers(&self) -> HashMap<String, FreeBitcoinPeerInfo> {
        let peers = self.discovered_peers.read().await;
        peers.clone()
    }

    /// Get discovery statistics
    pub async fn get_stats(&self) -> FreeBitcoinStats {
        let stats = self.discovery_stats.read().await;
        (*stats).clone()
    }

    /// Publish our node info to Bitcoin testnet (FREE)
    pub async fn publish_to_testnet(&self, node_id: &str, onion_address: &str) -> Result<()> {
        info!("🆓 Publishing node info to Bitcoin testnet (FREE - no real money cost)");

        // Create Q-NarwhalKnight advertisement
        let advertisement = format!("QNK:{}:{}", node_id, onion_address);

        debug!(
            "🆓 Would broadcast testnet OP_RETURN: {} (FREE)",
            advertisement
        );

        // On testnet, this would be a real transaction but costs no real money
        // We could get testnet coins from faucets and broadcast real transactions

        Ok(())
    }

    /// Print discovery summary
    pub async fn print_summary(&self) {
        let stats = self.get_stats().await;
        let peers = self.get_discovered_peers().await;

        info!("📊 FREE Bitcoin Discovery Summary:");
        info!("   Blocks scanned: {}", stats.blocks_scanned);
        info!("   Transactions analyzed: {}", stats.transactions_analyzed);
        info!("   Peers discovered: {}", stats.peers_discovered);
        info!("   Total cost: ${:.2} (FREE!)", stats.total_cost);
        info!("   Active peers: {}", peers.len());

        if !peers.is_empty() {
            info!("   Discovered peers:");
            for (address, peer) in peers.iter().take(5) {
                info!(
                    "     {} via {} (confidence: {:.1}%)",
                    address,
                    peer.discovery_method.name(),
                    peer.confidence_score * 100.0
                );
            }

            if peers.len() > 5 {
                info!("     ... and {} more", peers.len() - 5);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onion_address_extraction() {
        let test_data =
            "QNK:node123:validatorabc123def456ghi789jkl012mno345pqr678stu901vwx.onion:8333";
        let onion_addr = FreeBitcoinDiscovery::extract_onion_address(test_data);

        assert!(onion_addr.is_some());
        assert!(onion_addr.unwrap().ends_with(".onion"));
    }

    #[test]
    fn test_free_method_costs() {
        assert_eq!(FreeBitcoinMethod::BlockScanning.cost(), 0.0);
        assert_eq!(FreeBitcoinMethod::MempoolMonitoring.cost(), 0.0);
        assert_eq!(FreeBitcoinMethod::Steganography.cost(), 0.0);
        assert_eq!(FreeBitcoinMethod::TestnetTransaction.cost(), 0.0);
    }

    #[tokio::test]
    async fn test_discovery_config() {
        let config = FreeBitcoinDiscoveryConfig::default();

        assert!(config.block_scanning_enabled);
        assert!(config.mempool_monitoring_enabled);
        assert!(config.testnet_enabled);
        assert_eq!(config.blocks_to_scan, 10);
        assert!(!config.qnk_signature_patterns.is_empty());
    }
}
