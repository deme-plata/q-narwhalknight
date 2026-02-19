/// Real Bitcoin RPC Client - Production Implementation
/// 
/// Connects to actual Bitcoin networks (mainnet, testnet, regtest) via JSON-RPC
/// Supports both direct connection and Tor proxy routing
use anyhow::{anyhow, Result};
use bitcoin::{
    Address, Block, BlockHash, Transaction, Txid, 
    network::constants::Network,
    consensus::encode::deserialize,
    hashes::hex::FromHex,
};
use reqwest::{Client, Proxy};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    str::FromStr,
    time::Duration,
};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
// url crate removed - using string-based URL handling instead

/// Bitcoin network configuration
#[derive(Debug, Clone)]
pub struct BitcoinConfig {
    pub network: Network,
    pub rpc_url: String,
    pub rpc_user: String,
    pub rpc_password: String,
    pub tor_proxy: Option<String>,
    pub connection_timeout: Duration,
    pub request_timeout: Duration,
    pub max_retries: u32,
}

impl Default for BitcoinConfig {
    fn default() -> Self {
        Self {
            network: Network::Testnet,
            rpc_url: "http://127.0.0.1:18332".to_string(),
            rpc_user: "bitcoin".to_string(),
            rpc_password: "password".to_string(),
            tor_proxy: None,
            connection_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            max_retries: 3,
        }
    }
}

/// Bitcoin RPC response structure
#[derive(Debug, Deserialize)]
struct RpcResponse<T> {
    result: Option<T>,
    error: Option<RpcError>,
    id: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct RpcError {
    code: i32,
    message: String,
}

/// Block information from Bitcoin RPC
#[derive(Debug, Deserialize)]
pub struct BlockInfo {
    pub hash: String,
    pub height: u64,
    pub time: u64,
    pub tx: Vec<String>,
    pub size: u64,
    pub weight: u64,
    pub difficulty: f64,
}

/// Transaction information
#[derive(Debug, Deserialize)]
pub struct TxInfo {
    pub txid: String,
    pub size: u64,
    pub vsize: u64,
    pub weight: u64,
    pub version: u32,
    pub locktime: u32,
    pub vin: Vec<TxInput>,
    pub vout: Vec<TxOutput>,
    pub hex: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TxInput {
    pub txid: Option<String>,
    pub vout: Option<u32>,
    pub scriptSig: Option<ScriptSig>,
    pub sequence: u32,
}

#[derive(Debug, Deserialize)]
pub struct TxOutput {
    pub value: f64,
    pub n: u32,
    pub scriptPubKey: ScriptPubKey,
}

#[derive(Debug, Deserialize)]
pub struct ScriptSig {
    pub asm: String,
    pub hex: String,
}

#[derive(Debug, Deserialize)]
pub struct ScriptPubKey {
    pub asm: String,
    pub hex: String,
    #[serde(rename = "type")]
    pub script_type: String,
    pub addresses: Option<Vec<String>>,
}

/// Mempool information
#[derive(Debug, Deserialize)]
pub struct MempoolInfo {
    pub loaded: bool,
    pub size: u64,
    pub bytes: u64,
    pub usage: u64,
    pub maxmempool: u64,
    pub mempoolminfee: f64,
    pub minrelaytxfee: f64,
}

/// Network information
#[derive(Debug, Deserialize)]
pub struct NetworkInfo {
    pub version: u64,
    pub subversion: String,
    pub protocolversion: u64,
    pub localservices: String,
    pub localrelay: bool,
    pub timeoffset: i64,
    pub networkactive: bool,
    pub connections: u32,
    pub connections_in: u32,
    pub connections_out: u32,
    pub networks: Vec<NetworkData>,
}

#[derive(Debug, Deserialize)]
pub struct NetworkData {
    pub name: String,
    pub limited: bool,
    pub reachable: bool,
    pub proxy: String,
    pub proxy_randomize_credentials: bool,
}

/// Peer information
#[derive(Debug, Clone, Deserialize)]
pub struct PeerInfo {
    pub id: u32,
    pub addr: String,
    pub addrbind: String,
    pub addrlocal: Option<String>,
    pub services: String,
    pub relaytxes: bool,
    pub lastsend: u64,
    pub lastrecv: u64,
    pub conntime: u64,
    pub timeoffset: i64,
    pub pingtime: Option<f64>,
    pub minping: Option<f64>,
    pub version: u32,
    pub subver: String,
    pub inbound: bool,
    pub addnode: bool,
    pub startingheight: u64,
    pub bytessent: u64,
    pub bytesrecv: u64,
}

/// Production Bitcoin RPC Client
pub struct RealBitcoinClient {
    config: BitcoinConfig,
    client: Client,
    request_id: std::sync::atomic::AtomicU64,
}

impl RealBitcoinClient {
    /// Create a new Bitcoin RPC client
    pub async fn new(config: BitcoinConfig) -> Result<Self> {
        let mut client_builder = Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connection_timeout)
            .user_agent("Q-NarwhalKnight/1.0");

        // Configure Tor proxy if specified
        if let Some(proxy_url) = &config.tor_proxy {
            let proxy = Proxy::all(proxy_url)?;
            client_builder = client_builder.proxy(proxy);
            info!("Bitcoin client configured with Tor proxy: {}", proxy_url);
        }

        let client = client_builder.build()?;

        // Test connection
        let bitcoin_client = Self {
            config,
            client,
            request_id: std::sync::atomic::AtomicU64::new(1),
        };

        // Verify connection by getting network info
        match bitcoin_client.get_network_info().await {
            Ok(info) => {
                info!("Connected to Bitcoin {} node: {}", 
                      bitcoin_client.config.network, info.subversion);
                info!("Protocol version: {}, Connections: {}", 
                      info.protocolversion, info.connections);
            }
            Err(e) => {
                error!("Failed to connect to Bitcoin node: {}", e);
                return Err(anyhow!("Bitcoin connection failed: {}", e));
            }
        }

        Ok(bitcoin_client)
    }

    /// Make an RPC call to the Bitcoin node
    async fn rpc_call<T: for<'de> Deserialize<'de>>(
        &self, 
        method: &str, 
        params: Value
    ) -> Result<T> {
        let id = self.request_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        
        let payload = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });

        let mut attempts = 0;
        let mut last_error = None;

        while attempts < self.config.max_retries {
            attempts += 1;

            let response = self.client
                .post(&self.config.rpc_url)
                .basic_auth(&self.config.rpc_user, Some(&self.config.rpc_password))
                .json(&payload)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let rpc_response: RpcResponse<T> = resp.json().await?;
                        
                        if let Some(error) = rpc_response.error {
                            return Err(anyhow!("RPC error {}: {}", error.code, error.message));
                        }
                        
                        if let Some(result) = rpc_response.result {
                            debug!("RPC call {} succeeded on attempt {}", method, attempts);
                            return Ok(result);
                        } else {
                            return Err(anyhow!("RPC call returned null result"));
                        }
                    } else {
                        let error = anyhow!("HTTP error: {}", resp.status());
                        last_error = Some(error);
                    }
                }
                Err(e) => {
                    last_error = Some(anyhow!("Request failed: {}", e));
                }
            }

            if attempts < self.config.max_retries {
                warn!("RPC call {} failed (attempt {}), retrying...", method, attempts);
                sleep(Duration::from_millis(1000 * attempts as u64)).await;
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("All RPC attempts failed")))
    }

    /// Get blockchain information
    pub async fn get_blockchain_info(&self) -> Result<Value> {
        self.rpc_call("getblockchaininfo", json!([])).await
    }

    /// Get network information
    pub async fn get_network_info(&self) -> Result<NetworkInfo> {
        self.rpc_call("getnetworkinfo", json!([])).await
    }

    /// Get mempool information
    pub async fn get_mempool_info(&self) -> Result<MempoolInfo> {
        self.rpc_call("getmempoolinfo", json!([])).await
    }

    /// Get connected peers
    pub async fn get_peer_info(&self) -> Result<Vec<PeerInfo>> {
        self.rpc_call("getpeerinfo", json!([])).await
    }

    /// Get best block hash
    pub async fn get_best_block_hash(&self) -> Result<String> {
        self.rpc_call("getbestblockhash", json!([])).await
    }

    /// Get block by hash
    pub async fn get_block(&self, block_hash: &str, verbosity: u8) -> Result<Value> {
        self.rpc_call("getblock", json!([block_hash, verbosity])).await
    }

    /// Get block info by hash
    pub async fn get_block_info(&self, block_hash: &str) -> Result<BlockInfo> {
        self.rpc_call("getblock", json!([block_hash, 1])).await
    }

    /// Get raw block
    pub async fn get_raw_block(&self, block_hash: &str) -> Result<Vec<u8>> {
        let hex: String = self.rpc_call("getblock", json!([block_hash, 0])).await?;
        Ok(Vec::from_hex(&hex)?)
    }

    /// Get block hash by height
    pub async fn get_block_hash(&self, height: u64) -> Result<String> {
        self.rpc_call("getblockhash", json!([height])).await
    }

    /// Get block count (current height)
    pub async fn get_block_count(&self) -> Result<u64> {
        self.rpc_call("getblockcount", json!([])).await
    }

    /// Get transaction by ID
    pub async fn get_transaction(&self, txid: &str) -> Result<TxInfo> {
        self.rpc_call("getrawtransaction", json!([txid, true])).await
    }

    /// Get raw transaction
    pub async fn get_raw_transaction(&self, txid: &str) -> Result<Vec<u8>> {
        let hex: String = self.rpc_call("getrawtransaction", json!([txid, false])).await?;
        Ok(Vec::from_hex(&hex)?)
    }

    /// Send raw transaction
    pub async fn send_raw_transaction(&self, tx_hex: &str) -> Result<String> {
        self.rpc_call("sendrawtransaction", json!([tx_hex])).await
    }

    /// Get mempool entries
    pub async fn get_raw_mempool(&self, verbose: bool) -> Result<Value> {
        self.rpc_call("getrawmempool", json!([verbose])).await
    }

    /// Get transaction output (UTXO)
    pub async fn get_tx_out(&self, txid: &str, vout: u32, unconfirmed: bool) -> Result<Value> {
        self.rpc_call("gettxout", json!([txid, vout, unconfirmed])).await
    }

    /// Estimate smart fee
    pub async fn estimate_smart_fee(&self, conf_target: u32) -> Result<Value> {
        self.rpc_call("estimatesmartfee", json!([conf_target])).await
    }

    /// Get wallet info (if wallet is loaded)
    pub async fn get_wallet_info(&self) -> Result<Value> {
        self.rpc_call("getwalletinfo", json!([])).await
    }

    /// Generate new address (if wallet available)
    pub async fn get_new_address(&self, label: Option<&str>) -> Result<String> {
        let params = if let Some(l) = label {
            json!([l])
        } else {
            json!([])
        };
        self.rpc_call("getnewaddress", params).await
    }

    /// Get balance (if wallet available)
    pub async fn get_balance(&self) -> Result<f64> {
        self.rpc_call("getbalance", json!([])).await
    }

    /// List unspent outputs (if wallet available)
    pub async fn list_unspent(
        &self, 
        min_conf: Option<u32>, 
        max_conf: Option<u32>
    ) -> Result<Value> {
        let params = match (min_conf, max_conf) {
            (Some(min), Some(max)) => json!([min, max]),
            (Some(min), None) => json!([min]),
            _ => json!([]),
        };
        self.rpc_call("listunspent", params).await
    }

    /// Add a peer node
    pub async fn add_node(&self, node: &str, command: &str) -> Result<()> {
        let _: Option<Value> = self.rpc_call("addnode", json!([node, command])).await?;
        Ok(())
    }

    /// Disconnect a peer
    pub async fn disconnect_node(&self, address: &str) -> Result<()> {
        let _: Option<Value> = self.rpc_call("disconnectnode", json!([address])).await?;
        Ok(())
    }

    /// Get added node info
    pub async fn get_added_node_info(&self, node: Option<&str>) -> Result<Value> {
        let params = if let Some(n) = node {
            json!([n])
        } else {
            json!([])
        };
        self.rpc_call("getaddednodeinfo", params).await
    }

    /// Test mempool acceptance of a transaction
    pub async fn test_mempool_accept(&self, raw_txs: Vec<&str>) -> Result<Value> {
        self.rpc_call("testmempoolaccept", json!([raw_txs])).await
    }

    /// Decode a raw transaction
    pub async fn decode_raw_transaction(&self, tx_hex: &str) -> Result<TxInfo> {
        self.rpc_call("decoderawtransaction", json!([tx_hex])).await
    }

    /// Get connection count
    pub async fn get_connection_count(&self) -> Result<u32> {
        self.rpc_call("getconnectioncount", json!([])).await
    }

    /// Check if Bitcoin Core is running
    pub async fn ping(&self) -> Result<()> {
        let _: Option<Value> = self.rpc_call("ping", json!([])).await?;
        Ok(())
    }

    /// Get node uptime
    pub async fn uptime(&self) -> Result<u64> {
        self.rpc_call("uptime", json!([])).await
    }
}

/// Bitcoin network scanner for peer discovery
pub struct BitcoinNetworkScanner {
    client: RealBitcoinClient,
    discovered_peers: HashMap<String, PeerInfo>,
}

impl BitcoinNetworkScanner {
    pub fn new(client: RealBitcoinClient) -> Self {
        Self {
            client,
            discovered_peers: HashMap::new(),
        }
    }

    /// Scan the Bitcoin network for Q-NarwhalKnight nodes
    pub async fn scan_for_q_nodes(&mut self) -> Result<Vec<String>> {
        info!("Scanning Bitcoin network for Q-NarwhalKnight nodes");

        // Get current peers
        let peers = self.client.get_peer_info().await?;
        info!("Found {} connected Bitcoin peers", peers.len());

        let mut q_nodes = Vec::new();

        for peer in peers {
            self.discovered_peers.insert(peer.addr.clone(), peer.clone());

            // Check if peer might be a Q-NarwhalKnight node
            // Look for specific version strings or behaviors
            if self.is_potential_q_node(&peer) {
                info!("Potential Q-NarwhalKnight node detected: {}", peer.addr);
                q_nodes.push(peer.addr);
            }
        }

        // Try to discover additional peers through network crawling
        self.crawl_network().await?;

        Ok(q_nodes)
    }

    /// Check if a peer might be running Q-NarwhalKnight
    fn is_potential_q_node(&self, peer: &PeerInfo) -> bool {
        // Check version string for Q-NarwhalKnight identifiers
        if peer.subver.contains("Q-Knight") || peer.subver.contains("Quantum") {
            return true;
        }

        // Check for specific port patterns
        if let Some(port) = peer.addr.split(':').nth(1) {
            if let Ok(port_num) = port.parse::<u16>() {
                // Q-NarwhalKnight uses specific port ranges
                if (8333..8340).contains(&port_num) || (18333..18340).contains(&port_num) {
                    return true;
                }
            }
        }

        // Check connection patterns that might indicate Q-NarwhalKnight
        if peer.services.starts_with("0000040") { // Custom service bits
            return true;
        }

        false
    }

    /// Crawl the network to discover more peers
    async fn crawl_network(&mut self) -> Result<()> {
        debug!("Crawling Bitcoin network for additional peers");

        // Get addresses from getaddednodeinfo
        if let Ok(node_info) = self.client.get_added_node_info(None).await {
            debug!("Additional node info: {:?}", node_info);
        }

        // Try to connect to known Bitcoin DNS seeds for the network
        let seeds = match self.client.config.network {
            Network::Bitcoin => vec![
                "seed.bitcoin.sipa.be:8333",
                "dnsseed.bluematt.me:8333",
                "dnsseed.bitcoin.dashjr.org:8333",
            ],
            Network::Testnet => vec![
                "testnet-seed.bitcoin.jonasschnelli.ch:18333",
                "seed.tbtc.petertodd.org:18333",
                "testnet-seed.bluematt.me:18333",
            ],
            Network::Regtest => vec![], // No public seeds for regtest
            _ => vec![],
        };

        for seed in seeds {
            debug!("Attempting to discover peers from seed: {}", seed);
            // In production, you'd implement DNS seeding here
            // For now, just try to add the node
            if let Err(e) = self.client.add_node(seed, "add").await {
                debug!("Failed to add seed node {}: {}", seed, e);
            }
        }

        Ok(())
    }

    /// Get discovered peer information
    pub fn get_discovered_peers(&self) -> &HashMap<String, PeerInfo> {
        &self.discovered_peers
    }

    /// Get network statistics
    pub async fn get_network_stats(&self) -> Result<(NetworkInfo, MempoolInfo)> {
        let network_info = self.client.get_network_info().await?;
        let mempool_info = self.client.get_mempool_info().await?;
        Ok((network_info, mempool_info))
    }
}

/// Create a production Bitcoin client with automatic network detection
pub async fn create_bitcoin_client(
    rpc_url: &str,
    username: &str, 
    password: &str,
    tor_proxy: Option<&str>
) -> Result<RealBitcoinClient> {
    
    let network = if rpc_url.contains(":18332") || rpc_url.contains("testnet") {
        Network::Testnet
    } else if rpc_url.contains(":18443") || rpc_url.contains("regtest") {
        Network::Regtest
    } else {
        Network::Bitcoin
    };

    let config = BitcoinConfig {
        network,
        rpc_url: rpc_url.to_string(),
        rpc_user: username.to_string(),
        rpc_password: password.to_string(),
        tor_proxy: tor_proxy.map(|s| s.to_string()),
        ..Default::default()
    };

    RealBitcoinClient::new(config).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires running Bitcoin node
    async fn test_real_bitcoin_connection() {
        let client = create_bitcoin_client(
            "http://127.0.0.1:18332",
            "bitcoin", 
            "password",
            None
        ).await.expect("Failed to create Bitcoin client");

        let info = client.get_network_info().await.expect("Failed to get network info");
        println!("Connected to: {}", info.subversion);
        
        let block_count = client.get_block_count().await.expect("Failed to get block count");
        println!("Current block height: {}", block_count);
    }

    #[tokio::test]
    #[ignore] // Requires running Bitcoin node with Tor
    async fn test_bitcoin_with_tor() {
        let client = create_bitcoin_client(
            "http://127.0.0.1:18332",
            "bitcoin",
            "password", 
            Some("socks5://127.0.0.1:9050")
        ).await.expect("Failed to create Bitcoin client with Tor");

        let info = client.get_network_info().await.expect("Failed to get network info");
        println!("Connected via Tor to: {}", info.subversion);
    }
}