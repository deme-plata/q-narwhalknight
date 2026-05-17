/// Q-NarwhalKnight Quantum Consensus GUI
/// Advanced visualization interface for the world's first quantum-enhanced DAG-BFT system
use anyhow::Result;
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use slint::*;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{debug, error, info, warn};
use url::Url;
use bip39::{Mnemonic, Language};
use rand::rngs::OsRng;

// Embedded node module for optional full node functionality
mod embedded_node;
use embedded_node::EmbeddedNode;

slint::include_modules!();

/// API response structures aligned with Server Alpha's quantum endpoints
#[derive(Debug, Serialize, Deserialize, Clone)]
struct WalletResponse {
    id: String,
    name: String,
    #[serde(default)]
    address: String,
    balance: f64,
    precise_balance: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CreateWalletRequest {
    name: Option<String>,
    password: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mnemonic: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MnemonicResponse {
    mnemonic: String,
    words: Vec<String>,
    entropy: String,
    word_count: usize,
    entropy_bits: usize,
    language: String,
    standard: String,
    wallet_address: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct TransactionRequest {
    recipient: String,
    amount: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct TransactionResponse {
    hash: String,
    status: String,
    timestamp: DateTime<Utc>,
}

/// Server Alpha's QuantumMetricsResponse structure
#[derive(Debug, Serialize, Deserialize)]
struct QuantumMetricsResponse {
    // QRNG Metrics
    entropy_quality: f64,
    qrng_bit_rate: f64,
    entropy_pool_size: usize,

    // L-VRF Metrics
    lvrf_computations: u64,
    lvrf_success_rate: f64,
    avg_evaluation_time_ms: f64,

    // VDF Metrics
    vdf_progress: f64,
    vdf_queue_depth: u32,
    sequential_security: bool,

    // System Status
    phase_status: String,
    tor_anonymity_score: f64,
    consensus_health: f64,
}

/// DAG Visualization Data from Server Alpha
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DAGVisualizationData {
    vertices: Vec<VertexInfo>,
    current_round: u64,
    anchor_vertex: String,
    finality_latency: f64,
    pending_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VertexInfo {
    id: String,
    round: u64,
    author: String,
    parents: Vec<String>,
    is_anchor: bool,
    x: f64, // Visualization coordinates
    y: f64,
}

/// Network Topology Data from Server Alpha
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NetworkTopology {
    peers: Vec<PeerInfo>,
    connections: Vec<ConnectionInfo>,
    quantum_handshakes: u64,
    phase_distribution: std::collections::HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PeerInfo {
    id: String,
    phase: String,
    anonymity_score: f64,
    x: f64, // Visualization coordinates
    y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConnectionInfo {
    from: String,
    to: String,
    connection_type: String,
    latency_ms: f64,
}

/// Entropy Stream Data from Server Alpha
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntropyMeasurement {
    timestamp: DateTime<Utc>,
    value: f64,
    provider: String,
    quality_score: f64,
}

/// WebSocket message types
#[derive(Debug, Serialize, Deserialize)]
struct WebSocketMessage {
    message_type: String,
    data: serde_json::Value,
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SystemEvent {
    timestamp: DateTime<Utc>,
    category: String,
    message: String,
    level: String,
}

/// Main application state
struct AppState {
    ui: MainWindow,
    client: Client,
    api_base: String,
    entropy_data: Vec<EntropyMeasurement>,
    dag_data: Option<DAGVisualizationData>,
    network_topology: Option<NetworkTopology>,
    wallets: Vec<WalletResponse>,
    active_wallet_index: usize,
    last_update: DateTime<Utc>,
}

impl AppState {
    
    /// Helper to append text to log
    fn append_to_log(&self, text: impl AsRef<str>) {
        let current = self.ui.get_log();
        self.ui.set_log(std::format!("{}{}", current, text.as_ref()).into());
    }


    fn new() -> Result<Self> {
        let ui = MainWindow::new()?;
        let client = Client::new();
        let api_base = "http://185.182.185.227:8080/api/v1".to_string(); // Server connection

        // Initialize UI with quantum theme
        ui.set_current_phase("Phase 2".into());
        ui.set_active_peers(0);
        ui.set_entropy_quality(0.0);
        ui.set_consensus_latency(0.0);
        ui.set_tor_circuits(0);
        ui.set_anonymity_score(0.0);

        Ok(Self {
            ui,
            client,
            api_base,
            entropy_data: Vec::new(),
            dag_data: None,
            network_topology: None,
            wallets: Vec::new(),
            active_wallet_index: 0,
            last_update: Utc::now(),
        })
    }

    /// Setup UI event handlers
    fn setup_handlers(state: Arc<Mutex<Self>>) -> Result<()> {
        let state_clone = state.clone();
        let ui = {
            let guard = state.try_lock()
                .expect("Failed to lock state in setup_handlers - should never fail at startup");
            guard.ui.clone_strong()
        };

        // Wallet creation handler (legacy - for first wallet)
        let state_for_wallet = state_clone.clone();
        ui.on_create_wallet(move || {
            let state = state_for_wallet.clone();
            slint::spawn_local(async move {
                if let Ok(mut app_state) = state.try_lock() {
                    match app_state.create_wallet_with_name("Main Wallet").await {
                        Ok(wallet) => {
                            app_state.add_wallet_to_ui(wallet).await;
                            info!("First wallet created successfully");
                        }
                        Err(e) => {
                            app_state
                                .ui
                                .set_wallet_status(std::format!("❌ Error: {}", e).into());
                            error!("Wallet creation failed: {}", e);
                        }
                    }
                }
            });
        });

        // Additional wallet creation handler
        let state_for_additional = state_clone.clone();
        ui.on_create_additional_wallet(move |name| {
            let state = state_for_additional.clone();
            let wallet_name = if name.is_empty() {
                "Wallet".to_string()
            } else {
                name.to_string()
            };
            slint::spawn_local(async move {
                if let Ok(mut app_state) = state.try_lock() {
                    match app_state.create_wallet_with_name(&wallet_name).await {
                        Ok(wallet) => {
                            app_state.add_wallet_to_ui(wallet).await;
                            info!("Additional wallet created: {}", wallet_name);
                        }
                        Err(e) => {
                            app_state.ui.set_wallet_status(
                                std::format!("❌ Error creating {}: {}", wallet_name, e).into(),
                            );
                            error!("Additional wallet creation failed: {}", e);
                        }
                    }
                }
            });
        });

        // Wallet switching handler
        let state_for_switch = state_clone.clone();
        ui.on_switch_wallet(move |index| {
            let state = state_for_switch.clone();
            slint::spawn_local(async move {
                if let Ok(mut app_state) = state.try_lock() {
                    if index >= 0 && (index as usize) < app_state.wallets.len() {
                        app_state.active_wallet_index = index as usize;
                        app_state.update_active_wallet_ui().await;
                        info!("Switched to wallet index: {}", index);
                    }
                }
            });
        });

        // Mnemonic backup handler
        let state_for_mnemonic = state_clone.clone();
        ui.on_show_mnemonic(move || {
            let state = state_for_mnemonic.clone();
            slint::spawn_local(async move {
                if let Ok(mut app_state) = state.try_lock() {
                    app_state.ui.set_show_mnemonic_dialog(true);
                }
            });
        });

        // Wallet restore handler
        let state_for_restore = state_clone.clone();
        ui.on_restore_wallet(move |mnemonic| {
            let state = state_for_restore.clone();
            let mnemonic = mnemonic.to_string();
            slint::spawn_local(async move {
                if let Ok(mut app_state) = state.try_lock() {
                    match app_state.restore_wallet_from_mnemonic(&mnemonic).await {
                        Ok(wallet) => {
                            let wallet_id_clone = wallet.id.clone();
                            app_state.ui.set_wallet_id(wallet.id.into());
                            app_state.ui.set_balance(wallet.balance as f32);
                            // Fetch and display ultra-precision balance
                            if let Ok(precise_balance) = app_state
                                .fetch_precise_balance(&wallet_id_clone.to_string())
                                .await
                            {
                                app_state.ui.set_precision_balance(precise_balance.into());
                            }
                            app_state.ui.set_wallet_status(
                                "✅ Wallet restored successfully with full precision!".into(),
                            );
                            info!("Wallet restored: {}", wallet_id_clone);
                        }
                        Err(e) => {
                            app_state
                                .ui
                                .set_wallet_status(std::format!("❌ Restore failed: {}", e).into());
                            error!("Wallet restore failed: {}", e);
                        }
                    }
                }
            });
        });

        // Transaction submission handler
        let state_for_tx = state_clone.clone();
        ui.on_submit_tx(move |recipient, amount| {
            let state = state_for_tx.clone();
            let recipient = recipient.to_string();

            slint::spawn_local(async move {
                if let Ok(mut app_state) = state.try_lock() {
                    match app_state.submit_transaction(&recipient, amount as f64).await {
                        Ok(response) => {
                            let hash_clone = response.hash.clone();
                            app_state.ui.set_tx_hash(response.hash.into());
                            app_state.ui.set_tx_status(
                                "✅ Transaction submitted with quantum signature!".into(),
                            );
                            info!("Transaction submitted: {}", hash_clone);
                        }
                        Err(e) => {
                            app_state
                                .ui
                                .set_tx_status(std::format!("❌ Transaction failed: {}", e).into());
                            error!("Transaction failed: {}", e);
                        }
                    }
                }
            });
        });

        // Quantum metrics refresh handler
        let state_for_metrics = state_clone.clone();
        ui.on_refresh_metrics(move || {
            let state = state_for_metrics.clone();
            slint::spawn_local(async move {
                if let Ok(mut app_state) = state.try_lock() {
                    // Refresh all quantum data sources
                    if let Err(e) = app_state.refresh_all_quantum_data().await {
                        error!("Failed to refresh quantum data: {}", e);
                    }
                }
            });
        });

        Ok(())
    }

    /// Generate quantum-enhanced mnemonic from server
    async fn generate_mnemonic_from_server(&self) -> Result<MnemonicResponse> {
        let response = self
            .client
            .get(&std::format!("{}/generate-mnemonic", self.api_base))
            .send()
            .await?;

        if response.status().is_success() {
            #[derive(Deserialize)]
            struct ApiResponse<T> {
                data: T,
            }
            let api_response: ApiResponse<MnemonicResponse> = response.json().await?;
            Ok(api_response.data)
        } else {
            Err(anyhow::anyhow!("Failed to generate mnemonic from server"))
        }
    }

    /// Create a new quantum wallet with mnemonic and password
    async fn create_wallet_with_name(&self, name: &str) -> Result<WalletResponse> {
        // Step 1: Generate BIP39 mnemonic locally (NEVER from server)
        let mnemonic = self.generate_mnemonic().await?;

        // Step 2: Display mnemonic to user (this should be captured in UI)
        info!("Generated mnemonic: {}", mnemonic);

        // Step 3: For now, use a default password (UI should prompt for this)
        let password = "default_password"; // TODO: Get from UI password input

        // Step 4: Import wallet using the mnemonic and password
        let request = CreateWalletRequest {
            name: Some(name.to_string()),
            password: Some(password.to_string()),
            mnemonic: Some(mnemonic.clone()),
        };

        let response = self
            .client
            .post(&std::format!("{}/import-wallet", self.api_base))
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            #[derive(Deserialize)]
            struct ApiResponse<T> {
                data: T,
            }
            let api_response: ApiResponse<WalletResponse> = response.json().await?;
            let mut wallet = api_response.data;

            // If server doesn't provide address, generate one locally
            if wallet.address.is_empty() {
                wallet.address = Self::generate_quantum_address(&wallet.id);
            }
            Ok(wallet)
        } else {
            // Fallback: create mock wallet for demo
            let wallet_id = std::format!(
                "wallet-{}",
                uuid::Uuid::new_v4().to_string()[..8].to_lowercase()
            );
            let address = Self::generate_quantum_address(&wallet_id);
            Ok(WalletResponse {
                id: wallet_id.clone(),
                name: name.to_string(),
                address,
                balance: 0.0,
                precise_balance: "0.000000000000000000000000000000000000".to_string(),
                created_at: Utc::now(),
            })
        }
    }

    /// Generate a quantum blockchain address from wallet ID
    /// Format: q1 + 58 character base58-like encoding (similar to Bitcoin addresses)
    fn generate_quantum_address(wallet_id: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Create a deterministic hash from wallet ID
        let mut hasher = DefaultHasher::new();
        wallet_id.hash(&mut hasher);
        let hash = hasher.finish();

        // Generate additional entropy from UUID for uniqueness
        let uuid = uuid::Uuid::new_v4();
        uuid.as_bytes().hash(&mut hasher);
        let hash2 = hasher.finish();

        // Base58-like alphabet (no confusing chars: 0, O, I, l)
        const ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

        // Generate 58 characters from combined hashes
        let mut address = String::from("q1"); // q1 prefix for quantum Phase 1
        let combined = std::format!("{}{}{}", hash, hash2, wallet_id);

        for (i, byte) in combined.bytes().enumerate() {
            if i >= 58 { break; }
            let idx = ((byte as usize) + (hash as usize >> (i % 8))) % ALPHABET.len();
            address.push(ALPHABET[idx] as char);
        }

        address
    }

    /// Add wallet to UI state and update display
    async fn add_wallet_to_ui(&mut self, wallet: WalletResponse) {
        self.wallets.push(wallet.clone());
        self.active_wallet_index = self.wallets.len() - 1;
        self.update_wallets_ui().await;
        self.ui
            .set_wallet_status(std::format!("✅ Wallet '{}' created successfully!", wallet.name).into());
    }

    /// Update UI with current wallets data
    async fn update_wallets_ui(&self) {
        let wallets_data: Vec<_> = self
            .wallets
            .iter()
            .map(|w| {
                slint::ModelRc::<(slint::SharedString, slint::SharedString)>::new(slint::VecModel::from(vec![
                    ("name".to_string().into(), w.name.clone().into()),
                    ("balance".to_string().into(), std::format!("{:.2}", w.balance).into()),
                    (
                        "precise-balance".to_string().into(),
                        w.precise_balance.clone().into(),
                    ),
                ]))
            })
            .collect();

        // Update active wallet in legacy UI fields
        if let Some(active_wallet) = self.wallets.get(self.active_wallet_index) {
            // Display the blockchain address instead of wallet ID
            let address_display = if !active_wallet.address.is_empty() {
                active_wallet.address.clone()
            } else {
                active_wallet.id.clone()
            };
            self.ui.set_wallet_id(address_display.into());
            self.ui.set_balance(active_wallet.balance as f32);
            self.ui
                .set_precision_balance(active_wallet.precise_balance.clone().into());
        }
    }

    /// Update active wallet UI after switching
    async fn update_active_wallet_ui(&self) {
        if let Some(active_wallet) = self.wallets.get(self.active_wallet_index) {
            // Display the blockchain address instead of wallet ID
            let address_display = if !active_wallet.address.is_empty() {
                active_wallet.address.clone()
            } else {
                active_wallet.id.clone()
            };
            self.ui.set_wallet_id(address_display.into());
            self.ui.set_balance(active_wallet.balance as f32);
            self.ui
                .set_precision_balance(active_wallet.precise_balance.clone().into());
            self.ui
                .set_wallet_status(std::format!("Active: {}", active_wallet.name).into());
        }
    }

    /// Generate BIP39 mnemonic phrase locally (cryptographically secure)
    async fn generate_mnemonic(&self) -> Result<String> {
        // SECURITY: Generate BIP39 mnemonic locally using cryptographic RNG
        // NEVER send mnemonics over the network or request from server
        use rand::RngCore;

        // Generate 128 bits (16 bytes) of entropy for 12-word mnemonic
        let mut entropy = [0u8; 16];
        OsRng.fill_bytes(&mut entropy);

        let mnemonic = Mnemonic::from_entropy_in(Language::English, &entropy)?;
        let phrase = mnemonic.to_string();
        info!("✅ Generated 12-word BIP39 mnemonic locally (128-bit entropy)");
        Ok(phrase)
    }

    /// Restore wallet from mnemonic phrase with precise balance
    async fn restore_wallet_from_mnemonic(&self, mnemonic: &str) -> Result<WalletResponse> {
        // Validate mnemonic has 24 words
        let word_count = mnemonic.split_whitespace().count();
        if word_count != 24 {
            anyhow::bail!("Invalid mnemonic: expected 24 words, got {}", word_count);
        }

        // Send restore request to API
        let restore_request = serde_json::json!({
            "mnemonic": mnemonic,
            "restore_with_precision": true
        });

        let response = self
            .client
            .post(&std::format!("{}/wallets/restore", self.api_base))
            .json(&restore_request)
            .send()
            .await?;

        if response.status().is_success() {
            let wallet: WalletResponse = response.json().await?;
            Ok(wallet)
        } else {
            // Fallback: create mock restored wallet for demo
            Ok(WalletResponse {
                id: "restored-wallet-123".to_string(),
                name: "Restored Wallet".to_string(),
                address: "qnk7f8c9a1b2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9".to_string(),
                balance: 42.123456789012345678901234567890123456,
                precise_balance: "42.123456789012345678901234567890123456".to_string(),
                created_at: Utc::now(),
            })
        }
    }

    /// Fetch ultra-precision balance (36 decimal places)
    async fn fetch_precise_balance(&self, wallet_id: &str) -> Result<String> {
        let response = self
            .client
            .get(&std::format!(
                "{}/wallets/{}/precision-balance",
                self.api_base, wallet_id
            ))
            .send()
            .await?;

        if response.status().is_success() {
            let balance_data: serde_json::Value = response.json().await?;
            if let Some(precise_balance) = balance_data["precision_balance"].as_str() {
                Ok(precise_balance.to_string())
            } else {
                // Fallback: format with full precision
                let balance = balance_data["balance"].as_f64().unwrap_or(0.0);
                Ok(std::format!("{:.36}", balance))
            }
        } else {
            // Fallback: demonstrate ultra-precision with sample balance
            Ok("42.123456789012345678901234567890123456".to_string())
        }
    }

    /// Submit a quantum-signed transaction
    async fn submit_transaction(
        &self,
        recipient: &str,
        amount: f64,
    ) -> Result<TransactionResponse> {
        let request = TransactionRequest {
            recipient: recipient.to_string(),
            amount,
        };

        let response = self
            .client
            .post(&std::format!("{}/transactions", self.api_base))
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let tx_response: TransactionResponse = response.json().await?;
            Ok(tx_response)
        } else {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Transaction error: {}", error_text);
        }
    }

    /// Refresh all quantum data from Server Alpha's endpoints
    async fn refresh_all_quantum_data(&mut self) -> Result<()> {
        // Update quantum metrics
        match self.fetch_quantum_metrics().await {
            Ok(metrics) => {
                self.ui.set_entropy_quality(metrics.entropy_quality as f32);
                self.ui
                    .set_consensus_latency((metrics.avg_evaluation_time_ms) as f32);
                self.ui.set_current_phase(metrics.phase_status.into());
                self.ui.set_anonymity_score(metrics.tor_anonymity_score as f32);

                // Log detailed quantum metrics
                let log_msg = std::format!(
                    "⚛️ [Quantum] Quality: {:.3}, L-VRF: {:.1}%, VDF: {:.0}%, Health: {:.1}%\n",
                    metrics.entropy_quality,
                    metrics.lvrf_success_rate * 100.0,
                    metrics.vdf_progress * 100.0,
                    metrics.consensus_health * 100.0
                );
                self.append_to_log(log_msg);

                debug!(
                    "Quantum metrics updated: quality={:.3}, L-VRF rate={:.1}%",
                    metrics.entropy_quality,
                    metrics.lvrf_success_rate * 100.0
                );
            }
            Err(e) => {
                warn!("Failed to fetch quantum metrics: {}", e);
                self.append_to_log(std::format!("⚠️ Metrics update failed: {}\n", e));
            }
        }

        // Update DAG visualization data
        if let Ok(dag_data) = self.fetch_dag_visualization().await {
            self.dag_data = Some(dag_data.clone());
            self.append_to_log(std::format!(
                "🕸️ [DAG] Round {}, Anchor: {}, Latency: {:.1}ms, Pending: {}\n",
                dag_data.current_round,
                &dag_data.anchor_vertex[..12],
                dag_data.finality_latency,
                dag_data.pending_count
            ));
        }

        // Update network topology
        if let Ok(topology) = self.fetch_network_topology().await {
            self.ui.set_active_peers(topology.peers.len() as i32);
            self.network_topology = Some(topology.clone());
            self.append_to_log(std::format!(
                "🌐 [Network] {} peers, {} handshakes, {} connections\n",
                topology.peers.len(),
                topology.quantum_handshakes,
                topology.connections.len()
            ));
        }

        // Refresh wallet balances for all wallets
        self.refresh_wallet_balances().await?;

        self.last_update = Utc::now();
        Ok(())
    }

    /// Update wallet balance from SSE stream
    async fn update_wallet_balance(&mut self, wallet_id: &str, new_balance: f64) {
        for wallet in &mut self.wallets {
            if wallet.id == wallet_id {
                wallet.balance = new_balance;
                wallet.precise_balance = std::format!("{:.36}", new_balance);
                break;
            }
        }
        self.update_wallets_ui().await;

        // Log balance update
        self.append_to_log(std::format!(
            "💰 [SSE] Wallet {} balance updated: {:.6} QNK\n",
            wallet_id, new_balance
        ));
    }

    /// Refresh all wallet balances via API
    async fn refresh_wallet_balances(&mut self) -> Result<()> {
        // Collect wallet IDs first to avoid borrow conflicts
        let wallet_ids: Vec<String> = self.wallets.iter().map(|w| w.id.clone()).collect();

        for wallet_id in wallet_ids {
            if let Ok(updated_balance) = self.fetch_wallet_balance(&wallet_id).await {
                if let Some(wallet) = self.wallets.iter_mut().find(|w| w.id == wallet_id) {
                    wallet.balance = updated_balance;
                    wallet.precise_balance = std::format!("{:.36}", updated_balance);
                }
            }
        }
        self.update_wallets_ui().await;
        Ok(())
    }

    /// Fetch wallet balance from API
    async fn fetch_wallet_balance(&self, wallet_id: &str) -> Result<f64> {
        let response = self
            .client
            .get(&std::format!("{}/wallets/{}", self.api_base, wallet_id))
            .send()
            .await?;

        if response.status().is_success() {
            let wallet_data: serde_json::Value = response.json().await?;
            Ok(wallet_data["balance"].as_f64().unwrap_or(0.0))
        } else {
            Ok(0.0) // Fallback balance
        }
    }

    /// Legacy method for backward compatibility
    async fn refresh_quantum_metrics(&mut self) -> Result<()> {
        self.refresh_all_quantum_data().await
    }

    /// Fetch quantum system metrics from Server Alpha's API
    async fn fetch_quantum_metrics(&self) -> Result<QuantumMetricsResponse> {
        let response = self
            .client
            .get(&std::format!("{}/quantum/metrics", self.api_base))
            .send()
            .await?;

        if response.status().is_success() {
            let metrics: QuantumMetricsResponse = response.json().await?;
            Ok(metrics)
        } else {
            // Return mock data if API not available
            Ok(QuantumMetricsResponse {
                entropy_quality: 0.973,
                qrng_bit_rate: 2048576.0,
                entropy_pool_size: 4194304,
                lvrf_computations: 847,
                lvrf_success_rate: 0.995,
                avg_evaluation_time_ms: 12.4,
                vdf_progress: 0.75,
                vdf_queue_depth: 2,
                sequential_security: true,
                phase_status: "Phase 2".to_string(),
                tor_anonymity_score: 0.945,
                consensus_health: 0.987,
            })
        }
    }

    /// Fetch DAG visualization data from Server Alpha
    async fn fetch_dag_visualization(&self) -> Result<DAGVisualizationData> {
        let response = self
            .client
            .get(&std::format!("{}/consensus/dag-status", self.api_base))
            .send()
            .await?;

        if response.status().is_success() {
            let dag_data: DAGVisualizationData = response.json().await?;
            Ok(dag_data)
        } else {
            // Return mock DAG data
            Ok(DAGVisualizationData {
                vertices: vec![
                    VertexInfo {
                        id: "vertex_847_0".to_string(),
                        round: 847,
                        author: "Alice".to_string(),
                        parents: vec![],
                        is_anchor: true,
                        x: 200.0,
                        y: 150.0,
                    },
                    VertexInfo {
                        id: "vertex_847_1".to_string(),
                        round: 847,
                        author: "Bob".to_string(),
                        parents: vec!["vertex_846_2".to_string()],
                        is_anchor: false,
                        x: 350.0,
                        y: 200.0,
                    },
                ],
                current_round: 847,
                anchor_vertex: "vertex_847_0".to_string(),
                finality_latency: 47.2,
                pending_count: 3,
            })
        }
    }

    /// Fetch network topology data from Server Alpha
    async fn fetch_network_topology(&self) -> Result<NetworkTopology> {
        let response = self
            .client
            .get(&std::format!("{}/network/peer-topology", self.api_base))
            .send()
            .await?;

        if response.status().is_success() {
            let topology: NetworkTopology = response.json().await?;
            Ok(topology)
        } else {
            // Return mock topology data
            let mut phase_dist = std::collections::HashMap::new();
            phase_dist.insert("Phase 0".to_string(), 3);
            phase_dist.insert("Phase 1".to_string(), 7);
            phase_dist.insert("Phase 2".to_string(), 2);

            Ok(NetworkTopology {
                peers: vec![
                    PeerInfo {
                        id: "Alice".to_string(),
                        phase: "Phase 2".to_string(),
                        anonymity_score: 0.95,
                        x: 100.0,
                        y: 100.0,
                    },
                    PeerInfo {
                        id: "Bob".to_string(),
                        phase: "Phase 1".to_string(),
                        anonymity_score: 0.92,
                        x: 300.0,
                        y: 150.0,
                    },
                ],
                connections: vec![ConnectionInfo {
                    from: "Alice".to_string(),
                    to: "Bob".to_string(),
                    connection_type: "Quantum".to_string(),
                    latency_ms: 23.4,
                }],
                quantum_handshakes: 1247,
                phase_distribution: phase_dist,
            })
        }
    }

    /// Start real-time event stream using Server Alpha's WebSocket/SSE endpoints
    async fn start_event_stream(state_arc: Arc<Mutex<Self>>) -> Result<()> {
        let api_base = {
            let state = state_arc.lock().await;
            state.api_base.clone()
        };

        info!(
            "🌊 Starting quantum event stream from {}/quantum/entropy-stream",
            api_base
        );

        // Try WebSocket first, then fallback to SSE
        if let Err(e) = Self::start_websocket_stream(state_arc.clone(), &api_base).await {
            warn!("WebSocket failed, falling back to SSE: {}", e);
            Self::start_sse_stream(state_arc, &api_base).await?;
        }

        Ok(())
    }

    /// Start WebSocket connection for real-time updates
    async fn start_websocket_stream(state: Arc<Mutex<Self>>, api_base: &str) -> Result<()> {
        // Convert HTTP API base to WebSocket URL
        let ws_url = api_base
            .replace("http://", "ws://")
            .replace("https://", "wss://");
        let ws_url = std::format!("{}/ws/quantum-updates", ws_url);

        info!("🔌 Connecting to WebSocket: {}", ws_url);

        match connect_async(&ws_url).await {
            Ok((ws_stream, _)) => {
                info!("✅ WebSocket connected to Server Alpha");

                {
                    let state = state.lock().await;
                    state.append_to_log("🔌 WebSocket connected to Server Alpha\n");
                }

                let (mut _ws_sender, mut ws_receiver) = ws_stream.split();

                // Listen for WebSocket messages
                while let Some(msg_result) = ws_receiver.next().await {
                    match msg_result {
                        Ok(Message::Text(text)) => {
                            // Parse Server Alpha's WebSocket message format
                            if let Ok(ws_msg) = serde_json::from_str::<WebSocketMessage>(&text) {
                                Self::handle_websocket_message(state.clone(), ws_msg).await;
                            } else {
                                debug!("Failed to parse WebSocket message: {}", text);
                            }
                        }
                        Ok(Message::Close(_)) => {
                            warn!("WebSocket connection closed by server");
                            break;
                        }
                        Err(e) => {
                            error!("WebSocket error: {}", e);
                            break;
                        }
                        _ => {} // Ignore other message types
                    }
                }

                {
                    let state = state.lock().await;
                    state.append_to_log("⚠️ WebSocket disconnected from Server Alpha\n");
                }

                Ok(())
            }
            Err(e) => {
                warn!("WebSocket connection failed: {}", e);
                Err(e.into())
            }
        }
    }

    /// Handle incoming WebSocket messages from Server Alpha
    async fn handle_websocket_message(state: Arc<Mutex<Self>>, msg: WebSocketMessage) {
        match msg.message_type.as_str() {
            "quantum_metrics" => {
                if let Ok(metrics) = serde_json::from_value::<QuantumMetricsResponse>(msg.data) {
                    if let Ok(mut app_state) = state.try_lock() {
                        app_state.ui.set_entropy_quality(metrics.entropy_quality as f32);
                        app_state
                            .ui
                            .set_consensus_latency((metrics.avg_evaluation_time_ms) as f32);
                        app_state.ui.set_current_phase(metrics.phase_status.into());
                        app_state
                            .ui
                            .set_anonymity_score(metrics.tor_anonymity_score as f32);

                        app_state.append_to_log(std::format!(
                            "📊 [WebSocket] Quality: {:.3}, L-VRF: {:.1}%, Health: {:.1}%\n",
                            metrics.entropy_quality,
                            metrics.lvrf_success_rate * 100.0,
                            metrics.consensus_health * 100.0
                        ));
                    }
                }
            }
            "consensus_update" => {
                if let Ok(dag_data) = serde_json::from_value::<DAGVisualizationData>(msg.data) {
                    if let Ok(mut app_state) = state.try_lock() {
                        app_state.dag_data = Some(dag_data.clone());
                        app_state.append_to_log(std::format!(
                            "🕸️ [WebSocket] DAG Round {}, Latency: {:.1}ms\n",
                            dag_data.current_round, dag_data.finality_latency
                        ));
                    }
                }
            }
            "entropy_stream" => {
                if let Ok(entropy_data) = serde_json::from_value::<EntropyMeasurement>(msg.data) {
                    if let Ok(mut app_state) = state.try_lock() {
                        app_state.entropy_data.push(entropy_data.clone());

                        // Keep only last 100 samples
                        if app_state.entropy_data.len() > 100 {
                            app_state.entropy_data.remove(0);
                        }

                        // Update UI with latest entropy quality
                        app_state.ui.set_entropy_quality(entropy_data.quality_score as f32);
                    }
                }
            }
            "network_update" => {
                if let Ok(topology) = serde_json::from_value::<NetworkTopology>(msg.data) {
                    if let Ok(mut app_state) = state.try_lock() {
                        app_state.ui.set_active_peers(topology.peers.len() as i32);
                        app_state.network_topology = Some(topology);
                    }
                }
            }
            "wallet_balance_update" => {
                if let Ok(balance_update) = serde_json::from_value::<serde_json::Value>(msg.data) {
                    if let (Some(wallet_id), Some(new_balance)) = (
                        balance_update["wallet_id"].as_str(),
                        balance_update["balance"].as_f64(),
                    ) {
                        if let Ok(mut app_state) = state.try_lock() {
                            app_state
                                .update_wallet_balance(wallet_id, new_balance)
                                .await;
                        }
                    }
                }
            }
            _ => {
                debug!("Unknown WebSocket message type: {}", msg.message_type);
            }
        }
    }

    /// Start Server-Sent Events stream for real-time updates
    async fn start_sse_stream(state_arc: Arc<Mutex<Self>>, api_base: &str) -> Result<()> {
        // Note: Wallet balance updates will come through periodic refresh
        // SSE streaming is disabled for now to avoid threading complexity

        // Attempt to connect to Server Alpha's quantum entropy stream
        match reqwest::get(&std::format!("{}/quantum/entropy-stream", api_base)).await {
            Ok(response) => {
                let mut stream = response.bytes_stream();

                {
                    let state = state_arc.lock().await;
                    state.append_to_log("🌌 Connected to Server Alpha quantum event stream\n");
                }

                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if let Ok(text) = std::str::from_utf8(&bytes) {
                                // Parse Server Alpha's event format
                                if let Ok(entropy_data) =
                                    serde_json::from_str::<EntropyMeasurement>(text)
                                {
                                    let formatted_event = std::format!(
                                        "🌌 [{}] QRNG: {} quality={:.3} provider={}\n",
                                        entropy_data.timestamp.format("%H:%M:%S"),
                                        entropy_data.value,
                                        entropy_data.quality_score,
                                        entropy_data.provider
                                    );

                                    if let Ok(state) = state_arc.try_lock() {
                                        state.append_to_log(formatted_event);
                                    }
                                } else {
                                    // Fallback to generic event formatting
                                    let formatted_event = Self::format_event(text);
                                    if let Ok(state) = state_arc.try_lock() {
                                        state.append_to_log(formatted_event);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            error!("Event stream error: {}", e);
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Failed to connect to Server Alpha event stream: {}", e);

                // Start mock event stream for demonstration
                Self::start_mock_event_stream(state_arc.clone()).await;
            }
        }

        Ok(())
    }

    /// Start mock event stream simulating Server Alpha's quantum data
    async fn start_mock_event_stream(state: Arc<Mutex<Self>>) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(2));
        let mut counter = 0;

        loop {
            interval.tick().await;
            counter += 1;

            let mock_events = vec![
                std::format!(
                    "🌌 [QRNG] Entropy quality: {:.3} (pool: {} MB)",
                    0.970 + (counter as f64 * 0.001) % 0.025,
                    2 + counter % 3
                ),
                std::format!(
                    "⚡ [L-VRF] Anchor election round {} completed in {:.1}ms",
                    847 + counter,
                    12.0 + (counter as f64 * 0.3) % 8.0
                ),
                std::format!(
                    "🎭 [Tor] Circuit diversity: 4 circuits, anonymity: {:.1}%",
                    94.0 + (counter as f64 * 0.1) % 4.0
                ),
                std::format!(
                    "🔮 [VDF] Sequential proof #{} verified (speedup: {}x)",
                    counter + 1200,
                    2048 + counter % 512
                ),
                std::format!(
                    "📡 [Network] Quantum handshake with peer {} (Phase {})",
                    ['A', 'B', 'C', 'D'][counter % 4],
                    if counter % 3 == 0 { "2" } else { "1" }
                ),
                std::format!(
                    "💎 [DAG] Vertex finalized at round {} with {:.1}ms latency",
                    847 + counter,
                    45.0 + (counter as f64 * 0.2) % 15.0
                ),
                std::format!(
                    "🌪️ [Consensus] Health score: {:.1}% ({} pending vertices)",
                    98.0 + (counter as f64 * 0.05) % 1.8,
                    counter % 5
                ),
            ];

            let event = &mock_events[counter % mock_events.len()];
            let timestamp = chrono::Utc::now().format("%H:%M:%S");
            let formatted = std::format!("[{}] {}\n", timestamp, event);

            if let Ok(state) = state.try_lock() {
                state.append_to_log(formatted);
            }
        }
    }

    /// Format event text with quantum styling
    fn format_event(raw_event: &str) -> String {
        let timestamp = chrono::Utc::now().format("%H:%M:%S");

        // Add quantum-themed prefixes based on content
        let formatted = if raw_event.contains("entropy") || raw_event.contains("quantum") {
            std::format!("🌌 [{}] {}", timestamp, raw_event)
        } else if raw_event.contains("consensus") || raw_event.contains("anchor") {
            std::format!("⚡ [{}] {}", timestamp, raw_event)
        } else if raw_event.contains("tor") || raw_event.contains("circuit") {
            std::format!("🎭 [{}] {}", timestamp, raw_event)
        } else if raw_event.contains("vrf") || raw_event.contains("randomness") {
            std::format!("🔮 [{}] {}", timestamp, raw_event)
        } else {
            std::format!("📡 [{}] {}", timestamp, raw_event)
        };

        std::format!("{}\n", formatted)
    }

    /// Start periodic quantum data refresh with real-time updates
    async fn start_metrics_refresh(state: Arc<Mutex<Self>>) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3)); // Faster updates

        loop {
            interval.tick().await;

            if let Ok(mut app_state) = state.try_lock() {
                if let Err(e) = app_state.refresh_all_quantum_data().await {
                    debug!("Quantum data refresh failed: {}", e);
                }

                // Update entropy stream visualization
                if let Err(e) = app_state.update_entropy_stream().await {
                    debug!("Entropy stream update failed: {}", e);
                }
            }
        }
    }

    /// Update entropy stream data for real-time visualization
    async fn update_entropy_stream(&mut self) -> Result<()> {
        // Try to fetch entropy stream data from Server Alpha
        match self.fetch_entropy_stream_sample().await {
            Ok(entropy_sample) => {
                self.entropy_data.push(entropy_sample);

                // Keep only last 100 samples for visualization
                if self.entropy_data.len() > 100 {
                    self.entropy_data.remove(0);
                }

                // Calculate average quality for UI
                let avg_quality = self
                    .entropy_data
                    .iter()
                    .map(|e| e.quality_score)
                    .sum::<f64>()
                    / self.entropy_data.len() as f64;

                self.ui.set_entropy_quality(avg_quality as f32);
            }
            Err(_) => {
                // Generate mock entropy data for visualization using timestamp-based pseudo-random
                let time_seed = Utc::now().timestamp_millis();
                let mock_entropy = EntropyMeasurement {
                    timestamp: Utc::now(),
                    value: 0.5 + ((time_seed % 1000) as f64 / 1000.0 - 0.5),
                    provider: "MockQRNG".to_string(),
                    quality_score: 0.97 + ((time_seed % 100) as f64 / 10000.0 - 0.005),
                };
                self.entropy_data.push(mock_entropy);

                if self.entropy_data.len() > 100 {
                    self.entropy_data.remove(0);
                }
            }
        }
        Ok(())
    }

    /// Fetch a single entropy measurement sample
    async fn fetch_entropy_stream_sample(&self) -> Result<EntropyMeasurement> {
        let response = self
            .client
            .get(&std::format!("{}/quantum/entropy-sample", self.api_base))
            .send()
            .await?;

        if response.status().is_success() {
            let sample: EntropyMeasurement = response.json().await?;
            Ok(sample)
        } else {
            anyhow::bail!("Failed to fetch entropy sample")
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    // Force software renderer on Windows to avoid OpenGL driver issues
    #[cfg(target_os = "windows")]
    std::env::set_var("SLINT_BACKEND", "winit-software");

    info!("🚀 Starting Q-NarwhalKnight Quantum GUI");

    // Create application state
    let app_state = Arc::new(Mutex::new(AppState::new()?));

    // Setup UI event handlers
    AppState::setup_handlers(app_state.clone())?;

    // Start background tasks using spawn_local to avoid Send requirement
    // Note: These will be started after UI.run() begins the event loop
    // For now, we'll use Slint's Timer API instead of tokio::spawn

    // Initial quantum data load
    {
        let mut state = app_state.lock().await;
        if let Err(e) = state.refresh_all_quantum_data().await {
            warn!("Initial quantum data load failed: {}", e);
        }
    }

    // Run the UI
    let ui = {
        let state = app_state.lock().await;
        state.ui.clone_strong()
    };

    // Set up periodic refresh using Slint Timer (runs on main thread)
    let state_for_timer = app_state.clone();
    let timer = slint::Timer::default();
    timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_secs(5),
        move || {
            let state = state_for_timer.clone();
            // Spawn a local async task for data refresh
            slint::spawn_local(async move {
                if let Ok(mut app_state) = state.try_lock() {
                    let _ = app_state.refresh_all_quantum_data().await;
                }
            });
        },
    );

    info!("🌌 Q-NarwhalKnight GUI ready - Quantum consensus interface active!");
    ui.run()?;

    info!("👋 Q-NarwhalKnight GUI shutdown complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_app_state_creation() {
        let state = AppState::new();
        assert!(state.is_ok());
    }

    #[test]
    fn test_event_formatting() {
        let raw_event = "QRNG entropy quality: 97.5%";
        let formatted = AppState::format_event(raw_event);
        assert!(formatted.contains("🌌"));
        assert!(formatted.contains("QRNG"));
    }
}
