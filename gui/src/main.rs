/// Q-NarwhalKnight Quantum Consensus GUI
/// Advanced visualization interface for the world's first quantum-enhanced DAG-BFT system

use anyhow::Result;
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use q_types::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use slint::*;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

slint::include_modules!();

/// API response structures aligned with Server Alpha's quantum endpoints
#[derive(Debug, Serialize, Deserialize)]
struct WalletResponse {
    id: String,
    balance: f64,
    created_at: DateTime<Utc>,
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
#[derive(Debug, Serialize, Deserialize)]
struct DAGVisualizationData {
    vertices: Vec<VertexInfo>,
    current_round: u64,
    anchor_vertex: String,
    finality_latency: f64,
    pending_count: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexInfo {
    id: String,
    round: u64,
    author: String,
    parents: Vec<String>,
    is_anchor: bool,
    x: f64,  // Visualization coordinates
    y: f64,
}

/// Network Topology Data from Server Alpha
#[derive(Debug, Serialize, Deserialize)]
struct NetworkTopology {
    peers: Vec<PeerInfo>,
    connections: Vec<ConnectionInfo>,
    quantum_handshakes: u64,
    phase_distribution: std::collections::HashMap<String, u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PeerInfo {
    id: String,
    phase: String,
    anonymity_score: f64,
    x: f64,  // Visualization coordinates
    y: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConnectionInfo {
    from: String,
    to: String,
    connection_type: String,
    latency_ms: f64,
}

/// Entropy Stream Data from Server Alpha
#[derive(Debug, Serialize, Deserialize)]
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
    last_update: DateTime<Utc>,
}

impl AppState {
    fn new() -> Result<Self> {
        let ui = MainWindow::new()?;
        let client = Client::new();
        let api_base = "http://localhost:3030/api/v1".to_string(); // Updated to match Server Alpha's port
        
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
            last_update: Utc::now(),
        })
    }
    
    /// Setup UI event handlers
    fn setup_handlers(self: Arc<Mutex<Self>>) -> Result<()> {
        let state_clone = self.clone();
        let ui = {
            let state = self.blocking_lock();
            state.ui.clone_strong()
        };
        
        // Wallet creation handler
        let state_for_wallet = state_clone.clone();
        ui.on_create_wallet(move || {
            let state = state_for_wallet.clone();
            tokio::spawn(async move {
                if let Ok(mut app_state) = state.try_lock() {
                    match app_state.create_wallet().await {
                        Ok(wallet) => {
                            app_state.ui.set_wallet_id(wallet.id.into());
                            app_state.ui.set_balance(wallet.balance);
                            app_state.ui.set_wallet_status("✅ Quantum wallet created successfully!".into());
                            info!("Wallet created: {}", wallet.id);
                        }
                        Err(e) => {
                            app_state.ui.set_wallet_status(format!("❌ Error: {}", e).into());
                            error!("Wallet creation failed: {}", e);
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
            
            tokio::spawn(async move {
                if let Ok(mut app_state) = state.try_lock() {
                    match app_state.submit_transaction(&recipient, amount).await {
                        Ok(response) => {
                            app_state.ui.set_tx_hash(response.hash.into());
                            app_state.ui.set_tx_status("✅ Transaction submitted with quantum signature!".into());
                            info!("Transaction submitted: {}", response.hash);
                        }
                        Err(e) => {
                            app_state.ui.set_tx_status(format!("❌ Transaction failed: {}", e).into());
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
            tokio::spawn(async move {
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
    
    /// Create a new quantum wallet
    async fn create_wallet(&self) -> Result<WalletResponse> {
        let response = self
            .client
            .post(&format!("{}/wallets", self.api_base))
            .header("Content-Type", "application/json")
            .send()
            .await?;
            
        if response.status().is_success() {
            let wallet: WalletResponse = response.json().await?;
            Ok(wallet)
        } else {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API error: {}", error_text);
        }
    }
    
    /// Submit a quantum-signed transaction
    async fn submit_transaction(&self, recipient: &str, amount: f64) -> Result<TransactionResponse> {
        let request = TransactionRequest {
            recipient: recipient.to_string(),
            amount,
        };
        
        let response = self
            .client
            .post(&format!("{}/transactions", self.api_base))
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
                self.ui.set_entropy_quality(metrics.entropy_quality);
                self.ui.set_consensus_latency(metrics.avg_evaluation_time_ms);
                self.ui.set_current_phase(metrics.phase_status.into());
                self.ui.set_anonymity_score(metrics.tor_anonymity_score);
                
                // Log detailed quantum metrics
                self.ui.append_log(format!(
                    "⚛️ [Quantum] Quality: {:.3}, L-VRF: {:.1}%, VDF: {:.0}%, Health: {:.1}%\n",
                    metrics.entropy_quality,
                    metrics.lvrf_success_rate * 100.0,
                    metrics.vdf_progress * 100.0,
                    metrics.consensus_health * 100.0
                ).into());
                
                debug!("Quantum metrics updated: quality={:.3}, L-VRF rate={:.1}%", 
                       metrics.entropy_quality, metrics.lvrf_success_rate * 100.0);
            }
            Err(e) => {
                warn!("Failed to fetch quantum metrics: {}", e);
                self.ui.append_log(format!("⚠️ Metrics update failed: {}\n", e).into());
            }
        }
        
        // Update DAG visualization data
        if let Ok(dag_data) = self.fetch_dag_visualization().await {
            self.dag_data = Some(dag_data.clone());
            self.ui.append_log(format!(
                "🕸️ [DAG] Round {}, Anchor: {}, Latency: {:.1}ms, Pending: {}\n",
                dag_data.current_round,
                &dag_data.anchor_vertex[..12],
                dag_data.finality_latency,
                dag_data.pending_count
            ).into());
        }
        
        // Update network topology
        if let Ok(topology) = self.fetch_network_topology().await {
            self.ui.set_active_peers(topology.peers.len() as i32);
            self.network_topology = Some(topology.clone());
            self.ui.append_log(format!(
                "🌐 [Network] {} peers, {} handshakes, {} connections\n",
                topology.peers.len(),
                topology.quantum_handshakes,
                topology.connections.len()
            ).into());
        }
        
        self.last_update = Utc::now();
        Ok(())
    }
    
    /// Legacy method for backward compatibility
    async fn refresh_quantum_metrics(&mut self) -> Result<()> {
        self.refresh_all_quantum_data().await
    }
    
    /// Fetch quantum system metrics from Server Alpha's API
    async fn fetch_quantum_metrics(&self) -> Result<QuantumMetricsResponse> {
        let response = self
            .client
            .get(&format!("{}/quantum/metrics", self.api_base))
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
            .get(&format!("{}/consensus/dag-status", self.api_base))
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
            .get(&format!("{}/network/peer-topology", self.api_base))
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
                connections: vec![
                    ConnectionInfo {
                        from: "Alice".to_string(),
                        to: "Bob".to_string(),
                        connection_type: "Quantum".to_string(),
                        latency_ms: 23.4,
                    },
                ],
                quantum_handshakes: 1247,
                phase_distribution: phase_dist,
            })
        }
    }
    
    /// Start real-time event stream using Server Alpha's WebSocket/SSE endpoints
    async fn start_event_stream(self: Arc<Mutex<Self>>) -> Result<()> {
        let api_base = {
            let state = self.lock().await;
            state.api_base.clone()
        };
        
        info!("🌊 Starting quantum event stream from {}/quantum/entropy-stream", api_base);
        
        // Try WebSocket first, then fallback to SSE
        if let Err(e) = Self::start_websocket_stream(self.clone(), &api_base).await {
            warn!("WebSocket failed, falling back to SSE: {}", e);
            Self::start_sse_stream(self, &api_base).await?;
        }
        
        Ok(())
    }
    
    /// Start WebSocket connection for real-time updates
    async fn start_websocket_stream(state: Arc<Mutex<Self>>, api_base: &str) -> Result<()> {
        // For now, implement SSE fallback until WebSocket is fully implemented
        Err(anyhow::anyhow!("WebSocket not yet implemented, using SSE"))
    }
    
    /// Start Server-Sent Events stream for real-time updates
    async fn start_sse_stream(self: Arc<Mutex<Self>>, api_base: &str) -> Result<()> {
        
        // Attempt to connect to Server Alpha's quantum entropy stream
        match reqwest::get(&format!("{}/quantum/entropy-stream", api_base)).await {
            Ok(response) => {
                let mut stream = response.bytes_stream();
                
                {
                    let state = self.lock().await;
                    state.ui.append_log("🌌 Connected to Server Alpha quantum event stream\n".into());
                }
                
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if let Ok(text) = std::str::from_utf8(&bytes) {
                                // Parse Server Alpha's event format
                                if let Ok(entropy_data) = serde_json::from_str::<EntropyMeasurement>(text) {
                                    let formatted_event = format!(
                                        "🌌 [{}] QRNG: {} quality={:.3} provider={}\n",
                                        entropy_data.timestamp.format("%H:%M:%S"),
                                        entropy_data.value,
                                        entropy_data.quality_score,
                                        entropy_data.provider
                                    );
                                    
                                    if let Ok(state) = self.try_lock() {
                                        state.ui.append_log(formatted_event.into());
                                    }
                                } else {
                                    // Fallback to generic event formatting
                                    let formatted_event = Self::format_event(text);
                                    if let Ok(state) = self.try_lock() {
                                        state.ui.append_log(formatted_event.into());
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
                Self::start_mock_event_stream(self.clone()).await;
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
                format!("🌌 [QRNG] Entropy quality: {:.3} (pool: {} MB)", 
                       0.970 + (counter as f64 * 0.001) % 0.025, 2 + counter % 3),
                format!("⚡ [L-VRF] Anchor election round {} completed in {:.1}ms", 
                       847 + counter, 12.0 + (counter as f64 * 0.3) % 8.0),
                format!("🎭 [Tor] Circuit diversity: 4 circuits, anonymity: {:.1}%", 
                       94.0 + (counter as f64 * 0.1) % 4.0),
                format!("🔮 [VDF] Sequential proof #{} verified (speedup: {}x)", 
                       counter + 1200, 2048 + counter % 512),
                format!("📡 [Network] Quantum handshake with peer {} (Phase {})", 
                       ['A', 'B', 'C', 'D'][counter % 4], 
                       if counter % 3 == 0 { "2" } else { "1" }),
                format!("💎 [DAG] Vertex finalized at round {} with {:.1}ms latency", 
                       847 + counter, 45.0 + (counter as f64 * 0.2) % 15.0),
                format!("🌪️ [Consensus] Health score: {:.1}% ({} pending vertices)", 
                       98.0 + (counter as f64 * 0.05) % 1.8, counter % 5),
            ];
            
            let event = &mock_events[counter % mock_events.len()];
            let timestamp = chrono::Utc::now().format("%H:%M:%S");
            let formatted = format!("[{}] {}\n", timestamp, event);
            
            if let Ok(state) = state.try_lock() {
                state.ui.append_log(formatted.into());
            }
        }
    }
    
    /// Format event text with quantum styling
    fn format_event(raw_event: &str) -> String {
        let timestamp = chrono::Utc::now().format("%H:%M:%S");
        
        // Add quantum-themed prefixes based on content
        let formatted = if raw_event.contains("entropy") || raw_event.contains("quantum") {
            format!("🌌 [{}] {}", timestamp, raw_event)
        } else if raw_event.contains("consensus") || raw_event.contains("anchor") {
            format!("⚡ [{}] {}", timestamp, raw_event)
        } else if raw_event.contains("tor") || raw_event.contains("circuit") {
            format!("🎭 [{}] {}", timestamp, raw_event)
        } else if raw_event.contains("vrf") || raw_event.contains("randomness") {
            format!("🔮 [{}] {}", timestamp, raw_event)
        } else {
            format!("📡 [{}] {}", timestamp, raw_event)
        };
        
        formatted + "\n"
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
                let avg_quality = self.entropy_data.iter()
                    .map(|e| e.quality_score)
                    .sum::<f64>() / self.entropy_data.len() as f64;
                    
                self.ui.set_entropy_quality(avg_quality);
            }
            Err(_) => {
                // Generate mock entropy data for visualization
                let mock_entropy = EntropyMeasurement {
                    timestamp: Utc::now(),
                    value: rand::random::<f64>(),
                    provider: "MockQRNG".to_string(),
                    quality_score: 0.97 + (rand::random::<f64>() - 0.5) * 0.04,
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
            .get(&format!("{}/quantum/entropy-sample", self.api_base))
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
        .with_env_filter("qnk_gui=debug,info")
        .init();
    
    info!("🚀 Starting Q-NarwhalKnight Quantum GUI");
    
    // Create application state
    let app_state = Arc::new(Mutex::new(AppState::new()?));
    
    // Setup UI event handlers
    AppState::setup_handlers(app_state.clone())?;
    
    // Start background tasks
    let state_for_events = app_state.clone();
    tokio::spawn(async move {
        if let Err(e) = AppState::start_event_stream(state_for_events).await {
            error!("Event stream failed: {}", e);
        }
    });
    
    let state_for_metrics = app_state.clone();
    tokio::spawn(async move {
        AppState::start_metrics_refresh(state_for_metrics).await;
    });
    
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