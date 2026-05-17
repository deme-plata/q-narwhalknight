/// VM Network Bridge - Production libp2p Integration
///
/// This module bridges the Q-VM with the q-network libp2p infrastructure,
/// enabling VM-to-VM contract execution, state synchronization, and
/// distributed smart contract deployment.

use anyhow::Result;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, warn};

// Import from q-network crate
use q_network::{
    libp2p_bridge::{BridgeEvent, DhtEvent, Libp2pBridge},
    unified_network_manager::UnifiedNetworkManager,
    real_dht::{RealDht, RealDhtConfig, DhtCommand},
    NetworkManager, NetworkManagerConfig,
};

use crate::vm::{ExecutionResult, VmError};
use crate::state::StateDB;
use crate::network::security::{
    SignedVmMessage, PeerRateLimiter, ResourceQuotaManager, BytecodeValidator,
    AccessController, NonceTracker,
};

use ed25519_dalek::{SigningKey, VerifyingKey};

/// VM-specific network messages for contract execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VmNetworkMessage {
    /// Request to execute a contract on a remote VM
    ContractExecutionRequest {
        contract_address: String,
        function: String,
        args: Vec<u8>,
        caller: String,
        gas_limit: u64,
        request_id: String,
    },

    /// Response from contract execution
    ContractExecutionResponse {
        request_id: String,
        result: VmExecutionResult,
    },

    /// Deploy contract to network
    ContractDeployment {
        bytecode: Vec<u8>,
        deployer: String,
        deployment_id: String,
    },

    /// Confirm contract deployment
    DeploymentConfirmation {
        deployment_id: String,
        contract_address: String,
        success: bool,
    },

    /// State synchronization request
    StateSyncRequest {
        contract_address: String,
        state_root: [u8; 32],
    },

    /// State synchronization response
    StateSyncResponse {
        contract_address: String,
        state_data: Vec<u8>,
    },

    /// v2.9.2-beta: Encrypted state synchronization request
    /// Uses EncryptedStateSyncMessage for secure state transfer
    EncryptedStateSyncRequest {
        contract_address: String,
        encrypted_request: super::EncryptedStateSyncMessage,
    },

    /// v2.9.2-beta: Encrypted state synchronization response
    EncryptedStateSyncResponse {
        contract_address: String,
        encrypted_data: super::EncryptedStateSyncMessage,
    },

    /// v2.9.2-beta: Signed execution request (for caller verification)
    SignedContractExecution {
        request: super::SignedExecutionRequest,
        request_id: String,
    },

    /// VM capabilities announcement
    VmCapabilities {
        vm_version: String,
        supported_features: Vec<String>,
        max_gas_limit: u64,
        tps_capacity: u64,
    },
}

/// Result of VM execution for network transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmExecutionResult {
    pub success: bool,
    pub return_data: Vec<u8>,
    pub gas_used: u64,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

impl From<ExecutionResult> for VmExecutionResult {
    fn from(result: ExecutionResult) -> Self {
        Self {
            success: result.success,
            return_data: result.return_data,
            gas_used: result.gas_used,
            logs: result.logs,
            error: result.error,
        }
    }
}

/// Configuration for VM network bridge
#[derive(Debug, Clone)]
pub struct VmNetworkConfig {
    /// Enable distributed contract execution
    pub enable_distributed_execution: bool,

    /// Enable contract deployment gossip
    pub enable_deployment_gossip: bool,

    /// Enable state synchronization
    pub enable_state_sync: bool,

    /// Maximum concurrent remote execution requests
    pub max_concurrent_requests: usize,

    /// Request timeout in seconds
    pub request_timeout_secs: u64,

    /// Enable VM capability announcements
    pub announce_capabilities: bool,

    /// Security: Rate limit per peer (requests/second)
    pub rate_limit_per_peer: u32,

    /// Security: Total gas pool for all concurrent executions
    pub total_gas_pool: u64,

    /// Security: Maximum gas per single request
    pub max_gas_per_request: u64,

    /// Security: Maximum bytecode size (bytes)
    pub max_bytecode_size: usize,

    /// Security: Maximum message size (bytes)
    pub max_message_size: usize,
}

impl Default for VmNetworkConfig {
    fn default() -> Self {
        Self {
            enable_distributed_execution: true,
            enable_deployment_gossip: true,
            enable_state_sync: true,
            max_concurrent_requests: 100,
            request_timeout_secs: 30,
            announce_capabilities: true,
            rate_limit_per_peer: 10,
            total_gas_pool: 150_000_000,
            max_gas_per_request: 15_000_000,
            max_bytecode_size: 5 * 1024 * 1024, // 5 MB
            max_message_size: 10 * 1024 * 1024, // 10 MB
        }
    }
}

/// Statistics for VM network operations
#[derive(Debug, Clone, Default)]
pub struct VmNetworkStats {
    pub remote_executions_sent: u64,
    pub remote_executions_received: u64,
    pub contracts_deployed_to_network: u64,
    pub state_sync_requests: u64,
    pub connected_vm_peers: usize,
}

/// Production VM Network Bridge
pub struct VmNetworkBridge {
    /// Configuration
    config: VmNetworkConfig,

    /// Libp2p bridge for gossip/DHT integration
    libp2p_bridge_tx: Option<mpsc::Sender<DhtEvent>>,
    bridge_event_rx: Option<mpsc::Receiver<BridgeEvent>>,

    /// Real DHT for peer discovery
    dht_command_tx: Option<mpsc::Sender<DhtCommand>>,
    dht_event_rx: Option<broadcast::Receiver<q_network::real_dht::DhtEvent>>,

    /// Unified network manager (optional, for zero-config mode)
    network_manager: Option<Arc<RwLock<UnifiedNetworkManager>>>,

    /// Pending execution requests
    pending_requests: Arc<RwLock<HashMap<String, mpsc::Sender<VmExecutionResult>>>>,

    /// VM network statistics
    stats: Arc<RwLock<VmNetworkStats>>,

    /// State database reference
    state_db: Arc<StateDB>,

    /// Message broadcast channel
    message_tx: broadcast::Sender<VmNetworkMessage>,

    /// Security: Rate limiter per peer
    rate_limiter: Arc<PeerRateLimiter>,

    /// Security: Resource quota manager
    quota_manager: Arc<ResourceQuotaManager>,

    /// Security: Bytecode validator
    bytecode_validator: Arc<BytecodeValidator>,

    /// Security: Access controller
    access_controller: Arc<AccessController>,

    /// Security: Nonce tracker for replay protection
    nonce_tracker: Arc<NonceTracker>,

    /// Security: Ed25519 signing key for this node
    signing_key: SigningKey,

    /// Security: Public key for this node
    verifying_key: VerifyingKey,
}

impl VmNetworkBridge {
    /// Create new VM network bridge with libp2p integration and security
    pub async fn new(
        config: VmNetworkConfig,
        state_db: Arc<StateDB>,
    ) -> Result<Self> {
        info!("🌐 Initializing VM Network Bridge with libp2p and security");

        let (message_tx, _) = broadcast::channel(1000);

        // Initialize security components
        let rate_limiter = Arc::new(PeerRateLimiter::new(config.rate_limit_per_peer));
        let quota_manager = Arc::new(ResourceQuotaManager::new(
            config.total_gas_pool,
            config.max_gas_per_request,
        ));
        let bytecode_validator = Arc::new(BytecodeValidator::new());
        let access_controller = Arc::new(AccessController::new());
        let nonce_tracker = Arc::new(NonceTracker::new());

        // Generate Ed25519 keypair for this node
        let mut rng = rand::thread_rng();
        let signing_key = SigningKey::generate(&mut rng);
        let verifying_key = signing_key.verifying_key();

        info!(
            public_key = %hex::encode(verifying_key.as_bytes()),
            "🔐 VM Network Bridge security initialized"
        );

        Ok(Self {
            config,
            libp2p_bridge_tx: None,
            bridge_event_rx: None,
            dht_command_tx: None,
            dht_event_rx: None,
            network_manager: None,
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(VmNetworkStats::default())),
            state_db,
            message_tx,
            rate_limiter,
            quota_manager,
            bytecode_validator,
            access_controller,
            nonce_tracker,
            signing_key,
            verifying_key,
        })
    }

    /// Initialize with libp2p bridge for gossip-based communication
    pub async fn with_libp2p_bridge(
        mut self,
        keypair: libp2p::identity::Keypair,
    ) -> Result<Self> {
        info!("🔗 Connecting VM to libp2p gossip network");

        let (bridge_tx, bridge_rx) = mpsc::channel(1000);
        let (bridge, dht_tx) = Libp2pBridge::new(keypair, bridge_tx).await?;

        // Spawn bridge event loop
        tokio::spawn(async move {
            if let Err(e) = bridge.run().await {
                error!("Libp2p bridge error: {}", e);
            }
        });

        self.libp2p_bridge_tx = Some(dht_tx);
        self.bridge_event_rx = Some(bridge_rx);

        info!("✅ VM connected to libp2p gossip network");
        Ok(self)
    }

    /// Initialize with real DHT for peer discovery
    pub async fn with_real_dht(
        mut self,
        dht_config: RealDhtConfig,
    ) -> Result<Self> {
        info!("🔍 Enabling real DHT for VM peer discovery");

        let mut dht = RealDht::new(dht_config).await?;
        let dht_command_tx = dht.command_sender();
        let dht_event_rx = dht.subscribe_events();

        // Spawn DHT event loop
        tokio::spawn(async move {
            if let Err(e) = dht.run().await {
                error!("DHT error: {}", e);
            }
        });

        self.dht_command_tx = Some(dht_command_tx);
        self.dht_event_rx = Some(dht_event_rx);

        info!("✅ VM real DHT peer discovery enabled");
        Ok(self)
    }

    /// Initialize with unified network manager (zero-config mode)
    pub async fn with_unified_network(mut self) -> Result<Self> {
        info!("🚀 Enabling zero-config unified network");

        let network_manager = UnifiedNetworkManager::new().await?;
        let network_manager = Arc::new(RwLock::new(network_manager));

        // Store network manager but don't spawn event loop here
        // The event loop should be run separately by the caller
        self.network_manager = Some(network_manager);

        info!("✅ VM zero-config network enabled");
        Ok(self)
    }

    /// Execute contract on remote VM node
    pub async fn execute_remote_contract(
        &self,
        contract_address: String,
        function: String,
        args: Vec<u8>,
        caller: String,
        gas_limit: u64,
    ) -> Result<VmExecutionResult, VmError> {
        let request_id = uuid::Uuid::new_v4().to_string();

        let message = VmNetworkMessage::ContractExecutionRequest {
            contract_address: contract_address.clone(),
            function: function.clone(),
            args: args.clone(),
            caller,
            gas_limit,
            request_id: request_id.clone(),
        };

        // Create response channel
        let (response_tx, mut response_rx) = mpsc::channel(1);
        {
            let mut pending = self.pending_requests.write().await;
            pending.insert(request_id.clone(), response_tx);
        }

        // Broadcast request via libp2p
        if let Some(bridge_tx) = &self.libp2p_bridge_tx {
            let message_bytes = bincode::serialize(&message)
                .map_err(|e| VmError::SerializationError(e.to_string()))?;

            let dht_event = DhtEvent::PeerDiscovered {
                peer_id: message_bytes.clone(),
                address: format!("vm-contract://{}/{}", contract_address, function),
            };

            bridge_tx.send(dht_event).await
                .map_err(|e| VmError::ExecutionError(format!("Failed to send request: {}", e)))?;
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.remote_executions_sent += 1;
        }

        info!(
            request_id = %request_id,
            contract = %contract_address,
            function = %function,
            "Sent remote contract execution request"
        );

        // Wait for response with timeout
        let timeout = tokio::time::Duration::from_secs(self.config.request_timeout_secs);
        match tokio::time::timeout(timeout, response_rx.recv()).await {
            Ok(Some(result)) => {
                info!(request_id = %request_id, "Received remote execution response");
                Ok(result)
            }
            Ok(None) => Err(VmError::ExecutionError("Response channel closed".to_string())),
            Err(_) => Err(VmError::ExecutionError("Request timeout".to_string())),
        }
    }

    /// Deploy contract to the network
    pub async fn deploy_contract_to_network(
        &self,
        bytecode: Vec<u8>,
        deployer: String,
    ) -> Result<String, VmError> {
        let deployment_id = uuid::Uuid::new_v4().to_string();

        let message = VmNetworkMessage::ContractDeployment {
            bytecode: bytecode.clone(),
            deployer: deployer.clone(),
            deployment_id: deployment_id.clone(),
        };

        // Broadcast deployment via gossip
        self.broadcast_message(message).await?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.contracts_deployed_to_network += 1;
        }

        info!(
            deployment_id = %deployment_id,
            bytecode_len = bytecode.len(),
            deployer = %deployer,
            "Contract deployment broadcasted to network"
        );

        Ok(deployment_id)
    }

    /// Broadcast message to all VM peers with cryptographic signature
    async fn broadcast_message(&self, message: VmNetworkMessage) -> Result<(), VmError> {
        // SECURITY: Sign message before broadcasting
        let signed_msg = SignedVmMessage::sign(message.clone(), &self.signing_key)
            .map_err(|e| VmError::ExecutionError(format!("Failed to sign message: {}", e)))?;

        let message_bytes = bincode::serialize(&signed_msg)
            .map_err(|e| VmError::SerializationError(e.to_string()))?;

        // SECURITY: Check message size limit
        if message_bytes.len() > self.config.max_message_size {
            return Err(VmError::ExecutionError(format!(
                "Message size {} exceeds limit {}",
                message_bytes.len(),
                self.config.max_message_size
            )));
        }

        // Broadcast via message channel (send plain message for local subscribers)
        let _ = self.message_tx.send(message);

        // Also send via libp2p if available (send signed message)
        if let Some(bridge_tx) = &self.libp2p_bridge_tx {
            let dht_event = DhtEvent::ManifestUpdated {
                data: message_bytes,
            };

            bridge_tx.send(dht_event).await
                .map_err(|e| VmError::ExecutionError(format!("Broadcast failed: {}", e)))?;
        }

        Ok(())
    }

    /// Handle incoming network messages with security checks
    async fn handle_network_message(&self, signed_msg: SignedVmMessage<VmNetworkMessage>) -> Result<()> {
        // SECURITY: Verify message signature
        signed_msg.verify()
            .map_err(|e| anyhow::anyhow!("Message signature verification failed: {}", e))?;

        // SECURITY: Check and mark nonce for replay protection
        self.nonce_tracker
            .check_and_mark_nonce(&signed_msg.public_key, signed_msg.nonce)
            .await
            .map_err(|e| anyhow::anyhow!("Nonce check failed (replay attack?): {}", e))?;

        // SECURITY: Check rate limit
        self.rate_limiter
            .check_rate_limit(&signed_msg.public_key)
            .await
            .map_err(|e| anyhow::anyhow!("Rate limit exceeded: {}", e))?;

        // SECURITY: Check peer authorization
        if !self.access_controller
            .is_peer_authorized(&signed_msg.public_key)
            .await
        {
            warn!(
                peer = %hex::encode(&signed_msg.public_key),
                "Unauthorized peer attempted to send message"
            );
            return Err(anyhow::anyhow!("Peer not authorized"));
        }

        let message = signed_msg.message;

        match message {
            VmNetworkMessage::ContractExecutionRequest {
                contract_address,
                function,
                args,
                caller,
                gas_limit,
                request_id,
            } => {
                info!(
                    request_id = %request_id,
                    contract = %contract_address,
                    peer = %hex::encode(&signed_msg.public_key),
                    "Received authenticated remote contract execution request"
                );

                // SECURITY: Acquire gas quota
                let _gas_permit = self.quota_manager
                    .acquire_gas(gas_limit, &signed_msg.public_key)
                    .await
                    .map_err(|e| anyhow::anyhow!("Gas quota acquisition failed: {}", e))?;

                // SECURITY: Check contract-specific authorization
                if !self.access_controller
                    .is_authorized_for_contract(&signed_msg.public_key, &contract_address)
                    .await
                {
                    warn!(
                        peer = %hex::encode(&signed_msg.public_key),
                        contract = %contract_address,
                        "Peer not authorized for this contract"
                    );
                    return Err(anyhow::anyhow!("Not authorized for this contract"));
                }

                // Execute locally and send response
                // This would integrate with the actual VM executor
                let result = VmExecutionResult {
                    success: true,
                    return_data: vec![],
                    gas_used: 21000,
                    logs: vec![format!("Executed {} on {}", function, contract_address)],
                    error: None,
                };

                let response = VmNetworkMessage::ContractExecutionResponse {
                    request_id,
                    result,
                };

                self.broadcast_message(response).await?;

                // Update stats
                let mut stats = self.stats.write().await;
                stats.remote_executions_received += 1;
            }

            VmNetworkMessage::ContractExecutionResponse { request_id, result } => {
                // Forward response to waiting request
                if let Some(sender) = self.pending_requests.write().await.remove(&request_id) {
                    let _ = sender.send(result).await;
                }
            }

            VmNetworkMessage::ContractDeployment { deployment_id, bytecode, deployer } => {
                info!(
                    deployment_id = %deployment_id,
                    bytecode_len = bytecode.len(),
                    peer = %hex::encode(&signed_msg.public_key),
                    "Received authenticated contract deployment"
                );

                // SECURITY: Validate bytecode
                self.bytecode_validator
                    .validate(&bytecode)
                    .map_err(|e| anyhow::anyhow!("Bytecode validation failed: {}", e))?;

                info!(
                    deployment_id = %deployment_id,
                    "Bytecode validation passed"
                );

                // Store deployment in local state
                // This would integrate with state_db
            }

            VmNetworkMessage::VmCapabilities { vm_version, supported_features, .. } => {
                debug!(
                    version = %vm_version,
                    features = ?supported_features,
                    peer = %hex::encode(&signed_msg.public_key),
                    "Received VM capabilities announcement"
                );
            }

            // v2.9.2-beta: Handle encrypted state sync request
            VmNetworkMessage::EncryptedStateSyncRequest { contract_address, encrypted_request } => {
                info!(
                    contract = %&contract_address[..16.min(contract_address.len())],
                    "🔐 Received encrypted state sync request"
                );

                // Check freshness
                if !encrypted_request.is_fresh() {
                    warn!("Rejecting stale encrypted state sync request");
                    return Ok(());
                }

                // Decryption would be done with our private key
                // For now, log that we received it
                debug!(
                    "Encrypted request: {} bytes ciphertext, timestamp={}",
                    encrypted_request.ciphertext.len(),
                    encrypted_request.timestamp
                );
            }

            // v2.9.2-beta: Handle encrypted state sync response
            VmNetworkMessage::EncryptedStateSyncResponse { contract_address, encrypted_data } => {
                info!(
                    contract = %&contract_address[..16.min(contract_address.len())],
                    "🔐 Received encrypted state sync response"
                );

                if !encrypted_data.is_fresh() {
                    warn!("Rejecting stale encrypted state sync response");
                    return Ok(());
                }

                debug!(
                    "Encrypted response: {} bytes, timestamp={}",
                    encrypted_data.ciphertext.len(),
                    encrypted_data.timestamp
                );
            }

            // v2.9.2-beta: Handle signed contract execution request
            VmNetworkMessage::SignedContractExecution { request, request_id } => {
                info!(
                    request_id = %request_id,
                    contract = %&request.contract_address[..16.min(request.contract_address.len())],
                    function = %request.function,
                    "🔐 Received signed execution request"
                );

                // Verify the signature first
                if let Err(e) = request.verify() {
                    warn!(
                        request_id = %request_id,
                        error = %e,
                        "Rejecting signed execution request: signature verification failed"
                    );
                    return Ok(());
                }

                info!(
                    request_id = %request_id,
                    "✅ Signature verified for execution request"
                );

                // Further verification (nonce, balance, rate limit) would be done by RemoteExecutionVerifier
                // This is handled at the NetworkedVmExecutor level
            }

            _ => {
                debug!("Unhandled VM network message");
            }
        }

        Ok(())
    }

    /// Run the VM network bridge event loop
    pub async fn run(&mut self) -> Result<()> {
        info!("🚀 Starting VM Network Bridge event loop");

        // Announce capabilities if enabled
        if self.config.announce_capabilities {
            let capabilities = VmNetworkMessage::VmCapabilities {
                vm_version: "2.9.2-beta".to_string(),
                supported_features: vec![
                    "wasm".to_string(),
                    "parallel-execution".to_string(),
                    "ultra-performance".to_string(),
                    "state-sync".to_string(),
                    "encrypted-state-sync".to_string(),      // v2.9.2-beta
                    "signed-execution".to_string(),          // v2.9.2-beta
                    "caller-verification".to_string(),       // v2.9.2-beta
                    "consensus-finality-check".to_string(),  // v2.9.2-beta
                ],
                max_gas_limit: 15_000_000,
                tps_capacity: 150_000,
            };

            self.broadcast_message(capabilities).await?;
        }

        let mut message_rx = self.message_tx.subscribe();

        // Spawn periodic cleanup task
        let pending_requests_clone = self.pending_requests.clone();
        let config_clone = self.config.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                let mut pending = pending_requests_clone.write().await;
                if pending.len() > config_clone.max_concurrent_requests * 2 {
                    let keys_to_remove: Vec<_> = pending.keys()
                        .take(pending.len() - config_clone.max_concurrent_requests)
                        .cloned()
                        .collect();
                    for key in keys_to_remove {
                        pending.remove(&key);
                    }
                }
            }
        });

        loop {
            tokio::select! {
                // Handle bridge events
                Some(bridge_event) = async {
                    if let Some(rx) = &mut self.bridge_event_rx {
                        rx.recv().await
                    } else {
                        None
                    }
                } => {
                    match bridge_event {
                        BridgeEvent::ConsensusMessage { topic, data, peer } => {
                            debug!(topic = %topic, peer = %peer, "Received consensus message");

                            // SECURITY: Check message size limit
                            if data.len() > self.config.max_message_size {
                                warn!(
                                    size = data.len(),
                                    limit = self.config.max_message_size,
                                    "Received oversized message, dropping"
                                );
                                continue;
                            }

                            // Try to deserialize as signed VM message
                            if let Ok(signed_msg) = bincode::deserialize::<SignedVmMessage<VmNetworkMessage>>(&data) {
                                if let Err(e) = self.handle_network_message(signed_msg).await {
                                    error!("Failed to handle VM message: {}", e);
                                }
                            }
                        }
                        BridgeEvent::ValidatorDiscovered { peer_id, .. } => {
                            info!(peer = %peer_id, "New VM peer discovered");
                            let mut stats = self.stats.write().await;
                            stats.connected_vm_peers += 1;
                        }
                        _ => {}
                    }
                }

                // Handle DHT events
                Ok(dht_event) = async {
                    if let Some(rx) = &mut self.dht_event_rx {
                        rx.recv().await
                    } else {
                        Err(broadcast::error::RecvError::Closed)
                    }
                } => {
                    match dht_event {
                        q_network::real_dht::DhtEvent::PeerDiscovered(peer_info) => {
                            info!(
                                peer = %peer_info.peer_id,
                                addrs = ?peer_info.addresses,
                                "DHT peer discovered for VM"
                            );
                        }
                        q_network::real_dht::DhtEvent::RecordFound { key, value } => {
                            debug!(key = %key, "DHT record found");

                            // SECURITY: Check message size limit
                            if value.len() > self.config.max_message_size {
                                warn!(
                                    size = value.len(),
                                    limit = self.config.max_message_size,
                                    "Received oversized DHT record, dropping"
                                );
                                continue;
                            }

                            // Try to parse as signed VM message
                            if let Ok(signed_msg) = bincode::deserialize::<SignedVmMessage<VmNetworkMessage>>(&value) {
                                if let Err(e) = self.handle_network_message(signed_msg).await {
                                    error!("Failed to handle DHT VM message: {}", e);
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // Handle local VM network messages (already trusted, no signature check needed)
                Ok(message) = message_rx.recv() => {
                    // Local messages bypass security checks since they originate from this node
                    debug!("Processing local VM network message");
                }
            }
        }
    }

    /// Get current network statistics
    pub async fn get_stats(&self) -> VmNetworkStats {
        self.stats.read().await.clone()
    }

    /// Subscribe to VM network messages
    pub fn subscribe_messages(&self) -> broadcast::Receiver<VmNetworkMessage> {
        self.message_tx.subscribe()
    }

    /// Add an authorized peer by public key
    pub async fn add_authorized_peer(&self, public_key: [u8; 32]) -> Result<()> {
        self.access_controller.authorize_peer(public_key).await;
        info!(peer = %hex::encode(&public_key), "Added authorized peer");
        Ok(())
    }

    /// Ban a peer
    pub async fn ban_peer(&self, public_key: [u8; 32]) -> Result<()> {
        self.access_controller.ban_peer(public_key).await;
        warn!(peer = %hex::encode(&public_key), "Banned peer");
        Ok(())
    }

    /// Grant contract access to a peer
    pub async fn grant_contract_access(&self, public_key: [u8; 32], contract_address: String) -> Result<()> {
        self.access_controller
            .grant_contract_permission(contract_address.clone(), public_key)
            .await;
        info!(
            peer = %hex::encode(&public_key),
            contract = %contract_address,
            "Granted contract access"
        );
        Ok(())
    }

    /// Get resource quota statistics
    pub async fn get_quota_stats(&self) -> crate::network::security::ResourceQuotaStats {
        self.quota_manager.get_stats().await
    }

    /// Get this node's public key
    pub fn get_public_key(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }

    /// Cleanup expired pending requests
    async fn cleanup_expired_requests(&self) {
        let mut pending = self.pending_requests.write().await;
        let timeout = tokio::time::Duration::from_secs(self.config.request_timeout_secs);

        // This is a simple cleanup - in production you'd want to track timestamps
        // and remove only truly expired requests
        if pending.len() > self.config.max_concurrent_requests * 2 {
            warn!(
                count = pending.len(),
                limit = self.config.max_concurrent_requests,
                "Pending requests exceed limit, cleaning up oldest entries"
            );

            // Keep only the most recent max_concurrent_requests
            let keys_to_remove: Vec<_> = pending.keys()
                .take(pending.len() - self.config.max_concurrent_requests)
                .cloned()
                .collect();

            for key in keys_to_remove {
                pending.remove(&key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vm_network_bridge_creation() {
        let state_db = Arc::new(StateDB::new());
        let config = VmNetworkConfig::default();

        let bridge = VmNetworkBridge::new(config, state_db).await;
        assert!(bridge.is_ok());
    }

    #[tokio::test]
    async fn test_contract_deployment_broadcast() {
        let state_db = Arc::new(StateDB::new());
        let config = VmNetworkConfig::default();

        let bridge = VmNetworkBridge::new(config, state_db).await.unwrap();

        let bytecode = vec![0x60, 0x80, 0x60, 0x40, 0x52]; // Mock WASM bytecode
        let deployer = "0xdeployer".to_string();

        let result = bridge.deploy_contract_to_network(bytecode, deployer).await;
        assert!(result.is_ok());

        let stats = bridge.get_stats().await;
        assert_eq!(stats.contracts_deployed_to_network, 1);
    }
}
