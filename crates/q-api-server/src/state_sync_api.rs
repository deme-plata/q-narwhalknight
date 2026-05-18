/// P2P State Sync Protocol for Q-NarwhalKnight
///
/// v5.3.0: Gossipsub-based state synchronization (primary) with HTTP fallback.
///
/// Solves the "missed gossipsub" problem: if a node is offline when contract/pool/balance
/// state is broadcast via P2P, that state is permanently lost. This module provides:
///
/// 1. **P2P primary**: Gossipsub request/response on `/state-sync-requests` and `/state-sync-responses`
/// 2. **HTTP fallback**: `GET /api/v1/sync/full-state` (kept for backward compat and when P2P unavailable)
/// 3. **Background task**: On startup, tries P2P sync first, falls back to HTTP, then periodic every 5 min
///
/// Merge strategy is conservative: never overwrite existing local state, only add missing entries.

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use crate::AppState;

/// Pending P2P sync requests awaiting responses
use std::sync::Mutex as StdMutex;

struct PendingSync {
    response: Option<q_types::state_sync::StateSnapshotResponse>,
    notify: Arc<tokio::sync::Notify>,
}

static PENDING_SYNCS: once_cell::sync::Lazy<StdMutex<HashMap<u64, Arc<StdMutex<PendingSync>>>>> =
    once_cell::sync::Lazy::new(|| StdMutex::new(HashMap::new()));

// ============================================================================
// Types (kept for HTTP endpoint backward compat)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullStateSnapshot {
    pub contracts: HashMap<String, serde_json::Value>,
    pub liquidity_pools: HashMap<String, serde_json::Value>,
    pub wallet_balances: HashMap<String, String>,
    pub token_balances: HashMap<String, String>,
    pub symbol_to_address: HashMap<String, String>,
    pub block_height: u64,
    pub version: String,
    pub timestamp: u64,
    /// v7.3.0: Network ID for cross-network contamination prevention
    #[serde(default)]
    pub network_id: Option<String>,
    /// BFT-finalized balance records (Bracha RB + DAG-Knight).
    /// Includes both anchored and pending-anchor records that have reached 2f+1 READY.
    /// Fresh nodes apply these AFTER block-derived balances to get the authoritative value.
    #[serde(default)]
    pub finality_records: Vec<q_types::BalanceFinalityRecord>,
}

// ============================================================================
// HTTP Endpoint: GET /api/v1/sync/full-state (kept as fallback)
// ============================================================================

pub async fn get_full_state(
    State(app_state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let start = std::time::Instant::now();

    // Gather contracts
    let mut contracts = HashMap::new();
    {
        let deployed = app_state.orobit_ecosystem.deployed_contracts.read().await;
        for (addr, contract) in deployed.iter() {
            let addr_hex = hex::encode(addr.0);
            if let Ok(val) = serde_json::to_value(contract) {
                contracts.insert(addr_hex, val);
            }
        }
    }

    // Gather liquidity pools
    let mut liquidity_pools = HashMap::new();
    {
        let pools = app_state.liquidity_pools.read().await;
        for (pool_id, pool) in pools.iter() {
            if let Ok(val) = serde_json::to_value(pool) {
                liquidity_pools.insert(pool_id.clone(), val);
            }
        }
    }

    // Gather wallet balances
    let mut wallet_balances = HashMap::new();
    {
        let balances = app_state.wallet_balances.read().await;
        for (addr, amount) in balances.iter() {
            let addr_hex = hex::encode(addr);
            wallet_balances.insert(addr_hex, amount.to_string());
        }
    }

    // Gather token balances
    let mut token_balances = HashMap::new();
    {
        let balances = app_state.token_balances.read().await;
        for ((wallet, token), amount) in balances.iter() {
            let key = format!("{}_{}", hex::encode(wallet), hex::encode(token));
            token_balances.insert(key, amount.to_string());
        }
    }

    // Gather symbol_to_address
    let mut symbol_to_address = HashMap::new();
    for entry in app_state.symbol_to_address.iter() {
        symbol_to_address.insert(entry.key().clone(), entry.value().clone());
    }

    let block_height = app_state
        .current_height_atomic
        .load(std::sync::atomic::Ordering::SeqCst);

    let our_network_id = std::env::var("Q_NETWORK_ID")
        .unwrap_or_else(|_| "mainnet-genesis".to_string());

    // Gather BFT-finalized balance records for fresh-node seeding.
    // Includes both anchored records (in DB) and pending-anchor records (delivered but not yet in vertex).
    let finality_records = if let Some(ref engine) = app_state.balance_finality_engine {
        let mut recs = engine.load_anchored_records().await.unwrap_or_default();
        recs.extend(engine.pending_anchor_snapshot().await);
        recs
    } else {
        Vec::new()
    };

    let snapshot = FullStateSnapshot {
        contracts,
        liquidity_pools,
        wallet_balances,
        token_balances,
        symbol_to_address,
        block_height,
        version: crate::VERSION.to_string(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        network_id: Some(our_network_id),
        finality_records,
    };

    let elapsed = start.elapsed();
    debug!(
        "🔄 [STATE SYNC] Served full-state snapshot: {} contracts, {} pools, {} wallets, {} tokens in {:?}",
        snapshot.contracts.len(),
        snapshot.liquidity_pools.len(),
        snapshot.wallet_balances.len(),
        snapshot.token_balances.len(),
        elapsed,
    );

    Json(serde_json::json!({
        "success": true,
        "data": snapshot,
    }))
}

// ============================================================================
// Background sync task: P2P primary, HTTP fallback
// ============================================================================

/// Spawn the periodic state sync background task.
/// Runs initial sync after 10s, then every 5 minutes.
pub fn spawn_state_sync_task(app_state: Arc<AppState>, our_port: u16) {
    tokio::spawn(async move {
        // Wait for server to be fully ready (P2P needs time to connect)
        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

        info!("🔄 [STATE SYNC] Starting initial state sync (P2P primary, HTTP fallback)...");
        do_combined_state_sync(&app_state, our_port).await;

        // 🔧 v9.0.0: ONE-TIME AUTHORITATIVE BALANCE SYNC
        // When Q_BALANCE_AUTHORITY_PEER is set (e.g., "http://89.149.241.126:8080"),
        // fetch wallet balances from that peer and OVERWRITE local RocksDB values.
        // This fixes balance divergence after chain replay (which misses DEX protocol fees).
        // The env var is consumed and the flag is set so it only runs ONCE.
        if let Ok(authority_peer) = std::env::var("Q_BALANCE_AUTHORITY_PEER") {
            info!("🔑 [AUTHORITY SYNC] Q_BALANCE_AUTHORITY_PEER set to {}", authority_peer);
            info!("   Fetching authoritative wallet balances from trusted peer...");
            match do_authoritative_balance_sync(&app_state, &authority_peer).await {
                Ok(count) => {
                    info!("✅ [AUTHORITY SYNC] Imported {} wallet balances from {}", count, authority_peer);
                }
                Err(e) => {
                    error!("❌ [AUTHORITY SYNC] Failed to sync from {}: {}", authority_peer, e);
                }
            }
        }

        // v10.3.2: Mark startup sync as complete — balance API can now return real values
        // Before this point, balance API returns null/syncing to prevent ghost 4200 QUG display
        app_state.startup_sync_complete.store(true, std::sync::atomic::Ordering::Release);
        info!("✅ [STARTUP SYNC v10.3.2] Sync complete — balance API now serving authoritative values");

        // Periodic sync every 5 minutes
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300));
        interval.tick().await; // consume the first immediate tick
        loop {
            interval.tick().await;
            debug!("🔄 [STATE SYNC] Periodic state sync triggered");
            do_combined_state_sync(&app_state, our_port).await;
        }
    });
}

/// Try P2P sync first, fall back to HTTP if P2P unavailable or times out
async fn do_combined_state_sync(app_state: &Arc<AppState>, our_port: u16) {
    // Try P2P first
    if app_state.libp2p_command_tx.is_some() {
        info!("🔄 [STATE SYNC] Attempting P2P gossipsub state sync...");
        match do_p2p_state_sync(app_state).await {
            Ok(true) => {
                info!("✅ [STATE SYNC] P2P state sync completed successfully");
                return;
            }
            Ok(false) => {
                info!("🔄 [STATE SYNC] P2P sync returned no new data, trying HTTP fallback...");
            }
            Err(e) => {
                warn!("🔄 [STATE SYNC] P2P sync failed: {}, falling back to HTTP...", e);
            }
        }
    } else {
        debug!("🔄 [STATE SYNC] No P2P available, using HTTP fallback directly");
    }

    // HTTP fallback
    do_http_state_sync(app_state, our_port).await;
}

// ============================================================================
// P2P State Sync (primary method)
// ============================================================================

/// Send a signed state sync request via gossipsub and wait for response
async fn do_p2p_state_sync(app_state: &Arc<AppState>) -> anyhow::Result<bool> {
    use q_types::state_sync::StateSnapshotRequest;

    let network_tx = app_state.libp2p_command_tx.as_ref()
        .ok_or_else(|| anyhow::anyhow!("No P2P command channel"))?;

    // Get our public key
    let our_pubkey: [u8; 32] = app_state.node_signing_key.verifying_key().to_bytes();

    // Gather current counts
    let known_contracts = {
        let contracts = app_state.orobit_ecosystem.deployed_contracts.read().await;
        contracts.len() as u32
    };
    let known_pools = {
        let pools = app_state.liquidity_pools.read().await;
        pools.len() as u32
    };
    let current_height = app_state
        .current_height_atomic
        .load(std::sync::atomic::Ordering::SeqCst);

    // Create and sign request
    let mut request = StateSnapshotRequest::new(
        our_pubkey,
        current_height,
        known_contracts,
        known_pools,
    );
    request.sign(&app_state.node_signing_key)?;

    let request_id = request.request_id;

    // Register pending sync
    let pending = Arc::new(StdMutex::new(PendingSync {
        response: None,
        notify: Arc::new(tokio::sync::Notify::new()),
    }));
    let notify = {
        let guard = pending.lock().unwrap();
        guard.notify.clone()
    };
    {
        let mut syncs = PENDING_SYNCS.lock().unwrap();
        syncs.insert(request_id, pending.clone());
    }

    // Serialize and publish
    let request_bytes = serde_json::to_vec(&request)?;

    let network_id = std::env::var("Q_NETWORK_ID")
        .ok()
        .and_then(|s| s.parse::<q_types::NetworkId>().ok())
        .unwrap_or(q_types::NetworkId::MainnetGenesis);
    let topic = network_id.state_sync_requests_topic();

    info!(
        "🔄 [STATE SYNC P2P] Broadcasting request id={} (we have {} contracts, {} pools, height {})",
        request_id, known_contracts, known_pools, current_height,
    );

    if let Err(e) = network_tx.send(q_network::NetworkCommand::PublishStateSyncRequest {
        topic,
        request_bytes,
    }) {
        // Clean up
        let mut syncs = PENDING_SYNCS.lock().unwrap();
        syncs.remove(&request_id);
        return Err(anyhow::anyhow!("Failed to send P2P request: {}", e));
    }

    // Wait for response with 15-second timeout
    let result = tokio::time::timeout(
        tokio::time::Duration::from_secs(15),
        notify.notified(),
    ).await;

    // Extract response and clean up
    let response = {
        let mut syncs = PENDING_SYNCS.lock().unwrap();
        let pending = syncs.remove(&request_id);
        pending.and_then(|p| {
            let guard = p.lock().unwrap();
            guard.response.clone()
        })
    };

    match result {
        Ok(()) => {
            // Got notified — response should be present
            if let Some(resp) = response {
                info!(
                    "🔄 [STATE SYNC P2P] Received response from {} — {}",
                    hex::encode(&resp.responder[..8]),
                    resp.summary(),
                );
                let merge_result = merge_p2p_response(app_state, &resp).await;
                if merge_result.anything_changed() {
                    info!(
                        "✅ [STATE SYNC P2P] Merged: contracts +{}/{}, pools +{}/{}, wallets +{}, tokens +{}",
                        merge_result.contracts_added, merge_result.contracts_skipped,
                        merge_result.pools_added, merge_result.pools_updated,
                        merge_result.wallets_added, merge_result.tokens_added,
                    );
                    // Emit SSE
                    let _ = app_state.event_broadcaster.broadcast(
                        crate::streaming::StreamEvent::StateSyncComplete {
                            contracts_added: merge_result.contracts_added,
                            pools_added: merge_result.pools_added,
                            balances_added: merge_result.wallets_added + merge_result.tokens_added,
                            timestamp: chrono::Utc::now(),
                        },
                    ).await;
                    return Ok(true);
                } else {
                    debug!("🔄 [STATE SYNC P2P] No new state in response");
                    return Ok(false);
                }
            }
            // Notified but no response
            Ok(false)
        }
        Err(_) => {
            debug!("🔄 [STATE SYNC P2P] Timeout waiting for response (15s)");
            Ok(false)
        }
    }
}

// ============================================================================
// Request Handler — called from main.rs when we receive a /state-sync-requests message
// ============================================================================

/// Handle an incoming state sync request from P2P.
/// Gathers our state, signs a response, and publishes it.
pub async fn handle_state_sync_request(
    app_state: &Arc<AppState>,
    request: &q_types::state_sync::StateSnapshotRequest,
) {
    use q_types::state_sync::{ContractSyncEntry, PoolSyncEntry, StateSnapshotResponse};

    // Skip if requester is ourselves
    let our_pubkey: [u8; 32] = app_state.node_signing_key.verifying_key().to_bytes();
    if request.requester == our_pubkey {
        debug!("🔄 [STATE SYNC] Skipping self-request");
        return;
    }

    // Check freshness
    if !request.is_fresh() {
        warn!("🔄 [STATE SYNC] Ignoring stale request (timestamp too old)");
        return;
    }

    let our_height = app_state
        .current_height_atomic
        .load(std::sync::atomic::Ordering::SeqCst);

    info!(
        "🔄 [STATE SYNC] Received request id={} from {} (their height={}, contracts={}, pools={})",
        request.request_id,
        hex::encode(&request.requester[..8]),
        request.current_height,
        request.known_contracts,
        request.known_pools,
    );

    // Build contracts list
    let mut contracts = Vec::new();
    {
        let deployed = app_state.orobit_ecosystem.deployed_contracts.read().await;
        for (addr, contract) in deployed.iter() {
            let symbol = contract.metadata.symbol.clone().unwrap_or_default();
            let name = contract.metadata.name.clone();
            let decimals = contract.deployment_params.get("decimals")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as u8;
            let total_supply = contract.deployment_params.get("initialSupply")
                .map(|v| match v {
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::String(s) => s.clone(),
                    _ => "0".to_string(),
                })
                .unwrap_or_else(|| "0".to_string());
            let contract_type = format!("{:?}", contract.contract_type);

            contracts.push(ContractSyncEntry {
                address: addr.0,
                symbol,
                name,
                decimals,
                total_supply,
                deployer: contract.deployer,
                contract_type,
                deployed_at: contract.deployed_at,
                deployment_params: contract.deployment_params.clone(),
            });
        }
    }

    // Build pools list
    let mut pools = Vec::new();
    {
        let pool_map = app_state.liquidity_pools.read().await;
        for (pool_id, pool) in pool_map.iter() {
            pools.push(PoolSyncEntry {
                pool_id: pool_id.clone(),
                token0: pool.token0.clone(),
                token1: pool.token1.clone(),
                reserve0: pool.reserve0.to_string(),
                reserve1: pool.reserve1.to_string(),
                lp_token_supply: pool.lp_token_supply.to_string(),
                provider: pool.provider,
                created_at_unix: pool.created_at.timestamp() as u64,
                token0_decimals: pool.token0_decimals,
                token1_decimals: pool.token1_decimals,
            });
        }
    }

    // Build wallet balances
    let mut wallet_balances = HashMap::new();
    {
        let balances = app_state.wallet_balances.read().await;
        for (addr, amount) in balances.iter() {
            wallet_balances.insert(hex::encode(addr), amount.to_string());
        }
    }

    // Build token balances
    let mut token_balances = HashMap::new();
    {
        let balances = app_state.token_balances.read().await;
        for ((wallet, token), amount) in balances.iter() {
            let key = format!("{}_{}", hex::encode(wallet), hex::encode(token));
            token_balances.insert(key, amount.to_string());
        }
    }

    // Build symbol_to_address
    let mut symbol_to_address = HashMap::new();
    for entry in app_state.symbol_to_address.iter() {
        symbol_to_address.insert(entry.key().clone(), entry.value().clone());
    }

    // v8.7.4: Include vault data for one-time migration to new nodes
    let vault_data = {
        let vault = app_state.collateral_vault.read().await;
        // Only include if vault has any positions
        if vault.total_qug_locked > 0 || vault.total_qugusd_minted > 0 {
            match bincode::serialize(&*vault) {
                Ok(bytes) => {
                    info!("🏦 [STATE SYNC v8.7.4] Including vault data ({} bytes, locked={}, minted={})",
                          bytes.len(), vault.total_qug_locked, vault.total_qugusd_minted);
                    Some(bytes)
                }
                Err(e) => {
                    warn!("⚠️ [STATE SYNC] Failed to serialize vault data: {}", e);
                    None
                }
            }
        } else {
            None
        }
    };

    // Build and sign response
    let mut response = StateSnapshotResponse::new(request.request_id, our_pubkey, our_height);
    response.contracts = contracts;
    response.pools = pools;
    response.wallet_balances = wallet_balances.clone();
    response.token_balances = token_balances;
    response.symbol_to_address = symbol_to_address;
    response.vault_data = vault_data;

    // v1.0.3: Compute balance state hash for divergence detection
    // Hash is computed from sorted wallet balances to be deterministic
    {
        let mut sorted_balances: Vec<(&String, &String)> = wallet_balances.iter().collect();
        sorted_balances.sort_by_key(|(k, _)| k.clone());
        let mut hasher = blake3::Hasher::new();
        for (addr, amount) in &sorted_balances {
            hasher.update(addr.as_bytes());
            hasher.update(b":");
            hasher.update(amount.as_bytes());
            hasher.update(b"\n");
        }
        response.balance_state_hash = Some(hasher.finalize().to_hex().to_string());
    }

    if let Err(e) = response.sign(&app_state.node_signing_key) {
        error!("🔄 [STATE SYNC] Failed to sign response: {}", e);
        return;
    }

    info!(
        "🔄 [STATE SYNC] Sending response: {}",
        response.summary(),
    );

    // Serialize and publish
    let response_bytes = match serde_json::to_vec(&response) {
        Ok(bytes) => bytes,
        Err(e) => {
            error!("🔄 [STATE SYNC] Failed to serialize response: {}", e);
            return;
        }
    };

    let network_id = std::env::var("Q_NETWORK_ID")
        .ok()
        .and_then(|s| s.parse::<q_types::NetworkId>().ok())
        .unwrap_or(q_types::NetworkId::MainnetGenesis);
    let topic = network_id.state_sync_responses_topic();

    if let Some(ref network_tx) = app_state.libp2p_command_tx {
        if let Err(e) = network_tx.send(q_network::NetworkCommand::PublishStateSyncResponse {
            topic,
            response_bytes,
        }) {
            warn!("🔄 [STATE SYNC] Failed to publish response: {}", e);
        }
    }
}

// ============================================================================
// Response Handler — called from main.rs when we receive a /state-sync-responses message
// ============================================================================

/// Handle an incoming state sync response from P2P.
/// If it matches a pending request, store it and notify the waiter.
pub async fn handle_state_sync_response(
    app_state: &Arc<AppState>,
    response: &q_types::state_sync::StateSnapshotResponse,
) {
    // Skip if responder is ourselves
    let our_pubkey: [u8; 32] = app_state.node_signing_key.verifying_key().to_bytes();
    if response.responder == our_pubkey {
        debug!("🔄 [STATE SYNC] Skipping self-response");
        return;
    }

    info!(
        "🔄 [STATE SYNC] Received P2P response for request_id={} from {} — {}",
        response.request_id,
        hex::encode(&response.responder[..8]),
        response.summary(),
    );

    // Check if we have a pending sync for this request_id
    let pending = {
        let syncs = PENDING_SYNCS.lock().unwrap();
        syncs.get(&response.request_id).cloned()
    };

    if let Some(pending) = pending {
        // Store response and notify waiter
        let notify = {
            let mut guard = pending.lock().unwrap();
            guard.response = Some(response.clone());
            guard.notify.clone()
        };
        notify.notify_one();
        debug!("🔄 [STATE SYNC] Delivered response for request_id={}", response.request_id);
    } else {
        // No pending request — this is a response to someone else's request,
        // but we can still opportunistically merge the data
        info!(
            "🔄 [STATE SYNC] Opportunistic merge from unsolicited response (request_id={})",
            response.request_id,
        );
        let merge_result = merge_p2p_response(app_state, response).await;
        if merge_result.anything_changed() {
            info!(
                "✅ [STATE SYNC P2P] Opportunistic merge: contracts +{}, pools +{}, wallets +{}, tokens +{}",
                merge_result.contracts_added,
                merge_result.pools_added,
                merge_result.wallets_added,
                merge_result.tokens_added,
            );
            let _ = app_state.event_broadcaster.broadcast(
                crate::streaming::StreamEvent::StateSyncComplete {
                    contracts_added: merge_result.contracts_added,
                    pools_added: merge_result.pools_added,
                    balances_added: merge_result.wallets_added + merge_result.tokens_added,
                    timestamp: chrono::Utc::now(),
                },
            ).await;
        }
    }
}

// ============================================================================
// Merge logic — shared between P2P and HTTP paths
// ============================================================================

#[derive(Debug, Default)]
struct MergeResult {
    contracts_added: usize,
    contracts_skipped: usize,
    pools_added: usize,
    pools_updated: usize,
    wallets_added: usize,
    tokens_added: usize,
    symbols_added: usize,
    vaults_imported: bool,
}

impl MergeResult {
    fn anything_changed(&self) -> bool {
        self.contracts_added > 0
            || self.pools_added > 0
            || self.pools_updated > 0
            || self.wallets_added > 0
            || self.tokens_added > 0
            || self.symbols_added > 0
            || self.vaults_imported
    }
}

/// Merge a P2P StateSnapshotResponse into local state (add-only)
async fn merge_p2p_response(
    app_state: &Arc<AppState>,
    response: &q_types::state_sync::StateSnapshotResponse,
) -> MergeResult {
    let mut result = MergeResult::default();

    // ---- Merge contracts ----
    {
        let mut deployed = app_state.orobit_ecosystem.deployed_contracts.write().await;
        for entry in &response.contracts {
            let contract_addr = q_vm::contracts::orobit_smart_contracts::ContractAddress(entry.address);

            if deployed.contains_key(&contract_addr) {
                result.contracts_skipped += 1;
                continue;
            }

            // Parse contract type
            let contract_type = match entry.contract_type.to_lowercase().as_str() {
                "securetoken" => q_vm::contracts::ContractType::SecureToken,
                "advancedtoken" => q_vm::contracts::ContractType::AdvancedToken,
                "rwatoken" => q_vm::contracts::ContractType::RwaToken,
                "governance" | "governancetoken" => q_vm::contracts::ContractType::Governance,
                _ => q_vm::contracts::ContractType::SecureToken,
            };

            let metadata = q_vm::contracts::orobit_smart_contracts::ContractMetadata {
                name: entry.name.clone(),
                symbol: Some(entry.symbol.clone()),
                description: format!("Synced via P2P state sync from {}", hex::encode(&response.responder[..8])),
                features: std::collections::HashMap::new(),
                governance_enabled: false,
                upgrade_history: Vec::new(),
            };

            let contract_state = q_vm::contracts::orobit_smart_contracts::ContractState {
                active: true,
                paused: false,
                total_calls: 0,
                last_interaction: entry.deployed_at,
                storage_root: [0u8; 32],
            };

            let deployed_contract = q_vm::contracts::DeployedSmartContract {
                address: contract_addr.clone(),
                contract_type,
                deployer: entry.deployer,
                deployed_at: entry.deployed_at,
                deployment_tx: hex::encode(&entry.address),
                deployment_params: entry.deployment_params.clone(),
                verified: false,
                contract_state,
                metadata,
            };

            // Persist to storage
            if let Ok(contract_data) = serde_json::to_vec(&deployed_contract) {
                if let Err(e) = app_state.storage_engine.save_contract(&entry.address, &contract_data).await {
                    warn!("🔄 [STATE SYNC] Failed to persist contract {}: {}", entry.symbol, e);
                }
            }

            // Update symbol_to_address
            if !entry.symbol.is_empty() {
                let addr_str = format!("qnk{}", hex::encode(contract_addr.0));
                app_state.symbol_to_address.insert(entry.symbol.to_uppercase(), addr_str);
            }

            deployed.insert(contract_addr, deployed_contract);
            result.contracts_added += 1;
            info!("🪙 [STATE SYNC] Added contract: {} ({})", entry.symbol, entry.name);
        }
    }

    // ---- Merge liquidity pools ----
    {
        let our_height = app_state
            .current_height_atomic
            .load(std::sync::atomic::Ordering::SeqCst);

        let mut pools = app_state.liquidity_pools.write().await;
        for entry in &response.pools {
            let reserve0: u128 = entry.reserve0.parse().unwrap_or(0);
            let reserve1: u128 = entry.reserve1.parse().unwrap_or(0);
            let lp_supply: u128 = entry.lp_token_supply.parse().unwrap_or(0);

            // v10.2.2: Insertion-time validation — reject dust/broken pools from state sync
            const SYNC_MIN_POOL_RESERVE: u128 = 10_000_000_000_000_000_000_000; // 10^22 = 0.01 display
            if reserve0 < SYNC_MIN_POOL_RESERVE || reserve1 < SYNC_MIN_POOL_RESERVE {
                tracing::debug!(
                    "🚫 [STATE SYNC] Skipping dust pool: {} ({}/{}) r0={} r1={}",
                    entry.pool_id, entry.token0, entry.token1, reserve0, reserve1
                );
                continue;
            }

            if !pools.contains_key(&entry.pool_id) {
                // New pool
                let pool = crate::LiquidityPool {
                    pool_id: entry.pool_id.clone(),
                    token0: entry.token0.clone(),
                    token1: entry.token1.clone(),
                    reserve0,
                    reserve1,
                    provider: entry.provider,
                    created_at: chrono::DateTime::from_timestamp(entry.created_at_unix as i64, 0)
                        .unwrap_or_else(chrono::Utc::now),
                    lp_token_supply: lp_supply,
                    token0_decimals: entry.token0_decimals,
                    token1_decimals: entry.token1_decimals,
                };

                if let Ok(data) = serde_json::to_vec(&pool) {
                    if let Err(e) = app_state.storage_engine.save_liquidity_pool(&entry.pool_id, &data).await {
                        warn!("🔄 [STATE SYNC] Failed to persist pool {}: {}", entry.pool_id, e);
                    }
                }
                pools.insert(entry.pool_id.clone(), pool);
                result.pools_added += 1;
                info!("💧 [STATE SYNC] Added pool: {} ({}/{})", entry.pool_id, entry.token0, entry.token1);
            } else if let Some(local_pool) = pools.get_mut(&entry.pool_id) {
                // v8.7.2: Update reserves if peer is ahead OR local reserves are zero (post-restart recovery)
                let local_empty = local_pool.reserve0 == 0 && local_pool.reserve1 == 0;
                let peer_has_data = reserve0 > 0 || reserve1 > 0;
                if response.block_height > our_height || (local_empty && peer_has_data) {
                    if local_empty && peer_has_data {
                        info!("🔄 [STATE SYNC P2P] Restoring zero-reserve pool {} from peer (r0={}, r1={})",
                              entry.pool_id, reserve0, reserve1);
                    }
                    local_pool.reserve0 = reserve0;
                    local_pool.reserve1 = reserve1;
                    local_pool.lp_token_supply = lp_supply;
                    if let Ok(data) = serde_json::to_vec(local_pool) {
                        let _ = app_state.storage_engine.save_liquidity_pool(&entry.pool_id, &data).await;
                    }
                    result.pools_updated += 1;
                }
            }
        }
    }

    // ---- Wallet balances: ONE-TIME BOOTSTRAP SYNC (v8.8.1) ----
    // v8.5.4: Ongoing wallet balance import disabled (DEX swap debit erasure).
    // v8.8.1: ONE-TIME bootstrap import for nodes that have never imported before.
    // Safe because during initial state sync there are no local DEX swaps to protect.
    // v10.7.2: CHECKPOINT GATE — if the node applied the balance checkpoint, P2P bootstrap
    //   must be skipped entirely. Block replay from the checkpoint is the sole balance source.
    //   Without this gate the P2P snapshot (at live-network height) overwrites checkpoint
    //   balances, and subsequent block replay double-counts every coinbase since the checkpoint
    //   warp-floor (~7,000 QUG phantom inflation observed in sync tests).
    // Capture BEFORE wallet bootstrap runs — used for QUGUSD gating below.
    let bootstrap_was_done_before_this_sync = app_state.bootstrap_wallet_sync_done
        .load(std::sync::atomic::Ordering::SeqCst);
    if !response.wallet_balances.is_empty() {
        let already_done = bootstrap_was_done_before_this_sync;
        // v10.7.2: also treat checkpoint-applied nodes as "already done"
        let checkpoint_applied = app_state.storage_engine.is_checkpoint_applied().await;

        if already_done || checkpoint_applied {
            if checkpoint_applied && !already_done {
                debug!("🔒 [STATE SYNC v10.7.2] Skipping {} wallet balances — balance checkpoint applied; block replay is sole source",
                       response.wallet_balances.len());
                // Mark as done so future checks skip the is_checkpoint_applied() DB read
                app_state.bootstrap_wallet_sync_done
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            } else {
            debug!("🔒 [STATE SYNC v8.5.4] Skipping {} wallet balances (bootstrap already done)",
                   response.wallet_balances.len());
            }

            // v10.9.6: After replay is done, import any wallet addresses that are completely
            // absent locally. These are transfer-only wallets missed during coinbase-only turbo
            // sync. Adding them never violates max-wins (existing=0 < peer_balance).
            // Gate on replay-done so we don't race with SYNC-006 building the correct balance map.
            if checkpoint_applied && app_state.storage_engine.is_balance_replay_done().await {
                let missing: Vec<([u8; 32], u128)> = {
                    let wb = app_state.wallet_balances.read().await;
                    response.wallet_balances.iter()
                        .filter_map(|(addr_hex, amount_str)| {
                            let addr = hex_to_32bytes(addr_hex)?;
                            if wb.contains_key(&addr) { return None; }
                            let amount: u128 = amount_str.parse().ok().filter(|&a| a > 0)?;
                            Some((addr, amount))
                        })
                        .collect()
                };
                if !missing.is_empty() {
                    let count = missing.len();
                    info!("🔧 [NEW-WALLET IMPORT v10.9.6] Importing {} wallets absent locally (coinbase-only sync gap, peer={})",
                          count, response.block_height);
                    let mut wb = app_state.wallet_balances.write().await;
                    for (addr, amount) in &missing {
                        if let Err(e) = app_state.storage_engine.save_wallet_balance(addr, *amount).await {
                            warn!("⚠️ [NEW-WALLET IMPORT] Failed to persist {}: {}", hex::encode(&addr[..8]), e);
                            continue;
                        }
                        wb.insert(*addr, *amount);
                    }
                    let new_total = wb.len();
                    drop(wb);
                    let total: u128 = {
                        let wb = app_state.wallet_balances.read().await;
                        wb.values().sum()
                    };
                    {
                        let mut supply = app_state.total_minted_supply.write().await;
                        *supply = total;
                    }
                    info!("✅ [NEW-WALLET IMPORT v10.9.6] Added {} missing wallets → {} total wallets", count, new_total);
                    result.wallets_added += count;
                }
            }
        } else {
            let our_height = app_state.current_height_atomic
                .load(std::sync::atomic::Ordering::SeqCst);
            let our_wallet_count = {
                let wb = app_state.wallet_balances.read().await;
                wb.len()
            };
            let peer_wallet_count = response.wallet_balances.len();
            // Only import if peer is ahead AND has more wallets
            let peer_has_more = peer_wallet_count > our_wallet_count + 5;

            if response.block_height > our_height && peer_has_more {
                info!("🚀 [BOOTSTRAP SYNC v8.8.1] One-time wallet balance import: {} wallets from peer \
                       at height {} (we have {} wallets at height {})",
                      peer_wallet_count, response.block_height, our_wallet_count, our_height);

                let mut imported = 0u64;
                let mut updated = 0u64;
                {
                    let mut balances = app_state.wallet_balances.write().await;
                    for (addr_hex, amount_str) in &response.wallet_balances {
                        let addr_bytes = match hex_to_32bytes(addr_hex) {
                            Some(b) => b,
                            None => continue,
                        };
                        let amount: u128 = match amount_str.parse() {
                            Ok(a) if a > 0 => a,
                            _ => continue,
                        };
                        let current = balances.get(&addr_bytes).copied().unwrap_or(0);
                        if amount > current {
                            // 🔴 [BALANCE WRITE DEBUG] Bootstrap sync overwrite
                            let addr_hex_dbg = hex::encode(&addr_bytes);
                            warn!(
                                "🔴 [BALANCE WRITE] bootstrap_sync(): wallet={} old={} new={} delta=+{} caller=BOOTSTRAP_P2P_SYNC height={}",
                                &addr_hex_dbg[..16], current, amount, amount - current, response.block_height
                            );
                            if let Err(e) = app_state.storage_engine
                                .save_wallet_balance(&addr_bytes, amount).await
                            {
                                warn!("⚠️ [BOOTSTRAP SYNC] Failed to persist: {}", e);
                                continue;
                            }
                            if current == 0 { imported += 1; } else { updated += 1; }
                            balances.insert(addr_bytes, amount);
                        }
                    }
                }
                // Recalculate total supply
                {
                    let balances = app_state.wallet_balances.read().await;
                    let total: u128 = balances.values().sum();
                    let mut supply = app_state.total_minted_supply.write().await;
                    *supply = total;
                    info!("✅ [BOOTSTRAP SYNC v8.8.1] Imported {} new + updated {} wallets. \
                           Total supply: {} QUG",
                          imported, updated, total / 1_000_000_000_000_000_000_000_000u128);
                }
                // Set migration flag — never do this again
                let _ = app_state.storage_engine
                    .set_migration_flag(crate::BOOTSTRAP_WALLET_SYNC_FLAG).await;
                app_state.bootstrap_wallet_sync_done
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                result.wallets_added = imported as usize;
            } else {
                debug!("🔒 [BOOTSTRAP v8.8.1] Skipping: peer not ahead enough or insufficient extra wallets \
                        (peer height={}, our height={}, peer wallets={}, our wallets={})",
                       response.block_height, our_height, peer_wallet_count, our_wallet_count);
            }
        }
    }

    // ---- Merge token balances (add-only for NEW tokens, never overwrite existing) ----
    // v8.5.2 FIX: Only insert token balances for keys that don't exist locally at all.
    // Previously, if a user swapped QUGUSD→QUG (balance set to 0 in HashMap),
    // P2P sync from a stale peer would re-insert the old balance → infinite money glitch.
    // Now: if we have ANY record (even 0), we trust our local state over peers.
    //
    // v8.5.6 FIX: REJECT ALL QUGUSD token balances from P2P state sync.
    // Nodes running older versions have a ghost QUGUSD balance (172K+ QUGUSD) that
    // resurrects on every restart and propagates to the entire network via state sync.
    // QUGUSD is only legitimately created at runtime via vault minting or DEX swaps.
    {
        let qugusd_addr = q_types::QUGUSD_TOKEN_ADDRESS;
        let mut qugusd_rejected = 0u64;
        let mut balances = app_state.token_balances.write().await;
        for (composite_key, amount_str) in &response.token_balances {
            let parts: Vec<&str> = composite_key.splitn(2, '_').collect();
            if parts.len() != 2 {
                continue;
            }
            let wallet_bytes = match hex_to_32bytes(parts[0]) {
                Some(b) => b,
                None => continue,
            };
            let token_bytes = match hex_to_32bytes(parts[1]) {
                Some(b) => b,
                None => continue,
            };
            let amount: u128 = match amount_str.parse() {
                Ok(a) => a,
                Err(_) => continue,
            };

            // v8.5.6: Block QUGUSD from P2P — ghost balance propagation prevention
            // v8.8.2: Allow QUGUSD only on FIRST state sync (before wallet bootstrap completes).
            // v1.0.3: After convergence migration, QUGUSD is chain-derived and correct.
            //         Ghost prevention only needed for pre-migration nodes.
            if token_bytes == qugusd_addr && bootstrap_was_done_before_this_sync {
                let convergence_done = app_state.storage_engine
                    .has_migration_flag(b"migration_safe_convergence_v103_done").await;
                if !convergence_done {
                    qugusd_rejected += 1;
                    continue;
                }
                // Post-convergence: allow QUGUSD from peers (chain is source of truth)
            }

            let key = (wallet_bytes, token_bytes);
            // v8.5.2: Check BOTH in-memory HashMap AND RocksDB.
            // If we have any local record (even 0 from a swap), don't overwrite.
            let has_local = balances.contains_key(&key);
            let has_persisted = if !has_local {
                // Check RocksDB — if key exists at all (even 0), trust local state
                app_state.storage_engine.has_token_balance_key(&wallet_bytes, &token_bytes).await
            } else {
                true
            };

            if !has_local && !has_persisted && amount > 0 {
                if let Err(e) = app_state
                    .storage_engine
                    .save_token_balance(&wallet_bytes, &token_bytes, amount)
                    .await
                {
                    warn!("🔄 [STATE SYNC] Failed to persist token balance: {}", e);
                }
                balances.insert(key, amount);
                result.tokens_added += 1;
            }
        }
        if qugusd_rejected > 0 {
            info!("🛡️ [STATE SYNC P2P v8.5.6] Rejected {} QUGUSD token balances from peer (ghost prevention)", qugusd_rejected);
        }
    }

    // ---- Merge symbol_to_address ----
    for (symbol, address) in &response.symbol_to_address {
        if !app_state.symbol_to_address.contains_key(symbol) {
            app_state.symbol_to_address.insert(symbol.clone(), address.clone());
            result.symbols_added += 1;
        }
    }

    // ---- v1.0.3: Balance state hash divergence detection ----
    if let Some(ref peer_hash) = response.balance_state_hash {
        let our_height = app_state.current_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
        // Only compare if we're at similar heights (within 100 blocks)
        if our_height > 0 && response.block_height > 0
            && (our_height as i64 - response.block_height as i64).unsigned_abs() < 100
        {
            // Compute our own balance state hash (same format as response builder)
            let our_hash = {
                let wb = app_state.wallet_balances.read().await;
                let mut sorted: Vec<(String, String)> = wb.iter()
                    .map(|(addr, amount)| (hex::encode(addr), amount.to_string()))
                    .collect();
                sorted.sort_by(|(a, _), (b, _)| a.cmp(b));
                let mut hasher = blake3::Hasher::new();
                for (addr, amount) in &sorted {
                    hasher.update(addr.as_bytes());
                    hasher.update(b":");
                    hasher.update(amount.as_bytes());
                    hasher.update(b"\n");
                }
                hasher.finalize().to_hex().to_string()
            };

            if &our_hash == peer_hash {
                info!("✅ [DIVERGENCE CHECK] Balance hash MATCHES peer at heights ~{}/{} (hash={}..)",
                      our_height, response.block_height, &peer_hash[..12]);
            } else {
                error!("🚨 [DIVERGENCE CHECK] CRITICAL: Balance hash MISMATCH with peer!");
                error!("   Our height: {}, peer height: {}", our_height, response.block_height);
                error!("   Our hash:  {}", &our_hash[..24]);
                error!("   Peer hash: {}", &peer_hash[..24]);
                // v10.9.55 (2026-05-18): the previous advice to "delete the
                // convergence migration flag and restart" has been REMOVED.
                // The migration replayed the chain to rebuild wallet balances;
                // on a sparse chain that destroys real mining rewards. See
                // CLAUDE.md "BALANCE INTEGRITY" rules + the v10.9.55 removal
                // tombstones in crates/q-storage/src/lib.rs around line 7910
                // and crates/q-api-server/src/main.rs around line 4220.
                error!("   DO NOT replay-rebuild balances on this node. Recovery procedure:");
                error!("     1. Identify which side is correct via independent audit (block-replay arithmetic on a known wallet)");
                error!("     2. If THIS node is wrong: stop, clean-resync from a known-good peer with empty DB");
                error!("     3. If PEER is wrong: continue serving; peer should clean-resync");
                error!("     Never run any migration that bulk-deletes wallet_balance_* keys.");
            }
        }
    }

    // ---- v8.7.4 / v1.0.3: Merge vault data ----
    // v1.0.3: Import vault from peer if local is empty OR if peer has significantly
    // more vault activity (processed more blocks with vault txs). After convergence
    // migration, vault state is chain-derived so the "empty-only" restriction is relaxed.
    if let Some(ref vault_bytes) = response.vault_data {
        let should_import = {
            let vault = app_state.collateral_vault.read().await;
            let local_empty = vault.total_qug_locked == 0 && vault.total_qugusd_minted == 0;
            let our_height = app_state.current_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
            // Import if empty OR if peer is significantly ahead (processed more vault txs)
            local_empty || (response.block_height > our_height + 1000)
        };

        if should_import && !vault_bytes.is_empty() {
            match bincode::deserialize::<q_vm::contracts::CollateralVault>(vault_bytes) {
                Ok(peer_vault) => {
                    if peer_vault.total_qug_locked > 0 || peer_vault.total_qugusd_minted > 0 {
                        info!(
                            "🏦 [STATE SYNC v1.0.3] Importing vault state from peer: locked={}, minted={}, {} positions (peer height={})",
                            peer_vault.total_qug_locked,
                            peer_vault.total_qugusd_minted,
                            peer_vault.locked_qug.len(),
                            response.block_height,
                        );
                        // Persist to storage
                        if let Err(e) = app_state.storage_engine.save_collateral_vault_data(vault_bytes).await {
                            warn!("⚠️ [STATE SYNC v1.0.3] Failed to persist vault data: {}", e);
                        }
                        // Update in-memory vault
                        let mut vault = app_state.collateral_vault.write().await;
                        *vault = peer_vault;
                        result.vaults_imported = true;
                    }
                }
                Err(e) => {
                    warn!("⚠️ [STATE SYNC v1.0.3] Failed to deserialize vault data ({} bytes): {}", vault_bytes.len(), e);
                }
            }
        }
    }

    result
}

// ============================================================================
// HTTP fallback sync (original implementation)
// ============================================================================

async fn do_http_state_sync(app_state: &Arc<AppState>, our_port: u16) {
    let bootstrap_peers: &[&str] = &[
        "http://89.149.241.126:8080",   // Epsilon — authoritative 10Gbit supernode, most complete balance state
        "http://185.182.185.227:8080",  // Beta
        "http://109.205.176.60:8080",   // Gamma
        "http://161.35.219.10:8080",    // Alpha
    ];

    let our_ips = get_local_ips();

    // Collect snapshots from all reachable peers, then use the most complete one.
    // "Most complete" = highest wallet_balances count (most state coverage).
    let mut best_snapshot: Option<(String, FullStateSnapshot)> = None;

    for peer_url in bootstrap_peers {
        if is_self(peer_url, our_port, &our_ips) {
            debug!("🔄 [STATE SYNC HTTP] Skipping self: {}", peer_url);
            continue;
        }

        let url = format!("{}/api/v1/sync/full-state", peer_url);
        info!("🔄 [STATE SYNC HTTP] Fetching state from {}", peer_url);

        match fetch_with_timeout(&url).await {
            Ok(snapshot) => {
                info!(
                    "🔄 [STATE SYNC HTTP] Received snapshot from {}: {} contracts, {} pools, {} wallets, {} tokens (height {}, network={:?})",
                    peer_url,
                    snapshot.contracts.len(),
                    snapshot.liquidity_pools.len(),
                    snapshot.wallet_balances.len(),
                    snapshot.token_balances.len(),
                    snapshot.block_height,
                    snapshot.network_id.as_deref().unwrap_or("unknown"),
                );
                let is_better = match &best_snapshot {
                    None => true,
                    Some((_, prev)) => snapshot.wallet_balances.len() > prev.wallet_balances.len(),
                };
                if is_better {
                    best_snapshot = Some((peer_url.to_string(), snapshot));
                }
            }
            Err(e) => {
                warn!("🔄 [STATE SYNC HTTP] Failed to fetch from {}: {}", peer_url, e);
            }
        }
    }

    if let Some((peer_url, snapshot)) = best_snapshot {
        info!("🔄 [STATE SYNC HTTP] Using best snapshot from {} ({} wallets)", peer_url, snapshot.wallet_balances.len());

        // v10.5.4: Log wallet count BEFORE applying so we can see the delta clearly.
        // This is critical for diagnosing Q_SKIP_CHECKPOINT balance bugs.
        let before_count = {
            let balances = app_state.wallet_balances.read().await;
            balances.len()
        };
        info!("📊 [BALANCE SYNC] Before: {} wallets in RAM. Applying {} wallets from {}.",
              before_count, snapshot.wallet_balances.len(), peer_url);

        let result = merge_http_snapshot(app_state, &snapshot).await;

        // v10.5.4: Log wallet count AFTER merge to confirm balances were populated.
        let after_count = {
            let balances = app_state.wallet_balances.read().await;
            balances.len()
        };
        info!("📊 [BALANCE SYNC] After: {} wallets in RAM (+{} new wallets applied).",
              after_count, after_count.saturating_sub(before_count));

        if result.anything_changed() {
            info!(
                "🔄 [STATE SYNC HTTP] Merged from {}: contracts +{}/{}, pools +{}/{}, wallets +{}, tokens +{}",
                peer_url,
                result.contracts_added, result.contracts_skipped,
                result.pools_added, result.pools_updated,
                result.wallets_added,
                result.tokens_added,
            );
            let _ = app_state.event_broadcaster.broadcast(
                crate::streaming::StreamEvent::StateSyncComplete {
                    contracts_added: result.contracts_added,
                    pools_added: result.pools_added,
                    balances_added: result.wallets_added + result.tokens_added,
                    timestamp: chrono::Utc::now(),
                },
            ).await;
        } else {
            debug!("🔄 [STATE SYNC HTTP] No new state from {}", peer_url);
        }
    } else {
        warn!("🔄 [STATE SYNC] Could not reach any peer (P2P or HTTP) for state sync");
    }
}

/// v9.0.0: One-time authoritative balance sync from a trusted peer.
/// Fetches ALL wallet balances from the authority peer and overwrites local RocksDB values.
/// This fixes balance divergence after incomplete chain replay (which misses DEX protocol fees).
async fn do_authoritative_balance_sync(app_state: &Arc<AppState>, authority_url: &str) -> anyhow::Result<usize> {
    let url = format!("{}/api/v1/sync/full-state", authority_url.trim_end_matches('/'));
    info!("🔑 [AUTHORITY SYNC] Fetching state from {}", url);

    let snapshot = fetch_with_timeout(&url).await?;
    info!("🔑 [AUTHORITY SYNC] Received snapshot: {} wallet balances, height {}",
          snapshot.wallet_balances.len(), snapshot.block_height);

    if snapshot.wallet_balances.is_empty() {
        anyhow::bail!("Authority peer returned 0 wallet balances");
    }

    let qug_unit: u128 = 1_000_000_000_000_000_000_000_000;
    let mut imported = 0usize;
    let mut total_supply: u128 = 0;

    // Overwrite ALL wallet balances in RocksDB
    for (address_hex, balance_str) in &snapshot.wallet_balances {
        let balance: u128 = match balance_str.parse() {
            Ok(b) => b,
            Err(_) => {
                warn!("🔑 [AUTHORITY SYNC] Invalid balance for {}: {}", &address_hex[..16], balance_str);
                continue;
            }
        };

        let key = format!("wallet_balance_{}", address_hex);
        // 🔴 [BALANCE WRITE DEBUG] Read old value BEFORE authority sync overwrite
        let old_balance_authority = {
            let addr_bytes_check: Option<[u8; 32]> = hex::decode(address_hex).ok().and_then(|b| {
                if b.len() == 32 { let mut arr = [0u8; 32]; arr.copy_from_slice(&b); Some(arr) } else { None }
            });
            if let Some(ab) = addr_bytes_check {
                app_state.storage_engine.load_wallet_balance(&ab).await.ok().flatten().unwrap_or(0)
            } else { 0u128 }
        };
        if old_balance_authority != balance {
            let delta_abs = if balance >= old_balance_authority { balance - old_balance_authority } else { old_balance_authority - balance };
            let direction = if balance >= old_balance_authority { "+" } else { "-" };
            if balance < old_balance_authority {
                error!(
                    "🔴 [BALANCE WRITE] authority_sync_db_put(): wallet={} old={} new={} delta={}{} caller=AUTHORITY_SYNC height={}",
                    &address_hex[..16.min(address_hex.len())], old_balance_authority, balance, direction, delta_abs, snapshot.block_height
                );
            } else {
                warn!(
                    "🔴 [BALANCE WRITE] authority_sync_db_put(): wallet={} old={} new={} delta={}{} caller=AUTHORITY_SYNC height={}",
                    &address_hex[..16.min(address_hex.len())], old_balance_authority, balance, direction, delta_abs, snapshot.block_height
                );
            }
        }
        if let Err(e) = app_state.storage_engine.db_put("manifest", key.as_bytes(), &balance.to_le_bytes()).await {
            warn!("🔑 [AUTHORITY SYNC] Failed to write balance for {}: {}", &address_hex[..16], e);
            continue;
        }

        total_supply = total_supply.saturating_add(balance);
        imported += 1;

        // Log significant balances
        if balance > qug_unit {
            info!("🔑 [AUTHORITY SYNC] {} → {} QUG",
                  &address_hex[..16], balance / qug_unit);
        }
    }

    // Update in-memory wallet_balances HashMap
    {
        let mut balances = app_state.wallet_balances.write().await;
        for (address_hex, balance_str) in &snapshot.wallet_balances {
            let balance: u128 = match balance_str.parse() {
                Ok(b) => b,
                Err(_) => continue,
            };
            let addr_bytes: [u8; 32] = match hex::decode(address_hex) {
                Ok(bytes) if bytes.len() == 32 => {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(&bytes);
                    arr
                }
                _ => continue,
            };
            balances.insert(addr_bytes, balance);
        }
    }

    // Update total supply in storage
    if let Err(e) = app_state.storage_engine.db_put(
        "manifest",
        b"total_supply",
        &total_supply.to_le_bytes(),
    ).await {
        warn!("🔑 [AUTHORITY SYNC] Failed to update total_supply: {}", e);
    }

    // v10.3.3: SKIP token balance import — protect QUGUSD ($24M) and all other tokens.
    // DeepSeek + ChatGPT peer review: "If the incident scope is QUG wallet balances corrupted,
    // token balances intact, then do not touch token state during emergency restoration."
    // Token balances on Epsilon are already correct (reorg handler never touched them).
    let mut token_imported = 0usize;
    if !snapshot.token_balances.is_empty() {
        info!("🛡️ [AUTHORITY SYNC v10.3.3] SKIPPING {} token balances — QUGUSD/token protection active", snapshot.token_balances.len());

        // v10.3.3: Token import DISABLED — protect QUGUSD and all tokens
        if false { // DISABLED for QUGUSD protection
        let mut token_bals = app_state.token_balances.write().await;
        for (composite_key, balance_str) in &snapshot.token_balances {
            let balance: u128 = match balance_str.parse() {
                Ok(b) => b,
                Err(_) => continue,
            };

            // Key format: "{wallet_hex}_{token_hex}" (64+1+64 = 129 chars min)
            let parts: Vec<&str> = composite_key.splitn(2, '_').collect();
            if parts.len() != 2 { continue; }

            let wallet_bytes: [u8; 32] = match hex::decode(parts[0]) {
                Ok(bytes) if bytes.len() == 32 => {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(&bytes);
                    arr
                }
                _ => continue,
            };

            let token_bytes: [u8; 32] = match hex::decode(parts[1]) {
                Ok(bytes) if bytes.len() == 32 => {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(&bytes);
                    arr
                }
                Ok(bytes) if bytes.len() < 32 => {
                    // Pad short token IDs (e.g. 8-byte token symbols)
                    let mut arr = [0u8; 32];
                    arr[..bytes.len()].copy_from_slice(&bytes);
                    arr
                }
                _ => continue,
            };

            // Persist to RocksDB state_sync storage
            let db_key = format!("token_balance_{}_{}", parts[0], parts[1]);
            let _ = app_state.storage_engine.db_put(
                "manifest", db_key.as_bytes(), &balance.to_le_bytes()
            ).await;

            token_bals.insert((wallet_bytes, token_bytes), balance);
            token_imported += 1;
        }
        drop(token_bals);

        info!("🔑 [AUTHORITY SYNC] Imported {} token balances", token_imported);
        } // end if false (DISABLED v10.3.3 — QUGUSD protection)
    }

    info!("🔑 [AUTHORITY SYNC] Complete: {} wallets + {} token balances imported, total supply: {} QUG",
          imported, token_imported, total_supply / qug_unit);

    Ok(imported)
}

async fn fetch_with_timeout(url: &str) -> anyhow::Result<FullStateSnapshot> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let resp = client.get(url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {}", resp.status());
    }

    let body: serde_json::Value = resp.json().await?;
    let data = body
        .get("data")
        .ok_or_else(|| anyhow::anyhow!("Missing 'data' field"))?;

    let snapshot: FullStateSnapshot = serde_json::from_value(data.clone())?;
    Ok(snapshot)
}

/// Merge an HTTP FullStateSnapshot (uses serde_json::Value for contracts/pools)
async fn merge_http_snapshot(app_state: &Arc<AppState>, snapshot: &FullStateSnapshot) -> MergeResult {
    let mut result = MergeResult::default();

    // v7.3.0: Reject snapshots from different networks (prevents cross-network contamination)
    let our_network_id = std::env::var("Q_NETWORK_ID")
        .unwrap_or_else(|_| "mainnet-genesis".to_string());
    match &snapshot.network_id {
        Some(their_network_id) if their_network_id != &our_network_id => {
            warn!("🚫 [STATE SYNC HTTP] REJECTED snapshot: network_id mismatch (ours={}, theirs={})",
                  our_network_id, their_network_id);
            return result; // Return empty = no changes
        }
        None => {
            warn!("🚫 [STATE SYNC HTTP] REJECTED snapshot: no network_id field (old binary). \
                   Only accepting snapshots from v7.3.0+ nodes.");
            return result;
        }
        _ => {} // network_id matches — proceed
    }

    // ---- Merge contracts ----
    {
        let mut deployed = app_state.orobit_ecosystem.deployed_contracts.write().await;
        for (addr_hex, contract_val) in &snapshot.contracts {
            let addr_bytes = match hex_to_32bytes(addr_hex) {
                Some(b) => b,
                None => continue,
            };
            let contract_addr = q_vm::contracts::orobit_smart_contracts::ContractAddress(addr_bytes);

            if deployed.contains_key(&contract_addr) {
                result.contracts_skipped += 1;
                continue;
            }

            match serde_json::from_value::<q_vm::contracts::orobit_smart_contracts::DeployedSmartContract>(
                contract_val.clone(),
            ) {
                Ok(contract) => {
                    if let Ok(data) = serde_json::to_vec(&contract) {
                        if let Err(e) = app_state.storage_engine.save_contract(&addr_bytes, &data).await {
                            warn!("🔄 [STATE SYNC HTTP] Failed to persist contract {}: {}", addr_hex, e);
                        }
                    }
                    if let Some(symbol) = &contract.metadata.symbol {
                        if !symbol.is_empty() {
                            app_state.symbol_to_address.insert(symbol.clone(), addr_hex.clone());
                        }
                    }
                    deployed.insert(contract_addr, contract);
                    result.contracts_added += 1;
                }
                Err(e) => {
                    warn!("🔄 [STATE SYNC HTTP] Failed to deserialize contract {}: {}", addr_hex, e);
                }
            }
        }
    }

    // ---- Merge liquidity pools ----
    {
        let our_height = app_state
            .current_height_atomic
            .load(std::sync::atomic::Ordering::SeqCst);
        let mut pools = app_state.liquidity_pools.write().await;
        for (pool_id, pool_val) in &snapshot.liquidity_pools {
            match serde_json::from_value::<crate::LiquidityPool>(pool_val.clone()) {
                Ok(peer_pool) => {
                    // v10.2.2: Insertion-time validation — reject dust/broken pools
                    const HTTP_MIN_POOL_RESERVE: u128 = 10_000_000_000_000_000_000_000; // 10^22
                    if peer_pool.reserve0 < HTTP_MIN_POOL_RESERVE || peer_pool.reserve1 < HTTP_MIN_POOL_RESERVE {
                        tracing::debug!(
                            "🚫 [STATE SYNC HTTP] Skipping dust pool: {} ({}/{}) r0={} r1={}",
                            pool_id, peer_pool.token0, peer_pool.token1, peer_pool.reserve0, peer_pool.reserve1
                        );
                        continue;
                    }

                    if !pools.contains_key(pool_id) {
                        if let Ok(data) = serde_json::to_vec(&peer_pool) {
                            if let Err(e) = app_state.storage_engine.save_liquidity_pool(pool_id, &data).await {
                                warn!("🔄 [STATE SYNC HTTP] Failed to persist pool {}: {}", pool_id, e);
                            }
                        }
                        pools.insert(pool_id.clone(), peer_pool);
                        result.pools_added += 1;
                    } else if let Some(local_pool) = pools.get_mut(pool_id) {
                        // v8.7.2: Update reserves if peer is ahead OR local reserves are zero (post-restart recovery)
                        let local_empty = local_pool.reserve0 == 0 && local_pool.reserve1 == 0;
                        let peer_has_data = peer_pool.reserve0 > 0 || peer_pool.reserve1 > 0;
                        if snapshot.block_height > our_height || (local_empty && peer_has_data) {
                            if local_empty && peer_has_data {
                                info!("🔄 [STATE SYNC HTTP] Restoring zero-reserve pool {} from peer (r0={}, r1={})",
                                      pool_id, peer_pool.reserve0, peer_pool.reserve1);
                            }
                            local_pool.reserve0 = peer_pool.reserve0;
                            local_pool.reserve1 = peer_pool.reserve1;
                            local_pool.lp_token_supply = peer_pool.lp_token_supply;
                            if let Ok(data) = serde_json::to_vec(local_pool) {
                                let _ = app_state.storage_engine.save_liquidity_pool(pool_id, &data).await;
                            }
                            result.pools_updated += 1;
                        }
                    }
                }
                Err(e) => {
                    warn!("🔄 [STATE SYNC HTTP] Failed to deserialize pool {}: {}", pool_id, e);
                }
            }
        }

        // v7.3.0: Update vault QUG price from QUG/QUGUSD pool after pool merge
        // Without this, the vault price stays stale after HTTP state sync overwrites pool reserves.
        if result.pools_added > 0 || result.pools_updated > 0 {
            for p in pools.values() {
                let t0 = p.token0.to_uppercase();
                let t1 = p.token1.to_uppercase();
                let t0_qug = t0 == "QUG" || t0 == "NATIVE-QUG";
                let t1_qug = t1 == "QUG" || t1 == "NATIVE-QUG";
                let t0_usd = t0 == "QUGUSD";
                let t1_usd = t1 == "QUGUSD";
                if (t0_qug && t1_usd) || (t0_usd && t1_qug) {
                    let (qug_r, usd_r) = if t0_qug {
                        (p.reserve0 as f64, p.reserve1 as f64)
                    } else {
                        (p.reserve1 as f64, p.reserve0 as f64)
                    };
                    if qug_r > 0.0 {
                        let pool_price = usd_r / qug_r;
                        // v8.0.1: Reject stale pool prices from old $42.50 era
                        if pool_price >= 100.0 && pool_price < 1_000_000.0 {
                            let mut vault = app_state.collateral_vault.write().await;
                            let old_price = vault.qug_price_usd;
                            vault.qug_price_usd = pool_price;
                            vault.last_price_update = chrono::Utc::now().timestamp();
                            drop(vault);

                            if let Ok(vault_bytes) = bincode::serialize(&*app_state.collateral_vault.read().await) {
                                let _ = app_state.storage_engine.save_collateral_vault_data(&vault_bytes).await;
                            }

                            if (pool_price - old_price).abs() > 0.01 {
                                info!("💱 [STATE SYNC v7.3.0] Updated vault QUG price: ${:.4} → ${:.4} from synced pool",
                                      old_price, pool_price);
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    // ---- Wallet balances: DISABLED (v8.5.4 money glitch fix) ----
    // v8.5.4: See P2P path above. Wallet balance import permanently disabled to prevent
    // DEX swap debits from being overwritten by stale peer balances.
    if !snapshot.wallet_balances.is_empty() {
        debug!("🔒 [STATE SYNC HTTP v8.5.4] Skipping {} wallet balances (disabled — prevents DEX swap debit erasure)",
               snapshot.wallet_balances.len());
    }

    // ---- BFT finality records: AUTHORITATIVE overwrite ----
    // Finality records have 2f+1 Bracha READY signatures — they override block-derived balances.
    // Applied AFTER the disabled wallet_balances section so they always take precedence.
    if !snapshot.finality_records.is_empty() {
        info!("🔐 [STATE SYNC HTTP] Applying {} BFT-finalized balance records from peer",
              snapshot.finality_records.len());
        let mut wb = app_state.wallet_balances.write().await;
        for record in &snapshot.finality_records {
            // Write to in-memory cache
            wb.insert(record.wallet_address, record.new_balance);
            // Persist to RocksDB
            if let Err(e) = app_state.storage_engine.save_wallet_balance(&record.wallet_address, record.new_balance).await {
                warn!("🔐 [STATE SYNC HTTP] Failed to persist finality record for {}: {e}",
                      hex::encode(&record.wallet_address[..8]));
            }
            // Persist the finality proof itself
            if let Err(e) = app_state.storage_engine.put_manifest_sync(
                q_types::BalanceFinalityRecord::db_key(&record.wallet_address).as_bytes(),
                &record.to_cbor().unwrap_or_default(),
            ).await {
                warn!("🔐 [STATE SYNC HTTP] Failed to persist finality proof: {e}");
            }
            result.wallets_added += 1;
        }
        info!("✅ [STATE SYNC HTTP] Applied {} BFT-finalized balances", snapshot.finality_records.len());
    }

    // ---- Merge token balances (add-only for NEW tokens, never overwrite existing) ----
    // v8.5.2 FIX: Same protection as P2P path — don't restore spent token balances
    // v8.5.6 FIX: REJECT ALL QUGUSD token balances (ghost propagation from older nodes)
    {
        let qugusd_addr = q_types::QUGUSD_TOKEN_ADDRESS;
        let mut qugusd_rejected = 0u64;
        let mut balances = app_state.token_balances.write().await;
        for (composite_key, amount_str) in &snapshot.token_balances {
            let parts: Vec<&str> = composite_key.splitn(2, '_').collect();
            if parts.len() != 2 { continue; }
            let wallet_bytes = match hex_to_32bytes(parts[0]) { Some(b) => b, None => continue };
            let token_bytes = match hex_to_32bytes(parts[1]) { Some(b) => b, None => continue };
            let amount: u128 = match amount_str.parse() { Ok(a) => a, Err(_) => continue };

            // v8.5.6: Block QUGUSD from HTTP state sync — ghost balance propagation prevention
            // v1.0.3: After convergence migration, QUGUSD is chain-derived. Allow from peers.
            if token_bytes == qugusd_addr {
                let convergence_done = app_state.storage_engine
                    .has_migration_flag(b"migration_safe_convergence_v103_done").await;
                if !convergence_done {
                    qugusd_rejected += 1;
                    continue;
                }
            }

            let key = (wallet_bytes, token_bytes);
            let has_local = balances.contains_key(&key);
            let has_persisted = if !has_local {
                app_state.storage_engine.get_token_balance(&wallet_bytes, &token_bytes).await.unwrap_or(0) > 0
            } else {
                true
            };

            if !has_local && !has_persisted && amount > 0 {
                if let Err(e) = app_state.storage_engine.save_token_balance(&wallet_bytes, &token_bytes, amount).await {
                    warn!("🔄 [STATE SYNC HTTP] Failed to persist token balance: {}", e);
                }
                balances.insert(key, amount);
                result.tokens_added += 1;
            }
        }
        if qugusd_rejected > 0 {
            info!("🛡️ [STATE SYNC HTTP v8.5.6] Rejected {} QUGUSD token balances from peer (ghost prevention)", qugusd_rejected);
        }
    }

    // ---- Merge symbol_to_address ----
    for (symbol, address) in &snapshot.symbol_to_address {
        if !app_state.symbol_to_address.contains_key(symbol) {
            app_state.symbol_to_address.insert(symbol.clone(), address.clone());
            result.symbols_added += 1;
        }
    }

    result
}

// ============================================================================
// Helpers
// ============================================================================

fn hex_to_32bytes(hex_str: &str) -> Option<[u8; 32]> {
    let bytes = hex::decode(hex_str).ok()?;
    if bytes.len() != 32 {
        return None;
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Some(arr)
}

fn get_local_ips() -> Vec<String> {
    match std::process::Command::new("hostname").arg("-I").output() {
        Ok(output) => {
            let s = String::from_utf8_lossy(&output.stdout);
            s.split_whitespace().map(|ip| ip.to_string()).collect()
        }
        Err(_) => vec![],
    }
}

fn is_self(peer_url: &str, our_port: u16, our_ips: &[String]) -> bool {
    let without_scheme = peer_url
        .strip_prefix("http://")
        .or_else(|| peer_url.strip_prefix("https://"))
        .unwrap_or(peer_url);

    let (peer_host, peer_port) = if let Some(idx) = without_scheme.rfind(':') {
        let host = &without_scheme[..idx];
        let port: u16 = without_scheme[idx + 1..]
            .trim_end_matches('/')
            .parse()
            .unwrap_or(8080);
        (host, port)
    } else {
        (without_scheme.trim_end_matches('/'), 8080u16)
    };

    if peer_port != our_port {
        return false;
    }

    for ip in our_ips {
        if ip == peer_host {
            return true;
        }
    }

    if peer_host == "127.0.0.1" || peer_host == "localhost" || peer_host == "0.0.0.0" {
        return true;
    }

    false
}
