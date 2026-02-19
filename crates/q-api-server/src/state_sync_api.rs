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
        .unwrap_or_else(|_| "mainnet2026.2".to_string());

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
        .unwrap_or(q_types::NetworkId::Mainnet2026_2);
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

    // Build and sign response
    let mut response = StateSnapshotResponse::new(request.request_id, our_pubkey, our_height);
    response.contracts = contracts;
    response.pools = pools;
    response.wallet_balances = wallet_balances;
    response.token_balances = token_balances;
    response.symbol_to_address = symbol_to_address;

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
        .unwrap_or(q_types::NetworkId::Mainnet2026_2);
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
}

impl MergeResult {
    fn anything_changed(&self) -> bool {
        self.contracts_added > 0
            || self.pools_added > 0
            || self.pools_updated > 0
            || self.wallets_added > 0
            || self.tokens_added > 0
            || self.symbols_added > 0
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
            } else if response.block_height > our_height {
                // Peer is ahead — update reserves
                if let Some(local_pool) = pools.get_mut(&entry.pool_id) {
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

    // ---- Merge wallet balances (add-only) ----
    {
        let mut balances = app_state.wallet_balances.write().await;
        for (addr_hex, amount_str) in &response.wallet_balances {
            let addr_bytes = match hex_to_32bytes(addr_hex) {
                Some(b) => b,
                None => continue,
            };
            let amount: u128 = match amount_str.parse() {
                Ok(a) => a,
                Err(_) => continue,
            };
            if !balances.contains_key(&addr_bytes) && amount > 0 {
                if let Err(e) = app_state.storage_engine.save_wallet_balance(&addr_bytes, amount).await {
                    warn!("🔄 [STATE SYNC] Failed to persist wallet balance {}: {}", addr_hex, e);
                }
                balances.insert(addr_bytes, amount);
                result.wallets_added += 1;
            }
        }
    }

    // ---- Merge token balances (add-only) ----
    {
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

            let key = (wallet_bytes, token_bytes);
            if !balances.contains_key(&key) && amount > 0 {
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
    }

    // ---- Merge symbol_to_address ----
    for (symbol, address) in &response.symbol_to_address {
        if !app_state.symbol_to_address.contains_key(symbol) {
            app_state.symbol_to_address.insert(symbol.clone(), address.clone());
            result.symbols_added += 1;
        }
    }

    result
}

// ============================================================================
// HTTP fallback sync (original implementation)
// ============================================================================

async fn do_http_state_sync(app_state: &Arc<AppState>, our_port: u16) {
    let bootstrap_peers: &[&str] = &[
        "http://185.182.185.227:8080",
        "http://109.205.176.60:8080",
        "http://161.35.219.10:8080",
    ];

    let our_ips = get_local_ips();

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
                let result = merge_http_snapshot(app_state, &snapshot).await;
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
                return;
            }
            Err(e) => {
                warn!("🔄 [STATE SYNC HTTP] Failed to fetch from {}: {}", peer_url, e);
                continue;
            }
        }
    }

    warn!("🔄 [STATE SYNC] Could not reach any peer (P2P or HTTP) for state sync");
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
        .unwrap_or_else(|_| "mainnet2026.2".to_string());
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
                    if !pools.contains_key(pool_id) {
                        if let Ok(data) = serde_json::to_vec(&peer_pool) {
                            if let Err(e) = app_state.storage_engine.save_liquidity_pool(pool_id, &data).await {
                                warn!("🔄 [STATE SYNC HTTP] Failed to persist pool {}: {}", pool_id, e);
                            }
                        }
                        pools.insert(pool_id.clone(), peer_pool);
                        result.pools_added += 1;
                    } else if snapshot.block_height > our_height {
                        if let Some(local_pool) = pools.get_mut(pool_id) {
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
                        if pool_price > 0.0 && pool_price < 1_000_000.0 {
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

    // ---- Merge wallet balances ----
    {
        let mut balances = app_state.wallet_balances.write().await;
        for (addr_hex, amount_str) in &snapshot.wallet_balances {
            let addr_bytes = match hex_to_32bytes(addr_hex) {
                Some(b) => b,
                None => continue,
            };
            let amount: u128 = match amount_str.parse() {
                Ok(a) => a,
                Err(_) => continue,
            };
            if !balances.contains_key(&addr_bytes) && amount > 0 {
                if let Err(e) = app_state.storage_engine.save_wallet_balance(&addr_bytes, amount).await {
                    warn!("🔄 [STATE SYNC HTTP] Failed to persist wallet: {}", e);
                }
                balances.insert(addr_bytes, amount);
                result.wallets_added += 1;
            }
        }
    }

    // ---- Merge token balances ----
    {
        let mut balances = app_state.token_balances.write().await;
        for (composite_key, amount_str) in &snapshot.token_balances {
            let parts: Vec<&str> = composite_key.splitn(2, '_').collect();
            if parts.len() != 2 { continue; }
            let wallet_bytes = match hex_to_32bytes(parts[0]) { Some(b) => b, None => continue };
            let token_bytes = match hex_to_32bytes(parts[1]) { Some(b) => b, None => continue };
            let amount: u128 = match amount_str.parse() { Ok(a) => a, Err(_) => continue };

            let key = (wallet_bytes, token_bytes);
            if !balances.contains_key(&key) && amount > 0 {
                if let Err(e) = app_state.storage_engine.save_token_balance(&wallet_bytes, &token_bytes, amount).await {
                    warn!("🔄 [STATE SYNC HTTP] Failed to persist token balance: {}", e);
                }
                balances.insert(key, amount);
                result.tokens_added += 1;
            }
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
