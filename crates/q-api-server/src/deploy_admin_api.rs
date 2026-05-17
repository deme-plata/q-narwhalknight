//! v5.7.0: Deploy Admin API - CCC Convergence-Aware Rolling Deployments
//!
//! Integrates the K-Kristensen Convergence Readiness framework from the
//! "Cosmic Arcology Mission" paper into the 3-server HA deployment pipeline.
//!
//! The deployment pipeline maps to Penrose's Conformal Cyclic Cosmology phases:
//!   - Isolation (k: 0.0-0.5) → Alpha canary evolving independently
//!   - Convergence (k: 0.5-0.7) → Gamma verifying, syncing with Beta
//!   - Aeon Transition (k: 0.7-0.9) → Beta deploying (conformal boundary crossing)
//!   - Harmony (k: 0.9-1.0) → All servers unified, same version, synced
//!
//! K-Kristensen formula: k = G^0.25 × Q^0.20 × T^0.20 × I^0.15 × R^0.20
//!   G = Genetic Stability (version consistency)
//!   Q = Quantum Coherence (height synchronization)
//!   T = Thermodynamic Efficiency (uptime ratio)
//!   I = Information Density (sync completeness)
//!   R = Network Resilience (peer connectivity)
//!
//! Endpoints:
//! - GET  /api/v1/admin/deploy/status       - All servers' status
//! - GET  /api/v1/admin/deploy/convergence  - CCC phase + K-parameter + readiness
//! - POST /api/v1/admin/deploy/verify       - Trigger verification on Gamma
//! - GET  /api/v1/admin/deploy/progress     - SSE stream of verification progress
//! - POST /api/v1/admin/deploy/promote      - Rolling upgrade: Gamma→Beta
//! - POST /api/v1/admin/deploy/rollback     - Rollback to previous binary

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        Json,
    },
};
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use crate::aegis_auth_middleware::FOUNDER_WALLET;
use crate::handlers::ApiResponse;
use crate::AppState;

/// Server configuration
const ALPHA_URL: &str = "http://161.35.219.10:8080";
const GAMMA_URL: &str = "http://109.205.176.60:8080";
const DELTA_URL: &str = "http://5.79.79.158:8080";
const EPSILON_URL: &str = "http://89.149.241.126:8080";
const GAMMA_IP: &str = "109.205.176.60";

/// Status of a single server node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDeployStatus {
    pub name: String,
    pub url: String,
    pub online: bool,
    pub version: String,
    pub height: u64,
    pub network_height: u64,
    pub peers: usize,
    pub uptime_secs: u64,
    pub status: String, // "ready", "syncing", "starting", "offline"
    /// v1.0.2: Detailed sync status (chunks, speed, ETA) — None if unavailable
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_details: Option<q_storage::DetailedSyncStatus>,
}

/// Combined deploy status for all servers
#[derive(Debug, Clone, Serialize)]
pub struct DeployStatus {
    pub alpha: NodeDeployStatus,
    pub beta: NodeDeployStatus,
    pub gamma: NodeDeployStatus,
    pub delta: NodeDeployStatus,
    pub epsilon: NodeDeployStatus,
    pub height_delta: i64,
    pub versions_match: bool,
}

/// Verification progress event
#[derive(Debug, Clone, Serialize)]
pub struct VerifyProgressEvent {
    pub step: String,
    pub status: String, // "running", "passed", "failed"
    pub message: String,
    pub timestamp: u64,
}

/// v1.0.2: Mining capacity metrics for a single server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningCapacityLocal {
    /// Queue used / capacity
    pub queue_used: u64,
    pub queue_capacity: u64,
    pub queue_pct: f64,
    /// Network hashrate (H/s) and active miners
    pub hashrate_hs: u64,
    pub active_miners: usize,
    /// Acceptance rate
    pub solutions_submitted: u64,
    pub solutions_accepted: u64,
    pub acceptance_pct: f64,
    /// Health
    pub is_healthy: bool,
    pub last_solution_secs_ago: u64,
    /// Shard count (informational)
    pub shard_count: usize,
}

/// v1.0.2: Aggregated mining capacity for all servers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningCapacityAll {
    pub beta: Option<MiningCapacityLocal>,
    pub gamma: Option<MiningCapacityLocal>,
    pub delta: Option<MiningCapacityLocal>,
    pub epsilon: Option<MiningCapacityLocal>,
    pub alpha: Option<MiningCapacityLocal>,
}

/// v9.0.6: Decentralization Index — composite network health metric
/// Sqrt scaling, EMA smoothing, wealth Gini, Shannon entropy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecentralizationMetrics {
    pub unique_wallets: usize,
    pub total_workers: usize,
    pub top_miner_pct: f64,
    pub top3_miners_pct: f64,
    pub nakamoto_coefficient: usize,
    pub gini_coefficient: f64,
    pub hhi: f64,
    pub node_count: usize,
    pub peer_count: usize,
    /// Wealth Gini coefficient (balance distribution, 0=equal, 1=one wallet has all)
    pub wealth_gini: f64,
    /// Shannon entropy of mining power (bits), normalized 0-100
    pub entropy_score: f64,
    /// Infrastructure nodes (team-operated bootstrap servers)
    pub infrastructure_nodes: usize,
    /// Community nodes (unique peers not in bootstrap list)
    pub community_nodes: usize,
    /// Raw DI before EMA smoothing
    pub decentralization_index_raw: f64,
    /// EMA-smoothed DI (alpha=0.1)
    pub decentralization_index: f64,
    pub grade: String,
}

/// Shared verification state for SSE streaming
pub struct DeployState {
    pub verification_running: bool,
    pub verification_events: Vec<VerifyProgressEvent>,
    pub last_result: Option<crate::upgrade_verifier::VerificationResult>,
    /// v5.6.0: Prevents concurrent pipeline executions
    pub pipeline_running: bool,
}

impl DeployState {
    pub fn new() -> Self {
        Self {
            verification_running: false,
            verification_events: Vec::new(),
            last_result: None,
            pipeline_running: false,
        }
    }
}

/// Extract wallet address from request headers (same pattern as other admin APIs)
pub fn extract_wallet_from_headers(headers: &HeaderMap) -> Option<String> {
    // Try X-Wallet-Auth header first
    if let Some(auth) = headers.get("x-wallet-auth") {
        if let Ok(auth_str) = auth.to_str() {
            // Format: "wallet_address:signature" or just "wallet_address"
            let wallet = auth_str.split(':').next().unwrap_or("");
            let clean = wallet.replace("qnk", "").replace("qug", "");
            if clean.len() == 64 {
                return Some(clean);
            }
        }
    }

    // Try Authorization header
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_str) = auth.to_str() {
            let token = auth_str.strip_prefix("Bearer ").unwrap_or(auth_str);
            let clean = token.replace("qnk", "").replace("qug", "");
            if clean.len() == 64 {
                return Some(clean);
            }
        }
    }

    None
}

/// Check if the requesting wallet is the admin wallet (configurable via --admin-wallet)
fn is_master_wallet(headers: &HeaderMap, state: &AppState) -> bool {
    match extract_wallet_from_headers(headers) {
        Some(wallet) => wallet == state.admin_wallet,
        None => false,
    }
}

/// GET /api/v1/admin/deploy/status
/// Returns status of both Beta and Gamma servers
/// v8.6.4: Made public (read-only) — all logged-in users can see node status
pub async fn deploy_status(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<DeployStatus>>, StatusCode> {
    // v8.6.4: Status is read-only, allow all authenticated users
    // Deploy/rollback actions still require master wallet

    // Get Beta (local) status directly from AppState
    let beta_height = state
        .current_height_atomic
        .load(std::sync::atomic::Ordering::Relaxed);
    let beta_network_height = state
        .highest_network_height
        .load(std::sync::atomic::Ordering::Relaxed);
    let beta_peers = state
        .libp2p_peer_count
        .as_ref()
        .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(0);
    let beta_uptime = state.start_time.elapsed().as_secs();
    let beta_version = env!("CARGO_PKG_VERSION").to_string();

    // v8.2.9: Update peak height (only increases, never decreases)
    let peak = state.peak_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
    if beta_height > peak {
        state.peak_height_atomic.store(beta_height, std::sync::atomic::Ordering::Relaxed);
    }
    // Report the max of current and peak — users never see a decrease
    let display_height = beta_height.max(peak);

    let beta_status = if beta_height == 0 {
        "starting"
    } else if beta_network_height > 0 && beta_height + 10 < beta_network_height {
        if peak > beta_height {
            "recovering" // v8.2.9: Node restarted, catching up to previous peak
        } else {
            "syncing"
        }
    } else {
        "ready"
    };

    // v1.0.2: Get detailed sync status from local TurboSync
    let beta_sync_details = if let Some(ref turbo_sync) = state.turbo_sync {
        let mut details = turbo_sync.get_detailed_sync_status().await;
        // Enrich with FlightComputer telemetry
        if let Some(ref fc) = state.flight_computer {
            if let Ok(fc_guard) = fc.try_read() {
                let pc = turbo_sync.cached_peer_count.load(std::sync::atomic::Ordering::Relaxed);
                let telem = fc_guard.telemetry(pc);
                details.starship_phase = telem.phase;
                details.phase_duration_secs = telem.phase_duration_secs;
                details.orbit_stable = telem.orbit_stable;
                details.station_keeping_peer_health = telem.peer_health;
                details.mission_elapsed_secs = telem.mission_elapsed_secs;
            }
        }
        Some(details)
    } else {
        None
    };

    let beta = NodeDeployStatus {
        name: "Server Beta".to_string(),
        url: "https://quillon.xyz".to_string(),
        online: true,
        version: beta_version.clone(),
        height: display_height, // v8.2.9: Use peak height — never show a decrease
        network_height: beta_network_height,
        peers: beta_peers,
        uptime_secs: beta_uptime,
        status: beta_status.to_string(),
        sync_details: beta_sync_details,
    };

    // Get Alpha, Gamma, Delta, and Epsilon status via HTTP (parallel)
    let (alpha, gamma, delta, epsilon) = tokio::join!(
        fetch_node_status("Server Alpha", ALPHA_URL),
        fetch_node_status("Server Gamma", GAMMA_URL),
        fetch_node_status("Server Delta", DELTA_URL),
        fetch_node_status("Server Epsilon", EPSILON_URL),
    );

    let height_delta = beta.height as i64 - gamma.height as i64;
    let versions_match = beta.version == gamma.version
        && (alpha.version == beta.version || !alpha.online);

    Ok(Json(ApiResponse::success(DeployStatus {
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        height_delta,
        versions_match,
    })))
}

/// Fetch status from a remote node (health + sync details in parallel)
async fn fetch_node_status(name: &str, base_url: &str) -> NodeDeployStatus {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    let health_url = format!("{}/api/v1/health", base_url);
    let sync_url = format!("{}/api/v1/sync/detailed", base_url);

    // Fetch health and sync details in parallel
    let (health_res, sync_res) = tokio::join!(
        client.get(&health_url).send(),
        client.get(&sync_url).send(),
    );

    // Parse sync details (best-effort — None if endpoint doesn't exist on old versions)
    let sync_details: Option<q_storage::DetailedSyncStatus> = match sync_res {
        Ok(resp) if resp.status().is_success() => {
            #[derive(Deserialize)]
            struct SyncResp {
                data: Option<q_storage::DetailedSyncStatus>,
            }
            resp.json::<SyncResp>().await.ok().and_then(|r| r.data)
        }
        _ => None,
    };

    match health_res {
        Ok(resp) if resp.status().is_success() => {
            #[derive(Deserialize)]
            struct HealthResp {
                data: Option<HealthData>,
            }
            #[derive(Deserialize)]
            struct HealthData {
                status: Option<String>,
                height: Option<u64>,
                network_height: Option<u64>,
                peers: Option<usize>,
                version: Option<String>,
                uptime_secs: Option<u64>,
            }

            match resp.json::<HealthResp>().await {
                Ok(health) => {
                    if let Some(data) = health.data {
                        NodeDeployStatus {
                            name: name.to_string(),
                            url: base_url.to_string(),
                            online: true,
                            version: data.version.unwrap_or_else(|| "unknown".to_string()),
                            height: data.height.unwrap_or(0),
                            network_height: data.network_height.unwrap_or(0),
                            peers: data.peers.unwrap_or(0),
                            uptime_secs: data.uptime_secs.unwrap_or(0),
                            status: data.status.unwrap_or_else(|| "ready".to_string()),
                            sync_details,
                        }
                    } else {
                        // Legacy health endpoint returns "OK"
                        NodeDeployStatus {
                            name: name.to_string(),
                            url: base_url.to_string(),
                            online: true,
                            version: "unknown".to_string(),
                            height: 0,
                            network_height: 0,
                            peers: 0,
                            uptime_secs: 0,
                            status: "ready".to_string(),
                            sync_details,
                        }
                    }
                }
                Err(_) => NodeDeployStatus {
                    name: name.to_string(),
                    url: base_url.to_string(),
                    online: true,
                    version: "unknown (legacy)".to_string(),
                    height: 0,
                    network_height: 0,
                    peers: 0,
                    uptime_secs: 0,
                    status: "ready".to_string(),
                    sync_details: None,
                },
            }
        }
        _ => NodeDeployStatus {
            name: name.to_string(),
            url: base_url.to_string(),
            online: false,
            version: String::new(),
            height: 0,
            network_height: 0,
            peers: 0,
            uptime_secs: 0,
            status: "offline".to_string(),
            sync_details: None,
        },
    }
}

/// POST /api/v1/admin/deploy/verify
/// Triggers verification on Gamma (runs upgrade_verifier checks against Beta as reference)
pub async fn deploy_verify(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    if !is_master_wallet(&headers, &state) {
        return Err(StatusCode::FORBIDDEN);
    }

    // Check if Gamma is reachable
    let gamma_status = fetch_node_status("Gamma", GAMMA_URL).await;
    if !gamma_status.online {
        return Ok(Json(ApiResponse::success(
            "Gamma node is offline - cannot verify".to_string(),
        )));
    }

    // Run verification checks against Gamma from Beta's perspective
    let deploy_state = state.deploy_state.clone();

    {
        let mut ds = deploy_state.write().await;
        if ds.verification_running {
            return Ok(Json(ApiResponse::success(
                "Verification already in progress".to_string(),
            )));
        }
        ds.verification_running = true;
        ds.verification_events.clear();
    }

    let storage = state.storage_engine.clone();
    let ds_clone = deploy_state.clone();

    // Run verification in background
    tokio::spawn(async move {
        let config = crate::upgrade_verifier::VerifyConfig {
            reference_url: GAMMA_URL.to_string(),
            verify_only: false,
            max_height_delta: 10,
            sync_timeout_secs: 120,
        };

        let result = crate::upgrade_verifier::run_upgrade_verification(&config, &storage, 8080).await;

        let mut ds = ds_clone.write().await;
        ds.verification_running = false;

        // Push result events
        for check in &result.checks {
            ds.verification_events.push(VerifyProgressEvent {
                step: check.name.clone(),
                status: if check.passed {
                    "passed".to_string()
                } else {
                    "failed".to_string()
                },
                message: check.message.clone(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            });
        }

        // Final result
        ds.verification_events.push(VerifyProgressEvent {
            step: "RESULT".to_string(),
            status: if result.passed {
                "passed".to_string()
            } else {
                "failed".to_string()
            },
            message: format!(
                "Verification {} in {:.1}s",
                if result.passed { "PASSED" } else { "FAILED" },
                result.total_duration_ms as f64 / 1000.0
            ),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        });

        ds.last_result = Some(result);
    });

    Ok(Json(ApiResponse::success(
        "Verification started - monitor via /api/v1/admin/deploy/progress".to_string(),
    )))
}

/// GET /api/v1/admin/deploy/progress (SSE)
/// Streams verification/pipeline progress events.
/// NOTE: Auth dropped here because EventSource cannot send custom headers.
/// The POST trigger endpoints still require auth. This only streams log lines.
pub async fn deploy_progress(
    State(state): State<Arc<AppState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let deploy_state = state.deploy_state.clone();

    let stream = async_stream::stream! {
        let mut sent_count = 0;

        loop {
            let (events, running, pipeline) = {
                let ds = deploy_state.read().await;
                (ds.verification_events.clone(), ds.verification_running, ds.pipeline_running)
            };

            // Send new events
            while sent_count < events.len() {
                let event = &events[sent_count];
                let json = serde_json::to_string(event).unwrap_or_default();
                yield Ok(Event::default()
                    .event("verify-progress")
                    .data(json));
                sent_count += 1;
            }

            // If nothing is running and all events sent, we're done
            if !running && !pipeline && sent_count >= events.len() && sent_count > 0 {
                yield Ok(Event::default()
                    .event("verify-complete")
                    .data("done"));
                break;
            }

            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Strip ANSI color codes from script output
fn strip_ansi(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_escape = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
            continue;
        }
        if in_escape {
            if c.is_ascii_alphabetic() {
                in_escape = false;
            }
            continue;
        }
        result.push(c);
    }
    result
}

/// Parse a pipeline output line into a step name
fn parse_pipeline_step(line: &str) -> &str {
    let lower = line.to_lowercase();
    if lower.contains("alpha") || lower.contains("canary") {
        "alpha-canary"
    } else if lower.contains("gamma") && (lower.contains("verify") || lower.contains("deploy") || lower.contains("scp") || lower.contains("restart")) {
        "gamma-verify"
    } else if lower.contains("soak") || lower.contains("wait") || lower.contains("health check") {
        "soak-test"
    } else if lower.contains("beta") && (lower.contains("deploy") || lower.contains("upgrade") || lower.contains("restart")) {
        "beta-deploy"
    } else if lower.contains("restore") || lower.contains("weight") || lower.contains("primary") {
        "restore"
    } else if lower.contains("rollback") {
        "rollback"
    } else {
        "pipeline"
    }
}

/// Push a progress event into deploy_state
async fn push_event(deploy_state: &Arc<RwLock<DeployState>>, step: &str, status: &str, message: String) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut ds = deploy_state.write().await;
    ds.verification_events.push(VerifyProgressEvent {
        step: step.to_string(),
        status: status.to_string(),
        message,
        timestamp: ts,
    });
}

/// v5.6.0: Spawn ha-deploy.sh (or rollback) and stream output to deploy_state
async fn spawn_deploy_script(
    deploy_state: Arc<RwLock<DeployState>>,
    command_arg: &str,
    broadcaster: Arc<crate::streaming::EventBroadcaster>,
) {
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    let is_full = command_arg == "full";
    let script_path = "/opt/orobit/shared/q-narwhalknight/scripts/ha-deploy.sh";

    info!("🚀 [DEPLOY] Spawning pipeline: ha-deploy.sh {}", command_arg);
    push_event(&deploy_state, "pipeline", "running",
        format!("Starting pipeline: ha-deploy.sh {}", command_arg)).await;

    let child = if is_full {
        // Pipe "y" to auto-confirm
        Command::new("bash")
            .arg("-c")
            .arg(format!("echo 'y' | {} {}", script_path, command_arg))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
    } else {
        Command::new(script_path)
            .arg(command_arg)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
    };

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            error!("🚨 [DEPLOY] Failed to spawn ha-deploy.sh: {}", e);
            push_event(&deploy_state, "pipeline", "failed",
                format!("Failed to spawn script: {}", e)).await;
            let mut ds = deploy_state.write().await;
            ds.pipeline_running = false;
            return;
        }
    };

    // Read stdout line by line
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let clean = strip_ansi(&line);
            if clean.trim().is_empty() {
                continue;
            }
            let step = parse_pipeline_step(&clean);
            info!("🔧 [DEPLOY] [{}] {}", step, clean);
            push_event(&deploy_state, step, "running", clean.clone()).await;
        }
    }

    // Wait for process exit
    let exit_status = child.wait().await;
    let (success, code) = match &exit_status {
        Ok(status) => (status.success(), status.code().unwrap_or(-1)),
        Err(_) => (false, -1),
    };

    if success {
        info!("✅ [DEPLOY] Pipeline completed successfully");
        push_event(&deploy_state, "RESULT", "passed",
            "Pipeline completed successfully".to_string()).await;

        // Spawn post-deploy health monitor
        if is_full {
            spawn_post_deploy_monitor(deploy_state.clone(), broadcaster.clone()).await;
        }
    } else {
        error!("🚨 [DEPLOY] Pipeline failed with exit code {}", code);
        push_event(&deploy_state, "RESULT", "failed",
            format!("Pipeline failed (exit code {})", code)).await;
    }

    let mut ds = deploy_state.write().await;
    ds.pipeline_running = false;
}

/// v5.6.0: Post-deploy health monitor — checks local /health every 30s for 5 minutes.
/// Triggers auto-rollback on height regression, stall, or consecutive failures.
async fn spawn_post_deploy_monitor(
    deploy_state: Arc<RwLock<DeployState>>,
    broadcaster: Arc<crate::streaming::EventBroadcaster>,
) {
    info!("🩺 [DEPLOY] Starting post-deploy health monitor (5 min, 30s interval)");

    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        let mut prev_height: Option<u64> = None;
        let mut last_height_change = std::time::Instant::now();
        let mut consecutive_failures: u32 = 0;
        let monitor_end = std::time::Instant::now() + std::time::Duration::from_secs(300);

        while std::time::Instant::now() < monitor_end {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;

            #[derive(serde::Deserialize)]
            struct HealthResp {
                data: Option<HealthData>,
            }
            #[derive(serde::Deserialize)]
            struct HealthData {
                height: Option<u64>,
            }

            match client.get("http://127.0.0.1:8080/api/v1/health").send().await {
                Ok(resp) if resp.status().is_success() => {
                    consecutive_failures = 0;
                    if let Ok(health) = resp.json::<HealthResp>().await {
                        let height = health.data.and_then(|d| d.height).unwrap_or(0);

                        // Check height regression
                        if let Some(prev) = prev_height {
                            if height < prev {
                                let reason = format!(
                                    "Height regression detected: {} → {} (lost {} blocks)",
                                    prev, height, prev - height
                                );
                                warn!("🚨 [DEPLOY MONITOR] {}", reason);
                                trigger_auto_rollback(&deploy_state, &broadcaster, &reason).await;
                                return;
                            }
                            if height > prev {
                                last_height_change = std::time::Instant::now();
                            }
                        } else {
                            last_height_change = std::time::Instant::now();
                        }

                        prev_height = Some(height);

                        // Check stall (no height change for 2 minutes)
                        if last_height_change.elapsed() > std::time::Duration::from_secs(120) {
                            let reason = format!(
                                "Block production stalled for 2+ minutes at height {}",
                                height
                            );
                            warn!("🚨 [DEPLOY MONITOR] {}", reason);
                            trigger_auto_rollback(&deploy_state, &broadcaster, &reason).await;
                            return;
                        }
                    }
                }
                _ => {
                    consecutive_failures += 1;
                    warn!(
                        "🚨 [DEPLOY MONITOR] Health check failed ({}/3 consecutive)",
                        consecutive_failures
                    );
                    if consecutive_failures >= 3 {
                        let reason = "3 consecutive health check failures".to_string();
                        trigger_auto_rollback(&deploy_state, &broadcaster, &reason).await;
                        return;
                    }
                }
            }
        }

        info!("🩺 [DEPLOY] Post-deploy health monitor completed — no issues detected");
        push_event(&deploy_state, "health-monitor", "passed",
            "5-minute post-deploy monitoring passed".to_string()).await;
    });
}

/// Trigger auto-rollback and notify clients
async fn trigger_auto_rollback(
    deploy_state: &Arc<RwLock<DeployState>>,
    broadcaster: &Arc<crate::streaming::EventBroadcaster>,
    reason: &str,
) {
    error!("🚨 [AUTO-ROLLBACK] Triggering rollback: {}", reason);
    push_event(deploy_state, "auto-rollback", "failed",
        format!("Auto-rollback triggered: {}", reason)).await;

    // Broadcast security alert to all clients
    let _ = broadcaster.broadcast(crate::streaming::StreamEvent::SecurityAlert {
        alert_type: "auto-rollback".to_string(),
        description: format!("Automatic rollback triggered: {}", reason),
        risk_level: 0.9,
        timestamp: chrono::Utc::now(),
    }).await;

    // Execute rollback — this will kill the current process (old binary restarts via systemctl)
    let script_path = "/opt/orobit/shared/q-narwhalknight/scripts/ha-deploy.sh";
    match tokio::process::Command::new(script_path)
        .arg("rollback")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(mut child) => {
            let _ = child.wait().await;
            info!("🔄 [AUTO-ROLLBACK] Rollback script finished");
        }
        Err(e) => {
            error!("🚨 [AUTO-ROLLBACK] Failed to spawn rollback: {}", e);
        }
    }
}

/// POST /api/v1/admin/deploy/promote
/// v5.6.0: Actually executes `echo 'y' | ./scripts/ha-deploy.sh full`
pub async fn deploy_promote(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    if !is_master_wallet(&headers, &state) {
        return Err(StatusCode::FORBIDDEN);
    }

    // Prevent concurrent pipelines
    {
        let mut ds = state.deploy_state.write().await;
        if ds.pipeline_running {
            return Ok(Json(ApiResponse::success(
                "Pipeline already running — check /api/v1/admin/deploy/progress".to_string(),
            )));
        }
        ds.pipeline_running = true;
        ds.verification_events.clear();
    }

    let deploy_state = state.deploy_state.clone();
    let broadcaster = state.event_broadcaster.clone();

    tokio::spawn(async move {
        spawn_deploy_script(deploy_state, "full", broadcaster).await;
    });

    Ok(Json(ApiResponse::success(
        "Rolling upgrade started — monitor via /api/v1/admin/deploy/progress".to_string(),
    )))
}

/// POST /api/v1/admin/deploy/rollback
/// v5.6.0: Actually executes `./scripts/ha-deploy.sh rollback`
pub async fn deploy_rollback(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    if !is_master_wallet(&headers, &state) {
        return Err(StatusCode::FORBIDDEN);
    }

    // Prevent concurrent pipelines
    {
        let mut ds = state.deploy_state.write().await;
        if ds.pipeline_running {
            return Ok(Json(ApiResponse::success(
                "Pipeline already running — cannot rollback concurrently".to_string(),
            )));
        }
        ds.pipeline_running = true;
        ds.verification_events.clear();
    }

    let deploy_state = state.deploy_state.clone();
    let broadcaster = state.event_broadcaster.clone();

    tokio::spawn(async move {
        spawn_deploy_script(deploy_state, "rollback", broadcaster).await;
    });

    Ok(Json(ApiResponse::success(
        "Rollback started — monitor via /api/v1/admin/deploy/progress".to_string(),
    )))
}

// ============================================================================
// CCC CONVERGENCE READINESS SYSTEM (Cosmic Arcology Mission v2.4.0)
// ============================================================================

/// K-Kristensen convergence readiness metrics for a single node
/// Based on the K-formula: k = G^0.25 × Q^0.20 × T^0.20 × I^0.15 × R^0.20
#[derive(Debug, Clone, Serialize)]
pub struct NodeKMetrics {
    pub name: String,
    /// G: Genetic Stability — version consistency (1.0 = matches reference)
    pub genetic_stability: f64,
    /// Q: Quantum Coherence — height sync ratio (local_height / network_height)
    pub quantum_coherence: f64,
    /// T: Thermodynamic Efficiency — uptime health (uptime / expected)
    pub thermodynamic_efficiency: f64,
    /// I: Information Density — sync completeness (height / max_height)
    pub information_density: f64,
    /// R: Network Resilience — peer connectivity (peers / expected_peers)
    pub network_resilience: f64,
    /// Computed K-Kristensen parameter
    pub k_parameter: f64,
}

/// Convergence outcome prediction (from paper Table 3)
#[derive(Debug, Clone, Serialize)]
pub enum ConvergenceOutcome {
    /// k > 0.9: Peaceful merger with synergy bonus
    Communion { synergy_bonus: f64 },
    /// 0.7 < k <= 0.9: Safe limited contact
    Observation { interaction_distance: f64 },
    /// 0.5 < k <= 0.7: Resource equilibrium
    Competition { equilibrium: String },
    /// 0.3 < k <= 0.5: Potential casualties
    Conflict { risk: f64 },
    /// k <= 0.3: Dominant entity prevails
    Absorption { dominant: String },
}

/// CCC Cosmic Phase for the deployment pipeline
#[derive(Debug, Clone, Serialize)]
pub enum DeployCosmicPhase {
    /// Universe expanding — Alpha canary evolving independently
    Isolation {
        canary_version: String,
        isolation_duration_secs: u64,
        expansion_rate: f64,
    },
    /// Universe contracting — Gamma verifying, syncing with Beta
    Convergence {
        merging_servers: Vec<String>,
        contraction_rate: f64,
        blocks_to_unity: u64,
    },
    /// Conformal boundary — Beta deploying new binary
    AeonTransition {
        entropy_state: f64,
        old_version: String,
        new_version: String,
        hawking_points: Vec<u64>, // Block heights as consensus anchors
    },
    /// All servers unified and synchronized
    Harmony {
        collective_k: f64,
        harmony_duration_secs: u64,
        version: String,
    },
}

/// Full convergence status response
#[derive(Debug, Clone, Serialize)]
pub struct ConvergenceStatus {
    /// Current cosmic phase of the deployment pipeline
    pub cosmic_phase: DeployCosmicPhase,
    /// Per-node K-Kristensen metrics
    pub nodes: Vec<NodeKMetrics>,
    /// Collective K-parameter across all online nodes
    pub collective_k: f64,
    /// Predicted convergence outcome
    pub predicted_outcome: ConvergenceOutcome,
    /// Is it safe to deploy? (based on K-threshold from paper Section 8.3)
    pub convergence_safe: bool,
    /// The Cosmic Gardener's advice
    pub gardener_wisdom: String,
    /// Phase transition ETA (seconds until next phase, if applicable)
    pub phase_transition_eta: Option<u64>,
}

/// Calculate K-Kristensen parameter from the 5 component metrics
/// Formula: k = G^0.25 × Q^0.20 × T^0.20 × I^0.15 × R^0.20
fn calculate_k_parameter(g: f64, q: f64, t: f64, i: f64, r: f64) -> f64 {
    let g = g.clamp(0.001, 1.0); // Avoid zero (log domain)
    let q = q.clamp(0.001, 1.0);
    let t = t.clamp(0.001, 1.0);
    let i = i.clamp(0.001, 1.0);
    let r = r.clamp(0.001, 1.0);

    g.powf(0.25) * q.powf(0.20) * t.powf(0.20) * i.powf(0.15) * r.powf(0.20)
}

/// Calculate node K-metrics from NodeDeployStatus
fn node_to_k_metrics(
    node: &NodeDeployStatus,
    reference_version: &str,
    max_height: u64,
    expected_peers: usize,
) -> NodeKMetrics {
    if !node.online {
        return NodeKMetrics {
            name: node.name.clone(),
            genetic_stability: 0.0,
            quantum_coherence: 0.0,
            thermodynamic_efficiency: 0.0,
            information_density: 0.0,
            network_resilience: 0.0,
            k_parameter: 0.0,
        };
    }

    // G: Genetic Stability — version match with reference
    let g = if node.version == reference_version { 1.0 } else { 0.3 };

    // Q: Quantum Coherence — how close is this node's height to network height
    let q = if node.network_height > 0 {
        let ratio = node.height as f64 / node.network_height as f64;
        ratio.min(1.0)
    } else if node.height > 0 {
        0.8 // Has blocks but no network height info
    } else {
        0.1
    };

    // T: Thermodynamic Efficiency — uptime ratio (target: 1 hour minimum for stability)
    let t = if node.uptime_secs > 3600 {
        1.0
    } else if node.uptime_secs > 300 {
        node.uptime_secs as f64 / 3600.0
    } else {
        0.1 // Just started
    };

    // I: Information Density — sync completeness relative to highest known block
    let i = if max_height > 0 {
        (node.height as f64 / max_height as f64).min(1.0)
    } else {
        0.5
    };

    // R: Network Resilience — peer count relative to expected
    let expected = expected_peers.max(1) as f64;
    let r = (node.peers as f64 / expected).min(1.0);

    let k = calculate_k_parameter(g, q, t, i, r);

    NodeKMetrics {
        name: node.name.clone(),
        genetic_stability: g,
        quantum_coherence: q,
        thermodynamic_efficiency: t,
        information_density: i,
        network_resilience: r,
        k_parameter: k,
    }
}

/// Predict convergence outcome from collective K (paper Table 3)
fn predict_outcome(k: f64) -> ConvergenceOutcome {
    if k > 0.9 {
        ConvergenceOutcome::Communion {
            synergy_bonus: (k - 0.9) * 10.0, // 0.0 to 1.0 bonus
        }
    } else if k > 0.7 {
        ConvergenceOutcome::Observation {
            interaction_distance: (1.0 - k) * 100.0,
        }
    } else if k > 0.5 {
        ConvergenceOutcome::Competition {
            equilibrium: "nash_equilibrium".to_string(),
        }
    } else if k > 0.3 {
        ConvergenceOutcome::Conflict {
            risk: 1.0 - k,
        }
    } else {
        ConvergenceOutcome::Absorption {
            dominant: "rollback_required".to_string(),
        }
    }
}

/// Determine the cosmic phase based on current deploy pipeline state
fn determine_cosmic_phase(
    status: &DeployStatus,
    pipeline_running: bool,
    verification_running: bool,
) -> DeployCosmicPhase {
    // If pipeline is actively running, we're in Aeon Transition
    if pipeline_running {
        return DeployCosmicPhase::AeonTransition {
            entropy_state: 0.5, // Mid-transition
            old_version: status.beta.version.clone(),
            new_version: if status.gamma.version != status.beta.version {
                status.gamma.version.clone()
            } else {
                "deploying...".to_string()
            },
            hawking_points: vec![status.beta.height, status.gamma.height],
        };
    }

    // If verification is running, we're in Convergence
    if verification_running {
        return DeployCosmicPhase::Convergence {
            merging_servers: vec!["Gamma".to_string(), "Beta".to_string()],
            contraction_rate: 0.7,
            blocks_to_unity: status.height_delta.unsigned_abs(),
        };
    }

    // If versions differ, Alpha is in Isolation (canary diverged)
    if !status.versions_match {
        let canary_ver = if status.alpha.online && status.alpha.version != status.beta.version {
            status.alpha.version.clone()
        } else if status.gamma.online && status.gamma.version != status.beta.version {
            status.gamma.version.clone()
        } else {
            "diverged".to_string()
        };

        return DeployCosmicPhase::Isolation {
            canary_version: canary_ver,
            isolation_duration_secs: 0, // Unknown without tracking start time
            expansion_rate: 0.01,
        };
    }

    // All versions match, heights close — Harmony!
    let collective_k = if status.beta.online && status.gamma.online {
        let height_sync = if status.beta.height > 0 && status.gamma.height > 0 {
            let min_h = status.beta.height.min(status.gamma.height) as f64;
            let max_h = status.beta.height.max(status.gamma.height) as f64;
            (min_h / max_h).min(1.0)
        } else {
            0.5
        };
        height_sync
    } else {
        0.5
    };

    let min_uptime = [status.beta.uptime_secs, status.gamma.uptime_secs]
        .into_iter()
        .chain(if status.epsilon.online { Some(status.epsilon.uptime_secs) } else { None })
        .min()
        .unwrap_or(0);

    DeployCosmicPhase::Harmony {
        collective_k,
        harmony_duration_secs: min_uptime,
        version: status.beta.version.clone(),
    }
}

/// The Cosmic Gardener's wisdom based on K-parameter (inspired by Section 10.3)
fn gardener_wisdom(k: f64, phase: &DeployCosmicPhase) -> String {
    match phase {
        DeployCosmicPhase::Isolation { .. } => {
            if k < 0.3 {
                "The seeds need more time in their garden pots. Let the canary mature before transplanting.".to_string()
            } else {
                "The canary grows strong in isolation. When ready, it will guide the flock.".to_string()
            }
        }
        DeployCosmicPhase::Convergence { .. } => {
            if k > 0.7 {
                "The roots are strong enough to intertwine. Convergence may proceed safely.".to_string()
            } else {
                "Caution: the plants are still fragile. Verify thoroughly before merging.".to_string()
            }
        }
        DeployCosmicPhase::AeonTransition { .. } => {
            "The conformal boundary is being crossed. A new aeon begins — old rules give way to new.".to_string()
        }
        DeployCosmicPhase::Harmony { collective_k, .. } => {
            if *collective_k > 0.95 {
                "Perfect harmony. The cosmic garden flourishes in unified consensus.".to_string()
            } else {
                format!("Harmony achieved (k={:.3}). The garden tends itself, but the gardener watches.", collective_k)
            }
        }
    }
}

/// GET /api/v1/admin/deploy/convergence
/// Returns CCC convergence phase, K-parameters, and readiness assessment.
/// v8.6.4: Made public (read-only) — all logged-in users can see convergence status
pub async fn deploy_convergence(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<ConvergenceStatus>>, StatusCode> {
    // v8.6.4: Convergence is read-only status, allow all users

    // Gather status from all nodes (same as deploy_status)
    let beta_height = state
        .current_height_atomic
        .load(std::sync::atomic::Ordering::Relaxed);
    let beta_network_height = state
        .highest_network_height
        .load(std::sync::atomic::Ordering::Relaxed);
    let beta_peers = state
        .libp2p_peer_count
        .as_ref()
        .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(0);
    let beta_uptime = state.start_time.elapsed().as_secs();
    let beta_version = env!("CARGO_PKG_VERSION").to_string();

    // v8.2.9: Same peak height logic as deploy_status
    let conv_peak = state.peak_height_atomic.load(std::sync::atomic::Ordering::Relaxed);
    if beta_height > conv_peak {
        state.peak_height_atomic.store(beta_height, std::sync::atomic::Ordering::Relaxed);
    }
    let conv_display_height = beta_height.max(conv_peak);

    let beta_status_str = if beta_height == 0 {
        "starting"
    } else if beta_network_height > 0 && beta_height + 10 < beta_network_height {
        if conv_peak > beta_height { "recovering" } else { "syncing" }
    } else {
        "ready"
    };

    let conv_beta_sync_details = if let Some(ref turbo_sync) = state.turbo_sync {
        let mut details = turbo_sync.get_detailed_sync_status().await;
        if let Some(ref fc) = state.flight_computer {
            if let Ok(fc_guard) = fc.try_read() {
                let pc = turbo_sync.cached_peer_count.load(std::sync::atomic::Ordering::Relaxed);
                let telem = fc_guard.telemetry(pc);
                details.starship_phase = telem.phase;
                details.phase_duration_secs = telem.phase_duration_secs;
                details.orbit_stable = telem.orbit_stable;
                details.station_keeping_peer_health = telem.peer_health;
                details.mission_elapsed_secs = telem.mission_elapsed_secs;
            }
        }
        Some(details)
    } else {
        None
    };

    let beta = NodeDeployStatus {
        name: "Server Beta".to_string(),
        url: "https://quillon.xyz".to_string(),
        online: true,
        version: beta_version.clone(),
        height: conv_display_height, // v8.2.9: Use peak height — never show a decrease
        network_height: beta_network_height,
        peers: beta_peers,
        uptime_secs: beta_uptime,
        status: beta_status_str.to_string(),
        sync_details: conv_beta_sync_details,
    };

    let (alpha, gamma, delta, epsilon) = tokio::join!(
        fetch_node_status("Server Alpha", ALPHA_URL),
        fetch_node_status("Server Gamma", GAMMA_URL),
        fetch_node_status("Server Delta", DELTA_URL),
        fetch_node_status("Server Epsilon", EPSILON_URL),
    );

    let height_delta = beta.height as i64 - gamma.height as i64;
    let versions_match = beta.version == gamma.version
        && (alpha.version == beta.version || !alpha.online);

    let deploy_status = DeployStatus {
        alpha: alpha.clone(),
        beta: beta.clone(),
        gamma: gamma.clone(),
        delta: delta.clone(),
        epsilon: epsilon.clone(),
        height_delta,
        versions_match,
    };

    // Calculate K-Kristensen for each node
    let max_height = [beta.height, gamma.height, alpha.height, delta.height, epsilon.height]
        .iter()
        .copied()
        .max()
        .unwrap_or(1);
    let expected_peers = 5usize; // 5-server network

    let k_alpha = node_to_k_metrics(&alpha, &beta_version, max_height, expected_peers);
    let k_beta = node_to_k_metrics(&beta, &beta_version, max_height, expected_peers);
    let k_gamma = node_to_k_metrics(&gamma, &beta_version, max_height, expected_peers);
    let k_delta = node_to_k_metrics(&delta, &beta_version, max_height, expected_peers);
    let k_epsilon = node_to_k_metrics(&epsilon, &beta_version, max_height, expected_peers);

    // Collective K: average of online nodes (paper Section 4.1)
    let online_nodes: Vec<&NodeKMetrics> = [&k_alpha, &k_beta, &k_gamma, &k_delta, &k_epsilon]
        .iter()
        .filter(|n| n.k_parameter > 0.0)
        .copied()
        .collect();

    let collective_k = if online_nodes.is_empty() {
        0.0
    } else {
        online_nodes.iter().map(|n| n.k_parameter).sum::<f64>() / online_nodes.len() as f64
    };

    // Check pipeline state
    let (pipeline_running, verification_running) = {
        let ds = state.deploy_state.read().await;
        (ds.pipeline_running, ds.verification_running)
    };

    // Determine cosmic phase
    let cosmic_phase = determine_cosmic_phase(&deploy_status, pipeline_running, verification_running);

    // Predict outcome
    let predicted_outcome = predict_outcome(collective_k);

    // Convergence safety check (paper Section 8.3)
    // Safe if min(local_k, remote_k) predicts Communion or Observation with k >= threshold
    let convergence_safe = collective_k > 0.7;

    // Phase transition ETA
    let phase_transition_eta = match &cosmic_phase {
        DeployCosmicPhase::Isolation { .. } => Some(300), // ~5 min to verify
        DeployCosmicPhase::Convergence { blocks_to_unity, .. } => {
            Some(blocks_to_unity * 3) // ~3s per block sync
        }
        DeployCosmicPhase::AeonTransition { .. } => Some(60), // ~1 min deploy
        DeployCosmicPhase::Harmony { .. } => None, // Stable
    };

    let wisdom = gardener_wisdom(collective_k, &cosmic_phase);

    Ok(Json(ApiResponse::success(ConvergenceStatus {
        cosmic_phase,
        nodes: vec![k_alpha, k_beta, k_gamma, k_delta, k_epsilon],
        collective_k,
        predicted_outcome,
        convergence_safe,
        gardener_wisdom: wisdom,
        phase_transition_eta,
    })))
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEV FEE ADMIN API (v7.1.5)
// ═══════════════════════════════════════════════════════════════════════════════

/// Dev fee status response
#[derive(Debug, Clone, Serialize)]
pub struct DevFeeStatus {
    /// Current dev fee in basis points (100 = 1%)
    pub fee_bps: u64,
    /// Fee as percentage string
    pub fee_percent: String,
    /// Founder wallet address
    pub founder_wallet: String,
    /// Founder wallet QUG balance (display units)
    pub founder_balance_qug: f64,
    /// Total dev fees collected this session (from consensus stats)
    pub total_dev_fees_collected: f64,
    /// Total mining rewards this session
    pub total_mining_rewards: f64,
    /// Actual fee ratio (dev_fees / total_rewards) — for verification
    pub actual_fee_ratio: f64,
    /// Expected fee ratio based on current bps setting
    pub expected_fee_ratio: f64,
    /// Whether actual matches expected (within 0.1% tolerance)
    pub fee_verified: bool,
    /// Blocks processed
    pub blocks_processed: u64,
    /// Today's dev fee in QUG
    pub today_dev_fee_qug: f64,
    /// Today's expected dev fee based on emission target
    pub today_expected_dev_fee_qug: f64,
}

/// Dev fee config update request
#[derive(Debug, Clone, Deserialize)]
pub struct DevFeeConfigRequest {
    /// New fee in basis points (0-1000, i.e. 0%-10%)
    pub fee_bps: u64,
}

/// GET /api/v1/admin/dev-fee
/// Returns dev fee verification: actual collected vs expected, balance, health
pub async fn admin_dev_fee_status(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<DevFeeStatus>>, StatusCode> {
    if !is_master_wallet(&headers, &state) {
        return Err(StatusCode::FORBIDDEN);
    }

    let fee_bps = state.dev_fee_bps.load(std::sync::atomic::Ordering::Relaxed);

    // Get founder wallet balance
    const FOUNDER_HEX: &str = "efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";
    let founder_balance = {
        let balances = state.wallet_balances.read().await;
        let mut addr = [0u8; 32];
        if let Ok(bytes) = hex::decode(FOUNDER_HEX) {
            if bytes.len() == 32 {
                addr.copy_from_slice(&bytes);
            }
        }
        balances.get(&addr).copied().unwrap_or(0) as f64 / 1_000_000_000_000_000_000_000_000.0
    };

    // Get consensus stats
    let stats = state.balance_consensus_engine.get_stats().await;
    let total_dev_fees = stats.total_dev_fees as f64 / 1_000_000_000_000_000_000_000_000.0;
    let total_rewards = stats.total_rewards as f64 / 1_000_000_000_000_000_000_000_000.0;

    // Calculate actual vs expected ratio
    let actual_ratio = if total_rewards + total_dev_fees > 0.0 {
        total_dev_fees / (total_rewards + total_dev_fees)
    } else {
        0.0
    };
    let expected_ratio = fee_bps as f64 / 10_000.0;
    let fee_verified = (actual_ratio - expected_ratio).abs() < 0.001; // 0.1% tolerance

    // Today's emission target → expected dev fee
    let (today_dev_fee, today_expected) = match state.balance_consensus_engine.get_emission_summary().await {
        Ok(emission_summary) => {
            let today_emitted = emission_summary.today_emitted as f64 / 1_000_000_000_000_000_000_000_000.0;
            let today_exp = emission_summary.daily_target as f64 / 1_000_000_000_000_000_000_000_000.0 * expected_ratio;
            (today_emitted * expected_ratio, today_exp)
        }
        Err(_) => (0.0, 0.0),
    };

    Ok(Json(ApiResponse::success(DevFeeStatus {
        fee_bps,
        fee_percent: format!("{:.2}%", fee_bps as f64 / 100.0),
        founder_wallet: format!("qnk{}", FOUNDER_HEX),
        founder_balance_qug: founder_balance,
        total_dev_fees_collected: total_dev_fees,
        total_mining_rewards: total_rewards,
        actual_fee_ratio: actual_ratio,
        expected_fee_ratio: expected_ratio,
        fee_verified,
        blocks_processed: stats.blocks_processed,
        today_dev_fee_qug: today_dev_fee,
        today_expected_dev_fee_qug: today_expected,
    })))
}

/// POST /api/v1/admin/dev-fee/config
/// Update the dev fee percentage (master wallet only)
pub async fn admin_dev_fee_config(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
    Json(req): Json<DevFeeConfigRequest>,
) -> Result<Json<ApiResponse<DevFeeStatus>>, StatusCode> {
    if !is_master_wallet(&headers, &state) {
        return Err(StatusCode::FORBIDDEN);
    }

    // Validate: 0-1000 bps (0% to 10% max)
    if req.fee_bps > 1000 {
        return Ok(Json(ApiResponse::error(
            "Dev fee must be 0-1000 basis points (0%-10%)".to_string(),
        )));
    }

    let old_bps = state.dev_fee_bps.swap(req.fee_bps, std::sync::atomic::Ordering::SeqCst);
    info!(
        "💰 [ADMIN] Dev fee updated: {} bps ({:.2}%) → {} bps ({:.2}%)",
        old_bps, old_bps as f64 / 100.0,
        req.fee_bps, req.fee_bps as f64 / 100.0
    );

    // Return updated status
    admin_dev_fee_status(headers, State(state)).await
}

// ========================================
// 🔄 v8.5.0: UPDATE ANNOUNCEMENT API
// ========================================

/// Request body for POST /api/v1/admin/update/announce
#[derive(Debug, Deserialize)]
pub struct AnnounceUpdateRequest {
    pub version: String,
    pub sha256_checksum: String,
    pub blake3_checksum: String,
    pub binary_size: u64,
    pub download_url: String,
    #[serde(default)]
    pub mandatory: bool,
    #[serde(default)]
    pub release_notes: String,
}

/// Response for POST /api/v1/admin/update/announce
#[derive(Debug, Serialize)]
pub struct AnnounceUpdateResponse {
    pub success: bool,
    pub message: String,
    pub version: String,
    pub signer_pubkey: String,
    pub topic: String,
}

/// POST /api/v1/admin/update/announce
/// Signs and broadcasts an update announcement to the P2P gossipsub network.
/// Localhost-only (defense-in-depth alongside iptables).
/// Called by ha-deploy.sh after successful rolling deployment.
pub async fn admin_announce_update(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnnounceUpdateRequest>,
) -> Result<Json<AnnounceUpdateResponse>, StatusCode> {
    // Localhost-only restriction
    let ip = addr.ip();
    let is_localhost = ip.is_loopback()
        || ip == std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1))
        || ip == std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST);

    if !is_localhost {
        warn!(
            "🔄 [AUTO-UPDATE] Rejecting announce request from non-localhost: {}",
            addr
        );
        return Err(StatusCode::FORBIDDEN);
    }

    // Get peer ID
    let peer_id = {
        let info = state.libp2p_peer_info.read().await;
        info.0.clone()
    };

    // Determine network ID
    let network_id_str = std::env::var("Q_NETWORK_ID")
        .unwrap_or_else(|_| "mainnet-genesis".to_string());

    // Create announcement
    let release_notes = req.release_notes.clone();
    let mut announcement = q_types::update_announcement::UpdateAnnouncement::new(
        req.version.clone(),
        req.sha256_checksum.clone(),
        req.blake3_checksum.clone(),
        req.binary_size,
        req.download_url.clone(),
        network_id_str.clone(),
        peer_id,
        req.mandatory,
        req.release_notes,
    );

    // Sign with node signing key
    announcement.sign(&state.node_signing_key);

    let signer_pubkey = announcement.signer_pubkey.clone();
    let topic = network_id_str.parse::<q_types::NetworkId>()
        .map(|nid| nid.update_announcements_topic())
        .unwrap_or_else(|_| format!("/qnk/{}/update-announcements", network_id_str));

    // Publish via gossipsub
    if let Some(ref cmd_tx) = state.libp2p_command_tx {
        match serde_json::to_vec(&announcement) {
            Ok(data) => {
                let data_len = data.len();
                let _ = cmd_tx.send(q_network::NetworkCommand::PublishMessage {
                    topic: topic.clone(),
                    data,
                });
                info!(
                    "🔄 [AUTO-UPDATE] Update announcement published: v{} ({} bytes) to {}",
                    req.version, data_len, topic
                );
            }
            Err(e) => {
                error!("🔄 [AUTO-UPDATE] Failed to serialize announcement: {}", e);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
        }
    } else {
        warn!("🔄 [AUTO-UPDATE] No gossipsub command channel — announcement not broadcast");
    }

    // Send email notification to admin if configured
    send_update_notification_email(
        &state,
        &req.version,
        &req.download_url,
        &release_notes,
        req.mandatory,
    ).await;

    Ok(Json(AnnounceUpdateResponse {
        success: true,
        message: format!("Update v{} announced to P2P network", req.version),
        version: req.version,
        signer_pubkey,
        topic,
    }))
}

/// GET /api/v1/admin/update/status
/// Returns the current auto-update state
pub async fn admin_update_status(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let update_state = state
        .auto_update_state
        .as_ref()
        .map(|rx| rx.borrow().clone());

    let notification_email = state.admin_notification_email.read().await.clone();

    Json(serde_json::json!({
        "auto_update_enabled": state.auto_update_enabled.load(std::sync::atomic::Ordering::Relaxed),
        "current_version": env!("CARGO_PKG_VERSION"),
        "state": update_state,
        "notification_email": notification_email,
    }))
}

/// POST /api/v1/admin/update/toggle
/// Toggles auto-update on/off at runtime. Admin-only (requires admin wallet auth).
pub async fn admin_update_toggle(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: Option<Json<serde_json::Value>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Admin auth check
    let wallet = headers
        .get("x-wallet-auth")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let clean_wallet = wallet.replace("qnk", "").replace("qug", "");
    if clean_wallet != state.admin_wallet && clean_wallet != crate::aegis_auth_middleware::FOUNDER_WALLET {
        return Err(StatusCode::FORBIDDEN);
    }

    // If body has explicit "enabled" field, use it; otherwise toggle
    let new_value = if let Some(Json(body)) = body {
        if let Some(enabled) = body.get("enabled").and_then(|v| v.as_bool()) {
            enabled
        } else {
            !state.auto_update_enabled.load(std::sync::atomic::Ordering::Relaxed)
        }
    } else {
        !state.auto_update_enabled.load(std::sync::atomic::Ordering::Relaxed)
    };

    state.auto_update_enabled.store(new_value, std::sync::atomic::Ordering::Relaxed);

    info!(
        "🔄 [AUTO-UPDATE] Auto-update {} by admin {}",
        if new_value { "ENABLED" } else { "DISABLED" },
        &clean_wallet[..16.min(clean_wallet.len())]
    );

    Ok(Json(serde_json::json!({
        "auto_update_enabled": new_value,
        "message": format!("Auto-update {}", if new_value { "enabled" } else { "disabled" }),
    })))
}

// ============================================================
// v8.5.1: Admin notification email for node updates
// ============================================================

/// GET /api/v1/admin/update/notification-email
/// Returns the current notification email setting
pub async fn admin_get_notification_email(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Admin auth check
    let wallet = headers
        .get("x-wallet-auth")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let clean_wallet = wallet.replace("qnk", "").replace("qug", "");
    if clean_wallet != state.admin_wallet && clean_wallet != FOUNDER_WALLET {
        return Err(StatusCode::FORBIDDEN);
    }

    let email = state.admin_notification_email.read().await.clone();

    Ok(Json(serde_json::json!({
        "notification_email": email,
        "enabled": email.is_some(),
    })))
}

/// POST /api/v1/admin/update/notification-email
/// Set or clear the admin notification email address.
/// Body: { "email": "admin@example.com" } to set, or { "email": null } to clear.
pub async fn admin_set_notification_email(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Admin auth check
    let wallet = headers
        .get("x-wallet-auth")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let clean_wallet = wallet.replace("qnk", "").replace("qug", "");
    if clean_wallet != state.admin_wallet && clean_wallet != FOUNDER_WALLET {
        return Err(StatusCode::FORBIDDEN);
    }

    let email = body.get("email").and_then(|v| v.as_str()).map(|s| s.to_string());

    // Basic email validation if provided
    if let Some(ref e) = email {
        if !e.contains('@') || !e.contains('.') || e.len() < 5 {
            return Ok(Json(serde_json::json!({
                "success": false,
                "message": "Invalid email address format",
            })));
        }
    }

    let was_set = email.is_some();
    *state.admin_notification_email.write().await = email.clone();

    info!(
        "🔄 [AUTO-UPDATE] Notification email {}: {}",
        if was_set { "set" } else { "cleared" },
        email.as_deref().unwrap_or("(none)")
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "notification_email": email,
        "enabled": was_set,
        "message": if was_set {
            format!("Update notifications will be sent to {}", email.unwrap_or_default())
        } else {
            "Update email notifications disabled".to_string()
        },
    })))
}

/// Send an update notification email to the admin (called from auto-updater or announce handler).
/// Uses the existing MTA outbound queue — delivery happens automatically every 30 seconds.
pub async fn send_update_notification_email(
    state: &AppState,
    version: &str,
    download_url: &str,
    release_notes: &str,
    mandatory: bool,
) {
    let email_addr = {
        let guard = state.admin_notification_email.read().await;
        match guard.clone() {
            Some(e) => e,
            None => return, // No notification email configured
        }
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let urgency = if mandatory { "MANDATORY SECURITY UPDATE" } else { "Software Update Available" };

    let subject = format!("[Q-NarwhalKnight] {} — v{}", urgency, version);

    let body = format!(
        "{}\n\nA new node version v{} is available for your Q-NarwhalKnight node.\n\n\
         Download: {}\n\n\
         {}\
         Release Notes:\n{}\n\n\
         ---\n\
         This is an automated notification from your node at {}.\n\
         To stop receiving these emails, disable notifications in Node Settings > Updates.\n\
         \n\
         — system@quillon.xyz",
        urgency,
        version,
        download_url,
        if mandatory { "This is a mandatory security update. Please update as soon as possible.\n\n" } else { "" },
        if release_notes.is_empty() { "(no release notes provided)" } else { release_notes },
        env!("CARGO_PKG_VERSION"),
    );

    let body_html = Some(format!(
        "<div style=\"font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;\">\
         <div style=\"background: linear-gradient(135deg, #0f172a, #1e293b); border-radius: 12px; padding: 24px; color: #e2e8f0;\">\
         <h2 style=\"color: {}; margin-top: 0;\">{}</h2>\
         <p>A new node version <strong>v{}</strong> is available for your Q-NarwhalKnight node.</p>\
         <a href=\"{}\" style=\"display: inline-block; margin: 16px 0; padding: 12px 24px; \
         background: linear-gradient(135deg, #3b82f6, #06b6d4); color: white; text-decoration: none; \
         border-radius: 8px; font-weight: 600;\">Download v{}</a>\
         {}\
         <div style=\"margin-top: 16px; padding-top: 16px; border-top: 1px solid #334155;\">\
         <p style=\"color: #94a3b8; font-size: 14px;\"><strong>Release Notes:</strong></p>\
         <pre style=\"background: #0f172a; padding: 12px; border-radius: 8px; color: #94a3b8; \
         font-size: 13px; white-space: pre-wrap;\">{}</pre>\
         </div>\
         <p style=\"color: #64748b; font-size: 12px; margin-top: 16px;\">\
         Sent from your node running v{}. \
         Disable in Node Settings &gt; Updates.</p>\
         </div></div>",
        if mandatory { "#ef4444" } else { "#3b82f6" },
        urgency,
        version,
        download_url,
        version,
        if mandatory { "<p style=\"color: #fbbf24; font-weight: 600;\">This is a mandatory security update. Please update as soon as possible.</p>" } else { "" },
        if release_notes.is_empty() { "(no release notes provided)" } else { release_notes },
        env!("CARGO_PKG_VERSION"),
    ));

    let outbound = q_types::OutboundEmail {
        id: format!("update-notify-{}-{}", version, now),
        from_wallet: [0u8; 32], // System wallet
        from_email: "system@quillon.xyz".to_string(),
        to_email: email_addr.clone(),
        subject,
        body,
        body_html,
        timestamp: now,
        status: q_types::OutboundStatus::Pending,
        retry_count: 0,
        last_error: None,
        next_retry_at: None,
        email_id: None,
    };

    match state.storage_engine.save_outbound_email(&outbound).await {
        Ok(_) => {
            info!(
                "🔄 [AUTO-UPDATE] Update notification email queued for {} (v{})",
                email_addr, version
            );
        }
        Err(e) => {
            error!(
                "🔄 [AUTO-UPDATE] Failed to queue notification email: {}",
                e
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// v1.0.2: Mining Capacity Metrics — real-time queue/hashrate/acceptance stats
// ═══════════════════════════════════════════════════════════════════════════════

/// GET /api/v1/mining/capacity-local
/// No auth required — returns aggregate mining stats for this node only.
/// All data from in-memory atomics (<1ms response).
pub async fn mining_capacity_local(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<MiningCapacityLocal>> {
    use std::sync::atomic::Ordering;

    // --- Queue health ---
    let (queue_used, queue_capacity, shard_count) = if let Some(ref txs) = state.mining_submission_txs {
        let count = txs.len();
        // tokio mpsc Sender doesn't expose len(), so we estimate from capacity.
        // We stored capacity at init time; measure used via capacity() - max_capacity() delta.
        // Actually, Sender has no len(). We can only report capacity.
        // The best proxy: track submitted - accepted (in-flight approximation).
        let submitted = state.mining_solutions_submitted.load(Ordering::Relaxed);
        let accepted = state.mining_solutions_accepted.load(Ordering::Relaxed);
        // In-flight = submitted but not yet accepted/rejected
        let in_flight = submitted.saturating_sub(accepted);
        // Capacity: reconstruct from the same formula used in main.rs
        let cpus = num_cpus::get().min(64).max(4);
        let total_cap = if cpus > 16 {
            1_000_000 + (cpus - 16) * 50_000
        } else {
            1_000_000
        };
        (in_flight, total_cap as u64, count)
    } else {
        (0u64, 0u64, 0)
    };
    let queue_pct = if queue_capacity > 0 {
        (queue_used as f64 / queue_capacity as f64 * 100.0).min(100.0)
    } else {
        0.0
    };

    // --- Hashrate ---
    let (hashrate_hs, active_miners) = if let Some(ref mining_stats_arc) = state.mining_statistics {
        if let Ok(mut stats) = mining_stats_arc.try_write() {
            let hr = stats.calculate_network_hashrate() as u64;
            let mc = stats.active_miner_count();
            (hr, mc)
        } else {
            (0, 0)
        }
    } else {
        (0, 0)
    };

    // --- Acceptance rate ---
    let solutions_submitted = state.mining_solutions_submitted.load(Ordering::Relaxed);
    let solutions_accepted = state.mining_solutions_accepted.load(Ordering::Relaxed);
    let acceptance_pct = if solutions_submitted > 0 {
        (solutions_accepted as f64 / solutions_submitted as f64 * 100.0).min(100.0)
    } else {
        100.0 // No submissions yet = healthy
    };

    // --- Health ---
    let is_healthy = state.mining_is_healthy.load(Ordering::Relaxed);
    let last_solution_time = state.last_mining_solution_time.load(Ordering::Relaxed);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let last_solution_secs_ago = if last_solution_time > 0 {
        now.saturating_sub(last_solution_time)
    } else {
        u64::MAX // Never received a solution
    };

    Json(ApiResponse::success(MiningCapacityLocal {
        queue_used,
        queue_capacity,
        queue_pct,
        hashrate_hs,
        active_miners,
        solutions_submitted,
        solutions_accepted,
        acceptance_pct,
        is_healthy,
        last_solution_secs_ago,
        shard_count,
    }))
}

/// GET /api/v1/admin/mining/capacity
/// Admin-only — fetches local + all 4 remote servers in parallel (3s timeout).
pub async fn mining_capacity(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<MiningCapacityAll>>, StatusCode> {
    // Admin auth check
    if !is_master_wallet(&headers, &state) {
        return Err(StatusCode::FORBIDDEN);
    }

    // Get local (Beta) capacity
    let beta_local = {
        let Json(resp) = mining_capacity_local(State(state.clone())).await;
        resp.data
    };

    // Fetch remote servers in parallel with 3s timeout
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap_or_default();

    let fetch_remote = |url: &'static str| {
        let c = client.clone();
        async move {
            match c.get(format!("{}/api/v1/mining/capacity-local", url)).send().await {
                Ok(resp) if resp.status().is_success() => {
                    #[derive(Deserialize)]
                    struct Wrap { data: Option<MiningCapacityLocal> }
                    resp.json::<Wrap>().await.ok().and_then(|w| w.data)
                }
                _ => None,
            }
        }
    };

    let (gamma, delta, epsilon, alpha) = tokio::join!(
        fetch_remote(GAMMA_URL),
        fetch_remote(DELTA_URL),
        fetch_remote(EPSILON_URL),
        fetch_remote(ALPHA_URL),
    );

    Ok(Json(ApiResponse::success(MiningCapacityAll {
        beta: beta_local,
        gamma,
        delta,
        epsilon,
        alpha,
    })))
}

// =============================================================================
// v9.0.6: Caddy Reverse Proxy Metrics API (replaces nginx stats)
// =============================================================================

/// Per-status-code request counts parsed from Caddy Prometheus metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CaddyRequestsByStatus {
    pub ok_2xx: u64,
    pub redirect_3xx: u64,
    pub client_err_4xx: u64,
    pub server_err_5xx: u64,
    pub websocket_101: u64,
}

/// Caddy upstream health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaddyUpstream {
    pub address: String,
    pub healthy: bool,
}

/// Caddy metrics for a single server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaddyStats {
    /// Total requests handled since Caddy (re)start
    pub total_requests: u64,
    /// Requests broken down by status category
    pub requests_by_status: CaddyRequestsByStatus,
    /// Computed requests/sec (delta between polls)
    pub requests_per_second: f64,
    /// Average response time in ms (from histogram sum/count)
    pub avg_response_ms: f64,
    /// p99 response time estimate in ms (from histogram buckets)
    pub p99_response_ms: f64,
    /// Current Go goroutines (proxy for active connections)
    pub goroutines: u64,
    /// Caddy process heap memory in MB
    pub memory_mb: f64,
    /// Reverse proxy upstream health
    pub upstreams: Vec<CaddyUpstream>,
    /// Server hostname
    pub server_name: String,
    /// Caddy last reload timestamp
    pub last_reload: f64,
    /// Whether Caddy metrics endpoint is reachable
    pub online: bool,
}

/// All-servers Caddy stats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaddyStatsAll {
    pub epsilon: Option<CaddyStats>,
}

/// Parse Caddy Prometheus metrics text into CaddyStats
fn parse_caddy_metrics(text: &str) -> CaddyStats {
    let mut total_requests: u64 = 0;
    let mut by_status = CaddyRequestsByStatus::default();
    let mut duration_sum: f64 = 0.0;
    let mut duration_count: u64 = 0;
    let mut goroutines: u64 = 0;
    let mut heap_bytes: f64 = 0.0;
    let mut upstreams: Vec<CaddyUpstream> = Vec::new();
    let mut last_reload: f64 = 0.0;

    // Histogram buckets for p99 estimation
    let mut buckets: Vec<(f64, u64)> = Vec::new();
    let mut bucket_total_count: u64 = 0;

    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() { continue; }

        // caddy_http_request_duration_seconds_count{code="200",...} 1234
        if line.starts_with("caddy_http_request_duration_seconds_count{") {
            if let Some(count) = extract_metric_value(line) {
                let c = count as u64;
                total_requests += c;

                if let Some(code) = extract_label(line, "code") {
                    match code.chars().next() {
                        Some('2') => by_status.ok_2xx += c,
                        Some('3') => by_status.redirect_3xx += c,
                        Some('4') => by_status.client_err_4xx += c,
                        Some('5') => by_status.server_err_5xx += c,
                        _ => {
                            if code == "101" { by_status.websocket_101 += c; }
                        }
                    }
                }
            }
        }

        // caddy_http_request_duration_seconds_sum{...} 123.456
        if line.starts_with("caddy_http_request_duration_seconds_sum{") {
            if let Some(val) = extract_metric_value(line) {
                duration_sum += val;
            }
        }
        if line.starts_with("caddy_http_request_duration_seconds_count{") {
            if let Some(val) = extract_metric_value(line) {
                duration_count += val as u64;
            }
        }

        // Histogram buckets for p99 — only from subroute handler (avoids double-counting)
        if line.starts_with("caddy_http_request_duration_seconds_bucket{") {
            if line.contains("handler=\"subroute\"") && line.contains("server=\"srv0\"") {
                if let (Some(le), Some(count)) = (extract_label(line, "le"), extract_metric_value(line)) {
                    if let Ok(le_val) = le.parse::<f64>() {
                        buckets.push((le_val, count as u64));
                        if le_val == f64::INFINITY || le == "+Inf" {
                            bucket_total_count = count as u64;
                        }
                    }
                }
            }
        }

        // go_goroutines 123
        if line.starts_with("go_goroutines ") {
            if let Some(val) = line.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()) {
                goroutines = val;
            }
        }

        // go_memstats_heap_inuse_bytes 12345678
        if line.starts_with("go_memstats_heap_inuse_bytes ") {
            if let Some(val) = line.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()) {
                heap_bytes = val;
            }
        }

        // caddy_reverse_proxy_upstreams_healthy{upstream="localhost:8080"} 1
        if line.starts_with("caddy_reverse_proxy_upstreams_healthy{") {
            if let (Some(addr), Some(val)) = (extract_label(line, "upstream"), extract_metric_value(line)) {
                upstreams.push(CaddyUpstream {
                    address: addr.to_string(),
                    healthy: val >= 1.0,
                });
            }
        }

        // caddy_config_last_reload_success_timestamp_seconds 1.772e+09
        if line.starts_with("caddy_config_last_reload_success_timestamp_seconds ") {
            if let Some(val) = line.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()) {
                last_reload = val;
            }
        }
    }

    // Compute average response time
    let avg_response_ms = if duration_count > 0 {
        (duration_sum / duration_count as f64) * 1000.0
    } else { 0.0 };

    // Estimate p99 from histogram buckets
    let p99_response_ms = estimate_percentile(&buckets, bucket_total_count, 0.99) * 1000.0;

    // Deduplicate upstreams (may appear multiple times in metrics)
    upstreams.sort_by(|a, b| a.address.cmp(&b.address));
    upstreams.dedup_by(|a, b| {
        if a.address == b.address {
            b.healthy = b.healthy || a.healthy; // healthy if ANY says healthy
            true
        } else { false }
    });

    // We count _count lines twice (once for total, once for duration_count), fix by halving
    // Actually the first loop counts total_requests, the second just accumulates
    // duration_count. They parse the same lines so total_requests == duration_count. That's fine.

    CaddyStats {
        total_requests,
        requests_by_status: by_status,
        requests_per_second: compute_caddy_rps(total_requests),
        avg_response_ms,
        p99_response_ms,
        goroutines,
        memory_mb: heap_bytes / (1024.0 * 1024.0),
        upstreams,
        server_name: String::new(), // filled by caller
        last_reload,
        online: true,
    }
}

/// Extract a Prometheus metric value (the number after the last space/brace)
fn extract_metric_value(line: &str) -> Option<f64> {
    line.split_whitespace().last().and_then(|v| v.parse::<f64>().ok())
}

/// Extract a label value from a Prometheus metric line: `label="value"`
fn extract_label<'a>(line: &'a str, label: &str) -> Option<&'a str> {
    let pattern = format!("{}=\"", label);
    if let Some(start) = line.find(&pattern) {
        let start = start + pattern.len();
        if let Some(end) = line[start..].find('"') {
            return Some(&line[start..start + end]);
        }
    }
    None
}

/// Estimate a percentile from histogram buckets using linear interpolation
fn estimate_percentile(buckets: &[(f64, u64)], total: u64, percentile: f64) -> f64 {
    if total == 0 || buckets.is_empty() { return 0.0; }
    let target = (total as f64 * percentile) as u64;
    let mut sorted: Vec<(f64, u64)> = buckets.iter().copied()
        .filter(|(le, _)| le.is_finite())
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for &(le, count) in &sorted {
        if count >= target {
            return le;
        }
    }
    sorted.last().map(|(le, _)| *le).unwrap_or(0.0)
}

/// Compute requests/sec from two snapshots (delta-based, like the old nginx version)
fn compute_caddy_rps(current_requests: u64) -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static PREV_REQUESTS: AtomicU64 = AtomicU64::new(0);
    static PREV_TIME_MS: AtomicU64 = AtomicU64::new(0);

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let prev_req = PREV_REQUESTS.swap(current_requests, Ordering::Relaxed);
    let prev_time = PREV_TIME_MS.swap(now_ms, Ordering::Relaxed);

    if prev_time == 0 || now_ms <= prev_time { return 0.0; }

    let dt_secs = (now_ms - prev_time) as f64 / 1000.0;
    let dreqs = current_requests.saturating_sub(prev_req) as f64;
    dreqs / dt_secs
}

/// GET /api/v1/admin/caddy/stats-local — local Caddy metrics (no auth, for cross-server aggregation)
pub async fn caddy_stats_local() -> Json<ApiResponse<CaddyStats>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();

    let stats = match client.get("http://127.0.0.1:2019/metrics").send().await {
        Ok(resp) if resp.status().is_success() => {
            let text = resp.text().await.unwrap_or_default();
            let mut s = parse_caddy_metrics(&text);
            s.server_name = std::env::var("HOSTNAME").unwrap_or_else(|_| {
                std::fs::read_to_string("/etc/hostname")
                    .unwrap_or_else(|_| "unknown".to_string())
                    .trim().to_string()
            });
            s
        }
        _ => CaddyStats {
            total_requests: 0,
            requests_by_status: CaddyRequestsByStatus::default(),
            requests_per_second: 0.0,
            avg_response_ms: 0.0,
            p99_response_ms: 0.0,
            goroutines: 0,
            memory_mb: 0.0,
            upstreams: vec![],
            server_name: "unknown".into(),
            last_reload: 0.0,
            online: false,
        },
    };

    Json(ApiResponse::success(stats))
}

/// GET /api/v1/admin/caddy/stats — Caddy metrics from Epsilon (admin auth)
pub async fn caddy_stats(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<CaddyStatsAll>>, StatusCode> {
    if !is_master_wallet(&headers, &state) {
        return Err(StatusCode::FORBIDDEN);
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap_or_default();

    // Fetch Epsilon Caddy stats (Epsilon is the primary frontend server)
    let epsilon_stats = match client.get(format!("{}/api/v1/admin/caddy/stats-local", EPSILON_URL)).send().await {
        Ok(resp) if resp.status().is_success() => {
            #[derive(Deserialize)]
            struct Wrap { data: Option<CaddyStats> }
            resp.json::<Wrap>().await.ok().and_then(|w| w.data)
        }
        _ => None,
    };

    Ok(Json(ApiResponse::success(CaddyStatsAll {
        epsilon: epsilon_stats,
    })))
}

// Keep old nginx endpoints as aliases that return empty data (backwards compat)
/// GET /api/v1/admin/nginx/stats-local — deprecated, returns Caddy stats
pub async fn nginx_stats_local() -> Json<ApiResponse<CaddyStats>> {
    caddy_stats_local().await
}

/// GET /api/v1/admin/nginx/stats — deprecated, returns Caddy stats
pub async fn nginx_stats(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<CaddyStatsAll>>, StatusCode> {
    caddy_stats(headers, State(state)).await
}

// ─────────────────────────────────────────────────────────────────────────────
// q-flux reverse proxy stats (admin server on 127.0.0.1:9090)
// ─────────────────────────────────────────────────────────────────────────────

/// JSON shape returned by q-flux `GET /status`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxStats {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub worker_count: u64,
    #[serde(default)]
    pub uptime_secs: u64,
    #[serde(default)]
    pub active_connections: u64,
    #[serde(default)]
    pub total_connections: u64,
    #[serde(default)]
    pub tls_handshakes: u64,
    #[serde(default)]
    pub tls_handshake_failures: u64,
    #[serde(default)]
    pub total_requests: u64,
    #[serde(default)]
    pub requests_2xx: u64,
    #[serde(default)]
    pub requests_4xx: u64,
    #[serde(default)]
    pub requests_5xx: u64,
    #[serde(default)]
    pub upstream_active: u64,
    #[serde(default)]
    pub upstream_connect_failures: u64,
    #[serde(default)]
    pub upstream_timeouts: u64,
    #[serde(default)]
    pub rate_limited: u64,
    #[serde(default)]
    pub active_websockets: u64,
    #[serde(default)]
    pub websocket_upgrades: u64,
    #[serde(default)]
    pub bytes_received: u64,
    #[serde(default)]
    pub bytes_sent: u64,
    #[serde(default)]
    pub tls_reload_count: u64,
    #[serde(default)]
    pub h2_connections: u64,
    #[serde(default)]
    pub h2_streams_opened: u64,
    #[serde(default)]
    pub h2_streams_closed: u64,
    // Cluster health info (from q-flux admin /status)
    #[serde(default)]
    pub cluster: Option<FluxClusterInfo>,
    // Computed fields (not from q-flux, added by us)
    #[serde(default)]
    pub online: bool,
    #[serde(default)]
    pub requests_per_second: f64,
    #[serde(default)]
    pub error_rate_pct: f64,
}

/// Super-cluster health information from q-flux
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxClusterInfo {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub local_backends: Vec<FluxBackendHealth>,
    #[serde(default)]
    pub cluster_peers: Vec<FluxBackendHealth>,
}

/// Health status of a single backend/peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxBackendHealth {
    #[serde(default)]
    pub addr: String,
    #[serde(default)]
    pub healthy: bool,
    #[serde(default)]
    pub failures: u32,
    #[serde(default)]
    pub last_check_ms_ago: u64,
}

/// Compute flux requests/sec from two snapshots (delta-based)
fn compute_flux_rps(current_requests: u64) -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static PREV_REQUESTS: AtomicU64 = AtomicU64::new(0);
    static PREV_TIME_MS: AtomicU64 = AtomicU64::new(0);

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let prev_req = PREV_REQUESTS.swap(current_requests, Ordering::Relaxed);
    let prev_time = PREV_TIME_MS.swap(now_ms, Ordering::Relaxed);

    if prev_time == 0 || now_ms <= prev_time { return 0.0; }

    let dt_secs = (now_ms - prev_time) as f64 / 1000.0;
    let dreqs = current_requests.saturating_sub(prev_req) as f64;
    dreqs / dt_secs
}

/// GET /api/v1/admin/flux/stats-local — local q-flux metrics (no auth, for cross-server aggregation)
pub async fn flux_stats_local() -> Json<ApiResponse<FluxStats>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();

    let stats = match client.get("http://127.0.0.1:9090/status").send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<FluxStats>().await {
                Ok(mut s) => {
                    s.online = true;
                    s.requests_per_second = compute_flux_rps(s.total_requests);
                    let total_by_status = s.requests_2xx + s.requests_4xx + s.requests_5xx;
                    s.error_rate_pct = if total_by_status > 0 {
                        (s.requests_5xx as f64 / total_by_status as f64) * 100.0
                    } else {
                        0.0
                    };
                    s
                }
                Err(_) => FluxStats::offline(),
            }
        }
        _ => FluxStats::offline(),
    };

    Json(ApiResponse::success(stats))
}

/// GET /api/v1/admin/flux/stats — q-flux metrics (admin auth)
pub async fn flux_stats(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<FluxStats>>, StatusCode> {
    if !is_master_wallet(&headers, &state) {
        return Err(StatusCode::FORBIDDEN);
    }

    // q-flux runs on the same machine, just proxy local
    let resp = flux_stats_local().await;
    Ok(resp)
}

impl FluxStats {
    fn offline() -> Self {
        Self {
            version: String::new(),
            worker_count: 0,
            uptime_secs: 0,
            active_connections: 0,
            total_connections: 0,
            tls_handshakes: 0,
            tls_handshake_failures: 0,
            total_requests: 0,
            requests_2xx: 0,
            requests_4xx: 0,
            requests_5xx: 0,
            upstream_active: 0,
            upstream_connect_failures: 0,
            upstream_timeouts: 0,
            rate_limited: 0,
            active_websockets: 0,
            websocket_upgrades: 0,
            bytes_received: 0,
            bytes_sent: 0,
            tls_reload_count: 0,
            h2_connections: 0,
            h2_streams_opened: 0,
            h2_streams_closed: 0,
            cluster: None,
            online: false,
            requests_per_second: 0.0,
            error_rate_pct: 0.0,
        }
    }
}

/// GET /api/v1/admin/decentralization
/// Returns decentralization index and sub-metrics computed from active mining data.
pub async fn decentralization_metrics(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<DecentralizationMetrics>>, StatusCode> {
    if !is_master_wallet(&headers, &state) {
        return Err(StatusCode::FORBIDDEN);
    }

    // Collect per-wallet hashrates from active_miners (key = "address:worker_id")
    let mut wallet_hashrates: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut total_workers = 0usize;

    if let Some(ref mining_stats_arc) = state.mining_statistics {
        if let Ok(stats) = mining_stats_arc.try_read() {
            total_workers = stats.active_miners.len();
            for (key, miner) in &stats.active_miners {
                // Key format: "address:worker_id" — extract wallet address
                let wallet = key.split(':').next().unwrap_or(key).to_string();
                *wallet_hashrates.entry(wallet).or_insert(0.0) += miner.last_hashrate.max(0.0);
            }
        }
    }

    let unique_wallets = wallet_hashrates.len();
    let total_hashrate: f64 = wallet_hashrates.values().sum();

    // Sort wallets by hashrate descending
    let mut sorted: Vec<f64> = wallet_hashrates.values().copied().collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Top miner % and top 3 %
    let top_miner_pct = if total_hashrate > 0.0 {
        (sorted.first().copied().unwrap_or(0.0) / total_hashrate * 100.0)
    } else { 0.0 };

    let top3_miners_pct = if total_hashrate > 0.0 {
        (sorted.iter().take(3).sum::<f64>() / total_hashrate * 100.0)
    } else { 0.0 };

    // Nakamoto coefficient: min entities to control >= 51%
    let nakamoto_coefficient = if total_hashrate > 0.0 {
        let threshold = total_hashrate * 0.51;
        let mut cumulative = 0.0;
        let mut count = 0usize;
        for &hr in &sorted {
            cumulative += hr;
            count += 1;
            if cumulative >= threshold { break; }
        }
        count
    } else { 0 };

    // Gini coefficient (standard algorithm on sorted ascending)
    let gini_coefficient = if unique_wallets > 1 && total_hashrate > 0.0 {
        let mut asc: Vec<f64> = sorted.clone();
        asc.reverse(); // sorted was desc, reverse to asc
        let n = asc.len() as f64;
        let mut numerator = 0.0;
        for (i, &val) in asc.iter().enumerate() {
            numerator += (2.0 * (i as f64 + 1.0) - n - 1.0) * val;
        }
        (numerator / (n * total_hashrate)).abs()
    } else { 0.0 };

    // HHI: sum of (market_share_pct^2)
    let hhi = if total_hashrate > 0.0 {
        sorted.iter().map(|&hr| {
            let share = hr / total_hashrate * 100.0;
            share * share
        }).sum::<f64>()
    } else { 0.0 };

    // Node count and peer count from deploy status (quick local check)
    let node_count = {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap_or_default();
        let urls = [ALPHA_URL, GAMMA_URL, DELTA_URL, EPSILON_URL];
        let mut online = 1usize; // Beta is always online
        let futs: Vec<_> = urls.iter().map(|u| {
            let c = client.clone();
            let url = format!("{}/api/v1/health", u);
            async move { c.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false) }
        }).collect();
        let results = futures_util::future::join_all(futs).await;
        online += results.iter().filter(|&&r| r).count();
        online
    };

    let peer_count = state.libp2p_peer_count.as_ref()
        .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(0);

    // --- Wealth Gini (from wallet_balances) ---
    let wealth_gini = {
        let balances = state.wallet_balances.read().await;
        let mut vals: Vec<f64> = balances.values()
            .filter(|&&b| b > 0)
            .map(|&b| b as f64)
            .collect();
        if vals.len() > 1 {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = vals.len() as f64;
            let total: f64 = vals.iter().sum();
            let mut numerator = 0.0;
            for (i, &val) in vals.iter().enumerate() {
                numerator += (2.0 * (i as f64 + 1.0) - n - 1.0) * val;
            }
            (numerator / (n * total)).abs()
        } else {
            0.0
        }
    };

    // --- Shannon entropy of mining power ---
    let (entropy_score, _raw_entropy) = if unique_wallets > 1 && total_hashrate > 0.0 {
        let shares: Vec<f64> = sorted.iter()
            .map(|&hr| hr / total_hashrate)
            .filter(|&s| s > 0.0)
            .collect();
        let entropy: f64 = -shares.iter().map(|&p| p * p.ln()).sum::<f64>();
        let max_entropy = (unique_wallets as f64).ln();
        let score = if max_entropy > 0.0 { (entropy / max_entropy * 100.0).min(100.0) } else { 0.0 };
        (score, entropy)
    } else {
        (0.0, 0.0)
    };

    // --- Infrastructure vs community nodes ---
    // All health-checked nodes (Alpha/Beta/Gamma/Delta/Epsilon) are team-operated infra.
    // Community nodes = unique libp2p peers minus those infra servers.
    let infrastructure_nodes = node_count;
    let community_nodes = if peer_count > infrastructure_nodes {
        peer_count - infrastructure_nodes
    } else {
        0
    };

    // --- Sqrt scaling helper ---
    fn sqrt_score(value: f64, target: f64) -> f64 {
        ((value / target).sqrt() * 100.0).min(100.0)
    }

    // --- Composite DI score with sqrt scaling (0-100) ---
    // Weights: Nakamoto 25%, Mining Gini 15%, Wealth Gini 10%, Entropy 10%,
    //          Miner Diversity 15%, Nodes 15%, Peers 10%
    let nakamoto_score = sqrt_score(nakamoto_coefficient as f64, 10.0);
    let gini_score = (1.0 - gini_coefficient) * 100.0; // already 0-100
    let wealth_gini_score = (1.0 - wealth_gini) * 100.0;
    let miner_diversity = sqrt_score(unique_wallets as f64, 100.0);
    let node_score = sqrt_score(node_count as f64, 20.0);
    let peer_score = sqrt_score(peer_count as f64, 100.0);

    let di_raw = nakamoto_score * 0.25
        + gini_score * 0.15
        + wealth_gini_score * 0.10
        + entropy_score * 0.10
        + miner_diversity * 0.15
        + node_score * 0.15
        + peer_score * 0.10;
    let di_raw = di_raw.min(100.0).max(0.0);

    // --- EMA smoothing (alpha=0.1) ---
    let alpha = 0.1_f64;
    let prev_bits = state.di_ema.load(std::sync::atomic::Ordering::Relaxed);
    let prev_ema = f64::from_bits(prev_bits);
    let di = if prev_ema == 0.0 || prev_bits == 0 {
        // First call — seed with raw value
        di_raw
    } else {
        alpha * di_raw + (1.0 - alpha) * prev_ema
    };
    state.di_ema.store(di.to_bits(), std::sync::atomic::Ordering::Relaxed);

    let grade = if di >= 90.0 { "A+" }
        else if di >= 75.0 { "A" }
        else if di >= 60.0 { "B" }
        else if di >= 40.0 { "C" }
        else if di >= 20.0 { "D" }
        else { "F" }.to_string();

    Ok(Json(ApiResponse::success(DecentralizationMetrics {
        unique_wallets,
        total_workers,
        top_miner_pct,
        top3_miners_pct,
        nakamoto_coefficient,
        gini_coefficient,
        hhi,
        node_count,
        peer_count,
        wealth_gini,
        entropy_score,
        infrastructure_nodes,
        community_nodes,
        decentralization_index_raw: di_raw,
        decentralization_index: di,
        grade,
    })))
}

