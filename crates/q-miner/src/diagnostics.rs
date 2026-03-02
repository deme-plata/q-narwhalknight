// ═══════════════════════════════════════════════════════════════════
// MinerDiagnostics: 10 health checks with fix suggestions
// Auto-runs every 10s + on-demand with `R` key
// ═══════════════════════════════════════════════════════════════════

use crate::shared_state::SharedMinerState;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone)]
pub enum CheckStatus {
    Pass,
    Warn(String),
    Fail(String),
}

impl CheckStatus {
    pub fn symbol(&self) -> &'static str {
        match self {
            CheckStatus::Pass => "PASS",
            CheckStatus::Warn(_) => "WARN",
            CheckStatus::Fail(_) => "FAIL",
        }
    }

    pub fn detail(&self) -> Option<&str> {
        match self {
            CheckStatus::Pass => None,
            CheckStatus::Warn(s) | CheckStatus::Fail(s) => Some(s),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub name: &'static str,
    pub status: CheckStatus,
    pub fix_suggestion: Option<String>,
}

pub struct MinerDiagnostics {
    pub checks: Vec<HealthCheck>,
    pub last_run: Instant,
    pub server_reachable: bool,
    /// Minimum miner version required by server (set via UpdateAvailable event)
    pub min_miner_version: Option<String>,
}

impl MinerDiagnostics {
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            last_run: Instant::now(),
            server_reachable: false,
            min_miner_version: None,
        }
    }

    pub fn passed_count(&self) -> usize {
        self.checks.iter().filter(|c| matches!(c.status, CheckStatus::Pass)).count()
    }

    pub fn total_count(&self) -> usize {
        self.checks.len()
    }

    /// Run all health checks against current state (non-async, reads atomics)
    pub fn run_checks(&mut self, state: &Arc<SharedMinerState>) {
        self.checks.clear();
        self.last_run = Instant::now();

        // 1. Server Reachable
        self.check_server_reachable(state);

        // 2. Server Synced
        self.check_server_synced(state);

        // 3. Wallet Valid
        self.check_wallet_valid(state);

        // 4. Threads Running
        self.check_threads_running(state);

        // 5. Challenge Fetching
        self.check_challenge_fetching(state);

        // 6. Solution Acceptance
        self.check_solution_acceptance(state);

        // 7. SSE Connected
        self.check_sse_connected(state);

        // 8. Hashrate Non-Zero
        self.check_hashrate(state);

        // 9. Fallback Usage
        self.check_fallback_usage(state);

        // 10. Version Current
        self.check_version(state);

        // 11. Proxy Health (only when proxy is configured)
        self.check_proxy(state);
    }

    fn check_server_reachable(&mut self, state: &Arc<SharedMinerState>) {
        // If any thread is mining, the server was reachable at some point
        let active = state.active_thread_count();
        let latency = state.last_challenge_latency_us.load(Ordering::Relaxed);

        let status = if active > 0 || latency > 0 {
            self.server_reachable = true;
            if latency > 5_000_000 {
                // > 5 seconds
                CheckStatus::Warn(format!("High latency: {}ms", latency / 1000))
            } else {
                CheckStatus::Pass
            }
        } else if state.errored_thread_count() == state.num_threads {
            self.server_reachable = false;
            CheckStatus::Fail("All threads failed to connect".into())
        } else {
            CheckStatus::Pass // Still starting up
        };

        let fix = match &status {
            CheckStatus::Fail(_) => Some(format!(
                "Check server URL: {}\nTry: --server https://quillon.xyz\nVerify firewall allows outbound HTTPS",
                state.server_url
            )),
            CheckStatus::Warn(_) => Some("High latency may cause stale work. Try a closer server.".into()),
            _ => None,
        };

        self.checks.push(HealthCheck {
            name: "Server Reachable",
            status,
            fix_suggestion: fix,
        });
    }

    fn check_server_synced(&mut self, state: &Arc<SharedMinerState>) {
        let any_syncing = state.thread_states.iter().any(|ts| {
            matches!(ts.get_status(), crate::shared_state::ThreadStatus::WaitingForSync { .. })
        });

        let status = if any_syncing {
            CheckStatus::Warn("Server is catching up to network".into())
        } else {
            CheckStatus::Pass
        };

        self.checks.push(HealthCheck {
            name: "Server Synced",
            status,
            fix_suggestion: if any_syncing {
                Some("Mining starts automatically when sync completes.\nNo action needed.".into())
            } else {
                None
            },
        });
    }

    fn check_wallet_valid(&mut self, state: &Arc<SharedMinerState>) {
        let w = &state.wallet_address;
        let is_qug = w.starts_with("qnk") && !w.starts_with("qnka") && w.len() == 67;
        let is_aqua = w.starts_with("qnka") && w.len() == 66;

        let status = if is_qug || is_aqua {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail(format!("Invalid wallet format (len={})", w.len()))
        };

        self.checks.push(HealthCheck {
            name: "Wallet Valid",
            status: status.clone(),
            fix_suggestion: if matches!(status, CheckStatus::Fail(_)) {
                Some("QUG wallet: 'qnk' + 64 hex chars (67 total)\nAQUA wallet: 'qnka' + 62 hex chars (66 total)".into())
            } else {
                None
            },
        });
    }

    fn check_threads_running(&mut self, state: &Arc<SharedMinerState>) {
        let active = state.active_thread_count();
        let errored = state.errored_thread_count();
        let total = state.num_threads;

        let status = if active == 0 && errored == total && total > 0 {
            CheckStatus::Fail(format!("All {} threads errored", total))
        } else if active == 0 && errored > 0 {
            CheckStatus::Warn(format!("{}/{} threads errored", errored, total))
        } else if active < total && errored > 0 {
            CheckStatus::Warn(format!("{}/{} active, {} errored", active, total, errored))
        } else {
            CheckStatus::Pass
        };

        self.checks.push(HealthCheck {
            name: "Threads Running",
            status: status.clone(),
            fix_suggestion: if matches!(status, CheckStatus::Fail(_)) {
                Some("Check server URL and restart the miner.\nTry: --server https://quillon.xyz".into())
            } else {
                None
            },
        });
    }

    fn check_challenge_fetching(&mut self, state: &Arc<SharedMinerState>) {
        let latency = state.last_challenge_latency_us.load(Ordering::Relaxed);
        let any_fetching = state.thread_states.iter().any(|ts| {
            matches!(ts.get_status(), crate::shared_state::ThreadStatus::FetchingChallenge)
        });

        let status = if latency > 0 || state.active_thread_count() > 0 {
            CheckStatus::Pass
        } else if any_fetching {
            CheckStatus::Warn("Still fetching initial challenge...".into())
        } else if state.errored_thread_count() > 0 {
            CheckStatus::Fail("No successful challenge fetch".into())
        } else {
            CheckStatus::Pass // Still starting
        };

        self.checks.push(HealthCheck {
            name: "Challenge Fetching",
            status: status.clone(),
            fix_suggestion: if matches!(status, CheckStatus::Fail(_)) {
                Some("Try: --server https://quillon.xyz\nThe server must be running q-api-server".into())
            } else {
                None
            },
        });
    }

    fn check_solution_acceptance(&mut self, state: &Arc<SharedMinerState>) {
        let solutions = state.solutions_found.load(Ordering::Relaxed);
        let blocks = state.blocks_mined.load(Ordering::Relaxed);

        let status = if solutions == 0 {
            // No solutions found yet — only warn if running for a while
            if state.start_time.elapsed().as_secs() > 300 {
                CheckStatus::Warn("No solutions in 5+ minutes (normal at low hashrate)".into())
            } else {
                CheckStatus::Pass
            }
        } else if blocks == 0 && solutions > 10 {
            CheckStatus::Warn(format!("{} solutions but 0 blocks — check difficulty", solutions))
        } else {
            CheckStatus::Pass
        };

        self.checks.push(HealthCheck {
            name: "Solution Acceptance",
            status: status.clone(),
            fix_suggestion: if matches!(status, CheckStatus::Warn(_)) {
                Some("Check clock sync (NTP).\nUpdate to latest miner version.\nSolutions are normal to be rare.".into())
            } else {
                None
            },
        });
    }

    fn check_sse_connected(&mut self, state: &Arc<SharedMinerState>) {
        let connected = state.sse_connected.load(Ordering::Relaxed);

        let status = if connected {
            CheckStatus::Pass
        } else if state.start_time.elapsed().as_secs() > 15 {
            CheckStatus::Warn("SSE not connected — mining on timer-based refresh".into())
        } else {
            CheckStatus::Pass // Still starting
        };

        self.checks.push(HealthCheck {
            name: "SSE Connected",
            status: status.clone(),
            fix_suggestion: if matches!(status, CheckStatus::Warn(_)) {
                Some("SSE provides instant new-block notifications.\nWithout it, mining uses periodic refresh (slower).\nMay indicate server firewall blocking SSE.".into())
            } else {
                None
            },
        });
    }

    fn check_hashrate(&mut self, state: &Arc<SharedMinerState>) {
        let khs = state.get_hashrate_khs();

        let status = if khs > 0.0 {
            CheckStatus::Pass
        } else if state.start_time.elapsed().as_secs() > 10 && state.active_thread_count() > 0 {
            CheckStatus::Warn("Hashrate is 0 — may still be warming up".into())
        } else if state.start_time.elapsed().as_secs() > 30 {
            CheckStatus::Fail("Hashrate is 0 after 30 seconds".into())
        } else {
            CheckStatus::Pass
        };

        self.checks.push(HealthCheck {
            name: "Hashrate Non-Zero",
            status: status.clone(),
            fix_suggestion: if matches!(status, CheckStatus::Fail(_)) {
                Some("Try lowering intensity: --intensity 5\nCheck CPU usage — another process may be competing.".into())
            } else {
                None
            },
        });
    }

    fn check_fallback_usage(&mut self, state: &Arc<SharedMinerState>) {
        let using_fb = state.using_fallback.load(Ordering::Relaxed);

        let status = if using_fb {
            CheckStatus::Warn("Using fallback server (primary unreachable)".into())
        } else {
            CheckStatus::Pass
        };

        self.checks.push(HealthCheck {
            name: "Fallback Usage",
            status: status.clone(),
            fix_suggestion: if using_fb {
                Some(format!(
                    "Primary server {} is unreachable.\nFallback quillon.xyz is being used.\nCheck if primary server is running.",
                    state.server_url
                ))
            } else {
                None
            },
        });
    }

    fn check_version(&mut self, _state: &Arc<SharedMinerState>) {
        let my_ver = env!("CARGO_PKG_VERSION");

        let status = if let Some(ref min_ver) = self.min_miner_version {
            CheckStatus::Warn(format!("Minimum v{} required, you have v{}", min_ver, my_ver))
        } else {
            CheckStatus::Pass // No update required (or server doesn't report min version)
        };

        self.checks.push(HealthCheck {
            name: "Version Current",
            status: status.clone(),
            fix_suggestion: if matches!(status, CheckStatus::Warn(_)) {
                Some(format!(
                    "Download latest: wget https://quillon.xyz/downloads/q-miner-v{}\nchmod +x q-miner-v{}",
                    self.min_miner_version.as_deref().unwrap_or(my_ver),
                    self.min_miner_version.as_deref().unwrap_or(my_ver),
                ))
            } else {
                None
            },
        });
    }

    fn check_proxy(&mut self, state: &Arc<SharedMinerState>) {
        let proxy = match &state.proxy_url {
            Some(p) => p.clone(),
            None => return, // No proxy configured — skip this check entirely
        };

        let active = state.active_thread_count();
        let elapsed = state.start_time.elapsed().as_secs();

        let status = if active > 0 {
            CheckStatus::Pass
        } else if elapsed > 20 {
            CheckStatus::Fail("Proxy configured but no threads connected after 20s".into())
        } else {
            CheckStatus::Pass // Still starting up
        };

        self.checks.push(HealthCheck {
            name: "Proxy Health",
            status: status.clone(),
            fix_suggestion: if matches!(status, CheckStatus::Fail(_)) {
                Some(format!(
                    "Check proxy is running: {}\nTest with: curl --socks5 {} https://quillon.xyz/api/v1/status\nOr try without proxy: remove --proxy/--tor flag",
                    proxy,
                    proxy.trim_start_matches("socks5://"),
                ))
            } else {
                None
            },
        });
    }
}

/// Run an async server reachability check (for initial diagnostics)
pub async fn check_server_reachable_async(server_url: &str) -> Result<(bool, Option<String>, u64), String> {
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("Client build error: {}", e))?;

    let url = format!("{}/api/v1/status", server_url);
    let start = Instant::now();

    match client.get(&url).send().await {
        Ok(resp) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            if resp.status().is_success() {
                if let Ok(body) = resp.json::<serde_json::Value>().await {
                    let version = body.get("data")
                        .and_then(|d| d.get("version"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    Ok((true, version, latency_ms))
                } else {
                    Ok((true, None, latency_ms))
                }
            } else {
                Err(format!("HTTP {}", resp.status()))
            }
        }
        Err(e) => {
            if e.is_timeout() {
                Err("Connection timed out".into())
            } else if e.is_connect() {
                Err("Connection refused".into())
            } else {
                Err(format!("{}", e))
            }
        }
    }
}
