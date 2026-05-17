use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, warn};

/// Maximum time a backend can stay unhealthy before auto-reset.
/// Prevents permanent unhealthy state when health checker is starved.
const MAX_UNHEALTHY_SECS: u64 = 30;

/// Default drain timeout: 300 seconds (5 minutes).
/// If deploy script crashes after drain, auto-undrain prevents permanent lockout.
const DRAIN_TIMEOUT_SECS: u64 = 300;

/// Number of consecutive health check successes needed to auto-clear admin drain.
const DRAIN_AUTO_CLEAR_SUCCESSES: u32 = 2;

/// Health status of a single backend.
#[derive(Debug, Clone)]
pub struct BackendHealth {
    /// Whether this backend is currently considered healthy (true for Healthy and HalfOpen).
    pub is_healthy: bool,
    /// Whether this backend is in half-open (recovering) state.
    /// HalfOpen backends accept traffic but haven't proven full recovery yet.
    pub half_open: bool,
    /// When the last health check completed (success or failure).
    pub last_check: Instant,
    /// When the last successful health check occurred.
    pub last_success: Option<Instant>,
    /// Number of consecutive failures (resets to 0 on success).
    pub consecutive_failures: u32,
    /// Number of consecutive successes since last failure.
    pub consecutive_successes: u32,
    /// When the backend was first marked unhealthy (for auto-recovery).
    pub unhealthy_since: Option<Instant>,
    /// Last health probe response time in milliseconds.
    pub last_response_time_ms: Option<u64>,
    /// v1.0.5: Admin drain flag — set by POST /admin/drain.
    /// When true, the backend is excluded from routing even if health checks pass.
    /// Auto-cleared after DRAIN_TIMEOUT_SECS or DRAIN_AUTO_CLEAR_SUCCESSES consecutive
    /// successful health checks (whichever comes first).
    pub is_admin_drained: bool,
    /// v1.0.5: When admin drain was activated (for timeout auto-clear).
    pub drain_started: Option<Instant>,
    /// v1.0.5: Consecutive successes since drain started (for auto-clear).
    pub drain_success_count: u32,
}

impl BackendHealth {
    fn new() -> Self {
        Self {
            is_healthy: true, // optimistic: assume healthy until proven otherwise
            half_open: false,
            last_check: Instant::now(),
            last_success: None,
            consecutive_failures: 0,
            consecutive_successes: 0,
            unhealthy_since: None,
            last_response_time_ms: None,
            is_admin_drained: false,
            drain_started: None,
            drain_success_count: 0,
        }
    }
}

/// Shared health state accessible from both the health-check loop and the
/// upstream pool's backend selection logic.
pub type HealthMap = Arc<DashMap<String, BackendHealth>>;

/// Create a new empty health map, pre-populated with entries for each backend.
pub fn new_health_map(backends: &[String]) -> HealthMap {
    let map = DashMap::with_capacity(backends.len());
    for backend in backends {
        map.insert(backend.clone(), BackendHealth::new());
    }
    Arc::new(map)
}

/// Configuration for the health checker, extracted from the TOML config.
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// How often to probe each backend.
    pub interval: Duration,
    /// TCP connect + HTTP timeout for a single probe.
    pub timeout: Duration,
    /// Path to GET for the HTTP health check.
    /// Empty string means TCP-only (no HTTP request).
    pub path: String,
    /// Number of consecutive failures before marking unhealthy.
    pub failure_threshold: u32,
    /// Number of consecutive successes to promote from half-open to healthy.
    /// When an unhealthy backend gets its first success, it enters half-open state.
    /// After `healthy_threshold` consecutive successes, it becomes fully healthy.
    pub healthy_threshold: u32,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(5),
            timeout: Duration::from_secs(3),
            path: "/api/v1/status".to_string(),
            failure_threshold: 3,
            healthy_threshold: 2,
        }
    }
}

/// Spawn a background task that periodically probes every backend and updates
/// the shared `HealthMap`.
///
/// The returned `JoinHandle` can be used to abort the checker on shutdown.
pub fn spawn_health_checker(
    backends: Vec<String>,
    health_map: HealthMap,
    config: HealthCheckConfig,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        info!(
            interval_ms = config.interval.as_millis() as u64,
            timeout_ms = config.timeout.as_millis() as u64,
            path = %config.path,
            backends = backends.len(),
            "Health checker started"
        );

        let mut ticker = interval(config.interval);
        // The first tick fires immediately -- skip it so we don't flood logs
        // during startup. The initial optimistic `is_healthy = true` covers
        // the first interval.
        ticker.tick().await;

        loop {
            ticker.tick().await;

            // Auto-recovery: if ALL backends are unhealthy for >MAX_UNHEALTHY_SECS,
            // force-reset them all to healthy. This breaks the deadlock where memory
            // pressure starves the health checker, keeping backends permanently down.
            let all_unhealthy = backends.iter().all(|b| {
                health_map.get(b.as_str()).map(|e| !e.is_healthy).unwrap_or(false)
            });
            if all_unhealthy {
                let any_stale = backends.iter().any(|b| {
                    health_map.get(b.as_str()).map(|e| {
                        e.unhealthy_since
                            .map(|t| t.elapsed() > Duration::from_secs(MAX_UNHEALTHY_SECS))
                            .unwrap_or(false)
                    }).unwrap_or(false)
                });
                if any_stale {
                    warn!(
                        "All backends unhealthy for >{MAX_UNHEALTHY_SECS}s — auto-resetting to healthy"
                    );
                    for backend in &backends {
                        if let Some(mut entry) = health_map.get_mut(backend.as_str()) {
                            entry.is_healthy = true;
                            entry.half_open = false;
                            entry.consecutive_failures = 0;
                            entry.consecutive_successes = 0;
                            entry.unhealthy_since = None;
                            info!(backend = backend.as_str(), "Auto-recovered backend to healthy");
                        }
                    }
                }
            }

            // Probe all backends concurrently (one slow probe can't block others)
            let mut probes = Vec::with_capacity(backends.len());
            for backend in &backends {
                let b = backend.clone();
                let cfg = config.clone();
                probes.push(tokio::spawn(async move {
                    let probe_start = Instant::now();
                    let healthy = probe_backend(&b, &cfg).await;
                    let response_time_ms = probe_start.elapsed().as_millis() as u64;
                    (b, healthy, response_time_ms)
                }));
            }

            for handle in probes {
                if let Ok((backend, healthy, response_time_ms)) = handle.await {
                    update_health(&health_map, &backend, healthy, response_time_ms, config.failure_threshold, config.healthy_threshold);
                }
            }
        }
    })
}

/// Run a single health probe against one backend.
///
/// 1. Attempt a TCP connect within the configured timeout.
/// 2. If `config.path` is non-empty, send a minimal HTTP/1.1 GET and check
///    that the response status line starts with "HTTP/1.1 2" (any 2xx).
async fn probe_backend(backend: &str, config: &HealthCheckConfig) -> bool {
    // --- Step 1: TCP connect ---
    let mut stream = match timeout(config.timeout, TcpStream::connect(backend)).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            debug!(backend, error = %e, "Health check TCP connect failed");
            return false;
        }
        Err(_) => {
            debug!(backend, "Health check TCP connect timed out");
            return false;
        }
    };

    // TCP-only mode: if no path is configured, TCP connect is enough.
    if config.path.is_empty() {
        return true;
    }

    // --- Step 2: Minimal HTTP GET ---
    // We avoid pulling in a full HTTP client just for health checks. A raw
    // request on the already-connected stream keeps things lightweight and
    // avoids an extra connection-pool dependency.
    // Extract host (strip port for the Host header is optional but correct)
    let host = backend;
    let request = format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nUser-Agent: q-flux-healthcheck\r\n\r\n",
        config.path, host
    );

    // Remaining time budget after the TCP connect
    let remaining = config.timeout.saturating_sub(Duration::from_millis(50));

    let result = timeout(remaining, async {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        // Send request — do NOT call shutdown() after writing. Hyper/axum
        // treats a TCP FIN (half-close) as connection abort and drops the
        // response. The Connection: close header tells the server we're done
        // after this request; we rely on the read timeout to clean up.
        if let Err(e) = stream.write_all(request.as_bytes()).await {
            debug!(backend, error = %e, "Health check HTTP write failed");
            return false;
        }

        // Read enough of the response to see the status line.
        // "HTTP/1.1 200 OK\r\n" is 17 bytes; read up to 64 to be safe.
        let mut buf = [0u8; 64];
        let n = match stream.read(&mut buf).await {
            Ok(n) if n > 0 => n,
            Ok(_) => {
                debug!(backend, "Health check got empty response");
                return false;
            }
            Err(e) => {
                debug!(backend, error = %e, "Health check HTTP read failed");
                return false;
            }
        };

        let response_start = std::str::from_utf8(&buf[..n]).unwrap_or("");

        // Accept any 2xx status code
        if response_start.starts_with("HTTP/1.1 2") || response_start.starts_with("HTTP/1.0 2") {
            true
        } else {
            debug!(
                backend,
                response_start = &response_start[..response_start.len().min(40)],
                "Health check got non-2xx response"
            );
            false
        }
    })
    .await;

    match result {
        Ok(healthy) => healthy,
        Err(_) => {
            debug!(backend, "Health check HTTP timed out");
            false
        }
    }
}

/// Update the health map for a single backend after a probe result.
///
/// State machine:
///   Healthy → (failure_threshold consecutive failures) → Unhealthy
///   Unhealthy → (1 success) → HalfOpen
///   HalfOpen → (healthy_threshold consecutive successes) → Healthy
///   HalfOpen → (1 failure) → Unhealthy
fn update_health(
    health_map: &HealthMap,
    backend: &str,
    probe_ok: bool,
    response_time_ms: u64,
    failure_threshold: u32,
    healthy_threshold: u32,
) {
    let now = Instant::now();

    let mut entry = health_map.entry(backend.to_string()).or_insert_with(BackendHealth::new);
    let health = entry.value_mut();

    let was_healthy = health.is_healthy;
    let was_half_open = health.half_open;
    health.last_check = now;
    health.last_response_time_ms = Some(response_time_ms);

    if probe_ok {
        health.consecutive_failures = 0;
        health.consecutive_successes += 1;
        health.last_success = Some(now);

        // v1.0.5: Auto-clear admin drain after N consecutive successes or timeout
        if health.is_admin_drained {
            health.drain_success_count += 1;
            let timed_out = health.drain_started
                .map(|t| t.elapsed() > Duration::from_secs(DRAIN_TIMEOUT_SECS))
                .unwrap_or(false);
            if health.drain_success_count >= DRAIN_AUTO_CLEAR_SUCCESSES || timed_out {
                info!(
                    backend,
                    drain_successes = health.drain_success_count,
                    timed_out,
                    "Auto-clearing admin drain (backend recovered after deploy)"
                );
                health.is_admin_drained = false;
                health.drain_started = None;
                health.drain_success_count = 0;
            }
        }

        if !was_healthy && !was_half_open {
            // Unhealthy → HalfOpen (first success after being down)
            health.is_healthy = true;
            health.half_open = true;
            health.consecutive_successes = 1;
            info!(
                backend,
                response_time_ms,
                healthy_threshold,
                "Backend entering HALF-OPEN state (recovering)"
            );
        } else if was_half_open {
            // HalfOpen → check if we've reached healthy_threshold
            if health.consecutive_successes >= healthy_threshold {
                health.half_open = false;
                health.unhealthy_since = None;
                info!(
                    backend,
                    consecutive_successes = health.consecutive_successes,
                    response_time_ms,
                    "Backend is now HEALTHY (promoted from half-open)"
                );
            } else {
                debug!(
                    backend,
                    consecutive_successes = health.consecutive_successes,
                    healthy_threshold,
                    "Backend still HALF-OPEN (awaiting more successes)"
                );
            }
        }
        // Fully healthy + probe_ok: nothing to change
    } else {
        health.consecutive_failures += 1;
        health.consecutive_successes = 0;

        if was_half_open {
            // HalfOpen → Unhealthy (failure during recovery)
            health.is_healthy = false;
            health.half_open = false;
            health.unhealthy_since = Some(now);
            warn!(
                backend,
                response_time_ms,
                "Backend fell back to UNHEALTHY from half-open (probe failed during recovery)"
            );
        } else if was_healthy && health.consecutive_failures >= failure_threshold {
            health.is_healthy = false;
            health.half_open = false;
            health.unhealthy_since = Some(now);
            error!(
                backend,
                consecutive_failures = health.consecutive_failures,
                "Backend is now UNHEALTHY (exceeded failure threshold)"
            );
        } else if was_healthy {
            warn!(
                backend,
                consecutive_failures = health.consecutive_failures,
                threshold = failure_threshold,
                "Backend probe failed (still healthy, below threshold)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_map_creation() {
        let backends = vec![
            "127.0.0.1:8080".to_string(),
            "127.0.0.1:8081".to_string(),
        ];
        let map = new_health_map(&backends);
        assert_eq!(map.len(), 2);
        assert!(map.get("127.0.0.1:8080").unwrap().is_healthy);
        assert!(map.get("127.0.0.1:8081").unwrap().is_healthy);
    }

    #[test]
    fn test_update_health_failure_threshold() {
        let map: HealthMap = Arc::new(DashMap::new());
        map.insert("backend1".to_string(), BackendHealth::new());

        // First two failures: still healthy
        update_health(&map, "backend1", false, 5, 3, 2);
        assert!(map.get("backend1").unwrap().is_healthy);
        assert_eq!(map.get("backend1").unwrap().consecutive_failures, 1);

        update_health(&map, "backend1", false, 5, 3, 2);
        assert!(map.get("backend1").unwrap().is_healthy);
        assert_eq!(map.get("backend1").unwrap().consecutive_failures, 2);

        // Third failure: now unhealthy
        update_health(&map, "backend1", false, 5, 3, 2);
        assert!(!map.get("backend1").unwrap().is_healthy);
        assert_eq!(map.get("backend1").unwrap().consecutive_failures, 3);

        // One more failure while already unhealthy: stays unhealthy
        update_health(&map, "backend1", false, 5, 3, 2);
        assert!(!map.get("backend1").unwrap().is_healthy);
        assert_eq!(map.get("backend1").unwrap().consecutive_failures, 4);
    }

    #[test]
    fn test_half_open_recovery() {
        let map: HealthMap = Arc::new(DashMap::new());
        map.insert("backend1".to_string(), BackendHealth::new());

        // Drive to unhealthy
        update_health(&map, "backend1", false, 5, 3, 2);
        update_health(&map, "backend1", false, 5, 3, 2);
        update_health(&map, "backend1", false, 5, 3, 2);
        assert!(!map.get("backend1").unwrap().is_healthy);
        assert!(!map.get("backend1").unwrap().half_open);

        // First success: enters half-open (is_healthy=true, half_open=true)
        update_health(&map, "backend1", true, 10, 3, 2);
        assert!(map.get("backend1").unwrap().is_healthy);
        assert!(map.get("backend1").unwrap().half_open);
        assert_eq!(map.get("backend1").unwrap().consecutive_successes, 1);

        // Second success: promoted to fully healthy (healthy_threshold=2)
        update_health(&map, "backend1", true, 10, 3, 2);
        assert!(map.get("backend1").unwrap().is_healthy);
        assert!(!map.get("backend1").unwrap().half_open);
        assert_eq!(map.get("backend1").unwrap().consecutive_failures, 0);
    }

    #[test]
    fn test_half_open_failure_drops_to_unhealthy() {
        let map: HealthMap = Arc::new(DashMap::new());
        map.insert("backend1".to_string(), BackendHealth::new());

        // Drive to unhealthy
        update_health(&map, "backend1", false, 5, 3, 2);
        update_health(&map, "backend1", false, 5, 3, 2);
        update_health(&map, "backend1", false, 5, 3, 2);
        assert!(!map.get("backend1").unwrap().is_healthy);

        // Enter half-open with one success
        update_health(&map, "backend1", true, 10, 3, 2);
        assert!(map.get("backend1").unwrap().half_open);

        // Failure while half-open: drops immediately to unhealthy
        update_health(&map, "backend1", false, 5, 3, 2);
        assert!(!map.get("backend1").unwrap().is_healthy);
        assert!(!map.get("backend1").unwrap().half_open);
    }

    #[test]
    fn test_update_health_resets_failures_on_success() {
        let map: HealthMap = Arc::new(DashMap::new());
        map.insert("b1".to_string(), BackendHealth::new());

        // 2 failures (below threshold)
        update_health(&map, "b1", false, 5, 3, 2);
        update_health(&map, "b1", false, 5, 3, 2);
        assert_eq!(map.get("b1").unwrap().consecutive_failures, 2);

        // Success resets counter
        update_health(&map, "b1", true, 10, 3, 2);
        assert_eq!(map.get("b1").unwrap().consecutive_failures, 0);
        assert!(map.get("b1").unwrap().is_healthy);
    }

    #[test]
    fn test_update_health_unknown_backend_inserted() {
        let map: HealthMap = Arc::new(DashMap::new());

        // Updating a backend that was never inserted should auto-create it
        update_health(&map, "new-backend", true, 10, 3, 2);
        assert!(map.get("new-backend").unwrap().is_healthy);
    }

    #[test]
    fn test_response_time_tracked() {
        let map: HealthMap = Arc::new(DashMap::new());
        map.insert("b1".to_string(), BackendHealth::new());

        update_health(&map, "b1", true, 42, 3, 2);
        assert_eq!(map.get("b1").unwrap().last_response_time_ms, Some(42));

        update_health(&map, "b1", false, 100, 3, 2);
        assert_eq!(map.get("b1").unwrap().last_response_time_ms, Some(100));
    }

    #[test]
    fn test_higher_healthy_threshold() {
        let map: HealthMap = Arc::new(DashMap::new());
        map.insert("b1".to_string(), BackendHealth::new());

        // Drive to unhealthy
        for _ in 0..3 {
            update_health(&map, "b1", false, 5, 3, 5); // healthy_threshold=5
        }
        assert!(!map.get("b1").unwrap().is_healthy);

        // First success: half-open
        update_health(&map, "b1", true, 10, 3, 5);
        assert!(map.get("b1").unwrap().half_open);

        // 2nd, 3rd, 4th successes: still half-open
        for _ in 0..3 {
            update_health(&map, "b1", true, 10, 3, 5);
            assert!(map.get("b1").unwrap().half_open);
        }

        // 5th success: promoted to healthy
        update_health(&map, "b1", true, 10, 3, 5);
        assert!(map.get("b1").unwrap().is_healthy);
        assert!(!map.get("b1").unwrap().half_open);
    }
}
