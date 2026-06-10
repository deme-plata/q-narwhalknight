//! WireGuard fail-honest gate for Tor outbound routing.
//!
//! ## Why this exists
//!
//! When Tor outbound is enabled (`Q_TOR_OUTBOUND=1`) or Dandelion-stem-via-Tor
//! is enabled (`Q_TOR_STEM=1`), the operator's ISP would normally see
//! "this machine talks to the Tor network" — fine for some threat models,
//! catastrophic for others. Operators who care strongly about ISP-level
//! anonymity pair Tor with a WireGuard tunnel: traffic egresses through
//! the VPN provider, the ISP sees only encrypted VPN traffic.
//!
//! The risk: if WireGuard drops (provider issue, kill-switch failure, link
//! flap), Tor traffic silently falls back to the bare ISP path. The
//! operator THINKS they're anonymous; they're not.
//!
//! The fail-honest gate: when `Q_TOR_REQUIRE_WIREGUARD=1` is set, the Tor
//! activation code in `tor_integration.rs` checks this module at startup
//! AND periodically. If WireGuard is not up, the Tor routing is **refused**
//! (not silently downgraded), with a loud log line explaining the reason.
//! Operators must explicitly opt out (`Q_TOR_REQUIRE_WIREGUARD=0`) to
//! accept the leak risk.
//!
//! ## Detection
//!
//! Reads `/sys/class/net/<interface>/operstate`. No shell-out, no new deps.
//! Default interface name: `wg0`. Override via `Q_WIREGUARD_INTERFACE` env.
//! Result is cached for `WG_CACHE_TTL_SECS` to avoid syscall-storm on hot
//! paths.
//!
//! ## What this does NOT check
//!
//! - The interface is `up` per `operstate`, but we don't verify the tunnel
//!   is actually carrying traffic. A misconfigured route could still leak.
//!   Use `mitmproxy` or `tcpdump` on the WAN interface to verify end-to-end.
//! - We don't query the WireGuard handshake state. `operstate=up` means the
//!   kernel netdev is up; the cryptographic handshake could be stale.
//!   Sophisticated operators should pair this with `wg show <iface> latest-handshakes`.

use std::sync::atomic::{AtomicI64, AtomicU8, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::{debug, info, warn};

/// Default WireGuard interface name. Override via `Q_WIREGUARD_INTERFACE`.
pub const DEFAULT_WG_INTERFACE: &str = "wg0";

/// How long to cache a check result before re-querying sysfs.
const WG_CACHE_TTL_SECS: i64 = 5;

/// WireGuard interface status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WireguardStatus {
    /// Interface is up and operational per `/sys/class/net/<iface>/operstate`.
    /// Note: this does NOT verify the WireGuard cryptographic handshake is
    /// fresh — see module docs.
    Up { interface: String },

    /// Interface exists but is not up. `operstate` returned a value other
    /// than "up" (typically "down", "dormant", or "unknown").
    Down { interface: String, operstate: String },

    /// Interface does not exist. `/sys/class/net/<iface>/` is not present.
    Missing { interface: String },

    /// sysfs read failed for some other reason (permissions, IO).
    /// Treated as "not up" for gating purposes — when in doubt, deny.
    Error { interface: String, reason: String },
}

impl WireguardStatus {
    /// Convenience: is the interface up?
    pub fn is_up(&self) -> bool {
        matches!(self, WireguardStatus::Up { .. })
    }

    /// One-line human-readable summary.
    pub fn summary(&self) -> String {
        match self {
            WireguardStatus::Up { interface } => format!("UP on {}", interface),
            WireguardStatus::Down { interface, operstate } => {
                format!("DOWN on {} (operstate={})", interface, operstate)
            }
            WireguardStatus::Missing { interface } => {
                format!("MISSING — no kernel netdev named {}", interface)
            }
            WireguardStatus::Error { interface, reason } => {
                format!("ERROR checking {} — {}", interface, reason)
            }
        }
    }
}

/// Read `/sys/class/net/<interface>/operstate` and return a parsed status.
///
/// This is the raw uncached check. Most callers want `cached_check`.
pub fn check_wireguard_status_raw(interface: &str) -> WireguardStatus {
    let path = format!("/sys/class/net/{}/operstate", interface);
    match std::fs::read_to_string(&path) {
        Ok(s) => {
            let operstate = s.trim().to_string();
            if operstate == "up" {
                WireguardStatus::Up { interface: interface.to_string() }
            } else {
                WireguardStatus::Down {
                    interface: interface.to_string(),
                    operstate,
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            WireguardStatus::Missing { interface: interface.to_string() }
        }
        Err(e) => WireguardStatus::Error {
            interface: interface.to_string(),
            reason: e.to_string(),
        },
    }
}

/// Cached check — returns the previous result if it was queried within the
/// last `WG_CACHE_TTL_SECS` seconds. Avoids hot-path syscalls when the
/// caller polls frequently.
///
/// The cache is process-global. State encoded as:
///   - `WG_STATUS_CACHED_AT`: unix seconds at last cache write (-1 = empty)
///   - `WG_STATUS_KIND`: enum discriminant (0 = Up, 1 = Down, 2 = Missing, 3 = Error)
/// Cache stores only the boolean (is_up); for the typed enum, callers that
/// need the full status should call `check_wireguard_status_raw` directly.
pub fn cached_is_up(interface: &str) -> bool {
    static CACHED_AT: AtomicI64 = AtomicI64::new(-1);
    static CACHED_UP: AtomicU8 = AtomicU8::new(0);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let cached_at = CACHED_AT.load(Ordering::Acquire);
    if cached_at >= 0 && now - cached_at < WG_CACHE_TTL_SECS {
        return CACHED_UP.load(Ordering::Acquire) == 1;
    }

    // Cache miss — refresh.
    let status = check_wireguard_status_raw(interface);
    let is_up = status.is_up();
    CACHED_UP.store(if is_up { 1 } else { 0 }, Ordering::Release);
    CACHED_AT.store(now, Ordering::Release);
    is_up
}

/// Resolve the WireGuard interface name from env, with a fallback default.
pub fn interface_from_env() -> String {
    std::env::var("Q_WIREGUARD_INTERFACE")
        .unwrap_or_else(|_| DEFAULT_WG_INTERFACE.to_string())
}

/// Activation-time check: should we proceed with Tor outbound routing?
///
/// Returns `Ok(())` if the gate is satisfied:
///   - `require_wireguard == false` (operator opted out — leak risk accepted), OR
///   - `require_wireguard == true` AND WireGuard is up.
///
/// Returns `Err(WireguardStatus)` if `require_wireguard` is true but WG is
/// not up. Callers should refuse to activate Tor outbound and log loudly.
///
/// Side effect: emits an `info!` or `warn!` log line describing the decision.
/// The log message is the operator-facing fail-honest signal — it has to be
/// readable, not just machine-parseable.
pub fn activation_gate(require_wireguard: bool) -> Result<WireguardStatus, WireguardStatus> {
    let interface = interface_from_env();
    let status = check_wireguard_status_raw(&interface);

    if !require_wireguard {
        // Operator did not require WG — log status for visibility, allow.
        info!(
            "🛡️  [WG GATE] WireGuard status: {} — Tor outbound NOT gated (Q_TOR_REQUIRE_WIREGUARD=0 or unset)",
            status.summary()
        );
        return Ok(status);
    }

    // Operator requires WG.
    if status.is_up() {
        info!(
            "🛡️  [WG GATE] WireGuard {} — Tor outbound permitted (Q_TOR_REQUIRE_WIREGUARD=1 satisfied)",
            status.summary()
        );
        Ok(status)
    } else {
        warn!(
            "🛡️  [WG GATE] WireGuard {} — Tor outbound REFUSED. \
             Q_TOR_REQUIRE_WIREGUARD=1 is set but the WG interface is not up. \
             Bring WireGuard up OR explicitly opt out (Q_TOR_REQUIRE_WIREGUARD=0) \
             to accept ISP-level visibility risk.",
            status.summary()
        );
        Err(status)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_interface_returns_missing_status() {
        let s = check_wireguard_status_raw("nonexistent_iface_zzz");
        assert!(matches!(s, WireguardStatus::Missing { .. }));
        assert!(!s.is_up());
    }

    #[test]
    fn summary_strings_are_human_readable() {
        let s = WireguardStatus::Up { interface: "wg0".to_string() };
        assert_eq!(s.summary(), "UP on wg0");
        let s = WireguardStatus::Down {
            interface: "wg0".to_string(),
            operstate: "down".to_string(),
        };
        assert!(s.summary().contains("DOWN"));
        assert!(s.summary().contains("operstate=down"));
    }

    #[test]
    fn is_up_reflects_status_correctly() {
        assert!(WireguardStatus::Up { interface: "x".to_string() }.is_up());
        assert!(!WireguardStatus::Down { interface: "x".to_string(), operstate: "down".to_string() }.is_up());
        assert!(!WireguardStatus::Missing { interface: "x".to_string() }.is_up());
        assert!(!WireguardStatus::Error { interface: "x".to_string(), reason: "io".to_string() }.is_up());
    }

    #[test]
    fn interface_from_env_default() {
        // Save + restore env var so test is hermetic.
        let saved = std::env::var("Q_WIREGUARD_INTERFACE").ok();
        std::env::remove_var("Q_WIREGUARD_INTERFACE");
        assert_eq!(interface_from_env(), "wg0");
        if let Some(v) = saved {
            std::env::set_var("Q_WIREGUARD_INTERFACE", v);
        }
    }

    #[test]
    fn activation_gate_passes_when_not_required() {
        // require_wireguard=false → always Ok regardless of actual WG state.
        let r = activation_gate(false);
        assert!(r.is_ok());
    }

    #[test]
    fn activation_gate_with_required_returns_err_when_iface_missing() {
        // Force a known-missing interface, require WG → must err.
        let saved = std::env::var("Q_WIREGUARD_INTERFACE").ok();
        std::env::set_var("Q_WIREGUARD_INTERFACE", "nonexistent_iface_zzz_test");
        let r = activation_gate(true);
        assert!(r.is_err());
        if let Err(s) = r {
            assert!(matches!(s, WireguardStatus::Missing { .. }));
        }
        if let Some(v) = saved {
            std::env::set_var("Q_WIREGUARD_INTERFACE", v);
        } else {
            std::env::remove_var("Q_WIREGUARD_INTERFACE");
        }
    }
}
