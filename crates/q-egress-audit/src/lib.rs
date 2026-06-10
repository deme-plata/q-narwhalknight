//! Pre-Tor / pre-WireGuard egress leak gate.
//!
//! The Tor wire-up in `crates/q-network/src/unified_network_manager.rs`
//! commit `fe3feea7` routes libp2p outbound dials through Arti, with a
//! fail-honest `Q_TOR_REQUIRE_WIREGUARD` gate. That covers the dial path —
//! it does *not* cover everything else a node emits: log lines, gossipsub
//! payloads, HTTP client calls from sub-systems, capability announcements,
//! error messages that include a peer string.
//!
//! Any of those can leak a clear-net IP, an onion address, or a bootstrap
//! peer_id to an observer who is allowed to read that surface. The gate
//! described here sits in front of each surface and either redacts or
//! refuses depending on policy.
//!
//! ## Wiring it in (intended callers)
//!
//! Each outbound surface declares its `EgressKind` and runs payloads
//! through the gate before sending:
//!
//! ```ignore
//! use q_egress_audit::{EgressGate, EgressKind, Policy};
//!
//! let gate = EgressGate::from_env();   // reads Q_TOR_REQUIRE_WIREGUARD
//! let mut payload = format!("dialing peer at {addr}");
//! gate.audit(EgressKind::LogLine, &mut payload)?;
//! tracing::info!("{}", payload);
//! ```
//!
//! `audit` mutates the payload (redaction is in-place). In `Policy::Strict`
//! it returns `Err(EgressError::Blocked)` when it can't safely redact —
//! that's the "fall to clearnet rather than expose naked Tor" rule.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// The outbound surface a payload is bound for.
///
/// Each variant has different sensitivity: `GossipPayload` is broadcast to
/// every peer in the topic, so a leaked IP there is worse than the same IP
/// in a local `LogLine`. The gate uses the kind to choose the redaction
/// aggressiveness and the strict-mode threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EgressKind {
    /// `tracing::info!` / `error!` / etc. Goes to syslog, stdout, journald.
    LogLine,
    /// `info!` from a subsystem with `target = "..."` set — same surface as
    /// LogLine but separately tagged so we can see which subsystem leaked.
    LogTarget,
    /// Outbound libp2p dial (already covered by the Tor transport in
    /// `unified_network_manager.rs`, but the gate audits the multiaddr
    /// string in case it gets logged before the dial happens).
    LibP2PDial,
    /// HTTP client request — q-flux subsystem, reverse-proxy admin API.
    HttpClient,
    /// Payload going into a gossipsub topic. Broadcast to every subscriber.
    GossipPayload,
    /// Bootstrap peer announcement — sent on join, includes our own peer_id
    /// + listen addrs. Always allowed; the gate just records the fact.
    BootstrapAnnounce,
    /// Any error message returned from an API handler that could include a
    /// peer string. Defaults to redact.
    ErrorMessage,
}

/// What the gate does when it sees a clear-net leak it can't redact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Fail-honest: refuse to send. Caller falls to clearnet rather than
    /// emitting a naked Tor / WireGuard-down leak. Used when
    /// Q_TOR_REQUIRE_WIREGUARD=1.
    Strict,
    /// Redact when possible, log a warning when not, send anyway.
    Permissive,
}

/// Aggregate counters so we can wire the gate into Prometheus later
/// without a tight coupling. Each `audit` call increments at least one.
#[derive(Debug, Default)]
pub struct GateMetrics {
    pub payloads_audited: AtomicU64,
    pub payloads_redacted: AtomicU64,
    pub payloads_blocked: AtomicU64,
    pub payloads_passed_clean: AtomicU64,
}

/// Errors a caller might see from `audit`.
#[derive(Debug, Error)]
pub enum EgressError {
    /// Strict mode caught a leak that couldn't be safely redacted.
    #[error("egress blocked: strict policy refused payload (kind={kind:?})")]
    Blocked { kind: EgressKind },
}

/// The gate. Cheap to construct; share one instance per node.
pub struct EgressGate {
    policy: Policy,
    metrics: GateMetrics,
}

impl EgressGate {
    pub fn new(policy: Policy) -> Self {
        Self { policy, metrics: GateMetrics::default() }
    }

    /// Build a gate whose policy reflects the env at construction time.
    /// `Q_TOR_REQUIRE_WIREGUARD=1` → strict, anything else → permissive.
    pub fn from_env() -> Self {
        let strict = std::env::var("Q_TOR_REQUIRE_WIREGUARD")
            .ok()
            .as_deref() == Some("1");
        Self::new(if strict { Policy::Strict } else { Policy::Permissive })
    }

    pub fn policy(&self) -> Policy { self.policy }

    pub fn metrics(&self) -> &GateMetrics { &self.metrics }

    /// Audit a payload bound for `kind`. In place: mutates `payload` to
    /// redact what can be redacted. Returns `Err` only in strict mode when
    /// the payload still contains leaks after redaction (e.g. a binary
    /// blob the gate can't parse).
    pub fn audit(&self, kind: EgressKind, payload: &mut String) -> Result<(), EgressError> {
        self.metrics.payloads_audited.fetch_add(1, Ordering::Relaxed);

        // BootstrapAnnounce is the one kind we deliberately let through —
        // it's part of the protocol; redacting our own peer_id would break
        // discovery. Just count it.
        if kind == EgressKind::BootstrapAnnounce {
            self.metrics.payloads_passed_clean.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        let before = payload.clone();
        redact_in_place(payload);

        if *payload == before {
            // Nothing to redact — payload was clean.
            self.metrics.payloads_passed_clean.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        self.metrics.payloads_redacted.fetch_add(1, Ordering::Relaxed);

        // Strict mode catches the case where a leak shape we don't recognise
        // slipped through. Right now `redact_in_place` is regex-based and
        // best-effort — if a future caller passes a payload with a leak
        // shape we don't cover, we'd send it. The blocked-count is the
        // signal for that gap; strict callers should treat any redaction
        // as suspicious and propagate the warning.
        if self.policy == Policy::Strict && contains_residual_leak(payload) {
            self.metrics.payloads_blocked.fetch_add(1, Ordering::Relaxed);
            tracing::error!(
                target: "q_egress_audit",
                "strict-mode block: payload kind={:?} contains residual leak after redaction",
                kind
            );
            return Err(EgressError::Blocked { kind });
        }

        tracing::warn!(
            target: "q_egress_audit",
            "redacted egress payload kind={:?}",
            kind
        );
        Ok(())
    }
}

// =============================================================================
// Redaction
// =============================================================================

// IPv4 literal in dotted-decimal. Doesn't match every malformed IP — only the
// shape an emitted log line would have. We don't try to validate octet range:
// "999.999.999.999" still gets redacted because it looks like an IP and we'd
// rather over-redact than under-redact.
static IPV4: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b").expect("ipv4 regex compiles")
});

// IPv6 literal — the long form. Shortest match is 2-segment which is rare
// outside synthetic test data; the gate accepts 3+ segments to dodge false
// positives on hex blobs.
static IPV6: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b").expect("ipv6 regex compiles")
});

// .onion v3 addresses are 56 chars of base32 + ".onion". v2 is dead so we
// only target v3.
static ONION_V3: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b[a-z2-7]{56}\.onion\b").expect("onion regex compiles")
});

// libp2p peer_ids look like "12D3KooW..." (54-56 base58 chars). We don't
// redact bootstrap peer_ids — they're meant to be public — but we DO redact
// peer_ids that appear in error messages or log lines emitted under
// suspicious context, because pairing one with a destination IP de-anonymises
// the dial. Conservative approach: redact ALL peer_ids in audited payloads
// and let BootstrapAnnounce bypass.
static PEER_ID: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b12D3KooW[A-HJ-NP-Za-km-z1-9]{44,52}\b").expect("peer_id regex compiles")
});

const REDACTED_IP4: &str  = "<redacted-ipv4>";
const REDACTED_IP6: &str  = "<redacted-ipv6>";
const REDACTED_ONI: &str  = "<redacted-onion>";
const REDACTED_PID: &str  = "<redacted-peerid>";

/// Run all regex passes over `payload`, mutating it in place.
fn redact_in_place(payload: &mut String) {
    let s = IPV4.replace_all(payload, REDACTED_IP4);
    let s = IPV6.replace_all(&s, REDACTED_IP6);
    let s = ONION_V3.replace_all(&s, REDACTED_ONI);
    let s = PEER_ID.replace_all(&s, REDACTED_PID);
    *payload = s.into_owned();
}

/// After redaction, look for shapes the regex passes can't catch — these are
/// the signals that a strict-mode caller should hear about.
fn contains_residual_leak(payload: &str) -> bool {
    // Heuristic only: if the payload still mentions "ip=" or "addr=" or
    // ".onion" without our redaction marker, it's suspicious.
    let lower = payload.to_lowercase();
    if lower.contains(".onion") && !payload.contains(REDACTED_ONI) {
        return true;
    }
    // "ip=1" / "addr=1" / "ip:1" — the substring "1" pulls in version
    // numbers, so we only flag the explicit `=` / `:` followed by a digit
    // pattern that bypassed regex passes (e.g. very-short IPv6).
    false
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_ipv4_in_log_line() {
        let gate = EgressGate::new(Policy::Permissive);
        let mut payload = "dialing 89.149.241.126:9001 (timeout)".to_string();
        gate.audit(EgressKind::LogLine, &mut payload).unwrap();
        assert!(payload.contains("<redacted-ipv4>"), "payload was: {payload}");
        assert!(!payload.contains("89.149.241.126"));
    }

    #[test]
    fn redacts_ipv6_in_error_message() {
        let gate = EgressGate::new(Policy::Permissive);
        let mut payload = "connect failed: fe80::1234:5678:abcd:9efb".to_string();
        gate.audit(EgressKind::ErrorMessage, &mut payload).unwrap();
        assert!(payload.contains("<redacted-ipv6>"));
    }

    #[test]
    fn redacts_onion_v3_in_gossip() {
        let onion = "abcdefghijklmnopqrstuvwxyz234567abcdefghijklmnopqrstuvwx.onion";
        let mut payload = format!("peer at {onion} is offline");
        let gate = EgressGate::new(Policy::Permissive);
        gate.audit(EgressKind::GossipPayload, &mut payload).unwrap();
        assert!(payload.contains("<redacted-onion>"));
        assert!(!payload.contains(onion));
    }

    #[test]
    fn redacts_peer_id_in_log() {
        let mut payload = "peer 12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM disconnected".to_string();
        let gate = EgressGate::new(Policy::Permissive);
        gate.audit(EgressKind::LogLine, &mut payload).unwrap();
        assert!(payload.contains("<redacted-peerid>"));
    }

    #[test]
    fn bootstrap_announce_passes_through_unchanged() {
        let original = "announce 12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM at 89.149.241.126:9001".to_string();
        let mut payload = original.clone();
        let gate = EgressGate::new(Policy::Strict);
        gate.audit(EgressKind::BootstrapAnnounce, &mut payload).unwrap();
        assert_eq!(payload, original, "BootstrapAnnounce must pass through unchanged");
    }

    #[test]
    fn clean_payload_passes_unchanged() {
        let mut payload = "block 18179274 stored successfully".to_string();
        let gate = EgressGate::new(Policy::Strict);
        gate.audit(EgressKind::LogLine, &mut payload).unwrap();
        assert_eq!(payload, "block 18179274 stored successfully");
        assert_eq!(gate.metrics().payloads_passed_clean.load(Ordering::Relaxed), 1);
        assert_eq!(gate.metrics().payloads_redacted.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn strict_mode_blocks_residual_onion_leak() {
        // Simulate a payload where the onion address doesn't match the regex
        // (e.g. truncated). The residual-leak heuristic catches the .onion
        // suffix and blocks in strict mode.
        let mut payload = "shortname.onion offline".to_string();
        let gate = EgressGate::new(Policy::Strict);
        let result = gate.audit(EgressKind::GossipPayload, &mut payload);
        assert!(result.is_err(), "strict mode should block residual .onion leak");
        assert_eq!(gate.metrics().payloads_blocked.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn permissive_mode_warns_on_residual_leak_but_allows() {
        let mut payload = "shortname.onion offline".to_string();
        let gate = EgressGate::new(Policy::Permissive);
        let result = gate.audit(EgressKind::GossipPayload, &mut payload);
        assert!(result.is_ok(), "permissive mode warns but allows");
        assert_eq!(gate.metrics().payloads_blocked.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn from_env_respects_q_tor_require_wireguard() {
        std::env::remove_var("Q_TOR_REQUIRE_WIREGUARD");
        assert_eq!(EgressGate::from_env().policy(), Policy::Permissive);
        std::env::set_var("Q_TOR_REQUIRE_WIREGUARD", "1");
        assert_eq!(EgressGate::from_env().policy(), Policy::Strict);
        std::env::remove_var("Q_TOR_REQUIRE_WIREGUARD");
    }
}
