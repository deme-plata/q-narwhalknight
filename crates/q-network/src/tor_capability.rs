//! 🧅 Tor OOTB capability negotiation + onion auto-discovery (out-of-the-box layer).
//!
//! THE problem with "just turn Tor on": if a node forces Tor but its peer doesn't speak it,
//! the link breaks. The fix is *implicit capability negotiation* carried in the libp2p
//! Identify `agent_version` string — no new protocol, no handshake round-trip:
//!
//!   - A Tor-capable node that has a live onion service advertises it:
//!         "qnk/10.11.71 tor;onion=<base32>.onion:<port>"
//!   - A node without Tor (or with it off) advertises just its base version:
//!         "qnk/10.11.71"
//!
//! On every Identify::Received we parse the peer's agent_version. If it carries an onion,
//! the peer is Tor-capable and we record its onion as a Dandelion stem target. Because ONLY
//! Tor-capable peers advertise an onion, stems can *only ever* route to peers that can
//! receive them — capability negotiation falls out for free, and clearnet peers are simply
//! never chosen as Tor stem targets (no broken links, safe to leave on by default).
//!
//! This module is pure logic (build + parse + a registry) so it unit-tests with no network.
//! Wiring: q-network sets `identify::Config::with_agent_version(build_agent_version(..))` and
//! calls `TorPeerRegistry::note_identify(..)` in the Identify::Received arm; the registry's
//! `onion_targets()` feeds `QTorClient::send_over_tor` stem-target selection.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Marker placed in the Identify agent_version to signal Tor capability + reachability.
const TOR_TAG: &str = "tor;onion=";

/// Build our advertised Identify agent_version.
///
/// `base` is the normal version token (e.g. "qnk/10.11.71"). If we have a live onion
/// service, `onion` is `Some("<addr>.onion:<port>")` and we append the tor tag so peers
/// learn they can stem-relay to us over Tor. With `None` we advertise plain (no tag) — i.e.
/// "I am not Tor-reachable", so peers won't try to onion-route to us. Safe by default.
pub fn build_agent_version(base: &str, onion: Option<&str>) -> String {
    match onion {
        Some(o) if !o.is_empty() => format!("{base} {TOR_TAG}{o}"),
        _ => base.to_string(),
    }
}

/// What we learned about a peer from its advertised agent_version.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PeerTorInfo {
    /// Peer advertised a usable onion → it can receive Tor stem relays.
    pub tor_capable: bool,
    /// The peer's onion target ("<addr>.onion:<port>"), if advertised.
    pub onion: Option<String>,
}

/// Parse a peer's Identify agent_version into Tor capability info.
pub fn parse_peer(agent_version: &str) -> PeerTorInfo {
    if let Some(idx) = agent_version.find(TOR_TAG) {
        let rest = &agent_version[idx + TOR_TAG.len()..];
        // onion token ends at first whitespace
        let onion = rest.split_whitespace().next().unwrap_or("");
        if onion.contains(".onion") && onion.len() > ".onion".len() {
            return PeerTorInfo {
                tor_capable: true,
                onion: Some(onion.to_string()),
            };
        }
    }
    PeerTorInfo { tor_capable: false, onion: None }
}

/// Registry of Tor-capable peers learned via Identify. Cheap to clone (Arc inside).
#[derive(Clone, Default)]
pub struct TorPeerRegistry {
    inner: Arc<RwLock<HashMap<String, String>>>, // peer_id -> onion target
}

impl TorPeerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record what we learned from a peer's Identify. Returns the parsed info.
    /// `peer_id` is any stable peer identifier (e.g. PeerId.to_string()).
    pub fn note_identify(&self, peer_id: &str, agent_version: &str) -> PeerTorInfo {
        let info = parse_peer(agent_version);
        match &info.onion {
            Some(onion) => {
                self.inner.write().insert(peer_id.to_string(), onion.clone());
            }
            None => {
                self.inner.write().remove(peer_id);
            }
        }
        info
    }

    /// Forget a peer (e.g. on disconnect).
    pub fn forget(&self, peer_id: &str) {
        self.inner.write().remove(peer_id);
    }

    /// Is this peer Tor-capable (advertised an onion)?
    pub fn is_capable(&self, peer_id: &str) -> bool {
        self.inner.read().contains_key(peer_id)
    }

    /// All known Tor-capable peer onion targets (Dandelion stem candidates).
    pub fn onion_targets(&self) -> Vec<String> {
        self.inner.read().values().cloned().collect()
    }

    /// Number of Tor-capable peers currently known.
    pub fn capable_count(&self) -> usize {
        self.inner.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_plain_when_no_onion() {
        assert_eq!(build_agent_version("qnk/10.11.71", None), "qnk/10.11.71");
        assert_eq!(build_agent_version("qnk/10.11.71", Some("")), "qnk/10.11.71");
    }

    #[test]
    fn build_and_parse_roundtrip() {
        let onion = "vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd.onion:9055";
        let av = build_agent_version("qnk/10.11.71", Some(onion));
        assert_eq!(av, format!("qnk/10.11.71 tor;onion={onion}"));
        let info = parse_peer(&av);
        assert!(info.tor_capable);
        assert_eq!(info.onion.as_deref(), Some(onion));
    }

    #[test]
    fn parse_plain_is_not_capable() {
        let info = parse_peer("qnk/10.11.71");
        assert!(!info.tor_capable);
        assert!(info.onion.is_none());
    }

    #[test]
    fn parse_rejects_garbage_onion() {
        // tag present but no real onion → not capable
        assert!(!parse_peer("qnk/10.11.71 tor;onion=").tor_capable);
        assert!(!parse_peer("qnk/10.11.71 tor;onion=.onion").tor_capable);
    }

    #[test]
    fn registry_tracks_capable_peers_and_targets() {
        let reg = TorPeerRegistry::new();
        let onion = "abcdefghijklmnopqrstuvwxyz234567abcdefghijklmnopqrstuvwxy.onion:9055";
        reg.note_identify("peerA", &format!("qnk/10.11.71 tor;onion={onion}"));
        reg.note_identify("peerB", "qnk/10.11.71"); // clearnet-only
        assert!(reg.is_capable("peerA"));
        assert!(!reg.is_capable("peerB"));
        assert_eq!(reg.capable_count(), 1);
        assert_eq!(reg.onion_targets(), vec![onion.to_string()]);
        // peer A drops Tor → removed
        reg.note_identify("peerA", "qnk/10.11.71");
        assert_eq!(reg.capable_count(), 0);
        assert!(reg.onion_targets().is_empty());
    }
}
