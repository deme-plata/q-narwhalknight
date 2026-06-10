//! Address Filtering for Docker/Container Network Optimization
//!
//! v1.2.2-beta: Filters non-routable addresses to prevent P2P sync failures
//! in containerized environments.
//!
//! CRITICAL: This filter is DIRECTIONAL - it only filters PEER addresses,
//! NOT our own listen addresses. Filtering our own addresses would cause
//! Identify to advertise no usable addresses.
//!
//! Environment Variables:
//! - Q_FILTER_DOCKER_ADDRESSES: Enable/disable filtering (default: true)
//! - Q_ALLOW_PRIVATE_ADDRESSES: Allow 192.168.x.x addresses (default: false)
//! - Q_EXTERNAL_ADDRESS: Explicitly set external address for NAT traversal

use libp2p::Multiaddr;
use libp2p::multiaddr::Protocol;
use std::net::Ipv4Addr;
use tracing::{debug, info, warn};

/// Check if an address is routable for P2P connections
///
/// IMPORTANT: Only use this for PEER addresses, not our own listen addresses.
/// Filtering our own addresses causes Identify to advertise no usable addresses.
pub fn is_routable_peer_address(addr: &Multiaddr) -> bool {
    // Check if filtering is disabled entirely
    let filter_enabled = std::env::var("Q_FILTER_DOCKER_ADDRESSES")
        .map(|v| v != "false")
        .unwrap_or(true);

    if !filter_enabled {
        return true;
    }

    // Check Q_ALLOW_PRIVATE_ADDRESSES env var for private network deployments
    let allow_private = std::env::var("Q_ALLOW_PRIVATE_ADDRESSES")
        .map(|v| v == "true")
        .unwrap_or(false);

    for protocol in addr.iter() {
        match protocol {
            Protocol::Ip4(ip) => {
                // Always filter loopback
                if ip.is_loopback() {
                    debug!("🔒 [ADDR-FILTER] Filtering loopback: {}", addr);
                    return false;
                }

                // Filter link-local (169.254.0.0/16)
                if ip.octets()[0] == 169 && ip.octets()[1] == 254 {
                    debug!("🔒 [ADDR-FILTER] Filtering link-local: {}", addr);
                    return false;
                }

                // Filter container networks unless explicitly allowed
                if !allow_private {
                    // Docker default bridge: 172.17.0.0/16
                    if is_docker_bridge_network(ip) {
                        debug!("🐳 [ADDR-FILTER] Filtering Docker bridge: {}", addr);
                        return false;
                    }

                    // Docker custom networks: 172.16.0.0/12 (172.16-31.x.x)
                    if is_docker_custom_network(ip) {
                        debug!("🐳 [ADDR-FILTER] Filtering Docker custom: {}", addr);
                        return false;
                    }

                    // Kubernetes/cloud networks: 10.0.0.0/8
                    if ip.octets()[0] == 10 {
                        debug!("☸️  [ADDR-FILTER] Filtering Kubernetes: {}", addr);
                        return false;
                    }

                    // Carrier-grade NAT: 100.64.0.0/10
                    if ip.octets()[0] == 100 && (ip.octets()[1] >= 64 && ip.octets()[1] <= 127) {
                        debug!("🔒 [ADDR-FILTER] Filtering CGNAT: {}", addr);
                        return false;
                    }
                }
            }
            Protocol::Ip6(ip) => {
                // Filter IPv6 loopback (::1)
                if ip.is_loopback() {
                    debug!("🔒 [ADDR-FILTER] Filtering IPv6 loopback: {}", addr);
                    return false;
                }

                // Filter IPv6 link-local (fe80::/10)
                if ip.segments()[0] & 0xffc0 == 0xfe80 {
                    debug!("🔒 [ADDR-FILTER] Filtering IPv6 link-local: {}", addr);
                    return false;
                }

                // Filter IPv6 unique local addresses (fc00::/7) unless allowed
                if !allow_private && (ip.segments()[0] & 0xfe00 == 0xfc00) {
                    debug!("🔒 [ADDR-FILTER] Filtering IPv6 ULA: {}", addr);
                    return false;
                }
            }
            Protocol::Dns(_) | Protocol::Dns4(_) | Protocol::Dns6(_) => {
                // DNS addresses are always considered routable
                return true;
            }
            _ => {}
        }
    }
    true
}

/// Check if IP is on Docker's default bridge network (172.17.0.0/16)
fn is_docker_bridge_network(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    octets[0] == 172 && octets[1] == 17
}

/// Check if IP is on Docker's custom network range (172.16.0.0/12, excluding 172.17.x.x)
fn is_docker_custom_network(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    // 172.16.0.0/12 = 172.16.x.x through 172.31.x.x
    // Excludes 172.17.x.x which is handled by is_docker_bridge_network
    octets[0] == 172 && (octets[1] >= 16 && octets[1] <= 31) && octets[1] != 17
}

/// Score an address for quality (higher is better)
/// Used to prefer public IPs over private ones when multiple addresses available
pub fn score_address(addr: &Multiaddr) -> i32 {
    let mut score = 0;

    for protocol in addr.iter() {
        match protocol {
            Protocol::Ip4(ip) => {
                if ip.is_loopback() {
                    score -= 1000; // Worst
                } else if ip.octets()[0] == 172 && (ip.octets()[1] >= 16 && ip.octets()[1] <= 31) {
                    score -= 100; // Docker networks - heavily penalized
                } else if ip.octets()[0] == 10 {
                    score -= 100; // Kubernetes - heavily penalized
                } else if ip.octets()[0] == 169 && ip.octets()[1] == 254 {
                    score -= 500; // Link-local
                } else if ip.octets()[0] == 192 && ip.octets()[1] == 168 {
                    score += 50; // Home network - OK but not ideal
                } else if !ip.is_private() {
                    score += 100; // Public IPv4 - best
                } else {
                    score += 10; // Other private
                }
            }
            Protocol::Ip6(ip) => {
                if ip.is_loopback() {
                    score -= 1000;
                } else if ip.segments()[0] & 0xffc0 == 0xfe80 {
                    score -= 500; // Link-local
                } else if ip.segments()[0] & 0xfe00 == 0xfc00 {
                    score += 10; // ULA
                } else {
                    score += 90; // Global IPv6 - very good
                }
            }
            Protocol::Dns(_) | Protocol::Dns4(_) | Protocol::Dns6(_) => {
                score += 80; // DNS - reliable
            }
            _ => {}
        }
    }

    score
}

/// Filter a list of peer addresses, returning only routable ones
pub fn filter_peer_addresses(addresses: Vec<Multiaddr>) -> Vec<Multiaddr> {
    let original_count = addresses.len();
    let filtered: Vec<Multiaddr> = addresses
        .into_iter()
        .filter(|addr| is_routable_peer_address(addr))
        .collect();

    let filtered_count = original_count - filtered.len();
    if filtered_count > 0 {
        debug!(
            "🔍 [ADDR-FILTER] Filtered {}/{} non-routable addresses",
            filtered_count, original_count
        );
    }

    filtered
}

/// Get the best address from a list (highest scored routable address)
pub fn get_best_address(addresses: &[Multiaddr]) -> Option<&Multiaddr> {
    addresses
        .iter()
        .filter(|addr| is_routable_peer_address(addr))
        .max_by_key(|addr| score_address(addr))
}

/// Parse and validate external address from environment
pub fn get_external_address() -> Option<Multiaddr> {
    match std::env::var("Q_EXTERNAL_ADDRESS") {
        Ok(addr_str) if !addr_str.is_empty() => {
            match addr_str.parse::<Multiaddr>() {
                Ok(addr) => {
                    info!("📢 [EXTERNAL] Configured external address: {}", addr);
                    Some(addr)
                }
                Err(e) => {
                    warn!("⚠️ [EXTERNAL] Failed to parse Q_EXTERNAL_ADDRESS '{}': {}", addr_str, e);
                    None
                }
            }
        }
        _ => None,
    }
}

/// Parse and validate WebSocket Secure (WSS) external address from environment
///
/// This is critical for browser P2P clients that connect via nginx WSS proxy.
/// Format: /dns4/quillon.xyz/tcp/9443/wss
///
/// The peer ID will be appended automatically by the caller.
pub fn get_external_wss_address() -> Option<Multiaddr> {
    match std::env::var("Q_EXTERNAL_WSS_ADDRESS") {
        Ok(addr_str) if !addr_str.is_empty() => {
            match addr_str.parse::<Multiaddr>() {
                Ok(addr) => {
                    info!("🌐 [EXTERNAL-WSS] Configured external WSS address for browser P2P: {}", addr);
                    Some(addr)
                }
                Err(e) => {
                    warn!("⚠️ [EXTERNAL-WSS] Failed to parse Q_EXTERNAL_WSS_ADDRESS '{}': {}", addr_str, e);
                    None
                }
            }
        }
        _ => {
            // Fallback: Try to derive from Q_EXTERNAL_ADDRESS if it's a DNS address
            if let Some(tcp_addr) = get_external_address() {
                // Check if it's a DNS-based address we can convert
                let addr_str = tcp_addr.to_string();
                if addr_str.contains("/dns4/") || addr_str.contains("/dns6/") {
                    // Try to construct WSS address from TCP address
                    // /dns4/quillon.xyz/tcp/9001 -> /dns4/quillon.xyz/tcp/9443/wss
                    if let Some(wss_addr) = derive_wss_address(&tcp_addr) {
                        info!("🌐 [EXTERNAL-WSS] Derived WSS address from TCP: {}", wss_addr);
                        return Some(wss_addr);
                    }
                }
            }
            None
        }
    }
}

/// Derive a WSS address from a TCP address by replacing port with 9443 and adding /wss
fn derive_wss_address(tcp_addr: &Multiaddr) -> Option<Multiaddr> {
    use std::fmt::Write;

    let mut result = String::new();
    let mut found_tcp = false;

    for protocol in tcp_addr.iter() {
        match protocol {
            Protocol::Dns(host) => {
                write!(result, "/dns4/{}", host).ok()?;
            }
            Protocol::Dns4(host) => {
                write!(result, "/dns4/{}", host).ok()?;
            }
            Protocol::Dns6(host) => {
                write!(result, "/dns6/{}", host).ok()?;
            }
            Protocol::Tcp(_port) => {
                // Replace with WSS port 9443
                write!(result, "/tcp/9443/wss").ok()?;
                found_tcp = true;
            }
            Protocol::P2p(peer_id) => {
                // Preserve peer ID if present
                write!(result, "/p2p/{}", peer_id).ok()?;
            }
            _ => {
                // Skip other protocols like /ws, /wss, /ip4, etc.
            }
        }
    }

    if found_tcp && !result.is_empty() {
        result.parse().ok()
    } else {
        None
    }
}

/// Log the current filter configuration at startup
pub fn log_filter_configuration() {
    let filter_enabled = std::env::var("Q_FILTER_DOCKER_ADDRESSES")
        .map(|v| v != "false")
        .unwrap_or(true);

    let allow_private = std::env::var("Q_ALLOW_PRIVATE_ADDRESSES")
        .map(|v| v == "true")
        .unwrap_or(false);

    let external_addr = std::env::var("Q_EXTERNAL_ADDRESS").ok();
    let external_wss_addr = std::env::var("Q_EXTERNAL_WSS_ADDRESS").ok();

    if filter_enabled {
        info!("🔍 [ADDR-FILTER] Docker/container address filtering: ENABLED");
        info!("   Private addresses (192.168.x.x): {}", if allow_private { "ALLOWED" } else { "FILTERED" });
        info!("   Docker networks (172.x.x.x): FILTERED");
        info!("   Kubernetes networks (10.x.x.x): FILTERED");
    } else {
        warn!("⚠️ [ADDR-FILTER] Address filtering DISABLED - may experience Docker sync issues");
    }

    if let Some(ref addr) = external_addr {
        info!("📢 [EXTERNAL-TCP] External TCP address: {}", addr);
    } else {
        info!("📢 [EXTERNAL-TCP] No external TCP address configured (auto-detection via Identify)");
    }

    if let Some(ref addr) = external_wss_addr {
        info!("🌐 [EXTERNAL-WSS] External WSS address for browser P2P: {}", addr);
    } else if external_addr.is_some() {
        info!("🌐 [EXTERNAL-WSS] Will derive WSS address from TCP external address");
    } else {
        info!("🌐 [EXTERNAL-WSS] No browser P2P address configured (set Q_EXTERNAL_WSS_ADDRESS)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filters_docker_bridge() {
        let addr: Multiaddr = "/ip4/172.17.0.1/tcp/9001".parse().unwrap();
        assert!(!is_routable_peer_address(&addr));
    }

    #[test]
    fn test_filters_docker_custom_network() {
        let addr: Multiaddr = "/ip4/172.18.0.5/tcp/9001".parse().unwrap();
        assert!(!is_routable_peer_address(&addr));

        let addr2: Multiaddr = "/ip4/172.31.255.1/tcp/9001".parse().unwrap();
        assert!(!is_routable_peer_address(&addr2));
    }

    #[test]
    fn test_allows_public_ip() {
        let addr: Multiaddr = "/ip4/185.182.185.227/tcp/9001".parse().unwrap();
        assert!(is_routable_peer_address(&addr));
    }

    #[test]
    fn test_filters_kubernetes() {
        let addr: Multiaddr = "/ip4/10.244.1.5/tcp/9001".parse().unwrap();
        assert!(!is_routable_peer_address(&addr));

        let addr2: Multiaddr = "/ip4/10.0.0.1/tcp/9001".parse().unwrap();
        assert!(!is_routable_peer_address(&addr2));
    }

    #[test]
    fn test_filters_link_local() {
        let addr: Multiaddr = "/ip4/169.254.1.1/tcp/9001".parse().unwrap();
        assert!(!is_routable_peer_address(&addr));
    }

    #[test]
    fn test_filters_loopback() {
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/9001".parse().unwrap();
        assert!(!is_routable_peer_address(&addr));
    }

    #[test]
    fn test_filters_cgnat() {
        let addr: Multiaddr = "/ip4/100.64.0.1/tcp/9001".parse().unwrap();
        assert!(!is_routable_peer_address(&addr));

        let addr2: Multiaddr = "/ip4/100.127.255.255/tcp/9001".parse().unwrap();
        assert!(!is_routable_peer_address(&addr2));
    }

    #[test]
    fn test_allows_home_network_by_default() {
        // 192.168.x.x should be allowed by default (common home networks)
        let addr: Multiaddr = "/ip4/192.168.1.100/tcp/9001".parse().unwrap();
        assert!(is_routable_peer_address(&addr));
    }

    #[test]
    fn test_score_prefers_public() {
        let public: Multiaddr = "/ip4/185.182.185.227/tcp/9001".parse().unwrap();
        let private: Multiaddr = "/ip4/192.168.1.1/tcp/9001".parse().unwrap();
        let docker: Multiaddr = "/ip4/172.17.0.1/tcp/9001".parse().unwrap();

        assert!(score_address(&public) > score_address(&private));
        assert!(score_address(&private) > score_address(&docker));
    }

    #[test]
    fn test_get_best_address() {
        let addresses: Vec<Multiaddr> = vec![
            "/ip4/172.17.0.1/tcp/9001".parse().unwrap(),
            "/ip4/185.182.185.227/tcp/9001".parse().unwrap(),
            "/ip4/192.168.1.1/tcp/9001".parse().unwrap(),
        ];

        let best = get_best_address(&addresses);
        assert!(best.is_some());
        assert_eq!(
            best.unwrap().to_string(),
            "/ip4/185.182.185.227/tcp/9001"
        );
    }

    #[test]
    fn test_filter_returns_only_routable() {
        let addresses: Vec<Multiaddr> = vec![
            "/ip4/172.17.0.1/tcp/9001".parse().unwrap(),   // Docker - filtered
            "/ip4/185.182.185.227/tcp/9001".parse().unwrap(), // Public - kept
            "/ip4/10.0.0.1/tcp/9001".parse().unwrap(),     // K8s - filtered
            "/ip4/192.168.1.1/tcp/9001".parse().unwrap(),  // Home - kept
        ];

        let filtered = filter_peer_addresses(addresses);
        assert_eq!(filtered.len(), 2);
    }
}
