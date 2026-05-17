//! IP Allowlist / Blocklist with CIDR support (Issue #027).
//!
//! Provides pre-parsed CIDR-based access control that runs BEFORE the TLS
//! handshake — blocked IPs consume zero TLS resources.
//!
//! Modes:
//! - `disabled` (default): all IPs allowed
//! - `blocklist`: listed IPs/CIDRs are blocked, all others allowed
//! - `allowlist`: only listed IPs/CIDRs are allowed, all others blocked

use std::net::IpAddr;

/// Pre-parsed access control rules. Built once at config load time.
#[derive(Debug, Clone)]
pub struct AccessControl {
    mode: AccessMode,
    /// Pre-parsed CIDR networks for allowlist.
    allow_nets: Vec<IpNetwork>,
    /// Pre-parsed CIDR networks for blocklist.
    block_nets: Vec<IpNetwork>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    Disabled,
    Allowlist,
    Blocklist,
}

/// A parsed IP network (either a single address or a CIDR range).
#[derive(Debug, Clone)]
struct IpNetwork {
    /// Network address (masked).
    addr: IpAddr,
    /// Prefix length in bits.
    prefix_len: u8,
}

impl IpNetwork {
    /// Parse "192.168.1.0/24" or "10.0.0.1" (treated as /32 or /128).
    fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if let Some((addr_str, prefix_str)) = s.split_once('/') {
            let addr: IpAddr = addr_str.parse()
                .map_err(|e| format!("invalid IP in '{}': {}", s, e))?;
            let prefix_len: u8 = prefix_str.parse()
                .map_err(|e| format!("invalid prefix in '{}': {}", s, e))?;
            let max_prefix = match addr {
                IpAddr::V4(_) => 32,
                IpAddr::V6(_) => 128,
            };
            if prefix_len > max_prefix {
                return Err(format!("prefix /{} exceeds max /{} for {}", prefix_len, max_prefix, s));
            }
            // Mask the address to the network
            let masked = mask_ip(addr, prefix_len);
            Ok(IpNetwork { addr: masked, prefix_len })
        } else {
            let addr: IpAddr = s.parse()
                .map_err(|e| format!("invalid IP '{}': {}", s, e))?;
            let prefix_len = match addr {
                IpAddr::V4(_) => 32,
                IpAddr::V6(_) => 128,
            };
            Ok(IpNetwork { addr, prefix_len })
        }
    }

    /// Check if the given IP is within this network.
    fn contains(&self, ip: IpAddr) -> bool {
        // Must be same family
        match (&self.addr, &ip) {
            (IpAddr::V4(net), IpAddr::V4(candidate)) => {
                let net_bits = u32::from(*net);
                let cand_bits = u32::from(*candidate);
                if self.prefix_len == 0 {
                    return true;
                }
                let mask = u32::MAX.checked_shl(32 - self.prefix_len as u32).unwrap_or(0);
                (net_bits & mask) == (cand_bits & mask)
            }
            (IpAddr::V6(net), IpAddr::V6(candidate)) => {
                let net_bits = u128::from(*net);
                let cand_bits = u128::from(*candidate);
                if self.prefix_len == 0 {
                    return true;
                }
                let mask = u128::MAX.checked_shl(128 - self.prefix_len as u32).unwrap_or(0);
                (net_bits & mask) == (cand_bits & mask)
            }
            _ => false, // v4 vs v6 mismatch
        }
    }
}

/// Mask an IP address to its network prefix.
fn mask_ip(addr: IpAddr, prefix_len: u8) -> IpAddr {
    match addr {
        IpAddr::V4(v4) => {
            if prefix_len >= 32 {
                return addr;
            }
            let bits = u32::from(v4);
            let mask = if prefix_len == 0 { 0 } else {
                u32::MAX.checked_shl(32 - prefix_len as u32).unwrap_or(0)
            };
            IpAddr::V4(std::net::Ipv4Addr::from(bits & mask))
        }
        IpAddr::V6(v6) => {
            if prefix_len >= 128 {
                return addr;
            }
            let bits = u128::from(v6);
            let mask = if prefix_len == 0 { 0 } else {
                u128::MAX.checked_shl(128 - prefix_len as u32).unwrap_or(0)
            };
            IpAddr::V6(std::net::Ipv6Addr::from(bits & mask))
        }
    }
}

impl AccessControl {
    /// Build access control rules from config. Parses all CIDR entries upfront.
    /// Returns an error if any entry is invalid CIDR (fail-fast at startup).
    pub fn new(
        mode: &str,
        allowlist: &[String],
        blocklist: &[String],
    ) -> Result<Self, String> {
        let mode = match mode.to_lowercase().as_str() {
            "disabled" | "" => AccessMode::Disabled,
            "allowlist" | "allow" => AccessMode::Allowlist,
            "blocklist" | "block" | "deny" => AccessMode::Blocklist,
            other => return Err(format!("invalid access_control mode: '{}' (expected disabled/allowlist/blocklist)", other)),
        };

        let allow_nets: Vec<IpNetwork> = allowlist.iter()
            .map(|s| IpNetwork::parse(s))
            .collect::<Result<Vec<_>, _>>()?;

        let block_nets: Vec<IpNetwork> = blocklist.iter()
            .map(|s| IpNetwork::parse(s))
            .collect::<Result<Vec<_>, _>>()?;

        if mode == AccessMode::Allowlist && allow_nets.is_empty() {
            return Err("allowlist mode requires at least one allowlist entry".to_string());
        }

        Ok(Self { mode, allow_nets, block_nets })
    }

    /// Check if an IP address is allowed. This is called in the accept loop
    /// BEFORE the TLS handshake, so it must be fast (no allocations).
    #[inline]
    pub fn is_allowed(&self, ip: IpAddr) -> bool {
        match self.mode {
            AccessMode::Disabled => true,
            AccessMode::Blocklist => {
                // Blocked if IP matches any blocklist entry
                !self.block_nets.iter().any(|net| net.contains(ip))
            }
            AccessMode::Allowlist => {
                // Allowed only if IP matches an allowlist entry
                self.allow_nets.iter().any(|net| net.contains(ip))
            }
        }
    }

    /// Returns true if access control is active (not disabled).
    pub fn is_active(&self) -> bool {
        self.mode != AccessMode::Disabled
    }

    /// Get the mode name for logging.
    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            AccessMode::Disabled => "disabled",
            AccessMode::Allowlist => "allowlist",
            AccessMode::Blocklist => "blocklist",
        }
    }

    /// Number of rules configured.
    pub fn rule_count(&self) -> usize {
        self.allow_nets.len() + self.block_nets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_allows_everything() {
        let ac = AccessControl::new("disabled", &[], &[]).unwrap();
        assert!(ac.is_allowed("1.2.3.4".parse().unwrap()));
        assert!(ac.is_allowed("::1".parse().unwrap()));
        assert!(!ac.is_active());
    }

    #[test]
    fn test_blocklist_exact_ip() {
        let ac = AccessControl::new(
            "blocklist",
            &[],
            &["10.0.0.1".to_string()],
        ).unwrap();
        assert!(!ac.is_allowed("10.0.0.1".parse().unwrap()));
        assert!(ac.is_allowed("10.0.0.2".parse().unwrap()));
        assert!(ac.is_allowed("192.168.1.1".parse().unwrap()));
    }

    #[test]
    fn test_blocklist_cidr() {
        let ac = AccessControl::new(
            "blocklist",
            &[],
            &["192.168.1.0/24".to_string()],
        ).unwrap();
        assert!(!ac.is_allowed("192.168.1.1".parse().unwrap()));
        assert!(!ac.is_allowed("192.168.1.254".parse().unwrap()));
        assert!(ac.is_allowed("192.168.2.1".parse().unwrap()));
        assert!(ac.is_allowed("10.0.0.1".parse().unwrap()));
    }

    #[test]
    fn test_allowlist_allows_only_listed() {
        let ac = AccessControl::new(
            "allowlist",
            &["10.0.0.0/8".to_string(), "192.168.1.100".to_string()],
            &[],
        ).unwrap();
        assert!(ac.is_allowed("10.0.0.1".parse().unwrap()));
        assert!(ac.is_allowed("10.255.255.255".parse().unwrap()));
        assert!(ac.is_allowed("192.168.1.100".parse().unwrap()));
        assert!(!ac.is_allowed("192.168.1.101".parse().unwrap()));
        assert!(!ac.is_allowed("8.8.8.8".parse().unwrap()));
    }

    #[test]
    fn test_allowlist_blocks_unmatched() {
        let ac = AccessControl::new(
            "allowlist",
            &["127.0.0.1".to_string()],
            &[],
        ).unwrap();
        assert!(ac.is_allowed("127.0.0.1".parse().unwrap()));
        assert!(!ac.is_allowed("127.0.0.2".parse().unwrap()));
    }

    #[test]
    fn test_ipv6_cidr() {
        let ac = AccessControl::new(
            "blocklist",
            &[],
            &["fd00::/8".to_string()],
        ).unwrap();
        assert!(!ac.is_allowed("fd00::1".parse().unwrap()));
        assert!(!ac.is_allowed("fd12:3456::1".parse().unwrap()));
        assert!(ac.is_allowed("2001:db8::1".parse().unwrap()));
        assert!(ac.is_allowed("::1".parse().unwrap()));
    }

    #[test]
    fn test_ipv6_exact() {
        let ac = AccessControl::new(
            "blocklist",
            &[],
            &["::1".to_string()],
        ).unwrap();
        assert!(!ac.is_allowed("::1".parse().unwrap()));
        assert!(ac.is_allowed("::2".parse().unwrap()));
    }

    #[test]
    fn test_invalid_cidr_rejected() {
        let result = AccessControl::new(
            "blocklist",
            &[],
            &["not-an-ip".to_string()],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid IP"));
    }

    #[test]
    fn test_invalid_prefix_rejected() {
        let result = AccessControl::new(
            "blocklist",
            &[],
            &["10.0.0.0/33".to_string()],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("prefix"));
    }

    #[test]
    fn test_invalid_mode_rejected() {
        let result = AccessControl::new("badmode", &[], &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid access_control mode"));
    }

    #[test]
    fn test_empty_allowlist_rejected() {
        let result = AccessControl::new("allowlist", &[], &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one"));
    }

    #[test]
    fn test_empty_string_mode_is_disabled() {
        let ac = AccessControl::new("", &[], &[]).unwrap();
        assert!(!ac.is_active());
    }

    #[test]
    fn test_mode_aliases() {
        assert!(AccessControl::new("allow", &["1.0.0.0/8".into()], &[]).is_ok());
        assert!(AccessControl::new("block", &[], &[]).is_ok());
        assert!(AccessControl::new("deny", &[], &[]).is_ok());
    }

    #[test]
    fn test_multiple_blocklist_entries() {
        let ac = AccessControl::new(
            "blocklist",
            &[],
            &[
                "10.0.0.0/8".to_string(),
                "172.16.0.0/12".to_string(),
                "192.168.0.0/16".to_string(),
            ],
        ).unwrap();
        assert!(!ac.is_allowed("10.1.2.3".parse().unwrap()));
        assert!(!ac.is_allowed("172.20.1.1".parse().unwrap()));
        assert!(!ac.is_allowed("192.168.100.50".parse().unwrap()));
        assert!(ac.is_allowed("8.8.8.8".parse().unwrap()));
        assert!(ac.is_allowed("1.1.1.1".parse().unwrap()));
    }

    #[test]
    fn test_slash_zero_matches_all() {
        let ac = AccessControl::new(
            "blocklist",
            &[],
            &["0.0.0.0/0".to_string()],
        ).unwrap();
        assert!(!ac.is_allowed("1.2.3.4".parse().unwrap()));
        assert!(!ac.is_allowed("255.255.255.255".parse().unwrap()));
        // IPv6 should NOT be matched by IPv4 /0
        assert!(ac.is_allowed("::1".parse().unwrap()));
    }

    #[test]
    fn test_v4_v6_no_crossover() {
        let ac = AccessControl::new(
            "blocklist",
            &[],
            &["10.0.0.1".to_string()],
        ).unwrap();
        // IPv6 address should not match an IPv4 blocklist entry
        assert!(ac.is_allowed("::ffff:10.0.0.1".parse().unwrap()));
    }

    #[test]
    fn test_rule_count() {
        let ac = AccessControl::new(
            "blocklist",
            &[],
            &["10.0.0.0/8".into(), "192.168.0.0/16".into()],
        ).unwrap();
        assert_eq!(ac.rule_count(), 2);
    }

    #[test]
    fn test_is_active() {
        let disabled = AccessControl::new("disabled", &[], &[]).unwrap();
        assert!(!disabled.is_active());

        let blocklist = AccessControl::new("blocklist", &[], &["10.0.0.1".into()]).unwrap();
        assert!(blocklist.is_active());

        let allowlist = AccessControl::new("allowlist", &["10.0.0.0/8".into()], &[]).unwrap();
        assert!(allowlist.is_active());
    }
}
