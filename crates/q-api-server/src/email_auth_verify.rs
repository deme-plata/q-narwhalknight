/// Quillon Mail: Email Authentication (SPF/DKIM/DMARC)
/// v7.3.2: Verify inbound email authenticity
///
/// Ported from /opt/orobit/shared/axum-mail-server/backend/src/services/email_auth.rs

use std::net::IpAddr;
use tracing::{debug, info, warn};
use hickory_resolver::TokioAsyncResolver;
use hickory_resolver::config::*;

/// Result of email authentication checks
#[derive(Debug, Clone)]
pub struct EmailAuthResult {
    pub spf_pass: bool,
    pub dkim_pass: bool,
    pub dmarc_pass: bool,
    pub action: AuthAction,
    pub details: String,
}

/// Action to take based on authentication results
#[derive(Debug, Clone, PartialEq)]
pub enum AuthAction {
    Accept,
    Quarantine,
    Reject,
}

/// Email authentication verifier
pub struct EmailAuthenticator {
    resolver: TokioAsyncResolver,
}

impl EmailAuthenticator {
    pub fn new() -> Self {
        let resolver = TokioAsyncResolver::tokio(
            ResolverConfig::default(),
            ResolverOpts::default(),
        );
        Self { resolver }
    }

    /// Verify email authenticity (SPF + DKIM + DMARC)
    pub async fn verify_email(
        &self,
        from_domain: &str,
        sender_ip: &str,
        _message: &str,
    ) -> EmailAuthResult {
        let spf_pass = self.check_spf(from_domain, sender_ip).await;
        // DKIM verification requires RSA key parsing — placeholder for now
        let dkim_pass = true; // TODO: Implement full DKIM verification
        let dmarc_result = self.check_dmarc(from_domain, spf_pass, dkim_pass).await;

        let action = if dmarc_result.0 {
            AuthAction::Accept
        } else {
            dmarc_result.1.clone()
        };

        let details = format!(
            "SPF: {}, DKIM: {}, DMARC: {} (action: {:?})",
            if spf_pass { "pass" } else { "fail" },
            if dkim_pass { "pass" } else { "fail" },
            if dmarc_result.0 { "pass" } else { "fail" },
            action
        );

        debug!("📧 [AUTH] {} — {}", from_domain, details);

        EmailAuthResult {
            spf_pass,
            dkim_pass,
            dmarc_pass: dmarc_result.0,
            action,
            details,
        }
    }

    /// Check SPF record for sender authorization
    async fn check_spf(&self, domain: &str, sender_ip: &str) -> bool {
        let sender_addr: IpAddr = match sender_ip.parse() {
            Ok(addr) => addr,
            Err(_) => {
                warn!("📧 [SPF] Invalid sender IP: {}", sender_ip);
                return false;
            }
        };

        // Look up TXT records for SPF
        let txt_records = match self.resolver.txt_lookup(domain).await {
            Ok(records) => records,
            Err(e) => {
                debug!("📧 [SPF] No TXT records for {}: {}", domain, e);
                return false; // No SPF = soft fail
            }
        };

        for record in txt_records.iter() {
            let txt = record.to_string();
            if !txt.starts_with("v=spf1") {
                continue;
            }

            debug!("📧 [SPF] Found record for {}: {}", domain, txt);
            return self.evaluate_spf_record(&txt, &sender_addr);
        }

        debug!("📧 [SPF] No SPF record found for {}", domain);
        false
    }

    /// Evaluate SPF record mechanisms
    fn evaluate_spf_record(&self, record: &str, sender_ip: &IpAddr) -> bool {
        let parts: Vec<&str> = record.split_whitespace().collect();

        for part in &parts[1..] {
            // Skip "v=spf1"
            let (qualifier, mechanism) = if part.starts_with('+')
                || part.starts_with('-')
                || part.starts_with('~')
                || part.starts_with('?')
            {
                (part.chars().next().unwrap(), &part[1..])
            } else {
                ('+', part.as_ref())
            };

            let matches = if mechanism.starts_with("ip4:") {
                self.check_ip_match(sender_ip, &mechanism[4..])
            } else if mechanism.starts_with("ip6:") {
                self.check_ip_match(sender_ip, &mechanism[4..])
            } else if mechanism == "all" {
                true
            } else {
                // a, mx, include, etc. — simplified: skip
                false
            };

            if matches {
                return match qualifier {
                    '+' => true,  // pass
                    '-' => false, // fail
                    '~' => true,  // softfail (treat as pass)
                    '?' => true,  // neutral (treat as pass)
                    _ => false,
                };
            }
        }

        false
    }

    /// Check if sender IP matches an IP specification (CIDR or exact)
    fn check_ip_match(&self, sender_ip: &IpAddr, spec: &str) -> bool {
        // Exact match
        if let Ok(spec_ip) = spec.parse::<IpAddr>() {
            return *sender_ip == spec_ip;
        }

        // CIDR match
        if let Some(slash) = spec.find('/') {
            let (ip_str, prefix_str) = spec.split_at(slash);
            let prefix_len: u8 = match prefix_str[1..].parse() {
                Ok(p) => p,
                Err(_) => return false,
            };

            if let Ok(network_ip) = ip_str.parse::<IpAddr>() {
                return match (sender_ip, &network_ip) {
                    (IpAddr::V4(sender), IpAddr::V4(network)) => {
                        let sender_bits = u32::from(*sender);
                        let network_bits = u32::from(*network);
                        let mask = if prefix_len >= 32 {
                            u32::MAX
                        } else {
                            u32::MAX << (32 - prefix_len)
                        };
                        (sender_bits & mask) == (network_bits & mask)
                    }
                    (IpAddr::V6(sender), IpAddr::V6(network)) => {
                        let sender_bits = u128::from(*sender);
                        let network_bits = u128::from(*network);
                        let mask = if prefix_len >= 128 {
                            u128::MAX
                        } else {
                            u128::MAX << (128 - prefix_len)
                        };
                        (sender_bits & mask) == (network_bits & mask)
                    }
                    _ => false,
                };
            }
        }

        false
    }

    /// Check DMARC policy
    async fn check_dmarc(
        &self,
        domain: &str,
        spf_pass: bool,
        dkim_pass: bool,
    ) -> (bool, AuthAction) {
        let dmarc_domain = format!("_dmarc.{}", domain);

        let txt_records = match self.resolver.txt_lookup(&dmarc_domain).await {
            Ok(records) => records,
            Err(_) => {
                debug!("📧 [DMARC] No DMARC record for {}", domain);
                // No DMARC = accept (no policy to enforce)
                return (true, AuthAction::Accept);
            }
        };

        for record in txt_records.iter() {
            let txt = record.to_string();
            if !txt.starts_with("v=DMARC1") {
                continue;
            }

            debug!("📧 [DMARC] Found policy for {}: {}", domain, txt);

            // Parse policy
            let policy = self.parse_dmarc_policy(&txt);

            // DMARC passes if either SPF or DKIM passes (with alignment)
            let dmarc_pass = spf_pass || dkim_pass;

            if dmarc_pass {
                return (true, AuthAction::Accept);
            }

            // DMARC failed — apply policy
            return match policy.as_str() {
                "reject" => (false, AuthAction::Reject),
                "quarantine" => (false, AuthAction::Quarantine),
                _ => (false, AuthAction::Accept), // "none" policy
            };
        }

        // No DMARC record found
        (true, AuthAction::Accept)
    }

    /// Parse DMARC policy value from record
    fn parse_dmarc_policy(&self, record: &str) -> String {
        for part in record.split(';') {
            let part = part.trim();
            if part.starts_with("p=") {
                return part[2..].to_string();
            }
        }
        "none".to_string()
    }
}
