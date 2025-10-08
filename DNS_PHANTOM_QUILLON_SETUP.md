/// DNS-Phantom configuration for quillon.xyz domain
use crate::q_dns_phantom::DnsConfig;
use std::time::Duration;

pub fn create_quillon_dns_config() -> DnsConfig {
    DnsConfig {
        // Use reliable public DNS servers
        primary_servers: vec![
            "8.8.8.8:53".parse().unwrap(),     // Google
            "1.1.1.1:53".parse().unwrap(),     // Cloudflare
            "208.67.222.222:53".parse().unwrap(), // OpenDNS
            "9.9.9.9:53".parse().unwrap(),     // Quad9
        ],
        backup_servers: vec![
            "8.8.4.4:53".parse().unwrap(),
            "1.0.0.1:53".parse().unwrap(),
        ],
        timeout: Duration::from_secs(10),      // Increased timeout
        retries: 5,                            // More retries
        use_tcp: false,                        // Start with UDP
        use_tls: false,                        // Can enable later
        tor_proxy: None,                       // Direct first, Tor later
        steganography_enabled: true,           // Enable steganography
        phantom_domains: vec![
            // Primary domain for Q-NarwhalKnight
            "quillon.xyz".to_string(),

            // Cover traffic domains (mix with real domains)
            "cloudflare.com".to_string(),
            "google.com".to_string(),
            "github.com".to_string(),
            "microsoft.com".to_string(),
            "amazon.com".to_string(),
        ],
        cover_traffic_interval: Duration::from_secs(45), // Every 45 seconds
    }
}

/// Example peer advertisement for quillon.xyz
pub fn create_sample_peer_advertisement() -> String {
    format!(
        "v=qnk1;node={};onion={}.onion;caps=consensus,quantum,tor;port=8333;proto=1.0.0;ts={}",
        hex::encode(&rand::random::<[u8; 16]>()), // 16-byte node ID
        generate_onion_address(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    )
}

fn generate_onion_address() -> String {
    // Generate a realistic-looking onion address
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let chars: String = (0..16)
        .map(|_| {
            let charset = b"abcdefghijklmnopqrstuvwxyz234567";
            charset[rng.gen_range(0..charset.len())] as char
        })
        .collect();
    format!("{}abc123", chars)
}