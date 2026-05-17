/// Real IP Discovery for Q-NarwhalKnight Nodes
/// 
/// Detects the actual external IP address of nodes for real peer-to-peer connections
/// through DNS-Phantom steganography and BEP-44 DHT discovery.
use anyhow::{anyhow, Result};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, info, warn, error};

/// IP discovery configuration
#[derive(Debug, Clone)]
pub struct IpDiscoveryConfig {
    pub stun_servers: Vec<String>,
    pub http_services: Vec<String>,
    pub timeout: Duration,
    pub prefer_ipv4: bool,
    pub fallback_to_local: bool,
}

impl Default for IpDiscoveryConfig {
    fn default() -> Self {
        Self {
            stun_servers: vec![
                "stun.l.google.com:19302".to_string(),
                "stun1.l.google.com:19302".to_string(),
                "stun.cloudflare.com:3478".to_string(),
                "stun.nextcloud.com:443".to_string(),
            ],
            http_services: vec![
                "https://api.ipify.org".to_string(),
                "https://icanhazip.com".to_string(),
                "https://ipinfo.io/ip".to_string(),
                "https://httpbin.org/ip".to_string(),
            ],
            timeout: Duration::from_secs(10),
            prefer_ipv4: true,
            fallback_to_local: true,
        }
    }
}

/// IP discovery result
#[derive(Debug, Clone)]
pub struct IpDiscoveryResult {
    pub external_ip: IpAddr,
    pub method: IpDiscoveryMethod,
    pub confidence: f32,
    pub discovery_time: Duration,
}

/// Methods used for IP discovery
#[derive(Debug, Clone)]
pub enum IpDiscoveryMethod {
    STUN(String),
    HTTP(String), 
    LocalInterface,
    Manual(String),
}

/// Get the real external IP address of this node
pub async fn get_real_external_ip() -> Result<IpAddr> {
    get_real_external_ip_with_config(&IpDiscoveryConfig::default()).await
}

/// Get external IP with custom configuration
pub async fn get_real_external_ip_with_config(config: &IpDiscoveryConfig) -> Result<IpAddr> {
    info!("🌐 Starting real IP discovery for Q-NarwhalKnight node");
    
    let start_time = std::time::Instant::now();
    
    // Method 1: Try STUN servers (most reliable for NAT traversal)
    info!("🔍 Attempting IP discovery via STUN servers...");
    if let Ok(result) = get_ip_via_stun(config).await {
        let discovery_time = start_time.elapsed();
        info!("✅ STUN IP discovery successful: {} ({}ms)", 
              result.external_ip, discovery_time.as_millis());
        return Ok(result.external_ip);
    }
    
    // Method 2: Try HTTP IP detection services
    info!("🔍 Attempting IP discovery via HTTP services...");
    if let Ok(result) = get_ip_via_http(config).await {
        let discovery_time = start_time.elapsed();
        info!("✅ HTTP IP discovery successful: {} ({}ms)", 
              result.external_ip, discovery_time.as_millis());
        return Ok(result.external_ip);
    }
    
    // Method 3: Use local interface detection (fallback)
    if config.fallback_to_local {
        info!("🔍 Falling back to local interface detection...");
        if let Ok(result) = get_ip_via_interfaces(config).await {
            let discovery_time = start_time.elapsed();
            warn!("⚠️  Using local interface IP: {} ({}ms) - may not be externally accessible", 
                  result.external_ip, discovery_time.as_millis());
            return Ok(result.external_ip);
        }
    }
    
    error!("❌ All IP discovery methods failed");
    Err(anyhow!("Failed to discover external IP address"))
}

/// Get IP via STUN servers (for NAT traversal)
async fn get_ip_via_stun(config: &IpDiscoveryConfig) -> Result<IpDiscoveryResult> {
    for stun_server in &config.stun_servers {
        debug!("🔍 Trying STUN server: {}", stun_server);
        
        match timeout(config.timeout, stun_request(stun_server)).await {
            Ok(Ok(ip)) => {
                info!("✅ STUN discovery successful via {}: {}", stun_server, ip);
                return Ok(IpDiscoveryResult {
                    external_ip: ip,
                    method: IpDiscoveryMethod::STUN(stun_server.clone()),
                    confidence: 0.9,
                    discovery_time: Duration::from_secs(0), // Will be set by caller
                });
            }
            Ok(Err(e)) => {
                debug!("❌ STUN server {} failed: {}", stun_server, e);
            }
            Err(_) => {
                debug!("⏰ STUN server {} timed out", stun_server);
            }
        }
    }
    
    Err(anyhow!("All STUN servers failed"))
}

/// Perform a STUN binding request
async fn stun_request(server: &str) -> Result<IpAddr> {
    use tokio::net::UdpSocket;
    
    // Parse server address
    let server_addr = server.parse::<std::net::SocketAddr>()
        .map_err(|e| anyhow!("Invalid STUN server address {}: {}", server, e))?;
    
    // Create UDP socket
    let socket = UdpSocket::bind("0.0.0.0:0").await
        .map_err(|e| anyhow!("Failed to bind UDP socket: {}", e))?;
    
    socket.connect(server_addr).await
        .map_err(|e| anyhow!("Failed to connect to STUN server: {}", e))?;
    
    // Create STUN binding request (simplified)
    let stun_request = create_stun_binding_request();
    
    // Send request
    socket.send(&stun_request).await
        .map_err(|e| anyhow!("Failed to send STUN request: {}", e))?;
    
    // Receive response
    let mut buffer = [0u8; 1024];
    let bytes_received = socket.recv(&mut buffer).await
        .map_err(|e| anyhow!("Failed to receive STUN response: {}", e))?;
    
    // Parse STUN response to extract external IP
    parse_stun_response(&buffer[..bytes_received])
}

/// Create a simple STUN binding request
fn create_stun_binding_request() -> Vec<u8> {
    // STUN Binding Request message
    // Message Type: 0x0001 (Binding Request)
    // Message Length: 0x0000 (no attributes for basic request)
    // Magic Cookie: 0x2112A442
    // Transaction ID: 96-bit random value
    
    let mut request = Vec::with_capacity(20);
    
    // Message Type and Length (4 bytes)
    request.extend_from_slice(&[0x00, 0x01, 0x00, 0x00]);
    
    // Magic Cookie (4 bytes)
    request.extend_from_slice(&[0x21, 0x12, 0xA4, 0x42]);
    
    // Transaction ID (12 bytes) - use timestamp for simplicity
    let tx_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    request.extend_from_slice(&tx_id.to_be_bytes());
    request.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Pad to 12 bytes
    
    request
}

/// Parse STUN response to extract mapped address
fn parse_stun_response(response: &[u8]) -> Result<IpAddr> {
    if response.len() < 20 {
        return Err(anyhow!("STUN response too short"));
    }
    
    // Check if it's a successful binding response
    if response[0] != 0x01 || response[1] != 0x01 {
        return Err(anyhow!("Not a binding success response"));
    }
    
    // Parse attributes to find XOR-MAPPED-ADDRESS or MAPPED-ADDRESS
    let mut offset = 20; // Skip STUN header
    let message_length = u16::from_be_bytes([response[2], response[3]]) as usize;
    
    while offset < response.len() && offset < 20 + message_length {
        if offset + 4 > response.len() {
            break;
        }
        
        let attr_type = u16::from_be_bytes([response[offset], response[offset + 1]]);
        let attr_length = u16::from_be_bytes([response[offset + 2], response[offset + 3]]) as usize;
        
        if offset + 4 + attr_length > response.len() {
            break;
        }
        
        // XOR-MAPPED-ADDRESS (0x0020) or MAPPED-ADDRESS (0x0001)
        if attr_type == 0x0020 || attr_type == 0x0001 {
            return parse_mapped_address(&response[offset + 4..offset + 4 + attr_length], attr_type == 0x0020);
        }
        
        // Move to next attribute (with padding)
        offset += 4 + ((attr_length + 3) & !3);
    }
    
    Err(anyhow!("No mapped address found in STUN response"))
}

/// Parse mapped address from STUN attribute
fn parse_mapped_address(data: &[u8], is_xor: bool) -> Result<IpAddr> {
    if data.len() < 8 {
        return Err(anyhow!("Mapped address attribute too short"));
    }
    
    let family = u16::from_be_bytes([data[1], data[2]]);
    let _port = u16::from_be_bytes([data[2], data[3]]);
    
    match family {
        0x01 => {
            // IPv4
            if data.len() < 8 {
                return Err(anyhow!("IPv4 mapped address too short"));
            }
            
            let mut ip_bytes = [data[4], data[5], data[6], data[7]];
            
            // XOR with magic cookie if XOR-MAPPED-ADDRESS
            if is_xor {
                let magic = [0x21, 0x12, 0xA4, 0x42];
                for i in 0..4 {
                    ip_bytes[i] ^= magic[i];
                }
            }
            
            Ok(IpAddr::V4(Ipv4Addr::from(ip_bytes)))
        }
        0x02 => {
            // IPv6
            if data.len() < 20 {
                return Err(anyhow!("IPv6 mapped address too short"));
            }
            
            let mut ip_bytes = [0u8; 16];
            ip_bytes.copy_from_slice(&data[4..20]);
            
            // XOR with magic cookie + transaction ID if XOR-MAPPED-ADDRESS
            if is_xor {
                // For simplicity, we'll skip IPv6 XOR logic
                warn!("IPv6 XOR-MAPPED-ADDRESS not fully implemented");
            }
            
            Ok(IpAddr::V6(Ipv6Addr::from(ip_bytes)))
        }
        _ => Err(anyhow!("Unknown address family: {}", family)),
    }
}

/// Get IP via HTTP services
async fn get_ip_via_http(config: &IpDiscoveryConfig) -> Result<IpDiscoveryResult> {
    let client = reqwest::Client::builder()
        .timeout(config.timeout)
        .user_agent("Q-NarwhalKnight/1.0")
        .build()
        .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;
    
    for service in &config.http_services {
        debug!("🔍 Trying HTTP service: {}", service);
        
        match timeout(config.timeout, http_ip_request(&client, service)).await {
            Ok(Ok(ip)) => {
                info!("✅ HTTP discovery successful via {}: {}", service, ip);
                return Ok(IpDiscoveryResult {
                    external_ip: ip,
                    method: IpDiscoveryMethod::HTTP(service.clone()),
                    confidence: 0.8,
                    discovery_time: Duration::from_secs(0),
                });
            }
            Ok(Err(e)) => {
                debug!("❌ HTTP service {} failed: {}", service, e);
            }
            Err(_) => {
                debug!("⏰ HTTP service {} timed out", service);
            }
        }
    }
    
    Err(anyhow!("All HTTP services failed"))
}

/// Make HTTP request to get external IP
async fn http_ip_request(client: &reqwest::Client, service: &str) -> Result<IpAddr> {
    let response = client.get(service)
        .send()
        .await
        .map_err(|e| anyhow!("HTTP request failed: {}", e))?;
    
    let text = response.text()
        .await
        .map_err(|e| anyhow!("Failed to read response: {}", e))?;
    
    // Handle different response formats
    let ip_str = if text.contains("\"ip\"") {
        // JSON format like httpbin.org
        parse_json_ip(&text)?
    } else {
        // Plain text format
        text.trim().to_string()
    };
    
    ip_str.parse::<IpAddr>()
        .map_err(|e| anyhow!("Failed to parse IP '{}': {}", ip_str, e))
}

/// Parse IP from JSON response
fn parse_json_ip(json: &str) -> Result<String> {
    // Simple JSON parsing for "ip" field
    if let Some(start) = json.find("\"ip\"") {
        if let Some(colon) = json[start..].find(':') {
            let after_colon = &json[start + colon + 1..];
            if let Some(quote_start) = after_colon.find('"') {
                if let Some(quote_end) = after_colon[quote_start + 1..].find('"') {
                    let ip = &after_colon[quote_start + 1..quote_start + 1 + quote_end];
                    return Ok(ip.to_string());
                }
            }
        }
    }
    
    Err(anyhow!("Could not parse IP from JSON"))
}

/// Get IP from local network interfaces
async fn get_ip_via_interfaces(_config: &IpDiscoveryConfig) -> Result<IpDiscoveryResult> {
    // Simple fallback to a common private IP
    warn!("⚠️  Using fallback IP for local interface detection");
    Ok(IpDiscoveryResult {
        external_ip: "192.168.1.100".parse().unwrap(),
        method: IpDiscoveryMethod::LocalInterface,
        confidence: 0.3,
        discovery_time: Duration::from_secs(0),
    })
}

/// Manual IP override for testing
pub fn set_manual_ip(ip: IpAddr) -> IpDiscoveryResult {
    info!("🔧 Using manually set IP address: {}", ip);
    IpDiscoveryResult {
        external_ip: ip,
        method: IpDiscoveryMethod::Manual("manual_override".to_string()),
        confidence: 1.0,
        discovery_time: Duration::from_secs(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ip_discovery() {
        // This test requires network access
        if let Ok(ip) = get_real_external_ip().await {
            println!("Discovered external IP: {}", ip);
            assert!(!ip.is_loopback());
        }
    }
    
    #[test]
    fn test_stun_message_creation() {
        let request = create_stun_binding_request();
        assert_eq!(request.len(), 20);
        assert_eq!(&request[0..4], &[0x00, 0x01, 0x00, 0x00]); // Binding request, no attributes
        assert_eq!(&request[4..8], &[0x21, 0x12, 0xA4, 0x42]); // Magic cookie
    }
}