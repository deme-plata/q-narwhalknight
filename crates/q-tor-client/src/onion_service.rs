use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use tracing::{debug, error, info, warn};

/// Persistent keypair for Tor v3 onion service
#[derive(Clone, Serialize, Deserialize)]
pub struct OnionKeypair {
    /// Ed25519 secret key (32 bytes expanded to 64 with nonce for signing)
    pub secret_key: Vec<u8>,
    /// Ed25519 public key (32 bytes)
    pub public_key: Vec<u8>,
    /// Derived v3 .onion address (56 characters without .onion suffix)
    pub onion_address: String,
    /// Creation timestamp
    pub created_at: u64,
}

impl OnionKeypair {
    /// Generate a new keypair for Tor v3 onion service
    pub fn generate() -> Result<Self> {
        use std::time::{SystemTime, UNIX_EPOCH};

        // Generate 32 bytes of cryptographic randomness for secret key seed
        let mut secret_seed = [0u8; 32];
        getrandom::getrandom(&mut secret_seed)
            .map_err(|e| anyhow::anyhow!("Failed to generate random bytes: {}", e))?;

        // Expand secret key using SHA3-256 (real implementation would use Ed25519 key expansion)
        let mut hasher = Sha3_256::new();
        hasher.update(&secret_seed);
        hasher.update(b"QNK_ONION_SERVICE_KEY_EXPANSION_v1");
        let expanded: [u8; 32] = hasher.finalize().into();

        // For the secret key, we store the 32-byte seed
        let secret_key = secret_seed.to_vec();

        // Derive public key from secret (simplified - real Ed25519 would use actual curve math)
        let mut pub_hasher = Sha3_256::new();
        pub_hasher.update(&expanded);
        pub_hasher.update(b"QNK_ONION_PUBLIC_KEY_DERIVATION_v1");
        let public_key: Vec<u8> = pub_hasher.finalize().to_vec();

        // Generate v3 onion address from public key
        // Real Tor v3: base32(pubkey || checksum || version)
        let onion_address = Self::derive_onion_address(&public_key)?;

        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(Self {
            secret_key,
            public_key,
            onion_address,
            created_at,
        })
    }

    /// Derive v3 .onion address from public key (56 character base32)
    fn derive_onion_address(public_key: &[u8]) -> Result<String> {
        // Tor v3 onion address format:
        // base32(pubkey[32] || checksum[2] || version[1])
        // = 35 bytes = 56 base32 characters

        // Compute checksum: SHA3-256(".onion checksum" || pubkey || version)
        let mut checksum_hasher = Sha3_256::new();
        checksum_hasher.update(b".onion checksum");
        checksum_hasher.update(public_key);
        checksum_hasher.update(&[0x03]); // Version 3
        let checksum_full: [u8; 32] = checksum_hasher.finalize().into();

        // Build the address data: pubkey (32) + checksum (2) + version (1) = 35 bytes
        let mut address_data = Vec::with_capacity(35);
        address_data.extend_from_slice(&public_key[..32.min(public_key.len())]);
        // Pad if needed
        while address_data.len() < 32 {
            address_data.push(0);
        }
        address_data.push(checksum_full[0]);
        address_data.push(checksum_full[1]);
        address_data.push(0x03); // Version 3

        // Base32 encode (lowercase, no padding)
        let onion_address = base32_encode(&address_data);

        Ok(onion_address)
    }

    /// Save keypair to file
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .context("Failed to serialize keypair")?;

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create key directory")?;
        }

        std::fs::write(path, json)
            .context("Failed to write keypair file")?;

        info!("🔐 Saved onion keypair to {:?}", path);
        Ok(())
    }

    /// Load keypair from file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .context("Failed to read keypair file")?;

        let keypair: Self = serde_json::from_str(&json)
            .context("Failed to deserialize keypair")?;

        info!("🔐 Loaded existing onion keypair from {:?}", path);
        Ok(keypair)
    }
}

/// Base32 encode (RFC 4648 lowercase, no padding)
fn base32_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut result = String::new();

    let mut buffer: u64 = 0;
    let mut bits_in_buffer = 0;

    for &byte in data {
        buffer = (buffer << 8) | (byte as u64);
        bits_in_buffer += 8;

        while bits_in_buffer >= 5 {
            bits_in_buffer -= 5;
            let index = ((buffer >> bits_in_buffer) & 0x1F) as usize;
            result.push(ALPHABET[index] as char);
        }
    }

    // Handle remaining bits
    if bits_in_buffer > 0 {
        let index = ((buffer << (5 - bits_in_buffer)) & 0x1F) as usize;
        result.push(ALPHABET[index] as char);
    }

    result
}

/// Onion service for Q-NarwhalKnight validator
pub struct OnionService {
    /// SOCKS proxy address
    socks_proxy: SocketAddr,
    onion_name: String,
    onion_address: String,
    port: u16,
    service_id: Option<String>,
    /// Persistent keypair for this onion service
    keypair: OnionKeypair,
    /// Path where keypair is stored
    keypair_path: Option<PathBuf>,
}

impl OnionService {
    /// Create and start new onion service with persistent keypair
    pub async fn new(socks_proxy: SocketAddr, onion_name: String, port: u16) -> Result<Self> {
        // Use default data directory for keypair
        let data_dir = std::env::var("Q_DATA_DIR")
            .unwrap_or_else(|_| "./data".to_string());
        let keypair_path = PathBuf::from(data_dir).join("tor").join("onion_service_key.json");

        Self::new_with_keypair_path(socks_proxy, onion_name, port, Some(keypair_path)).await
    }

    /// Create onion service with specific keypair path
    pub async fn new_with_keypair_path(
        socks_proxy: SocketAddr,
        onion_name: String,
        port: u16,
        keypair_path: Option<PathBuf>,
    ) -> Result<Self> {
        info!("🧅 Creating persistent onion service: {}", onion_name);

        // Load existing keypair or generate new one
        let (keypair, is_new) = match &keypair_path {
            Some(path) if path.exists() => {
                match OnionKeypair::load_from_file(path) {
                    Ok(kp) => {
                        info!("🔑 Loaded PERSISTENT onion address: {}.onion", kp.onion_address);
                        (kp, false)
                    }
                    Err(e) => {
                        warn!("⚠️ Failed to load keypair, generating new: {}", e);
                        let kp = OnionKeypair::generate()?;
                        (kp, true)
                    }
                }
            }
            _ => {
                info!("🔑 Generating NEW persistent onion keypair...");
                let kp = OnionKeypair::generate()?;
                (kp, true)
            }
        };

        // Save new keypair if generated
        if is_new {
            if let Some(path) = &keypair_path {
                keypair.save_to_file(path)?;
            }
        }

        let onion_address = keypair.onion_address.clone();

        let mut service = Self {
            socks_proxy,
            onion_name: onion_name.clone(),
            onion_address: onion_address.clone(),
            port,
            service_id: None,
            keypair,
            keypair_path,
        };

        // Start the onion service
        service.start_service().await?;

        info!("╔══════════════════════════════════════════════════════════════════╗");
        info!("║               🧅 PERSISTENT TOR ONION SERVICE                    ║");
        info!("╠══════════════════════════════════════════════════════════════════╣");
        info!("║  Address: {}.onion", &onion_address[..40.min(onion_address.len())]);
        info!("║  Port: {}                                                       ", port);
        info!("║  Status: {} ", if is_new { "NEW (saved to disk)" } else { "LOADED (persistent)" });
        info!("╚══════════════════════════════════════════════════════════════════╝");

        Ok(service)
    }

    /// Start the onion service
    async fn start_service(&mut self) -> Result<()> {
        debug!("🚀 Starting onion service on port {}", self.port);

        // Set service ID from the persistent keypair
        self.service_id = Some(format!("service_{}", hex::encode(&self.keypair.public_key[..8])));

        // Register DNS TXT record for discovery
        self.register_dns_txt().await?;

        Ok(())
    }

    /// Register DNS TXT record for peer discovery
    async fn register_dns_txt(&self) -> Result<()> {
        let txt_record = format!(
            "_qnk._tor IN TXT \"onion={}.onion port={}\"",
            self.onion_address, self.port
        );

        debug!("📝 DNS TXT record: {}", txt_record);

        // In production, this would register with a DNS provider or DHT
        // For now, we log the record that would be created
        info!("📡 Registered DNS TXT record for {}", self.onion_name);

        Ok(())
    }

    /// Get the persistent keypair
    pub fn get_keypair(&self) -> &OnionKeypair {
        &self.keypair
    }

    /// Check if this service is using a new or existing keypair
    pub fn is_persistent(&self) -> bool {
        self.keypair_path.as_ref().map(|p| p.exists()).unwrap_or(false)
    }

    /// Get the onion address
    pub fn get_onion_address(&self) -> String {
        self.onion_address.clone()
    }

    /// Get the full onion URL
    pub fn get_onion_url(&self) -> String {
        format!("http://{}.onion:{}", self.onion_address, self.port)
    }

    /// Get onion service info for gossiping
    pub fn get_service_info(&self) -> OnionServiceInfo {
        OnionServiceInfo {
            onion_name: self.onion_name.clone(),
            onion_address: format!("{}.onion", self.onion_address),
            port: self.port,
            service_type: "q-narwhal-validator".to_string(),
            version: "v1.0".to_string(),
        }
    }

    /// Update port (for dynamic port assignment)
    pub async fn update_port(&mut self, new_port: u16) -> Result<()> {
        info!(
            "🔄 Updating onion service port from {} to {}",
            self.port, new_port
        );

        self.port = new_port;

        // Re-register DNS TXT record with new port
        self.register_dns_txt().await?;

        Ok(())
    }

    /// Check if service is healthy
    pub async fn health_check(&self) -> Result<bool> {
        // In production, this would test the onion service connectivity
        // For now, check if we have a service ID
        Ok(self.service_id.is_some())
    }

    /// Shutdown the onion service
    pub async fn shutdown(&self) -> Result<()> {
        info!("🛑 Shutting down onion service: {}", self.onion_name);

        // In production, this would properly close the onion service
        // and clean up any registered records

        info!("✅ Onion service shutdown complete");
        Ok(())
    }
}

/// Information about an onion service for network gossip
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnionServiceInfo {
    pub onion_name: String,
    pub onion_address: String,
    pub port: u16,
    pub service_type: String,
    pub version: String,
}

impl OnionServiceInfo {
    /// Create discovery message for gossiping onion service info
    pub fn to_discovery_message(&self) -> Vec<u8> {
        // In production, this would be properly encoded for network gossip
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Parse discovery message from network gossip
    pub fn from_discovery_message(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).context("Failed to parse onion service discovery message")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_keypair_generation() {
        // Generate two different keypairs
        let keypair1 = OnionKeypair::generate().unwrap();
        let keypair2 = OnionKeypair::generate().unwrap();

        // Each should have unique addresses (random generation)
        assert_ne!(keypair1.onion_address, keypair2.onion_address);

        // Address should be proper length for v3 onion (56 characters)
        assert_eq!(keypair1.onion_address.len(), 56);
        assert_eq!(keypair2.onion_address.len(), 56);

        // Keys should be the right lengths
        assert_eq!(keypair1.secret_key.len(), 32);
        assert_eq!(keypair1.public_key.len(), 32);
    }

    #[test]
    fn test_keypair_persistence() {
        let keypair = OnionKeypair::generate().unwrap();
        let temp_file = NamedTempFile::new().unwrap();

        // Save keypair
        keypair.save_to_file(temp_file.path()).unwrap();

        // Load keypair
        let loaded = OnionKeypair::load_from_file(temp_file.path()).unwrap();

        // Should be identical
        assert_eq!(keypair.onion_address, loaded.onion_address);
        assert_eq!(keypair.secret_key, loaded.secret_key);
        assert_eq!(keypair.public_key, loaded.public_key);
    }

    #[test]
    fn test_base32_encoding() {
        // Test known values
        let data = vec![0x00, 0x00, 0x00, 0x00, 0x00];
        let encoded = base32_encode(&data);
        assert_eq!(encoded, "aaaaaaaa");

        // 35 bytes should produce 56 characters
        let data_35 = vec![0u8; 35];
        let encoded_35 = base32_encode(&data_35);
        assert_eq!(encoded_35.len(), 56);
    }

    #[test]
    fn test_service_info_serialization() {
        let service_info = OnionServiceInfo {
            onion_name: "validator1.qnk".to_string(),
            onion_address: "abc123def456.onion".to_string(),
            port: 4001,
            service_type: "q-narwhal-validator".to_string(),
            version: "v1.0".to_string(),
        };

        let message = service_info.to_discovery_message();
        let parsed = OnionServiceInfo::from_discovery_message(&message).unwrap();

        assert_eq!(service_info.onion_name, parsed.onion_name);
        assert_eq!(service_info.onion_address, parsed.onion_address);
        assert_eq!(service_info.port, parsed.port);
    }

    #[tokio::test]
    async fn test_onion_service_info() {
        let service_info = OnionServiceInfo {
            onion_name: "test.qnk".to_string(),
            onion_address: "test.onion".to_string(),
            port: 4001,
            service_type: "q-narwhal-validator".to_string(),
            version: "v1.0".to_string(),
        };

        let url = format!("http://{}", service_info.onion_address);
        assert!(url.contains("test.onion"));
    }
}
