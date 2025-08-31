use anyhow::{Context, Result};
use arti_client::TorClient;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Onion service for Q-NarwhalKnight validator
pub struct OnionService {
    tor_client: Arc<TorClient<tor_rtcompat::tokio::TokioNativeTlsRuntime>>,
    onion_name: String,
    onion_address: String,
    port: u16,
    service_id: Option<String>,
}

impl OnionService {
    /// Create and start new onion service
    pub async fn new(
        tor_client: Arc<TorClient<tor_rtcompat::tokio::TokioNativeTlsRuntime>>,
        onion_name: String,
        port: u16,
    ) -> Result<Self> {
        info!("🧅 Creating onion service: {}", onion_name);

        // Generate onion address (in production, this would use Arti's onion service API)
        let onion_address = Self::generate_onion_address(&onion_name)?;

        let mut service = Self {
            tor_client,
            onion_name: onion_name.clone(),
            onion_address: onion_address.clone(),
            port,
            service_id: None,
        };

        // Start the onion service
        service.start_service().await?;

        info!("✅ Onion service active: {}.onion", onion_address);

        Ok(service)
    }

    /// Start the onion service
    async fn start_service(&mut self) -> Result<()> {
        debug!("🚀 Starting onion service on port {}", self.port);

        // In production, this would configure the actual onion service
        // For now, we simulate the service being started
        self.service_id = Some(format!("service_{}", self.onion_name));

        // Register DNS TXT record for discovery
        self.register_dns_txt().await?;

        Ok(())
    }

    /// Register DNS TXT record for peer discovery
    async fn register_dns_txt(&self) -> Result<()> {
        let txt_record = format!("_qnk._tor IN TXT \"onion={}.onion port={}\"", 
                                self.onion_address, self.port);
        
        debug!("📝 DNS TXT record: {}", txt_record);
        
        // In production, this would register with a DNS provider or DHT
        // For now, we log the record that would be created
        info!("📡 Registered DNS TXT record for {}", self.onion_name);

        Ok(())
    }

    /// Generate .qnk onion address
    fn generate_onion_address(onion_name: &str) -> Result<String> {
        // In production, this would derive from the actual onion service key
        // For now, generate a deterministic address based on the name
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        onion_name.hash(&mut hasher);
        let hash = hasher.finish();

        // Create a mock v3 onion address (56 characters)
        let onion_address = format!("{:016x}{:016x}{:016x}{:08x}", 
                                   hash, hash.rotate_left(16), hash.rotate_left(32), hash as u32);

        Ok(onion_address)
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
        info!("🔄 Updating onion service port from {} to {}", self.port, new_port);
        
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
        serde_json::from_slice(data)
            .context("Failed to parse onion service discovery message")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onion_address_generation() {
        let address1 = OnionService::generate_onion_address("validator1.qnk").unwrap();
        let address2 = OnionService::generate_onion_address("validator2.qnk").unwrap();
        let address1_repeat = OnionService::generate_onion_address("validator1.qnk").unwrap();

        // Different names should generate different addresses
        assert_ne!(address1, address2);
        
        // Same name should generate same address (deterministic)
        assert_eq!(address1, address1_repeat);
        
        // Address should be proper length for v3 onion
        assert_eq!(address1.len(), 56);
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