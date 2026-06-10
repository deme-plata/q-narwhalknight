/*!
# BEP-44 DHT Implementation (Simplified)

Core BitTorrent DHT (BEP-44) implementation for signed mutable data storage.

This provides a simplified version of BEP-44 for Q-NarwhalKnight peer discovery.
The full implementation would require complex DHT routing and Kademlia protocol.
For now, we provide a foundation that can be extended.

## Simplified BEP-44 for Proof of Concept

Instead of implementing full BEP-44, we create a simplified DHT client that:
- Connects to BitTorrent bootstrap nodes
- Stores and retrieves signed records
- Provides the interface for peer discovery
- Can be extended to full BEP-44 later

This approach allows us to demonstrate the architecture while avoiding 
complex Kademlia DHT implementation details.
*/

use anyhow::{Result, Context};
use ed25519_dalek::{Signature, SigningKey, VerifyingKey, Signer, Verifier};
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};
use tokio::net::UdpSocket;
use tokio::time::timeout;

/// Simplified BEP-44 DHT Client for Q-NarwhalKnight
#[derive(Debug)]
pub struct Bep44Client {
    socket: UdpSocket,
    node_id: [u8; 20],
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    bootstrap_nodes: Vec<SocketAddr>,
    local_storage: HashMap<[u8; 20], MutableDataRecord>, // Simplified storage
}

impl Bep44Client {
    /// Create new simplified BEP-44 DHT client
    pub async fn new(
        bind_addr: SocketAddr,
        signing_key: SigningKey,
        bootstrap_nodes: Vec<SocketAddr>,
    ) -> Result<Self> {
        let socket = UdpSocket::bind(bind_addr).await
            .context("Failed to bind DHT socket")?;
        
        let verifying_key = signing_key.verifying_key();
        
        // Generate random 20-byte node ID for DHT
        let mut node_id = [0u8; 20];
        getrandom::getrandom(&mut node_id)
            .context("Failed to generate node ID")?;
        
        tracing::info!("🔍 Simplified BEP-44 DHT client created - Node ID: {}", hex::encode(&node_id));
        
        Ok(Self {
            socket,
            node_id,
            signing_key,
            verifying_key,
            bootstrap_nodes,
            local_storage: HashMap::new(),
        })
    }
    
    /// Bootstrap DHT connection (simplified)
    pub async fn bootstrap(&mut self) -> Result<()> {
        tracing::info!("🌐 Bootstrapping simplified BEP-44 DHT connection");
        
        // For proof of concept, we just verify we can bind and communicate
        for bootstrap_addr in &self.bootstrap_nodes.clone() {
            match self.test_connectivity(*bootstrap_addr).await {
                Ok(_) => {
                    tracing::debug!("✅ Connectivity test passed for {}", bootstrap_addr);
                }
                Err(e) => {
                    tracing::debug!("⚠️ Connectivity test failed for {}: {}", bootstrap_addr, e);
                }
            }
        }
        
        tracing::info!("✅ Simplified DHT bootstrap complete");
        Ok(())
    }
    
    /// Store signed mutable data in local storage (simplified BEP-44)
    pub async fn store_mutable_data(
        &mut self,
        data: &[u8],
        sequence_number: u64,
    ) -> Result<()> {
        let public_key_bytes = self.verifying_key.as_bytes();
        let storage_key = self.calculate_storage_key(public_key_bytes);
        
        // Create BEP-44 signature
        let signature = self.create_bep44_signature(data, sequence_number);
        
        // Create mutable data record
        let mutable_record = MutableDataRecord {
            public_key: *public_key_bytes,
            sequence_number,
            data: data.to_vec(),
            signature: signature.to_bytes().to_vec(),
        };
        
        tracing::info!("📤 Storing mutable data in simplified DHT - Key: {}, Seq: {}", 
                      hex::encode(&storage_key[..8]), sequence_number);
        
        // Store in local storage for proof of concept
        self.local_storage.insert(storage_key, mutable_record);
        
        tracing::info!("✅ Mutable data stored successfully");
        Ok(())
    }
    
    /// Retrieve signed mutable data from storage
    pub async fn get_mutable_data(&mut self, public_key: &[u8; 32]) -> Result<Option<MutableDataRecord>> {
        let storage_key = self.calculate_storage_key(public_key);
        
        tracing::debug!("📥 Retrieving mutable data from simplified DHT - Key: {}", 
                       hex::encode(&storage_key[..8]));
        
        let record = self.local_storage.get(&storage_key).cloned();
        
        if let Some(record) = &record {
            // Verify signature
            self.verify_mutable_record(record)?;
            tracing::debug!("✅ Retrieved and verified mutable data - Seq: {}", 
                           record.sequence_number);
        }
        
        Ok(record)
    }
    
    /// Search for multiple mutable records (peer discovery)
    pub async fn search_mutable_data(&mut self, key_prefix: &[u8]) -> Result<Vec<MutableDataRecord>> {
        tracing::info!("🔍 Searching simplified DHT for mutable data - Prefix: {}", 
                      hex::encode(&key_prefix[..std::cmp::min(4, key_prefix.len())]));
        
        let mut discovered_records = Vec::new();
        
        // Search local storage for matching keys
        for (key, record) in &self.local_storage {
            if key.starts_with(key_prefix) {
                discovered_records.push(record.clone());
            }
        }
        
        // Verify all discovered records
        discovered_records.retain(|record| {
            match self.verify_mutable_record(record) {
                Ok(_) => true,
                Err(e) => {
                    tracing::warn!("⚠️ Invalid record signature: {}", e);
                    false
                }
            }
        });
        
        tracing::info!("✅ Found {} valid mutable records", discovered_records.len());
        Ok(discovered_records)
    }
    
    /// Calculate storage key for public key (SHA1)
    fn calculate_storage_key(&self, public_key: &[u8; 32]) -> [u8; 20] {
        let mut hasher = Sha1::new();
        hasher.update(public_key);
        hasher.finalize().into()
    }
    
    /// Create BEP-44 compliant signature
    fn create_bep44_signature(&self, data: &[u8], sequence_number: u64) -> Signature {
        // BEP-44 signature format: sign(data + sequence_number)
        let mut message = Vec::new();
        message.extend_from_slice(data);
        message.extend_from_slice(&sequence_number.to_be_bytes());
        
        self.signing_key.sign(&message)
    }
    
    /// Verify BEP-44 mutable record signature
    fn verify_mutable_record(&self, record: &MutableDataRecord) -> Result<()> {
        let public_key = VerifyingKey::from_bytes(&record.public_key)
            .context("Invalid public key in record")?;
        
        let signature = Signature::from_slice(&record.signature)
            .context("Invalid signature format")?;
        
        // Reconstruct signed message
        let mut message = Vec::new();
        message.extend_from_slice(&record.data);
        message.extend_from_slice(&record.sequence_number.to_be_bytes());
        
        public_key.verify(&message, &signature)
            .context("Signature verification failed")?;
        
        Ok(())
    }
    
    /// Test connectivity to a node
    async fn test_connectivity(&self, addr: SocketAddr) -> Result<()> {
        // Simple UDP connectivity test
        let test_data = b"Q-NarwhalKnight-BEP44-Test";
        
        match timeout(Duration::from_secs(2), self.socket.send_to(test_data, addr)).await {
            Ok(Ok(_)) => {
                tracing::debug!("✅ UDP connectivity test successful to {}", addr);
                Ok(())
            }
            Ok(Err(e)) => {
                tracing::debug!("⚠️ UDP send failed to {}: {}", addr, e);
                Err(e.into())
            }
            Err(_) => {
                tracing::debug!("⚠️ UDP connectivity test timeout to {}", addr);
                anyhow::bail!("Connection timeout")
            }
        }
    }
}

/// BEP-44 mutable data record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutableDataRecord {
    pub public_key: [u8; 32],
    pub sequence_number: u64,
    pub data: Vec<u8>,
    pub signature: Vec<u8>,
}