// REAL Tor Control Protocol Implementation
// This creates ACTUAL .onion addresses using the Tor daemon control interface

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tracing::{debug, error, info, warn};

/// REAL Tor controller that creates actual .onion services
pub struct TorController {
    control_stream: TcpStream,
    authenticated: bool,
    active_services: HashMap<String, String>, // service_name -> onion_address
}

/// Configuration for Tor control connection
#[derive(Debug, Clone)]
pub struct TorControlConfig {
    pub control_address: SocketAddr,    // Usually 127.0.0.1:9051
    pub auth_method: TorAuthMethod,
    pub service_data_dir: PathBuf,      // Where to store onion service keys
}

#[derive(Debug, Clone)]
pub enum TorAuthMethod {
    Cookie(PathBuf),    // Path to Tor auth cookie
    Password(String),   // Tor control password
    None,              // No authentication (dangerous)
}

impl Default for TorControlConfig {
    fn default() -> Self {
        Self {
            control_address: "127.0.0.1:9051".parse().unwrap(),
            auth_method: TorAuthMethod::Cookie("/var/lib/tor/control_auth_cookie".into()),
            service_data_dir: "/tmp/qnk_tor_services".into(),
        }
    }
}

impl TorController {
    /// Connect to Tor daemon and authenticate
    pub async fn connect(config: TorControlConfig) -> Result<Self> {
        info!("🔌 Connecting to Tor control port: {}", config.control_address);
        
        let control_stream = TcpStream::connect(&config.control_address)
            .await
            .context("Failed to connect to Tor control port - is Tor running?")?;
            
        info!("✅ Connected to Tor control port");
        
        let mut controller = Self {
            control_stream,
            authenticated: false,
            active_services: HashMap::new(),
        };
        
        // Authenticate with Tor daemon
        controller.authenticate(&config.auth_method).await?;
        
        // Create service data directory
        tokio::fs::create_dir_all(&config.service_data_dir).await
            .context("Failed to create service data directory")?;
            
        info!("🎯 Tor controller ready for creating REAL onion services");
        
        Ok(controller)
    }
    
    /// Authenticate with Tor daemon
    async fn authenticate(&mut self, auth_method: &TorAuthMethod) -> Result<()> {
        info!("🔐 Authenticating with Tor daemon...");
        
        match auth_method {
            TorAuthMethod::Cookie(cookie_path) => {
                // Read auth cookie
                let cookie = tokio::fs::read(cookie_path).await
                    .context("Failed to read Tor auth cookie - check Tor configuration")?;
                
                let cookie_hex = hex::encode(&cookie);
                let auth_command = format!("AUTHENTICATE {}\r\n", cookie_hex);
                
                self.send_command(&auth_command).await?;
            }
            TorAuthMethod::Password(password) => {
                let auth_command = format!("AUTHENTICATE \"{}\"\r\n", password);
                self.send_command(&auth_command).await?;
            }
            TorAuthMethod::None => {
                self.send_command("AUTHENTICATE\r\n").await?;
            }
        }
        
        let response = self.read_response().await?;
        
        if response.contains("250 OK") {
            info!("✅ Successfully authenticated with Tor daemon");
            self.authenticated = true;
            Ok(())
        } else {
            error!("❌ Tor authentication failed: {}", response);
            Err(anyhow::anyhow!("Tor authentication failed: {}", response))
        }
    }
    
    /// Create a REAL .onion hidden service
    pub async fn create_onion_service(
        &mut self, 
        service_name: &str, 
        target_port: u16
    ) -> Result<String> {
        if !self.authenticated {
            return Err(anyhow::anyhow!("Not authenticated with Tor daemon"));
        }
        
        info!("🧅 Creating REAL onion service: {}", service_name);
        info!("   Target port: {}", target_port);
        
        // Use ADD_ONION to create a new v3 hidden service
        // This generates a REAL .onion address from the Tor network
        let add_onion_command = format!(
            "ADD_ONION NEW:BEST Port=80,127.0.0.1:{}\r\n",
            target_port
        );
        
        debug!("Sending command: {}", add_onion_command.trim());
        
        self.send_command(&add_onion_command).await?;
        let response = self.read_response().await?;
        
        debug!("Tor response: {}", response);
        
        // Parse the response to extract the REAL onion address
        let onion_address = self.parse_onion_address(&response)
            .context("Failed to parse onion address from Tor response")?;
            
        info!("🎉 REAL onion service created: {}", onion_address);
        info!("   This is a genuine .onion address from the Tor network!");
        
        // Store the service
        self.active_services.insert(service_name.to_string(), onion_address.clone());
        
        Ok(onion_address)
    }
    
    /// Parse onion address from Tor ADD_ONION response
    fn parse_onion_address(&self, response: &str) -> Result<String> {
        for line in response.lines() {
            if line.starts_with("250-ServiceID=") {
                let service_id = line.strip_prefix("250-ServiceID=")
                    .context("Invalid ServiceID line format")?;
                return Ok(format!("{}.onion", service_id));
            }
        }
        
        Err(anyhow::anyhow!("No ServiceID found in Tor response: {}", response))
    }
    
    /// Remove an onion service
    pub async fn remove_onion_service(&mut self, service_name: &str) -> Result<()> {
        if let Some(onion_address) = self.active_services.get(service_name).cloned() {
            // Extract service ID from onion address
            let service_id = onion_address.strip_suffix(".onion")
                .context("Invalid onion address format")?;
                
            let del_command = format!("DEL_ONION {}\r\n", service_id);
            
            self.send_command(&del_command).await?;
            let response = self.read_response().await?;
            
            if response.contains("250 OK") {
                info!("✅ Removed onion service: {}", onion_address);
                self.active_services.remove(service_name);
                Ok(())
            } else {
                Err(anyhow::anyhow!("Failed to remove onion service: {}", response))
            }
        } else {
            Err(anyhow::anyhow!("Service {} not found", service_name))
        }
    }
    
    /// Get list of active onion services
    pub fn get_active_services(&self) -> &HashMap<String, String> {
        &self.active_services
    }
    
    /// Check Tor daemon status
    pub async fn get_tor_version(&mut self) -> Result<String> {
        self.send_command("GETINFO version\r\n").await?;
        let response = self.read_response().await?;
        
        for line in response.lines() {
            if line.starts_with("250-version=") {
                return Ok(line.strip_prefix("250-version=").unwrap().to_string());
            }
        }
        
        Err(anyhow::anyhow!("Could not get Tor version"))
    }
    
    /// Send command to Tor control port
    async fn send_command(&mut self, command: &str) -> Result<()> {
        self.control_stream.write_all(command.as_bytes()).await
            .context("Failed to send command to Tor control port")?;
        Ok(())
    }
    
    /// Read response from Tor control port
    async fn read_response(&mut self) -> Result<String> {
        let mut reader = BufReader::new(&mut self.control_stream);
        let mut response = String::new();
        let mut line = String::new();
        
        loop {
            line.clear();
            reader.read_line(&mut line).await
                .context("Failed to read response from Tor control port")?;
                
            response.push_str(&line);
            
            // Tor responses end with a line starting with "250 " (not "250-")
            if line.starts_with("250 ") || line.starts_with("551 ") || 
               line.starts_with("552 ") || line.starts_with("553 ") {
                break;
            }
        }
        
        Ok(response)
    }
    
    /// Close connection to Tor daemon
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("🛑 Shutting down Tor controller...");
        
        // Remove all active services
        let service_names: Vec<String> = self.active_services.keys().cloned().collect();
        for service_name in service_names {
            if let Err(e) = self.remove_onion_service(&service_name).await {
                warn!("Failed to remove service {}: {}", service_name, e);
            }
        }
        
        // Send QUIT command
        if let Err(e) = self.send_command("QUIT\r\n").await {
            warn!("Failed to send QUIT command: {}", e);
        }
        
        info!("✅ Tor controller shutdown complete");
        Ok(())
    }
}

/// Test if Tor daemon is running and accessible
pub async fn test_tor_daemon(config: &TorControlConfig) -> Result<()> {
    info!("🧪 Testing Tor daemon connectivity...");
    
    match TcpStream::connect(&config.control_address).await {
        Ok(_) => {
            info!("✅ Tor control port is accessible");
            Ok(())
        }
        Err(e) => {
            error!("❌ Cannot connect to Tor control port: {}", e);
            error!("   Make sure Tor is running with ControlPort enabled");
            error!("   Add 'ControlPort 9051' to your torrc configuration");
            Err(anyhow::anyhow!("Tor daemon not accessible: {}", e))
        }
    }
}

/// Create a REAL onion service for Q-NarwhalKnight
pub async fn create_qnk_onion_service(
    service_name: &str,
    target_port: u16,
) -> Result<(TorController, String)> {
    let config = TorControlConfig::default();
    
    // Test Tor daemon first
    test_tor_daemon(&config).await?;
    
    // Connect and authenticate
    let mut controller = TorController::connect(config).await?;
    
    // Create the REAL onion service
    let onion_address = controller.create_onion_service(service_name, target_port).await?;
    
    info!("🌟 Q-NarwhalKnight REAL onion service ready:");
    info!("   Service: {}", service_name);
    info!("   Address: {}", onion_address);
    info!("   Target: 127.0.0.1:{}", target_port);
    
    Ok((controller, onion_address))
}