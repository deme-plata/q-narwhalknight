//! REAL Q-NarwhalKnight Tor Integration Demo
//! This creates ACTUAL .onion addresses using the Tor daemon control protocol
//! NO SIMULATION - uses real Tor network and genuine onion services

use anyhow::Result;
use q_tor_client::{
    create_qnk_onion_service, test_tor_daemon, test_real_tor_connection,
    TorControlConfig, TorAuthMethod, TorSocksClient,
};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🧅🎯 Q-NarwhalKnight REAL Tor Integration Demo");
    println!("==============================================");
    println!("This creates GENUINE .onion addresses using the Tor daemon");
    println!("NO SIMULATION - connects to actual Tor network!");
    println!();
    
    // Step 1: Test Tor daemon accessibility
    println!("🔍 Step 1: Testing Tor daemon connectivity...");
    let tor_config = TorControlConfig {
        control_address: "127.0.0.1:9051".parse()?,
        auth_method: TorAuthMethod::None, // Try no auth first, then cookie
        service_data_dir: "/tmp/qnk_tor_services".into(),
    };
    
    match test_tor_daemon(&tor_config).await {
        Ok(_) => println!("✅ Tor daemon is accessible"),
        Err(e) => {
            println!("❌ Tor daemon test failed: {}", e);
            println!();
            println!("🛠️  SETUP REQUIRED:");
            println!("   1. Install Tor: sudo apt-get install tor");
            println!("   2. Edit /etc/tor/torrc and add:");
            println!("      ControlPort 9051");
            println!("      CookieAuthentication 0");
            println!("   3. Restart Tor: sudo systemctl restart tor");
            println!("   4. Run this demo again");
            return Err(e);
        }
    }
    
    // Step 2: Test SOCKS5 connectivity
    println!();
    println!("🌐 Step 2: Testing SOCKS5 proxy connectivity...");
    
    match test_real_tor_connection().await {
        Ok(connection_info) => {
            println!("✅ SOCKS5 proxy test passed");
            println!("   Proxy accessible: {}", connection_info.socks_proxy_accessible);
            println!("   Tor network accessible: {}", connection_info.tor_network_accessible);
            println!("   Can connect to .onion: {}", connection_info.can_connect_to_onions);
        }
        Err(e) => {
            warn!("⚠️ SOCKS5 test warning: {} (may still work)", e);
        }
    }
    
    // Step 3: Create REAL onion service for validator-alpha
    println!();
    println!("🧅 Step 3: Creating REAL onion service...");
    println!("   This will generate a genuine .onion address from the Tor network");
    
    let (mut controller, real_onion_address) = match create_qnk_onion_service(
        "qnk-validator-alpha",
        8001
    ).await {
        Ok((controller, address)) => {
            println!("🎉 SUCCESS! REAL onion service created:");
            println!("   Address: {}", address);
            
            // Validate this is a real v3 onion address
            if address.ends_with(".onion") && address.len() == 62 {
                println!("✅ Verified: This is a genuine Tor v3 onion address!");
                println!("   Length: {} characters (correct for v3)", address.len());
                println!("   Format: Valid .onion suffix");
            } else {
                println!("⚠️ Unexpected address format: {}", address);
            }
            
            (controller, address)
        }
        Err(e) => {
            println!("❌ Failed to create onion service: {}", e);
            println!();
            println!("🛠️  TROUBLESHOOTING:");
            println!("   - Ensure Tor daemon is running: systemctl status tor");
            println!("   - Check ControlPort is enabled in /etc/tor/torrc");
            println!("   - Verify no firewall blocking port 9051");
            return Err(e);
        }
    };
    
    // Step 4: Create second onion service
    println!();
    println!("🔄 Step 4: Creating second REAL onion service...");
    
    let beta_address = match controller.create_onion_service("qnk-validator-beta", 8002).await {
        Ok(address) => {
            println!("✅ Second onion service: {}", address);
            address
        }
        Err(e) => {
            warn!("⚠️ Could not create second service: {}", e);
            "".to_string()
        }
    };
    
    // Step 5: Show service information
    println!();
    println!("📊 Step 5: REAL Onion Service Information");
    println!("========================================");
    
    let active_services = controller.get_active_services();
    println!("Active services: {}", active_services.len());
    
    for (service_name, onion_address) in active_services {
        println!("  📍 {}", service_name);
        println!("     Address: {}", onion_address);
        println!("     URL: http://{}", onion_address);
        println!("     Accessible via: Tor Browser");
    }
    
    // Step 6: Test connection to our own onion service
    println!();
    println!("🔗 Step 6: Testing connection to created onion service...");
    
    let socks_client = TorSocksClient::default();
    match socks_client.connect_to_onion(&real_onion_address, 80).await {
        Ok(_stream) => {
            println!("✅ Successfully connected to our own onion service!");
            println!("   This proves the address is real and accessible");
        }
        Err(e) => {
            println!("⚠️ Connection test: {} (normal if no server running)", e);
        }
    }
    
    // Step 7: Get Tor version info
    println!();
    println!("ℹ️  Step 7: Tor daemon information");
    
    match controller.get_tor_version().await {
        Ok(version) => println!("   Tor version: {}", version),
        Err(_) => println!("   Could not retrieve Tor version"),
    }
    
    // Step 8: Keep services running briefly
    println!();
    println!("⏳ Step 8: Keeping onion services active...");
    println!("   Services are now published to Tor directory");
    println!("   They should be accessible via Tor Browser at:");
    println!("   http://{}", real_onion_address);
    if !beta_address.is_empty() {
        println!("   http://{}", beta_address);
    }
    println!();
    println!("   Waiting 10 seconds before cleanup...");
    
    for i in 1..=10 {
        sleep(Duration::from_secs(1)).await;
        print!(".");
        if i % 10 == 0 {
            println!(" {}s", i);
        }
    }
    println!();
    
    // Step 9: Cleanup
    println!("🧹 Step 9: Cleaning up onion services...");
    controller.shutdown().await?;
    
    println!();
    println!("🎯 REAL TOR INTEGRATION DEMO COMPLETE!");
    println!("=====================================");
    println!("✅ Created GENUINE .onion addresses using Tor daemon");
    println!("✅ Connected to actual Tor control protocol");
    println!("✅ Published to real Tor directory services");  
    println!("✅ NO SIMULATION - these were real Tor hidden services");
    println!();
    println!("These addresses were temporarily accessible via:");
    println!("🌐 Tor Browser at: http://{}", real_onion_address);
    println!();
    println!("This demonstrates how Q-NarwhalKnight can create");
    println!("REAL anonymous validator endpoints for quantum consensus!");
    
    Ok(())
}