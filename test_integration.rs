/// Test the high-level developer integration API
use q_narwhalknight::DNSPhantomMesh;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    println!("🧪 Testing Q-NarwhalKnight high-level API...");
    
    // Test the 3-line integration that developers will use
    let mesh = DNSPhantomMesh::new().await?;
    mesh.start_autonomous_discovery().await?;
    mesh.connect_discovered_peers().await?;
    
    // Test status and metrics
    println!("🎉 Mesh network operational: {} peers", mesh.peer_count().await);
    println!("📊 Status: {}", mesh.status_string().await);
    
    let health = mesh.mesh_health().await;
    println!("🏥 Health: discovered={}, connected={}, anomalies={}", 
             health.discovered_peer_count, 
             health.connected_peer_count, 
             health.dns_anomaly_count);
    
    println!("✅ High-level API test completed successfully!");
    println!("🌟 Developers can now integrate DNS-Phantom mesh networking with 3 lines of code!");
    
    Ok(())
}