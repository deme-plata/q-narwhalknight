use q_narwhalknight::DNSPhantomMesh;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("🌐 Starting basic DNS-Phantom mesh network...");

    let mesh = DNSPhantomMesh::new().await?;
    mesh.start_autonomous_discovery().await?;
    mesh.connect_discovered_peers().await?;

    println!(
        "🎉 Mesh network operational: {} peers",
        mesh.peer_count().await
    );
    println!("📊 Status: {}", mesh.status_string().await);

    tokio::time::sleep(std::time::Duration::from_secs(60)).await;

    println!("🏁 Basic mesh demo completed!");
    Ok(())
}
