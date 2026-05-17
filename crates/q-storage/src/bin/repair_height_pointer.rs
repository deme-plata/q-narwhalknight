/// Repair tool for v0.9.99-beta height pointer desync issue
/// Updates qblock:latest pointer to match actual highest block

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🔧 Q-NarwhalKnight Height Pointer Repair Utility");
    println!("=================================================\n");

    // Get database path from environment or use default
    let db_path = std::env::var("Q_DB_PATH").unwrap_or_else(|_| "./data-mine10".to_string());
    println!("📂 Database path: {}\n", db_path);

    // Open database
    println!("🔓 Opening database...");
    let config = q_storage::StorageConfig {
        db_path: db_path.clone(),
        hot_db_path: format!("{}/hot", db_path),
        enable_metrics: false,
        sync_writes: false,
        cache_size_mb: 512,
        max_open_files: 1000,
    };
    let storage = q_storage::QStorage::new(config).await?;
    println!("✅ Database opened\n");

    // Use built-in repair method
    println!("🔧 Running height pointer repair...\n");
    let repaired_height = storage.repair_height_pointer().await?;

    println!("\n✅ Repair complete - height pointer is now at: {}", repaired_height);
    println!("   You can now restart the q-api-server service");

    Ok(())
}
