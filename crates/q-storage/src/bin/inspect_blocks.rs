/// Database inspection tool for v0.9.99-beta height stuck issue
/// Checks blocks 690-710 and height pointer

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🔍 Q-NarwhalKnight Database Inspection");
    println!("======================================\n");

    // Get database path from environment or use default
    let db_path = std::env::var("Q_DB_PATH").unwrap_or_else(|_| "./data-mine10".to_string());
    println!("Database path: {}\n", db_path);

    // Open database
    println!("Opening database...");
    let config = q_storage::StorageConfig {
        db_path: db_path.clone(),
        hot_db_path: format!("{}/hot", db_path),
        enable_metrics: false,
        sync_writes: false,
        cache_size_mb: 512,
        max_open_files: 1000,
    };
    let storage = q_storage::QStorage::new(config).await?;

    println!("\n📊 Checking blocks 690-710:");
    println!("----------------------------");

    for height in 690..=710 {
        match storage.get_qblock_by_height(height).await {
            Ok(Some(block)) => {
                let hash = block.calculate_hash();
                let hash_hex = hex::encode(&hash[..8]);
                println!("✅ Block {} exists (hash: {}, solutions: {})",
                         height, hash_hex, block.mining_solutions.len());
            }
            Ok(None) => {
                println!("❌ Block {} NOT FOUND", height);
            }
            Err(e) => {
                println!("⚠️  Block {} ERROR: {}", height, e);
            }
        }
    }

    println!("\n📍 Height pointer:");
    println!("------------------");
    match storage.get_latest_qblock_height().await {
        Ok(Some(height)) => {
            println!("✅ Latest height = {}", height);
        }
        Ok(None) => {
            println!("⚠️  No height pointer found (empty database?)");
        }
        Err(e) => {
            println!("⚠️  Failed to get latest height: {}", e);
        }
    }

    println!("\n📈 Highest contiguous block:");
    println!("----------------------------");
    match storage.get_highest_contiguous_block().await {
        Ok(height) => {
            println!("✅ Highest contiguous = {}", height);
        }
        Err(e) => {
            println!("⚠️  Failed to get highest contiguous: {}", e);
        }
    }

    println!("\n✅ Inspection complete");

    Ok(())
}
