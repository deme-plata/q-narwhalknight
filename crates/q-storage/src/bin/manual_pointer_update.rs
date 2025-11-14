/// Manual height pointer update tool for v0.9.99-beta height desync
/// Directly updates qblock:latest to 710

use q_storage::{QStorage, StorageConfig, KVStore};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🔧 Q-NarwhalKnight Manual Height Pointer Update");
    println!("===============================================\n");

    // Get database path from environment or use default
    let db_path = std::env::var("Q_DB_PATH").unwrap_or_else(|_| "./data-mine10".to_string());
    println!("📂 Database path: {}\n", db_path);

    // Open database
    println!("🔓 Opening database...");
    let config = StorageConfig {
        db_path: db_path.clone(),
        hot_db_path: format!("{}/hot", db_path),
        enable_metrics: false,
        sync_writes: false,
        cache_size_mb: 512,
        max_open_files: 1000,
    };
    let storage = QStorage::new(config).await?;
    println!("✅ Database opened\n");

    // Get current pointer value
    println!("📍 Current state:");
    let current_pointer = storage.get_latest_qblock_height().await?.unwrap_or(0);
    println!("   Height pointer: {}", current_pointer);

    // Verify blocks 704-710 exist
    println!("\n🔍 Verifying blocks 704-710 exist...");
    for height in 704..=710 {
        match storage.get_qblock_by_height(height).await {
            Ok(Some(block)) => {
                let hash = block.calculate_hash();
                let hash_hex = hex::encode(&hash[..8]);
                println!("   ✅ Block {} exists (hash: {})", height, hash_hex);
            }
            Ok(None) => {
                eprintln!("   ❌ Block {} NOT FOUND - cannot update pointer!", height);
                return Err(anyhow::anyhow!("Block {} not found", height));
            }
            Err(e) => {
                eprintln!("   ⚠️  Error checking block {}: {}", height, e);
                return Err(e.into());
            }
        }
    }

    // Update pointer to 710
    let target_height = 710u64;
    println!("\n🔧 Updating height pointer to {}...", target_height);

    // Direct database write
    let height_bytes = target_height.to_be_bytes();
    storage.get_hot_db().put("blocks", b"qblock:latest", &height_bytes).await?;

    // Verify update
    let new_pointer = storage.get_latest_qblock_height().await?.unwrap_or(0);
    if new_pointer == target_height {
        println!("✅ Success! Height pointer updated: {} → {}", current_pointer, new_pointer);
    } else {
        eprintln!("❌ Update failed! Expected {}, got {}", target_height, new_pointer);
        return Err(anyhow::anyhow!("Height pointer update verification failed"));
    }

    println!("\n✅ Repair complete");
    println!("   You can now restart the q-api-server service");
    println!("   Service should resume from block 711");

    Ok(())
}
