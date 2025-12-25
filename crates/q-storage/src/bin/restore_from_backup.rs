/// Quick restore utility for Q-NarwhalKnight
/// Restores blocks from backup files
use anyhow::Result;
use std::path::PathBuf;
use q_storage::CheckpointStorage;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔧 Q-NarwhalKnight Backup Restore Utility");
    println!("==========================================\n");

    // Get database path from environment or use default
    let db_path = std::env::var("Q_DB_PATH").unwrap_or_else(|_| "./data-mine16".to_string());
    let target_height: u64 = std::env::var("RESTORE_HEIGHT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(61147);

    println!("📂 Database path: {}", db_path);
    println!("🎯 Target height: {}\n", target_height);

    // Read backup manifests
    let manifests_path = PathBuf::from(&db_path).join("backup_manifests.json");
    println!("📋 Reading manifests from: {}", manifests_path.display());

    let manifests_data = tokio::fs::read_to_string(&manifests_path).await?;
    let manifests: Vec<serde_json::Value> = serde_json::from_str(&manifests_data)?;
    println!("   Found {} backup manifests\n", manifests.len());

    // Open database with all column families
    println!("🔓 Opening database...");
    let config = q_storage::StorageConfig {
        db_path: db_path.clone(),
        hot_db_path: format!("{}/hot", db_path),
        enable_metrics: false,
        sync_writes: true,
        cache_size_mb: 512,
        max_open_files: 1000,
    };
    let storage = q_storage::QStorage::new(config).await?;
    println!("✅ Database opened\n");

    // Get current height
    let current_height = storage.get_latest_height().await.unwrap_or(0);
    println!("📊 Current database height: {}", current_height);

    if current_height >= target_height {
        println!("✅ Already at or above target height. Nothing to restore.");
        return Ok(());
    }

    println!("\n📥 Starting restore from height {} to {}\n", current_height + 1, target_height);

    // Find and restore from needed backups
    let mut restored_count = 0u64;
    for manifest in &manifests {
        let start = manifest["start_height"].as_u64().unwrap_or(0);
        let end = manifest["end_height"].as_u64().unwrap_or(0);
        let ipfs_cid = manifest["ipfs_cid"].as_str().unwrap_or("");
        let checksum = manifest["checksum"].as_str().unwrap_or("");

        // Skip if already have these blocks
        if end <= current_height {
            continue;
        }

        // Skip if beyond target
        if start > target_height {
            continue;
        }

        // Load backup file
        let backup_filename = if ipfs_cid.starts_with("local://") {
            &ipfs_cid[8..]
        } else {
            continue; // Skip IPFS backups for now
        };

        let backup_path = PathBuf::from(&db_path).join("backups").join(backup_filename);
        println!("📦 Restoring {} (heights {}-{})...", backup_filename, start, end);

        let data = match tokio::fs::read(&backup_path).await {
            Ok(d) => d,
            Err(e) => {
                println!("   ❌ Failed to read backup: {}", e);
                continue;
            }
        };

        // Verify checksum
        let computed = hex::encode(blake3::hash(&data).as_bytes());
        if computed != checksum {
            println!("   ❌ Checksum mismatch! Skipping.");
            continue;
        }

        // Import blocks
        match storage.import_blocks(&data).await {
            Ok(count) => {
                println!("   ✅ Imported {} blocks", count);
                restored_count += count;
            }
            Err(e) => {
                println!("   ❌ Import failed: {}", e);
            }
        }
    }

    // Update height pointer
    let final_height = storage.get_latest_height().await.unwrap_or(0);
    println!("\n✅ Restore complete!");
    println!("   Restored {} blocks", restored_count);
    println!("   Final height: {}", final_height);

    Ok(())
}
