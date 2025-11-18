/// Emergency fix for corrupted qblock:latest pointer (u64::MAX bug)
/// Scans for highest contiguous block and updates pointer

use q_storage::{QStorage, StorageConfig, KVStore};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🔧 Q-NarwhalKnight Emergency Pointer Fix - v1.0.17-beta");
    println!("=======================================================\n");

    // Get database path from environment or use default
    let db_path = std::env::var("Q_DB_PATH").unwrap_or_else(|_| "./data-mine12".to_string());
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
    let current_pointer_bytes = storage.get_hot_db().get("blocks", b"qblock:latest").await?;
    let current_pointer = if let Some(bytes) = current_pointer_bytes {
        if bytes.len() >= 8 {
            u64::from_be_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7]
            ])
        } else {
            0
        }
    } else {
        0
    };
    println!("   qblock:latest pointer: {}", current_pointer);

    if current_pointer == 18446744073709551615 {
        println!("   ⚠️  CORRUPTED: Pointer is u64::MAX!");
    }

    // Scan for highest contiguous block using binary search
    println!("\n🔍 Scanning for highest contiguous block...");
    let mut low = 0u64;
    let mut high = if current_pointer == 18446744073709551615 {
        10000u64 // Reasonable upper bound for scanning
    } else {
        current_pointer
    };

    let mut highest_found = 0u64;

    // Binary search for highest block
    while low <= high && high < 1000000 {
        let mid = low + (high - low) / 2;

        let exists = storage.get_qblock_by_height(mid).await?.is_some();

        if exists {
            highest_found = mid;
            low = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            high = mid - 1;
        }
    }

    println!("   Highest contiguous block found: {}", highest_found);

    // Verify blocks exist around highest_found
    if highest_found > 0 {
        println!("\n📋 Verifying blocks {} to {}...", highest_found.saturating_sub(2), highest_found);
        for height in highest_found.saturating_sub(2)..=highest_found {
            match storage.get_qblock_by_height(height).await {
                Ok(Some(block)) => {
                    let hash = block.calculate_hash();
                    let hash_hex = hex::encode(&hash[..8]);
                    println!("   ✅ Block {} exists (hash: {})", height, hash_hex);
                }
                Ok(None) => {
                    eprintln!("   ❌ Block {} NOT FOUND!", height);
                }
                Err(e) => {
                    eprintln!("   ⚠️  Error checking block {}: {}", height, e);
                }
            }
        }
    }

    // Update pointer to highest_found
    println!("\n🔧 Updating qblock:latest pointer to {}...", highest_found);

    // Direct database write with proper bytes
    let height_bytes = highest_found.to_be_bytes();
    storage.get_hot_db().put("blocks", b"qblock:latest", &height_bytes).await?;

    // Verify update
    let new_pointer_bytes = storage.get_hot_db().get("blocks", b"qblock:latest").await?;
    let new_pointer = if let Some(bytes) = new_pointer_bytes {
        if bytes.len() >= 8 {
            u64::from_be_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7]
            ])
        } else {
            0
        }
    } else {
        0
    };

    if new_pointer == highest_found {
        println!("✅ Success! qblock:latest pointer updated: {} → {}", current_pointer, new_pointer);
    } else {
        eprintln!("❌ Update failed! Expected {}, got {}", highest_found, new_pointer);
        return Err(anyhow::anyhow!("Pointer update verification failed"));
    }

    println!("\n✅ Database repair complete!");
    println!("   Blocks preserved: 0 to {}", highest_found);
    println!("   You can now restart the q-api-server service");
    if highest_found > 0 {
        println!("   Service should resume from block {}", highest_found + 1);
    } else {
        println!("   Service will start fresh from block 0");
    }

    Ok(())
}
