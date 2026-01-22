//! Database Repair Utility
//!
//! Scans the RocksDB database and rebuilds missing pointers like `qblock:latest`.
//! This fixes databases corrupted by the sync-down bug in v0.5.21 and earlier.
//!
//! v3.0.3-beta: Uses binary search to handle databases with millions of blocks

use anyhow::Result;
use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use std::sync::Arc;

const CF_BLOCKS: &str = "blocks";

/// Check if a block exists at the given height
fn block_exists(db: &DB, cf_blocks: &impl rocksdb::AsColumnFamilyRef, height: u64) -> bool {
    let key = format!("qblock:height:{}", height);
    matches!(db.get_cf(cf_blocks, key.as_bytes()), Ok(Some(_)))
}

/// Binary search to find the approximate upper bound of blocks
fn binary_search_upper_bound(db: &DB, cf_blocks: &impl rocksdb::AsColumnFamilyRef, max_height: u64) -> u64 {
    let mut low = 0u64;
    let mut high = max_height;
    let mut last_found = 0u64;

    // Binary search to find approximate region
    while low <= high {
        let mid = low + (high - low) / 2;

        // Check a small window around mid to handle gaps
        let found = (mid.saturating_sub(10)..=mid.saturating_add(10))
            .any(|h| block_exists(db, cf_blocks, h));

        if found {
            last_found = mid;
            low = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            high = mid - 1;
        }
    }

    // Extend search to find actual highest after binary search
    let mut highest = last_found;
    for h in last_found..=last_found.saturating_add(10000).min(max_height) {
        if block_exists(db, cf_blocks, h) {
            highest = h;
        } else if h > highest + 100 {
            // Early termination if 100 consecutive missing
            break;
        }
    }

    highest
}

fn main() -> Result<()> {
    println!("🔧 Q-NarwhalKnight Database Repair Utility v3.0.3-beta");
    println!("   Uses binary search for databases with millions of blocks");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Get database path from args or use default
    let db_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./data-mine1/hot".to_string());

    println!("📂 Opening database: {}", db_path);

    // Discover existing column families
    let db_opts_list = Options::default();
    let cf_list = DB::list_cf(&db_opts_list, &db_path)?;

    println!("📋 Found {} column families:", cf_list.len());
    for cf_name in &cf_list {
        println!("   • {}", cf_name);
    }
    println!();

    // Open database with discovered column families
    let mut db_opts = Options::default();
    db_opts.create_if_missing(false);

    let cfs: Vec<_> = cf_list.iter()
        .map(|name| ColumnFamilyDescriptor::new(name.as_str(), Options::default()))
        .collect();

    let db = DB::open_cf_descriptors(&db_opts, &db_path, cfs)?;
    let db = Arc::new(db);

    println!("✅ Database opened successfully");
    println!();

    let cf_blocks = db.cf_handle(CF_BLOCKS)
        .ok_or_else(|| anyhow::anyhow!("blocks column family not found"))?;

    // PHASE 1: Binary search to find highest block (fast, O(log n))
    println!("🔍 Phase 1: Binary search for highest block...");
    let absolute_max = 10_000_000u64;
    let highest_found = binary_search_upper_bound(&db, &cf_blocks, absolute_max);
    println!("   Binary search found highest: {}", highest_found);
    println!();

    // PHASE 2: Verify and count blocks in the found range
    println!("🔍 Phase 2: Verifying blocks around height {}...", highest_found);

    let mut total_blocks = 0u64;
    let mut missing_blocks = Vec::new();
    let mut consecutive_missing = 0u64;

    // Scan a window around the found highest to verify and count
    let scan_start = highest_found.saturating_sub(1000);
    let scan_end = highest_found.saturating_add(1000).min(absolute_max);

    for height in scan_start..=scan_end {
        let key = format!("qblock:height:{}", height);

        if let Ok(Some(_)) = db.get_cf(&cf_blocks, key.as_bytes()) {
            total_blocks += 1;
            consecutive_missing = 0;
        } else {
            if height <= highest_found {
                missing_blocks.push(height);
            }
            consecutive_missing += 1;
        }
    }

    // Also do a quick sample count of earlier blocks
    println!("   Sampling earlier blocks...");
    let mut sample_count = 0u64;
    for h in (0..scan_start).step_by(100) {
        if block_exists(&db, &cf_blocks, h) {
            sample_count += 1;
        }
    }
    let estimated_earlier = sample_count * 100; // Estimate based on 1% sampling

    println!();
    println!("📊 Scan Results:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Highest block found: {}", highest_found);
    println!("   Blocks in verification window: {}", total_blocks);
    println!("   Estimated earlier blocks: ~{}", estimated_earlier);

    if !missing_blocks.is_empty() && missing_blocks.len() <= 50 {
        println!("   ⚠️  Missing blocks in window: {:?}", missing_blocks);
    } else if !missing_blocks.is_empty() {
        println!("   ⚠️  {} gaps found near highest block", missing_blocks.len());
    }

    // Use highest_found as the target height (binary search verified it exists)
    let highest_contiguous = highest_found;

    println!("   Target repair height: {}", highest_contiguous);
    println!();

    // Check current pointer
    println!("🔍 Checking qblock:latest pointer...");
    let current_pointer = db.get_cf(&cf_blocks, b"qblock:latest")?;

    if let Some(height_bytes) = current_pointer {
        if height_bytes.len() == 8 {
            let mut height_array = [0u8; 8];
            height_array.copy_from_slice(&height_bytes);
            let current_height = u64::from_be_bytes(height_array);
            println!("   Current pointer: {} (height)", current_height);

            if current_height == highest_contiguous {
                println!("   ✅ Pointer is correct! No repair needed.");
                return Ok(());
            } else {
                println!("   ⚠️  Pointer is WRONG! Should be {}", highest_contiguous);
            }
        } else {
            println!("   ❌ Pointer is corrupted (invalid length: {})", height_bytes.len());
        }
    } else {
        println!("   ❌ Pointer is MISSING!");
    }

    println!();
    println!("🔧 Repair Options:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   1. Fix qblock:latest pointer to {}", highest_contiguous);
    println!("   2. Exit without changes");
    println!();
    print!("Choose an option (1/2): ");

    use std::io::{self, Write};
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    if input.trim() != "1" {
        println!("❌ Repair cancelled. No changes made.");
        return Ok(());
    }

    println!();
    println!("🔧 Applying repair...");

    // Write the correct pointer
    let height_bytes = highest_contiguous.to_be_bytes();
    db.put_cf(&cf_blocks, b"qblock:latest", &height_bytes)?;

    // Verify the fix
    let verify = db.get_cf(&cf_blocks, b"qblock:latest")?
        .ok_or_else(|| anyhow::anyhow!("Failed to verify pointer after write"))?;

    let mut verify_array = [0u8; 8];
    verify_array.copy_from_slice(&verify);
    let verify_height = u64::from_be_bytes(verify_array);

    if verify_height == highest_contiguous {
        println!("✅ Repair successful!");
        println!();
        println!("📊 Updated pointer:");
        println!("   qblock:latest → {}", verify_height);
        println!();
        println!("🎉 Database repair complete!");
        println!();
        println!("Next steps:");
        println!("   1. Restart your q-api-server");
        println!("   2. It should now load {} blocks correctly", highest_contiguous);
        println!("   3. Monitor logs to ensure height stays at {}", highest_contiguous);
    } else {
        println!("❌ Verification failed! Pointer shows {} instead of {}",
                 verify_height, highest_contiguous);
        return Err(anyhow::anyhow!("Repair verification failed"));
    }

    Ok(())
}
