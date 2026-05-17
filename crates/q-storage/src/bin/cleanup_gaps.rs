//! Gap Cleanup Utility - Remove Orphan Blocks Above Contiguous Height
//!
//! This tool removes blocks that would create gaps in the blockchain:
//! - Finds the highest contiguous height (no missing blocks before it)
//! - Deletes all blocks above that height (orphan blocks)
//! - Resets the `qblock:latest` pointer to the contiguous height
//!
//! Use this after a fork/gap situation caused by data loss or network split.
//!
//! Usage: cleanup-gaps [--db-path ./data-mine1/hot] [--dry-run] [--force]

use anyhow::Result;
use rocksdb::{DB, Options, ColumnFamilyDescriptor, WriteBatch};
use std::collections::HashSet;

const CF_BLOCKS: &str = "blocks";

fn main() -> Result<()> {
    println!("🧹 Q-NarwhalKnight Gap Cleanup Utility v1.0.79-beta");
    println!("   Purpose: Remove orphan blocks above contiguous height");
    println!("   This fixes fork/gap situations caused by data loss");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Parse command line args
    let args: Vec<String> = std::env::args().collect();

    let mut db_path = "./data-mine1/hot".to_string();
    let mut dry_run = false;
    let mut force = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--db-path" => {
                if i + 1 < args.len() {
                    db_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--dry-run" => dry_run = true,
            "--force" => force = true,
            "-h" | "--help" => {
                println!("Usage: cleanup-gaps [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --db-path <PATH>  Path to RocksDB database (default: ./data-mine1/hot)");
                println!("  --dry-run         Show what would be deleted without making changes");
                println!("  --force           Skip confirmation prompt");
                println!("  -h, --help        Show this help");
                println!();
                println!("Examples:");
                println!("  cleanup-gaps --db-path /opt/orobit/data-node1/hot --dry-run");
                println!("  cleanup-gaps --db-path ./data-mine1/hot --force");
                return Ok(());
            }
            _ => {
                // Assume it's the db path if it doesn't start with --
                if !args[i].starts_with("--") {
                    db_path = args[i].clone();
                }
            }
        }
        i += 1;
    }

    if dry_run {
        println!("⚠️  DRY RUN MODE - No changes will be made");
        println!();
    }

    println!("📂 Opening database: {}", db_path);

    // Discover existing column families
    let db_opts_list = Options::default();
    let cf_list = match DB::list_cf(&db_opts_list, &db_path) {
        Ok(list) => list,
        Err(e) => {
            println!("❌ Failed to open database: {}", e);
            println!("   Make sure the database path is correct and the server is stopped.");
            return Err(e.into());
        }
    };

    println!("📋 Found {} column families", cf_list.len());

    // Open database with discovered column families
    let mut db_opts = Options::default();
    db_opts.create_if_missing(false);

    let cfs: Vec<_> = cf_list.iter()
        .map(|name| ColumnFamilyDescriptor::new(name.as_str(), Options::default()))
        .collect();

    let db = DB::open_cf_descriptors(&db_opts, &db_path, cfs)?;

    println!("✅ Database opened successfully");
    println!();

    let cf_blocks = db.cf_handle(CF_BLOCKS)
        .ok_or_else(|| anyhow::anyhow!("'blocks' column family not found"))?;

    // Phase 1: Find contiguous height and scan for all stored blocks
    println!("🔍 Phase 1: Scanning for contiguous chain...");
    println!();

    let mut highest_contiguous = 0u64;
    let mut all_stored_heights: HashSet<u64> = HashSet::new();
    let mut highest_stored = 0u64;

    // First, find the contiguous chain starting from genesis (height 1 in Q-NarwhalKnight)
    // Note: Q-NarwhalKnight blockchain starts at height 1, not 0
    for height in 1..=10_000_000 {
        let key = format!("qblock:height:{}", height);

        if db.get_cf(&cf_blocks, key.as_bytes())?.is_some() {
            all_stored_heights.insert(height);
            if height > highest_stored {
                highest_stored = height;
            }
            highest_contiguous = height;
        } else {
            // Found first gap - stop contiguous search here
            break;
        }

        if height % 50_000 == 0 && height > 0 {
            println!("   ... scanned to height {}", height);
        }
    }

    println!("   Contiguous chain: 1 → {}", highest_contiguous);

    // Phase 2: Continue scanning to find orphan blocks above contiguous
    println!();
    println!("🔍 Phase 2: Scanning for orphan blocks above {}...", highest_contiguous);

    let mut orphan_heights: Vec<u64> = Vec::new();
    let mut consecutive_missing = 0u64;

    // Scan beyond contiguous to find orphans
    for height in (highest_contiguous + 1)..=10_000_000 {
        let key = format!("qblock:height:{}", height);

        if db.get_cf(&cf_blocks, key.as_bytes())?.is_some() {
            orphan_heights.push(height);
            all_stored_heights.insert(height);
            if height > highest_stored {
                highest_stored = height;
            }
            consecutive_missing = 0;
        } else {
            consecutive_missing += 1;
            // Stop after 10000 consecutive missing blocks
            if consecutive_missing >= 10000 {
                break;
            }
        }

        if height % 50_000 == 0 {
            println!("   ... scanned to height {} (found {} orphans so far)", height, orphan_heights.len());
        }
    }

    println!();
    println!("📊 Scan Results:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Contiguous height:    {}", highest_contiguous);
    println!("   Highest stored:       {}", highest_stored);
    println!("   Total blocks stored:  {}", all_stored_heights.len());
    println!("   Orphan blocks found:  {}", orphan_heights.len());

    if !orphan_heights.is_empty() {
        let gap_size = highest_stored - highest_contiguous;
        println!("   Gap size:             {} blocks", gap_size);

        if orphan_heights.len() <= 10 {
            println!("   Orphan heights:       {:?}", orphan_heights);
        } else {
            println!("   First 5 orphans:      {:?}", &orphan_heights[..5]);
            println!("   Last 5 orphans:       {:?}", &orphan_heights[orphan_heights.len()-5..]);
        }
    }

    println!();

    // Check current pointer
    println!("🔍 Checking qblock:latest pointer...");
    let current_pointer = db.get_cf(&cf_blocks, b"qblock:latest")?;

    let current_latest = if let Some(height_bytes) = current_pointer {
        if height_bytes.len() == 8 {
            let mut height_array = [0u8; 8];
            height_array.copy_from_slice(&height_bytes);
            let h = u64::from_be_bytes(height_array);
            println!("   Current pointer: {}", h);
            h
        } else {
            println!("   ⚠️  Pointer corrupted (invalid length)");
            0
        }
    } else {
        println!("   ⚠️  Pointer MISSING");
        0
    };

    // Determine if cleanup is needed
    if orphan_heights.is_empty() && current_latest == highest_contiguous {
        println!();
        println!("✅ No cleanup needed!");
        println!("   Chain is contiguous and pointer is correct.");
        return Ok(());
    }

    println!();
    println!("🧹 Cleanup Plan:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Will DELETE {} orphan blocks", orphan_heights.len());
    if current_latest != highest_contiguous {
        println!("   Will UPDATE qblock:latest: {} → {}", current_latest, highest_contiguous);
    }
    println!();

    // Confirmation
    if !force && !dry_run {
        println!("⚠️  WARNING: This operation is IRREVERSIBLE!");
        println!("   The {} orphan blocks will be permanently deleted.", orphan_heights.len());
        println!();
        print!("Type 'YES' to confirm: ");

        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if input.trim() != "YES" {
            println!("❌ Cleanup cancelled. No changes made.");
            return Ok(());
        }
    }

    if dry_run {
        println!("🔍 DRY RUN - Would delete the following:");
        for (i, height) in orphan_heights.iter().enumerate() {
            if i < 20 {
                println!("   - qblock:height:{}", height);
            } else if i == 20 {
                println!("   ... and {} more blocks", orphan_heights.len() - 20);
                break;
            }
        }
        if current_latest != highest_contiguous {
            println!("   - Would update qblock:latest: {} → {}", current_latest, highest_contiguous);
        }
        println!();
        println!("✅ DRY RUN complete. Run without --dry-run to apply changes.");
        return Ok(());
    }

    // Execute cleanup
    println!();
    println!("🧹 Executing cleanup...");

    let mut batch = WriteBatch::default();
    let mut deleted = 0u64;

    for height in &orphan_heights {
        let key = format!("qblock:height:{}", height);
        batch.delete_cf(&cf_blocks, key.as_bytes());
        deleted += 1;

        if deleted % 10000 == 0 {
            println!("   ... deleted {} blocks", deleted);
        }
    }

    // Update pointer
    let height_bytes = highest_contiguous.to_be_bytes();
    batch.put_cf(&cf_blocks, b"qblock:latest", &height_bytes);

    // Apply batch
    db.write(batch)?;

    println!("   ✅ Deleted {} orphan blocks", deleted);
    println!("   ✅ Updated qblock:latest → {}", highest_contiguous);

    // Verify
    println!();
    println!("🔍 Verifying cleanup...");

    let verify = db.get_cf(&cf_blocks, b"qblock:latest")?
        .ok_or_else(|| anyhow::anyhow!("Failed to verify pointer"))?;

    let mut verify_array = [0u8; 8];
    verify_array.copy_from_slice(&verify);
    let verify_height = u64::from_be_bytes(verify_array);

    if verify_height == highest_contiguous {
        println!("   ✅ Pointer verified: {}", verify_height);
    } else {
        println!("   ❌ Pointer verification FAILED!");
        return Err(anyhow::anyhow!("Pointer verification failed"));
    }

    // Verify a sample orphan was deleted
    if !orphan_heights.is_empty() {
        let sample = orphan_heights[0];
        let key = format!("qblock:height:{}", sample);
        if db.get_cf(&cf_blocks, key.as_bytes())?.is_none() {
            println!("   ✅ Orphan deletion verified (checked height {})", sample);
        } else {
            println!("   ⚠️  Warning: Sample orphan {} still exists", sample);
        }
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎉 Gap cleanup complete!");
    println!();
    println!("   Contiguous chain: 1 → {}", highest_contiguous);
    println!("   Orphan blocks removed: {}", deleted);
    println!();
    println!("Next steps:");
    println!("   1. Restart your q-api-server");
    println!("   2. The node will sync missing blocks from peers");
    println!("   3. Monitor logs for normal operation");
    println!();

    Ok(())
}
