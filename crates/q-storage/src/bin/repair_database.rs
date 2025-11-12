//! Database Repair Utility
//!
//! Scans the RocksDB database and rebuilds missing pointers like `qblock:latest`.
//! This fixes databases corrupted by the sync-down bug in v0.5.21 and earlier.

use anyhow::Result;
use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use std::path::PathBuf;
use std::sync::Arc;

const CF_BLOCKS: &str = "blocks";

fn main() -> Result<()> {
    println!("🔧 Q-NarwhalKnight Database Repair Utility v0.5.22");
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

    // Scan for highest block
    println!("🔍 Scanning for highest contiguous block...");
    println!("   This may take a few seconds...");
    println!();

    let cf_blocks = db.cf_handle(CF_BLOCKS)
        .ok_or_else(|| anyhow::anyhow!("blocks column family not found"))?;

    let mut highest_found = 0u64;
    let mut total_blocks = 0u64;
    let mut missing_blocks = Vec::new();

    // Check blocks from 0 to 200,000
    for height in 0..=200_000 {
        if height % 10_000 == 0 {
            println!("   Scanning height {}...", height);
        }

        let key = format!("qblock:height:{}", height);

        if let Ok(Some(_)) = db.get_cf(&cf_blocks, key.as_bytes()) {
            total_blocks += 1;
            if height > highest_found {
                highest_found = height;
            }
        } else if height < highest_found {
            // Found a gap
            missing_blocks.push(height);
        } else {
            // Reached the end of contiguous blocks
            break;
        }
    }

    println!();
    println!("📊 Scan Results:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Total blocks found: {}", total_blocks);
    println!("   Highest block: {}", highest_found);

    if !missing_blocks.is_empty() {
        println!("   ⚠️  Missing blocks: {} gaps found", missing_blocks.len());
        if missing_blocks.len() <= 10 {
            println!("   Missing heights: {:?}", missing_blocks);
        } else {
            println!("   First 10 missing: {:?}", &missing_blocks[..10]);
        }
    } else {
        println!("   ✅ No gaps detected - chain is contiguous!");
    }

    // Find highest contiguous block (no gaps before it)
    let mut highest_contiguous = 0u64;
    for height in 0..=highest_found {
        let key = format!("qblock:height:{}", height);
        if db.get_cf(&cf_blocks, key.as_bytes())?.is_some() {
            highest_contiguous = height;
        } else {
            // Found first gap
            break;
        }
    }

    println!("   Highest contiguous: {}", highest_contiguous);
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
