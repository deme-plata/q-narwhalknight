//! Database Key Inspector
//!
//! Lists ALL keys in the blocks column family to find anomalies

use anyhow::Result;
use rocksdb::{DB, Options, ColumnFamilyDescriptor, IteratorMode};

const CF_BLOCKS: &str = "blocks";

fn main() -> Result<()> {
    println!("🔍 Q-NarwhalKnight Database Key Inspector");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let db_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./data-mine12/hot".to_string());

    println!("📂 Opening database: {}\n", db_path);

    // Discover existing column families
    let cf_list = DB::list_cf(&Options::default(), &db_path)?;
    let cfs: Vec<_> = cf_list.iter()
        .map(|name| ColumnFamilyDescriptor::new(name.as_str(), Options::default()))
        .collect();

    let db = DB::open_cf_descriptors(&Options::default(), &db_path, cfs)?;
    let cf_blocks = db.cf_handle(CF_BLOCKS)
        .ok_or_else(|| anyhow::anyhow!("blocks column family not found"))?;

    println!("🔍 Scanning all keys in blocks column family...\n");

    let mut qblock_latest_count = 0;
    let mut qblock_height_count = 0;
    let mut qblock_hash_count = 0;
    let mut other_count = 0;

    let mut highest_height = 0u64;
    let mut lowest_height = u64::MAX;
    let mut suspicious_keys = Vec::new();
    let mut heights = Vec::new();

    for item in db.iterator_cf(&cf_blocks, IteratorMode::Start) {
        let (key_bytes, value_bytes) = item?;
        let key = String::from_utf8_lossy(&key_bytes);

        if key == "qblock:latest" {
            qblock_latest_count += 1;
            // Decode the value
            if value_bytes.len() == 8 {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&value_bytes);
                let pointer_value = u64::from_be_bytes(bytes);
                println!("✓ Found qblock:latest = {}", pointer_value);
            } else {
                println!("⚠️  Found qblock:latest with invalid length: {} bytes", value_bytes.len());
            }
        } else if key.starts_with("qblock:height:") {
            qblock_height_count += 1;
            if let Some(height_str) = key.strip_prefix("qblock:height:") {
                if let Ok(height) = height_str.parse::<u64>() {
                    heights.push(height);
                    if height > highest_height {
                        highest_height = height;
                    }
                    if height < lowest_height {
                        lowest_height = height;
                    }
                } else {
                    suspicious_keys.push(key.to_string());
                }
            }
        } else if key.starts_with("qblock:hash:") {
            qblock_hash_count += 1;
        } else {
            other_count += 1;
            if other_count <= 20 {
                // Show hex bytes of key for debugging
                let key_hex: String = key_bytes.iter()
                    .take(32)
                    .map(|b| format!("{:02x}", b))
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("? Unknown key (len={}): {} (value len: {})",
                         key_bytes.len(), key_hex, value_bytes.len());
            }
        }
    }

    println!("\n📊 Summary:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   qblock:latest keys: {}", qblock_latest_count);
    println!("   qblock:height:* keys: {}", qblock_height_count);
    println!("   qblock:hash:* keys: {}", qblock_hash_count);
    println!("   Other keys: {}", other_count);

    if lowest_height != u64::MAX {
        println!("\n📈 Height range:");
        println!("   Lowest: {}", lowest_height);
        println!("   Highest: {}", highest_height);
        println!("   Range: {}", highest_height - lowest_height + 1);
        println!("   Actual blocks: {}", qblock_height_count);

        // Find gaps
        heights.sort_unstable();
        let mut gaps = Vec::new();
        for i in 0..heights.len().saturating_sub(1) {
            if heights[i + 1] > heights[i] + 1 {
                gaps.push((heights[i] + 1, heights[i + 1] - 1));
            }
        }

        if !gaps.is_empty() {
            println!("\n⚠️  Gaps found: {}", gaps.len());
            for (start, end) in gaps.iter().take(10) {
                if start == end {
                    println!("   Missing: {}", start);
                } else {
                    println!("   Missing: {} to {} ({} blocks)", start, end, end - start + 1);
                }
            }
        } else {
            println!("\n✅ No gaps - chain is contiguous from {} to {}", lowest_height, highest_height);
        }
    }

    if !suspicious_keys.is_empty() {
        println!("\n⚠️  Suspicious keys (unparseable heights): {}", suspicious_keys.len());
        for key in suspicious_keys.iter().take(10) {
            println!("   {}", key);
        }
    }

    Ok(())
}
