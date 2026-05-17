//! Block Diagnostic Utility - Examine raw block data format
//!
//! This tool reads raw block bytes from the database and analyzes their format
//! to help diagnose deserialization issues.
//!
//! Usage: diagnose-block [--db-path ./data-mine12/hot] [--height 299905]

use anyhow::Result;
use rocksdb::{DB, Options, ColumnFamilyDescriptor};

const CF_BLOCKS: &str = "blocks";

fn main() -> Result<()> {
    println!("🔬 Q-NarwhalKnight Block Diagnostic Utility v1.0.80-beta");
    println!("   Purpose: Analyze raw block data format");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Parse command line args
    let args: Vec<String> = std::env::args().collect();

    let mut db_path = "./data-mine12/hot".to_string();
    let mut height: u64 = 299905;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--db-path" => {
                if i + 1 < args.len() {
                    db_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--height" => {
                if i + 1 < args.len() {
                    height = args[i + 1].parse().unwrap_or(299905);
                    i += 1;
                }
            }
            "-h" | "--help" => {
                println!("Usage: diagnose-block [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --db-path <PATH>   Path to RocksDB database (default: ./data-mine12/hot)");
                println!("  --height <NUM>     Block height to examine (default: 299905)");
                println!("  -h, --help         Show this help");
                return Ok(());
            }
            _ => {
                if !args[i].starts_with("--") {
                    height = args[i].parse().unwrap_or(299905);
                }
            }
        }
        i += 1;
    }

    println!("📂 Opening database: {}", db_path);
    println!("🎯 Target height: {}", height);
    println!();

    // Discover existing column families
    let db_opts_list = Options::default();
    let cf_list = match DB::list_cf(&db_opts_list, &db_path) {
        Ok(list) => list,
        Err(e) => {
            println!("❌ Failed to open database: {}", e);
            return Err(e.into());
        }
    };

    // Open database with discovered column families
    let mut db_opts = Options::default();
    db_opts.create_if_missing(false);

    let cfs: Vec<_> = cf_list.iter()
        .map(|name| ColumnFamilyDescriptor::new(name.as_str(), Options::default()))
        .collect();

    let db = DB::open_cf_descriptors(&db_opts, &db_path, cfs)?;

    let cf_blocks = db.cf_handle(CF_BLOCKS)
        .ok_or_else(|| anyhow::anyhow!("'blocks' column family not found"))?;

    // Read raw block data
    let key = format!("qblock:height:{}", height);
    println!("🔍 Reading key: {}", key);

    match db.get_cf(&cf_blocks, key.as_bytes())? {
        Some(block_data) => {
            println!("✅ Found block data: {} bytes", block_data.len());
            println!();

            // Analyze first bytes
            println!("📊 Raw Data Analysis:");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            // First 64 bytes hex dump
            let display_len = std::cmp::min(64, block_data.len());
            println!("   First {} bytes (hex):", display_len);
            for chunk in block_data[..display_len].chunks(16) {
                let hex_str: String = chunk.iter().map(|b| format!("{:02x} ", b)).collect();
                let ascii_str: String = chunk.iter().map(|b| {
                    if *b >= 32 && *b <= 126 { *b as char } else { '.' }
                }).collect();
                println!("   {} | {}", hex_str, ascii_str);
            }
            println!();

            // Check for common serialization format signatures
            println!("📌 Format Detection:");

            // MessagePack array starts with 0x90-0x9f (fixarray) or 0xdc/0xdd (array16/32)
            if block_data.len() > 0 {
                let first = block_data[0];
                if first >= 0x90 && first <= 0x9f {
                    println!("   → Looks like MessagePack fixarray (elements: {})", first - 0x90);
                } else if first == 0xdc {
                    println!("   → Looks like MessagePack array16");
                } else if first == 0xdd {
                    println!("   → Looks like MessagePack array32");
                } else if first >= 0x80 && first <= 0x8f {
                    println!("   → Looks like MessagePack fixmap (elements: {})", first - 0x80);
                } else if first == 0xde {
                    println!("   → Looks like MessagePack map16");
                } else if first == 0xdf {
                    println!("   → Looks like MessagePack map32");
                }

                // Postcard uses varint encoding - no clear signature
                // Bincode uses fixed-size encoding for numbers

                // Check if first 8 bytes look like a u64 (height)
                if block_data.len() >= 8 {
                    let mut height_bytes = [0u8; 8];
                    height_bytes.copy_from_slice(&block_data[..8]);
                    let height_le = u64::from_le_bytes(height_bytes);
                    let height_be = u64::from_be_bytes(height_bytes);
                    println!("   → First 8 bytes as u64 (LE): {}", height_le);
                    println!("   → First 8 bytes as u64 (BE): {}", height_be);
                }
            }
            println!();

            // Try different deserializers
            println!("🧪 Deserialization Tests:");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            // Try bincode
            match bincode::deserialize::<q_types::block::QBlock>(&block_data) {
                Ok(block) => {
                    println!("   ✅ bincode: SUCCESS");
                    println!("      Height: {}", block.header.height);
                    println!("      Tx count: {}", block.transactions.len());
                    if !block.transactions.is_empty() {
                        println!("      First tx type: {:?}", block.transactions[0].tx_type);
                    }
                }
                Err(e) => {
                    println!("   ❌ bincode: FAILED");
                    println!("      Error: {}", e);
                }
            }

            // v1.0.80-beta: Try legacy fallback
            match q_types::legacy::deserialize_qblock_with_fallback(&block_data) {
                Ok(block) => {
                    println!("   ✅ legacy_fallback: SUCCESS");
                    println!("      Height: {}", block.header.height);
                    println!("      Tx count: {}", block.transactions.len());
                    if !block.transactions.is_empty() {
                        println!("      First tx type: {:?}", block.transactions[0].tx_type);
                    }
                }
                Err(e) => {
                    println!("   ❌ legacy_fallback: FAILED");
                    println!("      Error: {}", e);
                }
            }

            // Try postcard
            match postcard::from_bytes::<q_types::block::QBlock>(&block_data) {
                Ok(block) => {
                    println!("   ✅ postcard: SUCCESS");
                    println!("      Height: {}", block.header.height);
                    println!("      Tx count: {}", block.transactions.len());
                }
                Err(e) => {
                    println!("   ❌ postcard: FAILED");
                    println!("      Error: {}", e);
                }
            }

            // Try rmp_serde (MessagePack)
            match rmp_serde::from_slice::<q_types::block::QBlock>(&block_data) {
                Ok(block) => {
                    println!("   ✅ rmp_serde: SUCCESS");
                    println!("      Height: {}", block.header.height);
                    println!("      Tx count: {}", block.transactions.len());
                }
                Err(e) => {
                    println!("   ❌ rmp_serde: FAILED");
                    println!("      Error: {}", e);
                }
            }

            // Check for compression
            println!();
            println!("🗜️  Compression Detection:");
            if block_data.len() >= 4 {
                // LZ4 frame magic: 0x04224D18
                if block_data[0] == 0x04 && block_data[1] == 0x22 && block_data[2] == 0x4D && block_data[3] == 0x18 {
                    println!("   → LZ4 frame format detected!");

                    // Try decompressing
                    match lz4::block::decompress(&block_data[4..], None) {
                        Ok(decompressed) => {
                            println!("   → Decompressed to {} bytes", decompressed.len());
                            // Try deserializing decompressed data
                            match bincode::deserialize::<q_types::block::QBlock>(&decompressed) {
                                Ok(block) => {
                                    println!("   ✅ bincode after LZ4: SUCCESS");
                                    println!("      Height: {}", block.header.height);
                                }
                                Err(e) => {
                                    println!("   ❌ bincode after LZ4: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            println!("   ❌ LZ4 decompress failed: {}", e);
                        }
                    }
                }
                // Zstd magic: 0x28B52FFD
                else if block_data[0] == 0x28 && block_data[1] == 0xB5 && block_data[2] == 0x2F && block_data[3] == 0xFD {
                    println!("   → ZSTD format detected!");

                    match zstd::decode_all(&block_data[..]) {
                        Ok(decompressed) => {
                            println!("   → Decompressed to {} bytes", decompressed.len());
                            match bincode::deserialize::<q_types::block::QBlock>(&decompressed) {
                                Ok(block) => {
                                    println!("   ✅ bincode after ZSTD: SUCCESS");
                                    println!("      Height: {}", block.header.height);
                                }
                                Err(e) => {
                                    println!("   ❌ bincode after ZSTD: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            println!("   ❌ ZSTD decompress failed: {}", e);
                        }
                    }
                } else {
                    println!("   → No compression detected (raw data)");
                }
            }

            // Scan nearby blocks for comparison
            println!();
            println!("📊 Scanning Nearby Blocks:");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            let check_heights = [
                height.saturating_sub(10),
                height.saturating_sub(5),
                height.saturating_sub(1),
                height,
                height + 1,
                height + 5,
                height + 10,
            ];

            for h in check_heights {
                let k = format!("qblock:height:{}", h);
                match db.get_cf(&cf_blocks, k.as_bytes())? {
                    Some(data) => {
                        let status = match bincode::deserialize::<q_types::block::QBlock>(&data) {
                            Ok(_) => "✅ OK",
                            Err(_) => "❌ FAIL",
                        };
                        let first_bytes: String = data.iter().take(8).map(|b| format!("{:02x}", b)).collect();
                        println!("   Height {}: {} bytes, {}, first: {}", h, data.len(), status, first_bytes);
                    }
                    None => {
                        println!("   Height {}: ⚠️  MISSING", h);
                    }
                }
            }

            // Find first failing block
            println!();
            println!("🔍 Finding First Failing Block:");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            let mut first_fail = None;
            let mut last_success = 0u64;

            for h in 1..=height + 100 {
                let k = format!("qblock:height:{}", h);
                match db.get_cf(&cf_blocks, k.as_bytes())? {
                    Some(data) => {
                        match bincode::deserialize::<q_types::block::QBlock>(&data) {
                            Ok(_) => {
                                last_success = h;
                            }
                            Err(e) => {
                                if first_fail.is_none() {
                                    first_fail = Some((h, e.to_string(), data.len()));
                                }
                                break;
                            }
                        }
                    }
                    None => break,
                }

                if h % 50000 == 0 {
                    println!("   ... scanned to height {}", h);
                }
            }

            if let Some((fail_height, error, size)) = first_fail {
                println!("   Last successful: height {}", last_success);
                println!("   First failure:   height {} ({} bytes)", fail_height, size);
                println!("   Error: {}", error);
            } else {
                println!("   ✅ All blocks up to {} deserialize successfully!", height + 100);
            }

        }
        None => {
            println!("❌ No block found at height {}", height);
        }
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔬 Diagnostic complete");

    Ok(())
}
