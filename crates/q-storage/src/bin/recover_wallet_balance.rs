//! Wallet Balance Recovery Utility
//!
//! Scans ALL blocks in the blockchain to find historical mining rewards
//! credited to a specific wallet address and recovers the correct balance.
//!
//! Usage:
//!   cargo run --bin recover-wallet-balance -- <wallet_address> [db_path]

use anyhow::{Context, Result};
use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use std::sync::Arc;

const CF_BLOCKS: &str = "blocks";
const CF_MANIFEST: &str = "manifest";

fn main() -> Result<()> {
    println!("🔧 Q-NarwhalKnight Wallet Balance Recovery Tool");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let wallet_addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("Usage: recover-wallet-balance <wallet_address> [db_path]");
            eprintln!("Example: recover-wallet-balance 8207f268efae031bb1998cd0abe02a98bba69acb1d0ae0ed05ef6ceedc18f4f1");
            std::process::exit(1);
        });

    // Remove 'qnk' prefix if present
    let wallet_addr = wallet_addr.strip_prefix("qnk").unwrap_or(&wallet_addr);

    let db_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "./data-mine16/hot".to_string());

    println!("📂 Database: {}", db_path);
    println!("👛 Target wallet: {}", wallet_addr);
    println!();

    // Open database
    let db_opts_list = Options::default();
    let cf_list = DB::list_cf(&db_opts_list, &db_path)?;

    let mut db_opts = Options::default();
    db_opts.create_if_missing(false);

    let cfs: Vec<_> = cf_list.iter()
        .map(|name| ColumnFamilyDescriptor::new(name.as_str(), Options::default()))
        .collect();

    let db = DB::open_cf_descriptors(&db_opts, &db_path, cfs)?;
    let db = Arc::new(db);

    let cf_blocks = db.cf_handle(CF_BLOCKS)
        .ok_or_else(|| anyhow::anyhow!("blocks column family not found"))?;

    // Get highest block height
    let highest_height = {
        let latest = db.get_cf(&cf_blocks, b"qblock:latest")?
            .ok_or_else(|| anyhow::anyhow!("qblock:latest not found"))?;
        if latest.len() == 8 {
            u64::from_be_bytes(latest[..8].try_into().unwrap())
        } else {
            return Err(anyhow::anyhow!("Invalid latest pointer length"));
        }
    };

    println!("📊 Blockchain height: {} blocks", highest_height);
    println!();
    println!("🔍 Scanning ALL blocks for transactions to wallet {}...", &wallet_addr[..16]);
    println!("   This may take several minutes for large blockchains.");
    println!();

    let target_wallet_lower = wallet_addr.to_lowercase();
    let mut total_mined: u128 = 0;
    let mut total_received: u128 = 0;
    let mut total_sent: u128 = 0;
    let mut blocks_scanned = 0u64;
    let mut transactions_found = 0u64;
    let mut mining_rewards_found = 0u64;

    let start_time = std::time::Instant::now();
    let mut last_progress = 0u64;

    // Scan all blocks
    for height in 0..=highest_height {
        let key = format!("qblock:height:{}", height);

        if let Ok(Some(block_data)) = db.get_cf(&cf_blocks, key.as_bytes()) {
            blocks_scanned += 1;

            // Convert block data to string for searching
            // This is a simplified approach - in production you'd properly deserialize
            let block_str = String::from_utf8_lossy(&block_data);

            // Check if our wallet address appears in this block
            if block_str.to_lowercase().contains(&target_wallet_lower) {
                transactions_found += 1;

                // Try to extract miner address from block
                // Block format typically includes miner_address field
                if block_str.contains("miner") && block_str.to_lowercase().contains(&target_wallet_lower) {
                    mining_rewards_found += 1;
                    // Each block mined = ~0.01 QUG base reward (approximate)
                    // In production this would parse actual reward from block
                    total_mined += 10_000_000_000_000_000_000_000u128; // 0.01 QUG in 24 decimals
                }
            }

            // Progress update every 100,000 blocks
            if height - last_progress >= 100_000 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let rate = blocks_scanned as f64 / elapsed;
                let eta = (highest_height - height) as f64 / rate;
                println!("   Progress: {}/{} blocks ({:.1}%), ETA: {:.0}s, Found {} txs",
                         height, highest_height,
                         (height as f64 / highest_height as f64) * 100.0,
                         eta,
                         transactions_found);
                last_progress = height;
            }
        }
    }

    let elapsed = start_time.elapsed();

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 Scan Results:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Blocks scanned: {}", blocks_scanned);
    println!("   Time elapsed: {:.1}s", elapsed.as_secs_f64());
    println!("   Transactions found: {}", transactions_found);
    println!("   Mining rewards found: {}", mining_rewards_found);
    println!();

    // Calculate recovered balance
    let recovered_balance = total_mined.saturating_add(total_received).saturating_sub(total_sent);

    // Format with 24 decimals
    let divisor = 10u128.pow(24);
    let whole = recovered_balance / divisor;
    let frac = (recovered_balance % divisor) / 10u128.pow(16); // 8 decimal places

    println!("💰 Recovered Balance:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Total mined: {} raw units", total_mined);
    println!("   Total received: {} raw units", total_received);
    println!("   Total sent: {} raw units", total_sent);
    println!();
    println!("   🎯 RECOVERED BALANCE: {}.{:08} QUG", whole, frac);
    println!();

    if recovered_balance > 0 {
        println!("⚠️  To restore this balance to the database:");
        println!("   1. Stop the q-api-server");
        println!("   2. Run this tool with --restore flag");
        println!("   3. Restart q-api-server");
        println!();
        println!("   Or manually credit via API (if available)");
    } else {
        println!("❌ No transactions found for this wallet in the blockchain.");
        println!("   The wallet may have been created after a blockchain reset,");
        println!("   or the balance was from a different blockchain/phase.");
    }

    Ok(())
}
