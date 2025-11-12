//! Balance Reset Utility
//!
//! CRITICAL: When blockchain is reset/corrupted, balances MUST be reset too.
//! Keeping old balances without blocks creates:
//! 1. Consensus failures
//! 2. Double-spending vulnerabilities
//! 3. Network rejection
//! 4. Invalid state

use anyhow::Result;
use rocksdb::{DB, Options, ColumnFamilyDescriptor, WriteBatch};
use std::sync::Arc;

const CF_MANIFEST: &str = "manifest";

fn main() -> Result<()> {
    println!("🔧 Q-NarwhalKnight Balance Reset Utility");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("⚠️  WARNING: This will DELETE ALL WALLET BALANCES!");
    println!("   Use this when blockchain is reset/corrupted.");
    println!();

    // Get database path
    let db_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./data-mine1/hot".to_string());

    println!("📂 Database: {}", db_path);
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

    let cf_manifest = db.cf_handle(CF_MANIFEST)
        .ok_or_else(|| anyhow::anyhow!("manifest column family not found"))?;

    // Count current balances
    let mut balance_count = 0u64;
    let mut total_balance = 0u128;
    let iter = db.iterator_cf(&cf_manifest, rocksdb::IteratorMode::Start);

    println!("🔍 Scanning current balances...");
    println!();

    for item in iter {
        let (key, value) = item?;
        let key_str = String::from_utf8_lossy(&key);

        if key_str.starts_with("wallet_balance:") {
            balance_count += 1;

            if value.len() >= 16 {
                let mut amount_bytes = [0u8; 16];
                amount_bytes.copy_from_slice(&value[..16]);
                let amount = u128::from_be_bytes(amount_bytes);
                total_balance += amount;

                if balance_count <= 5 {
                    let wallet = &key_str[15..35]; // First 20 chars of address
                    println!("   💰 {}: {} units", wallet, amount);
                }
            }
        }
    }

    if balance_count > 5 {
        println!("   ... and {} more wallets", balance_count - 5);
    }

    println!();
    println!("📊 Current State:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Total wallets: {}", balance_count);
    println!("   Total balance: {} units", total_balance);
    println!();

    if balance_count == 0 {
        println!("✅ No balances found. Database is already clean.");
        return Ok(());
    }

    println!("🚨 CRITICAL DECISION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Deleting balances is IRREVERSIBLE!");
    println!("   ");
    println!("   Why this is necessary:");
    println!("   1. Balances without blocks = NO cryptographic proof");
    println!("   2. Consensus will REJECT invalid balances");
    println!("   3. Other nodes will ban you for invalid state");
    println!("   4. Balances will be rebuilt from blockchain");
    println!("   ");
    println!("   Type 'DELETE' to confirm (case-sensitive):");
    println!();

    use std::io::{self, Write};
    print!("Confirmation: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    if input.trim() != "DELETE" {
        println!();
        println!("❌ Reset cancelled. No changes made.");
        return Ok(());
    }

    println!();
    println!("🔧 Deleting all wallet balances...");

    // Create batch delete
    let mut batch = WriteBatch::default();
    let iter = db.iterator_cf(&cf_manifest, rocksdb::IteratorMode::Start);

    let mut deleted = 0u64;
    for item in iter {
        let (key, _value) = item?;
        let key_str = String::from_utf8_lossy(&key);

        if key_str.starts_with("wallet_balance:") {
            batch.delete_cf(&cf_manifest, &key);
            deleted += 1;
        }
    }

    // Execute deletion
    db.write(batch)?;

    println!("✅ Deleted {} wallet balances", deleted);
    println!();
    println!("📊 Final State:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Total wallets: 0");
    println!("   Total balance: 0 units");
    println!();
    println!("🎉 Balance reset complete!");
    println!();
    println!("Next steps:");
    println!("   1. Restart q-api-server");
    println!("   2. Let it sync blockchain from genesis");
    println!("   3. Balances will be rebuilt from blocks");
    println!("   4. Final balances will match blockchain state");
    println!();
    println!("✅ Your node will now achieve consensus with network");

    Ok(())
}
