//! Wallet Balance Query Utility
//!
//! Check the balance of a specific wallet address in the database.
//! Searches ALL known storage formats for wallet balances.

use anyhow::Result;
use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use std::sync::Arc;

const CF_MANIFEST: &str = "manifest";
const CF_TOKEN_BALANCES: &str = "token_balances";
const CF_BALANCES: &str = "balances";

fn decode_balance(value: &[u8]) -> Option<(u128, &'static str)> {
    if value.len() >= 16 {
        // u128 format (16 bytes) - try both BE and LE
        let be = u128::from_be_bytes(value[..16].try_into().unwrap());
        let le = u128::from_le_bytes(value[..16].try_into().unwrap());
        // Return whichever is more reasonable (non-astronomical)
        if be > 0 && be < 10u128.pow(36) {
            Some((be, "u128-BE"))
        } else if le > 0 && le < 10u128.pow(36) {
            Some((le, "u128-LE"))
        } else if be > 0 {
            Some((be, "u128-BE"))
        } else {
            Some((le, "u128-LE"))
        }
    } else if value.len() >= 8 {
        // Legacy u64 format (8 bytes)
        let be = u64::from_be_bytes(value[..8].try_into().unwrap());
        let le = u64::from_le_bytes(value[..8].try_into().unwrap());
        if be > 0 && be < 10u64.pow(18) as u64 {
            Some((be as u128, "u64-BE"))
        } else {
            Some((le as u128, "u64-LE"))
        }
    } else {
        None
    }
}

fn format_balance(amount: u128, decimals: u32) -> String {
    let divisor = 10u128.pow(decimals);
    let whole = amount / divisor;
    let frac = amount % divisor;
    format!("{}.{:0width$}", whole, frac, width = decimals as usize)
}

fn main() -> Result<()> {
    println!("🔍 Q-NarwhalKnight Comprehensive Wallet Balance Query");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Get wallet address from args
    let wallet_addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("Usage: check_wallet_balance <wallet_address> [db_path]");
            eprintln!("Example: check_wallet_balance 8207f268efae031bb1998cd0abe02a98bba69acb1d0ae0ed05ef6ceedc18f4f1");
            std::process::exit(1);
        });

    // Remove 'qnk' prefix if present
    let wallet_addr = wallet_addr.strip_prefix("qnk").unwrap_or(&wallet_addr);

    let db_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "./data-mine16/hot".to_string());

    println!("📂 Database: {}", db_path);
    println!("👛 Wallet: {}", wallet_addr);
    println!();

    // Open database
    let db_opts_list = Options::default();
    let cf_list = DB::list_cf(&db_opts_list, &db_path)?;

    println!("📋 Found column families: {:?}", cf_list);

    let mut db_opts = Options::default();
    db_opts.create_if_missing(false);

    let cfs: Vec<_> = cf_list.iter()
        .map(|name| ColumnFamilyDescriptor::new(name.as_str(), Options::default()))
        .collect();

    let db = DB::open_cf_descriptors(&db_opts, &db_path, cfs)?;
    let db = Arc::new(db);

    println!();
    println!("🔍 Searching ALL storage locations for wallet balance...");
    println!();

    let mut found_any = false;

    // 1. Check manifest CF with underscore format: wallet_balance_<hex>
    if let Some(cf) = db.cf_handle(CF_MANIFEST) {
        let key = format!("wallet_balance_{}", wallet_addr);
        println!("📍 Checking manifest CF: {}", key);
        if let Ok(Some(value)) = db.get_cf(&cf, key.as_bytes()) {
            if let Some((amount, format)) = decode_balance(&value) {
                println!("   ✅ FOUND! Format: {}, Raw: {}", format, amount);
                println!("   💰 With 24 decimals: {} QUG", format_balance(amount, 24));
                println!("   💰 With 8 decimals: {} QUG", format_balance(amount, 8));
                found_any = true;
            }
        } else {
            println!("   ❌ Not found");
        }

        // 2. Check manifest CF with colon format: wallet_balance:<hex>
        let key = format!("wallet_balance:{}", wallet_addr);
        println!("📍 Checking manifest CF: {}", key);
        if let Ok(Some(value)) = db.get_cf(&cf, key.as_bytes()) {
            if let Some((amount, format)) = decode_balance(&value) {
                println!("   ✅ FOUND! Format: {}, Raw: {}", format, amount);
                println!("   💰 With 24 decimals: {} QUG", format_balance(amount, 24));
                println!("   💰 With 8 decimals: {} QUG", format_balance(amount, 8));
                found_any = true;
            }
        } else {
            println!("   ❌ Not found");
        }
    }

    // 3. Check balances CF: <hex> directly
    if let Some(cf) = db.cf_handle(CF_BALANCES) {
        println!("📍 Checking balances CF: {}", wallet_addr);
        if let Ok(Some(value)) = db.get_cf(&cf, wallet_addr.as_bytes()) {
            if let Some((amount, format)) = decode_balance(&value) {
                println!("   ✅ FOUND! Format: {}, Raw: {}", format, amount);
                println!("   💰 With 24 decimals: {} QUG", format_balance(amount, 24));
                println!("   💰 With 8 decimals: {} QUG", format_balance(amount, 8));
                found_any = true;
            }
        } else {
            println!("   ❌ Not found");
        }

        // Also try with qnk prefix
        let key = format!("qnk{}", wallet_addr);
        println!("📍 Checking balances CF: {}", key);
        if let Ok(Some(value)) = db.get_cf(&cf, key.as_bytes()) {
            if let Some((amount, format)) = decode_balance(&value) {
                println!("   ✅ FOUND! Format: {}, Raw: {}", format, amount);
                println!("   💰 With 24 decimals: {} QUG", format_balance(amount, 24));
                println!("   💰 With 8 decimals: {} QUG", format_balance(amount, 8));
                found_any = true;
            }
        } else {
            println!("   ❌ Not found");
        }
    }

    // 4. Check token_balances CF: <wallet 32 bytes><QUG token 32 bytes>
    if let Some(cf) = db.cf_handle(CF_TOKEN_BALANCES) {
        // QUG token address (native token)
        let qug_token = [0u8; 32]; // Native QUG is usually all zeros

        if let Ok(wallet_bytes) = hex::decode(wallet_addr) {
            if wallet_bytes.len() == 32 {
                let mut key = Vec::with_capacity(64);
                key.extend_from_slice(&wallet_bytes);
                key.extend_from_slice(&qug_token);

                println!("📍 Checking token_balances CF for native QUG");
                if let Ok(Some(value)) = db.get_cf(&cf, &key) {
                    if let Some((amount, format)) = decode_balance(&value) {
                        println!("   ✅ FOUND! Format: {}, Raw: {}", format, amount);
                        println!("   💰 With 24 decimals: {} QUG", format_balance(amount, 24));
                        println!("   💰 With 8 decimals: {} QUG", format_balance(amount, 8));
                        found_any = true;
                    }
                } else {
                    println!("   ❌ Not found");
                }
            }
        }
    }

    println!();

    // 5. Search for any key containing wallet address
    println!("🔍 Searching for ANY keys containing wallet address prefix...");
    let search_prefix = &wallet_addr[..16.min(wallet_addr.len())];
    let mut matches_found = 0;

    if let Some(cf) = db.cf_handle(CF_MANIFEST) {
        let iter = db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            if key_str.contains(search_prefix) {
                matches_found += 1;
                if let Some((amount, format)) = decode_balance(&value) {
                    println!("   📦 Key: {} ({})", key_str, format);
                    println!("      Raw: {}", amount);
                    println!("      24 dec: {} | 8 dec: {}",
                             format_balance(amount, 24), format_balance(amount, 8));
                    found_any = true;
                }
            }
        }
    }

    if matches_found == 0 {
        println!("   No matching keys found in manifest CF");
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if !found_any {
        println!("❌ NO BALANCE FOUND for wallet {}", wallet_addr);
        println!();
        println!("📊 Showing ALL wallets with balances for comparison:");
        println!();

        if let Some(cf) = db.cf_handle(CF_MANIFEST) {
            let iter = db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
            let mut count = 0;

            for item in iter {
                let (key, value) = item?;
                let key_str = String::from_utf8_lossy(&key);

                if key_str.starts_with("wallet_balance_") || key_str.starts_with("wallet_balance:") {
                    count += 1;
                    let addr = key_str.replace("wallet_balance_", "").replace("wallet_balance:", "");

                    if let Some((amount, _format)) = decode_balance(&value) {
                        let human = format_balance(amount, 24);
                        println!("   💰 qnk{}: {} QUG", &addr[..20.min(addr.len())], human);
                    }

                    if count >= 30 {
                        println!("   ... (more wallets exist)");
                        break;
                    }
                }
            }

            println!();
            println!("Total wallets found: {}", count);
        }
    }

    Ok(())
}
