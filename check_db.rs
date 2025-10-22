use rocksdb::{DB, Options};
use std::path::Path;

fn main() {
    let db_path = "./data-mine1";
    
    println!("🔍 Opening RocksDB at: {}", db_path);
    
    let mut opts = Options::default();
    opts.create_if_missing(false);
    
    match DB::open_for_read_only(&opts, db_path, false) {
        Ok(db) => {
            println!("✅ Database opened successfully");
            
            // Try to find wallet balance keys
            let iter = db.iterator(rocksdb::IteratorMode::Start);
            
            println!("\n📊 Searching for wallet_balance entries:");
            for item in iter {
                match item {
                    Ok((key, value)) => {
                        let key_str = String::from_utf8_lossy(&key);
                        if key_str.contains("wallet_balance") {
                            let amount = if value.len() == 8 {
                                u64::from_le_bytes(value.as_ref().try_into().unwrap())
                            } else {
                                0
                            };
                            println!("  🔑 {}: {} units ({} QUG)", 
                                key_str, 
                                amount,
                                amount as f64 / 100_000_000.0
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("❌ Error reading entry: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to open database: {}", e);
        }
    }
}
