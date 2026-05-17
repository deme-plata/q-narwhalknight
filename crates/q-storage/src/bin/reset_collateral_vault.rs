/// Reset corrupted CollateralVault data
///
/// This tool deletes the collateral_vault key from the manifest CF,
/// causing the server to create a fresh vault on next startup.
///
/// Usage:
///   1. Stop the server: systemctl stop q-api-server
///   2. Run this tool: Q_DB_PATH=./data-mine16 cargo run --release --bin reset_collateral_vault
///   3. Restart server: systemctl start q-api-server
///
/// The corruption happened due to an unsigned integer underflow where
/// total_qugusd_minted wrapped from 0 to near u64::MAX.

use anyhow::{Context, Result};
use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("🔧 Q-NarwhalKnight CollateralVault Reset Tool");
    println!("   Fixing corrupted stablecoin data (u64 underflow bug)");
    println!();

    // Get database path from environment variable
    let db_path_str = env::var("Q_DB_PATH")
        .unwrap_or_else(|_| {
            eprintln!("❌ Error: Q_DB_PATH environment variable not set");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  1. Stop server: systemctl stop q-api-server");
            eprintln!("  2. Run: Q_DB_PATH=./data-mine16 cargo run --release --bin reset_collateral_vault");
            eprintln!("  3. Restart: systemctl start q-api-server");
            std::process::exit(1);
        });

    let db_path = PathBuf::from(&db_path_str);
    let hot_db_path = db_path.join("hot");

    if !hot_db_path.exists() {
        anyhow::bail!("❌ Database path does not exist: {}", hot_db_path.display());
    }

    println!("📂 Database path: {}", hot_db_path.display());
    println!();

    // List existing column families
    let existing_cfs = DB::list_cf(&Options::default(), &hot_db_path)
        .context("Failed to list existing column families")?;

    println!("📋 Found {} column families", existing_cfs.len());

    // Check if manifest CF exists
    if !existing_cfs.contains(&"manifest".to_string()) {
        anyhow::bail!("❌ 'manifest' column family not found!");
    }

    // Open database with all column families
    let mut opts = Options::default();
    opts.create_if_missing(false);
    opts.create_missing_column_families(false);

    let cf_descriptors: Vec<ColumnFamilyDescriptor> = existing_cfs
        .iter()
        .map(|name| ColumnFamilyDescriptor::new(name, Options::default()))
        .collect();

    let db = DB::open_cf_descriptors(&opts, &hot_db_path, cf_descriptors)
        .context("Failed to open database")?;

    let manifest_cf = db.cf_handle("manifest")
        .context("Failed to get manifest CF handle")?;

    // Check current value
    let key = b"collateral_vault";
    if let Some(value) = db.get_cf(&manifest_cf, key)? {
        println!("🔍 Found collateral_vault key ({} bytes)", value.len());

        // Try to deserialize to see current state
        if let Ok(vault) = bincode::deserialize::<CollateralVaultCheck>(&value) {
            println!("   - total_qug_locked: {}", vault.total_qug_locked);
            println!("   - total_qugusd_minted: {}", vault.total_qugusd_minted);
            println!("   - qug_price_usd: ${:.2}", vault.qug_price_usd);

            if vault.total_qugusd_minted > 1_000_000_000_000_000 {
                println!();
                println!("   ⚠️  CORRUPTION DETECTED: total_qugusd_minted is impossibly large!");
                println!("   ⚠️  This is near u64::MAX ({}) due to underflow bug", u64::MAX);
            }
        }

        println!();
        println!("🗑️  Deleting corrupted collateral_vault key...");
        db.delete_cf(&manifest_cf, key)?;
        println!("✅ Key deleted successfully!");
        println!();
        println!("🚀 Now restart the server:");
        println!("   systemctl start q-api-server");
        println!();
        println!("   The server will create a fresh CollateralVault with:");
        println!("   - total_qug_locked: 0");
        println!("   - total_qugusd_minted: 0");
        println!("   - qug_price_usd: $3000.00");
    } else {
        println!("ℹ️  No collateral_vault key found - vault is already clean");
    }

    Ok(())
}

/// Minimal struct to check vault state (doesn't need all fields)
#[derive(serde::Deserialize)]
struct CollateralVaultCheck {
    total_qug_locked: u64,
    total_qugusd_minted: u64,
    qug_price_usd: f64,
    // ... other fields ignored for checking
}
