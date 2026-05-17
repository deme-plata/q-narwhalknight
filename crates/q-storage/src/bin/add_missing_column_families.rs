/// Add missing column families to existing RocksDB database
///
/// This tool adds missing column families to databases created before v0.9.18-beta WITHOUT deleting existing data.
/// Missing CFs: sync_certificates, peer_trust, banned_peers, ai_attachments
///
/// Usage:
///   Q_DB_PATH=./data cargo run --release --bin add_missing_column_families
///   Q_DB_PATH=/opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/data cargo run --release --bin add_missing_column_families

use anyhow::{Context, Result};
use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("🔧 Q-NarwhalKnight Database Migration Tool");
    println!("   Adding missing column families to existing database");
    println!();

    // Get database path from environment variable
    let db_path_str = env::var("Q_DB_PATH")
        .unwrap_or_else(|_| {
            eprintln!("❌ Error: Q_DB_PATH environment variable not set");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  Q_DB_PATH=./data cargo run --release --bin add_missing_column_families");
            std::process::exit(1);
        });

    let db_path = PathBuf::from(db_path_str);

    println!("   Database path: {}", db_path.display());
    println!();

    // Construct hot DB path - try both old and new layout
    let hot_db_path = if db_path.join("q-narwhal-db").join("hot").exists() {
        db_path.join("q-narwhal-db").join("hot")
    } else if db_path.join("hot").exists() {
        db_path.join("hot")
    } else {
        anyhow::bail!("❌ Database hot path does not exist at either {}/q-narwhal-db/hot or {}/hot",
                     db_path.display(), db_path.display());
    };

    println!("📂 Opening database at: {}", hot_db_path.display());

    // List existing column families
    let existing_cfs = DB::list_cf(&Options::default(), &hot_db_path)
        .context("Failed to list existing column families")?;

    println!("📊 Existing column families ({}):", existing_cfs.len());
    for cf in &existing_cfs {
        println!("   ✅ {}", cf);
    }
    println!();

    // Check if missing column families
    let has_sync_certificates = existing_cfs.contains(&"sync_certificates".to_string());
    let has_peer_trust = existing_cfs.contains(&"peer_trust".to_string());
    let has_banned_peers = existing_cfs.contains(&"banned_peers".to_string());
    let has_ai_attachments = existing_cfs.contains(&"ai_attachments".to_string());
    let has_swap_history = existing_cfs.contains(&"cf_swap_history".to_string());
    // v2.7.9-beta: Perpetual trading column families
    let has_perp_positions = existing_cfs.contains(&"cf_perp_positions".to_string());
    let has_perp_trades = existing_cfs.contains(&"cf_perp_trades".to_string());

    if has_sync_certificates && has_peer_trust && has_banned_peers && has_ai_attachments && has_swap_history && has_perp_positions && has_perp_trades {
        println!("✅ Database already has all required column families!");
        println!("   • sync_certificates: EXISTS");
        println!("   • peer_trust: EXISTS");
        println!("   • banned_peers: EXISTS");
        println!("   • ai_attachments: EXISTS");
        println!("   • cf_swap_history: EXISTS");
        println!("   • cf_perp_positions: EXISTS");
        println!("   • cf_perp_trades: EXISTS");
        println!();
        println!("🎯 No migration needed!");
        return Ok(());
    }

    println!("🔍 Missing column families detected:");
    if !has_sync_certificates {
        println!("   ❌ sync_certificates");
    } else {
        println!("   ✅ sync_certificates (already exists)");
    }
    if !has_peer_trust {
        println!("   ❌ peer_trust");
    } else {
        println!("   ✅ peer_trust (already exists)");
    }
    if !has_banned_peers {
        println!("   ❌ banned_peers");
    } else {
        println!("   ✅ banned_peers (already exists)");
    }
    if !has_ai_attachments {
        println!("   ❌ ai_attachments");
    } else {
        println!("   ✅ ai_attachments (already exists)");
    }
    if !has_swap_history {
        println!("   ❌ cf_swap_history");
    } else {
        println!("   ✅ cf_swap_history (already exists)");
    }
    if !has_perp_positions {
        println!("   ❌ cf_perp_positions");
    } else {
        println!("   ✅ cf_perp_positions (already exists)");
    }
    if !has_perp_trades {
        println!("   ❌ cf_perp_trades");
    } else {
        println!("   ✅ cf_perp_trades (already exists)");
    }
    println!();

    // Open database with existing column families
    let mut db_opts = Options::default();
    db_opts.create_if_missing(false);
    db_opts.create_missing_column_families(false);

    println!("🔓 Opening database with existing column families...");

    let cf_descriptors: Vec<ColumnFamilyDescriptor> = existing_cfs
        .iter()
        .map(|name| {
            let opts = Options::default();
            ColumnFamilyDescriptor::new(name, opts)
        })
        .collect();

    let db = DB::open_cf_descriptors(&db_opts, &hot_db_path, cf_descriptors)
        .context("Failed to open database")?;

    println!("✅ Database opened successfully");
    println!();

    // Add missing column families
    if !has_sync_certificates {
        println!("➕ Adding sync_certificates column family...");

        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
        opts.set_max_write_buffer_number(2);

        db.create_cf("sync_certificates", &opts)
            .context("Failed to create sync_certificates column family")?;

        println!("   ✅ sync_certificates created successfully");
        println!("      • Compression: LZ4");
        println!("      • Write buffer: 16MB");
    }

    if !has_peer_trust {
        println!("➕ Adding peer_trust column family...");

        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(8 * 1024 * 1024); // 8MB
        opts.set_max_write_buffer_number(2);

        db.create_cf("peer_trust", &opts)
            .context("Failed to create peer_trust column family")?;

        println!("   ✅ peer_trust created successfully");
        println!("      • Compression: LZ4");
        println!("      • Write buffer: 8MB");
    }

    if !has_banned_peers {
        println!("➕ Adding banned_peers column family...");

        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(8 * 1024 * 1024); // 8MB
        opts.set_max_write_buffer_number(2);

        db.create_cf("banned_peers", &opts)
            .context("Failed to create banned_peers column family")?;

        println!("   ✅ banned_peers created successfully");
        println!("      • Compression: LZ4");
        println!("      • Write buffer: 8MB");
    }

    if !has_ai_attachments {
        println!("➕ Adding ai_attachments column family...");

        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB (larger for attachments)
        opts.set_max_write_buffer_number(2);

        db.create_cf("ai_attachments", &opts)
            .context("Failed to create ai_attachments column family")?;

        println!("   ✅ ai_attachments created successfully");
        println!("      • Compression: LZ4");
        println!("      • Write buffer: 16MB");
    }

    // v2.3.9-beta: Swap history for Token Details Modal transaction history
    if !has_swap_history {
        println!("➕ Adding cf_swap_history column family...");

        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB - many small records
        opts.set_max_write_buffer_number(2);

        db.create_cf("cf_swap_history", &opts)
            .context("Failed to create cf_swap_history column family")?;

        println!("   ✅ cf_swap_history created successfully");
        println!("      • Compression: LZ4");
        println!("      • Write buffer: 32MB");
    }

    // v2.7.9-beta: Perpetual positions column family
    if !has_perp_positions {
        println!("➕ Adding cf_perp_positions column family...");

        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
        opts.set_max_write_buffer_number(2);

        db.create_cf("cf_perp_positions", &opts)
            .context("Failed to create cf_perp_positions column family")?;

        println!("   ✅ cf_perp_positions created successfully");
        println!("      • Compression: LZ4");
        println!("      • Write buffer: 32MB");
    }

    // v2.7.9-beta: Perpetual trades column family
    if !has_perp_trades {
        println!("➕ Adding cf_perp_trades column family...");

        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB
        opts.set_max_write_buffer_number(2);

        db.create_cf("cf_perp_trades", &opts)
            .context("Failed to create cf_perp_trades column family")?;

        println!("   ✅ cf_perp_trades created successfully");
        println!("      • Compression: LZ4");
        println!("      • Write buffer: 32MB");
    }

    println!();
    println!("🎉 MIGRATION COMPLETE!");
    println!();
    println!("📊 Final column family count:");

    let final_cfs = DB::list_cf(&Options::default(), &hot_db_path)
        .context("Failed to list final column families")?;

    println!("   Total: {} column families", final_cfs.len());
    for cf in &final_cfs {
        let marker = if cf == "sync_certificates" || cf == "peer_trust" || cf == "banned_peers" || cf == "ai_attachments" || cf == "cf_swap_history" {
            "🆕"
        } else {
            "  "
        };
        println!("   {} {}", marker, cf);
    }

    println!();
    println!("✅ Database migration successful!");
    println!("   You can now restart the node to use TurboSync.");
    println!();
    println!("🚀 Next steps:");
    println!("   1. Stop the node: systemctl stop q-api-server");
    println!("   2. Restart the node: systemctl start q-api-server");
    println!("   3. Verify TurboSync works in the logs");

    Ok(())
}
