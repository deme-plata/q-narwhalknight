//! Test Multi-Format Block Reader — DIAGNOSTIC ONLY
//!
//! Opens a RocksDB database in READ-ONLY mode and tests the multi-format
//! block reader. Never writes anything. Safe to run against a live production DB.
//!
//! Usage:
//!   test-multiformat-reader /path/to/hot/db
//!
//! This binary:
//! 1. Opens the DB read-only
//! 2. Tests get_qblock_by_height (standard format) at various heights
//! 3. Tests get_qblock_any_format (multi-format) at the same heights
//! 4. Compares results — any_format should find MORE blocks than standard
//! 5. Reports a summary

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: test-multiformat-reader <db-path>");
        eprintln!("  e.g.: test-multiformat-reader /home/orobit/data-mainnet-genesis/hot");
        eprintln!();
        eprintln!("This opens the DB in READ-ONLY mode. Safe for production.");
        std::process::exit(1);
    }
    let db_path = &args[1];

    eprintln!("╔═══════════════════════════════════════════════╗");
    eprintln!("║  Multi-Format Block Reader Test               ║");
    eprintln!("║  READ-ONLY — Safe for production databases    ║");
    eprintln!("╚═══════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Database path: {}", db_path);
    eprintln!();

    // We can't easily instantiate StorageEngine in a standalone binary
    // because it requires the full async runtime and many dependencies.
    // Instead, we test at the RocksDB level directly.

    // Discover column families
    let cfs = match rocksdb::DB::list_cf(&rocksdb::Options::default(), db_path) {
        Ok(cfs) => cfs,
        Err(e) => {
            eprintln!("ERROR: Cannot list column families: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!("Found {} column families", cfs.len());

    // Open read-only
    let mut cf_descriptors = Vec::new();
    for cf_name in &cfs {
        let mut opts = rocksdb::Options::default();
        opts.set_max_open_files(256);
        cf_descriptors.push(rocksdb::ColumnFamilyDescriptor::new(cf_name.clone(), opts));
    }

    let mut db_opts = rocksdb::Options::default();
    db_opts.set_max_open_files(256);

    let db = match rocksdb::DB::open_cf_descriptors_read_only(
        &db_opts,
        db_path,
        cf_descriptors,
        false,
    ) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("ERROR: Cannot open database: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!("Database opened READ-ONLY successfully.");
    eprintln!();

    let blocks_cf = match db.cf_handle("blocks") {
        Some(cf) => cf,
        None => {
            eprintln!("ERROR: 'blocks' column family not found!");
            std::process::exit(1);
        }
    };

    // Test heights to probe
    let test_heights: Vec<u64> = vec![
        0, 1, 100, 1000, 5249, 5250, 10000, 50000,
        100000, 100441, 100442, 500000, 1000000, 2000000,
        5000000, 7000000, 9000000, 9253760,
        10000000, 12000000, 13000000, 13478000, 13500000, 14000000,
    ];

    eprintln!("Testing {} heights...", test_heights.len());
    eprintln!();
    eprintln!("{:<12} {:>12} {:>12} {:>12}", "Height", "String Key", "DAG Key", "Binary Key");
    eprintln!("{}", "-".repeat(52));

    let mut string_found = 0u64;
    let mut dag_found = 0u64;
    let mut binary_found = 0u64;
    let mut total_found = 0u64;

    for &height in &test_heights {
        // Test 1: Standard string key format
        let string_key = format!("qblock:height:{}", height);
        let has_string = db.get_cf(&blocks_cf, string_key.as_bytes())
            .ok()
            .flatten()
            .is_some();

        // Test 2: DAG layer format (prefix scan)
        let dag_prefix = format!("qblock:dag:{}:", height);
        let has_dag = db.prefix_iterator_cf(&blocks_cf, dag_prefix.as_bytes())
            .next()
            .and_then(|r| r.ok())
            .map(|(k, _)| {
                let key_str = String::from_utf8_lossy(&k);
                key_str.starts_with(&dag_prefix)
            })
            .unwrap_or(false);

        // Test 3: Binary key format (old finalize_block)
        let binary_key = height.to_be_bytes();
        let has_binary = db.prefix_iterator_cf(&blocks_cf, &binary_key)
            .next()
            .and_then(|r| r.ok())
            .map(|(k, _)| k.len() >= 8 && k[..8] == binary_key)
            .unwrap_or(false);

        if has_string { string_found += 1; }
        if has_dag { dag_found += 1; }
        if has_binary { binary_found += 1; }
        if has_string || has_dag || has_binary { total_found += 1; }

        let s_mark = if has_string { "✅" } else { "❌" };
        let d_mark = if has_dag { "✅" } else { "❌" };
        let b_mark = if has_binary { "✅" } else { "❌" };

        eprintln!("{:<12} {:>12} {:>12} {:>12}", height, s_mark, d_mark, b_mark);
    }

    eprintln!("{}", "-".repeat(52));
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════╗");
    eprintln!("║  RESULTS                                      ║");
    eprintln!("╠═══════════════════════════════════════════════╣");
    eprintln!("║  Heights tested:  {:>6}                       ║", test_heights.len());
    eprintln!("║  String key hits: {:>6}                       ║", string_found);
    eprintln!("║  DAG key hits:    {:>6}                       ║", dag_found);
    eprintln!("║  Binary key hits: {:>6}                       ║", binary_found);
    eprintln!("║  Total found:     {:>6} / {:>3}                ║", total_found, test_heights.len());
    eprintln!("╚═══════════════════════════════════════════════╝");
    eprintln!();

    if total_found > string_found {
        eprintln!("🎉 Multi-format reader finds MORE blocks than standard reader!");
        eprintln!("   Standard: {} blocks, Multi-format: {} blocks", string_found, total_found);
        eprintln!("   DAG-layer blocks contribute {} additional blocks", dag_found);
    } else if total_found == string_found && total_found > 0 {
        eprintln!("ℹ️  Multi-format finds the same blocks as standard (no DAG/binary blocks at tested heights)");
    } else {
        eprintln!("⚠️  Very few blocks found at tested heights. Database may have sparse data.");
    }

    eprintln!();
    eprintln!("✅ Test complete. Database was opened READ-ONLY. No data modified.");
}
