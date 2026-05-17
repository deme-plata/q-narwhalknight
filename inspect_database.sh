#!/bin/bash
# Database inspection script for v0.9.99-beta height stuck issue

echo "🔍 Q-NarwhalKnight Database Inspection"
echo "======================================"
echo ""

DB_PATH="./data-mine10/hot"

if [ ! -d "$DB_PATH" ]; then
    echo "❌ Database not found at: $DB_PATH"
    exit 1
fi

echo "Database path: $DB_PATH"
echo ""

# We'll use a simple Rust binary to inspect RocksDB
# Since we can't directly read RocksDB from bash

cat > /tmp/inspect_db.rs << 'EOF'
use rocksdb::{DB, Options};
use std::path::Path;

fn main() {
    let db_path = std::env::args().nth(1).expect("Missing DB path");

    println!("Opening database: {}", db_path);

    let mut opts = Options::default();
    let db = DB::open_for_read_only(&opts, &db_path, false)
        .expect("Failed to open database");

    println!("\n📊 Checking blocks 690-710:");
    println!("----------------------------");

    for height in 690..=710 {
        let key = format!("qblock:height:{}", height);
        match db.get(key.as_bytes()) {
            Ok(Some(data)) => {
                println!("✅ Block {} exists ({} bytes)", height, data.len());
            }
            Ok(None) => {
                println!("❌ Block {} NOT FOUND", height);
            }
            Err(e) => {
                println!("⚠️  Block {} ERROR: {}", height, e);
            }
        }
    }

    println!("\n📍 Height pointer:");
    println!("------------------");
    match db.get(b"qblock:latest") {
        Ok(Some(data)) => {
            if data.len() == 8 {
                let height = u64::from_be_bytes(data.try_into().unwrap());
                println!("✅ qblock:latest = {}", height);
            } else {
                println!("⚠️  qblock:latest has wrong size: {} bytes", data.len());
            }
        }
        Ok(None) => {
            println!("❌ qblock:latest NOT FOUND");
        }
        Err(e) => {
            println!("⚠️  qblock:latest ERROR: {}", e);
        }
    }
}
EOF

echo "Compiling database inspection tool..."
rustc /tmp/inspect_db.rs -o /tmp/inspect_db --edition 2021 -C opt-level=0 \
    --extern rocksdb=/opt/orobit/shared/q-narwhalknight/target/release/deps/librocksdb-*.rlib 2>&1 | head -20

if [ ! -f /tmp/inspect_db ]; then
    echo "❌ Failed to compile inspection tool"
    echo "Falling back to manual checks..."
    echo ""
    echo "Database directory contents:"
    ls -lh "$DB_PATH" | head -20
    echo ""
    echo "Database size:"
    du -sh "$DB_PATH"
    exit 1
fi

echo "Running inspection..."
/tmp/inspect_db "$DB_PATH"
