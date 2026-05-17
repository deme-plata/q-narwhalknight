//! RocksDB Block Key Format Scanner
//!
//! Diagnostic tool to scan ALL keys in the CF_BLOCKS ("blocks") column family
//! and categorize them by format. Designed to discover where 13.4M+ blocks are
//! stored when only 585K are findable via the `qblock:height:N` string key format.
//!
//! Usage:
//!   scan-blocks --db-path /home/orobit/data-mainnet-genesis/hot
//!
//! This tool is READ-ONLY. It never writes to the database.

use anyhow::Result;
use rocksdb::{ColumnFamilyDescriptor, IteratorMode, Options, DB};
use std::collections::HashMap;
use std::time::Instant;

const CF_BLOCKS: &str = "blocks";

/// Maximum number of sample keys to store per category
const MAX_SAMPLES: usize = 5;

/// Progress reporting interval
const PROGRESS_INTERVAL: u64 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum KeyCategory {
    /// Starts with b"qblock:height:" — current primary block storage format
    StringHeight,
    /// Starts with b"qblock:dag:" — DAG layer blocks (multi-proposer)
    StringDag,
    /// Starts with b"qblock:hash:" — hash-to-height reverse index
    StringHash,
    /// Exactly b"qblock:latest" — latest height pointer
    StringLatest,
    /// Exactly b"qblock:tip_height" — tip height pointer
    StringTip,
    /// Exactly b"qblock:contiguous_verified" — contiguous verified pointer
    StringContiguousVerified,
    /// Starts with b"qblock:" but doesn't match any known pattern
    StringOther,
    /// Exactly 8 bytes — could be height.to_be_bytes()
    Binary8,
    /// Exactly 40 bytes — could be height(8) + hash(32)
    Binary40,
    /// Exactly 32 bytes — could be a raw hash
    Binary32,
    /// Binary key that doesn't start with "qblock" and doesn't match known sizes
    BinaryOther,
    /// Anything else
    Unknown,
}

impl std::fmt::Display for KeyCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyCategory::StringHeight => write!(f, "STRING_HEIGHT  (qblock:height:N)"),
            KeyCategory::StringDag => write!(f, "STRING_DAG     (qblock:dag:H:P)"),
            KeyCategory::StringHash => write!(f, "STRING_HASH    (qblock:hash:HEX)"),
            KeyCategory::StringLatest => write!(f, "STRING_LATEST  (qblock:latest)"),
            KeyCategory::StringTip => write!(f, "STRING_TIP     (qblock:tip_height)"),
            KeyCategory::StringContiguousVerified => {
                write!(f, "STRING_CONTV   (qblock:contiguous_verified)")
            }
            KeyCategory::StringOther => write!(f, "STRING_OTHER   (qblock:???)"),
            KeyCategory::Binary8 => write!(f, "BINARY_8       (8 bytes, BE u64?)"),
            KeyCategory::Binary40 => write!(f, "BINARY_40      (40 bytes, u64+hash?)"),
            KeyCategory::Binary32 => write!(f, "BINARY_32      (32 bytes, raw hash?)"),
            KeyCategory::BinaryOther => write!(f, "BINARY_OTHER   (non-qblock binary)"),
            KeyCategory::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

struct CategoryStats {
    count: u64,
    total_value_bytes: u64,
    total_key_bytes: u64,
    min_height: Option<u64>,
    max_height: Option<u64>,
    sample_keys: Vec<String>,
    sample_values_preview: Vec<String>,
    min_value_size: u64,
    max_value_size: u64,
}

impl CategoryStats {
    fn new() -> Self {
        Self {
            count: 0,
            total_value_bytes: 0,
            total_key_bytes: 0,
            min_height: None,
            max_height: None,
            sample_keys: Vec::new(),
            sample_values_preview: Vec::new(),
            min_value_size: u64::MAX,
            max_value_size: 0,
        }
    }

    fn record(&mut self, key_hex: String, value_len: u64, key_len: u64, height: Option<u64>, value_preview: String) {
        self.count += 1;
        self.total_value_bytes += value_len;
        self.total_key_bytes += key_len;

        if value_len < self.min_value_size {
            self.min_value_size = value_len;
        }
        if value_len > self.max_value_size {
            self.max_value_size = value_len;
        }

        if let Some(h) = height {
            match self.min_height {
                Some(min) if h < min => self.min_height = Some(h),
                None => self.min_height = Some(h),
                _ => {}
            }
            match self.max_height {
                Some(max) if h > max => self.max_height = Some(h),
                None => self.max_height = Some(h),
                _ => {}
            }
        }

        if self.sample_keys.len() < MAX_SAMPLES {
            self.sample_keys.push(key_hex);
            self.sample_values_preview.push(value_preview);
        }
    }
}

fn categorize_key(key: &[u8]) -> KeyCategory {
    // Try to interpret as UTF-8 string first
    if let Ok(s) = std::str::from_utf8(key) {
        if s == "qblock:latest" {
            return KeyCategory::StringLatest;
        }
        if s == "qblock:tip_height" {
            return KeyCategory::StringTip;
        }
        if s == "qblock:contiguous_verified" {
            return KeyCategory::StringContiguousVerified;
        }
        if s.starts_with("qblock:height:") {
            return KeyCategory::StringHeight;
        }
        if s.starts_with("qblock:dag:") {
            return KeyCategory::StringDag;
        }
        if s.starts_with("qblock:hash:") {
            return KeyCategory::StringHash;
        }
        if s.starts_with("qblock:") {
            return KeyCategory::StringOther;
        }
    }

    // Binary key classification by size
    match key.len() {
        8 => KeyCategory::Binary8,
        32 => KeyCategory::Binary32,
        40 => KeyCategory::Binary40,
        _ => {
            // Check if it starts with "qblock" bytes even if the rest isn't valid UTF-8
            if key.starts_with(b"qblock") {
                KeyCategory::StringOther
            } else {
                KeyCategory::BinaryOther
            }
        }
    }
}

fn extract_height(key: &[u8], category: &KeyCategory) -> Option<u64> {
    match category {
        KeyCategory::StringHeight => {
            let s = std::str::from_utf8(key).ok()?;
            s.strip_prefix("qblock:height:")?.parse::<u64>().ok()
        }
        KeyCategory::StringDag => {
            let s = std::str::from_utf8(key).ok()?;
            let rest = s.strip_prefix("qblock:dag:")?;
            // Format: height:proposer_hex
            rest.split(':').next()?.parse::<u64>().ok()
        }
        KeyCategory::Binary8 => {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(key);
            let val = u64::from_be_bytes(buf);
            // Plausible block height range: 0 to 20M
            if val <= 20_000_000 {
                Some(val)
            } else {
                None
            }
        }
        KeyCategory::Binary40 => {
            // First 8 bytes might be BE height
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&key[..8]);
            let val = u64::from_be_bytes(buf);
            if val <= 20_000_000 {
                Some(val)
            } else {
                None
            }
        }
        KeyCategory::StringHash => {
            // Hash keys don't encode a height in the key
            None
        }
        _ => None,
    }
}

fn format_key_hex(key: &[u8], max_bytes: usize) -> String {
    let display_len = key.len().min(max_bytes);
    let hex: String = key[..display_len]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();
    if key.len() > max_bytes {
        format!("{}... ({} bytes total)", hex, key.len())
    } else {
        format!("{} ({} bytes)", hex, key.len())
    }
}

fn format_value_preview(value: &[u8], max_bytes: usize) -> String {
    let display_len = value.len().min(max_bytes);
    let hex: String = value[..display_len]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();
    if value.len() > max_bytes {
        format!("{}... ({} bytes)", hex, value.len())
    } else {
        format!("{} ({} bytes)", hex, value.len())
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000_000 {
        format!("{:.2} TB", bytes as f64 / 1_000_000_000_000.0)
    } else if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

fn main() -> Result<()> {
    println!("=====================================================");
    println!("  Q-NarwhalKnight Block Key Format Scanner");
    println!("  READ-ONLY diagnostic tool");
    println!("=====================================================");
    println!();

    // Parse --db-path argument
    let args: Vec<String> = std::env::args().collect();
    let db_path = if let Some(pos) = args.iter().position(|a| a == "--db-path") {
        args.get(pos + 1)
            .cloned()
            .unwrap_or_else(|| {
                eprintln!("Error: --db-path requires a value");
                std::process::exit(1);
            })
    } else if args.len() == 2 {
        // Allow positional argument for convenience
        args[1].clone()
    } else {
        eprintln!("Usage: scan-blocks --db-path /path/to/hot/db");
        eprintln!("   or: scan-blocks /path/to/hot/db");
        std::process::exit(1);
    };

    println!("Database path: {}", db_path);
    println!();

    // Discover existing column families
    println!("Discovering column families...");
    let cf_list = DB::list_cf(&Options::default(), &db_path)?;
    println!("Found {} column families:", cf_list.len());
    for cf in &cf_list {
        println!("  - {}", cf);
    }
    println!();

    if !cf_list.iter().any(|cf| cf == CF_BLOCKS) {
        eprintln!("ERROR: '{}' column family not found in database!", CF_BLOCKS);
        std::process::exit(1);
    }

    // Open database in read-only mode
    println!("Opening database in READ-ONLY mode...");
    let mut opts = Options::default();
    opts.set_max_open_files(256); // Conservative to avoid fd exhaustion on large DBs

    let cfs: Vec<_> = cf_list
        .iter()
        .map(|name| ColumnFamilyDescriptor::new(name.as_str(), Options::default()))
        .collect();

    let db = DB::open_cf_descriptors_read_only(&opts, &db_path, cfs, false)?;
    let cf_blocks = db
        .cf_handle(CF_BLOCKS)
        .ok_or_else(|| anyhow::anyhow!("'blocks' column family handle not found"))?;

    println!("Database opened successfully.");
    println!();

    // Scan all keys
    println!("Scanning ALL keys in '{}' column family...", CF_BLOCKS);
    println!("(This may take a while for a 154GB database with 31K+ SST files)");
    println!();

    let mut stats: HashMap<KeyCategory, CategoryStats> = HashMap::new();
    let mut total_keys: u64 = 0;
    let start = Instant::now();
    let mut last_progress = Instant::now();

    // Track binary-8 height distribution in buckets
    let mut binary8_height_buckets: HashMap<u64, u64> = HashMap::new(); // bucket (millions) -> count
    let mut binary8_non_height_samples: Vec<String> = Vec::new();

    for item in db.iterator_cf(&cf_blocks, IteratorMode::Start) {
        let (key_bytes, value_bytes) = match item {
            Ok(kv) => kv,
            Err(e) => {
                eprintln!("WARNING: Iterator error at key #{}: {}", total_keys, e);
                continue;
            }
        };

        total_keys += 1;

        let category = categorize_key(&key_bytes);
        let height = extract_height(&key_bytes, &category);
        let key_hex = format_key_hex(&key_bytes, 48);
        let value_preview = format_value_preview(&value_bytes, 16);

        let entry = stats.entry(category).or_insert_with(CategoryStats::new);
        entry.record(
            key_hex,
            value_bytes.len() as u64,
            key_bytes.len() as u64,
            height,
            value_preview,
        );

        // Extra analysis for Binary8 keys
        if category == KeyCategory::Binary8 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&key_bytes);
            let val = u64::from_be_bytes(buf);
            if val <= 20_000_000 {
                let bucket = val / 1_000_000;
                *binary8_height_buckets.entry(bucket).or_insert(0) += 1;
            } else if binary8_non_height_samples.len() < 10 {
                binary8_non_height_samples.push(format!(
                    "0x{:016x} (decimal: {})",
                    val, val
                ));
            }
        }

        // Progress reporting
        if total_keys % PROGRESS_INTERVAL == 0 {
            let elapsed = start.elapsed();
            let rate = total_keys as f64 / elapsed.as_secs_f64();
            let since_last = last_progress.elapsed();
            println!(
                "  ... scanned {} keys ({:.0} keys/sec, elapsed: {:.1}s, last 1M in {:.1}s)",
                total_keys,
                rate,
                elapsed.as_secs_f64(),
                since_last.as_secs_f64()
            );
            last_progress = Instant::now();
        }
    }

    let elapsed = start.elapsed();

    // Print report
    println!();
    println!("=====================================================");
    println!("                   SCAN RESULTS");
    println!("=====================================================");
    println!();
    println!("Total keys scanned: {}", total_keys);
    println!("Scan duration: {:.2}s ({:.0} keys/sec)", elapsed.as_secs_f64(), total_keys as f64 / elapsed.as_secs_f64());
    println!();

    // Sort categories by count (descending)
    let mut categories: Vec<_> = stats.iter().collect();
    categories.sort_by(|a, b| b.1.count.cmp(&a.1.count));

    println!("{:<45} {:>12} {:>12} {:>12} {:>12}  Height Range",
             "Category", "Count", "Key Size", "Value Size", "Avg Val");
    println!("{}", "-".repeat(120));

    for (cat, st) in &categories {
        let avg_value = if st.count > 0 {
            st.total_value_bytes / st.count
        } else {
            0
        };

        let height_range = match (st.min_height, st.max_height) {
            (Some(min), Some(max)) => format!("{} - {}", min, max),
            _ => "N/A".to_string(),
        };

        println!(
            "{:<45} {:>12} {:>12} {:>12} {:>12}  {}",
            format!("{}", cat),
            st.count,
            format_bytes(st.total_key_bytes),
            format_bytes(st.total_value_bytes),
            format_bytes(avg_value),
            height_range
        );
    }

    println!("{}", "-".repeat(120));
    let total_key_bytes: u64 = stats.values().map(|s| s.total_key_bytes).sum();
    let total_value_bytes: u64 = stats.values().map(|s| s.total_value_bytes).sum();
    println!(
        "{:<45} {:>12} {:>12} {:>12}",
        "TOTAL",
        total_keys,
        format_bytes(total_key_bytes),
        format_bytes(total_value_bytes)
    );

    // Print samples for each category
    println!();
    println!("=====================================================");
    println!("                  KEY SAMPLES");
    println!("=====================================================");

    for (cat, st) in &categories {
        if st.sample_keys.is_empty() {
            continue;
        }
        println!();
        println!("--- {} (count: {}) ---", cat, st.count);
        for (i, (key, val)) in st.sample_keys.iter().zip(st.sample_values_preview.iter()).enumerate() {
            println!("  [{}] key: {}", i + 1, key);
            println!("       val: {}", val);
        }
        if st.count > MAX_SAMPLES as u64 {
            println!("  ... and {} more", st.count - MAX_SAMPLES as u64);
        }

        // Value size stats
        if st.min_value_size != u64::MAX {
            println!("  Value size range: {} - {}", format_bytes(st.min_value_size), format_bytes(st.max_value_size));
        }
    }

    // Binary-8 height distribution
    if !binary8_height_buckets.is_empty() {
        println!();
        println!("=====================================================");
        println!("  BINARY_8 HEIGHT DISTRIBUTION (if BE u64)");
        println!("=====================================================");
        println!();

        let mut buckets: Vec<_> = binary8_height_buckets.iter().collect();
        buckets.sort_by_key(|&(k, _)| *k);

        for (bucket, count) in &buckets {
            let range_start = *bucket * 1_000_000;
            let range_end = range_start + 999_999;
            println!(
                "  {:>10} - {:>10}: {:>12} keys",
                range_start, range_end, count
            );
        }

        let total_plausible: u64 = binary8_height_buckets.values().sum();
        let total_binary8 = stats
            .get(&KeyCategory::Binary8)
            .map(|s| s.count)
            .unwrap_or(0);
        println!();
        println!(
            "  Plausible heights (0-20M): {} / {} ({:.1}%)",
            total_plausible,
            total_binary8,
            if total_binary8 > 0 {
                total_plausible as f64 / total_binary8 as f64 * 100.0
            } else {
                0.0
            }
        );

        if !binary8_non_height_samples.is_empty() {
            println!();
            println!("  Non-height binary-8 samples (value > 20M):");
            for s in &binary8_non_height_samples {
                println!("    {}", s);
            }
        }
    }

    // Summary analysis
    println!();
    println!("=====================================================");
    println!("                  ANALYSIS");
    println!("=====================================================");
    println!();

    let string_height_count = stats.get(&KeyCategory::StringHeight).map(|s| s.count).unwrap_or(0);
    let string_dag_count = stats.get(&KeyCategory::StringDag).map(|s| s.count).unwrap_or(0);
    let string_hash_count = stats.get(&KeyCategory::StringHash).map(|s| s.count).unwrap_or(0);
    let binary8_count = stats.get(&KeyCategory::Binary8).map(|s| s.count).unwrap_or(0);
    let binary40_count = stats.get(&KeyCategory::Binary40).map(|s| s.count).unwrap_or(0);
    let binary32_count = stats.get(&KeyCategory::Binary32).map(|s| s.count).unwrap_or(0);
    let binary_other_count = stats.get(&KeyCategory::BinaryOther).map(|s| s.count).unwrap_or(0);

    println!("Block data keys (contain serialized blocks):");
    println!("  qblock:height:N   = {:>12}  (current turbo-sync format)", string_height_count);
    println!("  qblock:dag:H:P    = {:>12}  (DAG layer multi-proposer)", string_dag_count);
    println!("  BINARY_8          = {:>12}  (potential old BE u64 format)", binary8_count);
    println!();
    println!("Index/pointer keys:");
    println!("  qblock:hash:HEX   = {:>12}  (hash -> height reverse index)", string_hash_count);
    println!("  BINARY_32          = {:>12}  (potential raw hash keys)", binary32_count);
    println!("  BINARY_40          = {:>12}  (potential height+hash composite)", binary40_count);
    println!();
    println!("Other/unknown:");
    println!("  BINARY_OTHER       = {:>12}", binary_other_count);
    let string_other_count = stats.get(&KeyCategory::StringOther).map(|s| s.count).unwrap_or(0);
    let unknown_count = stats.get(&KeyCategory::Unknown).map(|s| s.count).unwrap_or(0);
    println!("  STRING_OTHER       = {:>12}", string_other_count);
    println!("  UNKNOWN            = {:>12}", unknown_count);
    println!();

    let potential_block_data = string_height_count + string_dag_count + binary8_count;
    println!("Potential block data keys total: {}", potential_block_data);
    println!("(qblock:height + qblock:dag + binary-8 that look like heights)");

    println!();
    println!("Done.");

    Ok(())
}
