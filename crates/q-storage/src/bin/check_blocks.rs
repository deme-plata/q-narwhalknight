use rocksdb::{DB, Options, ColumnFamilyDescriptor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = std::env::args().nth(1).unwrap_or_else(|| "./data-mine12/hot".to_string());
    let db_opts_list = Options::default();
    let cf_list = DB::list_cf(&db_opts_list, &db_path)?;
    
    let mut db_opts = Options::default();
    db_opts.create_if_missing(false);
    let cfs: Vec<_> = cf_list.iter()
        .map(|name| ColumnFamilyDescriptor::new(name.as_str(), Options::default()))
        .collect();
    let db = DB::open_cf_descriptors(&db_opts, &db_path, cfs)?;
    
    let cf_blocks = db.cf_handle("blocks").ok_or("blocks CF not found")?;
    
    // Check blocks 1-20
    println!("Checking blocks 1-20:");
    for height in 1..=20 {
        let key = format!("qblock:height:{}", height);
        if let Ok(Some(_)) = db.get_cf(&cf_blocks, key.as_bytes()) {
            println!("  Block {} ✅", height);
        } else {
            println!("  Block {} ❌ MISSING", height);
        }
    }
    
    // Check blocks 12100-12114
    println!("\nChecking blocks 12100-12114:");
    for height in 12100..=12114 {
        let key = format!("qblock:height:{}", height);
        if let Ok(Some(_)) = db.get_cf(&cf_blocks, key.as_bytes()) {
            println!("  Block {} ✅", height);
        } else {
            println!("  Block {} ❌ MISSING", height);
        }
    }
    
    Ok(())
}
