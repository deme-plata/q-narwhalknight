/// Emergency tool to force height pointer to 710
/// For v0.9.99-beta height desync issue

use rocksdb::{DB, Options};

fn main() -> anyhow::Result<()> {
    println!("🔧 Emergency Height Pointer Fix");
    println!("================================\n");

    let db_path = std::env::var("Q_DB_PATH")
        .unwrap_or_else(|_| "./data-mine10".to_string());
    let hot_path = format!("{}/hot", db_path);

    println!("📂 Database: {}\n", hot_path);

    // Open RocksDB
    println!("🔓 Opening database...");
    let mut opts = Options::default();
    let db = DB::open(&opts, &hot_path)?;
    println!("✅ Database opened\n");

    // Read current pointer
    println!("📍 Current state:");
    let current_bytes = db.get(b"qblock:latest")?;
    let current_height = if let Some(bytes) = &current_bytes {
        if bytes.len() == 8 {
            u64::from_be_bytes(bytes[..8].try_into().unwrap())
        } else {
            0
        }
    } else {
        0
    };
    println!("   Height pointer: {}\n", current_height);

    // Verify blocks 704-710 exist
    println!("🔍 Verifying blocks 704-710...");
    for height in 704..=710 {
        let key = format!("qblock:height:{}", height);
        match db.get(key.as_bytes())? {
            Some(_) => println!("   ✅ Block {} exists", height),
            None => {
                eprintln!("   ❌ Block {} NOT FOUND", height);
                return Err(anyhow::anyhow!("Block {} missing", height));
            }
        }
    }

    // Update pointer to 710
    let target_height = 710u64;
    println!("\n🔧 Updating pointer to {}...", target_height);
    db.put(b"qblock:latest", &target_height.to_be_bytes())?;

    // Verify
    let verify_bytes = db.get(b"qblock:latest")?;
    let verify_height = if let Some(bytes) = verify_bytes {
        u64::from_be_bytes(bytes[..8].try_into().unwrap())
    } else {
        0
    };

    if verify_height == target_height {
        println!("✅ Success! {} → {}", current_height, verify_height);
    } else {
        eprintln!("❌ Failed! Expected {}, got {}", target_height, verify_height);
        return Err(anyhow::anyhow!("Verification failed"));
    }

    println!("\n✅ Repair complete");
    println!("   Restart: systemctl start q-api-server");

    Ok(())
}
