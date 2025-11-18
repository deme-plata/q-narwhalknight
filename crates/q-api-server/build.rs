// build.rs - Compile-time metadata generation
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    // Get current timestamp
    let build_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("System time before UNIX epoch")
        .as_secs();

    // Set BUILD_TIMESTAMP environment variable for use in code
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", build_time);

    // Also set a human-readable timestamp
    let now = chrono::Utc::now();
    println!(
        "cargo:rustc-env=BUILD_DATE={}",
        now.format("%Y-%m-%d %H:%M:%S UTC")
    );

    // Rerun if this file changes
    println!("cargo:rerun-if-changed=build.rs");
}
