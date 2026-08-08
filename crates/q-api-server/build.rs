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

    // 2026-07-16: force this build script to re-run on EVERY build so BUILD_DATE /
    // BUILD_TIMESTAMP are always fresh. The old `rerun-if-changed=build.rs` only re-ran
    // when build.rs itself changed, so incremental release rebuilds reported a stale
    // build_date (e.g. an incremental v10.11.84 build still said "2026-07-13"). Emitting
    // a rerun-if-changed for a path that never exists makes cargo re-run us every time.
    println!("cargo:rerun-if-changed=.build-always-rerun");
}
