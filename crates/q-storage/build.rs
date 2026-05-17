// Q-NarwhalKnight q-storage build script
// v1.0.40-beta: Compile C++ EncryptionProvider for RocksDB

use std::env;
use std::path::PathBuf;

fn main() {
    // 🚧 Phase 2: C++ encryption temporarily disabled (missing RocksDB headers)
    // TODO Phase 3: Re-enable when RocksDB development headers are installed
    //
    // if cfg!(not(target_os = "windows")) {
    //     compile_cpp_encryption_provider();
    // }

    println!("cargo:rerun-if-changed=cpp/encryption_provider.h");
    println!("cargo:rerun-if-changed=cpp/encryption_provider.cpp");
    println!("cargo:rerun-if-changed=build.rs");
}

fn compile_cpp_encryption_provider() {
    let mut build = cc::Build::new();

    build
        .cpp(true)
        .std("c++17")
        .file("cpp/encryption_provider.cpp")
        .include("cpp")
        .warnings(true)
        .extra_warnings(true)
        .flag_if_supported("-Wno-unused-parameter");

    // Link against RocksDB
    println!("cargo:rustc-link-lib=dylib=rocksdb");

    // Add RocksDB include paths (common locations)
    let rocksdb_include_paths = vec![
        "/usr/include",
        "/usr/local/include",
        "/opt/homebrew/include",  // macOS ARM
    ];

    for path in rocksdb_include_paths {
        if PathBuf::from(path).join("rocksdb").exists() {
            build.include(path);
            break;
        }
    }

    // Compile
    build.compile("qnk_encryption");

    println!("cargo:rustc-link-search=native={}", env::var("OUT_DIR").unwrap());
}
