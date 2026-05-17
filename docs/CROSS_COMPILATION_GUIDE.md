# Q-NarwhalKnight Cross-Compilation Guide

Complete guide for building the miner (`q-miner`) and node (`q-api-server`) for Linux ARM64, Windows x64, and macOS ARM64 from an x86_64 Linux host.

---

## Prerequisites

### System Packages

```bash
# ARM64 cross-compiler toolchain
apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# ARM64 libraries needed for linking
dpkg --add-architecture arm64
echo "deb [arch=arm64] http://deb.debian.org/debian bookworm main" > /etc/apt/sources.list.d/arm64.list
apt-get update
apt-get install -y liblzma-dev:arm64

# Windows cross-compiler toolchain (MinGW)
apt-get install -y gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64-posix mingw-w64-x86-64-dev
apt-get install -y gfortran-mingw-w64-x86-64  # For LAPACK/numerical libs
# IMPORTANT: Runtime DLLs are in this package (libstdc++-6.dll, libgcc_s_seh-1.dll)
apt-get install -y gcc-mingw-w64-x86-64-posix-runtime

# Verify cc1plus exists (required for C++ deps like esaxx-rs)
ls /usr/lib/gcc/x86_64-w64-mingw32/12-posix/cc1plus
# If missing: apt-get install --reinstall g++-mingw-w64-x86-64-posix
```

### Rust Targets

```bash
rustup target add aarch64-unknown-linux-gnu
rustup target add x86_64-pc-windows-gnu
```

### Cargo Config

Add to `.cargo/config.toml`:

```toml
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"

[target.x86_64-pc-windows-gnu]
linker = "x86_64-w64-mingw32-gcc"
```

---

## Build Matrix

| Target | Miner | Node |
|--------|-------|------|
| Linux x86_64 | Working | Working (primary platform) |
| Linux ARM64 | Working | Working |
| Windows x64 | Working | Working (Sled backend, no RocksDB) |
| macOS ARM64 | Working (miner only) | Not attempted |

---

## 1. Linux ARM64 Miner

### Build

```bash
export CC_aarch64_linux_gnu=aarch64-linux-gnu-gcc
export CXX_aarch64_linux_gnu=aarch64-linux-gnu-g++

cargo build --release \
  --target aarch64-unknown-linux-gnu \
  --package q-miner \
  --features vendored-openssl
```

### Key Fixes Applied

1. **OpenSSL**: Uses `vendored-openssl` feature to compile OpenSSL from source (avoids needing ARM64 libssl-dev)
2. **raw-cpuid**: Gated with `#[cfg(target_arch = "x86_64")]` in `crates/q-miner/src/main.rs` - provides ARM fallback for hardware detection
3. **q-storage**: Made optional in miner's Cargo.toml (miner doesn't use it)

### Strip & Deploy

```bash
aarch64-linux-gnu-strip target/aarch64-unknown-linux-gnu/release/q-miner
cp target/aarch64-unknown-linux-gnu/release/q-miner \
   gui/quantum-wallet/dist-final/downloads/q-miner-linux-arm64
```

---

## 2. Linux ARM64 Node

### Build

```bash
export CC_aarch64_linux_gnu=aarch64-linux-gnu-gcc
export CXX_aarch64_linux_gnu=aarch64-linux-gnu-g++
export PKG_CONFIG_ALLOW_CROSS=1
export PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig

cargo build --release \
  --target aarch64-unknown-linux-gnu \
  --package q-api-server \
  --no-default-features \
  --features "tui,vendored-openssl"
```

**IMPORTANT**: Must use `--no-default-features` to exclude `resonance` feature, then re-enable `tui` and `vendored-openssl`.

### Key Fixes Applied

1. **OpenSSL**: Same vendored approach as miner

2. **q-resonance (lax crate i8/u8 mismatch)**: ARM64 `c_char` is `u8` but x86 is `i8`. The `lax` crate (via `ndarray-linalg`) fails on ARM64. Fixed by making `q-resonance` optional:
   - `crates/q-api-server/Cargo.toml`: `q-resonance = { ..., optional = true }`, feature `resonance = ["q-resonance", "q-network/resonance"]`
   - `crates/q-network/Cargo.toml`: `q-resonance = { ..., optional = true }`, feature `resonance = ["q-resonance"]`
   - `crates/q-api-server/src/lib.rs`: `#[cfg(feature = "resonance")]` on imports and struct fields
   - `crates/q-api-server/src/main.rs`: `#[cfg(feature = "resonance")]` on K-Parameter init and shadow mode
   - `crates/q-api-server/src/handlers.rs`: `#[cfg(feature = "resonance")]` on 6 handler locations
   - `crates/q-network/src/lib.rs`: `#[cfg(feature = "resonance")]` on `pub mod resonance_protocol` and its re-exports

3. **hidapi/libudev (q-quantum-rng)**: Hardware USB deps fail cross-compilation. Fixed by making them optional behind the `hardware` feature:
   - `crates/q-quantum-rng/Cargo.toml`: `hidapi`, `udev`, `serialport` all `optional = true`, enabled by `hardware = ["serialport", "hidapi", "udev"]`

4. **liblzma**: Linker needs ARM64 version. Install with `apt-get install liblzma-dev:arm64` (requires arm64 architecture added to dpkg and Debian official repo in sources.list)

### Strip & Deploy

```bash
aarch64-linux-gnu-strip target/aarch64-unknown-linux-gnu/release/q-api-server
cp target/aarch64-unknown-linux-gnu/release/q-api-server \
   gui/quantum-wallet/dist-final/downloads/q-api-server-linux-arm64
```

---

## 3. Windows x64 Miner

### Build

```bash
export CC_x86_64_pc_windows_gnu=x86_64-w64-mingw32-gcc
export CXX_x86_64_pc_windows_gnu=x86_64-w64-mingw32-g++

cargo build --release \
  --target x86_64-pc-windows-gnu \
  --package q-miner \
  --features vendored-openssl
```

### Key Fixes Applied

1. **OpenSSL**: Same vendored approach
2. **q-storage**: Made optional (same as ARM64 miner) - miner doesn't need the storage layer
3. **MinGW headers**: If `pqcrypto-internals` fails with "stdlib.h not found", run: `apt-get install --reinstall mingw-w64-x86-64-dev`
4. **cc1plus missing**: If `esaxx-rs` fails with "cannot execute cc1plus", run: `apt-get install --reinstall g++-mingw-w64-x86-64-posix`

### Strip & Deploy

```bash
x86_64-w64-mingw32-strip target/x86_64-pc-windows-gnu/release/q-miner.exe
cp target/x86_64-pc-windows-gnu/release/q-miner.exe \
   gui/quantum-wallet/dist-final/downloads/q-miner-windows-x64.exe
```

---

## 4. Windows x64 Node

### Build

```bash
# Recommended build command (tested 2026-02-19):
# IMPORTANT: LZMA_API_STATIC=1 forces lzma-sys to build from vendored source
# instead of using pkg-config (which finds the Linux liblzma and fails to link)
LZMA_API_STATIC=1 cargo build --release \
  --target x86_64-pc-windows-gnu \
  --package q-api-server \
  --no-default-features \
  --features tui

# Strip the binary (204MB → 145MB):
x86_64-w64-mingw32-strip target/x86_64-pc-windows-gnu/release/q-api-server.exe
```

**IMPORTANT**: Must use `--no-default-features` to exclude `llama-cpp` (default feature). The `llama-cpp` feature compiles ggml C++ code which fails on MinGW cross-compilation (`repack.cpp` stringop-overflow errors in CMake build). The `tui` feature works fine on Windows. The `resonance` feature is already disabled in defaults (OOM issues).

**Feature flags explained:**
- `--no-default-features`: Excludes `tui` and `llama-cpp` (both default)
- `--features tui`: Re-enables TUI (works on Windows)
- Do NOT add `llama-cpp`: Fails with MinGW (ggml/repack.cpp build errors)
- `vendored-openssl`: Optional, only needed if OpenSSL linking fails

### Architecture: Sled Instead of RocksDB

On Windows, the node uses **Sled** (pure-Rust embedded database) instead of **RocksDB** (C++ library with complex build requirements). This is controlled by platform-specific dependencies in `q-storage/Cargo.toml`:

```toml
[target.'cfg(not(target_os = "windows"))'.dependencies]
rocksdb = { workspace = true }
libc = "0.2"
memmap2 = "0.9"

[target.'cfg(target_os = "windows")'.dependencies]
sled = "0.34"
```

The `KVStore` trait in `crates/q-storage/src/kv.rs` provides a unified interface. RocksDB implements it via `kv_rocksdb.rs` (Linux/macOS), Sled via `kv_sled.rs` (Windows).

### Features Disabled on Windows

The following features are gated with `#[cfg(not(target_os = "windows"))]` and unavailable on Windows:

| Feature | Reason | Impact |
|---------|--------|--------|
| QNO Oracle Storage | Uses RocksDB column families directly | Oracle predictions unavailable |
| AsyncStorageEngine | Wraps RocksDB write batching | Uses synchronous writes instead |
| SafeBatchedWriter | RocksDB-specific batch optimization | Uses standard KVStore writes |
| Pointer Integrity Checks | RocksDB DB handle access | Skipped on startup |
| Preflight DB Verification | RocksDB-specific checks | Skipped on startup |
| Integrity Checker | RocksDB column family iteration | Skipped |
| DAG Sync Manager | Uses async storage engine | Falls back to standard sync |
| mmap Zero-Copy Blocks | memmap2 crate (Linux-only) | Standard file I/O instead |
| mdns Peer Discovery | Linux-only network feature | Disabled, uses bootstrap peers |

**Core functionality preserved**: Block validation, P2P sync, wallet, DEX, mining, API server all work on Windows.

### Key Fixes Applied (16 files, ~120 locations)

#### q-storage crate (storage layer)

| File | Changes |
|------|---------|
| `lib.rs` | `#[cfg]` guards on `get_rocks_db_handle()`, token balance iterator, RocksDB-specific column family methods |
| `kv_sled.rs` | Added 6 missing `KVStore` trait methods: `write_batch_turbo`, `create_checkpoint`, `sync_wal`, `shutdown_gracefully`, `verify_checkpoint`, `multi_get`. Added `get_raw_db()` returning `Arc<()>` |
| `zerocopy_blocks.rs` | Conditional `memmap2::Mmap` field → `Option<()>` on Windows, dual `enable_mmap()` implementations |
| `transaction.rs` | Conditional `RocksDBKV`/`SledKV` import |
| `turbo_sync.rs` | Gated `memmap2` import |
| `async_engine.rs` | Entire module gated (RocksDB-only) |
| `async_pipeline.rs` | Entire module gated |
| `integrity.rs` | Entire module gated |
| `pointer_integrity.rs` | Entire module gated |
| `preflight_check.rs` | Entire module gated |
| `safe_batched_writer.rs` | Entire module gated |
| `qno_storage.rs` | Entire module gated |
| `state_applicator.rs` | Gated RocksDB imports |
| `block_state_processor.rs` | Gated RocksDB imports |
| `db_util.rs` | Gated RocksDB imports |
| `token_registry.rs` | `new()` takes `Arc<dyn KVStore>` on Windows (not `Arc<RocksDB>`); iterator methods return empty vecs |
| `price_history.rs` | Windows stub with no-op methods; fixed `get_historical_candles` signature to match Linux (5 params, by-value DateTime, Option<usize> limit) |

#### q-network crate

| File | Changes |
|------|---------|
| `lib.rs` | `#[cfg(not(target_os = "windows"))]` on `mod mdns_discovery` |

#### q-dex crate

| File | Changes |
|------|---------|
| `token_registry.rs` | Ungated from Windows - uses `KVStore` trait, not RocksDB directly |
| `price_history.rs` | Windows stub implementation |

#### q-api-server crate (main application)

| File | Changes |
|------|---------|
| `lib.rs` | Gated `AppState` fields: `fast_sync_metrics`, `async_storage`, `qno_storage`. Gated their construction sites (2 locations). Gated integrity check block |
| `main.rs` | Gated imports: `AsyncStorageEngine`, `QnoStorage`. Gated blocks: SafeBatchedWriter init, AsyncStorageEngine init, pointer integrity, preflight check, integrity checker, QNO storage init, QNO routes, DagSyncManager, block pruning, shutdown cleanup. Fixed TUI error to use `anyhow::anyhow!()` |
| `handlers.rs` | Gated `async_storage` metrics access, `fast_sync_metrics` handler. Made `SyncMetricsResponse.metrics` use `serde_json::Value` on Windows |
| `oracle_integration.rs` | Added Windows stub types for `OutcomeType` enum and `PredictionOutcome` struct |
| `oauth2_provider.rs` | Fixed `rng.gen()` to `rng.gen::<u8>()` (type inference fails on Windows target) |
| `qno_api.rs` | Module declaration gated. Added type annotation for `resolve_domain_predictions` result |

### Gating Patterns Used

```rust
// Pattern 1: Import guard
#[cfg(not(target_os = "windows"))]
use rocksdb::{DB, ColumnFamily, Options};

// Pattern 2: Entire module gate
#[cfg(not(target_os = "windows"))]
mod qno_api;

// Pattern 3: Struct field gate (must also gate all construction sites!)
pub struct AppState {
    pub storage: Arc<KVStore>,  // Always present
    #[cfg(not(target_os = "windows"))]
    pub async_storage: Option<Arc<AsyncStorageEngine>>,  // Linux only
}

// Pattern 4: Block gate in function body
fn init() {
    // ... common init ...
    #[cfg(not(target_os = "windows"))]
    {
        let engine = AsyncStorageEngine::new(db.clone());
        // ... RocksDB-specific init ...
    }
}

// Pattern 5: Route chain breaking for conditional routes
let app = app.route("/api/common", get(handler));
#[cfg(not(target_os = "windows"))]
let app = app.route("/api/v1/qno/stake", post(qno_api::stake));
let app = app.route("/api/other", get(other_handler));

// Pattern 6: Dual return type
#[cfg(not(target_os = "windows"))]
pub fn get_raw_db(&self) -> Arc<rocksdb::DB> { ... }
#[cfg(target_os = "windows")]
pub fn get_raw_db(&self) -> Arc<()> { Arc::new(()) }
```

### CRITICAL: Cascading Gate Rule

When you gate a struct field with `#[cfg]`, you MUST also gate:
1. **All construction sites** - every place that creates the struct
2. **All field accesses** - every `instance.field_name` usage
3. **All imports of the field's type** - if the type itself is gated

Failure to gate all three causes cascading compilation errors. Use `cargo check --target x86_64-pc-windows-gnu` iteratively to find all locations.

### Strip & Deploy

```bash
x86_64-w64-mingw32-strip target/x86_64-pc-windows-gnu/release/q-api-server.exe
cp target/x86_64-pc-windows-gnu/release/q-api-server.exe \
   gui/quantum-wallet/dist-final/downloads/q-api-server-windows-x64.exe
```

### Required Windows DLLs

The Windows node binary requires MinGW runtime DLLs in the same directory:

| DLL | Size | Source | Purpose |
|-----|------|--------|---------|
| `libstdc++-6.dll` | 23 MB | MinGW C++ runtime | C++ standard library (used by PQ crypto, RocksDB stubs) |
| `libgcc_s_seh-1.dll` | 651 KB | MinGW GCC runtime | GCC exception handling |
| `libwinpthread-1.dll` | 312 KB | MinGW pthread | POSIX thread compatibility |

```bash
# Extract DLLs from MinGW package (they may not be installed on disk)
cd /tmp
apt download gcc-mingw-w64-x86-64-posix-runtime
mkdir -p mingw-dlls
dpkg-deb -x gcc-mingw-w64-x86-64-posix-runtime_*.deb mingw-dlls/

# Copy to downloads
DLLS=/tmp/mingw-dlls/usr/lib/gcc/x86_64-w64-mingw32/12-posix
DOWNLOADS=gui/quantum-wallet/dist-final/downloads
cp $DLLS/libstdc++-6.dll $DOWNLOADS/
cp $DLLS/libgcc_s_seh-1.dll $DOWNLOADS/
cp /usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll $DOWNLOADS/
```

### Packaging ZIP (Complete Build + Package Pipeline)

```bash
# Full pipeline: build, strip, package, deploy (tested 2026-02-16)

# 1. Build both binaries
cargo build --release --target x86_64-pc-windows-gnu --package q-api-server --no-default-features --features tui
cargo build --release --target x86_64-pc-windows-gnu --package q-miner

# 2. Strip binaries (reduces ~30-40% size)
x86_64-w64-mingw32-strip target/x86_64-pc-windows-gnu/release/q-api-server.exe
x86_64-w64-mingw32-strip target/x86_64-pc-windows-gnu/release/q-miner.exe

# 3. Create package directory
mkdir -p /tmp/q-narwhalknight-windows
cp target/x86_64-pc-windows-gnu/release/q-api-server.exe /tmp/q-narwhalknight-windows/
cp target/x86_64-pc-windows-gnu/release/q-miner.exe /tmp/q-narwhalknight-windows/
cp gui/quantum-wallet/dist-final/downloads/lib*.dll /tmp/q-narwhalknight-windows/
cp gui/quantum-wallet/dist-final/downloads/install-windows.ps1 /tmp/q-narwhalknight-windows/

# 4. Create ZIP (use -j to flatten paths)
cd /tmp && zip -j q-narwhalknight-windows-x64.zip q-narwhalknight-windows/*

# 5. Deploy to downloads folder
DOWNLOADS=gui/quantum-wallet/dist-final/downloads
cp /tmp/q-narwhalknight-windows-x64.zip $DOWNLOADS/
cp target/x86_64-pc-windows-gnu/release/q-api-server.exe $DOWNLOADS/q-api-server-windows-x64.exe
cp target/x86_64-pc-windows-gnu/release/q-miner.exe $DOWNLOADS/q-miner-windows-x64.exe
```

**Expected output sizes (v7.1.3, 2026-02-16):**
| File | Unstripped | Stripped |
|------|-----------|---------|
| q-api-server.exe | 222 MB | 147 MB |
| q-miner.exe | 15 MB | 15 MB |
| ZIP package | - | 68 MB |

### Windows User Quick Start

```powershell
# Download all files to the same folder
Invoke-WebRequest -Uri "https://quillon.xyz/downloads/q-api-server-windows-x64.exe" -OutFile "q-api-server.exe"
Invoke-WebRequest -Uri "https://quillon.xyz/downloads/libstdc++-6.dll" -OutFile "libstdc++-6.dll"
Invoke-WebRequest -Uri "https://quillon.xyz/downloads/libgcc_s_seh-1.dll" -OutFile "libgcc_s_seh-1.dll"
Invoke-WebRequest -Uri "https://quillon.xyz/downloads/libwinpthread-1.dll" -OutFile "libwinpthread-1.dll"

# Run (use a port that's not blocked - 8080 may need admin rights)
.\q-api-server.exe --port 9090 --p2p-port 9001

# If port 8080 is needed, run as Administrator (right-click → Run as administrator)
.\q-api-server.exe --port 8080 --p2p-port 9001
```

**Windows Port Notes**:
- Port 8080 may require Administrator privileges (OS error 10013 = `WSAEACCES`)
- Hyper-V reserves random port ranges - check with `netsh interface ipv4 show excludedportrange protocol=tcp`
- Use `--port 9090` or another high port to avoid permission issues
- Windows Firewall may need an inbound rule for the P2P port (9001)

---

## 5. macOS ARM64 Miner

### Build (from macOS host)

```bash
cargo build --release --package q-miner
```

Note: macOS cross-compilation from Linux requires osxcross toolchain (not documented here). The macOS miner was built natively.

### Deploy

```bash
strip target/release/q-miner
cp target/release/q-miner gui/quantum-wallet/dist-final/downloads/q-miner-macos-arm64
```

---

## Troubleshooting

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `cannot find -llzma` (ARM64) | Missing ARM64 liblzma | `apt-get install liblzma-dev:arm64` |
| `expected *const u8, found *const i8` (ARM64) | lax crate c_char signedness | Exclude `resonance` feature with `--no-default-features` |
| `Unable to find libudev` (ARM64) | hidapi cross-compilation | Already fixed: hidapi is optional |
| `undefined reference to lzma_*` (Windows) | pkg-config finds Linux liblzma | Set `LZMA_API_STATIC=1` to force vendored build |
| `cannot execute cc1plus` (Windows) | Broken mingw g++ | `apt-get install --reinstall g++-mingw-w64-x86-64-posix` |
| `stdlib.h not found` (Windows) | Broken mingw headers | `apt-get install --reinstall mingw-w64-x86-64-dev` |
| `CpuId::new() not found` (ARM64) | raw-cpuid is x86-only | Already fixed: cfg-gated in miner |
| OpenSSL headers not found | Missing cross-platform headers | Use `--features vendored-openssl` |
| `rng.gen()` type inference (Windows) | Compiler can't infer type | Use `rng.gen::<u8>()` turbofish syntax |
| `libstdc++-6.dll not found` (Windows runtime) | Missing MinGW DLL | Extract from `gcc-mingw-w64-x86-64-posix-runtime` package |
| OS error 10013 (Windows runtime) | Port access denied | Run as Administrator or use `--port 9090` |
| `task was cancelled` (Windows runtime) | Cascading from port error | Fix the port issue first |

### Build Times (approximate, release mode on x86_64 Linux)

| Target | Miner | Node |
|--------|-------|------|
| Linux x86_64 (first build) | ~20 min | ~30 min |
| Linux ARM64 (first build) | ~20 min | ~15 min (deps cached from x86) |
| Linux ARM64 (incremental) | ~2 min | ~5 min |
| Windows x64 (first build) | ~25 min | ~30 min |
| Windows x64 (incremental) | ~3 min | ~9 min |

**NOTE**: Post-quantum crypto crates (pqcrypto-dilithium, pqcrypto-kyber, pqcrypto-sphincsplus) dominate build time. Always use `timeout 36000` for build commands.

### Binary Sizes (stripped)

| Target | Miner | Node |
|--------|-------|------|
| Linux x86_64 | ~15 MB | ~110 MB |
| Linux ARM64 | ~14 MB | ~112 MB |
| Windows x64 | ~22 MB | ~191 MB |
| macOS ARM64 | ~8 MB | N/A |

The Windows node binary is larger because:
- MinGW linking includes more runtime overhead
- Sled database adds overhead vs RocksDB's static C++ library
- Debug symbols are partially preserved even after stripping PE binaries

---

## Deployment Checklist

After building, deploy binaries to the downloads directory:

```bash
DOWNLOADS=/opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads

# ARM64 Miner
aarch64-linux-gnu-strip target/aarch64-unknown-linux-gnu/release/q-miner
cp target/aarch64-unknown-linux-gnu/release/q-miner $DOWNLOADS/q-miner-linux-arm64

# ARM64 Node
aarch64-linux-gnu-strip target/aarch64-unknown-linux-gnu/release/q-api-server
cp target/aarch64-unknown-linux-gnu/release/q-api-server $DOWNLOADS/q-api-server-linux-arm64

# Windows Miner
x86_64-w64-mingw32-strip target/x86_64-pc-windows-gnu/release/q-miner.exe
cp target/x86_64-pc-windows-gnu/release/q-miner.exe $DOWNLOADS/q-miner-windows-x64.exe

# Windows Node + DLLs
x86_64-w64-mingw32-strip target/x86_64-pc-windows-gnu/release/q-api-server.exe
cp target/x86_64-pc-windows-gnu/release/q-api-server.exe $DOWNLOADS/q-api-server-windows-x64.exe
# DLLs (extract from package if not on disk - see Section 4)
cp /tmp/mingw-dlls/usr/lib/gcc/x86_64-w64-mingw32/12-posix/libstdc++-6.dll $DOWNLOADS/
cp /tmp/mingw-dlls/usr/lib/gcc/x86_64-w64-mingw32/12-posix/libgcc_s_seh-1.dll $DOWNLOADS/
cp /usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll $DOWNLOADS/

# macOS Miner (built natively)
# cp target/release/q-miner $DOWNLOADS/q-miner-macos-arm64

# Verify all files
ls -lh $DOWNLOADS/q-miner-linux-arm64 \
       $DOWNLOADS/q-api-server-linux-arm64 \
       $DOWNLOADS/q-miner-windows-x64.exe \
       $DOWNLOADS/q-api-server-windows-x64.exe \
       $DOWNLOADS/libstdc++-6.dll \
       $DOWNLOADS/libgcc_s_seh-1.dll \
       $DOWNLOADS/libwinpthread-1.dll
```

Download URLs (served by nginx at quillon.xyz):
```
# Linux ARM64
wget https://quillon.xyz/downloads/q-miner-linux-arm64
wget https://quillon.xyz/downloads/q-api-server-linux-arm64

# Linux x86_64
wget https://quillon.xyz/downloads/q-api-server-linux-x86_64

# Windows x64 (download all 4 files to same folder!)
wget https://quillon.xyz/downloads/q-api-server-windows-x64.exe
wget https://quillon.xyz/downloads/q-miner-windows-x64.exe
wget https://quillon.xyz/downloads/libstdc++-6.dll
wget https://quillon.xyz/downloads/libgcc_s_seh-1.dll
wget https://quillon.xyz/downloads/libwinpthread-1.dll

# macOS ARM64
wget https://quillon.xyz/downloads/q-miner-macos-arm64
```

---

## Cargo.toml Changes Reference

### q-miner/Cargo.toml
```toml
# Added:
openssl = { version = "0.10", features = ["vendored"], optional = true }
q-storage = { path = "../q-storage", optional = true }

[features]
vendored-openssl = ["openssl"]
```

### q-api-server/Cargo.toml
```toml
# Added:
openssl = { version = "0.10", features = ["vendored"], optional = true }
q-resonance = { path = "../q-resonance", optional = true }

[features]
default = ["tui", "resonance"]
vendored-openssl = ["openssl"]
resonance = ["q-resonance", "q-network/resonance"]
```

### q-network/Cargo.toml
```toml
# Changed:
q-resonance = { path = "../q-resonance", optional = true }

[features]
resonance = ["q-resonance"]
```

### q-quantum-rng/Cargo.toml
```toml
# Changed (all made optional):
[target.'cfg(target_os = "linux")'.dependencies]
serialport = { version = "4.2", optional = true }
hidapi = { version = "2.4", optional = true }
udev = { version = "0.7", optional = true }

[features]
hardware = ["serialport", "hidapi", "udev"]
```

### q-storage/Cargo.toml
```toml
# Platform-specific storage backends:
[target.'cfg(not(target_os = "windows"))'.dependencies]
rocksdb = { workspace = true }
libc = "0.2"
memmap2 = "0.9"

[target.'cfg(target_os = "windows")'.dependencies]
sled = "0.34"
```

---

## Development: Adding New Code

When adding new code to q-api-server or q-storage, keep these rules in mind:

1. **Never use `rocksdb` types directly in q-api-server** - always go through the `KVStore` trait or `StorageEngine` abstraction
2. **If you must use RocksDB-specific features** (column families, iterators, etc.), wrap with `#[cfg(not(target_os = "windows"))]`
3. **New AppState fields using RocksDB types** must be gated AND all construction sites updated
4. **New modules using `q_storage::qno_storage` or other gated modules** must also be gated
5. **Test with both targets** after changes:
   ```bash
   # Quick check (catches all compilation errors)
   cargo check --package q-api-server                                              # Linux
   cargo check --target x86_64-pc-windows-gnu --package q-api-server --no-default-features --features vendored-openssl  # Windows
   ```
6. **Windows stubs must match Linux signatures exactly** - parameter count, types, and return types must be identical
7. **Use `#[serde(default)]` on new Optional fields** for backward compatibility with existing DB data
