# Cross-Platform Build Guide - Q-NarwhalKnight Miner

This guide explains how to build Q-NarwhalKnight miner binaries for macOS and Windows using Docker cross-compilation.

## Quick Start

### Build Docker Image (One-Time Setup)

```bash
cd /opt/orobit/shared/q-narwhalknight
docker build -f Dockerfile.cross-compile -t q-narwhalknight-cross:latest .
```

**Time:** ~15-20 minutes (downloads Rust, OSXCross, macOS SDK, MinGW)

### Build All Platform Binaries

```bash
docker run --rm \
  -v $(pwd):/workspace \
  q-narwhalknight-cross:latest \
  /usr/local/bin/build-cross.sh
```

**Output:**
- `target/x86_64-apple-darwin/release/q-miner` - macOS Intel (x86_64)
- `target/aarch64-apple-darwin/release/q-miner` - macOS Apple Silicon (ARM64)
- `target/x86_64-pc-windows-gnu/release/q-miner.exe` - Windows (x86_64)

## What's Included in the Docker Image

### Cross-Compilation Tools:
- **OSXCross** - macOS cross-compilation toolchain with SDK 14.5
- **MinGW-w64** - Windows cross-compilation (POSIX threads)
- **Rust** with targets: x86_64-apple-darwin, aarch64-apple-darwin, x86_64-pc-windows-gnu

### Resolved Issues:
- ✅ OpenBLAS cross-compilation (disabled for miner, using pure Rust alternatives)
- ✅ macOS SDK frameworks (CoreFoundation, SystemConfiguration)
- ✅ MinGW cc1 compiler (properly configured POSIX threads variant)
- ✅ Bindgen for macOS headers
- ✅ aws-lc-sys linker compatibility (forced OSXCross linker via RUSTFLAGS)
- ✅ All platform-specific dependencies

## Build Individual Platforms

### macOS Intel (x86_64)

```bash
docker run --rm \
  -v $(pwd):/workspace \
  q-narwhalknight-cross:latest \
  cargo build --release --target x86_64-apple-darwin --package q-miner \
  --no-default-features --features "cpu-mining,cli"
```

### macOS Apple Silicon (ARM64)

```bash
docker run --rm \
  -v $(pwd):/workspace \
  q-narwhalknight-cross:latest \
  cargo build --release --target aarch64-apple-darwin --package q-miner \
  --no-default-features --features "cpu-mining,cli"
```

### Windows (x86_64)

```bash
docker run --rm \
  -v $(pwd):/workspace \
  q-narwhalknight-cross:latest \
  cargo build --release --target x86_64-pc-windows-gnu --package q-miner \
  --no-default-features --features "cpu-mining,cli"
```

## Interactive Docker Shell

For debugging or manual builds:

```bash
docker run --rm -it \
  -v $(pwd):/workspace \
  q-narwhalknight-cross:latest \
  /bin/bash
```

Inside the container:
```bash
# Check toolchains
x86_64-apple-darwin23.5-clang --version
x86_64-w64-mingw32-gcc --version

# Manual build
cargo build --release --target x86_64-apple-darwin --package q-miner

# Check build logs
ls -lh /tmp/*.log
```

## Features Built

The Docker cross-compilation builds the miner with:
- ✅ **CPU Mining** - Multi-threaded CPU mining
- ✅ **CLI** - Command-line interface
- ❌ **Network** - Disabled to avoid OpenBLAS dependency
- ❌ **GUI** - Disabled (platform-specific)
- ❌ **CUDA** - Disabled (requires platform-specific libraries)

## Troubleshooting

### Docker Build Fails

```bash
# Clean and rebuild
docker rmi q-narwhalknight-cross:latest
docker build --no-cache -f Dockerfile.cross-compile -t q-narwhalknight-cross:latest .
```

### Cargo Build Fails in Container

```bash
# Enter container and check logs
docker run --rm -it \
  -v $(pwd):/workspace \
  q-narwhalknight-cross:latest \
  /bin/bash

# Inside container:
tail -100 /tmp/macos-x86_64-miner.log
tail -100 /tmp/windows-miner.log
```

### Permission Issues

```bash
# Fix ownership of built binaries
sudo chown -R $USER:$USER target/
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Cross-Platform Miner Build

on: [push, pull_request]

jobs:
  build-miner:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -f Dockerfile.cross-compile -t q-narwhalknight-cross:latest .

      - name: Build all platforms
        run: |
          docker run --rm \
            -v $(pwd):/workspace \
            q-narwhalknight-cross:latest \
            /usr/local/bin/build-cross.sh

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: miner-binaries
          path: |
            target/x86_64-apple-darwin/release/q-miner
            target/aarch64-apple-darwin/release/q-miner
            target/x86_64-pc-windows-gnu/release/q-miner.exe
```

## Performance Notes

**Build Times (on typical CI/CD runner):**
- Docker image build: ~15-20 minutes (one-time)
- macOS x86_64 miner: ~10-15 minutes
- macOS ARM64 miner: ~10-15 minutes
- Windows miner: ~10-15 minutes
- **Total:** ~30-45 minutes for all platforms

**Caching:**
The Docker image caches:
- Rust installation
- OSXCross toolchain
- macOS SDK
- All build tools

Subsequent builds only recompile changed code.

## Binary Distribution

After building, create release archives:

```bash
# macOS Intel
tar -czf q-miner-macos-x86_64.tar.gz \
  -C target/x86_64-apple-darwin/release q-miner

# macOS Apple Silicon
tar -czf q-miner-macos-aarch64.tar.gz \
  -C target/aarch64-apple-darwin/release q-miner

# Windows
zip q-miner-windows-x86_64.zip \
  target/x86_64-pc-windows-gnu/release/q-miner.exe
```

## Support

For issues with cross-compilation:
- GitHub: https://github.com/deme-plata/q-narwhalknight/issues
- Discord: https://discord.gg/jEhaYtAhfx

---

**Built with Docker cross-compilation** ⚡️🐳
