#!/bin/bash
set -e

VERSION=$(grep '^version' /src/Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "=== Q-NarwhalKnight Build Environment (Debian 12) ==="
echo "    Version: $VERSION"
echo "    Target: $1"
echo ""

build_arm64() {
    echo ">>> Building ARM64 (aarch64-unknown-linux-gnu)..."
    # PKG_CONFIG_ALLOW_CROSS: let pkg-config find native .pc files during cross-compile
    # PKG_CONFIG_SYSROOT_DIR: don't prefix paths for cross sysroot
    PKG_CONFIG_ALLOW_CROSS=1 \
    cargo build --release --package q-api-server \
        --target aarch64-unknown-linux-gnu \
        --no-default-features --features tui,vendored-openssl

    cp target/aarch64-unknown-linux-gnu/release/q-api-server /output/q-api-server-v${VERSION}-linux-arm64
    cp target/aarch64-unknown-linux-gnu/release/q-api-server /output/q-api-server-linux-arm64
    echo ">>> ARM64 binary: /output/q-api-server-v${VERSION}-linux-arm64"

    # Also build miner if it exists
    if cargo build --release --package q-miner \
        --target aarch64-unknown-linux-gnu \
        --no-default-features 2>/dev/null; then
        cp target/aarch64-unknown-linux-gnu/release/q-miner /output/q-miner-v${VERSION}-linux-arm64
        cp target/aarch64-unknown-linux-gnu/release/q-miner /output/q-miner-linux-arm64
        echo ">>> ARM64 miner: /output/q-miner-v${VERSION}-linux-arm64"
    fi
}

build_windows() {
    echo ">>> Building Windows (x86_64-pc-windows-gnu)..."
    cargo build --release --package q-api-server \
        --target x86_64-pc-windows-gnu \
        --no-default-features --features tui,vendored-openssl

    cp target/x86_64-pc-windows-gnu/release/q-api-server.exe /output/q-api-server-v${VERSION}-windows-x64.exe
    cp target/x86_64-pc-windows-gnu/release/q-api-server.exe /output/q-api-server-windows-x64.exe
    echo ">>> Windows binary: /output/q-api-server-v${VERSION}-windows-x64.exe"

    # Also build miner if it exists
    if cargo build --release --package q-miner \
        --target x86_64-pc-windows-gnu \
        --no-default-features 2>/dev/null; then
        cp target/x86_64-pc-windows-gnu/release/q-miner.exe /output/q-miner-v${VERSION}-windows-x64.exe
        cp target/x86_64-pc-windows-gnu/release/q-miner.exe /output/q-miner-windows-x64.exe
        echo ">>> Windows miner: /output/q-miner-v${VERSION}-windows-x64.exe"
    fi
}

mkdir -p /output

case "$1" in
    arm64)
        build_arm64
        ;;
    windows)
        build_windows
        ;;
    all)
        build_arm64
        build_windows
        ;;
    *)
        echo "Usage: $0 {arm64|windows|all}"
        exit 1
        ;;
esac

echo ""
echo "=== Build complete ==="
ls -lh /output/q-*
