#!/bin/bash
# Quick Core System Test

echo "🚀 Q-NarwhalKnight Quick Core Test"
echo "==================================="
echo ""

# Test core packages only
CORE_PACKAGES="q-precision q-dag-knight q-narwhal-core q-storage q-network"

for package in $CORE_PACKAGES; do
    echo "Testing $package..."
    cargo test --package $package --lib 2>&1 | tail -5
    echo ""
done

echo "🎯 Running Consensus Simulation..."
cargo run --bin mitochondria-sim --release -- --validators 3 --rounds 5 2>&1 | head -30

echo ""
echo "✅ Core System Test Complete"
