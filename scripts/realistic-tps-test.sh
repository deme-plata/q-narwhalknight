#!/bin/bash
# Realistic TPS Benchmark Script
#
# This tests ACTUAL blockchain throughput with:
# - Signature verification
# - Block inclusion
# - Finality measurement
#
# Usage:
#   ./scripts/realistic-tps-test.sh              # Test local node
#   ./scripts/realistic-tps-test.sh production   # Test production node
#   ./scripts/realistic-tps-test.sh quick        # Quick test (fewer transactions)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Configuration based on argument
case "${1:-local}" in
    production|prod)
        export Q_NODE_URL="http://185.182.185.227:8080"
        export Q_PHASE1_TX=5000
        export Q_PHASE2_TX=500
        export Q_PHASE3_TX=50
        echo "Testing PRODUCTION node: $Q_NODE_URL"
        ;;
    quick)
        export Q_NODE_URL="${Q_NODE_URL:-http://localhost:8080}"
        export Q_PHASE1_TX=1000
        export Q_PHASE2_TX=100
        export Q_PHASE3_TX=20
        echo "Quick test mode: $Q_NODE_URL"
        ;;
    local|*)
        export Q_NODE_URL="${Q_NODE_URL:-http://localhost:8080}"
        export Q_PHASE1_TX=10000
        export Q_PHASE2_TX=1000
        export Q_PHASE3_TX=100
        echo "Testing LOCAL node: $Q_NODE_URL"
        ;;
esac

export Q_BATCH_SIZE=100
export Q_WAIT_BLOCKS=3
export Q_FINALITY_TIMEOUT=30

echo ""
echo "=========================================="
echo "  REALISTIC TPS BENCHMARK"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Node URL:      $Q_NODE_URL"
echo "  Phase 1 TX:    $Q_PHASE1_TX (verified mempool)"
echo "  Phase 2 TX:    $Q_PHASE2_TX (block inclusion)"
echo "  Phase 3 TX:    $Q_PHASE3_TX (finality)"
echo "  Batch Size:    $Q_BATCH_SIZE"
echo ""

# Check if node is accessible
echo "Checking node connectivity..."
if curl -s --connect-timeout 5 "$Q_NODE_URL/health" > /dev/null 2>&1; then
    echo "  Node is accessible"
else
    echo "  ERROR: Cannot connect to $Q_NODE_URL"
    echo "  Make sure the node is running"
    exit 1
fi

# Get current height
HEIGHT=$(curl -s "$Q_NODE_URL/api/v1/status" 2>/dev/null | grep -o '"height":[0-9]*' | grep -o '[0-9]*' || echo "unknown")
echo "  Current height: $HEIGHT"
echo ""

# Run the benchmark
echo "Running benchmark..."
echo ""

cargo test --release -p q-tps-benchmark --test realistic_verified_tps -- --nocapture 2>&1

echo ""
echo "Benchmark complete!"
