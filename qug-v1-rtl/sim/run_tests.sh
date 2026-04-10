#!/bin/bash
# QUG-V1 RTL — Run All Tests
set -e

echo "╔═══════════════════════════════════════════════════╗"
echo "║     QUG-V1 Mining SoC — RTL Test Suite           ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

PASS=0
FAIL=0

run_test() {
    local name=$1
    echo "━━━ Running: $name ━━━"
    if make "$name" 2>&1; then
        echo "✅ $name PASSED"
        ((PASS++))
    else
        echo "❌ $name FAILED"
        ((FAIL++))
    fi
    echo ""
}

run_test blake3
run_test core

echo "╔═══════════════════════════════════════════════════╗"
echo "║  Results: $PASS passed, $FAIL failed              "
echo "╚═══════════════════════════════════════════════════╝"

[ $FAIL -eq 0 ] && exit 0 || exit 1
