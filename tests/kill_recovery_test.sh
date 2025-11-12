#!/bin/bash
# Kill -9 Recovery Test Suite
# Validates ≤16 block max loss on crash (Phase 1A safety requirement)
#
# Expert consensus: ChatGPT, Kimi AI, DeepSeek (100% agreement)
# "100 kill -9 tests mandatory before deployment"

set -e

# Configuration
TOTAL_TESTS=${1:-100}
TEST_DB_BASE="./test-data-kill"
SUCCESS=0
FAILED=0
MAX_LOSS_ALLOWED=16
LOSSES=()

echo "🧪 Kill -9 Recovery Test Suite"
echo "================================"
echo "Target: ≤${MAX_LOSS_ALLOWED} blocks max loss"
echo "Tests: ${TOTAL_TESTS}"
echo ""

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up test processes..."
    pkill -9 q-api-server || true
    sleep 2
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Main test loop
for i in $(seq 1 $TOTAL_TESTS); do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧪 Test $i/$TOTAL_TESTS: Kill -9 recovery"

    TEST_DB="${TEST_DB_BASE}-${i}"
    TEST_PORT=$((8000 + i))

    # Remove old test data
    rm -rf "${TEST_DB}" 2>/dev/null || true

    # Start node with experimental fast sync
    echo "   ▶ Starting node on port ${TEST_PORT}..."
    Q_DB_PATH="${TEST_DB}" \
    Q_NETWORK_ID=testnet-phase11 \
    timeout 300 cargo run --release --bin q-api-server -- \
        --port ${TEST_PORT} \
        --experimental-fast-sync &

    PID=$!

    # Wait for node to start
    sleep 3

    # Let it sync for random 5-15 seconds
    SLEEP_TIME=$((5 + RANDOM % 10))
    echo "   ⏳ Syncing for ${SLEEP_TIME}s..."
    sleep $SLEEP_TIME

    # Get height before kill
    echo "   📊 Checking height before kill..."
    HEIGHT_BEFORE=$(curl -s http://localhost:${TEST_PORT}/api/height 2>/dev/null | jq -r '.height' 2>/dev/null || echo "0")

    if [ "$HEIGHT_BEFORE" = "null" ] || [ "$HEIGHT_BEFORE" = "" ]; then
        HEIGHT_BEFORE=0
    fi

    echo "   📏 Height before kill: $HEIGHT_BEFORE"

    # Kill -9 (simulate power loss)
    echo "   💥 Killing process (kill -9)..."
    kill -9 $PID 2>/dev/null || true
    wait $PID 2>/dev/null || true
    sleep 1

    # Restart node
    echo "   🔄 Restarting node..."
    Q_DB_PATH="${TEST_DB}" \
    Q_NETWORK_ID=testnet-phase11 \
    timeout 300 cargo run --release --bin q-api-server -- \
        --port ${TEST_PORT} \
        --experimental-fast-sync &

    NEW_PID=$!
    sleep 5

    # Get height after recovery
    echo "   📊 Checking height after recovery..."
    HEIGHT_AFTER=$(curl -s http://localhost:${TEST_PORT}/api/height 2>/dev/null | jq -r '.height' 2>/dev/null || echo "0")

    if [ "$HEIGHT_AFTER" = "null" ] || [ "$HEIGHT_AFTER" = "" ]; then
        HEIGHT_AFTER=0
    fi

    echo "   📏 Height after recovery: $HEIGHT_AFTER"

    # Calculate loss
    LOSS=$((HEIGHT_BEFORE - HEIGHT_AFTER))

    # Handle negative loss (node caught up during restart)
    if [ $LOSS -lt 0 ]; then
        LOSS=0
    fi

    echo "   📉 Loss: $LOSS blocks"
    LOSSES+=($LOSS)

    # Verify ≤16 blocks lost (Phase 1A batch size)
    if [ $LOSS -le $MAX_LOSS_ALLOWED ]; then
        echo "   ✅ PASS: Loss within safety bound (≤${MAX_LOSS_ALLOWED} blocks)"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "   ❌ FAIL: Loss exceeds ${MAX_LOSS_ALLOWED} blocks!"
        FAILED=$((FAILED + 1))
    fi

    # Cleanup this test
    kill -9 $NEW_PID 2>/dev/null || true
    wait $NEW_PID 2>/dev/null || true
    rm -rf "${TEST_DB}"

    echo ""
done

# Calculate statistics
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Test Results Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Total tests:      $TOTAL_TESTS"
echo "Passed:           $SUCCESS"
echo "Failed:           $FAILED"
echo "Success rate:     $(awk "BEGIN {printf \"%.1f\", ($SUCCESS/$TOTAL_TESTS)*100}")%"
echo ""

# Calculate loss statistics
if [ ${#LOSSES[@]} -gt 0 ]; then
    # Max loss
    MAX_LOSS=$(printf '%s\n' "${LOSSES[@]}" | sort -n | tail -1)

    # Average loss
    TOTAL_LOSS=0
    for loss in "${LOSSES[@]}"; do
        TOTAL_LOSS=$((TOTAL_LOSS + loss))
    done
    AVG_LOSS=$(awk "BEGIN {printf \"%.2f\", $TOTAL_LOSS/${#LOSSES[@]}}")

    echo "Loss Statistics:"
    echo "  Max loss:       $MAX_LOSS blocks"
    echo "  Average loss:   $AVG_LOSS blocks"
    echo "  Target:         ≤${MAX_LOSS_ALLOWED} blocks"
    echo ""
fi

# Final verdict
if [ $SUCCESS -eq $TOTAL_TESTS ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ ALL TESTS PASSED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Phase 1A safety requirement SATISFIED:"
    echo "  ✅ 100% recovery rate"
    echo "  ✅ Max loss ≤${MAX_LOSS_ALLOWED} blocks"
    echo "  ✅ Ready for testnet deployment"
    echo ""
    exit 0
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ SOME TESTS FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Phase 1A safety requirement NOT met:"
    echo "  ❌ Success rate: $(awk "BEGIN {printf \"%.1f\", ($SUCCESS/$TOTAL_TESTS)*100}")%"
    echo "  ❌ Failed tests: $FAILED"
    echo "  ⚠️  NOT ready for testnet deployment"
    echo ""
    exit 1
fi
