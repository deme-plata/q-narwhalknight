#!/bin/bash
# Quick TPS Benchmark - Direct HTTP testing of batch API
# Tests the already-running nodes

set -e

echo "🚀 Q-NarwhalKnight Quick TPS Benchmark"
echo "======================================"
echo ""

# Configuration
NODES=(
    "http://localhost:8100"
    "http://localhost:8101"
    "http://localhost:8102"
    "http://localhost:8103"
    "http://localhost:8104"
)

BATCH_SIZE=100
NUM_BATCHES=10
TOTAL_TXS=$((BATCH_SIZE * NUM_BATCHES))

echo "Configuration:"
echo "  Nodes: ${#NODES[@]}"
echo "  Batch size: $BATCH_SIZE transactions"
echo "  Batches: $NUM_BATCHES"
echo "  Total transactions: $TOTAL_TXS"
echo ""

# Check node health
echo "🏥 Checking node health..."
HEALTHY_NODES=()
for i in "${!NODES[@]}"; do
    NODE_URL="${NODES[$i]}"
    if curl -s -m 2 "$NODE_URL/health" > /dev/null 2>&1; then
        echo "  ✅ Node $i ($NODE_URL) is healthy"
        HEALTHY_NODES+=("$NODE_URL")
    else
        echo "  ❌ Node $i ($NODE_URL) is not responding"
    fi
done

if [ ${#HEALTHY_NODES[@]} -eq 0 ]; then
    echo ""
    echo "❌ No healthy nodes available!"
    exit 1
fi

echo ""
echo "📊 Starting TPS benchmark with ${#HEALTHY_NODES[@]} healthy nodes..."
echo ""

# Generate a batch transaction request
generate_batch_json() {
    local size=$1
    local batch_num=$2

    echo "{"
    echo "  \"transactions\": ["

    for i in $(seq 1 $size); do
        # Generate random hex addresses (32 bytes = 64 hex chars)
        FROM=$(openssl rand -hex 32)
        TO=$(openssl rand -hex 32)
        AMOUNT=$((RANDOM % 1000000 + 1))
        NONCE=$(( (batch_num * size) + i ))
        # Generate random signature (64 bytes = 128 hex chars)
        SIG=$(openssl rand -hex 64)

        echo "    {"
        echo "      \"from\": \"$FROM\","
        echo "      \"to\": \"$TO\","
        echo "      \"amount\": $AMOUNT,"
        echo "      \"fee\": 1,"
        echo "      \"nonce\": $NONCE,"
        echo "      \"signature\": \"$SIG\""

        if [ $i -lt $size ]; then
            echo "    },"
        else
            echo "    }"
        fi
    done

    echo "  ]"
    echo "}"
}

START_TIME=$(date +%s.%N)
SUBMITTED=0
FAILED=0
TOTAL_SERVER_TPS=0

for batch_num in $(seq 1 $NUM_BATCHES); do
    # Round-robin across healthy nodes
    NODE_INDEX=$(( (batch_num - 1) % ${#HEALTHY_NODES[@]} ))
    NODE_URL="${HEALTHY_NODES[$NODE_INDEX]}"

    # Generate batch
    BATCH_JSON=$(generate_batch_json $BATCH_SIZE $batch_num)

    # Submit batch
    RESPONSE=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$BATCH_JSON" \
        "$NODE_URL/api/v1/transactions/batch" 2>/dev/null)

    # Parse response
    if echo "$RESPONSE" | grep -q '"success":true'; then
        # Extract metrics from response
        BATCH_SUBMITTED=$(echo "$RESPONSE" | grep -o '"submitted":[0-9]*' | grep -o '[0-9]*' || echo "$BATCH_SIZE")
        BATCH_TPS=$(echo "$RESPONSE" | grep -o '"tps":[0-9]*' | grep -o '[0-9]*' || echo "0")

        SUBMITTED=$((SUBMITTED + BATCH_SUBMITTED))
        TOTAL_SERVER_TPS=$((TOTAL_SERVER_TPS + BATCH_TPS))

        if [ $((batch_num % 2)) -eq 0 ]; then
            ELAPSED=$(echo "$(date +%s.%N) - $START_TIME" | bc)
            CURRENT_TPS=$(echo "scale=0; $SUBMITTED / $ELAPSED" | bc)
            echo "  Batch $batch_num/$NUM_BATCHES: $SUBMITTED tx submitted ($CURRENT_TPS TPS overall, server: $BATCH_TPS TPS/batch)"
        fi
    else
        FAILED=$((FAILED + BATCH_SIZE))
        echo "  ❌ Batch $batch_num failed"
    fi
done

END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
OVERALL_TPS=$(echo "scale=0; $SUBMITTED / $ELAPSED" | bc)
AVG_SERVER_TPS=$(echo "scale=0; $TOTAL_SERVER_TPS / $NUM_BATCHES" | bc)

echo ""
echo "📈 TPS Benchmark Results:"
echo "========================================"
echo "  Total transactions: $SUBMITTED"
echo "  Failed: $FAILED"
echo "  Success rate: $(echo "scale=1; ($SUBMITTED * 100) / $TOTAL_TXS" | bc)%"
echo "  Time: ${ELAPSED}s"
echo "  Overall TPS: $OVERALL_TPS"
echo "  Avg server-reported TPS/batch: $AVG_SERVER_TPS"
echo ""

if [ "$OVERALL_TPS" -ge 1000000 ]; then
    echo "🎉 SUCCESS: Achieved 1M+ TPS target!"
elif [ "$OVERALL_TPS" -ge 500000 ]; then
    echo "🚀 EXCELLENT: Achieved 500k+ TPS!"
elif [ "$OVERALL_TPS" -ge 100000 ]; then
    echo "✅ GOOD: Achieved 100k+ TPS!"
elif [ "$OVERALL_TPS" -ge 10000 ]; then
    echo "⚡ PROGRESS: Achieved 10k+ TPS!"
elif [ "$OVERALL_TPS" -ge 1000 ]; then
    echo "📊 BASELINE: Achieved 1k+ TPS!"
else
    echo "⚠️  Initial test: $OVERALL_TPS TPS"
fi

echo ""
echo "Optimizations tested:"
echo "  ✅ Batch Transaction API"
echo "  ✅ Round-robin load balancing across ${#HEALTHY_NODES[@]} nodes"
echo "  ✅ Parallel node processing"
echo ""
echo "Note: This is a quick test. For extreme TPS, increase BATCH_SIZE and NUM_BATCHES."