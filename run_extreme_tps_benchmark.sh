#!/bin/bash
# Extreme TPS Benchmark - Testing 1M+ TPS with ParallelWorkerPool
# Tests: SIMD + Kernel I/O + 10 Parallel Workers + Batch API

set -e

echo "🚀 Q-NarwhalKnight Extreme TPS Benchmark"
echo "========================================"
echo ""
echo "Target: 1,000,000+ TPS"
echo "Architecture: DAG-Knight + Narwhal + Bullshark"
echo "Optimizations:"
echo "  - SIMD Crypto (10x speedup)"
echo "  - Kernel I/O (100x speedup)"
echo "  - ParallelWorkerPool (10 workers)"
echo "  - Batch Transaction API (10k-50k tx/batch)"
echo ""

# Configuration
NUM_NODES=5
BASE_PORT=8100
BASE_P2P_PORT=9100
BATCH_SIZE=10000
NUM_BATCHES=100
TOTAL_TXS=$((BATCH_SIZE * NUM_BATCHES))

echo "Configuration:"
echo "  Nodes: $NUM_NODES"
echo "  Batch size: $BATCH_SIZE transactions"
echo "  Batches: $NUM_BATCHES"
echo "  Total transactions: $TOTAL_TXS"
echo ""

# Clean up previous runs
echo "🧹 Cleaning up previous test data..."
for i in $(seq 0 $((NUM_NODES - 1))); do
    rm -rf "./data-tps-node$i"
done
pkill -f "q-api-server.*--port 8[1-2][0-9][0-9]" || true
sleep 2

# Build if needed
if [ ! -f "./target/x86_64-unknown-linux-gnu/release/q-api-server" ]; then
    echo "⚙️  Building q-api-server..."
    timeout 36000 cargo build --release --package q-api-server
fi

echo ""
echo "🌟 Starting $NUM_NODES validator nodes with extreme optimizations..."
echo ""

# Start nodes with all optimizations enabled
for i in $(seq 0 $((NUM_NODES - 1))); do
    NODE_ID="extreme-node$i"
    PORT=$((BASE_PORT + i))
    P2P_PORT=$((BASE_P2P_PORT + i))
    DB_PATH="./data-tps-node$i"

    echo "Starting node $i: $NODE_ID (port $PORT, p2p $P2P_PORT)"

    # Enable ALL optimizations
    ENABLE_SIMD=1 \
    ENABLE_KERNEL_IO=1 \
    PARALLEL_WORKERS=10 \
    SKIP_TOR=1 \
    SKIP_BITCOIN=1 \
    SKIP_DNS=1 \
    Q_DB_PATH="$DB_PATH" \
    Q_P2P_PORT="$P2P_PORT" \
    ./target/x86_64-unknown-linux-gnu/release/q-api-server \
        --port "$PORT" \
        --node-id "$NODE_ID" \
        > "/tmp/extreme-node$i.log" 2>&1 &

    echo "  PID: $!"
done

echo ""
echo "⏳ Waiting 10 seconds for nodes to initialize..."
sleep 10

# Check node health
echo ""
echo "🏥 Checking node health..."
HEALTHY_NODES=0
for i in $(seq 0 $((NUM_NODES - 1))); do
    PORT=$((BASE_PORT + i))
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "  ✅ Node $i (port $PORT) is healthy"
        HEALTHY_NODES=$((HEALTHY_NODES + 1))
    else
        echo "  ❌ Node $i (port $PORT) is not responding"
    fi
done

if [ "$HEALTHY_NODES" -lt "$NUM_NODES" ]; then
    echo ""
    echo "⚠️  Only $HEALTHY_NODES/$NUM_NODES nodes are healthy. Continuing anyway..."
fi

echo ""
echo "📊 Starting extreme TPS benchmark..."
echo ""

# Create benchmark script
cat > /tmp/extreme_tps_test.sh << 'EOF'
#!/bin/bash
set -e

BASE_PORT=$1
BATCH_SIZE=$2
NUM_BATCHES=$3

# Generate transaction batch
generate_batch() {
    local size=$1
    echo "{"
    echo "  \"transactions\": ["
    for j in $(seq 1 $size); do
        # Generate random transaction
        FROM_ADDR=$(openssl rand -hex 32)
        TO_ADDR=$(openssl rand -hex 32)
        AMOUNT=$((RANDOM % 1000000 + 1))

        echo "    {"
        echo "      \"from\": \"$FROM_ADDR\","
        echo "      \"to\": \"$TO_ADDR\","
        echo "      \"amount\": $AMOUNT,"
        echo "      \"fee\": 1,"
        echo "      \"nonce\": $j"
        if [ $j -lt $size ]; then
            echo "    },"
        else
            echo "    }"
        fi
    done
    echo "  ]"
    echo "}"
}

echo "Submitting $NUM_BATCHES batches of $BATCH_SIZE transactions..."

START_TIME=$(date +%s)
SUBMITTED=0
FAILED=0

for i in $(seq 1 $NUM_BATCHES); do
    # Round-robin across nodes
    NODE_INDEX=$(( (i - 1) % 5 ))
    PORT=$((BASE_PORT + NODE_INDEX))

    # Generate batch
    BATCH=$(generate_batch $BATCH_SIZE)

    # Submit batch
    RESPONSE=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$BATCH" \
        "http://localhost:$PORT/api/v1/transactions/batch")

    # Check response
    if echo "$RESPONSE" | grep -q '"success":true'; then
        SUBMITTED=$((SUBMITTED + BATCH_SIZE))
        if [ $((i % 10)) -eq 0 ]; then
            ELAPSED=$(($(date +%s) - START_TIME))
            CURRENT_TPS=$((SUBMITTED / ELAPSED))
            echo "  Batch $i/$NUM_BATCHES: $SUBMITTED tx submitted ($CURRENT_TPS TPS)"
        fi
    else
        FAILED=$((FAILED + BATCH_SIZE))
        echo "  ❌ Batch $i failed"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
TOTAL_TPS=$((SUBMITTED / ELAPSED))

echo ""
echo "📈 Extreme TPS Benchmark Results:"
echo "  Total transactions: $SUBMITTED"
echo "  Failed: $FAILED"
echo "  Time: ${ELAPSED}s"
echo "  Average TPS: $TOTAL_TPS"
echo ""

if [ $TOTAL_TPS -gt 1000000 ]; then
    echo "🎉 SUCCESS: Achieved 1M+ TPS target!"
elif [ $TOTAL_TPS -gt 500000 ]; then
    echo "🚀 EXCELLENT: Achieved 500k+ TPS!"
elif [ $TOTAL_TPS -gt 100000 ]; then
    echo "✅ GOOD: Achieved 100k+ TPS!"
else
    echo "⚠️  Below target: $TOTAL_TPS TPS"
fi
EOF

chmod +x /tmp/extreme_tps_test.sh

# Run benchmark
/tmp/extreme_tps_test.sh $BASE_PORT $BATCH_SIZE $NUM_BATCHES

echo ""
echo "🔍 Node Performance Analysis..."
echo ""

# Collect node metrics
for i in $(seq 0 $((NUM_NODES - 1))); do
    PORT=$((BASE_PORT + i))
    echo "Node $i metrics:"

    # Get node stats
    METRICS=$(curl -s "http://localhost:$PORT/metrics/system" || echo "{}")

    if echo "$METRICS" | grep -q "tx_pool_size"; then
        TX_POOL=$(echo "$METRICS" | jq -r '.tx_pool_size // 0')
        echo "  Transaction pool: $TX_POOL"
    fi

    # Check log for optimization status
    if grep -q "SIMD.*enabled" "/tmp/extreme-node$i.log" 2>/dev/null; then
        echo "  ✅ SIMD optimization active"
    fi

    if grep -q "Kernel I/O.*enabled" "/tmp/extreme-node$i.log" 2>/dev/null; then
        echo "  ✅ Kernel I/O optimization active"
    fi

    if grep -q "ParallelWorkerPool.*10 workers" "/tmp/extreme-node$i.log" 2>/dev/null; then
        echo "  ✅ ParallelWorkerPool active (10 workers)"
    fi

    echo ""
done

echo "📝 Detailed logs available:"
for i in $(seq 0 $((NUM_NODES - 1))); do
    echo "  Node $i: /tmp/extreme-node$i.log"
done

echo ""
echo "🎯 Benchmark complete!"
echo ""
echo "To analyze results:"
echo "  tail -100 /tmp/extreme-node0.log"
echo ""
echo "To stop nodes:"
echo "  pkill -f 'q-api-server.*extreme-node'"