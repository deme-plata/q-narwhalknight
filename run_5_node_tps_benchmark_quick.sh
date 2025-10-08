#!/bin/bash
# Real 5-Node TPS Benchmark Test with Quantum Transport
# Quick version - uses existing binary

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

cd "$(dirname "${BASH_SOURCE[0]}")"

# Config
NUM_NODES=5
BASE_API_PORT=9081  # Changed to avoid conflict with port 8080-8081
BASE_P2P_PORT=7001
RESULTS_DIR="tps-benchmark-results-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Real 5-Node TPS Benchmark with Quantum Transport         ║${NC}"
echo -e "${CYAN}║  Kyber1024 + Dilithium5 (Phase 1)                         ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"

# Check binary exists
if [ ! -f "./target/x86_64-unknown-linux-gnu/release/q-api-server" ]; then
    echo -e "${RED}❌ Binary not found. Please run: cargo build --release --package q-api-server${NC}"
    exit 1
fi

# Cleanup
echo -e "${YELLOW}🧹 Cleaning up...${NC}"
killall q-api-server 2>/dev/null || true
sleep 2
for i in $(seq 1 $NUM_NODES); do
    rm -rf "data-tps-node$i"
    mkdir -p "data-tps-node$i"
done

# Start nodes
echo -e "${YELLOW}🚀 Starting $NUM_NODES nodes...${NC}"
declare -a NODE_PIDS

for i in $(seq 1 $NUM_NODES); do
    API_PORT=$((BASE_API_PORT + i - 1))
    P2P_PORT=$((BASE_P2P_PORT + i - 1))
    DATA_DIR="data-tps-node$i"
    LOG_FILE="$RESULTS_DIR/node$i.log"

    echo -e "${BLUE}  Starting Node $i (API: $API_PORT, P2P: $P2P_PORT)${NC}"

    Q_DB_PATH="./$DATA_DIR" \
    Q_P2P_PORT="$P2P_PORT" \
    RUST_LOG=info \
    timeout 36000 ./target/x86_64-unknown-linux-gnu/release/q-api-server \
        --port "$API_PORT" \
        --node-id "tps-benchmark-node-$i" \
        > "$LOG_FILE" 2>&1 &

    NODE_PIDS[$i]=$!
    echo -e "${GREEN}  ✅ Node $i started (PID: ${NODE_PIDS[$i]})${NC}"
    sleep 2  # Stagger startup
done

# Wait for initialization (Tor bootstrap + network setup)
echo -e "${YELLOW}⏳ Waiting for nodes to initialize...${NC}"
echo -e "${BLUE}  Phase 1: Tor bootstrap (60s)${NC}"
sleep 60
echo -e "${BLUE}  Phase 2: Network initialization (30s)${NC}"
sleep 30
echo -e "${BLUE}  Phase 3: Quantum transport setup (30s)${NC}"
sleep 30

# Health check
echo -e "${YELLOW}🔍 Checking node health...${NC}"
ALL_HEALTHY=true
for i in $(seq 1 $NUM_NODES); do
    API_PORT=$((BASE_API_PORT + i - 1))
    if curl -s -m 5 "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✅ Node $i healthy${NC}"
    else
        echo -e "${RED}  ❌ Node $i unhealthy${NC}"
        ALL_HEALTHY=false
    fi
done

if [ "$ALL_HEALTHY" = false ]; then
    echo -e "${RED}❌ Not all nodes healthy. Check logs in $RESULTS_DIR/${NC}"
    for pid in "${NODE_PIDS[@]}"; do kill $pid 2>/dev/null || true; done
    exit 1
fi

# Check quantum transport
echo -e "${CYAN}⚛️  Checking quantum transport...${NC}"
sleep 2
for i in $(seq 1 $NUM_NODES); do
    if grep -q "Quantum Transport" "$RESULTS_DIR/node$i.log" 2>/dev/null; then
        echo -e "${GREEN}  ✅ Node $i: Quantum transport active${NC}"
    fi
done

# TPS test function
test_tps() {
    local tps=$1
    local duration=$2
    local phase=$3

    echo -e "${CYAN}📊 $phase: Target $tps TPS for ${duration}s${NC}"

    local node_port=$BASE_API_PORT
    local delay=$(awk "BEGIN {print 1.0/$tps}")
    local count=0
    local success=0
    local start=$(date +%s)

    while [ $(($(date +%s) - start)) -lt $duration ]; do
        curl -s -X POST "http://localhost:$node_port/api/quillon-bank/faucet" \
            -H "Content-Type: application/json" \
            -d "{\"wallet_address\":\"bench-$RANDOM\"}" \
            > /dev/null 2>&1 && ((success++)) || true
        ((count++))

        if [ $count -eq 50 ]; then
            actual_tps=$(awk "BEGIN {print $success / ($(date +%s) - $start + 1)}")
            echo -e "${BLUE}  Progress: $count tx, ~${actual_tps%.*} TPS${NC}"
            count=0
        fi

        sleep "$delay"
    done

    local elapsed=$(($(date +%s) - start))
    local actual_tps=$(awk "BEGIN {print $success / $elapsed}")
    echo -e "${GREEN}✅ $phase: $success tx in ${elapsed}s = ${actual_tps%.*} TPS${NC}"
    echo "$phase,$tps,$duration,$success,$elapsed,${actual_tps%.*}" >> "$RESULTS_DIR/results.csv"
}

# Initialize results
echo "Phase,Target_TPS,Duration,Success_Count,Actual_Duration,Actual_TPS" > "$RESULTS_DIR/results.csv"

# Run test phases
test_tps 50 20 "Warmup-50"
test_tps 100 30 "Ramp-100"
test_tps 200 30 "Ramp-200"
test_tps 500 60 "Peak-500"
test_tps 1000 60 "Peak-1000"

# Cooldown
echo -e "${YELLOW}⏳ Cool-down (10s)...${NC}"
sleep 10

# Collect metrics
echo -e "${CYAN}📊 Collecting metrics...${NC}"
echo ""
echo "Quantum Transport Stats:"
for i in $(seq 1 $NUM_NODES); do
    LOG="$RESULTS_DIR/node$i.log"
    HANDSHAKES=$(grep -c "quantum.*handshake" "$LOG" 2>/dev/null || echo "0")
    BROADCASTS=$(grep -c "Broadcasting.*quantum" "$LOG" 2>/dev/null || echo "0")
    echo "  Node $i: $HANDSHAKES handshakes, $BROADCASTS broadcasts"
done

# Generate report
cat > "$RESULTS_DIR/REPORT.md" << EOF
# 5-Node TPS Benchmark Report

Date: $(date)
Nodes: $NUM_NODES
Quantum Transport: Kyber1024 + Dilithium5

## Results
\`\`\`
$(cat "$RESULTS_DIR/results.csv" | column -t -s,)
\`\`\`

## Quantum Stats
$(for i in $(seq 1 $NUM_NODES); do
    LOG="$RESULTS_DIR/node$i.log"
    echo "Node $i: $(grep -c "quantum" "$LOG" 2>/dev/null || echo "0") quantum operations"
done)

## Average TPS
$(awk -F, 'NR>1 {sum+=$6; count++} END {print sum/count " TPS"}' "$RESULTS_DIR/results.csv")
EOF

# Shutdown
echo -e "${YELLOW}🛑 Stopping nodes...${NC}"
for pid in "${NODE_PIDS[@]}"; do kill $pid 2>/dev/null || true; done
sleep 3
killall q-api-server 2>/dev/null || true

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Benchmark Complete!                                       ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}Results: $RESULTS_DIR/REPORT.md${NC}"
echo ""
cat "$RESULTS_DIR/results.csv" | column -t -s,
echo ""
echo -e "${CYAN}⚛️  REAL Quantum Physics - Kyber1024 + Dilithium5${NC}"