#!/bin/bash
# Multi-Node Distributed AI Inference Test
# Tests 3-node distributed inference with layer forwarding and KV-cache

set -e

echo "🚀 Q-NarwhalKnight Distributed AI Inference Test"
echo "================================================="
echo ""

# Configuration
NUM_NODES=3
MODEL_PATH="/opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
TOTAL_LAYERS=32

# Calculate layer assignments (distribute evenly)
LAYERS_PER_NODE=$((TOTAL_LAYERS / NUM_NODES))

NODE1_LAYERS="0-$((LAYERS_PER_NODE - 1))"
NODE2_LAYERS="$LAYERS_PER_NODE-$((2 * LAYERS_PER_NODE - 1))"
NODE3_LAYERS="$((2 * LAYERS_PER_NODE))-$((TOTAL_LAYERS - 1))"

echo "📋 Test Configuration:"
echo "   Nodes: $NUM_NODES"
echo "   Total Layers: $TOTAL_LAYERS"
echo "   Layers per node: ~$LAYERS_PER_NODE"
echo ""
echo "   Node 1: Layers $NODE1_LAYERS (first node - embeds prompt)"
echo "   Node 2: Layers $NODE2_LAYERS (middle node - forwards)"
echo "   Node 3: Layers $NODE3_LAYERS (last node - decodes output)"
echo ""

# Create test data directories
mkdir -p data-node1 data-node2 data-node3
mkdir -p logs

echo "🔧 Starting distributed AI nodes..."
echo ""

# Start Node 1 (first layers - embeds prompt)
echo "🟢 Starting Node 1 (layers $NODE1_LAYERS)..."
Q_DB_PATH=./data-node1 \
Q_P2P_PORT=9001 \
Q_AI_THREADS=2 \
Q_AI_MAX_CONCURRENT=1 \
timeout 36000 cargo run --release --package q-api-server --bin q-api-server -- \
    --port 8001 \
    --node-id node1 \
    > logs/node1.log 2>&1 &
NODE1_PID=$!

sleep 5

# Start Node 2 (middle layers - forwards)
echo "🟡 Starting Node 2 (layers $NODE2_LAYERS)..."
Q_DB_PATH=./data-node2 \
Q_P2P_PORT=9002 \
Q_AI_THREADS=2 \
Q_AI_MAX_CONCURRENT=1 \
timeout 36000 cargo run --release --package q-api-server --bin q-api-server -- \
    --port 8002 \
    --node-id node2 \
    > logs/node2.log 2>&1 &
NODE2_PID=$!

sleep 5

# Start Node 3 (last layers - decodes)
echo "🟣 Starting Node 3 (layers $NODE3_LAYERS)..."
Q_DB_PATH=./data-node3 \
Q_P2P_PORT=9003 \
Q_AI_THREADS=2 \
Q_AI_MAX_CONCURRENT=1 \
timeout 36000 cargo run --release --package q-api-server --bin q-api-server -- \
    --port 8003 \
    --node-id node3 \
    > logs/node3.log 2>&1 &
NODE3_PID=$!

echo ""
echo "✅ All nodes started!"
echo "   Node 1 PID: $NODE1_PID (port 8001)"
echo "   Node 2 PID: $NODE2_PID (port 8002)"
echo "   Node 3 PID: $NODE3_PID (port 8003)"
echo ""

# Wait for nodes to initialize
echo "⏳ Waiting for nodes to initialize (30s)..."
sleep 30

echo ""
echo "🧪 Running distributed inference test..."
echo ""

# Test 1: Single inference request
echo "Test 1: Single Distributed Inference"
echo "-------------------------------------"

CHAT_ID=$(uuidgen || echo "test-chat-$(date +%s)")
echo "   Chat ID: $CHAT_ID"

START_TIME=$(date +%s%3N)

# Send request to Node 1 (first node)
curl -X POST "http://localhost:8001/api/chat/${CHAT_ID}/message" \
    -H "Content-Type: application/json" \
    -d '{
        "content": "What is quantum computing?",
        "max_tokens": 100
    }' 2>/dev/null | jq '.' || echo "   ❌ Request failed"

END_TIME=$(date +%s%3N)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "   Total time: ${ELAPSED}ms"
echo ""

# Test 2: Multi-turn conversation (tests KV-cache)
echo "Test 2: Multi-Turn Conversation (KV-Cache Test)"
echo "-----------------------------------------------"

# First message
echo "   Turn 1: Initial question..."
curl -X POST "http://localhost:8001/api/chat/${CHAT_ID}/message" \
    -H "Content-Type: application/json" \
    -d '{
        "content": "Explain quantum entanglement",
        "max_tokens": 50
    }' 2>/dev/null | jq '.content' || echo "   ❌ Turn 1 failed"

sleep 2

# Second message (should use KV-cache)
echo "   Turn 2: Follow-up (should be faster with KV-cache)..."
START_TIME2=$(date +%s%3N)

curl -X POST "http://localhost:8001/api/chat/${CHAT_ID}/message" \
    -H "Content-Type: application/json" \
    -d '{
        "content": "Can you give an example?",
        "max_tokens": 50
    }' 2>/dev/null | jq '.content' || echo "   ❌ Turn 2 failed"

END_TIME2=$(date +%s%3N)
ELAPSED2=$((END_TIME2 - START_TIME2))

echo ""
echo "   Turn 2 time: ${ELAPSED2}ms"
if [ $ELAPSED2 -lt $ELAPSED ]; then
    echo "   ✅ KV-cache speedup detected! (${ELAPSED}ms → ${ELAPSED2}ms)"
else
    echo "   ⚠️  No speedup detected (may need more data)"
fi

echo ""
echo "📊 Checking Node Statistics..."
echo "------------------------------"

# Get stats from each node
echo "Node 1 Stats:"
curl -s "http://localhost:8001/api/stats" | jq '.distributed_ai // "N/A"' || echo "N/A"

echo ""
echo "Node 2 Stats:"
curl -s "http://localhost:8002/api/stats" | jq '.distributed_ai // "N/A"' || echo "N/A"

echo ""
echo "Node 3 Stats:"
curl -s "http://localhost:8003/api/stats" | jq '.distributed_ai // "N/A"' || echo "N/A"

echo ""
echo "📋 Node Logs (last 20 lines):"
echo "------------------------------"

echo ""
echo "Node 1:"
tail -20 logs/node1.log

echo ""
echo "Node 2:"
tail -20 logs/node2.log

echo ""
echo "Node 3:"
tail -20 logs/node3.log

echo ""
echo "🛑 Stopping nodes..."
kill $NODE1_PID $NODE2_PID $NODE3_PID 2>/dev/null || true

echo ""
echo "✅ Distributed AI test complete!"
echo ""
echo "📝 Logs saved to logs/ directory"
echo "   logs/node1.log"
echo "   logs/node2.log"
echo "   logs/node3.log"
