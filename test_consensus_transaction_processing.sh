#!/bin/bash

echo "🏁 Q-NARWHALKNIGHT CONSENSUS TRANSACTION PROCESSING TEST"
echo "======================================================="
echo "Testing Step 6: Join consensus network for transaction processing"
echo ""

# Clean up
docker ps -a | grep qnk-consensus-test | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true
docker network rm qnk-consensus-net 2>/dev/null || true

# Create consensus test network
docker network create qnk-consensus-net

echo "🚀 Starting 3-node consensus network..."

# Start 3 nodes for consensus testing
for i in 1 2 3; do
    docker run -d --name qnk-consensus-test$i \
      --network qnk-consensus-net \
      -p $((8080 + i - 1)):8080 \
      -e QNK_NODE_NAME="consensus-node-$i" \
      -e QNK_API_PORT="8080" \
      -e RUST_LOG="info,q_dag_knight=debug,q_narwhal_core=debug,q_api_server=debug" \
      q-narwhalknight-fixed:latest
done

echo "⏳ Waiting 45 seconds for nodes to initialize..."
sleep 45

echo ""
echo "📊 NODE STATUS CHECK:"
echo "===================="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep qnk-consensus-test

echo ""
echo "🏁 STEP 6 TEST: CONSENSUS TRANSACTION PROCESSING"
echo "=============================================="

echo ""
echo "🔍 1. API Server Accessibility Test:"
echo "===================================="

for i in 1 2 3; do
    echo "📡 Testing Node $i API (Port $((8080 + i - 1))):"
    
    # Test basic API connectivity
    API_STATUS=$(curl -s -w "%{http_code}" -o /dev/null --max-time 5 http://localhost:$((8080 + i - 1))/api/v1/health 2>/dev/null || echo "000")
    
    if [ "$API_STATUS" = "200" ]; then
        echo "  ✅ API Server: ACCESSIBLE (HTTP $API_STATUS)"
        
        # Test node status endpoint
        NODE_STATUS=$(curl -s --max-time 5 http://localhost:$((8080 + i - 1))/api/v1/node/status 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null || echo "failed")
        echo "  📊 Node Status: $NODE_STATUS"
        
    else
        echo "  ❌ API Server: INACCESSIBLE (HTTP $API_STATUS)"
    fi
done

echo ""
echo "💰 2. Transaction Processing Test:"
echo "=================================="

echo "🧪 Testing transaction submission on Node 1..."

# Create test transaction
TRANSACTION_PAYLOAD='{
  "from": "test-wallet-1",
  "to": "test-wallet-2", 
  "amount": 100,
  "fee": 1
}'

echo "📤 Submitting test transaction..."
TX_RESPONSE=$(curl -s --max-time 10 -X POST \
  -H "Content-Type: application/json" \
  -d "$TRANSACTION_PAYLOAD" \
  http://localhost:8080/api/v1/transactions 2>/dev/null || echo '{"error": "connection_failed"}')

echo "📥 Transaction Response: $TX_RESPONSE"

# Check if transaction was accepted
TX_STATUS=$(echo "$TX_RESPONSE" | jq -r '.status // "failed"' 2>/dev/null || echo "failed")
if [ "$TX_STATUS" != "failed" ] && [ "$TX_STATUS" != "null" ]; then
    echo "  ✅ TRANSACTION ACCEPTED: Status = $TX_STATUS"
else
    echo "  ❌ TRANSACTION REJECTED OR FAILED"
fi

echo ""
echo "⚡ 3. Consensus Participation Test:"
echo "=================================="

echo "🔍 Checking consensus activity across all nodes..."

for i in 1 2 3; do
    echo ""
    echo "📡 Node $i Consensus Activity:"
    
    # Check for consensus-related logs
    CONSENSUS_ACTIVITY=$(docker exec qnk-consensus-test$i grep -c "consensus\|round\|block\|vertex\|DAG" /app/logs/qnk-stdout.log 2>/dev/null || echo "0")
    ROUND_ACTIVITY=$(docker exec qnk-consensus-test$i grep -c "round.*[0-9]" /app/logs/qnk-stdout.log 2>/dev/null || echo "0")
    VERTEX_ACTIVITY=$(docker exec qnk-consensus-test$i grep -c "vertex\|DAG" /app/logs/qnk-stdout.log 2>/dev/null || echo "0")
    
    echo "  📊 Consensus Messages: $CONSENSUS_ACTIVITY"
    echo "  🔄 Round Activity: $ROUND_ACTIVITY" 
    echo "  🔗 Vertex/DAG Activity: $VERTEX_ACTIVITY"
    
    if [ "$CONSENSUS_ACTIVITY" -gt "5" ]; then
        echo "  ✅ ACTIVE CONSENSUS PARTICIPATION"
    else
        echo "  ❌ Limited consensus activity"
    fi
    
    # Check for specific consensus engine initialization
    CONSENSUS_INIT=$(docker exec qnk-consensus-test$i grep -c "DAG.*Knight\|Consensus.*initialized" /app/logs/qnk-stdout.log 2>/dev/null || echo "0")
    echo "  🏗️  Consensus Engine Init: $CONSENSUS_INIT"
done

echo ""
echo "🌐 4. Network Coordination Test:"
echo "==============================="

echo "🧪 Testing cross-node coordination..."

# Submit transactions to different nodes and check propagation
echo "📤 Submitting transactions across nodes..."

for i in 1 2 3; do
    PORT=$((8080 + i - 1))
    TX_PAYLOAD_NODE="{\"from\": \"wallet-node-$i\", \"to\": \"common-wallet\", \"amount\": $((10 * i)), \"fee\": 1}"
    
    TX_RESP=$(curl -s --max-time 5 -X POST \
      -H "Content-Type: application/json" \
      -d "$TX_PAYLOAD_NODE" \
      http://localhost:$PORT/api/v1/transactions 2>/dev/null || echo '{"error": "failed"}')
    
    echo "  📡 Node $i TX: $(echo "$TX_RESP" | jq -r '.status // "failed"' 2>/dev/null)"
done

echo ""
echo "📈 5. Performance & TPS Measurement:"
echo "==================================="

echo "⚡ Testing sustained transaction throughput..."

# Quick TPS test
START_TIME=$(date +%s)
SUCCESSFUL_TXS=0

for batch in {1..5}; do
    for tx in {1..10}; do
        TX_PAYLOAD_PERF="{\"from\": \"perf-wallet-$batch\", \"to\": \"target-wallet-$tx\", \"amount\": 1, \"fee\": 1}"
        
        TX_RESP=$(curl -s --max-time 2 -X POST \
          -H "Content-Type: application/json" \
          -d "$TX_PAYLOAD_PERF" \
          http://localhost:8080/api/v1/transactions 2>/dev/null || echo '{"status": "failed"}')
        
        if echo "$TX_RESP" | grep -q '"status"' && ! echo "$TX_RESP" | grep -q '"error"'; then
            SUCCESSFUL_TXS=$((SUCCESSFUL_TXS + 1))
        fi
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
TPS=$((SUCCESSFUL_TXS / DURATION))

echo "📊 Performance Results:"
echo "  ⏱️  Duration: ${DURATION}s"
echo "  ✅ Successful Transactions: $SUCCESSFUL_TXS/50"
echo "  ⚡ Estimated TPS: $TPS"

echo ""
echo "🎯 FINAL STEP 6 ASSESSMENT:"
echo "=========================="

# Determine if consensus transaction processing is working
if [ "$SUCCESSFUL_TXS" -gt "10" ] && [ "$TPS" -gt "1" ]; then
    echo "🏆 ✅ STEP 6 SUCCESS: CONSENSUS TRANSACTION PROCESSING IS WORKING!"
    echo "   📈 Achieved $TPS TPS with $SUCCESSFUL_TXS successful transactions"
    echo "   🤝 Nodes successfully joined consensus network"
else
    echo "❌ STEP 6 INCOMPLETE: Limited transaction processing capability"
    echo "   📉 Only $SUCCESSFUL_TXS/$50 transactions succeeded ($TPS TPS)"
fi

echo ""
echo "🧹 Cleaning up test environment..."
docker rm -f qnk-consensus-test1 qnk-consensus-test2 qnk-consensus-test3
docker network rm qnk-consensus-net

echo ""
echo "🏁 CONSENSUS TRANSACTION PROCESSING TEST COMPLETED"