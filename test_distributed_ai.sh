#!/bin/bash
# Test Distributed AI Horizontal Scaling

set -e

echo "🤖 Q-NarwhalKnight Distributed AI Test"
echo "======================================"
echo ""

# Step 1: Create a new chat with distributed AI enabled
echo "📝 Step 1: Creating new chat with distributed AI enabled..."
CHAT_RESPONSE=$(curl -s -X POST http://localhost:8080/api/chat/create \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Distributed AI Test",
    "model": "mistral-7b-instruct-v0.3-q4",
    "distributed_enabled": true,
    "enable_kv_cache": true
  }')

CHAT_ID=$(echo "$CHAT_RESPONSE" | jq -r '.chat_id')
echo "✅ Chat created: $CHAT_ID"
echo ""

# Step 2: Check main node logs for distributed AI coordinator
echo "📊 Step 2: Checking nodes for distributed AI capability..."
echo "Main Node (localhost:8080):"
journalctl -u q-api-server --since "10 seconds ago" 2>&1 | grep -E "(distributed|coordinator|🤖)" | tail -5 || echo "  (No recent distributed AI activity)"
echo ""

echo "Test Node (docker container):"
docker logs q-test-node 2>&1 | grep -E "(distributed|coordinator|🤖)" | tail -5 || echo "  (No recent distributed AI activity)"
echo ""

# Step 3: Send a test message with distributed inference
echo "💬 Step 3: Sending test message with distributed AI inference..."
echo "  Watching logs for distributed inference activity..."
echo ""

# Start log monitoring in background
(journalctl -u q-api-server -f --since "1 second ago" 2>&1 | grep -E "(DISTRIBUTED|🌐|network nodes|request_id)" | head -10) &
JOURNAL_PID=$!

(docker logs -f q-test-node 2>&1 | grep -E "(inference-request|🤖|Distributed)" | head -10) &
DOCKER_PID=$!

sleep 2

# Send the message
curl -s -X POST "http://localhost:8080/api/chat/$CHAT_ID/message" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Explain quantum consensus in 2 sentences.",
    "max_tokens": 100
  }' > /tmp/distributed_ai_response.json &

# Wait for inference to complete
sleep 15

# Kill log monitors
kill $JOURNAL_PID 2>/dev/null || true
kill $DOCKER_PID 2>/dev/null || true

echo ""
echo "📋 Step 4: Checking inference results..."
if [ -f /tmp/distributed_ai_response.json ]; then
  cat /tmp/distributed_ai_response.json | jq '.' 2>/dev/null || cat /tmp/distributed_ai_response.json
fi

echo ""
echo "✅ Distributed AI test complete!"
echo ""
echo "📊 Summary:"
echo "  - Main node: http://localhost:8080"
echo "  - Test node: http://localhost:8090 (Docker)"
echo "  - Chat ID: $CHAT_ID"
echo "  - Distributed AI: $(echo "$CHAT_RESPONSE" | jq -r '.metadata.distributed_enabled')"
echo ""
echo "🔍 To check detailed logs:"
echo "  Main node: journalctl -u q-api-server -f"
echo "  Test node: docker logs -f q-test-node"
