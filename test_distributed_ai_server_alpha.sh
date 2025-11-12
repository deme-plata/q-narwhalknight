#!/bin/bash
# Distributed AI Testing Script for Server Alpha Connection
# Tests P2P distributed inference, model manager, and performance metrics
# Created: 2025-10-31

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Q-NarwhalKnight Distributed AI Test${NC}"
echo -e "${BLUE}  Testing Connection to Server Alpha${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration - UPDATE THIS WITH SERVER ALPHA IP/HOSTNAME
# Options:
# 1. If Server Alpha is accessible via domain: SERVER_ALPHA="quillon.xyz"
# 2. If Server Alpha is on same network: SERVER_ALPHA="192.168.1.X"
# 3. If Server Alpha is remote: SERVER_ALPHA="IP_ADDRESS"
SERVER_ALPHA="${1:-localhost}"  # First argument or default to localhost
PORT="${2:-8080}"               # Second argument or default to 8080

echo -e "${YELLOW}📡 Target Server: ${SERVER_ALPHA}:${PORT}${NC}"
echo ""

# Test 1: Check if Server Alpha is reachable
echo -e "${BLUE}[1/7] Testing Server Alpha connectivity...${NC}"
if wget -q --spider --timeout=5 "http://${SERVER_ALPHA}:${PORT}/health" 2>/dev/null; then
    echo -e "${GREEN}✅ Server Alpha is reachable${NC}"
else
    echo -e "${RED}❌ Cannot reach Server Alpha at http://${SERVER_ALPHA}:${PORT}${NC}"
    echo -e "${YELLOW}💡 Usage: $0 <server-alpha-ip> <port>${NC}"
    echo -e "${YELLOW}💡 Example: $0 192.168.1.10 8080${NC}"
    exit 1
fi
echo ""

# Test 2: Get Server Alpha Peer ID (for P2P bootstrap)
echo -e "${BLUE}[2/7] Fetching Server Alpha Peer ID (libp2p)...${NC}"
PEER_ID=$(wget -qO- "http://${SERVER_ALPHA}:${PORT}/api/v1/peer-id" 2>/dev/null || echo "{}")
if echo "$PEER_ID" | grep -q "peer_id"; then
    echo -e "${GREEN}✅ Server Alpha Peer ID obtained${NC}"
    echo "$PEER_ID" | jq '.' 2>/dev/null || echo "$PEER_ID"
else
    echo -e "${YELLOW}⚠️  Could not get peer ID (server may not have P2P enabled)${NC}"
fi
echo ""

# Test 3: Check General Performance Metrics
echo -e "${BLUE}[3/7] Fetching Performance Metrics...${NC}"
METRICS=$(wget -qO- "http://${SERVER_ALPHA}:${PORT}/metrics" 2>/dev/null || echo "{}")
if [ ! -z "$METRICS" ]; then
    echo -e "${GREEN}✅ Metrics retrieved${NC}"
    echo "$METRICS" | head -20
    echo ""

    # Parse key metrics
    if echo "$METRICS" | grep -q "peer_count"; then
        PEER_COUNT=$(echo "$METRICS" | grep "peer_count" | head -1)
        echo -e "${YELLOW}📊 $PEER_COUNT${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No metrics available${NC}"
fi
echo ""

# Test 4: Check AI Treasury Stats (distributed AI payment tracking)
echo -e "${BLUE}[4/7] Checking AI Treasury Stats...${NC}"
TREASURY=$(wget -qO- "http://${SERVER_ALPHA}:${PORT}/api/treasury/stats" 2>/dev/null || echo "{}")
if echo "$TREASURY" | grep -q "success"; then
    echo -e "${GREEN}✅ AI Treasury Stats retrieved${NC}"
    echo "$TREASURY" | jq '.' 2>/dev/null || echo "$TREASURY"
else
    echo -e "${YELLOW}⚠️  AI Treasury not available${NC}"
fi
echo ""

# Test 5: Check AI Wallet Usage (distributed inference payments)
echo -e "${BLUE}[5/7] Checking AI Wallet Usage...${NC}"
WALLET_USAGE=$(wget -qO- "http://${SERVER_ALPHA}:${PORT}/api/wallet/usage" 2>/dev/null || echo "{}")
if echo "$WALLET_USAGE" | grep -q "success"; then
    echo -e "${GREEN}✅ AI Wallet Usage retrieved${NC}"
    echo "$WALLET_USAGE" | jq '.' 2>/dev/null || echo "$WALLET_USAGE"
else
    echo -e "${YELLOW}⚠️  AI Wallet not available${NC}"
fi
echo ""

# Test 6: Test Model Manager Status (if available)
echo -e "${BLUE}[6/7] Testing Model Manager Status...${NC}"
# Try to get list of available AI models
MODELS=$(wget -qO- "http://${SERVER_ALPHA}:${PORT}/api/models" 2>/dev/null || echo "{}")
if echo "$MODELS" | grep -q "Mistral"; then
    echo -e "${GREEN}✅ Model Manager active${NC}"
    echo "$MODELS" | jq '.' 2>/dev/null || echo "$MODELS"
else
    echo -e "${YELLOW}⚠️  Model Manager endpoint not available${NC}"
fi
echo ""

# Test 7: Test Distributed AI Inference (if chat available)
echo -e "${BLUE}[7/7] Testing Distributed AI Chat Inference...${NC}"
echo -e "${YELLOW}📝 Creating test chat session...${NC}"

# Create a new chat
CHAT_ID=$(wget -qO- --post-data='{
  "title": "Distributed AI Test",
  "initial_model": "Mistral-7B-Instruct-v0.3"
}' \
--header='Content-Type: application/json' \
"http://${SERVER_ALPHA}:${PORT}/api/chat" 2>/dev/null | jq -r '.data.id' 2>/dev/null || echo "")

if [ ! -z "$CHAT_ID" ] && [ "$CHAT_ID" != "null" ]; then
    echo -e "${GREEN}✅ Chat created: $CHAT_ID${NC}"
    echo ""

    # Send a test message
    echo -e "${YELLOW}💬 Sending test inference request...${NC}"
    RESPONSE=$(wget -qO- --post-data='{
      "content": "What is quantum computing in one sentence?"
    }' \
    --header='Content-Type: application/json' \
    "http://${SERVER_ALPHA}:${PORT}/api/chat/${CHAT_ID}/message" 2>/dev/null || echo "{}")

    if echo "$RESPONSE" | grep -q "success"; then
        echo -e "${GREEN}✅ Distributed AI inference successful${NC}"
        echo "$RESPONSE" | jq '.data.content' 2>/dev/null || echo "$RESPONSE"
    else
        echo -e "${RED}❌ Inference failed${NC}"
        echo "$RESPONSE"
    fi
else
    echo -e "${YELLOW}⚠️  Could not create chat (chat API may be disabled)${NC}"
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}✅ Connection Status: Successful${NC}"
echo -e "${YELLOW}📡 Server: http://${SERVER_ALPHA}:${PORT}${NC}"
echo ""
echo -e "${BLUE}🎯 Next Steps for P2P Distributed AI:${NC}"
echo "1. Export Server Alpha as bootstrap peer:"
echo "   export Q_BOOTSTRAP_PEERS=\"/ip4/${SERVER_ALPHA}/tcp/9001\""
echo ""
echo "2. Start your local node with P2P enabled:"
echo "   Q_DB_PATH=./data-local Q_P2P_PORT=9002 ./q-api-server --port 8090"
echo ""
echo "3. Test distributed inference across both nodes:"
echo "   curl -X POST http://localhost:8090/api/chat/[id]/message \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"content\": \"Test distributed AI\"}'"
echo ""
echo -e "${GREEN}🚀 Distributed AI Testing Complete!${NC}"
