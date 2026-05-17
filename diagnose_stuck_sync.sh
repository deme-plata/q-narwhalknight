#!/bin/bash
# Diagnostic script for stuck blockchain sync at height 4810
# v0.9.66-beta

echo "=========================================="
echo "Q-NarwhalKnight Sync Diagnostic Tool"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check current node status
echo -e "${YELLOW}1. Current Node Status:${NC}"
if command -v curl &> /dev/null; then
    STATUS=$(curl -s http://localhost:8080/api/v1/node/status 2>/dev/null || curl -s http://localhost:9010/api/v1/node/status 2>/dev/null || echo "{}")

    if [ -n "$STATUS" ] && [ "$STATUS" != "{}" ]; then
        echo "$STATUS" | python3 -m json.tool 2>/dev/null || echo "$STATUS"

        CURRENT_HEIGHT=$(echo "$STATUS" | grep -o '"current_height":[0-9]*' | grep -o '[0-9]*')
        echo -e "\n${GREEN}Current Height: $CURRENT_HEIGHT${NC}"
    else
        echo -e "${RED}❌ Cannot connect to API server${NC}"
        echo "   Trying ports: 8080, 9010, 9008..."
    fi
else
    echo -e "${RED}❌ curl not found${NC}"
fi

echo ""
echo "=========================================="

# 2. Check if node is actually stuck
echo -e "${YELLOW}2. Checking if sync is progressing:${NC}"
echo "   Waiting 10 seconds to measure sync rate..."

HEIGHT1=$(curl -s http://localhost:8080/api/v1/node/status 2>/dev/null | grep -o '"current_height":[0-9]*' | grep -o '[0-9]*' || echo "0")
sleep 10
HEIGHT2=$(curl -s http://localhost:8080/api/v1/node/status 2>/dev/null | grep -o '"current_height":[0-9]*' | grep -o '[0-9]*' || echo "0")

if [ "$HEIGHT1" = "0" ] || [ "$HEIGHT2" = "0" ]; then
    echo -e "${RED}❌ Cannot measure sync rate (API not responding)${NC}"
elif [ "$HEIGHT2" -gt "$HEIGHT1" ]; then
    RATE=$(( (HEIGHT2 - HEIGHT1) / 10 ))
    echo -e "${GREEN}✅ Sync is progressing: $HEIGHT1 → $HEIGHT2 ($RATE blocks/sec)${NC}"
else
    echo -e "${RED}❌ Sync is STUCK: Height has not changed ($HEIGHT1 → $HEIGHT2)${NC}"
fi

echo ""
echo "=========================================="

# 3. Check for errors in logs
echo -e "${YELLOW}3. Recent Errors in Logs:${NC}"

# Try different log locations
if [ -f "/var/log/q-api-server.log" ]; then
    LOG_FILE="/var/log/q-api-server.log"
elif journalctl -u q-api-server -n 1 &>/dev/null; then
    echo "   Using journalctl..."
    journalctl -u q-api-server --no-pager -n 50 | grep -i -E "error|failed|stuck|panic" | tail -10
    LOG_FILE=""
else
    echo -e "${RED}❌ Cannot find logs${NC}"
    LOG_FILE=""
fi

if [ -n "$LOG_FILE" ]; then
    echo "   Log file: $LOG_FILE"
    grep -i -E "error|failed|stuck|panic" "$LOG_FILE" | tail -10
fi

echo ""
echo "=========================================="

# 4. Check Turbo Sync status
echo -e "${YELLOW}4. Turbo Sync Status:${NC}"

if [ -n "$LOG_FILE" ]; then
    echo "   Last Turbo Sync activities:"
    grep "TURBO SYNC" "$LOG_FILE" | tail -15
elif journalctl -u q-api-server -n 1 &>/dev/null; then
    journalctl -u q-api-server --no-pager -n 200 | grep "TURBO SYNC" | tail -15
else
    echo -e "${RED}❌ No logs available${NC}"
fi

echo ""
echo "=========================================="

# 5. Check height 4810 specifically
echo -e "${YELLOW}5. Activity at Height 4810:${NC}"

if [ -n "$LOG_FILE" ]; then
    grep "4810" "$LOG_FILE" | tail -20
elif journalctl -u q-api-server -n 1 &>/dev/null; then
    journalctl -u q-api-server --no-pager | grep "4810" | tail -20
else
    echo -e "${RED}❌ No logs available${NC}"
fi

echo ""
echo "=========================================="

# 6. Check for block producer issues
echo -e "${YELLOW}6. Block Producer Status:${NC}"

if [ -n "$LOG_FILE" ]; then
    grep -E "block producer|producing block" "$LOG_FILE" -i | tail -10
elif journalctl -u q-api-server -n 1 &>/dev/null; then
    journalctl -u q-api-server --no-pager -n 200 | grep -i -E "block producer|producing block" | tail -10
else
    echo -e "${RED}❌ No logs available${NC}"
fi

echo ""
echo "=========================================="

# 7. Check peer connectivity
echo -e "${YELLOW}7. P2P Network Status:${NC}"

PEERS=$(curl -s http://localhost:8080/api/v1/p2p/peers 2>/dev/null || curl -s http://localhost:9010/api/v1/p2p/peers 2>/dev/null || echo "{}")
if [ -n "$PEERS" ] && [ "$PEERS" != "{}" ]; then
    echo "$PEERS" | python3 -m json.tool 2>/dev/null || echo "$PEERS"
else
    echo -e "${RED}❌ Cannot get peer info${NC}"
fi

echo ""
echo "=========================================="

# 8. Check for deserialization errors
echo -e "${YELLOW}8. Deserialization Errors (MessagePack issue):${NC}"

if [ -n "$LOG_FILE" ]; then
    grep -i "deserialize.*failed\|incompatible binary" "$LOG_FILE" | tail -10
elif journalctl -u q-api-server -n 1 &>/dev/null; then
    journalctl -u q-api-server --no-pager -n 200 | grep -i "deserialize.*failed\|incompatible binary" | tail -10
else
    echo -e "${RED}❌ No logs available${NC}"
fi

echo ""
echo "=========================================="

# 9. Database check
echo -e "${YELLOW}9. Database Status:${NC}"

if [ -d "./data/q-narwhal-db" ]; then
    echo "   Database directory: ./data/q-narwhal-db"
    du -sh ./data/q-narwhal-db

    # Check for corruption indicators
    if [ -f "./data/q-narwhal-db/LOCK" ]; then
        echo -e "   ${GREEN}✅ Database lock file present${NC}"
    else
        echo -e "   ${RED}❌ Database lock file missing${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  Database directory not found in current location${NC}"
    echo "   Looking in common locations..."
    find /opt /var -name "q-narwhal-db" -type d 2>/dev/null | head -5
fi

echo ""
echo "=========================================="

# 10. Recommendations
echo -e "${YELLOW}10. Diagnostic Summary:${NC}"

if [ "$HEIGHT2" = "$HEIGHT1" ]; then
    echo -e "${RED}❌ SYNC IS STUCK${NC}"
    echo ""
    echo "Possible causes:"
    echo "1. Turbo Sync deserialization failure (MessagePack incompatibility)"
    echo "2. No peers available with blocks > 4810"
    echo "3. Block producer waiting for more transactions"
    echo "4. Database corruption at height 4810"
    echo ""
    echo "Recommended actions:"
    echo "1. Check if you need v0.9.66-beta fix (MessagePack deserialization)"
    echo "2. Restart the node: systemctl restart q-api-server"
    echo "3. Check peer connectivity"
    echo "4. If persistent, may need to reset database"
else
    echo -e "${GREEN}✅ Sync appears to be working${NC}"
fi

echo ""
echo "=========================================="
echo "Diagnostic complete. Please share this output for analysis."
echo "=========================================="
