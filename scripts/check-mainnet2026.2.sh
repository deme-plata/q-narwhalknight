#!/bin/bash
# ============================================================================
# Mainnet 2026.2 Health Check Script
# Run on Delta (5.79.79.158) to verify canary soak test
# Usage: ./check-mainnet2026.2.sh [--continuous]
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

pass() { echo -e "  ${GREEN}✅ PASS${NC}: $1"; PASS=$((PASS+1)); }
fail() { echo -e "  ${RED}❌ FAIL${NC}: $1"; FAIL=$((FAIL+1)); }
warn() { echo -e "  ${YELLOW}⚠️  WARN${NC}: $1"; WARN=$((WARN+1)); }

API="http://localhost:8080"
GENESIS_TS=1771761600  # Feb 22, 2026 12:00 UTC
TARGET_ANNUAL=2625000  # QUG/year Era 0
TARGET_DAILY=$(echo "scale=2; $TARGET_ANNUAL / 365.25" | bc)  # ~7186.9 QUG/day
PROJECT_DIR="/opt/orobit/shared/q-narwhalknight"

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  MAINNET 2026.2 HEALTH CHECK — $(date -u '+%Y-%m-%d %H:%M:%S UTC')${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# ---- 1. Process Running ----
echo -e "${CYAN}[1/10] Process${NC}"
if pgrep -f "q-api-server" > /dev/null 2>&1; then
    PID=$(pgrep -f "q-api-server" | head -1)
    UPTIME=$(ps -o etimes= -p "$PID" 2>/dev/null | tr -d ' ')
    UPTIME_H=$(echo "scale=1; ${UPTIME:-0} / 3600" | bc)
    pass "q-api-server running (PID: $PID, uptime: ${UPTIME_H}h)"
else
    fail "q-api-server NOT running"
fi

# ---- 2. API Responding ----
echo -e "${CYAN}[2/10] API Health${NC}"
STATUS_RESP=$(curl -s --max-time 5 "$API/api/v1/status" 2>/dev/null || echo "FAIL")
if echo "$STATUS_RESP" | grep -q "success\|height\|version"; then
    VERSION=$(echo "$STATUS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('version',d.get('version','unknown')))" 2>/dev/null || echo "unknown")
    pass "API responding (version: $VERSION)"
else
    fail "API not responding or returned error"
fi

# ---- 3. Network ID ----
echo -e "${CYAN}[3/10] Network ID${NC}"
NETWORK_ID=$(echo "$STATUS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('network_id',d.get('network_id','unknown')))" 2>/dev/null || echo "unknown")
if [ "$NETWORK_ID" = "mainnet2026.2" ]; then
    pass "Network ID: mainnet2026.2"
elif [ "$NETWORK_ID" = "unknown" ]; then
    ENV_NET=$(grep Q_NETWORK_ID /etc/systemd/system/q-api-server.service 2>/dev/null | grep -o 'mainnet2026\.[0-9]*' || echo "unknown")
    if [ "$ENV_NET" = "mainnet2026.2" ]; then
        pass "Network ID from service file: mainnet2026.2"
    else
        warn "Cannot determine network ID (API returned: $NETWORK_ID)"
    fi
else
    fail "Wrong network ID: $NETWORK_ID (expected mainnet2026.2)"
fi

# ---- 4. Block Height ----
echo -e "${CYAN}[4/10] Block Production${NC}"
HEIGHT=$(echo "$STATUS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('height',d.get('height',0)))" 2>/dev/null || echo "0")
if [ "$HEIGHT" -gt 0 ] 2>/dev/null; then
    pass "Block height: $HEIGHT"
else
    NOW=$(date +%s)
    if [ "$NOW" -lt "$GENESIS_TS" ]; then
        REMAINING_H=$(( (GENESIS_TS - NOW) / 3600 ))
        warn "Block height 0 — genesis hasn't occurred yet (${REMAINING_H}h remaining)"
    else
        fail "Block height 0 — no blocks produced since genesis"
    fi
fi

# ---- 5. Mining / Emission ----
echo -e "${CYAN}[5/10] Mining & Emission${NC}"
if [ "$HEIGHT" -gt 10 ] 2>/dev/null; then
    REWARD_LOG=$(journalctl -u q-api-server --since "10 minutes ago" --no-pager 2>/dev/null | grep -oP 'reward[=: ]+\K[0-9.]+' | tail -5 || true)
    if [ -n "$REWARD_LOG" ]; then
        LAST_REWARD=$(echo "$REWARD_LOG" | tail -1)
        pass "Mining active (last reward: $LAST_REWARD QUG)"

        NOW=$(date +%s)
        ELAPSED_S=$((NOW - GENESIS_TS))
        if [ "$ELAPSED_S" -gt 3600 ]; then
            MINTED=$(curl -s --max-time 5 "$API/api/v1/status" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('total_minted',0))" 2>/dev/null || echo "0")
            if [ "$MINTED" != "0" ] && [ -n "$MINTED" ]; then
                DAILY_RATE=$(echo "scale=2; $MINTED / ($ELAPSED_S / 86400)" | bc 2>/dev/null || echo "unknown")
                if [ "$DAILY_RATE" != "unknown" ]; then
                    echo -e "     Daily emission rate: ~${DAILY_RATE} QUG/day (target: ${TARGET_DAILY})"
                    RATIO=$(echo "scale=2; $DAILY_RATE / $TARGET_DAILY" | bc 2>/dev/null || echo "1")
                    if [ "$(echo "$RATIO > 5" | bc -l 2>/dev/null)" = "1" ]; then
                        warn "Emission rate ${RATIO}x above target — check emission controller"
                    elif [ "$(echo "$RATIO < 0.2" | bc -l 2>/dev/null)" = "1" ]; then
                        warn "Emission rate ${RATIO}x below target — check block rate"
                    else
                        pass "Emission rate within expected range (${RATIO}x target)"
                    fi
                fi
            fi
        fi
    else
        warn "No recent mining reward logs found"
    fi
else
    if [ "$HEIGHT" = "0" ]; then
        warn "Skipping emission check — no blocks yet"
    else
        warn "Too few blocks ($HEIGHT) to check emission"
    fi
fi

# ---- 6. Memory Usage ----
echo -e "${CYAN}[6/10] Memory${NC}"
if pgrep -f "q-api-server" > /dev/null 2>&1; then
    PID=$(pgrep -f "q-api-server" | head -1)
    RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
    RSS_MB=$((RSS_KB / 1024))
    TOTAL_MB=$(free -m | awk '/^Mem:/{print $2}')
    PCT=$((RSS_MB * 100 / TOTAL_MB))

    if [ "$RSS_MB" -lt 2000 ]; then
        pass "Memory: ${RSS_MB}MB (${PCT}% of ${TOTAL_MB}MB)"
    elif [ "$RSS_MB" -lt 5000 ]; then
        warn "Memory: ${RSS_MB}MB (${PCT}% of ${TOTAL_MB}MB) — elevated"
    else
        fail "Memory: ${RSS_MB}MB (${PCT}% of ${TOTAL_MB}MB) — possible leak"
    fi
fi

# ---- 7. Disk Usage ----
echo -e "${CYAN}[7/10] Disk${NC}"
DATA_DIR="$PROJECT_DIR/data-mainnet2026.2"
if [ -d "$DATA_DIR" ]; then
    DATA_SIZE=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
    pass "Data directory: $DATA_DIR ($DATA_SIZE)"
else
    warn "Data directory not found at $DATA_DIR"
fi

# ---- 8. P2P Peers ----
echo -e "${CYAN}[8/10] P2P Network${NC}"
PEERS=$(echo "$STATUS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('peers',d.get('peers',d.get('data',{}).get('peer_count',0))))" 2>/dev/null || echo "0")
if [ "$PEERS" -gt 0 ] 2>/dev/null; then
    pass "Connected peers: $PEERS"
else
    warn "No P2P peers connected (expected on isolated canary)"
fi

# ---- 9. No Panics/Errors ----
echo -e "${CYAN}[9/10] Error Check${NC}"
PANIC_COUNT=$(journalctl -u q-api-server --since "1 hour ago" --no-pager -n 5000 2>/dev/null | grep -ci "panic\|SIGSEGV\|stack overflow\|thread.*panicked" 2>/dev/null || echo "0")
PANIC_COUNT=$(echo "$PANIC_COUNT" | tr -d '[:space:]')
PANIC_COUNT=${PANIC_COUNT:-0}
CRITICAL_COUNT=$(journalctl -u q-api-server --since "1 hour ago" --no-pager -n 5000 2>/dev/null | grep -ci "CRITICAL\|FATAL\|OOM\|Out of memory" 2>/dev/null || echo "0")
CRITICAL_COUNT=$(echo "$CRITICAL_COUNT" | tr -d '[:space:]')
CRITICAL_COUNT=${CRITICAL_COUNT:-0}
ERROR_COUNT=$(journalctl -u q-api-server --since "1 hour ago" --no-pager -n 5000 2>/dev/null | grep -ci "ERROR" 2>/dev/null || echo "0")
ERROR_COUNT=$(echo "$ERROR_COUNT" | tr -d '[:space:]')
ERROR_COUNT=${ERROR_COUNT:-0}

if [ "$PANIC_COUNT" -gt 0 ] 2>/dev/null; then
    fail "Found $PANIC_COUNT panic(s) in last hour!"
    journalctl -u q-api-server --since "1 hour ago" --no-pager -n 1000 2>/dev/null | grep -i "panic\|thread.*panicked" | tail -3
elif [ "$CRITICAL_COUNT" -gt 0 ] 2>/dev/null; then
    fail "Found $CRITICAL_COUNT CRITICAL error(s) in last hour!"
else
    if [ "$ERROR_COUNT" -gt 50 ] 2>/dev/null; then
        warn "High error count: $ERROR_COUNT errors in last hour"
    else
        pass "No panics, $ERROR_COUNT errors in last hour"
    fi
fi

# ---- 10. Service Config ----
echo -e "${CYAN}[10/10] Service Config${NC}"
if [ -f /etc/systemd/system/q-api-server.service ]; then
    SVC_NET=$(grep Q_NETWORK_ID /etc/systemd/system/q-api-server.service 2>/dev/null | grep -o 'mainnet2026\.[0-9]*' || echo "unknown")
    SVC_DB=$(grep Q_DB_PATH /etc/systemd/system/q-api-server.service 2>/dev/null | grep -o 'data-mainnet2026\.[0-9]*' || echo "unknown")
    if [ "$SVC_NET" = "mainnet2026.2" ] && [ "$SVC_DB" = "data-mainnet2026.2" ]; then
        pass "Service config: network=$SVC_NET, db=$SVC_DB"
    else
        fail "Service config mismatch: network=$SVC_NET, db=$SVC_DB"
    fi
else
    warn "No systemd service file found"
fi

# ---- Summary ----
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
TOTAL=$((PASS + FAIL + WARN))
echo -e "  Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${WARN} warnings${NC} (${TOTAL} checks)"
if [ "$FAIL" -gt 0 ]; then
    echo -e "  ${RED}STATUS: ISSUES DETECTED — investigate before launch${NC}"
elif [ "$WARN" -gt 3 ]; then
    echo -e "  ${YELLOW}STATUS: MOSTLY OK — review warnings${NC}"
else
    echo -e "  ${GREEN}STATUS: HEALTHY — canary looks good${NC}"
fi
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# ---- Continuous mode ----
if [ "$1" = "--continuous" ] || [ "$1" = "-c" ]; then
    echo "Running in continuous mode (every 5 minutes). Ctrl+C to stop."
    echo ""
    while true; do
        sleep 300
        echo ""
        echo "--- Re-checking at $(date -u '+%H:%M:%S UTC') ---"
        exec "$0" "--continuous"
    done
fi
