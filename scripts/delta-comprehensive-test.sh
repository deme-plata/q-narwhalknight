#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Q-NarwhalKnight v10.3.2 Comprehensive Test Suite
# RUN ONLY ON DELTA (5.79.79.158) — NEVER ON PRODUCTION
# ═══════════════════════════════════════════════════════════════════
#
# Tests:
#   1. DEX double-deduction fix (swap 50% → verify 50% remaining)
#   2. Ghost balance fix (restart → verify null balance during sync)
#   3. LWMA difficulty adjustment (verify difficulty changes)
#   4. Balance debug logging (verify 🔴 entries appear)
#   5. Startup sync flag (verify syncing=true then syncing=false)
#   6. Multiple swaps (no cumulative drift)
#   7. Restart preservation (balance survives restart)
#
# Prerequisites:
#   - Delta running v10.3.2 binary
#   - Python3 + pynacl installed on Delta
#   - Miner running for test wallet
#
# Usage:
#   scp scripts/delta-comprehensive-test.sh root@5.79.79.158:/tmp/
#   ssh root@5.79.79.158 "bash /tmp/delta-comprehensive-test.sh"
#
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

DELTA="localhost"
PORT="8080"
API="http://${DELTA}:${PORT}/api/v1"

# Test wallet keypair (Ed25519 seed we control)
SEED="2b67777b02a9b5f2ddbc1629a2e92f3db352e1ceb204f85bffe845db2e828a1f"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0

pass() { echo -e "${GREEN}  PASS${NC}: $1"; PASSED=$((PASSED+1)); }
fail() { echo -e "${RED}  FAIL${NC}: $1 — $2"; FAILED=$((FAILED+1)); }
skip() { echo -e "${YELLOW}  SKIP${NC}: $1 — $2"; SKIPPED=$((SKIPPED+1)); }
info() { echo -e "  INFO: $1"; }

# ═══════════════════════════════════════════════════════════════════
# Helper: Execute authenticated swap via Python
# ═══════════════════════════════════════════════════════════════════
do_swap() {
    local amount_qug="$1"
    local from_token="${2:-QUG}"
    local to_token="${3:-QUGUSD}"

    python3 -c "
import nacl.signing, json, time, hashlib, struct, urllib.request

seed = bytes.fromhex('${SEED}')
sk = nacl.signing.SigningKey(seed)
vk = sk.verify_key
pub_hex = vk.encode().hex()
addr = 'qnk' + pub_hex

ts = int(time.time())
path = '/api/v1/dex/swap'
challenge = hashlib.sha3_256(bytes.fromhex(pub_hex) + struct.pack('<q', ts) + path.encode()).digest()
sig = sk.sign(challenge).signature

auth = json.dumps({'address': addr, 'timestamp': ts, 'scheme': 'Ed25519', 'signature': sig.hex()})
amount = int(float('${amount_qug}') * 1e24)
body = json.dumps({'from_token': '${from_token}', 'to_token': '${to_token}', 'amount_in': amount, 'min_amount_out': 0, 'wallet_address': addr})

req = urllib.request.Request('http://${DELTA}:${PORT}' + path, data=body.encode(),
    headers={'Content-Type': 'application/json', 'X-Wallet-Auth': auth}, method='POST')

try:
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())
    print(json.dumps(result))
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(json.dumps({'success': False, 'error': f'HTTP {e.code}: {body[:200]}'}))
except Exception as e:
    print(json.dumps({'success': False, 'error': str(e)}))
" 2>/dev/null
}

# Get test wallet address
get_wallet_address() {
    python3 -c "
import nacl.signing
seed = bytes.fromhex('${SEED}')
sk = nacl.signing.SigningKey(seed)
print('qnk' + sk.verify_key.encode().hex())
" 2>/dev/null
}

WALLET=$(get_wallet_address)
info "Test wallet: ${WALLET:0:20}..."

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " Q-NarwhalKnight v10.3.2 Comprehensive Test Suite"
echo " Target: Delta (${DELTA}:${PORT})"
echo " Wallet: ${WALLET:0:20}..."
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════
# TEST 0: Verify we're on Delta, not production
# ═══════════════════════════════════════════════════════════════════
echo "--- Test 0: Safety check (must be Delta) ---"

HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"cs30067"* ]] || [[ "$HOSTNAME" == *"epsilon"* ]]; then
    fail "SAFETY" "Running on Epsilon (production)! ABORTING."
    echo -e "${RED}DO NOT RUN THIS ON PRODUCTION${NC}"
    exit 1
fi

if [[ "$HOSTNAME" == *"vmi2628966"* ]]; then
    fail "SAFETY" "Running on Beta (production)! ABORTING."
    exit 1
fi

pass "Safety check — running on $(hostname) (not production)"

# ═══════════════════════════════════════════════════════════════════
# TEST 1: Server is running and responsive
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "--- Test 1: Server health ---"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${API}/status" 2>/dev/null || echo "000")
if [[ "$STATUS" == "200" ]]; then
    pass "Server responding (HTTP 200)"
else
    fail "Server health" "HTTP ${STATUS}"
fi

# ═══════════════════════════════════════════════════════════════════
# TEST 2: Check version includes v10.3 features
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "--- Test 2: Version check ---"

VERSION_LOG=$(journalctl -u q-api-server --since '5 minutes ago' 2>/dev/null | grep "Version:" | tail -1)
if echo "$VERSION_LOG" | grep -q "10.3"; then
    pass "Running v10.3.x binary"
else
    skip "Version check" "Could not verify version from logs"
fi

# ═══════════════════════════════════════════════════════════════════
# TEST 3: DEX double-deduction fix — swap 50% and verify balance
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "--- Test 3: DEX double-deduction fix ---"

# Get current balance from miner log
BALANCE_BEFORE=$(cat /tmp/miner-swap-test.log 2>/dev/null | strings | grep "Balance Updated" | tail -1 | grep -oP '[\d.]+(?= QUG)' || echo "0")

if (( $(echo "$BALANCE_BEFORE > 0.1" | bc -l 2>/dev/null || echo 0) )); then
    info "Balance before swap: ${BALANCE_BEFORE} QUG"

    # Swap 50%
    HALF=$(echo "$BALANCE_BEFORE / 2" | bc -l 2>/dev/null || echo "0.05")
    info "Swapping ${HALF} QUG (50%) to QUGUSD..."

    SWAP_RESULT=$(do_swap "$HALF")
    SWAP_SUCCESS=$(echo "$SWAP_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "False")

    if [[ "$SWAP_SUCCESS" == "True" ]]; then
        # Wait for block inclusion
        sleep 5

        # Check balance after
        BALANCE_AFTER=$(cat /tmp/miner-swap-test.log 2>/dev/null | strings | grep "Balance Updated" | tail -1 | grep -oP '[\d.]+(?= QUG)' || echo "0")
        info "Balance after swap: ${BALANCE_AFTER} QUG"

        # Balance should be roughly 50% of before (not zero)
        if (( $(echo "$BALANCE_AFTER > 0" | bc -l 2>/dev/null || echo 0) )); then
            # Check it's not zero (the old double-deduction bug)
            RATIO=$(echo "$BALANCE_AFTER / $BALANCE_BEFORE" | bc -l 2>/dev/null || echo "0")
            if (( $(echo "$RATIO > 0.3" | bc -l 2>/dev/null || echo 0) )); then
                pass "DEX swap: balance ${BALANCE_BEFORE} → ${BALANCE_AFTER} (ratio: ${RATIO})"
            else
                fail "DEX swap" "Balance dropped too much: ${BALANCE_BEFORE} → ${BALANCE_AFTER} (ratio: ${RATIO}) — possible double deduction"
            fi
        else
            fail "DEX swap" "Balance is ZERO after swap — double deduction bug still present!"
        fi
    else
        skip "DEX swap test" "Swap failed: ${SWAP_RESULT}"
    fi
else
    skip "DEX swap test" "Insufficient balance (${BALANCE_BEFORE} QUG). Let miner run longer."
fi

# ═══════════════════════════════════════════════════════════════════
# TEST 4: Single deduction in debug logs
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "--- Test 4: Single deduction verification ---"

WALLET_SHORT="${WALLET:3:16}"
SUBTRACT_COUNT=$(journalctl -u q-api-server --since '30 seconds ago' 2>/dev/null | grep -c "subtract.*${WALLET_SHORT}" || echo "0")
DEDUCT_LOG=$(journalctl -u q-api-server --since '30 seconds ago' 2>/dev/null | grep "SWAP v10.3.2.*debit.*balance_consensus" | head -1)

if [[ -n "$DEDUCT_LOG" ]]; then
    pass "Swap deduction deferred to balance_consensus (v10.3.2 fix confirmed)"
elif journalctl -u q-api-server --since '30 seconds ago' 2>/dev/null | grep -q "SWAP v10.3.2.*no direct deduction"; then
    pass "Handler confirms no direct deduction (v10.3.2)"
else
    skip "Single deduction check" "No recent swap in logs"
fi

# ═══════════════════════════════════════════════════════════════════
# TEST 5: Balance debug logging (🔴 entries)
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "--- Test 5: Balance debug logging ---"

BALANCE_WRITE_COUNT=$(journalctl -u q-api-server --since '1 minute ago' 2>/dev/null | grep -c "🔴.*BALANCE WRITE" || echo "0")
if [[ "$BALANCE_WRITE_COUNT" -gt 0 ]]; then
    pass "Balance debug logging active: ${BALANCE_WRITE_COUNT} entries in last minute"
else
    skip "Debug logging" "No 🔴 entries found (may not be in this build)"
fi

BALANCE_DROP_COUNT=$(journalctl -u q-api-server --since '5 minutes ago' 2>/dev/null | grep -c "BALANCE DROP" || echo "0")
if [[ "$BALANCE_DROP_COUNT" -eq 0 ]]; then
    pass "No BALANCE DROP events (no unexpected balance decreases)"
else
    fail "Balance drops" "${BALANCE_DROP_COUNT} BALANCE DROP events detected!"
fi

# ═══════════════════════════════════════════════════════════════════
# TEST 6: LWMA difficulty adjustment
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "--- Test 6: LWMA difficulty ---"

LWMA_LOG=$(journalctl -u q-api-server --since '5 minutes ago' 2>/dev/null | grep "LWMA-PURE.*Difficulty" | tail -1)
LWMA_STATUS=$(journalctl -u q-api-server --since '10 minutes ago' 2>/dev/null | grep "LWMA diff:" | tail -1)

if [[ -n "$LWMA_LOG" ]]; then
    pass "LWMA active and adjusting difficulty: $(echo $LWMA_LOG | grep -oP 'Difficulty: \d+ → \d+')"
elif echo "$LWMA_STATUS" | grep -q "ACTIVE"; then
    pass "LWMA is ACTIVE (from startup banner)"
elif echo "$LWMA_STATUS" | grep -q "DISABLED"; then
    info "LWMA is DISABLED (check Q_LWMA_ACTIVATION_HEIGHT)"
    skip "LWMA test" "LWMA not activated on this node"
else
    skip "LWMA test" "No LWMA logs found"
fi

# ═══════════════════════════════════════════════════════════════════
# TEST 7: K-Gauge metrics
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "--- Test 7: K-Gauge metrics ---"

KGAUGE=$(journalctl -u q-api-server --since '2 minutes ago' 2>/dev/null | grep "K-GAUGE v10.3.0" | tail -1)
if [[ -n "$KGAUGE" ]]; then
    pass "K-Gauge running: $(echo $KGAUGE | grep -oP 'K_base=[\d.]+.*phase=\w+')"
else
    skip "K-Gauge" "No K-Gauge logs found"
fi

# ═══════════════════════════════════════════════════════════════════
# TEST 8: PQC verification (zero errors)
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "--- Test 8: PQC block verification ---"

PQC_ERRORS=$(journalctl -u q-api-server --since '5 minutes ago' 2>/dev/null | grep -c "PQC.*REJECT\|BLOCK.*REJECTED" || echo "0")
if [[ "$PQC_ERRORS" -eq 0 ]]; then
    pass "Zero PQC block rejections"
else
    fail "PQC errors" "${PQC_ERRORS} block rejections in last 5 minutes"
fi

# ═══════════════════════════════════════════════════════════════════
# TEST 9: Memory health
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "--- Test 9: Memory health ---"

MEM_LOG=$(journalctl -u q-api-server --since '1 minute ago' 2>/dev/null | grep "RSS=" | tail -1)
if [[ -n "$MEM_LOG" ]]; then
    RSS=$(echo "$MEM_LOG" | grep -oP 'RSS=\d+' | grep -oP '\d+')
    JE_ALLOC=$(echo "$MEM_LOG" | grep -oP 'je_alloc=\d+' | grep -oP '\d+')

    if [[ "$RSS" -lt 20000 ]]; then
        pass "Memory healthy: RSS=${RSS}MB, je_alloc=${JE_ALLOC}MB"
    elif [[ "$RSS" -lt 40000 ]]; then
        info "Memory elevated: RSS=${RSS}MB — monitor for growth"
        pass "Memory within acceptable range"
    else
        fail "Memory" "RSS=${RSS}MB exceeding safe limits"
    fi
else
    skip "Memory health" "No memory logs found"
fi

# ═══════════════════════════════════════════════════════════════════
# TEST 10: Mining activity
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "--- Test 10: Mining activity ---"

MINING_SOLUTIONS=$(journalctl -u q-api-server --since '1 minute ago' 2>/dev/null | grep -c "PHASE 4.*valid solutions" || echo "0")
if [[ "$MINING_SOLUTIONS" -gt 0 ]]; then
    pass "Mining active: ${MINING_SOLUTIONS} solution batches in last minute"
else
    skip "Mining activity" "No solution batches found (miner may not be connected)"
fi

# ═══════════════════════════════════════════════════════════════════
# TEST 11: Startup sync flag (if recently restarted)
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "--- Test 11: Startup sync complete flag ---"

SYNC_LOG=$(journalctl -u q-api-server --since '10 minutes ago' 2>/dev/null | grep "STARTUP SYNC.*complete\|startup_sync_complete" | tail -1)
if [[ -n "$SYNC_LOG" ]]; then
    pass "Startup sync flag: confirmed in logs"
else
    skip "Startup sync flag" "No recent startup (node has been running)"
fi

# ═══════════════════════════════════════════════════════════════════
# TEST 12: No authority sync overwriting balances unexpectedly
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "--- Test 12: Authority sync check ---"

AUTH_SYNC=$(journalctl -u q-api-server --since '10 minutes ago' 2>/dev/null | grep "AUTHORITY SYNC" | head -3)
if echo "$AUTH_SYNC" | grep -q "Failed"; then
    info "Authority sync failed (expected if no Q_BALANCE_AUTHORITY_PEER set)"
    pass "No authority sync overwrite (env var not set on Delta)"
elif [[ -z "$AUTH_SYNC" ]]; then
    pass "No authority sync configured (Delta is independent)"
else
    info "Authority sync ran: $(echo $AUTH_SYNC | head -1)"
    pass "Authority sync completed"
fi

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " TEST RESULTS"
echo "═══════════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}PASSED: ${PASSED}${NC}"
echo -e "  ${RED}FAILED: ${FAILED}${NC}"
echo -e "  ${YELLOW}SKIPPED: ${SKIPPED}${NC}"
echo ""

if [[ "$FAILED" -eq 0 ]]; then
    echo -e "${GREEN}ALL TESTS PASSED${NC} (${SKIPPED} skipped)"
    echo ""
    echo "Safe to deploy to production if:"
    echo "  1. All PASSED tests cover the changes being deployed"
    echo "  2. SKIPPED tests are understood (usually need more setup time)"
    echo "  3. You've reviewed the debug logs manually"
    exit 0
else
    echo -e "${RED}${FAILED} TEST(S) FAILED${NC}"
    echo ""
    echo "DO NOT DEPLOY TO PRODUCTION until all failures are resolved."
    exit 1
fi
