#!/bin/bash
# Balance Database Diagnostic Tool
# Check what balances are actually stored in RocksDB

echo "=== Q-NarwhalKnight Balance Diagnostic ==="
echo "Database: ./data-mine1"
echo ""

# Query balances via API
echo "=== Current Balances (via API) ==="
curl -s "http://localhost:8080/api/v1/wallet/list" 2>/dev/null | jq -r '.wallets[]? | "\(.label // "Unknown"): QUG=\(.balance // 0)"' || echo "API not responding"

echo ""
echo "=== Token Balances (via API) ==="
curl -s "http://localhost:8080/api/v1/tokens" 2>/dev/null | jq -r '.data[]? | select(.symbol == "QUGUSD" or .symbol == "QUG") | "\(.symbol): supply=\(.total_supply // 0)"' || echo "API not responding"

echo ""
echo "=== Checking for wallet addresses in logs ==="
grep -oE "qnk[a-f0-9]{64}" /tmp/api-server-restart.log | sort -u | head -10

echo ""
echo "=== Database Storage Analysis ==="
echo "Database files:"
ls -lh ./data-mine1/hot/*.sst 2>/dev/null | tail -5
echo ""
echo "Latest database log (last 20 lines):"
tail -20 ./data-mine1/hot/LOG 2>/dev/null || echo "No database log found"
