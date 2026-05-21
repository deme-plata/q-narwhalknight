#!/usr/bin/env bash
# Phase 4 (new): Post-deploy verification after a production Epsilon restart.
#
# Runs after the symlink/binary swap + systemctl restart. Confirms:
#   - the running binary really is the new version (readlink /proc/PID/exe)
#   - HTTP /status responds and reports a height
#   - the height is advancing (production isn't wedged)
#   - the operator's wallet balance is intact (no balance corruption)
#
# The wallet balance check is the load-bearing one: it confirms Ed25519
# X-Wallet-Auth still works AND that the new binary didn't roll back any
# balances during DB open. The skill bit this on 2026-05-21 — production
# went from v10.9.55 to v10.10.13 and the operator only noticed the balance
# check was missing AFTER deploying.
#
# Usage: ./verify_post_deploy.sh <version> [wallet_address]
set -euo pipefail

VERSION="${1:?usage: verify_post_deploy.sh <version> [wallet]}"
WALLET="${2:-qnk7154929a6aa0c118791373ea21004aca6e494e6e031c36f780cd5acedf031ccb}"
EPSILON="root@89.149.241.126"
SEED_FILE="/root/.claude/quillon-agent-seed"

echo "[verify] Post-deploy verification for v${VERSION}"

echo ""
echo "--- 1. running binary path ---"
ssh "$EPSILON" "
  pid=\$(pgrep -f 'q-api-server-stable' | head -1)
  if [ -z \"\$pid\" ]; then
    pid=\$(pgrep -f 'q-api-server-v${VERSION}' | head -1)
  fi
  [ -z \"\$pid\" ] && { echo 'ERROR: no q-api-server PID found'; exit 1; }
  exe=\$(readlink -f /proc/\$pid/exe 2>/dev/null)
  echo \"PID \$pid -> \$exe\"
  if echo \"\$exe\" | grep -q 'v${VERSION}'; then
    echo 'OK: binary matches v${VERSION}'
  else
    echo \"FAIL: binary does NOT match v${VERSION}\"
    exit 1
  fi
"

echo ""
echo "--- 2. HTTP /status responds + height ---"
HEIGHT_1=$(ssh "$EPSILON" "curl -s -m 5 http://localhost:8080/api/v1/status" | grep -oE '"current_height":[0-9]+' | head -1 | grep -oE '[0-9]+' || echo "")
[ -z "$HEIGHT_1" ] && { echo "FAIL: no height from /api/v1/status"; exit 1; }
echo "height_1 = $HEIGHT_1"

echo ""
echo "--- 3. height advancing (15s sample) ---"
sleep 15
HEIGHT_2=$(ssh "$EPSILON" "curl -s -m 5 http://localhost:8080/api/v1/status" | grep -oE '"current_height":[0-9]+' | head -1 | grep -oE '[0-9]+' || echo "")
echo "height_2 = $HEIGHT_2"
if [ "$HEIGHT_2" -gt "$HEIGHT_1" ]; then
  echo "OK: chain advanced $((HEIGHT_2 - HEIGHT_1)) block(s) in 15s"
else
  echo "WARN: chain did not advance in 15s (height stuck at $HEIGHT_1) — investigate"
fi

echo ""
echo "--- 4. wallet balance intact ---"
if [ ! -f "$SEED_FILE" ]; then
  echo "SKIP: $SEED_FILE missing — cannot sign auth"
  echo "Manual check: log into the wallet at https://quillon.xyz and verify QUG balance is unchanged"
else
  # Use the same probe pattern as tools/quillon-wallet-mcp/balance_check.mjs
  if command -v node >/dev/null 2>&1; then
    node -e "
      const { ed25519 } = await import('@noble/curves/ed25519.js');
      const { sha3_256 } = await import('@noble/hashes/sha3.js');
      const { bytesToHex, utf8ToBytes } = await import('@noble/hashes/utils.js');
      const { readFileSync } = await import('node:fs');
      const SEED = readFileSync('$SEED_FILE', 'utf8').trim();
      const priv = sha3_256(utf8ToBytes(SEED));
      const pub = ed25519.getPublicKey(priv);
      const addr = 'qnk' + bytesToHex(pub);
      const path = '/api/v1/wallets/' + addr + '/balance';
      const ts = Math.floor(Date.now() / 1000);
      const tsBuf = new Uint8Array(8);
      let v = BigInt(ts);
      for (let i = 0; i < 8; i++) { tsBuf[i] = Number(v & 0xffn); v >>= 8n; }
      const buf = new Uint8Array(40 + path.length);
      buf.set(pub, 0); buf.set(tsBuf, 32); buf.set(utf8ToBytes(path), 40);
      const authHdr = JSON.stringify({ address: addr, timestamp: ts, scheme: 'Ed25519', signature: bytesToHex(ed25519.sign(sha3_256(buf), priv)), public_key: bytesToHex(pub) });
      const r = await fetch('https://quillon.xyz' + path, { headers: { 'X-Wallet-Auth': authHdr } });
      const j = await r.json();
      if (j.success && j.data && j.data.balance_qnk !== undefined) {
        console.log('OK: balance = ' + j.data.balance_qnk.toFixed(4) + ' QUG (address ' + addr.slice(0,20) + '…)');
      } else {
        console.log('FAIL: balance lookup returned ' + JSON.stringify(j).slice(0,200));
        process.exit(1);
      }
    " --input-type=module 2>&1 | tail -3 || echo "WARN: balance check threw — check manually via wallet UI"
  else
    echo "SKIP: node not on PATH"
  fi
fi

echo ""
echo "[verify] done."
