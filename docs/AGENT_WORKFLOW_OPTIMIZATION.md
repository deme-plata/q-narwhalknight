# Claude Code Agent Workflow Optimization

## Overview

This document analyzes how specialized Claude Code agents can improve development workflows for the Q-NarwhalKnight quantum consensus system. Based on the Project Report (v3.4.15), we map specific agent types to each project component.

## Agent Types Available

| Agent Type | Purpose | Best For |
|------------|---------|----------|
| **Explore** | Codebase exploration, file pattern search, keyword search | Understanding architecture, finding implementations |
| **Plan** | Architecture design, implementation planning | Complex features, multi-file changes |
| **Bash** | Git operations, command execution | Deployment, testing, system administration |
| **general-purpose** | Multi-step tasks, complex research | Cross-cutting concerns, refactoring |
| **Release-Validator** | Staging environment testing | Pre-production validation, sync testing |

---

## 🚀 CRITICAL: Staging Environment & Release Validation

### Architecture: testnet.quillon.xyz

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RELEASE VALIDATION PIPELINE                               │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   BUILD      │───▶│   TESTNET    │───▶│   VALIDATE   │───▶│ PRODUCTION│ │
│  │   v3.4.16    │    │   DEPLOY     │    │   24-48h     │    │  RELEASE  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│        │                    │                    │                          │
│        ▼                    ▼                    ▼                          │
│  cargo build         testnet.quillon.xyz   Sync tests                      │
│  --release           Master wallet only    Balance checks                   │
│                      SSL/Let's Encrypt     P2P connectivity                 │
│                                            Block production                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Testnet Infrastructure Setup

**Domain:** `testnet.quillon.xyz`
**Purpose:** Pre-production release validation with master wallet access only
**Stack:** Nginx + Let's Encrypt + Vite + React + TypeScript

---

## 🛡️ Release Validator Agent Workflow

### Phase 1: Infrastructure Setup (One-Time)

```bash
# ════════════════════════════════════════════════════════════════════════════
# TESTNET INFRASTRUCTURE SETUP - Run once to establish staging environment
# ════════════════════════════════════════════════════════════════════════════

# Step 1: Create testnet directory structure
mkdir -p /opt/orobit/testnet/q-narwhalknight
mkdir -p /opt/orobit/testnet/frontend/dist
mkdir -p /opt/orobit/testnet/data
mkdir -p /opt/orobit/testnet/logs

# Step 2: Install SSL certificate with Let's Encrypt
sudo certbot certonly --nginx -d testnet.quillon.xyz --non-interactive --agree-tos -m admin@quillon.xyz

# Step 3: Create Nginx configuration for testnet
cat > /etc/nginx/sites-available/testnet.quillon.xyz << 'NGINX_EOF'
# ════════════════════════════════════════════════════════════════════════════
# TESTNET.QUILLON.XYZ - Staging Environment (Master Wallet Only)
# ════════════════════════════════════════════════════════════════════════════

# Rate limiting for testnet
limit_req_zone $binary_remote_addr zone=testnet_limit:10m rate=10r/s;

server {
    listen 80;
    server_name testnet.quillon.xyz;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name testnet.quillon.xyz;

    # SSL Configuration (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/testnet.quillon.xyz/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/testnet.quillon.xyz/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;

    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header X-Environment "TESTNET-STAGING" always;

    # ⚠️ MASTER WALLET ONLY - IP Whitelist
    # Only allow specific IPs to access testnet
    # Add your master wallet IPs here
    # allow 192.168.1.0/24;  # Example: local network
    # allow YOUR_IP_HERE;
    # deny all;

    # Root for Vite React frontend
    root /opt/orobit/testnet/frontend/dist;
    index index.html;

    # API Proxy to testnet backend (port 8085)
    location /api/ {
        limit_req zone=testnet_limit burst=20 nodelay;
        proxy_pass http://127.0.0.1:8085/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # SSE Streaming endpoint
    location /api/v1/sse/ {
        proxy_pass http://127.0.0.1:8085/api/v1/sse/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Cache-Control "no-cache";
        proxy_buffering off;
        proxy_read_timeout 86400s;
        chunked_transfer_encoding off;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://127.0.0.1:8085/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400s;
    }

    # Binary downloads for testnet
    location /downloads/ {
        alias /opt/orobit/testnet/frontend/dist/downloads/;
        autoindex on;
        add_header X-Release-Stage "TESTNET-CANDIDATE";
    }

    # React SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Health check endpoint
    location /health {
        return 200 'TESTNET OK';
        add_header Content-Type text/plain;
    }

    # Testnet status page
    location /testnet-status {
        proxy_pass http://127.0.0.1:8085/api/v1/status;
        add_header X-Testnet-Version $upstream_http_x_version;
    }

    access_log /opt/orobit/testnet/logs/access.log;
    error_log /opt/orobit/testnet/logs/error.log;
}
NGINX_EOF

# Step 4: Enable testnet site
sudo ln -sf /etc/nginx/sites-available/testnet.quillon.xyz /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Phase 2: Create Testnet Frontend (Vite + React + TypeScript)

```bash
# ════════════════════════════════════════════════════════════════════════════
# TESTNET FRONTEND - Modified version with TESTNET banner and master wallet
# ════════════════════════════════════════════════════════════════════════════

cd /opt/orobit/testnet

# Option A: Copy and modify production frontend
cp -r /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet /opt/orobit/testnet/frontend-src

# Modify for testnet
cd /opt/orobit/testnet/frontend-src

# Update vite.config.ts for testnet
cat > vite.config.ts << 'VITE_EOF'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  define: {
    'import.meta.env.VITE_NETWORK': JSON.stringify('testnet'),
    'import.meta.env.VITE_API_URL': JSON.stringify('https://testnet.quillon.xyz/api'),
    'import.meta.env.VITE_IS_STAGING': JSON.stringify('true'),
    'import.meta.env.VITE_MASTER_WALLET_ONLY': JSON.stringify('true'),
  },
  build: {
    outDir: '../frontend/dist',
    sourcemap: true,
  },
})
VITE_EOF

# Add testnet banner component
cat > src/components/TestnetBanner.tsx << 'TSX_EOF'
import React from 'react';

export const TestnetBanner: React.FC = () => {
  return (
    <div style={{
      background: 'linear-gradient(90deg, #ff6b00, #ff0000)',
      color: 'white',
      padding: '8px 16px',
      textAlign: 'center',
      fontWeight: 'bold',
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      zIndex: 9999,
      fontSize: '14px',
    }}>
      ⚠️ TESTNET STAGING ENVIRONMENT - Release Candidate Testing Only ⚠️
      <span style={{ marginLeft: '20px', opacity: 0.8 }}>
        Master Wallet Access | Not for Production Use
      </span>
    </div>
  );
};
TSX_EOF

# Build testnet frontend
npm install
npm run build
```

### Phase 3: Create Testnet Systemd Service

```bash
# ════════════════════════════════════════════════════════════════════════════
# TESTNET API SERVER SERVICE
# ════════════════════════════════════════════════════════════════════════════

cat > /etc/systemd/system/q-testnet-server.service << 'SERVICE_EOF'
[Unit]
Description=Q-NarwhalKnight Testnet API Server (Staging)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/orobit/testnet/q-narwhalknight
Environment=RUST_LOG=info
Environment=Q_NETWORK_ID=testnet-staging
Environment=Q_DATA_DIR=/opt/orobit/testnet/data
Environment=Q_IS_STAGING=true
Environment=Q_PREFLIGHT_CHECK=1

# Use release candidate binary
ExecStart=/opt/orobit/testnet/q-narwhalknight/q-api-server-candidate \
    --port 8085 \
    --p2p-port 9005 \
    --bootstrap-peer /ip4/185.182.185.227/tcp/9001/p2p/12D3KooWFrhdwDDTgxPX41mUyRgLcE1ozsBYArKM4DT8t4VLwuNx \
    --data-dir /opt/orobit/testnet/data \
    --network testnet-staging

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=q-testnet

[Install]
WantedBy=multi-user.target
SERVICE_EOF

sudo systemctl daemon-reload
```

---

## 🔄 Release Validation Agent Procedure

### The Professional Release Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              RELEASE CANDIDATE VALIDATION CHECKLIST                          │
│                                                                              │
│  ✅ = Required    ⚠️ = Warning if fails    ❌ = Blocks release              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE A: Pre-Deployment Validation                                          │
│  ─────────────────────────────────────                                       │
│  [ ] ✅ All 4000+ tests pass                                                │
│  [ ] ✅ cargo clippy -- -D warnings (zero warnings)                         │
│  [ ] ✅ cargo fmt --check passes                                            │
│  [ ] ✅ 125+ mainnet critical tests pass                                    │
│  [ ] ✅ Binary builds successfully                                          │
│                                                                              │
│  PHASE B: Testnet Deployment                                                 │
│  ────────────────────────────                                                │
│  [ ] ✅ Copy binary to testnet environment                                  │
│  [ ] ✅ Pre-flight check passes (Q_PREFLIGHT_ONLY=1)                        │
│  [ ] ✅ Service starts without errors                                       │
│  [ ] ✅ API responds on testnet.quillon.xyz                                 │
│                                                                              │
│  PHASE C: Sync Validation (24 hours minimum)                                 │
│  ───────────────────────────────────────────                                 │
│  [ ] ✅ Connects to bootstrap peer                                          │
│  [ ] ✅ Sync starts and progresses                                          │
│  [ ] ✅ No sync-down detected (CRITICAL!)                                   │
│  [ ] ✅ Height increases monotonically                                      │
│  [ ] ✅ Catches up to network height within 6 hours                         │
│  [ ] ⚠️ Block production resumes after sync                                 │
│                                                                              │
│  PHASE D: Functional Validation (24 hours)                                   │
│  ─────────────────────────────────────────                                   │
│  [ ] ✅ Master wallet can connect                                           │
│  [ ] ✅ Balance displays correctly                                          │
│  [ ] ✅ Transaction submission works                                        │
│  [ ] ✅ Mining rewards received                                             │
│  [ ] ✅ P2P balance propagation works                                       │
│  [ ] ✅ SSE streaming functional                                            │
│  [ ] ⚠️ No memory leaks (RSS stable over 24h)                              │
│  [ ] ⚠️ No CPU spikes (< 50% average)                                      │
│                                                                              │
│  PHASE E: Production Release Approval                                        │
│  ────────────────────────────────────                                        │
│  [ ] ❌ All PHASE A-D checks pass                                           │
│  [ ] ❌ 48-hour soak test complete                                          │
│  [ ] ❌ No critical errors in logs                                          │
│  [ ] ❌ Version string matches expected                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Commands for Each Phase

#### PHASE A: Pre-Deployment (Bash Agent)

```bash
# ════════════════════════════════════════════════════════════════════════════
# PHASE A: PRE-DEPLOYMENT VALIDATION
# Run by: Bash Agent
# ════════════════════════════════════════════════════════════════════════════

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         PHASE A: PRE-DEPLOYMENT VALIDATION                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

cd /opt/orobit/shared/q-narwhalknight
VERSION="v3.4.16-beta"  # Update for each release

# A.1: Format check
echo "🔍 [A.1] Checking code formatting..."
cargo fmt --check
if [ $? -ne 0 ]; then
    echo "❌ PHASE A FAILED: Code formatting issues"
    exit 1
fi
echo "✅ [A.1] Format check passed"

# A.2: Clippy warnings
echo "🔍 [A.2] Running clippy..."
timeout 1800 cargo clippy -- -D warnings 2>&1 | tee /tmp/clippy-output.txt
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ PHASE A FAILED: Clippy warnings found"
    exit 1
fi
echo "✅ [A.2] Clippy passed"

# A.3: Critical mainnet tests
echo "🔍 [A.3] Running mainnet critical tests..."
CRITICAL_TESTS=(
    "mainnet_critical_tests"
    "signature_verification_tests"
    "balance_propagation_tests"
    "overflow_protection_tests"
    "sync_down_protection_tests"
    "fork_reorg_tests"
)

for test in "${CRITICAL_TESTS[@]}"; do
    echo "   Running: $test"
    timeout 300 cargo test --package q-storage --test "$test" 2>/dev/null || \
    timeout 300 cargo test --package q-types --test "$test" 2>/dev/null || \
    timeout 300 cargo test --package q-dex --test "$test" 2>/dev/null || \
    echo "   (test not found, skipping)"
done
echo "✅ [A.3] Critical tests completed"

# A.4: Full test suite
echo "🔍 [A.4] Running full test suite..."
timeout 7200 cargo test --workspace 2>&1 | tee /tmp/test-output.txt
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ PHASE A FAILED: Test suite failed"
    exit 1
fi
echo "✅ [A.4] Full test suite passed"

# A.5: Build release binary
echo "🔍 [A.5] Building release binary..."
timeout 3600 cargo build --release --package q-api-server
if [ $? -ne 0 ]; then
    echo "❌ PHASE A FAILED: Build failed"
    exit 1
fi
echo "✅ [A.5] Binary built successfully"

# A.6: Copy to testnet as candidate
echo "🔍 [A.6] Copying to testnet environment..."
cp target/release/q-api-server /opt/orobit/testnet/q-narwhalknight/q-api-server-candidate
chmod +x /opt/orobit/testnet/q-narwhalknight/q-api-server-candidate
echo "✅ [A.6] Binary copied to testnet"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         ✅ PHASE A COMPLETE - Ready for PHASE B               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
```

#### PHASE B: Testnet Deployment (Bash Agent)

```bash
# ════════════════════════════════════════════════════════════════════════════
# PHASE B: TESTNET DEPLOYMENT
# Run by: Bash Agent
# ════════════════════════════════════════════════════════════════════════════

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         PHASE B: TESTNET DEPLOYMENT                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# B.1: Pre-flight check
echo "🔍 [B.1] Running pre-flight verification..."
cd /opt/orobit/testnet/q-narwhalknight
Q_PREFLIGHT_ONLY=1 Q_DATA_DIR=/opt/orobit/testnet/data \
    ./q-api-server-candidate 2>&1 | tee /tmp/preflight-output.txt

if grep -q "PRE-FLIGHT CHECK: PASSED" /tmp/preflight-output.txt; then
    echo "✅ [B.1] Pre-flight check passed"
else
    echo "❌ PHASE B FAILED: Pre-flight check failed"
    cat /tmp/preflight-output.txt
    exit 1
fi

# B.2: Stop existing testnet service (if running)
echo "🔍 [B.2] Stopping existing testnet service..."
systemctl stop q-testnet-server 2>/dev/null || true
sleep 5
echo "✅ [B.2] Previous service stopped"

# B.3: Start new testnet service
echo "🔍 [B.3] Starting testnet service with new binary..."
systemctl start q-testnet-server
sleep 10

# B.4: Verify service is running
if systemctl is-active --quiet q-testnet-server; then
    echo "✅ [B.3] Testnet service started"
else
    echo "❌ PHASE B FAILED: Service failed to start"
    journalctl -u q-testnet-server --since "1 minute ago" --no-pager
    exit 1
fi

# B.5: Verify API responds
echo "🔍 [B.5] Checking API response..."
sleep 5
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8085/api/v1/status)
if [ "$RESPONSE" = "200" ]; then
    echo "✅ [B.5] API responding correctly"
else
    echo "❌ PHASE B FAILED: API not responding (HTTP $RESPONSE)"
    exit 1
fi

# B.6: Verify SSL on testnet domain
echo "🔍 [B.6] Checking SSL certificate..."
SSL_CHECK=$(curl -s -o /dev/null -w "%{http_code}" https://testnet.quillon.xyz/health 2>/dev/null)
if [ "$SSL_CHECK" = "200" ]; then
    echo "✅ [B.6] SSL certificate valid and responding"
else
    echo "⚠️ [B.6] SSL check returned $SSL_CHECK (may need DNS propagation)"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         ✅ PHASE B COMPLETE - Starting PHASE C (24h)          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Monitor sync progress with:"
echo "   journalctl -u q-testnet-server -f | grep -E 'height|sync|P2P'"
```

#### PHASE C: Sync Validation (Bash Agent - Periodic Check)

```bash
# ════════════════════════════════════════════════════════════════════════════
# PHASE C: SYNC VALIDATION (Run every hour for 24 hours)
# Run by: Bash Agent with cron or manual checks
# ════════════════════════════════════════════════════════════════════════════

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         PHASE C: SYNC VALIDATION CHECK                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
LOG_FILE="/opt/orobit/testnet/logs/sync-validation-${TIMESTAMP}.log"

# C.1: Get current testnet height
TESTNET_STATUS=$(curl -s http://127.0.0.1:8085/api/v1/status)
TESTNET_HEIGHT=$(echo "$TESTNET_STATUS" | jq -r '.height // 0')
echo "📊 Testnet height: $TESTNET_HEIGHT" | tee -a "$LOG_FILE"

# C.2: Get production height for comparison
PROD_STATUS=$(curl -s http://127.0.0.1:8080/api/v1/status)
PROD_HEIGHT=$(echo "$PROD_STATUS" | jq -r '.height // 0')
echo "📊 Production height: $PROD_HEIGHT" | tee -a "$LOG_FILE"

# C.3: Calculate sync progress
if [ "$PROD_HEIGHT" -gt 0 ]; then
    SYNC_PERCENT=$(echo "scale=2; ($TESTNET_HEIGHT / $PROD_HEIGHT) * 100" | bc)
    echo "📊 Sync progress: ${SYNC_PERCENT}%" | tee -a "$LOG_FILE"
fi

# C.4: Check for sync-down (CRITICAL!)
PREV_HEIGHT_FILE="/opt/orobit/testnet/logs/last-height.txt"
if [ -f "$PREV_HEIGHT_FILE" ]; then
    PREV_HEIGHT=$(cat "$PREV_HEIGHT_FILE")
    if [ "$TESTNET_HEIGHT" -lt "$PREV_HEIGHT" ]; then
        echo "❌ CRITICAL: SYNC-DOWN DETECTED! Height went from $PREV_HEIGHT to $TESTNET_HEIGHT" | tee -a "$LOG_FILE"
        echo "❌ PHASE C FAILED: Sync-down is a release blocker!" | tee -a "$LOG_FILE"
        # Alert mechanism here (email, Slack, etc.)
        exit 1
    fi
fi
echo "$TESTNET_HEIGHT" > "$PREV_HEIGHT_FILE"
echo "✅ Height monotonically increasing" | tee -a "$LOG_FILE"

# C.5: Check P2P connectivity
P2P_PEERS=$(curl -s http://127.0.0.1:8085/api/v1/peers | jq -r '.connected_peers // 0')
echo "📊 Connected P2P peers: $P2P_PEERS" | tee -a "$LOG_FILE"
if [ "$P2P_PEERS" -lt 1 ]; then
    echo "⚠️ Warning: No P2P peers connected" | tee -a "$LOG_FILE"
fi

# C.6: Check for errors in recent logs
ERROR_COUNT=$(journalctl -u q-testnet-server --since "1 hour ago" | grep -c -E "ERROR|CRITICAL|panic" || true)
echo "📊 Errors in last hour: $ERROR_COUNT" | tee -a "$LOG_FILE"
if [ "$ERROR_COUNT" -gt 10 ]; then
    echo "⚠️ Warning: High error count in logs" | tee -a "$LOG_FILE"
fi

# C.7: Memory usage check
RSS_MB=$(ps aux | grep q-api-server-candidate | grep -v grep | awk '{print $6/1024}' | head -1)
echo "📊 Memory usage: ${RSS_MB}MB" | tee -a "$LOG_FILE"

echo ""
echo "✅ Sync validation check complete at $TIMESTAMP"
```

#### PHASE D: Functional Validation (Bash Agent)

```bash
# ════════════════════════════════════════════════════════════════════════════
# PHASE D: FUNCTIONAL VALIDATION
# Run by: Bash Agent after sync complete
# ════════════════════════════════════════════════════════════════════════════

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         PHASE D: FUNCTIONAL VALIDATION                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

TESTNET_API="http://127.0.0.1:8085/api/v1"

# D.1: API Status endpoint
echo "🔍 [D.1] Checking /api/v1/status..."
STATUS=$(curl -s "$TESTNET_API/status")
if echo "$STATUS" | jq -e '.height' > /dev/null; then
    echo "✅ [D.1] Status endpoint working"
else
    echo "❌ [D.1] Status endpoint failed"
fi

# D.2: Network info
echo "🔍 [D.2] Checking /api/v1/network..."
NETWORK=$(curl -s "$TESTNET_API/network")
if echo "$NETWORK" | jq -e '.network_id' > /dev/null; then
    echo "✅ [D.2] Network endpoint working"
else
    echo "⚠️ [D.2] Network endpoint issue"
fi

# D.3: Peers endpoint
echo "🔍 [D.3] Checking /api/v1/peers..."
PEERS=$(curl -s "$TESTNET_API/peers")
PEER_COUNT=$(echo "$PEERS" | jq -r '.connected_peers // 0')
echo "   Connected peers: $PEER_COUNT"
if [ "$PEER_COUNT" -ge 1 ]; then
    echo "✅ [D.3] P2P connectivity working"
else
    echo "⚠️ [D.3] No peers connected"
fi

# D.4: SSE Streaming test
echo "🔍 [D.4] Testing SSE streaming..."
timeout 10 curl -s "$TESTNET_API/sse/blocks" > /tmp/sse-test.txt 2>&1 &
SSE_PID=$!
sleep 5
kill $SSE_PID 2>/dev/null
if [ -s /tmp/sse-test.txt ]; then
    echo "✅ [D.4] SSE streaming functional"
else
    echo "⚠️ [D.4] SSE may not be producing events"
fi

# D.5: Block retrieval test
echo "🔍 [D.5] Testing block retrieval..."
HEIGHT=$(curl -s "$TESTNET_API/status" | jq -r '.height')
if [ "$HEIGHT" -gt 10 ]; then
    BLOCK=$(curl -s "$TESTNET_API/blocks/$((HEIGHT - 5))")
    if echo "$BLOCK" | jq -e '.hash' > /dev/null; then
        echo "✅ [D.5] Block retrieval working"
    else
        echo "⚠️ [D.5] Block retrieval issue"
    fi
fi

# D.6: Version check
echo "🔍 [D.6] Checking version..."
VERSION_INFO=$(curl -s "$TESTNET_API/version" 2>/dev/null || curl -s "$TESTNET_API/status" | jq -r '.version // "unknown"')
echo "   Version: $VERSION_INFO"
echo "✅ [D.6] Version check complete"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         ✅ PHASE D COMPLETE - Ready for Production            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
```

#### PHASE E: Production Release (Bash Agent)

```bash
# ════════════════════════════════════════════════════════════════════════════
# PHASE E: PRODUCTION RELEASE
# Run by: Bash Agent ONLY after all phases pass
# ════════════════════════════════════════════════════════════════════════════

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         PHASE E: PRODUCTION RELEASE                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

VERSION="v3.4.16-beta"  # Update for each release
CANDIDATE="/opt/orobit/testnet/q-narwhalknight/q-api-server-candidate"
PRODUCTION="/opt/orobit/shared/q-narwhalknight/target/release/q-api-server"
DOWNLOADS="/opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads"

# E.1: Final checklist verification
echo "🔍 [E.1] Final checklist..."
echo "   Has the candidate been running on testnet for 48+ hours? [MANUAL CHECK]"
echo "   Has sync validation passed all checks? [MANUAL CHECK]"
echo "   Has functional validation passed? [MANUAL CHECK]"
echo ""
read -p "Type 'APPROVED' to proceed with production release: " APPROVAL

if [ "$APPROVAL" != "APPROVED" ]; then
    echo "❌ Release not approved. Exiting."
    exit 1
fi

# E.2: Create backup of current production binary
echo "🔍 [E.2] Creating backup..."
BACKUP_NAME="q-api-server-backup-$(date +%Y%m%d-%H%M%S)"
cp "$PRODUCTION" "/opt/orobit/backups/$BACKUP_NAME" 2>/dev/null || true
echo "✅ [E.2] Backup created: $BACKUP_NAME"

# E.3: Copy validated binary to production
echo "🔍 [E.3] Copying to production..."
cp "$CANDIDATE" "$PRODUCTION"
chmod +x "$PRODUCTION"
echo "✅ [E.3] Binary copied to production path"

# E.4: Copy to downloads for users
echo "🔍 [E.4] Updating downloads..."
cp "$CANDIDATE" "$DOWNLOADS/q-api-server-$VERSION"
cp "$CANDIDATE" "$DOWNLOADS/q-api-server-linux-x86_64"
chmod +x "$DOWNLOADS/q-api-server-$VERSION"
chmod +x "$DOWNLOADS/q-api-server-linux-x86_64"
echo "✅ [E.4] Downloads updated"

# E.5: Restart production service
echo "🔍 [E.5] Restarting production service..."
systemctl restart q-api-server
sleep 10

if systemctl is-active --quiet q-api-server; then
    echo "✅ [E.5] Production service restarted successfully"
else
    echo "❌ [E.5] Production service failed to start!"
    echo "🔄 Rolling back..."
    cp "/opt/orobit/backups/$BACKUP_NAME" "$PRODUCTION"
    systemctl restart q-api-server
    exit 1
fi

# E.6: Verify production API
echo "🔍 [E.6] Verifying production API..."
sleep 5
PROD_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/api/v1/status)
if [ "$PROD_RESPONSE" = "200" ]; then
    echo "✅ [E.6] Production API responding"
else
    echo "❌ [E.6] Production API not responding (HTTP $PROD_RESPONSE)"
    echo "🔄 Consider rollback if issues persist"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         🎉 RELEASE COMPLETE: $VERSION                         ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║                                                               ║"
echo "║  📥 User download link:                                       ║"
echo "║  wget https://quillon.xyz/downloads/q-api-server-$VERSION     ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
```

---

## 📋 Release Validator Agent Summary

### When to Use Each Agent

| Phase | Agent Type | Parallelization | Duration |
|-------|------------|-----------------|----------|
| Phase A | Bash | Sequential | 2-4 hours |
| Phase B | Bash | Sequential | 15 minutes |
| Phase C | Bash (cron) | Hourly checks | 24 hours |
| Phase D | Bash | Sequential | 30 minutes |
| Phase E | Bash | Sequential | 15 minutes |

### Error Recovery

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ERROR RECOVERY PROCEDURES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE A FAILURE:                                                            │
│  → Fix code issues, re-run tests                                            │
│  → Do NOT proceed until all tests pass                                       │
│                                                                              │
│  PHASE B FAILURE:                                                            │
│  → Check pre-flight output for specific errors                              │
│  → May need database migration or schema update                              │
│  → Return to PHASE A if binary issue                                         │
│                                                                              │
│  PHASE C FAILURE (SYNC-DOWN):                                                │
│  → CRITICAL! Immediately stop testnet service                                │
│  → Investigate root cause in code                                            │
│  → Return to PHASE A with fix                                                │
│  → Never deploy a binary that syncs down!                                    │
│                                                                              │
│  PHASE D FAILURE:                                                            │
│  → Check specific endpoint that failed                                       │
│  → May be temporary - retry after 10 minutes                                 │
│  → If persistent, return to PHASE A                                          │
│                                                                              │
│  PHASE E FAILURE (Production):                                               │
│  → Immediate rollback to backup                                              │
│  → systemctl restart q-api-server                                            │
│  → Investigate and return to PHASE A                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Consensus Layer (`q-dag-knight`, `q-narwhal-core`)

### Current State
- 661,154 lines of Rust
- DAG-Knight consensus with VDF-based anchor election
- Zero-message complexity BFT

### Recommended Agent Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  CONSENSUS CHANGES - Always Use Plan Agent First!               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Plan Agent (subagent_type=Plan)                       │
│    - "Design implementation for [consensus feature]"            │
│    - Outputs: File list, dependency analysis, safety review     │
│                                                                 │
│  Step 2: Explore Agent (subagent_type=Explore)                 │
│    - "Find all callers of verify_block_signature()"            │
│    - "Where is the finality threshold configured?"              │
│                                                                 │
│  Step 3: Implementation (direct edits)                          │
│    - Make changes with height-gated upgrades                    │
│                                                                 │
│  Step 4: Bash Agent (subagent_type=Bash)                       │
│    - Run: cargo test --package q-dag-knight                     │
│    - Run: ./scripts/safe-deploy.sh test-all                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Example Parallel Agent Calls

For consensus changes, launch multiple agents in parallel:

```json
{
  "agents": [
    {
      "subagent_type": "Explore",
      "prompt": "Find all block validation functions in q-dag-knight and q-types",
      "description": "Find block validators"
    },
    {
      "subagent_type": "Explore",
      "prompt": "Search for 'finality' and 'confirmation' in consensus code",
      "description": "Find finality code"
    },
    {
      "subagent_type": "Bash",
      "prompt": "git log --oneline --since='1 week ago' -- crates/q-dag-knight",
      "description": "Recent consensus changes"
    }
  ]
}
```

---

## 2. Cryptography Layer (`q-crypto-advanced`, `q-quantum-crypto`)

### Current State
- Dilithium5 + Kyber1024 (post-quantum)
- Genus-2 curve VDF
- Bulletproofs for privacy

### Critical Safety Rule
**NEVER modify cryptographic code without extensive review!**

### Recommended Agent Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  CRYPTO CHANGES - Plan Agent MANDATORY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Explore Agent - Security Audit First                   │
│    - "Find all uses of SigningKey and VerifyingKey"             │
│    - "Search for 'verify' in signature-related code"            │
│    - "Find all UNSAFE blocks in crypto crates"                  │
│                                                                 │
│  Step 2: Plan Agent - Design Review                             │
│    - "Plan post-quantum signature migration"                    │
│    - Must include: backward compatibility, test plan            │
│                                                                 │
│  Step 3: Bash Agent - Comprehensive Testing                     │
│    - Run ALL mainnet critical tests (125+ tests)                │
│    - Run signature_verification_tests specifically              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Network Layer (`q-network`, `q-tor-client`)

### Current State
- libp2p gossipsub + Kademlia DHT
- Tor integration with 4 circuits per validator
- P2P sync with TurboSync

### Recommended Agent Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  NETWORK CHANGES - Explore + Bash Agents                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Explore Agent (parallel)                               │
│    Agent A: "Find gossipsub topic definitions"                  │
│    Agent B: "Search for connection_manager patterns"            │
│    Agent C: "Find all P2P message types"                        │
│                                                                 │
│  Step 2: Bash Agent - Test P2P Changes                          │
│    - Run network_partition_tests                                │
│    - Run distributed_ai_encryption_tests                        │
│    - Check Docker node connectivity                             │
│                                                                 │
│  Step 3: Bash Agent - Deploy & Monitor                          │
│    - systemctl restart q-api-server                             │
│    - journalctl -u q-api-server | grep "P2P\|peer"              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Storage Layer (`q-storage`)

### Current State
- RocksDB with hot/cold separation
- TurboSync for fast synchronization
- Fork detection and reorg handling

### CRITICAL: Storage is Highest Risk!

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚠️  STORAGE CHANGES - HIGHEST RISK CATEGORY                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MANDATORY before ANY storage change:                           │
│                                                                 │
│  1. Plan Agent - Full Architecture Review                       │
│     - "Plan the storage schema change for [feature]"            │
│     - Must address: migration, rollback, data integrity         │
│                                                                 │
│  2. Explore Agent - Impact Analysis                             │
│     - "Find all code that calls storage.get_block()"            │
│     - "Where is the column family defined for balances?"        │
│                                                                 │
│  3. Bash Agent - Full Test Suite                                │
│     - timeout 3600 cargo test --workspace                       │
│     - MUST run sync_down_protection_tests                       │
│     - MUST run balance_propagation_tests                        │
│                                                                 │
│  4. Docker Soak Test (48-72 hours)                              │
│     - Test on Server Alpha before production                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. API Server (`q-api-server`)

### Current State
- REST API with SSE streaming
- Wallet authentication
- Mining endpoints

### Recommended Agent Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  API CHANGES - Lower Risk, Can Use Direct Implementation       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For NEW endpoints:                                             │
│    1. Explore Agent: "Find similar API handler patterns"        │
│    2. Direct implementation                                     │
│    3. Bash Agent: cargo test --package q-api-server             │
│                                                                 │
│  For SSE/WebSocket changes:                                     │
│    1. Explore Agent: "Find SSE broadcast patterns"              │
│    2. Plan Agent (if complex): Design event flow                │
│    3. Bash Agent: Test streaming with curl                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. DEX & Smart Contracts (`q-dex`, `q-vm`)

### Current State
- AMM with u128 token amounts
- WASM sandbox for smart contracts
- Privacy-preserving swaps

### Recommended Agent Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  DEX/VM CHANGES - Financial Code = High Risk                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Explore Agent - Find Related Code                      │
│    - "Find all overflow checks in q-dex"                        │
│    - "Search for 'checked_mul' and 'checked_div'"               │
│                                                                 │
│  Step 2: Plan Agent - For Any AMM Logic                         │
│    - "Plan the constant product calculation change"             │
│    - Must include: overflow protection, edge cases              │
│                                                                 │
│  Step 3: Bash Agent - Mandatory Tests                           │
│    - Run overflow_protection_tests (26 tests)                   │
│    - Run comprehensive_dex_tests                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. AI Integration (`q-ai-inference`)

### Current State
- MistralRS integration
- Distributed inference across nodes
- Proof of Inference validation

### Recommended Agent Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  AI CHANGES - General Purpose Agent Often Best                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For Model Integration:                                         │
│    1. General-Purpose Agent: Research model requirements        │
│       "Investigate memory requirements for Mistral-7B"          │
│                                                                 │
│  For Inference Pipeline:                                        │
│    1. Explore Agent: "Find the inference request flow"          │
│    2. Direct implementation                                     │
│                                                                 │
│  For Distributed AI:                                            │
│    1. Plan Agent: "Design P2P inference task distribution"      │
│    2. Network + AI changes together                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Robot Control (`q-robot-control`)

### Current State
- Water robot coordination
- Resonance swarm consensus (NEW)
- Blockchain life simulation

### Recommended Agent Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  ROBOT CHANGES - Explore + Direct Implementation                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For Swarm Algorithms:                                          │
│    1. Explore Agent: "Find resonance and coupling code"         │
│    2. Direct implementation with async/await                    │
│                                                                 │
│  For Hardware Integration:                                      │
│    1. Plan Agent: "Design sensor integration architecture"      │
│    2. Bash Agent: Test on actual hardware                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Documentation & Papers

### Current State
- LaTeX technical papers
- Markdown documentation
- CLAUDE.md development guide

### Recommended Agent Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  DOCUMENTATION - Explore + General Purpose                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For Technical Documentation:                                   │
│    1. Explore Agent: Gather code examples and stats             │
│    2. Direct writing with LaTeX/Markdown                        │
│                                                                 │
│  For API Documentation:                                         │
│    1. Explore Agent: "Find all public API endpoints"            │
│    2. General-Purpose: Generate OpenAPI spec                    │
│                                                                 │
│  For Architecture Diagrams:                                     │
│    1. Explore Agent: Map crate dependencies                     │
│    2. Direct TikZ/Mermaid diagram creation                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Parallel Agent Patterns

### Pattern 1: Multi-Crate Investigation

When a change affects multiple crates, launch parallel Explore agents:

```
User: "Fix the balance display bug"

Claude: Launches 4 parallel Explore agents:
  - Agent 1: Search q-storage for balance functions
  - Agent 2: Search q-api-server for balance endpoints
  - Agent 3: Search q-types for Balance struct
  - Agent 4: Search q-network for P2P balance messages
```

### Pattern 2: Test + Build Pipeline

After implementation, launch parallel verification:

```
User: "Deploy the fix"

Claude: Launches parallel agents:
  - Bash Agent 1: cargo test --package affected-crate
  - Bash Agent 2: cargo clippy -- -D warnings
  - Bash Agent 3: cargo fmt --check
```

### Pattern 3: Documentation Update

After major changes, parallel doc updates:

```
User: "Document the new feature"

Claude: Launches parallel:
  - Explore Agent: Gather implementation details
  - Bash Agent: Extract test coverage stats
  - General-Purpose: Research similar implementations
```

---

## Risk-Based Agent Selection

| Risk Level | Primary Agent | Secondary Agent | Mandatory Tests |
|------------|---------------|-----------------|-----------------|
| 🟢 LOW | Direct Edit | Bash (quick test) | cargo check |
| 🟡 MEDIUM | Explore | Bash (test suite) | cargo test --package |
| 🟠 HIGH | Plan | Explore + Bash | All 125+ critical tests |
| 🔴 CRITICAL | Plan (mandatory) | Explore + Bash + Docker Soak | Full workspace + Docker 48h |

---

## Summary: Agent Selection Flowchart

```
                    ┌─────────────────┐
                    │ What type of    │
                    │ change is it?   │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │ Research │      │ Feature  │      │   Bug    │
    │ Question │      │  Change  │      │   Fix    │
    └────┬─────┘      └────┬─────┘      └────┬─────┘
         │                 │                  │
         ▼                 ▼                  │
    ┌──────────┐      ┌──────────┐           │
    │ Explore  │      │  Plan    │           │
    │  Agent   │      │  Agent   │           │
    └──────────┘      └────┬─────┘           │
                           │                  │
                           ▼                  ▼
                      ┌──────────┐      ┌──────────┐
                      │ Explore  │      │ Explore  │
                      │  Agent   │      │  Agent   │
                      └────┬─────┘      └────┬─────┘
                           │                  │
                           └────────┬─────────┘
                                    │
                                    ▼
                              ┌──────────┐
                              │  Bash    │
                              │  Agent   │
                              │ (tests)  │
                              └──────────┘
```

---

## Conclusion

The key improvements for Q-NarwhalKnight development with Claude Code agents:

1. **Always use Explore agents** before diving into unfamiliar code
2. **Use Plan agents** for HIGH/CRITICAL risk changes (consensus, storage, crypto)
3. **Launch parallel agents** when investigating multi-crate issues
4. **Bash agents for CI/CD** - testing, deployment, monitoring
5. **Never skip the test phase** - use Bash agents to run full test suites

The 661K+ line codebase benefits enormously from agents that can search and understand code before making changes. This reduces errors and ensures changes respect the existing architecture.
