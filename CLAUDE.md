# CLAUDE.md - Multi-Server Development Guide

## 🚨 BALANCE INTEGRITY — NON-NEGOTIABLE RULES

These rules exist because balance-corrupting code was written and destroyed correct wallet balances on Epsilon in May 2026. Do not repeat this.

### Rule 1: save_wallet_balances MUST be max-wins
Any code that writes wallet balances to RocksDB MUST check `existing >= new` first and skip the write if the existing value is higher. `save_wallet_balances` is a batch overwrite — it will silently destroy correct higher balances if called with stale/partial data. This is how a replay bug dropped a user wallet from 3200 QUG → 1484 QUG.

### Rule 2: Replay code MUST gate on is_checkpoint_applied()
Balance replay (`replay_post_checkpoint_balances`) exists ONLY for nodes that bootstrapped from the checkpoint snapshot. It MUST check `is_checkpoint_applied()` before running — if that returns false, the node ran from genesis and already has correct balances. Never run replay on Epsilon.

### Rule 3: Epsilon's wallet balances are authoritative — never overwrite them
Epsilon has been running since genesis. Its RocksDB wallet balances are the ground truth for the network. No code path should ever write a lower balance to any wallet on Epsilon. If a feature needs to modify balances on checkpoint nodes (Beta/Gamma), add an explicit `is_genesis_node()` guard that skips Epsilon.

### Rule 4: Test balance-modifying code on Alpha Docker ONLY
Any new code that touches `save_wallet_balance`, `save_wallet_balances`, or the replay path must be tested in a fresh Docker container on Alpha before it gets near Beta, Gamma, or Epsilon. Verify wallet counts and total supply match expected values before deploying.

---

## Claude Code Distributed Development for Q-NarwhalKnight

This guide explains how to set up distributed development with multiple Claude Code servers working collaboratively on the Q-NarwhalKnight quantum consensus system.

## 🌐 **NETWORK INFRASTRUCTURE**

### **Server Configuration:**

#### **Server Alpha (Testing/Development Node)**
- **IP Address**: `161.35.219.10`
- **Role**: Testing node for development builds, Docker container hosting
- **Environment**: Docker containers for isolated testing
- **Purpose**: Test new features before Server Beta deployment
- **Docker Test Container**: `q-v1098` (or similar) - used for sync testing
- **⚠️ IMPORTANT**: Do NOT SSH into Server Alpha from Server Beta
- **Protocol**: Ask the user for information from Server Alpha (logs, status, etc.)
- **Note**: When checking sync test status, the container runs on Server Alpha, not Beta
- **To get sync status**: Ask user to run `docker logs q-v1098 2>&1 | tail -50` on Server Alpha

#### **Server Beta (Production/Bootstrap Node)**
- **IP Address**: `185.182.185.227`
- **Role**: Production bootstrap node, network anchor
- **API Port**: `8080` (HTTP REST API)
- **P2P Port**: `9001` (libp2p gossipsub + Kademlia DHT)
- **Working Directory**: `/opt/orobit/shared/q-narwhalknight`
- **Service**: `systemd` service at `/etc/systemd/system/q-api-server.service`
- **Frontend**: Nginx serving from `gui/quantum-wallet/dist-final/`
- **Domain**: `quillon.xyz`

#### **Server Gamma (Backup/Failover Node)**
- **IP Address**: `109.205.176.60`
- **Role**: Backup node for HA failover, rolling upgrade target
- **API Port**: `8080` (HTTP REST API)
- **P2P Port**: `9001` (libp2p gossipsub + Kademlia DHT)
- **SSH**: `root@109.205.176.60` (SSH key auth from Beta, no password needed)
- **Service**: `systemd` service at `/etc/systemd/system/q-api-server.service`
- **Binary**: `/opt/orobit/shared/q-narwhalknight/q-api-server` (SCP'd from Beta)
- **Working Directory**: `/opt/orobit/shared/q-narwhalknight`
- **Peer ID**: `12D3KooWFqPX9TkvF43eyDeH9wwxYTSfnBn8AobLJeA7xRnmpPcv`
- **RAM**: 7.8GB + 4GB swap (swap required to prevent OOM during sync)
- **Note**: Has Claude Code installed for remote administration

#### **Server Epsilon (10Gbit SUPERNODE — Primary Sync Target)**
- **IP Address**: `89.149.241.126`
- **Role**: Primary bootstrap node, 10Gbit supernode, fastest sync source
- **API Port**: `8080` (HTTP REST API)
- **P2P Port**: `9001` (libp2p gossipsub + Kademlia DHT)
- **SSH**: `root@89.149.241.126` (SSH key auth from Beta)
- **Service**: `systemd` service at `/etc/systemd/system/q-api-server.service`
- **Binary**: `/opt/orobit/shared/q-narwhalknight/q-api-server-v889` (name in systemd service)
- **Working Directory**: `/opt/orobit/shared/q-narwhalknight`
- **Reverse Proxy**: q-flux (NOT nginx, NOT Caddy — nginx is DISABLED on Epsilon)
- **Static Files Root**: `/home/orobit/q-narwhalknight/dist-final/` (DIFFERENT from Beta!)
- **Peer ID**: `12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM` (changed 2026-05-09 restart; was `12D3KooWAbrVw892T8RSenWy1j89NBrd7p4aXKsSMKAYpH47YbgD`)
- **⚠️ CRITICAL: ALWAYS use /home paths on Epsilon, NEVER /tmp or /root!**
  - `/tmp` is on a tiny 40GB root partition (always near full)
  - `/home` is on a 1.8TB NVMe partition with ~800GB free
  - **Git clone destination**: `/home/orobit/q-narwhalknight/` (NOT /tmp/q-source)
  - **Build/temp files**: `/home/orobit/tmp/` (git tmpdir configured here)
  - **Frontend deploy**: `/home/orobit/q-narwhalknight/dist-final/`
  - **Binary deploy**: `/opt/orobit/shared/q-narwhalknight/q-api-server-v889`
- **NEVER**: `git clone ... /tmp/...` or write large files to `/tmp` or `/root`

- **🗄️ EPSILON DATABASE — CRITICAL FACTS (learned 2026-04-24 incident):**
  - **Authoritative DB path**: `/home/orobit/data-mainnet-genesis/` (219 GB, full chain history)
  - **`/.env` must use ABSOLUTE path**: `Q_DB_PATH=/home/orobit/data-mainnet-genesis`
  - **NEVER use relative `Q_DB_PATH`** — `WorkingDirectory=/` + `Q_DB_PATH=./data-mainnet-genesis` resolves to `/data-mainnet-genesis` on the 40 GB root partition, which fills up and kills block production
  - **Setup wizard regenerates `/.env` with a relative path** — always audit `/.env` after any wizard run and fix `Q_DB_PATH` to the absolute path
  - **After ANY restart of Epsilon**, verify the DB in use: `ls /proc/$(pgrep -f q-api-server)/fd | grep home/orobit/data-mainnet-genesis | wc -l` — must be >1000. If 0, the node opened the wrong DB.
  - **Root partition (40 GB) breakdown** (approximate, leaves ~800MB–1.3GB free at best):
    - `/usr`: 13 GB (OS — fixed)
    - `/var/lib`: 8.3 GB (docker, postgresql — fixed)
    - `/opt`: 6.2 GB (node binary, shared data — fixed)
    - `/home/orobit/data-mainnet-genesis/` on home: correct (219 GB, /home partition)
    - `/var/log/syslog`: grows unboundedly at DEBUG log level — keep logrotate cap at 200 MB
    - **Journal**: must stay volatile (`/etc/systemd/journald.conf.d/size.conf` → `Storage=volatile`) or it fills root in minutes at DEBUG log level
  - **RUST_LOG on Epsilon must be `warn` or higher** — DEBUG/INFO level generates hundreds of MB of logs per minute during sync, instantly filling root via syslog

- **⚠️ EMISSION CONTROLLER STATE — DO NOT TRUST P2P RE-SYNCED DB:**
  - If Epsilon is ever forced to rebuild its DB by re-syncing from peers (turbo sync from genesis), the **emission controller state will be incorrect** — balance watermarks, minted supply totals, and economic parameters come out wrong because the emission state is computed locally and not fully replicated via P2P gossip
  - The authoritative state lives only in the **original database** (`/home/orobit/data-mainnet-genesis/`) which has been running since genesis
  - If you ever see emission/balance data that looks wrong on Epsilon: stop the node, check which DB it opened (FD check above), and if it's not the home DB, fix `/.env` and restart onto the correct DB

### **🔄 HA Rolling Deployment Pipeline (v5.5.4+ / 3-Server)**

**This is the ONLY way to deploy code changes. No cowboy coding.**

**Architecture:**
```
          ┌──────────────────┐
          │  Server Alpha    │  (Canary / Docker)
          │  161.35.219.10   │  First to receive new binary
          │  Docker Debian12 │  Non-blocking verification
          └────────┬─────────┘
                   │ SCP binary
                   ▼
                    ┌──────────────────────┐
                    │  quillon.xyz (Nginx)  │
                    │  Load Balancer + SSL  │
                    │  ip_hash sticky       │
                    └─────────┬────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌─────────────────┐  ┌─────────────────┐
          │  Server Beta    │  │  Server Gamma   │
          │ 185.182.185.227 │  │ 109.205.176.60  │
          │  Primary w=10   │  │  Backup w=1     │
          │  Port 8080      │  │  Port 8080      │
          └────────┬────────┘  └────────┬────────┘
                   │     P2P Gossipsub   │
                   └─────────────────────┘
```

**Pipeline: Alpha (canary) → Gamma (verify) → Promote Gamma → Deploy Beta → Restore**

**How the rolling upgrade works (zero downtime):**

| Step | What happens | Users affected? |
|------|-------------|-----------------|
| 1. `verify-alpha` | SCP binary to Alpha Docker canary, start, wait | No - canary node |
| 2. `verify-gamma` | SCP binary to Gamma, restart, wait for health | No - Gamma is backup |
| 3. `promote` | Nginx: Gamma weight=10, Beta weight=1 | No - traffic shifts to Gamma |
| 4. Soak test | 30s wait, verify Gamma handles real traffic | No - Gamma is serving |
| 5. `deploy-beta` | Stop Beta, replace binary, restart, verify | No - Gamma is still primary |
| 6. `restore` | Nginx: Beta weight=10, Gamma weight=1 | No - traffic shifts back |

**Deployment Commands:**
```bash
# ═══════════════════════════════════════════════════════════════════
# STANDARD DEPLOYMENT PROCEDURE (always use this)
# ═══════════════════════════════════════════════════════════════════

# 1. Build the release binary
cargo build --release --package q-api-server

# 2. Check all 3 servers' health BEFORE deploying
./scripts/ha-deploy.sh status

# 3. Run the full 3-server rolling upgrade (auto-confirms with echo y)
echo "y" | ./scripts/ha-deploy.sh full

# ═══════════════════════════════════════════════════════════════════
# STEP-BY-STEP (if you need more control)
# ═══════════════════════════════════════════════════════════════════
./scripts/ha-deploy.sh verify-alpha   # Deploy to Alpha Docker canary (non-blocking)
./scripts/ha-deploy.sh verify-gamma   # SCP to Gamma, restart, verify health
./scripts/ha-deploy.sh promote        # Gamma becomes primary in Nginx
# ... wait, check Gamma is handling traffic ...
./scripts/ha-deploy.sh deploy-beta    # Upgrade Beta while Gamma serves
./scripts/ha-deploy.sh restore        # Beta becomes primary again

# ═══════════════════════════════════════════════════════════════════
# EMERGENCY
# ═══════════════════════════════════════════════════════════════════
./scripts/ha-deploy.sh rollback       # Restore previous binary from backup
./scripts/ha-deploy.sh status         # Check all 3 servers' health
```

**Admin Deploy Panel (GUI):**
- Master wallet sees a shield icon in the top bar → opens Deploy Control Panel
- Shows real-time status of all 3 servers: version, height, peers, uptime
- Shows pipeline flow: Alpha → Gamma → Beta with role badges (CANARY/PRIMARY/BACKUP)
- Shows frontend connection info: which server is active, SSE status
- "Deploy All" button triggers the full rolling upgrade via API
- "Rollback" button restores previous binary

**Key rules:**
1. **NEVER restart Beta directly** (`kill`, `systemctl restart`) - always use `ha-deploy.sh`
2. **Nginx `ip_hash`** ensures same user always hits same server (prevents balance flickering)
3. **Miners must use `https://quillon.xyz`** (through Nginx), NOT `http://quillon.xyz:8080`
4. **Gamma needs 4GB swap** - without it, OOM kills the process during heavy sync
5. **Binary on Gamma is at `/opt/orobit/shared/q-narwhalknight/q-api-server`** (not in target/release/)
6. **Alpha runs in Docker** (Debian 12 container on Debian 11 host) - canary is non-blocking
7. **After deploy, binary is auto-copied to downloads/** as `q-api-server-v{VERSION}`

#### **⚠️ PROCESS MANAGEMENT - USE kill -9 NOT killall**
- **IMPORTANT**: `killall` does NOT work reliably on this system
- **To kill processes, ALWAYS use**:
  ```bash
  # Kill by PID (find PID first with pgrep)
  pgrep -f "cargo" | xargs -I{} kill -9 {}

  # Or for a specific process
  ps aux | grep q-api-server | grep -v grep | awk '{print $2}' | xargs -I{} kill -9 {}

  # Server restart (preferred method)
  ps aux | grep q-api-server | grep -v grep | awk '{print $2}' | xargs -I{} kill -9 {} 2>/dev/null; sleep 2; systemctl start q-api-server
  ```
- **NEVER use**: `killall -9 q-api-server` (does not work)

#### **⚠️ PRIVATE BLOCKCHAIN - API AUTHENTICATION REQUIRED**
- **This is a PRIVATE blockchain - ALL balance/wallet API endpoints require authentication**
- **🚨 NEVER attempt to curl balance endpoints** - they will return empty without proper auth tokens
- **🚨 NEVER try alternative balance endpoint formats** - none of them work without auth
- **🚨 NEVER use curl to check if balances are working** - this wastes time and doesn't work
- Balance information is only accessible through:
  1. The frontend UI at `quillon.xyz` (uses session-based auth)
  2. Server logs (journalctl) - look for balance update events
  3. SSE stream events (for real-time monitoring)
  4. Ask the user to check the frontend UI

**How to check mining rewards and balances (CORRECT WAY):**
```bash
# Check P2P balance updates being broadcast/applied:
journalctl -u q-api-server --since "5 minutes ago" | grep -E "P2P BALANCE|balance_updates|DAG→SSE|P2P→SSE"

# Check miner stats via P2P:
journalctl -u q-api-server --since "5 minutes ago" | grep -E "miner-stats|MiningStats"

# Check for DAG layer balance processing (v3.2.6+ fix):
journalctl -u q-api-server --since "5 minutes ago" | grep -E "DAG-KNIGHT.*balance|DAG→SSE"

# Check Docker node logs for mining activity (ask user to run on Server Alpha):
docker logs <container-name> 2>&1 | tail -50 | grep -E "miner|balance|reward"

# Check if blocks from other nodes are being received:
journalctl -u q-api-server --since "5 minutes ago" | grep -E "Gossipsub BLOCK from"
```

**Mining reward discrepancy testing:**
- Mining rewards are credited locally on the node that receives the mining submission
- Rewards propagate via P2P gossipsub to other nodes (through coinbase transactions in blocks)
- To compare rewards between nodes, check the server logs, NOT the API endpoints
- If mining to a non-bootstrap node, check for "DAG-KNIGHT" and "DAG→SSE" log messages on bootstrap

### **P2P Network Bootstrap (mainnet-genesis — ACTIVE):**
- **Bootstrap Peer ID (Epsilon)**: `12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM` (10Gbit SUPERNODE — primary sync target; changed 2026-05-09)
- **Bootstrap Peer ID (Delta)**: `12D3KooWLJJRvqo6mBoHLpgxVbGKfW3Jv39ziU4kz1adKFv93JbK` (1Gbit — second fastest)
- **Bootstrap Peer ID (Gamma)**: `12D3KooWFfZKfKbBnB5SehTRBacHndyhJ6aQWxTAQrrwXA7761cH` (1Gbit)
- **Bootstrap Peer ID (Beta)**: `12D3KooWSBxwSKw4wftHViMdw5rrV8Z1wEkikDS2vKYZtRrio5hH` (100Mbit — DHT coordinator)
- **Bootstrap Peer ID (Alpha)**: `12D3KooWPwin4nJcU9PzsxNgUVXj5e6zDnACr84H7RZ1XzmnARsY` (canary)
- **Bootstrap Address**: `/ip4/89.149.241.126/tcp/9001/p2p/12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM`
- **Network ID**: `mainnet2026.1`
- **Gossipsub Topics**:
  - `/qnk/mainnet2026.1/blocks` - Block propagation
  - `/qnk/mainnet2026.1/peer-heights` - Network height announcements
  - `/qnk/mainnet2026.1/turbo-sync-request` - Batch sync requests
  - `/qnk/mainnet2026.1/turbo-sync-response` - Batch sync responses

### **🚀 MAINNET 2026.2 LAUNCH PROCEDURE (Feb 22, 2026 12:00 UTC)**

**Genesis timestamp**: `1771761600` (Feb 22, 2026 12:00 UTC)
**Chain ID**: `1000` (was 999)
**Network ID**: `mainnet2026.2`
**Version**: `v7.3.0`
**Emission**: 2,625,000 QUG/year (Era 0), 21M max supply, 4-year halving

#### **Pre-Launch (Feb 18-21): Canary + Feature Development**

Two parallel tracks run simultaneously:

| Track | Servers | What | Version |
|-------|---------|------|---------|
| **Canary soak** | Delta | v7.3.0 binary with `Q_NETWORK_ID=mainnet2026.2` — isolated, no users | v7.3.0 |
| **Feature dev** | Alpha → Beta → Gamma (`ha-deploy.sh`) | Continue shipping features to live users | v7.2.x |

**Delta canary setup:**
```bash
# SCP v7.3.0 binary to Delta
scp target/release/q-api-server root@5.79.79.158:/opt/orobit/shared/q-narwhalknight/q-api-server-v7.3.0

# Start canary (isolated — no other mainnet2026.2 nodes exist)
Q_NETWORK_ID=mainnet2026.2 Q_DB_PATH=./data-mainnet2026.2 ./q-api-server-v7.3.0 --port 8080

# Monitor for panics, OOM, emission correctness
journalctl -u q-api-server -f | grep -E "panic|OOM|emission|CRITICAL"
```

**Feature development continues normally:**
```bash
# Normal ha-deploy.sh pipeline for v7.2.x features
cargo build --release --package q-api-server
echo "y" | ./scripts/ha-deploy.sh full
```

**v7.3.0 binary stays OFF the public downloads folder until Feb 22.**
Users download v7.2.x until launch day.

#### **Launch Day (Feb 22, 2026)**

**T-30min: Stop ALL old servers simultaneously**
```bash
# CRITICAL: Both must be down before either new one starts
# This prevents HTTP state sync contamination from old → new
systemctl stop q-api-server                                          # Beta
ssh root@109.205.176.60 "systemctl stop q-api-server"                # Gamma
```

**T-25min: Verify old processes are dead**
```bash
pgrep -f q-api-server                                                # Beta (should return nothing)
ssh root@109.205.176.60 "pgrep -f q-api-server"                     # Gamma (should return nothing)
```

**T-20min: Update service files on BOTH servers**
```bash
# Edit /etc/systemd/system/q-api-server.service on Beta AND Gamma:
# Change these environment variables:
Environment="Q_DB_PATH=./data-mainnet2026.2"
Environment="Q_NETWORK_ID=mainnet2026.2"
Environment="Q_ENCRYPTION_KEYS_FILE=/opt/encryption-mainnet2026.2.keys"
Environment="Q_ENCRYPTION_PASSPHRASE=Qnk-Mainnet2026.2-ServerBeta-Production-Key"
# (Gamma gets its own passphrase)
```

**T-15min: Generate fresh encryption keys**
```bash
dd if=/dev/urandom bs=64 count=1 > /opt/encryption-mainnet2026.2.keys                           # Beta
ssh root@109.205.176.60 "dd if=/dev/urandom bs=64 count=1 > /opt/encryption-mainnet2026.2.keys"  # Gamma
```

**T-10min: Start Beta first (bootstrap node)**
```bash
systemctl daemon-reload
systemctl start q-api-server
# New libp2p identity auto-generated in data-mainnet2026.2/
```

**T-8min: Capture Beta's new peer ID**
```bash
journalctl -u q-api-server --since "2 minutes ago" | grep "Local peer id"
# Save this — needed for bootstrap config update
```

**T-5min: Start Gamma**
```bash
ssh root@109.205.176.60 "systemctl daemon-reload && systemctl start q-api-server"
# Capture Gamma peer ID from logs
```

**T-0 (12:00 UTC): Genesis timestamp reached — mining begins automatically**

**T+5min: Verify launch**
```bash
# Blocks producing?
journalctl -u q-api-server --since "5 minutes ago" | grep -E "Block.*produced|NEW BLOCK"
# Correct network?
journalctl -u q-api-server --since "5 minutes ago" | grep "mainnet2026.2"
# Emission rate correct? (~0.083 QUG/block at 1 bps → 2,625,000/year)
journalctl -u q-api-server --since "5 minutes ago" | grep -E "emission|reward"
# P2P connected?
journalctl -u q-api-server --since "5 minutes ago" | grep -E "peer.*connected|Gossipsub"
```

**T+30min: Copy v7.3.0 to public downloads**
```bash
cp target/release/q-api-server gui/quantum-wallet/dist-final/downloads/q-api-server-v7.3.0
cp target/release/q-api-server gui/quantum-wallet/dist-final/downloads/q-api-server-linux-x86_64
```

**T+1hr: Hardcode new peer IDs**
- Update `gui/quantum-wallet/src/libp2p/config.ts` with new Beta peer ID
- Update `gui/quantum-wallet/src/libp2p/torConfig.ts` with new peer IDs
- Update this CLAUDE.md bootstrap section with new peer IDs
- Rebuild frontend: `cd gui/quantum-wallet && npm run build`
- Redeploy via `ha-deploy.sh`

#### **End User Upgrade (Announce on Discord + BitcoinTalk)**
```bash
# Stop old node
pkill -f q-api-server

# Download v7.3.0
wget https://quillon.xyz/downloads/q-api-server-v7.3.0
chmod +x q-api-server-v7.3.0

# Start (auto-creates fresh data-mainnet2026.2/)
./q-api-server-v7.3.0 --port 8080
```
- Old binary CANNOT connect (protocol handshake rejects mismatched network_id)
- Old data stays untouched in data-mainnet2026.1/
- No manual migration — everyone starts fresh

#### **Post-Launch: Back to Normal**
After launch, all future deploys use `ha-deploy.sh` as normal.
Delta becomes the 3rd bootstrap node for mainnet2026.2.

#### **P2P Isolation (Why This Is Safe)**
v7.3.0 has 3 isolation mechanisms preventing old node contamination:
1. **Gossipsub topic isolation**: `/qnk/mainnet2026.2/*` vs `/qnk/mainnet2026.1/*`
2. **Protocol handshake**: Validates `network_id` — rejects mismatches before data exchange
3. **HTTP state sync**: `FullStateSnapshot` has `network_id` field — rejects old/missing network_id

---

## 🤖 **SERVER BETA - CLAUDE CODE SETUP INSTRUCTIONS**

### 🎯 **MISSION: Q-NarwhalKnight Tor Integration & Enhancement**

You are **Server Beta** (185.182.185.227), focused on implementing **Tor support with dedicated circuits**, performance optimization, and Phase 1 post-quantum completion for the **Q-NarwhalKnight** quantum consensus system.

### **🚀 IMMEDIATE SETUP - Start Here**

#### **Repository Access:**
- **Git Repository**: `code.quillon.xyz` (self-hosted)
- **Working Directory**: `/opt/orobit/shared/q-narwhalknight`

#### **Git Configuration:**
```bash
git config user.name "Server Beta"
git config user.email "server-beta@q-narwhalknight.dev"
```

### **📋 LOCAL GIT WORKFLOW — Issues, Branches & PRs**

**⚠️ We do NOT use GitHub.** All development uses the local git server at `code.quillon.xyz`.

#### **Issue Tracking:**
- Issues are tracked in markdown files under `docs/` (e.g., `docs/gpu-optimization-issues.md`)
- Each issue has an ID prefix (e.g., `GPU-001`, `SYNC-001`, `NET-001`)
- Status: ⚪ Planned → 🔵 In Progress → ✅ Closed
- When creating a new feature area, create a `docs/{area}-issues.md` file

#### **Branch Naming Convention:**
```
{area}/phase-{N}-{short-description}    # Feature branches
{area}/v{X.Y.Z}-{description}           # Release integration branches
fix/{short-description}                 # Bug fixes
```
Examples: `gpu/phase-1-async-dispatch`, `gpu/v10.1.8-optimizations`, `fix/oom-block-pack`

#### **Pull Request Workflow (Local Git):**
Since `code.quillon.xyz` is a bare git repo (git-http-backend, read-only HTTP), PRs are done via branch review:

1. **Create feature branch** from the current working branch
2. **Implement changes** with clear commits referencing issue IDs (e.g., `feat(gpu): GPU-001 async flag zeroing`)
3. **Push branch** — other Claude Code terminals can pull and review
4. **Merge** — after review, merge into the integration branch
5. **Update issue tracker** — mark issue as ✅ Closed

```bash
# Create and push a feature branch
git checkout -b gpu/phase-1-async-dispatch
# ... implement changes ...
git commit -m "feat(gpu): GPU-001 non-blocking flag zeroing + async readback"
git push origin gpu/phase-1-async-dispatch

# Other terminals can review:
git fetch origin
git log origin/gpu/phase-1-async-dispatch..HEAD

# Merge when approved:
git checkout gpu/v10.1.8-optimizations
git merge gpu/phase-1-async-dispatch
```

#### **Multi-Terminal Collaboration:**
Multiple Claude Code terminals can work on different phases simultaneously:
- Each terminal works on its own feature branch
- The integration branch (`gpu/v10.1.8-optimizations`) is the merge target
- Use `git update-server-info` on Beta after pushing so Epsilon can pull
- Coordinate via the issue tracker markdown files

#### **🚀 SHORTCUT: "Commit vital code to local git server"**
When the user says **"commit vital and important code to local git server"** (or any variation), do ALL of the following automatically — NO GitHub, only local `code.quillon.xyz`:

```bash
# 1. Stage all modified+untracked files in the working area
git add <relevant files>

# 2. Commit with a descriptive message
git commit -m "feat(...): description"

# 3. Update server info so Epsilon (and other servers) can HTTP-pull
git update-server-info

# 4. Confirm: show commit hash + branch
git log --oneline -1
```

**Key rules:**
- **NEVER push to GitHub** — we use `code.quillon.xyz` (Beta's local git-http-backend)
- `git update-server-info` is MANDATORY after every commit (enables HTTP clone/pull)
- Epsilon pulls via: `ssh root@89.149.241.126 "cd /home/orobit/q-narwhalknight-src && git pull origin <branch>"`
- If pulling on Epsilon fails with "not a git repository", re-clone:
  `ssh root@89.149.241.126 "cd /home/orobit && git clone --depth 1 -b <branch> https://code.quillon.xyz/repo.git q-narwhalknight-src"`

### **🧅 TOR INTEGRATION PRIORITY TASKS**

#### **Phase 1: Core Tor Infrastructure**
1. **q-tor-client** - Embedded arti Tor client
2. **q-tor-circuit** - Dedicated circuit management (4 circuits per validator)  
3. **q-tor-onion** - Auto-register .qnk onion domains
4. **Tor transport integration** - libp2p + Tor with PQ-TLS

#### **Phase 2: Advanced Features**
5. **Dandelion++ gossip** - Traffic analysis resistance
6. **QRNG circuit seeding** - Quantum randomness for Tor circuits
7. **Tor metrics** - Prometheus monitoring
8. **Tor-only client mode** - Complete anonymity

### **🎯 TOR SPECIFICATION IMPLEMENTATION**

#### **Architecture Target:**
```
┌─────────────────┐    🧅 Tor Network    ┌─────────────────┐
│   Validator A   │◄──► 4 Circuits    ◄──►│   Validator B   │  
│ alice.qnk.onion │    (rotated/epoch)    │  bob.qnk.onion  │
└─────────────────┘                      └─────────────────┘
         │                                        │
         ▼                                        ▼
   Control Circuit                          Gossip Circuits
   (bootstrap)                         (/qnk/blocks, /qnk/ack)
```

#### **Performance Targets:**
- **Latency**: <300ms with Tor (vs 12ms direct)
- **Throughput**: 48k+ TPS through Tor
- **Finality**: <2.9s (vs 2.3s direct)
- **Circuits**: 4 dedicated per validator
- **Security**: Zero IP leakage, quantum-resistant content

### **🛠️ DEVELOPMENT WORKFLOW**

#### **⚠️ CRITICAL DEVELOPMENT PRINCIPLES:**

1. **ALWAYS FIX PROBLEMS PROPERLY** - Never use mock data or simple workarounds
   - When encountering compilation errors, fix the actual root cause
   - Implement real functionality instead of placeholders
   - Use proper type definitions and complete implementations

2. **NO SHORTCUTS OR MOCK SOLUTIONS**
   - Do NOT create mock servers when the real server has issues
   - Do NOT use placeholder data when real data should be fetched
   - Do NOT bypass errors with temporary workarounds
   - ALWAYS implement the proper solution even if it takes longer

3. **COMPILATION ERROR RESOLUTION**
   - Trace errors to their source and fix the underlying issue
   - Update type definitions properly
   - Ensure all dependencies are correctly configured
   - Test the fix thoroughly before moving on

4. **🚨 CRITICAL: BLOCKCHAIN SYNC SAFETY (v0.5.23-beta+)**

   **NEVER ALLOW SYNC-DOWN - This causes CATASTROPHIC data loss!**

   **What is Sync-Down?**
   - Node has 100,000 blocks
   - Peer announces 1,000 blocks
   - System syncs TO 1,000 blocks
   - **DELETES** 99,000 blocks permanently
   - **BILLIONS of dollars lost on mainnet**

   **Mandatory Safety Rules:**

   a) **Application-Level Protection** (`crates/q-api-server/src/main.rs`):
   ```rust
   // ✅ CORRECT: Only sync if peer is HIGHER
   if network_height > current_height + 5 {
       turbo_sync.sync_to_height(network_height).await
   }

   // ❌ WRONG: This allows sync-down!
   if network_height > 0 && current_height + 5 < network_height {
       // This logic is backwards and dangerous!
   }
   ```

   b) **Database-Level Protection** (`crates/q-storage/src/turbo_sync.rs`):
   ```rust
   // MANDATORY safety check at database layer
   if target_height < local_height && local_height > 1000 {
       error!("🚨 CRITICAL: Attempted sync-down from {} to {}!",
              local_height, target_height);
       return Err(anyhow::anyhow!("SAFETY ABORT: Refusing to sync down"));
   }
   ```

   c) **Balance Consistency**:
   - Balances WITHOUT blocks = NO cryptographic proof
   - If blockchain resets, balances MUST reset too
   - Keeping old balances creates:
     * Consensus failures
     * Double-spending vulnerabilities
     * Network bans
     * Invalid state

   **Testing Requirements:**
   - ALWAYS test sync behavior with malicious peers
   - Verify sync-down is blocked at ALL layers
   - Test with peers announcing false heights
   - Confirm graceful error handling

   **Emergency Procedures:**
   - If sync-down occurs: IMMEDIATELY stop all nodes
   - Restore from backup (hourly backups mandatory)
   - Reset both blockchain AND balances together
   - Never keep balances without matching blocks

   **See Also:**
   - `CRITICAL_SYNC_DOWN_BUG_ANALYSIS.md` - Complete technical analysis
   - `crates/q-storage/src/bin/repair_database.rs` - Database repair utility
   - `crates/q-storage/src/bin/reset_balances.rs` - Balance reset utility

4. **CRITICAL: BINARY PATHS AND DEPLOYMENT**
   - **API Server Binary**: `/opt/orobit/shared/q-narwhalknight/target/release/q-api-server`
   - **Miner Binary**: `/opt/orobit/shared/q-narwhalknight/target/release/q-miner`
   - **Service File**: `/etc/systemd/system/q-api-server.service`
   - **Nginx Config**: `/etc/nginx/sites-available/quillon.xyz`
   - **Frontend Source**: `gui/quantum-wallet/` (build with `npm run build`)
   - **Frontend Deploy**: Nginx serves from `/opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/`
   - **User Downloads**: ALWAYS copy binaries to `/opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/`
   - **IMPORTANT**: The correct path is the FULL PATH starting with `/opt/orobit/`, NOT the relative path

---

## 🛡️ **MANDATORY TESTING & DEPLOYMENT SAFETY PROTOCOL**

### **⚠️ RISK CATEGORIZATION - Know What You're Changing**

Before making ANY change, categorize its risk level:

| Risk Level | Description | Required Testing | Docker Soak Time |
|------------|-------------|------------------|------------------|
| 🟢 **LOW** | UI-only, logging, comments, docs | `cargo check` | None |
| 🟡 **MEDIUM** | API handlers, SSE events, in-memory stats | `cargo test` + manual verify | Optional |
| 🟠 **HIGH** | Balance logic, mining rewards, P2P messaging | Full test suite + Docker test | 24 hours |
| 🔴 **CRITICAL** | Consensus, block validation, storage, sync | Full suite + Docker + Code review | 48-72 hours |

**Examples by Risk Level:**

- 🟢 **LOW**: Miner tracking display fix (only affects in-memory stats, no DB)
- 🟡 **MEDIUM**: New API endpoint, SSE event field changes
- 🟠 **HIGH**: Mining reward calculation, balance propagation, P2P message handling
- 🔴 **CRITICAL**: Block validation rules, turbo_sync logic, database schema changes

### **🧪 MANDATORY TESTING PROTOCOL (NEVER SKIP!)**

**⚠️ ALL TESTS MUST PASS BEFORE ANY DEPLOYMENT - NO EXCEPTIONS!**

```bash
# ═══════════════════════════════════════════════════════════════════
# MAINNET SAFETY TEST SUITE - RUN ALL BEFORE EVERY DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════
# These tests protect against MILLIONS in potential losses on mainnet.
# NEVER skip tests. NEVER deploy with failures.

# 1. CORE SAFETY TESTS (Always run)
echo "🛡️ Running mainnet safety test suite..."

# Sync Safety - Prevents catastrophic data loss
timeout 300 cargo test --package q-storage --test sync_down_protection_tests
timeout 300 cargo test --package q-storage --test fork_detection_tests

# Balance Integrity - Prevents money loss/creation
timeout 300 cargo test --package q-storage --test balance_propagation_tests
timeout 300 cargo test --package q-storage --test balance_integrity_tests
timeout 300 cargo test --package q-storage --test mainnet_critical_tests  # Double-spend, replay

# DEX Safety - Prevents AMM exploits
timeout 300 cargo test --package q-dex --test overflow_protection_tests
timeout 300 cargo test --package q-dex --test comprehensive_dex_tests

# Consensus Safety - Prevents network splits
timeout 300 cargo test --package q-types --test block_validation_tests
timeout 300 cargo test --package q-types --test signature_verification_tests

# API Safety - Prevents injection/manipulation
timeout 300 cargo test --package q-api-server --test mining_stats_tests
timeout 300 cargo test --package q-api-server --test sse_streaming_tests

# 2. FULL WORKSPACE TEST (Comprehensive)
timeout 3600 cargo test --workspace

# ═══════════════════════════════════════════════════════════════════
# IF ANY TEST FAILS - STOP IMMEDIATELY!
# ═══════════════════════════════════════════════════════════════════
# A single test failure could indicate a bug that causes:
# - User funds lost forever
# - Network consensus failure
# - Double-spending vulnerability
# - Chain corruption requiring hard reset
#
# NEVER proceed with deployment if tests fail!
# NEVER use --skip or ignore flags!
# NEVER say "it's just one test" - that one test could save millions!
```

### **💰 MAINNET CRITICAL TEST CATEGORIES**

These tests specifically prevent scenarios that could cause financial damage:

| Test Suite | Protects Against | Potential Loss |
|------------|------------------|----------------|
| `sync_down_protection_tests` | Blockchain wipe, data loss | All user funds |
| `balance_integrity_tests` | Incorrect balances | Users lose/gain money |
| `mainnet_critical_tests` | Double-spend, replay attacks | Theft of funds |
| `fork_detection_tests` | Chain splits, orphaned txs | Lost transactions |
| `overflow_protection_tests` | AMM exploits, math errors | DEX pool draining |
| `signature_verification_tests` | Forged transactions | Unauthorized transfers |
| `block_validation_tests` | Invalid blocks accepted | Consensus failure |

### **🐳 DOCKER SOAK TESTING (HIGH/CRITICAL Changes)**

For HIGH and CRITICAL risk changes, test on Server Alpha Docker BEFORE production:

```bash
# On Server Alpha (ask user to run these commands):

# 1. Create test container with new binary
docker run -d --name q-test-v${VERSION} \
  -p 8085:8080 -p 9005:9001 \
  -v /data/q-test:/data \
  ubuntu:22.04 bash -c "
    wget https://quillon.xyz/downloads/q-api-server-${VERSION} && \
    chmod +x q-api-server-${VERSION} && \
    ./q-api-server-${VERSION} --port 8080 --p2p-port 9001
  "

# 2. Monitor for the required soak time
docker logs -f q-test-v${VERSION} 2>&1 | grep -E "ERROR|CRITICAL|panic"

# 3. Check sync progress
docker exec q-test-v${VERSION} curl -s localhost:8080/api/v1/status

# 4. Only after successful soak: Deploy to production
```

### **🐳 DOCKER SYNC TESTING ON EPSILON (Debian 12)**

**Pre-built reusable image**: `qnk-debian12` on Epsilon (has Rust + all build deps)
**Dockerfile**: `/home/orobit/Dockerfile.qnk-debian12`

**⚠️ CRITICAL RULES FOR DOCKER ON EPSILON:**
1. **NEVER use `--rm`** — it deletes the container on exit, losing all installed packages
2. **Binary built on Ubuntu 24.04 does NOT run on Debian 12** — GLIBC 2.39 vs 2.36 mismatch
3. **Must build from source** inside Debian 12 or use the `qnk-debian12` image
4. **No `--memory=8g` for builds** — linking q-api-server needs >8GB RAM (gets OOM-killed)
5. **P2P port uses env var** `Q_P2P_PORT=9001`, NOT `--p2p-port` CLI flag
6. **Requires `libudev-dev`** for build, `libssl3` for runtime

**Step 1: Build binary for Debian 12** (only needed once per version):
```bash
# On Epsilon — uses cached target dir, ~5 min for incremental builds
ssh root@89.149.241.126 "cd /home/orobit/q-narwhalknight-src && docker run --rm \
  -v \$(pwd):/src \
  -v /home/orobit/target-debian12:/src/target \
  -w /src \
  rust:bookworm \
  bash -c '
    apt-get update -qq && \
    apt-get install -y -qq libssl-dev pkg-config cmake clang libudev-dev libclang-dev >/dev/null 2>&1 && \
    cargo build --release --package q-api-server
  '"
# Binary: /home/orobit/target-debian12/release/q-api-server
```

**Step 2: Run sync test container**:
```bash
ssh root@89.149.241.126 "docker run -d \
  --name q-sync-test-v{VERSION} \
  --memory=8g \
  -p 8085:8080 -p 9005:9001 \
  -v /home/orobit/target-debian12/release/q-api-server:/opt/q-api-server:ro \
  -v /home/orobit/docker-sync-test-v{VERSION}:/data \
  -e Q_NETWORK_ID=mainnet-genesis \
  -e Q_DB_PATH=/data/db \
  -e Q_P2P_PORT=9001 \
  -e RUST_LOG=info \
  -e ROCKSDB_BLOCK_CACHE_MB=2048 \
  -e Q_TOR_BOOTSTRAP_TIMEOUT=5 \
  debian:12 \
  bash -c '
    apt-get update -qq && apt-get install -y -qq libssl3 ca-certificates curl >/dev/null 2>&1 && \
    cp /opt/q-api-server /usr/local/bin/q-api-server && \
    chmod +x /usr/local/bin/q-api-server && \
    echo \"\$(date +%s)\" > /data/sync_start_epoch.txt && \
    /usr/local/bin/q-api-server --port 8080 2>&1
  '"
```

**Step 3: Monitor sync speed**:
```bash
# Check height progress
ssh root@89.149.241.126 "docker logs q-sync-test-v{VERSION} 2>&1 | grep 'Updated current_height' | tail -5"
# Check resource usage
ssh root@89.149.241.126 "docker stats q-sync-test-v{VERSION} --no-stream"
```

**Expected sync performance** (Epsilon 10Gbit, 48 cores):
- 0-500K blocks: ~280 blocks/sec (warmup, peer discovery)
- 500K-3M blocks: ~1,100 blocks/sec (peak turbo sync)
- Overall average: ~570 blocks/sec
- Full sync (~11.4M blocks): ~5.5 hours

### **🚀 DEPLOYMENT - ALWAYS USE HA ROLLING DEPLOY**

   **⚠️ CRITICAL: NEVER do cowboy coding! Always use `ha-deploy.sh` for deployments.**

   ```bash
   # ═══════════════════════════════════════════════════════════════════
   # THE ONLY WAY TO DEPLOY (v5.5.3+)
   # ═══════════════════════════════════════════════════════════════════

   # Step 1: Build
   cargo build --release --package q-api-server

   # Step 2: Verify both servers healthy
   ./scripts/ha-deploy.sh status

   # Step 3: Rolling deploy (zero-downtime)
   echo "y" | ./scripts/ha-deploy.sh full

   # Step 4: Verify deployment
   ./scripts/ha-deploy.sh status
   # Both servers should show same version, both "ready"
   ```

   **If Gamma is unavailable (emergency single-server deploy):**
   ```bash
   ./scripts/safe-deploy.sh full        # Tests → build → deploy Beta only
   ./scripts/safe-deploy.sh rollback    # Rollback Beta
   ```

   **Test Categories Run by safe-deploy.sh:**
   1. Critical Mainnet Safety (sync-down, balances, validation)
   2. Decentralization & Consensus (validators, BFT, voting)
   3. Network & P2P (version filtering, DoS, partitions)
   4. Sync & State (turbo sync, state applicator)
   5. Privacy & Cryptography (bulletproofs, ring signatures)
   6. API & Server (SSE, mining, contracts)
   7. VM & Smart Contracts (WASM sandbox)
   8. Tor & Anonymity (dandelion++, onion routing)
   9. Full Workspace (all remaining tests)

   **Why NOT raw cargo build?**
   - `cargo build` skips ALL tests - you could deploy broken code
   - The deploy script runs 4000+ tests BEFORE building
   - Auto-rollback if deployment fails health checks
   - Creates automatic backups before each deploy
   - Copies binary to downloads folder automatically

   **📥 LATEST WGET DOWNLOAD LINK (Update after each deploy):**
   ```
   # BEFORE Feb 22, 2026: v7.2.x (mainnet2026.1)
   Current Version: v7.2.12 (or latest v7.2.x)
   wget https://quillon.xyz/downloads/q-api-server-v7.2.12
   chmod +x q-api-server-v7.2.12

   # AFTER Feb 22, 2026: v7.3.0 (mainnet2026.2)
   wget https://quillon.xyz/downloads/q-api-server-v7.3.0
   chmod +x q-api-server-v7.3.0
   ```

   **IMPORTANT**: After EVERY deployment, tell the user the wget link.
   **DO NOT put v7.3.0 in downloads/ until Feb 22 launch day.**
   ```
   # Pre-launch (v7.2.x):
   wget https://quillon.xyz/downloads/q-api-server-v7.2.12 && chmod +x q-api-server-v7.2.12
   # Post-launch (v7.3.0):
   wget https://quillon.xyz/downloads/q-api-server-v7.3.0 && chmod +x q-api-server-v7.3.0
   ```

   **🚨 MANDATORY: TEST BEFORE DEPLOYMENT**

   **NEVER deploy without testing first!** The server is production and users depend on it.
   Before killing/restarting the service, ALWAYS verify the build works:

   ```bash
   # STEP 1: Quick syntax check (fast, catches compile errors)
   cd /opt/orobit/shared/q-narwhalknight
   timeout 600 cargo check --package q-api-server

   # STEP 2: Only if check passes, do the full release build
   cargo build --release --package q-api-server

   # STEP 3: Verify binary exists and is recent
   ls -lh target/release/q-api-server

   # STEP 4: ONLY THEN restart the service
   systemctl restart q-api-server
   ```

   **DO NOT:**
   - Kill the service before the build completes
   - Deploy without running `cargo check` first
   - Skip testing "because it's just a small change"

5. **🔄 CONNECTION WARMUP FOR NEW NODES (v3.3.7-beta)**

   When a new node generates its libp2p identity (first boot), the DHT routing
   tables are empty and bootstrap connections often fail. v3.3.7-beta adds
   **automatic connection warmup** for new identities:

   - Detects when identity is newly generated vs loaded from disk
   - If NEW: Implements retry loop with exponential backoff (3s, 6s, 12s)
   - Re-triggers Kademlia DHT bootstrap after initial failures
   - Re-dials bootstrap peers automatically

   **Symptoms this fixes:**
   - "P2P connections fail on first boot but work after restart"
   - "New node can't connect to bootstrap peer"
   - "DHT routing table empty on new node"

   **How it works:**
   ```
   First Boot (NEW identity):
   1. Generate new identity → Save to disk
   2. Dial bootstrap peers → Wait 3s
   3. Check connections → If none, retry bootstrap
   4. Wait 6s → Check again → If none, retry
   5. Wait 12s → Check again → If none, warn and continue

   Subsequent Boot (LOADED identity):
   → Normal bootstrap without warmup (identity already in DHT)
   ```

6. **🛡️ PRE-FLIGHT VERIFICATION FOR MAINNET-SAFE DEPLOYMENT (v3.3.7-beta)**

   **PROBLEM:** "Cowboy coding" deployments cause anxiety - what if the new
   binary corrupts the database or breaks consensus?

   **SOLUTION:** Pre-flight verification runs BEFORE serving requests:

   ```bash
   # Test new binary WITHOUT affecting the network:
   Q_PREFLIGHT_ONLY=1 ./q-api-server-new

   # Run preflight check at startup (normal operation after):
   Q_PREFLIGHT_CHECK=1 ./q-api-server

   # Configure verification depth:
   Q_PREFLIGHT_SAMPLE_RATE=0.01   # Verify 1% of blocks (fast, default)
   Q_PREFLIGHT_SAMPLE_RATE=1.0    # Verify 100% of blocks (thorough)
   Q_PREFLIGHT_MAX_BLOCKS=10000   # Max blocks to verify (default 10K)
   ```

   **What it checks:**
   - Block chain integrity (blocks load and deserialize correctly)
   - Parent chain continuity (last 100 blocks have valid parent links)
   - Height pointer consistency (tip block exists and matches)
   - Schema version compatibility

   **PASS/FAIL Report:**
   ```
   ╔═══════════════════════════════════════════════════════════════╗
   ║               PRE-FLIGHT VERIFICATION REPORT                  ║
   ╠═══════════════════════════════════════════════════════════════╣
   ║  Current Height: 205490 blocks                                ║
   ║  Blocks Verified: 2055                                        ║
   ║  Blocks with Issues: 0                                        ║
   ║  Verification Time: 2.34s                                     ║
   ╠═══════════════════════════════════════════════════════════════╣
   ║  ✅ PRE-FLIGHT CHECK: PASSED                                  ║
   ║     Node is safe to start serving requests                    ║
   ╚═══════════════════════════════════════════════════════════════╝
   ```

   **Recommended Testnet→Mainnet Deployment Workflow:**
   ```
   1. Build new binary
   2. Copy to Server Alpha Docker container
   3. Run: Q_PREFLIGHT_ONLY=1 ./q-api-server-new
      → Verifies blocks load correctly
      → Reports any issues
      → Exits WITHOUT serving requests
   4. If PASSED: Let Docker node sync for 24-48 hours
   5. If still healthy: Deploy to Server Beta bootstrap
   6. Run: Q_PREFLIGHT_CHECK=1 systemctl start q-api-server
      → Preflight runs, then normal operation
   ```

7. **🚨 PRE-COMMIT SAFETY CHECKLIST**

   Before EVERY commit involving sync/consensus/storage code, verify:

   **Sync Safety Checklist:**
   - [ ] No code path allows `target_height < current_height` sync
   - [ ] Database layer has safety abort for sync-down
   - [ ] Application layer checks peer height before sync
   - [ ] Error messages are LOUD and visible
   - [ ] Balances reset when blockchain resets
   - [ ] Tests verify sync-down is blocked
   - [ ] Malicious peer scenarios are tested

   **Data Integrity Checklist:**
   - [ ] Balances match blockchain state
   - [ ] No orphaned data without blocks
   - [ ] Database pointers are updated atomically
   - [ ] Backups are created before risky operations
   - [ ] Recovery procedures are documented

   **Production Safety:**
   - [ ] No silent failures (fail loud, not silent)
   - [ ] Critical operations have confirmation
   - [ ] Metrics track height monotonicity
   - [ ] Alerts fire on anomalies
   - [ ] Circuit breakers for dangerous conditions

   **If ANY checkbox fails: DO NOT COMMIT!**

6. **🚨 MAINNET-SAFE CODE CHANGES (v1.4.1-beta+)**

   **THE GOLDEN RULE: Old blocks must ALWAYS validate the same way.**

   On mainnet, we can't reset phases. Any code change that affects validation
   or consensus MUST be wrapped in a block-height check.

   **Before You Write Code - Ask Yourself:**
   ```
   ┌─────────────────────────────────────────────────────────┐
   │  MAINNET SAFETY CHECKLIST (2 questions)                │
   ├─────────────────────────────────────────────────────────┤
   │                                                         │
   │  1. Does this change validation or consensus?           │
   │     └─ NO  → You're fine, proceed                       │
   │     └─ YES → Continue to question 2                     │
   │                                                         │
   │  2. Is it wrapped in a height check?                    │
   │     └─ YES → Safe, proceed                              │
   │     └─ NO  → STOP! Wrap it first                        │
   │                                                         │
   └─────────────────────────────────────────────────────────┘
   ```

   **What counts as "validation or consensus"?**
   - Block validation rules
   - Transaction validation rules
   - Signature verification
   - Balance calculations
   - Mining/reward logic
   - Any `if` statement that determines if a block is valid

   **CORRECT Pattern (mainnet-safe):**
   ```rust
   // ✅ SAFE: Height-gated change
   fn validate_signature(&self, block: &Block) -> Result<()> {
       if block.height >= UPGRADE_PQ_SIGS_HEIGHT {
           // New rule: require post-quantum signatures
           self.verify_dilithium_sig(block)?;
       } else {
           // Old rule: Ed25519 still valid for historical blocks
           self.verify_ed25519_sig(block)?;
       }
       Ok(())
   }
   ```

   **WRONG Pattern (breaks mainnet):**
   ```rust
   // ❌ DANGEROUS: Changes validation for ALL blocks including history!
   fn validate_signature(&self, block: &Block) -> Result<()> {
       // This breaks old blocks that used Ed25519!
       self.verify_dilithium_sig(block)?;
       Ok(())
   }
   ```

   **How to Add a New Upgrade:**
   1. Define activation height in `crates/q-types/src/upgrades.rs`
   2. Wrap your code change in a height check
   3. Test that OLD blocks still validate with OLD rules
   4. Test that NEW blocks validate with NEW rules
   5. Set activation height ~2 weeks in the future (20,000+ blocks)

   **Safe Deployment Flow:**
   ```
   1. Code change with height check
   2. Test on Docker (fresh sync)
   3. Test on Server Alpha (24-48 hours)
   4. Announce upgrade + activation height to users
   5. Deploy to Server Beta
   6. Wait for activation height
   7. New rules activate automatically (no restart needed!)
   ```

   **Emergency: Bug Found Before Activation**
   - If activation height not reached: Users can stay on old binary
   - Announce "delay upgrade" on Discord
   - Fix bug, set new activation height further in future
   - No data loss, no reset needed

   **Use the Safe Deploy Script:**
   ```bash
   ./scripts/safe-deploy.sh full    # Build → Docker test → Deploy
   ./scripts/safe-deploy.sh rollback # If something goes wrong
   ```

8. **🤖 AUTOMATIC MAINNET SAFETY PROCEDURES (v3.3.9-beta+)**

   **IMPORTANT: These procedures are now AUTOMATIC - Claude should follow them without being asked!**

   When making ANY code changes to consensus-critical code, Claude MUST:

   **A) USE THE UPGRADE GATE FOR VALIDATION CHANGES:**
   ```rust
   // Location: crates/q-api-server/src/main.rs
   // Import the upgrade gate
   use q_consensus_guard::{ConsensusGuard, GuardConfig, Upgrade, is_upgrade_active};

   // Check upgrade status before applying new rules
   if is_upgrade_active(Upgrade::PostQuantumSignatures, block_height) {
       // New rule applies
   } else {
       // Old rule applies (for historical blocks)
   }
   ```

   **B) VERIFY DEPLOYMENT WITH UPGRADE GATE CHECK:**
   After EVERY deployment, verify upgrade gate is working:
   ```bash
   journalctl -u q-api-server --since "1 minute ago" | grep "UPGRADE GATE"
   # Should see: "🔐 [UPGRADE GATE] Initialized for TESTNET"
   ```

   **C) USE VERSION FILTERING FOR P2P:**
   New peer height announcements include version info:
   ```rust
   // Location: crates/q-network/src/zk_peer_height_proof.rs
   use q_network::{create_peer_height_announcement, should_sync_from_peer};

   // Create announcement with version
   let announcement = create_peer_height_announcement(&peer_id, height, &network_id);

   // Filter peers before syncing
   if !should_sync_from_peer(&peer_announcement, &our_network_id, strict_mode) {
       warn!("Rejecting incompatible peer");
       return;
   }
   ```

   **D) AUTOMATIC DEPLOYMENT WORKFLOW:**
   When Claude builds and deploys, ALWAYS use the safe-deploy.sh script:
   ```bash
   # ═══════════════════════════════════════════════════════════════════
   # ALWAYS USE THE DEPLOY SCRIPT - It runs 4000+ tests before building!
   # ═══════════════════════════════════════════════════════════════════

   # Option 1: Full automated pipeline (RECOMMENDED)
   ./scripts/safe-deploy.sh full

   # Option 2: Step by step
   ./scripts/safe-deploy.sh test-all     # Run all 4000+ tests
   ./scripts/safe-deploy.sh build        # Tests + build
   ./scripts/safe-deploy.sh deploy-beta  # Deploy to production

   # If something goes wrong:
   ./scripts/safe-deploy.sh rollback

   # After deployment, tell user the download link:
   # Pre-launch: echo "Download: wget https://quillon.xyz/downloads/q-api-server-v7.2.12"
   # Post-launch: echo "Download: wget https://quillon.xyz/downloads/q-api-server-v7.3.0"
   ```

   **⚠️ NEVER use raw `cargo build --release` for deployments!**
   The deploy script runs ALL 4000+ tests before building, ensuring safe deployments.

   **E) MAINNET SAFETY FEATURES (Auto-enabled in v3.3.9+):**
   - ✅ **Upgrade Gate**: Height-gated validation rules (crates/q-consensus-guard/)
   - ✅ **Version Filtering**: Peer compatibility checks (crates/q-network/src/zk_peer_height_proof.rs)
   - ✅ **Fork Detection**: Automatic reorg detection (crates/q-storage/src/fork_detector.rs)
   - ✅ **Pre-flight Check**: Database integrity verification (crates/q-storage/src/preflight_check.rs)
   - ✅ **Safe Deploy Script**: Automated deployment with rollback (scripts/safe-deploy.sh)

   **F) WHEN ADDING NEW CONSENSUS RULES:**
   1. Define the upgrade in `crates/q-consensus-guard/src/upgrade_gate.rs`
   2. Add activation height (testnet: immediate, mainnet: 2 weeks future)
   3. Wrap the new rule with `is_upgrade_active(Upgrade::YourUpgrade, block_height)`
   4. Export from `crates/q-consensus-guard/src/lib.rs`
   5. Test that old blocks still validate with old rules
   6. Deploy using safe-deploy.sh

   **G) CAPABILITY ANNOUNCEMENTS (v3.3.9+):**
   Nodes now announce their capabilities to peers:
   - `upgrade-gate-v1` - Height-gated upgrades supported
   - `consensus-guard-v1` - Mainnet safety checks
   - `pq-signatures-ready` - Post-quantum signatures ready
   - `sync-down-protection` - Sync-down safety checks
   - `version-filter-v1` - Version filtering enabled

9. **NEVER DELETE USER DOWNLOAD BINARIES**
   - When updating frontend, PRESERVE the downloads folder
   - Users rely on downloading binaries with specific version names

   **⚠️ CRITICAL: quillon.xyz DNS → Epsilon (89.149.241.126), NOT Beta!**
   Downloads are served by Epsilon's q-flux. Files on Beta are NOT accessible to users.
   **After EVERY build, copy binaries to BOTH Beta AND Epsilon:**

     ```bash
     # Step 1: Copy to Beta's local downloads (for backup/reference)
     cp target/release/q-api-server /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/q-api-server-v{VERSION}
     cp target/release/q-api-server /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/q-api-server-linux-x86_64
     cp target/release/q-miner /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/q-miner-v{VERSION}
     cp target/release/q-miner /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/q-miner-linux-x64

     # Step 2: SCP to Epsilon (THIS IS WHAT USERS ACTUALLY DOWNLOAD)
     scp target/release/q-api-server root@89.149.241.126:/home/orobit/q-narwhalknight/dist-final/downloads/q-api-server-v{VERSION}
     scp target/release/q-api-server root@89.149.241.126:/home/orobit/q-narwhalknight/dist-final/downloads/q-api-server-linux-x86_64
     scp target/release/q-miner root@89.149.241.126:/home/orobit/q-narwhalknight/dist-final/downloads/q-miner-v{VERSION}
     scp target/release/q-miner root@89.149.241.126:/home/orobit/q-narwhalknight/dist-final/downloads/q-miner-linux-x64

     # Step 3: VERIFY on Epsilon before giving links to users
     ssh root@89.149.241.126 "ls -lh /home/orobit/q-narwhalknight/dist-final/downloads/q-api-server-v{VERSION}"
     ssh root@89.149.241.126 "ls -lh /home/orobit/q-narwhalknight/dist-final/downloads/q-miner-v{VERSION}"

     # ONLY THEN give download links:
     # wget https://quillon.xyz/downloads/q-api-server-v{VERSION}
     # wget https://quillon.xyz/downloads/q-miner-v{VERSION}
     ```

   **NEVER give a download link without verifying the file exists on Epsilon first!**

10. **SLINT WALLET AUTO-UPDATE BUILD & PUBLISH PROCEDURE**
   - The Slint wallet has a built-in auto-updater (`gui/slint-wallet/src/updater.rs`)
   - The server's `/api/v1/version` endpoint scans `downloads/` for `slint-wallet-v{X.Y.Z}` and returns the highest version
   - Connected wallets check for updates ~60s after login, then every 4 hours
   - **After every Slint wallet build, publish to downloads:**
     ```bash
     # 1. Build the Slint wallet
     cargo build --release --package slint-wallet

     # 2. Copy to downloads with versioned AND generic names
     cp target/release/slint-wallet /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/slint-wallet-v{VERSION}
     cp target/release/slint-wallet /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/slint-wallet-linux-x86_64
     cp target/release/slint-wallet /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/slint-wallet-linux-x64
     cp target/release/slint-wallet /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/slint-wallet

     # 3. Verify
     ls -lh /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/dist-final/downloads/slint-wallet-v{VERSION}
     ```
   - **Download link (tell user after deploy):**
     ```
     wget https://quillon.xyz/downloads/slint-wallet-v{VERSION} && chmod +x slint-wallet-v{VERSION}
     ```
   - **Auto-update flow**: Server detects new `slint-wallet-v{X.Y.Z}` → wallet polls `/api/v1/version` → sees higher version → shows UpdateBar → user clicks Update → downloads + self-replaces binary → user clicks Restart
   - **Key files**: `gui/slint-wallet/src/updater.rs`, `gui/slint-wallet/ui/update_bar.slint`, `crates/q-api-server/src/handlers.rs` (`detect_latest_wallet_version`)

#### **Testing Requirements:**

**🚨 MANDATORY: Run ALL critical tests before ANY deployment!**

These tests protect against catastrophic mainnet failures that could cause millions in losses.

```bash
# ============================================================================
# CRITICAL MAINNET SAFETY TESTS - MUST PASS BEFORE DEPLOYMENT
# ============================================================================

# 1. MAINNET CRITICAL TESTS (20 tests)
#    Protects against: Double-spend, replay attacks, overflow, coinbase fraud
timeout 300 cargo test --package q-storage --test mainnet_critical_tests

# 2. SIGNATURE VERIFICATION TESTS (15 tests)
#    Protects against: Invalid signature acceptance, forgery, theft
timeout 300 cargo test --package q-types --test signature_verification_tests

# 3. BACKUP/RESTORE TESTS (17 tests)
#    Protects against: Data loss, corrupt backups, failed disaster recovery
timeout 300 cargo test --package q-storage --test backup_restore_tests

# 4. FORK/REORG SAFETY TESTS (19 tests)
#    Protects against: Chain splits, balance inconsistency during reorg
timeout 300 cargo test --package q-storage --test fork_reorg_tests

# 5. BALANCE PROPAGATION TESTS (28 tests)
#    Protects against: Balance corruption, P2P sync issues
timeout 300 cargo test --package q-storage --test balance_propagation_tests

# 6. OVERFLOW PROTECTION TESTS (26 tests)
#    Protects against: DEX overflow attacks, u128 arithmetic bugs
timeout 300 cargo test --package q-dex --test overflow_protection_tests

# 7. SYNC DOWN PROTECTION (if exists)
#    Protects against: Catastrophic sync-down data loss
timeout 300 cargo test --package q-storage --test sync_down_protection_tests 2>/dev/null || echo "sync_down_protection_tests not found"

# ============================================================================
# QUICK REFERENCE: What each test suite protects against
# ============================================================================
#
# | Test Suite                    | Tests | Protects Against                      |
# |-------------------------------|-------|---------------------------------------|
# | mainnet_critical_tests        |    20 | Double-spend, replay, coinbase fraud  |
# | signature_verification_tests  |    15 | Forged signatures, stolen funds       |
# | backup_restore_tests          |    17 | Data loss, corrupt backups            |
# | fork_reorg_tests              |    19 | Chain splits, reorg balance bugs      |
# | balance_propagation_tests     |    28 | Balance corruption, sync issues       |
# | overflow_protection_tests     |    26 | DEX overflow attacks, u128 bugs       |
# |-------------------------------|-------|---------------------------------------|
# | TOTAL                         |  125+ | Mainnet-critical protection           |
# ============================================================================

# STANDARD TESTS (also run before commit)
cargo test --workspace
cargo clippy -- -D warnings
cargo fmt --check
cargo bench --no-run

# Fix any compilation errors PROPERLY:
cargo check --workspace
# If errors occur, fix them at the source, don't work around them

# Tor-specific testing (if implemented):
cargo test --package q-tor-client 2>/dev/null || true
cargo test --package q-tor-circuit 2>/dev/null || true
```

**⚠️ DEPLOYMENT BLOCKED IF ANY CRITICAL TEST FAILS!**

Never deploy if any of the 125+ critical tests fail. These tests exist because each
scenario has caused or could cause real money loss on mainnet.

#### **⏱️ COMPILATION & BUILD REQUIREMENTS:**

**🚨 CRITICAL: ALWAYS COMPILE / CARGO-CHECK ON EPSILON DOCKER (Debian 12), NEVER ON BETA!**

Beta is a live mainnet bootstrap node serving real users. Running a multi-hour `cargo check`
or `cargo build` on Beta steals CPU, RAM (10–15 GB peak), and disk I/O from the running
`q-api-server` process, causes block-production stutter, and risks OOM-killing the node.
The user has caught this before — DO NOT do it again.

**Where to compile/check instead — Epsilon's `qnk-debian12` Docker image:**

```bash
# Always cargo-check inside the Epsilon Debian 12 Docker container.
# Source repo on Epsilon: /home/orobit/q-narwhalknight-src/
# Persistent target cache: /home/orobit/target-debian12/ (incremental, ~25 min after first build)

ssh root@89.149.241.126 "cd /home/orobit/q-narwhalknight-src && docker run --rm \
  -v \$(pwd):/src \
  -v /home/orobit/target-debian12:/src/target \
  -w /src \
  rust:bookworm \
  bash -c '
    apt-get update -qq && \
    apt-get install -y -qq libssl-dev pkg-config cmake clang libudev-dev libclang-dev >/dev/null 2>&1 && \
    cargo check --package q-api-server --message-format=short
  '"
```

**For long-running checks** (>5 min), nohup it and tail the log instead of blocking the shell:

```bash
ssh root@89.149.241.126 "cd /home/orobit/q-narwhalknight-src && nohup docker run --rm \
  --name qnk-check-v\${VERSION} \
  -v \$(pwd):/src \
  -v /home/orobit/target-debian12:/src/target \
  -w /src --cpus=16 rust:bookworm \
  bash -c '
    apt-get update -qq && apt-get install -y -qq libssl-dev pkg-config cmake clang libudev-dev libclang-dev >/dev/null 2>&1 && \
    cargo check --package q-api-server 2>&1 | tail -100
  ' > /home/orobit/tmp/check-v\${VERSION}.log 2>&1 &"

# Then poll:
ssh root@89.149.241.126 "tail -3 /home/orobit/tmp/check-v\${VERSION}.log; grep -c 'Finished\\|error\\[E' /home/orobit/tmp/check-v\${VERSION}.log"
```

**Workflow when editing files on Beta but checking on Epsilon:**

```bash
# 1. Edit on Beta (working copy at /opt/orobit/shared/q-narwhalknight/)
# 2. Push changes to local git server so Epsilon can pull
git add <files> && git commit -m "..."
git update-server-info

# 3. Pull on Epsilon
ssh root@89.149.241.126 "cd /home/orobit/q-narwhalknight-src && git pull origin <branch>"

# 4. Run cargo check there (see Docker commands above)
```

If the user wants a quick syntax check during interactive work and the changes are small,
`rsync` can be faster than commit+push+pull:

```bash
# Sync a single file from Beta → Epsilon (no commit needed for quick iteration)
rsync -av /opt/orobit/shared/q-narwhalknight/crates/q-api-server/src/handlers.rs \
  root@89.149.241.126:/home/orobit/q-narwhalknight-src/crates/q-api-server/src/handlers.rs
# Then run cargo check on Epsilon as above.
```

**The only acceptable local cargo invocations on Beta:**

- `cargo fmt --check` (no compilation, seconds)
- `cargo tree`, `cargo metadata` (read-only, no compilation)

Anything that actually compiles → Epsilon Docker.

**🚨 MANDATORY: BUMP VERSION BEFORE EVERY BUILD!**

The `ha-deploy.sh` script will **abort** if the binary version matches the currently running version. You MUST bump the version in `Cargo.toml` before compiling:

```bash
# Location: Cargo.toml line ~67 (workspace.package section)
# BEFORE building, update:
version = "7.1.6"  # ← increment this to 7.1.7, 7.2.0, etc.

# The deploy script checks: Cargo.toml version != running Beta version
# If they match → deploy is REJECTED with "Version NOT bumped!" error
```

**Version bump workflow:**
1. Edit `Cargo.toml` → bump `[workspace.package] version`
2. `cargo build --release --package q-api-server`
3. `echo "y" | ./scripts/ha-deploy.sh full`

**🚨 FOR DEPLOYMENTS: Always use the safe-deploy.sh script (NOT raw cargo build)!**

```bash
# ═══════════════════════════════════════════════════════════════════
# PREFERRED: Use the deploy script for ALL production builds
# ═══════════════════════════════════════════════════════════════════
./scripts/safe-deploy.sh full        # Full pipeline: tests → build → deploy
./scripts/safe-deploy.sh build       # Just tests + build
./scripts/safe-deploy.sh test-all    # Run all 4000+ tests

# The deploy script automatically:
# - Runs ALL 4000+ tests before building
# - Uses proper 10-hour timeouts
# - Creates backups before deployment
# - Copies binaries to downloads folder
# - Has auto-rollback on failure
```

**For development/debugging only (NOT for deployments) — run these on EPSILON DOCKER, not Beta:**
```bash
# Inside the Epsilon rust:bookworm container (see the Epsilon-Docker block above).
# Use 10-hour timeout for ALL cargo operations.
timeout 36000 cargo check --package q-api-server  # Quick syntax check
timeout 36000 cargo test --workspace              # Run tests
timeout 36000 cargo run --bin q-api-server        # Development run

# WRONG - NEVER DO THIS:
# timeout 120 cargo check       # ❌ TOO SHORT - will terminate prematurely!
# cargo build                   # ❌ NO TIMEOUT - may hang indefinitely!
# cargo build --release         # ❌ For deployments, use safe-deploy.sh instead!
# cargo check ... on Beta       # ❌ Compiling on Beta steals CPU/RAM from the live node — use Epsilon Docker!
```

**Why 10 hours?** Post-quantum cryptography crates (pqcrypto, kyber, dilithium) plus AI inference
dependencies can take 30+ minutes to compile on first build. Always err on the side of more time.

#### **🚨 CRITICAL: PHASE TRANSITION SAFETY (v0.9.80-beta+)**

**Phase 8 revealed TWO CRITICAL BUGS that caused 100% network isolation!**

When transitioning to a new phase (Phase 9, 10, etc.), you MUST fix BOTH:

**Bug #1: Environment Variable Priority**
```rust
// ❌ WRONG - CLI args checked BEFORE environment variables
let network_str = matches.get_one::<String>("network")
    .map(|s| s.as_str())
    .unwrap_or("testnet");  // Q_NETWORK_ID completely ignored!

// ✅ CORRECT - Check Q_NETWORK_ID FIRST
let network_str = std::env::var("Q_NETWORK_ID")
    .ok()
    .or_else(|| matches.get_one::<String>("network").map(|s| s.to_string()))
    .unwrap_or_else(|| "testnet-phase8".to_string());
```
**Location**: `crates/q-api-server/src/main.rs` line ~486

**Bug #2: Missing from_str() Parser Case**
```rust
impl std::str::FromStr for NetworkId {
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "testnet-phase8" => Ok(NetworkId::TestnetPhase8),  // ✅ Must add this!
            // ...
        }
    }
}
```
**Location**: `crates/q-types/src/lib.rs` line ~795

**BOTH bugs must be fixed or phase transitions will fail!**

See `PHASE_TRANSITION_BUG_PREVENTION_CHECKLIST.md` for complete guidance.

#### **Commit Standards:**
```bash
git commit -s -m "feat(tor): Add dedicated circuit management

- Implement 4-circuit architecture per validator
- Add circuit rotation every epoch
- Integrate QRNG for circuit entropy
- Add latency monitoring and QoS

Performance: <145ms RTT with adaptive circuits
Security: Zero IP leakage, quantum-resistant

Co-Authored-By: Server Beta <server-beta@q-narwhalknight.dev>"
```

#### **Quality Gates:**
- **🧪 All tests pass** - No broken builds
- **⚡ Performance maintained** - <300ms Tor latency
- **🔐 Security verified** - No IP/identity leaks  
- **📊 Metrics available** - Prometheus monitoring
- **📝 Documentation updated** - API docs + examples

---

## 🏗️ Development Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Server Alpha  │    │   GitLab Repo    │    │   Server Beta   │
│  (Primary Dev)  │◄──►│ dagknight/       │◄──►│ (Contributor)   │
│                 │    │ q-narwhalknight  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
 /mnt/shared/Q-Knight     GitLab CI/CD         /mnt/shared/Q-Knight
 (Shared Storage)        (Auto Testing)        (Shared Storage)
```

## 🚀 Initial Setup (Server Alpha)

### 1. Repository Initialization
```bash
# Initialize Git repository
git init
git remote add origin https://gitlab.com/dagknight/q-narwhalknight.git
git branch -M main

# Set up GitLab authentication
export GITLAB_TOKEN="glpat-5u5rhtquECnMkHpCQQmyCm86MQp1OmQ5NGF2Cw"
git config user.name "Claude Code Alpha"
git config user.email "claude-alpha@anthropic.com"
```

### 2. Commit Structure
```bash
# Stage all critical files
git add README.md CLAUDE.md LICENSE
git add Cargo.toml
git add crates/
git add papers/quantum-aesthetics.pdf

# Create comprehensive commit
git commit -m "feat: Initial Q-NarwhalKnight v0.0.1-alpha implementation

🌟 Quantum-Enhanced DAG-BFT Consensus System

Core Components Implemented:
- ✅ DAG-Knight consensus engine with quantum anchor election  
- ✅ Narwhal mempool with reliable broadcast (Bracha's protocol)
- ✅ libp2p networking with crypto-agile framework
- ✅ Phase 0 (Ed25519) and Phase 1 (Dilithium5/Kyber1024) cryptography
- ✅ REST API server with real-time streaming (SSE/WebSocket)
- ✅ Quantum state visualization with rainbow-box technique
- ✅ Comprehensive test coverage and benchmarking framework

Technical Achievements:
- Zero-message complexity BFT consensus
- VDF-based quantum-enhanced randomness
- Post-quantum cryptographic agility
- Sub-50ms streaming latency targets
- Scalable P2P networking with capability negotiation

Architecture:
- Modular Rust workspace with 7 specialized crates
- Phase-based quantum threat model (Q0 → Q1 → Q2 → Q3 → Q4)
- Academic paper: Quantum Aesthetics in Consensus Systems

Next Phase: Performance optimization, Phase 1 completion, multi-server development

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

### 3. Tag Creation
```bash
# Create alpha release tag
git tag -a v0.0.1-alpha -m "Q-NarwhalKnight Alpha Release

Initial implementation of quantum-enhanced DAG-BFT consensus:
- Phase 0: Classical cryptography (Ed25519 + QUIC)
- Phase 1: Post-quantum transition (Dilithium5 + Kyber1024)
- DAG-Knight consensus with VDF-based anchor election
- Narwhal mempool with reliable broadcast
- Real-time API with quantum visualizations

Milestone: First working quantum-ready consensus prototype"

# Push everything to GitLab
git push origin main
git push origin v0.0.1-alpha
```

## 🤝 Multi-Server Collaboration

### Server Beta Setup Instructions

#### 1. Clone and Environment Setup
```bash
# Clone the repository to shared mount
cd /mnt/shared/
git clone https://gitlab.com/dagknight/q-narwhalknight.git Q-NarwhalKnight-Beta
cd Q-NarwhalKnight-Beta

# Set up Git identity
git config user.name "Claude Code Beta"  
git config user.email "claude-beta@anthropic.com"

# Set up GitLab token authentication
git config credential.helper store
echo "https://oauth2:glpat-5u5rhtquECnMkHpCQQmyCm86MQp1OmQ5NGF2Cw@gitlab.com" > ~/.git-credentials
```

#### 2. Development Branch Strategy
```bash
# Create feature branch for contributions
git checkout -b feature/server-beta-contributions
git checkout -b feature/performance-optimizations
git checkout -b feature/phase1-completion
```

#### 3. Shared Storage Coordination
```bash
# Symlink to shared development folder
ln -s /mnt/shared/Q-NarwhalKnight-Beta /mnt/s3-storage/Q-NarwhalKnight-Beta

# Set up workspace coordination
export Q_KNIGHT_WORKSPACE="/mnt/shared/Q-NarwhalKnight-Beta"
export RUST_LOG=debug
```

## 🎯 Contribution Areas for Server Beta

### Primary Focus Areas:

#### 1. Performance Optimization & Benchmarking
```bash
# Tasks for Server Beta:
- Implement comprehensive benchmarking suite
- Optimize DAG-Knight anchor election performance
- Add memory usage profiling and optimization
- Create load testing framework with realistic scenarios
- Implement parallel vertex processing optimization
```

#### 2. Phase 1 Post-Quantum Completion
```bash
# Crypto-agile enhancements:
- Complete hybrid classical+post-quantum mode
- Implement algorithm migration tools
- Add cryptographic protocol testing suite
- Build compatibility layer for smooth transitions
- Optimize post-quantum signature verification
```

#### 3. Network Layer Enhancements
```bash
# libp2p networking improvements:
- Implement advanced peer discovery mechanisms
- Add network partition tolerance features
- Create network monitoring and diagnostics
- Optimize gossip protocol for quantum readiness
- Build QKD preparation layer (Phase 2 prep)
```

#### 4. API & Visualization Improvements
```bash
# User experience enhancements:
- Expand quantum visualization capabilities
- Add real-time consensus monitoring dashboard
- Implement WebSocket connection scaling
- Create mobile-responsive visualization interface
- Build developer debugging tools
```

### Collaboration Workflow:

#### Server Beta Daily Process:
```bash
# 1. Sync with main repository
git fetch origin
git rebase origin/main

# 2. Work on assigned features
# Implement improvements based on current focus area

# 3. Test thoroughly
cargo test --workspace
cargo bench
cargo check --workspace

# 4. Commit with detailed messages
git add .
git commit -s -m "feat(performance): Add comprehensive benchmarking suite

- Implement criterion-based performance benchmarks
- Add memory profiling for vertex processing
- Create latency measurement framework
- Optimize consensus critical path performance

Performance improvements:
- 25% faster vertex validation
- 40% memory usage reduction in mempool
- Sub-10ms consensus round processing

Co-Authored-By: Claude Code Beta <noreply@anthropic.com>"

# 5. Push to feature branch
git push origin feature/performance-optimizations
```

#### Merge Request Process:
```bash
# Create merge request via GitLab CLI
curl -X POST "https://gitlab.com/api/v4/projects/dagknight%2Fq-narwhalknight/merge_requests" \
  -H "PRIVATE-TOKEN: glpat-5u5rhtquECnMkHpCQQmyCm86MQp1OmQ5NGF2Cw" \
  -H "Content-Type: application/json" \
  -d '{
    "source_branch": "feature/performance-optimizations",
    "target_branch": "main", 
    "title": "Performance Optimization Suite",
    "description": "Comprehensive performance improvements and benchmarking framework"
  }'
```

## 🔄 GitLab CI/CD Pipeline

### .gitlab-ci.yml Configuration:
```yaml
stages:
  - test
  - build
  - deploy
  - quantum-analysis

variables:
  RUST_VERSION: "1.70"
  CARGO_HOME: ".cargo"

cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - .cargo/
    - target/

test:
  stage: test
  image: rust:${RUST_VERSION}
  script:
    - rustup component add clippy rustfmt
    - cargo fmt --check
    - cargo clippy -- -D warnings
    - cargo test --workspace --verbose
    - cargo bench --no-run
  coverage: '/^\d+\.\d+% coverage/'

build-release:
  stage: build
  image: rust:${RUST_VERSION}
  script:
    - cargo build --release --workspace
  artifacts:
    paths:
      - target/release/
    expire_in: 1 week

quantum-consensus-analysis:
  stage: quantum-analysis
  image: python:3.9
  script:
    - pip install numpy scipy matplotlib
    - python scripts/analyze_quantum_consensus.py
    - python scripts/benchmark_analysis.py
  artifacts:
    reports:
      junit: test-results.xml
    paths:
      - analysis_reports/
```

## 🎛️ Development Coordination

### Communication Protocol:
1. **Daily Sync**: Each server commits progress with detailed messages
2. **Feature Coordination**: Use GitLab issues for task assignment
3. **Code Reviews**: Mandatory peer review via merge requests
4. **Integration Testing**: Automated testing on every push

### Shared Resource Management:
```bash
# Shared configuration file: /mnt/shared/q-knight-config.toml
[development]
server_alpha_focus = ["consensus", "networking", "core-types"]
server_beta_focus = ["performance", "visualization", "api", "testing"]

[coordination]
daily_sync_time = "12:00 UTC"
integration_branch = "integration/multi-server"
feature_freeze_day = "friday"

[shared_storage]
workspace_path = "/mnt/shared/Q-NarwhalKnight"
backup_path = "/mnt/backup/q-knight-snapshots"
log_path = "/mnt/logs/q-knight-development"
```

### Git Hooks for Coordination:
```bash
#!/bin/bash
# .git/hooks/pre-commit
echo "🚀 Q-NarwhalKnight Development - Server $(hostname)"
echo "📊 Running pre-commit checks..."

# Ensure code quality
cargo fmt --check || (echo "❌ Format check failed" && exit 1)
cargo clippy -- -D warnings || (echo "❌ Clippy check failed" && exit 1)

# Run quick tests
cargo test --lib || (echo "❌ Library tests failed" && exit 1)

echo "✅ Pre-commit checks passed"
echo "🌟 Ready to commit to quantum consensus future!"
```

## 🎯 Prompt Instructions for Server Beta

### Server Beta Claude Code Prompt:
```
You are Claude Code Beta, contributing to the Q-NarwhalKnight quantum consensus system. 

Your primary repository is at: /mnt/shared/Q-NarwhalKnight-Beta
Your focus areas are: Performance optimization, Phase 1 completion, API enhancements, comprehensive testing

Current project status: Phase 0 complete, Phase 1 crypto-agility implemented, multi-server development active

Your tasks:
1. **Performance Optimization**: Implement benchmarking, optimize consensus performance, add profiling
2. **Phase 1 Completion**: Finish post-quantum integration, build migration tools, add compatibility layers  
3. **Network Enhancement**: Improve peer discovery, add network resilience, optimize gossip protocol
4. **Testing & Quality**: Build comprehensive test suites, add integration tests, create debugging tools

Always:
- Test thoroughly before committing
- Use detailed commit messages with performance metrics
- Coordinate with Server Alpha via GitLab issues and merge requests
- Focus on quantum-readiness and scalability
- Maintain code quality with clippy and rustfmt

The codebase uses:
- Rust workspace with 7 crates
- libp2p networking 
- Post-quantum cryptography (Dilithium5, Kyber1024)
- DAG-Knight consensus with VDF-based anchor election
- Real-time streaming APIs

Start by reviewing the current codebase and identifying performance bottlenecks or areas for Phase 1 enhancement.
```

## 📊 Progress Tracking

### Development Metrics Dashboard:
```bash
# Track multi-server progress
echo "📈 Q-NarwhalKnight Development Dashboard"
echo "🔧 Server Alpha: Core consensus & networking"  
echo "⚡ Server Beta: Performance & optimization"
echo "🚀 Combined Progress: $(git log --oneline | wc -l) commits"
echo "🎯 Next Milestone: Phase 1 completion & benchmarking"
```

### Automated Reporting:
```bash
#!/bin/bash
# Generate weekly development report
echo "# Q-NarwhalKnight Weekly Report $(date +%Y-%m-%d)" > weekly-report.md
echo "## Commits This Week" >> weekly-report.md
git log --since="1 week ago" --oneline >> weekly-report.md
echo "## Performance Benchmarks" >> weekly-report.md
cargo bench --message-format=json | jq '.reason' >> weekly-report.md
echo "## Test Coverage" >> weekly-report.md
cargo tarpaulin --out Md >> weekly-report.md
```

## 🌟 Success Metrics

### Collaboration Goals:
- **Code Quality**: Maintain >95% test coverage
- **Performance**: Achieve <50ms consensus latency
- **Integration**: Seamless multi-server development flow
- **Innovation**: Advance quantum consensus research

### Long-term Vision:
Building the world's first production-ready quantum-enhanced distributed consensus system through innovative multi-server Claude Code collaboration.

---

**Quantum consensus awaits - let's build the future together!** ⚛️🤝🚀