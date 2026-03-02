# Epsilon Supernode Setup Guide

## Server Specs
- **2x Xeon Gold 5118** (24 cores / 48 threads)
- **64GB DDR4 RAM**
- **2TB NVMe** (OS + blockchain data)
- **88TB HDD** (backups, archives)
- **10 Gbit network** (10x faster than existing 1Gbit servers)

## What's Already Prepared (in codebase)

### 1. BandwidthTier Enum (`crates/q-network/src/unified_network_manager.rs`)
- `BandwidthTier::Supernode` (>=5000 Mbps) with 10x sync boost
- `max_serve_chunk_size()` returns 5000 blocks for supernodes (vs 500 for standard)
- `SUPERNODE_PEER_IDS` lazy_static from `Q_SUPERNODE_PEERS` env var

### 2. Bootstrap Peer Lists (same file)
- Epsilon placeholder as FIRST entry in `HARDCODED_BOOTSTRAP_PEERS` (commented)
- Epsilon placeholder as FIRST entry in `BOOTSTRAP_HTTP_ENDPOINTS` (commented)
- Ordering: Epsilon 10Gbit -> Gamma 1Gbit -> Delta 1Gbit -> Beta 100Mbit

### 3. Adaptive Sync (`crates/q-storage/src/turbo_sync.rs`)
- `supernode_peers` config field from `Q_SUPERNODE_PEERS` env var
- Tiered boost in `apollo_select_peer()`: 10x supernode, 3x preferred, 1x default
- `get_peer_chunk_size()` method: 5000 for supernodes, 1000 for standard, 500 for fallback
- When supernode detected among qualified peers, chunk size cap raised to 5000

### 4. Handshake (`crates/q-network/src/protocol_handshake.rs`)
- `"supernode"` feature announced when `Q_BANDWIDTH_MBPS >= 5000`

### 5. ServerRole Enum (`crates/q-network/src/handshake.rs`)
- `Delta` and `Epsilon` variants added

### 6. Peer Stats (`crates/q-storage/src/peer_momentum.rs`)
- `bandwidth_tier: String` field in `PeerStats` ("SUPERNODE"/"STANDARD"/"FALLBACK"/"UNKNOWN")

### 7. Nginx (`/etc/nginx/sites-available/quillon.xyz`)
- Epsilon placeholder with `weight=20` in all upstream groups (commented, marked `down`)

### 8. Frontend (`gui/quantum-wallet/src/services/api.ts`)
- Epsilon placeholder in `API_SERVERS` failover list (commented)

---

## What To Do When We Get Root SSH Access

### Step 1: OS Setup (5 min)
```bash
ssh root@<EPSILON_IP>

# Update system
apt update && apt upgrade -y

# Install build essentials
apt install -y build-essential curl git pkg-config libssl-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Create working directory
mkdir -p /opt/orobit/shared/q-narwhalknight
```

### Step 2: Configure Swap (even with 64GB, good safety margin)
```bash
fallocate -l 8G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

### Step 3: SCP Binary From Beta (2 min)
```bash
# From Beta:
scp /opt/orobit/shared/q-narwhalknight/target/release/q-api-server root@<EPSILON_IP>:/opt/orobit/shared/q-narwhalknight/q-api-server
```

### Step 4: Create Systemd Service
```bash
cat > /etc/systemd/system/q-api-server.service << 'EOF'
[Unit]
Description=Q-NarwhalKnight API Server (Epsilon Supernode)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/orobit/shared/q-narwhalknight
ExecStart=/opt/orobit/shared/q-narwhalknight/q-api-server --port 8080
Restart=always
RestartSec=5
LimitNOFILE=65535

# Epsilon Supernode Configuration
Environment="Q_NETWORK_ID=mainnet-genesis"
Environment="Q_DB_PATH=./data-mainnet-genesis"
Environment="Q_BANDWIDTH_MBPS=10000"
Environment="ROCKSDB_BLOCK_CACHE_MB=16384"
Environment="Q_TURBO_PARALLEL_STREAMS=64"
Environment="Q_TURBO_CHUNK_SIZE=5000"
Environment="Q_TURBO_CHUNK_TIMEOUT_SECS=30"
Environment="Q_SYNC_RAYON_THREADS=16"
Environment="Q_TOR_BOOTSTRAP_TIMEOUT=5"

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable q-api-server
```

### Step 5: First Boot & Record PeerId
```bash
systemctl start q-api-server

# Wait 10 seconds for identity generation
sleep 10

# Capture the PeerId
journalctl -u q-api-server --since "1 minute ago" | grep "Local peer id"
# Output: Local peer id: 12D3KooW...

# SAVE THIS PEER ID — needed for bootstrap lists
```

### Step 6: Fill In Placeholders (on Beta) — DONE (v8.7.4, 2026-03-02)
Epsilon PeerId: `12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM`
Epsilon IP: `89.149.241.126`

All placeholders filled:
- `crates/q-network/src/unified_network_manager.rs` — HARDCODED_BOOTSTRAP_PEERS (Epsilon FIRST), BOOTSTRAP_HTTP_ENDPOINTS (Epsilon FIRST), HARDCODED_BOOTSTRAP_PEER (Epsilon)
- `crates/q-storage/src/turbo_sync.rs` — Hardcoded Epsilon peer ID in `apollo_select_peer()` (10x boost) and `get_peer_chunk_size()` (5000 blocks)
- `gui/quantum-wallet/src/services/api.ts` — API_SERVERS (Epsilon direct as #2 after nginx LB)
- `/etc/nginx/sites-available/quillon.xyz` (Beta) — All 3 upstreams have Epsilon IP with weight=20
- `CLAUDE.md` — Bootstrap peer IDs updated (Epsilon first)

### Step 7: Set Q_SUPERNODE_PEERS on Other Servers — DONE via hardcoded peer ID
Epsilon's peer ID is now hardcoded in `turbo_sync.rs` for 10x boost + 5000-block chunks,
so `Q_SUPERNODE_PEERS` env var is no longer required (but still works as an override).

For maximum effect, optionally add to Beta/Gamma/Delta service files:
```bash
Environment="Q_SUPERNODE_PEERS=12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM"
```

### Step 8: Wait For Sync (monitor progress)
```bash
# On Epsilon:
journalctl -u q-api-server -f | grep -E "TURBO|height|BPS|SYNC"

# Expected: ~2000+ BPS from Delta/Gamma (10Gbit download)
# Full sync of 4.6M blocks should take ~30-40 minutes
```

### Step 9: Enable In Nginx (ONLY after fully synced!)
```bash
# On Beta — edit /etc/nginx/sites-available/quillon.xyz
# Remove 'down' from all 3 Epsilon lines

nginx -t && systemctl reload nginx
```

### Step 10: Rebuild & Deploy v8.6.3+
```bash
# On Beta — fill in placeholders, rebuild
cargo build --release --package q-api-server
echo "y" | ./scripts/ha-deploy.sh full -y
```

---

## Verification Checklist

1. `journalctl -u q-api-server | grep "SUPERNODE"` — should see supernode feature announced
2. Fresh test node logs show Epsilon selected first with 10x boost
3. Chunk size logs: 5000 blocks/chunk from Epsilon vs 500 from Gamma
4. Kill Epsilon -> fallback to Gamma/Delta within 30 seconds (success_rate EMA drops)
5. `curl http://<EPSILON_IP>:8080/api/v1/status` returns correct height

## Epsilon Env Vars Summary

| Variable | Value | Purpose |
|----------|-------|---------|
| `Q_BANDWIDTH_MBPS` | `10000` | Announce 10Gbit to peers |
| `ROCKSDB_BLOCK_CACHE_MB` | `16384` | 16GB block cache (of 64GB RAM) |
| `Q_TURBO_PARALLEL_STREAMS` | `64` | Max parallel sync streams |
| `Q_TURBO_CHUNK_SIZE` | `5000` | Serve 5000 blocks per chunk |
| `Q_SYNC_RAYON_THREADS` | `16` | 16 CPU threads for sync verification |
| `Q_TOR_BOOTSTRAP_TIMEOUT` | `5` | Skip long Tor wait |
