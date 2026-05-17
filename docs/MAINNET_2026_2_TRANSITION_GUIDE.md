# Transitioning to Mainnet 2026.2 (Feb 22, 2026 12:00 UTC)

## What Changed?

Q-NarwhalKnight is transitioning from `mainnet2026.1.1` to `mainnet2026.2` — the official full mainnet launch with:

- Fresh genesis block at timestamp `1771761600` (Feb 22 2026 12:00 UTC)
- New chain ID: `1000` (was `999`)
- Network ID: `mainnet2026.2`
- Emission: 2,625,000 QUG/year (Era 0), 21M max supply, 4-year halving
- Full DeFi stack: DEX, QUGUSD stablecoin, Quillon Bank, QNK10 Index Fund
- Fixed block storage (QRAW format — no more LZ4 compression bugs)
- Fixed emission controller (34× overshoot corrected)

**Your mainnet2026.1.1 data is NOT compatible.** Start fresh.

---

## ⚠️ WARNING: Rogue Pre-Launch Nodes

There is at least one rogue node already running `Q_NETWORK_ID=mainnet2026.2` before the
official launch date. This node has peer ID `12D3KooWDspLtKpQwSxqTNZXKAFZcRdSXAVxPEyrCLjLDyMVZday`
and claims to be at height ~311,755.

**This is NOT a valid mainnet2026.2 node.** Its blocks:
- Have timestamps BEFORE the genesis timestamp (Feb 22 12:00 UTC)
- Will be **automatically rejected** by the genesis filter in v7.3.6+
- Cannot contaminate the official launch chain

**Do NOT sync from this peer.** If you download the official v7.3.6+ binary and launch
on Feb 22 or later, you will automatically be on the correct chain.

---

## Quick Start (Feb 22 or later)

```bash
# 1. Download the v7.3.6+ binary (available at launch)
wget https://quillon.xyz/downloads/q-api-server-v7.3.6
chmod +x q-api-server-v7.3.6

# 2. Create fresh data directory (IMPORTANT: must be new!)
mkdir -p data-mainnet2026.2

# 3. Start the node (set env vars, NOT CLI flags)
Q_NETWORK_ID=mainnet2026.2 \
Q_DB_PATH=./data-mainnet2026.2 \
./q-api-server-v7.3.6 --port 8080
```

> **Why environment variables?** The binary checks `Q_NETWORK_ID` before CLI args.
> Using `--network mainnet2026.2` does NOT work reliably (see Bug #2 in the checklist).

---

## Upgrade from mainnet2026.1.1

### Step 1: Wait for Feb 22 12:00 UTC

The genesis block cannot be produced before this timestamp. Starting your node early
is fine — it will wait and then auto-mine the genesis block at the correct time.

### Step 2: Stop your current node

```bash
# If running as systemd service (ALWAYS use systemctl, never kill -9):
sudo systemctl stop q-api-server

# If running manually:
# Ctrl+C or kill the process
```

### Step 3: Back up old data (recommended)

```bash
mv data-mainnet2026.1.1 data-mainnet2026.1.1-backup
# Or delete it to free disk space:
# rm -rf data-mainnet2026.1.1
```

### Step 4: Download the new binary

```bash
wget https://quillon.xyz/downloads/q-api-server-v7.3.6
chmod +x q-api-server-v7.3.6
```

### Step 5: Create fresh data directory

```bash
mkdir -p data-mainnet2026.2
```

Do NOT copy files from the old data directory. Start completely fresh.

### Step 6: Start the node

```bash
Q_NETWORK_ID=mainnet2026.2 \
Q_DB_PATH=./data-mainnet2026.2 \
./q-api-server-v7.3.6 --port 8080
```

### Step 7: Update your miner config (if mining)

Point your miner to the same node as before. The mining API hasn't changed.

---

## Systemd Service Setup

```bash
sudo tee /etc/systemd/system/q-api-server.service > /dev/null << 'EOF'
[Unit]
Description=Q-NarwhalKnight Node (Mainnet 2026.2)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/user/q-narwhalknight
ExecStart=/home/user/q-narwhalknight/q-api-server-v7.3.6 --port 8080
Restart=on-failure
RestartSec=10
Environment=RUST_LOG=info
Environment=Q_NETWORK_ID=mainnet2026.2
Environment=Q_DB_PATH=./data-mainnet2026.2

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable q-api-server
sudo systemctl start q-api-server

# Monitor:
sudo journalctl -u q-api-server -f
```

---

## Bootstrap Peers

Your node will automatically connect to the official bootstrap nodes.
These peer IDs will be published on launch day (Feb 22 12:00 UTC) after the
genesis block is produced. Check https://quillon.xyz for the updated IDs.

> **Note**: The old mainnet2026.1.1 peer IDs will NOT work on mainnet2026.2.

| Server | IP | Role |
|--------|-----|------|
| Beta | 185.182.185.227 | Primary bootstrap |
| Gamma | 109.205.176.60 | Backup bootstrap |
| Delta | 5.79.79.158 | Secondary bootstrap |

---

## Verify Your Node

After starting, confirm your node is on the right network:

```bash
curl -s http://localhost:8080/api/v1/status | python3 -m json.tool
```

Look for:
- `"network_id": "mainnet2026.2"` ✅
- `"current_height"`: increasing over time ✅
- `"connected_peers"`: 1 or more ✅

If you see `mainnet2026.1.1` — you're still running the old binary.

---

## FAQ

**Q: Will I lose my balance?**
A: Yes. Mainnet 2026.2 is a fresh blockchain. Earn QUG by mining.

**Q: Can I keep my wallet address?**
A: Yes! Your wallet address (derived from your private key) works on any network.

**Q: What if I connect to a node claiming height 300,000+ before Feb 22?**
A: That's the rogue pre-launch node. v7.3.6+ automatically rejects its blocks
(genesis timestamp filter). You cannot be contaminated if using the official binary.

**Q: Can I start my node before Feb 22?**
A: Yes. The node will start, connect to peers, and wait until 12:00 UTC on Feb 22
before producing the genesis block.

**Q: What if I see "waiting for genesis" in the logs?**
A: Normal behavior before Feb 22 12:00 UTC. The node is correctly waiting for launch time.

**Q: What is the correct data directory name?**
A: `data-mainnet2026.2` — this is the official name hardcoded in the server configs.

**Q: What happened to the old `data-mainnet2026.2` folder if I ran a canary?**
A: Delete it and start fresh. The canary accumulated corrupt or contaminated blocks.
```bash
rm -rf data-mainnet2026.2
mkdir data-mainnet2026.2
```
