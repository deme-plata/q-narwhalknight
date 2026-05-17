# Transitioning to Mainnet 2026.1

## What Changed?

Q-NarwhalKnight has transitioned from `mainnet2026` to `mainnet2026.1` — a clean relaunch with:
- Fresh genesis block (height 0)
- Fixed emission controller (no more 2x overshoot)
- Fixed RocksDB batch write failures
- Cleaned testnet contract contamination
- New bootstrap peer IDs

**Your old testnet/mainnet2026 data is NOT compatible.** You must start fresh.

---

## Quick Start (New Install)

```bash
# Download the latest binary
wget https://quillon.xyz/downloads/q-api-server-v7.1.3-mainnet2026.1
chmod +x q-api-server-v7.1.3-mainnet2026.1

# Create data directory
mkdir -p data-mainnet2026.1

# Start the node
./q-api-server-v7.1.3-mainnet2026.1 \
  --port 8080 \
  --p2p-port 9001 \
  --data-dir ./data-mainnet2026.1 \
  --network mainnet2026.1
```

---

## Upgrade from Previous Version

### Step 1: Stop your current node

```bash
# If running as systemd service:
sudo systemctl stop q-api-server

# If running manually:
# Press Ctrl+C or kill the process
```

### Step 2: Back up old data (optional)

```bash
# Rename your old data directory (keeps it as backup)
mv data-mainnet2026 data-mainnet2026-old
# Or if you were on testnet:
mv data-mine23 data-mine23-old
```

### Step 3: Download the new binary

```bash
wget https://quillon.xyz/downloads/q-api-server-v7.1.3-mainnet2026.1
chmod +x q-api-server-v7.1.3-mainnet2026.1
```

### Step 4: Create fresh data directory

```bash
mkdir -p data-mainnet2026.1
```

### Step 5: Start the node

```bash
./q-api-server-v7.1.3-mainnet2026.1 \
  --port 8080 \
  --p2p-port 9001 \
  --data-dir ./data-mainnet2026.1 \
  --network mainnet2026.1
```

### Step 6: Update your miner config (if mining)

Point your miner to the same node as before. The mining API hasn't changed — only the network ID and bootstrap peers are different.

---

## Systemd Service Setup

If you run your node as a systemd service:

```bash
sudo tee /etc/systemd/system/q-api-server.service > /dev/null << 'EOF'
[Unit]
Description=Q-NarwhalKnight Node (Mainnet 2026.1)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/orobit/shared/q-narwhalknight
ExecStart=/opt/orobit/shared/q-narwhalknight/q-api-server-v7.1.3-mainnet2026.1 \
  --port 8080 \
  --p2p-port 9001 \
  --data-dir ./data-mainnet2026.1 \
  --network mainnet2026.1
Restart=on-failure
RestartSec=10
Environment=RUST_LOG=info
Environment=Q_NETWORK_ID=mainnet2026.1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable q-api-server
sudo systemctl start q-api-server

# Check status:
sudo journalctl -u q-api-server -f
```

---

## Bootstrap Peers

Your node will automatically connect to these bootstrap peers:

| Server | IP | Peer ID |
|--------|-----|---------|
| Beta (Primary) | 185.182.185.227 | `12D3KooWBHTC9FhwwXmvH7YA17YHTLdcxbtLWg2U5xEtxSeqX7jc` |
| Gamma (Backup) | 109.205.176.60 | `12D3KooWFqPX9TkvF43eyDeH9wwxYTSfnBn8AobLJeA7xRnmpPcv` |
| Delta | 5.79.79.158 | `12D3KooWQZZAyLA4VQmwNozCBTZXXoWfvKE86ebbaPhSKu6XVmJJ` |
| Alpha | 161.35.219.10 | `12D3KooWPwin4nJcU9PzsxNgUVXj5e6zDnACr84H7RZ1XzmnARsY` |

These are hardcoded in the binary. No manual configuration needed.

---

## Verify Your Node

After starting, check that your node is syncing:

```bash
curl -s http://localhost:8080/api/v1/status | python3 -m json.tool
```

You should see:
- `network_id`: `"mainnet2026.1"`
- `current_height`: increasing over time
- `connected_peers`: 1 or more

---

## FAQ

**Q: Will I lose my balance?**
A: Mainnet 2026.1 is a fresh start. All balances from mainnet2026/testnet are reset. Start mining to earn QUG on the new network.

**Q: Can I keep my old wallet address?**
A: Yes! Your wallet address (derived from your private key) works on any network. Only the blockchain data is reset.

**Q: Do I need to change my miner?**
A: No. The same q-miner binary works. Just make sure it points to your node running mainnet2026.1.

**Q: What if I see "incompatible network" errors?**
A: You're still running the old binary or connecting to an old-network node. Download v7.1.3 and use a fresh data directory.

**Q: Can I run both old and new nodes?**
A: Yes, use different ports and data directories. But the old network is deprecated.
