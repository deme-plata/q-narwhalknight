# @everyone MAINNET GENESIS IS LIVE

**Q-NarwhalKnight Mainnet Genesis launched today, February 22, 2026.**

The production mainnet is running, blocks are being produced, and mining is open to everyone.

---

## What's New

- **Network**: `mainnet-genesis` (Chain ID: 1000)
- **Version**: v8.1.6
- **Consensus**: DAG-Knight BFT with quantum-resistant cryptography
- **Emission**: 2,625,000 QUG/year (Era 0), 21M max supply, 4-year halving (Bitcoin-style)
- **Block Reward**: ~0.083 QUG/block (adaptive difficulty)
- **3 Bootstrap Nodes**: Beta (EU), Gamma (EU), Delta (EU) — all online and synced

---

## How to Start Mining

### Option 1: Download & Run (Linux)

```bash
wget https://quillon.xyz/downloads/q-api-server-v8.1.6
chmod +x q-api-server-v8.1.6
./q-api-server-v8.1.6 --port 8080 --tui --admin-wallet YOUR_WALLET_ADDRESS
```

Your node will:
- Auto-discover peers via P2P gossipsub
- WarpSync to catch up with the network in minutes
- Start mining automatically once synced

### Option 2: Use the Web Wallet

Visit **https://quillon.xyz** — the wallet connects directly to the network via WebSocket. You can mine from your browser.

---

## Important Notes

- **Fresh start**: Mainnet Genesis is a brand new chain. All previous testnet/rehearsal balances do not carry over.
- **Old binaries won't connect**: The protocol handshake rejects mismatched network IDs. You must use v8.1.6+.
- **Your old data is safe**: Previous data directories (`data-mainnet2026.1.3/`, etc.) are untouched. The new chain uses `data-mainnet-genesis/`.
- **Admin panel**: Use `--admin-wallet YOUR_WALLET` flag to enable the Node Admin dashboard with sync stats, peer info, and fee earnings.

---

## For Existing Node Operators

If you were running a previous network:

```bash
# Stop old node
pkill -f q-api-server

# Download new binary
wget https://quillon.xyz/downloads/q-api-server-v8.1.6
chmod +x q-api-server-v8.1.6

# Start on mainnet-genesis (auto-creates data-mainnet-genesis/)
./q-api-server-v8.1.6 --port 8080 --tui --admin-wallet YOUR_WALLET_ADDRESS
```

No migration needed. Everyone starts fresh on the genesis chain.

---

## Network Status

- Block height: growing (mining is active with 10+ miners already)
- Bootstrap nodes: 3/3 online
- P2P: gossipsub + Kademlia DHT, automatic peer discovery
- Download: https://quillon.xyz/downloads/q-api-server-v8.1.6

Welcome to Mainnet Genesis. Let's build.
