# Docker P2P Setup Guide - Q-NarwhalKnight v0.3.1

## Problem: Docker Container Can't Sync Blocks

When running Q-NarwhalKnight in Docker, the container needs **two ports** to work properly:
- **Port 8080**: HTTP API (for mining, transactions, status)
- **Port 8081**: P2P libp2p (for block propagation, gossipsub, peer discovery)

If you only expose port 8080, the container can't receive blocks from the P2P network!

## Solution: Expose Both Ports

### Option 1: Host Networking (Recommended - Fastest)

This gives the container direct access to the host's network:

```bash
# Download latest version
wget https://quillon.xyz/downloads/q-api-server-v0.3.1-beta -O q-api-server

# Create Dockerfile
cat > Dockerfile <<EOF
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY q-api-server ./q-api-server
RUN chmod +x q-api-server
CMD ["./q-api-server"]
EOF

# Build image
docker build -t quillon-api:v0.3.1 .

# Run with host networking
docker run -d \
  --name quillon-node \
  --network host \
  quillon-api:v0.3.1

# Access:
# HTTP API: http://localhost:8080
# P2P: Direct connection on host's network
```

### Option 2: Port Mapping (Alternative)

If you can't use host networking:

```bash
# Same download and Dockerfile as above, then:

# Run with BOTH ports mapped
docker run -d \
  --name quillon-node \
  -p 9080:8080 \
  -p 9081:8081 \
  quillon-api:v0.3.1

# Access:
# HTTP API: http://YOUR_IP:9080
# P2P: tcp://YOUR_IP:9081
```

### Option 3: Custom P2P Port

If you need a different port:

```bash
# Run with custom P2P port
docker run -d \
  --name quillon-node \
  -p 9080:8080 \
  -p 9001:8081 \
  -e Q_P2P_PORT=8081 \
  quillon-api:v0.3.1

# The container listens on 8081 internally
# Your firewall sees it on 9001 externally
```

## How It Works

### With Both Ports Exposed ✅

```
┌─────────────────────────────────────────┐
│  Bootstrap Node (localhost:8080)        │
│  Height: 108,000                         │
│  Producing blocks...                     │
└───────────────┬─────────────────────────┘
                │
                │ libp2p gossipsub
                │ /qnk/testnet/blocks
                │
                ▼
┌─────────────────────────────────────────┐
│  Docker Container (161.35.219.10:9080)  │
│  Port 8080 → HTTP API                   │
│  Port 8081 → P2P libp2p ✅              │
│                                          │
│  📥 Receives blocks via gossipsub        │
│  📈 Syncs to height 108,000              │
│  ⏸️  Pauses mining until synced          │
│  ✅ Resumes mining after sync complete   │
└─────────────────────────────────────────┘
```

### Without P2P Port ❌

```
┌─────────────────────────────────────────┐
│  Bootstrap Node (localhost:8080)        │
│  Height: 108,000                         │
│  Producing blocks...                     │
└───────────────┬─────────────────────────┘
                │
                │ libp2p gossipsub
                │ (blocked by Docker NAT)
                │
                ✗ BLOCKED
┌─────────────────────────────────────────┐
│  Docker Container (161.35.219.10:9080)  │
│  Port 8080 → HTTP API                   │
│  Port 8081 → NOT EXPOSED ❌             │
│                                          │
│  ❌ Can't receive blocks                 │
│  ❌ Stuck at height 1                    │
│  ⏸️  Mining paused (sync-first mode)     │
│  ❌ Never catches up                     │
└─────────────────────────────────────────┘
```

## Verification

After starting the container, verify P2P connectivity:

```bash
# Check if container is running
docker ps | grep quillon

# Check logs for P2P initialization
docker logs quillon-node | grep -E "libp2p|P2P|gossipsub"

# Expected output:
# ✅ libp2p initialized on /ip4/0.0.0.0/tcp/8081
# ✅ Subscribed to gossipsub topic: /qnk/testnet/blocks
# 📥 GOSSIPSUB: topic=/qnk/testnet/blocks, size=...

# Check if syncing
docker logs quillon-node | grep -E "Received block|Network height"

# Expected output:
# 📦 Received block 1000 (height=1000) from network
# 📈 Network height updated: 999 -> 1000
# 🚀 FAST SYNC: 107000 blocks behind
```

## Troubleshooting

### Container stuck at height 1

**Cause:** P2P port (8081) not exposed
**Solution:** Restart with both ports mapped or use `--network host`

### "Not receiving blocks from network"

**Cause:** Firewall blocking port 8081 or 9081
**Solution:**
```bash
# Check if port is listening
netstat -tulpn | grep 8081

# Open firewall
ufw allow 9081/tcp
```

### "Connected peers: 0"

**Cause:** Bootstrap peer not reachable
**Solution:** Ensure bootstrap nodes are accessible and on same network

## Performance Expectations

### Sync Speed with P2P:
- **Local network**: 1,000-5,000 blocks/second
- **Internet**: 100-500 blocks/second
- **Full sync (100k blocks)**: 20 seconds - 10 minutes

### Without P2P:
- **No sync**: Stuck at genesis block forever ❌

## Version Info

- **Version**: v0.3.1-beta
- **Features**:
  - ✅ Sync-first mode (no mining until caught up)
  - ✅ Active block sync loop (detects when behind)
  - ✅ HTTP fallback sync (`/api/v1/blocks/range`)
  - ✅ Fast gossipsub propagation
  - ✅ Real-time miner block notifications

## Recommended Docker Run Command

```bash
docker run -d \
  --name quillon-node \
  --network host \
  --restart unless-stopped \
  -v ./data:/app/data \
  quillon-api:v0.3.1
```

This setup:
- Uses host networking for maximum P2P performance
- Auto-restarts if it crashes
- Persists blockchain data to `./data` directory
- No port mapping needed (direct host access)

---

**Key Takeaway:** Always expose port 8081 for P2P, or use `--network host`!
