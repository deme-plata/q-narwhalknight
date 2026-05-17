# Q-NarwhalKnight Multi-Bootstrap Decentralization

## Overview

This directory contains the Docker setup for running a secondary bootstrap node on Server Alpha (161.35.219.10), creating a decentralized bootstrap architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                  MULTI-BOOTSTRAP ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────┐         ┌─────────────────┐              │
│   │  Server Beta    │◄───────►│  Server Alpha   │              │
│   │ 185.182.185.227 │  sync   │ 161.35.219.10   │              │
│   │   (Primary)     │         │  (Secondary)    │              │
│   └────────┬────────┘         └────────┬────────┘              │
│            │                           │                        │
│            │     ┌──────────────┐     │                        │
│            └────►│  New Nodes   │◄────┘                        │
│                  │ (Failover)   │                              │
│                  └──────────────┘                              │
│                                                                 │
│   If Primary fails → Automatic failover to Secondary           │
│   If Secondary fails → Primary continues serving                │
│   Both available → Load balanced, highest height wins          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### On Server Alpha (161.35.219.10):

```bash
# 1. Clone or copy the project
git clone <repo> /opt/q-narwhalknight
cd /opt/q-narwhalknight/docker/bootstrap-node

# 2. Deploy the bootstrap node
./deploy-alpha.sh all

# 3. Check status
./deploy-alpha.sh status
```

### On Server Beta (update existing node):

The multi-bootstrap support is already built into the API server. Just ensure the environment is set:

```bash
# Optional: Explicitly configure both bootstrap servers
export Q_BOOTSTRAP_URLS="http://185.182.185.227:8080,http://161.35.219.10:8080"

# Restart the service
systemctl restart q-api-server
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `Q_BOOTSTRAP_URLS` | Both servers | Comma-separated list of bootstrap URLs |
| `Q_BOOTSTRAP_URL` | Server Beta | Legacy single-URL (still works) |
| `Q_BOOTSTRAP_TIMEOUT_MS` | 5000 | Timeout per server in milliseconds |
| `Q_NETWORK_ID` | testnet-phase16 | Network identifier |

## Commands

```bash
./deploy-alpha.sh build   # Build Docker image
./deploy-alpha.sh start   # Start bootstrap node
./deploy-alpha.sh stop    # Stop bootstrap node
./deploy-alpha.sh restart # Restart node
./deploy-alpha.sh logs    # Show logs (follow mode)
./deploy-alpha.sh status  # Show sync status
./deploy-alpha.sh all     # Full deployment (build + start)
```

## Verifying Decentralization

After deployment, verify both bootstrap servers are operational:

```bash
# Check Server Beta
curl -s http://185.182.185.227:8080/api/v1/status | jq '.data.height'

# Check Server Alpha
curl -s http://161.35.219.10:8080/api/v1/status | jq '.data.height'

# Both should show similar heights (within a few blocks)
```

## Testing Failover

To test automatic failover:

```bash
# 1. Start a new node with both bootstrap servers
export Q_BOOTSTRAP_URLS="http://185.182.185.227:8080,http://161.35.219.10:8080"
./q-api-server --data-dir /tmp/test-node

# 2. In another terminal, watch the logs
# You should see: "Using multi-bootstrap with failover..."

# 3. Temporarily stop Server Beta
# The new node should automatically use Server Alpha

# 4. Restart Server Beta
# Both are now available for new connections
```

## Future Decentralization

For true decentralization, add more bootstrap nodes:

1. **Community Bootstrap Nodes**: Allow trusted community members to run bootstrap nodes
2. **DNS-based Discovery**: Add DNS seed records (like Bitcoin's DNS seeds)
3. **DHT-only Mode**: Enable Kademlia DHT as primary discovery (no bootstrap required)

## Files

```
docker/bootstrap-node/
├── docker-compose.yml  # Docker Compose configuration
├── Dockerfile          # Multi-stage build for lean image
├── deploy-alpha.sh     # Deployment script for Server Alpha
└── README.md          # This file
```

## Troubleshooting

### Node not syncing

```bash
# Check connectivity to primary bootstrap
curl -sf http://185.182.185.227:8080/api/v1/status

# Check container logs
docker logs q-bootstrap-alpha --tail 100
```

### Peer ID changes on restart

The peer ID is derived from the libp2p identity key. To preserve it:

```bash
# Mount a persistent identity volume
-v /var/lib/q-narwhalknight/identity:/identity
```

### Port conflicts

If port 8080 or 9001 is in use:

```bash
# Use different ports
docker run -e Q_API_PORT=8081 -e Q_P2P_PORT=9002 ...
```
