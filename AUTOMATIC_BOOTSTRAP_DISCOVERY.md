# Automatic Bootstrap Discovery - User Guide

## 🚀 Zero-Configuration Peer Connection

Q-NarwhalKnight now supports **automatic bootstrap peer discovery** via HTTP API. Users no longer need to manually configure bootstrap peers - simply point to the masternode API and the node will automatically discover and connect to peers.

## 📡 How It Works

1. **User starts node** with just the masternode URL
2. **Node queries** the masternode's `/api/v1/status` endpoint
3. **Masternode returns** its libp2p peer ID and listen addresses
4. **User node automatically connects** to the masternode
5. **Network discovery continues** via Kademlia DHT and mDNS

## 🎯 Quick Start

### Zero-Configuration Startup (Automatic!)
```bash
./q-api-server --port 8080
```

That's it! The node will **automatically**:
1. Connect to the Q-NarwhalKnight Testnet Masternode (185.182.185.227:8080)
2. Discover libp2p peer addresses via HTTP API
3. Join the global peer-to-peer network
4. Start syncing blockchain state

### Custom Bootstrap Node (Optional)
To use a different bootstrap node:
```bash
Q_BOOTSTRAP_URL=http://your-bootstrap-node.com:8080 ./q-api-server --port 8080
```

## 🔧 Advanced Configuration

### Manual Bootstrap Peers (Advanced Users)
If you prefer to manually specify bootstrap peers:
```bash
Q_BOOTSTRAP_PEERS="/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWHgRygY58vccKTJ5kyQhfZr4TTZxMfbNtdb9To6uENj9p" ./q-api-server
```

### Local Network Only (No Bootstrap)
For local testing without connecting to the testnet:
```bash
./q-api-server --port 8080
# Will use mDNS for local peer discovery only
```

## 📊 API Endpoint Format

The `/api/v1/status` endpoint now includes libp2p peer information:

```json
{
  "success": true,
  "data": {
    "node_id": "9029883faa...",
    "current_height": 20093,
    "libp2p": {
      "peer_id": "12D3KooWHgRygY58vccKTJ5kyQhfZr4TTZxMfbNtdb9To6uENj9p",
      "listen_addresses": [
        "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWHgRygY58vccKTJ5kyQhfZr4TTZxMfbNtdb9To6uENj9p",
        "/ip4/172.17.0.1/tcp/39383/p2p/12D3KooWHgRygY58vccKTJ5kyQhfZr4TTZxMfbNtdb9To6uENj9p"
      ]
    }
  }
}
```

## 🌐 Network Discovery Layers

Q-NarwhalKnight uses **three discovery mechanisms** for maximum resilience:

1. **mDNS** - Local network discovery (< 1 second, zero-config)
2. **Kademlia DHT** - Global internet discovery (via bootstrap peers)
3. **Gossipsub** - Consensus message propagation across the network

## ✅ Verification

To verify your node connected successfully:

```bash
curl http://localhost:8080/api/v1/status | jq '.data.connected_peers'
```

You should see `connected_peers` increase from 0 to 1+ within 30 seconds.

## 🎓 Technical Details

### Bootstrap Discovery Process

1. **Environment Check**: System checks for `Q_BOOTSTRAP_URL` environment variable
2. **HTTP Request**: Makes GET request to `{URL}/api/v1/status`
3. **JSON Parsing**: Extracts `data.libp2p.listen_addresses`
4. **Filter**: Removes localhost addresses (127.0.0.1, ::1)
5. **Configuration**: Passes addresses to libp2p UnifiedNetworkManager
6. **Connection**: libp2p dials the bootstrap peer multiaddrs
7. **DHT Bootstrap**: Kademlia DHT uses bootstrap peer to discover more nodes

### Security Considerations

- **TLS Not Required**: Initial discovery uses HTTP (not HTTPS)
- **Trust Model**: You must trust the bootstrap URL you provide
- **Production**: Consider using HTTPS for bootstrap in production
- **Tor Support**: Coming in Phase 2 - automatic .onion bootstrap discovery

## 🔧 Troubleshooting

### "No bootstrap peers discovered"
- Check that the masternode URL is correct
- Verify the masternode is running: `curl http://185.182.185.227:8080/health`
- Check firewall allows HTTP (port 8080) and P2P (port 9001)

### "Failed to fetch bootstrap peers"
- Masternode may be temporarily unavailable
- Network connectivity issue
- Firewall blocking outbound HTTP requests

### "Connected peers = 0 after 60 seconds"
- Check P2P port 9001 is not firewalled
- Verify NAT/firewall allows incoming connections
- Try setting explicit P2P port: `Q_P2P_PORT=9001`

## 📚 Related Documentation

- [Network Architecture](docs/technical/NETWORK_ARCHITECTURE_ANALYSIS.md)
- [Multi-Server Testing Guide](docs/guides/MULTI_SERVER_TESTING_GUIDE.md)
- [Production Deployment Guide](docs/guides/PRODUCTION_DEPLOYMENT_GUIDE.md)

## 🚀 Future Enhancements

- **HTTPS Bootstrap**: Secure bootstrap discovery with TLS
- **Tor Bootstrap**: Automatic .onion peer discovery
- **DNS Seed Nodes**: DNS-based peer discovery (Bitcoin-style)
- **BEP-44 DHT**: BitTorrent DHT integration for censorship resistance
- **Default Bootstrap**: Built-in default bootstrap URLs

---

**Quantum consensus made simple - just point and connect!** 🌟
