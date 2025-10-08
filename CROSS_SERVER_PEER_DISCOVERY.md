# Cross-Server Peer Discovery for Q-NarwhalKnight

## ✅ SOLUTION IMPLEMENTED

The Q-NarwhalKnight DHT-to-Gossip bridge now supports cross-server peer discovery through the `Q_PEER_SERVERS` environment variable.

## 🚀 How to Enable Cross-Server Peer Discovery

### 1. Set the Q_PEER_SERVERS Environment Variable

When starting your Q-NarwhalKnight node, specify the IP addresses or hostnames of other servers running Q-NarwhalKnight nodes:

```bash
# Example for Server Beta
export Q_PEER_SERVERS="server-alpha,192.168.1.10,node3.example.com"
export Q_DB_PATH=./data-node-beta
export Q_P2P_PORT=9001
./target/x86_64-unknown-linux-gnu/release/q-api-server --port 8001 --node-id node-beta --production
```

### 2. Multiple Server Configuration

For multi-server deployments, each server should set Q_PEER_SERVERS to point to the other servers:

**Server Alpha (IP: 10.0.0.1):**
```bash
export Q_PEER_SERVERS="10.0.0.2,10.0.0.3"  # Points to Beta and Gamma
./q-api-server --port 8001 --node-id alpha --production
```

**Server Beta (IP: 10.0.0.2):**
```bash
export Q_PEER_SERVERS="10.0.0.1,10.0.0.3"  # Points to Alpha and Gamma
./q-api-server --port 8001 --node-id beta --production
```

**Server Gamma (IP: 10.0.0.3):**
```bash
export Q_PEER_SERVERS="10.0.0.1,10.0.0.2"  # Points to Alpha and Beta
./q-api-server --port 8001 --node-id gamma --production
```

## 🔍 How It Works

1. **BEP-44 DHT Discovery** scans the specified server IPs for Q-NarwhalKnight nodes
2. **Port Scanning** checks these ports on each server: `[8001, 8002, 8003, 8080, 8081, 8082, 8090, 8091, 8092, 8093, 8094, 8095, 25001, 25002, 27000, 27001, 28000, 28001]`
3. **DHT-to-Gossip Bridge** converts discovered peers to libp2p gossip messages
4. **Gossipsub Protocol** propagates peer information across the network
5. **Consensus Integration** uses gossip topics for DAG-Knight consensus

## 📊 Monitoring Cross-Server Discovery

When configured correctly, you'll see these log messages:

```
🌐 Added peer server for cross-server discovery: server-alpha
🔍 BEP-44 found potential peer on server-alpha:8001
🔗 Forwarded DHT peer discovery to gossip network
📤 Forwarded peer (validator_abc123) to consensus layer
```

## 🎯 Troubleshooting

### Nodes Not Discovering Each Other?

1. **Check Firewall Rules** - Ensure ports 8001-8095 are open between servers
2. **Verify Q_PEER_SERVERS** - Check the environment variable is set correctly
3. **Test Connectivity** - Try `curl http://other-server:8001/health` between servers
4. **Check Logs** - Look for "Added peer server for cross-server discovery" messages
5. **Verify Node Health** - Ensure all nodes show `"success": true` on `/health` endpoint

### Example Test Commands

```bash
# Test if Server Alpha can reach Server Beta
curl -s http://server-beta:8001/health

# Check discovered peers on a node
curl -s http://localhost:8001/api/network/status | jq '.data.discovered_peers'

# Monitor discovery logs
tail -f /tmp/node.log | grep -E "discovered|peer|BEP-44|gossip"
```

## 🌟 Complete Pipeline

```
Q_PEER_SERVERS → BEP-44 DHT Discovery → DHT Events → Libp2p Bridge → Gossip Network → Consensus
```

## 💡 Tips

- Use IP addresses instead of hostnames for faster discovery
- Keep all nodes on the same ports (e.g., 8001) for simpler configuration
- The discovery interval is 15 seconds, so allow time for initial discovery
- Monitor the DHT-to-Gossip coordinator logs for detailed debugging

## 🚧 Future Improvements

- Automatic peer discovery via mDNS for local networks
- Bootstrap node configuration for initial network entry
- Real BitTorrent DHT integration for global peer discovery
- Peer exchange protocol for dynamic network growth