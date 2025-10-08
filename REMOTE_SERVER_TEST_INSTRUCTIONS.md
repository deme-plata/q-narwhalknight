# Remote Server Connection Test

## 🎯 Objective
Test REAL quantum physics integration (Kyber1024 + Dilithium5) between:
- **Bootstrap Server** (185.182.185.227) - This server
- **Remote Node** - Your other server with different IP

## 📋 Remote Server Setup

### 1. Copy Binary to Remote Server
```bash
# On remote server, download the compiled binary
scp root@185.182.185.227:/opt/orobit/shared/q-narwhalknight/target/x86_64-unknown-linux-gnu/release/q-api-server /usr/local/bin/q-api-server
chmod +x /usr/local/bin/q-api-server
```

### 2. Run Remote Node
```bash
# On remote server
export Q_DB_PATH=./data-remote-node
export Q_P2P_PORT=9002
export Q_NARWHAL_BOOTSTRAP_NODE=185.182.185.227:6881
export Q_BOOTSTRAP_PEERS=185.182.185.227:6881

echo "🚀 Launching Q-NarwhalKnight Remote Node"
echo "Environment variables set:"
echo "  Q_P2P_PORT=$Q_P2P_PORT"
echo "  Q_NARWHAL_BOOTSTRAP_NODE=$Q_NARWHAL_BOOTSTRAP_NODE"
echo "  Q_BOOTSTRAP_PEERS=$Q_BOOTSTRAP_PEERS"
echo "  Q_DB_PATH=$Q_DB_PATH"
echo ""
echo "Expected ports:"
echo "  API: 8090"
echo "  librqbit DHT: 9002"
echo "  libp2p: 9102 (Q_P2P_PORT + 100)"
echo "  P2P listener: 8091"
echo ""
echo "Starting server with REAL quantum cryptography..."
echo ""

/usr/local/bin/q-api-server \
  --port 8090 \
  --node-id remote-node \
  --production
```

## 🔬 What to Look For

### On Remote Node
Look for these log messages indicating REAL quantum physics:
```
✅ "Initializing REAL quantum transport for Phase Phase1"
✅ "Using Kyber1024 (NIST ML-KEM) + Dilithium5 (NIST ML-DSA)"
✅ "Generated REAL Kyber1024 keypair"
✅ "Establishing REAL quantum channel with peer"
✅ "Connected to bootstrap peer via quantum-secure channel"
```

### On Bootstrap Server (185.182.185.227)
Check for incoming connection:
```bash
curl http://localhost:8090/api/v1/network/peers
```

Expected output:
```json
{
  "peers": [
    {
      "node_id": "remote-node",
      "address": "<remote-ip>:9102",
      "connection_quality": 1.0,
      "quantum_secure": true,
      "crypto_phase": "Phase1"
    }
  ]
}
```

## ✅ Success Criteria

1. ✅ Remote node connects to bootstrap (185.182.185.227:6881)
2. ✅ Quantum handshake completes (<50ms)
3. ✅ Kyber1024 key exchange succeeds
4. ✅ libp2p gossipsub messages flow
5. ✅ NO MOCK DATA - all using REAL post-quantum cryptography

## 🎉 This Tests

- **REAL Kyber1024** key exchange across internet
- **REAL Dilithium5** signature verification
- **REAL libp2p** networking with quantum transport
- **NO simulation** - actual production cryptography
- **Cross-server** quantum-secure communication

