# Q-NarwhalKnight v0.0.9-beta

**Quantum-Enhanced DAG-BFT Consensus System with Fixed Peer Discovery**

## Quick Start

### 1. Run Full Node
```bash
cd bin
./q-api-server --port 8080
```

### 2. Run with Terminal UI
```bash
./q-api-server --port 8080 --tui
```

### 3. Check Status
```bash
curl http://localhost:8080/api/v1/status
```

## What's New in v0.0.9-beta

- ✅ **Fixed peer discovery** - Nodes now reliably connect via mDNS and Kademlia DHT
- ✅ **Real-time peer counting** - Accurate peer count in console and frontend
- ✅ **Enhanced logging** - Better visibility into connection attempts
- ✅ **TUI enabled by default** - Beautiful terminal interface

## Features

- **Peer Discovery**: Automatic via mDNS (local) and Kademlia DHT (global)
- **Austrian Economics**: 21M fixed supply, 0.5 QUG block reward, 1s block time
- **Real-Time API**: WebSocket/SSE streaming for live updates
- **Privacy First**: ZK-SNARK/STARK ready architecture
- **Mining**: CPU mining with Blake3 + VDF

## Configuration

### Environment Variables
```bash
Q_DB_PATH=./data              # Database location
Q_P2P_PORT=9001               # P2P port (random if not set)
Q_BOOTSTRAP_PEERS=...         # Custom bootstrap nodes
```

### Command Line
```bash
./q-api-server --help

Options:
  --port <PORT>           API port (default: 8080)
  --node-id <ID>          Node identifier
  --tui                   Enable terminal UI
```

## API Endpoints

- `GET /api/v1/status` - Node status and peer count
- `GET /api/v1/blocks/recent` - Recent blocks
- `GET /api/v1/transactions/recent` - Recent transactions
- `GET /api/v1/statistics/network` - Network statistics
- `POST /api/v1/transactions/send` - Submit transaction

## Mining

```bash
# Download miner from https://quillon.xyz/mining
./q-miner --wallet <YOUR_WALLET> --server http://localhost:8080
```

## Support

- **Website**: https://quillon.xyz
- **Explorer**: https://quillon.xyz/explorer
- **Issues**: https://github.com/deme-plata/q-narwhalknight/issues

## License

MIT License - See LICENSE file
