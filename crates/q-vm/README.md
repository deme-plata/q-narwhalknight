# DAGKnight VM

A high-performance virtual machine for the DAGKnight blockchain platform.

## Features

- WASM-based smart contract execution
- Comprehensive state management
- PBFT consensus with block ordering and finality
- Complete P2P network with gossipsub protocol
- Transaction mempool with prioritization
- Robust testing framework

## Getting Started

```bash
# Build the project
cargo build

# Run tests
cargo test

# Run the node
cargo run -- --node-id 0 --peers "/ip4/127.0.0.1/tcp/8081"
```

## Architecture

The DAGKnight VM consists of several key components:

1. **VM Core** - Executes WASM smart contracts
2. **State Management** - Handles the blockchain state
3. **Consensus** - PBFT consensus with finality guarantees
4. **Network** - P2P network stack for communication
5. **Mempool** - Manages pending transactions
6. **Storage** - Persistent storage using RocksDB

## License

MIT
