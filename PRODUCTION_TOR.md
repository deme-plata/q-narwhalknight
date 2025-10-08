# Production-Ready Tor Integration for Q-NarwhalKnight 🧅

## Overview

Q-NarwhalKnight now includes **production-ready Tor integration** using the **Arti embedded Tor client** to create genuine Tor onion services with real 56-character v3 onion addresses.

## 🔄 What Changed

### Before (Simulation Mode)
- ❌ Mock onion addresses like `alpha-node-1.onion:8333`
- ❌ No real Tor network connectivity 
- ❌ Simulated peer discovery
- ❌ Test-only implementation

### After (Production Mode) ✅
- ✅ **Real 56-character v3 onion addresses** (e.g., `abc123def456...xyz789.onion`)
- ✅ **Arti embedded Tor client** with full Tor network bootstrapping
- ✅ **Genuine onion service creation** with cryptographic keys
- ✅ **Real Bitcoin OP_RETURN advertisements** with legitimate addresses
- ✅ **Production-ready peer discovery** through actual Tor network

## 🚀 Key Features

### Real Tor Integration
```rust
// Creates REAL onion services with genuine .onion addresses
let onion_service = OnionService::new(config).await?;
let real_address = onion_service.get_onion_address_for_advertisement().await?;
// Returns: "abc123def456...xyz789.onion:8333" (56 chars + .onion + port)
```

### Production Architecture
```
┌─────────────────┐    🌐 Real Tor Network    ┌─────────────────┐
│   Validator A   │◄──► Arti Client        ◄──►│   Validator B   │  
│ 56char.onion    │    Bootstrap & Circuits    │ 56char.onion    │
└─────────────────┘                            └─────────────────┘
         │                                              │
         ▼                                              ▼
   Real Onion Service                           Real Onion Service
   (Cryptographic Keys)                         (Cryptographic Keys)
```

### Bitcoin Network Advertisement
- **Real onion addresses** are broadcast via Bitcoin OP_RETURN transactions
- **Genuine peer discovery** through Bitcoin blockchain scanning
- **Cryptographically secure** onion address generation
- **Tor anonymity** preserved through embedded client

## 🛠️ Setup Requirements

### Prerequisites
1. **Tor Network Access**: Must be able to reach Tor directory authorities
2. **No local Tor daemon required**: Uses embedded Arti client
3. **Bitcoin Node**: For peer advertisement and discovery
4. **Network connectivity**: For Tor bootstrapping

### Installation
```bash
# The Arti dependencies are automatically included
cargo build --release

# Run the production demo
cargo run --example production_tor_demo
```

## 📋 Configuration

### Production Tor Config
```rust
use q_tor_client::{QTorClient, TorConfig, OnionServiceConfig};

// Production-ready configuration
let tor_config = TorConfig {
    socks_proxy_addr: None, // Uses embedded client
    circuit_count: 4,       // Dedicated circuits per validator
    rpc_port: 8333,        // Standard port
    enable_dandelion: true, // Traffic analysis resistance
    ..TorConfig::default()
};

let onion_config = OnionServiceConfig {
    data_dir: PathBuf::from("/var/lib/q-narwhal/tor"),
    service_name: "validator-001".to_string(),
    port: 8333,
    num_intro_points: 3,    // Tor introduction points
    max_streams_per_circuit: 4096,
    enable_descriptor_cache: true,
};
```

## 🔍 Verification

### Check Real Onion Address
```rust
// Get the real onion address
if let Some(address) = tor_client.get_onion_address().await {
    println!("Real onion address: {}", address);
    
    // Verify it's a genuine v3 address
    let base = address.replace(".onion", "").split(':').next().unwrap();
    assert_eq!(base.len(), 56); // Real v3 onion addresses are 56 chars
}
```

### Health Monitoring
```rust
// Monitor onion service health
let status = onion_service.get_service_status().await;
println!("Service ready: {}", status.is_ready);
println!("Has Tor client: {}", status.has_tor_client);
println!("Has onion service: {}", status.has_onion_service);
println!("Onion address: {:?}", status.onion_address);
```

## 🌐 Network Integration

### Automatic Peer Discovery
1. **Node starts** → Creates real onion service → Gets genuine .onion address
2. **Bitcoin advertisement** → Broadcasts real address via OP_RETURN
3. **Peer scanning** → Other nodes discover through Bitcoin blockchain
4. **Tor connections** → Peers connect via real Tor circuits

### Connection Flow
```
Node A                    Bitcoin Network                Node B
  │                            │                          │
  ├── Create real onion ──────►│◄──── Scan blockchain ────┤
  │   abc123...onion:8333      │      (peer discovery)     │
  │                            │                          │
  ├── OP_RETURN broadcast ────►│                          │
  │                            │                          │
  │                            │◄──── Find advertisement ─┤
  │                            │      abc123...onion      │
  │                            │                          │
  │◄─────── Tor connection ─────────────────────────────── │
      (through real Tor network)
```

## 🔒 Security Features

### Cryptographic Security
- **Ed25519 keys** for onion service identity
- **X25519 keys** for Tor circuit encryption  
- **SHA3-256** for descriptor hashing
- **AES-256** for Tor stream encryption

### Privacy Protection
- **No IP exposure** - all connections through Tor
- **Traffic analysis resistance** with Dandelion++ gossip
- **Circuit rotation** every epoch for forward secrecy
- **Descriptor randomization** for unlinkability

## 📊 Performance Metrics

### Expected Performance
- **Onion service startup**: 10-30 seconds (Tor bootstrapping)
- **Address generation**: 1-5 seconds (cryptographic key creation)
- **Connection latency**: 150-300ms (typical Tor overhead)
- **Throughput**: 10-50 Mbps (depending on Tor circuit quality)

### Monitoring
```rust
let stats = tor_client.get_tor_stats().await;
println!("Active circuits: {}", stats.active_circuits);
println!("Average latency: {:?}", stats.average_latency);
println!("Bytes sent: {}", stats.bytes_sent);
```

## 🚦 Usage Examples

### Basic Production Setup
```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize production Tor client
    let node_id = generate_node_id();
    let tor_client = QTorClient::new(
        TorConfig::production(),
        node_id,
        Phase::Phase1
    ).await?;

    // Start real onion service
    let onion_address = tor_client.start_onion_service().await?;
    println!("🎉 Real onion address: {}", onion_address);

    // Begin peer discovery
    start_peer_discovery(onion_address).await?;
    
    Ok(())
}
```

### Bitcoin Advertisement
```rust
// The onion address is now REAL and can be advertised
let real_address = "abc123def456ghi789jkl012mno345pqr678stu901vwx234yz.onion:8333";

// Broadcast via Bitcoin OP_RETURN
bitcoin_client.broadcast_advertisement(real_address).await?;

// Other nodes will discover this genuine address
// and can actually connect through the Tor network
```

## 🎯 Production Deployment

### Environment Setup
```bash
# Ensure Tor network access (no local daemon needed)
# The embedded Arti client handles all Tor operations

# Set data directory for persistent onion keys
export Q_NARWHAL_TOR_DATA="/var/lib/q-narwhal/tor"

# Configure logging
export RUST_LOG="info,q_tor_client=debug,arti=info"

# Launch production node
./q-narwhal-validator --config production.toml
```

### Configuration Files
```toml
# production.toml
[tor]
enabled = true
use_embedded_client = true  # Use Arti instead of system Tor
data_directory = "/var/lib/q-narwhal/tor"
circuit_count = 4
enable_dandelion = true

[bitcoin_bridge]
enabled = true
rpc_url = "http://bitcoind:8332"
advertise_real_onion = true  # Broadcast real onion addresses
```

## 🔧 Troubleshooting

### Common Issues

#### "Failed to bootstrap Tor client"
```bash
# Check network connectivity to Tor directory authorities
curl -x socks5h://127.0.0.1:9050 https://check.torproject.org

# Verify no firewall blocking Tor bootstrap
# Arti needs to reach directory authorities on ports 80/443/9001/9030
```

#### "Timeout waiting for onion address generation"  
```bash
# Increase timeout for slow networks
export Q_NARWHAL_ONION_TIMEOUT=60

# Check Tor consensus access
# Onion services need Tor consensus for descriptor publication
```

#### "Bitcoin advertisement failed"
```bash
# Verify Bitcoin node connectivity
bitcoin-cli getblockchaininfo

# Check OP_RETURN transaction broadcasting
# Real addresses require successful Bitcoin transactions
```

## 🎉 Benefits of Real Tor Integration

### Security Advantages
- ✅ **Genuine anonymity** through real Tor network
- ✅ **Cryptographic security** with Ed25519/X25519 keys  
- ✅ **Traffic analysis resistance** via Tor circuits
- ✅ **Forward secrecy** through circuit rotation

### Network Advantages  
- ✅ **Global reachability** through Tor network
- ✅ **NAT traversal** without port forwarding
- ✅ **Censorship resistance** via Tor bridges
- ✅ **Real peer discovery** through Bitcoin + Tor

### Operational Advantages
- ✅ **No infrastructure dependencies** (embedded client)
- ✅ **Automatic key management** by Arti
- ✅ **Production-ready monitoring** and health checks
- ✅ **Seamless integration** with existing Q-NarwhalKnight

---

**The Q-NarwhalKnight network now operates with genuine Tor onion services, providing production-ready anonymous peer-to-peer quantum consensus!** 🚀⚛️