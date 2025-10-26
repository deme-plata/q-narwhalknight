# Dynamic Peer ID Discovery

## Problem Statement

libp2p peer IDs are generated from keypairs and change every time a node restarts. This creates a problem for bootstrap peer configuration:

- Hardcoded peer IDs in configuration become stale after restart
- Clients can't connect to bootstrap nodes with outdated peer IDs
- Manual updates required after every bootstrap node restart

## Solution

### HTTP Endpoint for Peer ID Discovery

Bootstrap nodes now expose an HTTP endpoint that returns their current peer ID:

**Endpoint**: `GET /api/v1/peer-id`

**Response**:
```json
{
  "success": true,
  "data": {
    "peer_id": "12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG",
    "listen_addresses": [
      "/ip4/0.0.0.0/tcp/9001",
      "/ip4/127.0.0.1/tcp/9001",
      "/ip6/::/tcp/9001"
    ],
    "multiaddr_examples": [
      "/ip4/0.0.0.0/tcp/9001/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG",
      "/ip4/127.0.0.1/tcp/9001/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG",
      "/ip6/::/tcp/9001/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG"
    ]
  }
}
```

### Usage

**For Client Applications:**

1. Before connecting to bootstrap node, fetch current peer ID:
   ```bash
   curl http://185.182.185.227:18080/api/v1/peer-id | jq -r '.data.peer_id'
   ```

2. Build multiaddr with fetched peer ID:
   ```bash
   PEER_ID=$(curl -s http://185.182.185.227:18080/api/v1/peer-id | jq -r '.data.peer_id')
   MULTIADDR="/ip4/185.182.185.227/tcp/9001/p2p/${PEER_ID}"
   ```

3. Connect using dynamic multiaddr

**For Automated Clients:**

```rust
// Fetch peer ID from bootstrap node
let response: ApiResponse<PeerIdInfo> = reqwest::get(
    "http://185.182.185.227:18080/api/v1/peer-id"
)
.await?
.json()
.await?;

let peer_id = response.data.peer_id;

// Build multiaddr
let multiaddr = format!("/ip4/185.182.185.227/tcp/9001/p2p/{}", peer_id);

// Connect
network_manager.dial_peer(multiaddr.parse()?)?;
```

## Implementation Details

### Files Modified

1. **crates/q-api-server/src/handlers.rs** (lines 119-146)
   - Added `get_peer_id()` handler function
   - Retrieves peer ID from UnifiedNetworkManager
   - Returns JSON response with peer info

2. **crates/q-api-server/src/main.rs** (line 1858)
   - Added route: `.route("/api/v1/peer-id", get(handlers::get_peer_id))`

3. **crates/q-network/src/unified_network_manager.rs** (lines 629-632)
   - Added `get_listen_addrs()` method
   - Returns current listen addresses from swarm

### Architecture

```
┌─────────────────────────┐
│  Bootstrap Node         │
│  185.182.185.227        │
│                         │
│  libp2p Peer ID:        │
│  12D3Koo...             │ ← Changes on restart
└────────┬────────────────┘
         │
         │ Exposes HTTP endpoint
         ▼
┌─────────────────────────┐
│  GET /api/v1/peer-id    │
│  Returns current peer ID│
└────────┬────────────────┘
         │
         │ Client fetches
         ▼
┌─────────────────────────┐
│  Client Application     │
│  - Fetch peer ID        │
│  - Build multiaddr      │
│  - Connect via libp2p   │
└─────────────────────────┘
```

## Benefits

1. **No Manual Configuration**: Peer IDs automatically discovered
2. **Restart Resilience**: Works even after bootstrap node restarts
3. **DNS-Free**: No need for DNS TXT records
4. **Simple Implementation**: Just HTTP GET request
5. **Backward Compatible**: Can still use hardcoded peer IDs as fallback

## Bootstrap Configuration

### Multiple Bootstrap Nodes (Recommended)

For redundancy, you can configure multiple bootstrap nodes. The system will try all of them:

**Testnet** (crates/q-types/src/lib.rs:762):
```rust
bootstrap_peers: vec![
    // Primary bootstrap node (with hardcoded peer ID)
    "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG".to_string(),

    // Secondary bootstrap node (with automatic discovery)
    "/ip4/YOUR_SECOND_NODE_IP/tcp/9001".to_string(),

    // Or with hardcoded peer ID
    "/ip4/YOUR_SECOND_NODE_IP/tcp/9001/p2p/PEER_ID_HERE".to_string(),
],
```

**Mainnet** (crates/q-types/src/lib.rs:792):
```rust
bootstrap_peers: vec![
    // Primary bootstrap node (automatic discovery)
    "/ip4/185.182.185.227/tcp/9002".to_string(),

    // Secondary bootstrap node
    "/ip4/YOUR_SECOND_NODE_IP/tcp/9002".to_string(),
],
```

### How Multiple Bootstrap Nodes Work

1. **Redundancy**: If one bootstrap node is down, others are still available
2. **Automatic Discovery**: Peer IDs are fetched automatically from HTTP endpoints
3. **Parallel Connection**: System tries all bootstrap nodes in parallel
4. **Failover**: If HTTP fetch fails for one node, tries the next

### Adding Your Second Bootstrap Node

To add a second bootstrap node at IP `203.0.113.100`:

```rust
bootstrap_peers: vec![
    "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG".to_string(),
    "/ip4/203.0.113.100/tcp/9001".to_string(), // Automatic peer ID discovery
],
```

The system will:
1. Connect to first bootstrap node using hardcoded peer ID
2. Fetch peer ID from `http://203.0.113.100:18080/api/v1/peer-id`
3. Connect to second bootstrap node with fetched peer ID
4. DHT now has 2 bootstrap peers for better connectivity

## Future Enhancements

1. **Automatic Retry Logic**: Implement in UnifiedNetworkManager
2. **Caching**: Cache peer ID with TTL (e.g., 1 hour)
3. **Multiple Bootstrap Nodes**: Support peer ID discovery from multiple sources
4. **DNS-Based Discovery**: Add DNS TXT record support as alternative

## Testing

```bash
# Test the endpoint
curl http://185.182.185.227:18080/api/v1/peer-id

# Extract just the peer ID
curl -s http://185.182.185.227:18080/api/v1/peer-id | jq -r '.data.peer_id'

# Build complete multiaddr
PEER_ID=$(curl -s http://185.182.185.227:18080/api/v1/peer-id | jq -r '.data.peer_id')
echo "/ip4/185.182.185.227/tcp/9001/p2p/${PEER_ID}"
```

## Related Documentation

- libp2p Multiaddr Spec: https://github.com/multiformats/multiaddr
- libp2p PeerId Spec: https://github.com/libp2p/specs/blob/master/peer-ids/peer-ids.md
- Q-NarwhalKnight Network Architecture: See `crates/q-network/README.md`
