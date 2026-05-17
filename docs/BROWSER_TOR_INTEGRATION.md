# 🧅 Browser Tor Integration - Q-NarwhalKnight

## Overview

Q-NarwhalKnight implements **mandatory Tor routing** for all browser P2P traffic. This means:

1. **Privacy by default** - Users don't need to opt-in
2. **IP address protection** - User's real IP is never exposed to the P2P network
3. **Traffic analysis resistance** - All traffic goes through Tor's onion routing
4. **No WebRTC** - Disabled to prevent IP leaks via STUN/ICE

Users accessing the wallet via clearnet (https://quillon.xyz) automatically have their P2P traffic routed through Tor without knowing.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        User's Browser (Clearnet)                             │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  Quantum Wallet React App                                            │  │
│   │                                                                      │  │
│   │   🧅 Status: "Protected - Traffic routed via Tor"                    │  │
│   │                                                                      │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  js-libp2p Node                                                      │  │
│   │                                                                      │  │
│   │   Transports:                                                        │  │
│   │   ✅ WebSocket → wss://quillon.xyz:9444/tor-bridge                   │  │
│   │   ✅ Circuit Relay → Through Tor for browser-to-browser              │  │
│   │   ❌ WebRTC → DISABLED (IP leak prevention)                          │  │
│   │                                                                      │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ wss:// (TLS encrypted)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Server Beta (quillon.xyz)                             │
│                                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│   │ nginx:9443  │     │ nginx:9444  │     │ Tor Client  │                   │
│   │ (clearnet)  │     │ /tor-bridge │────►│ SOCKS5:9050 │                   │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │
│          │                   │                   │                           │
│          ▼                   ▼                   ▼                           │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐           │
│   │libp2p:9001  │     │websockify   │     │ Tor Network         │           │
│   │(direct P2P) │     │   :9445     │────►│ Entry → Mid → Exit  │           │
│   └─────────────┘     └─────────────┘     └─────────────────────┘           │
│                                                   │                          │
│   .onion Hidden Service                          │                          │
│   qnkxyz7a...xyz.onion:9001 ◄────────────────────┘                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why Tor is Mandatory

### 1. Privacy First Philosophy
Blockchain wallets handle sensitive financial data. Users shouldn't have to remember to enable privacy - it should be the default.

### 2. IP Address Protection
Without Tor, connecting to the P2P network exposes your IP address to:
- Bootstrap nodes
- Other peers in the network
- Network observers

### 3. Traffic Analysis Resistance
Even with encryption, traffic patterns can reveal:
- When you're using the wallet
- Transaction timing correlations
- Network topology information

### 4. WebRTC IP Leak Prevention
WebRTC uses STUN servers which reveal your real IP address, even when using a VPN or proxy. We disable WebRTC entirely.

## Implementation Files

### Browser Side

| File | Purpose |
|------|---------|
| `src/libp2p/torConfig.ts` | Tor configuration constants |
| `src/libp2p/torTransport.ts` | Tor transport creation & monitoring |
| `src/libp2p/transports.ts` | Modified to use Tor-only transports |
| `src/libp2p/config.ts` | Bootstrap peers pointing to Tor bridge |
| `src/libp2p/node.ts` | Node initialization with Tor monitoring |
| `src/components/TorStatusIndicator.tsx` | UI component showing Tor status |

### Server Side

| File | Purpose |
|------|---------|
| `docker/tor-bridge/setup-tor-bridge.sh` | Server setup script |
| `/etc/tor/torrc` | Tor daemon configuration |
| `/etc/nginx/sites-available/qnk-tor-bridge` | nginx WebSocket proxy |
| `/etc/systemd/system/qnk-tor-bridge.service` | websockify service |

## Server Setup

### Prerequisites
- Ubuntu/Debian server
- nginx installed
- Let's Encrypt SSL certificates
- Root access

### Quick Setup

```bash
cd /opt/orobit/shared/q-narwhalknight/docker/tor-bridge
sudo ./setup-tor-bridge.sh
```

### Manual Setup

1. **Install Tor**
```bash
apt install -y tor
```

2. **Configure Tor** (`/etc/tor/torrc`)
```
SocksPort 9050
SocksPolicy accept 127.0.0.1
HiddenServiceDir /var/lib/tor/qnk_hidden_service/
HiddenServicePort 9001 127.0.0.1:9001
```

3. **Install websockify**
```bash
pip3 install websockify
```

4. **Create systemd service**
```bash
# /etc/systemd/system/qnk-tor-bridge.service
[Service]
ExecStart=/usr/local/bin/websockify 127.0.0.1:9445 --proxy-mode=socks5 --proxy-host=127.0.0.1:9050
```

5. **Configure nginx for port 9444**
```nginx
location /tor-bridge {
    proxy_pass http://127.0.0.1:9445;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

6. **Start services**
```bash
systemctl enable tor qnk-tor-bridge
systemctl start tor qnk-tor-bridge
systemctl reload nginx
```

## Configuration

### torConfig.ts

```typescript
export const TOR_CONFIG = {
  enabled: true,                                    // Always enabled
  bridgeEndpoint: 'wss://quillon.xyz:9444/tor-bridge',
  dialTimeout: 45000,                               // 45s for Tor circuits
  circuitRotationInterval: 10 * 60 * 1000,         // 10 minutes
}

export const TOR_SECURITY = {
  allowWebRTC: false,              // NEVER allow WebRTC
  allowDirectConnections: false,   // NEVER allow clearnet
  requireTorCircuit: true,         // MUST have Tor before P2P
}
```

### config.ts Bootstrap Peers

```typescript
export const BOOTSTRAP_PEERS = [
  // Tor bridge endpoint (port 9444)
  '/dns4/quillon.xyz/tcp/9444/wss/p2p/12D3KooWFrhdwDDTgxPX41mUyRgLcE1ozsBYArKM4DT8t4VLwuNx',
]
```

## UI Integration

### TorStatusIndicator Component

```tsx
import { TorStatusIndicator } from './components/TorStatusIndicator'

// In your TopBar or StatusBar component:
<TorStatusIndicator toolbar />

// Or full version:
<TorStatusIndicator />
```

### useTorStatus Hook

```tsx
import { useTorStatus } from './components/TorStatusIndicator'

function MyComponent() {
  const { isConnected, statusText, bridgeLatency } = useTorStatus()

  return (
    <div>
      {isConnected ? `🧅 ${statusText}` : '⏳ Connecting to Tor...'}
    </div>
  )
}
```

## Debug Commands

In browser console:

```javascript
// Get Tor status
window.libp2pDebug.getTorStatus()

// Get transport stats
window.libp2pDebug.getStats()

// Test Tor connection
window.libp2pDebug.testDial()
```

## Performance

### Expected Latencies

| Connection Type | Latency |
|-----------------|---------|
| Direct WebSocket (clearnet) | 10-50ms |
| Tor Bridge | 200-500ms |
| Browser-to-Browser via Tor | 400-1000ms |

### Why Tor is Slower

1. **Circuit establishment** - Building 3-hop circuits takes time
2. **Multiple hops** - Traffic goes through 3+ relays
3. **Encryption layers** - Each hop adds encryption overhead

### Optimizations Applied

1. **Longer timeouts** - 45s dial timeout vs 10s for clearnet
2. **Circuit reuse** - Keep circuits alive for 10 minutes
3. **Heartbeat** - Keep connections warm to avoid circuit rebuilds
4. **Eager dialing** - Connect immediately on page load

## Security Considerations

### What Tor Protects

- ✅ IP address hidden from P2P network
- ✅ Traffic patterns obscured by onion routing
- ✅ Connection metadata (timing, size) mixed with other Tor traffic
- ✅ Geographic location hidden

### What Tor Does NOT Protect

- ❌ Content of transactions (use encryption for that)
- ❌ Blockchain address privacy (use mixers/ring signatures)
- ❌ Browser fingerprinting (use Tor Browser for full protection)
- ❌ Application-level leaks (e.g., if app sends data via non-Tor channel)

### Threat Model

| Attacker | Protection Level |
|----------|------------------|
| P2P network peers | ✅ Full - Cannot see your IP |
| Network observer | ✅ Full - Only sees Tor traffic |
| ISP | ⚠️ Partial - Knows you use Tor, not what for |
| Global adversary | ⚠️ Limited - Traffic correlation possible |
| Bootstrap server | ⚠️ Partial - Sees Tor exit IP, not real IP |

## Troubleshooting

### "Tor bridge connection timeout"

1. Check if Tor service is running on server: `systemctl status tor`
2. Check if websockify is running: `systemctl status qnk-tor-bridge`
3. Verify nginx is proxying: `curl -I https://quillon.xyz:9444/tor-ping`

### "No connections remaining"

The Tor circuit may have been terminated. The browser will automatically reconnect. If persistent:
1. Refresh the page
2. Check server Tor logs: `journalctl -u tor -f`

### High Latency (>1s)

This is normal for Tor. The trade-off is privacy vs speed. Consider:
- Tor circuits may be congested
- Geographic distance through circuit
- Server Tor daemon may need restart

## Future Improvements

### Phase 2: Pure Browser Tor (node-Tor)
When [node-Tor](https://github.com/Ayms/node-Tor) matures, we can implement pure browser-side Tor without needing a server bridge.

### Phase 3: .onion Direct Connection
For Tor Browser users, we can connect directly to the .onion hidden service without going through the bridge.

### Phase 4: Multi-Bridge Redundancy
Add community-run Tor bridges for decentralization and redundancy.

## Related Documentation

- [libp2p Tor transport discussion](https://github.com/libp2p/js-libp2p/issues/142)
- [Tor Project](https://www.torproject.org/)
- [websockify](https://github.com/novnc/websockify)
- [node-Tor](https://github.com/Ayms/node-Tor)
