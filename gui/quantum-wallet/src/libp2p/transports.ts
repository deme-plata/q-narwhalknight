/**
 * Transport Layer Configuration for Browser P2P Node
 *
 * Implements a multi-transport strategy:
 * 1. WebSocket - Browser to Rust nodes (primary)
 * 2. WebRTC - Browser to Browser (future, direct P2P)
 * 3. Circuit Relay - Fallback for restrictive NATs
 */

import { webSockets } from '@libp2p/websockets'

/**
 * Create WebSocket transport for browser-to-Rust-node connections
 *
 * WebSocket Configuration:
 * - Supports both ws:// (development) and wss:// (production)
 * - Automatic protocol upgrade from HTTP to WebSocket
 * - Compatible with nginx WebSocket proxy
 *
 * @returns WebSocket transport configuration
 */
export function createWebSocketTransport() {
  return webSockets()
}

/**
 * Create all transports for the browser node
 *
 * Phase 1: WebSocket only
 * Phase 5: Add WebRTC for browser-to-browser
 * Phase 6: Add Circuit Relay for NAT traversal
 *
 * @returns Array of transport configurations
 */
export function createTransports() {
  const transports = []

  // Phase 1: WebSocket transport (browser ↔ Rust node)
  transports.push(createWebSocketTransport())

  // TODO Phase 5: Add WebRTC transport (browser ↔ browser)
  // import { webRTC } from '@libp2p/webrtc'
  // transports.push(webRTC({
  //   rtcConfiguration: {
  //     iceServers: [
  //       { urls: 'stun:stun.l.google.com:19302' },
  //       { urls: 'stun:global.stun.twilio.com:3478' }
  //     ]
  //   }
  // }))

  // TODO Phase 6: Add Circuit Relay transport (NAT traversal)
  // import { circuitRelayTransport } from '@libp2p/circuit-relay-v2'
  // transports.push(circuitRelayTransport({
  //   discoverRelays: 2
  // }))

  return transports
}

/**
 * Validate transport connectivity
 *
 * Tests if the transport layer can establish connections
 * Used for diagnostics and health checks
 *
 * @param multiaddr - Address to test
 * @returns Promise<boolean> - True if connection succeeds
 */
export async function testTransportConnectivity(multiaddr: string): Promise<boolean> {
  try {
    // Parse multiaddr to check if it's a valid WebSocket address
    if (!multiaddr.includes('/wss/') && !multiaddr.includes('/ws/')) {
      console.warn('⚠️  [TRANSPORT] Not a WebSocket address:', multiaddr)
      return false
    }

    console.log('🔍 [TRANSPORT] Testing connectivity to:', multiaddr)

    // Extract WebSocket URL from multiaddr
    // Example: /dns4/quillon.xyz/tcp/9001/wss/p2p/12D3... → wss://quillon.xyz:9001
    const parts = multiaddr.split('/')
    const protocol = parts.includes('wss') ? 'wss' : 'ws'
    const host = parts[2]
    const port = parts[4]

    const wsUrl = `${protocol}://${host}:${port}/p2p`

    // Attempt WebSocket connection
    const ws = new WebSocket(wsUrl)

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        ws.close()
        console.error('❌ [TRANSPORT] Connection timeout')
        resolve(false)
      }, 10000) // 10 second timeout

      ws.onopen = () => {
        console.log('✅ [TRANSPORT] Connection successful')
        clearTimeout(timeout)
        ws.close()
        resolve(true)
      }

      ws.onerror = (error) => {
        console.error('❌ [TRANSPORT] Connection failed:', error)
        clearTimeout(timeout)
        resolve(false)
      }
    })
  } catch (error) {
    console.error('❌ [TRANSPORT] Test failed:', error)
    return false
  }
}

/**
 * Get transport statistics
 *
 * Returns information about active transports and connections
 *
 * @returns Transport statistics object
 */
export function getTransportStats() {
  return {
    websocket: {
      enabled: true,
      description: 'WebSocket transport for browser-to-Rust connections',
    },
    webrtc: {
      enabled: false, // TODO: Enable in Phase 5
      description: 'WebRTC transport for browser-to-browser direct connections',
    },
    relay: {
      enabled: false, // TODO: Enable in Phase 6
      description: 'Circuit Relay for NAT traversal fallback',
    },
  }
}
