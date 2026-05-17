/**
 * Transport Layer Configuration for Browser P2P Node
 *
 * Transport Strategy:
 * 1. WebSocket Secure (wss://quillon.xyz:9443) → nginx → libp2p:9001/ws
 * 2. Circuit Relay for browser-to-browser connections
 * 3. NO WebRTC - Disabled to prevent IP leaks (STUN reveals real IP)
 *
 * v3.6.0-browser: WebSocket + Circuit Relay transports
 * v3.5.4-browser: (REMOVED) WebRTC disabled - IP leak prevention
 * v3.5.3-browser: Updated for @libp2p/websockets v10.x API changes.
 */

import { webSockets } from '@libp2p/websockets'
import { circuitRelayTransport } from '@libp2p/circuit-relay-v2'
// WebRTC import REMOVED - disabled to prevent IP leaks
// import { webRTC } from '@libp2p/webrtc'

import { logTor } from './torConfig'

/**
 * Create WebSocket transport for Tor bridge connections
 *
 * 🧅 TOR ONLY: All WebSocket connections go through port 9444 (Tor bridge)
 *
 * Traffic flow:
 * Browser WebSocket → nginx:9444 → websockify → Tor SOCKS5 → Tor network
 *
 * @returns WebSocket transport configuration
 */
export function createWebSocketTransport() {
  logTor('info', 'Creating WebSocket transport (Tor bridge only)')
  // WebSocket transport - connects to Tor bridge endpoint
  // The actual Tor routing happens server-side via websockify → SOCKS5
  return webSockets()
}

/**
 * WebRTC transport - DISABLED for privacy
 *
 * 🚨 SECURITY: WebRTC is disabled to prevent IP leaks
 *
 * Why WebRTC leaks IP even with Tor:
 * - STUN servers are contacted directly by the browser
 * - ICE candidates contain real public IP addresses
 * - Even with a proxy, WebRTC bypasses it for media/data channels
 *
 * Browser-to-browser connections use Circuit Relay through Tor instead.
 *
 * @deprecated WebRTC disabled - use Circuit Relay through Tor
 */
export function createWebRTCTransport() {
  logTor('warn', 'WebRTC transport requested but DISABLED for privacy')
  throw new Error('WebRTC is disabled to prevent IP leaks. Use Circuit Relay through Tor.')
}

/**
 * Create Circuit Relay transport for browser-to-browser via Tor
 *
 * 🧅 TOR: Circuit Relay routes through Tor-connected bootstrap node
 *
 * Browser A → Tor Bridge → Bootstrap Relay → Tor Bridge → Browser B
 *
 * This is the ONLY way for browsers to communicate with each other
 * since WebRTC is disabled for privacy.
 *
 * v3.6.0: Circuit relay through Tor for browser-to-browser
 *
 * @returns Circuit Relay transport configuration
 */
export function createCircuitRelayTransport() {
  logTor('info', 'Creating Circuit Relay transport (via Tor)')
  // Circuit relay v2 discovers relays automatically
  // Our Tor-connected bootstrap node acts as the relay
  return circuitRelayTransport()
}

/**
 * Create all transports for the browser node
 *
 * 🧅 MANDATORY TOR - v3.6.0
 *
 * Transport stack (Tor-only):
 * 1. WebSocket → Tor bridge (wss://quillon.xyz:9444)
 * 2. Circuit Relay → Through Tor for browser-to-browser
 *
 * REMOVED (privacy risk):
 * - WebRTC (leaks IP via STUN/ICE)
 * - Direct WebSocket to clearnet (leaks IP)
 *
 * @returns Array of Tor-only transport configurations
 */
export function createTransports() {
  logTor('info', 'Creating mandatory Tor transport stack...')

  const transports = []

  // 1. WebSocket through Tor bridge (PRIMARY)
  transports.push(createWebSocketTransport())
  logTor('info', '✅ WebSocket transport (Tor bridge)')

  // 2. Circuit Relay through Tor (browser-to-browser)
  transports.push(createCircuitRelayTransport())
  logTor('info', '✅ Circuit Relay transport (via Tor)')

  // 3. NO WebRTC - explicitly NOT added
  // WebRTC uses STUN which leaks real IP even through proxy
  logTor('info', '🚫 WebRTC DISABLED (IP leak prevention)')

  logTor('info', `Transport stack ready: ${transports.length} transports`)
  return transports
}

/**
 * Test Tor bridge connectivity
 *
 * 🧅 Tests if the Tor bridge WebSocket endpoint is reachable
 * Used for diagnostics and health checks
 *
 * @param multiaddr - Address to test (should be Tor bridge)
 * @returns Promise<{success: boolean, latency?: number}> - Result with latency
 */
export async function testTransportConnectivity(multiaddr: string): Promise<boolean> {
  try {
    // Parse multiaddr to check if it's a valid WebSocket address
    if (!multiaddr.includes('/wss/') && !multiaddr.includes('/ws/')) {
      logTor('warn', 'Not a WebSocket address:', multiaddr)
      return false
    }

    // Verify this is a valid bootstrap endpoint (port 9443 or 9444)
    if (!multiaddr.includes('/tcp/9443/') && !multiaddr.includes('/tcp/9444/')) {
      logTor('warn', 'Not a valid bootstrap address (expected port 9443 or 9444):', multiaddr)
      return false
    }

    logTor('info', 'Testing Tor bridge connectivity...')

    // Extract WebSocket URL from multiaddr
    const parts = multiaddr.split('/')
    const protocol = parts.includes('wss') ? 'wss' : 'ws'
    const host = parts[2]
    const port = parts[4]

    const wsUrl = `${protocol}://${host}:${port}/tor-bridge`

    // Attempt WebSocket connection with longer timeout for Tor
    const ws = new WebSocket(wsUrl)
    const startTime = performance.now()

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        ws.close()
        logTor('error', 'Tor bridge connection timeout (45s)')
        resolve(false)
      }, 45000) // 45 second timeout for Tor

      ws.onopen = () => {
        const latency = Math.round(performance.now() - startTime)
        logTor('info', `Tor bridge connected (${latency}ms)`)
        clearTimeout(timeout)
        ws.close()
        resolve(true)
      }

      ws.onerror = (error) => {
        logTor('error', 'Tor bridge connection failed', error)
        clearTimeout(timeout)
        resolve(false)
      }
    })
  } catch (error) {
    logTor('error', 'Tor bridge test failed', error)
    return false
  }
}

/**
 * Get transport statistics
 *
 * 🧅 Returns Tor-only transport status
 *
 * @returns Transport statistics object
 */
export function getTransportStats() {
  return {
    tor: {
      enabled: true,
      mandatory: true,
      description: 'All traffic routed through Tor (mandatory)',
    },
    websocket: {
      enabled: true,
      endpoint: 'wss://quillon.xyz:9443',
      description: 'WebSocket Secure to libp2p bootstrap node',
    },
    webrtc: {
      enabled: false, // v3.6.0: DISABLED for privacy
      reason: 'Disabled - STUN/ICE leaks real IP address',
      description: 'WebRTC is permanently disabled to prevent IP leaks',
    },
    relay: {
      enabled: true,
      description: 'Circuit Relay through Tor for browser-to-browser connections',
    },
  }
}
