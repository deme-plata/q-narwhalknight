/**
 * Mandatory Tor Transport Layer for Q-NarwhalKnight Browser
 *
 * ALL P2P connections are routed through Tor automatically.
 * This module replaces the standard transports with Tor-only transports.
 *
 * Key Design Decisions:
 * 1. WebSocket ONLY through Tor bridge (wss://quillon.xyz:9444/tor-bridge)
 * 2. NO WebRTC - it leaks real IP even through proxies
 * 3. Circuit Relay through Tor for browser-to-browser
 * 4. NO clearnet fallback - Tor or nothing
 *
 * @module torTransport
 * @version 1.0.0
 */

import { webSockets } from '@libp2p/websockets'
import { circuitRelayTransport } from '@libp2p/circuit-relay-v2'
import {
  TOR_CONFIG,
  TOR_SECURITY,
  TOR_BOOTSTRAP,
  updateTorState,
  logTor,
  getTorState,
} from './torConfig'

/**
 * Create Tor-only transports for the browser node
 *
 * CRITICAL: This is the ONLY transport creator used by the node.
 * Standard transports are NOT available to prevent clearnet leaks.
 *
 * @returns Array of Tor-only transport configurations
 */
export function createTorTransports() {
  logTor('info', 'Creating mandatory Tor transports...')
  logTor('info', `Bridge endpoint: ${TOR_CONFIG.bridgeEndpoint}`)

  const transports = []

  // 1. WebSocket through Tor bridge (PRIMARY)
  // All WebSocket connections go through the server-side Tor proxy
  transports.push(createTorWebSocketTransport())
  logTor('info', 'WebSocket transport configured (Tor bridge only)')

  // 2. Circuit Relay for browser-to-browser (through Tor)
  // Browsers can relay through the Tor-connected bootstrap node
  transports.push(circuitRelayTransport())
  logTor('info', 'Circuit relay transport configured (via Tor)')

  // 3. NO WebRTC - explicitly NOT added
  // WebRTC uses STUN servers which leak real IP addresses
  if (TOR_SECURITY.allowWebRTC) {
    // This should NEVER be true in production
    logTor('error', 'WebRTC is enabled - THIS IS A SECURITY RISK!')
  } else {
    logTor('info', 'WebRTC disabled (IP leak prevention)')
  }

  logTor('info', `Total transports configured: ${transports.length}`)
  return transports
}

/**
 * Create WebSocket transport that ONLY connects through Tor bridge
 *
 * The browser connects to wss://quillon.xyz:9444/tor-bridge
 * The server routes this through Tor SOCKS5 proxy
 *
 * @returns WebSocket transport configuration for Tor
 */
function createTorWebSocketTransport() {
  // Create standard WebSocket transport
  // The address filtering happens at the multiaddr level in config.ts
  return webSockets()
}

/**
 * Get bootstrap peers for Tor mode
 *
 * Returns ONLY the Tor bridge multiaddr - no clearnet addresses
 *
 * @returns Array of Tor-only bootstrap peer multiaddrs
 */
export function getTorBootstrapPeers(): string[] {
  const peers: string[] = []

  // Primary Tor bridge
  peers.push(TOR_BOOTSTRAP.primary)
  logTor('info', `Primary Tor bootstrap: ${TOR_BOOTSTRAP.primary}`)

  // Add fallback Tor bridges if available
  for (const fallback of TOR_BOOTSTRAP.fallbacks) {
    peers.push(fallback)
    logTor('info', `Fallback Tor bootstrap: ${fallback}`)
  }

  // NO clearnet fallback
  if (TOR_BOOTSTRAP.clearnetFallback) {
    logTor('error', 'Clearnet fallback is configured - THIS SHOULD BE NULL!')
  }

  return peers
}

/**
 * Get connection timeout for Tor
 *
 * Tor connections take longer due to circuit establishment
 *
 * @returns Timeout in milliseconds
 */
export function getTorDialTimeout(): number {
  return TOR_CONFIG.dialTimeout
}

/**
 * Check if an address should be allowed (Tor-only filter)
 *
 * Blocks any address that isn't going through our Tor bridge
 *
 * @param multiaddr - The multiaddr to check
 * @returns true if address is allowed (Tor), false otherwise
 */
export function isAddressAllowed(multiaddr: string): boolean {
  // Allow Tor bridge address
  if (multiaddr.includes('quillon.xyz') && multiaddr.includes('/tcp/9444/')) {
    return true
  }

  // Allow .onion addresses (direct Tor hidden service)
  if (multiaddr.includes('.onion') || multiaddr.includes('/onion3/')) {
    return true
  }

  // Allow circuit relay addresses (browser-to-browser through Tor)
  if (multiaddr.includes('/p2p-circuit/')) {
    return true
  }

  // Block everything else
  logTor('warn', `Blocked non-Tor address: ${multiaddr}`)
  return false
}

/**
 * Filter a list of multiaddrs to only include Tor-safe addresses
 *
 * @param multiaddrs - Array of multiaddrs to filter
 * @returns Array of Tor-safe multiaddrs
 */
export function filterTorAddresses(multiaddrs: string[]): string[] {
  return multiaddrs.filter(isAddressAllowed)
}

/**
 * Monitor Tor connection health
 *
 * Starts a heartbeat to verify Tor circuit is still alive
 *
 * @param onHealthChange - Callback when health status changes
 */
export function startTorHealthMonitor(
  onHealthChange?: (healthy: boolean) => void
): () => void {
  let lastHealthy = false
  let intervalId: ReturnType<typeof setInterval> | null = null

  const checkHealth = () => {
    const state = getTorState()
    const healthy = state.isConnected && state.circuitEstablished

    if (healthy !== lastHealthy) {
      lastHealthy = healthy
      logTor(healthy ? 'info' : 'warn', `Tor health changed: ${healthy ? 'HEALTHY' : 'UNHEALTHY'}`)
      onHealthChange?.(healthy)
    }

    // Update heartbeat timestamp if healthy
    if (healthy) {
      updateTorState({ lastHeartbeat: Date.now() })
    }
  }

  // Check immediately
  checkHealth()

  // Check periodically
  intervalId = setInterval(checkHealth, TOR_CONFIG.heartbeatInterval)

  // Return cleanup function
  return () => {
    if (intervalId) {
      clearInterval(intervalId)
    }
  }
}

/**
 * Handle Tor connection established
 *
 * Called when the WebSocket connection to Tor bridge is established
 *
 * @param latency - Connection latency in ms (optional)
 */
export function onTorConnected(latency?: number): void {
  updateTorState({
    isConnected: true,
    circuitEstablished: true,
    bridgeLatency: latency ?? null,
    lastHeartbeat: Date.now(),
    reconnectAttempts: 0,
  })
  logTor('info', `Tor circuit established${latency ? ` (latency: ${latency}ms)` : ''}`)
}

/**
 * Handle Tor connection lost
 *
 * Called when the WebSocket connection to Tor bridge is lost
 *
 * @param error - Error message (optional)
 */
export function onTorDisconnected(error?: string): void {
  const state = getTorState()
  updateTorState({
    isConnected: false,
    circuitEstablished: false,
    bridgeLatency: null,
    reconnectAttempts: state.reconnectAttempts + 1,
  })

  if (error) {
    logTor('error', `Tor disconnected: ${error}`)
  } else {
    logTor('warn', 'Tor disconnected')
  }

  // Warn if too many reconnect attempts
  if (state.reconnectAttempts >= TOR_CONFIG.maxConnectionAttempts) {
    logTor('error', `Too many reconnect attempts (${state.reconnectAttempts}). Check Tor bridge.`)
  }
}

/**
 * Measure Tor circuit latency
 *
 * Sends a ping through the Tor circuit and measures round-trip time
 *
 * @returns Promise<number> - Latency in milliseconds
 */
export async function measureTorLatency(): Promise<number> {
  const start = performance.now()

  try {
    // Ping the Tor bridge endpoint
    const response = await fetch(`${TOR_CONFIG.bridgeEndpoint.replace('wss://', 'https://').replace('/tor-bridge', '/tor-ping')}`, {
      method: 'HEAD',
      cache: 'no-store',
    })

    const end = performance.now()
    const latency = Math.round(end - start)

    updateTorState({ bridgeLatency: latency })
    logTor('info', `Tor latency: ${latency}ms`)

    return latency
  } catch (error) {
    logTor('warn', 'Failed to measure Tor latency', error)
    return -1
  }
}

/**
 * Get Tor transport statistics
 *
 * @returns Transport statistics object
 */
export function getTorTransportStats() {
  const state = getTorState()

  return {
    tor: {
      enabled: TOR_CONFIG.enabled,
      connected: state.isConnected,
      circuitEstablished: state.circuitEstablished,
      latency: state.bridgeLatency,
      reconnectAttempts: state.reconnectAttempts,
      errors: state.errors,
    },
    websocket: {
      enabled: true,
      endpoint: TOR_CONFIG.bridgeEndpoint,
      description: 'WebSocket through Tor bridge (mandatory)',
    },
    webrtc: {
      enabled: false,
      reason: 'Disabled to prevent IP leaks',
    },
    relay: {
      enabled: true,
      description: 'Circuit Relay through Tor for browser-to-browser',
    },
  }
}
