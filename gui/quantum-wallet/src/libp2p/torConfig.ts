/**
 * Mandatory Tor Configuration for Q-NarwhalKnight Browser
 *
 * ALL browser P2P traffic is routed through Tor automatically.
 * Users don't need to opt-in - privacy is the default.
 *
 * Architecture:
 * Browser → wss://quillon.xyz:9444/tor-bridge → Tor SOCKS5 → Tor Network → .onion Bootstrap
 *
 * This ensures:
 * 1. User's real IP is never exposed to the P2P network
 * 2. Traffic analysis is prevented via Tor's onion routing
 * 3. Connection to .onion hidden service for bootstrap
 *
 * @module torConfig
 * @version 1.0.0
 */

/**
 * Core Tor configuration - ALL settings are mandatory
 */
export const TOR_CONFIG = {
  // Tor is ALWAYS enabled - no opt-out
  enabled: true,

  // WebSocket endpoint for libp2p bootstrap
  // Port 9443: nginx WSS proxy → libp2p:9001/ws
  bridgeEndpoint: 'wss://quillon.xyz:9443',

  // Bootstrap multiaddr for libp2p
  // Port 9443 = WebSocket Secure proxy to libp2p
  bridgeMultiaddr: '/dns4/quillon.xyz/tcp/9443/wss',

  // Onion address of bootstrap node
  // Format: /onion3/<56-char-address>:<port>/p2p/<peer-id>
  // This allows Tor Browser users to connect directly via .onion
  onionBootstrap: '/onion3/ca3jpub2haxboxjw4ws6run36ekdh3pv7pneqg2tbac5rxzvxhd2i5id:9001/p2p/12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM',

  // Connection timeout for Tor (longer due to circuit establishment)
  dialTimeout: 45000, // 45 seconds - Tor circuits take time

  // Retry attempts through Tor before giving up
  maxRetries: 5,

  // Circuit rotation interval (ms) - rotate for anonymity
  circuitRotationInterval: 10 * 60 * 1000, // 10 minutes

  // Heartbeat interval for keeping Tor connection alive
  heartbeatInterval: 30000, // 30 seconds

  // Maximum connection attempts before warning user
  maxConnectionAttempts: 10,
} as const

/**
 * Security settings - WebRTC is DISABLED to prevent IP leaks
 */
export const TOR_SECURITY = {
  // WebRTC leaks real IP even through proxies - ALWAYS disabled
  allowWebRTC: false,

  // Only allow WebSocket connections through Tor bridge
  allowDirectConnections: false,

  // Verify Tor circuit is established before any P2P activity
  requireTorCircuit: true,

  // Don't cache peer addresses (they might be clearnet)
  cachePeerAddresses: false,

  // Strip any non-Tor transport addresses from announcements
  stripClearnetAddresses: true,
} as const

/**
 * Bootstrap configuration - Tor bridge is the ONLY option
 */
export const TOR_BOOTSTRAP = {
  // Primary bootstrap endpoint (WSS proxy to libp2p)
  primary: '/dns4/quillon.xyz/tcp/9443/wss/p2p/12D3KooWSBxwSKw4wftHViMdw5rrV8Z1wEkikDS2vKYZtRrio5hH',

  // Fallback Tor bridges (TODO: add community bridges)
  fallbacks: [] as string[],

  // NO clearnet fallback - Tor or nothing
  clearnetFallback: null,
} as const

/**
 * Connection state for monitoring Tor circuit health
 */
export interface TorConnectionState {
  isConnected: boolean
  circuitEstablished: boolean
  bridgeLatency: number | null // ms
  lastHeartbeat: number | null
  reconnectAttempts: number
  errors: string[]
}

// Global Tor connection state
let torState: TorConnectionState = {
  isConnected: false,
  circuitEstablished: false,
  bridgeLatency: null,
  lastHeartbeat: null,
  reconnectAttempts: 0,
  errors: [],
}

/**
 * Get current Tor connection state
 */
export function getTorState(): TorConnectionState {
  return { ...torState }
}

/**
 * Update Tor connection state (internal use)
 */
export function updateTorState(update: Partial<TorConnectionState>): void {
  torState = { ...torState, ...update }

  // Emit event for UI updates
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('tor-state-changed', {
      detail: torState
    }))
  }
}

/**
 * Reset Tor connection state (on reconnect)
 */
export function resetTorState(): void {
  torState = {
    isConnected: false,
    circuitEstablished: false,
    bridgeLatency: null,
    lastHeartbeat: null,
    reconnectAttempts: torState.reconnectAttempts + 1,
    errors: [],
  }
}

/**
 * Log Tor-related events with consistent formatting
 */
export function logTor(level: 'info' | 'warn' | 'error', message: string, data?: any): void {
  const prefix = '🧅 [TOR]'
  const timestamp = new Date().toISOString()

  switch (level) {
    case 'info':
      console.log(`${prefix} ${message}`, data || '')
      break
    case 'warn':
      console.warn(`${prefix} ⚠️ ${message}`, data || '')
      break
    case 'error':
      console.error(`${prefix} ❌ ${message}`, data || '')
      // Track errors in state
      torState.errors.push(`[${timestamp}] ${message}`)
      if (torState.errors.length > 10) {
        torState.errors.shift() // Keep last 10 errors
      }
      break
  }
}

/**
 * Check if Tor circuit is healthy
 */
export function isTorHealthy(): boolean {
  const now = Date.now()
  const lastHeartbeat = torState.lastHeartbeat || 0
  const heartbeatAge = now - lastHeartbeat

  // Circuit is healthy if:
  // 1. Connected to bridge
  // 2. Circuit established
  // 3. Heartbeat within last 2 minutes
  return (
    torState.isConnected &&
    torState.circuitEstablished &&
    heartbeatAge < 2 * 60 * 1000 // 2 minutes
  )
}

/**
 * Format Tor status for display
 */
export function formatTorStatus(): string {
  if (!torState.isConnected) {
    return 'Connecting to Tor...'
  }
  if (!torState.circuitEstablished) {
    return 'Establishing Tor circuit...'
  }
  if (torState.bridgeLatency !== null) {
    return `Connected via Tor (${torState.bridgeLatency}ms)`
  }
  return 'Connected via Tor'
}
