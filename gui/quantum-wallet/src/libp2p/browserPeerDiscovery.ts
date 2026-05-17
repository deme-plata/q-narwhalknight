/**
 * Browser Peer Discovery - v3.5.8
 *
 * Enables browsers to discover and track other browser peers in the network.
 * Uses gossipsub to announce presence and discover other browsers.
 *
 * This allows the Network Map to show browser-to-browser connections
 * even when browsers connect through the bootstrap relay.
 */

import type { Libp2p } from 'libp2p'
import type { GossipSub } from '@libp2p/gossipsub'
import { multiaddr } from '@multiformats/multiaddr'
import { encode as msgpackEncode, decode as msgpackDecode } from '@msgpack/msgpack'
import { TOPICS, NETWORK_ID, PROTOCOL_VERSION, CONNECTION_CONFIG } from './config'

/**
 * Browser peer announcement message
 */
export interface BrowserPeerAnnouncement {
  // Peer ID of the announcing browser
  peerId: string

  // User agent / browser type
  userAgent: string

  // Current block height this browser has
  blockHeight: number

  // Number of peers this browser is connected to
  peerCount: number

  // Timestamp of announcement
  timestamp: number

  // Network ID for filtering
  networkId: string

  // Protocol version for compatibility
  protocolVersion: string

  // Whether this is a Tor browser
  isTorBrowser: boolean

  // Relay address through bootstrap (for dialing)
  relayAddress?: string
}

/**
 * Known browser peer info (tracked locally)
 */
export interface KnownBrowserPeer {
  peerId: string
  userAgent: string
  blockHeight: number
  peerCount: number
  lastSeen: number
  isTorBrowser: boolean
  relayAddress?: string
}

/**
 * Browser Peer Discovery Manager
 */
class BrowserPeerDiscoveryManager {
  private node: Libp2p | null = null
  private knownBrowserPeers: Map<string, KnownBrowserPeer> = new Map()
  private announceInterval: ReturnType<typeof setInterval> | null = null
  private cleanupInterval: ReturnType<typeof setInterval> | null = null
  private isInitialized = false
  private myPeerId: string | null = null
  private currentBlockHeight = 0

  // Configuration
  private readonly ANNOUNCE_INTERVAL = 30000 // Announce every 30 seconds
  private readonly PEER_EXPIRY = 120000 // Remove peers not seen for 2 minutes
  private readonly CLEANUP_INTERVAL = 60000 // Cleanup every minute

  // v10.2.3: Browser-to-browser mesh — dial discovered peers via circuit relay
  private readonly MAX_BROWSER_CONNECTIONS = 3 // Don't overwhelm relay with too many circuits
  private readonly DIAL_COOLDOWN = 60000 // Don't re-dial a peer within 60s
  private dialedPeers: Map<string, number> = new Map() // peerId → last dial attempt timestamp
  private connectedBrowserPeers: Set<string> = new Set()

  /**
   * Initialize the browser peer discovery
   */
  initialize(node: Libp2p): void {
    if (this.isInitialized) {
      console.warn('[BROWSER DISCOVERY] Already initialized')
      return
    }

    this.node = node
    this.myPeerId = node.peerId.toString()
    this.isInitialized = true

    console.log('🌐 [BROWSER DISCOVERY] Initializing browser peer discovery...')

    // Subscribe to browser peers topic
    const pubsub = node.services.pubsub as GossipSub
    pubsub.subscribe(TOPICS.BROWSER_PEERS)

    // Listen for browser peer announcements
    pubsub.addEventListener('message', this.handleMessage.bind(this))

    // v10.2.3: Track browser peer disconnections to refill mesh
    node.addEventListener('peer:disconnect', (event: any) => {
      const disconnectedPeerId = event.detail.toString()
      if (this.connectedBrowserPeers.delete(disconnectedPeerId)) {
        console.log(`🌐 [BROWSER MESH] Browser peer disconnected: ${disconnectedPeerId.substring(0, 12)}... (mesh: ${this.connectedBrowserPeers.size}/${this.MAX_BROWSER_CONNECTIONS})`)
      }
    })

    console.log(`🌐 [BROWSER DISCOVERY] Subscribed to ${TOPICS.BROWSER_PEERS}`)
  }

  /**
   * Start announcing and discovering
   */
  start(): void {
    if (!this.isInitialized || !this.node) {
      console.error('[BROWSER DISCOVERY] Not initialized')
      return
    }

    console.log('🌐 [BROWSER DISCOVERY] Starting browser peer discovery...')

    // Announce immediately
    this.announce()

    // Set up periodic announcement
    this.announceInterval = setInterval(() => {
      this.announce()
    }, this.ANNOUNCE_INTERVAL)

    // Set up periodic cleanup
    this.cleanupInterval = setInterval(() => {
      this.cleanupExpiredPeers()
    }, this.CLEANUP_INTERVAL)

    console.log('🌐 [BROWSER DISCOVERY] Browser peer discovery started')
  }

  /**
   * Stop discovery
   */
  stop(): void {
    if (this.announceInterval) {
      clearInterval(this.announceInterval)
      this.announceInterval = null
    }
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval)
      this.cleanupInterval = null
    }

    console.log('🌐 [BROWSER DISCOVERY] Browser peer discovery stopped')
  }

  /**
   * Update current block height (called by useRealtimeBlocks)
   */
  setBlockHeight(height: number): void {
    this.currentBlockHeight = height
  }

  /**
   * Get all known browser peers
   */
  getKnownBrowserPeers(): KnownBrowserPeer[] {
    return Array.from(this.knownBrowserPeers.values())
  }

  /**
   * Get count of known browser peers
   */
  getBrowserPeerCount(): number {
    return this.knownBrowserPeers.size
  }

  /**
   * Announce this browser's presence
   */
  private async announce(): Promise<void> {
    if (!this.node || !this.myPeerId) return

    try {
      const pubsub = this.node.services.pubsub as GossipSub
      const connections = this.node.getConnections()

      // Detect Tor browser
      const isTorBrowser = this.detectTorBrowser()

      // Build relay address through bootstrap
      let relayAddress: string | undefined
      for (const conn of connections) {
        const remoteAddr = conn.remoteAddr.toString()
        if (remoteAddr.includes('/wss/')) {
          // Extract bootstrap peer ID from connection
          const match = remoteAddr.match(/\/p2p\/([^/]+)$/)
          if (match) {
            const bootstrapPeerId = match[1]
            // Our relay address is: /p2p/BOOTSTRAP/p2p-circuit/p2p/OUR_PEER_ID
            relayAddress = `/p2p/${bootstrapPeerId}/p2p-circuit/p2p/${this.myPeerId}`
            break
          }
        }
      }

      const announcement: BrowserPeerAnnouncement = {
        peerId: this.myPeerId,
        userAgent: navigator.userAgent.substring(0, 100),
        blockHeight: this.currentBlockHeight,
        peerCount: connections.length,
        timestamp: Date.now(),
        networkId: NETWORK_ID,
        protocolVersion: PROTOCOL_VERSION,
        isTorBrowser,
        relayAddress,
      }

      const encoded = msgpackEncode(announcement)
      await pubsub.publish(TOPICS.BROWSER_PEERS, encoded)

      console.log(`🌐 [BROWSER DISCOVERY] Announced presence (height=${this.currentBlockHeight}, peers=${connections.length})`)
    } catch (error) {
      console.error('[BROWSER DISCOVERY] Failed to announce:', error)
    }
  }

  /**
   * Handle incoming messages
   */
  private handleMessage(event: any): void {
    const topic = event.detail.topic
    if (topic !== TOPICS.BROWSER_PEERS) return

    try {
      const data = event.detail.data
      const announcement = msgpackDecode(data) as BrowserPeerAnnouncement

      // Ignore our own announcements
      if (announcement.peerId === this.myPeerId) return

      // Validate network ID
      if (announcement.networkId !== NETWORK_ID) {
        console.warn(`[BROWSER DISCOVERY] Ignoring peer from different network: ${announcement.networkId}`)
        return
      }

      // Update known peers
      this.knownBrowserPeers.set(announcement.peerId, {
        peerId: announcement.peerId,
        userAgent: announcement.userAgent,
        blockHeight: announcement.blockHeight,
        peerCount: announcement.peerCount,
        lastSeen: Date.now(),
        isTorBrowser: announcement.isTorBrowser,
        relayAddress: announcement.relayAddress,
      })

      console.log(`🌐 [BROWSER DISCOVERY] Discovered browser peer: ${announcement.peerId.substring(0, 12)}... (height=${announcement.blockHeight})`)

      // v10.2.3: Try to connect to discovered browser peer via circuit relay
      this.maybeDialBrowserPeer(announcement)

      // Emit custom event for UI updates
      window.dispatchEvent(new CustomEvent('browser-peer-discovered', {
        detail: { peerId: announcement.peerId, peerCount: this.knownBrowserPeers.size }
      }))
    } catch (error) {
      console.error('[BROWSER DISCOVERY] Failed to parse announcement:', error)
    }
  }

  /**
   * v10.2.3: Dial a discovered browser peer via circuit relay
   * This forms the browser-to-browser gossipsub mesh, so blocks propagate
   * peer-to-peer without every browser needing to get them from Epsilon.
   */
  private async maybeDialBrowserPeer(announcement: BrowserPeerAnnouncement): Promise<void> {
    if (!this.node || !announcement.relayAddress) return

    const peerId = announcement.peerId

    // Already connected to this peer?
    if (this.connectedBrowserPeers.has(peerId)) return

    // Enough browser connections already?
    if (this.connectedBrowserPeers.size >= this.MAX_BROWSER_CONNECTIONS) return

    // Recently tried this peer?
    const lastDial = this.dialedPeers.get(peerId) || 0
    if (Date.now() - lastDial < this.DIAL_COOLDOWN) return

    // Already connected via libp2p?
    try {
      const conns = this.node.getConnections()
      for (const conn of conns) {
        if (conn.remotePeer.toString() === peerId) {
          this.connectedBrowserPeers.add(peerId)
          return // Already connected
        }
      }
    } catch { /* ignore */ }

    // Mark dial attempt
    this.dialedPeers.set(peerId, Date.now())

    try {
      const relayAddr = multiaddr(announcement.relayAddress)
      console.log(`🌐 [BROWSER MESH] Dialing browser peer ${peerId.substring(0, 12)}... via relay`)
      console.log(`   Address: ${announcement.relayAddress}`)

      await this.node.dial(relayAddr, {
        signal: AbortSignal.timeout(15000), // 15s timeout for relay circuit setup
      })

      this.connectedBrowserPeers.add(peerId)
      console.log(`✅ [BROWSER MESH] Connected to browser peer ${peerId.substring(0, 12)}...!`)
      console.log(`   Browser mesh: ${this.connectedBrowserPeers.size}/${this.MAX_BROWSER_CONNECTIONS} peers`)
      console.log(`   Gossipsub will now relay blocks through this peer`)

      // Emit event for UI
      window.dispatchEvent(new CustomEvent('browser-mesh-peer-connected', {
        detail: { peerId, meshSize: this.connectedBrowserPeers.size }
      }))
    } catch (err) {
      // Expected to fail sometimes (peer offline, relay busy, etc.)
      console.log(`🌐 [BROWSER MESH] Could not dial ${peerId.substring(0, 12)}...: ${err}`)
    }
  }

  /**
   * Remove peers not seen recently
   */
  private cleanupExpiredPeers(): void {
    const now = Date.now()
    const expiredPeers: string[] = []

    for (const [peerId, peer] of this.knownBrowserPeers) {
      if (now - peer.lastSeen > this.PEER_EXPIRY) {
        expiredPeers.push(peerId)
      }
    }

    for (const peerId of expiredPeers) {
      this.knownBrowserPeers.delete(peerId)
      console.log(`🌐 [BROWSER DISCOVERY] Removed expired peer: ${peerId.substring(0, 12)}...`)
    }

    if (expiredPeers.length > 0) {
      // Emit update event
      window.dispatchEvent(new CustomEvent('browser-peers-updated', {
        detail: { peerCount: this.knownBrowserPeers.size }
      }))
    }
  }

  /**
   * Detect if user is using Tor Browser
   */
  private detectTorBrowser(): boolean {
    try {
      // Check 1: WebRTC disabled (Tor Browser disables it)
      const rtcDisabled = typeof RTCPeerConnection === 'undefined' ||
        typeof navigator.mediaDevices === 'undefined'

      // Check 2: User Agent contains Tor indicators
      const ua = navigator.userAgent.toLowerCase()
      const torUserAgent = ua.includes('tor') || ua.includes('torbrowser')

      // Check 3: Firefox ESR (Tor Browser is based on Firefox ESR)
      const isFirefoxESR = ua.includes('firefox/') && !ua.includes('chrome')

      if (rtcDisabled && isFirefoxESR) return true
      if (torUserAgent) return true

      return false
    } catch {
      return false
    }
  }

  /**
   * Get stats for debugging
   */
  getStats(): object {
    return {
      initialized: this.isInitialized,
      myPeerId: this.myPeerId?.substring(0, 12) + '...',
      knownBrowserPeers: this.knownBrowserPeers.size,
      connectedBrowserPeers: this.connectedBrowserPeers.size,
      maxBrowserConnections: this.MAX_BROWSER_CONNECTIONS,
      currentBlockHeight: this.currentBlockHeight,
      peers: Array.from(this.knownBrowserPeers.values()).map(p => ({
        peerId: p.peerId.substring(0, 12) + '...',
        blockHeight: p.blockHeight,
        lastSeen: new Date(p.lastSeen).toISOString(),
        isTorBrowser: p.isTorBrowser,
        connected: this.connectedBrowserPeers.has(p.peerId),
      })),
    }
  }

  /** v10.2.3: Get connected browser mesh peer count */
  getBrowserMeshSize(): number {
    return this.connectedBrowserPeers.size
  }
}

// Export singleton instance
export const browserPeerDiscovery = new BrowserPeerDiscoveryManager()

// Export convenience functions
export function getKnownBrowserPeers(): KnownBrowserPeer[] {
  return browserPeerDiscovery.getKnownBrowserPeers()
}

export function getBrowserPeerCount(): number {
  return browserPeerDiscovery.getBrowserPeerCount()
}
