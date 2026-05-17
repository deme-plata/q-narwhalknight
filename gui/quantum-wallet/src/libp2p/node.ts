/**
 * Browser LibP2P Node - Core P2P Functionality
 *
 * 🧅 MANDATORY TOR ROUTING - v3.6.0
 * 🔐 POST-QUANTUM CRYPTOGRAPHY - v3.7.4
 *
 * This module creates and manages the browser's libp2p node,
 * transforming the React app into a first-class P2P peer.
 *
 * ALL traffic is routed through Tor automatically - users don't opt-in,
 * privacy is the default. This ensures:
 * 1. User's real IP is never exposed to the P2P network
 * 2. Traffic analysis is prevented via Tor's onion routing
 * 3. Connection to .onion hidden service for bootstrap
 *
 * POST-QUANTUM SECURITY (v3.7.4):
 * - Dilithium5 signatures for authentication (NIST Level 5)
 * - Kyber1024 key exchange for session encryption
 * - Hybrid mode: Classical + Post-Quantum for maximum security
 * - Future-proof against quantum computer attacks
 *
 * Architecture:
 * - Transport: WebSocket through Tor bridge (wss://quillon.xyz:9444)
 * - Security: Hybrid (Noise + PQ-Noise with Dilithium5/Kyber1024) + Tor
 * - Multiplexing: Yamux (stream multiplexing)
 * - PubSub: Gossipsub (real-time messaging)
 * - DHT: Kademlia (peer discovery, light mode)
 * - Protocols: Identify, Ping, Bootstrap
 * - NO WebRTC: Disabled to prevent IP leaks
 */

import { createLibp2p } from 'libp2p'
import type { Libp2p } from 'libp2p'
import { noise } from '@chainsafe/libp2p-noise'
import { yamux } from '@chainsafe/libp2p-yamux'
import { gossipsub } from '@libp2p/gossipsub'
import type { GossipSub } from '@libp2p/gossipsub'
import { kadDHT } from '@libp2p/kad-dht'
import { bootstrap } from '@libp2p/bootstrap'
import { identify } from '@libp2p/identify'
import { ping } from '@libp2p/ping'
import { multiaddr } from '@multiformats/multiaddr'
import { encode as msgpackEncode, decode as msgpackDecode } from '@msgpack/msgpack'

import { createTransports, getTransportStats } from './transports'
import {
  BOOTSTRAP_PEERS,
  CONNECTION_CONFIG,
  DHT_CONFIG,
  GOSSIPSUB_CONFIG,
  NETWORK_ID,
  PROTOCOL_VERSION,
  TOPICS,
  PERFORMANCE_CONFIG,
} from './config'
import { DECODER_METRICS, getDecoderMetricsSummary } from './decoder'
import { telemetryReporter } from './telemetry'
import { verificationReporter } from './verificationReporter'
import { blockServer } from './blockServer'
import { blockCache, getBlockCacheStats } from './blockCache'
import { getRelayStats } from '../hooks/useRealtimeBlocks'
import { browserPeerDiscovery, getKnownBrowserPeers, getBrowserPeerCount } from './browserPeerDiscovery'
import { propagationQueue, getPropagationStats } from './blockPropagationQueue'
import { p2pDataService, getP2PDataStats } from './p2pDataService'

// 🧅 Tor integration imports
import {
  TOR_CONFIG,
  logTor,
  updateTorState,
  getTorState,
  formatTorStatus,
  isTorHealthy,
} from './torConfig'
import {
  onTorConnected,
  onTorDisconnected,
  startTorHealthMonitor,
  getTorTransportStats,
} from './torTransport'

// 🔐 v3.7.4: Post-Quantum Cryptography imports
import {
  loadPQCrypto,
  isPQCryptoAvailable,
  getPQCryptoStatus,
  generateHybridKeypair,
  type HybridKeypair,
} from './postQuantumCrypto'
import { pqNoise, getPQNoiseStatus } from './pqConnectionEncrypter'

/**
 * Helper to get pubsub service with proper typing
 */
function getPubSub(node: Libp2p): GossipSub {
  return node.services.pubsub as GossipSub
}

/**
 * Helper to get ping service with proper typing
 */
function getPing(node: Libp2p): any {
  return node.services.ping
}

/**
 * Browser P2P Node State
 */
export interface BrowserNodeState {
  node: Libp2p | null
  peerId: string | null
  peerCount: number
  topics: string[]
  isStarted: boolean
}

/**
 * v3.5.22: Connection Pre-warming ⚡
 *
 * Pre-warms connections by initiating WebSocket connections to bootstrap peers
 * BEFORE the libp2p node is fully initialized. This reduces perceived latency
 * by starting the TCP/TLS handshake early.
 *
 * The actual libp2p connection will reuse the warmed socket or establish
 * faster due to DNS caching and connection pooling.
 */
let prewarmPromise: Promise<void> | null = null
let prewarmResults: { url: string; success: boolean; latency: number }[] = []

export function prewarmConnections(): void {
  if (prewarmPromise) return // Already prewarming

  console.log('⚡ [PREWARM] Starting connection pre-warming...')
  const startTime = performance.now()

  prewarmPromise = (async () => {
    const results: { url: string; success: boolean; latency: number }[] = []

    // Extract WebSocket URLs from bootstrap multiaddrs
    for (const peerAddr of BOOTSTRAP_PEERS) {
      try {
        const parts = peerAddr.split('/')
        const hostIndex = parts.indexOf('dns4') + 1
        const portIndex = parts.indexOf('tcp') + 1
        const host = parts[hostIndex]
        const port = parts[portIndex]
        const protocol = parts.includes('wss') ? 'wss' : 'ws'

        const wsUrl = `${protocol}://${host}:${port}`

        // Create a test WebSocket connection to warm the socket
        const ws = new WebSocket(wsUrl)
        const connStart = performance.now()

        await new Promise<void>((resolve) => {
          const timeout = setTimeout(() => {
            ws.close()
            results.push({ url: wsUrl, success: false, latency: -1 })
            resolve()
          }, 5000) // 5 second timeout for prewarm

          ws.onopen = () => {
            const latency = Math.round(performance.now() - connStart)
            console.log(`⚡ [PREWARM] ${host}:${port} ready (${latency}ms)`)
            results.push({ url: wsUrl, success: true, latency })
            clearTimeout(timeout)
            ws.close() // Close immediately - we just wanted to warm the connection
            resolve()
          }

          ws.onerror = () => {
            results.push({ url: wsUrl, success: false, latency: -1 })
            clearTimeout(timeout)
            resolve()
          }
        })
      } catch (err) {
        console.warn('⚡ [PREWARM] Failed to parse peer address:', peerAddr)
      }
    }

    prewarmResults = results
    const totalTime = Math.round(performance.now() - startTime)
    const successCount = results.filter(r => r.success).length
    console.log(`⚡ [PREWARM] Complete: ${successCount}/${results.length} connections warmed in ${totalTime}ms`)
  })()
}

/**
 * Get pre-warming results (for diagnostics)
 */
export function getPrewarmResults() {
  return {
    completed: prewarmPromise !== null,
    results: prewarmResults,
  }
}

/**
 * v3.7.4: Post-quantum keypair for this browser node
 * Generated once at startup, used for all PQ authentication
 */
let browserPQKeypair: HybridKeypair | null = null

/**
 * Get the browser's post-quantum keypair
 * Returns null if PQ crypto is not initialized
 */
export function getBrowserPQKeypair(): HybridKeypair | null {
  return browserPQKeypair
}

/**
 * Create and initialize a browser P2P node
 *
 * This is the main entry point for creating the libp2p node.
 * It configures all transports, protocols, and services needed
 * for the browser to participate in the P2P network.
 *
 * v3.7.4: Now includes post-quantum cryptography initialization
 * with Dilithium5 signatures and Kyber1024 key exchange.
 *
 * @returns Promise<Libp2p> - The initialized libp2p node
 */
export async function createBrowserNode(): Promise<Libp2p> {
  console.log('🚀 [LIBP2P] Initializing browser P2P node...')
  console.log(`📡 [LIBP2P] Network: ${NETWORK_ID}`)
  console.log(`📡 [LIBP2P] Protocol Version: ${PROTOCOL_VERSION}`)

  // 🔐 v3.7.4: Initialize post-quantum cryptography
  console.log('🔐 [LIBP2P] Initializing post-quantum cryptography...')
  const pqStartTime = performance.now()

  try {
    await loadPQCrypto()
    if (isPQCryptoAvailable()) {
      // Generate hybrid keypair (Ed25519 + Dilithium5 + Kyber1024)
      browserPQKeypair = await generateHybridKeypair()
      const pqElapsed = Math.round(performance.now() - pqStartTime)
      const pqStatus = getPQCryptoStatus()
      console.log(`✅ [LIBP2P] Post-quantum crypto initialized in ${pqElapsed}ms`)
      console.log(`   🔐 Algorithm: Dilithium5 (signatures) + Kyber1024 (key exchange)`)
      console.log(`   🛡️ Security: NIST Level 5 (256-bit post-quantum)`)
      console.log(`   📊 Crypto type: ${pqStatus.type}`)
    } else {
      console.warn('⚠️ [LIBP2P] Post-quantum crypto not available, using classical only')
    }
  } catch (pqError) {
    console.warn('⚠️ [LIBP2P] Failed to initialize post-quantum crypto:', pqError)
    console.warn('   Falling back to classical cryptography')
  }

  // ⚡ v3.5.22: Start connection pre-warming immediately
  // This warms TCP/TLS connections while we set up the node
  prewarmConnections()

  // 🧅 Log Tor status
  logTor('info', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
  logTor('info', 'MANDATORY TOR ROUTING ENABLED')
  logTor('info', 'All P2P traffic will be routed through Tor')
  logTor('info', 'Your IP address is protected from the network')
  logTor('info', `Bridge endpoint: ${TOR_CONFIG.bridgeEndpoint}`)
  logTor('info', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')

  try {
    const node = await createLibp2p({
      // Addresses: Browser nodes listen via circuit relay (through bootstrap)
      // This makes the browser reachable by other browsers via relay
      // v10.2.3: ENABLED — previously empty, which meant browsers couldn't
      // receive incoming connections from other browsers. Now browsers listen
      // on circuit relay through the bootstrap node, forming a real P2P mesh.
      addresses: {
        // v10.3.0: Listen addresses empty initially — circuit relay is added
        // AFTER connecting to bootstrap. Setting it here before connection
        // causes a fatal "failed to listen" error.
        listen: [],
        announce: [],
      },

      // v10.3.0: Don't crash if a listen address fails (e.g. relay not ready)
      transportManager: {
        faultTolerance: 1, // NO_FATAL — log warning but don't crash
      },

      // Transport Layer: WebSocket + Circuit Relay
      transports: createTransports(),

      // Connection Encryption: Noise protocol
      // v3.7.4: PQ-Noise encrypter prepared but disabled until backend supports it
      // To enable: connectionEncrypters: [pqNoise(), noise()],
      // The pqNoise() uses Kyber1024 key exchange + Dilithium5 authentication
      // Falls back to classical Noise when connecting to non-PQ peers
      connectionEncrypters: [noise()],

      // Stream Multiplexing: Yamux
      streamMuxers: [yamux()],

      // Peer Discovery: Bootstrap nodes
      peerDiscovery: [
        bootstrap({
          list: BOOTSTRAP_PEERS,
          timeout: CONNECTION_CONFIG.DIAL_TIMEOUT,
        }),
      ],

      // Connection Manager: Control connection lifecycle
      connectionManager: {
        maxConnections: CONNECTION_CONFIG.MAX_CONNECTIONS,
      },

      // Services: Core libp2p protocols
      services: {
        // Identify: Exchange peer info
        identify: identify(),

        // Ping: Keep connections alive
        // Use default /ipfs/ping/1.0.0 to match Rust libp2p server
        ping: ping(),

        // PubSub: Gossipsub for real-time messaging
        pubsub: gossipsub({
          emitSelf: false, // Don't receive our own messages
          floodPublish: GOSSIPSUB_CONFIG.FLOOD_PUBLISH,
          fallbackToFloodsub: true, // Ensure messages reach all subscribers even without mesh
          allowPublishToZeroTopicPeers: true, // Don't error when no mesh peers
          // Mesh parameters - tuned for browser with 1 bootstrap peer
          D: GOSSIPSUB_CONFIG.D,
          Dlo: 1, // Allow mesh with just 1 peer (bootstrap server)
          Dhi: GOSSIPSUB_CONFIG.D_HIGH,
          heartbeatInterval: GOSSIPSUB_CONFIG.HEARTBEAT_INTERVAL,
          seenTTL: GOSSIPSUB_CONFIG.SEEN_TTL,
          // Ensure we accept messages from the server (strict signing)
          globalSignaturePolicy: 'StrictSign' as any,
        }),

        // DHT: Kademlia for peer/content discovery (light mode)
        dht: kadDHT({
          clientMode: DHT_CONFIG.CLIENT_MODE, // Don't store DHT data
          kBucketSize: DHT_CONFIG.K_BUCKET_SIZE,
        }),
      },
    })

    // Start the node
    await node.start()

    const peerId = node.peerId.toString()
    console.log('✅ [LIBP2P] Node started successfully!')
    console.log(`🆔 [LIBP2P] Peer ID: ${peerId}`)
    console.log(`📊 [LIBP2P] Bootstrap peers configured: ${BOOTSTRAP_PEERS.length}`)

    // 🧅 Eager dial to Tor bridge bootstrap
    // Use longer timeout for Tor (circuit establishment takes time)
    logTor('info', 'Connecting to Tor bridge...')
    const dialStartTime = performance.now()

    for (const peerAddr of BOOTSTRAP_PEERS) {
      try {
        const ma = multiaddr(peerAddr)
        // Longer timeout for Tor connections (45s instead of 10s)
        node.dial(ma, { signal: AbortSignal.timeout(TOR_CONFIG.dialTimeout) })
          .then(() => {
            const latency = Math.round(performance.now() - dialStartTime)
            logTor('info', `Connected to Tor bridge (${latency}ms)`)
            onTorConnected(latency)
          })
          .catch((err) => {
            logTor('error', `Failed to connect to Tor bridge: ${err.message}`)
            onTorDisconnected(err.message)
          })
      } catch (err) {
        logTor('error', `Invalid Tor bridge address: ${peerAddr}`, err)
      }
    }

    // Register custom protocols (handshake, block-sync, etc.)
    const { registerProtocols } = await import('./protocols')
    await registerProtocols(node)

    // Set up event listeners
    setupEventListeners(node)

    // Set up graceful shutdown
    setupGracefulShutdown(node)

    // v3.5.x: Initialize Browser P2P Network Contribution features
    console.log('🚀 [LIBP2P] Initializing Browser P2P Network Contribution features...')

    // Initialize telemetry reporter
    telemetryReporter.initialize(node)
    telemetryReporter.start()
    console.log('📊 [LIBP2P] Telemetry reporter started')

    // Initialize verification reporter
    verificationReporter.initialize(node)
    console.log('🔍 [LIBP2P] Verification reporter initialized')

    // Initialize block server for serving blocks to peers
    await blockServer.start(node)
    console.log('🗄️ [LIBP2P] Block server started')

    // v3.5.23: Initialize block propagation queue for faster P2P sync
    propagationQueue.initialize(node)
    console.log('📦 [LIBP2P] Block propagation queue started')

    // v3.5.24: Initialize P2P data service for P2P-first data fetching
    p2pDataService.initialize(node)
    console.log('🌐 [LIBP2P] P2P data service started')

    // Subscribe to topics during node initialization
    // CRITICAL: Topics MUST be subscribed here (not deferred to React hooks)
    // to ensure gossipsub mesh forms immediately after connection
    //
    // v10.2.3: Subscribe to blocks, transactions, and browser-peers.
    // DO NOT subscribe to peer-heights — it generates ~900 msgs/min and floods
    // the Rust node's per-peer gossipsub send queue, causing "Send Queue full"
    // drops that BLOCK delivery of actual block messages (55% of queue pressure).
    // Browsers don't need peer-heights (they don't sync the chain).
    const pubsub = getPubSub(node)
    pubsub.subscribe(TOPICS.BLOCKS)
    pubsub.subscribe(TOPICS.TRANSACTIONS)
    pubsub.subscribe(TOPICS.BROWSER_PEERS)
    console.log('📡 [LIBP2P] Subscribed to blocks + transactions + browser-peers')
    console.log('⚡ [LIBP2P] Gossipsub mesh pre-warmed for block delivery')

    // v3.5.8: Initialize browser peer discovery
    browserPeerDiscovery.initialize(node)
    browserPeerDiscovery.start()
    console.log('🌐 [LIBP2P] Browser peer discovery started')

    // 🧅 Start Tor health monitoring
    const stopTorMonitor = startTorHealthMonitor((healthy) => {
      if (!healthy) {
        logTor('warn', 'Tor circuit unhealthy - may need to reconnect')
      }
    })
    logTor('info', 'Tor health monitor started')

    // Store cleanup function for shutdown
    ;(node as any)._torMonitorCleanup = stopTorMonitor

    return node
  } catch (error) {
    console.error('❌ [LIBP2P] Failed to create node:', error)
    throw error
  }
}

/**
 * Set up event listeners for node events
 *
 * Monitors connection state, peer discovery, and protocol events
 *
 * @param node - The libp2p node
 */
function setupEventListeners(node: Libp2p) {
  // Peer discovery events
  node.addEventListener('peer:discovery', (event) => {
    const peerId = event.detail.id.toString()
    console.log('👤 [LIBP2P] Discovered peer:', peerId.substring(0, 16) + '...')
  })

  // Connection events - 🧅 Track Tor status
  node.addEventListener('peer:connect', (event) => {
    const peerId = event.detail.toString()
    const connections = node.getConnections(event.detail)

    // Check connection type
    let isRelayConnection = false
    let isTorBridge = false
    for (const conn of connections) {
      const remoteAddr = conn.remoteAddr.toString()
      if (remoteAddr.includes('/p2p-circuit/')) {
        isRelayConnection = true
      }
      // Check if this is a bootstrap connection (port 9443 or 9444)
      if (remoteAddr.includes('/tcp/9443/') || remoteAddr.includes('/tcp/9444/') || remoteAddr.includes('quillon.xyz')) {
        isTorBridge = true
      }
    }

    if (isTorBridge) {
      // 🧅 Tor bridge connection established
      logTor('info', `Connected to bootstrap via Tor`)
      onTorConnected()
    } else if (isRelayConnection) {
      logTor('info', `Connected to browser peer via Tor relay: ${peerId.substring(0, 12)}...`)
    } else {
      console.log('🔗 [LIBP2P] Connected to peer:', peerId.substring(0, 16) + '...')
    }
    console.log(`📊 [LIBP2P] Total connections: ${node.getConnections().length}`)
  })

  node.addEventListener('peer:disconnect', (event) => {
    const peerId = event.detail.toString()
    const remainingConnections = node.getConnections().length

    console.log('❌ [LIBP2P] Disconnected from peer:', peerId.substring(0, 16) + '...')
    console.log(`📊 [LIBP2P] Remaining connections: ${remainingConnections}`)

    // 🧅 Track Tor disconnection if no connections remain
    if (remainingConnections === 0) {
      logTor('warn', 'All Tor connections lost - attempting reconnect')
      onTorDisconnected('No connections remaining')
    }
  })

  // Protocol events (identify)
  node.addEventListener('peer:identify', (event) => {
    const peerId = event.detail.peerId.toString()
    const protocols = event.detail.protocols
    console.log('🔍 [LIBP2P] Identified peer:', peerId.substring(0, 16) + '...')
    console.log('   Protocols:', protocols.slice(0, 5).join(', '))
  })

  // Gossipsub events
  const pubsub = getPubSub(node)
  pubsub.addEventListener('subscription-change', (event: any) => {
    console.log('📡 [LIBP2P] Subscription change:', event.detail)
  })
  pubsub.addEventListener('gossipsub:graft', (event: any) => {
    console.log('🌿 [LIBP2P] Gossipsub GRAFT (mesh formed):', event.detail)
  })
  pubsub.addEventListener('gossipsub:prune', (event: any) => {
    console.log('✂️ [LIBP2P] Gossipsub PRUNE:', event.detail)
  })

  // Global message listener to verify gossipsub delivery
  pubsub.addEventListener('message', (event: any) => {
    const topic = event.detail?.topic || 'unknown'
    const dataSize = event.detail?.data?.length || 0
    const from = event.detail?.from?.toString()?.substring(0, 16) || 'unknown'
    console.log(`📬 [LIBP2P] Gossipsub message received: topic=${topic}, size=${dataSize}B, from=${from}`)
  })

  // Log initial state + gossipsub mesh status
  setTimeout(() => {
    const connections = node.getConnections()
    const topics = pubsub.getTopics()
    const peers = pubsub.getPeers()
    console.log(`📊 [LIBP2P] Current state:`)
    console.log(`   Connections: ${connections.length}`)
    console.log(`   Peer Store Size: ${node.getPeers().length}`)
    console.log(`   Gossipsub topics: ${topics.join(', ')}`)
    console.log(`   Gossipsub peers: ${peers.length} [${peers.map((p: any) => p.toString().substring(0, 12)).join(', ')}]`)

    // Check mesh status for each topic
    for (const topic of topics) {
      try {
        const meshPeers = (pubsub as any).getMeshPeers?.(topic) || []
        console.log(`   Mesh[${topic.split('/').pop()}]: ${meshPeers.length} peers`)
      } catch (e) {
        // getMeshPeers might not be available
      }
    }
  }, 5000) // Give time for initial connections
}

/**
 * Set up graceful shutdown handler
 *
 * Ensures clean disconnection when the page is closed
 *
 * @param node - The libp2p node
 */
function setupGracefulShutdown(node: Libp2p) {
  // Handle page unload
  window.addEventListener('beforeunload', async () => {
    console.log('🛑 [LIBP2P] Shutting down node...')

    try {
      // 🧅 Stop Tor health monitor
      const torCleanup = (node as any)._torMonitorCleanup
      if (torCleanup) {
        torCleanup()
        logTor('info', 'Tor health monitor stopped')
      }

      // v3.5.x: Stop Browser P2P Network Contribution features
      telemetryReporter.stop()
      browserPeerDiscovery.stop()
      propagationQueue.stop()
      await blockServer.stop()
      console.log('🛑 [LIBP2P] P2P contribution features stopped')

      // Unsubscribe from all PubSub topics
      const pubsub = getPubSub(node)
      const topics = pubsub.getTopics()
      for (const topic of topics) {
        pubsub.unsubscribe(topic)
      }

      // Close all connections gracefully
      const connections = node.getConnections()
      for (const conn of connections) {
        await conn.close()
      }

      // Stop the node
      await node.stop()

      console.log('✅ [LIBP2P] Node stopped successfully')
    } catch (error) {
      console.error('❌ [LIBP2P] Error during shutdown:', error)
    }
  })

  // Handle visibility change (tab backgrounding)
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      console.log('📴 [LIBP2P] Tab backgrounded - reducing activity')
      // TODO: Reduce connection count, unsubscribe from non-critical topics
    } else {
      console.log('📶 [LIBP2P] Tab foregrounded - resuming full activity')
      // TODO: Restore connections, re-subscribe to topics
    }
  })
}

/**
 * Get current node statistics
 *
 * Returns information about the node's current state
 *
 * @param node - The libp2p node
 * @returns Node statistics object
 */
export function getNodeStats(node: Libp2p) {
  const connections = node.getConnections()
  const peers = node.getPeers()
  const topics = getPubSub(node).getTopics()

  return {
    peerId: node.peerId.toString(),
    peerCount: peers.length,
    connectionCount: connections.length,
    topics: topics,
    connections: connections.map((conn) => ({
      peerId: conn.remotePeer.toString(),
      status: conn.status,
      direction: conn.direction,
      timeline: {
        open: conn.timeline.open,
        upgraded: conn.timeline.upgraded,
      },
    })),
  }
}

/**
 * Stop the libp2p node
 *
 * Gracefully shuts down the node and cleans up resources
 *
 * @param node - The libp2p node to stop
 */
export async function stopNode(node: Libp2p): Promise<void> {
  console.log('🛑 [LIBP2P] Stopping node...')

  try {
    // Unsubscribe from all topics
    const pubsub = getPubSub(node)
    const topics = pubsub.getTopics()
    for (const topic of topics) {
      pubsub.unsubscribe(topic)
    }

    // Close all connections
    const connections = node.getConnections()
    await Promise.all(connections.map((conn) => conn.close()))

    // Stop the node
    await node.stop()

    console.log('✅ [LIBP2P] Node stopped successfully')
  } catch (error) {
    console.error('❌ [LIBP2P] Error stopping node:', error)
    throw error
  }
}

/**
 * Debug utilities for development and Tor status monitoring
 * Exposed on window.libp2pDebug in all modes (needed for Tor status)
 */
export function exposeDebugUtilities(node: Libp2p): void {
  // Always expose debug utilities (needed for Tor status monitoring)
  // In production, users can check: window.libp2pDebug.getTorStatus()
  ;(window as any).libp2pDebug = {
      // Get current node stats
      getStats: () => getNodeStats(node),

      // Get peer list with details
      getPeers: () => {
        return node.getPeers().map((peerId) => ({
          peerId: peerId.toString(),
          connections: node.getConnections(peerId).map((conn) => ({
            status: conn.status,
            direction: conn.direction,
            multiaddr: conn.remoteAddr.toString(),
          })),
        }))
      },

      // Test dial to bootstrap peer
      testDial: async () => {
        try {
          console.log('🧪 Testing dial to bootstrap peer...')
          const ma = multiaddr(BOOTSTRAP_PEERS[0])
          await node.dial(ma)
          console.log('✅ Dial succeeded!')
          return { success: true, message: 'Dial succeeded' }
        } catch (error) {
          console.error('❌ Dial failed:', error)
          return { success: false, error: (error as Error).message }
        }
      },

      // Force reconnect (stop and restart)
      forceReconnect: async () => {
        console.log('🔄 Forcing reconnect...')
        await stopNode(node)
        const newNode = await createBrowserNode()
        console.log('✅ Reconnect complete')
        return newNode
      },

      // Subscribe to topic
      subscribe: (topic: string) => {
        console.log(`📡 Subscribing to topic: ${topic}`)
        getPubSub(node).subscribe(topic)
      },

      // Unsubscribe from topic
      unsubscribe: (topic: string) => {
        console.log(`📡 Unsubscribing from topic: ${topic}`)
        getPubSub(node).unsubscribe(topic)
      },

      // Publish to topic
      publish: (topic: string, data: string) => {
        console.log(`📤 Publishing to topic: ${topic}`)
        const encoder = new TextEncoder()
        getPubSub(node).publish(topic, encoder.encode(data))
      },

      // Get connection details
      getConnections: () => {
        return node.getConnections().map((conn) => ({
          peerId: conn.remotePeer.toString(),
          status: conn.status,
          direction: conn.direction,
          multiaddr: conn.remoteAddr.toString(),
          timeline: conn.timeline,
        }))
      },

      // Ping a peer
      ping: async (peerIdStr: string) => {
        try {
          const pingService = getPing(node)
          const latency = await pingService.ping(peerIdStr as any)
          console.log(`🏓 Ping to ${peerIdStr}: ${latency}ms`)
          return { success: true, latency }
        } catch (error) {
          console.error('❌ Ping failed:', error)
          return { success: false, error: (error as Error).message }
        }
      },

      // Get decoder metrics
      getMetrics: () => {
        console.log(getDecoderMetricsSummary())
        return DECODER_METRICS
      },

      // v3.5.x: Get P2P contribution stats
      getContributionStats: () => {
        const stats = {
          telemetry: telemetryReporter.getStats(),
          verification: verificationReporter.getStats(),
          blockCache: getBlockCacheStats(),
          blockServer: blockServer.getStats(),
          relay: getRelayStats(),
          browserPeers: browserPeerDiscovery.getStats(),
          propagation: getPropagationStats(), // v3.5.23
        }
        console.log('📊 [DEBUG] P2P Contribution Stats:', stats)
        return stats
      },

      // v3.5.23: Get propagation queue stats
      getPropagationStats: () => {
        const stats = getPropagationStats()
        console.log('📦 [DEBUG] Propagation Queue Stats:', stats)
        return stats
      },

      // v3.5.23: Accelerate gossip for a specific block
      accelerateGossip: async (height: number) => {
        console.log(`🚀 [DEBUG] Accelerating gossip for block ${height}...`)
        const success = await propagationQueue.accelerateGossip(height)
        console.log(success ? '✅ Gossip accelerated' : '❌ Acceleration failed')
        return success
      },

      // v3.5.24: P2P Data Service commands
      getP2PDataStats: () => {
        const stats = getP2PDataStats()
        console.log('🌐 [DEBUG] P2P Data Service Stats:', stats)
        return stats
      },

      // v3.5.24: Fetch block via P2P-first strategy
      fetchBlockP2P: async (height: number) => {
        console.log(`📦 [DEBUG] Fetching block ${height} (P2P-first)...`)
        const result = await p2pDataService.fetchBlock(height)
        console.log(`Result: ${result.success ? '✅' : '❌'} Source: ${result.source} (${result.latencyMs}ms)`)
        return result
      },

      // v3.5.24: Verify TX confirmation from multiple peers
      verifyTxConfirmation: async (txHash: string, blockHeight?: number) => {
        console.log(`🔍 [DEBUG] Verifying TX ${txHash.substring(0, 16)}...`)
        const result = await p2pDataService.verifyTransactionConfirmation(txHash, blockHeight)
        console.log(`Consensus: ${result.confirmed ? '✅ CONFIRMED' : '❌ NOT CONFIRMED'} (${result.confidence}% confidence)`)
        return result
      },

      // v3.5.24: Search for a transaction
      findTransaction: async (txHash: string) => {
        console.log(`🔍 [DEBUG] Searching for TX ${txHash.substring(0, 16)}...`)
        const result = await p2pDataService.findTransaction(txHash)
        if (result) {
          console.log(`✅ Found in block ${result.block.header.height}`)
        } else {
          console.log('❌ Transaction not found')
        }
        return result
      },

      // v3.5.24: Check if in offline mode
      isOfflineMode: () => {
        const offline = p2pDataService.isOfflineMode()
        console.log(`📴 [DEBUG] Offline mode: ${offline ? 'YES' : 'NO'}`)
        return offline
      },

      // 🧅 v3.6.0: Get Tor status
      getTorStatus: () => {
        const torState = getTorState()
        const torTransports = getTorTransportStats()
        const status = {
          state: torState,
          transports: torTransports,
          healthy: isTorHealthy(),
          status: formatTorStatus(),
        }
        console.log('🧅 [DEBUG] Tor Status:', status)
        return status
      },

      // 🔐 v3.7.4: Get post-quantum crypto status
      getPQCryptoStatus: () => {
        const status = {
          pqNoise: getPQNoiseStatus(),
          crypto: getPQCryptoStatus(),
          keypairGenerated: browserPQKeypair !== null,
          fingerprint: browserPQKeypair
            ? Array.from(browserPQKeypair.fingerprint.slice(0, 8))
                .map(b => b.toString(16).padStart(2, '0'))
                .join('')
            : null,
        }
        console.log('🔐 [DEBUG] Post-Quantum Crypto Status:', status)
        return status
      },

      // 🔐 v3.7.4: Get PQ keypair info (public only!)
      getPQKeypairInfo: () => {
        if (!browserPQKeypair) {
          console.log('❌ [DEBUG] No PQ keypair generated')
          return null
        }
        const info = {
          ed25519PublicKeyHex: Array.from(browserPQKeypair.ed25519PublicKey)
            .map(b => b.toString(16).padStart(2, '0'))
            .join(''),
          dilithium5PublicKeySize: browserPQKeypair.dilithium5PublicKey.length,
          kyber1024PublicKeySize: browserPQKeypair.kyber1024PublicKey.length,
          fingerprint: Array.from(browserPQKeypair.fingerprint)
            .map(b => b.toString(16).padStart(2, '0'))
            .join(''),
        }
        console.log('🔐 [DEBUG] PQ Keypair Info:', info)
        return info
      },

      // v3.5.8: Get known browser peers
      getBrowserPeers: () => {
        const peers = getKnownBrowserPeers()
        console.log('🌐 [DEBUG] Known Browser Peers:', peers)
        return peers
      },

      // v3.5.22: Get connection pre-warm results
      getPrewarmResults: () => {
        const results = getPrewarmResults()
        console.log('⚡ [DEBUG] Connection Pre-warm Results:', results)
        return results
      },

      // v3.5.x: Force send telemetry report
      forceTelemetry: async () => {
        console.log('📤 [DEBUG] Forcing telemetry report...')
        const success = await telemetryReporter.sendReport()
        console.log(success ? '✅ Telemetry sent' : '❌ Telemetry failed')
        return success
      },

      // v3.5.x: Get block cache contents
      getBlockCache: () => {
        const stats = getBlockCacheStats()
        const heights = blockCache.getHeights()
        console.log('🗄️ [DEBUG] Block Cache:', { stats, heights })
        return { stats, heights }
      },

      // Simulate network partition (for testing)
      simulateNetworkPartition: () => {
        console.warn('🧪 [DEBUG] Simulating network partition...')
        const connections = node.getConnections()
        connections.forEach((conn) => {
          conn.close().catch((err) => console.error('Failed to close connection:', err))
        })
        console.warn('🚨 [DEBUG] All connections closed - network partitioned')
      },

      // Force missed block simulation (for testing)
      forceMissedBlock: () => {
        console.warn('🧪 [DEBUG] Simulating missed blocks (setting lastBlockTime to 2 minutes ago)')
        window.dispatchEvent(new CustomEvent('debug-force-missed-block'))
      },
    }

    // Only log help messages in development mode
    if (import.meta.env.DEV) {
      console.log('🛠️ [LIBP2P DEBUG] Debug utilities exposed on window.libp2pDebug')
      console.log('Available commands:')
      console.log('  - window.libp2pDebug.getStats()')
      console.log('  - window.libp2pDebug.getPeers()')
      console.log('  - window.libp2pDebug.getMetrics()')
      console.log('  - window.libp2pDebug.testDial()')
      console.log('  - window.libp2pDebug.forceReconnect()')
      console.log('  - window.libp2pDebug.subscribe(topic)')
      console.log('  - window.libp2pDebug.publish(topic, data)')
      console.log('  - window.libp2pDebug.ping(peerId)')
      console.log('  - window.libp2pDebug.simulateNetworkPartition() [TEST]')
      console.log('  - window.libp2pDebug.forceMissedBlock() [TEST]')
      console.log('  v3.5.x P2P Contribution:')
      console.log('  - window.libp2pDebug.getContributionStats()')
      console.log('  - window.libp2pDebug.forceTelemetry()')
      console.log('  - window.libp2pDebug.getBlockCache()')
      console.log('  🧅 v3.6.0 Tor Status:')
      console.log('  - window.libp2pDebug.getTorStatus()')
      console.log('  ⚡ v3.5.22 Connection Optimization:')
      console.log('  - window.libp2pDebug.getPrewarmResults()')
      console.log('  🔐 v3.7.4 Post-Quantum Cryptography:')
      console.log('  - window.libp2pDebug.getPQCryptoStatus()')
      console.log('  - window.libp2pDebug.getPQKeypairInfo()')
    }
}
