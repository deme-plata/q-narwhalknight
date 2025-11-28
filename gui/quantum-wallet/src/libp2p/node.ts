/**
 * Browser LibP2P Node - Core P2P Functionality
 *
 * This module creates and manages the browser's libp2p node,
 * transforming the React app into a first-class P2P peer.
 *
 * Architecture:
 * - Transport: WebSocket (browser ↔ Rust nodes)
 * - Security: Noise protocol (encryption)
 * - Multiplexing: Yamux (stream multiplexing)
 * - PubSub: Gossipsub (real-time messaging)
 * - DHT: Kademlia (peer discovery, light mode)
 * - Protocols: Identify, Ping, Bootstrap
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

import { createTransports } from './transports'
import {
  BOOTSTRAP_PEERS,
  CONNECTION_CONFIG,
  DHT_CONFIG,
  GOSSIPSUB_CONFIG,
  NETWORK_ID,
  PROTOCOL_VERSION,
} from './config'
import { DECODER_METRICS, getDecoderMetricsSummary } from './decoder'

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
 * Create and initialize a browser P2P node
 *
 * This is the main entry point for creating the libp2p node.
 * It configures all transports, protocols, and services needed
 * for the browser to participate in the P2P network.
 *
 * @returns Promise<Libp2p> - The initialized libp2p node
 */
export async function createBrowserNode(): Promise<Libp2p> {
  console.log('🚀 [LIBP2P] Initializing browser P2P node...')
  console.log(`📡 [LIBP2P] Network: ${NETWORK_ID}`)
  console.log(`📡 [LIBP2P] Protocol Version: ${PROTOCOL_VERSION}`)

  try {
    const node = await createLibp2p({
      // Addresses: Browsers can't listen (no incoming connections)
      addresses: {
        listen: [], // Browser nodes are dial-only
      },

      // Transport Layer: WebSocket for now
      transports: createTransports(),

      // Connection Encryption: Noise protocol
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
        ping: ping({
          protocolPrefix: 'qnk',
        }),

        // PubSub: Gossipsub for real-time messaging
        pubsub: gossipsub({
          emitSelf: false, // Don't receive our own messages
          floodPublish: GOSSIPSUB_CONFIG.FLOOD_PUBLISH,
          // Mesh parameters
          D: GOSSIPSUB_CONFIG.D,
          Dlo: GOSSIPSUB_CONFIG.D_LOW,
          Dhi: GOSSIPSUB_CONFIG.D_HIGH,
          heartbeatInterval: GOSSIPSUB_CONFIG.HEARTBEAT_INTERVAL,
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

    // Register custom protocols (handshake, block-sync, etc.)
    const { registerProtocols } = await import('./protocols')
    await registerProtocols(node)

    // Set up event listeners
    setupEventListeners(node)

    // Set up graceful shutdown
    setupGracefulShutdown(node)

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

  // Connection events
  node.addEventListener('peer:connect', (event) => {
    const peerId = event.detail.toString()
    console.log('🔗 [LIBP2P] Connected to peer:', peerId.substring(0, 16) + '...')
    console.log(`📊 [LIBP2P] Total connections: ${node.getConnections().length}`)
  })

  node.addEventListener('peer:disconnect', (event) => {
    const peerId = event.detail.toString()
    console.log('❌ [LIBP2P] Disconnected from peer:', peerId.substring(0, 16) + '...')
    console.log(`📊 [LIBP2P] Remaining connections: ${node.getConnections().length}`)
  })

  // Protocol events (identify)
  node.addEventListener('peer:identify', (event) => {
    const peerId = event.detail.peerId.toString()
    const protocols = event.detail.protocols
    console.log('🔍 [LIBP2P] Identified peer:', peerId.substring(0, 16) + '...')
    console.log('   Protocols:', protocols.slice(0, 5).join(', '))
  })

  // Log initial state
  setTimeout(() => {
    const connections = node.getConnections()
    console.log(`📊 [LIBP2P] Current state:`)
    console.log(`   Connections: ${connections.length}`)
    console.log(`   Peer Store Size: ${node.getPeers().length}`)
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
 * Debug utilities for development
 * Exposed on window.libp2pDebug in development mode
 */
export function exposeDebugUtilities(node: Libp2p): void {
  if (import.meta.env.DEV) {
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
  }
}
