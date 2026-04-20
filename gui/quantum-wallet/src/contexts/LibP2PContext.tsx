/**
 * LibP2P React Context Provider
 *
 * Provides the libp2p node to the entire React application.
 * This makes the P2P node accessible from any component via useLibP2P hook.
 *
 * v3.4.2-browser: Re-enabled browser P2P with correct PeerID and network config
 *
 * Usage:
 * ```tsx
 * import { useLibP2P } from '../contexts/LibP2PContext'
 *
 * function MyComponent() {
 *   const { node, peerId, peerCount, isReady } = useLibP2P()
 *
 *   if (!isReady) return <div>Connecting to P2P network...</div>
 *
 *   return <div>Connected as {peerId}</div>
 * }
 * ```
 */

console.log('📦 [LIBP2P] LibP2PContext.tsx module loading...')

// ⚡ v3.5.22: Start connection pre-warming immediately on module load
// This warms TCP/TLS connections to bootstrap peers before the node is created
// giving us a head start on connection establishment
setTimeout(() => {
  console.log('⚡ [LIBP2P] Starting early connection pre-warming...')
  prewarmConnections()
}, 100) // Small delay to not block initial render

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
} from 'react'
import type { ReactNode } from 'react'
import type { Libp2p } from 'libp2p'
import { createBrowserNode, getNodeStats, stopNode, exposeDebugUtilities, prewarmConnections } from '../libp2p/node'
import { BOOTSTRAP_PEERS, NETWORK_ID } from '../libp2p/config'
import { multiaddr } from '@multiformats/multiaddr'
import { submitTransactionWithFallback, getTransactionStats } from '../libp2p/transactionSubmitter'
import { getTelemetryStats, telemetryReporter } from '../libp2p/telemetry'
import { getBlockCacheStats, blockCache } from '../libp2p/blockCache'
import { verificationReporter, getReporterStats } from '../libp2p/verificationReporter'
import type { SignedTransaction, TelemetryReport, BlockCacheStats } from '../libp2p/types'
import type { TransactionSubmitResult, TransactionStats } from '../libp2p/transactionSubmitter'
import type { ReporterStats } from '../libp2p/verificationReporter'

/**
 * Force reconnection to bootstrap peers
 */
async function forceReconnect(node: Libp2p): Promise<boolean> {
  console.log('🔄 [CONTEXT] Force reconnecting to bootstrap peers...')
  let connected = false

  for (const peerAddr of BOOTSTRAP_PEERS) {
    try {
      const ma = multiaddr(peerAddr)
      console.log(`🔌 [CONTEXT] Dialing ${peerAddr}...`)
      // v3.5.5: Reduced timeout from 30s to 10s for faster connection
      await node.dial(ma, { signal: AbortSignal.timeout(10000) })
      console.log(`✅ [CONTEXT] Successfully dialed ${peerAddr}`)
      connected = true
    } catch (err) {
      console.warn(`⚠️  [CONTEXT] Failed to dial ${peerAddr}:`, err)
    }
  }

  return connected
}

/**
 * LibP2P Context State
 */
interface LibP2PContextState {
  // The libp2p node instance
  node: Libp2p | null

  // Node identification
  peerId: string | null

  // Network statistics
  peerCount: number
  connectionCount: number
  topics: string[]

  // Node state
  isReady: boolean
  isConnecting: boolean
  error: Error | null

  // Functions
  refresh: () => void

  // v3.5.x: Browser P2P Network Contribution functions
  submitTransaction: (tx: SignedTransaction) => Promise<TransactionSubmitResult>
  getTelemetryStats: () => Omit<TelemetryReport, 'peerId' | 'timestamp'>
  getBlockCacheStats: () => BlockCacheStats
  getTransactionStats: () => TransactionStats
  getVerificationReporterStats: () => ReporterStats
}

/**
 * Default context value (before initialization)
 */
const defaultContextValue: LibP2PContextState = {
  node: null,
  peerId: null,
  peerCount: 0,
  connectionCount: 0,
  topics: [],
  isReady: false,
  isConnecting: false,
  error: null,
  refresh: () => {},
  // v3.5.x: Default implementations for P2P contribution functions
  submitTransaction: async () => ({ success: false, method: 'none' as const, error: 'Node not ready', timestamp: Date.now() }),
  getTelemetryStats: () => getTelemetryStats(),
  getBlockCacheStats: () => getBlockCacheStats(),
  getTransactionStats: () => getTransactionStats(),
  getVerificationReporterStats: () => getReporterStats(),
}

/**
 * LibP2P Context
 */
const LibP2PContext = createContext<LibP2PContextState>(defaultContextValue)

/**
 * LibP2P Provider Props
 */
interface LibP2PProviderProps {
  children: ReactNode
  // Optional: delay node initialization (useful for testing)
  autoStart?: boolean
}

/**
 * LibP2P Provider Component
 *
 * v3.4.2-browser: Browser P2P ENABLED
 * - Connects to Server Beta via WebSocket (wss://quillon.xyz:9443)
 * - Uses correct PeerID: 12D3KooWSBxwSKw4wftHViMdw5rrV8Z1wEkikDS2vKYZtRrio5hH
 * - Network: mainnet2026.2
 *
 * @param props - Provider props
 */
export function LibP2PProvider({
  children,
  autoStart = true,
}: LibP2PProviderProps) {
  // State with proper setters for P2P mode
  const [node, setNode] = useState<Libp2p | null>(null)
  const [peerId, setPeerId] = useState<string | null>(null)
  const [peerCount, setPeerCount] = useState(0)
  const [connectionCount, setConnectionCount] = useState(0)
  const [topics, setTopics] = useState<string[]>([])
  const [isReady, setIsReady] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  // Refs for cleanup
  const nodeRef = useRef<Libp2p | null>(null)
  const statsIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptRef = useRef<number>(0)

  /**
   * Initialize the libp2p node
   */
  useEffect(() => {
    console.log('🔄 [CONTEXT] LibP2PProvider useEffect running, autoStart:', autoStart)

    if (!autoStart) {
      console.log('📡 [CONTEXT] P2P autoStart disabled, waiting for manual start')
      return
    }

    let mounted = true

    /**
     * 🔄 v3.5.4-browser: Schedule reconnection with exponential backoff
     * Automatically reconnects when bootstrap node restarts
     */
    function scheduleReconnect(browserNode: Libp2p) {
      // Clear any existing reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }

      // Exponential backoff: 2s, 4s, 8s, 16s, 30s max
      const attempt = reconnectAttemptRef.current
      const delay = Math.min(2000 * Math.pow(2, attempt), 30000)
      reconnectAttemptRef.current += 1

      console.log(`🔄 [CONTEXT] Reconnect attempt ${attempt + 1} scheduled in ${delay / 1000}s...`)

      reconnectTimeoutRef.current = setTimeout(async () => {
        if (!mounted) return

        try {
          console.log('🔄 [CONTEXT] Attempting to reconnect to bootstrap peers...')
          const connected = await forceReconnect(browserNode)

          if (connected) {
            console.log('✅ [CONTEXT] Reconnected successfully!')
            reconnectAttemptRef.current = 0 // Reset backoff on success
          } else {
            console.warn('⚠️  [CONTEXT] Reconnect failed, will retry...')
            scheduleReconnect(browserNode) // Try again with longer delay
          }
        } catch (err) {
          console.error('❌ [CONTEXT] Reconnect error:', err)
          scheduleReconnect(browserNode) // Try again
        }
      }, delay)
    }

    async function initNode() {
      // v10.3.0: VISIBLE error reporting for P2P init debugging
      const p2pDebugDiv = document.createElement('div')
      p2pDebugDiv.id = 'p2p-debug'
      p2pDebugDiv.style.cssText = 'position:fixed;bottom:10px;left:10px;background:rgba(0,0,0,0.85);color:#0f0;font-family:monospace;font-size:11px;padding:8px 12px;border-radius:6px;z-index:99999;max-width:500px;pointer-events:none;'
      p2pDebugDiv.textContent = '🔄 P2P: Initializing...'
      document.body.appendChild(p2pDebugDiv)

      const p2pLog = (msg: string) => {
        console.log(msg)
        if (p2pDebugDiv) p2pDebugDiv.textContent = msg
      }

      p2pLog('🔄 P2P: Starting initNode()')
      console.log('🚀 [CONTEXT] Initializing browser P2P node...')
      console.log(`   Network: ${NETWORK_ID}`)
      console.log(`   Bootstrap peers: ${BOOTSTRAP_PEERS.length}`)

      setIsConnecting(true)
      setError(null)

      try {
        p2pLog('🔄 P2P: Calling createBrowserNode()...')
        // Create the browser libp2p node
        const browserNode = await createBrowserNode()
        p2pLog(`✅ P2P: Node created! PeerId: ${browserNode.peerId.toString().slice(0, 20)}...`)

        if (!mounted) {
          console.log('🛑 [CONTEXT] Component unmounted during init, stopping node')
          await stopNode(browserNode)
          return
        }

        nodeRef.current = browserNode
        setNode(browserNode)
        setPeerId(browserNode.peerId.toString())

        console.log(`✅ [CONTEXT] Node created with PeerId: ${browserNode.peerId}`)

        // Expose debug utilities (always - needed for Tor status monitoring)
        exposeDebugUtilities(browserNode)
        console.log('🔧 [CONTEXT] Debug utilities exposed on window.libp2pDebug')

        // Set up connection event listeners
        browserNode.addEventListener('peer:connect', (evt) => {
          console.log(`🔗 [CONTEXT] Peer connected: ${evt.detail}`)
          updateStats(browserNode)
        })

        browserNode.addEventListener('peer:disconnect', (evt) => {
          console.log(`🔌 [CONTEXT] Peer disconnected: ${evt.detail}`)
          updateStats(browserNode)

          // 🔄 v3.5.4-browser: Auto-reconnect when all connections lost
          const connections = browserNode.getConnections()
          if (connections.length === 0) {
            console.log('⚠️  [CONTEXT] All connections lost - scheduling auto-reconnect...')
            scheduleReconnect(browserNode)
          }
        })

        // Try to connect to bootstrap peers
        console.log('🌐 [CONTEXT] Connecting to bootstrap peers...')
        const connected = await forceReconnect(browserNode)

        if (!mounted) return

        if (connected) {
          console.log('✅ [CONTEXT] Connected to at least one bootstrap peer!')
          setIsReady(true)
        } else {
          console.warn('⚠️  [CONTEXT] Failed to connect to any bootstrap peers')
          console.log('   Browser will retry automatically via DHT discovery')
          // Still mark as ready - node is running, just no peers yet
          setIsReady(true)
        }

        // Update initial stats
        updateStats(browserNode)

        // Set up periodic stats refresh
        statsIntervalRef.current = setInterval(() => {
          if (nodeRef.current) {
            updateStats(nodeRef.current)
          }
        }, 5000)

      } catch (err) {
        const errMsg = err instanceof Error ? `${err.message}\n${err.stack?.slice(0, 300)}` : String(err)
        console.error('❌ [CONTEXT] Failed to initialize P2P node:', err)
        p2pLog(`❌ P2P ERROR: ${errMsg.slice(0, 200)}`)
        // Keep the debug div visible for 30 seconds on error
        if (p2pDebugDiv) {
          p2pDebugDiv.style.color = '#f55'
          p2pDebugDiv.style.pointerEvents = 'auto'
          setTimeout(() => p2pDebugDiv?.remove(), 30000)
        }
        if (mounted) {
          setError(err instanceof Error ? err : new Error(String(err)))
          setIsReady(false)
        }
      } finally {
        if (mounted) {
          setIsConnecting(false)
        }
      }
    }

    function updateStats(n: Libp2p) {
      try {
        const stats = getNodeStats(n)
        setPeerCount(stats.peerCount)
        setConnectionCount(stats.connectionCount)
        setTopics(stats.topics)
      } catch (err) {
        console.warn('[CONTEXT] Failed to get node stats:', err)
      }
    }

    initNode()

    // Cleanup on unmount
    return () => {
      mounted = false

      // Clear reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }

      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current)
        statsIntervalRef.current = null
      }

      if (nodeRef.current) {
        console.log('🛑 [CONTEXT] Stopping P2P node on cleanup...')
        stopNode(nodeRef.current).catch(err => {
          console.warn('[CONTEXT] Error stopping node:', err)
        })
        nodeRef.current = null
      }
    }
  }, [autoStart])

  /**
   * Manually refresh statistics
   */
  const refresh = useCallback(() => {
    if (!node) return

    try {
      const stats = getNodeStats(node)
      setPeerCount(stats.peerCount)
      setConnectionCount(stats.connectionCount)
      setTopics(stats.topics)
    } catch (err) {
      console.warn('[CONTEXT] Failed to refresh stats:', err)
    }
  }, [node])

  /**
   * v3.5.x: Submit transaction via P2P
   */
  const submitTransaction = useCallback(async (tx: SignedTransaction): Promise<TransactionSubmitResult> => {
    return submitTransactionWithFallback(node, tx)
  }, [node])

  /**
   * Context value
   */
  const contextValue: LibP2PContextState = {
    node,
    peerId,
    peerCount,
    connectionCount,
    topics,
    isReady,
    isConnecting,
    error,
    refresh,
    // v3.5.x: Browser P2P Network Contribution functions
    submitTransaction,
    getTelemetryStats,
    getBlockCacheStats,
    getTransactionStats,
    getVerificationReporterStats: getReporterStats,
  }

  return (
    <LibP2PContext.Provider value={contextValue}>
      {children}
    </LibP2PContext.Provider>
  )
}

/**
 * Hook to access LibP2P context
 *
 * @returns LibP2P context state
 * @throws Error if used outside LibP2PProvider
 */
export function useLibP2P(): LibP2PContextState {
  const context = useContext(LibP2PContext)

  if (context === undefined) {
    throw new Error('useLibP2P must be used within a LibP2PProvider')
  }

  return context
}

/**
 * HOC to inject LibP2P into a component
 *
 * @param Component - Component to wrap
 * @returns Wrapped component with LibP2P props
 */
export function withLibP2P<P extends object>(
  Component: React.ComponentType<P & { libp2p: LibP2PContextState }>
) {
  return function WrappedComponent(props: P) {
    const libp2p = useLibP2P()
    return <Component {...props} libp2p={libp2p} />
  }
}
