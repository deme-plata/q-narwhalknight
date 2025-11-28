/**
 * LibP2P React Context Provider
 *
 * Provides the libp2p node to the entire React application.
 * This makes the P2P node accessible from any component via useLibP2P hook.
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

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
} from 'react'
import type { ReactNode } from 'react'
import type { Libp2p } from 'libp2p'
import { createBrowserNode, getNodeStats, stopNode, exposeDebugUtilities } from '../libp2p/node'
import { BOOTSTRAP_PEERS } from '../libp2p/config'
import { multiaddr } from '@multiformats/multiaddr'

/**
 * Force reconnection to bootstrap peers
 */
async function forceReconnect(node: Libp2p): Promise<void> {
  console.log('🔄 [CONTEXT] Force reconnecting to bootstrap peers...')

  for (const peerAddr of BOOTSTRAP_PEERS) {
    try {
      const ma = multiaddr(peerAddr)
      console.log(`🔌 [CONTEXT] Dialing ${peerAddr}...`)
      await node.dial(ma)
      console.log(`✅ [CONTEXT] Successfully dialed ${peerAddr}`)
    } catch (err) {
      console.warn(`⚠️  [CONTEXT] Failed to dial ${peerAddr}:`, err)
    }
  }
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
 * Wraps the application and provides P2P functionality to all child components.
 * Automatically creates and starts the libp2p node on mount.
 *
 * @param props - Provider props
 */
export function LibP2PProvider({
  children,
  autoStart = true,
}: LibP2PProviderProps) {
  const [node, setNode] = useState<Libp2p | null>(null)
  const [peerId, setPeerId] = useState<string | null>(null)
  const [peerCount, setPeerCount] = useState(0)
  const [connectionCount, setConnectionCount] = useState(0)
  const [topics, setTopics] = useState<string[]>([])
  const [isReady, setIsReady] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [lastBlockTime, setLastBlockTime] = useState(Date.now())
  const [networkPartitioned, setNetworkPartitioned] = useState(false)

  /**
   * Initialize the libp2p node
   */
  useEffect(() => {
    if (!autoStart) return

    let mounted = true
    let statsInterval: ReturnType<typeof setInterval> | null = null

    async function initNode() {
      try {
        setIsConnecting(true)
        setError(null)

        console.log('🚀 [CONTEXT] Initializing libp2p node...')

        // Create and start the node
        const newNode = await createBrowserNode()

        if (!mounted) {
          // Component unmounted during initialization
          await stopNode(newNode)
          return
        }

        setNode(newNode)
        setPeerId(newNode.peerId.toString())
        setIsReady(true)
        setIsConnecting(false)

        // Expose debug utilities in development mode
        exposeDebugUtilities(newNode)

        console.log('✅ [CONTEXT] Node initialized successfully')

        // Listen for block-received events to update lastBlockTime
        blockReceivedHandler = () => {
          setLastBlockTime(Date.now())
          if (networkPartitioned) {
            console.log('✅ [CONTEXT] Block received - network partition resolved')
            setNetworkPartitioned(false)
            setError(null)
          }
        }
        window.addEventListener('block-received', blockReceivedHandler as EventListener)

        // Update stats periodically
        statsInterval = setInterval(() => {
          if (!mounted) return

          const stats = getNodeStats(newNode)
          setPeerCount(stats.peerCount)
          setConnectionCount(stats.connectionCount)
          setTopics(stats.topics)

          // Network partition detection
          const timeSinceLastBlock = Date.now() - lastBlockTime
          const NO_BLOCKS_TIMEOUT = 60000 // 60 seconds
          const partitioned = timeSinceLastBlock > NO_BLOCKS_TIMEOUT && stats.peerCount === 0

          if (partitioned && !networkPartitioned) {
            console.error(
              '🚨 [CONTEXT] Network partition detected:',
              `\n  - No blocks for ${(timeSinceLastBlock / 1000).toFixed(0)}s`,
              `\n  - Peer count: ${stats.peerCount}`,
              '\n  - Attempting reconnection...'
            )
            setNetworkPartitioned(true)
            setError(new Error('Network partition detected - reconnecting...'))

            // Force reconnect by dialing bootstrap peers
            forceReconnect(newNode).catch((err) => {
              console.error('❌ [CONTEXT] Reconnection failed:', err)
            })
          } else if (!partitioned && networkPartitioned) {
            console.log('✅ [CONTEXT] Network partition resolved')
            setNetworkPartitioned(false)
            setError(null)
          }
        }, 5000) // Update every 5 seconds
      } catch (err) {
        console.error('❌ [CONTEXT] Failed to initialize node:', err)

        if (mounted) {
          setError(err instanceof Error ? err : new Error(String(err)))
          setIsConnecting(false)
          setIsReady(false)
        }
      }
    }

    // Store handleBlockReceived in a ref so cleanup can access it
    let blockReceivedHandler: (() => void) | null = null

    initNode()

    // Cleanup on unmount
    return () => {
      mounted = false

      if (statsInterval) {
        clearInterval(statsInterval)
      }

      // Remove block-received event listener (proper cleanup)
      if (blockReceivedHandler) {
        window.removeEventListener('block-received', blockReceivedHandler as EventListener)
      }

      if (node) {
        console.log('🛑 [CONTEXT] Cleaning up node...')
        stopNode(node).catch((err) => {
          console.error('❌ [CONTEXT] Error during cleanup:', err)
        })
      }
    }
  }, [autoStart]) // Only depend on autoStart - networkPartitioned is handled via state updates

  /**
   * Manually refresh statistics
   */
  const refresh = () => {
    if (!node) return

    const stats = getNodeStats(node)
    setPeerCount(stats.peerCount)
    setConnectionCount(stats.connectionCount)
    setTopics(stats.topics)
  }

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
