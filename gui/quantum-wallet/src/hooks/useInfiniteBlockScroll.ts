/**
 * useInfiniteBlockScroll Hook
 *
 * Magical infinite scrolling for blockchain explorer.
 * Fetches blocks on-demand via P2P network as user scrolls.
 *
 * Features:
 * - Zero loading states (seamless streaming)
 * - P2P block requests from multiple peers
 * - Parallel loading from 3 peers simultaneously
 * - Automatic peer selection based on proximity
 * - Graceful fallback to HTTP API if P2P unavailable
 *
 * Usage:
 * ```typescript
 * const { blocks, loadMore, isLoading, stats } = useInfiniteBlockScroll()
 * ```
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import { useLibP2P } from '../contexts/LibP2PContext'
import { qnkAPI } from '../services/api'
import { requestBlocksFromPeers } from '../libp2p/blockRequest'
import type { QBlock } from '../libp2p/types'

interface BlockScrollStats {
  totalLoaded: number
  p2pSuccesses: number
  httpFallbacks: number
  avgLoadTime: number
  activePeers: number
}

interface UseInfiniteBlockScrollResult {
  blocks: QBlock[]
  loadMore: (startHeight: number, count: number) => Promise<void>
  isLoading: boolean
  stats: BlockScrollStats
  currentHeight: number
}

const BLOCKS_PER_PAGE = 20

/**
 * Infinite scroll blockchain explorer hook
 */
export function useInfiniteBlockScroll(): UseInfiniteBlockScrollResult {
  const { node, isReady, peerCount } = useLibP2P()
  const [blocks, setBlocks] = useState<QBlock[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [currentHeight, setCurrentHeight] = useState(0)
  const [stats, setStats] = useState<BlockScrollStats>({
    totalLoaded: 0,
    p2pSuccesses: 0,
    httpFallbacks: 0,
    avgLoadTime: 0,
    activePeers: 0,
  })

  const loadTimesRef = useRef<number[]>([])

  /**
   * Fetch current blockchain height
   */
  useEffect(() => {
    async function fetchHeight() {
      try {
        const response = await qnkAPI.getNetworkStatistics()
        if (response.success && response.data) {
          setCurrentHeight(response.data.currentHeight || 0)
        }
      } catch (error) {
        console.error('[INFINITE SCROLL] Failed to fetch height:', error)
      }
    }

    fetchHeight()
    const interval = setInterval(fetchHeight, 30000) // Update every 30s

    return () => clearInterval(interval)
  }, [])

  /**
   * Load blocks from P2P network with HTTP fallback
   */
  const loadMore = useCallback(
    async (startHeight: number, count: number = BLOCKS_PER_PAGE) => {
      if (isLoading) return

      setIsLoading(true)
      const loadStartTime = performance.now()

      console.log(`📜 [INFINITE SCROLL] Loading blocks ${startHeight} to ${startHeight - count}`)

      try {
        let loadedBlocks: QBlock[] = []

        // Try P2P first if available
        if (node && isReady && peerCount > 0) {
          console.log(`🔗 [INFINITE SCROLL] Attempting P2P load from ${peerCount} peers`)

          try {
            loadedBlocks = await loadBlocksFromP2P(node, startHeight, count)

            if (loadedBlocks.length > 0) {
              console.log(`✅ [INFINITE SCROLL] P2P success: ${loadedBlocks.length} blocks`)

              setStats((prev) => ({
                ...prev,
                p2pSuccesses: prev.p2pSuccesses + loadedBlocks.length,
                totalLoaded: prev.totalLoaded + loadedBlocks.length,
                activePeers: peerCount,
              }))
            } else {
              throw new Error('No blocks from P2P')
            }
          } catch (p2pError) {
            console.warn('[INFINITE SCROLL] P2P failed, falling back to HTTP:', p2pError)
            loadedBlocks = await loadBlocksFromHTTP(startHeight, count)

            setStats((prev) => ({
              ...prev,
              httpFallbacks: prev.httpFallbacks + loadedBlocks.length,
              totalLoaded: prev.totalLoaded + loadedBlocks.length,
            }))
          }
        } else {
          // HTTP fallback if P2P not ready
          console.log('📡 [INFINITE SCROLL] Using HTTP (P2P not ready)')
          loadedBlocks = await loadBlocksFromHTTP(startHeight, count)

          setStats((prev) => ({
            ...prev,
            httpFallbacks: prev.httpFallbacks + loadedBlocks.length,
            totalLoaded: prev.totalLoaded + loadedBlocks.length,
          }))
        }

        // Update blocks state
        setBlocks((prev) => {
          // Avoid duplicates
          const existingHeights = new Set(prev.map((b) => b.header.height))
          const newBlocks = loadedBlocks.filter((b) => !existingHeights.has(b.header.height))
          return [...prev, ...newBlocks].sort((a, b) => b.header.height - a.header.height)
        })

        // Update load time stats
        const loadTime = performance.now() - loadStartTime
        loadTimesRef.current.push(loadTime)

        if (loadTimesRef.current.length > 10) {
          loadTimesRef.current.shift() // Keep last 10
        }

        const avgLoadTime =
          loadTimesRef.current.reduce((a, b) => a + b, 0) / loadTimesRef.current.length

        setStats((prev) => ({
          ...prev,
          avgLoadTime,
        }))

        console.log(`⚡ [INFINITE SCROLL] Load complete in ${loadTime.toFixed(0)}ms`)
      } catch (error) {
        console.error('[INFINITE SCROLL] Load failed:', error)
      } finally {
        setIsLoading(false)
      }
    },
    [node, isReady, peerCount, isLoading]
  )

  return {
    blocks,
    loadMore,
    isLoading,
    stats,
    currentHeight,
  }
}

/**
 * Load blocks from P2P network (parallel requests to multiple peers)
 */
async function loadBlocksFromP2P(
  node: any,
  startHeight: number,
  count: number
): Promise<QBlock[]> {
  if (!node) {
    throw new Error('LibP2P node not available')
  }

  try {
    const blocks = await requestBlocksFromPeers(node, startHeight, count)

    if (blocks.length === 0) {
      throw new Error('No blocks received from P2P network')
    }

    return blocks
  } catch (error) {
    console.error('[P2P LOAD] Failed to load blocks from P2P:', error)
    throw error
  }
}

/**
 * Load blocks from HTTP API (fallback)
 */
async function loadBlocksFromHTTP(startHeight: number, count: number): Promise<QBlock[]> {
  const blocks: QBlock[] = []

  // Fetch blocks in parallel (5 at a time)
  const PARALLEL_REQUESTS = 5
  const promises: Promise<any>[] = []

  for (let i = 0; i < count; i += PARALLEL_REQUESTS) {
    const batchPromises = []

    for (let j = 0; j < PARALLEL_REQUESTS && i + j < count; j++) {
      const height = startHeight - i - j

      if (height >= 0) {
        batchPromises.push(
          qnkAPI
            .getBlock(height)
            .then((block: any) => block)
            .catch((err: any) => {
              console.warn(`[HTTP FALLBACK] Failed to fetch block ${height}:`, err)
              return null
            })
        )
      }
    }

    promises.push(...batchPromises)
  }

  const results = await Promise.allSettled(promises)

  results.forEach((result) => {
    if (result.status === 'fulfilled' && result.value) {
      blocks.push(result.value)
    }
  })

  return blocks.filter((b) => b != null)
}
