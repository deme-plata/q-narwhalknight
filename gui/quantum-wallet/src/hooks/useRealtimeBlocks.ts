/**
 * useRealtimeBlocks Hook
 *
 * Real-time block streaming via libp2p Gossipsub.
 * Replaces HTTP polling with <100ms latency PubSub messages.
 *
 * Usage:
 * ```typescript
 * const { latestBlock, blockHistory, isSubscribed } = useRealtimeBlocks()
 * ```
 */

import { useState, useEffect, useCallback } from 'react'
import { useLibP2P } from '../contexts/LibP2PContext'
import { TOPICS } from '../libp2p/config'
import { decodeBlock, validateBlock, createBlockSummary, logBlockSummary } from '../libp2p/decoder'
import type { QBlock, BlockSummary } from '../libp2p/types'
import type { GossipSub } from '@libp2p/gossipsub'

/**
 * Backpressure Configuration
 * Prevents memory exhaustion if blocks arrive faster than browser can process
 */
const MAX_BLOCK_HISTORY = 100 // Cap at 100 blocks (~2MB memory)
const BUFFER_OVERFLOW_WARNING_THRESHOLD = 80 // Warn at 80% capacity

export interface UseRealtimeBlocksResult {
  // Latest block received
  latestBlock: QBlock | null

  // Block summary for UI display
  latestBlockSummary: BlockSummary | null

  // Recent block history (last 10 blocks)
  blockHistory: QBlock[]

  // Subscription status
  isSubscribed: boolean

  // Error state
  error: Error | null

  // Manually subscribe/unsubscribe
  subscribe: () => void
  unsubscribe: () => void
}

/**
 * Real-time block streaming hook
 */
export function useRealtimeBlocks(): UseRealtimeBlocksResult {
  const { node, isReady } = useLibP2P()

  const [latestBlock, setLatestBlock] = useState<QBlock | null>(null)
  const [latestBlockSummary, setLatestBlockSummary] = useState<BlockSummary | null>(null)
  const [blockHistory, setBlockHistory] = useState<QBlock[]>([])
  const [isSubscribed, setIsSubscribed] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  /**
   * Handle incoming block messages
   */
  const handleBlockMessage = useCallback((event: CustomEvent) => {
    try {
      const messageData = event.detail?.data

      if (!messageData) {
        console.warn('[REALTIME BLOCKS] Received message with no data')
        return
      }

      // Decode block from binary data
      const block = decodeBlock(messageData)

      if (!block) {
        console.warn('[REALTIME BLOCKS] Failed to decode block')
        return
      }

      // Validate block structure
      if (!validateBlock(block)) {
        console.warn('[REALTIME BLOCKS] Invalid block structure')
        return
      }

      // Log block summary in dev mode
      if (import.meta.env.DEV) {
        console.log('📦 [REALTIME BLOCKS] New block received via P2P!')
        logBlockSummary(block)
      }

      // Update latest block
      setLatestBlock(block)

      // Create block summary for UI
      const summary = createBlockSummary(block)
      setLatestBlockSummary(summary)

      // Update block history with backpressure limit
      setBlockHistory((prev) => {
        const newHistory = [block, ...prev]

        // Trim to MAX_BLOCK_HISTORY to prevent memory exhaustion
        const trimmed = newHistory.slice(0, MAX_BLOCK_HISTORY)

        // Warn if approaching buffer limit
        if (trimmed.length >= BUFFER_OVERFLOW_WARNING_THRESHOLD) {
          console.warn(
            `⚠️  [REALTIME BLOCKS] Buffer at ${trimmed.length}/${MAX_BLOCK_HISTORY} blocks (${Math.round((trimmed.length / MAX_BLOCK_HISTORY) * 100)}%)`
          )
        }

        return trimmed
      })

      // Dispatch custom event for other components
      window.dispatchEvent(
        new CustomEvent('block-received', {
          detail: { block, summary },
        })
      )

      console.log(
        `✅ [REALTIME BLOCKS] Block ${block.header.height} processed (${block.transactions.length} txs)`
      )
    } catch (err) {
      console.error('[REALTIME BLOCKS] Error handling block message:', err)
      setError(err instanceof Error ? err : new Error(String(err)))
    }
  }, [])

  /**
   * Subscribe to block topic
   */
  const subscribe = useCallback(() => {
    if (!node || !isReady) {
      console.warn('[REALTIME BLOCKS] Cannot subscribe: node not ready')
      return
    }

    try {
      console.log(`📡 [REALTIME BLOCKS] Subscribing to ${TOPICS.BLOCKS}`)

      const pubsub = node.services.pubsub as GossipSub

      // Subscribe to block topic
      pubsub.subscribe(TOPICS.BLOCKS)

      // Listen for messages
      pubsub.addEventListener('message', (event: any) => {
        // Filter by topic
        if (event.detail.topic === TOPICS.BLOCKS) {
          handleBlockMessage(event as CustomEvent)
        }
      })

      setIsSubscribed(true)
      console.log('✅ [REALTIME BLOCKS] Subscribed successfully')
    } catch (err) {
      console.error('[REALTIME BLOCKS] Subscription failed:', err)
      setError(err instanceof Error ? err : new Error(String(err)))
    }
  }, [node, isReady, handleBlockMessage])

  /**
   * Unsubscribe from block topic
   */
  const unsubscribe = useCallback(() => {
    if (!node || !isReady) {
      return
    }

    try {
      console.log(`📡 [REALTIME BLOCKS] Unsubscribing from ${TOPICS.BLOCKS}`)

      const pubsub = node.services.pubsub as GossipSub
      pubsub.unsubscribe(TOPICS.BLOCKS)

      setIsSubscribed(false)
      console.log('✅ [REALTIME BLOCKS] Unsubscribed successfully')
    } catch (err) {
      console.error('[REALTIME BLOCKS] Unsubscribe failed:', err)
      setError(err instanceof Error ? err : new Error(String(err)))
    }
  }, [node, isReady])

  /**
   * Auto-subscribe when node is ready
   */
  useEffect(() => {
    if (isReady && !isSubscribed) {
      subscribe()
    }

    // Cleanup on unmount
    return () => {
      if (isSubscribed) {
        unsubscribe()
      }
    }
  }, [isReady, isSubscribed, subscribe, unsubscribe])

  return {
    latestBlock,
    latestBlockSummary,
    blockHistory,
    isSubscribed,
    error,
    subscribe,
    unsubscribe,
  }
}

/**
 * useBlockHeight Hook
 *
 * Simplified hook that only tracks the latest block height.
 * Useful for components that don't need the full block data.
 */
export function useBlockHeight(): number {
  const { latestBlock } = useRealtimeBlocks()
  return latestBlock?.header.height || 0
}

/**
 * useTransactionCount Hook
 *
 * Track total number of transactions in recent blocks
 */
export function useTransactionCount(): number {
  const { blockHistory } = useRealtimeBlocks()

  return blockHistory.reduce(
    (total, block) => total + block.transactions.length,
    0
  )
}
