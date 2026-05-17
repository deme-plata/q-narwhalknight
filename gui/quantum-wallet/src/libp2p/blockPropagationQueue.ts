/**
 * Block Propagation Queue - v3.5.23
 *
 * Advanced queue-based block propagation system for browser P2P nodes.
 * Enables faster block synchronization through:
 *
 * 1. Priority Queue - Recent blocks are "hot" and propagate first
 * 2. Eager Push - Proactively push blocks to connected peers
 * 3. Request Batching - Queue and batch multiple block requests
 * 4. Gossip Acceleration - Re-broadcast blocks on demand
 * 5. Peer Need Tracking - Track which peers need which blocks
 *
 * Architecture:
 * - Hot blocks (last 10) are kept in fast-access queue
 * - Warm blocks (last 50) are in secondary queue
 * - Cold blocks fall back to standard cache
 */

import type { Libp2p } from 'libp2p'
import type { GossipSub } from '@libp2p/gossipsub'
import { encode as msgpackEncode, decode as msgpackDecode } from '@msgpack/msgpack'
import { TOPICS, PERFORMANCE_CONFIG } from './config'
import { blockCache } from './blockCache'
import type { VerifiedBlock, QBlock } from './types'

/**
 * Queue priority levels
 */
export const BlockPriority = {
  HOT: 0,    // Last 10 blocks - immediate propagation
  WARM: 1,   // Last 50 blocks - fast propagation
  COLD: 2,   // Older blocks - standard cache
} as const

export type BlockPriority = typeof BlockPriority[keyof typeof BlockPriority]

/**
 * Queued block entry
 */
interface QueuedBlock {
  block: VerifiedBlock
  priority: BlockPriority
  addedAt: number
  propagatedTo: Set<string> // Peer IDs we've already pushed to
  requestCount: number      // How many times this block was requested
}

/**
 * Peer need tracking
 */
interface PeerNeed {
  peerId: string
  lastHeight: number
  lastUpdate: number
  missedBlocks: number[]
}

/**
 * Propagation statistics
 */
export interface PropagationStats {
  hotQueueSize: number
  warmQueueSize: number
  totalPropagations: number
  eagerPushes: number
  batchedRequests: number
  gossipAccelerations: number
  avgPropagationTimeMs: number
  peersTracked: number
}

/**
 * Block Propagation Queue
 */
export class BlockPropagationQueue {
  private libp2p: Libp2p | null = null

  // Priority queues (Map for O(1) access, sorted by height)
  private hotQueue: Map<number, QueuedBlock> = new Map()
  private warmQueue: Map<number, QueuedBlock> = new Map()

  // Peer need tracking
  private peerNeeds: Map<string, PeerNeed> = new Map()

  // Pending batch requests
  private pendingRequests: Map<string, { heights: number[], resolve: (blocks: QBlock[]) => void }[]> = new Map()
  private batchFlushTimer: ReturnType<typeof setTimeout> | null = null

  // Statistics
  private stats = {
    totalPropagations: 0,
    eagerPushes: 0,
    batchedRequests: 0,
    gossipAccelerations: 0,
    propagationTimes: [] as number[],
  }

  // Configuration
  private readonly HOT_QUEUE_SIZE = 10
  private readonly WARM_QUEUE_SIZE = 50
  private readonly BATCH_DELAY_MS = PERFORMANCE_CONFIG.BATCH_TIMEOUT
  private readonly MAX_PROPAGATION_SAMPLES = 100
  private readonly EAGER_PUSH_INTERVAL_MS = 500

  // Eager push timer
  private eagerPushTimer: ReturnType<typeof setInterval> | null = null

  /**
   * Initialize the propagation queue
   */
  initialize(libp2p: Libp2p): void {
    this.libp2p = libp2p
    console.log('📦 [PROP QUEUE] Block propagation queue initialized')

    // Start eager push loop
    this.startEagerPushLoop()
  }

  /**
   * Add a block to the propagation queue
   */
  addBlock(block: VerifiedBlock): void {
    const height = block.header.height
    const now = Date.now()

    // Determine priority based on recency
    const highestHot = this.getHighestHeight(this.hotQueue)
    const priority = this.calculatePriority(height, highestHot)

    const entry: QueuedBlock = {
      block,
      priority,
      addedAt: now,
      propagatedTo: new Set(),
      requestCount: 0,
    }

    // Add to appropriate queue
    if (priority === BlockPriority.HOT) {
      this.hotQueue.set(height, entry)
      this.trimQueue(this.hotQueue, this.HOT_QUEUE_SIZE)
      console.log(`🔥 [PROP QUEUE] HOT block ${height} added (queue: ${this.hotQueue.size})`)
    } else if (priority === BlockPriority.WARM) {
      this.warmQueue.set(height, entry)
      this.trimQueue(this.warmQueue, this.WARM_QUEUE_SIZE)
      console.log(`♨️ [PROP QUEUE] WARM block ${height} added (queue: ${this.warmQueue.size})`)
    }

    // Also add to standard cache
    blockCache.add(block)

    // Trigger immediate propagation for HOT blocks
    if (priority === BlockPriority.HOT) {
      this.triggerEagerPush(height)
    }
  }

  /**
   * Get a block from the queue (fastest path first)
   */
  getBlock(height: number): VerifiedBlock | undefined {
    // Check hot queue first (fastest)
    const hot = this.hotQueue.get(height)
    if (hot) {
      hot.requestCount++
      return hot.block
    }

    // Check warm queue
    const warm = this.warmQueue.get(height)
    if (warm) {
      warm.requestCount++
      // Promote to hot if frequently requested
      if (warm.requestCount >= 3) {
        this.promoteToHot(height, warm)
      }
      return warm.block
    }

    // Fall back to standard cache
    return blockCache.get(height)
  }

  /**
   * Get multiple blocks efficiently (batch)
   */
  getBlocks(heights: number[]): QBlock[] {
    const blocks: QBlock[] = []

    for (const height of heights) {
      const block = this.getBlock(height)
      if (block) {
        // Strip verification for transfer
        const { verification, ...simple } = block
        blocks.push(simple as QBlock)
      }
    }

    return blocks
  }

  /**
   * Track peer height announcement
   */
  trackPeerHeight(peerId: string, height: number): void {
    const existing = this.peerNeeds.get(peerId)
    const now = Date.now()

    if (existing) {
      // Check for gaps (peer might have missed blocks)
      if (height > existing.lastHeight + 1) {
        const missed: number[] = []
        for (let h = existing.lastHeight + 1; h < height; h++) {
          if (this.hasBlock(h)) {
            missed.push(h)
          }
        }
        if (missed.length > 0) {
          existing.missedBlocks = [...new Set([...existing.missedBlocks, ...missed])].slice(-20)
          console.log(`📊 [PROP QUEUE] Peer ${peerId.substring(0, 12)} missed ${missed.length} blocks`)
        }
      }
      existing.lastHeight = height
      existing.lastUpdate = now
    } else {
      this.peerNeeds.set(peerId, {
        peerId,
        lastHeight: height,
        lastUpdate: now,
        missedBlocks: [],
      })
    }

    // Cleanup old peer tracking (older than 5 minutes)
    for (const [pid, need] of this.peerNeeds) {
      if (now - need.lastUpdate > 5 * 60 * 1000) {
        this.peerNeeds.delete(pid)
      }
    }
  }

  /**
   * Queue a batch request (will be flushed after delay)
   */
  queueBatchRequest(peerId: string, heights: number[]): Promise<QBlock[]> {
    return new Promise((resolve) => {
      if (!this.pendingRequests.has(peerId)) {
        this.pendingRequests.set(peerId, [])
      }

      this.pendingRequests.get(peerId)!.push({ heights, resolve })
      this.stats.batchedRequests++

      // Schedule batch flush
      if (!this.batchFlushTimer) {
        this.batchFlushTimer = setTimeout(() => this.flushBatchRequests(), this.BATCH_DELAY_MS)
      }
    })
  }

  /**
   * Flush all pending batch requests
   */
  private flushBatchRequests(): void {
    this.batchFlushTimer = null

    for (const [peerId, requests] of this.pendingRequests) {
      // Combine all requested heights
      const allHeights = new Set<number>()
      for (const req of requests) {
        for (const h of req.heights) {
          allHeights.add(h)
        }
      }

      // Get blocks for all heights at once
      const blocks = this.getBlocks(Array.from(allHeights).sort((a, b) => a - b))

      // Resolve all requests with the combined result
      for (const req of requests) {
        const requestedBlocks = blocks.filter(b => req.heights.includes(b.header.height))
        req.resolve(requestedBlocks)
      }

      console.log(`📦 [PROP QUEUE] Flushed batch: ${requests.length} requests, ${allHeights.size} unique heights for ${peerId.substring(0, 12)}`)
    }

    this.pendingRequests.clear()
  }

  /**
   * Accelerate gossip for a specific block
   * Re-broadcasts the block through gossipsub
   */
  async accelerateGossip(height: number): Promise<boolean> {
    if (!this.libp2p) return false

    const block = this.getBlock(height)
    if (!block) {
      console.warn(`⚠️ [PROP QUEUE] Cannot accelerate gossip: block ${height} not in queue`)
      return false
    }

    try {
      const pubsub = this.libp2p.services.pubsub as GossipSub
      const { verification, ...simpleBlock } = block
      const encoded = msgpackEncode(simpleBlock)

      await pubsub.publish(TOPICS.BLOCKS, encoded)

      this.stats.gossipAccelerations++
      console.log(`🚀 [PROP QUEUE] Accelerated gossip for block ${height}`)
      return true
    } catch (error) {
      console.error(`❌ [PROP QUEUE] Failed to accelerate gossip:`, error)
      return false
    }
  }

  /**
   * Start the eager push loop
   */
  private startEagerPushLoop(): void {
    if (this.eagerPushTimer) return

    this.eagerPushTimer = setInterval(() => {
      this.processEagerPush()
    }, this.EAGER_PUSH_INTERVAL_MS)

    console.log('🔄 [PROP QUEUE] Eager push loop started')
  }

  /**
   * Process eager push - send hot blocks to peers who need them
   */
  private processEagerPush(): void {
    if (!this.libp2p || this.hotQueue.size === 0) return

    const connections = this.libp2p.getConnections()
    if (connections.length === 0) return

    // Find peers with missed blocks
    for (const [peerId, need] of this.peerNeeds) {
      if (need.missedBlocks.length === 0) continue

      // Get hot blocks that this peer missed
      const toSend = need.missedBlocks.filter(h => this.hotQueue.has(h))
      if (toSend.length === 0) continue

      // Check if we've already pushed to this peer
      const entry = this.hotQueue.get(toSend[0])
      if (entry && !entry.propagatedTo.has(peerId)) {
        // Trigger accelerated gossip for missed blocks
        for (const height of toSend.slice(0, 3)) { // Max 3 blocks per cycle
          this.accelerateGossip(height)
          entry.propagatedTo.add(peerId)
          this.stats.eagerPushes++
        }

        // Clear sent blocks from missed list
        need.missedBlocks = need.missedBlocks.filter(h => !toSend.includes(h))
      }
    }
  }

  /**
   * Trigger immediate eager push for a specific block
   */
  private triggerEagerPush(height: number): void {
    const entry = this.hotQueue.get(height)
    if (!entry || !this.libp2p) return

    const startTime = Date.now()

    // Push to all connected peers who don't have it
    const connections = this.libp2p.getConnections()
    for (const conn of connections) {
      const peerId = conn.remotePeer.toString()
      const need = this.peerNeeds.get(peerId)

      // If peer height is below this block, they might need it
      if (need && need.lastHeight < height && !entry.propagatedTo.has(peerId)) {
        // Will be handled by next gossip cycle
        entry.propagatedTo.add(peerId)
      }
    }

    // Track propagation time
    const elapsed = Date.now() - startTime
    this.stats.propagationTimes.push(elapsed)
    if (this.stats.propagationTimes.length > this.MAX_PROPAGATION_SAMPLES) {
      this.stats.propagationTimes.shift()
    }
    this.stats.totalPropagations++
  }

  /**
   * Calculate block priority
   */
  private calculatePriority(height: number, highestHot: number | null): BlockPriority {
    if (highestHot === null) return BlockPriority.HOT

    const diff = highestHot - height
    if (diff < this.HOT_QUEUE_SIZE) return BlockPriority.HOT
    if (diff < this.WARM_QUEUE_SIZE) return BlockPriority.WARM
    return BlockPriority.COLD
  }

  /**
   * Promote a block from warm to hot queue
   */
  private promoteToHot(height: number, entry: QueuedBlock): void {
    this.warmQueue.delete(height)
    entry.priority = BlockPriority.HOT
    this.hotQueue.set(height, entry)
    this.trimQueue(this.hotQueue, this.HOT_QUEUE_SIZE)
    console.log(`⬆️ [PROP QUEUE] Promoted block ${height} to HOT (requested ${entry.requestCount}x)`)
  }

  /**
   * Trim queue to max size (remove lowest heights)
   */
  private trimQueue(queue: Map<number, QueuedBlock>, maxSize: number): void {
    while (queue.size > maxSize) {
      const lowestHeight = Math.min(...queue.keys())
      const entry = queue.get(lowestHeight)
      queue.delete(lowestHeight)

      // Demote to next tier if applicable
      if (entry && entry.priority === BlockPriority.HOT) {
        entry.priority = BlockPriority.WARM
        this.warmQueue.set(lowestHeight, entry)
        this.trimQueue(this.warmQueue, this.WARM_QUEUE_SIZE)
      }
    }
  }

  /**
   * Get highest height in a queue
   */
  private getHighestHeight(queue: Map<number, QueuedBlock>): number | null {
    if (queue.size === 0) return null
    return Math.max(...queue.keys())
  }

  /**
   * Check if block exists in any queue
   */
  private hasBlock(height: number): boolean {
    return this.hotQueue.has(height) || this.warmQueue.has(height) || blockCache.has(height)
  }

  /**
   * Get propagation statistics
   */
  getStats(): PropagationStats {
    const avgTime = this.stats.propagationTimes.length > 0
      ? this.stats.propagationTimes.reduce((a, b) => a + b, 0) / this.stats.propagationTimes.length
      : 0

    return {
      hotQueueSize: this.hotQueue.size,
      warmQueueSize: this.warmQueue.size,
      totalPropagations: this.stats.totalPropagations,
      eagerPushes: this.stats.eagerPushes,
      batchedRequests: this.stats.batchedRequests,
      gossipAccelerations: this.stats.gossipAccelerations,
      avgPropagationTimeMs: Math.round(avgTime * 100) / 100,
      peersTracked: this.peerNeeds.size,
    }
  }

  /**
   * Stop the propagation queue
   */
  stop(): void {
    if (this.eagerPushTimer) {
      clearInterval(this.eagerPushTimer)
      this.eagerPushTimer = null
    }
    if (this.batchFlushTimer) {
      clearTimeout(this.batchFlushTimer)
      this.flushBatchRequests() // Flush any pending
    }
    console.log('🛑 [PROP QUEUE] Block propagation queue stopped')
  }
}

// Global instance
export const propagationQueue = new BlockPropagationQueue()

/**
 * Get propagation queue statistics
 */
export function getPropagationStats(): PropagationStats {
  return propagationQueue.getStats()
}
