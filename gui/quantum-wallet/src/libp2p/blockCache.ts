/**
 * Block Cache - In-Memory Block Storage
 *
 * Stores recently verified blocks for serving to other browser nodes.
 * This enables browser-to-browser block synchronization without
 * requiring full node involvement.
 *
 * v3.5.x: Browser P2P Network Contribution - Feature 5 (Part 1)
 */

import { BLOCK_CACHE_CONFIG } from './config'
import type { VerifiedBlock, BlockCacheStats, QBlock } from './types'

/**
 * Block Cache Implementation
 *
 * Uses a Map for O(1) access by height and maintains blocks sorted by height.
 * Automatically evicts oldest blocks when capacity is reached.
 */
export class BlockCache {
  private cache: Map<number, VerifiedBlock> = new Map()
  private maxSize: number
  private estimatedMemoryPerBlock: number = 100 * 1024 // ~100KB estimate

  // Statistics
  private cacheHits: number = 0
  private cacheMisses: number = 0
  private requestsServed: number = 0

  constructor(maxSize: number = BLOCK_CACHE_CONFIG.MAX_SIZE) {
    this.maxSize = maxSize
    console.log(`🗄️ [BLOCK CACHE] Initialized with max size: ${maxSize} blocks`)
  }

  /**
   * Add a verified block to the cache
   *
   * @param block - The verified block to add
   */
  add(block: VerifiedBlock): void {
    const height = block.header.height

    // Don't add duplicates
    if (this.cache.has(height)) {
      return
    }

    // Evict oldest blocks if at capacity
    while (this.cache.size >= this.maxSize) {
      const oldestHeight = this.getLowestHeight()
      if (oldestHeight !== null) {
        this.cache.delete(oldestHeight)
        console.log(`🗑️ [BLOCK CACHE] Evicted block ${oldestHeight} (cache full)`)
      } else {
        break
      }
    }

    // Add the new block
    this.cache.set(height, block)
    console.log(`📥 [BLOCK CACHE] Added block ${height} (cache size: ${this.cache.size}/${this.maxSize})`)
  }

  /**
   * Get a block by height
   *
   * @param height - Block height to retrieve
   * @returns The block if found, undefined otherwise
   */
  get(height: number): VerifiedBlock | undefined {
    const block = this.cache.get(height)
    if (block) {
      this.cacheHits++
      return block
    }
    this.cacheMisses++
    return undefined
  }

  /**
   * Get a range of blocks
   *
   * @param startHeight - Start height (inclusive)
   * @param endHeight - End height (inclusive)
   * @returns Array of blocks in the range
   */
  getRange(startHeight: number, endHeight: number): VerifiedBlock[] {
    const blocks: VerifiedBlock[] = []

    // Limit range to max blocks per request
    const limitedEnd = Math.min(endHeight, startHeight + BLOCK_CACHE_CONFIG.MAX_BLOCKS_PER_REQUEST - 1)

    for (let height = startHeight; height <= limitedEnd; height++) {
      const block = this.cache.get(height)
      if (block) {
        blocks.push(block)
        this.cacheHits++
      } else {
        this.cacheMisses++
      }
    }

    this.requestsServed++
    return blocks
  }

  /**
   * Check if a block exists in cache
   *
   * @param height - Block height to check
   * @returns True if block is in cache
   */
  has(height: number): boolean {
    return this.cache.has(height)
  }

  /**
   * Get the lowest block height in cache
   */
  getLowestHeight(): number | null {
    if (this.cache.size === 0) return null
    return Math.min(...this.cache.keys())
  }

  /**
   * Get the highest block height in cache
   */
  getHighestHeight(): number | null {
    if (this.cache.size === 0) return null
    return Math.max(...this.cache.keys())
  }

  /**
   * Get current cache size
   */
  getSize(): number {
    return this.cache.size
  }

  /**
   * Get cache statistics
   */
  getStats(): BlockCacheStats {
    const lowestHeight = this.getLowestHeight()
    const highestHeight = this.getHighestHeight()

    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      lowestHeight: lowestHeight ?? 0,
      highestHeight: highestHeight ?? 0,
      memoryUsed: this.cache.size * this.estimatedMemoryPerBlock,
      hitRate: this.cacheHits + this.cacheMisses > 0
        ? (this.cacheHits / (this.cacheHits + this.cacheMisses)) * 100
        : 0,
      requestsServed: this.requestsServed,
    }
  }

  /**
   * Clear the cache
   */
  clear(): void {
    this.cache.clear()
    this.cacheHits = 0
    this.cacheMisses = 0
    this.requestsServed = 0
    console.log(`🗑️ [BLOCK CACHE] Cleared`)
  }

  /**
   * Get all heights in cache (sorted)
   */
  getHeights(): number[] {
    return Array.from(this.cache.keys()).sort((a, b) => a - b)
  }

  /**
   * Get blocks as simple QBlock array (without verification results)
   */
  getBlocksAsSimple(startHeight: number, endHeight: number): QBlock[] {
    const blocks = this.getRange(startHeight, endHeight)
    return blocks.map(vb => {
      // Create a copy without the verification property
      const { verification, ...block } = vb
      return block as QBlock
    })
  }
}

// Global block cache instance
export const blockCache = new BlockCache()

/**
 * Get global block cache statistics
 */
export function getBlockCacheStats(): BlockCacheStats {
  return blockCache.getStats()
}

/**
 * Add a block to the global cache
 */
export function cacheBlock(block: VerifiedBlock): void {
  blockCache.add(block)
}

/**
 * Get a block from the global cache
 */
export function getCachedBlock(height: number): VerifiedBlock | undefined {
  return blockCache.get(height)
}

/**
 * Get a range of blocks from the global cache
 */
export function getCachedBlockRange(startHeight: number, endHeight: number): VerifiedBlock[] {
  return blockCache.getRange(startHeight, endHeight)
}
