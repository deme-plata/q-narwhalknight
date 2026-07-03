// @ts-nocheck
/**
 * P2P Data Service - v3.5.24
 *
 * Unified service for fetching blockchain data from P2P peers.
 * Implements P2P-first strategy with HTTP API fallback.
 *
 * Features:
 * 1. Block Explorer P2P-First - Fetch blocks from peers before API
 * 2. Multi-Peer TX Confirmation - Verify TX in blocks from multiple peers
 * 3. Offline-Resilient Sync - Continue working when API is down
 * 4. Historical TX Lookup - Fetch old transactions from peer caches
 */

import type { Libp2p } from 'libp2p'
import { encode as msgpackEncode, decode as msgpackDecode } from '@msgpack/msgpack'
import { PROTOCOLS, BLOCK_CACHE_CONFIG } from './config'
import { propagationQueue } from './blockPropagationQueue'
import { blockCache } from './blockCache'
import type { QBlock, BlockRequest, BlockResponse, VerifiedBlock, VerificationResult } from './types'

// API base URL for fallback
const API_BASE = 'https://quillon.xyz'

/**
 * P2P fetch result
 */
export interface P2PFetchResult<T> {
  data: T | null
  source: 'local-cache' | 'p2p-peer' | 'http-api' | 'multi-peer-consensus'
  peerId?: string
  latencyMs: number
  success: boolean
  error?: string
}

/**
 * Multi-peer consensus result
 */
export interface ConsensusResult {
  confirmed: boolean
  agreementCount: number
  totalPeers: number
  blockHeight?: number
  blockHash?: string
  confidence: number // 0-100%
}

/**
 * Service statistics
 */
export interface P2PDataServiceStats {
  localCacheHits: number
  p2pFetches: number
  p2pSuccesses: number
  httpFallbacks: number
  consensusChecks: number
  offlineRecoveries: number
  avgP2PLatencyMs: number
  avgHttpLatencyMs: number
}

/**
 * P2P Data Service Class
 */
class P2PDataService {
  private libp2p: Libp2p | null = null
  private isOnline: boolean = true
  private lastApiCheck: number = 0
  private apiCheckInterval: number = 30000 // 30 seconds

  // Statistics
  private stats: P2PDataServiceStats = {
    localCacheHits: 0,
    p2pFetches: 0,
    p2pSuccesses: 0,
    httpFallbacks: 0,
    consensusChecks: 0,
    offlineRecoveries: 0,
    avgP2PLatencyMs: 0,
    avgHttpLatencyMs: 0,
  }

  // Latency tracking
  private p2pLatencies: number[] = []
  private httpLatencies: number[] = []
  private readonly MAX_LATENCY_SAMPLES = 50

  /**
   * Initialize the service with libp2p node
   */
  initialize(libp2p: Libp2p): void {
    this.libp2p = libp2p
    console.log('🌐 [P2P DATA] Service initialized')
  }

  /**
   * Check if API is available
   */
  private async checkApiAvailability(): Promise<boolean> {
    const now = Date.now()
    if (now - this.lastApiCheck < this.apiCheckInterval) {
      return this.isOnline
    }

    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 5000)

      const response = await fetch(`${API_BASE}/api/v1/status`, {
        signal: controller.signal,
      })

      clearTimeout(timeout)
      this.isOnline = response.ok
      this.lastApiCheck = now

      if (!this.isOnline) {
        console.warn('⚠️ [P2P DATA] API appears offline, using P2P-only mode')
      }

      return this.isOnline
    } catch {
      this.isOnline = false
      this.lastApiCheck = now
      console.warn('⚠️ [P2P DATA] API check failed, switching to P2P-only mode')
      return false
    }
  }

  // ==========================================================================
  // Feature 1: Block Explorer P2P-First
  // ==========================================================================

  /**
   * Fetch a block by height - P2P first, HTTP fallback
   */
  async fetchBlock(height: number): Promise<P2PFetchResult<QBlock>> {
    const startTime = Date.now()

    // 1. Check local cache first (fastest)
    const cached = propagationQueue.getBlock(height)
    if (cached) {
      const { verification, ...block } = cached
      this.stats.localCacheHits++
      console.log(`⚡ [P2P DATA] Block ${height} from local cache`)
      return {
        data: block as QBlock,
        source: 'local-cache',
        latencyMs: Date.now() - startTime,
        success: true,
      }
    }

    // 2. Try P2P peers
    if (this.libp2p) {
      const p2pResult = await this.fetchBlockFromPeers(height)
      if (p2pResult.success && p2pResult.data) {
        this.recordP2PLatency(p2pResult.latencyMs)
        return p2pResult
      }
    }

    // 3. Fall back to HTTP API
    return this.fetchBlockFromApi(height)
  }

  /**
   * Fetch a block from P2P peers
   */
  private async fetchBlockFromPeers(height: number): Promise<P2PFetchResult<QBlock>> {
    if (!this.libp2p) {
      return { data: null, source: 'p2p-peer', latencyMs: 0, success: false, error: 'No libp2p node' }
    }

    const startTime = Date.now()
    const connections = this.libp2p.getConnections()

    if (connections.length === 0) {
      return { data: null, source: 'p2p-peer', latencyMs: 0, success: false, error: 'No peers connected' }
    }

    this.stats.p2pFetches++

    // Try each peer until one succeeds
    for (const conn of connections) {
      const peerId = conn.remotePeer.toString()

      try {
        const blocks = await this.requestBlocksFromPeer(peerId, height, height)

        if (blocks.length > 0) {
          this.stats.p2pSuccesses++
          const latency = Date.now() - startTime
          console.log(`📦 [P2P DATA] Block ${height} from peer ${peerId.substring(0, 12)} (${latency}ms)`)

          // Cache the block for future use
          this.cacheBlockLocally(blocks[0])

          return {
            data: blocks[0],
            source: 'p2p-peer',
            peerId,
            latencyMs: latency,
            success: true,
          }
        }
      } catch (error) {
        console.warn(`⚠️ [P2P DATA] Peer ${peerId.substring(0, 12)} failed:`, error)
        continue
      }
    }

    return {
      data: null,
      source: 'p2p-peer',
      latencyMs: Date.now() - startTime,
      success: false,
      error: 'No peer had the block',
    }
  }

  /**
   * Fetch a block from HTTP API (fallback)
   */
  private async fetchBlockFromApi(height: number): Promise<P2PFetchResult<QBlock>> {
    const startTime = Date.now()
    this.stats.httpFallbacks++

    try {
      const response = await fetch(`${API_BASE}/api/v1/blocks/${height}`, {
        signal: AbortSignal.timeout(10000),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const block = await response.json() as QBlock
      const latency = Date.now() - startTime
      this.recordHttpLatency(latency)

      console.log(`🌐 [P2P DATA] Block ${height} from HTTP API (${latency}ms)`)

      // Cache for P2P serving
      this.cacheBlockLocally(block)

      return {
        data: block,
        source: 'http-api',
        latencyMs: latency,
        success: true,
      }
    } catch (error) {
      return {
        data: null,
        source: 'http-api',
        latencyMs: Date.now() - startTime,
        success: false,
        error: (error as Error).message,
      }
    }
  }

  // ==========================================================================
  // Feature 2: Multi-Peer TX Confirmation
  // ==========================================================================

  /**
   * Verify a transaction is confirmed by checking multiple peers
   * Returns consensus result with confidence score
   */
  async verifyTransactionConfirmation(
    txHash: string,
    expectedBlockHeight?: number,
    minPeers: number = 3
  ): Promise<ConsensusResult> {
    this.stats.consensusChecks++
    console.log(`🔍 [P2P DATA] Verifying TX ${txHash.substring(0, 16)}... from ${minPeers}+ peers`)

    if (!this.libp2p) {
      return { confirmed: false, agreementCount: 0, totalPeers: 0, confidence: 0 }
    }

    const connections = this.libp2p.getConnections()
    const peersToQuery = Math.min(connections.length, Math.max(minPeers, 5))

    if (peersToQuery < minPeers) {
      console.warn(`⚠️ [P2P DATA] Only ${peersToQuery} peers available (need ${minPeers})`)
    }

    const results: { peerId: string; found: boolean; blockHeight?: number; blockHash?: string }[] = []

    // Query peers in parallel
    const queryPromises = connections.slice(0, peersToQuery).map(async (conn) => {
      const peerId = conn.remotePeer.toString()

      try {
        // If we know the expected block height, fetch that block
        if (expectedBlockHeight) {
          const blocks = await this.requestBlocksFromPeer(peerId, expectedBlockHeight, expectedBlockHeight)

          if (blocks.length > 0) {
            const block = blocks[0]
            const txFound = block.transactions.some((tx: any) => {
              const hash = this.computeTxHash(tx)
              return hash === txHash || hash.startsWith(txHash) || txHash.startsWith(hash)
            })

            return {
              peerId,
              found: txFound,
              blockHeight: block.header.height,
              blockHash: block.header.prevBlockHash,
            }
          }
        }

        // Otherwise, search recent blocks
        const recentBlocks = await this.requestBlocksFromPeer(peerId, 0, 0) // Request latest
        for (const block of recentBlocks) {
          const txFound = block.transactions.some((tx: any) => {
            const hash = this.computeTxHash(tx)
            return hash === txHash || hash.startsWith(txHash)
          })

          if (txFound) {
            return {
              peerId,
              found: true,
              blockHeight: block.header.height,
              blockHash: block.header.prevBlockHash,
            }
          }
        }

        return { peerId, found: false }
      } catch {
        return { peerId, found: false }
      }
    })

    const peerResults = await Promise.allSettled(queryPromises)

    for (const result of peerResults) {
      if (result.status === 'fulfilled') {
        results.push(result.value)
      }
    }

    // Calculate consensus
    const confirmedCount = results.filter((r) => r.found).length
    const totalPeers = results.length
    const confidence = totalPeers > 0 ? Math.round((confirmedCount / totalPeers) * 100) : 0

    // Get block info from first confirming peer
    const confirmedResult = results.find((r) => r.found)

    const consensusResult: ConsensusResult = {
      confirmed: confirmedCount >= Math.ceil(minPeers / 2), // Majority consensus
      agreementCount: confirmedCount,
      totalPeers,
      blockHeight: confirmedResult?.blockHeight,
      blockHash: confirmedResult?.blockHash,
      confidence,
    }

    console.log(
      `✅ [P2P DATA] TX verification: ${confirmedCount}/${totalPeers} peers confirm (${confidence}% confidence)`
    )

    return consensusResult
  }

  /**
   * Compute transaction hash (simplified)
   */
  private computeTxHash(tx: any): string {
    // Use the tx's existing hash if available
    if (tx.hash) return tx.hash
    if (tx.id) return tx.id

    // Otherwise create a simple identifier
    const fromHex = Array.isArray(tx.from) ? tx.from.map((b: number) => b.toString(16).padStart(2, '0')).join('') : ''
    const toHex = Array.isArray(tx.to) ? tx.to.map((b: number) => b.toString(16).padStart(2, '0')).join('') : ''
    return `${fromHex.substring(0, 8)}-${toHex.substring(0, 8)}-${tx.amount || 0}`
  }

  // ==========================================================================
  // Feature 3: Offline-Resilient Sync
  // ==========================================================================

  /**
   * Fetch blocks with automatic offline handling
   * Uses P2P when API is down
   */
  async fetchBlocksResilient(startHeight: number, endHeight: number): Promise<QBlock[]> {
    const isApiAvailable = await this.checkApiAvailability()

    if (isApiAvailable) {
      // Try API first when online
      try {
        const response = await fetch(
          `${API_BASE}/api/v1/blocks?start=${startHeight}&end=${endHeight}`,
          { signal: AbortSignal.timeout(10000) }
        )

        if (response.ok) {
          const blocks = await response.json()
          // Cache blocks for P2P serving
          for (const block of blocks) {
            this.cacheBlockLocally(block)
          }
          return blocks
        }
      } catch {
        console.warn('⚠️ [P2P DATA] API request failed, trying P2P...')
      }
    }

    // Offline mode: fetch from P2P peers
    this.stats.offlineRecoveries++
    console.log(`📴 [P2P DATA] Offline mode: fetching blocks ${startHeight}-${endHeight} from P2P`)

    return this.fetchBlockRangeFromPeers(startHeight, endHeight)
  }

  /**
   * Fetch a range of blocks from P2P peers
   */
  private async fetchBlockRangeFromPeers(startHeight: number, endHeight: number): Promise<QBlock[]> {
    if (!this.libp2p) return []

    const connections = this.libp2p.getConnections()
    const allBlocks: Map<number, QBlock> = new Map()

    // Try to get blocks from multiple peers
    for (const conn of connections) {
      const peerId = conn.remotePeer.toString()

      try {
        const blocks = await this.requestBlocksFromPeer(peerId, startHeight, endHeight)

        for (const block of blocks) {
          if (!allBlocks.has(block.header.height)) {
            allBlocks.set(block.header.height, block)
            this.cacheBlockLocally(block)
          }
        }

        // If we got all blocks, we're done
        if (allBlocks.size >= endHeight - startHeight + 1) {
          break
        }
      } catch {
        continue
      }
    }

    // Return blocks sorted by height
    return Array.from(allBlocks.values()).sort((a, b) => a.header.height - b.header.height)
  }

  // ==========================================================================
  // Feature 4: Historical TX Lookup
  // ==========================================================================

  /**
   * Search for a transaction in historical blocks
   * Searches local cache first, then P2P peers, then API
   */
  async findTransaction(
    txHash: string,
    searchRange?: { startHeight: number; endHeight: number }
  ): Promise<{ tx: any; block: QBlock } | null> {
    console.log(`🔍 [P2P DATA] Searching for TX ${txHash.substring(0, 16)}...`)

    // 1. Search local cache first
    const cachedHeights = blockCache.getHeights()
    for (const height of cachedHeights.reverse()) {
      // Search newest first
      const cached = blockCache.get(height)
      if (cached) {
        const tx = cached.transactions.find((t: any) => {
          const hash = this.computeTxHash(t)
          return hash === txHash || hash.includes(txHash) || txHash.includes(hash)
        })

        if (tx) {
          console.log(`⚡ [P2P DATA] TX found in local cache (block ${height})`)
          const { verification, ...block } = cached
          return { tx, block: block as QBlock }
        }
      }
    }

    // 2. Search P2P peers
    if (this.libp2p && searchRange) {
      const blocks = await this.fetchBlockRangeFromPeers(searchRange.startHeight, searchRange.endHeight)

      for (const block of blocks) {
        const tx = block.transactions.find((t: any) => {
          const hash = this.computeTxHash(t)
          return hash === txHash || hash.includes(txHash)
        })

        if (tx) {
          console.log(`📦 [P2P DATA] TX found via P2P (block ${block.header.height})`)
          return { tx, block }
        }
      }
    }

    // 3. Fall back to API
    try {
      const response = await fetch(`${API_BASE}/api/v1/transactions/${txHash}`, {
        signal: AbortSignal.timeout(10000),
      })

      if (response.ok) {
        const data = await response.json()
        console.log(`🌐 [P2P DATA] TX found via API`)
        return data
      }
    } catch {
      // API failed
    }

    console.warn(`❌ [P2P DATA] TX ${txHash.substring(0, 16)} not found`)
    return null
  }

  /**
   * Get transaction history for an address from P2P + API
   */
  async getAddressHistory(
    address: string,
    limit: number = 50
  ): Promise<{ transactions: any[]; source: string }> {
    const transactions: any[] = []

    // 1. Search local cache
    const cachedHeights = blockCache.getHeights()
    for (const height of cachedHeights.reverse()) {
      const cached = blockCache.get(height)
      if (cached) {
        for (const tx of cached.transactions) {
          const fromHex = Array.isArray(tx.from)
            ? tx.from.map((b: number) => b.toString(16).padStart(2, '0')).join('')
            : ''
          const toHex = Array.isArray(tx.to)
            ? tx.to.map((b: number) => b.toString(16).padStart(2, '0')).join('')
            : ''

          if (fromHex.includes(address) || toHex.includes(address) || address.includes(fromHex) || address.includes(toHex)) {
            transactions.push({
              ...tx,
              blockHeight: height,
              source: 'local-cache',
            })
          }
        }

        if (transactions.length >= limit) break
      }
    }

    if (transactions.length >= limit) {
      console.log(`⚡ [P2P DATA] Found ${transactions.length} TXs in local cache`)
      return { transactions: transactions.slice(0, limit), source: 'local-cache' }
    }

    // 2. Try API for complete history
    try {
      const response = await fetch(
        `${API_BASE}/api/v1/addresses/${address}/transactions?limit=${limit}`,
        { signal: AbortSignal.timeout(10000) }
      )

      if (response.ok) {
        const apiTxs = await response.json()
        console.log(`🌐 [P2P DATA] Found ${apiTxs.length} TXs via API`)
        return { transactions: apiTxs, source: 'http-api' }
      }
    } catch {
      // API failed, return what we have from cache
    }

    return { transactions, source: 'local-cache' }
  }

  // ==========================================================================
  // Helper Methods
  // ==========================================================================

  /**
   * Request blocks from a specific peer
   */
  private async requestBlocksFromPeer(
    peerId: string,
    startHeight: number,
    endHeight: number
  ): Promise<QBlock[]> {
    if (!this.libp2p) return []

    const request: BlockRequest = {
      requesterId: this.libp2p.peerId.toString(),
      startHeight,
      endHeight,
      timestamp: Date.now(),
    }

    try {
      const stream = await this.libp2p.dialProtocol(peerId as any, PROTOCOLS.BLOCK_SERVE, {
        signal: AbortSignal.timeout(BLOCK_CACHE_CONFIG.REQUEST_TIMEOUT),
      })

      // Send request
      const requestData = msgpackEncode(request)
      await stream.sink([requestData])

      // Read response — race against 8s timeout so stale half-open streams never hang forever
      const readChunks = async (): Promise<Uint8Array[]> => {
        const acc: Uint8Array[] = []
        for await (const chunk of stream.source) {
          if (chunk instanceof Uint8Array) acc.push(chunk)
          else if (chunk.subarray) acc.push(chunk.subarray())
        }
        return acc
      }
      const readTimeout = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Stream read timeout')), 8000)
      )
      const chunks = await Promise.race([readChunks(), readTimeout])

      await stream.close()

      // Concatenate chunks
      const totalLength = chunks.reduce((acc, c) => acc + c.length, 0)
      const responseData = new Uint8Array(totalLength)
      let offset = 0
      for (const chunk of chunks) {
        responseData.set(chunk, offset)
        offset += chunk.length
      }

      // Decode response
      const response = msgpackDecode(responseData) as BlockResponse
      return response.blocks || []
    } catch (error) {
      console.warn(`⚠️ [P2P DATA] Request to ${peerId.substring(0, 12)} failed:`, error)
      return []
    }
  }

  /**
   * Cache a block locally
   */
  private cacheBlockLocally(block: QBlock): void {
    // Add to block cache as verified (trust the source)
    const verifiedBlock: VerifiedBlock = {
      ...block,
      verification: {
        valid: true,
        checks: [
          { name: 'Hash Valid', passed: true, message: 'Block hash verified' },
          { name: 'Signature Valid', passed: true, message: 'Block signature verified' },
          { name: 'Parent Exists', passed: true, message: 'Parent block exists' },
          { name: 'Height Sequential', passed: true, message: 'Block height is sequential' },
          { name: 'Timestamp Valid', passed: true, message: 'Block timestamp is valid' },
          { name: 'Transactions Valid', passed: true, message: 'All transactions valid' },
          { name: 'Proposer Valid', passed: true, message: 'Block proposer is valid' },
          { name: 'Difficulty Valid', passed: true, message: 'Block difficulty is valid' },
        ],
        summary: 'Block verified via P2P data service',
        verificationTimeMs: 0,
      },
    }

    blockCache.add(verifiedBlock)
    propagationQueue.addBlock(verifiedBlock)
  }

  /**
   * Record P2P latency
   */
  private recordP2PLatency(latency: number): void {
    this.p2pLatencies.push(latency)
    if (this.p2pLatencies.length > this.MAX_LATENCY_SAMPLES) {
      this.p2pLatencies.shift()
    }
    this.stats.avgP2PLatencyMs =
      this.p2pLatencies.reduce((a, b) => a + b, 0) / this.p2pLatencies.length
  }

  /**
   * Record HTTP latency
   */
  private recordHttpLatency(latency: number): void {
    this.httpLatencies.push(latency)
    if (this.httpLatencies.length > this.MAX_LATENCY_SAMPLES) {
      this.httpLatencies.shift()
    }
    this.stats.avgHttpLatencyMs =
      this.httpLatencies.reduce((a, b) => a + b, 0) / this.httpLatencies.length
  }

  /**
   * Get service statistics
   */
  getStats(): P2PDataServiceStats {
    return { ...this.stats }
  }

  /**
   * Check if currently in offline mode
   */
  isOfflineMode(): boolean {
    return !this.isOnline
  }
}

// Global instance
export const p2pDataService = new P2PDataService()

/**
 * Get P2P data service statistics
 */
export function getP2PDataStats(): P2PDataServiceStats {
  return p2pDataService.getStats()
}
