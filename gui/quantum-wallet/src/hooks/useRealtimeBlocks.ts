/**
 * useRealtimeBlocks Hook
 *
 * Real-time block streaming via libp2p Gossipsub with HTTP fallback.
 * Primary: P2P gossipsub for <100ms latency
 * Fallback: HTTP polling for reliability
 *
 * Usage:
 * ```typescript
 * const { latestBlock, blockHistory, isSubscribed } = useRealtimeBlocks()
 * ```
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { useLibP2P } from '../contexts/LibP2PContext'
import { TOPICS, RELAY_CONFIG } from '../libp2p/config'
import { decodeBlock, validateBlock, createBlockSummary, logBlockSummary } from '../libp2p/decoder'
import { verifyBlock, getVerificationMetrics, VERIFICATION_METRICS } from '../libp2p/verification'
import { verificationReporter } from '../libp2p/verificationReporter'
import { recordBlockReceived, recordVerification } from '../libp2p/telemetry'
import { cacheBlock } from '../libp2p/blockCache'
import { propagationQueue } from '../libp2p/blockPropagationQueue'
import { browserPeerDiscovery } from '../libp2p/browserPeerDiscovery'
import type { QBlock, BlockSummary, VerifiedBlock, VerificationResult, RelayStats } from '../libp2p/types'
import type { GossipSub } from '@libp2p/gossipsub'
import { encode as msgpackEncode } from '@msgpack/msgpack'

// HTTP API base URL
const API_BASE = 'https://quillon.xyz'

/**
 * Convert hex string to Uint8Array (browser-safe, no Buffer dependency)
 */
function hexToUint8Array(hex: string): Uint8Array {
  if (!hex || hex.length === 0) return new Uint8Array(32)
  // Remove '0x' prefix if present
  const cleanHex = hex.startsWith('0x') ? hex.slice(2) : hex
  const bytes = new Uint8Array(cleanHex.length / 2)
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(cleanHex.substr(i * 2, 2), 16)
  }
  return bytes
}

/**
 * Backpressure Configuration
 * Prevents memory exhaustion if blocks arrive faster than browser can process
 */
const MAX_BLOCK_HISTORY = 100 // Cap at 100 blocks (~2MB memory)
const BUFFER_OVERFLOW_WARNING_THRESHOLD = 80 // Warn at 80% capacity

// ============================================================================
// Block Relay State (v3.5.x: Browser P2P Network Contribution)
// ============================================================================

// Seen block cache for deduplication (prevents relay loops)
const seenBlockHashes = new Set<string>()

// Rate limiting state
const relayTimestamps: number[] = []

// Global relay statistics
const relayStats: RelayStats = {
  blocksRelayed: 0,
  blocksRelayedLastHour: 0,
  duplicatesFiltered: 0,
  invalidBlocksFiltered: 0,
  rateLimitDrops: 0,
  lastRelayTime: 0,
}

/**
 * Get block relay statistics
 */
export function getRelayStats(): RelayStats {
  return { ...relayStats }
}

/**
 * Generate a unique hash for a block (for deduplication)
 */
function generateBlockHash(block: QBlock): string {
  // Use height + timestamp + proposer as unique identifier
  return `${block.header.height}-${block.header.timestamp}-${block.header.proposer}`
}

/**
 * Check if relay is rate limited
 */
function isRelayRateLimited(): boolean {
  const now = Date.now()
  const windowStart = now - RELAY_CONFIG.RATE_LIMIT_WINDOW

  // Remove old timestamps
  while (relayTimestamps.length > 0 && relayTimestamps[0] < windowStart) {
    relayTimestamps.shift()
  }

  return relayTimestamps.length >= RELAY_CONFIG.MAX_RELAYS_PER_SECOND
}

/**
 * Clean up seen cache when it gets too large
 */
function cleanupSeenCache(): void {
  if (seenBlockHashes.size > RELAY_CONFIG.SEEN_CACHE_SIZE) {
    // Remove oldest entries (convert to array, slice, convert back)
    const entries = Array.from(seenBlockHashes)
    seenBlockHashes.clear()
    entries.slice(-RELAY_CONFIG.SEEN_CACHE_SIZE / 2).forEach(h => seenBlockHashes.add(h))
    console.log(`🧹 [RELAY] Cleaned seen cache: ${entries.length} -> ${seenBlockHashes.size}`)
  }
}

export interface UseRealtimeBlocksResult {
  // Latest block received
  latestBlock: QBlock | null

  // Block summary for UI display
  latestBlockSummary: BlockSummary | null

  // Recent block history (last 100 blocks) - with verification results
  blockHistory: VerifiedBlock[]

  // Subscription status
  isSubscribed: boolean

  // Error state
  error: Error | null

  // v3.5.10: Light client verification stats
  verificationStats: {
    blocksVerified: number
    blocksValid: number
    blocksInvalid: number
    verificationEnabled: boolean
  }

  // v3.5.x: Block relay stats
  relayStats: RelayStats

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
  const [blockHistory, setBlockHistory] = useState<VerifiedBlock[]>([])
  const [isSubscribed, setIsSubscribed] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [verificationStats, setVerificationStats] = useState({
    blocksVerified: 0,
    blocksValid: 0,
    blocksInvalid: 0,
    verificationEnabled: true, // v3.5.10: Light client verification enabled by default
  })

  // Track highest accepted block height for rogue peer detection
  const highestAcceptedHeightRef = useRef<number>(0)

  /**
   * Handle incoming block messages
   * v3.5.10: Now async for light client verification
   */
  const handleBlockMessage = useCallback(async (event: CustomEvent) => {
    try {
      const messageData = event.detail?.data

      if (!messageData) {
        console.warn('[REALTIME BLOCKS] Received message with no data')
        return
      }

      console.log(`📦 [REALTIME BLOCKS] Got ${messageData.length} bytes of block data`)

      // Decode block from binary data
      let block: QBlock | null = null
      try {
        block = decodeBlock(messageData)
        console.log(`🔓 [REALTIME BLOCKS] Decode returned:`, block ? `Block #${block.header.height}` : 'null')
      } catch (decodeError) {
        console.error('❌ [REALTIME BLOCKS] Decode threw exception:', decodeError)
        return
      }

      if (!block) {
        console.warn('[REALTIME BLOCKS] Failed to decode block - returned null')
        return
      }

      // Debug: log block structure before validation
      console.log('🎯 [REALTIME BLOCKS] Block decoded successfully! Adding to state...', {
        height: block.header?.height,
        timestamp: block.header?.timestamp,
        networkId: block.header?.networkId,
        phase: block.header?.phase,
        txCount: block.transactions?.length,
      })

      // Validate block structure
      if (!validateBlock(block)) {
        console.warn('[REALTIME BLOCKS] Invalid block structure - validation failed')
        console.warn('[REALTIME BLOCKS] Block that failed:', JSON.stringify({
          hasHeader: !!block.header,
          height: block.header?.height,
          timestamp: block.header?.timestamp,
          networkId: block.header?.networkId,
          phase: block.header?.phase,
        }))
        return
      }

      // HEIGHT SANITY CHECK: Reject blocks from rogue peers with unreasonable heights
      // Mirror server-side logic: max_reasonable = max(our_height * 3, our_height + 500_000)
      const ourHeight = Math.max(lastHttpBlockHeight.current, highestAcceptedHeightRef.current)
      if (ourHeight > 0) {
        const maxReasonable = Math.max(ourHeight * 3, ourHeight + 500_000)
        if (block.header.height > maxReasonable) {
          console.warn(
            `🚨 [REALTIME BLOCKS] REJECTED rogue block at height ${block.header.height.toLocaleString()} ` +
            `(our height: ${ourHeight.toLocaleString()}, max reasonable: ${maxReasonable.toLocaleString()}). ` +
            `Likely from a rogue peer.`
          )
          return
        }
      }
      // Update highest accepted height
      if (block.header.height > highestAcceptedHeightRef.current) {
        highestAcceptedHeightRef.current = block.header.height
      }

      // Log block summary in dev mode
      if (import.meta.env.DEV) {
        console.log('📦 [REALTIME BLOCKS] New block received via P2P!')
        logBlockSummary(block)
      }

      // v3.5.10: Verify the block (light client verification)
      let verificationResult: VerificationResult | undefined
      try {
        const startVerify = performance.now()
        verificationResult = await verifyBlock(block)
        const verifyTime = performance.now() - startVerify
        verificationResult.verificationTimeMs = verifyTime

        // Update verification stats
        setVerificationStats(prev => ({
          ...prev,
          blocksVerified: prev.blocksVerified + 1,
          blocksValid: prev.blocksValid + (verificationResult?.valid ? 1 : 0),
          blocksInvalid: prev.blocksInvalid + (verificationResult?.valid ? 0 : 1),
        }))

        // Log verification result
        const emoji = verificationResult.valid ? '✅' : '⚠️'
        console.log(`${emoji} [VERIFICATION] Block #${block.header.height}: ${verificationResult.summary} (${verifyTime.toFixed(1)}ms)`)
        if (!verificationResult.valid) {
          console.warn('[VERIFICATION] Failed checks:', verificationResult.checks.filter(c => !c.passed))
        }

        // v3.5.x: Record verification for telemetry
        recordVerification(verificationResult.valid)

        // v3.5.x: Report verification failures to network
        if (!verificationResult.valid && node) {
          verificationReporter.report(block, verificationResult).catch(err => {
            console.warn('[VERIFICATION] Failed to report verification failure:', err)
          })
        }
      } catch (verifyError) {
        console.warn('[VERIFICATION] Error verifying block:', verifyError)
        // Continue without verification result - don't block on verification failure
      }

      // v3.5.x: Record block for telemetry
      const receiveTime = Date.now()
      recordBlockReceived(block.header.height, block.header.timestamp, receiveTime)

      // Create verified block with attached result
      const verifiedBlock: VerifiedBlock = {
        ...block,
        verification: verificationResult,
      }

      // Update latest block
      setLatestBlock(block)

      // v3.5.8: Update browser peer discovery with current height
      browserPeerDiscovery.setBlockHeight(block.header.height)

      // Create block summary for UI
      const summary = createBlockSummary(block)
      setLatestBlockSummary(summary)

      // Update block history with backpressure limit
      setBlockHistory((prev) => {
        // Check for duplicate blocks
        if (prev.some(b => b.header.height === block.header.height)) {
          console.log(`🔁 [REALTIME BLOCKS] Skipping duplicate block ${block.header.height}`)
          return prev
        }

        const newHistory = [verifiedBlock, ...prev]

        // Trim to MAX_BLOCK_HISTORY to prevent memory exhaustion
        const trimmed = newHistory.slice(0, MAX_BLOCK_HISTORY)

        // Log successful addition
        console.log(`📥 [REALTIME BLOCKS] Block ${block.header.height} added to history. Total: ${trimmed.length} blocks`)

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

      // Track P2P block activity
      lastP2PBlockTime.current = Date.now()
      p2pBlockCount.current++

      console.log(
        `✅ [REALTIME BLOCKS] Block ${block.header.height} ADDED TO STATE via P2P (${block.transactions.length} txs) - P2P blocks: ${p2pBlockCount.current}`
      )

      // v3.5.x: Cache verified block for serving to peers
      cacheBlock(verifiedBlock)

      // v3.5.23: Add to propagation queue for faster P2P sync
      propagationQueue.addBlock(verifiedBlock)

      // v3.5.x: Relay verified blocks to peers
      if (RELAY_CONFIG.ENABLED && node && verificationResult) {
        const blockHash = generateBlockHash(block)

        // Skip if already seen (prevent relay loops)
        if (seenBlockHashes.has(blockHash)) {
          relayStats.duplicatesFiltered++
          console.log(`🔁 [RELAY] Skipping already-seen block ${block.header.height}`)
        }
        // Skip if verification failed and we require it
        else if (RELAY_CONFIG.REQUIRE_VERIFICATION && !verificationResult.valid) {
          relayStats.invalidBlocksFiltered++
          console.log(`⚠️  [RELAY] Skipping unverified block ${block.header.height}`)
        }
        // Skip if rate limited
        else if (isRelayRateLimited()) {
          relayStats.rateLimitDrops++
          console.log(`🚫 [RELAY] Rate limited, skipping block ${block.header.height}`)
        }
        // Relay the block
        else {
          try {
            const pubsub = node.services.pubsub as GossipSub
            const encoded = msgpackEncode(block)
            await pubsub.publish(TOPICS.BLOCKS, encoded)

            // Update relay stats
            seenBlockHashes.add(blockHash)
            relayTimestamps.push(Date.now())
            relayStats.blocksRelayed++
            relayStats.lastRelayTime = Date.now()

            // Clean up seen cache if needed
            cleanupSeenCache()

            console.log(`📡 [RELAY] Relayed block ${block.header.height} (total: ${relayStats.blocksRelayed})`)
          } catch (relayError) {
            console.warn(`[RELAY] Failed to relay block:`, relayError)
          }
        }
      }
    } catch (err) {
      console.error('[REALTIME BLOCKS] Error handling block message:', err)
      setError(err instanceof Error ? err : new Error(String(err)))
    }
  }, [])

  // Store message handler reference for cleanup
  const messageHandlerRef = useRef<((event: any) => void) | null>(null)

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

      // Remove any existing listener first
      if (messageHandlerRef.current) {
        pubsub.removeEventListener('message', messageHandlerRef.current)
        messageHandlerRef.current = null
      }

      // Subscribe to block topic
      pubsub.subscribe(TOPICS.BLOCKS)

      // Debug: Check current subscriptions
      const topics = pubsub.getTopics()
      console.log(`📋 [REALTIME BLOCKS] Currently subscribed topics:`, topics)

      // Debug: Check peers for this topic
      const peers = pubsub.getPeers()
      console.log(`🔗 [REALTIME BLOCKS] Connected gossipsub peers:`, peers.length, peers.map((p: any) => p.toString().substring(0, 12)))

      // Create message handler that handles ALL messages
      const messageHandler = (event: any) => {
        const topic = event.detail?.topic || 'unknown'
        const dataSize = event.detail?.data?.length || 0

        // Only log blocks topic to reduce noise
        if (topic.includes('/blocks')) {
          console.log(`📨 [REALTIME BLOCKS] BLOCKS message received!`)
          console.log(`   Topic: "${topic}"`)
          console.log(`   Expected: "${TOPICS.BLOCKS}"`)
          console.log(`   Match: ${topic === TOPICS.BLOCKS}`)
          console.log(`   Data: ${dataSize} bytes`)
        }

        // Handle blocks topic (also try partial match in case of encoding issues)
        if (topic === TOPICS.BLOCKS || topic.endsWith('/blocks')) {
          console.log(`✅ [REALTIME BLOCKS] Block message MATCHED - processing!`)
          handleBlockMessage(event as CustomEvent)
        }
      }

      // Store reference and add listener
      messageHandlerRef.current = messageHandler
      pubsub.addEventListener('message', messageHandler)

      // Also listen to 'gossipsub:message' event as backup
      pubsub.addEventListener('gossipsub:message', (evt: any) => {
        console.log(`🔔 [REALTIME BLOCKS] gossipsub:message event received!`, evt)
      })

      setIsSubscribed(true)
      console.log('✅ [REALTIME BLOCKS] Subscribed successfully')
      console.log(`   Listening on topic: ${TOPICS.BLOCKS}`)

      // Periodic mesh status check
      const meshCheckInterval = setInterval(() => {
        try {
          const allPeers = pubsub.getPeers()
          const allTopics = pubsub.getTopics()
          console.log(`🔍 [REALTIME BLOCKS] Gossipsub status:`, {
            peers: allPeers.length,
            topics: allTopics,
            subscribed: allTopics.includes(TOPICS.BLOCKS)
          })
        } catch (e) {
          // Ignore errors from mesh check
        }
      }, 10000) // Every 10 seconds

      // Store interval for cleanup
      ;(messageHandlerRef as any).meshCheckInterval = meshCheckInterval

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

      // Clear mesh check interval
      if ((messageHandlerRef as any).meshCheckInterval) {
        clearInterval((messageHandlerRef as any).meshCheckInterval)
      }

      // Remove event listener
      if (messageHandlerRef.current) {
        pubsub.removeEventListener('message', messageHandlerRef.current)
        messageHandlerRef.current = null
      }

      pubsub.unsubscribe(TOPICS.BLOCKS)

      setIsSubscribed(false)
      console.log('✅ [REALTIME BLOCKS] Unsubscribed successfully')
    } catch (err) {
      console.error('[REALTIME BLOCKS] Unsubscribe failed:', err)
      setError(err instanceof Error ? err : new Error(String(err)))
    }
  }, [node, isReady])

  // Track if we've already subscribed to prevent re-subscription loops
  const hasSubscribedRef = useRef(false)
  const lastP2PBlockTime = useRef<number>(0) // Track last P2P block time
  const p2pBlockCount = useRef<number>(0) // Count P2P blocks received

  // v10.2.3: HTTP fallback re-enabled as safety net for gossipsub send queue overflow
  const lastHttpBlockHeight = useRef<number>(0)
  const _fetchBlocksViaHttp = useCallback(async () => {
    console.log('🌐 [HTTP FALLBACK] Fetching blocks from API...')
    try {
      // Fetch recent blocks from API (correct endpoint is /api/v1/blocks/recent)
      const response = await fetch(`${API_BASE}/api/v1/blocks/recent`)
      console.log(`🌐 [HTTP FALLBACK] Response status: ${response.status}`)
      if (!response.ok) {
        console.warn(`🌐 [HTTP FALLBACK] Bad response: ${response.status}`)
        return
      }

      const data = await response.json()
      console.log('🌐 [HTTP FALLBACK] Response data:', { success: data.success, dataLength: data.data?.length })

      // API returns { success: true, data: [...blocks...] }
      const blocks = data.data || data.blocks || data || []

      if (!Array.isArray(blocks) || blocks.length === 0) {
        console.log('[HTTP FALLBACK] No blocks in response:', { hasData: !!data.data, hasBlocks: !!data.blocks })
        return
      }

      console.log(`🌐 [HTTP FALLBACK] Got ${blocks.length} blocks from API, lastHttpBlockHeight=${lastHttpBlockHeight.current}`)

      // Convert API blocks to QBlock format
      const emptyHash = new Uint8Array(32)
      const emptyVdfProof = {
        input: new Uint8Array(32),
        output: new Uint8Array(32),
        proof: new Uint8Array(0),
        iterations: 0,
      }

      const newBlocks: QBlock[] = blocks
        .filter((b: any) => b.height > lastHttpBlockHeight.current)
        .map((apiBlock: any) => ({
          header: {
            height: apiBlock.height || 0,
            phase: apiBlock.phase || 19,
            networkId: apiBlock.network_id || 'mainnet-genesis',
            prevBlockHash: apiBlock.prev_hash ? hexToUint8Array(apiBlock.prev_hash) : emptyHash,
            solutionsRoot: apiBlock.solutions_root ? hexToUint8Array(apiBlock.solutions_root) : emptyHash,
            txRoot: apiBlock.tx_root ? hexToUint8Array(apiBlock.tx_root) : emptyHash,
            stateRoot: apiBlock.state_root ? hexToUint8Array(apiBlock.state_root) : emptyHash,
            timestamp: apiBlock.timestamp || Math.floor(Date.now() / 1000),
            dagRound: apiBlock.dag_round || 0,
            vdfProof: apiBlock.vdf_proof || emptyVdfProof,
            anchorValidator: apiBlock.anchor_validator,
            proposer: apiBlock.proposer || apiBlock.miner || '',
            producerId: apiBlock.producer_id || 0,
            totalDifficulty: BigInt(apiBlock.total_difficulty || 0),
          },
          transactions: Array.isArray(apiBlock.transactions)
            ? apiBlock.transactions.map((tx: any) => ({
                from: tx.from || tx.sender || '',
                to: tx.to || tx.recipient || '',
                amount: tx.amount || tx.value || 0,
                timestamp: tx.timestamp || apiBlock.timestamp || 0,
                signature: tx.signature ? hexToUint8Array(tx.signature) : undefined,
                nonce: tx.nonce,
              }))
            : Array.from({ length: apiBlock.tx_count || 0 }, () => ({
                from: apiBlock.proposer || '',
                to: '',
                amount: 0,
                timestamp: apiBlock.timestamp || 0,
              })),
          miningSolutions: Array.isArray(apiBlock.mining_solutions)
            ? apiBlock.mining_solutions
            : [],
          dagParents: apiBlock.dag_parents || [],
          quantumMetadata: apiBlock.quantum_metadata || { coherence: 0, entanglement: 0, measurement: 0 },
          balanceUpdates: apiBlock.balance_updates || [],
          sizeBytes: apiBlock.size || 0,
        }))
        .reverse() // Oldest first so we add them in order

      console.log(`🌐 [HTTP FALLBACK] Converted ${newBlocks.length} new blocks (filtered from ${blocks.length} total)`)

      if (newBlocks.length > 0) {
        console.log(`📡 [HTTP FALLBACK] Adding ${newBlocks.length} new blocks to state via HTTP`)

        // Update last height
        const maxHeight = Math.max(...newBlocks.map(b => b.header.height))
        if (maxHeight > lastHttpBlockHeight.current) {
          lastHttpBlockHeight.current = maxHeight
        }

        // Add blocks to history WITH verification (same as P2P path)
        for (const block of newBlocks) {
          try {
            // Run light client verification on HTTP-fetched blocks
            let verificationResult: VerificationResult | undefined
            try {
              const startVerify = performance.now()
              verificationResult = await verifyBlock(block)
              const verifyTime = performance.now() - startVerify
              verificationResult.verificationTimeMs = verifyTime

              setVerificationStats(prev => ({
                ...prev,
                blocksVerified: prev.blocksVerified + 1,
                blocksValid: prev.blocksValid + (verificationResult?.valid ? 1 : 0),
                blocksInvalid: prev.blocksInvalid + (verificationResult?.valid ? 0 : 1),
              }))

              const emoji = verificationResult.valid ? '✅' : '⚠️'
              console.log(`${emoji} [HTTP VERIFY] Block #${block.header.height}: ${verificationResult.summary} (${verifyTime.toFixed(1)}ms)`)
              recordVerification(verificationResult.valid)
            } catch (verifyError) {
              console.warn('[HTTP VERIFY] Error verifying block:', verifyError)
            }

            const verifiedBlock: VerifiedBlock = {
              ...block,
              verification: verificationResult,
            }

            setLatestBlock(block)
            setLatestBlockSummary(createBlockSummary(block))
            setBlockHistory(prev => {
              // Avoid duplicates
              if (prev.some(b => b.header.height === block.header.height)) {
                return prev
              }
              const newHistory = [verifiedBlock, ...prev]
              return newHistory.slice(0, MAX_BLOCK_HISTORY)
            })
          } catch (blockErr) {
            console.warn(`[HTTP FALLBACK] Error processing block ${block?.header?.height}:`, blockErr)
          }
        }
      }
    } catch (err) {
      console.warn('[HTTP FALLBACK] Failed to fetch blocks:', err)
    }
  }, [])

  /**
   * Auto-subscribe when node is ready
   */
  useEffect(() => {
    // Only subscribe once when node becomes ready
    if (isReady && !hasSubscribedRef.current) {
      hasSubscribedRef.current = true
      subscribe()
    }

    // Cleanup on unmount only
    return () => {
      if (hasSubscribedRef.current && messageHandlerRef.current) {
        // Don't call unsubscribe() as it may not have node ready
        // Just mark that we need to resubscribe on next mount
        hasSubscribedRef.current = false
      }
    }
  }, [isReady, subscribe])

  // v10.2.3: HTTP fallback RE-ENABLED as safety net
  // Gossipsub send queue overflow can silently drop block messages.
  // Poll every 10s if no P2P blocks received in the last 15s.
  useEffect(() => {
    if (!isReady) return

    const HTTP_POLL_INTERVAL = 10000 // 10 seconds
    const P2P_SILENCE_THRESHOLD = 15000 // 15 seconds without P2P blocks → start HTTP polling
    let lastP2pBlockTime = Date.now()

    // Track when we last got a P2P block
    const p2pTracker = setInterval(() => {
      if (p2pBlockCount.current > 0) {
        // Check if any new P2P blocks arrived recently
        // We use the blockHistory length as proxy
      }
    }, 5000)

    const httpPoller = setInterval(async () => {
      const timeSinceP2p = Date.now() - lastP2pBlockTime
      // Only poll HTTP if P2P is silent and we haven't received any P2P blocks
      // OR if we've never received a P2P block at all
      if (p2pBlockCount.current > 0 && timeSinceP2p < P2P_SILENCE_THRESHOLD) {
        return // P2P is working, skip HTTP
      }

      try {
        await _fetchBlocksViaHttp()
      } catch (e) {
        console.warn('[HTTP FALLBACK] Error:', e)
      }
    }, HTTP_POLL_INTERVAL)

    // Update lastP2pBlockTime when P2P blocks arrive
    const originalCount = p2pBlockCount.current
    const p2pMonitor = setInterval(() => {
      if (p2pBlockCount.current > originalCount) {
        lastP2pBlockTime = Date.now()
      }
    }, 2000)

    return () => {
      clearInterval(p2pTracker)
      clearInterval(httpPoller)
      clearInterval(p2pMonitor)
    }
  }, [isReady, _fetchBlocksViaHttp])

  return {
    latestBlock,
    latestBlockSummary,
    blockHistory,
    isSubscribed,
    error,
    verificationStats, // v3.5.10: Light client verification stats
    relayStats: getRelayStats(), // v3.5.x: Block relay stats
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
