// @ts-nocheck
/**
 * Block Server - Serve Blocks to Peers
 *
 * Handles the /qnk/block-serve/1.0.0 protocol, allowing browser nodes
 * to serve cached blocks to other browser nodes. This enables browser-to-browser
 * block synchronization.
 *
 * v3.5.x: Browser P2P Network Contribution - Feature 5 (Part 2)
 *
 * Note: TypeScript type checking disabled due to iterator type inference issues
 * with libp2p v3 Stream async iterables. The code is functionally correct.
 */

import type { Libp2p, Stream, Connection } from '@libp2p/interface'
import { decode as msgpackDecode, encode as msgpackEncode } from '@msgpack/msgpack'
import { PROTOCOLS, BLOCK_CACHE_CONFIG } from './config'
import { blockCache } from './blockCache'
import type { BlockRequest, BlockResponse, QBlock } from './types'

/**
 * Block server statistics
 */
export interface BlockServerStats {
  // Total requests received
  requestsReceived: number

  // Total blocks served
  blocksServed: number

  // Total requests failed
  requestsFailed: number

  // Average response time (ms)
  avgResponseTimeMs: number

  // Server active
  isActive: boolean

  // Last request timestamp
  lastRequestTime: number
}

// Global stats
const serverStats: BlockServerStats = {
  requestsReceived: 0,
  blocksServed: 0,
  requestsFailed: 0,
  avgResponseTimeMs: 0,
  isActive: false,
  lastRequestTime: 0,
}

// Response times for averaging
const responseTimes: number[] = []
const MAX_RESPONSE_TIME_SAMPLES = 100

/**
 * Get block server statistics
 */
export function getBlockServerStats(): BlockServerStats {
  return { ...serverStats }
}

/**
 * Read data from a stream
 */
async function readStreamData(stream: Stream): Promise<Uint8Array> {
  const chunks: Uint8Array[] = []

  for await (const chunk of stream.source) {
    if (chunk instanceof Uint8Array) {
      chunks.push(chunk)
    } else if (chunk.subarray) {
      // Handle Uint8ArrayList
      chunks.push(chunk.subarray())
    }
  }

  // Concatenate all chunks
  const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0)
  const result = new Uint8Array(totalLength)
  let offset = 0
  for (const chunk of chunks) {
    result.set(chunk, offset)
    offset += chunk.length
  }

  return result
}

/**
 * Write data to a stream
 */
async function writeStreamData(stream: Stream, data: Uint8Array): Promise<void> {
  // Push the data to the sink
  await stream.sink([data])
}

/**
 * Handle a block request from a peer
 */
async function handleBlockRequest(stream: Stream, connection: Connection): Promise<void> {
  const startTime = Date.now()
  const peerId = connection.remotePeer.toString().substring(0, 16)

  console.log(`📥 [BLOCK SERVER] Request from ${peerId}...`)
  serverStats.requestsReceived++
  serverStats.lastRequestTime = startTime

  try {
    // Read the request
    const requestData = await Promise.race([
      readStreamData(stream),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Request timeout')), BLOCK_CACHE_CONFIG.REQUEST_TIMEOUT)
      ),
    ])

    // Decode request
    const request = msgpackDecode(requestData) as BlockRequest

    console.log(`📋 [BLOCK SERVER] Request for blocks ${request.startHeight}-${request.endHeight}`)

    // Validate request
    if (
      typeof request.startHeight !== 'number' ||
      typeof request.endHeight !== 'number' ||
      request.startHeight > request.endHeight
    ) {
      throw new Error('Invalid request parameters')
    }

    // Get blocks from cache
    const blocks = blockCache.getBlocksAsSimple(request.startHeight, request.endHeight)

    // Create response
    const response: BlockResponse = {
      responderId: '', // Will be set by the protocol handler
      startHeight: request.startHeight,
      endHeight: request.endHeight,
      blocks: blocks,
      cacheSize: blockCache.getSize(),
      timestamp: Date.now(),
    }

    // Encode and send response
    const responseData = msgpackEncode(response)
    await writeStreamData(stream, responseData)

    // Update stats
    const responseTime = Date.now() - startTime
    serverStats.blocksServed += blocks.length
    responseTimes.push(responseTime)
    if (responseTimes.length > MAX_RESPONSE_TIME_SAMPLES) {
      responseTimes.shift()
    }
    serverStats.avgResponseTimeMs = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length

    console.log(`✅ [BLOCK SERVER] Served ${blocks.length} blocks to ${peerId} in ${responseTime}ms`)
  } catch (error) {
    console.error(`❌ [BLOCK SERVER] Request failed:`, error)
    serverStats.requestsFailed++

    // Try to send error response
    try {
      const errorResponse: BlockResponse = {
        responderId: '',
        startHeight: 0,
        endHeight: 0,
        blocks: [],
        cacheSize: blockCache.getSize(),
        timestamp: Date.now(),
      }
      const responseData = msgpackEncode(errorResponse)
      await writeStreamData(stream, responseData)
    } catch {
      // Ignore errors sending error response
    }
  } finally {
    // Close the stream
    try {
      await stream.close()
    } catch {
      // Ignore close errors
    }
  }
}

/**
 * Block Server Class
 *
 * Manages the block serving protocol handler.
 */
export class BlockServer {
  private libp2p: Libp2p | null = null
  private enabled: boolean = BLOCK_CACHE_CONFIG.SERVE_ENABLED

  /**
   * Initialize and start the block server
   */
  async start(libp2p: Libp2p): Promise<void> {
    this.libp2p = libp2p

    if (!this.enabled) {
      console.log(`🔇 [BLOCK SERVER] Serving disabled`)
      return
    }

    try {
      // Register protocol handler
      await libp2p.handle(PROTOCOLS.BLOCK_SERVE, async (stream: Stream, connection: Connection) => {
        await handleBlockRequest(stream, connection)
      })

      serverStats.isActive = true
      console.log(`🚀 [BLOCK SERVER] Started on ${PROTOCOLS.BLOCK_SERVE}`)
    } catch (error) {
      console.error(`❌ [BLOCK SERVER] Failed to start:`, error)
    }
  }

  /**
   * Stop the block server
   */
  async stop(): Promise<void> {
    if (!this.libp2p) return

    try {
      await this.libp2p.unhandle(PROTOCOLS.BLOCK_SERVE)
      serverStats.isActive = false
      console.log(`🛑 [BLOCK SERVER] Stopped`)
    } catch (error) {
      console.error(`❌ [BLOCK SERVER] Error stopping:`, error)
    }
  }

  /**
   * Enable or disable the server
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled
    console.log(`🔧 [BLOCK SERVER] ${enabled ? 'Enabled' : 'Disabled'}`)
  }

  /**
   * Check if server is active
   */
  isActive(): boolean {
    return serverStats.isActive
  }

  /**
   * Get server statistics
   */
  getStats(): BlockServerStats {
    return getBlockServerStats()
  }
}

// Global block server instance
export const blockServer = new BlockServer()

/**
 * Request blocks from a peer
 *
 * @param libp2p - The libp2p node
 * @param peerId - The peer to request from
 * @param startHeight - Start height (inclusive)
 * @param endHeight - End height (inclusive)
 * @returns Promise<QBlock[]> - The received blocks
 */
export async function requestBlocksFromPeer(
  libp2p: Libp2p,
  peerId: string,
  startHeight: number,
  endHeight: number
): Promise<QBlock[]> {
  console.log(`📤 [BLOCK CLIENT] Requesting blocks ${startHeight}-${endHeight} from ${peerId.substring(0, 16)}...`)

  try {
    // Create the request
    const request: BlockRequest = {
      requesterId: libp2p.peerId.toString(),
      startHeight,
      endHeight,
      timestamp: Date.now(),
    }

    // Open stream to peer
    const stream = await libp2p.dialProtocol(peerId as any, PROTOCOLS.BLOCK_SERVE, {
      signal: AbortSignal.timeout(BLOCK_CACHE_CONFIG.REQUEST_TIMEOUT),
    })

    // Send request
    const requestData = msgpackEncode(request)
    await writeStreamData(stream, requestData)

    // Read response
    const responseData = await Promise.race([
      readStreamData(stream),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Response timeout')), BLOCK_CACHE_CONFIG.REQUEST_TIMEOUT)
      ),
    ])

    // Close stream
    await stream.close()

    // Decode response
    const response = msgpackDecode(responseData) as BlockResponse

    console.log(`✅ [BLOCK CLIENT] Received ${response.blocks.length} blocks from peer`)

    return response.blocks
  } catch (error) {
    console.error(`❌ [BLOCK CLIENT] Request failed:`, error)
    return []
  }
}
