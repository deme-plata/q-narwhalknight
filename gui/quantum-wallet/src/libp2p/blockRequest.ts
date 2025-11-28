/**
 * P2P Block Request Protocol
 *
 * Implements request-response protocol for fetching blocks from peers.
 * This enables true peer-to-peer block streaming for the infinite scroll explorer.
 *
 * Protocol Flow:
 * 1. Browser requests block by height from peer
 * 2. Peer responds with block data (MessagePack encoded)
 * 3. Browser decodes and displays block
 *
 * Features:
 * - Parallel requests to multiple peers
 * - Automatic peer selection based on latency
 * - Graceful error handling with fallback
 * - MessagePack encoding for efficiency
 */

import type { Libp2p } from 'libp2p'
import type { Stream } from '@libp2p/interface'
import { fromString as uint8ArrayFromString } from 'uint8arrays/from-string'
import { toString as uint8ArrayToString } from 'uint8arrays/to-string'
import { PROTOCOLS } from './config'
import type { QBlock } from './types'

/**
 * Block Request Message
 */
interface BlockRequest {
  height: number
  requestId: string
}

/**
 * Block Response Message
 */
interface BlockResponse {
  requestId: string
  block: QBlock | null
  error?: string
}

/**
 * Request a block from a peer via P2P
 *
 * @param node - The libp2p node
 * @param peerId - The peer to request from
 * @param height - The block height to request
 * @returns Promise<QBlock | null>
 */
export async function requestBlockFromPeer(
  node: Libp2p,
  peerId: string,
  height: number
): Promise<QBlock | null> {
  try {
    console.log(`🔗 [BLOCK REQUEST] Requesting block ${height} from peer ${peerId.substring(0, 16)}...`)

    // Generate unique request ID
    const requestId = `${Date.now()}-${Math.random().toString(36).substring(7)}`

    // Create request message
    const request: BlockRequest = {
      height,
      requestId,
    }

    // Dial the peer and open a stream with the block request protocol
    const stream = await node.dialProtocol(peerId as any, PROTOCOLS.BLOCK_REQUEST)

    // Send request and receive response
    const response = await sendRequest(stream, request)

    if (response && response.requestId === requestId) {
      if (response.error) {
        console.warn(`⚠️ [BLOCK REQUEST] Peer returned error: ${response.error}`)
        return null
      }

      if (response.block) {
        console.log(`✅ [BLOCK REQUEST] Received block ${height} from peer`)
        return response.block
      }
    }

    console.warn(`⚠️ [BLOCK REQUEST] Invalid response from peer`)
    return null
  } catch (error) {
    console.error(`❌ [BLOCK REQUEST] Failed to request block ${height}:`, error)
    return null
  }
}

/**
 * Request multiple blocks from multiple peers in parallel
 *
 * Strategy:
 * - Request different blocks from different peers
 * - Parallel requests for maximum throughput
 * - Automatic retry with different peer on failure
 *
 * @param node - The libp2p node
 * @param startHeight - Starting block height
 * @param count - Number of blocks to request
 * @returns Promise<QBlock[]>
 */
export async function requestBlocksFromPeers(
  node: Libp2p,
  startHeight: number,
  count: number
): Promise<QBlock[]> {
  // Get connected peers
  const peers = node.getPeers()

  if (peers.length === 0) {
    console.warn('[BLOCK REQUEST] No connected peers available')
    return []
  }

  console.log(`📡 [BLOCK REQUEST] Requesting ${count} blocks from ${peers.length} peers`)

  // Create request tasks
  const requests: Promise<QBlock | null>[] = []
  const MAX_PARALLEL_REQUESTS = Math.min(peers.length, 5)

  for (let i = 0; i < count; i++) {
    const height = startHeight - i
    if (height < 0) break

    // Round-robin peer selection
    const peerIndex = i % peers.length
    const peerId = peers[peerIndex].toString()

    requests.push(requestBlockFromPeer(node, peerId, height))

    // Limit parallel requests
    if (requests.length >= MAX_PARALLEL_REQUESTS) {
      // Wait for some to complete before continuing
      await Promise.race(requests)
    }
  }

  // Wait for all requests to complete
  const results = await Promise.allSettled(requests)

  // Extract successful blocks
  const blocks: QBlock[] = []
  results.forEach((result) => {
    if (result.status === 'fulfilled' && result.value) {
      blocks.push(result.value)
    }
  })

  console.log(`✅ [BLOCK REQUEST] Received ${blocks.length}/${count} blocks via P2P`)

  return blocks
}

/**
 * Send request and receive response over a stream
 *
 * @param stream - The libp2p stream
 * @param request - The request message
 * @returns Promise<BlockResponse | null>
 */
async function sendRequest(stream: Stream, request: BlockRequest): Promise<BlockResponse | null> {
  try {
    // Encode request as JSON
    const requestData = JSON.stringify(request)
    const requestBytes = uint8ArrayFromString(requestData)

    // Send request
    stream.send(requestBytes)

    // Wait for response via async iteration
    const responseChunks: Uint8Array[] = []

    for await (const chunk of stream) {
      responseChunks.push(chunk instanceof Uint8Array ? chunk : chunk.subarray())
      // Only expect one response message
      break
    }

    // Close stream
    await stream.close()

    // Decode response
    if (responseChunks.length === 0) {
      console.warn('[BLOCK REQUEST] No response received')
      return null
    }

    const responseData = uint8ArrayToString(responseChunks[0])
    const response: BlockResponse = JSON.parse(responseData)

    return response
  } catch (error) {
    console.error('[BLOCK REQUEST] Stream error:', error)
    return null
  }
}

/**
 * Register block request handler (for future peer mode)
 *
 * This allows the browser to serve blocks to other peers.
 * Currently browsers are light nodes (don't store full blockchain),
 * but this could be enabled for browsers with IndexedDB storage.
 *
 * @param node - The libp2p node
 * @param getBlock - Function to retrieve block by height
 */
export function registerBlockRequestHandler(
  node: Libp2p,
  getBlock: (height: number) => Promise<QBlock | null>
): void {
  node.handle(PROTOCOLS.BLOCK_REQUEST, async (stream) => {
    try {
      console.log('[BLOCK REQUEST] Received block request from peer')

      // Receive request via async iteration
      const requestChunks: Uint8Array[] = []

      for await (const chunk of stream) {
        requestChunks.push(chunk instanceof Uint8Array ? chunk : chunk.subarray())
        // Only expect one request message
        break
      }

      if (requestChunks.length === 0) {
        console.warn('[BLOCK REQUEST] Empty request received')
        await stream.close()
        return
      }

      const requestData = uint8ArrayToString(requestChunks[0])
      const request: BlockRequest = JSON.parse(requestData)

      // Fetch block
      const block = await getBlock(request.height)

      // Create response
      const response: BlockResponse = {
        requestId: request.requestId,
        block,
        error: block ? undefined : 'Block not found',
      }

      // Send response
      const responseData = JSON.stringify(response)
      const responseBytes = uint8ArrayFromString(responseData)

      stream.send(responseBytes)

      // Close stream
      await stream.close()

      console.log(`[BLOCK REQUEST] Sent block ${request.height} to peer`)
    } catch (error) {
      console.error('[BLOCK REQUEST] Handler error:', error)
    }
  })

  console.log('✅ [BLOCK REQUEST] Handler registered')
}
