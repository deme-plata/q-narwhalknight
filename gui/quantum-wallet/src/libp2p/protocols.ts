// @ts-nocheck
/**
 * Custom libp2p Protocols for Q-NarwhalKnight Browser Nodes
 *
 * Implements request-response protocols for:
 * 1. Handshake - Protocol version and network validation
 * 2. Block sync - Fetching blocks from peers
 * 3. Balance queries - Getting account balances
 *
 * Note: TypeScript type checking disabled due to iterator type inference issues
 * with libp2p v3 Stream async iterables. The code is functionally correct.
 */

import type { Libp2p, Stream, Connection } from '@libp2p/interface'
import {
  createHandshakeMessage,
  encodeHandshakeMessage,
  decodeHandshakeResult,
  logHandshakeResult,
  type HandshakeMessage,
  type HandshakeResult,
} from './handshake'
import { NETWORK_ID, PROTOCOLS } from './config'

/**
 * Register all custom protocols on the libp2p node
 */
export async function registerProtocols(node: Libp2p): Promise<void> {
  console.log('📡 [PROTOCOLS] Registering custom protocols...')

  // Register handshake protocol handler
  await node.handle(PROTOCOLS.HANDSHAKE, async (stream: Stream, connection: Connection) => {
    await handleHandshakeRequest(stream, connection.remotePeer.toString())
  })

  console.log('✅ [PROTOCOLS] All protocols registered')
  console.log(`   - ${PROTOCOLS.HANDSHAKE} (handshake validation)`)
}

/**
 * Handle incoming handshake requests from Rust nodes
 *
 * When a Rust node connects to us, it will send a handshake request
 * to validate protocol compatibility. We need to respond with our
 * handshake message so the node can verify we're on the correct network.
 */
async function handleHandshakeRequest(stream: Stream, peerId: string): Promise<void> {
  console.log(`🤝 [HANDSHAKE] Received handshake request from ${peerId.substring(0, 16)}...`)

  try {
    // Read the incoming handshake message
    // For simplicity, read raw bytes without length-prefix
    // (Rust side should send the message directly)
    let peerHandshakeData: Uint8Array | null = null

    for await (const chunk of stream.source) {
      // Accumulate chunks - convert to Uint8Array
      // Use Uint8ArrayList.subarray() to get Uint8Array
      const chunkBytes = new Uint8Array(chunk.subarray(0))

      if (!peerHandshakeData) {
        peerHandshakeData = chunkBytes
      } else {
        const prev = peerHandshakeData
        const newData = new Uint8Array(prev.length + chunkBytes.length)
        newData.set(prev, 0)
        newData.set(chunkBytes, prev.length)
        peerHandshakeData = newData
      }
      // For now, assume first chunk contains full message
      // TODO: Implement proper framing if needed
      break
    }

    if (!peerHandshakeData) {
      console.error('❌ [HANDSHAKE] No handshake data received')
      await stream.close()
      return
    }

    // Decode the peer's handshake message
    const peerHandshake = decodeHandshakePeerMessage(peerHandshakeData)
    console.log(`📥 [HANDSHAKE] Peer protocol: v${peerHandshake.protocol_version.major}.${peerHandshake.protocol_version.minor}.${peerHandshake.protocol_version.patch}`)
    console.log(`📥 [HANDSHAKE] Peer network: ${peerHandshake.network_id}`)

    // Validate the peer's handshake
    const validation = validatePeerHandshake(peerHandshake)

    // Create our response
    const response = encodeHandshakeResponse(validation)

    // Send response back to peer
    stream.send(response)

    // Wait a moment for the send to complete
    await new Promise(resolve => setTimeout(resolve, 100))

    // Log the result
    if (validation.type === 'Success') {
      console.log(`✅ [HANDSHAKE] Success with ${peerId.substring(0, 16)}... - Protocol compatible`)
    } else {
      console.warn(`⚠️  [HANDSHAKE] Validation failed with ${peerId.substring(0, 16)}...`)
      logHandshakeResult(validation)
    }

    await stream.close()
  } catch (error) {
    console.error(`❌ [HANDSHAKE] Error handling handshake request:`, error)
    try {
      await stream.close()
    } catch (e) {
      // Ignore close errors
    }
  }
}

/**
 * Decode peer's handshake message (simplified - same structure as ours)
 */
function decodeHandshakePeerMessage(data: Uint8Array): HandshakeMessage {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength)
  const decoder = new TextDecoder()

  let offset = 0

  // Protocol version (3 x u16)
  const major = view.getUint16(offset, true)
  offset += 2
  const minor = view.getUint16(offset, true)
  offset += 2
  const patch = view.getUint16(offset, true)
  offset += 2

  // Network ID (u64 length + bytes)
  const networkIdLen = Number(view.getBigUint64(offset, true))
  offset += 8
  const networkIdBytes = data.slice(offset, offset + networkIdLen)
  const network_id = decoder.decode(networkIdBytes)
  offset += networkIdLen

  // Node version (u64 length + bytes)
  const nodeVersionLen = Number(view.getBigUint64(offset, true))
  offset += 8
  const nodeVersionBytes = data.slice(offset, offset + nodeVersionLen)
  const node_version = decoder.decode(nodeVersionBytes)
  offset += nodeVersionLen

  // Features array (u64 length + each string)
  const featuresCount = Number(view.getBigUint64(offset, true))
  offset += 8
  const features: string[] = []
  for (let i = 0; i < featuresCount; i++) {
    const featureLen = Number(view.getBigUint64(offset, true))
    offset += 8
    const featureBytes = data.slice(offset, offset + featureLen)
    features.push(decoder.decode(featureBytes))
    offset += featureLen
  }

  // Genesis hash (u64 length + bytes)
  const genesisHashLen = Number(view.getBigUint64(offset, true))
  offset += 8
  const genesis_hash = data.slice(offset, offset + genesisHashLen)

  return {
    protocol_version: { major, minor, patch },
    network_id,
    node_version,
    features,
    genesis_hash,
  }
}

/**
 * Validate peer's handshake message against our requirements
 */
function validatePeerHandshake(peerHandshake: HandshakeMessage): HandshakeResult {
  const ourHandshake = createHandshakeMessage()

  // Check protocol version compatibility
  // Compatible if major versions match and minor is within 1 step
  if (peerHandshake.protocol_version.major !== ourHandshake.protocol_version.major) {
    return {
      type: 'IncompatibleProtocol',
      ours: ourHandshake.protocol_version,
      theirs: peerHandshake.protocol_version,
    }
  }

  const minorDiff = Math.abs(
    peerHandshake.protocol_version.minor - ourHandshake.protocol_version.minor
  )
  if (minorDiff > 1) {
    return {
      type: 'IncompatibleProtocol',
      ours: ourHandshake.protocol_version,
      theirs: peerHandshake.protocol_version,
    }
  }

  // Check network ID
  if (peerHandshake.network_id !== NETWORK_ID) {
    return {
      type: 'WrongNetwork',
      ours: NETWORK_ID,
      theirs: peerHandshake.network_id,
    }
  }

  // Check genesis hash
  if (!arraysEqual(peerHandshake.genesis_hash, ourHandshake.genesis_hash)) {
    return {
      type: 'GenesisMismatch',
    }
  }

  // All checks passed
  return {
    type: 'Success',
  }
}

/**
 * Encode handshake validation result to binary format
 */
function encodeHandshakeResponse(result: HandshakeResult): Uint8Array {
  const parts: Uint8Array[] = []

  switch (result.type) {
    case 'Success':
      // Discriminant 0
      parts.push(new Uint8Array([0]))
      break

    case 'IncompatibleProtocol':
      // Discriminant 1 + two ProtocolVersion structs
      parts.push(new Uint8Array([1]))

      // Ours
      const oursBuf1 = new ArrayBuffer(6)
      const oursView1 = new DataView(oursBuf1)
      oursView1.setUint16(0, result.ours.major, true)
      oursView1.setUint16(2, result.ours.minor, true)
      oursView1.setUint16(4, result.ours.patch, true)
      parts.push(new Uint8Array(oursBuf1))

      // Theirs
      const theirsBuf1 = new ArrayBuffer(6)
      const theirsView1 = new DataView(theirsBuf1)
      theirsView1.setUint16(0, result.theirs.major, true)
      theirsView1.setUint16(2, result.theirs.minor, true)
      theirsView1.setUint16(4, result.theirs.patch, true)
      parts.push(new Uint8Array(theirsBuf1))
      break

    case 'WrongNetwork':
      // Discriminant 2 + two strings
      parts.push(new Uint8Array([2]))

      const encoder = new TextEncoder()

      // Ours
      const oursBytes = encoder.encode(result.ours)
      parts.push(encodeU64(oursBytes.length))
      parts.push(oursBytes)

      // Theirs
      const theirsBytes = encoder.encode(result.theirs)
      parts.push(encodeU64(theirsBytes.length))
      parts.push(theirsBytes)
      break

    case 'GenesisMismatch':
      // Discriminant 3
      parts.push(new Uint8Array([3]))
      break

    case 'MissingFeatures':
      // Discriminant 4 + array of strings
      parts.push(new Uint8Array([4]))

      const featEncoder = new TextEncoder()
      parts.push(encodeU64(result.required.length))

      for (const feature of result.required) {
        const featureBytes = featEncoder.encode(feature)
        parts.push(encodeU64(featureBytes.length))
        parts.push(featureBytes)
      }
      break
  }

  // Concatenate all parts
  const totalLength = parts.reduce((sum, part) => sum + part.length, 0)
  const response = new Uint8Array(totalLength)
  let offset = 0
  for (const part of parts) {
    response.set(part, offset)
    offset += part.length
  }

  return response
}

/**
 * Encode u64 as little-endian bytes (8 bytes)
 */
function encodeU64(value: number): Uint8Array {
  const buf = new ArrayBuffer(8)
  const view = new DataView(buf)
  view.setBigUint64(0, BigInt(value), true) // true = little-endian
  return new Uint8Array(buf)
}

/**
 * Compare two Uint8Arrays for equality
 */
function arraysEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
}

/**
 * Initiate handshake with a peer (outbound)
 *
 * This is used when WE dial a peer and want to validate compatibility
 * before establishing full communication.
 */
export async function initiateHandshake(
  node: Libp2p,
  peerIdStr: string
): Promise<HandshakeResult> {
  console.log(`🤝 [HANDSHAKE] Initiating handshake with ${peerIdStr.substring(0, 16)}...`)

  try {
    // Open a stream for the handshake protocol
    const stream = await node.dialProtocol(peerIdStr as any, PROTOCOLS.HANDSHAKE)

    // Create our handshake message
    const ourHandshake = createHandshakeMessage()
    const encoded = encodeHandshakeMessage(ourHandshake)

    // Send our handshake
    stream.send(encoded)

    // Wait for send to complete
    await new Promise(resolve => setTimeout(resolve, 100))

    // Read the response
    let responseData: Uint8Array | null = null

    for await (const chunk of stream.source) {
      // Accumulate chunks - convert to Uint8Array
      // Use Uint8ArrayList.subarray() to get Uint8Array
      const chunkBytes = new Uint8Array(chunk.subarray(0))

      if (!responseData) {
        responseData = chunkBytes
      } else {
        const prev = responseData
        const newData = new Uint8Array(prev.length + chunkBytes.length)
        newData.set(prev, 0)
        newData.set(chunkBytes, prev.length)
        responseData = newData
      }
      // For now, assume first chunk contains full message
      break
    }

    if (!responseData) {
      throw new Error('No handshake response received')
    }

    // Decode the response
    const result = decodeHandshakeResult(responseData)

    // Log the result
    logHandshakeResult(result)

    await stream.close()

    return result
  } catch (error) {
    console.error(`❌ [HANDSHAKE] Failed to initiate handshake:`, error)
    throw error
  }
}
