/**
 * Browser Handshake Protocol for Q-NarwhalKnight
 *
 * Implements the /qnk/handshake/1.0.0 protocol to validate compatibility
 * with Rust nodes before establishing full P2P connection.
 */

import { NETWORK_ID, PROTOCOL_VERSION } from './config'

/**
 * Protocol version structure matching Rust ProtocolVersion
 */
export interface ProtocolVersion {
  major: number
  minor: number
  patch: number
}

/**
 * Handshake message structure matching Rust HandshakeMessage
 */
export interface HandshakeMessage {
  protocol_version: ProtocolVersion
  network_id: string
  node_version: string
  features: string[]
  genesis_hash: Uint8Array
}

/**
 * Handshake result matching Rust HandshakeResult
 */
export type HandshakeResult =
  | { type: 'Success' }
  | {
      type: 'IncompatibleProtocol'
      ours: ProtocolVersion
      theirs: ProtocolVersion
    }
  | {
      type: 'WrongNetwork'
      ours: string
      theirs: string
    }
  | { type: 'GenesisMismatch' }
  | {
      type: 'MissingFeatures'
      required: string[]
    }

/**
 * Parse protocol version from string like "1.0.20"
 */
function parseProtocolVersion(version: string): ProtocolVersion {
  const parts = version.split('.')
  if (parts.length !== 3) {
    throw new Error(`Invalid protocol version format: ${version}`)
  }

  return {
    major: parseInt(parts[0], 10),
    minor: parseInt(parts[1], 10),
    patch: parseInt(parts[2], 10),
  }
}

/**
 * Genesis block hash for Q-NarwhalKnight
 * This must match the Rust node's genesis hash exactly
 *
 * TODO: Fetch this from the API or hardcode the actual genesis hash
 * For now, using a placeholder that should be replaced
 */
const GENESIS_HASH = new Uint8Array([
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
])

/**
 * Create browser handshake message
 *
 * This message will be sent to Rust nodes during connection establishment
 * to verify protocol compatibility before full P2P communication.
 */
export function createHandshakeMessage(): HandshakeMessage {
  return {
    protocol_version: parseProtocolVersion(PROTOCOL_VERSION),
    network_id: NETWORK_ID,
    node_version: `browser-${PROTOCOL_VERSION}`,
    features: [
      'browser-node',
      'websocket-only',
      'gossipsub',
      'block-sync',
      'real-time-updates',
    ],
    genesis_hash: GENESIS_HASH,
  }
}

/**
 * Encode handshake message to binary format (bincode-compatible)
 *
 * IMPORTANT: Must match Rust bincode serialization format exactly:
 * - Protocol version: 3 x u16 (6 bytes)
 * - Network ID: length prefix (u64) + UTF-8 bytes
 * - Node version: length prefix (u64) + UTF-8 bytes
 * - Features: array length (u64) + each string (length prefix + UTF-8)
 * - Genesis hash: length prefix (u64) + bytes
 */
export function encodeHandshakeMessage(msg: HandshakeMessage): Uint8Array {
  const encoder = new TextEncoder()
  const parts: Uint8Array[] = []

  // Protocol version (3 x u16 = 6 bytes, little-endian)
  const versionBuf = new ArrayBuffer(6)
  const versionView = new DataView(versionBuf)
  versionView.setUint16(0, msg.protocol_version.major, true)
  versionView.setUint16(2, msg.protocol_version.minor, true)
  versionView.setUint16(4, msg.protocol_version.patch, true)
  parts.push(new Uint8Array(versionBuf))

  // Network ID (u64 length + bytes)
  const networkIdBytes = encoder.encode(msg.network_id)
  parts.push(encodeU64(networkIdBytes.length))
  parts.push(networkIdBytes)

  // Node version (u64 length + bytes)
  const nodeVersionBytes = encoder.encode(msg.node_version)
  parts.push(encodeU64(nodeVersionBytes.length))
  parts.push(nodeVersionBytes)

  // Features array (u64 length + each string)
  parts.push(encodeU64(msg.features.length))
  for (const feature of msg.features) {
    const featureBytes = encoder.encode(feature)
    parts.push(encodeU64(featureBytes.length))
    parts.push(featureBytes)
  }

  // Genesis hash (u64 length + bytes)
  parts.push(encodeU64(msg.genesis_hash.length))
  parts.push(msg.genesis_hash)

  // Concatenate all parts
  const totalLength = parts.reduce((sum, part) => sum + part.length, 0)
  const result = new Uint8Array(totalLength)
  let offset = 0
  for (const part of parts) {
    result.set(part, offset)
    offset += part.length
  }

  return result
}

/**
 * Encode u64 as little-endian bytes (8 bytes)
 */
function encodeU64(value: number): Uint8Array {
  const buf = new ArrayBuffer(8)
  const view = new DataView(buf)
  // JavaScript can only safely represent integers up to 2^53 - 1
  // For lengths, this is more than sufficient
  view.setBigUint64(0, BigInt(value), true) // true = little-endian
  return new Uint8Array(buf)
}

/**
 * Decode handshake result from binary format
 *
 * IMPORTANT: Must match Rust bincode deserialization format
 */
export function decodeHandshakeResult(data: Uint8Array): HandshakeResult {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength)

  // First byte is enum discriminant
  const discriminant = view.getUint8(0)

  switch (discriminant) {
    case 0: // Success
      return { type: 'Success' }

    case 1: // IncompatibleProtocol
      // Read two ProtocolVersion structs (3 x u16 each = 12 bytes total)
      const oursMajor = view.getUint16(1, true)
      const oursMinor = view.getUint16(3, true)
      const oursPatch = view.getUint16(5, true)
      const theirsMajor = view.getUint16(7, true)
      const theirsMinor = view.getUint16(9, true)
      const theirsPatch = view.getUint16(11, true)
      return {
        type: 'IncompatibleProtocol',
        ours: { major: oursMajor, minor: oursMinor, patch: oursPatch },
        theirs: { major: theirsMajor, minor: theirsMinor, patch: theirsPatch },
      }

    case 2: // WrongNetwork
      // Read two strings (length prefix + bytes for each)
      const decoder = new TextDecoder()
      let offset = 1
      const oursLen = Number(view.getBigUint64(offset, true))
      offset += 8
      const oursBytes = data.slice(offset, offset + oursLen)
      const ours = decoder.decode(oursBytes)
      offset += oursLen
      const theirsLen = Number(view.getBigUint64(offset, true))
      offset += 8
      const theirsBytes = data.slice(offset, offset + theirsLen)
      const theirs = decoder.decode(theirsBytes)
      return {
        type: 'WrongNetwork',
        ours,
        theirs,
      }

    case 3: // GenesisMismatch
      return { type: 'GenesisMismatch' }

    case 4: // MissingFeatures
      // Read array of strings
      let featureOffset = 1
      const requiredCount = Number(view.getBigUint64(featureOffset, true))
      featureOffset += 8
      const required: string[] = []
      const featureDecoder = new TextDecoder()
      for (let i = 0; i < requiredCount; i++) {
        const len = Number(view.getBigUint64(featureOffset, true))
        featureOffset += 8
        const bytes = data.slice(featureOffset, featureOffset + len)
        required.push(featureDecoder.decode(bytes))
        featureOffset += len
      }
      return {
        type: 'MissingFeatures',
        required,
      }

    default:
      throw new Error(`Unknown handshake result discriminant: ${discriminant}`)
  }
}

/**
 * Log handshake result for debugging
 */
export function logHandshakeResult(result: HandshakeResult): void {
  switch (result.type) {
    case 'Success':
      console.log('✅ [HANDSHAKE] Success - Protocol compatible')
      break

    case 'IncompatibleProtocol':
      console.error(
        `❌ [HANDSHAKE] Incompatible protocol: ours=v${result.ours.major}.${result.ours.minor}.${result.ours.patch}, theirs=v${result.theirs.major}.${result.theirs.minor}.${result.theirs.patch}`
      )
      break

    case 'WrongNetwork':
      console.error(
        `❌ [HANDSHAKE] Wrong network: ours=${result.ours}, theirs=${result.theirs}`
      )
      break

    case 'GenesisMismatch':
      console.error('❌ [HANDSHAKE] Genesis hash mismatch')
      break

    case 'MissingFeatures':
      console.error(
        `❌ [HANDSHAKE] Missing required features: ${result.required.join(', ')}`
      )
      break
  }
}
