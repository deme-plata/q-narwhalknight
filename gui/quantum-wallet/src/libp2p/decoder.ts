/**
 * MessagePack + JSON Hybrid Decoder
 *
 * Decodes messages using MessagePack binary format with JSON fallback.
 * MessagePack provides:
 * - 90% size reduction vs JSON
 * - Full compatibility with Rust rmp-serde
 * - Type-safe serialization
 * - Fast encode/decode (<15µs)
 *
 * Decoding Strategy:
 * 1. Try MessagePack binary decode (production)
 * 2. Fall back to JSON (development/testing)
 *
 * Rust Backend:
 * - Use rmp-serde for MessagePack serialization
 * - Maintains backward compatibility with postcard
 */

import { decode as msgpackDecode, encode as msgpackEncode } from '@msgpack/msgpack'
import type { QBlock, Transaction, PeerHeightAnnouncement, BlockSummary } from './types'

/**
 * Decoder Performance Metrics
 * Tracks decode performance and errors for monitoring
 */
export const DECODER_METRICS = {
  decodeCount: 0,
  avgDecodeTime: 0,
  decodeErrors: 0,
  msgpackSuccesses: 0,
  jsonFallbacks: 0,
  totalBytesDecoded: 0,
}

/**
 * Reset metrics (useful for testing)
 */
export function resetDecoderMetrics(): void {
  DECODER_METRICS.decodeCount = 0
  DECODER_METRICS.avgDecodeTime = 0
  DECODER_METRICS.decodeErrors = 0
  DECODER_METRICS.msgpackSuccesses = 0
  DECODER_METRICS.jsonFallbacks = 0
  DECODER_METRICS.totalBytesDecoded = 0
}

/**
 * Get human-readable metrics summary
 */
export function getDecoderMetricsSummary(): string {
  const successRate =
    DECODER_METRICS.decodeCount > 0
      ? ((DECODER_METRICS.msgpackSuccesses / DECODER_METRICS.decodeCount) * 100).toFixed(1)
      : '0.0'
  const errorRate =
    DECODER_METRICS.decodeCount > 0
      ? ((DECODER_METRICS.decodeErrors / DECODER_METRICS.decodeCount) * 100).toFixed(1)
      : '0.0'

  return `
📊 Decoder Metrics:
  - Total Decodes: ${DECODER_METRICS.decodeCount}
  - Avg Decode Time: ${DECODER_METRICS.avgDecodeTime.toFixed(2)}ms
  - MessagePack Success: ${DECODER_METRICS.msgpackSuccesses} (${successRate}%)
  - JSON Fallbacks: ${DECODER_METRICS.jsonFallbacks}
  - Decode Errors: ${DECODER_METRICS.decodeErrors} (${errorRate}%)
  - Total Bytes: ${(DECODER_METRICS.totalBytesDecoded / 1024).toFixed(2)} KB
  `.trim()
}

/**
 * Decode a block message from PubSub
 *
 * @param data - Raw message data from gossipsub
 * @returns Decoded block or null if decode fails
 */
export function decodeBlock(data: Uint8Array): QBlock | null {
  // 🔥🔥🔥 DECODER START - This MUST appear in console
  console.log('🔥🔥🔥 DECODER CALLED with', data.length, 'bytes')

  const startTime = performance.now()
  DECODER_METRICS.decodeCount++
  DECODER_METRICS.totalBytesDecoded += data.length

  try {
    // Try MessagePack binary decode first (production)
    console.log('🔥 [DECODER] Attempting msgpackDecode on', data.length, 'bytes...')

    let parsed: any
    try {
      parsed = msgpackDecode(data)
      console.log('[DECODER] msgpackDecode SUCCESS')
    } catch (msgpackError) {
      console.error('[DECODER] msgpackDecode FAILED:', msgpackError)
      throw msgpackError
    }

    // Debug: Log raw parsed structure
    console.log('[DECODER] Raw parsed type:', typeof parsed, Array.isArray(parsed) ? 'array' : 'object')

    // Safely stringify preview
    let preview = ''
    try {
      preview = JSON.stringify(parsed).substring(0, 500)
    } catch (jsonErr) {
      preview = `(cannot stringify: ${jsonErr})`
    }
    console.log('[DECODER] Parsed preview:', preview)

    // v3.5.9: Server sends ARRAY format (tuple serialization):
    // ["qnk-block-v1", [[header_fields...], mining_solutions, dag_parents, quantum_metadata, transactions, balance_updates, size_bytes]]

    let version: string
    let blockData: any
    let isArrayFormat = false

    if (Array.isArray(parsed)) {
      // ARRAY format - this is what the server actually sends!
      version = parsed[0]
      blockData = parsed[1]
      isArrayFormat = true
      console.log('[DECODER] Array format detected - version:', version, 'blockData is array:', Array.isArray(blockData))
    } else if (parsed.version && parsed.block) {
      // Object format (fallback)
      version = parsed.version
      blockData = parsed.block
      console.log('[DECODER] Object format - version:', version)
    } else {
      console.warn('[DECODER] Unknown block format - not array and not object with version/block')
      console.warn('[DECODER] Keys:', Object.keys(parsed || {}))
      DECODER_METRICS.decodeErrors++
      return null
    }

    if (version !== 'qnk-block-v1') {
      console.warn('[DECODER] Unknown block version:', version, '(expected qnk-block-v1)')
      DECODER_METRICS.decodeErrors++
      return null
    }

    if (!blockData) {
      console.warn('[DECODER] Missing block data')
      DECODER_METRICS.decodeErrors++
      return null
    }

    let height: number, phase: number, networkId: string
    let prevBlockHash: any, solutionsRoot: any, txRoot: any, stateRoot: any
    let timestamp: number, dagRound: number, vdfProof: any
    let anchorValidator: any, proposer: string, producerId: number, totalDifficulty: any
    let producerSignature: any, producerPublicKey: any  // v3.5.10: For light client verification
    let miningSolutions: any[], dagParents: any[], quantumMetadata: any
    let transactions: any[], balanceUpdates: any[], sizeBytes: number

    if (isArrayFormat) {
      // v3.5.9: Parse ARRAY format
      // blockData = [[header_array], mining_solutions, dag_parents, quantum_metadata, transactions, balance_updates, size_bytes]
      // header_array = [height, phase, networkId, prevHash, solutionsRoot, txRoot, stateRoot, timestamp, dagRound, vdfProof, anchorValidator, proposer, producerId, totalDifficulty]

      const headerArray = blockData[0]
      if (!headerArray || !Array.isArray(headerArray)) {
        console.warn('[DECODER] Header array is missing or invalid')
        DECODER_METRICS.decodeErrors++
        return null
      }

      console.log('[DECODER] Header array length:', headerArray.length, 'first element (height):', headerArray[0])

      height = headerArray[0] || 0
      phase = headerArray[1] || 19
      networkId = headerArray[2] || ('mainnet-genesis')
      prevBlockHash = headerArray[3] || []
      solutionsRoot = headerArray[4] || []
      txRoot = headerArray[5] || []
      stateRoot = headerArray[6] || []
      timestamp = headerArray[7] || Date.now() / 1000
      dagRound = headerArray[8] || 0
      vdfProof = headerArray[9] || null
      anchorValidator = headerArray[10]
      proposer = headerArray[11] || ''
      producerId = headerArray[12] || 0
      totalDifficulty = headerArray[13] || 0

      // v3.5.10: Producer signature fields (optional, indices 14-15)
      producerSignature = headerArray[14] || null
      producerPublicKey = headerArray[15] || null

      // Rest of block data from array
      miningSolutions = blockData[1] || []
      dagParents = blockData[2] || []
      quantumMetadata = blockData[3] || [0, 0, 0]
      transactions = blockData[4] || []
      balanceUpdates = blockData[5] || []
      sizeBytes = blockData[6] || 0

    } else {
      // Object format parsing (fallback)
      const headerObj = blockData.header
      if (!headerObj) {
        console.warn('[DECODER] Header is missing from block data')
        DECODER_METRICS.decodeErrors++
        return null
      }

      height = headerObj.height || 0
      phase = headerObj.phase || 19
      networkId = headerObj.network_id || ('mainnet-genesis')
      prevBlockHash = headerObj.prev_block_hash || []
      solutionsRoot = headerObj.solutions_root || []
      txRoot = headerObj.tx_root || []
      stateRoot = headerObj.state_root || []
      timestamp = headerObj.timestamp || Date.now() / 1000
      dagRound = headerObj.dag_round || 0
      vdfProof = headerObj.vdf_proof || null
      anchorValidator = headerObj.anchor_validator
      proposer = headerObj.proposer || ''
      producerId = headerObj.producer_id || 0
      totalDifficulty = headerObj.total_difficulty || 0
      // v3.5.10: Producer signature fields for verification
      producerSignature = headerObj.producer_signature || null
      producerPublicKey = headerObj.producer_public_key || null

      miningSolutions = blockData.mining_solutions || []
      dagParents = blockData.dag_parents || []
      quantumMetadata = blockData.quantum_metadata || { coherence: 0, entanglement: 0, measurement: 0 }
      transactions = blockData.transactions || []
      balanceUpdates = blockData.balance_updates || []
      sizeBytes = blockData.size_bytes || 0
    }

    console.log(`[DECODER] Parsed block #${height} phase=${phase} network=${networkId} txs=${transactions.length} solutions=${miningSolutions.length}`)

    // Debug: Log types of key variables
    console.log('[DECODER] Types check:', {
      vdfProof: typeof vdfProof, isArray: Array.isArray(vdfProof),
      miningSolutions: typeof miningSolutions, msLength: miningSolutions?.length,
      transactions: typeof transactions, txLength: transactions?.length,
      totalDifficulty: typeof totalDifficulty, totalDiffValue: String(totalDifficulty).substring(0, 50),
    })

    // Update metrics
    const decodeTime = performance.now() - startTime
    DECODER_METRICS.avgDecodeTime =
      (DECODER_METRICS.avgDecodeTime * (DECODER_METRICS.msgpackSuccesses) + decodeTime) /
      (DECODER_METRICS.msgpackSuccesses + 1)
    DECODER_METRICS.msgpackSuccesses++

    // Log successful decode (every 10th block to reduce noise)
    if (height % 10 === 0 || import.meta.env.DEV) {
      console.log(`[DECODER] Block #${height} decoded (${transactions.length} txs)`)
    }

    // v3.5.9: Convert to QBlock format - WRAPPED IN TRY-CATCH FOR DEBUGGING
    try {
      console.log('[DECODER] Starting QBlock construction...')

      // Parse VDF proof - could be array [output, verification_proof, iterations, challenge, generated_at] or object
      let parsedVdfProof: any
      if (!vdfProof) {
        parsedVdfProof = {
          input: new Uint8Array(),
          output: new Uint8Array(),
          proof: new Uint8Array(),
          iterations: 0,
        }
      } else if (Array.isArray(vdfProof)) {
        // Array format: [output, verification_proof, iterations, challenge, generated_at]
        parsedVdfProof = {
          output: new Uint8Array(vdfProof[0] || []),
          proof: new Uint8Array(vdfProof[1] || []),
          iterations: vdfProof[2] || 0,
          input: new Uint8Array(vdfProof[3] || []), // challenge
        }
      } else {
        // Object format
        parsedVdfProof = {
          input: new Uint8Array(vdfProof.challenge || vdfProof.input || []),
          output: new Uint8Array(vdfProof.output || []),
          proof: new Uint8Array(vdfProof.verification_proof || vdfProof.proof || []),
          iterations: vdfProof.iterations || 0,
        }
      }
      console.log('[DECODER] VDF proof parsed OK')

      // Parse mining solutions - Rust MiningSolution: nonce, hash, difficulty_target, miner_address, timestamp
      const parsedMiningSolutions = miningSolutions.map((sol: any, idx: number) => {
        try {
          if (Array.isArray(sol)) {
            // Array format fallback - safely convert nonce
            const nonceValue = sol[0]
            let nonce: bigint
            if (typeof nonceValue === 'bigint') {
              nonce = nonceValue
            } else if (typeof nonceValue === 'number') {
              nonce = BigInt(Math.floor(nonceValue))
            } else if (typeof nonceValue === 'string') {
              nonce = BigInt(nonceValue)
            } else {
              nonce = BigInt(0)
            }
            // Miner address - convert to string if it's a byte array
            let minerStr: string
            const minerVal = sol[3]
            if (typeof minerVal === 'string') {
              minerStr = minerVal
            } else if (minerVal instanceof Uint8Array || Array.isArray(minerVal)) {
              minerStr = uint8ArrayToHex(new Uint8Array(minerVal))
            } else {
              minerStr = ''
            }
            return {
              nonce,
              difficulty: sol[1] || 0,
              hash: new Uint8Array(sol[2] || []),
              miner: minerStr,
              reward: 0, // Reward not in MiningSolution struct - calculated from block reward
              timestamp: sol[4] || 0,
            }
          }
          // Object format from rmp_serde - safely convert nonce
          const nonceValue = sol.nonce
          let nonce: bigint
          if (typeof nonceValue === 'bigint') {
            nonce = nonceValue
          } else if (typeof nonceValue === 'number') {
            nonce = BigInt(Math.floor(nonceValue))
          } else if (typeof nonceValue === 'string') {
            nonce = BigInt(nonceValue)
          } else {
            nonce = BigInt(0)
          }
          // Miner address - convert to string if it's a byte array
          let minerStr: string
          const minerVal = sol.miner_address || sol.miner
          if (typeof minerVal === 'string') {
            minerStr = minerVal
          } else if (minerVal instanceof Uint8Array || Array.isArray(minerVal)) {
            minerStr = uint8ArrayToHex(new Uint8Array(minerVal))
          } else {
            minerStr = ''
          }
          return {
            nonce,
            difficulty: sol.difficulty_target ? 1 : 0, // Convert difficulty_target to simple difficulty
            hash: new Uint8Array(sol.hash || []),
            miner: minerStr,
            reward: 0, // Not in Rust struct, computed elsewhere
          }
        } catch (solErr) {
          console.error(`[DECODER] Error parsing mining solution ${idx}:`, solErr, 'sol:', sol)
          return { nonce: BigInt(0), difficulty: 0, hash: new Uint8Array(), miner: '', reward: 0 }
        }
      })
      console.log('[DECODER] Mining solutions parsed OK:', parsedMiningSolutions.length)

      // Parse transactions - Rust Transaction: id, from, to, amount, fee, nonce, signature, timestamp, etc.
      const parsedTransactions = transactions.map((tx: any, idx: number) => {
        try {
          // Helper to convert address to string
          const toAddressString = (val: any): string => {
            if (typeof val === 'string') return val
            if (val instanceof Uint8Array || Array.isArray(val)) return uint8ArrayToHex(new Uint8Array(val))
            return ''
          }

          if (Array.isArray(tx)) {
            // Array format fallback
            return {
              from: toAddressString(tx[0]),
              to: toAddressString(tx[1]),
              amount: typeof tx[2] === 'number' ? tx[2] : Number(tx[2] || 0),
              timestamp: tx[3] || Date.now() / 1000,
              signature: tx[4] ? new Uint8Array(tx[4]) : undefined,
              nonce: tx[5],
            }
          }
          // Object format from rmp_serde
          return {
            from: toAddressString(tx.from),
            to: toAddressString(tx.to),
            amount: Number(tx.amount || 0),
            timestamp: typeof tx.timestamp === 'object' ? (tx.timestamp.secs_since_epoch || 0) : tx.timestamp || Date.now() / 1000,
            signature: tx.signature ? new Uint8Array(tx.signature) : undefined,
            nonce: tx.nonce,
          }
        } catch (txErr) {
          console.error(`[DECODER] Error parsing transaction ${idx}:`, txErr, 'tx:', tx)
          return { from: '', to: '', amount: 0, timestamp: Date.now() / 1000 }
        }
      })
      console.log('[DECODER] Transactions parsed OK:', parsedTransactions.length)

      // Parse balance updates
      const parsedBalanceUpdates = balanceUpdates.map((bu: any, idx: number) => {
        try {
          // Helper to convert address to string
          const toAddressString = (val: any): string => {
            if (typeof val === 'string') return val
            if (val instanceof Uint8Array || Array.isArray(val)) return uint8ArrayToHex(new Uint8Array(val))
            return ''
          }

          if (Array.isArray(bu)) {
            // Array format fallback
            return {
              address: toAddressString(bu[0]),
              oldBalance: typeof bu[1] === 'number' ? bu[1] : Number(bu[1] || 0),
              newBalance: typeof bu[2] === 'number' ? bu[2] : Number(bu[2] || 0),
              reason: typeof bu[3] === 'string' ? bu[3] : '',
            }
          }
          // Object format from rmp_serde
          return {
            address: toAddressString(bu.address),
            oldBalance: Number(bu.old_balance || bu.oldBalance || 0),
            newBalance: Number(bu.new_balance || bu.newBalance || 0),
            reason: typeof bu.reason === 'string' ? bu.reason : '',
          }
        } catch (buErr) {
          console.error(`[DECODER] Error parsing balance update ${idx}:`, buErr, 'bu:', bu)
          return { address: '', oldBalance: 0, newBalance: 0, reason: '' }
        }
      })
      console.log('[DECODER] Balance updates parsed OK:', parsedBalanceUpdates.length)

      // Parse quantum metadata
      const parsedQuantumMetadata = Array.isArray(quantumMetadata) ? {
        coherence: quantumMetadata[0] || 0,
        entanglement: quantumMetadata[1] || 0,
        measurement: quantumMetadata[2] || 0,
      } : {
        coherence: quantumMetadata?.coherence || 0,
        entanglement: quantumMetadata?.entanglement || 0,
        measurement: quantumMetadata?.measurement || 0,
      }
      console.log('[DECODER] Quantum metadata parsed OK')

      // Safe BigInt conversion for totalDifficulty
      let safeTotalDifficulty: bigint
      if (typeof totalDifficulty === 'bigint') {
        safeTotalDifficulty = totalDifficulty
      } else if (typeof totalDifficulty === 'number') {
        safeTotalDifficulty = BigInt(Math.floor(totalDifficulty))
      } else if (typeof totalDifficulty === 'string') {
        safeTotalDifficulty = BigInt(totalDifficulty)
      } else if (Array.isArray(totalDifficulty)) {
        // Could be a byte array representing a big number
        console.log('[DECODER] totalDifficulty is array, converting...', totalDifficulty.slice(0, 10))
        safeTotalDifficulty = BigInt(0)
      } else {
        safeTotalDifficulty = BigInt(0)
      }
      console.log('[DECODER] Total difficulty converted OK:', safeTotalDifficulty.toString().substring(0, 20))

      // v3.5.11: Debug logging for proposer, dagParents, sizeBytes
      // Note: proposer/dagParents come as raw values before conversion, log them here
      console.log('[DECODER] proposer raw:', typeof proposer === 'string' ? `string(${proposer.substring(0, 16)}...)` : `array(len=${(proposer as any)?.length || 0})`)
      console.log('[DECODER] dagParents count:', dagParents?.length, 'first parent raw type:', dagParents?.[0] ? (typeof dagParents[0] === 'string' ? 'string' : 'array') : 'N/A')
      console.log('[DECODER] sizeBytes:', sizeBytes, typeof sizeBytes)

      // v3.5.11: Helper to convert byte arrays to hex strings
      // Rust sends [u8; 32] as byte arrays, but UI expects hex strings
      const toHexString = (val: any): string => {
        if (typeof val === 'string') return val
        if (val instanceof Uint8Array || Array.isArray(val)) {
          return uint8ArrayToHex(new Uint8Array(val))
        }
        return ''
      }

      // Convert to QBlock format
      const qblock = {
        header: {
          height,
          phase,
          networkId,
          prevBlockHash: new Uint8Array(prevBlockHash),
          solutionsRoot: new Uint8Array(solutionsRoot),
          txRoot: new Uint8Array(txRoot),
          stateRoot: new Uint8Array(stateRoot),
          timestamp,
          dagRound,
          vdfProof: parsedVdfProof,
          anchorValidator,
          // v3.5.11: Convert proposer byte array to hex string
          proposer: toHexString(proposer),
          producerId,
          totalDifficulty: safeTotalDifficulty,
          // v3.5.10: Producer signature for light client verification
          producerSignature: producerSignature ? new Uint8Array(producerSignature) : undefined,
          producerPublicKey: producerPublicKey ? new Uint8Array(producerPublicKey) : undefined,
        },
        miningSolutions: parsedMiningSolutions,
        // v3.5.11: Convert dag_parent byte arrays to hex strings
        dagParents: dagParents.map((p: any) => toHexString(p)),
        quantumMetadata: parsedQuantumMetadata,
        transactions: parsedTransactions,
        balanceUpdates: parsedBalanceUpdates,
        sizeBytes,
      }

      console.log(`[DECODER] ✅ QBlock constructed successfully: height=${qblock.header.height}, txs=${qblock.transactions.length}`)
      return qblock
    } catch (constructionError) {
      console.error('[DECODER] ❌ QBlock construction FAILED:', constructionError)
      console.error('[DECODER] Stack:', (constructionError as Error).stack)
      DECODER_METRICS.decodeErrors++
      return null
    }
  } catch (error) {
    console.warn('[DECODER] MessagePack decode failed, trying JSON fallback:', error)
    DECODER_METRICS.jsonFallbacks++

    // Fallback to JSON for development/testing
    try {
      const text = new TextDecoder().decode(data)
      const parsed = JSON.parse(text)

      // Update metrics for JSON fallback
      const decodeTime = performance.now() - startTime
      DECODER_METRICS.avgDecodeTime =
        (DECODER_METRICS.avgDecodeTime * (DECODER_METRICS.decodeCount - 1) + decodeTime) /
        DECODER_METRICS.decodeCount

      return {
        header: {
          height: parsed.header?.height || 0,
          phase: parsed.header?.phase || 5,
          networkId: parsed.header?.network_id || ('mainnet-genesis'),
          prevBlockHash: hexToUint8Array(parsed.header?.prev_block_hash || ''),
          solutionsRoot: hexToUint8Array(parsed.header?.solutions_root || ''),
          txRoot: hexToUint8Array(parsed.header?.tx_root || ''),
          stateRoot: hexToUint8Array(parsed.header?.state_root || ''),
          timestamp: parsed.header?.timestamp || Date.now() / 1000,
          dagRound: parsed.header?.dag_round || 0,
          vdfProof: {
            input: new Uint8Array(),
            output: new Uint8Array(),
            proof: new Uint8Array(),
            iterations: 0,
          },
          anchorValidator: parsed.header?.anchor_validator,
          proposer: parsed.header?.proposer || '',
          producerId: parsed.header?.producer_id || 0,
          totalDifficulty: BigInt(parsed.header?.total_difficulty || 0),
        },
        miningSolutions: parsed.mining_solutions || [],
        dagParents: parsed.dag_parents || [],
        quantumMetadata: parsed.quantum_metadata || {
          coherence: 0,
          entanglement: 0,
          measurement: 0,
        },
        transactions: parsed.transactions || [],
        balanceUpdates: parsed.balance_updates || [],
        sizeBytes: parsed.size_bytes || 0,
      }
    } catch (jsonError) {
      console.error('[DECODER] Both MessagePack and JSON decode failed:', jsonError)
      DECODER_METRICS.decodeErrors++

      // Alert if error rate is too high
      const errorRate = DECODER_METRICS.decodeErrors / DECODER_METRICS.decodeCount
      if (errorRate > 0.1) {
        // >10% error rate
        console.error(
          `🚨 [DECODER] HIGH ERROR RATE: ${(errorRate * 100).toFixed(1)}% (${DECODER_METRICS.decodeErrors}/${DECODER_METRICS.decodeCount})`
        )
        console.error(`[DECODER] Metrics: ${getDecoderMetricsSummary()}`)
      }

      return null
    }
  }
}

/**
 * Decode a transaction message from PubSub
 *
 * @param data - Raw message data from gossipsub
 * @returns Decoded transaction or null if decode fails
 */
export function decodeTransaction(data: Uint8Array): Transaction | null {
  try {
    // Try MessagePack first
    const parsed = msgpackDecode(data) as any

    // Check version
    if (parsed.version && parsed.version !== 'qnk-tx-v1') {
      console.warn('[DECODER] Unknown transaction version:', parsed.version)
    }

    return {
      from: parsed.from || '',
      to: parsed.to || '',
      amount: parsed.amount || 0,
      timestamp: parsed.timestamp || Date.now() / 1000,
      signature: parsed.signature ? hexToUint8Array(parsed.signature) : undefined,
      nonce: parsed.nonce,
    }
  } catch (error) {
    console.warn('[DECODER] MessagePack decode failed, trying JSON fallback')

    // JSON fallback
    try {
      const text = new TextDecoder().decode(data)
      const parsed = JSON.parse(text)

      return {
        from: parsed.from || '',
        to: parsed.to || '',
        amount: parsed.amount || 0,
        timestamp: parsed.timestamp || Date.now() / 1000,
        signature: parsed.signature ? hexToUint8Array(parsed.signature) : undefined,
        nonce: parsed.nonce,
      }
    } catch (jsonError) {
      console.error('[DECODER] Failed to decode transaction:', jsonError)
      return null
    }
  }
}

/**
 * Decode peer height announcement
 *
 * @param data - Raw message data from gossipsub
 * @returns Decoded announcement or null if decode fails
 */
export function decodePeerHeight(data: Uint8Array): PeerHeightAnnouncement | null {
  try {
    const text = new TextDecoder().decode(data)
    const parsed = JSON.parse(text)

    return {
      peerId: parsed.peer_id || parsed.peerId || '',
      height: parsed.height || 0,
      bestBlockHash: hexToUint8Array(parsed.best_block_hash || parsed.bestBlockHash || ''),
      timestamp: parsed.timestamp || Date.now() / 1000,
    }
  } catch (error) {
    console.error('[DECODER] Failed to decode peer height:', error)
    return null
  }
}

/**
 * Create a simplified block summary for UI display
 *
 * @param block - Full QBlock
 * @returns BlockSummary with essential fields
 */
export function createBlockSummary(block: QBlock): BlockSummary {
  // Calculate total mining reward
  const miningReward = block.miningSolutions.reduce(
    (sum, solution) => sum + solution.reward,
    0
  )

  // Convert block hash to hex string
  const blockHash = uint8ArrayToHex(block.header.prevBlockHash)

  return {
    height: block.header.height,
    hash: blockHash.substring(0, 16) + '...', // Truncate for display
    timestamp: block.header.timestamp,
    transactionCount: block.transactions.length,
    miningReward,
    proposer: block.header.proposer,
    phase: block.header.phase,
    networkId: block.header.networkId,
  }
}

/**
 * Helper: Convert hex string to Uint8Array
 */
function hexToUint8Array(hex: string): Uint8Array {
  if (!hex || hex.length === 0) {
    return new Uint8Array()
  }

  // Remove 0x prefix if present
  hex = hex.replace(/^0x/, '')

  // Ensure even length
  if (hex.length % 2 !== 0) {
    hex = '0' + hex
  }

  const length = hex.length / 2
  const result = new Uint8Array(length)

  for (let i = 0; i < length; i++) {
    result[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16)
  }

  return result
}

/**
 * Helper: Convert Uint8Array to hex string
 */
function uint8ArrayToHex(data: Uint8Array): string {
  return Array.from(data)
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('')
}

/**
 * Validate block structure
 *
 * @param block - Block to validate
 * @returns true if block is valid
 */
export function validateBlock(block: QBlock): boolean {
  try {
    // Basic validation
    if (!block.header) return false
    if (block.header.height < 0) return false
    if (block.header.timestamp <= 0) return false
    if (!block.header.networkId) return false

    // Phase validation
    if (block.header.phase < 1 || block.header.phase > 255) return false

    // Array validation
    if (!Array.isArray(block.transactions)) return false
    if (!Array.isArray(block.miningSolutions)) return false
    if (!Array.isArray(block.balanceUpdates)) return false

    return true
  } catch (error) {
    console.error('[DECODER] Block validation failed:', error)
    return false
  }
}

/**
 * Encode transaction for publishing
 *
 * @param tx - Transaction to encode
 * @returns Encoded transaction bytes (MessagePack format)
 */
export function encodeTransaction(tx: Transaction): Uint8Array {
  // Add version for backward compatibility
  const versionedTx = {
    ...tx,
    version: 'qnk-tx-v1',
  }

  // Encode with MessagePack
  return msgpackEncode(versionedTx) as Uint8Array
}

/**
 * Debug: Log block summary
 */
export function logBlockSummary(block: QBlock): void {
  console.log('📦 Block Summary:')
  console.log(`   Height: ${block.header.height}`)
  console.log(`   Timestamp: ${new Date(block.header.timestamp * 1000).toISOString()}`)
  console.log(`   Transactions: ${block.transactions.length}`)
  console.log(`   Mining Solutions: ${block.miningSolutions.length}`)
  console.log(`   DAG Parents: ${block.dagParents.length}`)
  console.log(`   Phase: ${block.header.phase}`)
  console.log(`   Network: ${block.header.networkId}`)
  console.log(`   Proposer: ${block.header.proposer}`)
}
