/**
 * Transaction Submitter - P2P Transaction Submission
 *
 * Submits signed transactions directly through gossipsub instead of HTTP API.
 * This enables true P2P transaction propagation where browser nodes contribute
 * to the network by broadcasting their transactions.
 *
 * v3.5.x: Browser P2P Network Contribution - Feature 1
 * v3.7.4: Post-Quantum signatures (Dilithium5) for quantum-resistant transactions
 * v3.7.5: ZK-STARK proof commitments for privacy-preserving P2P transactions
 * v3.9.1: CRITICAL FIX - Await publish result before returning success
 *         Bug: Fire-and-forget pattern caused "confirmed" transactions to never arrive
 *         Fix: Now awaits pubsub.publish() and verifies recipients > 0 before returning success
 */

import type { Libp2p } from 'libp2p'
import type { GossipSub } from '@libp2p/gossipsub'
import { encode as msgpackEncode } from '@msgpack/msgpack'
import { TOPICS, NETWORK_ID, PROTOCOL_VERSION } from './config'
import type { SignedTransaction, Transaction } from './types'
import {
  generateStarkProofCommitment,
  serializeStarkCommitment,
  type StarkProofCommitment,
} from './zkStarkProof'

/**
 * Result of a transaction submission attempt
 */
export interface TransactionSubmitResult {
  // Whether submission succeeded
  success: boolean

  // Method used for submission
  method: 'p2p' | 'http' | 'none'

  // Transaction hash (if successful)
  txHash?: string

  // Error message (if failed)
  error?: string

  // Number of peers the transaction was published to
  peerCount?: number

  // Timestamp of submission
  timestamp: number

  // v3.7.5: ZK-STARK proof included (for privacy-preserving P2P)
  starkProofIncluded?: boolean

  // v3.7.5: STARK proof generation time in milliseconds
  starkProvingTimeMs?: number
}

/**
 * Transaction submission statistics
 */
export interface TransactionStats {
  // Total transactions submitted
  totalSubmitted: number

  // Successful P2P submissions
  p2pSuccess: number

  // Successful HTTP fallback submissions
  httpSuccess: number

  // Failed submissions
  failed: number

  // Last submission timestamp
  lastSubmission: number

  // v3.7.5: Transactions with ZK-STARK proofs
  starkProofSubmissions: number

  // v3.7.5: Total STARK proving time (ms)
  totalStarkProvingTimeMs: number
}

// Global stats
const txStats: TransactionStats = {
  totalSubmitted: 0,
  p2pSuccess: 0,
  httpSuccess: 0,
  failed: 0,
  lastSubmission: 0,
  starkProofSubmissions: 0,
  totalStarkProvingTimeMs: 0,
}

/**
 * Get transaction submission statistics
 */
export function getTransactionStats(): TransactionStats {
  return { ...txStats }
}

/**
 * Result of encoding a transaction with optional STARK proof
 */
interface EncodeResult {
  encoded: Uint8Array
  starkProofIncluded: boolean
  starkProvingTimeMs?: number
}

/**
 * Encode a signed transaction for P2P transmission
 * Uses MessagePack for compact binary encoding
 *
 * v3.7.4: Now includes optional Dilithium5 post-quantum signatures
 * v3.7.5: Now generates ZK-STARK proof commitment for privacy-preserving transactions
 */
async function encodeTransaction(tx: SignedTransaction): Promise<EncodeResult> {
  // Convert to wire format matching Rust's postcard/msgpack expectations
  const wireFormat: Record<string, any> = {
    from: tx.from,
    to: tx.to,
    // Convert bigint to string for msgpack (bigint not natively supported)
    amount: tx.amount.toString(),
    nonce: tx.nonce,
    timestamp: tx.timestamp,
    // Convert Uint8Array to array for msgpack
    signature: Array.from(tx.signature),
    public_key: Array.from(tx.publicKey),
    token_address: tx.tokenAddress || null,
    memo: tx.memo || null,
    // Include network info for validation
    network_id: NETWORK_ID,
    protocol_version: PROTOCOL_VERSION,
  }

  // v3.7.4: Include post-quantum signature if present
  if (tx.dilithium5Signature && tx.dilithium5PublicKey) {
    wireFormat.dilithium5_signature = Array.from(tx.dilithium5Signature)
    wireFormat.dilithium5_public_key = Array.from(tx.dilithium5PublicKey)
    wireFormat.signature_mode = tx.signatureMode || 'hybrid'

    console.log(`🔐 [TX ENCODE] Including Dilithium5 PQ signature (${tx.dilithium5Signature.length} bytes)`)
  } else {
    wireFormat.signature_mode = 'ed25519'
  }

  // v3.7.5: Generate ZK-STARK proof commitment for privacy-preserving P2P
  let starkProofIncluded = false
  let starkProvingTimeMs: number | undefined

  try {
    // Use Dilithium5 public key if available, otherwise use Ed25519 public key
    const proverPubKey = tx.dilithium5PublicKey || tx.publicKey

    console.log(`🔐 [TX ENCODE] Generating ZK-STARK proof commitment...`)
    const starkResult = await generateStarkProofCommitment(tx, proverPubKey)

    if (starkResult.success && starkResult.commitment) {
      // Serialize the STARK commitment for transmission
      const serializedStark = serializeStarkCommitment(starkResult.commitment)

      wireFormat.stark_proof_commitment = Array.from(serializedStark)
      wireFormat.stark_proof_version = starkResult.commitment.version

      starkProofIncluded = true
      starkProvingTimeMs = starkResult.provingTimeMs

      console.log(`✅ [TX ENCODE] ZK-STARK proof commitment included (${serializedStark.length} bytes, ${starkProvingTimeMs}ms)`)
    } else {
      console.warn(`⚠️  [TX ENCODE] STARK proof generation failed: ${starkResult.error}`)
      console.warn(`   Transaction will be submitted without STARK proof`)
    }
  } catch (starkError) {
    console.warn(`⚠️  [TX ENCODE] STARK proof generation error:`, starkError)
    console.warn(`   Transaction will be submitted without STARK proof`)
  }

  return {
    encoded: msgpackEncode(wireFormat),
    starkProofIncluded,
    starkProvingTimeMs,
  }
}

/**
 * Submit a signed transaction via P2P gossipsub
 *
 * This is the primary method for submitting transactions. The transaction
 * is published to the /qnk/{networkId}/transactions topic where it will
 * be picked up by full nodes for inclusion in blocks.
 *
 * @param libp2p - The libp2p node instance
 * @param tx - The signed transaction to submit
 * @returns Promise<TransactionSubmitResult>
 */
export async function submitTransactionP2P(
  libp2p: Libp2p,
  tx: SignedTransaction
): Promise<TransactionSubmitResult> {
  const timestamp = Date.now()
  txStats.totalSubmitted++

  console.log(`📤 [TX SUBMIT] Submitting transaction via P2P...`)
  console.log(`   From: ${tx.from.substring(0, 16)}...`)
  console.log(`   To: ${tx.to.substring(0, 16)}...`)
  console.log(`   Amount: ${tx.amount.toString()}`)
  console.log(`   Nonce: ${tx.nonce}`)
  console.log(`   Network: ${NETWORK_ID}`)
  console.log(`   Target topic: ${TOPICS.TRANSACTIONS}`)

  try {
    // Get pubsub service
    const pubsub = libp2p.services.pubsub as GossipSub
    if (!pubsub) {
      console.error(`❌ [TX SUBMIT] PubSub service not available!`)
      txStats.failed++
      return {
        success: false,
        method: 'none',
        error: 'PubSub service not initialized',
        timestamp,
      }
    }

    // Check if we have peers
    const peers = pubsub.getPeers()
    console.log(`📊 [TX SUBMIT] Connected gossipsub peers: ${peers.length}`)

    // Log all subscribed topics for debugging
    const topics = pubsub.getTopics()
    console.log(`📊 [TX SUBMIT] Subscribed topics: ${topics.join(', ')}`)

    if (peers.length === 0) {
      console.warn(`⚠️  [TX SUBMIT] No gossipsub peers connected`)
      console.warn(`   Tip: Check if you're connected to the bootstrap node`)
      console.warn(`   Run window.libp2pDebug.getPeers() for connection status`)
      txStats.failed++
      return {
        success: false,
        method: 'none',
        error: 'No peers connected',
        timestamp,
      }
    }

    // Encode transaction with ZK-STARK proof
    const encodeResult = await encodeTransaction(tx)
    const encoded = encodeResult.encoded
    console.log(`📦 [TX SUBMIT] Encoded transaction: ${encoded.length} bytes`)
    console.log(`📦 [TX SUBMIT] First 50 bytes (hex): ${Array.from(encoded.slice(0, 50)).map(b => b.toString(16).padStart(2, '0')).join('')}`)
    if (encodeResult.starkProofIncluded) {
      console.log(`🔐 [TX SUBMIT] ZK-STARK proof included (proving time: ${encodeResult.starkProvingTimeMs}ms)`)
    }

    // Check if topic has subscribers (mesh formed)
    const subscribers = pubsub.getSubscribers(TOPICS.TRANSACTIONS)
    console.log(`📡 [TX SUBMIT] Topic ${TOPICS.TRANSACTIONS} has ${subscribers.length} subscribers in mesh`)
    if (subscribers.length > 0) {
      console.log(`   Subscribers: ${subscribers.map(p => p.toString().substring(0, 16) + '...').join(', ')}`)
    }

    // v3.5.24-beta: CRITICAL FIX - If no subscribers, fail P2P and let fallback handle it
    // This fixes the bug where fire-and-forget returns success but transaction never arrives
    // because there's no one listening on the transactions topic
    if (subscribers.length === 0) {
      console.warn(`⚠️  [TX SUBMIT] No subscribers on transactions topic - P2P mesh not formed`)
      console.warn(`   This typically means bootstrap node hasn't formed mesh yet`)
      console.warn(`   Falling back to HTTP API for reliable delivery`)
      txStats.failed++
      return {
        success: false,
        method: 'p2p',
        error: 'No peers subscribed to transactions topic - mesh not formed',
        timestamp,
      }
    }

    // v3.9.1-beta: CRITICAL FIX - Actually await publish and verify delivery
    // The previous fire-and-forget pattern caused transactions to show "confirmed"
    // but never actually arrive because we returned success before delivery was verified.
    // Bug report: tx fc41e3fddd486bf62c0354a0bd17dc721a509a2d5ba9b354aca42ad9551f5245
    console.log(`📤 [TX SUBMIT] Publishing to topic: ${TOPICS.TRANSACTIONS}...`)

    // Generate tx hash first (we need it for the response)
    const txHash = await generateTxHash(tx)

    // v3.9.1-beta: AWAIT the publish and verify recipients
    // This is critical - we must NOT return success until delivery is confirmed
    try {
      const publishResult = await pubsub.publish(TOPICS.TRANSACTIONS, encoded)
      const recipients = publishResult.recipients || []

      console.log(`📡 [TX SUBMIT] P2P publish complete! Recipients: ${recipients.length}`)
      if (recipients.length > 0) {
        console.log(`   Recipients: ${recipients.map(p => p.toString().substring(0, 16) + '...').join(', ')}`)
      }

      // v3.9.1-beta: CRITICAL - Fail if no recipients received the transaction
      // This ensures we only return success when transaction was actually delivered
      if (recipients.length === 0) {
        console.error(`❌ [TX SUBMIT] Transaction published but NO RECIPIENTS received it!`)
        console.error(`   This means the transaction was NOT delivered to the network.`)
        console.error(`   Falling back to HTTP API for reliable delivery.`)
        txStats.failed++
        return {
          success: false,
          method: 'p2p',
          error: 'Transaction published but no peers received it - delivery failed',
          txHash,
          peerCount: 0,
          timestamp,
        }
      }

      // Transaction was actually delivered to at least one peer
      txStats.p2pSuccess++
      txStats.lastSubmission = timestamp

      // v3.7.5: Track STARK proof stats
      if (encodeResult.starkProofIncluded) {
        txStats.starkProofSubmissions++
        if (encodeResult.starkProvingTimeMs) {
          txStats.totalStarkProvingTimeMs += encodeResult.starkProvingTimeMs
        }
      }

      console.log(`✅ [TX SUBMIT] Transaction confirmed delivered to ${recipients.length} peer(s)`)

      return {
        success: true,
        method: 'p2p',
        txHash,
        peerCount: recipients.length, // Use actual recipients count, not connected peers
        timestamp,
        starkProofIncluded: encodeResult.starkProofIncluded,
        starkProvingTimeMs: encodeResult.starkProvingTimeMs,
      }
    } catch (publishError) {
      console.error(`❌ [TX SUBMIT] P2P publish failed:`, publishError)
      txStats.failed++
      return {
        success: false,
        method: 'p2p',
        error: publishError instanceof Error ? publishError.message : String(publishError),
        txHash,
        timestamp,
      }
    }
  } catch (error) {
    console.error(`❌ [TX SUBMIT] P2P submission failed:`, error)
    txStats.failed++

    return {
      success: false,
      method: 'p2p',
      error: error instanceof Error ? error.message : String(error),
      timestamp,
    }
  }
}

/**
 * Submit a transaction with HTTP fallback
 *
 * Tries P2P first, falls back to HTTP if P2P fails.
 *
 * @param libp2p - The libp2p node instance (can be null for HTTP-only)
 * @param tx - The signed transaction to submit
 * @param apiBase - Base URL for HTTP API fallback
 * @returns Promise<TransactionSubmitResult>
 */
export async function submitTransactionWithFallback(
  libp2p: Libp2p | null,
  tx: SignedTransaction,
  apiBase: string = 'https://quillon.xyz'
): Promise<TransactionSubmitResult> {
  // Try P2P first if node is available
  if (libp2p) {
    const p2pResult = await submitTransactionP2P(libp2p, tx)

    if (p2pResult.success) {
      return p2pResult
    }

    // v3.5.24-beta: Enhanced logging for P2P failures
    console.log(`🌐 [TX SUBMIT] P2P failed: ${p2pResult.error || 'unknown'}`)
    console.log(`   Reason: ${p2pResult.error?.includes('mesh') ? 'Gossipsub mesh not formed with bootstrap node' : 'P2P connectivity issue'}`)
    console.log(`   Falling back to HTTP API for guaranteed delivery...`)
  } else {
    console.log(`🌐 [TX SUBMIT] No P2P node available, using HTTP API directly`)
  }

  // HTTP fallback - always works as long as API is reachable
  const httpResult = await submitTransactionHTTP(tx, apiBase)

  if (httpResult.success) {
    console.log(`✅ [TX SUBMIT] HTTP fallback succeeded! Transaction delivered to bootstrap node`)
  } else {
    console.error(`❌ [TX SUBMIT] Both P2P and HTTP failed - transaction NOT submitted`)
    console.error(`   HTTP error: ${httpResult.error}`)
  }

  return httpResult
}

/**
 * Submit a transaction via HTTP API
 *
 * Fallback method when P2P is unavailable or fails.
 * v3.7.4: Now includes optional Dilithium5 post-quantum signatures
 *
 * @param tx - The signed transaction to submit
 * @param apiBase - Base URL for HTTP API
 * @returns Promise<TransactionSubmitResult>
 */
export async function submitTransactionHTTP(
  tx: SignedTransaction,
  apiBase: string = 'https://quillon.xyz'
): Promise<TransactionSubmitResult> {
  const timestamp = Date.now()

  console.log(`🌐 [TX SUBMIT] Submitting transaction via HTTP...`)

  try {
    // Convert to API format
    const apiTx: Record<string, any> = {
      from: tx.from,
      to: tx.to,
      amount: tx.amount.toString(),
      nonce: tx.nonce,
      timestamp: tx.timestamp,
      signature: uint8ArrayToHex(tx.signature),
      public_key: uint8ArrayToHex(tx.publicKey),
      token_address: tx.tokenAddress || undefined,
      memo: tx.memo || undefined,
    }

    // v3.7.4: Include post-quantum signature if present
    if (tx.dilithium5Signature && tx.dilithium5PublicKey) {
      apiTx.dilithium5_signature = uint8ArrayToHex(tx.dilithium5Signature)
      apiTx.dilithium5_public_key = uint8ArrayToHex(tx.dilithium5PublicKey)
      apiTx.signature_mode = tx.signatureMode || 'hybrid'
      console.log(`🔐 [TX SUBMIT] Including Dilithium5 PQ signature in HTTP request`)
    }

    const response = await fetch(`${apiBase}/api/v1/transactions/submit`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(apiTx),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`HTTP ${response.status}: ${errorText}`)
    }

    const result = await response.json()

    txStats.httpSuccess++
    txStats.lastSubmission = timestamp

    return {
      success: true,
      method: 'http',
      txHash: result.tx_hash || result.hash,
      timestamp,
    }
  } catch (error) {
    console.error(`❌ [TX SUBMIT] HTTP submission failed:`, error)
    txStats.failed++

    return {
      success: false,
      method: 'http',
      error: error instanceof Error ? error.message : String(error),
      timestamp,
    }
  }
}

/**
 * Create a simple Transaction from SignedTransaction
 * (For compatibility with existing code)
 */
export function signedToSimpleTransaction(signed: SignedTransaction): Transaction {
  return {
    from: signed.from,
    to: signed.to,
    amount: Number(signed.amount),
    timestamp: signed.timestamp,
    signature: signed.signature,
    nonce: signed.nonce,
  }
}

/**
 * Generate a transaction hash for tracking
 * v3.5.15-beta: Fixed to match backend SHA3-256 binary hash computation
 * Backend computes: SHA3-256(from_bytes || to_bytes || amount_le_16 || nonce_le_8 || timestamp_le_8)
 */
async function generateTxHash(tx: SignedTransaction): Promise<string> {
  // Import sha3_256 from @noble/hashes (same as walletAuth.ts)
  const { sha3_256 } = await import('@noble/hashes/sha3')

  // Parse addresses (strip 'qnk' prefix if present)
  const fromHex = tx.from.startsWith('qnk') ? tx.from.substring(3) : tx.from
  const toHex = tx.to.startsWith('qnk') ? tx.to.substring(3) : tx.to

  const fromBytes = hexToBytes(fromHex)
  const toBytes = hexToBytes(toHex)

  // Amount as 16-byte little-endian (u128)
  const amountBytes = new Uint8Array(16)
  let amount = tx.amount
  for (let i = 0; i < 16; i++) {
    amountBytes[i] = Number(amount & BigInt(0xff))
    amount = amount >> BigInt(8)
  }

  // Nonce as 8-byte little-endian (u64)
  const nonceBytes = new Uint8Array(8)
  const nonceView = new DataView(nonceBytes.buffer)
  nonceView.setBigUint64(0, BigInt(tx.nonce), true)

  // Timestamp as 8-byte little-endian (u64)
  const timestampBytes = new Uint8Array(8)
  const timestampView = new DataView(timestampBytes.buffer)
  timestampView.setBigUint64(0, BigInt(tx.timestamp), true)

  // Concatenate all fields (same order as backend)
  const totalLength = fromBytes.length + toBytes.length + amountBytes.length + nonceBytes.length + timestampBytes.length
  const combined = new Uint8Array(totalLength)
  let offset = 0
  combined.set(fromBytes, offset); offset += fromBytes.length
  combined.set(toBytes, offset); offset += toBytes.length
  combined.set(amountBytes, offset); offset += amountBytes.length
  combined.set(nonceBytes, offset); offset += nonceBytes.length
  combined.set(timestampBytes, offset)

  // Hash with SHA3-256 (same as backend) - returns Uint8Array, convert to hex
  const hashBytes = sha3_256(combined)
  return uint8ArrayToHex(hashBytes)
}

/**
 * Convert hex string to Uint8Array
 */
function hexToBytes(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2)
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16)
  }
  return bytes
}

/**
 * Convert Uint8Array to hex string
 */
function uint8ArrayToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('')
}
