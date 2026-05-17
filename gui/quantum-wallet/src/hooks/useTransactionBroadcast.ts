/**
 * useTransactionBroadcast Hook
 *
 * Direct P2P transaction broadcasting via Gossipsub.
 * Enables censorship-resistant transaction submission without relying on centralized API.
 *
 * Usage:
 * ```typescript
 * const { broadcastTransaction, isPending } = useTransactionBroadcast()
 *
 * // Broadcast transaction directly to P2P network
 * await broadcastTransaction({
 *   from: 'qnk...',
 *   to: 'qnk...',
 *   amount: 100,
 *   timestamp: Date.now() / 1000
 * })
 * ```
 */

import { useState, useCallback } from 'react'
import { useLibP2P } from '../contexts/LibP2PContext'
import { TOPICS } from '../libp2p/config'
import { encodeTransaction } from '../libp2p/decoder'
import type { Transaction } from '../libp2p/types'
import type { GossipSub } from '@libp2p/gossipsub'

export interface UseTransactionBroadcastResult {
  // Broadcast a transaction to the P2P network
  broadcastTransaction: (tx: Transaction) => Promise<BroadcastResult>

  // Pending state
  isPending: boolean

  // Error state
  error: Error | null

  // Last broadcasted transaction
  lastTransaction: Transaction | null
}

export interface BroadcastResult {
  success: boolean
  transactionId?: string
  error?: string
}

/**
 * Transaction broadcasting hook
 */
export function useTransactionBroadcast(): UseTransactionBroadcastResult {
  const { node, isReady } = useLibP2P()

  const [isPending, setIsPending] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [lastTransaction, setLastTransaction] = useState<Transaction | null>(null)

  /**
   * Broadcast transaction to P2P network
   */
  const broadcastTransaction = useCallback(
    async (tx: Transaction): Promise<BroadcastResult> => {
      // Validate node is ready
      if (!node || !isReady) {
        const errorMsg = 'P2P node not ready for transaction broadcast'
        console.error(`❌ [TX BROADCAST] ${errorMsg}`)
        setError(new Error(errorMsg))
        return { success: false, error: errorMsg }
      }

      setIsPending(true)
      setError(null)

      try {
        console.log('📤 [TX BROADCAST] Broadcasting transaction via P2P...')
        console.log(`   From: ${tx.from}`)
        console.log(`   To: ${tx.to}`)
        console.log(`   Amount: ${tx.amount}`)

        // Encode transaction for P2P transmission
        const txBytes = encodeTransaction(tx)

        console.log(`   Encoded size: ${txBytes.length} bytes`)

        // Publish to transaction topic
        const pubsub = node.services.pubsub as GossipSub
        const result = await pubsub.publish(
          TOPICS.TRANSACTIONS,
          txBytes
        )

        // Calculate transaction ID (simple hash of key fields)
        const txId = await calculateTransactionId(tx)

        console.log(`✅ [TX BROADCAST] Transaction broadcast successful!`)
        console.log(`   Transaction ID: ${txId}`)
        console.log(`   Peers reached: ${result.recipients?.length || 'unknown'}`)

        // Store last transaction
        setLastTransaction(tx)

        // Dispatch custom event for UI updates
        window.dispatchEvent(
          new CustomEvent('transaction-broadcast', {
            detail: { transaction: tx, transactionId: txId },
          })
        )

        setIsPending(false)

        return {
          success: true,
          transactionId: txId,
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err)
        console.error(`❌ [TX BROADCAST] Broadcast failed:`, errorMsg)

        setError(err instanceof Error ? err : new Error(errorMsg))
        setIsPending(false)

        return {
          success: false,
          error: errorMsg,
        }
      }
    },
    [node, isReady]
  )

  return {
    broadcastTransaction,
    isPending,
    error,
    lastTransaction,
  }
}

/**
 * Calculate transaction ID (simple hash)
 */
async function calculateTransactionId(tx: Transaction): Promise<string> {
  // Create transaction string for hashing
  const txString = `${tx.from}-${tx.to}-${tx.amount}-${tx.timestamp}`

  // Use Web Crypto API for hashing
  const encoder = new TextEncoder()
  const data = encoder.encode(txString)
  const hashBuffer = await crypto.subtle.digest('SHA-256', data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('')

  return hashHex.substring(0, 16) // Return first 16 chars
}

/**
 * useTransactionListener Hook
 *
 * Listen for incoming transactions from the P2P network.
 * Useful for real-time mempool visualization.
 */
export function useTransactionListener() {
  const { node, isReady } = useLibP2P()
  const [transactions, setTransactions] = useState<Transaction[]>([])

  const subscribe = useCallback(() => {
    if (!node || !isReady) return

    console.log(`📡 [TX LISTENER] Subscribing to ${TOPICS.TRANSACTIONS}`)

    const pubsub = node.services.pubsub as GossipSub
    pubsub.subscribe(TOPICS.TRANSACTIONS)

    pubsub.addEventListener('message', (event: any) => {
      if (event.detail.topic === TOPICS.TRANSACTIONS) {
        try {
          const txData = event.detail.data
          const decoder = new TextDecoder()
          const txJson = JSON.parse(decoder.decode(txData))

          console.log('📥 [TX LISTENER] New transaction received:', txJson)

          setTransactions((prev) => {
            const newTxs = [txJson, ...prev]
            return newTxs.slice(0, 50) // Keep last 50 transactions
          })
        } catch (err) {
          console.error('[TX LISTENER] Failed to decode transaction:', err)
        }
      }
    })
  }, [node, isReady])

  return {
    transactions,
    subscribe,
  }
}

/**
 * useHybridTransactionSubmit Hook
 *
 * Hybrid approach: Try P2P first, fallback to HTTP API.
 * Provides maximum reliability while maintaining decentralization preference.
 */
export function useHybridTransactionSubmit() {
  const { broadcastTransaction } = useTransactionBroadcast()
  const [method, setMethod] = useState<'p2p' | 'http' | null>(null)

  const submitTransaction = useCallback(
    async (tx: Transaction): Promise<BroadcastResult> => {
      console.log('🔄 [HYBRID TX] Attempting P2P broadcast first...')

      // Try P2P broadcast first
      const p2pResult = await broadcastTransaction(tx)

      if (p2pResult.success) {
        console.log('✅ [HYBRID TX] P2P broadcast successful!')
        setMethod('p2p')
        return p2pResult
      }

      // Fallback to HTTP API
      console.warn('⚠️  [HYBRID TX] P2P failed, falling back to HTTP API...')
      setMethod('http')

      try {
        // Use existing HTTP API
        const response = await fetch('/api/v1/transactions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(tx),
        })

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const data = await response.json()

        console.log('✅ [HYBRID TX] HTTP fallback successful!')

        return {
          success: true,
          transactionId: data.transaction_id || data.txId,
        }
      } catch (err) {
        console.error('❌ [HYBRID TX] Both P2P and HTTP failed:', err)

        return {
          success: false,
          error: `Both P2P and HTTP failed: ${err}`,
        }
      }
    },
    [broadcastTransaction]
  )

  return {
    submitTransaction,
    method,
  }
}
