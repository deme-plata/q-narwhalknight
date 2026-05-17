/**
 * useP2PData Hook - v3.5.24
 *
 * React hook for P2P-first data fetching.
 * Provides easy access to:
 * - Block fetching (P2P-first, HTTP fallback)
 * - Transaction confirmation verification
 * - Offline-resilient sync
 * - Historical TX lookup
 *
 * Usage:
 * ```tsx
 * const { fetchBlock, verifyTransaction, isOffline, stats } = useP2PData()
 *
 * // Fetch a block (P2P-first)
 * const result = await fetchBlock(1429800)
 * console.log(`Block from ${result.source} in ${result.latencyMs}ms`)
 *
 * // Verify a transaction from multiple peers
 * const consensus = await verifyTransaction('abc123...')
 * console.log(`Confirmed by ${consensus.agreementCount} peers`)
 * ```
 */

import { useState, useCallback, useEffect } from 'react'
import { useLibP2P } from '../contexts/LibP2PContext'
import {
  p2pDataService,
  getP2PDataStats,
  type P2PFetchResult,
  type ConsensusResult,
  type P2PDataServiceStats,
} from '../libp2p/p2pDataService'
import type { QBlock } from '../libp2p/types'

export interface UseP2PDataResult {
  // Fetch a block (P2P-first, HTTP fallback)
  fetchBlock: (height: number) => Promise<P2PFetchResult<QBlock>>

  // Fetch multiple blocks (resilient to offline)
  fetchBlocks: (startHeight: number, endHeight: number) => Promise<QBlock[]>

  // Verify a transaction is confirmed by multiple peers
  verifyTransaction: (txHash: string, expectedBlockHeight?: number) => Promise<ConsensusResult>

  // Find a transaction in historical blocks
  findTransaction: (txHash: string, searchRange?: { startHeight: number; endHeight: number }) => Promise<{ tx: any; block: QBlock } | null>

  // Get transaction history for an address
  getAddressHistory: (address: string, limit?: number) => Promise<{ transactions: any[]; source: string }>

  // Is currently in offline mode
  isOffline: boolean

  // Service statistics
  stats: P2PDataServiceStats

  // Is P2P ready
  isP2PReady: boolean
}

/**
 * P2P Data Hook
 */
export function useP2PData(): UseP2PDataResult {
  const { node, isReady } = useLibP2P()
  const [isOffline, setIsOffline] = useState(false)
  const [stats, setStats] = useState<P2PDataServiceStats>(getP2PDataStats())

  // Update stats periodically
  useEffect(() => {
    const interval = setInterval(() => {
      setStats(getP2PDataStats())
      setIsOffline(p2pDataService.isOfflineMode())
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  // Fetch a single block (P2P-first)
  const fetchBlock = useCallback(async (height: number): Promise<P2PFetchResult<QBlock>> => {
    const result = await p2pDataService.fetchBlock(height)
    setStats(getP2PDataStats())
    return result
  }, [])

  // Fetch multiple blocks (offline-resilient)
  const fetchBlocks = useCallback(async (startHeight: number, endHeight: number): Promise<QBlock[]> => {
    const blocks = await p2pDataService.fetchBlocksResilient(startHeight, endHeight)
    setStats(getP2PDataStats())
    setIsOffline(p2pDataService.isOfflineMode())
    return blocks
  }, [])

  // Verify transaction confirmation from multiple peers
  const verifyTransaction = useCallback(async (
    txHash: string,
    expectedBlockHeight?: number
  ): Promise<ConsensusResult> => {
    const result = await p2pDataService.verifyTransactionConfirmation(txHash, expectedBlockHeight)
    setStats(getP2PDataStats())
    return result
  }, [])

  // Find a transaction
  const findTransaction = useCallback(async (
    txHash: string,
    searchRange?: { startHeight: number; endHeight: number }
  ) => {
    return p2pDataService.findTransaction(txHash, searchRange)
  }, [])

  // Get address history
  const getAddressHistory = useCallback(async (address: string, limit: number = 50) => {
    return p2pDataService.getAddressHistory(address, limit)
  }, [])

  return {
    fetchBlock,
    fetchBlocks,
    verifyTransaction,
    findTransaction,
    getAddressHistory,
    isOffline,
    stats,
    isP2PReady: isReady,
  }
}

export default useP2PData
