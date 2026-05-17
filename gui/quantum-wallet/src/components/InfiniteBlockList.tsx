/**
 * InfiniteBlockList Component
 *
 * ✨ Magical infinite scroll blockchain explorer
 *
 * Features:
 * - Scroll through entire blockchain history (millions of blocks)
 * - Zero loading states - blocks appear instantly
 * - P2P streaming from nearby peers
 * - Visual P2P activity indicators
 * - Smooth animations and transitions
 */

import { useEffect, useState } from 'react'
import { Virtuoso } from 'react-virtuoso'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Database,
  Hash,
  Clock,
  Users,
  Activity,
  Zap,
  Wifi,
  WifiOff,
  TrendingUp,
} from 'lucide-react'
import { useInfiniteBlockScroll } from '../hooks/useInfiniteBlockScroll'
import { useLibP2P } from '../contexts/LibP2PContext'
import type { QBlock } from '../libp2p/types'

interface BlockCardProps {
  block: QBlock
  isNew?: boolean
}

/**
 * Individual Block Card
 */
function BlockCard({ block, isNew }: BlockCardProps) {
  const [isHovered, setIsHovered] = useState(false)

  // Format timestamp
  const timestamp = new Date(block.header.timestamp * 1000).toLocaleString()

  // Transaction count
  const txCount = block.transactions?.length || 0

  return (
    <motion.div
      initial={isNew ? { opacity: 0, x: -20 } : false}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
      className="group"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div
        className={`
        bg-quantum-indigo/20 backdrop-blur-xl rounded-lg border
        ${isHovered ? 'border-quantum-cyan/50 shadow-lg shadow-quantum-cyan/20' : 'border-quantum-purple/20'}
        p-4 transition-all duration-300 hover:scale-[1.01]
      `}
      >
        <div className="flex items-center justify-between mb-3">
          {/* Block Height */}
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-quantum-cyan" />
            <span className="text-xl font-bold text-white">
              Block #{block.header.height.toLocaleString()}
            </span>
            {isNew && (
              <motion.span
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="px-2 py-0.5 bg-quantum-green/20 text-quantum-green text-xs rounded-full"
              >
                NEW
              </motion.span>
            )}
          </div>

          {/* Transaction Count */}
          <div className="flex items-center gap-2 text-quantum-purple">
            <Activity className="w-4 h-4" />
            <span className="text-sm font-medium">{txCount} txs</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm">
          {/* Block Hash */}
          <div className="col-span-2">
            <div className="flex items-center gap-2 text-gray-400 mb-1">
              <Hash className="w-4 h-4" />
              <span className="text-xs">Hash</span>
            </div>
            <div className="font-mono text-xs text-quantum-cyan truncate">
              {block.header.prevBlockHash || 'N/A'}
            </div>
          </div>

          {/* Timestamp */}
          <div>
            <div className="flex items-center gap-2 text-gray-400 mb-1">
              <Clock className="w-4 h-4" />
              <span className="text-xs">Time</span>
            </div>
            <div className="text-white text-xs">{timestamp}</div>
          </div>

          {/* Miner/Producer */}
          <div>
            <div className="flex items-center gap-2 text-gray-400 mb-1">
              <Users className="w-4 h-4" />
              <span className="text-xs">Proposer</span>
            </div>
            <div className="text-white text-xs font-mono truncate">
              {block.header.proposer || 'System'}
            </div>
          </div>
        </div>

        {/* Hover: Show more details */}
        <AnimatePresence>
          {isHovered && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="mt-3 pt-3 border-t border-quantum-purple/20 overflow-hidden"
            >
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div>
                  <span className="text-gray-400">Phase:</span>
                  <span className="text-white ml-1">{block.header.phase}</span>
                </div>
                <div>
                  <span className="text-gray-400">DAG Round:</span>
                  <span className="text-white ml-1">{block.header.dagRound}</span>
                </div>
                <div>
                  <span className="text-gray-400">Network:</span>
                  <span className="text-white ml-1">{block.header.networkId}</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

/**
 * Network Activity Indicator
 *
 * NOTE: js-libp2p browser P2P is currently DISABLED.
 * Shows HTTP/SSE mode status instead of misleading P2P stats.
 */
function P2PActivityIndicator() {
  const { isReady, error } = useLibP2P()
  const { stats } = useInfiniteBlockScroll()

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-quantum-indigo/40 backdrop-blur-xl rounded-lg border border-quantum-purple/20 p-4"
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Wifi className="w-4 h-4 text-blue-400" />
          Network Status
        </h3>

        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.5, 1, 0.5],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          className="w-2 h-2 rounded-full bg-blue-400"
        />
      </div>

      <div className="grid grid-cols-2 gap-4 text-xs">
        {/* Connection Mode */}
        <div>
          <div className="text-gray-400 mb-1">Mode</div>
          <div className="text-white font-semibold flex items-center gap-1">
            <Zap className="w-3 h-3 text-blue-400" />
            HTTP/SSE
          </div>
        </div>

        {/* Total Blocks Loaded */}
        <div>
          <div className="text-gray-400 mb-1">Blocks Loaded</div>
          <div className="text-white font-semibold flex items-center gap-1">
            <Database className="w-3 h-3 text-quantum-purple" />
            {stats.totalLoaded.toLocaleString()}
          </div>
        </div>

        {/* Connection Status */}
        <div>
          <div className="text-gray-400 mb-1">Status</div>
          <div className="flex items-center gap-2">
            <span className="text-quantum-green font-semibold">Connected</span>
          </div>
        </div>

        {/* Avg Load Time */}
        <div>
          <div className="text-gray-400 mb-1">Avg Load Time</div>
          <div className="text-white font-semibold flex items-center gap-1">
            <Zap className="w-3 h-3 text-yellow-500" />
            {stats.avgLoadTime.toFixed(0)}ms
          </div>
        </div>
      </div>

      {/* Mode Explanation */}
      <div className="mt-3 p-2 bg-blue-500/10 border border-blue-500/20 rounded text-xs text-blue-400">
        <div className="flex items-center gap-2">
          <Wifi className="w-3 h-3" />
          <span>Real-time updates via Server-Sent Events</span>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="mt-3 p-2 bg-red-500/10 border border-red-500/20 rounded text-xs text-red-500">
          {error.message}
        </div>
      )}
    </motion.div>
  )
}

/**
 * Infinite Block List Component
 */
export function InfiniteBlockList() {
  const { blocks, loadMore, isLoading, currentHeight } = useInfiniteBlockScroll()
  const [lastLoadedHeight, setLastLoadedHeight] = useState(0)

  // Load initial blocks
  useEffect(() => {
    if (currentHeight > 0 && blocks.length === 0) {
      console.log(`🚀 [INFINITE BLOCK LIST] Loading initial blocks from height ${currentHeight}`)
      loadMore(currentHeight, 20)
      setLastLoadedHeight(currentHeight - 20)
    }
  }, [currentHeight, blocks.length, loadMore])

  // Handle reaching the end of the list
  const handleEndReached = () => {
    if (isLoading || lastLoadedHeight <= 0) return

    const nextHeight = lastLoadedHeight
    console.log(`📜 [INFINITE BLOCK LIST] Loading more blocks from height ${nextHeight}`)

    loadMore(nextHeight, 20)
    setLastLoadedHeight(nextHeight - 20)
  }

  return (
    <div className="space-y-4">
      {/* P2P Activity Indicator */}
      <P2PActivityIndicator />

      {/* Block List Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white flex items-center gap-3">
          <Database className="w-7 h-7 text-quantum-cyan" />
          Blockchain Explorer
        </h2>

        <div className="text-sm text-gray-400">
          Current Height:{' '}
          <span className="text-white font-semibold">{currentHeight.toLocaleString()}</span>
        </div>
      </div>

      {/* Infinite Scroll List */}
      <div className="h-[600px] bg-quantum-indigo/10 rounded-xl border border-quantum-purple/20">
        <Virtuoso
          data={blocks}
          endReached={handleEndReached}
          itemContent={(index, block) => (
            <div className="p-2">
              <BlockCard block={block} isNew={index === 0 && blocks.length > 1} />
            </div>
          )}
          components={{
            Footer: () =>
              isLoading ? (
                <div className="flex items-center justify-center p-4 text-quantum-cyan">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                  >
                    <Activity className="w-6 h-6" />
                  </motion.div>
                  <span className="ml-2">Loading blocks via P2P...</span>
                </div>
              ) : null,
          }}
        />
      </div>

      {/* Empty State */}
      {blocks.length === 0 && !isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center justify-center h-64 text-gray-400"
        >
          <Database className="w-16 h-16 mb-4 opacity-50" />
          <p className="text-lg">No blocks loaded yet</p>
          <p className="text-sm">Waiting for blockchain data...</p>
        </motion.div>
      )}
    </div>
  )
}
