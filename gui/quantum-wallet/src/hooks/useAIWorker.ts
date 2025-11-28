/**
 * React Hook for AI Worker Node Management
 *
 * Manages lifecycle of browser-based libp2p AI compute worker:
 * - Worker initialization and shutdown
 * - Real-time stats polling
 * - Earnings tracking
 * - Connection health monitoring
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { AIWorkerNode, type WorkerCapability, type InferenceRequest } from '../libp2p/ai-worker-node'

export interface WorkerStats {
  peerId: string | null
  capability: WorkerCapability
  connectedPeers: number
  avgLatency: number
  requestsProcessed: number
  tokensGenerated: number
  earningsQUG: number
  uptime: number
  isRunning: boolean
}

export interface UseAIWorkerReturn {
  worker: AIWorkerNode | null
  stats: WorkerStats
  isInitializing: boolean
  error: Error | null
  startWorker: () => Promise<boolean>
  stopWorker: () => Promise<void>
  setInferenceHandler: (handler: (request: InferenceRequest) => Promise<void>) => void
}

const STATS_POLL_INTERVAL = 3000 // Update stats every 3 seconds
const QUG_PER_TOKEN = 0.0001 // 10,000 tokens = 1 QUG

export function useAIWorker(): UseAIWorkerReturn {
  const [worker, setWorker] = useState<AIWorkerNode | null>(null)
  const [stats, setStats] = useState<WorkerStats>({
    peerId: null,
    capability: { type: 'None' },
    connectedPeers: 0,
    avgLatency: 0,
    requestsProcessed: 0,
    tokensGenerated: 0,
    earningsQUG: 0,
    uptime: 0,
    isRunning: false
  })
  const [isInitializing, setIsInitializing] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const statsIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const startTimeRef = useRef<number>(0)
  const requestCountRef = useRef<number>(0)
  const tokenCountRef = useRef<number>(0)

  const startWorker = useCallback(async (): Promise<boolean> => {
    try {
      setIsInitializing(true)
      setError(null)

      console.log('🚀 Starting AI Worker Node...')

      const newWorker = new AIWorkerNode()

      // Set up error handler
      newWorker.onError = (err: Error) => {
        console.error('❌ Worker error:', err)
        setError(err)
      }

      // Set up peer connection handlers
      newWorker.onPeerConnected = (peerId: string, latency: number) => {
        console.log(`🔗 Connected to peer ${peerId.slice(0, 8)}... (${latency}ms)`)
      }

      newWorker.onPeerDisconnected = (peerId: string) => {
        console.log(`🔌 Disconnected from peer ${peerId.slice(0, 8)}...`)
      }

      // Default inference request handler with mock token generation
      // Phase 2: Replace with real inference engine (see PHASE2_INFERENCE_INTEGRATION.md)
      newWorker.onInferenceRequest = async (request: InferenceRequest) => {
        console.log('📥 Received inference request:', request.request_id)
        requestCountRef.current++

        // Mock token generation for Phase 1 demonstration
        // Phase 2 will replace this with WebGPU/WASM inference engine
        for (let i = 0; i < request.max_tokens; i++) {
          // Simulate token generation delay
          await new Promise(resolve => setTimeout(resolve, 50))

          // Generate mock token
          const mockToken = `token_${i} `
          tokenCountRef.current++

          // Publish token to network
          await newWorker.publishToken(request.request_id, mockToken, i)

          console.log(`🔤 Published token ${i + 1}/${request.max_tokens} for ${request.request_id}`)
        }

        console.log(`✅ Completed inference request: ${request.request_id}`)
      }

      // Initialize the worker
      const success = await newWorker.init()

      if (success) {
        setWorker(newWorker)
        startTimeRef.current = Date.now()

        // Start stats polling
        statsIntervalRef.current = setInterval(() => {
          if (newWorker) {
            const nodeStats = newWorker.getStats()
            const uptimeSeconds = Math.floor((Date.now() - startTimeRef.current) / 1000)

            setStats({
              peerId: nodeStats.peerId || null,
              capability: nodeStats.capability,
              connectedPeers: nodeStats.connectedPeers,
              avgLatency: Math.round(nodeStats.avgLatency),
              requestsProcessed: requestCountRef.current,
              tokensGenerated: tokenCountRef.current,
              earningsQUG: tokenCountRef.current * QUG_PER_TOKEN,
              uptime: uptimeSeconds,
              isRunning: nodeStats.isRunning
            })
          }
        }, STATS_POLL_INTERVAL)

        console.log('✅ AI Worker Node started successfully')
      } else {
        throw new Error('Worker initialization failed')
      }

      return success
    } catch (err) {
      console.error('❌ Failed to start worker:', err)
      setError(err as Error)
      return false
    } finally {
      setIsInitializing(false)
    }
  }, [])

  const stopWorker = useCallback(async () => {
    if (worker) {
      console.log('🛑 Stopping AI Worker Node...')

      // Stop stats polling
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current)
        statsIntervalRef.current = null
      }

      await worker.stop()
      setWorker(null)

      // Reset stats
      setStats({
        peerId: null,
        capability: { type: 'None' },
        connectedPeers: 0,
        avgLatency: 0,
        requestsProcessed: 0,
        tokensGenerated: 0,
        earningsQUG: 0,
        uptime: 0,
        isRunning: false
      })

      requestCountRef.current = 0
      tokenCountRef.current = 0
      startTimeRef.current = 0

      console.log('✅ AI Worker Node stopped')
    }
  }, [worker])

  const setInferenceHandler = useCallback((
    handler: (request: InferenceRequest) => Promise<void>
  ) => {
    if (worker) {
      worker.onInferenceRequest = async (request: InferenceRequest) => {
        requestCountRef.current++
        await handler(request)
      }
    }
  }, [worker])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current)
      }
      if (worker) {
        worker.stop()
      }
    }
  }, [worker])

  return {
    worker,
    stats,
    isInitializing,
    error,
    startWorker,
    stopWorker,
    setInferenceHandler
  }
}

/**
 * Utility function to format uptime display
 */
export function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`
  } else {
    return `${secs}s`
  }
}

/**
 * Utility function to format QUG earnings
 */
export function formatEarnings(qug: number): string {
  if (qug < 0.0001) {
    return '< 0.0001 QUG'
  }
  return `${qug.toFixed(4)} QUG`
}

/**
 * Utility function to get capability display name
 */
export function getCapabilityDisplayName(capability: WorkerCapability): string {
  switch (capability.type) {
    case 'WebGPU':
      return `WebGPU (${capability.vram_mb}MB VRAM)`
    case 'WASM_CPU':
      return `WASM CPU (${capability.threads} threads)`
    case 'None':
      return 'No Compute Available'
    default:
      return 'Unknown'
  }
}

/**
 * Utility function to get capability badge color
 */
export function getCapabilityColor(capability: WorkerCapability): string {
  switch (capability.type) {
    case 'WebGPU':
      return '#10b981' // green-500
    case 'WASM_CPU':
      return '#f59e0b' // amber-500
    case 'None':
      return '#6b7280' // gray-500
    default:
      return '#6b7280'
  }
}
