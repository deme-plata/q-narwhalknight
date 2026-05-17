/**
 * useMinerLink Hook
 *
 * Manages a WebSocket connection to the API server's miner-link relay,
 * providing real-time stats from the user's personal miner(s) and the
 * ability to send control commands (pause, resume, set threads, etc.).
 *
 * Usage:
 * ```typescript
 * const { isConnected, miners, totalHashrate, sendCommand } = useMinerLink(walletAddress)
 * ```
 */

import { useState, useEffect, useCallback, useRef } from 'react'

// ─── Types ──────────────────────────────────────────────────────────────────

export interface GpuDeviceInfo {
  name: string
  vendor: string
  compute_units: number
  memory_mb: number
  max_clock_mhz: number
  api: string
}

export interface MinerInfo {
  minerId: string
  minerName: string | null
  hashrate: number          // H/s
  totalHashes: number
  solutions: number
  blocksFound: number
  uptimeSecs: number
  threadsActive: number
  threadsTotal: number
  cpuVendor: string
  hasAvx2: boolean
  hasAvx512: boolean
  intensity: number
  isMining: boolean
  currentBlockHeight: number
  lastUpdate: Date
  hashrateHistory: number[] // last 60 readings for sparkline
  avgHashrate5m: number
  // GPU fields
  gpuActive: boolean
  gpuHashrate: number
  gpuDevices: GpuDeviceInfo[]
}

export type MinerCommandAction =
  | { action: 'Pause' }
  | { action: 'Resume' }
  | { action: 'SetThreads'; count: number }
  | { action: 'SetIntensity'; level: number }
  | { action: 'GetDetailedStats' }

export interface PendingCommand {
  commandId: string
  action: MinerCommandAction
  sentAt: Date
  status: 'pending' | 'success' | 'failed'
  message?: string
}

export interface UseMinerLinkReturn {
  isConnected: boolean
  miners: MinerInfo[]
  totalHashrate: number
  totalSolutions: number
  sendCommand: (minerId: string, cmd: MinerCommandAction) => void
  pendingCommands: PendingCommand[]
}

// ─── Message types (matching Rust MinerLinkMessage) ─────────────────────────

interface StatsMessage {
  type: 'Stats'
  miner_id: string
  hashrate: number
  total_hashes: number
  solutions: number
  blocks_found: number
  uptime_secs: number
  threads_active: number
  threads_total: number
  cpu_vendor: string
  has_avx2: boolean
  has_avx512: boolean
  intensity: number
  is_mining: boolean
  current_block_height: number
  temperature_estimate: number | null
  gpu_active?: boolean
  gpu_hashrate?: number
  gpu_devices?: GpuDeviceInfo[]
}

interface SolutionFoundMessage {
  type: 'SolutionFound'
  miner_id: string
  block_height: number
  nonce: number
  hash_preview: string
}

interface AckMessage {
  type: 'Ack'
  command_id: string
  success: boolean
  message: string
}

interface LinkEstablishedMessage {
  type: 'LinkEstablished'
  peer_type: string
  connected_miners: number
  connected_wallets: number
}

type MinerLinkMessage = StatsMessage | SolutionFoundMessage | AckMessage | LinkEstablishedMessage | { type: 'Pong' }

// ─── Constants ──────────────────────────────────────────────────────────────

const HASHRATE_HISTORY_SIZE = 60 // 60 seconds of history for sparkline
const MINER_OFFLINE_TIMEOUT_MS = 5000 // Mark miner offline if no stats for 5s
const RECONNECT_DELAYS = [2000, 4000, 8000, 16000, 30000] // Exponential backoff

// ─── Hook ───────────────────────────────────────────────────────────────────

export function useMinerLink(walletAddress: string | null): UseMinerLinkReturn {
  const [isConnected, setIsConnected] = useState(false)
  const [miners, setMiners] = useState<MinerInfo[]>([])
  const [pendingCommands, setPendingCommands] = useState<PendingCommand[]>([])

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttempt = useRef(0)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const offlineCheckTimer = useRef<ReturnType<typeof setInterval> | null>(null)

  // Build the WS URL
  const getWsUrl = useCallback(() => {
    if (!walletAddress) return null
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    return `${proto}//${host}/api/v1/miner-link/ws?role=wallet&wallet=${walletAddress}`
  }, [walletAddress])

  // Send a command to a specific miner
  const sendCommand = useCallback((minerId: string, cmd: MinerCommandAction) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
    const commandId = `cmd_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    const message = {
      type: 'Command',
      command_id: commandId,
      action: cmd,
    }
    wsRef.current.send(JSON.stringify(message))
    setPendingCommands(prev => [
      ...prev.slice(-19), // Keep last 20
      { commandId, action: cmd, sentAt: new Date(), status: 'pending' as const },
    ])
  }, [])

  // Connect WebSocket
  useEffect(() => {
    const wsUrl = getWsUrl()
    if (!wsUrl) return

    const connect = () => {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setIsConnected(true)
        reconnectAttempt.current = 0
      }

      ws.onmessage = (event) => {
        try {
          const msg: MinerLinkMessage = JSON.parse(event.data)
          switch (msg.type) {
            case 'Stats': {
              const stats = msg as StatsMessage
              setMiners(prev => {
                const existing = prev.find(m => m.minerId === stats.miner_id)
                const history = existing?.hashrateHistory ?? []
                const newHistory = [...history, stats.hashrate].slice(-HASHRATE_HISTORY_SIZE)
                const avg5m = newHistory.length > 0
                  ? newHistory.reduce((a, b) => a + b, 0) / newHistory.length
                  : 0

                const updated: MinerInfo = {
                  minerId: stats.miner_id,
                  minerName: null, // Will be set from Register if available
                  hashrate: stats.hashrate,
                  totalHashes: stats.total_hashes,
                  solutions: stats.solutions,
                  blocksFound: stats.blocks_found,
                  uptimeSecs: stats.uptime_secs,
                  threadsActive: stats.threads_active,
                  threadsTotal: stats.threads_total,
                  cpuVendor: stats.cpu_vendor,
                  hasAvx2: stats.has_avx2,
                  hasAvx512: stats.has_avx512,
                  intensity: stats.intensity,
                  isMining: stats.is_mining,
                  currentBlockHeight: stats.current_block_height,
                  lastUpdate: new Date(),
                  hashrateHistory: newHistory,
                  avgHashrate5m: avg5m,
                  gpuActive: stats.gpu_active ?? false,
                  gpuHashrate: stats.gpu_hashrate ?? 0,
                  gpuDevices: stats.gpu_devices ?? [],
                }

                if (existing) {
                  // Preserve minerName from existing
                  updated.minerName = existing.minerName
                  return prev.map(m => m.minerId === stats.miner_id ? updated : m)
                }
                return [...prev, updated]
              })
              break
            }
            case 'SolutionFound': {
              const sol = msg as SolutionFoundMessage
              setMiners(prev =>
                prev.map(m =>
                  m.minerId === sol.miner_id
                    ? { ...m, solutions: m.solutions + 1 }
                    : m
                )
              )
              break
            }
            case 'Ack': {
              const ack = msg as AckMessage
              setPendingCommands(prev =>
                prev.map(c =>
                  c.commandId === ack.command_id
                    ? { ...c, status: ack.success ? 'success' as const : 'failed' as const, message: ack.message }
                    : c
                )
              )
              break
            }
            case 'LinkEstablished': {
              // Initial connection info
              break
            }
            case 'Pong':
              break
          }
        } catch {
          // Ignore malformed messages
        }
      }

      ws.onclose = () => {
        setIsConnected(false)
        wsRef.current = null
        // Reconnect with backoff
        const delay = RECONNECT_DELAYS[Math.min(reconnectAttempt.current, RECONNECT_DELAYS.length - 1)]
        reconnectAttempt.current++
        reconnectTimer.current = setTimeout(connect, delay)
      }

      ws.onerror = () => {
        // onclose will fire after onerror
      }
    }

    connect()

    // Periodic check for offline miners
    offlineCheckTimer.current = setInterval(() => {
      const now = Date.now()
      setMiners(prev =>
        prev.map(m => {
          if (now - m.lastUpdate.getTime() > MINER_OFFLINE_TIMEOUT_MS) {
            return { ...m, isMining: false }
          }
          return m
        }).filter(m => now - m.lastUpdate.getTime() < 60000) // Remove after 60s stale
      )
    }, 2000)

    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (offlineCheckTimer.current) clearInterval(offlineCheckTimer.current)
      if (wsRef.current) {
        wsRef.current.onclose = null // Prevent reconnect on intentional close
        wsRef.current.close()
        wsRef.current = null
      }
      setIsConnected(false)
    }
  }, [getWsUrl])

  // Computed totals
  const totalHashrate = miners.reduce((sum, m) => sum + (m.isMining ? m.hashrate : 0), 0)
  const totalSolutions = miners.reduce((sum, m) => sum + m.solutions, 0)

  return {
    isConnected,
    miners,
    totalHashrate,
    totalSolutions,
    sendCommand,
    pendingCommands,
  }
}
