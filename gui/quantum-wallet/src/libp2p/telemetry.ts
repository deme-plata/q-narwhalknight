/**
 * Telemetry Reporter - Network Health Metrics
 *
 * Browser nodes periodically report network health metrics to help
 * the network understand overall health and performance. Metrics include
 * connection count, block latency, verification rate, and uptime.
 *
 * v3.5.x: Browser P2P Network Contribution - Feature 3
 */

import type { Libp2p } from 'libp2p'
import type { GossipSub } from '@libp2p/gossipsub'
import { encode as msgpackEncode } from '@msgpack/msgpack'
import { TOPICS, NETWORK_ID, PROTOCOL_VERSION, TELEMETRY_CONFIG } from './config'
import type { TelemetryReport } from './types'

/**
 * Internal telemetry state
 */
interface TelemetryState {
  // When the node started
  startTime: number

  // Blocks received (total)
  blocksReceived: number

  // Blocks received in last 24h (tracked by timestamps)
  blockTimestamps: number[]

  // Block latencies (ms) for averaging
  blockLatencies: number[]

  // Verification stats
  blocksVerified: number
  blocksValid: number
  blocksInvalid: number

  // Current block height
  currentHeight: number

  // Last report time
  lastReportTime: number

  // Report interval handle
  reportInterval: NodeJS.Timeout | null
}

// Global telemetry state
const state: TelemetryState = {
  startTime: Date.now(),
  blocksReceived: 0,
  blockTimestamps: [],
  blockLatencies: [],
  blocksVerified: 0,
  blocksValid: 0,
  blocksInvalid: 0,
  currentHeight: 0,
  lastReportTime: 0,
  reportInterval: null,
}

/**
 * Get current telemetry statistics
 */
export function getTelemetryStats(): Omit<TelemetryReport, 'peerId' | 'timestamp'> {
  const now = Date.now()
  const twentyFourHoursAgo = now - 24 * 60 * 60 * 1000

  // Count blocks in last 24h
  const blocksReceived24h = state.blockTimestamps.filter(t => t > twentyFourHoursAgo).length

  // Calculate average latency
  const avgBlockLatencyMs = state.blockLatencies.length > 0
    ? state.blockLatencies.reduce((a, b) => a + b, 0) / state.blockLatencies.length
    : 0

  // Calculate verification rate
  const verificationRate = state.blocksReceived > 0
    ? (state.blocksVerified / state.blocksReceived) * 100
    : 0

  // Calculate uptime
  const uptime = Math.floor((now - state.startTime) / 1000)

  return {
    nodeType: 'browser',
    connectedPeers: 0, // Will be filled in by reporter
    blocksReceived24h,
    avgBlockLatencyMs,
    verificationRate,
    uptime,
    currentHeight: state.currentHeight,
    blocksVerified: state.blocksVerified,
    blocksValid: state.blocksValid,
    blocksInvalid: state.blocksInvalid,
    networkId: NETWORK_ID,
    protocolVersion: PROTOCOL_VERSION,
  }
}

/**
 * Record a block received
 *
 * @param height - Block height
 * @param blockTimestamp - Block's timestamp (Unix seconds)
 * @param receiveTime - When the block was received (Date.now())
 */
export function recordBlockReceived(
  height: number,
  blockTimestamp: number,
  receiveTime: number = Date.now()
): void {
  state.blocksReceived++
  state.blockTimestamps.push(receiveTime)

  // Update current height
  if (height > state.currentHeight) {
    state.currentHeight = height
  }

  // Calculate latency (ms from block production to receipt)
  // blockTimestamp is in seconds, receiveTime is in ms
  const latencyMs = receiveTime - (blockTimestamp * 1000)

  // Only record positive, reasonable latencies (< 5 minutes)
  if (latencyMs > 0 && latencyMs < 5 * 60 * 1000) {
    state.blockLatencies.push(latencyMs)

    // Keep only last N samples
    if (state.blockLatencies.length > TELEMETRY_CONFIG.LATENCY_SAMPLE_SIZE) {
      state.blockLatencies.shift()
    }
  }

  // Clean up old block timestamps (older than 24h)
  const twentyFourHoursAgo = Date.now() - 24 * 60 * 60 * 1000
  state.blockTimestamps = state.blockTimestamps.filter(t => t > twentyFourHoursAgo)
}

/**
 * Record verification result
 *
 * @param valid - Whether the block passed verification
 */
export function recordVerification(valid: boolean): void {
  state.blocksVerified++
  if (valid) {
    state.blocksValid++
  } else {
    state.blocksInvalid++
  }
}

/**
 * Reset telemetry state (for testing)
 */
export function resetTelemetryState(): void {
  state.startTime = Date.now()
  state.blocksReceived = 0
  state.blockTimestamps = []
  state.blockLatencies = []
  state.blocksVerified = 0
  state.blocksValid = 0
  state.blocksInvalid = 0
  state.currentHeight = 0
  state.lastReportTime = 0
}

/**
 * Detect if running on mobile device
 */
function isMobile(): boolean {
  if (typeof navigator === 'undefined') return false
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  )
}

/**
 * Telemetry Reporter Class
 *
 * Manages periodic telemetry reporting to the network.
 */
export class TelemetryReporter {
  private libp2p: Libp2p | null = null
  private enabled: boolean = TELEMETRY_CONFIG.ENABLED
  private reportInterval: NodeJS.Timeout | null = null

  /**
   * Initialize the reporter with a libp2p node
   */
  initialize(libp2p: Libp2p): void {
    this.libp2p = libp2p
    console.log(`🔧 [TELEMETRY] Initialized`)
  }

  /**
   * Start periodic telemetry reporting
   */
  start(): void {
    if (!this.libp2p || !this.enabled) {
      console.log(`🔇 [TELEMETRY] Not starting (${!this.libp2p ? 'no node' : 'disabled'})`)
      return
    }

    // Stop any existing interval
    this.stop()

    // Determine report interval based on device type
    const interval = isMobile()
      ? TELEMETRY_CONFIG.MOBILE_REPORT_INTERVAL
      : TELEMETRY_CONFIG.REPORT_INTERVAL

    console.log(`📡 [TELEMETRY] Starting with ${interval / 1000}s interval`)

    // Send initial report after short delay
    setTimeout(() => {
      this.sendReport().catch(err => {
        console.warn(`[TELEMETRY] Initial report failed:`, err)
      })
    }, 5000)

    // Set up periodic reporting
    this.reportInterval = setInterval(() => {
      this.sendReport().catch(err => {
        console.warn(`[TELEMETRY] Report failed:`, err)
      })
    }, interval)
  }

  /**
   * Stop telemetry reporting
   */
  stop(): void {
    if (this.reportInterval) {
      clearInterval(this.reportInterval)
      this.reportInterval = null
      console.log(`🛑 [TELEMETRY] Stopped`)
    }
  }

  /**
   * Enable or disable telemetry
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled
    console.log(`🔧 [TELEMETRY] ${enabled ? 'Enabled' : 'Disabled'}`)

    if (enabled && this.libp2p) {
      this.start()
    } else {
      this.stop()
    }
  }

  /**
   * Check if reporter is active
   */
  isActive(): boolean {
    return this.reportInterval !== null
  }

  /**
   * Send a telemetry report
   */
  async sendReport(): Promise<boolean> {
    if (!this.libp2p) {
      return false
    }

    // Check minimum interval
    const now = Date.now()
    if (now - state.lastReportTime < TELEMETRY_CONFIG.MIN_REPORT_INTERVAL) {
      console.log(`⏳ [TELEMETRY] Skipping report (too soon)`)
      return false
    }

    try {
      // Get pubsub service
      const pubsub = this.libp2p.services.pubsub as GossipSub

      // Get current stats
      const stats = getTelemetryStats()

      // Create report
      const report: TelemetryReport = {
        ...stats,
        peerId: this.libp2p.peerId.toString(),
        connectedPeers: this.libp2p.getConnections().length,
        timestamp: now,
      }

      console.log(`📤 [TELEMETRY] Sending report...`)
      console.log(`   Uptime: ${report.uptime}s`)
      console.log(`   Blocks (24h): ${report.blocksReceived24h}`)
      console.log(`   Avg Latency: ${report.avgBlockLatencyMs.toFixed(1)}ms`)
      console.log(`   Verification Rate: ${report.verificationRate.toFixed(1)}%`)
      console.log(`   Connected Peers: ${report.connectedPeers}`)

      // Encode report
      const encoded = msgpackEncode(report)

      // Publish to telemetry topic
      const result = await pubsub.publish(TOPICS.TELEMETRY, encoded)

      const recipients = result.recipients || []
      console.log(`✅ [TELEMETRY] Report sent to ${recipients.length} peers`)

      state.lastReportTime = now
      return true
    } catch (error) {
      console.error(`❌ [TELEMETRY] Failed to send report:`, error)
      return false
    }
  }

  /**
   * Get current telemetry statistics
   */
  getStats(): ReturnType<typeof getTelemetryStats> {
    return getTelemetryStats()
  }
}

// Global telemetry reporter instance
export const telemetryReporter = new TelemetryReporter()
