/**
 * Verification Reporter - Report Blocks That Fail Verification
 *
 * Browser nodes perform light client verification on received blocks.
 * When verification fails, this module reports the failure to the network,
 * helping identify bad actors and invalid blocks.
 *
 * v3.5.x: Browser P2P Network Contribution - Feature 2
 */

import type { Libp2p } from 'libp2p'
import type { GossipSub } from '@libp2p/gossipsub'
import { encode as msgpackEncode } from '@msgpack/msgpack'
import { TOPICS, PROTOCOL_VERSION, VERIFICATION_REPORT_CONFIG } from './config'
import type { VerificationReport, VerificationResult, QBlock } from './types'

/**
 * Verification reporter statistics
 */
export interface ReporterStats {
  // Total reports sent
  totalReports: number

  // Critical reports
  criticalReports: number

  // Warning reports
  warningReports: number

  // Reports rate limited
  rateLimited: number

  // Reports in cooldown
  cooldownSkipped: number

  // Last report timestamp
  lastReportTime: number
}

// Global stats
const reporterStats: ReporterStats = {
  totalReports: 0,
  criticalReports: 0,
  warningReports: 0,
  rateLimited: 0,
  cooldownSkipped: 0,
  lastReportTime: 0,
}

// Rate limiting state
const reportTimestamps: number[] = []
const reportedBlocks = new Map<string, number>() // blockHash -> lastReportTime

/**
 * Get verification reporter statistics
 */
export function getReporterStats(): ReporterStats {
  return { ...reporterStats }
}

/**
 * Reset reporter statistics (for testing)
 */
export function resetReporterStats(): void {
  reporterStats.totalReports = 0
  reporterStats.criticalReports = 0
  reporterStats.warningReports = 0
  reporterStats.rateLimited = 0
  reporterStats.cooldownSkipped = 0
  reporterStats.lastReportTime = 0
  reportTimestamps.length = 0
  reportedBlocks.clear()
}

/**
 * Convert block hash to hex string
 */
function blockHashToHex(hash: Uint8Array): string {
  return Array.from(hash)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('')
}

/**
 * Check if we're rate limited
 */
function isRateLimited(): boolean {
  const now = Date.now()
  const oneMinuteAgo = now - 60000

  // Remove timestamps older than 1 minute
  while (reportTimestamps.length > 0 && reportTimestamps[0] < oneMinuteAgo) {
    reportTimestamps.shift()
  }

  return reportTimestamps.length >= VERIFICATION_REPORT_CONFIG.MAX_REPORTS_PER_MINUTE
}

/**
 * Check if block is in cooldown
 */
function isInCooldown(blockHash: string): boolean {
  const lastReport = reportedBlocks.get(blockHash)
  if (!lastReport) return false

  return Date.now() - lastReport < VERIFICATION_REPORT_CONFIG.REPORT_COOLDOWN
}

/**
 * Determine severity based on failed checks
 */
function determineSeverity(failedChecks: string[]): 'warning' | 'critical' {
  // Critical failures: signature, hash, structure
  const criticalChecks = [
    'Signature Valid',
    'Block Hash Valid',
    'Block Structure',
    'Parent Chain Valid',
  ]

  const hasCriticalFailure = failedChecks.some(check =>
    criticalChecks.some(critical => check.includes(critical))
  )

  return hasCriticalFailure ? 'critical' : 'warning'
}

/**
 * Report a verification failure to the network
 *
 * This function publishes a verification report to the gossipsub network
 * when a block fails light client verification. Other nodes can use these
 * reports to identify problematic peers or blocks.
 *
 * @param libp2p - The libp2p node instance
 * @param block - The block that failed verification
 * @param verificationResult - The verification result with failed checks
 * @returns Promise<boolean> - Whether the report was successfully sent
 */
export async function reportVerificationFailure(
  libp2p: Libp2p,
  block: QBlock,
  verificationResult: VerificationResult
): Promise<boolean> {
  // Check if reporting is enabled
  if (!VERIFICATION_REPORT_CONFIG.ENABLED) {
    console.log(`🔇 [VERIFICATION REPORTER] Reporting disabled`)
    return false
  }

  // Skip if verification passed
  if (verificationResult.valid) {
    return false
  }

  // Get failed checks
  const failedChecks = verificationResult.checks
    .filter(c => !c.passed)
    .map(c => c.name)

  // Check minimum failed checks threshold
  if (failedChecks.length < VERIFICATION_REPORT_CONFIG.MIN_FAILED_CHECKS) {
    return false
  }

  // Determine severity
  const severity = determineSeverity(failedChecks)

  // Skip warnings if configured
  if (VERIFICATION_REPORT_CONFIG.CRITICAL_ONLY && severity === 'warning') {
    console.log(`🔇 [VERIFICATION REPORTER] Skipping warning (critical only mode)`)
    return false
  }

  // Get block hash
  const blockHash = blockHashToHex(block.header.prevBlockHash)

  // Check cooldown
  if (isInCooldown(blockHash)) {
    console.log(`⏳ [VERIFICATION REPORTER] Block ${block.header.height} in cooldown, skipping`)
    reporterStats.cooldownSkipped++
    return false
  }

  // Check rate limit
  if (isRateLimited()) {
    console.log(`🚫 [VERIFICATION REPORTER] Rate limited, skipping report`)
    reporterStats.rateLimited++
    return false
  }

  const timestamp = Date.now()

  // Create report
  const report: VerificationReport = {
    reporterId: libp2p.peerId.toString(),
    blockHeight: block.header.height,
    blockHash,
    failedChecks,
    failureDetails: verificationResult.checks
      .filter(c => !c.passed)
      .map(c => c.details),
    timestamp,
    severity,
    nodeType: 'browser',
    protocolVersion: PROTOCOL_VERSION,
  }

  console.log(`📢 [VERIFICATION REPORTER] Reporting ${severity} failure for block ${block.header.height}`)
  console.log(`   Failed checks: ${failedChecks.join(', ')}`)

  try {
    // Get pubsub service
    const pubsub = libp2p.services.pubsub as GossipSub

    // Encode report
    const encoded = msgpackEncode(report)

    // Publish to verification reports topic
    const result = await pubsub.publish(TOPICS.VERIFICATION_REPORTS, encoded)

    const recipients = result.recipients || []
    console.log(`✅ [VERIFICATION REPORTER] Report sent to ${recipients.length} peers`)

    // Update stats
    reporterStats.totalReports++
    if (severity === 'critical') {
      reporterStats.criticalReports++
    } else {
      reporterStats.warningReports++
    }
    reporterStats.lastReportTime = timestamp

    // Update rate limiting state
    reportTimestamps.push(timestamp)
    reportedBlocks.set(blockHash, timestamp)

    // Clean up old cooldown entries
    const cooldownCutoff = timestamp - VERIFICATION_REPORT_CONFIG.REPORT_COOLDOWN
    for (const [hash, time] of reportedBlocks.entries()) {
      if (time < cooldownCutoff) {
        reportedBlocks.delete(hash)
      }
    }

    return true
  } catch (error) {
    console.error(`❌ [VERIFICATION REPORTER] Failed to send report:`, error)
    return false
  }
}

/**
 * Create a VerificationReporter instance that automatically reports failures
 *
 * This class wraps the verification process and automatically reports
 * failures to the network.
 */
export class VerificationReporter {
  private libp2p: Libp2p | null = null
  private enabled: boolean = VERIFICATION_REPORT_CONFIG.ENABLED

  /**
   * Initialize the reporter with a libp2p node
   */
  initialize(libp2p: Libp2p): void {
    this.libp2p = libp2p
    console.log(`🔧 [VERIFICATION REPORTER] Initialized`)
  }

  /**
   * Enable or disable reporting
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled
    console.log(`🔧 [VERIFICATION REPORTER] ${enabled ? 'Enabled' : 'Disabled'}`)
  }

  /**
   * Check if reporter is ready
   */
  isReady(): boolean {
    return this.libp2p !== null && this.enabled
  }

  /**
   * Report a verification failure
   */
  async report(block: QBlock, result: VerificationResult): Promise<boolean> {
    if (!this.libp2p || !this.enabled) {
      return false
    }

    return reportVerificationFailure(this.libp2p, block, result)
  }

  /**
   * Get current statistics
   */
  getStats(): ReporterStats {
    return getReporterStats()
  }
}

// Global reporter instance
export const verificationReporter = new VerificationReporter()
