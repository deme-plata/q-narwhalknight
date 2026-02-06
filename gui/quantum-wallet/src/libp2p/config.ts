/**
 * libp2p Configuration for Q-NarwhalKnight Browser Nodes
 *
 * This module contains all configuration for the browser's P2P node,
 * including bootstrap peers, network IDs, and protocol versions.
 */

// Network configuration
// 🔥 v3.4.2-browser: Updated to match Server Beta (testnet-phase19)
export const NETWORK_ID = 'testnet-phase19'
export const PROTOCOL_VERSION = '3.4.2'

/**
 * Bootstrap Peers - ALL connections routed through Tor bridge
 *
 * 🧅 MANDATORY TOR: All browser P2P traffic goes through Tor
 * - Port 9444: Dedicated Tor bridge WebSocket server
 * - Server-side Tor SOCKS5 proxy routes to Tor network
 * - User's IP is NEVER exposed to the P2P network
 *
 * Architecture:
 * Browser → wss://quillon.xyz:9444 → Tor SOCKS5 → Tor Network → .onion Bootstrap
 *
 * NO clearnet connections allowed - privacy is mandatory, not optional
 */
export const BOOTSTRAP_PEERS = [
  // 🧅 Server Beta (EU) - Tor Bridge Bootstrap
  // Port 9444: Tor bridge WebSocket endpoint
  // Traffic flows: Browser WebSocket → nginx:9444 → websockify → Tor SOCKS5 → Tor network
  // PeerID: 12D3KooWFrhdwDDTgxPX41mUyRgLcE1ozsBYArKM4DT8t4VLwuNx
  '/dns4/quillon.xyz/tcp/9444/wss/p2p/12D3KooWFrhdwDDTgxPX41mUyRgLcE1ozsBYArKM4DT8t4VLwuNx',

  // TODO: Add .onion bootstrap when Tor hidden service is configured
  // '/onion3/<56-char-onion-address>:9001/p2p/12D3KooWFrhdwDDTgxPX41mUyRgLcE1ozsBYArKM4DT8t4VLwuNx',

  // TODO: Add community Tor bridges for decentralization
  // '/dns4/community1.quillon.xyz/tcp/9444/wss/p2p/<Community-PeerID-1>',
]

/**
 * Legacy clearnet peers (DISABLED - kept for reference only)
 * DO NOT USE - these would leak user IPs
 */
export const CLEARNET_PEERS_DISABLED = [
  // '/dns4/quillon.xyz/tcp/9443/wss/p2p/12D3KooWFrhdwDDTgxPX41mUyRgLcE1ozsBYArKM4DT8t4VLwuNx',
]

/**
 * Gossipsub Topics - Pub/Sub channels for real-time updates
 */
export const TOPICS = {
  BLOCKS: `/qnk/${NETWORK_ID}/blocks`,
  TRANSACTIONS: `/qnk/${NETWORK_ID}/transactions`,
  PEER_HEIGHTS: `/qnk/${NETWORK_ID}/peer-heights`,
  TURBO_SYNC_REQUEST: `/qnk/${NETWORK_ID}/turbo-sync-request`,
  TURBO_SYNC_RESPONSE: `/qnk/${NETWORK_ID}/turbo-sync-response`,
  // v3.5.x: Browser P2P Network Contribution topics
  VERIFICATION_REPORTS: `/qnk/${NETWORK_ID}/verification-reports`,
  TELEMETRY: `/qnk/${NETWORK_ID}/telemetry`,
  // v3.5.8: Browser peer discovery - browsers announce themselves to find each other
  BROWSER_PEERS: `/qnk/${NETWORK_ID}/browser-peers`,
} as const

/**
 * Distributed AI Topics - For browser-based AI compute workers
 * (Phase 1 of js-libp2p distributed AI integration)
 */
export const AI_TOPICS = {
  // Worker nodes announce their capabilities (hardware, uptime, availability)
  NODES_ANNOUNCE: '/qnk/distributed-ai/nodes-announce',

  // Coordinator publishes inference requests to specific workers
  INFERENCE_REQUEST: '/qnk/distributed-ai/inference-request',

  // Workers publish token-by-token inference responses
  INFERENCE_RESPONSE: '/qnk/distributed-ai/inference-response',

  // Coordinator election (highest election_score becomes coordinator)
  COORDINATOR_ELECTION: '/qnk/distributed-ai/coordinator-election'
} as const

/**
 * Custom Request-Response Protocols
 */
export const PROTOCOLS = {
  BLOCK_REQUEST: '/qnk/block-request/1.0.0',
  BALANCE_QUERY: '/qnk/balance-query/1.0.0',
  TX_STATUS: '/qnk/tx-status/1.0.0',
  HANDSHAKE: '/qnk/handshake/1.0.0',
  // v3.5.x: Browser-to-browser block serving
  BLOCK_SERVE: '/qnk/block-serve/1.0.0',
} as const

/**
 * Connection Configuration
 *
 * IMPORTANT: Limits tuned for mobile browser constraints
 * - Mobile browsers have strict connection limits
 * - Too many connections can cause memory issues
 * - Balance between redundancy and resource usage
 */
export const CONNECTION_CONFIG = {
  // Maximum number of peer connections (mobile browser realistic limit)
  MAX_CONNECTIONS: 15,

  // Minimum connections to maintain (redundancy)
  MIN_CONNECTIONS: 3,

  // Auto-dial configuration
  AUTO_DIAL_INTERVAL: 5000, // 5 seconds (faster retry)

  // Connection timeout - v3.5.5: Reduced from 30s to 10s for faster initial connection
  DIAL_TIMEOUT: 10000, // 10 seconds
}

/**
 * DHT Configuration (Light Mode for browsers)
 * Browsers don't store DHT data, only query it
 */
export const DHT_CONFIG = {
  CLIENT_MODE: true, // Don't store DHT data
  K_BUCKET_SIZE: 10, // Reduced from default 20 for memory savings
  QUERY_TIMEOUT: 5000, // v3.5.5: Reduced from 10s to 5s for faster discovery
}

/**
 * Gossipsub Configuration
 * v3.5.22: Optimized for faster mesh recovery and propagation
 */
export const GOSSIPSUB_CONFIG = {
  // Enable flood publishing for critical messages
  FLOOD_PUBLISH: true,

  // Mesh parameters - v3.5.22: Increased D for faster propagation
  D: 8, // Desired number of peers in mesh (was 6)
  D_LOW: 5, // Minimum peers before grafting (was 4)
  D_HIGH: 14, // Maximum peers before pruning (was 12)

  // v3.5.22: Faster heartbeat for quicker mesh recovery
  HEARTBEAT_INTERVAL: 700, // 700ms (was 1000ms) - faster mesh maintenance

  // v3.5.22: Seen message TTL (reduces duplicate processing overhead)
  SEEN_TTL: 60000, // 60 seconds - messages older than this are forgotten
}

/**
 * Security Configuration
 */
export const SECURITY_CONFIG = {
  // Minimum peer score before banning
  MIN_PEER_SCORE: 20,

  // Initial peer score
  INITIAL_PEER_SCORE: 100,

  // Peer score adjustments
  INVALID_MESSAGE_PENALTY: -10,
  VALID_MESSAGE_REWARD: 1,

  // Geographic diversity requirement
  MIN_UNIQUE_REGIONS: 2,
}

/**
 * Performance Configuration
 * v3.5.22: Optimized for faster message delivery
 */
export const PERFORMANCE_CONFIG = {
  // Message batching - v3.5.22: Reduced timeout for faster delivery
  BATCH_SIZE: 10,
  BATCH_TIMEOUT: 50, // 50ms (was 100ms) - faster message delivery

  // Connection pooling
  POOL_SIZE: 5,
  POOL_TIMEOUT: 60000, // 60 seconds

  // v3.5.22: Connection pre-warming settings
  PREWARM_DELAY: 100, // ms - delay before starting pre-warm
  PREWARM_PARALLEL_DIALS: 3, // Number of parallel dial attempts
}

// ============================================================================
// Browser P2P Network Contribution Configuration (v3.5.x)
// ============================================================================

/**
 * Telemetry Configuration
 * Controls how often and what metrics browser nodes report
 */
export const TELEMETRY_CONFIG = {
  // How often to send telemetry reports (ms)
  REPORT_INTERVAL: 5 * 60 * 1000, // 5 minutes

  // Minimum interval between reports (spam prevention)
  MIN_REPORT_INTERVAL: 60 * 1000, // 1 minute

  // Reduce frequency on mobile (battery saving)
  MOBILE_REPORT_INTERVAL: 15 * 60 * 1000, // 15 minutes on mobile

  // Enable telemetry (can be disabled by user)
  ENABLED: true,

  // Track block latency for last N blocks
  LATENCY_SAMPLE_SIZE: 100,
}

/**
 * Block Relay Configuration
 * Controls how browser nodes relay verified blocks to peers
 */
export const RELAY_CONFIG = {
  // Enable block relay
  ENABLED: true,

  // Only relay blocks that pass verification
  REQUIRE_VERIFICATION: true,

  // Minimum verification score to relay (0-8 checks must pass)
  MIN_VERIFICATION_SCORE: 8, // All checks must pass

  // Seen block cache size (deduplication)
  SEEN_CACHE_SIZE: 1000,

  // Maximum relays per second (rate limiting)
  MAX_RELAYS_PER_SECOND: 10,

  // Time window for rate limiting (ms)
  RATE_LIMIT_WINDOW: 1000,
}

/**
 * Block Cache Configuration
 * Controls the browser's local block cache for serving peers
 */
export const BLOCK_CACHE_CONFIG = {
  // Maximum number of blocks to cache
  MAX_SIZE: 100,

  // Estimated max memory usage (100 blocks * ~100KB = ~10MB)
  MAX_MEMORY_MB: 10,

  // How long to keep blocks (ms) - 0 = forever (until evicted by size)
  TTL: 0,

  // Enable serving blocks to peers
  SERVE_ENABLED: true,

  // Maximum blocks per request
  MAX_BLOCKS_PER_REQUEST: 10,

  // Request timeout (ms)
  REQUEST_TIMEOUT: 10000,
}

/**
 * Verification Report Configuration
 * Controls when and how verification failures are reported
 */
export const VERIFICATION_REPORT_CONFIG = {
  // Enable verification failure reporting
  ENABLED: true,

  // Only report critical failures (not warnings)
  CRITICAL_ONLY: false,

  // Minimum number of failed checks to report
  MIN_FAILED_CHECKS: 1,

  // Rate limit reports (max reports per minute)
  MAX_REPORTS_PER_MINUTE: 10,

  // Cooldown period before reporting same block again (ms)
  REPORT_COOLDOWN: 60000, // 1 minute
}
