/**
 * libp2p Configuration for Q-NarwhalKnight Browser Nodes
 *
 * This module contains all configuration for the browser's P2P node,
 * including bootstrap peers, network IDs, and protocol versions.
 */

// Network configuration
export const NETWORK_ID = 'testnet-phase12'
export const PROTOCOL_VERSION = '1.0.20'

/**
 * Bootstrap Peers - Multiple diverse nodes for redundancy and eclipse attack prevention
 *
 * Geographic Distribution:
 * - Server Beta (EU): Production bootstrap node
 * - Server Alpha (US): Secondary bootstrap (TODO: add when available)
 * - Community nodes: Decentralized community-run bootstrap nodes
 *
 * 🌐 v1.0.21-browser: Updated to use HTTPS (443) with /p2p path
 * Browser connects via: wss://quillon.xyz:443/p2p
 * Nginx proxies to: ws://127.0.0.1:9001 (libp2p WebSocket listener)
 */
export const BOOTSTRAP_PEERS = [
  // Server Beta (EU) - Primary Bootstrap with WebSocket over HTTPS
  // 🔥 Port 9443: Dedicated WebSocket server (HTTP/1.1 only - HTTP/2 doesn't support WS)
  // ✅ Updated PeerID: 12D3KooWAixQBjCdjZiDvTeeXsz11F211RLVfpPNZMSAEpUX2zMS (2025-11-21 v1.0.21-browser-p2p)
  '/dns4/quillon.xyz/tcp/9443/wss/p2p/12D3KooWAixQBjCdjZiDvTeeXsz11F211RLVfpPNZMSAEpUX2zMS',

  // TODO: Add Server Alpha (US) when WebSocket support is enabled
  // '/dns4/alpha.quillon.xyz/tcp/9443/wss/p2p/<Server-Alpha-PeerID>',

  // TODO: Add community bootstrap nodes for decentralization
  // '/dns4/community1.quillon.xyz/tcp/9443/wss/p2p/<Community-PeerID-1>',
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
  AUTO_DIAL_INTERVAL: 10000, // 10 seconds

  // Connection timeout
  DIAL_TIMEOUT: 30000, // 30 seconds
}

/**
 * DHT Configuration (Light Mode for browsers)
 * Browsers don't store DHT data, only query it
 */
export const DHT_CONFIG = {
  CLIENT_MODE: true, // Don't store DHT data
  K_BUCKET_SIZE: 10, // Reduced from default 20 for memory savings
  QUERY_TIMEOUT: 10000, // 10 seconds
}

/**
 * Gossipsub Configuration
 */
export const GOSSIPSUB_CONFIG = {
  // Enable flood publishing for critical messages
  FLOOD_PUBLISH: true,

  // Mesh parameters (conservative for browsers)
  D: 6, // Desired number of peers in mesh
  D_LOW: 4, // Minimum peers before grafting
  D_HIGH: 12, // Maximum peers before pruning

  // Heartbeat interval
  HEARTBEAT_INTERVAL: 1000, // 1 second
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
 */
export const PERFORMANCE_CONFIG = {
  // Message batching
  BATCH_SIZE: 10,
  BATCH_TIMEOUT: 100, // ms

  // Connection pooling
  POOL_SIZE: 5,
  POOL_TIMEOUT: 60000, // 60 seconds
}
