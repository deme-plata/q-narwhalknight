/**
 * libp2p Message Types
 *
 * TypeScript interfaces matching the Rust blockchain types for P2P messaging.
 * These types are used for encoding/decoding messages over PubSub.
 *
 * Rust uses postcard serialization (compact binary format).
 * TypeScript needs to decode these messages for real-time updates.
 */

/**
 * Block Hash (blake3, 32 bytes)
 */
export type BlockHash = Uint8Array

/**
 * DAG Round Number
 */
export type DagRound = number

/**
 * Node ID (peer identifier)
 */
export type NodeId = string

/**
 * VDF Proof for anchor election
 */
export interface VDFProof {
  input: Uint8Array
  output: Uint8Array
  proof: Uint8Array
  iterations: number
}

/**
 * Quantum Metadata for consensus
 */
export interface QuantumMetadata {
  coherence: number
  entanglement: number
  measurement: number
}

/**
 * Mining Solution included in block
 */
export interface MiningSolution {
  nonce: bigint
  difficulty: number
  hash: BlockHash
  miner: string
  reward: number
}

/**
 * Balance Update (deterministic balance state)
 */
export interface BalanceUpdate {
  address: string
  oldBalance: number
  newBalance: number
  reason: string
}

/**
 * Transaction
 */
export interface Transaction {
  from: string
  to: string
  amount: number
  timestamp: number
  signature?: Uint8Array
  nonce?: number
}

/**
 * Block Header
 */
export interface BlockHeader {
  // Block height (monotonically increasing)
  height: number

  // Network phase identifier (prevents cross-phase contamination)
  phase: number

  // Network ID
  networkId: string

  // Previous block hash (chain backbone)
  prevBlockHash: BlockHash

  // Merkle root of mining solutions
  solutionsRoot: BlockHash

  // Merkle root of transactions
  txRoot: BlockHash

  // State root (world state after this block)
  stateRoot: BlockHash

  // Block creation timestamp (Unix epoch seconds)
  timestamp: number

  // DAG round number
  dagRound: DagRound

  // VDF proof for anchor election
  vdfProof: VDFProof

  // Anchor validator elected for this round
  anchorValidator?: string

  // Block proposer
  proposer: NodeId

  // Producer ID / Lane ID (0-7 for parallel production)
  producerId: number

  // Total difficulty accumulated to this block
  totalDifficulty: bigint

  // Producer Ed25519 signature (64 bytes) - for block verification
  producerSignature?: Uint8Array

  // Producer public key (32 bytes) - for signature verification
  producerPublicKey?: Uint8Array
}

/**
 * Complete Q-NarwhalKnight Block
 */
export interface QBlock {
  // Block header
  header: BlockHeader

  // Mining proof-of-work solutions
  miningSolutions: MiningSolution[]

  // DAG vertex parent references
  dagParents: string[]

  // Quantum consensus metadata
  quantumMetadata: QuantumMetadata

  // Transactions included in this block
  transactions: Transaction[]

  // Balance updates (deterministic balance state)
  balanceUpdates: BalanceUpdate[]

  // Block size in bytes
  sizeBytes: number
}

/**
 * Simplified Block for UI Display
 * (subset of QBlock with only essential fields)
 */
export interface BlockSummary {
  height: number
  hash: string
  timestamp: number
  transactionCount: number
  miningReward: number
  proposer: string
  phase: number
  networkId: string
}

/**
 * PubSub Message Wrapper
 *
 * All gossipsub messages are wrapped in this envelope
 * to provide metadata about the message type and routing.
 */
export interface PubSubMessage<T> {
  type: string
  data: T
  timestamp: number
  sender?: string
}

/**
 * Block Message (published to /qnk/testnet-phase12/blocks)
 */
export type BlockMessage = PubSubMessage<QBlock>

/**
 * Transaction Message (published to /qnk/testnet-phase12/transactions)
 */
export type TransactionMessage = PubSubMessage<Transaction>

/**
 * Peer Height Announcement (published to /qnk/testnet-phase12/peer-heights)
 */
export interface PeerHeightAnnouncement {
  peerId: string
  height: number
  bestBlockHash: BlockHash
  timestamp: number
}

/**
 * Turbo Sync Request
 */
export interface TurboSyncRequest {
  requesterId: string
  startHeight: number
  endHeight: number
  timestamp: number
}

/**
 * Turbo Sync Response
 */
export interface TurboSyncResponse {
  responderId: string
  blocks: QBlock[]
  startHeight: number
  endHeight: number
  timestamp: number
}

/**
 * Verification Result for Light Client Validation
 */
export interface VerificationResult {
  // Overall validity
  valid: boolean

  // Individual verification checks performed
  checks: VerificationCheck[]

  // Human-readable summary
  summary: string

  // Time taken to verify (ms)
  verificationTimeMs?: number
}

/**
 * Individual Verification Check
 */
export interface VerificationCheck {
  // Check name (e.g., "Block Structure", "Signature Valid")
  name: string

  // Whether this check passed
  passed: boolean

  // Details about the check result
  details: string
}

/**
 * Verified Block - QBlock with verification result attached
 */
export interface VerifiedBlock extends QBlock {
  // Verification result (undefined if not yet verified)
  verification?: VerificationResult
}

// ============================================================================
// Browser P2P Network Contribution Types (v3.5.x)
// ============================================================================

/**
 * Signed Transaction for P2P submission
 * Browser nodes submit transactions directly via gossipsub instead of HTTP
 *
 * v3.7.4: Now includes optional Dilithium5 post-quantum signatures for
 * quantum-resistant transaction authentication.
 * v3.7.5: Now includes optional ZK-STARK proof commitment for privacy-preserving
 * transaction verification.
 */
export interface SignedTransaction {
  // Sender address (hex string)
  from: string

  // Recipient address (hex string)
  to: string

  // Amount to transfer (in smallest unit)
  amount: bigint

  // Transaction nonce (prevents replay)
  nonce: number

  // Unix timestamp (seconds)
  timestamp: number

  // Ed25519 signature (64 bytes) - classical
  signature: Uint8Array

  // Public key of sender (32 bytes) - Ed25519 for verification
  publicKey: Uint8Array

  // v3.7.4: Dilithium5 post-quantum signature (4,627 bytes) - optional
  dilithium5Signature?: Uint8Array

  // v3.7.4: Dilithium5 public key (2,592 bytes) - optional, for PQ verification
  dilithium5PublicKey?: Uint8Array

  // v3.7.4: Signature mode - indicates what signatures are included
  signatureMode?: 'ed25519' | 'dilithium5' | 'hybrid'

  // v3.7.5: ZK-STARK proof commitment (201 bytes) - optional, for privacy-preserving P2P
  starkProofCommitment?: Uint8Array

  // v3.7.5: STARK proof version
  starkProofVersion?: number

  // Optional: token address for token transfers (empty for native)
  tokenAddress?: string

  // Optional: memo/data field
  memo?: string
}

/**
 * Verification Report - Report blocks that fail verification
 * Helps network detect bad actors and invalid blocks
 */
export interface VerificationReport {
  // Browser peer ID that performed verification
  reporterId: string

  // Block being reported
  blockHeight: number
  blockHash: string

  // Which verification checks failed
  failedChecks: string[]

  // Additional details about failures
  failureDetails: string[]

  // Unix timestamp (ms)
  timestamp: number

  // Severity level
  severity: 'warning' | 'critical'

  // Reporter's node type
  nodeType: 'browser' | 'full'

  // Protocol version
  protocolVersion: string
}

/**
 * Telemetry Report - Network health metrics from browser nodes
 * Helps nodes understand overall network health
 */
export interface TelemetryReport {
  // Peer ID of the reporting node
  peerId: string

  // Node type
  nodeType: 'browser'

  // Number of connected peers
  connectedPeers: number

  // Number of blocks received in last 24h
  blocksReceived24h: number

  // Average block latency (ms from production to receipt)
  avgBlockLatencyMs: number

  // Verification rate (percentage of blocks verified, 0-100)
  verificationRate: number

  // How long this node has been online (seconds)
  uptime: number

  // Current block height seen
  currentHeight: number

  // Verification stats
  blocksVerified: number
  blocksValid: number
  blocksInvalid: number

  // Network ID
  networkId: string

  // Unix timestamp (ms)
  timestamp: number

  // Protocol version
  protocolVersion: string
}

/**
 * Block Request - Request blocks from peer block cache
 * Used for browser-to-browser block serving
 */
export interface BlockRequest {
  // Requesting peer ID
  requesterId: string

  // Start height (inclusive)
  startHeight: number

  // End height (inclusive)
  endHeight: number

  // Unix timestamp (ms)
  timestamp: number
}

/**
 * Block Response - Response to block request
 */
export interface BlockResponse {
  // Responding peer ID
  responderId: string

  // Requested range
  startHeight: number
  endHeight: number

  // Blocks in the requested range (may be subset if some missing)
  blocks: QBlock[]

  // Total blocks available in cache
  cacheSize: number

  // Unix timestamp (ms)
  timestamp: number
}

/**
 * Relay Stats - Track block relay activity
 */
export interface RelayStats {
  // Total blocks relayed
  blocksRelayed: number

  // Blocks relayed in last hour
  blocksRelayedLastHour: number

  // Duplicate blocks filtered
  duplicatesFiltered: number

  // Failed verification blocks not relayed
  invalidBlocksFiltered: number

  // Rate limit drops
  rateLimitDrops: number

  // Last relay timestamp
  lastRelayTime: number
}

/**
 * Block Cache Stats - Statistics about the local block cache
 */
export interface BlockCacheStats {
  // Number of blocks in cache
  size: number

  // Maximum cache size
  maxSize: number

  // Lowest block height in cache
  lowestHeight: number

  // Highest block height in cache
  highestHeight: number

  // Total memory used (estimated bytes)
  memoryUsed: number

  // Cache hit rate (percentage)
  hitRate: number

  // Total requests served
  requestsServed: number
}

// ============================================================================
// Post-Quantum Cryptography Types (v3.7.4)
// ============================================================================

/**
 * Post-Quantum Signed Transaction
 * Includes both classical Ed25519 and post-quantum Dilithium5 signatures
 */
export interface PQSignedTransaction {
  // Sender address (hex string)
  from: string

  // Recipient address (hex string)
  to: string

  // Amount to transfer (in smallest unit)
  amount: bigint

  // Transaction nonce (prevents replay)
  nonce: number

  // Unix timestamp (seconds)
  timestamp: number

  // Ed25519 signature (64 bytes) - classical
  ed25519Signature: Uint8Array

  // Dilithium5 signature (4,627 bytes) - post-quantum
  dilithium5Signature: Uint8Array

  // Ed25519 public key of sender (32 bytes)
  ed25519PublicKey: Uint8Array

  // Dilithium5 public key of sender (2,592 bytes)
  dilithium5PublicKey: Uint8Array

  // Optional: token address for token transfers
  tokenAddress?: string

  // Optional: memo/data field
  memo?: string
}

/**
 * Post-Quantum Peer Identity
 * Exchanged during PQ handshake for quantum-resistant authentication
 */
export interface PQPeerIdentity {
  // Peer ID (libp2p peer ID string)
  peerId: string

  // Ed25519 public key (32 bytes) - classical
  ed25519PublicKey: Uint8Array

  // Dilithium5 public key (2,592 bytes) - post-quantum signatures
  dilithium5PublicKey: Uint8Array

  // Kyber1024 public key (1,568 bytes) - post-quantum key exchange
  kyber1024PublicKey: Uint8Array

  // Combined fingerprint (SHA3-256 of all public keys)
  fingerprint: Uint8Array

  // Protocol version
  protocolVersion: string

  // Timestamp of identity creation
  createdAt: number
}

/**
 * Post-Quantum Handshake Message
 * Sent at connection establishment for quantum-resistant session
 */
export interface PQHandshakeMessage {
  // Message type
  type: 'hello' | 'kyber-encapsulation' | 'auth-signature' | 'complete'

  // Protocol version
  version: string

  // Sender's PQ identity
  identity?: PQPeerIdentity

  // Kyber1024 ciphertext (1,568 bytes) - for key exchange
  kyberCiphertext?: Uint8Array

  // Hybrid signature (Ed25519 + Dilithium5)
  signature?: {
    ed25519: Uint8Array
    dilithium5: Uint8Array
    timestamp: number
  }

  // Random nonce for freshness
  nonce: Uint8Array

  // Timestamp
  timestamp: number

  // Success flag (for complete message)
  success?: boolean
}

/**
 * Post-Quantum Session State
 * Represents an established quantum-resistant session
 */
export interface PQSession {
  // Session ID (SHA3-256 hash)
  sessionId: Uint8Array

  // Remote peer's identity
  remotePeer: PQPeerIdentity

  // Shared secret derived from Kyber1024 (32 bytes)
  sharedSecret: Uint8Array

  // Session established timestamp
  establishedAt: number

  // Session expiry (for forward secrecy, typically 1 hour)
  expiresAt: number

  // Whether this is a hybrid session (classical + PQ)
  isHybrid: boolean

  // Connection quality metrics
  metrics: {
    handshakeLatencyMs: number
    lastActivityAt: number
    messagesExchanged: number
  }
}

/**
 * Post-Quantum Crypto Status
 * Returned by debug utilities
 */
export interface PQCryptoStatus {
  // Whether PQ crypto is loaded
  loaded: boolean

  // Crypto implementation type ('wasm', 'pure-js', 'simulated')
  type: string

  // Whether a keypair has been generated
  keypairGenerated: boolean

  // Keypair fingerprint (hex, first 16 chars)
  fingerprint: string | null

  // Algorithm constants
  constants: {
    dilithium5PublicKeyBytes: number
    dilithium5SecretKeyBytes: number
    dilithium5SignatureBytes: number
    kyber1024PublicKeyBytes: number
    kyber1024SecretKeyBytes: number
    kyber1024CiphertextBytes: number
    kyber1024SharedSecretBytes: number
  }
}

// ============================================================================
// ZK-STARK Proof Types (v3.7.5)
// ============================================================================

/**
 * ZK-STARK Proof Commitment for P2P Transactions (v3.8.0)
 * Production-ready commitment with Goldilocks field, FRI protocol, and PoW grinding
 *
 * Security Properties:
 * - 100-128 bit security (configurable)
 * - Transparent setup (no trusted setup)
 * - Post-quantum secure (hash-based)
 *
 * The commitment structure includes:
 * - Merkle commitment: Hash tree of execution trace (domain-separated)
 * - Polynomial commitment: Low-degree extension over Goldilocks field
 * - FRI commitment: Fast Reed-Solomon proximity proof with folding
 * - PoW nonce: Grinding resistance against proof spamming
 */
export interface StarkProofCommitment {
  // Protocol version (currently 2)
  version: number

  // Merkle root of execution trace (32 bytes, Blake3)
  merkleRoot: Uint8Array

  // Composition polynomial commitment (32 bytes)
  polyCommitment: Uint8Array

  // FRI final layer commitment (32 bytes)
  friCommitment: Uint8Array

  // Fiat-Shamir challenge seed (32 bytes)
  challengeSeed: Uint8Array

  // Proof generation timestamp
  timestamp: number

  // Transaction binding hash (32 bytes, SHA3-256)
  txBinding: Uint8Array

  // Prover's public key (32 bytes)
  proverPubKey: Uint8Array

  // Proof-of-work nonce for grinding resistance
  powNonce: number

  // AIR trace length (power of 2)
  traceLength: number

  // Number of FRI layers
  friLayers: number
}

/**
 * Full STARK Proof (expanded for validator verification)
 * Contains all data needed for complete verification on full nodes
 */
export interface FullStarkProof extends StarkProofCommitment {
  // Merkle authentication paths for trace queries
  authPaths: Uint8Array[]

  // FRI query responses (values and Merkle paths per layer)
  friResponses: Uint8Array[]

  // Polynomial evaluations at query points
  polyEvaluations: Uint8Array[]

  // Verification status (set by validator after verification)
  verified?: boolean

  // Proving time in milliseconds
  provingTimeMs?: number
}

/**
 * STARK Proof Generation Result
 */
export interface StarkProofResult {
  // Whether proof generation succeeded
  success: boolean

  // The lightweight commitment (for P2P)
  commitment?: StarkProofCommitment

  // The full proof (for validator)
  fullProof?: FullStarkProof

  // Time taken to generate proof (ms)
  provingTimeMs: number

  // Error message if generation failed
  error?: string
}

/**
 * STARK Verification Result
 */
export interface StarkVerificationResult {
  // Whether verification succeeded
  valid: boolean

  // Commitment binding valid
  bindingValid: boolean

  // Merkle proof valid
  merkleValid: boolean

  // FRI proof valid
  friValid: boolean

  // Verification time (ms)
  verificationTimeMs: number

  // Error details if verification failed
  error?: string
}
