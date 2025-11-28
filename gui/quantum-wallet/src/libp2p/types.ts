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
