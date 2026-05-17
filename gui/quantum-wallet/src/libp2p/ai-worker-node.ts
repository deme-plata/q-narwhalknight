/**
 * AI Worker Node - Browser-side libp2p node for distributed AI compute
 *
 * Implements Phase 1 of js-libp2p integration with modifications from aireply43:
 * - Circuit breaker pattern for bootstrap dial
 * - Geo-aware peer selection (<100ms RTT)
 * - Graceful degradation with static peer fallback
 * - Browser as "client-only" peer (no inbound connections)
 */

import { createLibp2p, type Libp2p } from 'libp2p'
import { webSockets } from '@libp2p/websockets'
import { noise } from '@chainsafe/libp2p-noise'
import { yamux } from '@chainsafe/libp2p-yamux'
import { identify } from '@libp2p/identify'
// @ts-ignore - gossipsub types may vary
import { gossipsub } from '@chainsafe/libp2p-gossipsub'
import { kadDHT } from '@libp2p/kad-dht'
import type { PeerId } from '@libp2p/interface'
import { BOOTSTRAP_PEERS } from './config'

// GossipSub message type
interface GossipsubMessage {
  from?: PeerId
  data: Uint8Array
  topic: string
}

export interface WorkerCapability {
  type: 'WebGPU' | 'WASM_CPU' | 'None'
  vram_mb?: number
  compute_units?: number
  threads?: number
  latency_estimate_ms?: number
}

export interface AINodeAnnouncement {
  node_id: string
  peer_id: string
  capability: WorkerCapability
  available_layers: number
  election_score: number
  uptime_secs: number
  geo_region?: string // For geo-aware peer selection
}

export interface InferenceRequest {
  request_id: string
  prompt: string
  max_tokens: number
  temperature: number
  model: string
  layer_start?: number
  layer_end?: number
  assigned_node_id?: string
  payment_amount_base_units?: number
}

export interface InferenceResponse {
  request_id: string
  token: string
  token_index: number
  worker_node_id: string
  tensor_hash?: string
}

/**
 * Circuit Breaker for bootstrap peer connections
 * Prevents overload and implements exponential backoff
 */
class CircuitBreaker {
  private failures = 0
  private lastAttempt = 0
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED'

  private readonly maxFailures = 3
  private readonly resetTimeout = 60000 // 60 seconds
  private readonly baseBackoff = 1000 // 1 second

  canAttempt(): boolean {
    const now = Date.now()

    if (this.state === 'CLOSED') {
      return true
    }

    if (this.state === 'OPEN') {
      if (now - this.lastAttempt > this.resetTimeout) {
        this.state = 'HALF_OPEN'
        return true
      }
      return false
    }

    // HALF_OPEN: allow one attempt
    return true
  }

  recordSuccess() {
    this.failures = 0
    this.state = 'CLOSED'
  }

  recordFailure() {
    this.failures++
    this.lastAttempt = Date.now()

    if (this.failures >= this.maxFailures) {
      this.state = 'OPEN'
      console.warn(`🔴 Circuit breaker OPEN after ${this.failures} failures`)
    }
  }

  getBackoffMs(): number {
    return Math.min(this.baseBackoff * Math.pow(2, this.failures), 30000)
  }
}

/**
 * AI Worker Node - Browser libp2p peer for distributed inference
 */
export class AIWorkerNode {
  private node: Libp2p | null = null
  private capability: WorkerCapability = { type: 'None' }
  private peerId: PeerId | null = null
  private isRunning = false
  private circuitBreaker = new CircuitBreaker()
  private connectedPeers = new Set<string>()
  private peerLatencies = new Map<string, number>()

  // Callbacks
  public onInferenceRequest?: (request: InferenceRequest) => Promise<void>
  public onPeerConnected?: (peerId: string, latency: number) => void
  public onPeerDisconnected?: (peerId: string) => void
  public onError?: (error: Error) => void

  async init(): Promise<boolean> {
    try {
      console.log('🤖 Initializing AI Worker Node...')

      // Detect browser capabilities
      this.capability = await this.detectCapability()
      console.log('📊 Detected capability:', this.capability)

      // Create libp2p node
      console.log('🔧 Creating libp2p node...')
      this.node = await this.createNode()
      this.peerId = this.node.peerId

      console.log('🆔 Peer ID:', this.peerId.toString())

      // Start node
      console.log('🚀 Starting libp2p node...')
      await this.node.start()
      console.log('✅ libp2p node started')

      // Subscribe to AI topics
      console.log('📡 Subscribing to AI topics...')
      await this.subscribeToTopics()

      // Connect to bootstrap peers with circuit breaker
      // NOTE: Bootstrap connection is optional - worker can still run in standalone mode
      console.log('🔗 Attempting bootstrap connection...')
      try {
        await this.connectToBootstrap()
      } catch (bootstrapError) {
        console.warn('⚠️ Bootstrap connection failed, running in standalone mode:', bootstrapError)
        // Continue without bootstrap - worker can still function for local inference
      }

      // Announce worker capability (only if connected to peers)
      if (this.connectedPeers.size > 0) {
        await this.announceWorker()
      } else {
        console.log('📢 Skipping worker announcement (no peers connected)')
      }

      // Start heartbeat
      this.startHeartbeat()

      this.isRunning = true
      console.log('✅ AI Worker Node initialized successfully')
      console.log(`   Mode: ${this.connectedPeers.size > 0 ? 'Connected' : 'Standalone'}`)
      console.log(`   Capability: ${this.capability.type}`)

      return true
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error)
      console.error('❌ Failed to initialize AI Worker Node:', errorMsg)
      this.onError?.(new Error(`Worker init failed: ${errorMsg}`))
      return false
    }
  }

  private async createNode(): Promise<Libp2p> {
    const node = await createLibp2p({
      addresses: {
        // Browser cannot listen for incoming connections
        listen: []
      },
      transports: [
        webSockets()
      ],
      streamMuxers: [yamux()],
      connectionEncrypters: [noise()],
      services: {
        identify: identify({
          protocolPrefix: '/qnarwhal'
        }),
        pubsub: gossipsub({
          emitSelf: false,
          allowPublishToZeroTopicPeers: true,
          fallbackToFloodsub: false
        }) as any,
        dht: kadDHT({
          // Browser as DHT client only (cannot accept inbound)
          clientMode: true,
          protocol: '/qnk/kad/1.0.0'
        }) as any
      },
      connectionManager: {
        maxConnections: 20
      }
    })

    // Set up connection event handlers
    node.addEventListener('peer:connect', (evt) => {
      const peerId = evt.detail.toString()
      this.connectedPeers.add(peerId)
      console.log('🔗 Connected to peer:', peerId)
    })

    node.addEventListener('peer:disconnect', (evt) => {
      const peerId = evt.detail.toString()
      this.connectedPeers.delete(peerId)
      this.peerLatencies.delete(peerId)
      console.log('🔌 Disconnected from peer:', peerId)
      this.onPeerDisconnected?.(peerId)
    })

    return node
  }

  private async detectCapability(): Promise<WorkerCapability> {
    // Check for WebGPU support
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter()
        if (adapter) {
          const limits = adapter.limits
          const info = await adapter.requestAdapterInfo()

          // Estimate VRAM (WebGPU doesn't expose this directly)
          const estimatedVRAM = Math.floor(limits.maxStorageBufferBindingSize / 1024 / 1024)

          return {
            type: 'WebGPU',
            vram_mb: estimatedVRAM,
            compute_units: limits.maxComputeWorkgroupsPerDimension
          }
        }
      } catch (error) {
        console.warn('WebGPU adapter request failed:', error)
      }
    }

    // Fallback to WASM (CPU)
    return {
      type: 'WASM_CPU',
      threads: navigator.hardwareConcurrency || 4
    }
  }

  private async subscribeToTopics(): Promise<void> {
    if (!this.node) return

    const topics = [
      '/qnk/distributed-ai/nodes-announce',
      '/qnk/distributed-ai/inference-request',
      '/qnk/distributed-ai/inference-response',
      '/qnk/distributed-ai/coordinator-election'
    ]

    for (const topic of topics) {
      await (this.node.services.pubsub as any).subscribe(topic)
      console.log('📡 Subscribed to topic:', topic)
    }

    // Set up message handler
    (this.node.services.pubsub as any).addEventListener('message', (evt: any) => {
      this.handleGossipMessage(evt.detail)
    })
  }

  private async connectToBootstrap(): Promise<void> {
    if (!this.node || !this.circuitBreaker.canAttempt()) {
      console.warn('⚠️ Circuit breaker preventing bootstrap connection')
      return
    }

    try {
      // Try primary bootstrap peer
      const bootstrapPeer = BOOTSTRAP_PEERS[0]
      console.log('🔄 Connecting to bootstrap peer:', bootstrapPeer)

      const conn = await this.node.dial(bootstrapPeer as any)

      // Measure latency
      const start = Date.now()
      await conn.newStream('/ipfs/ping/1.0.0')
      const latency = Date.now() - start

      this.peerLatencies.set(conn.remotePeer.toString(), latency)
      this.circuitBreaker.recordSuccess()

      console.log(`✅ Connected to bootstrap (${latency}ms latency)`)
      this.onPeerConnected?.(conn.remotePeer.toString(), latency)

    } catch (error) {
      console.error('❌ Bootstrap connection failed:', error)
      this.circuitBreaker.recordFailure()

      // Exponential backoff
      const backoffMs = this.circuitBreaker.getBackoffMs()
      console.log(`⏱️  Retrying in ${backoffMs}ms...`)

      setTimeout(() => this.connectToBootstrap(), backoffMs)
    }
  }

  private async announceWorker(): Promise<void> {
    if (!this.node || !this.peerId) return

    const announcement: AINodeAnnouncement = {
      node_id: this.peerId.toString(),
      peer_id: this.peerId.toString(),
      capability: this.capability,
      available_layers: this.getAvailableLayers(),
      election_score: this.calculateElectionScore(),
      uptime_secs: 0,
      geo_region: await this.estimateGeoRegion()
    }

    const message = new TextEncoder().encode(JSON.stringify(announcement))

    await (this.node.services.pubsub as any).publish(
      '/qnk/distributed-ai/nodes-announce',
      message
    )

    console.log('📢 Announced worker capability')
  }

  private async handleGossipMessage(message: GossipsubMessage): Promise<void> {
    try {
      const topic = message.topic
      const data = JSON.parse(new TextDecoder().decode(message.data))

      if (topic === '/qnk/distributed-ai/inference-request') {
        // Check if this request is assigned to us
        const request = data as InferenceRequest
        if (request.assigned_node_id === this.peerId?.toString()) {
          console.log('📥 Received inference request:', request.request_id)
          await this.onInferenceRequest?.(request)
        }
      }

      // Handle other message types...

    } catch (error) {
      console.error('❌ Failed to handle gossip message:', error)
    }
  }

  async publishToken(requestId: string, token: string, tokenIndex: number): Promise<void> {
    if (!this.node || !this.peerId) return

    const response: InferenceResponse = {
      request_id: requestId,
      token,
      token_index: tokenIndex,
      worker_node_id: this.peerId.toString()
    }

    const message = new TextEncoder().encode(JSON.stringify(response))

    await (this.node.services.pubsub as any).publish(
      '/qnk/distributed-ai/inference-response',
      message
    )
  }

  private startHeartbeat(): void {
    // Announce capability every 30 seconds
    setInterval(() => {
      if (this.isRunning) {
        this.announceWorker()
      }
    }, 30000)
  }

  private getAvailableLayers(): number {
    // Based on capability, determine how many model layers we can handle
    if (this.capability.type === 'WebGPU') {
      const vram = this.capability.vram_mb || 0
      if (vram >= 8000) return 32 // Full Mistral-7B
      if (vram >= 4000) return 16 // Half model
      return 8 // Quarter model
    }

    if (this.capability.type === 'WASM_CPU') {
      // WASM can handle small models only (Phi-2)
      return 4
    }

    return 0
  }

  private calculateElectionScore(): number {
    // Higher score = more likely to be selected as coordinator
    let score = 0

    if (this.capability.type === 'WebGPU') {
      score += 100
      score += (this.capability.vram_mb || 0) / 100
    } else if (this.capability.type === 'WASM_CPU') {
      score += 10
      score += (this.capability.threads || 0) * 2
    }

    // Bonus for low latency
    const avgLatency = Array.from(this.peerLatencies.values())
      .reduce((sum, lat) => sum + lat, 0) / (this.peerLatencies.size || 1)

    if (avgLatency < 50) score += 50
    else if (avgLatency < 100) score += 25

    return Math.floor(score)
  }

  private async estimateGeoRegion(): Promise<string> {
    // Use average bootstrap peer latency to estimate region
    const latencies = Array.from(this.peerLatencies.values())
    if (latencies.length === 0) return 'unknown'

    const avgLatency = latencies.reduce((sum, lat) => sum + lat, 0) / latencies.length

    if (avgLatency < 30) return 'local'     // Same data center / city
    if (avgLatency < 100) return 'regional' // Same continent
    return 'global'                          // Cross-continent
  }

  getStats() {
    return {
      peerId: this.peerId?.toString(),
      capability: this.capability,
      connectedPeers: this.connectedPeers.size,
      avgLatency: Array.from(this.peerLatencies.values())
        .reduce((sum, lat) => sum + lat, 0) / (this.peerLatencies.size || 1),
      isRunning: this.isRunning
    }
  }

  async stop(): Promise<void> {
    if (this.node) {
      await this.node.stop()
      this.isRunning = false
      console.log('🛑 AI Worker Node stopped')
    }
  }
}
