import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { motion, AnimatePresence, useAnimation } from 'framer-motion';
import {
  X, Shield, Wifi, WifiOff, Globe, Lock, Zap, Server, Activity, Eye, EyeOff,
  ArrowRight, Radio, Layers, RefreshCw, Database, Box, FileText, Users, Clock,
  Search, Filter, ArrowUpDown, ArrowUp, ArrowDown, ChevronLeft, ChevronRight,
  Download, Hash, CheckCircle, AlertCircle, ShieldCheck
} from 'lucide-react';
import { useLibP2P } from '../contexts/LibP2PContext';
import { useRealtimeBlocks } from '../hooks/useRealtimeBlocks';
import { BOOTSTRAP_PEERS } from '../libp2p/config';
import { DECODER_METRICS } from '../libp2p/decoder';
import { getKnownBrowserPeers, getBrowserPeerCount, type KnownBrowserPeer } from '../libp2p/browserPeerDiscovery';
import { getTransactionStats, type TransactionStats } from '../libp2p/transactionSubmitter';
import { getPQCryptoStatus, isPQCryptoAvailable } from '../libp2p/postQuantumCrypto';
import { getStarkConfig } from '../libp2p/zkStarkProof';
import type { QBlock, Transaction, VerifiedBlock } from '../libp2p/types';

// Local type for PQ crypto status (matches actual return type of getPQCryptoStatus)
interface LocalPQStatus {
  loaded: boolean;
  type: string;
  keypairGenerated: boolean;
  fingerprint: string | null;
  constants: {
    DILITHIUM5_PUBLIC_KEY_BYTES: number;
    DILITHIUM5_SECRET_KEY_BYTES: number;
    DILITHIUM5_SIGNATURE_BYTES: number;
    KYBER1024_PUBLIC_KEY_BYTES: number;
    KYBER1024_SECRET_KEY_BYTES: number;
    KYBER1024_CIPHERTEXT_BYTES: number;
    KYBER1024_SHARED_SECRET_BYTES: number;
  };
}
import { BrowserResonanceVisualization } from './BrowserResonanceVisualization';

interface PeerInfo {
  id: string;
  shortId: string;
  address: string;
  latency: number;
  status: string;
  direction: 'inbound' | 'outbound';
  isTor: boolean;
  isBootstrap: boolean;
  isBrowser: boolean;  // v3.5.x: Browser peer detection (WebRTC/WebSocket)
  connectedAt: number;
  protocols: string[];
}

interface NetworkMapModalProps {
  isOpen: boolean;
  onClose: () => void;
  peers: number;
  blockHeight: number;
}

type TabType = 'blocks' | 'transactions' | 'peers' | 'miners' | 'metrics';
type SortDirection = 'asc' | 'desc';

interface SortConfig {
  key: string;
  direction: SortDirection;
}

// Miner statistics from recent blocks
interface MinerStats {
  address: string;
  shortAddress: string;
  solutionCount: number;
  totalRewards: number;
  lastActive: number;
  avgDifficulty: number;
}

// Format timestamp
const formatTime = (timestamp: number) => {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  });
};

// Format large numbers
const formatNumber = (num: number) => {
  if (num >= 1000000) return `${(num / 1000000).toFixed(2)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(2)}K`;
  return num.toString();
};

// QNK has 9 decimal places - convert from atomic units to display units
const QNK_DECIMALS = 1e9;
const formatQNK = (atomicUnits: number): string => {
  const qnk = atomicUnits / QNK_DECIMALS;
  if (qnk >= 1000000) return `${(qnk / 1000000).toFixed(2)}M`;
  if (qnk >= 1000) return `${(qnk / 1000).toFixed(2)}K`;
  if (qnk >= 1) return qnk.toFixed(2);
  return qnk.toFixed(6);
};

// Truncate hash
const truncateHash = (hash: string, length = 8) => {
  if (!hash || hash.length <= length * 2) return hash;
  return `${hash.substring(0, length)}...${hash.substring(hash.length - length)}`;
};

// Convert bytes to hex string
const bytesToHex = (bytes: Uint8Array): string => {
  return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
};

// Animated connection line between nodes
const ConnectionLine = ({ from, to, isActive, isTor, direction }: {
  from: { x: number; y: number };
  to: { x: number; y: number };
  isActive: boolean;
  isTor: boolean;
  direction: 'inbound' | 'outbound';
}) => {
  const midX = (from.x + to.x) / 2;
  const midY = (from.y + to.y) / 2 - 20;
  const path = `M ${from.x} ${from.y} Q ${midX} ${midY} ${to.x} ${to.y}`;

  return (
    <g>
      <motion.path
        d={path}
        stroke={isTor ? "rgba(168, 85, 247, 0.4)" : "rgba(212, 175, 55, 0.4)"}
        strokeWidth="2"
        fill="none"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1, ease: "easeInOut" }}
      />
      {isActive && (
        <motion.circle
          r="3"
          fill={isTor ? "#a855f7" : "#d4af37"}
          filter="url(#glow)"
          initial={{ offsetDistance: "0%" }}
          animate={{ offsetDistance: direction === 'outbound' ? "100%" : "0%" }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "linear",
            repeatDelay: Math.random() * 0.5
          }}
          style={{ offsetPath: `path('${path}')` }}
        />
      )}
    </g>
  );
};

// Animated peer node
const PeerNode = ({ peer, position, isSelected, onClick, index }: {
  peer: PeerInfo;
  position: { x: number; y: number };
  isSelected: boolean;
  onClick: () => void;
  index: number;
}) => {
  const controls = useAnimation();

  useEffect(() => {
    controls.start({
      scale: [1, 1.05, 1],
      transition: { duration: 2, repeat: Infinity, delay: index * 0.2 }
    });
  }, [controls, index]);

  const getLatencyColor = (latency: number) => {
    if (latency === 0) return '#6b7280';
    if (latency < 100) return '#22c55e';
    if (latency < 300) return '#eab308';
    return '#ef4444';
  };

  return (
    <motion.g
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      style={{ cursor: 'pointer' }}
      onClick={onClick}
    >
      <motion.circle
        cx={position.x}
        cy={position.y}
        r={isSelected ? 28 : 22}
        fill="none"
        stroke={peer.isTor ? "rgba(168, 85, 247, 0.3)" : "rgba(212, 175, 55, 0.3)"}
        strokeWidth="2"
        animate={controls}
      />
      {peer.isBootstrap && (
        <motion.circle
          cx={position.x}
          cy={position.y}
          r={28}
          fill="none"
          stroke="rgba(34, 197, 94, 0.5)"
          strokeWidth="2"
          initial={{ r: 20, opacity: 1 }}
          animate={{ r: 35, opacity: 0 }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      )}
      <motion.circle
        cx={position.x}
        cy={position.y}
        r={18}
        fill={peer.isTor
          ? "url(#torGradient)"
          : peer.isBootstrap
            ? "url(#bootstrapGradient)"
            : peer.isBrowser
              ? "url(#browserGradient)"
              : "url(#peerGradient)"
        }
        stroke={isSelected ? "#fbbf24" : peer.isTor ? "#a855f7" : peer.isBrowser ? "#ec4899" : "#d4af37"}
        strokeWidth={isSelected ? 2 : 1}
        filter="url(#glow)"
        whileHover={{ scale: 1.15 }}
        whileTap={{ scale: 0.95 }}
      />
      <motion.circle
        cx={position.x + 14}
        cy={position.y - 14}
        r={5}
        fill={peer.direction === 'outbound' ? '#22c55e' : '#3b82f6'}
        stroke="#1e293b"
        strokeWidth="1.5"
      />
      <text
        x={position.x + 14}
        y={position.y - 11}
        textAnchor="middle"
        fill="#fff"
        fontSize="7"
        fontWeight="bold"
      >
        {peer.direction === 'outbound' ? '↑' : '↓'}
      </text>
      <motion.g transform={`translate(${position.x - 8}, ${position.y - 8})`}>
        {peer.isTor ? (
          <text fontSize="12" fill="#fff" textAnchor="middle" x="8" y="12">🧅</text>
        ) : peer.isBootstrap ? (
          <text fontSize="11" fill="#fff" textAnchor="middle" x="8" y="12">⚡</text>
        ) : peer.isBrowser ? (
          <text fontSize="11" fill="#fff" textAnchor="middle" x="8" y="12">🌐</text>
        ) : (
          <Server size={16} color="#fff" />
        )}
      </motion.g>
      <motion.text
        x={position.x}
        y={position.y + 32}
        textAnchor="middle"
        fill={peer.isTor ? "#c084fc" : peer.isBrowser ? "#f472b6" : "#fbbf24"}
        fontSize="8"
        fontFamily="monospace"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: index * 0.1 + 0.3 }}
      >
        {peer.shortId}
      </motion.text>
      <motion.text
        x={position.x}
        y={position.y + 42}
        textAnchor="middle"
        fill={getLatencyColor(peer.latency)}
        fontSize="7"
        fontFamily="monospace"
      >
        {peer.latency > 0 ? `${peer.latency}ms` : '...'}
      </motion.text>
    </motion.g>
  );
};

// Central node (this node)
const CentralNode = ({ position, peerId, isTorBrowser }: { position: { x: number; y: number }; peerId: string | null; isTorBrowser?: boolean }) => (
  <motion.g>
    <motion.circle
      cx={position.x}
      cy={position.y}
      r={40}
      fill="none"
      stroke={isTorBrowser ? "url(#torFieldGradient)" : "url(#quantumFieldGradient)"}
      strokeWidth="2"
      strokeDasharray="6 3"
      animate={{ rotate: 360 }}
      transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
      style={{ transformOrigin: `${position.x}px ${position.y}px` }}
    />
    {[0, 1, 2].map((i) => (
      <motion.circle
        key={i}
        cx={position.x}
        cy={position.y}
        r={25}
        fill="none"
        stroke={isTorBrowser ? "rgba(168, 85, 247, 0.3)" : "rgba(212, 175, 55, 0.3)"}
        strokeWidth="1"
        initial={{ r: 25, opacity: 0.8 }}
        animate={{ r: 50, opacity: 0 }}
        transition={{ duration: 3, repeat: Infinity, delay: i * 1 }}
      />
    ))}
    <motion.circle
      cx={position.x}
      cy={position.y}
      r={28}
      fill={isTorBrowser ? "url(#torCentralGradient)" : "url(#centralGradient)"}
      stroke={isTorBrowser ? "#a855f7" : "#fbbf24"}
      strokeWidth="2"
      filter="url(#strongGlow)"
      animate={{ scale: [1, 1.05, 1] }}
      transition={{ duration: 2, repeat: Infinity }}
    />
    {isTorBrowser ? (
      <motion.text
        x={position.x}
        y={position.y + 5}
        textAnchor="middle"
        fill="#fff"
        fontSize="14"
      >
        🧅
      </motion.text>
    ) : (
      <motion.text
        x={position.x}
        y={position.y + 4}
        textAnchor="middle"
        fill="#fff"
        fontSize="10"
        fontWeight="bold"
      >
        YOU
      </motion.text>
    )}
    <motion.text
      x={position.x}
      y={position.y + 45}
      textAnchor="middle"
      fill={isTorBrowser ? "#c084fc" : "#d4af37"}
      fontSize="8"
      fontWeight="bold"
    >
      {isTorBrowser ? "TOR" : (peerId ? `${peerId.substring(0, 8)}...` : 'Init...')}
    </motion.text>
  </motion.g>
);

// Connection log entry
interface ConnectionLog {
  time: string;
  message: string;
  type: 'info' | 'success' | 'error' | 'warning';
}

/**
 * Detect if user is browsing via Tor Browser
 * Uses multiple heuristics:
 * - WebRTC disabled (Tor blocks WebRTC for privacy)
 * - Canvas fingerprinting protection
 * - Known Tor User-Agent patterns
 */
function detectTorBrowser(): boolean {
  try {
    // Check 1: WebRTC disabled (most reliable for Tor Browser)
    // Tor Browser disables WebRTC to prevent IP leaks
    const rtcDisabled = typeof RTCPeerConnection === 'undefined' ||
      typeof navigator.mediaDevices === 'undefined';

    // Check 2: User Agent contains Tor indicators
    const ua = navigator.userAgent.toLowerCase();
    const torUserAgent = ua.includes('tor') || ua.includes('torbrowser');

    // Check 3: Check if Firefox ESR (Tor Browser is based on Firefox ESR)
    const isFirefoxESR = ua.includes('firefox/') && !ua.includes('chrome');

    // If WebRTC is disabled AND Firefox-based, likely Tor Browser
    if (rtcDisabled && isFirefoxESR) return true;
    if (torUserAgent) return true;

    return false;
  } catch {
    return false;
  }
}

export default function NetworkMapModal({ isOpen, onClose, peers: peerCount, blockHeight }: NetworkMapModalProps) {
  const { node, peerId, peerCount: libp2pPeerCount, connectionCount, topics, isReady, refresh } = useLibP2P();
  const { blockHistory, latestBlock, isSubscribed, verificationStats } = useRealtimeBlocks();

  const [peerList, setPeerList] = useState<PeerInfo[]>([]);
  const [selectedPeer, setSelectedPeer] = useState<PeerInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [connectionLogs, setConnectionLogs] = useState<ConnectionLog[]>([]);
  const [isTorBrowser, setIsTorBrowser] = useState(false);
  // v3.5.8: Known browser peers from gossipsub discovery
  const [knownBrowserPeers, setKnownBrowserPeers] = useState<KnownBrowserPeer[]>([]);
  // v3.5.8: View mode toggle - classic network map vs quantum resonance visualization
  const [viewMode, setViewMode] = useState<'classic' | 'quantum'>('classic');
  const containerRef = useRef<HTMLDivElement>(null);
  const pingTimeouts = useRef<Map<string, NodeJS.Timeout>>(new Map());
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Add log entry
  const addLog = useCallback((message: string, type: ConnectionLog['type'] = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    setConnectionLogs(prev => [...prev.slice(-50), { time, message, type }]);
  }, []);

  // Data stream state
  const [activeTab, setActiveTab] = useState<TabType>('blocks');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'height', direction: 'desc' });
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 15;

  // v3.8.0: ZK-STARK & Post-Quantum Metrics State
  const [txStats, setTxStats] = useState<TransactionStats>(getTransactionStats());
  const [pqStatus, setPqStatus] = useState<LocalPQStatus | null>(null);
  const [starkConfig] = useState(getStarkConfig());

  // Refresh metrics periodically
  useEffect(() => {
    const updateMetrics = () => {
      setTxStats(getTransactionStats());
      if (isPQCryptoAvailable()) {
        const rawStatus = getPQCryptoStatus();
        setPqStatus({
          loaded: rawStatus.loaded,
          type: rawStatus.type,
          keypairGenerated: rawStatus.loaded, // If loaded, keypair is available
          fingerprint: null, // Fingerprint not readily available without keypair access
          constants: rawStatus.constants
        });
      }
    };

    updateMetrics();
    const interval = setInterval(updateMetrics, 2000);
    return () => clearInterval(interval);
  }, []);

  // Extract bootstrap peer IDs
  const bootstrapPeerIds = useMemo(() => {
    return BOOTSTRAP_PEERS.map(addr => {
      const match = addr.match(/\/p2p\/([^/]+)$/);
      return match ? match[1] : '';
    }).filter(Boolean);
  }, []);

  // Fetch real peer data
  const fetchPeerData = useCallback(async () => {
    if (!node || !isReady) {
      setIsLoading(false);
      return;
    }

    setIsRefreshing(true);

    try {
      const connections = node.getConnections();
      const pingService = (node.services as any).ping;
      const peerInfos: PeerInfo[] = [];

      for (const conn of connections) {
        const remotePeerId = conn.remotePeer.toString();
        const remoteAddr = conn.remoteAddr.toString();
        const isTor = remoteAddr.includes('.onion') || remoteAddr.includes('/onion');
        const isBootstrap = bootstrapPeerIds.some(bpId => remotePeerId.includes(bpId));
        // v3.5.x: Detect browser peers - they connect via WebRTC or are relayed through circuit
        const isBrowser = remoteAddr.includes('/webrtc/') ||
                          remoteAddr.includes('/p2p-circuit/') ||
                          remoteAddr.includes('/webtransport/') ||
                          // Also check if it's an incoming WebSocket connection (likely browser)
                          (conn.direction === 'inbound' && remoteAddr.includes('/ws/'));

        let latency = 0;
        if (pingService) {
          try {
            const start = performance.now();
            await Promise.race([
              pingService.ping(conn.remotePeer),
              new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 5000))
            ]);
            latency = Math.round(performance.now() - start);
          } catch {
            // Ping failed
          }
        }

        peerInfos.push({
          id: remotePeerId,
          shortId: `${remotePeerId.substring(0, 6)}...`,
          address: remoteAddr,
          latency,
          status: conn.status,
          direction: conn.direction as 'inbound' | 'outbound',
          isTor,
          isBootstrap,
          isBrowser,
          connectedAt: conn.timeline.open || Date.now(),
          protocols: [],
        });
      }

      setPeerList(peerInfos);
    } catch (error) {
      console.error('Error fetching peer data:', error);
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [node, isReady, bootstrapPeerIds]);

  // Listen for connection events
  useEffect(() => {
    if (!node || !isOpen) return;

    const handlePeerConnect = (evt: any) => {
      const remotePeer = evt.detail?.remotePeer?.toString() || 'unknown';
      addLog(`Connected to peer: ${remotePeer.substring(0, 12)}...`, 'success');
    };

    const handlePeerDisconnect = (evt: any) => {
      const remotePeer = evt.detail?.remotePeer?.toString() || 'unknown';
      addLog(`Disconnected from: ${remotePeer.substring(0, 12)}...`, 'warning');
    };

    const handleConnectionOpen = (evt: any) => {
      const remoteAddr = evt.detail?.remoteAddr?.toString() || '';
      addLog(`Connection opened: ${remoteAddr.substring(0, 40)}...`, 'info');
    };

    node.addEventListener('peer:connect', handlePeerConnect);
    node.addEventListener('peer:disconnect', handlePeerDisconnect);
    node.addEventListener('connection:open', handleConnectionOpen);

    // Initial status log
    addLog(`Node initialized with PeerID: ${peerId?.substring(0, 12) || 'loading'}...`, 'info');
    addLog(`Attempting to connect to ${BOOTSTRAP_PEERS.length} bootstrap peers...`, 'info');

    return () => {
      node.removeEventListener('peer:connect', handlePeerConnect);
      node.removeEventListener('peer:disconnect', handlePeerDisconnect);
      node.removeEventListener('connection:open', handleConnectionOpen);
    };
  }, [node, isOpen, peerId, addLog]);

  useEffect(() => {
    if (!isOpen) return;
    fetchPeerData();
    const interval = setInterval(fetchPeerData, 10000);
    return () => {
      clearInterval(interval);
      pingTimeouts.current.forEach(timeout => clearTimeout(timeout));
      pingTimeouts.current.clear();
    };
  }, [isOpen, fetchPeerData]);

  // Log peer list changes
  useEffect(() => {
    if (peerList.length > 0) {
      addLog(`Now connected to ${peerList.length} peer(s)`, 'success');
    }
  }, [peerList.length, addLog]);

  // Detect Tor browser on mount
  useEffect(() => {
    const detected = detectTorBrowser();
    setIsTorBrowser(detected);
    if (detected) {
      addLog('🧅 Tor Browser detected - enhanced privacy enabled', 'success');
    }
  }, [addLog]);

  // v3.5.8: Listen for browser peer discovery events and refresh periodically
  useEffect(() => {
    if (!isOpen) return;

    // Initial load
    setKnownBrowserPeers(getKnownBrowserPeers());

    // Listen for discovery events
    const handlePeerDiscovered = (event: CustomEvent) => {
      setKnownBrowserPeers(getKnownBrowserPeers());
      addLog(`🌐 Discovered browser peer: ${event.detail.peerId.substring(0, 12)}...`, 'success');
    };

    const handlePeersUpdated = () => {
      setKnownBrowserPeers(getKnownBrowserPeers());
    };

    window.addEventListener('browser-peer-discovered', handlePeerDiscovered as EventListener);
    window.addEventListener('browser-peers-updated', handlePeersUpdated);

    // Refresh periodically
    const interval = setInterval(() => {
      setKnownBrowserPeers(getKnownBrowserPeers());
    }, 10000);

    return () => {
      window.removeEventListener('browser-peer-discovered', handlePeerDiscovered as EventListener);
      window.removeEventListener('browser-peers-updated', handlePeersUpdated);
      clearInterval(interval);
    };
  }, [isOpen, addLog]);

  // Auto-scroll logs to bottom
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [connectionLogs]);

  const handleRefresh = useCallback(() => {
    refresh();
    fetchPeerData();
  }, [refresh, fetchPeerData]);

  // Calculate peer positions
  const peerPositions = useMemo(() => {
    const centerX = 180;
    const centerY = 160;
    const radius = 110;

    return peerList.map((_, index) => {
      const angle = (index / Math.max(peerList.length, 1)) * 2 * Math.PI - Math.PI / 2;
      return {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      };
    });
  }, [peerList]);

  // v3.5.8: Calculate positions for discovered browser peers (outer ring)
  const browserPeerPositions = useMemo(() => {
    const centerX = 180;
    const centerY = 160;
    const radius = 140; // Outer ring for discovered peers

    return knownBrowserPeers.map((_, index) => {
      const angle = (index / Math.max(knownBrowserPeers.length, 1)) * 2 * Math.PI - Math.PI / 4;
      return {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      };
    });
  }, [knownBrowserPeers]);

  const centerPosition = { x: 180, y: 160 };
  const torPeers = peerList.filter(p => p.isTor).length;
  const avgLatency = peerList.length > 0
    ? Math.round(peerList.filter(p => p.latency > 0).reduce((sum, p) => sum + p.latency, 0) / Math.max(peerList.filter(p => p.latency > 0).length, 1))
    : 0;

  // Data stream logic
  const allTransactions = useMemo(() => {
    const txs: (Transaction & { blockHeight: number; blockTimestamp: number })[] = [];
    blockHistory.forEach(block => {
      block.transactions.forEach(tx => {
        txs.push({
          ...tx,
          blockHeight: block.header.height,
          blockTimestamp: block.header.timestamp
        });
      });
    });
    return txs;
  }, [blockHistory]);

  // Extract miner statistics from recent blocks
  // Block reward is 0.5 QNK = 500,000,000 atomic units, split among solutions in block
  const BLOCK_REWARD_ATOMIC = 500_000_000; // 0.5 QNK in atomic units

  const minerStats = useMemo(() => {
    const minerMap = new Map<string, MinerStats>();

    blockHistory.forEach(block => {
      // Calculate reward per solution for this block
      const numSolutions = block.miningSolutions.length;
      const rewardPerSolution = numSolutions > 0 ? BLOCK_REWARD_ATOMIC / numSolutions : 0;

      block.miningSolutions.forEach(solution => {
        // Ensure miner is a string (could be Uint8Array from P2P)
        const rawMiner = solution.miner as unknown;
        let minerAddr: string;
        if (typeof rawMiner === 'string') {
          minerAddr = rawMiner;
        } else if (rawMiner instanceof Uint8Array) {
          minerAddr = Array.from(rawMiner).map((b) => b.toString(16).padStart(2, '0')).join('');
        } else if (Array.isArray(rawMiner)) {
          minerAddr = rawMiner.map((b) => (typeof b === 'number' ? b.toString(16).padStart(2, '0') : '')).join('');
        } else {
          minerAddr = String(rawMiner || '');
        }

        const existing = minerMap.get(minerAddr);
        if (existing) {
          existing.solutionCount += 1;
          existing.totalRewards += rewardPerSolution;
          existing.lastActive = Math.max(existing.lastActive, block.header.timestamp);
          existing.avgDifficulty = (existing.avgDifficulty * (existing.solutionCount - 1) + solution.difficulty) / existing.solutionCount;
        } else {
          minerMap.set(minerAddr, {
            address: minerAddr,
            shortAddress: minerAddr.length > 12 ? `${minerAddr.substring(0, 6)}...${minerAddr.substring(minerAddr.length - 4)}` : minerAddr,
            solutionCount: 1,
            totalRewards: rewardPerSolution,
            lastActive: block.header.timestamp,
            avgDifficulty: solution.difficulty,
          });
        }
      });
    });

    // Convert to array and sort by solution count (more reliable than calculated rewards)
    return Array.from(minerMap.values()).sort((a, b) => b.solutionCount - a.solutionCount);
  }, [blockHistory]);

  // Calculate total network rewards
  const totalNetworkRewards = useMemo(() => {
    return minerStats.reduce((sum, m) => sum + m.totalRewards, 0);
  }, [minerStats]);

  const filteredBlocks = useMemo(() => {
    let filtered = [...blockHistory];
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(block =>
        block.header.height.toString().includes(query) ||
        block.header.proposer.toLowerCase().includes(query)
      );
    }
    filtered.sort((a, b) => {
      const aVal = sortConfig.key === 'height' ? a.header.height :
                   sortConfig.key === 'timestamp' ? a.header.timestamp :
                   sortConfig.key === 'txCount' ? a.transactions.length : a.sizeBytes;
      const bVal = sortConfig.key === 'height' ? b.header.height :
                   sortConfig.key === 'timestamp' ? b.header.timestamp :
                   sortConfig.key === 'txCount' ? b.transactions.length : b.sizeBytes;
      return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal;
    });
    return filtered;
  }, [blockHistory, searchQuery, sortConfig]);

  const filteredTransactions = useMemo(() => {
    let filtered = [...allTransactions];
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(tx =>
        tx.from.toLowerCase().includes(query) ||
        tx.to.toLowerCase().includes(query)
      );
    }
    filtered.sort((a, b) => {
      const aVal = sortConfig.key === 'amount' ? a.amount : a.timestamp;
      const bVal = sortConfig.key === 'amount' ? b.amount : b.timestamp;
      return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal;
    });
    return filtered;
  }, [allTransactions, searchQuery, sortConfig]);

  const totalPages = useMemo(() => {
    const data = activeTab === 'blocks' ? filteredBlocks :
                 activeTab === 'transactions' ? filteredTransactions : peerList;
    return Math.ceil(data.length / pageSize);
  }, [activeTab, filteredBlocks, filteredTransactions, peerList, pageSize]);

  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    const end = start + pageSize;
    if (activeTab === 'blocks') return filteredBlocks.slice(start, end);
    if (activeTab === 'transactions') return filteredTransactions.slice(start, end);
    return peerList.slice(start, end);
  }, [activeTab, filteredBlocks, filteredTransactions, peerList, currentPage, pageSize]);

  const handleSort = (key: string) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc'
    }));
  };

  const getSortIcon = (key: string) => {
    if (sortConfig.key !== key) return <ArrowUpDown size={12} className="text-gray-500" />;
    return sortConfig.direction === 'desc'
      ? <ArrowDown size={12} className="text-amber-400" />
      : <ArrowUp size={12} className="text-amber-400" />;
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/90 backdrop-blur-md flex items-center justify-center z-[9999] p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="network-map-modal relative bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border-2 border-amber-500/30 rounded-2xl w-full max-w-[1400px] h-[80vh] max-h-[750px] overflow-hidden flex flex-col"
          onClick={(e) => e.stopPropagation()}
          style={{ boxShadow: '0 0 80px rgba(212, 175, 55, 0.15)' }}
          ref={containerRef}
        >
          {/* Header */}
          <div className="flex-shrink-0 border-b border-amber-500/20 p-4 bg-slate-900/80">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <motion.div
                  className="p-2 rounded-lg bg-amber-500/20 border border-amber-500/30"
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                >
                  <Globe className="w-6 h-6 text-amber-400" />
                </motion.div>
                <div>
                  <h2 className="text-xl font-bold text-amber-100 flex items-center gap-2">
                    P2P Network Overview
                    {isTorBrowser && (
                      <span className="text-purple-400 text-sm font-normal flex items-center gap-1 bg-purple-500/20 px-2 py-0.5 rounded-full">
                        🧅 Tor Browser
                      </span>
                    )}
                  </h2>
                  <p className="text-amber-300/60 text-sm">
                    {isReady ? (
                      <>
                        {connectionCount > 0 ? connectionCount : (peerCount || 0)} connections • {topics.length} topics •
                        {pqStatus?.loaded ? ' PQ-Hybrid' : ' Noise'} encrypted
                        {txStats.starkProofSubmissions > 0 && ' • ZK-STARK active'}
                        {isTorBrowser ? ' • Tor privacy' : ''}
                      </>
                    ) : 'Connecting...'}
                  </p>
                </div>
              </div>

              {/* Global Stats */}
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-6 px-4 py-2 bg-slate-800/50 rounded-lg border border-amber-500/20">
                  <div className="text-center">
                    <div className="text-xs text-gray-400">Blocks</div>
                    <div className="text-lg font-bold text-amber-300">{blockHistory.length}</div>
                  </div>
                  <div className="w-px h-8 bg-amber-500/20" />
                  <div className="text-center">
                    <div className="text-xs text-gray-400">Transactions</div>
                    <div className="text-lg font-bold text-cyan-300">{allTransactions.length}</div>
                  </div>
                  <div className="w-px h-8 bg-amber-500/20" />
                  <div className="text-center">
                    <div className="text-xs text-gray-400">Peers</div>
                    <div className="text-lg font-bold text-green-300">{peerList.length}</div>
                  </div>
                  <div className="w-px h-8 bg-amber-500/20" />
                  <div className="text-center" title={`Discovered via gossipsub: ${knownBrowserPeers.length} browsers | Directly connected: ${peerList.filter(p => p.isBrowser).length}`}>
                    <div className="text-xs text-gray-400 flex items-center gap-1">
                      <Globe size={10} className="text-pink-400" />
                      Browsers
                    </div>
                    <div className="text-lg font-bold text-pink-300">
                      {knownBrowserPeers.length > 0 ? knownBrowserPeers.length : peerList.filter(p => p.isBrowser).length}
                    </div>
                  </div>
                  <div className="w-px h-8 bg-amber-500/20" />
                  <div className="text-center" title={`Verified: ${verificationStats.blocksVerified}, Valid: ${verificationStats.blocksValid}, Invalid: ${verificationStats.blocksInvalid}`}>
                    <div className="text-xs text-gray-400 flex items-center gap-1">
                      <ShieldCheck size={10} className="text-emerald-400" />
                      Verified
                    </div>
                    <div className={`text-lg font-bold ${verificationStats.blocksInvalid > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
                      {verificationStats.blocksValid}/{verificationStats.blocksVerified}
                    </div>
                  </div>
                  <div className="w-px h-8 bg-amber-500/20" />
                  <div className="text-center">
                    <div className="text-xs text-gray-400">Latency</div>
                    <div className={`text-lg font-bold ${avgLatency < 100 ? 'text-green-300' : avgLatency < 300 ? 'text-yellow-300' : 'text-red-300'}`}>
                      {avgLatency > 0 ? `${avgLatency}ms` : 'N/A'}
                    </div>
                  </div>
                </div>

                <motion.button
                  onClick={handleRefresh}
                  className="p-2 hover:bg-amber-500/20 rounded-lg transition-colors"
                  animate={isRefreshing ? { rotate: 360 } : {}}
                  transition={{ duration: 1, repeat: isRefreshing ? Infinity : 0 }}
                >
                  <RefreshCw className="w-5 h-5 text-amber-400" />
                </motion.button>

                <button onClick={onClose} className="p-2 hover:bg-amber-500/20 rounded-lg transition-colors">
                  <X className="w-6 h-6 text-amber-400" />
                </button>
              </div>
            </div>
          </div>

          {/* Main Content - Split View */}
          <div className="flex-1 flex overflow-hidden">
            {/* Left Panel - Network Map */}
            <div className="w-[400px] flex-shrink-0 border-r border-amber-500/20 relative">
              {isLoading ? (
                <div className="flex flex-col items-center justify-center h-full gap-4">
                  <motion.div animate={{ rotate: 360 }} transition={{ duration: 2, repeat: Infinity, ease: "linear" }}>
                    <Radio className="w-10 h-10 text-amber-400" />
                  </motion.div>
                  <p className="text-amber-300/60 text-sm">Discovering peers...</p>
                </div>
              ) : peerList.length === 0 ? (
                <div className="flex flex-col h-full">
                  {/* Status Header */}
                  <div className="flex flex-col items-center gap-3 pt-8 pb-4 border-b border-amber-500/20">
                    <motion.div
                      animate={{ scale: [1, 1.1, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      <WifiOff className="w-10 h-10 text-amber-400/50" />
                    </motion.div>
                    <p className="text-amber-300 font-medium">No peers connected</p>
                    <p className="text-amber-300/60 text-sm text-center px-4">
                      {isReady ? 'Searching for peers on the network...' : 'Initializing P2P node...'}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                      <motion.div
                        className="w-2 h-2 bg-amber-400 rounded-full"
                        animate={{ opacity: [1, 0.3, 1] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      />
                      <span className="text-amber-400/60 text-xs">Attempting connection...</span>
                    </div>
                  </div>

                  {/* Connection Logs */}
                  <div className="flex-1 overflow-hidden flex flex-col">
                    <div className="px-4 py-2 bg-slate-900/50 border-b border-amber-500/10">
                      <span className="text-amber-300/60 text-xs font-medium">Connection Log</span>
                    </div>
                    <div className="flex-1 overflow-y-auto p-2 space-y-1 font-mono text-xs">
                      {connectionLogs.length === 0 ? (
                        <div className="text-gray-500 text-center py-4">Waiting for connection events...</div>
                      ) : (
                        connectionLogs.map((log, idx) => (
                          <motion.div
                            key={idx}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            className={`flex gap-2 px-2 py-1 rounded ${
                              log.type === 'success' ? 'bg-green-500/10 text-green-300' :
                              log.type === 'error' ? 'bg-red-500/10 text-red-300' :
                              log.type === 'warning' ? 'bg-yellow-500/10 text-yellow-300' :
                              'bg-slate-800/50 text-gray-400'
                            }`}
                          >
                            <span className="text-gray-500 flex-shrink-0">{log.time}</span>
                            <span className="break-all">{log.message}</span>
                          </motion.div>
                        ))
                      )}
                      <div ref={logsEndRef} />
                    </div>
                  </div>

                  {/* Bootstrap Info */}
                  <div className="p-3 bg-slate-900/50 border-t border-amber-500/10">
                    <div className="text-amber-300/60 text-xs mb-2">Bootstrap Peers:</div>
                    <div className="space-y-1">
                      {BOOTSTRAP_PEERS.map((peer, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-xs">
                          <motion.div
                            className="w-1.5 h-1.5 bg-amber-400 rounded-full"
                            animate={{ opacity: [1, 0.3, 1] }}
                            transition={{ duration: 1, repeat: Infinity, delay: idx * 0.3 }}
                          />
                          <span className="text-gray-400 truncate">{peer.substring(0, 50)}...</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col h-full">
                  {/* View Mode Toggle */}
                  <div className="flex items-center justify-center gap-2 p-2 bg-slate-900/50 border-b border-amber-500/20">
                    <button
                      onClick={() => setViewMode('classic')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                        viewMode === 'classic'
                          ? 'bg-amber-500/30 text-amber-200 border border-amber-500/50'
                          : 'bg-slate-800/50 text-gray-400 hover:text-amber-300 border border-transparent'
                      }`}
                    >
                      <Globe size={12} />
                      Classic
                    </button>
                    <button
                      onClick={() => setViewMode('quantum')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                        viewMode === 'quantum'
                          ? 'bg-purple-500/30 text-purple-200 border border-purple-500/50'
                          : 'bg-slate-800/50 text-gray-400 hover:text-purple-300 border border-transparent'
                      }`}
                    >
                      <span className="text-sm">⚛️</span>
                      Quantum Resonance
                    </button>
                  </div>

                  {/* Visualization Area */}
                  <div className="flex-1 relative overflow-hidden">
                    {viewMode === 'quantum' ? (
                      <div className="absolute inset-0 flex items-center justify-center p-4">
                        <BrowserResonanceVisualization
                          width={380}
                          height={340}
                          showLabels={true}
                          theme="quantum"
                          currentHeight={blockHeight}
                        />
                      </div>
                    ) : (
                <svg width="100%" height="100%" viewBox="0 0 360 340" className="overflow-visible">
                  <defs>
                    <radialGradient id="centralGradient" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#fbbf24" />
                      <stop offset="70%" stopColor="#d97706" />
                      <stop offset="100%" stopColor="#92400e" />
                    </radialGradient>
                    <radialGradient id="peerGradient" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#4a5568" />
                      <stop offset="100%" stopColor="#2d3748" />
                    </radialGradient>
                    <radialGradient id="torGradient" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#9333ea" />
                      <stop offset="100%" stopColor="#581c87" />
                    </radialGradient>
                    <radialGradient id="bootstrapGradient" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#22c55e" />
                      <stop offset="100%" stopColor="#166534" />
                    </radialGradient>
                    <radialGradient id="browserGradient" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#ec4899" />
                      <stop offset="100%" stopColor="#be185d" />
                    </radialGradient>
                    <linearGradient id="quantumFieldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#fbbf24" />
                      <stop offset="50%" stopColor="#d97706" />
                      <stop offset="100%" stopColor="#fbbf24" />
                    </linearGradient>
                    {/* Tor Browser gradient definitions */}
                    <radialGradient id="torCentralGradient" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#c084fc" />
                      <stop offset="70%" stopColor="#9333ea" />
                      <stop offset="100%" stopColor="#581c87" />
                    </radialGradient>
                    <linearGradient id="torFieldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#c084fc" />
                      <stop offset="50%" stopColor="#9333ea" />
                      <stop offset="100%" stopColor="#c084fc" />
                    </linearGradient>
                    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                      <feGaussianBlur stdDeviation="2" result="blur" />
                      <feMerge>
                        <feMergeNode in="blur" />
                        <feMergeNode in="SourceGraphic" />
                      </feMerge>
                    </filter>
                    <filter id="strongGlow" x="-100%" y="-100%" width="300%" height="300%">
                      <feGaussianBlur stdDeviation="6" result="blur" />
                      <feMerge>
                        <feMergeNode in="blur" />
                        <feMergeNode in="blur" />
                        <feMergeNode in="SourceGraphic" />
                      </feMerge>
                    </filter>
                  </defs>

                  <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
                    <path d="M 30 0 L 0 0 0 30" fill="none" stroke="rgba(212, 175, 55, 0.05)" strokeWidth="1" />
                  </pattern>
                  <rect width="100%" height="100%" fill="url(#grid)" />

                  {peerList.map((peer, index) => (
                    <ConnectionLine
                      key={peer.id}
                      from={centerPosition}
                      to={peerPositions[index]}
                      isActive={peer.status === 'open'}
                      isTor={peer.isTor}
                      direction={peer.direction}
                    />
                  ))}

                  {peerList.map((peer, index) => (
                    <PeerNode
                      key={peer.id}
                      peer={peer}
                      position={peerPositions[index]}
                      isSelected={selectedPeer?.id === peer.id}
                      onClick={() => setSelectedPeer(peer)}
                      index={index}
                    />
                  ))}

                  {/* v3.5.8: Discovered browser peers (outer ring with dashed connections) */}
                  {knownBrowserPeers.map((browser, index) => {
                    const pos = browserPeerPositions[index];
                    const shortId = `${browser.peerId.substring(0, 6)}...`;
                    return (
                      <g key={browser.peerId}>
                        {/* Dashed connection line to center (via bootstrap) */}
                        <motion.path
                          d={`M ${centerPosition.x} ${centerPosition.y} Q ${(centerPosition.x + pos.x) / 2} ${(centerPosition.y + pos.y) / 2 - 15} ${pos.x} ${pos.y}`}
                          stroke="rgba(236, 72, 153, 0.3)"
                          strokeWidth="1.5"
                          strokeDasharray="4 3"
                          fill="none"
                          initial={{ pathLength: 0 }}
                          animate={{ pathLength: 1 }}
                          transition={{ duration: 1.5, delay: index * 0.1 }}
                        />
                        {/* Browser peer node (hollow circle to indicate discovered, not direct) */}
                        <motion.g
                          initial={{ scale: 0, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ duration: 0.5, delay: index * 0.15 + 0.5 }}
                        >
                          <motion.circle
                            cx={pos.x}
                            cy={pos.y}
                            r={14}
                            fill="rgba(236, 72, 153, 0.1)"
                            stroke={browser.isTorBrowser ? "#a855f7" : "#ec4899"}
                            strokeWidth="2"
                            strokeDasharray="3 2"
                            animate={{ scale: [1, 1.05, 1] }}
                            transition={{ duration: 2.5, repeat: Infinity, delay: index * 0.3 }}
                          />
                          <text
                            x={pos.x}
                            y={pos.y + 4}
                            textAnchor="middle"
                            fill="#fff"
                            fontSize="9"
                          >
                            {browser.isTorBrowser ? '🧅' : '🌐'}
                          </text>
                          <text
                            x={pos.x}
                            y={pos.y + 26}
                            textAnchor="middle"
                            fill="#f472b6"
                            fontSize="7"
                            fontFamily="monospace"
                          >
                            {shortId}
                          </text>
                          <text
                            x={pos.x}
                            y={pos.y + 35}
                            textAnchor="middle"
                            fill="#9ca3af"
                            fontSize="6"
                            fontFamily="monospace"
                          >
                            H:{browser.blockHeight}
                          </text>
                        </motion.g>
                      </g>
                    );
                  })}

                  <CentralNode position={centerPosition} peerId={peerId} isTorBrowser={isTorBrowser} />
                </svg>
                    )}
                  </div>
                </div>
              )}

              {/* Legend */}
              <div className="absolute bottom-4 left-4 bg-slate-900/90 border border-amber-500/20 rounded-lg p-3">
                <div className="text-amber-300/60 text-xs mb-2 font-medium">Legend</div>
                <div className="space-y-1.5 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-gradient-to-r from-green-400 to-green-600" />
                    <span className="text-gray-300">Bootstrap</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-gradient-to-r from-pink-400 to-pink-600" />
                    <span className="text-pink-300">Browser (Connected)</span>
                    {peerList.filter(p => p.isBrowser).length > 0 && (
                      <span className="text-pink-400 font-bold">({peerList.filter(p => p.isBrowser).length})</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full border-2 border-pink-400 bg-transparent" />
                    <span className="text-pink-200">Browser (Discovered)</span>
                    {knownBrowserPeers.length > 0 && (
                      <span className="text-pink-300 font-bold">({knownBrowserPeers.length})</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-gradient-to-r from-purple-400 to-purple-600" />
                    <span className="text-gray-300">Tor Peer</span>
                  </div>
                  {isTorBrowser && (
                    <div className="flex items-center gap-2">
                      <span className="text-sm">🧅</span>
                      <span className="text-purple-300 font-medium">You (Tor Browser)</span>
                    </div>
                  )}
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-gradient-to-r from-slate-400 to-slate-600" />
                    <span className="text-gray-300">Full Node</span>
                  </div>
                  <div className="flex items-center gap-2 pt-1 border-t border-amber-500/20">
                    <div className="w-3 h-3 rounded-full bg-green-500" />
                    <span className="text-gray-400">Out</span>
                    <div className="w-3 h-3 rounded-full bg-blue-500 ml-2" />
                    <span className="text-gray-400">In</span>
                  </div>
                </div>
              </div>

              {/* Selected Peer Details */}
              {selectedPeer && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="absolute top-4 left-4 right-4 bg-slate-900/95 border border-amber-500/30 rounded-lg p-3"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {selectedPeer.isBootstrap ? <Zap size={14} className="text-green-400" /> : selectedPeer.isBrowser ? <Globe size={14} className="text-pink-400" /> : <Server size={14} className="text-amber-400" />}
                      <span className={`text-sm font-bold ${selectedPeer.isBrowser ? 'text-pink-200' : 'text-amber-100'}`}>
                        {selectedPeer.isBootstrap ? 'Bootstrap' : selectedPeer.isTor ? 'Tor Peer' : selectedPeer.isBrowser ? 'Browser Peer' : 'Full Node'}
                      </span>
                      <span className={`px-1.5 py-0.5 text-xs rounded ${
                        selectedPeer.direction === 'outbound' ? 'bg-green-500/20 text-green-400' : 'bg-blue-500/20 text-blue-400'
                      }`}>
                        {selectedPeer.direction}
                      </span>
                    </div>
                    <button onClick={() => setSelectedPeer(null)} className="p-1 hover:bg-amber-500/20 rounded">
                      <X size={14} className="text-amber-400" />
                    </button>
                  </div>
                  <div className="space-y-1 text-xs">
                    <div><span className="text-gray-400">ID:</span> <code className="text-amber-200 ml-1">{truncateHash(selectedPeer.id, 10)}</code></div>
                    <div><span className="text-gray-400">Latency:</span> <span className={selectedPeer.latency < 100 ? 'text-green-400' : 'text-yellow-400'}>{selectedPeer.latency}ms</span></div>
                    <div><span className="text-gray-400">Connected:</span> <span className="text-gray-300">{Math.floor((Date.now() - selectedPeer.connectedAt) / 60000)}m ago</span></div>
                  </div>
                </motion.div>
              )}
            </div>

            {/* Right Panel - Data Stream */}
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Tabs */}
              <div className="flex-shrink-0 border-b border-amber-500/20 px-4 bg-slate-900/50">
                <div className="flex gap-1">
                  {[
                    { id: 'blocks' as TabType, label: 'Blocks', icon: Box, count: blockHistory.length },
                    { id: 'transactions' as TabType, label: 'Transactions', icon: FileText, count: allTransactions.length },
                    { id: 'peers' as TabType, label: 'Peers', icon: Users, count: peerList.length },
                    { id: 'miners' as TabType, label: 'Miners', icon: Zap, count: minerStats.length },
                    { id: 'metrics' as TabType, label: 'Metrics', icon: Activity, count: null },
                  ].map(tab => (
                    <button
                      key={tab.id}
                      onClick={() => { setActiveTab(tab.id); setCurrentPage(1); }}
                      className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                        activeTab === tab.id
                          ? 'border-amber-400 text-amber-100 bg-amber-500/10'
                          : 'border-transparent text-gray-400 hover:text-amber-200'
                      }`}
                    >
                      <tab.icon size={14} />
                      <span className="text-sm">{tab.label}</span>
                      {tab.count !== null && (
                        <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                          activeTab === tab.id ? 'bg-amber-500/30' : 'bg-gray-700'
                        }`}>
                          {tab.count}
                        </span>
                      )}
                    </button>
                  ))}
                </div>
              </div>

              {/* Search */}
              {activeTab !== 'metrics' && (
                <div className="flex-shrink-0 p-3 border-b border-amber-500/20 bg-slate-900/30">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={e => { setSearchQuery(e.target.value); setCurrentPage(1); }}
                      placeholder={`Search ${activeTab}...`}
                      className="w-full pl-9 pr-4 py-2 bg-slate-800 border border-amber-500/20 rounded-lg text-amber-100 text-sm placeholder-gray-500 focus:outline-none focus:border-amber-500/50"
                    />
                  </div>
                </div>
              )}

              {/* Data Table */}
              <div className="flex-1 overflow-auto p-3">
                {activeTab === 'blocks' && (
                  <table className="w-full">
                    <thead className="sticky top-0 bg-slate-900/95 backdrop-blur">
                      <tr className="text-left text-xs text-gray-400 uppercase">
                        <th className="pb-2 pr-3">
                          <button onClick={() => handleSort('height')} className="flex items-center gap-1 hover:text-amber-300">
                            Height {getSortIcon('height')}
                          </button>
                        </th>
                        <th className="pb-2 pr-3">
                          <button onClick={() => handleSort('timestamp')} className="flex items-center gap-1 hover:text-amber-300">
                            Time {getSortIcon('timestamp')}
                          </button>
                        </th>
                        <th className="pb-2 pr-3">
                          <button onClick={() => handleSort('txCount')} className="flex items-center gap-1 hover:text-amber-300">
                            Txs {getSortIcon('txCount')}
                          </button>
                        </th>
                        <th className="pb-2 pr-3">Proposer</th>
                        <th className="pb-2 pr-3">DAG</th>
                        <th className="pb-2 pr-3">
                          <button onClick={() => handleSort('size')} className="flex items-center gap-1 hover:text-amber-300">
                            Size {getSortIcon('size')}
                          </button>
                        </th>
                        <th className="pb-2">Mining</th>
                        <th className="pb-2">Verified</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-amber-500/10">
                      <AnimatePresence initial={false} mode="popLayout">
                        {(paginatedData as VerifiedBlock[]).map((block) => (
                          <motion.tr
                            key={block.header.height}
                            layout
                            initial={{ opacity: 0, backgroundColor: 'rgba(245, 158, 11, 0.2)' }}
                            animate={{ opacity: 1, backgroundColor: 'rgba(0, 0, 0, 0)' }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.3, layout: { duration: 0.2 } }}
                            className="text-sm hover:bg-amber-500/5"
                          >
                            <td className="py-2 pr-3">
                              <span className="font-mono text-amber-300 font-bold">#{block.header.height.toLocaleString()}</span>
                            </td>
                            <td className="py-2 pr-3 text-gray-300">{formatTime(block.header.timestamp)}</td>
                            <td className="py-2 pr-3">
                              <span className={`px-1.5 py-0.5 rounded text-xs ${
                                block.transactions.length > 0 ? 'bg-cyan-500/20 text-cyan-300' : 'bg-gray-500/20 text-gray-400'
                              }`}>
                                {block.transactions.length}
                              </span>
                            </td>
                            <td className="py-2 pr-3">
                              <code className="text-xs text-gray-400 bg-slate-800 px-1.5 py-0.5 rounded">{truncateHash(block.header.proposer, 5)}</code>
                            </td>
                            <td className="py-2 pr-3 text-purple-300">{block.dagParents.length}</td>
                            <td className="py-2 pr-3 text-gray-400">{formatNumber(block.sizeBytes)}B</td>
                            <td className="py-2 pr-3">
                              <div className="flex items-center gap-1">
                                <Zap size={10} className="text-yellow-400" />
                                <span className="text-yellow-300 text-xs">{block.miningSolutions.length}</span>
                              </div>
                            </td>
                            <td className="py-2">
                              {block.verification ? (
                                block.verification.valid ? (
                                  <div className="flex items-center gap-1" title={block.verification.summary}>
                                    <CheckCircle size={14} className="text-emerald-400" />
                                    <span className="text-emerald-400 text-xs">{block.verification.checks.filter(c => c.passed).length}/{block.verification.checks.length}</span>
                                  </div>
                                ) : (
                                  <div className="flex items-center gap-1" title={block.verification.checks.filter(c => !c.passed).map(c => c.name).join(', ')}>
                                    <AlertCircle size={14} className="text-amber-400" />
                                    <span className="text-amber-400 text-xs">{block.verification.checks.filter(c => c.passed).length}/{block.verification.checks.length}</span>
                                  </div>
                                )
                              ) : (
                                <span className="text-gray-500 text-xs">-</span>
                              )}
                            </td>
                          </motion.tr>
                        ))}
                      </AnimatePresence>
                      {filteredBlocks.length === 0 && (
                        <tr><td colSpan={8} className="py-8 text-center text-gray-400">No blocks received yet</td></tr>
                      )}
                    </tbody>
                  </table>
                )}

                {activeTab === 'transactions' && (
                  <table className="w-full">
                    <thead className="sticky top-0 bg-slate-900/95 backdrop-blur">
                      <tr className="text-left text-xs text-gray-400 uppercase">
                        <th className="pb-2 pr-3">Block</th>
                        <th className="pb-2 pr-3">From</th>
                        <th className="pb-2 pr-3">To</th>
                        <th className="pb-2 pr-3">
                          <button onClick={() => handleSort('amount')} className="flex items-center gap-1 hover:text-amber-300">
                            Amount {getSortIcon('amount')}
                          </button>
                        </th>
                        <th className="pb-2">Time</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-amber-500/10">
                      {(paginatedData as (Transaction & { blockHeight: number; blockTimestamp: number })[]).map((tx, idx) => (
                        <motion.tr
                          key={idx}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: idx * 0.02 }}
                          className="text-sm hover:bg-amber-500/5"
                        >
                          <td className="py-2 pr-3"><span className="font-mono text-amber-300">#{tx.blockHeight}</span></td>
                          <td className="py-2 pr-3"><code className="text-xs text-green-300 bg-slate-800 px-1.5 py-0.5 rounded">{truncateHash(tx.from, 6)}</code></td>
                          <td className="py-2 pr-3"><code className="text-xs text-blue-300 bg-slate-800 px-1.5 py-0.5 rounded">{truncateHash(tx.to, 6)}</code></td>
                          <td className="py-2 pr-3"><span className="text-cyan-300 font-bold">{tx.amount.toLocaleString()} QNK</span></td>
                          <td className="py-2 text-gray-400">{formatTime(tx.timestamp)}</td>
                        </motion.tr>
                      ))}
                      {filteredTransactions.length === 0 && (
                        <tr><td colSpan={5} className="py-8 text-center text-gray-400">No transactions yet</td></tr>
                      )}
                    </tbody>
                  </table>
                )}

                {activeTab === 'peers' && (
                  <table className="w-full">
                    <thead className="sticky top-0 bg-slate-900/95 backdrop-blur">
                      <tr className="text-left text-xs text-gray-400 uppercase">
                        <th className="pb-2 pr-3">Peer ID</th>
                        <th className="pb-2 pr-3">Address</th>
                        <th className="pb-2 pr-3">Direction</th>
                        <th className="pb-2 pr-3">Status</th>
                        <th className="pb-2">Connected</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-amber-500/10">
                      {peerList.map((peer, idx) => (
                        <motion.tr
                          key={peer.id}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: idx * 0.03 }}
                          className="text-sm hover:bg-amber-500/5"
                        >
                          <td className="py-2 pr-3"><code className="text-xs text-amber-300 bg-slate-800 px-1.5 py-0.5 rounded">{truncateHash(peer.id, 8)}</code></td>
                          <td className="py-2 pr-3"><code className="text-xs text-gray-400 max-w-[200px] truncate block">{peer.address}</code></td>
                          <td className="py-2 pr-3">
                            <span className={`px-1.5 py-0.5 rounded text-xs ${
                              peer.direction === 'outbound' ? 'bg-green-500/20 text-green-300' : 'bg-blue-500/20 text-blue-300'
                            }`}>
                              {peer.direction}
                            </span>
                          </td>
                          <td className="py-2 pr-3">
                            <span className={`px-1.5 py-0.5 rounded text-xs ${
                              peer.status === 'open' ? 'bg-green-500/20 text-green-300' : 'bg-yellow-500/20 text-yellow-300'
                            }`}>
                              {peer.status}
                            </span>
                          </td>
                          <td className="py-2 text-gray-400">{Math.floor((Date.now() - peer.connectedAt) / 60000)}m ago</td>
                        </motion.tr>
                      ))}
                      {peerList.length === 0 && (
                        <tr><td colSpan={5} className="py-8 text-center text-gray-400">No peers connected</td></tr>
                      )}
                    </tbody>
                  </table>
                )}

                {activeTab === 'miners' && (
                  <div className="space-y-4">
                    {/* Network Mining Summary */}
                    <div className="grid grid-cols-3 gap-3">
                      <div className="bg-slate-800/50 border border-amber-500/20 rounded-xl p-4 text-center">
                        <div className="text-xs text-gray-400 mb-1">Active Miners</div>
                        <div className="text-2xl font-bold text-amber-300">{minerStats.length}</div>
                      </div>
                      <div className="bg-slate-800/50 border border-green-500/20 rounded-xl p-4 text-center">
                        <div className="text-xs text-gray-400 mb-1">Total Solutions</div>
                        <div className="text-2xl font-bold text-green-300">
                          {minerStats.reduce((sum, m) => sum + m.solutionCount, 0)}
                        </div>
                      </div>
                      <div className="bg-slate-800/50 border border-purple-500/20 rounded-xl p-4 text-center">
                        <div className="text-xs text-gray-400 mb-1">Total Rewards</div>
                        <div className="text-2xl font-bold text-purple-300">
                          {formatQNK(totalNetworkRewards)} QNK
                        </div>
                      </div>
                    </div>

                    {/* Miners Table */}
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-left text-gray-400 border-b border-amber-500/20">
                          <th className="py-2 px-3">#</th>
                          <th className="py-2 px-3">Miner Address</th>
                          <th className="py-2 px-3 text-right">Solutions</th>
                          <th className="py-2 px-3 text-right">Rewards (QNK)</th>
                          <th className="py-2 px-3 text-right">Last Active</th>
                        </tr>
                      </thead>
                      <tbody>
                        {minerStats.map((miner, index) => (
                          <tr
                            key={miner.address}
                            className="border-b border-slate-700/50 hover:bg-amber-500/5 transition-all duration-300"
                          >
                            <td className="py-3 px-3 text-gray-400">{index + 1}</td>
                            <td className="py-3 px-3">
                              <div className="flex items-center gap-2">
                                <div className={`w-2 h-2 rounded-full transition-colors duration-500 ${
                                  Date.now() / 1000 - miner.lastActive < 60 ? 'bg-green-400 animate-pulse' : 'bg-gray-500'
                                }`} />
                                <code className="text-amber-200 text-xs">{miner.shortAddress}</code>
                              </div>
                            </td>
                            <td className="py-3 px-3 text-right">
                              <span className="text-cyan-300 transition-all duration-300">{miner.solutionCount}</span>
                            </td>
                            <td className="py-3 px-3 text-right">
                              <span className="text-green-300 font-medium transition-all duration-300">{formatQNK(miner.totalRewards)}</span>
                            </td>
                            <td className="py-3 px-3 text-right text-gray-400 text-xs">
                              {formatTime(miner.lastActive)}
                            </td>
                          </tr>
                        ))}
                        {minerStats.length === 0 && (
                          <tr>
                            <td colSpan={5} className="py-8 text-center text-gray-400">
                              No mining activity in recent blocks
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                )}

                {activeTab === 'metrics' && (
                  <div className="space-y-4 overflow-y-auto max-h-[calc(100vh-300px)]">
                    {/* ZK-STARK Proof Statistics - v3.8.0 */}
                    <div className="bg-gradient-to-br from-purple-900/30 to-slate-800/50 border border-purple-500/30 rounded-xl p-4">
                      <h3 className="text-purple-300 font-bold mb-3 flex items-center gap-2">
                        <ShieldCheck size={16} />
                        ZK-STARK Privacy Proofs
                        <span className="ml-auto text-xs px-2 py-0.5 bg-purple-500/20 text-purple-300 rounded-full">v3.8.0</span>
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="p-3 bg-slate-900/80 rounded-lg border border-purple-500/10">
                          <div className="text-xs text-gray-400">Proofs Generated</div>
                          <div className="text-xl font-bold text-purple-300">{txStats.starkProofSubmissions}</div>
                        </div>
                        <div className="p-3 bg-slate-900/80 rounded-lg border border-purple-500/10">
                          <div className="text-xs text-gray-400">Avg Proving Time</div>
                          <div className="text-xl font-bold text-cyan-300">
                            {txStats.starkProofSubmissions > 0
                              ? (txStats.totalStarkProvingTimeMs / txStats.starkProofSubmissions).toFixed(1)
                              : 0}ms
                          </div>
                        </div>
                        <div className="p-3 bg-slate-900/80 rounded-lg border border-purple-500/10">
                          <div className="text-xs text-gray-400">Security Level</div>
                          <div className="text-xl font-bold text-green-300">{starkConfig.securityBits}-bit</div>
                        </div>
                        <div className="p-3 bg-slate-900/80 rounded-lg border border-purple-500/10">
                          <div className="text-xs text-gray-400">FRI Queries</div>
                          <div className="text-xl font-bold text-amber-300">{starkConfig.numQueries}</div>
                        </div>
                      </div>
                      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                        <div className="flex items-center gap-1.5 text-gray-400">
                          <span className="w-2 h-2 bg-green-400 rounded-full" />
                          Goldilocks Field
                        </div>
                        <div className="flex items-center gap-1.5 text-gray-400">
                          <span className="w-2 h-2 bg-purple-400 rounded-full" />
                          FRI Protocol
                        </div>
                        <div className="flex items-center gap-1.5 text-gray-400">
                          <span className="w-2 h-2 bg-cyan-400 rounded-full" />
                          Transparent Setup
                        </div>
                      </div>
                    </div>

                    {/* Post-Quantum Cryptography Status - v3.7.4 */}
                    <div className="bg-gradient-to-br from-cyan-900/30 to-slate-800/50 border border-cyan-500/30 rounded-xl p-4">
                      <h3 className="text-cyan-300 font-bold mb-3 flex items-center gap-2">
                        <Lock size={16} />
                        Post-Quantum Cryptography
                        <span className="ml-auto text-xs px-2 py-0.5 bg-cyan-500/20 text-cyan-300 rounded-full">NIST Level 5</span>
                      </h3>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="p-3 bg-slate-900/80 rounded-lg border border-cyan-500/10">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-gray-400">Dilithium5 Signatures</span>
                            <span className={`text-xs px-1.5 py-0.5 rounded ${pqStatus?.loaded ? 'bg-green-500/20 text-green-300' : 'bg-yellow-500/20 text-yellow-300'}`}>
                              {pqStatus?.loaded ? 'Active' : 'Loading'}
                            </span>
                          </div>
                          <div className="text-sm text-gray-300">
                            <div className="flex justify-between"><span>Public Key:</span><span className="text-cyan-300">2,592 B</span></div>
                            <div className="flex justify-between"><span>Signature:</span><span className="text-cyan-300">4,627 B</span></div>
                          </div>
                        </div>
                        <div className="p-3 bg-slate-900/80 rounded-lg border border-cyan-500/10">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-gray-400">Kyber1024 Key Exchange</span>
                            <span className={`text-xs px-1.5 py-0.5 rounded ${pqStatus?.loaded ? 'bg-green-500/20 text-green-300' : 'bg-yellow-500/20 text-yellow-300'}`}>
                              {pqStatus?.loaded ? 'Active' : 'Loading'}
                            </span>
                          </div>
                          <div className="text-sm text-gray-300">
                            <div className="flex justify-between"><span>Public Key:</span><span className="text-cyan-300">1,568 B</span></div>
                            <div className="flex justify-between"><span>Shared Secret:</span><span className="text-cyan-300">32 B</span></div>
                          </div>
                        </div>
                      </div>
                      {pqStatus?.keypairGenerated && (
                        <div className="mt-3 p-2 bg-green-500/10 border border-green-500/20 rounded-lg">
                          <div className="flex items-center gap-2 text-xs text-green-300">
                            <CheckCircle size={14} />
                            <span>Hybrid keypair generated: {pqStatus.fingerprint?.slice(0, 16)}...</span>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Transaction Statistics */}
                    <div className="bg-slate-800/50 border border-green-500/20 rounded-xl p-4">
                      <h3 className="text-green-300 font-bold mb-3 flex items-center gap-2">
                        <Zap size={16} />
                        P2P Transaction Stats
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="p-3 bg-slate-900 rounded-lg">
                          <div className="text-xs text-gray-400">Total Submitted</div>
                          <div className="text-xl font-bold text-green-300">{txStats.totalSubmitted}</div>
                        </div>
                        <div className="p-3 bg-slate-900 rounded-lg">
                          <div className="text-xs text-gray-400">P2P Direct</div>
                          <div className="text-xl font-bold text-cyan-300">{txStats.p2pSuccess}</div>
                        </div>
                        <div className="p-3 bg-slate-900 rounded-lg">
                          <div className="text-xs text-gray-400">HTTP Fallback</div>
                          <div className="text-xl font-bold text-amber-300">{txStats.httpSuccess}</div>
                        </div>
                        <div className="p-3 bg-slate-900 rounded-lg">
                          <div className="text-xs text-gray-400">Failed</div>
                          <div className="text-xl font-bold text-red-300">{txStats.failed}</div>
                        </div>
                      </div>
                      {txStats.lastSubmission > 0 && (
                        <div className="mt-2 text-xs text-gray-400">
                          Last submission: {new Date(txStats.lastSubmission).toLocaleTimeString()}
                        </div>
                      )}
                    </div>

                    {/* Decoder Metrics */}
                    <div className="bg-slate-800/50 border border-amber-500/20 rounded-xl p-4">
                      <h3 className="text-amber-300 font-bold mb-3 flex items-center gap-2">
                        <Activity size={16} />
                        Decoder Metrics
                      </h3>
                      <div className="grid grid-cols-3 gap-3">
                        <div className="p-3 bg-slate-900 rounded-lg">
                          <div className="text-xs text-gray-400">Decodes</div>
                          <div className="text-xl font-bold text-amber-300">{DECODER_METRICS.decodeCount}</div>
                        </div>
                        <div className="p-3 bg-slate-900 rounded-lg">
                          <div className="text-xs text-gray-400">Avg Time</div>
                          <div className="text-xl font-bold text-cyan-300">{DECODER_METRICS.avgDecodeTime.toFixed(2)}ms</div>
                        </div>
                        <div className="p-3 bg-slate-900 rounded-lg">
                          <div className="text-xs text-gray-400">Success</div>
                          <div className="text-xl font-bold text-green-300">
                            {DECODER_METRICS.decodeCount > 0
                              ? ((DECODER_METRICS.msgpackSuccesses / DECODER_METRICS.decodeCount) * 100).toFixed(1)
                              : 0}%
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Connection Status */}
                    <div className="bg-slate-800/50 border border-amber-500/20 rounded-xl p-4">
                      <h3 className="text-amber-300 font-bold mb-3 flex items-center gap-2">
                        <Radio size={16} />
                        Connection Status
                      </h3>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center p-2 bg-slate-900 rounded-lg">
                          <span className="text-gray-400 text-sm">Peer ID</span>
                          <code className="text-xs text-amber-300">{peerId ? truncateHash(peerId, 10) : 'N/A'}</code>
                        </div>
                        <div className="flex justify-between items-center p-2 bg-slate-900 rounded-lg">
                          <span className="text-gray-400 text-sm">Status</span>
                          <span className={`flex items-center gap-2 ${isReady ? 'text-green-400' : 'text-yellow-400'}`}>
                            <span className={`w-2 h-2 rounded-full ${isReady ? 'bg-green-400' : 'bg-yellow-400'} animate-pulse`} />
                            {isReady ? 'Connected' : 'Connecting'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center p-2 bg-slate-900 rounded-lg">
                          <span className="text-gray-400 text-sm">Subscribed</span>
                          <span className={isSubscribed ? 'text-green-400' : 'text-yellow-400'}>{isSubscribed ? 'Yes' : 'No'}</span>
                        </div>
                        <div className="flex justify-between items-center p-2 bg-slate-900 rounded-lg">
                          <span className="text-gray-400 text-sm">Encryption</span>
                          <span className="flex items-center gap-1.5 text-purple-300">
                            <Lock size={12} />
                            {pqStatus?.loaded ? 'Hybrid (Noise + PQ)' : 'Noise Protocol'}
                          </span>
                        </div>
                        <div className="p-2 bg-slate-900 rounded-lg">
                          <div className="text-gray-400 text-sm mb-2">Topics:</div>
                          <div className="flex flex-wrap gap-1">
                            {topics.map(topic => (
                              <span key={topic} className="text-xs bg-cyan-500/20 text-cyan-300 px-2 py-0.5 rounded">
                                {topic.split('/').pop()}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Security Features Summary */}
                    <div className="bg-gradient-to-r from-green-900/20 to-cyan-900/20 border border-green-500/20 rounded-xl p-4">
                      <h3 className="text-green-300 font-bold mb-3 flex items-center gap-2">
                        <Shield size={16} />
                        Active Security Features
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                        <div className="flex items-center gap-2 p-2 bg-slate-900/50 rounded-lg">
                          <CheckCircle size={14} className="text-green-400" />
                          <span className="text-xs text-gray-300">Noise Encryption</span>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-slate-900/50 rounded-lg">
                          <CheckCircle size={14} className={pqStatus?.loaded ? 'text-green-400' : 'text-yellow-400'} />
                          <span className="text-xs text-gray-300">Post-Quantum Sigs</span>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-slate-900/50 rounded-lg">
                          <CheckCircle size={14} className="text-green-400" />
                          <span className="text-xs text-gray-300">ZK-STARK Proofs</span>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-slate-900/50 rounded-lg">
                          <CheckCircle size={14} className="text-green-400" />
                          <span className="text-xs text-gray-300">Gossipsub v1.1</span>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-slate-900/50 rounded-lg">
                          <CheckCircle size={14} className="text-green-400" />
                          <span className="text-xs text-gray-300">Fiat-Shamir</span>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-slate-900/50 rounded-lg">
                          <CheckCircle size={14} className="text-green-400" />
                          <span className="text-xs text-gray-300">Transparent Setup</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Pagination */}
              {activeTab !== 'metrics' && totalPages > 1 && (
                <div className="flex-shrink-0 border-t border-amber-500/20 p-3 bg-slate-900/50">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-400">
                      Page {currentPage} of {totalPages}
                    </span>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                        disabled={currentPage === 1}
                        className="p-1.5 hover:bg-amber-500/20 rounded disabled:opacity-50"
                      >
                        <ChevronLeft size={16} className="text-amber-400" />
                      </button>
                      {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                        let pageNum;
                        if (totalPages <= 5) {
                          pageNum = i + 1;
                        } else if (currentPage <= 3) {
                          pageNum = i + 1;
                        } else if (currentPage >= totalPages - 2) {
                          pageNum = totalPages - 4 + i;
                        } else {
                          pageNum = currentPage - 2 + i;
                        }
                        return (
                          <button
                            key={pageNum}
                            onClick={() => setCurrentPage(pageNum)}
                            className={`w-7 h-7 rounded text-xs transition-colors ${
                              currentPage === pageNum
                                ? 'bg-amber-500 text-black font-bold'
                                : 'hover:bg-amber-500/20 text-amber-300'
                            }`}
                          >
                            {pageNum}
                          </button>
                        );
                      })}
                      <button
                        onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                        disabled={currentPage === totalPages}
                        className="p-1.5 hover:bg-amber-500/20 rounded disabled:opacity-50"
                      >
                        <ChevronRight size={16} className="text-amber-400" />
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Live indicator */}
          {isSubscribed && (
            <div className="absolute bottom-4 left-4 flex items-center gap-2 bg-green-500/20 border border-green-500/30 rounded-full px-3 py-1.5">
              <motion.div
                className="w-2 h-2 bg-green-400 rounded-full"
                animate={{ scale: [1, 1.2, 1], opacity: [1, 0.7, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              />
              <span className="text-green-300 text-xs font-medium">LIVE</span>
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
