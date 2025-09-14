import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  Activity,
  Heart,
  Shield,
  Hash,
  Database,
  Cpu,
  Layers,
  BarChart3,
  Atom,
  X,
  Copy,
  Code,
  Info
} from 'lucide-react';
import { qnkAPI } from '../services/api';

interface NetworkStats {
  currentHeight: number;
  currentRound: number;
  currentTps: number;
  totalTransactions: number;
  activePeers: number;
  networkHealth: number;
  consensusParticipation: number;
  mempoolSize: number;
  quantumEntropy: number;
  avgBlockTime: number;
  networkHashRate: number;
  byzantineTolerance: number;
  postQuantumReady: number;
}

// StatCardProps interface removed - no longer needed

interface ActivityItem {
  type: 'transaction' | 'block' | 'vertex' | 'contract';
  id: string;
  amount?: string;
  time: string;
  status?: string;
  contractInfo?: ContractInfo;
}

interface ContractInfo {
  address: string;
  name?: string;
  type: 'evm' | 'wasm' | 'move' | 'native';
  bytecodeSize: number;
  storageUsed: number;
  callCount: number;
  gasUsed: number;
  creator: string;
  creationTime: string;
  isActive: boolean;
  balance: number;
  sourceCode?: string;
  abi?: any[];
}

// StatCard component removed - now using modal interface

const ActivityCard = ({ title, items }: { title: string; items: ActivityItem[] }) => {
  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'transaction': return <Hash className="w-4 h-4 text-quantum-green" />;
      case 'block': return <Database className="w-4 h-4 text-quantum-cyan" />;
      case 'vertex': return <Atom className="w-4 h-4 text-quantum-purple" />;
      case 'contract': return <Code className="w-4 h-4 text-yellow-500" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  return (
    <div className="bg-quantum-indigo/20 backdrop-blur-xl rounded-xl border border-quantum-purple/20 p-6">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Activity className="w-5 h-5 text-quantum-cyan" />
        {title}
      </h3>
      
      <div className="space-y-3 max-h-80 overflow-y-auto">
        {items.map((item, index) => (
          <motion.div
            key={`${item.type}-${item.id}-${index}`}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className="flex items-center justify-between p-3 bg-quantum-dark/30 rounded-lg hover:bg-quantum-dark/50 cursor-pointer transition-colors"
          >
            <div className="flex items-center gap-3">
              {getTypeIcon(item.type)}
              <div>
                <div className="text-white text-sm font-mono">{item.id}</div>
                {item.amount && <div className="text-quantum-green text-xs">{item.amount}</div>}
                {item.status && <div className="text-quantum-purple text-xs">{item.status}</div>}
                {item.contractInfo && (
                  <div className="text-yellow-500 text-xs">
                    {item.contractInfo.name || 'Contract'} ({item.contractInfo.type.toUpperCase()})
                  </div>
                )}
              </div>
            </div>
            <div className="text-gray-400 text-xs">{item.time}</div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

const DetailModal = ({ detail, onClose }: { detail: {type: string, data: any}, onClose: () => void }) => {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const renderDetailContent = () => {
    switch (detail.type) {
      case 'block':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Block Height</div>
                <div className="text-lg font-mono">{detail.data?.height || 'N/A'}</div>
              </div>
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Transactions</div>
                <div className="text-lg font-mono">{detail.data?.tx_count || 'N/A'}</div>
              </div>
            </div>
            <div className="p-3 bg-quantum-dark/30 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-400">Block Hash</div>
                <Copy className="w-4 h-4 text-gray-400 cursor-pointer hover:text-white" 
                      onClick={() => copyToClipboard(detail.data?.hash || '')} />
              </div>
              <div className="text-sm font-mono break-all">{detail.data?.hash || 'N/A'}</div>
            </div>
          </div>
        );

      case 'transaction':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Amount</div>
                <div className="text-lg font-mono text-quantum-green">{detail.data?.amount || 'N/A'} QNK</div>
              </div>
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Status</div>
                <div className="text-lg capitalize">{detail.data?.status || 'pending'}</div>
              </div>
            </div>
            <div className="p-3 bg-quantum-dark/30 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-400">Transaction Hash</div>
                <Copy className="w-4 h-4 text-gray-400 cursor-pointer hover:text-white" 
                      onClick={() => copyToClipboard(detail.data?.hash || '')} />
              </div>
              <div className="text-sm font-mono break-all">{detail.data?.hash || 'N/A'}</div>
            </div>
          </div>
        );

      case 'performance':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Current TPS</div>
                <div className="text-lg font-mono text-quantum-cyan">{detail.data?.current_tps?.toFixed(1) || 'N/A'}</div>
              </div>
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Peak TPS</div>
                <div className="text-lg font-mono">{detail.data?.peak_tps?.toFixed(0) || 'N/A'}</div>
              </div>
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Avg Latency</div>
                <div className="text-lg font-mono">{detail.data?.avg_latency_ms || 'N/A'}ms</div>
              </div>
            </div>
          </div>
        );

      case 'contract':
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Contract Type</div>
                <div className="text-lg font-mono text-yellow-500 uppercase">{detail.data?.type || 'N/A'}</div>
              </div>
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Status</div>
                <div className={`text-lg capitalize ${detail.data?.isActive ? 'text-quantum-green' : 'text-red-500'}`}>
                  {detail.data?.isActive ? 'Active' : 'Inactive'}
                </div>
              </div>
            </div>
            
            <div className="grid grid-cols-3 gap-4">
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Balance</div>
                <div className="text-lg font-mono text-quantum-green">{detail.data?.balance || 0} QNK</div>
              </div>
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Call Count</div>
                <div className="text-lg font-mono">{detail.data?.callCount || 0}</div>
              </div>
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Gas Used</div>
                <div className="text-lg font-mono">{detail.data?.gasUsed || 0}</div>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Bytecode Size</div>
                <div className="text-lg font-mono">{detail.data?.bytecodeSize || 0} bytes</div>
              </div>
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Storage Used</div>
                <div className="text-lg font-mono">{detail.data?.storageUsed || 0} KB</div>
              </div>
            </div>
            
            <div className="p-3 bg-quantum-dark/30 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-400">Contract Address</div>
                <Copy className="w-4 h-4 text-gray-400 cursor-pointer hover:text-white" 
                      onClick={() => copyToClipboard(detail.data?.address || '')} />
              </div>
              <div className="text-sm font-mono break-all">{detail.data?.address || 'N/A'}</div>
            </div>
            
            <div className="p-3 bg-quantum-dark/30 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-400">Creator</div>
                <Copy className="w-4 h-4 text-gray-400 cursor-pointer hover:text-white" 
                      onClick={() => copyToClipboard(detail.data?.creator || '')} />
              </div>
              <div className="text-sm font-mono break-all">{detail.data?.creator || 'N/A'}</div>
            </div>
            
            {detail.data?.name && (
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400">Contract Name</div>
                <div className="text-lg font-semibold text-white">{detail.data.name}</div>
              </div>
            )}
            
            {detail.data?.sourceCode && (
              <div className="p-3 bg-quantum-dark/30 rounded-lg">
                <div className="text-sm text-gray-400 mb-2">Source Code</div>
                <pre className="text-xs overflow-auto max-h-40 bg-black/20 p-3 rounded">
                  {detail.data.sourceCode}
                </pre>
              </div>
            )}
          </div>
        );

      default:
        return (
          <div className="p-4 bg-quantum-dark/30 rounded-lg">
            <pre className="text-sm overflow-auto">
              {JSON.stringify(detail.data, null, 2)}
            </pre>
          </div>
        );
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-quantum-indigo/90 backdrop-blur-xl rounded-xl border border-quantum-purple/30 p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-white capitalize">
            {detail.type} Details
          </h3>
          <button
            onClick={onClose}
            className="p-2 hover:bg-quantum-purple/20 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-400 hover:text-white" />
          </button>
        </div>

        {renderDetailContent()}

        <div className="mt-6 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-quantum-purple/20 text-white rounded-lg hover:bg-quantum-purple/30 transition-colors"
          >
            Close
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

const StatsModal = ({ networkStats, liveMetrics, onClose }: { 
  networkStats: NetworkStats, 
  liveMetrics: any, 
  onClose: () => void 
}) => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-quantum-indigo/90 backdrop-blur-xl rounded-xl border border-quantum-purple/30 p-6 max-w-6xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-2xl font-bold text-white flex items-center gap-3">
            <BarChart3 className="w-6 h-6 text-quantum-cyan" />
            📊 Complete Network Statistics
          </h3>
          <button
            onClick={onClose}
            className="p-2 hover:bg-quantum-purple/20 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-400 hover:text-white" />
          </button>
        </div>

        {/* Core Network Stats */}
        <div className="space-y-6">
          <div>
            <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Database className="w-5 h-5 text-quantum-green" />
              Core Network Metrics
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Current Height</div>
                <div className="text-2xl font-bold text-quantum-cyan">{networkStats.currentHeight}</div>
                <div className="text-xs text-gray-500">Latest committed block</div>
              </div>
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Consensus Round</div>
                <div className="text-2xl font-bold text-quantum-purple">{networkStats.currentRound}</div>
                <div className="text-xs text-gray-500">DAG-Knight round</div>
              </div>
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Current TPS</div>
                <div className="text-2xl font-bold text-quantum-green">{networkStats.currentTps.toFixed(1)}</div>
                <div className="text-xs text-gray-500">Transactions per second</div>
              </div>
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Total Transactions</div>
                <div className="text-2xl font-bold text-white">{networkStats.totalTransactions.toLocaleString()}</div>
                <div className="text-xs text-gray-500">Network lifetime</div>
              </div>
            </div>
          </div>

          {/* Network Health */}
          <div>
            <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Heart className="w-5 h-5 text-red-500" />
              Network Health & Peers
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Active Peers</div>
                <div className="text-2xl font-bold text-quantum-cyan">{networkStats.activePeers}</div>
                <div className="text-xs text-gray-500">Connected validators</div>
              </div>
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Network Health</div>
                <div className="text-2xl font-bold text-quantum-green">{(networkStats.networkHealth * 100).toFixed(1)}%</div>
                <div className="text-xs text-gray-500">Overall score</div>
              </div>
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Consensus Participation</div>
                <div className="text-2xl font-bold text-quantum-purple">{(networkStats.consensusParticipation * 100).toFixed(1)}%</div>
                <div className="text-xs text-gray-500">Validator participation</div>
              </div>
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Mempool Size</div>
                <div className="text-2xl font-bold text-yellow-500">{networkStats.mempoolSize}</div>
                <div className="text-xs text-gray-500">Pending transactions</div>
              </div>
            </div>
          </div>

          {/* Quantum & Security */}
          <div>
            <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Shield className="w-5 h-5 text-quantum-purple" />
              Quantum & Security Metrics
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Quantum Entropy</div>
                <div className="text-2xl font-bold text-quantum-green">{(networkStats.quantumEntropy * 100).toFixed(1)}%</div>
                <div className="text-xs text-gray-500">Randomness quality</div>
              </div>
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Post-Quantum Ready</div>
                <div className="text-2xl font-bold text-quantum-cyan">{(networkStats.postQuantumReady * 100).toFixed(1)}%</div>
                <div className="text-xs text-gray-500">PQ crypto adoption</div>
              </div>
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Byzantine Tolerance</div>
                <div className="text-2xl font-bold text-quantum-purple">{(networkStats.byzantineTolerance * 100).toFixed(1)}%</div>
                <div className="text-xs text-gray-500">Fault tolerance (f=1)</div>
              </div>
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">VDF Computations</div>
                <div className="text-2xl font-bold text-yellow-500">{liveMetrics.vdfComputations}</div>
                <div className="text-xs text-gray-500">Quantum anchor elections</div>
              </div>
            </div>
          </div>

          {/* Resource Usage */}
          <div>
            <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-quantum-cyan" />
              Resource Usage & Performance
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Memory Usage</div>
                <div className="text-2xl font-bold text-quantum-green">{liveMetrics.memoryUsage.toFixed(1)}%</div>
                <div className="text-xs text-gray-500">System memory</div>
              </div>
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Data Storage</div>
                <div className="text-2xl font-bold text-quantum-cyan">{liveMetrics.dataStorage.toFixed(1)} GB</div>
                <div className="text-xs text-gray-500">Total chain data</div>
              </div>
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Average Block Time</div>
                <div className="text-2xl font-bold text-quantum-purple">{networkStats.avgBlockTime.toFixed(1)}s</div>
                <div className="text-xs text-gray-500">Finalization time</div>
              </div>
              <div className="p-4 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                <div className="text-sm text-gray-400">Network Hash Rate</div>
                <div className="text-2xl font-bold text-yellow-500">{networkStats.networkHashRate.toLocaleString()} H/s</div>
                <div className="text-xs text-gray-500">Compute power</div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-6 py-3 bg-quantum-purple/20 text-white rounded-lg hover:bg-quantum-purple/30 transition-colors font-semibold"
          >
            Close Statistics
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default function ExplorerScreen() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDetail, setSelectedDetail] = useState<{type: string, data: any} | null>(null);
  const [showStatsModal, setShowStatsModal] = useState(false);
  const [networkStats, setNetworkStats] = useState<NetworkStats>({
    currentHeight: 0,
    currentRound: 0,
    currentTps: 0,
    totalTransactions: 0,
    activePeers: 0,
    networkHealth: 0,
    consensusParticipation: 0,
    mempoolSize: 0,
    quantumEntropy: 0,
    avgBlockTime: 0,
    networkHashRate: 0,
    byzantineTolerance: 0,
    postQuantumReady: 0
  });

  const [recentActivity, setRecentActivity] = useState({
    transactions: [] as ActivityItem[],
    blocks: [] as ActivityItem[],
    vertices: [] as ActivityItem[],
    contracts: [] as ActivityItem[]
  });
  const [liveMetrics, setLiveMetrics] = useState({
    vdfComputations: 0,
    memoryUsage: 0,
    dataStorage: 0,
    realTimeTps: 0,
    realTimeLatency: 0
  });

  useEffect(() => {
    // Fetch available data and provide fallbacks for missing endpoints
    const fetchAllData = async () => {
      try {
        // Only fetch available endpoint
        const nodeStatus = await qnkAPI.getNodeStatus();

        // Update network stats with available data and computed values
        setNetworkStats({
          currentHeight: nodeStatus.data?.current_height || 0,
          currentRound: nodeStatus.data?.current_round || 0,
          currentTps: nodeStatus.data?.tps_current || 0,
          totalTransactions: (nodeStatus.data?.current_height || 0) * 100, // Estimate
          activePeers: nodeStatus.data?.connected_peers || 0,
          networkHealth: nodeStatus.data?.is_validator ? 0.95 : 0.8,
          consensusParticipation: nodeStatus.data?.is_validator ? 1.0 : 0.0,
          mempoolSize: nodeStatus.data?.tx_pool_size || 0,
          quantumEntropy: 0.92, // Static high value for demo
          avgBlockTime: nodeStatus.data?.last_block_time ? nodeStatus.data.last_block_time / 1000 : 2.5,
          networkHashRate: (nodeStatus.data?.tps_current || 0) * 1000,
          byzantineTolerance: (nodeStatus.data?.connected_peers || 0) >= 4 ? 0.95 : 0.75,
          postQuantumReady: 0.88 // Static value for demo
        });

        // Generate mock activity data for demonstration

        // Generate mock activity data for demonstration
        const now = new Date();
        setRecentActivity({
          transactions: [
            {
              type: 'transaction' as const,
              id: 'tx_a1b2c3d4e5f6',
              amount: '1000 QNK',
              time: '2 min ago'
            },
            {
              type: 'transaction' as const,
              id: 'tx_f6e5d4c3b2a1',
              amount: '250 QNK',
              time: '5 min ago'
            }
          ],
          blocks: [
            {
              type: 'block' as const,
              id: String(nodeStatus.data?.current_height || 0),
              amount: '15 txs',
              time: '1 min ago'
            },
            {
              type: 'block' as const,
              id: String((nodeStatus.data?.current_height || 0) - 1),
              amount: '22 txs',
              time: '3 min ago'
            }
          ],
          vertices: [
            {
              type: 'vertex' as const,
              id: `vtx_round_${nodeStatus.data?.current_round || 0}`,
              time: '30 sec ago',
              status: 'committed'
            },
            {
              type: 'vertex' as const,
              id: `vtx_round_${(nodeStatus.data?.current_round || 0) - 1}`,
              time: '2 min ago',
              status: 'confirmed'
            }
          ],
          contracts: [
            {
              type: 'contract' as const,
              id: '0x742d35Cc631C0532925a3b8D922B45c',
              time: '1 hour ago',
              contractInfo: {
                address: '0x742d35Cc631C0532925a3b8D922B45c',
                name: 'Q-DeFi Pool',
                type: 'evm' as const,
                bytecodeSize: 24576,
                storageUsed: 1024,
                callCount: 47,
                gasUsed: 892000,
                creator: '0x1234567890123456789012345678901234567890',
                creationTime: new Date(now.getTime() - 3600000).toISOString(),
                isActive: true,
                balance: 15000
              }
            }
          ]
        });

        // Update live metrics from computed values
        setLiveMetrics({
          vdfComputations: Math.max(1, Math.floor(nodeStatus.data?.current_round || 0 / 10)),
          memoryUsage: Math.min(85, 45 + (nodeStatus.data?.tx_pool_size || 0) / 100),
          dataStorage: Math.max(1.2, (nodeStatus.data?.current_height || 0) * 0.01),
          realTimeTps: nodeStatus.data?.tps_current || 0,
          realTimeLatency: (nodeStatus.data?.connected_peers || 0) >= 4 ? 12 : 45
        });

      } catch (error) {
        console.error('Failed to fetch real data:', error);
        // On error, keep current state (all zeros initially)
      }
    };

    fetchAllData();
    const interval = setInterval(fetchAllData, 5000); // Update every 5 seconds for real-time feel

    return () => clearInterval(interval);
  }, []);

  const handleSearch = async (query: string) => {
    setSearchQuery(query);
    
    if (!query.trim()) return;
    
    try {
      // Determine search type based on query format
      let searchType = '';
      if (query.match(/^tx_[a-f0-9]+/i)) searchType = 'transaction';
      else if (query.match(/^vtx_[a-f0-9]+/i)) searchType = 'vertex';
      else if (query.match(/^0x[a-f0-9]{40}$/i)) searchType = 'contract'; // EVM contract address
      else if (query.match(/^qnk[a-z0-9]{39}$/i)) searchType = 'contract'; // Q-NarwhalKnight contract address
      else if (query.match(/^\d+$/)) searchType = 'block';
      else if (query.match(/^[a-f0-9]{64}$/i)) searchType = 'hash';
      
      // Simple search handling - just log the search for now
      console.log(`Searching for: ${query} (type: ${searchType})`);
      
      // For demo purposes, show a mock detail based on search type
      if (searchType === 'contract') {
        setSelectedDetail({
          type: 'contract',
          data: {
            address: query,
            name: 'Demo Contract',
            type: 'evm',
            isActive: true,
            balance: 5000,
            callCount: 23,
            gasUsed: 450000,
            bytecodeSize: 12288,
            storageUsed: 512,
            creator: '0x1234567890123456789012345678901234567890'
          }
        });
      } else if (searchType === 'transaction') {
        setSelectedDetail({
          type: 'transaction',
          data: {
            hash: query,
            amount: 1000,
            status: 'confirmed'
          }
        });
      }
    } catch (error) {
      console.error('Search failed:', error);
    }
  };

  // handleStatClick removed - using modal interface now

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-3xl font-bold bg-gradient-to-r from-quantum-cyan to-quantum-purple bg-clip-text text-transparent"
        >
          🔍 Q-NarwhalKnight Explorer
        </motion.h1>

        {/* Enhanced Search Bar */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="relative max-w-md"
        >
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search tx, block, contract address, vertex ID..."
            className="w-full pl-10 pr-4 py-3 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-quantum-cyan transition-colors"
          />
        </motion.div>
      </div>

      {/* Network Statistics Button */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <h2 className="text-xl font-semibold text-white">📊 Network Overview</h2>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowStatsModal(true)}
            className="px-6 py-3 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300 flex items-center gap-3"
          >
            <BarChart3 className="w-5 h-5" />
            View Complete Statistics
            <Info className="w-4 h-4" />
          </motion.button>
        </div>
        
        {/* Quick Stats Preview */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-quantum-indigo/20 backdrop-blur-xl rounded-lg border border-quantum-purple/20 p-4 text-center">
            <div className="text-2xl font-bold text-quantum-cyan">{networkStats.currentHeight}</div>
            <div className="text-sm text-gray-400">Current Height</div>
          </div>
          <div className="bg-quantum-indigo/20 backdrop-blur-xl rounded-lg border border-quantum-purple/20 p-4 text-center">
            <div className="text-2xl font-bold text-quantum-green">{networkStats.currentTps.toFixed(1)}</div>
            <div className="text-sm text-gray-400">TPS</div>
          </div>
          <div className="bg-quantum-indigo/20 backdrop-blur-xl rounded-lg border border-quantum-purple/20 p-4 text-center">
            <div className="text-2xl font-bold text-quantum-purple">{networkStats.activePeers}</div>
            <div className="text-sm text-gray-400">Active Peers</div>
          </div>
          <div className="bg-quantum-indigo/20 backdrop-blur-xl rounded-lg border border-quantum-purple/20 p-4 text-center">
            <div className="text-2xl font-bold text-yellow-500">{(networkStats.networkHealth * 100).toFixed(0)}%</div>
            <div className="text-sm text-gray-400">Health</div>
          </div>
        </div>
      </motion.section>


      {/* Live Visualizations Placeholder */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <h2 className="text-xl font-semibold text-white mb-6">📈 Live Network Visualizations</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-quantum-indigo/20 backdrop-blur-xl rounded-xl border border-quantum-purple/20 p-6 h-80">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Layers className="w-5 h-5 text-quantum-cyan" />
              🕸️ DAG-Knight Consensus
            </h3>
            <div className="relative h-60 flex items-center justify-center">
              {/* Live DAG Network Visualization */}
              <div className="relative w-48 h-48">
                {/* Central node */}
                <motion.div
                  className="absolute top-1/2 left-1/2 w-8 h-8 bg-quantum-cyan rounded-full transform -translate-x-1/2 -translate-y-1/2 z-10"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
                
                {/* Surrounding nodes */}
                {[0, 1, 2, 3].map((index) => {
                  const angle = (index * 90) * (Math.PI / 180);
                  const radius = 60;
                  const x = Math.cos(angle) * radius;
                  const y = Math.sin(angle) * radius;
                  
                  return (
                    <motion.div
                      key={index}
                      className="absolute w-6 h-6 bg-quantum-purple rounded-full"
                      style={{
                        top: '50%',
                        left: '50%',
                        transform: `translate(${x - 12}px, ${y - 12}px)`
                      }}
                      animate={{ 
                        opacity: [0.5, 1, 0.5],
                        scale: [0.8, 1, 0.8]
                      }}
                      transition={{ 
                        duration: 1.5, 
                        repeat: Infinity,
                        delay: index * 0.3
                      }}
                    />
                  );
                })}
                
                {/* Connection lines */}
                <svg className="absolute inset-0 w-full h-full">
                  {[0, 1, 2, 3].map((index) => {
                    const angle = (index * 90) * (Math.PI / 180);
                    const radius = 60;
                    const x = Math.cos(angle) * radius + 96; // 96 = half of 192px
                    const y = Math.sin(angle) * radius + 96;
                    
                    return (
                      <motion.line
                        key={index}
                        x1="96" y1="96"
                        x2={x} y2={y}
                        stroke="url(#gradient)"
                        strokeWidth="1"
                        animate={{ opacity: [0.3, 0.8, 0.3] }}
                        transition={{ duration: 2, repeat: Infinity, delay: index * 0.2 }}
                      />
                    );
                  })}
                  <defs>
                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#00f5ff" />
                      <stop offset="100%" stopColor="#8b5cf6" />
                    </linearGradient>
                  </defs>
                </svg>
              </div>
              
              <div className="absolute bottom-4 right-4 text-right text-xs text-gray-400">
                <div>Vertices: {networkStats.currentRound}</div>
                <div>Round: {networkStats.currentRound}</div>
                <div>Active: {networkStats.activePeers} nodes</div>
              </div>
            </div>
          </div>
          
          <div className="bg-quantum-indigo/20 backdrop-blur-xl rounded-xl border border-quantum-purple/20 p-6 h-80">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-quantum-green" />
              📊 Performance Metrics
            </h3>
            <div className="relative h-60 p-4">
              {/* Live Performance Chart */}
              <div className="flex items-end justify-between h-full gap-2">
                {[65, 80, 45, 90, 70, 85, 95, 88, 92, 78, 87, 94].map((height, index) => (
                  <motion.div
                    key={index}
                    className="bg-gradient-to-t from-quantum-green/60 to-quantum-cyan/60 rounded-t flex-1 min-w-0"
                    initial={{ height: 0 }}
                    animate={{ height: `${height}%` }}
                    transition={{ 
                      duration: 1.5, 
                      delay: index * 0.1,
                      repeat: Infinity,
                      repeatType: 'reverse',
                      repeatDelay: 2
                    }}
                  />
                ))}
              </div>
              
              {/* Chart Labels */}
              <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-gray-500 px-4">
                <span>00:00</span>
                <span>06:00</span>
                <span>12:00</span>
                <span>18:00</span>
                <span>24:00</span>
              </div>
              
              {/* Live Metrics Overlay */}
              <div className="absolute top-4 right-4 bg-quantum-dark/60 rounded-lg p-3 text-right">
                <div className="text-quantum-green text-xl font-bold">
                  {liveMetrics.realTimeTps.toFixed(0)}
                </div>
                <div className="text-xs text-gray-400">TPS</div>
                <div className="text-quantum-cyan text-sm font-semibold mt-1">
                  {liveMetrics.realTimeLatency.toFixed(0)}ms
                </div>
                <div className="text-xs text-gray-400">Latency</div>
              </div>
            </div>
          </div>
        </div>
      </motion.section>

      {/* Recent Activity */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <h2 className="text-xl font-semibold text-white mb-6">🔥 Recent Network Activity</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6">
          <ActivityCard
            title="💸 Recent Transactions"
            items={recentActivity.transactions}
          />
          <ActivityCard
            title="🧱 Recent Blocks"
            items={recentActivity.blocks}
          />
          <ActivityCard
            title="⚛️ Recent Vertices"
            items={recentActivity.vertices}
          />
          <ActivityCard
            title="📜 Smart Contracts"
            items={recentActivity.contracts}
          />
        </div>
      </motion.section>

      {/* Detail Modal */}
      <AnimatePresence>
        {selectedDetail && (
          <DetailModal 
            detail={selectedDetail} 
            onClose={() => setSelectedDetail(null)} 
          />
        )}
        {showStatsModal && (
          <StatsModal 
            networkStats={networkStats}
            liveMetrics={liveMetrics}
            onClose={() => setShowStatsModal(false)} 
          />
        )}
      </AnimatePresence>
    </div>
  );
}