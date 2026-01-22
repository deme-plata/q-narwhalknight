import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Pickaxe, Download, Cpu, Zap, Award, TrendingUp, AlertCircle, ExternalLink, Terminal, Users, User, Link as LinkIcon, RefreshCw, Clock, Activity, DollarSign, Copy, Check, Server } from 'lucide-react';
import MiningDashboard from './MiningDashboard';

type MiningTab = 'pool' | 'solo' | 'downloads' | 'links';

// Pool API Types
interface PoolStats {
  name: string;
  version: string;
  hashrate: number;
  workers: number;
  blocks_found: number;
  current_round: number;
  difficulty: number;
  fee_bps: number;
  min_payout: number;
  shares_this_round: number;
  uptime_seconds: number;
  stratum_port: number;
}

interface WorkerStats {
  worker_id: string;
  wallet_address: string;
  hashrate: number;
  difficulty: number;
  shares_submitted: number;
  shares_stale: number;
  shares_invalid: number;
  blocks_found: number;
  last_share_time: number;
  connected_since: number;
  is_connected: boolean;
}

interface PendingBalance {
  wallet_address: string;
  pending_balance: number;
  estimated_payout: string | null;
}

interface PayoutEntry {
  id: number;
  amount: number;
  tx_hash: string | null;
  status: string;
  timestamp: number;
}

export default function MiningScreen() {
  const [activeTab, setActiveTab] = useState<MiningTab>('solo');
  const walletAddress = localStorage.getItem('walletAddress') || '';

  // Pool Mining State
  const [poolStats, setPoolStats] = useState<PoolStats | null>(null);
  const [myWorkers, setMyWorkers] = useState<WorkerStats[]>([]);
  const [pendingBalance, setPendingBalance] = useState<PendingBalance | null>(null);
  const [recentPayouts, setRecentPayouts] = useState<PayoutEntry[]>([]);
  const [poolLoading, setPoolLoading] = useState(false);
  const [poolError, setPoolError] = useState<string | null>(null);
  const [copiedStratum, setCopiedStratum] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  // Fetch pool data
  const fetchPoolData = useCallback(async () => {
    setPoolLoading(true);
    setPoolError(null);

    try {
      // Fetch pool stats
      const statsRes = await fetch('/api/v1/pool/stats');
      if (statsRes.ok) {
        const stats = await statsRes.json();
        setPoolStats(stats);
      }

      // Fetch workers for current wallet
      if (walletAddress) {
        const workersRes = await fetch(`/api/v1/pool/workers?wallet=${walletAddress}`);
        if (workersRes.ok) {
          const workers = await workersRes.json();
          setMyWorkers(workers);
        }

        // Fetch pending balance
        const balanceRes = await fetch(`/api/v1/pool/balance/${walletAddress}`);
        if (balanceRes.ok) {
          const balance = await balanceRes.json();
          setPendingBalance(balance);
        }
      }

      // Fetch recent payouts
      const payoutsRes = await fetch('/api/v1/pool/payouts?limit=10');
      if (payoutsRes.ok) {
        const payouts = await payoutsRes.json();
        setRecentPayouts(payouts);
      }

      setLastRefresh(new Date());
    } catch (err) {
      setPoolError('Pool service unavailable - the pool may not be running');
    } finally {
      setPoolLoading(false);
    }
  }, [walletAddress]);

  // Auto-refresh pool data every 30 seconds when on pool tab
  useEffect(() => {
    if (activeTab === 'pool') {
      fetchPoolData();
      const interval = setInterval(fetchPoolData, 30000);
      return () => clearInterval(interval);
    }
  }, [activeTab, fetchPoolData]);

  // Format hashrate
  const formatHashrate = (h: number): string => {
    if (h >= 1e12) return `${(h / 1e12).toFixed(2)} TH/s`;
    if (h >= 1e9) return `${(h / 1e9).toFixed(2)} GH/s`;
    if (h >= 1e6) return `${(h / 1e6).toFixed(2)} MH/s`;
    if (h >= 1e3) return `${(h / 1e3).toFixed(2)} KH/s`;
    return `${h.toFixed(2)} H/s`;
  };

  // Format QUG amount
  const formatQUG = (atomic: number): string => {
    return (atomic / 1e9).toFixed(4);
  };

  // Format uptime
  const formatUptime = (seconds: number): string => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  };

  // Copy stratum URL
  const copyStratumUrl = () => {
    const url = `stratum+tcp://pool.quillon.xyz:${poolStats?.stratum_port || 3333}`;
    navigator.clipboard.writeText(url);
    setCopiedStratum(true);
    setTimeout(() => setCopiedStratum(false), 2000);
  };

  const handleDownloadMiner = (platform: 'linux' | 'windows' | 'macos-intel' | 'macos-arm') => {
    // Link to download the miner binary
    if (platform === 'windows') {
      window.open('/downloads/q-miner-windows-x64.exe', '_blank');
    } else if (platform === 'macos-intel') {
      window.open('/downloads/q-miner-macos-x64', '_blank');
    } else if (platform === 'macos-arm') {
      window.open('/downloads/q-miner-macos-arm64', '_blank');
    } else {
      window.open('/downloads/q-miner-linux-x64', '_blank');
    }
  };

  const copyCommand = (command: string) => {
    navigator.clipboard.writeText(command);
  };

  // v2.7.1-beta: CRITICAL FIX - Use dynamic server URL for decentralized mining
  // Previously hardcoded to bootstrap server, which broke mining on user's own nodes
  // Now uses the current host (e.g., localhost:8080 or user's node IP)
  const currentServerUrl = typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.host}`
    : 'http://localhost:8080';

  const miningCommand = `./q-miner --mode solo --wallet ${walletAddress} --threads 4 --intensity 7 --server ${currentServerUrl}`;

  const tabs = [
    { id: 'pool' as MiningTab, label: 'Pool Mining', icon: Users, color: 'quantum-purple' },
    { id: 'solo' as MiningTab, label: 'Solo Mining', icon: User, color: 'quantum-cyan' },
    { id: 'downloads' as MiningTab, label: 'Downloads', icon: Download, color: 'quantum-green' },
    { id: 'links' as MiningTab, label: 'Links', icon: LinkIcon, color: 'quantum-orange' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="p-3 rainbow-box rounded-xl">
          <Pickaxe className="w-8 h-8 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-quantum-cyan to-quantum-purple bg-clip-text text-transparent">
            Quantum Mining
          </h1>
          <p className="text-gray-400">
            Mine QUG with Austrian Economics & DAG-Knight VDF
          </p>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex flex-wrap gap-2 bg-quantum-dark/50 rounded-xl p-2 border border-quantum-purple/20">
        {tabs.map((tab) => (
          <motion.button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
              activeTab === tab.id
                ? `bg-${tab.color}/20 text-${tab.color} border border-${tab.color}/50`
                : 'text-gray-400 hover:text-white hover:bg-quantum-indigo/30'
            }`}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
          </motion.button>
        ))}
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'pool' && (
          <motion.div
            key="pool"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Pool Header with Refresh */}
            <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 bg-quantum-purple/20 rounded-xl">
                    <Users className="w-8 h-8 text-quantum-purple" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-white">Pool Mining Dashboard</h2>
                    <p className="text-gray-400">
                      {poolStats ? `${poolStats.name} - ${poolStats.version}` : 'PPLNS Stratum Mining Pool'}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {lastRefresh && (
                    <span className="text-xs text-gray-500">
                      Updated {lastRefresh.toLocaleTimeString()}
                    </span>
                  )}
                  <motion.button
                    onClick={fetchPoolData}
                    disabled={poolLoading}
                    className="p-2 bg-quantum-purple/20 hover:bg-quantum-purple/30 rounded-lg transition-colors disabled:opacity-50"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <RefreshCw className={`w-5 h-5 text-quantum-purple ${poolLoading ? 'animate-spin' : ''}`} />
                  </motion.button>
                </div>
              </div>

              {/* Error State */}
              {poolError && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6">
                  <div className="flex items-center gap-3">
                    <AlertCircle className="w-5 h-5 text-red-400" />
                    <div>
                      <p className="text-red-400 font-medium">{poolError}</p>
                      <p className="text-gray-400 text-sm mt-1">
                        The mining pool feature requires the pool module to be enabled. Solo mining is always available.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Pool Stats Grid */}
              {poolStats && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-cyan/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-4 h-4 text-quantum-cyan" />
                      <span className="text-gray-400 text-sm">Pool Hashrate</span>
                    </div>
                    <p className="text-xl font-bold text-quantum-cyan">{formatHashrate(poolStats.hashrate)}</p>
                  </div>
                  <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-green/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Users className="w-4 h-4 text-quantum-green" />
                      <span className="text-gray-400 text-sm">Active Workers</span>
                    </div>
                    <p className="text-xl font-bold text-quantum-green">{poolStats.workers}</p>
                  </div>
                  <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-purple/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Award className="w-4 h-4 text-quantum-purple" />
                      <span className="text-gray-400 text-sm">Blocks Found</span>
                    </div>
                    <p className="text-xl font-bold text-quantum-purple">{poolStats.blocks_found}</p>
                  </div>
                  <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-orange/20">
                    <div className="flex items-center gap-2 mb-2">
                      <Clock className="w-4 h-4 text-quantum-orange" />
                      <span className="text-gray-400 text-sm">Uptime</span>
                    </div>
                    <p className="text-xl font-bold text-quantum-orange">{formatUptime(poolStats.uptime_seconds)}</p>
                  </div>
                </div>
              )}

              {/* Pool Info Cards */}
              {poolStats && (
                <div className="grid md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-purple/20">
                    <h4 className="font-bold text-white mb-3 flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-quantum-purple" />
                      Current Round #{poolStats.current_round}
                    </h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Shares</span>
                        <span className="text-white font-mono">{poolStats.shares_this_round.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Difficulty</span>
                        <span className="text-white font-mono">{poolStats.difficulty.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>
                  <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-green/20">
                    <h4 className="font-bold text-white mb-3 flex items-center gap-2">
                      <DollarSign className="w-4 h-4 text-quantum-green" />
                      Pool Fees
                    </h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Pool Fee</span>
                        <span className="text-quantum-green font-mono">{(poolStats.fee_bps / 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Min Payout</span>
                        <span className="text-white font-mono">{formatQUG(poolStats.min_payout)} QUG</span>
                      </div>
                    </div>
                  </div>
                  <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-cyan/20">
                    <h4 className="font-bold text-white mb-3 flex items-center gap-2">
                      <Zap className="w-4 h-4 text-quantum-cyan" />
                      Features
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      <span className="text-xs bg-quantum-purple/20 text-quantum-purple px-2 py-1 rounded">PPLNS</span>
                      <span className="text-xs bg-quantum-cyan/20 text-quantum-cyan px-2 py-1 rounded">Vardiff</span>
                      <span className="text-xs bg-quantum-green/20 text-quantum-green px-2 py-1 rounded">Stratum</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Stratum Connection Info */}
              <div className="bg-gradient-to-r from-quantum-purple/10 to-quantum-cyan/10 rounded-xl p-5 border border-quantum-purple/30 mb-6">
                <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                  <Server className="w-5 h-5 text-quantum-purple" />
                  Stratum Connection
                </h4>
                <div className="space-y-3">
                  <div className="bg-quantum-dark/50 rounded-lg p-3 flex items-center justify-between">
                    <div>
                      <span className="text-gray-500 text-sm block">Stratum URL</span>
                      <code className="text-quantum-cyan font-mono">
                        stratum+tcp://pool.quillon.xyz:{poolStats?.stratum_port || 3333}
                      </code>
                    </div>
                    <motion.button
                      onClick={copyStratumUrl}
                      className="p-2 bg-quantum-purple/20 hover:bg-quantum-purple/30 rounded-lg transition-colors"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      {copiedStratum ? (
                        <Check className="w-4 h-4 text-quantum-green" />
                      ) : (
                        <Copy className="w-4 h-4 text-quantum-purple" />
                      )}
                    </motion.button>
                  </div>
                  <div className="bg-quantum-dark/50 rounded-lg p-3">
                    <span className="text-gray-500 text-sm block">Worker Name Format</span>
                    <code className="text-quantum-purple font-mono">
                      {walletAddress ? `${walletAddress.slice(0, 20)}...` : 'YOUR_WALLET_ADDRESS'}.rig1
                    </code>
                  </div>
                  <div className="bg-quantum-dark/50 rounded-lg p-3">
                    <span className="text-gray-500 text-sm block">Example Miner Command</span>
                    <code className="text-quantum-green font-mono text-sm block overflow-x-auto">
                      ./q-miner --mode pool --server stratum+tcp://pool.quillon.xyz:{poolStats?.stratum_port || 3333} --wallet {walletAddress || 'YOUR_WALLET'}.rig1
                    </code>
                  </div>
                </div>
              </div>

              {/* My Workers Section */}
              {walletAddress && myWorkers.length > 0 && (
                <div className="mb-6">
                  <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                    <User className="w-5 h-5 text-quantum-cyan" />
                    My Workers ({myWorkers.length})
                  </h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-gray-400 border-b border-quantum-purple/20">
                          <th className="text-left py-2 px-3">Worker</th>
                          <th className="text-right py-2 px-3">Hashrate</th>
                          <th className="text-right py-2 px-3">Shares</th>
                          <th className="text-right py-2 px-3">Stale</th>
                          <th className="text-center py-2 px-3">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {myWorkers.map((worker) => (
                          <tr key={worker.worker_id} className="border-b border-quantum-dark/50 hover:bg-quantum-purple/5">
                            <td className="py-3 px-3 font-mono text-quantum-cyan">{worker.worker_id}</td>
                            <td className="py-3 px-3 text-right text-white">{formatHashrate(worker.hashrate)}</td>
                            <td className="py-3 px-3 text-right text-white">{worker.shares_submitted.toLocaleString()}</td>
                            <td className="py-3 px-3 text-right text-quantum-orange">{worker.shares_stale}</td>
                            <td className="py-3 px-3 text-center">
                              <span className={`px-2 py-1 rounded text-xs ${
                                worker.is_connected
                                  ? 'bg-quantum-green/20 text-quantum-green'
                                  : 'bg-red-500/20 text-red-400'
                              }`}>
                                {worker.is_connected ? 'Online' : 'Offline'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Pending Balance */}
              {pendingBalance && pendingBalance.pending_balance > 0 && (
                <div className="bg-gradient-to-r from-quantum-green/10 to-quantum-cyan/10 rounded-xl p-5 border border-quantum-green/30 mb-6">
                  <h4 className="font-bold text-white mb-3 flex items-center gap-2">
                    <DollarSign className="w-5 h-5 text-quantum-green" />
                    Pending Balance
                  </h4>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-3xl font-bold text-quantum-green">
                        {formatQUG(pendingBalance.pending_balance)} QUG
                      </p>
                      {pendingBalance.estimated_payout && (
                        <p className="text-gray-400 text-sm mt-1">{pendingBalance.estimated_payout}</p>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Recent Payouts */}
              {recentPayouts.length > 0 && (
                <div>
                  <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                    <Award className="w-5 h-5 text-quantum-purple" />
                    Recent Pool Payouts
                  </h4>
                  <div className="space-y-2">
                    {recentPayouts.slice(0, 5).map((payout) => (
                      <div key={payout.id} className="bg-quantum-dark/50 rounded-lg p-3 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className={`w-2 h-2 rounded-full ${
                            payout.status === 'Completed' ? 'bg-quantum-green' : 'bg-quantum-yellow'
                          }`} />
                          <div>
                            <p className="text-white font-mono">{formatQUG(payout.amount)} QUG</p>
                            <p className="text-gray-500 text-xs">
                              {new Date(payout.timestamp * 1000).toLocaleString()}
                            </p>
                          </div>
                        </div>
                        {payout.tx_hash && (
                          <a
                            href={`/explorer/tx/${payout.tx_hash}`}
                            className="text-quantum-cyan text-xs hover:underline font-mono"
                          >
                            {payout.tx_hash.slice(0, 12)}...
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* No Pool Data - Show Features */}
              {!poolStats && !poolError && !poolLoading && (
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-purple/20">
                    <h4 className="font-bold text-white mb-2">PPLNS Rewards</h4>
                    <p className="text-gray-400 text-sm">Pay-Per-Last-N-Shares ensures fair distribution based on recent mining contribution</p>
                  </div>
                  <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-purple/20">
                    <h4 className="font-bold text-white mb-2">Low Pool Fee</h4>
                    <p className="text-gray-400 text-sm">1.5% pool fee + 1% dev fee with promotional periods</p>
                  </div>
                  <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-purple/20">
                    <h4 className="font-bold text-white mb-2">Vardiff Support</h4>
                    <p className="text-gray-400 text-sm">Variable difficulty adapts to your hashrate for optimal share submission</p>
                  </div>
                  <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-purple/20">
                    <h4 className="font-bold text-white mb-2">Quantum Security</h4>
                    <p className="text-gray-400 text-sm">Post-quantum cryptographic operations for future-proof pool mining</p>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}

        {activeTab === 'solo' && (
          <motion.div
            key="solo"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Mining Dashboard with Real-Time SSE Updates */}
            {walletAddress && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <MiningDashboard />
              </motion.div>
            )}

            {/* Austrian Economics Notice */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gradient-to-r from-quantum-yellow/10 to-quantum-orange/10 border border-quantum-yellow/30 rounded-xl p-6"
            >
              <div className="flex items-start gap-4">
                <AlertCircle className="w-6 h-6 text-quantum-yellow flex-shrink-0 mt-1" />
                <div>
                  <h3 className="text-lg font-bold text-quantum-yellow mb-2">Austrian Economics Enabled</h3>
                  <div className="space-y-2 text-gray-300 text-sm">
                    <p>
                      <strong>Fixed Supply:</strong> 21,000,000 QUG total (hard cap, immutable)
                    </p>
                    <p>
                      <strong>Block Reward:</strong> 0.5 QUG initially, halves every 210,000 blocks (~4 years)
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Quick Start Guide */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-cyan/30 rounded-xl p-6"
            >
              <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                <Terminal className="w-6 h-6 text-quantum-green" />
                Solo Mining Quick Start
              </h2>

              <div className="space-y-4">
                <div>
                  <p className="text-gray-300 mb-2">1. Download the miner for your platform from the <button onClick={() => setActiveTab('downloads')} className="text-quantum-cyan hover:underline">Downloads tab</button></p>
                </div>

                <div>
                  <p className="text-gray-300 mb-2">2. Make executable and run:</p>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-quantum-dark/50 rounded-lg p-3">
                      <p className="text-quantum-cyan text-sm font-bold mb-1">Linux:</p>
                      <code className="font-mono text-xs text-gray-300">chmod +x q-miner-linux-x64</code>
                    </div>
                    <div className="bg-quantum-dark/50 rounded-lg p-3">
                      <p className="text-quantum-purple text-sm font-bold mb-1">Windows:</p>
                      <code className="font-mono text-xs text-gray-300">q-miner-windows-x64.exe</code>
                    </div>
                  </div>
                </div>

                <div>
                  <p className="text-gray-300 mb-2">3. Run the miner with your wallet address:</p>
                  <div className="bg-quantum-dark/50 rounded-lg p-3 font-mono text-sm text-quantum-green border border-quantum-green/20 relative">
                    <code className="block overflow-x-auto">{miningCommand}</code>
                    <button
                      onClick={() => copyCommand(miningCommand)}
                      className="absolute top-2 right-2 bg-quantum-green/20 hover:bg-quantum-green/30 text-quantum-green px-2 py-1 rounded text-xs transition-colors"
                    >
                      Copy
                    </button>
                  </div>
                </div>

                <div>
                  <p className="text-gray-300 mb-2">4. Optional parameters:</p>
                  <div className="bg-quantum-dark/50 rounded-lg p-3 text-sm text-gray-300 space-y-1">
                    <p><code className="text-quantum-cyan">--threads 4</code> - Number of CPU threads to use (0 = all cores)</p>
                    <p><code className="text-quantum-cyan">--intensity 7</code> - Mining intensity (1-10)</p>
                    <p><code className="text-quantum-cyan">--server {currentServerUrl}</code> - Server URL (auto-detected from current page)</p>
                  </div>
                </div>

                <div className="bg-quantum-purple/10 border border-quantum-purple/30 rounded-lg p-4">
                  <p className="text-quantum-purple font-bold mb-2">Pro Tips:</p>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• Use <code className="text-quantum-cyan">--intensity 10</code> for maximum CPU utilization</li>
                    <li>• Mining rewards appear instantly in your wallet balance</li>
                    <li>• Miner shows hash rate statistics every 5 seconds</li>
                  </ul>
                </div>
              </div>
            </motion.div>

            {/* Mining Statistics */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="grid md:grid-cols-3 gap-4"
            >
              <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-green/30 rounded-xl p-6">
                <div className="flex items-center justify-between mb-3">
                  <Award className="w-6 h-6 text-quantum-green" />
                  <span className="text-2xl font-bold text-quantum-green">0.5 QUG</span>
                </div>
                <p className="text-gray-300 text-sm">Current Block Reward</p>
                <p className="text-gray-500 text-xs mt-1">Halves every 210,000 blocks</p>
              </div>

              <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-cyan/30 rounded-xl p-6">
                <div className="flex items-center justify-between mb-3">
                  <TrendingUp className="w-6 h-6 text-quantum-cyan" />
                  <span className="text-2xl font-bold text-quantum-cyan">21M</span>
                </div>
                <p className="text-gray-300 text-sm">Total Supply Cap</p>
                <p className="text-gray-500 text-xs mt-1">Fixed, immutable hard cap</p>
              </div>

              <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6">
                <div className="flex items-center justify-between mb-3">
                  <Zap className="w-6 h-6 text-quantum-purple" />
                  <span className="text-2xl font-bold text-quantum-purple">1s</span>
                </div>
                <p className="text-gray-300 text-sm">Target Block Time</p>
                <p className="text-gray-500 text-xs mt-1">After bootstrap phase</p>
              </div>
            </motion.div>
          </motion.div>
        )}

        {activeTab === 'downloads' && (
          <motion.div
            key="downloads"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Download Miner Section */}
            <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-green/30 rounded-xl p-6">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <Download className="w-6 h-6 text-quantum-green" />
                Download Q-NarwhalKnight Miner
              </h2>

              {/* Mining Types */}
              <div className="grid md:grid-cols-2 gap-4 mb-6">
                <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-cyan/20">
                  <div className="flex items-center gap-3 mb-3">
                    <Cpu className="w-5 h-5 text-quantum-green" />
                    <span className="font-bold text-white">CPU Mining</span>
                  </div>
                  <p className="text-gray-400 text-sm mb-3">
                    Optimized for multi-core CPUs with AVX2/AVX-512 acceleration
                  </p>
                  <ul className="text-sm text-gray-300 space-y-1">
                    <li>Multi-threaded support</li>
                    <li>Blake3 + VDF algorithm</li>
                    <li>Real-time hash rate monitoring</li>
                  </ul>
                </div>

                <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-purple/20 opacity-60">
                  <div className="flex items-center gap-3 mb-3">
                    <Zap className="w-5 h-5 text-quantum-purple" />
                    <span className="font-bold text-white">GPU Mining</span>
                    <span className="text-xs bg-quantum-purple/20 text-quantum-purple px-2 py-1 rounded">Coming Soon</span>
                  </div>
                  <p className="text-gray-400 text-sm mb-3">
                    CUDA, OpenCL, and Vulkan support (in development)
                  </p>
                  <ul className="text-sm text-gray-300 space-y-1">
                    <li>NVIDIA GPU support</li>
                    <li>AMD GPU support</li>
                    <li>Parallel VDF computation</li>
                  </ul>
                </div>
              </div>

              {/* Miner Downloads */}
              <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <Download className="w-5 h-5 text-quantum-cyan" />
                Latest Miner (v1.0.2-beta)
              </h3>

              <div className="grid md:grid-cols-2 gap-4 mb-6">
                <motion.button
                  onClick={() => handleDownloadMiner('linux')}
                  className="bg-gradient-to-r from-quantum-cyan to-quantum-blue hover:from-quantum-cyan/80 hover:to-quantum-blue/80 text-white font-bold py-4 px-6 rounded-xl transition-all flex flex-col items-center justify-center gap-2 shadow-lg shadow-quantum-cyan/20"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex items-center gap-3">
                    <Terminal className="w-5 h-5" />
                    <span>Linux x86_64</span>
                  </div>
                  <span className="text-xs text-quantum-cyan/80">v1.0.2-beta</span>
                </motion.button>

                <motion.button
                  onClick={() => handleDownloadMiner('windows')}
                  className="bg-gradient-to-r from-quantum-purple to-quantum-pink hover:from-quantum-purple/80 hover:to-quantum-pink/80 text-white font-bold py-4 px-6 rounded-xl transition-all flex flex-col items-center justify-center gap-2 shadow-lg shadow-quantum-purple/20"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex items-center gap-3">
                    <Download className="w-5 h-5" />
                    <span>Windows x64</span>
                  </div>
                  <span className="text-xs text-quantum-purple/80">v1.0.2-beta</span>
                </motion.button>

                <motion.button
                  onClick={() => handleDownloadMiner('macos-intel')}
                  className="bg-gradient-to-r from-quantum-green to-quantum-cyan hover:from-quantum-green/80 hover:to-quantum-cyan/80 text-white font-bold py-4 px-6 rounded-xl transition-all flex flex-col items-center justify-center gap-2 shadow-lg shadow-quantum-green/20"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex items-center gap-3">
                    <Download className="w-5 h-5" />
                    <span>macOS Intel (x64)</span>
                  </div>
                  <span className="text-xs text-quantum-green/80">v1.0.2-beta</span>
                </motion.button>

                <motion.button
                  onClick={() => handleDownloadMiner('macos-arm')}
                  className="bg-gradient-to-r from-quantum-orange to-quantum-yellow hover:from-quantum-orange/80 hover:to-quantum-yellow/80 text-white font-bold py-4 px-6 rounded-xl transition-all flex flex-col items-center justify-center gap-2 shadow-lg shadow-quantum-orange/20"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex items-center gap-3">
                    <Download className="w-5 h-5" />
                    <span>macOS Apple Silicon</span>
                  </div>
                  <span className="text-xs text-quantum-orange/80">v1.0.2-beta</span>
                </motion.button>
              </div>

              {/* Node Binary Downloads */}
              <h3 className="text-lg font-bold text-white mb-4 mt-8 flex items-center gap-2">
                <Terminal className="w-5 h-5 text-quantum-purple" />
                Node Binary (Run Your Own Node)
              </h3>

              <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-purple/20 mb-4">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <p className="font-bold text-white">q-api-server v2.2.1-beta</p>
                    <p className="text-gray-400 text-sm">Full node with mining, wallet, and P2P sync</p>
                  </div>
                  <motion.a
                    href="/downloads/q-api-server-v2.2.1-beta"
                    download
                    className="bg-gradient-to-r from-quantum-purple to-quantum-pink hover:from-quantum-purple/80 hover:to-quantum-pink/80 text-white font-bold py-2 px-4 rounded-lg transition-all flex items-center gap-2"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Download className="w-4 h-4" />
                    Linux x86_64
                  </motion.a>
                </div>
                <div className="bg-quantum-dark/80 rounded-lg p-3 font-mono text-sm text-quantum-cyan">
                  <code>wget https://quillon.xyz/downloads/q-api-server-v2.2.1-beta && chmod +x q-api-server-v2.2.1-beta</code>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {activeTab === 'links' && (
          <motion.div
            key="links"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Links and Resources */}
            <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-orange/30 rounded-xl p-6">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <LinkIcon className="w-6 h-6 text-quantum-orange" />
                Resources & Documentation
              </h2>

              <div className="grid md:grid-cols-2 gap-4">
                {/* Code & Development */}
                <div className="bg-quantum-dark/50 rounded-xl p-5 border border-quantum-cyan/20">
                  <h3 className="text-lg font-bold text-quantum-cyan mb-4">Development</h3>
                  <div className="space-y-3">
                    <a
                      href="https://code.quillon.xyz/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-3 text-gray-300 hover:text-quantum-cyan transition-colors p-2 rounded-lg hover:bg-quantum-cyan/10"
                    >
                      <ExternalLink className="w-4 h-4 flex-shrink-0" />
                      <div>
                        <p className="font-medium">GitHub Repository</p>
                        <p className="text-xs text-gray-500">Source code & contributions</p>
                      </div>
                    </a>
                    <a
                      href="https://github.com/dagknight/q-narwhalknight/issues"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-3 text-gray-300 hover:text-quantum-cyan transition-colors p-2 rounded-lg hover:bg-quantum-cyan/10"
                    >
                      <ExternalLink className="w-4 h-4 flex-shrink-0" />
                      <div>
                        <p className="font-medium">Bug Reports & Issues</p>
                        <p className="text-xs text-gray-500">Report bugs or request features</p>
                      </div>
                    </a>
                  </div>
                </div>

                {/* Whitepapers */}
                <div className="bg-quantum-dark/50 rounded-xl p-5 border border-quantum-purple/20">
                  <h3 className="text-lg font-bold text-quantum-purple mb-4">Whitepapers</h3>
                  <div className="space-y-3">
                    <a
                      href="https://drive.proton.me/urls/ZDQQ98GHKW#JpPgzckdlaGw"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-3 text-gray-300 hover:text-quantum-purple transition-colors p-2 rounded-lg hover:bg-quantum-purple/10"
                    >
                      <ExternalLink className="w-4 h-4 flex-shrink-0" />
                      <div>
                        <p className="font-medium">Mainnet Rewards Whitepaper</p>
                        <p className="text-xs text-gray-500">Mining economics & distribution</p>
                      </div>
                    </a>
                    <a
                      href="/papers/genus2-jacobian-vdf-mining-whitepaper.pdf"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-3 text-gray-300 hover:text-quantum-purple transition-colors p-2 rounded-lg hover:bg-quantum-purple/10"
                    >
                      <ExternalLink className="w-4 h-4 flex-shrink-0" />
                      <div>
                        <p className="font-medium">Genus-2 VDF Mining</p>
                        <p className="text-xs text-gray-500">Post-quantum mining algorithm</p>
                      </div>
                    </a>
                  </div>
                </div>

                {/* Community */}
                <div className="bg-quantum-dark/50 rounded-xl p-5 border border-quantum-green/20">
                  <h3 className="text-lg font-bold text-quantum-green mb-4">Community</h3>
                  <div className="space-y-3">
                    <a
                      href="https://discord.gg/quillon"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-3 text-gray-300 hover:text-quantum-green transition-colors p-2 rounded-lg hover:bg-quantum-green/10"
                    >
                      <ExternalLink className="w-4 h-4 flex-shrink-0" />
                      <div>
                        <p className="font-medium">Discord Server</p>
                        <p className="text-xs text-gray-500">Join the community chat</p>
                      </div>
                    </a>
                    <a
                      href="https://bitcointalk.org/index.php?topic=5526456"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-3 text-gray-300 hover:text-quantum-green transition-colors p-2 rounded-lg hover:bg-quantum-green/10"
                    >
                      <ExternalLink className="w-4 h-4 flex-shrink-0" />
                      <div>
                        <p className="font-medium">BitcoinTalk Announcement</p>
                        <p className="text-xs text-gray-500">Official ANN thread</p>
                      </div>
                    </a>
                  </div>
                </div>

                {/* Network Info */}
                <div className="bg-quantum-dark/50 rounded-xl p-5 border border-quantum-orange/20">
                  <h3 className="text-lg font-bold text-quantum-orange mb-4">Network Info</h3>
                  <div className="space-y-3">
                    <div className="p-2">
                      <p className="font-medium text-gray-300">Bootstrap Node</p>
                      <code className="text-xs text-quantum-orange break-all">185.182.185.227:9001</code>
                    </div>
                    <div className="p-2">
                      <p className="font-medium text-gray-300">API Server</p>
                      <code className="text-xs text-quantum-orange">https://quillon.xyz</code>
                    </div>
                    <div className="p-2">
                      <p className="font-medium text-gray-300">Network ID</p>
                      <code className="text-xs text-quantum-orange">testnet-phase16</code>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
