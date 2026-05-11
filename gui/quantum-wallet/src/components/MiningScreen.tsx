import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Pickaxe, Download, Cpu, Zap, Award, TrendingUp, AlertCircle, ExternalLink, Terminal, Users, User, Link as LinkIcon, RefreshCw, Clock, Activity, DollarSign, Copy, Check, Server } from 'lucide-react';
import MiningDashboard from './MiningDashboard';
import { useMinerLink } from '../hooks/useMinerLink';
import MinerLinkModal from './MinerLinkModal';

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

interface RoundEntry {
  round_id: number;
  block_height: number;
  block_hash: string;
  block_reward: number;
  pool_fee: number;
  dev_fee: number;
  miner_rewards: number;
  payout_count: number;
  found_by: string;
  timestamp: number;
  total_shares: number;
  total_difficulty: number;
}

interface PoolNode {
  peer_id: string;
  stratum_port: number;
  hashrate: number;
  worker_count: number;
  region: string;
  version: string;
  last_seen: number;
  accepting_connections: boolean;
}

interface HashrateEntry {
  hashrate: number;
  workers: number;
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
  const [roundHistory, setRoundHistory] = useState<RoundEntry[]>([]);
  const [poolNodes, setPoolNodes] = useState<PoolNode[]>([]);
  const [hashrateHistory, setHashrateHistory] = useState<HashrateEntry[]>([]);

  // Miner Link — real-time WebSocket connection to personal miner(s)
  const minerLink = useMinerLink(walletAddress || null);
  const [showMinerLinkModal, setShowMinerLinkModal] = useState(false);

  // Fetch pool data - all requests in parallel
  const fetchPoolData = useCallback(async () => {
    setPoolLoading(true);
    setPoolError(null);

    try {
      const [statsRes, workersRes, balanceRes, payoutsRes, roundsRes, nodesRes, hashrateRes] =
        await Promise.allSettled([
          fetch('/api/v1/pool/stats'),
          walletAddress ? fetch(`/api/v1/pool/workers?wallet=${walletAddress}`) : Promise.resolve(null),
          walletAddress ? fetch(`/api/v1/pool/balance/${walletAddress}`) : Promise.resolve(null),
          fetch('/api/v1/pool/payouts?limit=10'),
          fetch('/api/v1/pool/rounds?limit=20'),
          fetch('/api/v1/pool/nodes'),
          fetch('/api/v1/pool/hashrate/history'),
        ]);

      if (statsRes.status === 'fulfilled' && statsRes.value?.ok) {
        setPoolStats(await statsRes.value.json());
      }
      if (workersRes.status === 'fulfilled' && workersRes.value?.ok) {
        setMyWorkers(await workersRes.value.json());
      }
      if (balanceRes.status === 'fulfilled' && balanceRes.value?.ok) {
        setPendingBalance(await balanceRes.value.json());
      }
      if (payoutsRes.status === 'fulfilled' && payoutsRes.value?.ok) {
        setRecentPayouts(await payoutsRes.value.json());
      }
      if (roundsRes.status === 'fulfilled' && roundsRes.value?.ok) {
        setRoundHistory(await roundsRes.value.json());
      }
      if (nodesRes.status === 'fulfilled' && nodesRes.value?.ok) {
        setPoolNodes(await nodesRes.value.json());
      }
      if (hashrateRes.status === 'fulfilled' && hashrateRes.value?.ok) {
        setHashrateHistory(await hashrateRes.value.json());
      }

      setLastRefresh(new Date());
    } catch (err) {
      setPoolError('Pool service unavailable - the pool may not be running');
    } finally {
      setPoolLoading(false);
    }
  }, [walletAddress]);

  // Auto-refresh pool data when on pool tab + SSE for real-time updates
  useEffect(() => {
    if (activeTab !== 'pool') return;

    let isMounted = true;
    fetchPoolData();
    const interval = setInterval(fetchPoolData, 15000);

    // SSE listener for real-time pool updates
    const eventSource = new EventSource(`/api/v1/stream/events?filter=${walletAddress || ''}`);

    eventSource.addEventListener('pool-stats-updated', (e) => {
      if (!isMounted) return;
      try {
        const data = JSON.parse(e.data);
        if (data.data) {
          setPoolStats(prev => prev ? {
            ...prev,
            hashrate: data.data.hashrate ?? prev.hashrate,
            workers: data.data.workers ?? prev.workers,
            blocks_found: data.data.blocks_found ?? prev.blocks_found,
            current_round: data.data.current_round ?? prev.current_round,
            difficulty: data.data.difficulty ?? prev.difficulty,
            shares_this_round: data.data.total_shares ?? prev.shares_this_round,
          } : prev);
        }
      } catch { /* ignore parse errors */ }
    });

    eventSource.addEventListener('pool-block-found', (e) => {
      if (!isMounted) return;
      try {
        const data = JSON.parse(e.data);
        if (data.data) {
          fetch('/api/v1/pool/rounds?limit=20')
            .then(r => r.ok ? r.json() : [])
            .then(rounds => { if (isMounted) setRoundHistory(rounds); })
            .catch(() => {});
        }
      } catch { /* ignore parse errors */ }
    });

    return () => {
      isMounted = false;
      clearInterval(interval);
      eventSource.close();
    };
  }, [activeTab, walletAddress]); // Removed fetchPoolData to prevent interval recreation

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

  // Dynamic stratum host derived from current page
  const stratumHost = typeof window !== 'undefined' ? window.location.hostname : 'pool.quillon.xyz';
  const stratumPort = poolStats?.stratum_port || 3333;
  const stratumUrl = `stratum+tcp://${stratumHost}:${stratumPort}`;

  // Copy stratum URL
  const copyStratumUrl = () => {
    navigator.clipboard.writeText(stratumUrl);
    setCopiedStratum(true);
    setTimeout(() => setCopiedStratum(false), 2000);
  };

  // State for copy mining command button
  const [copiedMiningCmd, setCopiedMiningCmd] = useState(false);

  // Copy full pool mining command
  const copyPoolMiningCommand = () => {
    const cmd = `./q-miner --pool ${stratumUrl} --wallet ${walletAddress || 'YOUR_WALLET'} --threads $(nproc)`;
    navigator.clipboard.writeText(cmd);
    setCopiedMiningCmd(true);
    setTimeout(() => setCopiedMiningCmd(false), 2000);
  };

  const handleDownloadMiner = (platform: 'linux' | 'linux-arm64' | 'windows' | 'macos-intel' | 'macos-arm') => {
    // Link to download the miner binary
    if (platform === 'windows') {
      window.open('https://quillon.xyz/downloads/q-miner-windows-x64.exe', '_blank');
    } else if (platform === 'linux-arm64') {
      window.open('https://quillon.xyz/downloads/q-miner-linux-arm64', '_blank');
    } else if (platform === 'macos-intel') {
      window.open('https://quillon.xyz/downloads/q-miner-macos-x64', '_blank');
    } else if (platform === 'macos-arm') {
      window.open('https://quillon.xyz/downloads/q-miner-macos-arm64', '_blank');
    } else {
      window.open('https://quillon.xyz/downloads/q-miner-linux-x64', '_blank');
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

  const miningCommand = `./q-miner-v10.5.3 --mode solo --wallet ${walletAddress} --threads 4 --intensity 7 --server ${currentServerUrl}`;

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
        <div className="flex-1">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-quantum-cyan to-quantum-purple bg-clip-text text-transparent">
            Quantum Mining
          </h1>
          <p className="text-gray-400">
            Mine QUG with Austrian Economics & DAG-Knight VDF
          </p>
        </div>
        {/* Miner Link button */}
        <button
          onClick={() => setShowMinerLinkModal(true)}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all border ${
            minerLink.miners.length > 0
              ? 'bg-purple-500/10 text-purple-300 border-purple-500/20 hover:bg-purple-500/20'
              : 'bg-white/5 text-gray-400 border-white/10 hover:bg-white/10'
          }`}
        >
          <div className="relative">
            <Pickaxe className="w-4 h-4" />
            {minerLink.miners.length > 0 && (
              <div className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            )}
          </div>
          {minerLink.miners.length > 0
            ? `${minerLink.miners.length} Miner${minerLink.miners.length > 1 ? 's' : ''}`
            : 'My Miners'}
        </button>
      </div>

      {/* Inline miner status */}
      {minerLink.miners.length > 0 && (
        <div className="flex items-center gap-2 text-sm text-gray-400 -mt-2 mb-2">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span>
            {minerLink.miners.length} miner{minerLink.miners.length > 1 ? 's' : ''} connected
            {' — '}
            {minerLink.totalHashrate >= 1e6
              ? `${(minerLink.totalHashrate / 1e6).toFixed(2)} MH/s`
              : minerLink.totalHashrate >= 1e3
              ? `${(minerLink.totalHashrate / 1e3).toFixed(2)} KH/s`
              : `${minerLink.totalHashrate.toFixed(0)} H/s`
            }
          </span>
        </div>
      )}

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
                    <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                      Pool Mining Dashboard
                      {/* Pool Health Indicator */}
                      <span
                        className={`inline-block w-3 h-3 rounded-full ${
                          poolStats && poolStats.uptime_seconds > 0 && poolStats.workers > 0
                            ? 'bg-green-400 shadow-lg shadow-green-400/50'
                            : poolStats && poolStats.uptime_seconds > 0 && poolStats.workers === 0
                            ? 'bg-yellow-400 shadow-lg shadow-yellow-400/50 animate-pulse'
                            : 'bg-red-500 shadow-lg shadow-red-500/50'
                        }`}
                        title={
                          poolStats && poolStats.uptime_seconds > 0 && poolStats.workers > 0
                            ? 'Pool healthy - workers active'
                            : poolStats && poolStats.uptime_seconds > 0 && poolStats.workers === 0
                            ? 'Pool online - no workers connected'
                            : 'Pool offline or unreachable'
                        }
                      />
                    </h2>
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
                        {stratumUrl}
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
                      ./q-miner --mode pool --server {stratumUrl} --wallet {walletAddress || 'YOUR_WALLET'}.rig1
                    </code>
                  </div>
                  {/* Copy Mining Command Button */}
                  <motion.button
                    onClick={copyPoolMiningCommand}
                    className="w-full mt-3 bg-gradient-to-r from-quantum-purple/20 to-quantum-cyan/20 hover:from-quantum-purple/30 hover:to-quantum-cyan/30 border border-quantum-purple/40 rounded-lg px-4 py-3 flex items-center justify-center gap-2 transition-all"
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                  >
                    {copiedMiningCmd ? (
                      <>
                        <Check className="w-4 h-4 text-quantum-green" />
                        <span className="text-quantum-green font-medium text-sm">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Terminal className="w-4 h-4 text-quantum-purple" />
                        <span className="text-white font-medium text-sm">Copy Mining Command</span>
                        <code className="text-gray-500 text-xs ml-2 hidden md:inline">
                          ./q-miner --pool {stratumUrl} --wallet ... --threads $(nproc)
                        </code>
                      </>
                    )}
                  </motion.button>
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
                  {/* Payout Progress Bar */}
                  {(() => {
                    const minPayout = poolStats?.min_payout || 10000000;
                    const progressPct = Math.min((pendingBalance.pending_balance / minPayout) * 100, 100);
                    return (
                      <div className="mt-4">
                        <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                          <span>Payout Progress</span>
                          <span>{progressPct.toFixed(1)}% of {formatQUG(minPayout)} QUG minimum</span>
                        </div>
                        <div className="w-full bg-quantum-dark/60 rounded-full h-3 overflow-hidden border border-quantum-green/20">
                          <motion.div
                            className={`h-full rounded-full ${
                              progressPct >= 100
                                ? 'bg-gradient-to-r from-quantum-green to-quantum-cyan'
                                : progressPct >= 75
                                ? 'bg-gradient-to-r from-quantum-green/80 to-quantum-cyan/80'
                                : 'bg-gradient-to-r from-quantum-green/60 to-quantum-cyan/60'
                            }`}
                            initial={{ width: 0 }}
                            animate={{ width: `${progressPct}%` }}
                            transition={{ duration: 0.8, ease: 'easeOut' }}
                          />
                        </div>
                        {progressPct >= 100 && (
                          <p className="text-quantum-green text-xs mt-1 font-medium">Payout threshold reached - payout will be processed soon</p>
                        )}
                      </div>
                    );
                  })()}
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

              {/* Hashrate History Sparkline */}
              {hashrateHistory.length > 1 && (
                <div className="bg-quantum-dark/30 rounded-xl p-5 border border-quantum-cyan/20 mb-6">
                  <h4 className="font-bold text-white mb-3 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-quantum-cyan" />
                    Pool Hashrate (24h)
                  </h4>
                  <div className="h-20 flex items-end gap-px">
                    {(() => {
                      const maxH = Math.max(...hashrateHistory.map(h => h.hashrate), 1);
                      const display = hashrateHistory.slice(-60); // Last 60 entries
                      return display.map((entry, i) => (
                        <div
                          key={i}
                          className="flex-1 bg-quantum-cyan/60 hover:bg-quantum-cyan/90 rounded-t transition-colors"
                          style={{ height: `${Math.max((entry.hashrate / maxH) * 100, 2)}%` }}
                          title={`${formatHashrate(entry.hashrate)} - ${new Date(entry.timestamp * 1000).toLocaleTimeString()}`}
                        />
                      ));
                    })()}
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>{new Date(hashrateHistory[Math.max(0, hashrateHistory.length - 60)].timestamp * 1000).toLocaleTimeString()}</span>
                    <span>Now</span>
                  </div>
                </div>
              )}

              {/* Round History */}
              {roundHistory.length > 0 && (
                <div className="mb-6">
                  <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                    <Clock className="w-5 h-5 text-quantum-purple" />
                    Round History ({roundHistory.length})
                  </h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-gray-400 border-b border-quantum-purple/20">
                          <th className="text-left py-2 px-3">Round</th>
                          <th className="text-right py-2 px-3">Block</th>
                          <th className="text-right py-2 px-3">Reward</th>
                          <th className="text-right py-2 px-3">Shares</th>
                          <th className="text-right py-2 px-3">Payouts</th>
                          <th className="text-right py-2 px-3">Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {roundHistory.slice(0, 10).map((round) => (
                          <tr key={round.round_id} className="border-b border-quantum-dark/50 hover:bg-quantum-purple/5">
                            <td className="py-3 px-3 font-mono text-quantum-cyan">#{round.round_id}</td>
                            <td className="py-3 px-3 text-right text-white">{round.block_height.toLocaleString()}</td>
                            <td className="py-3 px-3 text-right text-quantum-green">{formatQUG(round.block_reward)} QUG</td>
                            <td className="py-3 px-3 text-right text-white">{round.total_shares.toLocaleString()}</td>
                            <td className="py-3 px-3 text-right text-quantum-purple">{round.payout_count}</td>
                            <td className="py-3 px-3 text-right text-gray-400 text-xs">
                              {new Date(round.timestamp * 1000).toLocaleString()}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Pool Nodes */}
              {poolNodes.length > 0 && (
                <div className="mb-6">
                  <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                    <Server className="w-5 h-5 text-quantum-green" />
                    Pool Nodes ({poolNodes.length})
                  </h4>
                  <div className="grid md:grid-cols-2 gap-3">
                    {poolNodes.map((node) => (
                      <div key={node.peer_id} className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-green/20">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-quantum-cyan font-mono text-sm">{node.peer_id.slice(0, 16)}...</span>
                          <span className={`px-2 py-1 rounded text-xs ${
                            node.accepting_connections
                              ? 'bg-quantum-green/20 text-quantum-green'
                              : 'bg-red-500/20 text-red-400'
                          }`}>
                            {node.accepting_connections ? 'Active' : 'Down'}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-gray-500">Hashrate</span>
                            <p className="text-white">{formatHashrate(node.hashrate)}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Workers</span>
                            <p className="text-white">{node.worker_count}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Port</span>
                            <p className="text-white font-mono">{node.stratum_port}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Region</span>
                            <p className="text-white">{node.region || 'Global'}</p>
                          </div>
                        </div>
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
                      <strong>Block Reward:</strong> Adaptive per block (Era 0: ~0.08 QUG at 1 bps), time-based halving every 4 years
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
                  <p className="text-gray-300 mb-3">2. Setup by platform:</p>
                  <div className="grid md:grid-cols-2 gap-3">
                    {/* Linux x86_64 */}
                    <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-cyan/20 relative">
                      <p className="text-quantum-cyan text-sm font-bold mb-3">Linux x86_64</p>
                      <div className="space-y-1 font-mono text-xs text-gray-300 overflow-x-auto">
                        <code className="block">wget https://quillon.xyz/downloads/q-miner-v10.5.3</code>
                        <code className="block">chmod +x q-miner-v10.5.3</code>
                        <code className="block text-quantum-green mt-2">./q-miner-v10.5.3 --mode solo \</code>
                        <code className="block text-quantum-green pl-2">--wallet {walletAddress || 'YOUR_WALLET'} \</code>
                        <code className="block text-quantum-green pl-2">--server {currentServerUrl}</code>
                      </div>
                      <button
                        onClick={() => copyCommand(`wget https://quillon.xyz/downloads/q-miner-v10.5.3 && chmod +x q-miner-v10.5.3 && ./q-miner-v10.5.3 --mode solo --wallet ${walletAddress || 'YOUR_WALLET'} --server ${currentServerUrl}`)}
                        className="absolute top-3 right-3 bg-quantum-cyan/20 hover:bg-quantum-cyan/30 text-quantum-cyan px-2 py-1 rounded text-xs transition-colors"
                      >
                        Copy
                      </button>
                    </div>

                    {/* Linux ARM64 */}
                    <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-green/20 relative">
                      <p className="text-quantum-green text-sm font-bold mb-3">Linux ARM64</p>
                      <div className="space-y-1 font-mono text-xs text-gray-300 overflow-x-auto">
                        <code className="block">wget https://quillon.xyz/downloads/q-miner-linux-arm64</code>
                        <code className="block">chmod +x q-miner-linux-arm64</code>
                        <code className="block text-quantum-green mt-2">./q-miner-linux-arm64 --mode solo \</code>
                        <code className="block text-quantum-green pl-2">--wallet {walletAddress || 'YOUR_WALLET'} \</code>
                        <code className="block text-quantum-green pl-2">--server {currentServerUrl}</code>
                      </div>
                      <button
                        onClick={() => copyCommand(`wget https://quillon.xyz/downloads/q-miner-linux-arm64 && chmod +x q-miner-linux-arm64 && ./q-miner-linux-arm64 --mode solo --wallet ${walletAddress || 'YOUR_WALLET'} --server ${currentServerUrl}`)}
                        className="absolute top-3 right-3 bg-quantum-green/20 hover:bg-quantum-green/30 text-quantum-green px-2 py-1 rounded text-xs transition-colors"
                      >
                        Copy
                      </button>
                    </div>

                    {/* Windows x64 */}
                    <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-purple/20 relative">
                      <p className="text-quantum-purple text-sm font-bold mb-3">Windows x64</p>
                      <div className="space-y-1 font-mono text-xs text-gray-300 overflow-x-auto">
                        <code className="block text-gray-400"># Download from Downloads tab or:</code>
                        <code className="block">Invoke-WebRequest -Uri https://quillon.xyz/downloads/q-miner-windows-x64.exe -OutFile q-miner.exe</code>
                        <code className="block text-quantum-green mt-2">.\\q-miner.exe --mode solo `</code>
                        <code className="block text-quantum-green pl-2">--wallet {walletAddress || 'YOUR_WALLET'} `</code>
                        <code className="block text-quantum-green pl-2">--server {currentServerUrl}</code>
                      </div>
                      <button
                        onClick={() => copyCommand(`.\\q-miner.exe --mode solo --wallet ${walletAddress || 'YOUR_WALLET'} --server ${currentServerUrl}`)}
                        className="absolute top-3 right-3 bg-quantum-purple/20 hover:bg-quantum-purple/30 text-quantum-purple px-2 py-1 rounded text-xs transition-colors"
                      >
                        Copy
                      </button>
                    </div>

                    {/* Windows CMD */}
                    <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-pink/20 relative">
                      <p className="text-quantum-pink text-sm font-bold mb-3">Windows CMD</p>
                      <div className="space-y-1 font-mono text-xs text-gray-300 overflow-x-auto">
                        <code className="block text-gray-400">REM Open CMD in the download folder</code>
                        <code className="block text-quantum-green">q-miner-windows-x64.exe --mode solo ^</code>
                        <code className="block text-quantum-green pl-2">--wallet {walletAddress || 'YOUR_WALLET'} ^</code>
                        <code className="block text-quantum-green pl-2">--server {currentServerUrl}</code>
                      </div>
                      <button
                        onClick={() => copyCommand(`q-miner-windows-x64.exe --mode solo --wallet ${walletAddress || 'YOUR_WALLET'} --server ${currentServerUrl}`)}
                        className="absolute top-3 right-3 bg-quantum-pink/20 hover:bg-quantum-pink/30 text-quantum-pink px-2 py-1 rounded text-xs transition-colors"
                      >
                        Copy
                      </button>
                    </div>
                  </div>
                </div>

                <div>
                  <p className="text-gray-300 mb-2">3. Full command (copy & paste):</p>
                  <div className="bg-quantum-dark/50 rounded-lg p-3 font-mono text-sm text-quantum-green border border-quantum-green/20 relative">
                    <code className="block overflow-x-auto">{miningCommand}</code>
                    <button
                      onClick={() => copyCommand(miningCommand)}
                      className="absolute top-2 right-2 bg-quantum-green/20 hover:bg-quantum-green/30 text-quantum-green px-2 py-1 rounded text-xs transition-colors"
                    >
                      Copy
                    </button>
                  </div>
                  <div className="bg-quantum-dark/50 rounded-lg p-3 font-mono text-sm text-quantum-purple border border-quantum-purple/20 relative mt-2">
                    <code className="block overflow-x-auto">{`.\\q-miner.exe --mode solo --wallet ${walletAddress} --threads 4 --intensity 7 --server ${currentServerUrl}`}</code>
                    <button
                      onClick={() => copyCommand(`.\\q-miner.exe --mode solo --wallet ${walletAddress} --threads 4 --intensity 7 --server ${currentServerUrl}`)}
                      className="absolute top-2 right-2 bg-quantum-purple/20 hover:bg-quantum-purple/30 text-quantum-purple px-2 py-1 rounded text-xs transition-colors"
                    >
                      Copy
                    </button>
                    <span className="absolute bottom-2 right-2 text-[10px] text-quantum-purple/60">Windows</span>
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
                  <span className="text-2xl font-bold text-quantum-green">~0.08 QUG</span>
                </div>
                <p className="text-gray-300 text-sm">Block Reward (Era 0)</p>
                <p className="text-gray-500 text-xs mt-1">2,625,000 QUG/year, halves every 4 years</p>
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
                <p className="text-gray-500 text-xs mt-1">Adaptive rate targeting</p>
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
              <h3 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
                <Download className="w-5 h-5 text-quantum-cyan" />
                Miner Downloads
                <span className="text-xs font-medium px-2 py-0.5 bg-quantum-green/20 text-quantum-green rounded-full">v10.5.3</span>
              </h3>
              <p className="text-sm text-gray-400 mb-4">Native threads, jemalloc/mimalloc allocator, batched atomic counters</p>

              <div className="grid md:grid-cols-3 gap-4 mb-6">
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
                  <span className="text-xs text-quantum-cyan/80">Servers & Desktops</span>
                </motion.button>

                <motion.button
                  onClick={() => handleDownloadMiner('linux-arm64')}
                  className="bg-gradient-to-r from-quantum-green to-quantum-cyan hover:from-quantum-green/80 hover:to-quantum-cyan/80 text-white font-bold py-4 px-6 rounded-xl transition-all flex flex-col items-center justify-center gap-2 shadow-lg shadow-quantum-green/20"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex items-center gap-3">
                    <Cpu className="w-5 h-5" />
                    <span>Linux ARM64</span>
                  </div>
                  <span className="text-xs text-quantum-green/80">Raspberry Pi / ARM Servers</span>
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
                  <span className="text-xs text-quantum-purple/80">Windows 10/11 — mimalloc optimized</span>
                </motion.button>
              </div>

              {/* Quick Start Commands */}
              <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-cyan/20 mb-6">
                <h4 className="text-sm font-bold text-quantum-cyan mb-3">Quick Start (wget)</h4>
                <div className="space-y-2">
                  <div className="bg-quantum-dark/80 rounded-lg p-2 font-mono text-xs text-gray-300 flex items-center justify-between">
                    <code>wget https://quillon.xyz/downloads/q-miner-v10.5.3 && chmod +x q-miner-v10.5.3</code>
                    <button onClick={() => copyCommand('wget https://quillon.xyz/downloads/q-miner-v10.5.3 && chmod +x q-miner-v10.5.3')} className="text-quantum-cyan hover:text-white ml-2 flex-shrink-0 text-xs px-2">Copy</button>
                  </div>
                  <div className="bg-quantum-dark/80 rounded-lg p-2 font-mono text-xs text-gray-300 flex items-center justify-between">
                    <code>wget https://quillon.xyz/downloads/q-miner-linux-arm64 && chmod +x q-miner-linux-arm64</code>
                    <button onClick={() => copyCommand('wget https://quillon.xyz/downloads/q-miner-linux-arm64 && chmod +x q-miner-linux-arm64')} className="text-quantum-green hover:text-white ml-2 flex-shrink-0 text-xs px-2">Copy</button>
                  </div>
                  <div className="bg-quantum-dark/80 rounded-lg p-2 font-mono text-xs text-gray-300 flex items-center justify-between">
                    <code>wget https://quillon.xyz/downloads/q-miner-windows-x64.exe</code>
                    <button onClick={() => copyCommand('wget https://quillon.xyz/downloads/q-miner-windows-x64.exe')} className="text-quantum-purple hover:text-white ml-2 flex-shrink-0 text-xs px-2">Copy</button>
                  </div>
                </div>
              </div>

              {/* Node Binary Downloads */}
              <h3 className="text-lg font-bold text-white mb-4 mt-8 flex items-center gap-2">
                <Terminal className="w-5 h-5 text-quantum-purple" />
                Node Binary (Run Your Own Node)
              </h3>

              <div className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-purple/20 mb-4">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <p className="font-bold text-white">q-api-server (Latest)</p>
                    <p className="text-gray-400 text-sm">Full node with mining, wallet, DEX, and P2P sync</p>
                  </div>
                  <motion.a
                    href="https://quillon.xyz/downloads/q-api-server-linux-x86_64"
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
                  <code>wget https://quillon.xyz/downloads/q-api-server-linux-x86_64 && chmod +x q-api-server-linux-x86_64</code>
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
                      <code className="text-xs text-quantum-orange">mainnet-genesis</code>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Miner Link Modal */}
      <MinerLinkModal
        isOpen={showMinerLinkModal}
        onClose={() => setShowMinerLinkModal(false)}
        minerLink={minerLink}
      />
    </div>
  );
}
