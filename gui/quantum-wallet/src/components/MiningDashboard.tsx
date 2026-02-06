import { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Trophy, TrendingUp, Zap, Clock, Award, Sparkles } from 'lucide-react';
import { qnkAPI, type MiningRewardEvent, type BalanceUpdateEvent, type MiningStatsEvent, type WalletMiningStats } from '../services/api';

interface MiningStats {
  totalRewards: number;
  blocksFound: number;
  currentBalance: number;
  avgHashRate: number;
  networkHashRate: number; // v1.1.9-beta: Network-wide hashrate
}

// v3.3.4-beta: Individual miner tracking for hash rate breakdown
interface MinerInfo {
  minerId: string;
  workerName: string | null;
  hashRate: number;
  lastSeen: Date;
  blocksFound: number;
  totalRewards: number;
}

interface RewardWithAnimation extends MiningRewardEvent {
  id: string;
  isNew: boolean;
}

export default function MiningDashboard() {
  const [rewards, setRewards] = useState<RewardWithAnimation[]>([]);
  const [stats, setStats] = useState<MiningStats>({
    totalRewards: 0,
    blocksFound: 0,
    currentBalance: 0,
    avgHashRate: 0,
    networkHashRate: 0,
  });
  const [showRewardPopup, setShowRewardPopup] = useState(false);
  const [latestReward, setLatestReward] = useState<MiningRewardEvent | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // v3.3.4-beta: Track individual miners for hash rate breakdown tooltip
  const [miners, setMiners] = useState<Map<string, MinerInfo>>(new Map());
  const [showMinerTooltip, setShowMinerTooltip] = useState(false);
  const [showNetworkAnimation, setShowNetworkAnimation] = useState(false);

  // Use the same wallet as Dashboard - from localStorage
  const [walletAddress, setWalletAddress] = useState('');

  // v3.4.21-beta: Track session rewards separately from total balance
  // This prevents jumps caused by mixing local accumulation with backend absolute values
  const [sessionRewardsTotal, setSessionRewardsTotal] = useState(0);
  const initialBalanceRef = useRef<number | null>(null);

  // v3.4.21-beta: Fetch authoritative balance from API
  const fetchBalance = async () => {
    if (!walletAddress) return;
    try {
      const balanceResponse = await qnkAPI.getWalletBalance(walletAddress);
      if (balanceResponse.success && balanceResponse.data) {
        const balance = balanceResponse.data.balance_qnk || 0;
        console.log('💰 [MiningDashboard] Fetched authoritative balance from API:', balance);

        // Store initial balance on first fetch
        if (initialBalanceRef.current === null) {
          initialBalanceRef.current = balance;
          console.log('💰 [MiningDashboard] Set initial balance reference:', balance);
        }

        setStats(prev => ({
          ...prev,
          currentBalance: balance,
          totalRewards: balance, // Total rewards = current balance (authoritative from API)
        }));
      }
    } catch (error) {
      console.error('Failed to fetch balance:', error);
    }
  };

  // v3.5.0-beta: Fetch mining stats from backend (blocks found, hash rate)
  // This allows stats to survive page refresh instead of resetting to 0
  const fetchMiningStats = async () => {
    if (!walletAddress) return;
    try {
      const miningStatsResponse = await qnkAPI.getMiningStats(walletAddress);
      if (miningStatsResponse.success && miningStatsResponse.data) {
        const serverStats = miningStatsResponse.data;
        console.log('⛏️ [MiningDashboard] Fetched mining stats from server:', serverStats);

        setStats(prev => ({
          ...prev,
          blocksFound: serverStats.blocks_found,
          avgHashRate: serverStats.hash_rate, // KH/s from server
        }));

        // Only log if there are actual mining stats
        if (serverStats.blocks_found > 0 || serverStats.hash_rate > 0) {
          console.log(`⛏️ [MiningDashboard] Restored: ${serverStats.blocks_found} blocks, ${serverStats.hash_rate.toFixed(2)} KH/s`);
        }
      }
    } catch (error) {
      console.error('Failed to fetch mining stats:', error);
    }
  };

  // Get wallet address from localStorage (same as Dashboard)
  useEffect(() => {
    const storedWallet = localStorage.getItem('walletAddress');
    if (storedWallet) {
      setWalletAddress(storedWallet);
      console.log('✅ Mining Dashboard tracking wallet:', storedWallet);
    } else {
      console.warn('⚠️ No wallet found in localStorage - user needs to login/create wallet first');
    }
  }, []);

  useEffect(() => {
    if (!walletAddress) {
      console.warn('⚠️ No wallet address found for mining dashboard');
      return;
    }

    console.log('🔌 Connecting to SSE for wallet:', walletAddress);

    // v3.4.21-beta: Fetch initial balance from API (single source of truth)
    fetchBalance();

    // v3.5.0-beta: Fetch mining stats (blocks found, hash rate) from backend
    // This restores stats on page refresh instead of starting from 0
    fetchMiningStats();

    // v3.4.21-beta: Refresh balance periodically to stay in sync with backend
    const balanceRefreshInterval = setInterval(fetchBalance, 10000); // Every 10s

    // v3.5.0-beta: Refresh mining stats periodically (every 30s)
    const miningStatsInterval = setInterval(fetchMiningStats, 30000);

    // v1.1.9-beta: Fetch network hashrate on load and periodically
    fetchNetworkHashrate();
    const networkHashrateInterval = setInterval(fetchNetworkHashrate, 30000); // Every 30s

    // Subscribe to mining rewards via SSE
    const eventSource = qnkAPI.subscribeToMiningRewards(
      walletAddress,
      handleMiningReward,
      handleBalanceUpdate,
      handleMiningStats
    );

    console.log('✅ SSE EventSource created:', eventSource.url);
    eventSourceRef.current = eventSource;

    // Request notification permissions
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }

    // Cleanup on unmount
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      clearInterval(networkHashrateInterval);
      clearInterval(balanceRefreshInterval);
      clearInterval(miningStatsInterval);
    };
  }, [walletAddress]);

  const handleMiningReward = (reward: MiningRewardEvent) => {
    console.log('🎉 Mining reward received:', reward);
    console.log('🎉 Reward details:', {
      miner: reward.miner_address,
      amount: reward.reward_qnk,
      nonce: reward.nonce,
      hash_rate: reward.hash_rate
    });

    // Add to rewards list with animation flag
    const rewardWithId: RewardWithAnimation = {
      ...reward,
      id: `${reward.block_height}-${reward.nonce}`,
      isNew: true,
    };

    setRewards(prev => {
      const updated = [rewardWithId, ...prev].slice(0, 10); // Keep last 10
      return updated;
    });

    // v3.4.21-beta: Track session rewards (small incremental values only)
    // This is separate from total balance which comes from API
    if (reward.reward_qnk > 0 && reward.reward_qnk < 10) { // Sanity check: individual rewards should be < 10 QUG
      setSessionRewardsTotal(prev => prev + reward.reward_qnk);
    }

    // v3.4.21-beta: Update hash rate and blocks found, but NOT balance
    // Balance is fetched from API periodically to stay authoritative
    setStats(prev => ({
      ...prev,
      blocksFound: prev.blocksFound + 1,
      avgHashRate: reward.hash_rate > 0 ? reward.hash_rate : prev.avgHashRate,
    }));

    // v3.4.21-beta: Trigger a balance refresh from API after receiving a reward
    // This ensures we show the authoritative balance, not a locally accumulated one
    setTimeout(() => fetchBalance(), 500);

    // v3.3.4-beta: Track individual miners for hash rate breakdown
    const minerId = reward.miner_id || reward.miner_address.substring(0, 16);
    setMiners(prev => {
      const newMiners = new Map(prev);
      const existing = newMiners.get(minerId);

      newMiners.set(minerId, {
        minerId,
        workerName: reward.worker_name || null,
        hashRate: reward.hash_rate > 0 ? reward.hash_rate : (existing?.hashRate || 0),
        lastSeen: new Date(),
        blocksFound: (existing?.blocksFound || 0) + 1,
        totalRewards: (existing?.totalRewards || 0) + reward.reward_qnk,
      });

      console.log('⛏️ [MiningDashboard] Updated miner tracking:', {
        minerId,
        workerName: reward.worker_name,
        hashRate: reward.hash_rate,
        totalMiners: newMiners.size
      });

      return newMiners;
    });

    // Show reward popup
    setLatestReward(reward);
    setShowRewardPopup(true);
    setTimeout(() => setShowRewardPopup(false), 5000);

    // Show browser notification
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('Mining Reward Received!', {
        body: `You earned ${reward.reward_qnk.toFixed(8)} QUG from block #${reward.block_height}`,
        icon: '/quillon-logo.png',
        tag: `mining-reward-${reward.block_height}`,
      });
    }

    // Remove animation flag after animation completes
    setTimeout(() => {
      setRewards(prev =>
        prev.map(r => (r.id === rewardWithId.id ? { ...r, isNew: false } : r))
      );
    }, 1000);
  };

  const handleBalanceUpdate = (update: BalanceUpdateEvent) => {
    // v3.4.21-beta: SIMPLIFIED - Just log and trigger API refresh
    // We no longer try to accumulate locally - API is the single source of truth
    console.log('🔔 [MiningDashboard] handleBalanceUpdate - triggering API refresh:', {
      wallet: update.wallet_address,
      old: update.old_balance,
      new: update.new_balance,
      reason: update.change_reason
    });

    // Check if this is a P2P mining reward (needs to be added to rewards list since no MiningReward event)
    const isP2PMiningReward = update.change_reason === 'p2p_mining_reward' ||
                              update.change_reason === 'pending_mining_reward';

    if (isP2PMiningReward) {
      // P2P mining rewards - add to rewards list for display only
      const rewardAmount = update.new_balance - update.old_balance;

      // Sanity check: individual rewards should be small (< 10 QUG)
      if (rewardAmount > 0 && rewardAmount < 10) {
        console.log('⛏️ [MiningDashboard] P2P Mining reward - adding to list:', rewardAmount);

        const rewardWithId: RewardWithAnimation = {
          id: `p2p-${update.timestamp}-${Math.random()}`,
          miner_address: update.wallet_address,
          reward_qnk: rewardAmount,
          nonce: 0,
          block_height: (update as { block_height?: number }).block_height || 0,
          difficulty: '0',
          hash_rate: 0,
          timestamp: update.timestamp,
          isNew: true,
        };

        setRewards(prev => [rewardWithId, ...prev].slice(0, 10));
        setSessionRewardsTotal(prev => prev + rewardAmount);
        setStats(prev => ({ ...prev, blocksFound: prev.blocksFound + 1 }));

        setTimeout(() => {
          setRewards(prev => prev.map(r => r.id === rewardWithId.id ? { ...r, isNew: false } : r));
        }, 1000);
      }
    }

    // v3.4.21-beta: Trigger API refresh to get authoritative balance
    // This is the ONLY place we update the displayed balance
    fetchBalance();
  };

  const handleMiningStats = (statsUpdate: MiningStatsEvent) => {
    // v3.4.21-beta: SIMPLIFIED - Only update hashrate, let API handle balance
    console.log('📊 [MiningDashboard] Mining stats received:', {
      miner: statsUpdate.miner_address,
      hash_rate: statsUpdate.avg_hash_rate,
      miner_id: statsUpdate.miner_id,
      worker_id: statsUpdate.worker_id
    });

    // Only update hashrate - balance comes from API
    setStats(prev => ({
      ...prev,
      avgHashRate: statsUpdate.avg_hash_rate,
    }));

    // Track individual miners for hash rate breakdown
    if (statsUpdate.worker_id || statsUpdate.miner_id) {
      const minerId = statsUpdate.miner_id || statsUpdate.worker_id || 'unknown';
      setMiners(prev => {
        const newMiners = new Map(prev);
        const existing = newMiners.get(minerId);

        newMiners.set(minerId, {
          minerId,
          workerName: statsUpdate.worker_id || null,
          hashRate: statsUpdate.avg_hash_rate,
          lastSeen: new Date(),
          blocksFound: existing?.blocksFound || 0,
          totalRewards: existing?.totalRewards || 0,
        });

        console.log('⛏️ [MiningDashboard] Updated miner hashrate:', {
          minerId,
          hashRate: statsUpdate.avg_hash_rate,
          totalMiners: newMiners.size
        });

        return newMiners;
      });
    }
  };

  // v1.1.9-beta: Fetch network-wide hashrate from /api/v1/network/supply
  const fetchNetworkHashrate = async () => {
    try {
      const response = await fetch('/api/v1/network/supply');
      if (response.ok) {
        const json = await response.json();
        // API returns { success: true, data: { network_hashrate: 186539, ... } }
        const networkHashrate = json.data?.network_hashrate || json.network_hashrate || 0;
        console.log('🌐 Network hashrate fetched:', networkHashrate, 'H/s');
        setStats(prev => ({
          ...prev,
          networkHashRate: networkHashrate,
        }));
      }
    } catch (error) {
      console.error('Failed to fetch network hashrate:', error);
    }
  };

  const formatHashRate = (hashRate: number) => {
    if (hashRate >= 1e9) return `${(hashRate / 1e9).toFixed(2)} GH/s`;
    if (hashRate >= 1e6) return `${(hashRate / 1e6).toFixed(2)} MH/s`;
    if (hashRate >= 1e3) return `${(hashRate / 1e3).toFixed(2)} KH/s`;
    return `${hashRate.toFixed(2)} H/s`;
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  // v3.3.4-beta: Calculate total hash rate from all tracked miners
  const totalMinerHashRate = Array.from(miners.values()).reduce(
    (sum, miner) => sum + miner.hashRate,
    0
  );
  // v3.5.4-beta: Use maximum of API-reported hashrate and SSE-tracked miners
  // SSE events may have hash_rate=0, but API endpoint calculates from solution timestamps
  const displayHashRate = Math.max(totalMinerHashRate, stats.avgHashRate);

  if (!walletAddress) {
    return (
      <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-yellow/30 rounded-xl p-6">
        <p className="text-quantum-yellow">
          No mining wallet configured. Please set VITE_DEFAULT_MINING_WALLET in .env or connect your wallet.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Reward Popup */}
      <AnimatePresence>
        {showRewardPopup && latestReward && (
          <motion.div
            initial={{ opacity: 0, y: -50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -50, scale: 0.9 }}
            className="fixed top-20 left-1/2 transform -translate-x-1/2 z-50 max-w-md w-full"
          >
            <div className="bg-gradient-to-r from-quantum-green/20 to-quantum-cyan/20 backdrop-blur-xl border-2 border-quantum-green rounded-2xl p-6 shadow-2xl">
              <div className="flex items-center gap-4">
                <div className="relative">
                  <Sparkles className="w-12 h-12 text-quantum-green animate-pulse" />
                  <div className="absolute inset-0 bg-quantum-green/20 rounded-full animate-ping" />
                </div>
                <div className="flex-1">
                  <h3 className="text-2xl font-bold text-quantum-green mb-1">
                    Mining Reward!
                  </h3>
                  <p className="text-white text-lg">
                    +{latestReward.reward_qnk.toFixed(8)} QUG
                  </p>
                  <p className="text-gray-300 text-sm">
                    Block #{latestReward.block_height}
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-cyan/30 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <TrendingUp className="w-6 h-6 text-quantum-cyan" />
            <span className="text-sm text-gray-400">Balance</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {stats.currentBalance.toFixed(4)}
          </div>
          <div className="text-sm text-quantum-cyan">QUG</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-green/30 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <Trophy className="w-6 h-6 text-quantum-green" />
            <span className="text-sm text-gray-400">Total Rewards</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {stats.totalRewards.toFixed(4)}
          </div>
          <div className="text-sm text-quantum-green">QUG</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <Award className="w-6 h-6 text-quantum-purple" />
            <span className="text-sm text-gray-400">Blocks Found</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {stats.blocksFound}
          </div>
          <div className="text-sm text-quantum-purple">Blocks</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-yellow/30 rounded-xl p-6 relative cursor-pointer overflow-visible"
          style={{ zIndex: showMinerTooltip ? 100 : 1 }}
          onMouseEnter={() => setShowMinerTooltip(true)}
          onMouseLeave={() => setShowMinerTooltip(false)}
        >
          <div className="flex items-center justify-between mb-3">
            <Zap className="w-6 h-6 text-quantum-yellow" />
            <span className="text-sm text-gray-400">Your Hash Rate</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {formatHashRate(displayHashRate)}
          </div>
          <div className="text-sm text-quantum-yellow flex items-center gap-2">
            Personal
            {miners.size > 0 && (
              <span className="text-xs bg-quantum-yellow/20 px-2 py-0.5 rounded-full">
                {miners.size} miner{miners.size !== 1 ? 's' : ''}
              </span>
            )}
          </div>

          {/* v3.3.4-beta: Miner List Tooltip */}
          <AnimatePresence>
            {showMinerTooltip && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                transition={{ duration: 0.2 }}
                className="absolute left-0 right-0 top-full mt-2 bg-quantum-dark/95 backdrop-blur-xl border border-quantum-yellow/40 rounded-xl p-4 shadow-2xl"
                style={{ zIndex: 9999 }}
              >
                <div className="text-sm font-semibold text-quantum-yellow mb-3 flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Active Miners ({miners.size})
                </div>
                {miners.size > 0 ? (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {Array.from(miners.values())
                      .sort((a, b) => b.hashRate - a.hashRate)
                      .map((miner) => (
                        <div
                          key={miner.minerId}
                          className="flex items-center justify-between bg-quantum-indigo/20 rounded-lg p-3 border border-quantum-purple/20"
                        >
                          <div className="flex flex-col">
                            <span className="text-white font-medium">
                              {miner.workerName || `Miner ${miner.minerId.substring(0, 8)}...`}
                            </span>
                            <span className="text-xs text-gray-400">
                              ID: {miner.minerId.substring(0, 12)}...
                            </span>
                          </div>
                          <div className="flex flex-col items-end">
                            <span className="text-quantum-cyan font-bold">
                              {formatHashRate(miner.hashRate)}
                            </span>
                            <span className="text-xs text-gray-400">
                              {miner.blocksFound} block{miner.blocksFound !== 1 ? 's' : ''} | {miner.totalRewards.toFixed(4)} QUG
                            </span>
                          </div>
                        </div>
                      ))}
                  </div>
                ) : (
                  <div className="text-gray-400 text-sm text-center py-4">
                    <div className="mb-2">No miners detected yet</div>
                    <div className="text-xs">Mining rewards will appear here as they are received via SSE</div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* Network Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="bg-gradient-to-br from-quantum-indigo/40 to-quantum-purple/20 backdrop-blur-xl border border-quantum-cyan/40 rounded-xl p-6 relative cursor-pointer overflow-visible"
          style={{ zIndex: showNetworkAnimation ? 100 : 1 }}
          onMouseEnter={() => setShowNetworkAnimation(true)}
          onMouseLeave={() => setShowNetworkAnimation(false)}
        >
          <div className="flex items-center justify-between mb-3">
            <TrendingUp className="w-6 h-6 text-quantum-cyan" />
            <span className="text-sm text-gray-400">Network Hash Rate</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {formatHashRate(stats.networkHashRate)}
          </div>
          <div className="text-sm text-quantum-cyan">Total Network Power</div>

          {/* v3.3.4-beta: Epic Mining Animation on Hover */}
          <AnimatePresence>
            {showNetworkAnimation && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ duration: 0.3 }}
                className="absolute inset-0 rounded-xl overflow-hidden pointer-events-none"
                style={{ zIndex: 9999 }}
              >
                {/* Animated Background Glow */}
                <div className="absolute inset-0 bg-gradient-to-br from-quantum-cyan/30 via-quantum-purple/20 to-quantum-yellow/30 animate-pulse" />

                {/* Mining Sparks */}
                {[...Array(12)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute w-2 h-2 rounded-full"
                    style={{
                      background: i % 3 === 0 ? '#00f0ff' : i % 3 === 1 ? '#ffd700' : '#ff6b00',
                      left: `${20 + Math.random() * 60}%`,
                      top: `${20 + Math.random() * 60}%`,
                      boxShadow: `0 0 10px ${i % 3 === 0 ? '#00f0ff' : i % 3 === 1 ? '#ffd700' : '#ff6b00'}`,
                    }}
                    animate={{
                      y: [-20, -40, -20],
                      x: [0, (i % 2 === 0 ? 10 : -10), 0],
                      opacity: [0, 1, 0],
                      scale: [0.5, 1.2, 0.5],
                    }}
                    transition={{
                      duration: 1 + Math.random() * 0.5,
                      repeat: Infinity,
                      delay: i * 0.1,
                      ease: "easeInOut",
                    }}
                  />
                ))}

                {/* Pickaxe Animation */}
                <motion.div
                  className="absolute bottom-4 left-1/2 transform -translate-x-1/2 text-4xl"
                  animate={{
                    rotate: [-30, 30, -30],
                    y: [0, -5, 0],
                  }}
                  transition={{
                    duration: 0.4,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                >
                  ⛏️
                </motion.div>

                {/* Hash Rate Pulse Rings */}
                {[...Array(3)].map((_, i) => (
                  <motion.div
                    key={`ring-${i}`}
                    className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 border-2 border-quantum-cyan/50 rounded-full"
                    style={{
                      width: 60 + i * 40,
                      height: 60 + i * 40,
                    }}
                    animate={{
                      scale: [1, 1.5, 1],
                      opacity: [0.5, 0, 0.5],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      delay: i * 0.4,
                      ease: "easeOut",
                    }}
                  />
                ))}

                {/* Mining Stats Overlay */}
                <motion.div
                  className="absolute inset-0 flex flex-col items-center justify-center bg-quantum-dark/80 backdrop-blur-sm"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  <motion.div
                    className="text-5xl mb-2"
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 0.5, repeat: Infinity }}
                  >
                    ⚡
                  </motion.div>
                  <div className="text-quantum-cyan font-bold text-2xl">
                    {formatHashRate(stats.networkHashRate)}
                  </div>
                  <div className="text-gray-400 text-sm mt-1">Network Mining Power</div>
                  <div className="flex gap-4 mt-3">
                    <div className="text-center">
                      <div className="text-quantum-yellow font-bold">{miners.size}</div>
                      <div className="text-xs text-gray-500">Your Miners</div>
                    </div>
                    <div className="text-center">
                      <div className="text-quantum-purple font-bold">
                        {stats.networkHashRate > 0
                          ? ((displayHashRate / stats.networkHashRate) * 100).toFixed(1)
                          : '0.0'}%
                      </div>
                      <div className="text-xs text-gray-500">Your Share</div>
                    </div>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-gradient-to-br from-quantum-indigo/40 to-quantum-green/20 backdrop-blur-xl border border-quantum-green/40 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <Sparkles className="w-6 h-6 text-quantum-green" />
            <span className="text-sm text-gray-400">Your Share</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {stats.networkHashRate > 0
              ? ((stats.avgHashRate / stats.networkHashRate) * 100).toFixed(2)
              : '0.00'}%
          </div>
          <div className="text-sm text-quantum-green">of Network Power</div>
        </motion.div>
      </div>

      {/* Recent Rewards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6"
      >
        <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Clock className="w-5 h-5 text-quantum-cyan" />
          Recent Mining Rewards
        </h3>

        {rewards.length === 0 ? (
          <div className="text-center py-8">
            <Zap className="w-12 h-12 text-gray-500 mx-auto mb-3 opacity-50" />
            <p className="text-gray-400">
              Waiting for mining rewards...
            </p>
            <p className="text-gray-500 text-sm mt-1">
              Rewards will appear here in real-time via SSE
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            <AnimatePresence>
              {rewards.map((reward) => (
                <motion.div
                  key={reward.id}
                  initial={reward.isNew ? { opacity: 0, x: -20, scale: 0.95 } : false}
                  animate={{ opacity: 1, x: 0, scale: 1 }}
                  exit={{ opacity: 0, x: 20, scale: 0.95 }}
                  className={`bg-quantum-dark/50 rounded-lg p-4 border transition-all ${
                    reward.isNew
                      ? 'border-quantum-green shadow-lg shadow-quantum-green/20'
                      : 'border-quantum-purple/20'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Trophy className="w-4 h-4 text-quantum-green" />
                        <span className="text-quantum-green font-bold">
                          +{reward.reward_qnk.toFixed(8)} QUG
                        </span>
                        {reward.isNew && (
                          <span className="text-xs bg-quantum-green/20 text-quantum-green px-2 py-0.5 rounded-full animate-pulse">
                            NEW
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-300">
                        Block #{reward.block_height} • Nonce: {reward.nonce}
                      </div>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {reward.origin_node_name && (
                          <span className="text-xs bg-quantum-purple/20 text-quantum-purple px-2 py-0.5 rounded">
                            {reward.origin_node_name}
                          </span>
                        )}
                        {reward.worker_name && (
                          <span className="text-xs bg-quantum-cyan/20 text-quantum-cyan px-2 py-0.5 rounded">
                            Miner: {reward.worker_name}
                          </span>
                        )}
                        {reward.miner_id && !reward.worker_name && (
                          <span className="text-xs bg-quantum-yellow/20 text-quantum-yellow px-2 py-0.5 rounded">
                            ID: {reward.miner_id.substring(0, 8)}...
                          </span>
                        )}
                        {reward.miner_id && reward.worker_name && (
                          <span className="text-xs bg-quantum-yellow/20 text-quantum-yellow px-2 py-0.5 rounded opacity-70">
                            [{reward.miner_id.substring(0, 8)}]
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        Difficulty: {reward.difficulty}
                        {reward.origin_node_id && (
                          <span className="ml-2 text-gray-600">
                            • Node: {reward.origin_node_id.substring(0, 12)}...
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-400">
                        {formatTime(reward.timestamp)}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}

        {/* Smart Accumulation Tip */}
        {rewards.length >= 2 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-4 bg-quantum-cyan/10 border border-quantum-cyan/30 rounded-lg p-4"
          >
            <div className="flex items-center gap-2 text-quantum-cyan text-sm">
              <TrendingUp className="w-4 h-4" />
              <span className="font-semibold">Accumulation Rate</span>
            </div>
            <div className="text-gray-300 text-sm mt-2">
              {(() => {
                // Calculate rewards per hour based on recent activity
                const recentRewards = rewards.slice(0, Math.min(5, rewards.length));
                const totalAmount = recentRewards.reduce((sum, r) => sum + r.reward_qnk, 0);
                const firstTime = new Date(recentRewards[recentRewards.length - 1]?.timestamp || Date.now()).getTime();
                const lastTime = new Date(recentRewards[0]?.timestamp || Date.now()).getTime();
                const timeDiffHours = Math.max((lastTime - firstTime) / (1000 * 60 * 60), 0.01);
                const ratePerHour = totalAmount / timeDiffHours;
                const ratePerDay = ratePerHour * 24;

                if (ratePerHour > 0.001) {
                  return (
                    <>
                      At your current hashrate, you're earning approximately{' '}
                      <span className="text-quantum-green font-bold">{ratePerHour.toFixed(4)} QUG/hour</span>
                      {' '}({ratePerDay.toFixed(2)} QUG/day)
                    </>
                  );
                }
                return 'Keep mining to calculate your accumulation rate!';
              })()}
            </div>
            <div className="text-gray-500 text-xs mt-2">
              All mining rewards are credited instantly across the network via P2P propagation
            </div>
          </motion.div>
        )}
      </motion.div>

      {/* Download Miner */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-quantum-green/10 border border-quantum-green/30 rounded-xl p-6"
      >
        <h4 className="text-lg font-bold text-quantum-green mb-3 flex items-center gap-2">
          <Zap className="w-5 h-5" />
          Download Optimized Miner v3.3.3
        </h4>
        <p className="text-gray-300 text-sm mb-4">
          v3.3.3: Miner identification + Lock-free multi-threading + P2P propagation
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <a
            href="/downloads/q-miner-v3.3.3-beta"
            download
            className="flex items-center justify-center gap-2 bg-quantum-green/20 hover:bg-quantum-green/30 border border-quantum-green/50 text-quantum-green font-bold py-3 px-4 rounded-lg transition-all"
          >
            <Zap className="w-4 h-4" />
            Linux x64 (Latest)
          </a>
          <a
            href="/downloads/q-miner-windows-x64.exe"
            download
            className="flex items-center justify-center gap-2 bg-quantum-cyan/20 hover:bg-quantum-cyan/30 border border-quantum-cyan/50 text-quantum-cyan font-bold py-3 px-4 rounded-lg transition-all"
          >
            <Zap className="w-4 h-4" />
            Windows x64
          </a>
          <a
            href="/downloads/q-miner-macos-arm64"
            download
            className="flex items-center justify-center gap-2 bg-quantum-purple/20 hover:bg-quantum-purple/30 border border-quantum-purple/50 text-quantum-purple font-bold py-3 px-4 rounded-lg transition-all"
          >
            <Zap className="w-4 h-4" />
            macOS ARM64
          </a>
          <a
            href="/downloads/q-miner-macos-x64"
            download
            className="flex items-center justify-center gap-2 bg-quantum-yellow/20 hover:bg-quantum-yellow/30 border border-quantum-yellow/50 text-quantum-yellow font-bold py-3 px-4 rounded-lg transition-all"
          >
            <Zap className="w-4 h-4" />
            macOS Intel x64
          </a>
        </div>
        <div className="mt-4 p-4 bg-black/30 rounded-lg space-y-3">
          <div>
            <p className="text-xs text-gray-400 mb-1">Connect to Network (with miner name):</p>
            <code className="text-xs text-quantum-cyan block">
              ./q-miner --wallet {walletAddress.slice(0, 20)}... --server http://quillon.xyz:8080 --miner-name "My Rig"
            </code>
          </div>
          <div>
            <p className="text-xs text-gray-400 mb-1">Solo Mining (Local Node):</p>
            <code className="text-xs text-quantum-yellow block">
              ./q-miner --wallet {walletAddress.slice(0, 20)}... --server http://localhost:8080 --miner-name "Local"
            </code>
            <p className="text-xs text-gray-500 mt-1">Start node with: Q_ALLOW_SOLO_MINING=true ./q-api-server</p>
          </div>
        </div>
      </motion.div>

      {/* Mining Tips */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-quantum-purple/10 border border-quantum-purple/30 rounded-xl p-6"
      >
        <h4 className="text-lg font-bold text-quantum-purple mb-3">Mining Tips</h4>
        <ul className="space-y-2 text-gray-300 text-sm">
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>Real-time rewards appear instantly via Server-Sent Events (SSE)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>Dashboard updates automatically when your miner finds a block</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>Browser notifications alert you to new rewards (enable in settings)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>Hash rate is calculated from your actual mining performance</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-cyan mt-0.5">•</span>
            <span>New optimized miner: 2.5x faster with AVX2 SIMD + CPU core pinning</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-quantum-yellow mt-0.5">⚡</span>
            <span>Solo mining: Balance updates in &lt;100ms with AEGIS-256 authenticated rewards</span>
          </li>
        </ul>
      </motion.div>
    </div>
  );
}
