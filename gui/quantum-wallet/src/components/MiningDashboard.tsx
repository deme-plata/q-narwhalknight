import { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Trophy, TrendingUp, Zap, Clock, Award, Sparkles } from 'lucide-react';
import { qnkAPI, type MiningRewardEvent, type BalanceUpdateEvent } from '../services/api';

interface MiningStats {
  totalRewards: number;
  blocksFound: number;
  currentBalance: number;
  avgHashRate: number;
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
  });
  const [showRewardPopup, setShowRewardPopup] = useState(false);
  const [latestReward, setLatestReward] = useState<MiningRewardEvent | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const walletAddress = localStorage.getItem('walletAddress') || '';

  useEffect(() => {
    if (!walletAddress) {
      console.warn('⚠️ No wallet address found for mining dashboard');
      return;
    }

    console.log('🔌 Connecting to SSE for wallet:', walletAddress);

    // Subscribe to mining rewards via SSE
    const eventSource = qnkAPI.subscribeToMiningRewards(
      walletAddress,
      handleMiningReward,
      handleBalanceUpdate
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

    // Update stats
    setStats(prev => ({
      ...prev,
      totalRewards: prev.totalRewards + reward.reward_qnk,
      blocksFound: prev.blocksFound + 1,
      avgHashRate: reward.hash_rate > 0 ? reward.hash_rate : prev.avgHashRate,
    }));

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
    console.log('💰 Balance updated:', update);
    console.log('💰 Balance details:', {
      wallet: update.wallet_address,
      old: update.old_balance,
      new: update.new_balance,
      reason: update.change_reason
    });
    setStats(prev => ({
      ...prev,
      currentBalance: update.new_balance,
    }));
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

  if (!walletAddress) {
    return (
      <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-yellow/30 rounded-xl p-6">
        <p className="text-quantum-yellow">
          Please connect your wallet to view mining dashboard
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
          className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-yellow/30 rounded-xl p-6"
        >
          <div className="flex items-center justify-between mb-3">
            <Zap className="w-6 h-6 text-quantum-yellow" />
            <span className="text-sm text-gray-400">Hash Rate</span>
          </div>
          <div className="text-3xl font-bold text-white mb-1">
            {formatHashRate(stats.avgHashRate)}
          </div>
          <div className="text-sm text-quantum-yellow">Average</div>
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
                      <div className="text-xs text-gray-500 mt-1">
                        Difficulty: {reward.difficulty}
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
        </ul>
      </motion.div>
    </div>
  );
}
