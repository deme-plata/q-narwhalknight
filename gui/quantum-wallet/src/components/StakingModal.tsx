import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Coins,
  TrendingUp,
  Lock,
  Zap,
  Brain,
  Shield,
  BarChart3,
  Clock,
  AlertTriangle,
  CheckCircle,
  Info,
  Loader2
} from 'lucide-react';
import { qnkAPI } from '../services/api';

interface StakingModalProps {
  isOpen: boolean;
  onClose: () => void;
  availableBalance: number;
  walletAddress: string;
  onStakeSuccess?: () => void;
}

interface PredictionDomain {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  apy: number;
  riskLevel: 'low' | 'medium' | 'high';
  totalStaked: number;
  accuracy: number;
}

interface ActiveStake {
  id: string;
  domain: string;
  domain_name: string;
  amount: number;
  confidence: number;
  lock_days: number;
  staked_at: number;
  unlocks_at: number;
  status: 'active' | 'unlocked' | 'claimed';
  reward: number;
  prediction_accuracy: number;
}

// Icon mapping for domains
const DOMAIN_ICONS: Record<string, React.ReactNode> = {
  'gas-fees': <Coins className="w-5 h-5" />,
  'block-time': <Clock className="w-5 h-5" />,
  'network-load': <BarChart3 className="w-5 h-5" />,
  'validator-uptime': <Shield className="w-5 h-5" />,
  'cross-chain': <Zap className="w-5 h-5" />,
  'defi-tvl': <TrendingUp className="w-5 h-5" />,
};

const LOCK_PERIODS = [
  { days: 7, multiplier: 1.0, label: '1 Week' },
  { days: 30, multiplier: 1.25, label: '1 Month' },
  { days: 90, multiplier: 1.5, label: '3 Months' },
  { days: 180, multiplier: 2.0, label: '6 Months' },
];

export default function StakingModal({
  isOpen,
  onClose,
  availableBalance,
  walletAddress,
  onStakeSuccess
}: StakingModalProps) {
  const [activeTab, setActiveTab] = useState<'stake' | 'positions' | 'resolution'>('stake');
  const [selectedDomain, setSelectedDomain] = useState<PredictionDomain | null>(null);
  const [amount, setAmount] = useState('');
  const [confidence, setConfidence] = useState(50);
  const [predictionValue, setPredictionValue] = useState('');  // v1.4.3: User's predicted value
  const [lockPeriod, setLockPeriod] = useState(LOCK_PERIODS[1]);
  const [isStaking, setIsStaking] = useState(false);
  const [stakeSuccess, setStakeSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeStakes, setActiveStakes] = useState<ActiveStake[]>([]);
  const [loadingStakes, setLoadingStakes] = useState(false);
  const [predictionDomains, setPredictionDomains] = useState<PredictionDomain[]>([]);
  const [loadingDomains, setLoadingDomains] = useState(false);
  const [resolutionHistory, setResolutionHistory] = useState<any[]>([]);  // v1.4.3
  const [loadingResolutions, setLoadingResolutions] = useState(false);  // v1.4.3

  // Fetch domains and stakes on mount
  useEffect(() => {
    if (isOpen) {
      fetchDomains();
      if (walletAddress) {
        fetchActiveStakes();
        fetchResolutionHistory();
      }
    }
  }, [isOpen, walletAddress]);

  const fetchDomains = async () => {
    setLoadingDomains(true);
    try {
      const response = await qnkAPI.getPredictionDomains();
      if (response.success && response.data) {
        const domains: PredictionDomain[] = response.data.map((d: any) => ({
          id: d.id,
          name: d.name,
          description: d.description,
          icon: DOMAIN_ICONS[d.id] || <Brain className="w-5 h-5" />,
          apy: d.apy || 0,
          riskLevel: (d.risk_level || 'medium') as 'low' | 'medium' | 'high',
          totalStaked: (d.total_staked || 0) / 1_000_000_000,
          accuracy: (d.accuracy_30d || 0) * 100,
        }));
        setPredictionDomains(domains);
      } else {
        setError('Failed to load prediction domains');
      }
    } catch (err) {
      console.error('Failed to fetch domains:', err);
      setError('Failed to load prediction domains');
    } finally {
      setLoadingDomains(false);
    }
  };

  const fetchActiveStakes = async () => {
    setLoadingStakes(true);
    try {
      const response = await qnkAPI.getStakingPositions(walletAddress);
      if (response.success && response.data) {
        // Map API response to frontend format
        const stakes: ActiveStake[] = response.data.map((s: any) => ({
          id: s.id,
          domain: s.domain,
          domain_name: s.domain_name,
          amount: (s.amount || 0) / 1e24, // Convert from base units to QUG (8 decimals)
          confidence: (s.confidence || 0) * 100, // Convert from decimal (0.1-1.0) to percentage
          lock_days: s.lock_days || 0,
          staked_at: (s.staked_at || 0) * 1000, // Convert to milliseconds
          unlocks_at: (s.unlocks_at || 0) * 1000, // Convert to milliseconds
          status: s.status || 'active',
          reward: (s.reward || 0) / 1e24, // Convert from base units to QUG (8 decimals)
          prediction_accuracy: (s.prediction_accuracy || 0) * 100,
        }));
        setActiveStakes(stakes);
      } else {
        setActiveStakes([]);
      }
    } catch (err) {
      console.error('Failed to fetch stakes:', err);
      setActiveStakes([]);
    } finally {
      setLoadingStakes(false);
    }
  };

  // v1.4.3: Fetch resolution history for all domains
  const fetchResolutionHistory = async () => {
    setLoadingResolutions(true);
    try {
      // Fetch resolution history from all domains
      const allResolutions: any[] = [];
      const domains = ['gas-fees', 'block-time', 'network-load', 'validator-uptime', 'cross-chain', 'defi-tvl'];

      for (const domain of domains) {
        try {
          const response = await qnkAPI.getResolutionHistory(domain);
          if (response.success && response.data) {
            allResolutions.push(...response.data.map((r: any) => ({
              ...r,
              domain,
              domain_name: domain.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
            })));
          }
        } catch (e) {
          // Domain might not have any resolutions yet
          console.debug(`No resolutions for ${domain}`);
        }
      }

      // Sort by resolved_at descending (most recent first)
      allResolutions.sort((a, b) => (b.resolved_at || 0) - (a.resolved_at || 0));
      setResolutionHistory(allResolutions.slice(0, 20)); // Show last 20
    } catch (err) {
      console.error('Failed to fetch resolution history:', err);
      setResolutionHistory([]);
    } finally {
      setLoadingResolutions(false);
    }
  };

  const calculateExpectedReward = () => {
    if (!selectedDomain || !amount) return 0;
    const stakeAmount = parseFloat(amount) || 0;
    const baseReward = (stakeAmount * selectedDomain.apy / 100) * (lockPeriod.days / 365);
    const confidenceMultiplier = 0.5 + (confidence / 100) * 0.5; // 50-100% of base
    return baseReward * lockPeriod.multiplier * confidenceMultiplier;
  };

  // Calculate expected reward for a staked position
  const calculatePositionReward = (stake: ActiveStake) => {
    // Find the domain to get APY
    const domain = predictionDomains.find(d => d.id === stake.domain);
    if (!domain) return stake.reward; // Fall back to actual reward if domain not found

    // Calculate lock multiplier based on lock_days
    let lockMultiplier = 1.0;
    if (stake.lock_days >= 180) lockMultiplier = 2.0;
    else if (stake.lock_days >= 90) lockMultiplier = 1.5;
    else if (stake.lock_days >= 30) lockMultiplier = 1.2;
    else if (stake.lock_days >= 7) lockMultiplier = 1.0;

    const baseReward = (stake.amount * domain.apy / 100) * (stake.lock_days / 365);
    const confidenceMultiplier = 0.5 + (stake.confidence / 100) * 0.5;
    return baseReward * lockMultiplier * confidenceMultiplier;
  };

  const getRiskColor = (risk: 'low' | 'medium' | 'high') => {
    switch (risk) {
      case 'low': return 'text-green-400 bg-green-500/20';
      case 'medium': return 'text-yellow-400 bg-yellow-500/20';
      case 'high': return 'text-red-400 bg-red-500/20';
    }
  };

  const handleStake = async () => {
    if (!selectedDomain || !amount || parseFloat(amount) <= 0) {
      setError('Please select a domain and enter a valid amount');
      return;
    }

    // v1.4.3: Validate prediction value
    const predValue = parseFloat(predictionValue);
    if (isNaN(predValue) || predValue <= 0) {
      setError('Please enter your predicted value for the oracle outcome');
      return;
    }

    // Backend validates balance - no client-side check needed
    setIsStaking(true);
    setError(null);

    try {
      const response = await qnkAPI.stakePrediction({
        domain: selectedDomain.id,
        amount: parseFloat(amount),
        confidence,
        lockDays: lockPeriod.days,
        walletAddress,
        predictionValue: predValue  // v1.4.3: User's predicted value
      });

      if (response.success) {
        setStakeSuccess(true);
        setTimeout(() => {
          setStakeSuccess(false);
          setAmount('');
          setSelectedDomain(null);
          setConfidence(50);
          setPredictionValue('');  // v1.4.3: Reset prediction value
          onStakeSuccess?.();
          fetchActiveStakes();
        }, 2000);
      } else {
        setError(response.error || 'Failed to stake. Please try again.');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to stake. Please try again.');
    } finally {
      setIsStaking(false);
    }
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(num);
  };

  const formatTimeRemaining = (timestamp: number) => {
    const diff = timestamp - Date.now();
    const days = Math.floor(diff / (24 * 60 * 60 * 1000));
    if (days > 0) return `${days}d remaining`;
    const hours = Math.floor(diff / (60 * 60 * 1000));
    return `${hours}h remaining`;
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 overflow-y-auto"
        onClick={onClose}
      >
        <div className="flex min-h-full items-center justify-center p-4">
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-gradient-to-br from-slate-900 via-purple-950/50 to-slate-900 rounded-2xl w-full max-w-2xl flex flex-col max-h-[90vh] border border-purple-500/20 shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-purple-500/20">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/30">
                <Brain className="w-6 h-6 text-purple-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">QNO Prediction Staking</h2>
                <p className="text-xs text-gray-400">Stake QUG to earn rewards from AI predictions</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/10 transition-colors"
            >
              <X className="w-5 h-5 text-gray-400" />
            </button>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-purple-500/20">
            <button
              onClick={() => setActiveTab('stake')}
              className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
                activeTab === 'stake'
                  ? 'text-purple-400 border-b-2 border-purple-400 bg-purple-500/10'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <Coins className="w-4 h-4 inline mr-2" />
              New Stake
            </button>
            <button
              onClick={() => setActiveTab('positions')}
              className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
                activeTab === 'positions'
                  ? 'text-purple-400 border-b-2 border-purple-400 bg-purple-500/10'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <BarChart3 className="w-4 h-4 inline mr-2" />
              Positions ({activeStakes.length})
            </button>
            <button
              onClick={() => setActiveTab('resolution')}
              className={`flex-1 py-3 px-4 text-sm font-medium transition-colors ${
                activeTab === 'resolution'
                  ? 'text-purple-400 border-b-2 border-purple-400 bg-purple-500/10'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <TrendingUp className="w-4 h-4 inline mr-2" />
              Resolution
            </button>
          </div>

          {/* Content */}
          <div className="p-4 overflow-y-auto flex-1">
            {activeTab === 'stake' ? (
              <div className="space-y-4">
                {/* Domain Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Select Prediction Domain
                  </label>
                  {loadingDomains ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="w-6 h-6 animate-spin text-purple-400" />
                    </div>
                  ) : predictionDomains.length === 0 ? (
                    <div className="text-center py-8 text-gray-400">
                      <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p>No prediction domains available</p>
                    </div>
                  ) : (
                  <div className="grid grid-cols-2 gap-2">
                    {predictionDomains.map((domain) => (
                      <motion.button
                        key={domain.id}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => setSelectedDomain(domain)}
                        className={`p-3 rounded-xl text-left transition-all ${
                          selectedDomain?.id === domain.id
                            ? 'bg-purple-500/30 border-2 border-purple-400'
                            : 'bg-white/5 border border-white/10 hover:border-purple-500/50'
                        }`}
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-purple-400">{domain.icon}</span>
                          <span className="text-sm font-medium text-white">{domain.name}</span>
                        </div>
                        <p className="text-xs text-gray-400 mb-2">{domain.description}</p>
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-green-400">APY: {domain.apy}%</span>
                          <span className={`px-1.5 py-0.5 rounded ${getRiskColor(domain.riskLevel)}`}>
                            {domain.riskLevel}
                          </span>
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                          Accuracy: {domain.accuracy.toFixed(1)}%
                        </div>
                      </motion.button>
                    ))}
                  </div>
                  )}
                </div>

                {/* Amount Input */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Stake Amount
                  </label>
                  <div className="relative">
                    <input
                      type="number"
                      value={amount}
                      onChange={(e) => setAmount(e.target.value)}
                      placeholder="0.00"
                      className="w-full p-3 pr-24 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:border-purple-500 focus:ring-1 focus:ring-purple-500 outline-none"
                    />
                    <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
                      <span className="text-gray-400 text-sm">QUG</span>
                      <button
                        onClick={() => setAmount(availableBalance.toString())}
                        className="px-2 py-1 rounded bg-purple-500/20 text-purple-400 text-xs hover:bg-purple-500/30 transition-colors"
                      >
                        MAX
                      </button>
                    </div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Available: {formatNumber(availableBalance)} QUG
                  </p>
                </div>

                {/* Prediction Value Input - v1.4.3 */}
                {selectedDomain && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                  >
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Your Prediction
                      <span className="text-xs text-gray-500 ml-2">
                        (What value do you predict for {selectedDomain.name}?)
                      </span>
                    </label>
                    <div className="relative">
                      <input
                        type="number"
                        value={predictionValue}
                        onChange={(e) => setPredictionValue(e.target.value)}
                        placeholder={selectedDomain.id === 'gas-fees' ? '45.0' :
                                    selectedDomain.id === 'block-time' ? '12.0' :
                                    selectedDomain.id === 'network-load' ? '65.0' :
                                    selectedDomain.id === 'validator-uptime' ? '99.5' : '50.0'}
                        className="w-full p-3 pr-24 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:border-purple-500 focus:ring-1 focus:ring-purple-500 outline-none"
                      />
                      <div className="absolute right-3 top-1/2 -translate-y-1/2">
                        <span className="text-gray-400 text-sm">
                          {selectedDomain.id === 'gas-fees' ? 'gwei' :
                           selectedDomain.id === 'block-time' ? 'sec' :
                           selectedDomain.id === 'network-load' ? '%' :
                           selectedDomain.id === 'validator-uptime' ? '%' :
                           selectedDomain.id === 'defi-tvl' ? 'M USD' : 'units'}
                        </span>
                      </div>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      Enter your predicted value. If accurate (within threshold), you'll earn bonus rewards!
                    </p>
                  </motion.div>
                )}

                {/* Confidence Slider */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-medium text-gray-300">
                      Confidence Level
                    </label>
                    <span className="text-sm text-purple-400">{confidence}%</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={confidence}
                    onChange={(e) => setConfidence(parseInt(e.target.value))}
                    className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Low risk, lower rewards</span>
                    <span>High risk, higher rewards</span>
                  </div>
                </div>

                {/* Lock Period */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Lock Period
                  </label>
                  <div className="grid grid-cols-4 gap-2">
                    {LOCK_PERIODS.map((period) => (
                      <button
                        key={period.days}
                        onClick={() => setLockPeriod(period)}
                        className={`p-2 rounded-lg text-center transition-all ${
                          lockPeriod.days === period.days
                            ? 'bg-purple-500/30 border border-purple-400 text-white'
                            : 'bg-white/5 border border-white/10 text-gray-400 hover:border-purple-500/50'
                        }`}
                      >
                        <div className="text-sm font-medium">{period.label}</div>
                        <div className="text-xs text-purple-400">{period.multiplier}x</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Expected Reward Preview */}
                {selectedDomain && amount && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/20"
                  >
                    <div className="flex items-center gap-2 mb-3">
                      <TrendingUp className="w-5 h-5 text-purple-400" />
                      <span className="text-sm font-medium text-white">Expected Returns</span>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Base APY</p>
                        <p className="text-white font-medium">{selectedDomain.apy}%</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Lock Multiplier</p>
                        <p className="text-white font-medium">{lockPeriod.multiplier}x</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Est. Reward</p>
                        <p className="text-green-400 font-medium">+{formatNumber(calculateExpectedReward())} QUG</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Effective APY</p>
                        <p className="text-purple-400 font-medium">
                          {((selectedDomain.apy * lockPeriod.multiplier * (0.5 + confidence/200))).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <div className="mt-3 p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/20 flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                      <p className="text-xs text-yellow-200">
                        Higher confidence = higher potential rewards but also higher slashing risk if predictions are wrong.
                      </p>
                    </div>
                  </motion.div>
                )}

                {/* Error Message */}
                {error && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="p-3 rounded-xl bg-red-500/20 border border-red-500/30 text-red-400 text-sm flex items-center gap-2"
                  >
                    <AlertTriangle className="w-4 h-4" />
                    {error}
                  </motion.div>
                )}

                {/* Success Message */}
                {stakeSuccess && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="p-3 rounded-xl bg-green-500/20 border border-green-500/30 text-green-400 text-sm flex items-center gap-2"
                  >
                    <CheckCircle className="w-4 h-4" />
                    Successfully staked! Your prediction is now active.
                  </motion.div>
                )}

                {/* Stake Button */}
                <button
                  type="button"
                  onClick={handleStake}
                  disabled={isStaking}
                  className={`w-full py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all ${
                    isStaking
                      ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                      : 'bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:from-purple-600 hover:to-pink-600'
                  }`}
                >
                  {isStaking ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Staking...
                    </>
                  ) : (
                    <>
                      <Lock className="w-5 h-5" />
                      Stake {amount ? formatNumber(parseFloat(amount) || 0) : '0'} QUG
                    </>
                  )}
                </button>
              </div>
            ) : activeTab === 'positions' ? (
              /* Positions Tab */
              <div className="space-y-3">
                {loadingStakes ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="w-6 h-6 animate-spin text-purple-400" />
                  </div>
                ) : activeStakes.length === 0 ? (
                  <div className="text-center py-8">
                    <Brain className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                    <p className="text-gray-400">No active staking positions</p>
                    <p className="text-xs text-gray-500 mt-1">
                      Stake QUG to start earning prediction rewards
                    </p>
                  </div>
                ) : (
                  activeStakes.map((stake) => (
                    <motion.div
                      key={stake.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-purple-500/30 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-white">{stake.domain_name}</span>
                        <span className={`px-2 py-0.5 rounded text-xs ${
                          stake.status === 'unlocked'
                            ? 'bg-green-500/20 text-green-400'
                            : stake.status === 'active'
                            ? 'bg-purple-500/20 text-purple-400'
                            : 'bg-gray-500/20 text-gray-400'
                        }`}>
                          {stake.status}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <p className="text-gray-500 text-xs">Staked</p>
                          <p className="text-white">{formatNumber(stake.amount)} QUG</p>
                        </div>
                        <div>
                          <p className="text-gray-500 text-xs">Confidence</p>
                          <p className="text-white">{stake.confidence.toFixed(0)}%</p>
                        </div>
                        <div>
                          <p className="text-gray-500 text-xs">Est. Reward</p>
                          <p className="text-green-400">+{formatNumber(calculatePositionReward(stake))} QUG</p>
                        </div>
                        <div>
                          <p className="text-gray-500 text-xs">Lock Period</p>
                          <p className="text-white flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {formatTimeRemaining(stake.unlocks_at)}
                          </p>
                        </div>
                      </div>
                      {stake.status === 'unlocked' && (
                        <motion.button
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className="w-full mt-3 py-2 rounded-lg bg-green-500/20 border border-green-500/30 text-green-400 text-sm font-medium hover:bg-green-500/30 transition-colors"
                        >
                          Claim Rewards
                        </motion.button>
                      )}
                    </motion.div>
                  ))
                )}

                {/* Total Stats */}
                {activeStakes.length > 0 && (
                  <div className="p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/20">
                    <div className="flex items-center gap-2 mb-3">
                      <BarChart3 className="w-5 h-5 text-purple-400" />
                      <span className="text-sm font-medium text-white">Portfolio Summary</span>
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Total Staked</p>
                        <p className="text-white font-medium">
                          {formatNumber(activeStakes.reduce((sum, s) => sum + s.amount, 0))} QUG
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-400">Total Rewards</p>
                        <p className="text-green-400 font-medium">
                          +{formatNumber(activeStakes.reduce((sum, s) => sum + s.reward, 0))} QUG
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-400">Active Positions</p>
                        <p className="text-purple-400 font-medium">{activeStakes.length}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : activeTab === 'resolution' ? (
              /* Resolution History Tab - v1.4.3 */
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-medium text-white">Oracle Resolution History</h3>
                  <button
                    onClick={fetchResolutionHistory}
                    className="p-2 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 transition-colors"
                    title="Refresh"
                  >
                    <Loader2 className={`w-4 h-4 text-purple-400 ${loadingResolutions ? 'animate-spin' : ''}`} />
                  </button>
                </div>

                {loadingResolutions ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="w-6 h-6 animate-spin text-purple-400" />
                  </div>
                ) : resolutionHistory.length === 0 ? (
                  <div className="text-center py-8">
                    <TrendingUp className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                    <p className="text-gray-400">No resolution history yet</p>
                    <p className="text-xs text-gray-500 mt-1">
                      Resolution events will appear here after oracle updates
                    </p>
                  </div>
                ) : (
                  resolutionHistory.map((result, idx) => (
                    <motion.div
                      key={result.stake_id || idx}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.05 }}
                      className={`p-4 rounded-xl border ${
                        result.is_accurate
                          ? 'bg-green-500/10 border-green-500/30'
                          : 'bg-red-500/10 border-red-500/30'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-white">{result.domain_name}</span>
                        <span className={`px-2 py-0.5 rounded text-xs ${
                          result.is_accurate
                            ? 'bg-green-500/20 text-green-400'
                            : 'bg-red-500/20 text-red-400'
                        }`}>
                          {result.is_accurate ? 'Accurate' : 'Inaccurate'}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <p className="text-gray-500 text-xs">Your Prediction</p>
                          <p className="text-white">{result.predicted_value?.toFixed(2) || 'N/A'}</p>
                        </div>
                        <div>
                          <p className="text-gray-500 text-xs">Actual Value</p>
                          <p className="text-white">{result.actual_value?.toFixed(2) || 'N/A'}</p>
                        </div>
                        <div>
                          <p className="text-gray-500 text-xs">Accuracy Score</p>
                          <p className={result.is_accurate ? 'text-green-400' : 'text-red-400'}>
                            {((result.accuracy_score || 0) * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div>
                          <p className="text-gray-500 text-xs">Reward Adjustment</p>
                          <p className={result.reward_adjustment >= 0 ? 'text-green-400' : 'text-red-400'}>
                            {result.reward_adjustment >= 0 ? '+' : ''}{(result.reward_adjustment / 1e24).toFixed(2)} QUG
                          </p>
                        </div>
                      </div>
                      {result.slashing_applied > 0 && (
                        <div className="mt-2 p-2 rounded-lg bg-red-500/20 flex items-center gap-2">
                          <AlertTriangle className="w-4 h-4 text-red-400" />
                          <span className="text-xs text-red-400">
                            Slashing applied: -{(result.slashing_applied / 1e24).toFixed(2)} QUG
                          </span>
                        </div>
                      )}
                      <div className="mt-2 text-xs text-gray-500">
                        Resolved: {new Date((result.resolved_at || 0) * 1000).toLocaleString()}
                      </div>
                    </motion.div>
                  ))
                )}

                {/* Resolution Info */}
                <div className="p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/20">
                  <div className="flex items-center gap-2 mb-3">
                    <Info className="w-5 h-5 text-purple-400" />
                    <span className="text-sm font-medium text-white">How Resolution Works</span>
                  </div>
                  <ul className="text-xs text-gray-400 space-y-1">
                    <li>• Oracle data is fetched hourly from multiple sources</li>
                    <li>• Your prediction is compared to the actual oracle value</li>
                    <li>• Accuracy within threshold = bonus rewards</li>
                    <li>• Consecutive failures may result in slashing</li>
                    <li>• Higher confidence = higher rewards OR penalties</li>
                  </ul>
                </div>
              </div>
            ) : null}
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-purple-500/20 bg-black/20">
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Info className="w-4 h-4" />
              <span>
                Powered by Quantum Neural Oracle (QNO) - Decentralized AI predictions with zkML proofs
              </span>
            </div>
          </div>
        </motion.div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
