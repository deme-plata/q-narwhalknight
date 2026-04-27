import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, PieChart, TrendingUp, TrendingDown, Clock, Shield,
  ArrowUpRight, ArrowDownRight, RefreshCw, Info, Wallet,
  BarChart3, Target, Layers, ChevronDown, ChevronUp, Plus, Minus, AlertCircle, Loader2
} from 'lucide-react';
import { qnkAPI } from '../services/api';

interface IndexComponent {
  symbol: string;
  name: string;
  weight: number;
  price: number;
  change24h: number;
}

interface IndexData {
  methodology: 'market_cap_weighted' | 'equal_weighted' | 'custom';
  rebalanceFrequency: string;
  managementFee: number;
  performanceFee: number;
  navPerShare: number;
  totalAUM: number;
  components: IndexComponent[];
  lastRebalance: string;
  nextRebalance: string;
  inceptionDate: string;
  ytdReturn: number;
  allTimeReturn: number;
}

interface Token {
  id: string;
  symbol: string;
  name: string;
  balance: number;
  price: number;
  change24h: number;
  indexData?: IndexData;
}

interface IndexFundModalProps {
  token: Token;
  onClose: () => void;
}

export default function IndexFundModal({ token, onClose }: IndexFundModalProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'components' | 'trade' | 'governance'>('overview');
  const [showComponentDetails, setShowComponentDetails] = useState<string | null>(null);
  const [tradeMode, setTradeMode] = useState<'mint' | 'redeem'>('mint');
  const [tradeAmount, setTradeAmount] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingData, setIsLoadingData] = useState(true);
  const [liveIndexData, setLiveIndexData] = useState<IndexData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [userQugusdBalance, setUserQugusdBalance] = useState(0);

  // Fetch real prices from oracle and calculate NAV
  const fetchLiveData = useCallback(async () => {
    setIsLoadingData(true);
    setError(null);

    try {
      // Define the base index configuration from token.indexData
      const baseConfig = token.indexData;
      if (!baseConfig) {
        throw new Error('Index configuration not found');
      }

      // Fetch live prices for all components
      const updatedComponents: IndexComponent[] = [];
      let totalWeightedPrice = 0;

      for (const component of baseConfig.components) {
        try {
          // Try to fetch real price from oracle
          const oracleResponse = await qnkAPI.getOraclePrice(`${component.symbol}/USD`);

          if (oracleResponse.success && oracleResponse.data) {
            const livePrice = oracleResponse.data.price;
            const change24h = oracleResponse.data.change_24h || 0;

            updatedComponents.push({
              ...component,
              price: livePrice,
              change24h: change24h,
            });

            totalWeightedPrice += livePrice * (component.weight / 100);
          } else {
            // Use static price if oracle fails
            updatedComponents.push(component);
            totalWeightedPrice += component.price * (component.weight / 100);
          }
        } catch (err) {
          // Use static price on error
          console.warn(`Failed to fetch price for ${component.symbol}:`, err);
          updatedComponents.push(component);
          totalWeightedPrice += component.price * (component.weight / 100);
        }
      }

      // Calculate live NAV based on weighted component prices
      // NAV = sum(component_price * weight) normalized to base price
      const baseNAV = baseConfig.navPerShare;
      const priceMultiplier = totalWeightedPrice / 100; // Normalize
      const liveNAV = baseNAV * (1 + (priceMultiplier - 1) * 0.1); // Dampened effect

      // v4.0.3: Fetch user's QUGUSD balance using authenticated multi-token API
      // Previously used getTokenBalance('QUGUSD') which failed because 'QUGUSD' is not a valid hex address
      try {
        const multiTokenResponse = await qnkAPI.getMultiTokenBalance();
        if (multiTokenResponse.success && multiTokenResponse.data) {
          const qugusdData = multiTokenResponse.data.tokens?.QUGUSD;
          if (qugusdData) {
            // Balance comes as base units (24 decimals), convert to display
            const rawBalance = parseFloat(qugusdData.balance || '0');
            setUserQugusdBalance(rawBalance > 1e15 ? rawBalance / 1e24 : rawBalance);
            console.log('💰 [IndexFund] QUGUSD balance:', rawBalance > 1e15 ? rawBalance / 1e24 : rawBalance);
          }
        }
      } catch (err) {
        console.warn('Failed to fetch QUGUSD balance:', err);
      }

      // Update index data with live values
      setLiveIndexData({
        ...baseConfig,
        components: updatedComponents,
        navPerShare: liveNAV,
        // Keep other values from config but could be enhanced with more API calls
      });
    } catch (err) {
      console.error('Failed to fetch live index data:', err);
      setError('Failed to load live data. Showing cached values.');
      // Fall back to static data
      setLiveIndexData(token.indexData || null);
    } finally {
      setIsLoadingData(false);
    }
  }, [token.indexData]);

  // Fetch live data on mount and every 30 seconds
  useEffect(() => {
    fetchLiveData();
    const interval = setInterval(fetchLiveData, 30000);
    return () => clearInterval(interval);
  }, [fetchLiveData]);

  // Use live data if available, otherwise use token data
  const indexData: IndexData = liveIndexData || token.indexData || {
    methodology: 'market_cap_weighted',
    rebalanceFrequency: 'Monthly',
    managementFee: 0.5,
    performanceFee: 10,
    navPerShare: token.price,
    totalAUM: 0,
    components: [],
    lastRebalance: new Date().toISOString().split('T')[0],
    nextRebalance: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    inceptionDate: '2025-01-01',
    ytdReturn: 0,
    allTimeReturn: 0,
  };

  const handleTrade = async () => {
    if (!tradeAmount || parseFloat(tradeAmount) <= 0) return;

    const walletAddress = localStorage.getItem('walletAddress');
    if (!walletAddress) {
      setError('Please connect your wallet first');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const amount = parseFloat(tradeAmount);

      if (tradeMode === 'mint') {
        // Check if user has enough QUGUSD
        if (amount > userQugusdBalance) {
          throw new Error(`Insufficient QUGUSD balance. You have ${userQugusdBalance.toFixed(2)} QUGUSD`);
        }

        // v4.0.3: Mint index shares by swapping QUGUSD → Index token
        const sharesToMint = amount / indexData.navPerShare;
        console.log(`Minting ${sharesToMint.toFixed(6)} ${token.symbol} shares for ${amount} QUGUSD`);

        const amountInBaseUnits = Math.floor(amount * 1e24);
        const minOut = Math.floor(sharesToMint * 0.95 * 1e24); // 5% slippage

        const result = await qnkAPI.executeSwap({
          from_token: 'QUGUSD',
          to_token: token.id,
          amount_in: amountInBaseUnits,
          min_amount_out: minOut,
          wallet_address: walletAddress,
        });

        if (result.success) {
          alert(`Successfully minted ${sharesToMint.toFixed(6)} ${token.symbol} shares for ${amount.toFixed(2)} QUGUSD`);
          fetchLiveData(); // Refresh balances
        } else {
          throw new Error(result.error || 'Mint transaction failed');
        }
      } else {
        // Check if user has enough index shares
        if (amount > token.balance) {
          throw new Error(`Insufficient ${token.symbol} balance. You have ${(Number(token.balance) || 0).toFixed(6)} shares`);
        }

        // v4.0.3: Redeem index shares by swapping Index token → QUGUSD
        const qugusdToReceive = amount * indexData.navPerShare;
        console.log(`Redeeming ${amount} ${token.symbol} shares for ${qugusdToReceive.toFixed(2)} QUGUSD`);

        const amountInBaseUnits = Math.floor(amount * 1e24);
        const minOut = Math.floor(qugusdToReceive * 0.95 * 1e24); // 5% slippage

        const result = await qnkAPI.executeSwap({
          from_token: token.id,
          to_token: 'QUGUSD',
          amount_in: amountInBaseUnits,
          min_amount_out: minOut,
          wallet_address: walletAddress,
        });

        if (result.success) {
          alert(`Successfully redeemed ${amount.toFixed(6)} ${token.symbol} shares for ${qugusdToReceive.toFixed(2)} QUGUSD`);
          fetchLiveData(); // Refresh balances
        } else {
          throw new Error(result.error || 'Redeem transaction failed');
        }
      }

      setTradeAmount('');
    } catch (err: any) {
      setError(err.message || 'Trade failed');
    } finally {
      setIsLoading(false);
    }
  };

  const formatCurrency = (value: number) => {
    if (value >= 1000000000) return `$${(value / 1000000000).toFixed(2)}B`;
    if (value >= 1000000) return `$${(value / 1000000).toFixed(2)}M`;
    if (value >= 1000) return `$${(value / 1000).toFixed(2)}K`;
    return `$${value.toFixed(2)}`;
  };

  const estimatedOutput = tradeMode === 'mint'
    ? parseFloat(tradeAmount || '0') / indexData.navPerShare
    : parseFloat(tradeAmount || '0') * indexData.navPerShare;

  // Calculate weighted 24h change
  const weightedChange24h = indexData.components.reduce(
    (sum, comp) => sum + (comp.change24h * comp.weight / 100),
    0
  );

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 overflow-y-auto bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      >
        <div className="flex min-h-full items-center justify-center p-4">
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="relative w-full max-w-4xl flex flex-col max-h-[90vh] rounded-2xl"
          style={{
            background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.98) 100%)',
            border: '2px solid rgba(139, 92, 246, 0.3)',
            boxShadow: '0 0 60px rgba(139, 92, 246, 0.2)',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="p-6 border-b border-purple-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div
                  className="w-14 h-14 rounded-xl flex items-center justify-center"
                  style={{
                    background: 'linear-gradient(135deg, #8B5CF6 0%, #A855F7 50%, #D946EF 100%)',
                    boxShadow: '0 0 20px rgba(139, 92, 246, 0.4)',
                  }}
                >
                  <PieChart className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-amber-50">{token.name}</h2>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-purple-400 font-semibold">{token.symbol}</span>
                    <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-purple-500/20 text-purple-300 border border-purple-500/30">
                      Index Fund
                    </span>
                    <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-violet-500/20 text-violet-300 border border-violet-500/30">
                      {indexData.methodology.replace(/_/g, ' ')}
                    </span>
                    {isLoadingData && (
                      <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-2xl font-bold text-amber-50">${indexData.navPerShare.toFixed(2)}</div>
                  <div className={`flex items-center gap-1 ${weightedChange24h >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {weightedChange24h >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                    <span className="font-medium">{Math.abs(weightedChange24h).toFixed(2)}% (24h)</span>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 rounded-xl text-amber-200/60 hover:text-amber-200 hover:bg-purple-500/10 transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>
            </div>

            {/* Error Banner */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4 p-3 rounded-xl bg-red-500/10 border border-red-500/30 flex items-center gap-2"
              >
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                <span className="text-red-300 text-sm">{error}</span>
              </motion.div>
            )}

            {/* Tab Navigation */}
            <div className="flex gap-2 mt-6">
              {(['overview', 'components', 'trade', 'governance'] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                    activeTab === tab
                      ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40'
                      : 'text-amber-200/60 hover:text-amber-200 hover:bg-purple-500/10'
                  }`}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto flex-1">
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Key Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { label: 'Total AUM', value: formatCurrency(indexData.totalAUM), icon: BarChart3, color: 'purple' },
                    { label: 'Your Balance', value: `${(Number(token.balance) || 0).toFixed(4)} ${token.symbol}`, icon: Wallet, color: 'blue' },
                    { label: 'YTD Return', value: `${indexData.ytdReturn >= 0 ? '+' : ''}${indexData.ytdReturn}%`, icon: TrendingUp, color: 'green' },
                    { label: 'All-Time Return', value: `${indexData.allTimeReturn >= 0 ? '+' : ''}${indexData.allTimeReturn}%`, icon: Target, color: 'violet' },
                  ].map((metric, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.1 }}
                      className="p-4 rounded-xl"
                      style={{
                        background: `linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%)`,
                        border: `1px solid rgba(139, 92, 246, 0.2)`,
                      }}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <metric.icon className="w-4 h-4 text-purple-400" />
                        <span className="text-xs text-amber-200/60">{metric.label}</span>
                      </div>
                      <div className="text-lg font-bold text-amber-50">{metric.value}</div>
                    </motion.div>
                  ))}
                </div>

                {/* Fund Details */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div
                    className="p-5 rounded-xl"
                    style={{
                      background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%)',
                      border: '1px solid rgba(139, 92, 246, 0.15)',
                    }}
                  >
                    <h3 className="text-lg font-semibold text-amber-50 mb-4 flex items-center gap-2">
                      <Info className="w-5 h-5 text-purple-400" />
                      Fund Information
                    </h3>
                    <div className="space-y-3">
                      {[
                        { label: 'Methodology', value: indexData.methodology.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) },
                        { label: 'Rebalance Frequency', value: indexData.rebalanceFrequency },
                        { label: 'Management Fee', value: `${indexData.managementFee}% annually` },
                        { label: 'Performance Fee', value: `${indexData.performanceFee}% of profits` },
                        { label: 'Inception Date', value: new Date(indexData.inceptionDate).toLocaleDateString() },
                      ].map((item, idx) => (
                        <div key={idx} className="flex justify-between items-center">
                          <span className="text-amber-200/60 text-sm">{item.label}</span>
                          <span className="text-amber-50 font-medium">{item.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div
                    className="p-5 rounded-xl"
                    style={{
                      background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%)',
                      border: '1px solid rgba(139, 92, 246, 0.15)',
                    }}
                  >
                    <h3 className="text-lg font-semibold text-amber-50 mb-4 flex items-center gap-2">
                      <RefreshCw className="w-5 h-5 text-purple-400" />
                      Rebalance Schedule
                    </h3>
                    <div className="space-y-4">
                      <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/20">
                        <div className="text-xs text-green-400 mb-1">Last Rebalance</div>
                        <div className="text-amber-50 font-medium">{new Date(indexData.lastRebalance).toLocaleDateString()}</div>
                      </div>
                      <div className="p-3 rounded-lg bg-purple-500/10 border border-purple-500/20">
                        <div className="text-xs text-purple-400 mb-1">Next Rebalance</div>
                        <div className="text-amber-50 font-medium">{new Date(indexData.nextRebalance).toLocaleDateString()}</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Composition Preview */}
                <div
                  className="p-5 rounded-xl"
                  style={{
                    background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%)',
                    border: '1px solid rgba(139, 92, 246, 0.15)',
                  }}
                >
                  <h3 className="text-lg font-semibold text-amber-50 mb-4 flex items-center gap-2">
                    <Layers className="w-5 h-5 text-purple-400" />
                    Composition ({indexData.components.length} assets)
                  </h3>
                  <div className="flex gap-1 h-8 rounded-lg overflow-hidden mb-4">
                    {indexData.components.map((comp, idx) => {
                      const colors = ['#8B5CF6', '#A855F7', '#D946EF', '#22C55E', '#3B82F6'];
                      return (
                        <motion.div
                          key={comp.symbol}
                          initial={{ width: 0 }}
                          animate={{ width: `${comp.weight}%` }}
                          transition={{ delay: idx * 0.1, duration: 0.5 }}
                          className="h-full flex items-center justify-center text-xs font-medium text-white"
                          style={{ backgroundColor: colors[idx % colors.length] }}
                          title={`${comp.symbol}: ${comp.weight}%`}
                        >
                          {comp.weight >= 10 && comp.symbol}
                        </motion.div>
                      );
                    })}
                  </div>
                  <div className="flex flex-wrap gap-3">
                    {indexData.components.map((comp, idx) => {
                      const colors = ['purple', 'violet', 'fuchsia', 'green', 'blue'];
                      return (
                        <div key={comp.symbol} className="flex items-center gap-2">
                          <div className={`w-3 h-3 rounded-full`} style={{ backgroundColor: ['#8B5CF6', '#A855F7', '#D946EF', '#22C55E', '#3B82F6'][idx % 5] }} />
                          <span className="text-sm text-amber-200/60">{comp.symbol}</span>
                          <span className="text-sm font-medium text-amber-50">{comp.weight}%</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'components' && (
              <div className="space-y-3">
                {indexData.components.length === 0 ? (
                  <div className="text-center py-12 text-amber-200/60">
                    <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No components configured</p>
                  </div>
                ) : (
                  indexData.components.map((component, idx) => (
                    <motion.div
                      key={component.symbol}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.05 }}
                      className="rounded-xl overflow-hidden"
                      style={{
                        background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%)',
                        border: '1px solid rgba(139, 92, 246, 0.15)',
                      }}
                    >
                      <button
                        onClick={() => setShowComponentDetails(showComponentDetails === component.symbol ? null : component.symbol)}
                        className="w-full p-4 flex items-center justify-between hover:bg-purple-500/5 transition-colors"
                      >
                        <div className="flex items-center gap-4">
                          <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                            <span className="text-purple-400 font-bold text-sm">{component.symbol.slice(0, 2)}</span>
                          </div>
                          <div className="text-left">
                            <div className="font-semibold text-amber-50">{component.name}</div>
                            <div className="text-sm text-amber-200/60">{component.symbol}</div>
                          </div>
                        </div>

                        <div className="flex items-center gap-6">
                          <div className="text-right">
                            <div className="font-semibold text-amber-50">{component.weight}%</div>
                            <div className="text-xs text-amber-200/60">Weight</div>
                          </div>
                          <div className="text-right">
                            <div className="font-semibold text-amber-50">${component.price.toLocaleString()}</div>
                            <div className={`text-xs ${component.change24h >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {component.change24h >= 0 ? '+' : ''}{component.change24h.toFixed(2)}%
                            </div>
                          </div>
                          {showComponentDetails === component.symbol ? (
                            <ChevronUp className="w-5 h-5 text-purple-400" />
                          ) : (
                            <ChevronDown className="w-5 h-5 text-amber-200/40" />
                          )}
                        </div>
                      </button>

                      <AnimatePresence>
                        {showComponentDetails === component.symbol && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="border-t border-purple-500/10"
                          >
                            <div className="p-4 grid grid-cols-3 gap-4">
                              <div className="p-3 rounded-lg bg-slate-800/50">
                                <div className="text-xs text-amber-200/60 mb-1">Value in Index</div>
                                <div className="font-semibold text-amber-50">
                                  {formatCurrency(indexData.totalAUM * (component.weight / 100))}
                                </div>
                              </div>
                              <div className="p-3 rounded-lg bg-slate-800/50">
                                <div className="text-xs text-amber-200/60 mb-1">Target Weight</div>
                                <div className="font-semibold text-amber-50">{component.weight}%</div>
                              </div>
                              <div className="p-3 rounded-lg bg-slate-800/50">
                                <div className="text-xs text-amber-200/60 mb-1">24h Change</div>
                                <div className={`font-semibold ${component.change24h >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                  {component.change24h >= 0 ? '+' : ''}{component.change24h.toFixed(2)}%
                                </div>
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  ))
                )}
              </div>
            )}

            {activeTab === 'trade' && (
              <div className="max-w-lg mx-auto space-y-6">
                {/* Trade Mode Toggle */}
                <div className="flex p-1 rounded-xl bg-slate-800/50">
                  <button
                    onClick={() => setTradeMode('mint')}
                    className={`flex-1 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
                      tradeMode === 'mint'
                        ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                        : 'text-amber-200/60 hover:text-amber-200'
                    }`}
                  >
                    <Plus className="w-4 h-4" />
                    Mint Shares
                  </button>
                  <button
                    onClick={() => setTradeMode('redeem')}
                    className={`flex-1 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
                      tradeMode === 'redeem'
                        ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                        : 'text-amber-200/60 hover:text-amber-200'
                    }`}
                  >
                    <Minus className="w-4 h-4" />
                    Redeem Shares
                  </button>
                </div>

                {/* Trade Input */}
                <div
                  className="p-5 rounded-xl"
                  style={{
                    background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%)',
                    border: '1px solid rgba(139, 92, 246, 0.2)',
                  }}
                >
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-sm text-amber-200/60">
                      {tradeMode === 'mint' ? 'You Pay (QUGUSD)' : `You Redeem (${token.symbol})`}
                    </span>
                    <span className="text-sm text-amber-200/60">
                      Balance: {tradeMode === 'mint' ? (Number(userQugusdBalance) || 0).toFixed(2) : (Number(token.balance) || 0).toFixed(4)}
                    </span>
                  </div>
                  <input
                    type="number"
                    value={tradeAmount}
                    onChange={(e) => setTradeAmount(e.target.value)}
                    placeholder="0.00"
                    className="w-full bg-transparent text-3xl font-bold text-amber-50 outline-none placeholder:text-amber-200/20"
                  />
                  {tradeMode === 'mint' && (
                    <button
                      onClick={() => setTradeAmount(userQugusdBalance.toString())}
                      className="mt-2 text-xs text-purple-400 hover:text-purple-300"
                    >
                      Max
                    </button>
                  )}
                  {tradeMode === 'redeem' && (
                    <button
                      onClick={() => setTradeAmount(token.balance.toString())}
                      className="mt-2 text-xs text-purple-400 hover:text-purple-300"
                    >
                      Max
                    </button>
                  )}
                </div>

                {/* Arrow */}
                <div className="flex justify-center">
                  <div className="p-3 rounded-full bg-purple-500/20 border border-purple-500/30">
                    <ArrowDownRight className="w-5 h-5 text-purple-400" />
                  </div>
                </div>

                {/* Output Preview */}
                <div
                  className="p-5 rounded-xl"
                  style={{
                    background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%)',
                    border: '1px solid rgba(139, 92, 246, 0.2)',
                  }}
                >
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-sm text-amber-200/60">
                      {tradeMode === 'mint' ? `You Receive (${token.symbol})` : 'You Receive (QUGUSD)'}
                    </span>
                  </div>
                  <div className="text-3xl font-bold text-amber-50">
                    {estimatedOutput.toFixed(4)}
                  </div>
                </div>

                {/* Trade Details */}
                <div className="space-y-2 p-4 rounded-xl bg-slate-800/30">
                  <div className="flex justify-between text-sm">
                    <span className="text-amber-200/60">NAV per Share</span>
                    <span className="text-amber-50">${indexData.navPerShare.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-amber-200/60">
                      {tradeMode === 'mint' ? 'Mint Fee' : 'Redemption Fee'}
                    </span>
                    <span className="text-amber-50">0.1%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-amber-200/60">Slippage Tolerance</span>
                    <span className="text-amber-50">0.5%</span>
                  </div>
                </div>

                {/* Trade Button */}
                <motion.button
                  onClick={handleTrade}
                  disabled={!tradeAmount || parseFloat(tradeAmount) <= 0 || isLoading}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`w-full py-4 rounded-xl font-bold text-lg transition-all ${
                    tradeMode === 'mint'
                      ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white'
                      : 'bg-gradient-to-r from-red-500 to-rose-500 text-white'
                  } ${(!tradeAmount || parseFloat(tradeAmount) <= 0 || isLoading) && 'opacity-50 cursor-not-allowed'}`}
                >
                  {isLoading ? (
                    <span className="flex items-center justify-center gap-2">
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Processing...
                    </span>
                  ) : (
                    `${tradeMode === 'mint' ? 'Mint' : 'Redeem'} ${token.symbol}`
                  )}
                </motion.button>
              </div>
            )}

            {activeTab === 'governance' && (
              <div className="space-y-6">
                <div
                  className="p-6 rounded-xl text-center"
                  style={{
                    background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%)',
                    border: '1px solid rgba(139, 92, 246, 0.15)',
                  }}
                >
                  <Shield className="w-12 h-12 text-purple-400 mx-auto mb-4" />
                  <h3 className="text-xl font-bold text-amber-50 mb-2">Index Governance</h3>
                  <p className="text-amber-200/60 mb-6">
                    {token.symbol} holders can participate in governance decisions including
                    rebalancing strategies, fee adjustments, and component additions.
                  </p>
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
                      <div className="text-2xl font-bold text-amber-50">0</div>
                      <div className="text-xs text-amber-200/60">Active Proposals</div>
                    </div>
                    <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                      <div className="text-2xl font-bold text-green-400">0</div>
                      <div className="text-xs text-amber-200/60">Passed This Month</div>
                    </div>
                    <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                      <div className="text-2xl font-bold text-blue-400">0</div>
                      <div className="text-xs text-amber-200/60">Total Voters</div>
                    </div>
                  </div>
                  <p className="text-sm text-amber-200/40 italic">
                    Governance features coming soon. Smart contract integration required.
                  </p>
                </div>
              </div>
            )}
          </div>
        </motion.div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
