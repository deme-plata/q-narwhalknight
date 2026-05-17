import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Plus, ArrowRight, AlertCircle, Minus, Loader2 } from 'lucide-react';
import TokenIcon from './TokenIcon';
import { qnkAPI } from '../services/api';

// v2.4.7: Pool interface for remove liquidity
interface UserPool {
  pool_id: string;
  token0: string;
  token1: string;
  reserve0: number;
  reserve1: number;
  lp_tokens?: number;
}

interface Token {
  id: string;
  symbol: string;
  name: string;
  balance: number | string;  // v1.4.9: Handle both number and string balances from API
  price: number;
  icon: string;
}

// v1.4.9: Helper function to safely convert balance to number
const toNum = (val: number | string | undefined | null): number => {
  if (val === undefined || val === null) return 0;
  const num = typeof val === 'string' ? parseFloat(val) : val;
  return isNaN(num) ? 0 : num;
};

// v4.0.15: Format large balances compactly (e.g., 1e29 → "100.00Sx")
const formatLargeBalance = (val: number): string => {
  if (!isFinite(val) || isNaN(val)) return '0';
  if (val >= 1e30) return `${(val / 1e30).toFixed(2)} Nonillion`;
  if (val >= 1e27) return `${(val / 1e27).toFixed(2)} Octillion`;
  if (val >= 1e24) return `${(val / 1e24).toFixed(2)} Septillion`;
  if (val >= 1e21) return `${(val / 1e21).toFixed(2)} Sextillion`;
  if (val >= 1e18) return `${(val / 1e18).toFixed(2)} Quintillion`;
  if (val >= 1e15) return `${(val / 1e15).toFixed(2)} Quadrillion`;
  if (val >= 1e12) return `${(val / 1e12).toFixed(2)} Trillion`;
  if (val >= 1e9) return `${(val / 1e9).toFixed(2)} Billion`;
  if (val >= 1e6) return `${(val / 1e6).toFixed(2)} Million`;
  if (val >= 10000) return `${(val / 1000).toFixed(1)}K`;
  return val.toFixed(4);
};

// v3.2.22-beta: Helper to format very small prices (e.g., 1e-28)
// toFixed(8) can't display prices smaller than 0.00000001
const formatSmallPrice = (price: number): string => {
  if (price === 0) return '0';
  if (price >= 0.00000001) {
    // Normal small prices: show 8 decimals
    return price.toFixed(8);
  } else {
    // Very tiny prices: use scientific notation
    return price.toExponential(4);
  }
};

// v3.2.22-beta: Helper to format USD prices (with $ symbol)
const formatUsdPrice = (price: number): string => {
  if (price === 0) return '$0.00';
  if (price >= 0.01) {
    return `$${price.toFixed(2)}`;
  } else if (price >= 0.000001) {
    return `$${price.toFixed(6)}`;
  } else {
    return `$${price.toExponential(2)}`;
  }
};

interface LiquidityModalProps {
  token: Token;
  availableTokens: Token[];
  onClose: () => void;
  // v3.2.20-beta: Use string amounts to preserve precision for large numbers
  // JavaScript Number loses precision above ~9×10^15 (Number.MAX_SAFE_INTEGER)
  onAddLiquidity: (tokenA: string, tokenB: string, amountA: string, amountB: string) => void;
}

export default function LiquidityModal({ token, availableTokens, onClose, onAddLiquidity }: LiquidityModalProps) {
  // v1.0.50-beta: FIX - Filter out the current token from pair options to prevent QUG/QUG pools
  // This fixes the bug where both tokens are QUG and balance gets deducted twice
  const validPairTokens = availableTokens.filter(t => t.symbol !== token.symbol);

  // Initialize pair token to first valid option (not the current token)
  const defaultPairToken = validPairTokens.length > 0 ? validPairTokens[0].symbol : '';

  const [selectedPairToken, setSelectedPairToken] = useState<string>(defaultPairToken);
  const [amount1, setAmount1] = useState('');
  const [amount2, setAmount2] = useState('');
  const [mode, setMode] = useState<'add' | 'remove'>('add');
  // v2.4.7: Allow manual price ratio control for new pools
  const [autoCalculate, setAutoCalculate] = useState(false);

  // v2.4.7: Remove liquidity state
  const [userPools, setUserPools] = useState<UserPool[]>([]);
  const [selectedPool, setSelectedPool] = useState<UserPool | null>(null);
  const [removePercentage, setRemovePercentage] = useState(50);
  const [isLoadingPools, setIsLoadingPools] = useState(false);
  const [isRemoving, setIsRemoving] = useState(false);

  // v1.0.50-beta: Use filtered list to find pair token - prevents same-token selection
  const pairToken = validPairTokens.find(t => t.symbol === selectedPairToken);

  // v2.4.7: Calculate the resulting price based on input amounts
  const resultingPrice = amount1 && amount2 && parseFloat(amount1) > 0
    ? parseFloat(amount2) / parseFloat(amount1)
    : 0;

  // Calculate equivalent amount based on price ratio (only if auto-calculate is ON)
  const handleAmount1Change = (value: string) => {
    setAmount1(value);
    if (autoCalculate && value && pairToken) {
      const ratio = token.price / pairToken.price;
      setAmount2((parseFloat(value) * ratio).toFixed(6));
    }
  };

  const handleAmount2Change = (value: string) => {
    setAmount2(value);
    if (autoCalculate && value && pairToken) {
      const ratio = pairToken.price / token.price;
      setAmount1((parseFloat(value) * ratio).toFixed(6));
    }
  };

  const handleSubmit = () => {
    if (amount1 && amount2) {
      // v3.2.20-beta: Use parseFloat ONLY for validation, not for the actual value
      // The actual value is passed as string to preserve precision for large numbers
      const amt1ForValidation = parseFloat(amount1);
      const amt2ForValidation = parseFloat(amount2);

      // Validate balances - v1.4.9: Use toNum() for safe comparison
      const tokenBal = toNum(token.balance);
      if (amt1ForValidation > tokenBal) {
        alert(`❌ Insufficient ${token.symbol} balance!\n\nYou need ${amt1ForValidation.toFixed(4)} ${token.symbol}\nBut you only have ${tokenBal.toFixed(4)} ${token.symbol}`);
        return;
      }

      const pairBal = toNum(pairToken?.balance);
      if (pairToken && amt2ForValidation > pairBal) {
        alert(`❌ Insufficient ${pairToken.symbol} balance!\n\nYou need ${amt2ForValidation.toFixed(4)} ${pairToken.symbol}\nBut you only have ${pairBal.toFixed(4)} ${pairToken.symbol}`);
        return;
      }

      // v3.2.20-beta: Pass raw string values to preserve precision
      // The conversion to BigInt happens in DexScreen with proper decimal handling
      onAddLiquidity(token.symbol, selectedPairToken, amount1, amount2);
      onClose();
    }
  };

  // v2.4.7: Fetch user's pools when switching to remove mode
  useEffect(() => {
    if (mode === 'remove') {
      const fetchPools = async () => {
        setIsLoadingPools(true);
        try {
          const walletAddress = localStorage.getItem('walletAddress');
          if (!walletAddress) return;

          const response = await qnkAPI.getLiquidityPools();
          if (response.success && response.data) {
            // Filter pools where user has liquidity (matching the token)
            const myPools = response.data.filter((p: any) =>
              p.token0 === token.symbol || p.token1 === token.symbol
            );
            setUserPools(myPools);
            if (myPools.length > 0) {
              setSelectedPool(myPools[0]);
            }
          }
        } catch (error) {
          console.error('Failed to fetch pools:', error);
        } finally {
          setIsLoadingPools(false);
        }
      };
      fetchPools();
    }
  }, [mode, token.symbol]);

  // v2.4.7: Handle remove liquidity
  const handleRemoveLiquidity = async () => {
    if (!selectedPool) return;

    const walletAddress = localStorage.getItem('walletAddress');
    if (!walletAddress) {
      alert('❌ Please login first');
      return;
    }

    setIsRemoving(true);
    try {
      const response = await qnkAPI.removeLiquidity({
        pool_id: selectedPool.pool_id,
        percentage: removePercentage,
        provider: walletAddress,
      });

      if (response.success && response.data) {
        const amt0 = (response.data.amount0_returned / 1e24).toFixed(4);
        const amt1 = (response.data.amount1_returned / 1e24).toFixed(4);
        alert(`✅ Liquidity removed!\n\n${amt0} ${selectedPool.token0}\n${amt1} ${selectedPool.token1}\n\nTx: ${response.data.transaction_id}`);
        onClose();
        window.location.reload();
      } else {
        alert(`❌ Failed: ${response.error || 'Unknown error'}`);
      }
    } catch (error: any) {
      alert(`❌ Error: ${error.message || 'Failed to remove liquidity'}`);
    } finally {
      setIsRemoving(false);
    }
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 overflow-y-auto">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="fixed inset-0 bg-black/80 backdrop-blur-sm"
        />

        <div className="flex min-h-full items-center justify-center p-4">
        {/* Modal */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="relative w-full max-w-lg"
        >
          <div className="relative group">
            {/* Glow effect */}
            <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink rounded-2xl blur-xl opacity-50" />

            <div className="relative bg-black border border-quantum-cyan/30 rounded-2xl p-4 max-h-[90vh] overflow-y-auto">
              {/* Header - v2.4.7: Compact */}
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h2 className="text-xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent">
                    {mode === 'add' ? 'Add Liquidity' : 'Remove Liquidity'}
                  </h2>
                  <p className="text-gray-400 text-xs">Create a pair for {token.symbol}</p>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <X className="w-5 h-5 text-gray-400" />
                </button>
              </div>

              {/* Mode Toggle - v2.4.7: Compact */}
              <div className="flex gap-2 mb-3">
                <button
                  onClick={() => setMode('add')}
                  className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${
                    mode === 'add'
                      ? 'bg-gradient-to-r from-quantum-cyan to-quantum-purple text-white'
                      : 'bg-white/5 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  Add
                </button>
                <button
                  onClick={() => setMode('remove')}
                  className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${
                    mode === 'remove'
                      ? 'bg-gradient-to-r from-quantum-cyan to-quantum-purple text-white'
                      : 'bg-white/5 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  Remove
                </button>
              </div>

              {mode === 'add' ? (
                <>
                  {/* Token 1 Input - v2.4.7: Compact */}
                  <div className="space-y-1 mb-2">
                    <label className="text-sm text-gray-400">First Token</label>
                    <div className="relative group">
                      <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl blur opacity-20 group-hover:opacity-30 transition-opacity" />
                      <div className="relative bg-quantum-dark/80 border border-quantum-cyan/20 rounded-xl p-3">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <TokenIcon symbol={token.symbol} icon={token.icon} logoUrl={(token as any).logoUrl} size={40} />
                            <div>
                              <div className="font-bold text-white">{token.symbol}</div>
                              <div className="text-xs text-gray-400">{token.name}</div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-xs text-gray-400">Balance</div>
                            <div className="text-sm text-white font-medium" title={toNum(token.balance).toLocaleString()}>{formatLargeBalance(toNum(token.balance))}</div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={amount1}
                            onChange={(e) => { if (/^[0-9]*\.?[0-9]*$/.test(e.target.value) || e.target.value === '') handleAmount1Change(e.target.value); }}
                            placeholder="0.0"
                            inputMode="decimal"
                            className={`flex-1 bg-transparent text-2xl font-bold focus:outline-none ${
                              parseFloat(amount1 || '0') > toNum(token.balance) ? 'text-red-500' : 'text-white'
                            }`}
                          />
                          <button
                            onClick={() => handleAmount1Change(toNum(token.balance).toString())}
                            className="px-3 py-1 text-xs font-bold bg-quantum-purple/30 hover:bg-quantum-purple/50 text-quantum-purple rounded-lg transition-colors"
                          >
                            MAX
                          </button>
                        </div>
                        <div className="flex items-center justify-between mt-1">
                          <div className="text-xs text-gray-500">
                            ≈ ${(parseFloat(amount1 || '0') * token.price).toFixed(2)} USD
                          </div>
                          {parseFloat(amount1 || '0') > toNum(token.balance) && (
                            <div className="text-xs text-red-500 font-medium">
                              Insufficient balance
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Plus Icon - v2.4.7: Compact */}
                  <div className="flex justify-center my-2">
                    <div className="p-2 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-full">
                      <Plus className="w-4 h-4 text-white" />
                    </div>
                  </div>

                  {/* Token 2 Input - v2.4.7: Compact */}
                  <div className="space-y-1 mb-3">
                    <label className="text-sm text-gray-400">Pair With</label>
                    <div className="relative group">
                      <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-purple to-quantum-pink rounded-xl blur opacity-20 group-hover:opacity-30 transition-opacity" />
                      <div className="relative bg-quantum-dark/80 border border-quantum-purple/20 rounded-xl p-3">
                        <div className="flex items-center justify-between mb-2">
                          <select
                            value={selectedPairToken}
                            onChange={(e) => setSelectedPairToken(e.target.value)}
                            className="bg-transparent text-white font-bold text-lg focus:outline-none cursor-pointer"
                          >
                            {validPairTokens.map(t => (
                              <option key={t.id} value={t.symbol} className="bg-quantum-dark">
                                {t.icon} {t.symbol} - {t.name}
                              </option>
                            ))}
                          </select>
                          <div className="text-right">
                            <div className="text-xs text-gray-400">Balance</div>
                            <div className="text-sm text-white font-medium" title={toNum(pairToken?.balance).toLocaleString()}>{formatLargeBalance(toNum(pairToken?.balance))}</div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={amount2}
                            onChange={(e) => { if (/^[0-9]*\.?[0-9]*$/.test(e.target.value) || e.target.value === '') handleAmount2Change(e.target.value); }}
                            placeholder="0.0"
                            inputMode="decimal"
                            className={`flex-1 bg-transparent text-2xl font-bold focus:outline-none ${
                              pairToken && parseFloat(amount2 || '0') > toNum(pairToken.balance) ? 'text-red-500' : 'text-white'
                            }`}
                          />
                          <button
                            onClick={() => handleAmount2Change(toNum(pairToken?.balance).toString())}
                            className="px-3 py-1 text-xs font-bold bg-quantum-pink/30 hover:bg-quantum-pink/50 text-quantum-pink rounded-lg transition-colors"
                          >
                            MAX
                          </button>
                        </div>
                        <div className="flex items-center justify-between mt-1">
                          <div className="text-xs text-gray-500">
                            ≈ ${(parseFloat(amount2 || '0') * (pairToken?.price || 0)).toFixed(2)} USD
                          </div>
                          {pairToken && parseFloat(amount2 || '0') > toNum(pairToken.balance) && (
                            <div className="text-xs text-red-500 font-medium">
                              Insufficient balance
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* v2.4.7: Custom Price Ratio Toggle */}
                  <div className="bg-quantum-purple/10 border border-quantum-purple/30 rounded-xl p-3 mb-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-gray-300">🎯 Custom Price Ratio</span>
                        <span className="text-xs text-gray-500">(for new pools)</span>
                      </div>
                      <button
                        onClick={() => setAutoCalculate(!autoCalculate)}
                        className={`relative w-12 h-6 rounded-full transition-colors ${
                          !autoCalculate ? 'bg-quantum-purple' : 'bg-gray-600'
                        }`}
                      >
                        <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                          !autoCalculate ? 'left-7' : 'left-1'
                        }`} />
                      </button>
                    </div>
                    {!autoCalculate && (
                      <p className="text-xs text-gray-400 mt-2">
                        💡 Enter any amounts to set your price. Put more tokens with less QUG for a lower price per token.
                      </p>
                    )}
                  </div>

                  {/* Pool Share Info */}
                  <div className="bg-quantum-cyan/10 border border-quantum-cyan/30 rounded-xl p-4 mb-6">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="w-5 h-5 text-quantum-cyan mt-0.5 flex-shrink-0" />
                      <div className="flex-1 space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Pool Share</span>
                          <span className="text-white font-medium">~0.01%</span>
                        </div>
                        {/* v2.4.7: Show RESULTING price from input amounts, not market price */}
                        {/* v3.2.22-beta: Use formatSmallPrice for very small ratios (e.g., 1e-28) */}
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Your Pool Price</span>
                          <span className={`font-medium ${resultingPrice > 0 ? 'text-quantum-cyan' : 'text-white'}`}>
                            {resultingPrice > 0
                              ? `1 ${token.symbol} = ${formatSmallPrice(resultingPrice)} ${selectedPairToken}`
                              : 'Enter amounts to see price'
                            }
                          </span>
                        </div>
                        {resultingPrice > 0 && pairToken && (
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-400">Price in USD</span>
                            <span className="text-quantum-pink font-medium">
                              {formatUsdPrice(resultingPrice * pairToken.price)} per {token.symbol}
                            </span>
                          </div>
                        )}
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">LP Tokens</span>
                          <span className="text-white font-medium">
                            {amount1 && amount2 ? Math.sqrt(parseFloat(amount1) * parseFloat(amount2)).toFixed(4) : '0.0000'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Add Liquidity Button */}
                  <button
                    onClick={handleSubmit}
                    disabled={
                      !amount1 ||
                      !amount2 ||
                      parseFloat(amount1) > toNum(token.balance) ||
                      (pairToken && parseFloat(amount2) > toNum(pairToken.balance))
                    }
                    className="w-full py-4 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-cyan/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    Add Liquidity
                    <ArrowRight className="w-5 h-5" />
                  </button>
                </>
              ) : (
                /* v2.4.7: Remove Liquidity UI */
                <div className="space-y-4">
                  {isLoadingPools ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="w-6 h-6 text-quantum-cyan animate-spin" />
                      <span className="ml-2 text-gray-400">Loading pools...</span>
                    </div>
                  ) : userPools.length === 0 ? (
                    <div className="text-center py-8">
                      <p className="text-gray-400">No liquidity pools found for {token.symbol}</p>
                      <p className="text-xs text-gray-500 mt-2">Add liquidity first to see your pools here</p>
                    </div>
                  ) : (
                    <>
                      {/* Pool Selector */}
                      <div className="space-y-1">
                        <label className="text-sm text-gray-400">Select Pool</label>
                        <select
                          value={selectedPool?.pool_id || ''}
                          onChange={(e) => setSelectedPool(userPools.find(p => p.pool_id === e.target.value) || null)}
                          className="w-full bg-quantum-dark/80 border border-quantum-purple/30 rounded-xl p-3 text-white focus:outline-none focus:border-quantum-purple"
                        >
                          {userPools.map(pool => (
                            <option key={pool.pool_id} value={pool.pool_id} className="bg-quantum-dark">
                              {pool.token0} / {pool.token1}
                            </option>
                          ))}
                        </select>
                      </div>

                      {selectedPool && (
                        <>
                          {/* Pool Reserves */}
                          <div className="bg-white/5 rounded-xl p-3">
                            <div className="text-xs text-gray-400 mb-2">Your Pool Position</div>
                            <div className="flex justify-between text-sm">
                              <span className="text-white">{(toNum(selectedPool.reserve0) / 1e24).toFixed(4)} {selectedPool.token0}</span>
                              <span className="text-white">{(toNum(selectedPool.reserve1) / 1e24).toFixed(4)} {selectedPool.token1}</span>
                            </div>
                          </div>

                          {/* Percentage Slider */}
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <label className="text-sm text-gray-400">Remove Amount</label>
                              <span className="text-quantum-pink font-bold">{removePercentage}%</span>
                            </div>
                            <input
                              type="range"
                              min="1"
                              max="100"
                              value={removePercentage}
                              onChange={(e) => setRemovePercentage(parseInt(e.target.value))}
                              className="w-full accent-quantum-pink"
                            />
                            <div className="flex justify-between">
                              {[25, 50, 75, 100].map(pct => (
                                <button
                                  key={pct}
                                  onClick={() => setRemovePercentage(pct)}
                                  className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                                    removePercentage === pct
                                      ? 'bg-quantum-pink text-white'
                                      : 'bg-white/10 text-gray-400 hover:bg-white/20'
                                  }`}
                                >
                                  {pct}%
                                </button>
                              ))}
                            </div>
                          </div>

                          {/* You Will Receive */}
                          <div className="bg-quantum-pink/10 border border-quantum-pink/30 rounded-xl p-3">
                            <div className="text-xs text-gray-400 mb-2">You Will Receive</div>
                            <div className="flex justify-between text-sm">
                              <span className="text-quantum-pink font-medium">
                                {((toNum(selectedPool.reserve0) / 1e24) * removePercentage / 100).toFixed(4)} {selectedPool.token0}
                              </span>
                              <span className="text-quantum-pink font-medium">
                                {((toNum(selectedPool.reserve1) / 1e24) * removePercentage / 100).toFixed(4)} {selectedPool.token1}
                              </span>
                            </div>
                          </div>

                          {/* Remove Button */}
                          <button
                            onClick={handleRemoveLiquidity}
                            disabled={isRemoving || removePercentage === 0}
                            className="w-full py-3 bg-gradient-to-r from-quantum-pink to-red-500 rounded-xl font-bold text-white hover:shadow-lg hover:shadow-quantum-pink/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                          >
                            {isRemoving ? (
                              <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                Removing...
                              </>
                            ) : (
                              <>
                                <Minus className="w-5 h-5" />
                                Remove {removePercentage}% Liquidity
                              </>
                            )}
                          </button>
                        </>
                      )}
                    </>
                  )}
                </div>
              )}

              {/* Info Banner */}
              <div className="mt-4 p-4 bg-quantum-purple/10 border border-quantum-purple/30 rounded-xl">
                <p className="text-sm text-gray-400">
                  <strong className="text-white">Note:</strong> By adding liquidity, you'll receive LP tokens representing your share of the pool. You'll earn a portion of the 0.3% trading fees proportional to your share.
                </p>
              </div>
            </div>
          </div>
        </motion.div>
        </div>
      </div>
    </AnimatePresence>
  );
}
