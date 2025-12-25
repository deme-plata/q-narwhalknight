import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Plus, ArrowRight, AlertCircle } from 'lucide-react';

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

interface LiquidityModalProps {
  token: Token;
  availableTokens: Token[];
  onClose: () => void;
  onAddLiquidity: (tokenA: string, tokenB: string, amountA: number, amountB: number) => void;
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

  // v1.0.50-beta: Use filtered list to find pair token - prevents same-token selection
  const pairToken = validPairTokens.find(t => t.symbol === selectedPairToken);

  // Calculate equivalent amount based on price ratio
  const handleAmount1Change = (value: string) => {
    setAmount1(value);
    if (value && pairToken) {
      const ratio = token.price / pairToken.price;
      setAmount2((parseFloat(value) * ratio).toFixed(6));
    } else {
      setAmount2('');
    }
  };

  const handleAmount2Change = (value: string) => {
    setAmount2(value);
    if (value && pairToken) {
      const ratio = pairToken.price / token.price;
      setAmount1((parseFloat(value) * ratio).toFixed(6));
    } else {
      setAmount1('');
    }
  };

  const handleSubmit = () => {
    if (amount1 && amount2) {
      const amt1 = parseFloat(amount1);
      const amt2 = parseFloat(amount2);

      // Validate balances - v1.4.9: Use toNum() for safe comparison
      const tokenBal = toNum(token.balance);
      if (amt1 > tokenBal) {
        alert(`❌ Insufficient ${token.symbol} balance!\n\nYou need ${amt1.toFixed(4)} ${token.symbol}\nBut you only have ${tokenBal.toFixed(4)} ${token.symbol}`);
        return;
      }

      const pairBal = toNum(pairToken?.balance);
      if (pairToken && amt2 > pairBal) {
        alert(`❌ Insufficient ${pairToken.symbol} balance!\n\nYou need ${amt2.toFixed(4)} ${pairToken.symbol}\nBut you only have ${pairBal.toFixed(4)} ${pairToken.symbol}`);
        return;
      }

      onAddLiquidity(token.symbol, selectedPairToken, amt1, amt2);
      onClose();
    }
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="absolute inset-0 bg-black/80 backdrop-blur-sm"
        />

        {/* Modal */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="relative w-full max-w-2xl"
        >
          <div className="relative group">
            {/* Glow effect */}
            <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink rounded-2xl blur-xl opacity-50" />

            <div className="relative bg-black border border-quantum-cyan/30 rounded-2xl p-6 max-h-[90vh] overflow-y-auto">
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent">
                    {mode === 'add' ? 'Add Liquidity' : 'Remove Liquidity'}
                  </h2>
                  <p className="text-gray-400 text-sm mt-1">Create a liquidity pair for {token.symbol}</p>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <X className="w-5 h-5 text-gray-400" />
                </button>
              </div>

              {/* Mode Toggle */}
              <div className="flex gap-2 mb-6">
                <button
                  onClick={() => setMode('add')}
                  className={`flex-1 py-3 rounded-xl font-medium transition-all ${
                    mode === 'add'
                      ? 'bg-gradient-to-r from-quantum-cyan to-quantum-purple text-white'
                      : 'bg-white/5 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  Add Liquidity
                </button>
                <button
                  onClick={() => setMode('remove')}
                  className={`flex-1 py-3 rounded-xl font-medium transition-all ${
                    mode === 'remove'
                      ? 'bg-gradient-to-r from-quantum-cyan to-quantum-purple text-white'
                      : 'bg-white/5 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  Remove Liquidity
                </button>
              </div>

              {mode === 'add' ? (
                <>
                  {/* Token 1 Input */}
                  <div className="space-y-2 mb-4">
                    <label className="text-sm text-gray-400">First Token</label>
                    <div className="relative group">
                      <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-xl blur opacity-20 group-hover:opacity-30 transition-opacity" />
                      <div className="relative bg-quantum-dark/80 border border-quantum-cyan/20 rounded-xl p-4">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-gradient-to-br from-quantum-cyan to-quantum-purple rounded-full flex items-center justify-center text-xl">
                              {token.icon}
                            </div>
                            <div>
                              <div className="font-bold text-white">{token.symbol}</div>
                              <div className="text-xs text-gray-400">{token.name}</div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-xs text-gray-400">Balance</div>
                            <div className="text-sm text-white font-medium">{toNum(token.balance).toFixed(4)}</div>
                          </div>
                        </div>
                        <input
                          type="number"
                          value={amount1}
                          onChange={(e) => handleAmount1Change(e.target.value)}
                          placeholder="0.0"
                          className={`w-full bg-transparent text-2xl font-bold focus:outline-none ${
                            parseFloat(amount1 || '0') > toNum(token.balance) ? 'text-red-500' : 'text-white'
                          }`}
                        />
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

                  {/* Plus Icon */}
                  <div className="flex justify-center my-4">
                    <div className="p-3 bg-gradient-to-r from-quantum-cyan to-quantum-purple rounded-full">
                      <Plus className="w-5 h-5 text-white" />
                    </div>
                  </div>

                  {/* Token 2 Input */}
                  <div className="space-y-2 mb-6">
                    <label className="text-sm text-gray-400">Pair With</label>
                    <div className="relative group">
                      <div className="absolute -inset-0.5 bg-gradient-to-r from-quantum-purple to-quantum-pink rounded-xl blur opacity-20 group-hover:opacity-30 transition-opacity" />
                      <div className="relative bg-quantum-dark/80 border border-quantum-purple/20 rounded-xl p-4">
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
                            <div className="text-sm text-white font-medium">{toNum(pairToken?.balance).toFixed(4)}</div>
                          </div>
                        </div>
                        <input
                          type="number"
                          value={amount2}
                          onChange={(e) => handleAmount2Change(e.target.value)}
                          placeholder="0.0"
                          className={`w-full bg-transparent text-2xl font-bold focus:outline-none ${
                            pairToken && parseFloat(amount2 || '0') > toNum(pairToken.balance) ? 'text-red-500' : 'text-white'
                          }`}
                        />
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

                  {/* Pool Share Info */}
                  <div className="bg-quantum-cyan/10 border border-quantum-cyan/30 rounded-xl p-4 mb-6">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="w-5 h-5 text-quantum-cyan mt-0.5 flex-shrink-0" />
                      <div className="flex-1 space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Pool Share</span>
                          <span className="text-white font-medium">~0.01%</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-400">Exchange Rate</span>
                          <span className="text-white font-medium">
                            1 {token.symbol} ≈ {pairToken ? (token.price / pairToken.price).toFixed(4) : '0'} {selectedPairToken}
                          </span>
                        </div>
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
                <div className="text-center py-12">
                  <p className="text-gray-400">
                    Remove liquidity feature coming soon
                  </p>
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
    </AnimatePresence>
  );
}
