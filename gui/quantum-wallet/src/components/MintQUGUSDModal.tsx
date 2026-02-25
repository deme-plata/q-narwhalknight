import { motion, AnimatePresence } from 'framer-motion';
import { X, DollarSign, Lock, TrendingUp, AlertCircle, CheckCircle, RefreshCw } from 'lucide-react';
import { createPortal } from 'react-dom';
import { useState, useEffect, useCallback } from 'react';
import { qnkAPI } from '../services/api';

interface MintQUGUSDModalProps {
  isOpen: boolean;
  onClose: () => void;
  userQUGBalance: number;
  onSuccess?: () => void;
}

export default function MintQUGUSDModal({ isOpen, onClose, userQUGBalance, onSuccess }: MintQUGUSDModalProps) {
  const [collateralAmount, setCollateralAmount] = useState<string>('');
  const [qugusdAmount, setQugusdAmount] = useState<string>('');
  const [collateralRatio, setCollateralRatio] = useState<number>(160); // Default 160%
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [txId, setTxId] = useState<string>('');

  // v2.3.6-beta: Fetch real QUG price from AMM oracle instead of hardcoded value
  const [qugPrice, setQugPrice] = useState<number>(3000.00); // Default fallback
  const [priceLoading, setPriceLoading] = useState(false);
  const [priceSource, setPriceSource] = useState<string>('default');

  const MIN_COLLATERAL_RATIO = 150; // Minimum 150% collateralization

  // Fetch real QUG price from AMM oracle
  const fetchQugPrice = useCallback(async () => {
    setPriceLoading(true);
    try {
      const response = await fetch('/api/v1/oracle/price/QUG');
      const data = await response.json();
      if (data.success && data.data) {
        setQugPrice(data.data.price_usd);
        setPriceSource(data.data.source);
        console.log(`📊 QUG price fetched: $${data.data.price_usd} (source: ${data.data.source})`);
      }
    } catch (err) {
      console.warn('Failed to fetch QUG price, using default:', err);
    } finally {
      setPriceLoading(false);
    }
  }, []);

  // Fetch price when modal opens
  useEffect(() => {
    if (isOpen) {
      fetchQugPrice();
    }
  }, [isOpen, fetchQugPrice]);

  // Calculate QUGUSD amount when collateral changes
  useEffect(() => {
    if (collateralAmount && !isNaN(parseFloat(collateralAmount))) {
      const collateralValue = parseFloat(collateralAmount) * qugPrice;
      const maxQugusd = collateralValue / (collateralRatio / 100);
      setQugusdAmount(maxQugusd.toFixed(2));
    } else {
      setQugusdAmount('');
    }
  }, [collateralAmount, collateralRatio, qugPrice]);

  // Calculate collateral ratio when amounts change
  const actualCollateralRatio = (() => {
    if (!collateralAmount || !qugusdAmount) return 0;
    const collateralValue = parseFloat(collateralAmount) * qugPrice;
    const qugusdValue = parseFloat(qugusdAmount);
    if (qugusdValue === 0) return 0;
    return (collateralValue / qugusdValue) * 100;
  })();

  const isValidMint = () => {
    if (!collateralAmount || !qugusdAmount) return false;
    const collateral = parseFloat(collateralAmount);
    const qugusd = parseFloat(qugusdAmount);
    if (isNaN(collateral) || isNaN(qugusd)) return false;
    if (collateral <= 0 || qugusd <= 0) return false;
    if (collateral > userQUGBalance) return false;
    if (actualCollateralRatio < MIN_COLLATERAL_RATIO) return false;
    return true;
  };

  const handleMint = async () => {
    if (!isValidMint()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await qnkAPI.mintQUGUSD({
        amount: parseInt(qugusdAmount),
        collateral_type: 'QUG',
        collateral_amount: parseFloat(collateralAmount),
        reason: 'User-initiated QUGUSD minting',
      });

      if (response.success && response.data) {
        setTxId(response.data.transaction_id);
        setSuccess(true);

        // Dispatch CDP mint event for Dashboard to catch
        const currentWalletAddress = localStorage.getItem('walletAddress');
        window.dispatchEvent(new CustomEvent('cdp-mint', {
          detail: {
            transaction_id: response.data.transaction_id,
            collateral_amount: parseFloat(collateralAmount),
            minted_amount: parseFloat(qugusdAmount),
            collateral_ratio: response.data.collateral_ratio,
            wallet_address: currentWalletAddress,
            timestamp: new Date().toISOString(),
          }
        }));

        setTimeout(() => {
          onSuccess?.();
          onClose();
          // Reset state
          setCollateralAmount('');
          setQugusdAmount('');
          setSuccess(false);
          setTxId('');
        }, 3000);
      } else {
        setError(response.error || 'Failed to mint QUGUSD');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  const modalContent = (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[10000] flex items-center justify-center p-4 overflow-y-auto">
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
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: "spring", damping: 20, stiffness: 300 }}
            className="relative w-full max-w-lg my-8"
          >
            <div className="relative group">
              {/* Animated glow effect */}
              <motion.div
                className="absolute -inset-1 bg-gradient-to-r from-green-500 via-emerald-500 to-green-500 rounded-3xl blur-2xl"
                animate={{
                  opacity: [0.5, 0.8, 0.5],
                  scale: [1, 1.05, 1],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              />

              <div className="relative bg-gradient-to-br from-black via-gray-900 to-black border-2 border-green-500/50 rounded-3xl p-8 shadow-2xl">
                {/* Close button */}
                <button
                  onClick={onClose}
                  className="absolute top-4 right-4 p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
                >
                  <X className="w-5 h-5 text-white" />
                </button>

                {success ? (
                  // Success State
                  <div className="text-center">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 200 }}
                      className="flex justify-center mb-6"
                    >
                      <div className="w-20 h-20 bg-gradient-to-br from-green-500 to-emerald-500 rounded-full flex items-center justify-center">
                        <CheckCircle className="w-10 h-10 text-white" />
                      </div>
                    </motion.div>

                    <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
                      QUGUSD Minted Successfully!
                    </h2>

                    <div className="bg-green-500/20 border border-green-500/30 rounded-xl p-4 space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-300">Collateral Locked</span>
                        <span className="text-white font-bold">{collateralAmount} QUG</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-300">QUGUSD Minted</span>
                        <span className="text-green-400 font-bold">${qugusdAmount}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-300">Collateral Ratio</span>
                        <span className="text-white font-bold">{actualCollateralRatio.toFixed(1)}%</span>
                      </div>
                    </div>

                    {txId && (
                      <div className="mt-4 bg-white/5 rounded-xl p-3">
                        <div className="text-xs text-gray-400 mb-1">Transaction ID</div>
                        <div className="font-mono text-sm text-white break-all">
                          {txId.slice(0, 32)}...
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  // Mint Form
                  <>
                    {/* Header */}
                    <div className="flex items-center gap-3 mb-6">
                      <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl flex items-center justify-center">
                        <DollarSign className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <h2 className="text-2xl font-bold text-white">Mint QUGUSD</h2>
                        <p className="text-sm text-gray-400">Lock QUG as collateral to mint stablecoin</p>
                      </div>
                    </div>

                    {/* Balance Display */}
                    <div className="mb-6 p-4 bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20 rounded-xl">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-300">Available QUG</span>
                        <span className="text-xl font-bold text-white">{userQUGBalance.toFixed(2)}</span>
                      </div>
                      <div className="text-xs text-gray-400 mt-1 flex items-center gap-1">
                        ≈ ${(userQUGBalance * qugPrice).toFixed(2)} USD
                        {priceLoading && <RefreshCw className="w-3 h-3 animate-spin" />}
                        {priceSource === 'amm_oracle' && (
                          <span className="text-green-400 ml-1">(live)</span>
                        )}
                      </div>
                    </div>

                    {/* Collateral Input */}
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Collateral Amount (QUG)
                      </label>
                      <div className="relative">
                        <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                        <input
                          type="number"
                          value={collateralAmount}
                          onChange={(e) => setCollateralAmount(e.target.value)}
                          placeholder="0.00"
                          className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-green-500/50"
                          step="0.01"
                          min="0"
                          max={userQUGBalance}
                        />
                        <button
                          onClick={() => setCollateralAmount(userQUGBalance.toFixed(2))}
                          className="absolute right-2 top-1/2 transform -translate-y-1/2 px-3 py-1 bg-green-500/20 hover:bg-green-500/30 rounded-lg text-green-400 text-xs font-medium transition-colors"
                        >
                          MAX
                        </button>
                      </div>
                      {collateralAmount && parseFloat(collateralAmount) > userQUGBalance && (
                        <div className="mt-2 text-xs text-red-400 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          Insufficient QUG balance
                        </div>
                      )}
                    </div>

                    {/* Collateral Ratio Slider */}
                    <div className="mb-4">
                      <div className="flex justify-between items-center mb-2">
                        <label className="text-sm font-medium text-gray-300">
                          Collateral Ratio
                        </label>
                        <span className="text-sm font-bold text-green-400">{collateralRatio}%</span>
                      </div>
                      <input
                        type="range"
                        min={MIN_COLLATERAL_RATIO}
                        max="300"
                        step="10"
                        value={collateralRatio}
                        onChange={(e) => setCollateralRatio(parseInt(e.target.value))}
                        className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-green-500"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>Min: {MIN_COLLATERAL_RATIO}%</span>
                        <span>Safe: 200%</span>
                        <span>Max: 300%</span>
                      </div>
                    </div>

                    {/* QUGUSD Output */}
                    <div className="mb-6">
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        QUGUSD to Mint
                      </label>
                      <div className="relative">
                        <TrendingUp className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                        <input
                          type="number"
                          value={qugusdAmount}
                          onChange={(e) => setQugusdAmount(e.target.value)}
                          placeholder="0.00"
                          className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-green-500/50"
                          step="0.01"
                          min="0"
                        />
                      </div>
                      {actualCollateralRatio > 0 && actualCollateralRatio < MIN_COLLATERAL_RATIO && (
                        <div className="mt-2 text-xs text-red-400 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          Collateral ratio too low (minimum {MIN_COLLATERAL_RATIO}%)
                        </div>
                      )}
                    </div>

                    {/* Info Box */}
                    <div className="mb-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl">
                      <div className="text-xs text-blue-300 space-y-1">
                        <div className="flex justify-between items-center">
                          <span>QUG Price:</span>
                          <span className="font-medium flex items-center gap-1">
                            ${qugPrice.toFixed(2)}
                            {priceSource === 'amm_oracle' && (
                              <span className="text-green-400 text-[10px]">LIVE</span>
                            )}
                            <button
                              onClick={fetchQugPrice}
                              disabled={priceLoading}
                              className="p-0.5 hover:bg-white/10 rounded"
                              title="Refresh price"
                            >
                              <RefreshCw className={`w-3 h-3 ${priceLoading ? 'animate-spin' : ''}`} />
                            </button>
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Collateral Value:</span>
                          <span className="font-medium">
                            ${collateralAmount ? (parseFloat(collateralAmount) * qugPrice).toFixed(2) : '0.00'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Actual Ratio:</span>
                          <span className={`font-medium ${
                            actualCollateralRatio >= MIN_COLLATERAL_RATIO ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {actualCollateralRatio > 0 ? `${actualCollateralRatio.toFixed(1)}%` : '-'}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Error Message */}
                    {error && (
                      <div className="mb-4 p-3 bg-red-500/20 border border-red-500/30 rounded-xl text-red-300 text-sm flex items-center gap-2">
                        <AlertCircle className="w-4 h-4 flex-shrink-0" />
                        {error}
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex gap-3">
                      <button
                        onClick={onClose}
                        className="flex-1 py-3 bg-white/5 hover:bg-white/10 rounded-xl text-white font-medium transition-all"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleMint}
                        disabled={!isValidMint() || isLoading}
                        className="flex-1 py-3 bg-gradient-to-r from-green-500 to-emerald-500 hover:shadow-lg hover:shadow-green-500/50 rounded-xl text-white font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none"
                      >
                        {isLoading ? 'Minting...' : 'Mint QUGUSD'}
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );

  return createPortal(modalContent, document.body);
}
