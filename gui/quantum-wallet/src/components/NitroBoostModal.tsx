import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Zap, CreditCard, Coins, TrendingUp, Award, Star } from 'lucide-react';
import TokenIcon from './TokenIcon';
import { TICKER_SYMBOL } from '../constants/ticker';
import { qnkAPI } from '../services/api';

interface Token {
  id: string;
  symbol: string;
  name: string;
  icon: string;
  price: number;
}

interface NitroBoostModalProps {
  isOpen: boolean;
  onClose: () => void;
  token: Token;
  onPurchase: (points: number, paymentMethod: 'QUG' | 'QUGUSD') => void;
}

const NITRO_PACKAGES = [
  {
    id: 'starter',
    name: 'Starter Boost',
    points: 100,
    priceQUG: 5,
    priceQUGUSD: 10,
    benefits: ['2x trading speed', 'Priority transactions', '24h boost duration'],
    icon: '🚀',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    id: 'pro',
    name: 'Pro Boost',
    points: 500,
    priceQUG: 20,
    priceQUGUSD: 45,
    benefits: ['5x trading speed', 'Zero gas fees', '7 days boost duration', 'Premium support'],
    icon: '⚡',
    color: 'from-purple-500 to-pink-500',
    popular: true,
  },
  {
    id: 'ultimate',
    name: 'Ultimate Boost',
    points: 1500,
    priceQUG: 50,
    priceQUGUSD: 120,
    benefits: ['10x trading speed', 'Zero fees forever', '30 days boost duration', 'VIP support', 'Exclusive features'],
    icon: '🔥',
    color: 'from-orange-500 to-red-500',
  },
];

export default function NitroBoostModal({ isOpen, onClose, token, onPurchase }: NitroBoostModalProps) {
  const [selectedPackage, setSelectedPackage] = useState(NITRO_PACKAGES[1]);
  const [paymentMethod, setPaymentMethod] = useState<'QUG' | 'QUGUSD'>('QUGUSD');
  const [processing, setProcessing] = useState(false);

  const handlePurchase = async () => {
    setProcessing(true);

    try {
      const walletAddress = localStorage.getItem('walletAddress');
      if (!walletAddress) {
        alert('Please connect your wallet first');
        setProcessing(false);
        return;
      }

      // Calculate cost based on payment method
      const cost = paymentMethod === 'QUG' ? selectedPackage.priceQUG : selectedPackage.priceQUGUSD;

      // Send transaction to burn address to pay for NITRO points
      const burnAddress = 'qnk0000000000000000000000000000000000000000000000000000000000000000';
      const txResponse = await qnkAPI.sendTransaction(
        walletAddress,
        burnAddress,
        cost,
        `Nitro Points Purchase: ${selectedPackage.points} points (${selectedPackage.name})`,
        paymentMethod
      );

      if (!txResponse.success) {
        alert(`Payment failed: ${txResponse.error || 'Unknown error'}\n\nYour ${paymentMethod} was not deducted.`);
        setProcessing(false);
        return;
      }

      // Payment successful - grant points
      onPurchase(selectedPackage.points, paymentMethod);
    } catch (error) {
      console.error('Nitro purchase failed:', error);
      alert(`Purchase failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setProcessing(false);
      onClose();
    }
  };

  const price = paymentMethod === 'QUG' ? selectedPackage.priceQUG : selectedPackage.priceQUGUSD;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 overflow-y-auto"
          onClick={onClose}
        >
          <div className="flex min-h-full items-center justify-center p-4">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
            className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-3xl"
            style={{
              background: 'linear-gradient(135deg, rgba(20, 20, 30, 0.98), rgba(30, 30, 45, 0.98))',
              border: '2px solid rgba(255, 140, 0, 0.3)',
              boxShadow: '0 0 60px rgba(255, 140, 0, 0.3)'
            }}
          >
            {/* Animated background effects */}
            <div className="absolute inset-0 overflow-hidden rounded-3xl pointer-events-none">
              <motion.div
                className="absolute inset-0 opacity-20"
                style={{
                  background: 'radial-gradient(circle at 50% 50%, rgba(255, 140, 0, 0.3), transparent 70%)'
                }}
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.2, 0.3, 0.2],
                }}
                transition={{
                  duration: 4,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              />
            </div>

            {/* Header */}
            <div className="relative p-8 pb-6 border-b border-white/10">
              <button
                onClick={onClose}
                className="absolute top-6 right-6 p-2 rounded-lg hover:bg-white/10 transition-colors"
              >
                <X className="w-6 h-6 text-gray-400" />
              </button>

              <div className="flex items-center gap-4 mb-4">
                <motion.div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center text-3xl"
                  style={{
                    background: 'linear-gradient(135deg, #ff8c00, #ff4500)',
                    boxShadow: '0 0 30px rgba(255, 140, 0, 0.5)'
                  }}
                  animate={{
                    boxShadow: [
                      '0 0 30px rgba(255, 140, 0, 0.5)',
                      '0 0 50px rgba(255, 140, 0, 0.8)',
                      '0 0 30px rgba(255, 140, 0, 0.5)',
                    ]
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  <Zap className="w-8 h-8 text-white" />
                </motion.div>

                <div>
                  <h2 className="text-3xl font-black bg-gradient-to-r from-orange-400 via-red-500 to-orange-600 bg-clip-text text-transparent">
                    Nitro Boost
                  </h2>
                  <p className="text-gray-400 mt-1">
                    Supercharge {token.symbol} trading with premium features
                  </p>
                </div>
              </div>

              {/* Token Info */}
              <div className="flex items-center gap-3 p-4 rounded-xl bg-white/5 border border-white/10">
                <TokenIcon symbol={token.symbol} icon={token.icon} logoUrl={(token as any).logoUrl} size={40} />
                <div>
                  <div className="font-bold text-white text-lg">{token.symbol}</div>
                  <div className="text-sm text-gray-400">{token.name}</div>
                </div>
                <div className="ml-auto text-right">
                  <div className="text-sm text-gray-400">Current Price</div>
                  <div className="font-bold text-white text-lg">${token.price.toLocaleString()}</div>
                </div>
              </div>
            </div>

            {/* Packages */}
            <div className="p-8">
              <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                <Award className="w-6 h-6 text-orange-400" />
                Choose Your Boost Package
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                {NITRO_PACKAGES.map((pkg) => (
                  <motion.div
                    key={pkg.id}
                    onClick={() => setSelectedPackage(pkg)}
                    className={`relative cursor-pointer rounded-2xl p-6 transition-all ${
                      selectedPackage.id === pkg.id
                        ? 'ring-2 ring-orange-500'
                        : 'hover:bg-white/5'
                    }`}
                    style={{
                      background: selectedPackage.id === pkg.id
                        ? 'linear-gradient(135deg, rgba(255, 140, 0, 0.15), rgba(255, 69, 0, 0.1))'
                        : 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid rgba(255, 255, 255, 0.1)'
                    }}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {/* Popular Badge */}
                    {pkg.popular && (
                      <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-4 py-1 rounded-full bg-gradient-to-r from-orange-500 to-red-500 text-white text-xs font-bold flex items-center gap-1">
                        <Star className="w-3 h-3" />
                        MOST POPULAR
                      </div>
                    )}

                    {/* Icon */}
                    <div className="text-4xl mb-3 text-center">{pkg.icon}</div>

                    {/* Package Name */}
                    <h4 className="text-lg font-bold text-white text-center mb-2">
                      {pkg.name}
                    </h4>

                    {/* Points */}
                    <div className="text-center mb-4">
                      <div className="text-3xl font-black bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent">
                        {pkg.points}
                      </div>
                      <div className="text-sm text-gray-400">Nitro Points</div>
                    </div>

                    {/* Price */}
                    <div className="text-center mb-4 p-3 rounded-xl bg-black/30">
                      <div className="text-sm text-gray-400 mb-1">Price</div>
                      <div className="font-bold text-white">
                        {pkg.priceQUG} QUG
                      </div>
                      <div className="text-xs text-gray-500">
                        or ${pkg.priceQUGUSD} QUGUSD
                      </div>
                    </div>

                    {/* Benefits */}
                    <div className="space-y-2">
                      {pkg.benefits.map((benefit, idx) => (
                        <div key={idx} className="flex items-start gap-2 text-sm">
                          <Zap className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" />
                          <span className="text-gray-300">{benefit}</span>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Payment Method */}
              <div className="mb-6">
                <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                  <CreditCard className="w-5 h-5 text-orange-400" />
                  Payment Method
                </h3>

                <div className="grid grid-cols-2 gap-4">
                  <motion.button
                    onClick={() => setPaymentMethod('QUG')}
                    className={`p-4 rounded-xl transition-all ${
                      paymentMethod === 'QUG'
                        ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white'
                        : 'bg-white/5 text-gray-400 hover:bg-white/10'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-center justify-center gap-2 mb-2">
                      <Coins className="w-5 h-5" />
                      <span className="font-bold">Pay with QUG</span>
                    </div>
                    <div className="text-sm opacity-80">
                      {selectedPackage.priceQUG} {TICKER_SYMBOL}
                    </div>
                  </motion.button>

                  <motion.button
                    onClick={() => setPaymentMethod('QUGUSD')}
                    className={`p-4 rounded-xl transition-all ${
                      paymentMethod === 'QUGUSD'
                        ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white'
                        : 'bg-white/5 text-gray-400 hover:bg-white/10'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-center justify-center gap-2 mb-2">
                      <Coins className="w-5 h-5" />
                      <span className="font-bold">Pay with QUGUSD</span>
                    </div>
                    <div className="text-sm opacity-80">
                      ${selectedPackage.priceQUGUSD} QUGUSD
                    </div>
                  </motion.button>
                </div>
              </div>

              {/* Summary */}
              <div className="p-6 rounded-2xl bg-gradient-to-r from-orange-500/10 to-red-500/10 border-2 border-orange-500/30 mb-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-orange-400" />
                    <span className="font-bold text-white">Order Summary</span>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Package</span>
                    <span className="text-white font-medium">{selectedPackage.name}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Nitro Points</span>
                    <span className="text-orange-400 font-bold">{selectedPackage.points} points</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Payment Method</span>
                    <span className="text-white font-medium">{paymentMethod}</span>
                  </div>
                  <div className="border-t border-white/10 pt-3 mt-3">
                    <div className="flex justify-between">
                      <span className="text-white font-bold">Total</span>
                      <span className="text-white font-bold text-lg">
                        {paymentMethod === 'QUG' ? `${price} ${TICKER_SYMBOL}` : `$${price} QUGUSD`}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Purchase Button */}
              <motion.button
                onClick={handlePurchase}
                disabled={processing}
                className="w-full py-4 rounded-xl font-bold text-white text-lg relative overflow-hidden disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  background: 'linear-gradient(135deg, #ff8c00, #ff4500)',
                  boxShadow: '0 0 30px rgba(255, 140, 0, 0.5)'
                }}
                whileHover={{ scale: processing ? 1 : 1.02 }}
                whileTap={{ scale: processing ? 1 : 0.98 }}
              >
                {processing ? (
                  <div className="flex items-center justify-center gap-2">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <Zap className="w-5 h-5" />
                    </motion.div>
                    Processing...
                  </div>
                ) : (
                  <div className="flex items-center justify-center gap-2">
                    <Zap className="w-5 h-5" />
                    Purchase Nitro Boost
                  </div>
                )}

                {/* Animated shine effect */}
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                  initial={{ x: '-100%' }}
                  animate={{ x: '200%' }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: "linear",
                    repeatDelay: 1
                  }}
                />
              </motion.button>

              {/* Disclaimer */}
              <p className="text-xs text-gray-500 text-center mt-4">
                Nitro points are non-refundable. Boost features will be activated immediately after purchase.
              </p>
            </div>
          </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
