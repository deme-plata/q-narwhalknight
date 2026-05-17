import { motion, AnimatePresence } from 'framer-motion';
import { X, Zap, Check, TrendingUp } from 'lucide-react';
import { createPortal } from 'react-dom';

interface NitroSuccessModalProps {
  isOpen: boolean;
  onClose: () => void;
  type: 'purchase' | 'activation';
  data: {
    points?: number;
    qugCost?: number;
    newBalance?: number;
    txId?: string;
    tokenSymbol?: string;
    remainingPoints?: number;
    totalBoost?: number;
  };
}

export default function NitroSuccessModal({ isOpen, onClose, type, data }: NitroSuccessModalProps) {
  if (!isOpen) return null;

  const isPurchase = type === 'purchase';

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
            className="relative w-full max-w-md my-8"
          >
            <div className="relative group">
              {/* Animated glow effect */}
              <motion.div
                className="absolute -inset-1 bg-gradient-to-r from-orange-500 via-yellow-500 to-orange-500 rounded-3xl blur-2xl"
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

              <div className="relative bg-gradient-to-br from-black via-gray-900 to-black border-2 border-orange-500/50 rounded-3xl p-8 shadow-2xl">
                {/* Close button */}
                <button
                  onClick={onClose}
                  className="absolute top-4 right-4 p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
                >
                  <X className="w-5 h-5 text-white" />
                </button>

                {/* Success icon */}
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                  className="flex items-center justify-center mb-6"
                >
                  <div className="relative">
                    <motion.div
                      className="absolute inset-0 bg-gradient-to-r from-orange-500 to-yellow-500 rounded-full blur-xl"
                      animate={{
                        scale: [1, 1.2, 1],
                        opacity: [0.5, 0.8, 0.5],
                      }}
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                      }}
                    />
                    <div className="relative w-20 h-20 bg-gradient-to-br from-orange-500 to-yellow-500 rounded-full flex items-center justify-center">
                      {isPurchase ? (
                        <Check className="w-10 h-10 text-white" />
                      ) : (
                        <Zap className="w-10 h-10 text-white" />
                      )}
                    </div>
                  </div>
                </motion.div>

                {/* Title */}
                <motion.h2
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="text-3xl font-bold text-center mb-6 bg-gradient-to-r from-orange-400 via-yellow-400 to-orange-400 bg-clip-text text-transparent"
                >
                  {isPurchase ? '🎉 Purchase Successful!' : '🚀 Nitro Activated!'}
                </motion.h2>

                {/* Content */}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                  className="space-y-4"
                >
                  {isPurchase ? (
                    <>
                      {/* Purchase details */}
                      <div className="bg-gradient-to-br from-orange-500/20 to-yellow-500/20 border border-orange-500/30 rounded-xl p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-gray-300">Points Purchased</span>
                          <span className="text-xl font-bold text-yellow-400 flex items-center gap-1">
                            <Zap className="w-5 h-5" />
                            {data.points?.toLocaleString()}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-gray-300">Cost</span>
                          <span className="text-lg font-bold text-orange-400">
                            {data.qugCost?.toFixed(2)} QUG
                          </span>
                        </div>
                        <div className="h-px bg-gradient-to-r from-transparent via-orange-500/50 to-transparent" />
                        <div className="flex items-center justify-between">
                          <span className="text-gray-300">New Balance</span>
                          <span className="text-2xl font-bold bg-gradient-to-r from-orange-400 to-yellow-500 bg-clip-text text-transparent">
                            {data.newBalance?.toLocaleString()} pts
                          </span>
                        </div>
                      </div>

                      {/* Transaction ID */}
                      {data.txId && (
                        <div className="bg-white/5 rounded-xl p-3">
                          <div className="text-xs text-gray-400 mb-1">Transaction ID</div>
                          <div className="font-mono text-sm text-white break-all">
                            {data.txId.slice(0, 32)}...
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <>
                      {/* Activation details */}
                      <div className="bg-gradient-to-br from-orange-500/20 to-yellow-500/20 border border-orange-500/30 rounded-xl p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-gray-300">Token</span>
                          <span className="text-xl font-bold text-white">
                            {data.tokenSymbol}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-gray-300">Points Spent</span>
                          <span className="text-lg font-bold text-orange-400 flex items-center gap-1">
                            <Zap className="w-5 h-5" />
                            {data.points}
                          </span>
                        </div>
                        <div className="h-px bg-gradient-to-r from-transparent via-orange-500/50 to-transparent" />
                        <div className="flex items-center justify-between">
                          <span className="text-gray-300">Remaining Points</span>
                          <span className="text-xl font-bold text-yellow-400">
                            {data.remainingPoints?.toLocaleString()} pts
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-gray-300 flex items-center gap-1">
                            <TrendingUp className="w-4 h-4" />
                            Total Boost
                          </span>
                          <span className="text-2xl font-bold bg-gradient-to-r from-orange-400 to-yellow-500 bg-clip-text text-transparent">
                            {data.totalBoost?.toLocaleString()} pts
                          </span>
                        </div>
                      </div>

                      {/* Benefits reminder */}
                      <div className="bg-gradient-to-br from-cyan-500/10 to-purple-500/10 border border-cyan-500/20 rounded-xl p-3">
                        <div className="text-sm text-cyan-300 mb-1 font-medium">✨ Boost Active</div>
                        <div className="text-xs text-gray-400">
                          Your token is now promoted to the top of the DEX listing with increased visibility!
                        </div>
                      </div>
                    </>
                  )}
                </motion.div>

                {/* Action button */}
                <motion.button
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                  onClick={onClose}
                  className="w-full mt-6 py-4 bg-gradient-to-r from-orange-500 via-yellow-500 to-orange-500 rounded-xl text-white font-bold text-lg hover:shadow-lg hover:shadow-orange-500/50 transition-all"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Awesome!
                </motion.button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );

  return createPortal(modalContent, document.body);
}
