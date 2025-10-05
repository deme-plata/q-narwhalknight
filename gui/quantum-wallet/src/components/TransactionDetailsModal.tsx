import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Copy, ExternalLink, Clock, Hash, Wallet, ArrowUpRight, ArrowDownLeft, Check, Coins, Code } from 'lucide-react';
import { TICKER_SYMBOL } from '../constants/ticker';

interface Transaction {
  id: string;
  type: 'send' | 'receive' | 'mining' | 'contract' | 'token_transfer' | 'staking_reward' | 'reflection_reward';
  amount: number;
  fee?: number; // Transaction fee in QUG
  from?: string;
  to?: string;
  timestamp: string;
  txHash: string;
  contractName?: string;
  contractType?: string;
  tokenAddress?: string;
  tokenSymbol?: string;
  tokenName?: string;
  rewardType?: 'staking' | 'reflection' | 'dividend';
}

interface TransactionDetailsModalProps {
  transaction: Transaction | null;
  isOpen: boolean;
  onClose: () => void;
}

export default function TransactionDetailsModal({ transaction, isOpen, onClose }: TransactionDetailsModalProps) {
  const [copiedField, setCopiedField] = React.useState<string | null>(null);

  const copyToClipboard = async (text: string, field: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedField(field);
      setTimeout(() => setCopiedField(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const formatDateTime = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return {
        date: date.toLocaleDateString('en-US', {
          year: 'numeric',
          month: 'long',
          day: 'numeric'
        }),
        time: date.toLocaleTimeString('en-US', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit'
        })
      };
    } catch (err) {
      return { date: 'Unknown', time: 'Unknown' };
    }
  };


  if (!transaction) return null;

  const dateTime = formatDateTime(transaction.timestamp);
  const isReceive = transaction.type === 'receive' || transaction.type === 'mining' || transaction.type === 'staking_reward' || transaction.type === 'reflection_reward';
  const isMining = transaction.type === 'mining';
  const isContract = transaction.type === 'contract';
  const isToken = transaction.type === 'token_transfer';
  const isStaking = transaction.type === 'staking_reward';
  const isReflection = transaction.type === 'reflection_reward';

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: "spring", duration: 0.3 }}
            className="bg-gradient-to-br from-quantum-indigo/90 to-quantum-purple/80 backdrop-blur-xl rounded-3xl p-8 max-w-lg w-full quantum-glow border border-quantum-cyan/30"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className={`p-3 rounded-xl ${
                  isMining ? 'bg-quantum-yellow/20' :
                  isContract ? 'bg-quantum-purple/20' :
                  isToken ? 'bg-quantum-cyan/20' :
                  isStaking ? 'bg-emerald-500/20' :
                  isReflection ? 'bg-lime-500/20' :
                  isReceive ? 'bg-quantum-green/20' : 'bg-quantum-pink/20'
                }`}>
                  {isMining ? (
                    <Coins className="w-6 h-6 text-quantum-yellow" />
                  ) : isContract ? (
                    <Code className="w-6 h-6 text-quantum-purple" />
                  ) : isToken ? (
                    <Coins className="w-6 h-6 text-quantum-cyan" />
                  ) : isStaking ? (
                    <Coins className="w-6 h-6 text-emerald-400" />
                  ) : isReflection ? (
                    <Coins className="w-6 h-6 text-lime-400" />
                  ) : isReceive ? (
                    <ArrowDownLeft className="w-6 h-6 text-quantum-green" />
                  ) : (
                    <ArrowUpRight className="w-6 h-6 text-quantum-pink" />
                  )}
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">Transaction Details</h2>
                  <p className={`text-sm ${
                    isMining ? 'text-quantum-yellow' :
                    isContract ? 'text-quantum-purple' :
                    isToken ? 'text-quantum-cyan' :
                    isStaking ? 'text-emerald-400' :
                    isReflection ? 'text-lime-400' :
                    isReceive ? 'text-quantum-green' : 'text-quantum-pink'
                  }`}>
                    {isMining ? '⛏️ Mining Reward' :
                     isContract ? '📜 Contract Deployment' :
                     isToken ? `🪙 ${transaction.tokenSymbol || 'Token'} Transfer` :
                     isStaking ? `🎁 Staking Reward` :
                     isReflection ? `💎 Reflection Reward` :
                     isReceive ? 'Received' : 'Sent'}
                  </p>
                </div>
              </div>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={onClose}
                className="p-2 rounded-xl bg-white/10 hover:bg-white/20 transition-colors"
              >
                <X className="w-5 h-5 text-white" />
              </motion.button>
            </div>

            {/* Amount */}
            <div className="bg-white/5 rounded-2xl p-6 mb-6">
              <div className="text-center">
                <p className="text-sm text-gray-300 mb-2">Amount</p>
                {(transaction as any).isPrivate ? (
                  <div>
                    <p className="text-3xl font-bold text-quantum-purple">🔒 PRIVATE</p>
                    <p className="text-sm text-gray-400 mt-1">
                      🛡️ ZK-SNARK Protected
                    </p>
                    <p className="text-xs text-gray-500 mt-2">
                      Amount hidden for quantum privacy
                    </p>
                  </div>
                ) : (
                  <div>
                    <p className={`text-3xl font-bold ${isReceive ? 'text-quantum-green' : 'text-quantum-pink'}`}>
                      {isReceive ? '+' : '-'}{transaction.amount?.toFixed(8) || '0'} {TICKER_SYMBOL}
                    </p>
                    <p className="text-sm text-gray-400 mt-1">
                      ≈ ${((transaction.amount || 0) * 0.01).toFixed(2)} USD
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Transaction Details */}
            <div className="space-y-4">
              {/* Transaction Hash */}
              <div className="bg-white/5 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Hash className="w-4 h-4 text-quantum-cyan" />
                    <span className="text-sm text-gray-300">Transaction Hash</span>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => copyToClipboard(transaction.txHash, 'hash')}
                    className="p-1 rounded-lg hover:bg-white/10 transition-colors"
                  >
                    {copiedField === 'hash' ? (
                      <Check className="w-4 h-4 text-quantum-green" />
                    ) : (
                      <Copy className="w-4 h-4 text-gray-400" />
                    )}
                  </motion.button>
                </div>
                <p className="font-mono text-sm text-white mt-2 break-all">
                  {transaction.txHash}
                </p>
              </div>

              {/* From Address */}
              <div className="bg-white/5 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Wallet className="w-4 h-4 text-quantum-purple" />
                    <span className="text-sm text-gray-300">From</span>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => copyToClipboard(transaction.from || '', 'from')}
                    className="p-1 rounded-lg hover:bg-white/10 transition-colors"
                  >
                    {copiedField === 'from' ? (
                      <Check className="w-4 h-4 text-quantum-green" />
                    ) : (
                      <Copy className="w-4 h-4 text-gray-400" />
                    )}
                  </motion.button>
                </div>
                <p className="font-mono text-sm text-white mt-2 break-all">
                  {(transaction as any).isPrivate ? '🔒 Protected by ZK-SNARK' : (transaction.from || 'Unknown')}
                </p>
              </div>

              {/* To Address */}
              <div className="bg-white/5 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Wallet className="w-4 h-4 text-quantum-cyan" />
                    <span className="text-sm text-gray-300">To</span>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => copyToClipboard(transaction.to || '', 'to')}
                    className="p-1 rounded-lg hover:bg-white/10 transition-colors"
                  >
                    {copiedField === 'to' ? (
                      <Check className="w-4 h-4 text-quantum-green" />
                    ) : (
                      <Copy className="w-4 h-4 text-gray-400" />
                    )}
                  </motion.button>
                </div>
                <p className="font-mono text-sm text-white mt-2 break-all">
                  {(transaction as any).isPrivate ? '🔒 Protected by ZK-SNARK' : (transaction.to || 'Unknown')}
                </p>
              </div>

              {/* Token Address for token/reward transactions */}
              {(isToken || isStaking || isReflection) && transaction.tokenAddress && (
                <div className="bg-white/5 rounded-xl p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Code className="w-4 h-4 text-quantum-cyan" />
                      <span className="text-sm text-gray-300">
                        {isToken ? 'Token Contract' : 'Reward Contract'}
                      </span>
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => copyToClipboard(transaction.tokenAddress || '', 'token')}
                      className="p-1 rounded-lg hover:bg-white/10 transition-colors"
                    >
                      {copiedField === 'token' ? (
                        <Check className="w-4 h-4 text-quantum-green" />
                      ) : (
                        <Copy className="w-4 h-4 text-gray-400" />
                      )}
                    </motion.button>
                  </div>
                  <p className="font-mono text-sm text-white mt-2 break-all">
                    {transaction.tokenAddress}
                  </p>
                  {transaction.tokenName && (
                    <p className="text-xs text-gray-400 mt-1">
                      {transaction.tokenName} ({transaction.tokenSymbol})
                    </p>
                  )}
                </div>
              )}

              {/* Timestamp */}
              <div className="bg-white/5 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-quantum-yellow" />
                  <span className="text-sm text-gray-300">Timestamp</span>
                </div>
                <div className="text-white">
                  <p className="text-sm">{dateTime.date}</p>
                  <p className="text-sm text-gray-400">{dateTime.time}</p>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3 mt-6">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => copyToClipboard(transaction.txHash, 'hash')}
                className="flex-1 flex items-center justify-center gap-2 bg-quantum-purple/30 hover:bg-quantum-purple/40 text-white py-3 px-4 rounded-xl transition-colors"
              >
                <Copy className="w-4 h-4" />
                Copy Hash
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => {
                  // Open transaction in new tab with block explorer URL
                  const explorerUrl = `${window.location.origin}/explorer/tx/${transaction.txHash}`;
                  window.open(explorerUrl, '_blank');
                }}
                className="flex-1 flex items-center justify-center gap-2 bg-quantum-cyan/30 hover:bg-quantum-cyan/40 text-white py-3 px-4 rounded-xl transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                Explorer
              </motion.button>
            </div>

            {/* Status Badge */}
            <div className="mt-4 text-center">
              <span className="inline-flex items-center gap-2 bg-quantum-green/20 text-quantum-green text-sm py-2 px-4 rounded-full border border-quantum-green/30">
                <div className="w-2 h-2 bg-quantum-green rounded-full animate-pulse"></div>
                Confirmed
              </span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}