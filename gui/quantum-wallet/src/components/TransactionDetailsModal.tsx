import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Copy, ExternalLink, Clock, Hash, Wallet, ArrowUpRight, ArrowDownLeft, ArrowRightLeft, Check, Coins, Code, MessageSquare } from 'lucide-react';
import { TICKER_SYMBOL } from '../constants/ticker';
import ScoreBreakdown, { type ScoreData } from './ScoreBreakdown';

interface Transaction {
  id: string;
  type: 'send' | 'receive' | 'mining' | 'contract' | 'token_transfer' | 'staking_reward' | 'reflection_reward' | 'swap';
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
  memo?: string;
  // v3.5.8-beta: Swap transaction fields
  amountOut?: string;
  tokenIn?: string;
  tokenOut?: string;
  score?: ScoreData;
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
  const isSwap = transaction.type === 'swap';
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
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 overflow-y-auto"
          onClick={onClose}
        >
          <div className="flex min-h-full items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: "spring", duration: 0.3 }}
            className="backdrop-blur-xl rounded-3xl p-8 max-w-2xl w-full border-2"
            style={{
              background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.98) 0%, rgba(50, 30, 80, 0.98) 100%)',
              borderColor: 'rgba(212, 175, 55, 0.3)',
              boxShadow: '0 0 40px rgba(212, 175, 55, 0.3), inset 0 0 20px rgba(212, 175, 55, 0.1)'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="p-3 rounded-xl"
                  style={{
                    background: isSwap
                      ? 'linear-gradient(135deg, rgba(34, 211, 238, 0.2), rgba(6, 182, 212, 0.15))'
                      : isReceive
                        ? 'linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.15))'
                        : 'linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.15))',
                    border: isSwap
                      ? '2px solid rgba(34, 211, 238, 0.3)'
                      : isReceive
                        ? '2px solid rgba(34, 197, 94, 0.3)'
                        : '2px solid rgba(239, 68, 68, 0.3)'
                  }}
                >
                  {isMining ? (
                    <Coins className="w-6 h-6 text-amber-400" />
                  ) : isContract ? (
                    <Code className="w-6 h-6 text-amber-400" />
                  ) : isToken ? (
                    <Coins className="w-6 h-6 text-amber-400" />
                  ) : isStaking ? (
                    <Coins className="w-6 h-6 text-emerald-400" />
                  ) : isReflection ? (
                    <Coins className="w-6 h-6 text-lime-400" />
                  ) : isSwap ? (
                    <ArrowRightLeft className="w-6 h-6 text-cyan-400" />
                  ) : isReceive ? (
                    <ArrowDownLeft className="w-6 h-6 text-green-400" />
                  ) : (
                    <ArrowUpRight className="w-6 h-6 text-red-400" />
                  )}
                </div>
                <div>
                  <h2 className="text-xl font-bold bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent">Transaction Details</h2>
                  <p className={`text-sm ${
                    isSwap ? 'text-cyan-400' : isReceive ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {isMining ? '⛏️ Mining Reward' :
                     isContract ? '📜 Contract Deployment' :
                     isToken ? `🪙 ${transaction.tokenSymbol || 'Token'} Transfer` :
                     isStaking ? `🎁 Staking Reward` :
                     isReflection ? `💎 Reflection Reward` :
                     isSwap ? `🔄 Swap${transaction.tokenIn && transaction.tokenOut ? ` (${transaction.tokenIn} → ${transaction.tokenOut})` : ''}` :
                     isReceive ? 'Received' : 'Sent'}
                  </p>
                </div>
              </div>
              <motion.button
                whileHover={{ scale: 1.1, rotate: 90 }}
                whileTap={{ scale: 0.9 }}
                onClick={onClose}
                className="p-2 rounded-xl hover:bg-amber-500/10 transition-colors"
              >
                <X className="w-5 h-5 text-amber-400" />
              </motion.button>
            </div>

            {/* Amount */}
            <div className="rounded-2xl p-6 mb-6"
              style={{
                background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.15), rgba(255, 215, 0, 0.1))',
                border: '2px solid rgba(212, 175, 55, 0.3)',
                boxShadow: '0 0 20px rgba(212, 175, 55, 0.2)'
              }}
            >
              <div className="text-center">
                <p className="text-sm text-amber-300/60 mb-2">Amount</p>
                {(transaction as any).isPrivate ? (
                  <div>
                    <p className="text-3xl font-bold text-amber-400">🔒 PRIVATE</p>
                    <p className="text-sm text-amber-300/60 mt-1">
                      🛡️ ZK-SNARK Protected
                    </p>
                    <p className="text-xs text-amber-300/40 mt-2">
                      Amount hidden for quantum privacy
                    </p>
                  </div>
                ) : (
                  <div>
                    <p className={`text-3xl font-bold ${isReceive ? 'text-green-400' : 'text-red-400'}`}>
                      {isReceive ? '+' : '-'}{((transaction.amount || 0) / 1e10).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 8 })} {TICKER_SYMBOL}
                    </p>
                    <p className="text-sm text-amber-300/60 mt-1">
                      ≈ ${(((transaction.amount || 0) / 1e10) * 0.01).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USD
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Transaction Details */}
            <div className="space-y-4">
              {/* Transaction Hash */}
              <div className="rounded-xl p-4"
                style={{
                  background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.8), rgba(50, 30, 80, 0.8))',
                  border: '1px solid rgba(212, 175, 55, 0.2)'
                }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Hash className="w-4 h-4 text-amber-400" />
                    <span className="text-sm text-amber-200">Transaction Hash</span>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => copyToClipboard(transaction.txHash, 'hash')}
                    className="p-1 rounded-lg hover:bg-amber-500/10 transition-colors"
                  >
                    {copiedField === 'hash' ? (
                      <Check className="w-4 h-4 text-green-400" />
                    ) : (
                      <Copy className="w-4 h-4 text-amber-400" />
                    )}
                  </motion.button>
                </div>
                <div className="mt-2">
                  <p className="font-mono text-xs text-amber-100 break-all">
                    {transaction.txHash}
                  </p>
                </div>
              </div>

              {/* From Address */}
              <div className="rounded-xl p-4"
                style={{
                  background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.8), rgba(50, 30, 80, 0.8))',
                  border: '1px solid rgba(212, 175, 55, 0.2)'
                }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Wallet className="w-4 h-4 text-amber-400" />
                    <span className="text-sm text-amber-200">From</span>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => copyToClipboard(transaction.from || '', 'from')}
                    className="p-1 rounded-lg hover:bg-amber-500/10 transition-colors"
                  >
                    {copiedField === 'from' ? (
                      <Check className="w-4 h-4 text-green-400" />
                    ) : (
                      <Copy className="w-4 h-4 text-amber-400" />
                    )}
                  </motion.button>
                </div>
                <p className="font-mono text-xs text-amber-100 mt-2 break-all">
                  {(transaction as any).isPrivate ? '🔒 Protected by ZK-SNARK' : (transaction.from || 'Unknown')}
                </p>
              </div>

              {/* To Address */}
              <div className="rounded-xl p-4"
                style={{
                  background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.8), rgba(50, 30, 80, 0.8))',
                  border: '1px solid rgba(212, 175, 55, 0.2)'
                }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Wallet className="w-4 h-4 text-amber-400" />
                    <span className="text-sm text-amber-200">To</span>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => copyToClipboard(transaction.to || '', 'to')}
                    className="p-1 rounded-lg hover:bg-amber-500/10 transition-colors"
                  >
                    {copiedField === 'to' ? (
                      <Check className="w-4 h-4 text-green-400" />
                    ) : (
                      <Copy className="w-4 h-4 text-amber-400" />
                    )}
                  </motion.button>
                </div>
                <p className="font-mono text-xs text-amber-100 mt-2 break-all">
                  {(transaction as any).isPrivate ? '🔒 Protected by ZK-SNARK' : (transaction.to || 'Unknown')}
                </p>
              </div>

              {/* Token Address for token/reward transactions */}
              {(isToken || isStaking || isReflection) && transaction.tokenAddress && (
                <div className="rounded-xl p-4"
                  style={{
                    background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.8), rgba(50, 30, 80, 0.8))',
                    border: '1px solid rgba(212, 175, 55, 0.2)'
                  }}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Code className="w-4 h-4 text-amber-400" />
                      <span className="text-sm text-amber-200">
                        {isToken ? 'Token Contract' : 'Reward Contract'}
                      </span>
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => copyToClipboard(transaction.tokenAddress || '', 'token')}
                      className="p-1 rounded-lg hover:bg-amber-500/10 transition-colors"
                    >
                      {copiedField === 'token' ? (
                        <Check className="w-4 h-4 text-green-400" />
                      ) : (
                        <Copy className="w-4 h-4 text-amber-400" />
                      )}
                    </motion.button>
                  </div>
                  <p className="font-mono text-xs text-amber-100 mt-2 break-all">
                    {transaction.tokenAddress}
                  </p>
                  {transaction.tokenName && (
                    <p className="text-xs text-amber-300/60 mt-1">
                      {transaction.tokenName} ({transaction.tokenSymbol})
                    </p>
                  )}
                </div>
              )}

              {/* Memo */}
              {(transaction as any).memo && (
                <div className="rounded-xl p-4"
                  style={{
                    background: 'linear-gradient(135deg, rgba(212,175,55,0.1), rgba(255,215,0,0.06))',
                    border: '1px solid rgba(212,175,55,0.3)'
                  }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <MessageSquare className="w-4 h-4 text-amber-400" />
                    <span className="text-sm text-amber-200">Message</span>
                  </div>
                  <p className="text-sm text-amber-100 leading-relaxed break-words">
                    {(transaction as any).memo}
                  </p>
                </div>
              )}

              {/* Timestamp */}
              <div className="rounded-xl p-4"
                style={{
                  background: 'linear-gradient(135deg, rgba(30, 20, 60, 0.8), rgba(50, 30, 80, 0.8))',
                  border: '1px solid rgba(212, 175, 55, 0.2)'
                }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-amber-400" />
                  <span className="text-sm text-amber-200">Timestamp</span>
                </div>
                <div className="text-amber-100">
                  <p className="text-sm">{dateTime.date}</p>
                  <p className="text-sm text-amber-300/60">{dateTime.time}</p>
                </div>
              </div>
            </div>

            {transaction.score && (
              <div className="mt-6">
                <ScoreBreakdown score={transaction.score} title="Transaction Score" />
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3 mt-6">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => copyToClipboard(transaction.txHash, 'hash')}
                className="flex-1 flex items-center justify-center gap-2 text-amber-100 py-3 px-4 rounded-xl transition-colors"
                style={{
                  background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.15))',
                  border: '2px solid rgba(212, 175, 55, 0.3)'
                }}
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
                className="flex-1 flex items-center justify-center gap-2 text-amber-100 py-3 px-4 rounded-xl transition-colors"
                style={{
                  background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(255, 215, 0, 0.15))',
                  border: '2px solid rgba(212, 175, 55, 0.3)'
                }}
              >
                <ExternalLink className="w-4 h-4" />
                Explorer
              </motion.button>
            </div>

            {/* Status Badge */}
            <div className="mt-4 text-center">
              <span className="inline-flex items-center gap-2 text-green-400 text-sm py-2 px-4 rounded-full"
                style={{
                  background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.1))',
                  border: '2px solid rgba(34, 197, 94, 0.3)'
                }}
              >
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                Confirmed
              </span>
            </div>
          </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
