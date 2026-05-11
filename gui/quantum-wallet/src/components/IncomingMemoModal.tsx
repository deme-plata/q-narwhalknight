import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, MessageSquare, ArrowDownLeft, Copy, Check, ExternalLink } from 'lucide-react';
import { TICKER_SYMBOL } from '../constants/ticker';

interface IncomingMemoTx {
  amount: number;
  fromAddress: string;
  memo: string;
  txHash: string;
  timestamp: number;
}

interface IncomingMemoModalProps {
  tx: IncomingMemoTx | null;
  onClose: () => void;
}

function truncateAddress(addr: string): string {
  if (!addr || addr.length < 12) return addr;
  return `${addr.slice(0, 8)}…${addr.slice(-6)}`;
}

export default function IncomingMemoModal({ tx, onClose }: IncomingMemoModalProps) {
  const [copied, setCopied] = React.useState<'addr' | 'hash' | null>(null);

  const copy = async (text: string, field: 'addr' | 'hash') => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(field);
      setTimeout(() => setCopied(null), 2000);
    } catch {}
  };

  const amountQug = tx ? (tx.amount / 1e10).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 8,
  }) : '0';

  return (
    <AnimatePresence>
      {tx && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[200] flex items-center justify-center p-4"
          style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(6px)' }}
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.88, y: 24 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.88, y: 24 }}
            transition={{ type: 'spring', duration: 0.35, bounce: 0.25 }}
            className="rounded-3xl p-7 max-w-md w-full border-2"
            style={{
              background: 'linear-gradient(135deg, rgba(20,14,50,0.99) 0%, rgba(40,22,72,0.99) 100%)',
              borderColor: 'rgba(34,197,94,0.4)',
              boxShadow: '0 0 48px rgba(34,197,94,0.25), inset 0 0 24px rgba(34,197,94,0.07)',
            }}
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl"
                  style={{
                    background: 'linear-gradient(135deg, rgba(34,197,94,0.2), rgba(22,163,74,0.12))',
                    border: '2px solid rgba(34,197,94,0.35)',
                  }}
                >
                  <ArrowDownLeft className="w-5 h-5 text-green-400" />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-green-400">Payment Received</h2>
                  <p className="text-xs text-green-400/60">with message</p>
                </div>
              </div>
              <motion.button
                whileHover={{ scale: 1.1, rotate: 90 }}
                whileTap={{ scale: 0.9 }}
                onClick={onClose}
                className="p-2 rounded-xl hover:bg-white/5 transition-colors"
              >
                <X className="w-4 h-4 text-amber-400/70" />
              </motion.button>
            </div>

            {/* Amount */}
            <div className="rounded-2xl p-5 mb-4 text-center"
              style={{
                background: 'linear-gradient(135deg, rgba(34,197,94,0.12), rgba(22,163,74,0.07))',
                border: '1.5px solid rgba(34,197,94,0.25)',
              }}
            >
              <p className="text-xs text-green-400/60 mb-1">You received</p>
              <p className="text-3xl font-bold text-green-400">
                +{amountQug} {TICKER_SYMBOL}
              </p>
            </div>

            {/* Memo */}
            <div className="rounded-2xl p-4 mb-4"
              style={{
                background: 'linear-gradient(135deg, rgba(212,175,55,0.1), rgba(255,215,0,0.06))',
                border: '1.5px solid rgba(212,175,55,0.3)',
              }}
            >
              <div className="flex items-center gap-2 mb-2">
                <MessageSquare className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-xs font-semibold text-amber-400 uppercase tracking-wide">Message</span>
              </div>
              <p className="text-sm text-amber-100 leading-relaxed break-words">{tx.memo}</p>
            </div>

            {/* From */}
            <div className="rounded-xl p-3.5 mb-5"
              style={{
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(212,175,55,0.15)',
              }}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-amber-400/50 mb-0.5">From</p>
                  <p className="font-mono text-xs text-amber-200">{truncateAddress(tx.fromAddress)}</p>
                </div>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => copy(tx.fromAddress, 'addr')}
                  className="p-1.5 rounded-lg hover:bg-amber-500/10 transition-colors"
                >
                  {copied === 'addr'
                    ? <Check className="w-3.5 h-3.5 text-green-400" />
                    : <Copy className="w-3.5 h-3.5 text-amber-400/60" />}
                </motion.button>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={onClose}
                className="flex-1 py-2.5 rounded-xl text-sm font-medium text-amber-200 transition-colors"
                style={{
                  background: 'rgba(255,255,255,0.05)',
                  border: '1.5px solid rgba(212,175,55,0.2)',
                }}
              >
                Dismiss
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => {
                  window.open(`${window.location.origin}/explorer/tx/${tx.txHash}`, '_blank');
                  onClose();
                }}
                className="flex-1 flex items-center justify-center gap-1.5 py-2.5 rounded-xl text-sm font-medium text-green-300 transition-colors"
                style={{
                  background: 'linear-gradient(135deg, rgba(34,197,94,0.15), rgba(22,163,74,0.1))',
                  border: '1.5px solid rgba(34,197,94,0.3)',
                }}
              >
                <ExternalLink className="w-3.5 h-3.5" />
                View Tx
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
