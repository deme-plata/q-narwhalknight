import { motion, AnimatePresence } from 'framer-motion';
import { X, AlertTriangle, CheckCircle, DollarSign, User, FileText, Shield, TrendingUp } from 'lucide-react';

interface TransactionIntent {
  action: string;
  recipient: string | null;
  recipient_address: string | null;
  amount: number | null;
  memo: string | null;
  priority: string;
}

interface SecurityChecks {
  recipient_verified: boolean;
  balance_sufficient: boolean;
  fraud_score: number;
  warnings: string[];
  recommendations: string[];
}

interface TransactionPreview {
  intent: TransactionIntent;
  from: string;
  to: string;
  amount: number;
  fee_estimate: number;
  total_cost: number;
  security_checks: SecurityChecks;
  requires_confirmation: boolean;
}

interface Props {
  preview: TransactionPreview | null;
  onClose: () => void;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function TransactionPreviewModal({ preview, onClose, onConfirm, onCancel }: Props) {
  if (!preview) return null;

  const { intent, from, to, amount, fee_estimate, total_cost, security_checks } = preview;

  // Determine risk level based on fraud score
  const getRiskLevel = (score: number) => {
    if (score >= 0.5) return { level: 'HIGH', color: 'red', bg: 'rgba(239, 68, 68, 0.1)' };
    if (score >= 0.2) return { level: 'MEDIUM', color: 'yellow', bg: 'rgba(234, 179, 8, 0.1)' };
    return { level: 'LOW', color: 'green', bg: 'rgba(34, 197, 94, 0.1)' };
  };

  const risk = getRiskLevel(security_checks.fraud_score);

  return (
    <AnimatePresence>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
      />

      {/* Modal */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl z-50"
      >
        <div
          className="rounded-2xl p-8 shadow-2xl"
          style={{
            background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%)',
            border: '1px solid rgba(212, 175, 55, 0.2)',
            boxShadow: '0 0 40px rgba(212, 175, 55, 0.15)'
          }}
        >
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div
                className="p-3 rounded-xl"
                style={{
                  background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #D4AF37 100%)',
                  boxShadow: '0 0 20px rgba(212, 175, 55, 0.4)'
                }}
              >
                <DollarSign className="w-6 h-6 text-slate-900" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">Transaction Preview</h2>
                <p className="text-sm text-slate-400">Review before confirming</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-slate-800/50 transition-colors"
            >
              <X className="w-6 h-6 text-slate-400" />
            </button>
          </div>

          {/* Transaction Details */}
          <div className="space-y-4 mb-6">
            {/* From/To */}
            <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-slate-400 mb-1">From</p>
                  <div className="flex items-center gap-2">
                    <User className="w-4 h-4 text-slate-400" />
                    <p className="text-sm text-white font-mono truncate">{from}</p>
                  </div>
                </div>
                <div>
                  <p className="text-sm text-slate-400 mb-1">To</p>
                  <div className="flex items-center gap-2">
                    <User className="w-4 h-4 text-slate-400" />
                    <div>
                      {intent.recipient && (
                        <p className="text-sm text-white font-semibold">{intent.recipient}</p>
                      )}
                      <p className="text-xs text-slate-400 font-mono truncate">{to}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Amount Breakdown */}
            <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-slate-400">Amount</span>
                  <span className="text-white font-semibold">{amount.toFixed(8)} QUG</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Network Fee</span>
                  <span className="text-white">{fee_estimate.toFixed(8)} QUG</span>
                </div>
                <div className="h-px bg-slate-700/50 my-2" />
                <div className="flex justify-between text-lg">
                  <span className="text-white font-semibold">Total Cost</span>
                  <span className="text-white font-bold">{total_cost.toFixed(8)} QUG</span>
                </div>
              </div>
            </div>

            {/* Memo */}
            {intent.memo && (
              <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
                <div className="flex items-start gap-2">
                  <FileText className="w-4 h-4 text-slate-400 mt-0.5" />
                  <div>
                    <p className="text-sm text-slate-400">Memo</p>
                    <p className="text-white">{intent.memo}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Security Checks */}
            <div
              className="p-4 rounded-xl border"
              style={{
                backgroundColor: risk.bg,
                borderColor: `rgba(${risk.color === 'red' ? '239, 68, 68' : risk.color === 'yellow' ? '234, 179, 8' : '34, 197, 94'}, 0.3)`
              }}
            >
              <div className="flex items-center gap-2 mb-3">
                <Shield className={`w-5 h-5 text-${risk.color}-400`} />
                <h3 className="text-white font-semibold">Security Analysis</h3>
                <span
                  className={`ml-auto px-3 py-1 rounded-full text-xs font-bold text-${risk.color}-400 bg-${risk.color}-500/20`}
                >
                  {risk.level} RISK
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-3">
                <div className="flex items-center gap-2">
                  {security_checks.recipient_verified ? (
                    <CheckCircle className="w-4 h-4 text-green-400" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  )}
                  <span className="text-sm text-slate-300">
                    {security_checks.recipient_verified ? 'Verified Recipient' : 'Unverified Recipient'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {security_checks.balance_sufficient ? (
                    <CheckCircle className="w-4 h-4 text-green-400" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 text-red-400" />
                  )}
                  <span className="text-sm text-slate-300">
                    {security_checks.balance_sufficient ? 'Sufficient Balance' : 'Insufficient Balance'}
                  </span>
                </div>
              </div>

              {/* Warnings */}
              {security_checks.warnings.length > 0 && (
                <div className="space-y-2">
                  {security_checks.warnings.map((warning, idx) => (
                    <div key={idx} className="flex items-start gap-2 text-sm text-yellow-300">
                      <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>{warning}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Recommendations */}
              {security_checks.recommendations.length > 0 && (
                <div className="mt-3 space-y-2">
                  {security_checks.recommendations.map((rec, idx) => (
                    <div key={idx} className="flex items-start gap-2 text-sm text-blue-300">
                      <TrendingUp className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>{rec}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4">
            <button
              onClick={onCancel}
              className="flex-1 px-6 py-4 rounded-xl font-semibold transition-all bg-slate-800/50 hover:bg-slate-800 text-white border border-slate-700/30"
            >
              Cancel
            </button>
            <button
              onClick={onConfirm}
              disabled={!security_checks.balance_sufficient}
              className="flex-1 px-6 py-4 rounded-xl font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              style={{
                background: security_checks.balance_sufficient
                  ? 'linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #D4AF37 100%)'
                  : 'rgba(100, 116, 139, 0.3)',
                color: security_checks.balance_sufficient ? '#0F172A' : '#94A3B8',
                boxShadow: security_checks.balance_sufficient ? '0 0 20px rgba(212, 175, 55, 0.4)' : 'none'
              }}
            >
              {security_checks.balance_sufficient ? 'Confirm Transaction' : 'Insufficient Balance'}
            </button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
