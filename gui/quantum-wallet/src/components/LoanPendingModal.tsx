import { motion, AnimatePresence } from 'framer-motion';
import { Clock, FileText, Info } from 'lucide-react';
import { useState, useEffect } from 'react';

interface LoanPendingModalProps {
  onClose: () => void;
  loanDetails: {
    amount: number;
    interestRate: number;
    termMonths: number;
    monthlyPayment: number;
    collateralAmount: number;
    collateralType: string;
  };
}

const LoanPendingModal: React.FC<LoanPendingModalProps> = ({ onClose, loanDetails }) => {
  const [pulseCount, setPulseCount] = useState(0);

  // Pulse animation counter
  useEffect(() => {
    const interval = setInterval(() => {
      setPulseCount(prev => prev + 1);
    }, 1500);

    return () => clearInterval(interval);
  }, []);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, y: 20 }}
          animate={{ scale: 1, y: 0 }}
          transition={{ type: 'spring', damping: 20, stiffness: 300 }}
          className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-3xl p-8"
          style={{
            background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.95))',
            border: '2px solid rgba(234, 179, 8, 0.4)',
            boxShadow: '0 0 60px rgba(234, 179, 8, 0.2)',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Pending Header */}
          <div className="flex flex-col items-center mb-8">
            <motion.div
              animate={{
                scale: [1, 1.1, 1],
                opacity: [0.7, 1, 0.7],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
              className="relative mb-4"
            >
              <div className="absolute inset-0 blur-xl opacity-60 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full" />
              <div className="relative p-6 rounded-full bg-gradient-to-br from-yellow-500 to-orange-500">
                <Clock className="w-16 h-16 text-white" />
              </div>
            </motion.div>

            <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-orange-400 mb-2">
              Application Submitted
            </h2>
            <p className="text-gray-400 text-center max-w-md">
              Your loan application is pending approval by the Quillon Bank
            </p>
          </div>

          {/* Application Status */}
          <div className="mb-6 p-6 rounded-2xl bg-gradient-to-br from-yellow-500/10 to-orange-500/10 border border-yellow-500/30">
            <div className="flex items-center gap-4 mb-4">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                className="p-3 rounded-xl bg-gradient-to-br from-yellow-500 to-orange-500"
              >
                <FileText className="w-6 h-6 text-white" />
              </motion.div>
              <div>
                <h3 className="font-semibold text-white">Awaiting Bank Approval</h3>
                <p className="text-sm text-gray-400">The founder will review your application using Quillon Bank CLI</p>
              </div>
            </div>

            <div className="pl-16 space-y-2 text-sm text-gray-400">
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
                <span>Collateral verification: Complete</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
                <span>Risk assessment: Complete</span>
              </div>
              <div className="flex items-center gap-2">
                <motion.div
                  key={pulseCount}
                  animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
                  transition={{ duration: 1.5 }}
                  className="w-1.5 h-1.5 rounded-full bg-yellow-400"
                />
                <span>Founder approval: Pending...</span>
              </div>
            </div>
          </div>

          {/* Application Summary */}
          <div className="mb-6 p-6 rounded-2xl bg-white/5 border border-white/10">
            <h3 className="text-lg font-semibold text-white mb-4">Application Summary</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-gray-400 mb-1">Requested Amount</div>
                <div className="text-xl font-bold text-white">{loanDetails.amount.toFixed(2)} QUGUSD</div>
              </div>
              <div>
                <div className="text-xs text-gray-400 mb-1">Estimated Rate</div>
                <div className="text-xl font-bold text-yellow-400">{loanDetails.interestRate.toFixed(2)}% APR</div>
              </div>
              <div>
                <div className="text-xs text-gray-400 mb-1">Collateral Locked</div>
                <div className="text-xl font-bold text-orange-400">{loanDetails.collateralAmount.toFixed(2)} {loanDetails.collateralType}</div>
              </div>
              <div>
                <div className="text-xs text-gray-400 mb-1">Loan Term</div>
                <div className="text-xl font-bold text-purple-400">{loanDetails.termMonths} months</div>
              </div>
            </div>
          </div>

          {/* What Happens Next */}
          <div className="mb-6 p-4 rounded-xl bg-blue-500/10 border border-blue-500/30">
            <div className="flex gap-3">
              <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="text-sm font-semibold text-blue-300 mb-2">What Happens Next?</h4>
                <ul className="text-xs text-gray-400 space-y-1">
                  <li>• The Quillon Bank founder will review your application</li>
                  <li>• They will use the <code className="text-cyan-400">quillon-bank</code> CLI to approve or reject</li>
                  <li>• If approved, QUGUSD will be minted to your wallet instantly</li>
                  <li>• You'll receive a notification when the decision is made</li>
                </ul>
              </div>
            </div>
          </div>

          {/* CLI Command Reference */}
          <div className="mb-6 p-4 rounded-xl bg-gray-800/50 border border-gray-700">
            <div className="text-xs text-gray-400 mb-2">Founder CLI Command:</div>
            <code className="text-xs text-cyan-400 font-mono block bg-black/30 p-2 rounded">
              ./target/release/quillon-bank list-applications
            </code>
            <code className="text-xs text-cyan-400 font-mono block bg-black/30 p-2 rounded mt-2">
              ./target/release/quillon-bank approve-loan --id &lt;loan_id&gt;
            </code>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 px-6 py-3 rounded-xl font-semibold bg-gradient-to-r from-yellow-500 to-orange-500 text-white hover:shadow-lg hover:shadow-yellow-500/50 transition-all"
            >
              Got it, Thanks!
            </button>
          </div>

          {/* Animated Background Glow */}
          <motion.div
            animate={{
              opacity: [0.2, 0.4, 0.2],
              scale: [1, 1.1, 1],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
            className="absolute -top-32 -right-32 w-64 h-64 bg-gradient-to-br from-yellow-500/20 to-orange-500/20 rounded-full blur-3xl pointer-events-none"
          />
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default LoanPendingModal;
