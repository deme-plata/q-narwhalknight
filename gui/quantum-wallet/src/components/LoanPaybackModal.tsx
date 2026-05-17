import { motion, AnimatePresence } from 'framer-motion';
import { X, DollarSign, Calendar, Info, CreditCard } from 'lucide-react';

interface LoanPaybackModalProps {
  loanId: string;
  onClose: () => void;
}

const LoanPaybackModal: React.FC<LoanPaybackModalProps> = ({ loanId, onClose }) => {
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, y: 20 }}
          animate={{ scale: 1, y: 0 }}
          exit={{ scale: 0.9, y: 20 }}
          className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl p-6"
          style={{
            background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.95))',
            border: '2px solid rgba(168, 85, 247, 0.3)',
            boxShadow: '0 0 60px rgba(168, 85, 247, 0.2)',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-gradient-to-br from-green-500 to-emerald-500">
                <DollarSign className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">Make Loan Payment</h2>
                <p className="text-sm text-gray-400">Pay down your outstanding loan balance</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/5 transition-colors"
            >
              <X className="w-6 h-6 text-gray-400" />
            </button>
          </div>

          {/* Coming Soon Notice */}
          <div className="mb-6 p-6 rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/30 text-center">
            <div className="text-6xl mb-4">🚧</div>
            <h3 className="text-xl font-bold text-purple-300 mb-3">Payment System Coming Soon</h3>
            <p className="text-gray-400 mb-4">
              The loan payment system is currently under development. You will be able to:
            </p>
            <ul className="text-left text-sm text-gray-400 space-y-2 max-w-md mx-auto">
              <li className="flex items-start gap-2">
                <CreditCard className="w-4 h-4 mt-0.5 text-green-400 flex-shrink-0" />
                <span>Make monthly payments or pay off your loan early</span>
              </li>
              <li className="flex items-start gap-2">
                <Calendar className="w-4 h-4 mt-0.5 text-blue-400 flex-shrink-0" />
                <span>View payment history and upcoming due dates</span>
              </li>
              <li className="flex items-start gap-2">
                <DollarSign className="w-4 h-4 mt-0.5 text-yellow-400 flex-shrink-0" />
                <span>Track interest accrual and principal reduction</span>
              </li>
            </ul>
          </div>

          {/* Info Box */}
          <div className="mb-6 p-4 rounded-xl bg-blue-500/10 border border-blue-500/30">
            <div className="flex gap-3">
              <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="text-sm font-semibold text-blue-300 mb-1">For Now: Founder CLI Approval</h4>
                <p className="text-xs text-gray-400">
                  Your loan has been submitted and is awaiting approval via the Quillon Bank CLI.
                  Once approved, you'll be notified and the payment system will be available for managing your loan.
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  Loan ID: <span className="text-purple-400 font-mono">{loanId}</span>
                </p>
              </div>
            </div>
          </div>

          {/* Close Button */}
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 px-6 py-3 rounded-xl border-2 border-white/10 text-gray-300 font-semibold hover:bg-white/5 transition-all"
            >
              Close
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default LoanPaybackModal;
