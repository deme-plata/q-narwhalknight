import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Lock, AlertCircle } from 'lucide-react';

interface SessionTimeoutModalProps {
  isOpen: boolean;
  onSubmit: (password: string) => void;
  onCancel: () => void;
  error?: string;
}

export default function SessionTimeoutModal({ isOpen, onSubmit, onCancel, error }: SessionTimeoutModalProps) {
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!password) return;

    setIsSubmitting(true);
    await onSubmit(password);
    setIsSubmitting(false);
    setPassword(''); // Clear password after submit
  };

  const handleCancel = () => {
    setPassword('');
    onCancel();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 flex items-center justify-center p-4"
          style={{ zIndex: 999999 }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {/* Backdrop */}
          <motion.div
            className="absolute inset-0 bg-black/90 backdrop-blur-md"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleCancel}
          />

          {/* Modal */}
          <motion.div
            className="relative w-full max-w-md bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 rounded-3xl p-8 shadow-2xl"
            style={{
              border: '2px solid',
              borderImage: 'linear-gradient(135deg, #D4AF37, #FFD700, #FFA500, #FFD700, #D4AF37) 1',
              boxShadow: '0 0 30px rgba(212, 175, 55, 0.3), inset 0 0 20px rgba(212, 175, 55, 0.1)'
            }}
            initial={{ scale: 0.9, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.9, y: 20 }}
          >
            {/* Icon */}
            <div className="flex justify-center mb-6">
              <div className="w-20 h-20 rounded-full bg-amber-500/20 flex items-center justify-center border-2 border-amber-500/40">
                <Lock className="w-10 h-10 text-amber-400" />
              </div>
            </div>

            {/* Title */}
            <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent">
              Session Expired
            </h2>
            <p className="text-center text-amber-200/70 mb-6">
              Enter your password to continue
            </p>

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Password Input */}
              <div>
                <label className="block text-sm font-medium text-amber-200 mb-2">
                  Wallet Password
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 bg-slate-900/70 border-2 border-amber-500/30 rounded-xl text-amber-50 placeholder-slate-400 focus:outline-none focus:border-amber-400 focus:shadow-[0_0_15px_rgba(251,191,36,0.3)] transition-all"
                  placeholder="Enter your password..."
                  autoFocus
                  disabled={isSubmitting}
                />
              </div>

              {/* Error Message */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-300 text-sm flex items-center gap-2"
                >
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span>{error}</span>
                </motion.div>
              )}

              {/* Buttons */}
              <div className="flex gap-3">
                <motion.button
                  type="button"
                  onClick={handleCancel}
                  disabled={isSubmitting}
                  className="flex-1 py-3 px-4 bg-slate-800/60 border-2 border-slate-600/40 rounded-xl text-slate-300 font-medium hover:border-slate-500/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  whileHover={{ scale: isSubmitting ? 1 : 1.02 }}
                  whileTap={{ scale: isSubmitting ? 1 : 0.98 }}
                >
                  Cancel
                </motion.button>

                <motion.button
                  type="submit"
                  disabled={!password || isSubmitting}
                  className="flex-1 py-3 px-4 bg-gradient-to-r from-amber-600 to-yellow-600 rounded-xl text-slate-900 font-bold disabled:opacity-50 disabled:cursor-not-allowed hover:from-amber-500 hover:to-yellow-500 transition-all"
                  whileHover={{ scale: (!password || isSubmitting) ? 1 : 1.02 }}
                  whileTap={{ scale: (!password || isSubmitting) ? 1 : 0.98 }}
                >
                  {isSubmitting ? (
                    <span className="flex items-center justify-center gap-2">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        className="w-4 h-4 border-2 border-slate-900 border-t-transparent rounded-full"
                      />
                      Unlocking...
                    </span>
                  ) : (
                    'Unlock'
                  )}
                </motion.button>
              </div>
            </form>

            {/* Security Note */}
            <p className="text-center text-amber-300/40 text-xs mt-4">
              🔐 Your password is never transmitted or stored
            </p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
