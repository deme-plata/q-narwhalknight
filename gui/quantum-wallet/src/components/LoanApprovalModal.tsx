import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, Clock, Shield, TrendingUp, Calendar, DollarSign, ArrowRight, Sparkles, Zap } from 'lucide-react';
import { useState, useEffect } from 'react';

interface LoanApprovalModalProps {
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

const LoanApprovalModal: React.FC<LoanApprovalModalProps> = ({ onClose, loanDetails }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number }>>([]);

  // Create particle burst on mount
  useEffect(() => {
    const newParticles = Array.from({ length: 20 }, (_, i) => ({
      id: i,
      x: Math.random() * 100 - 50,
      y: Math.random() * 100 - 50,
    }));
    setParticles(newParticles);

    // Auto-advance through approval steps
    const timer1 = setTimeout(() => setCurrentStep(1), 1500);
    const timer2 = setTimeout(() => setCurrentStep(2), 3000);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, []);

  const approvalSteps = [
    {
      icon: Shield,
      title: 'Collateral Locked',
      description: `${loanDetails.collateralAmount.toFixed(2)} ${loanDetails.collateralType} secured`,
      color: 'from-purple-500 to-pink-500',
      status: 'completed',
    },
    {
      icon: Zap,
      title: 'QUGUSD Minted',
      description: `${loanDetails.amount.toFixed(2)} QUGUSD credited to your wallet`,
      color: 'from-cyan-500 to-blue-500',
      status: currentStep >= 1 ? 'completed' : 'pending',
    },
    {
      icon: CheckCircle,
      title: 'Loan Active',
      description: 'Your loan is now active and earning begins',
      color: 'from-green-500 to-emerald-500',
      status: currentStep >= 2 ? 'completed' : 'pending',
    },
  ];

  const nextSteps = [
    {
      icon: Calendar,
      title: 'Monthly Payments',
      description: `Pay ${loanDetails.monthlyPayment.toFixed(2)} QUGUSD each month for ${loanDetails.termMonths} months`,
      color: 'text-purple-400',
    },
    {
      icon: TrendingUp,
      title: 'Monitor Collateral Ratio',
      description: 'Keep your ratio above 120% to avoid liquidation',
      color: 'text-cyan-400',
    },
    {
      icon: DollarSign,
      title: 'Early Repayment Bonus',
      description: 'Pay off early to save on interest and unlock collateral',
      color: 'text-green-400',
    },
  ];

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
          initial={{ scale: 0.8, y: 50 }}
          animate={{ scale: 1, y: 0 }}
          exit={{ scale: 0.8, y: 50 }}
          transition={{ type: 'spring', damping: 20, stiffness: 300 }}
          className="relative w-full max-w-3xl max-h-[90vh] overflow-y-auto rounded-3xl p-8"
          style={{
            background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.95))',
            border: '2px solid rgba(168, 85, 247, 0.4)',
            boxShadow: '0 0 80px rgba(168, 85, 247, 0.3), inset 0 0 60px rgba(168, 85, 247, 0.05)',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Particle Effects */}
          <div className="absolute inset-0 overflow-hidden rounded-3xl pointer-events-none">
            <AnimatePresence>
              {particles.map(particle => (
                <motion.div
                  key={particle.id}
                  initial={{
                    x: '50%',
                    y: '50%',
                    scale: 0,
                    opacity: 1
                  }}
                  animate={{
                    x: `calc(50% + ${particle.x}px)`,
                    y: `calc(50% + ${particle.y}px)`,
                    scale: [0, 2, 0],
                    opacity: [1, 0.8, 0]
                  }}
                  transition={{ duration: 1.5, ease: 'easeOut' }}
                  className="absolute w-2 h-2 rounded-full"
                  style={{
                    background: 'linear-gradient(135deg, #a855f7, #ec4899)',
                    boxShadow: '0 0 10px rgba(168, 85, 247, 0.8)',
                  }}
                />
              ))}
            </AnimatePresence>
          </div>

          {/* Success Header */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring', damping: 15 }}
            className="flex flex-col items-center mb-8"
          >
            <motion.div
              animate={{
                rotate: [0, 360],
                scale: [1, 1.1, 1],
              }}
              transition={{
                rotate: { duration: 2, repeat: Infinity, ease: 'linear' },
                scale: { duration: 2, repeat: Infinity, ease: 'easeInOut' }
              }}
              className="relative mb-4"
            >
              <div className="absolute inset-0 blur-xl opacity-60 bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 rounded-full" />
              <div className="relative p-6 rounded-full bg-gradient-to-br from-purple-500 to-pink-500">
                <Sparkles className="w-16 h-16 text-white" />
              </div>
            </motion.div>

            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 mb-2"
            >
              Loan Approved!
            </motion.h2>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="text-gray-400 text-center"
            >
              Your loan has been successfully processed
            </motion.p>
          </motion.div>

          {/* Approval Process Steps */}
          <div className="mb-8 space-y-4">
            {approvalSteps.map((step, index) => {
              const Icon = step.icon;
              const isCompleted = step.status === 'completed';

              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 + index * 0.15 }}
                  className="relative"
                >
                  <div
                    className={`flex items-center gap-4 p-4 rounded-xl border-2 transition-all duration-500 ${
                      isCompleted
                        ? 'border-green-500/50 bg-green-500/10'
                        : 'border-gray-700 bg-gray-800/30'
                    }`}
                  >
                    <motion.div
                      animate={isCompleted ? {
                        scale: [1, 1.2, 1],
                        rotate: [0, 360],
                      } : {}}
                      transition={{ duration: 0.5 }}
                      className={`p-3 rounded-xl bg-gradient-to-br ${step.color}`}
                    >
                      <Icon className="w-6 h-6 text-white" />
                    </motion.div>

                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold text-white">{step.title}</h3>
                        {isCompleted && (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: 'spring', damping: 15 }}
                          >
                            <CheckCircle className="w-5 h-5 text-green-400" />
                          </motion.div>
                        )}
                      </div>
                      <p className="text-sm text-gray-400">{step.description}</p>
                    </div>

                    {!isCompleted && (
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                      >
                        <Clock className="w-5 h-5 text-gray-500" />
                      </motion.div>
                    )}
                  </div>
                </motion.div>
              );
            })}
          </div>

          {/* Loan Summary */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2 }}
            className="mb-8 p-6 rounded-2xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/30"
          >
            <h3 className="text-lg font-semibold text-purple-300 mb-4 flex items-center gap-2">
              <DollarSign className="w-5 h-5" />
              Loan Summary
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-gray-400 mb-1">Loan Amount</div>
                <div className="text-xl font-bold text-white">{loanDetails.amount.toFixed(2)} QUGUSD</div>
              </div>
              <div>
                <div className="text-xs text-gray-400 mb-1">Interest Rate</div>
                <div className="text-xl font-bold text-cyan-400">{loanDetails.interestRate.toFixed(2)}% APR</div>
              </div>
              <div>
                <div className="text-xs text-gray-400 mb-1">Monthly Payment</div>
                <div className="text-xl font-bold text-green-400">{loanDetails.monthlyPayment.toFixed(2)} QUGUSD</div>
              </div>
              <div>
                <div className="text-xs text-gray-400 mb-1">Loan Term</div>
                <div className="text-xl font-bold text-purple-400">{loanDetails.termMonths} months</div>
              </div>
            </div>
          </motion.div>

          {/* What's Next Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.4 }}
            className="mb-6"
          >
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <ArrowRight className="w-5 h-5 text-cyan-400" />
              What's Next?
            </h3>
            <div className="space-y-3">
              {nextSteps.map((step, index) => {
                const Icon = step.icon;
                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 1.5 + index * 0.1 }}
                    className="flex items-start gap-3 p-3 rounded-xl bg-white/5 hover:bg-white/10 transition-colors"
                  >
                    <Icon className={`w-5 h-5 flex-shrink-0 mt-0.5 ${step.color}`} />
                    <div>
                      <div className="font-medium text-white mb-1">{step.title}</div>
                      <div className="text-sm text-gray-400">{step.description}</div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>

          {/* Action Button */}
          <motion.button
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.8 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onClose}
            className="w-full py-4 rounded-xl font-bold text-white text-lg relative overflow-hidden group"
            style={{
              background: 'linear-gradient(135deg, #a855f7, #ec4899, #06b6d4)',
              backgroundSize: '200% 200%',
            }}
          >
            <motion.div
              animate={{
                backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: 'linear',
              }}
              className="absolute inset-0"
              style={{
                background: 'linear-gradient(135deg, #a855f7, #ec4899, #06b6d4)',
                backgroundSize: '200% 200%',
              }}
            />
            <span className="relative flex items-center justify-center gap-2">
              Got it! Let's Go
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </span>
          </motion.button>

          {/* Subtle Background Glow Animation */}
          <motion.div
            animate={{
              opacity: [0.3, 0.6, 0.3],
              scale: [1, 1.1, 1],
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
            className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full blur-3xl pointer-events-none"
          />
          <motion.div
            animate={{
              opacity: [0.3, 0.6, 0.3],
              scale: [1, 1.1, 1],
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: 'easeInOut',
              delay: 2,
            }}
            className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-full blur-3xl pointer-events-none"
          />
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default LoanApprovalModal;
