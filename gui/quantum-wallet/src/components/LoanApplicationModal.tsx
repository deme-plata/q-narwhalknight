import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, DollarSign, AlertTriangle, TrendingUp, Calendar, CreditCard, Info } from 'lucide-react';
import LoanPendingModal from './LoanPendingModal';

interface LoanApplicationModalProps {
  onClose: () => void;
  walletBalances: Array<{
    symbol: string;
    name: string;
    balance: number;
    usdValue?: number;
    icon: 'qug' | 'usd' | 'btc' | 'eth' | 'sol' | 'zec' | 'iron' | 'custom';
    color: string;
    comingSoon?: boolean;
    shieldedOnly?: boolean;
  }>;
  walletAddress: string;
}

const LoanApplicationModal: React.FC<LoanApplicationModalProps> = ({
  onClose,
  walletBalances,
  walletAddress,
}) => {
  // Form state
  const [loanAmount, setLoanAmount] = useState<string>('1000');
  const [selectedTerm, setSelectedTerm] = useState<number>(12);
  const [collateralType, setCollateralType] = useState<'QUG' | 'QUGUSD'>('QUG');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showApprovalModal, setShowApprovalModal] = useState(false);

  // Calculated values
  const [requiredCollateral, setRequiredCollateral] = useState<number>(0);
  const [estimatedInterestRate, setEstimatedInterestRate] = useState<number>(0);
  const [monthlyPayment, setMonthlyPayment] = useState<number>(0);
  const [totalRepayment, setTotalRepayment] = useState<number>(0);

  // Fetch live QUG price from oracle
  const [qugPrice, setQugPrice] = useState<number>(3000.00);
  useEffect(() => {
    fetch('/api/v1/defi/oracle/price/QUG/USD')
      .then(r => r.json())
      .then(data => {
        const price = data?.data?.price || data?.price;
        if (price && price > 0) setQugPrice(price);
      })
      .catch(() => {}); // Fallback to default $3000.00
  }, []);
  const QUG_PRICE = qugPrice;

  // Get available QUG balance
  const qugWallet = walletBalances.find(w => w.symbol === 'QUG');
  const availableQUG = qugWallet?.balance || 0;

  // Calculate collateral ratio
  const MINIMUM_COLLATERAL_RATIO = 1.5; // 150%
  const LIQUIDATION_THRESHOLD = 1.2; // 120%

  // Calculate required collateral in real-time
  useEffect(() => {
    const loanAmountNum = parseFloat(loanAmount) || 0;

    if (collateralType === 'QUG') {
      // Calculate QUG collateral needed
      // Loan amount (QUGUSD) * 1.5 ratio / QUG price
      const collateralNeeded = (loanAmountNum * MINIMUM_COLLATERAL_RATIO) / QUG_PRICE;
      setRequiredCollateral(collateralNeeded);
    } else {
      // QUGUSD as collateral (for additional loans)
      const collateralNeeded = loanAmountNum * MINIMUM_COLLATERAL_RATIO;
      setRequiredCollateral(collateralNeeded);
    }

    // Calculate interest rate
    const baseRate = 0.05; // 5% APR
    const creditAdjustment = 0; // TODO: Calculate from credit score
    const collateralBonus = 0; // Additional collateral reduces rate
    const termPremium = (selectedTerm / 6) * 0.005; // +0.5% per 6 months

    const rate = baseRate + creditAdjustment + collateralBonus + termPremium;
    setEstimatedInterestRate(rate * 100); // Convert to percentage

    // Calculate monthly payment (simple interest for MVP)
    const totalInterest = loanAmountNum * rate * (selectedTerm / 12);
    const totalRepay = loanAmountNum + totalInterest;
    const monthly = totalRepay / selectedTerm;

    setMonthlyPayment(monthly);
    setTotalRepayment(totalRepay);
  }, [loanAmount, selectedTerm, collateralType]);

  // Check if user has sufficient collateral
  const hasSufficientCollateral = collateralType === 'QUG'
    ? availableQUG >= requiredCollateral
    : false; // TODO: Check QUGUSD balance

  const handleSubmit = async () => {
    if (!hasSufficientCollateral) {
      alert(`Insufficient ${collateralType} collateral. You need ${requiredCollateral.toFixed(2)} ${collateralType} but only have ${availableQUG.toFixed(2)}.`);
      return;
    }

    setIsSubmitting(true);
    try {
      console.log('🏦 Submitting loan application to backend...');

      // Convert frontend display amount to backend base units (24 decimals)
      // Must send as string to avoid JSON float precision loss (e.g., 1e27)
      const loanAmountBackend = (BigInt(Math.round(parseFloat(loanAmount))) * BigInt(10) ** BigInt(24)).toString();

      const response = await fetch('/api/v1/quillon-bank/lending/apply', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          wallet_address: walletAddress,
          loan_amount: loanAmountBackend,
          collateral_amount: requiredCollateral,
          collateral_type: collateralType,
          term_months: selectedTerm,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      console.log('✅ Loan application submitted successfully:', result);

      if (result.success) {
        // Show approval pending modal
        setShowApprovalModal(true);
      } else {
        throw new Error(result.error || 'Unknown error');
      }
    } catch (error) {
      console.error('❌ Loan application error:', error);
      alert(`Failed to submit loan application: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const termOptions = [
    { months: 3, label: '3 months' },
    { months: 6, label: '6 months' },
    { months: 12, label: '12 months' },
    { months: 24, label: '24 months' },
  ];

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
              <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500">
                <DollarSign className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">Apply for Loan</h2>
                <p className="text-sm text-gray-400">Borrow QUGUSD against your QUG collateral</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/5 transition-colors"
            >
              <X className="w-6 h-6 text-gray-400" />
            </button>
          </div>

          {/* Loan Amount Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Loan Amount (QUGUSD)
            </label>
            <div className="relative">
              <input
                type="number"
                value={loanAmount}
                onChange={(e) => setLoanAmount(e.target.value)}
                className="w-full px-4 py-3 rounded-xl bg-white/5 border-2 border-purple-500/30 text-white text-lg font-medium focus:outline-none focus:border-purple-500 transition-colors"
                placeholder="1000"
                min="100"
                step="100"
              />
              <div className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 font-medium">
                QUGUSD
              </div>
            </div>
          </div>

          {/* Collateral Type Selector */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Collateral Type
            </label>
            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => setCollateralType('QUG')}
                className={`p-4 rounded-xl border-2 transition-all ${
                  collateralType === 'QUG'
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-white/10 bg-white/5 hover:border-purple-500/50'
                }`}
              >
                <div className="text-left">
                  <div className="text-white font-semibold">QUG</div>
                  <div className="text-xs text-gray-400">Primary collateral</div>
                </div>
              </button>
              <button
                onClick={() => setCollateralType('QUGUSD')}
                className={`p-4 rounded-xl border-2 transition-all ${
                  collateralType === 'QUGUSD'
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-white/10 bg-white/5 hover:border-purple-500/50'
                }`}
                disabled
              >
                <div className="text-left">
                  <div className="text-gray-500 font-semibold">QUGUSD</div>
                  <div className="text-xs text-gray-600">Coming soon</div>
                </div>
              </button>
            </div>
          </div>

          {/* Term Length Selector */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              Loan Term
            </label>
            <div className="grid grid-cols-4 gap-2">
              {termOptions.map((option) => (
                <button
                  key={option.months}
                  onClick={() => setSelectedTerm(option.months)}
                  className={`py-3 px-2 rounded-lg border-2 transition-all text-sm font-medium ${
                    selectedTerm === option.months
                      ? 'border-purple-500 bg-purple-500/20 text-purple-300'
                      : 'border-white/10 bg-white/5 text-gray-400 hover:border-purple-500/50'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          {/* Calculated Values Display */}
          <div className="mb-6 p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/30">
            <h3 className="text-sm font-semibold text-purple-300 mb-3 flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Loan Terms
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-gray-400 mb-1">Required Collateral</div>
                <div className="text-white font-bold">
                  {requiredCollateral.toFixed(2)} {collateralType}
                </div>
                <div className="text-xs text-gray-500">
                  @ {MINIMUM_COLLATERAL_RATIO * 100}% ratio
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400 mb-1">Available Balance</div>
                <div className={`font-bold ${hasSufficientCollateral ? 'text-green-400' : 'text-red-400'}`}>
                  {availableQUG.toFixed(2)} QUG
                </div>
                <div className="text-xs text-gray-500">
                  {hasSufficientCollateral ? 'Sufficient' : 'Insufficient'}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400 mb-1">Interest Rate</div>
                <div className="text-white font-bold">
                  {estimatedInterestRate.toFixed(2)}% APR
                </div>
                <div className="text-xs text-gray-500">
                  {selectedTerm} months
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400 mb-1">Monthly Payment</div>
                <div className="text-white font-bold">
                  {monthlyPayment.toFixed(2)} QUGUSD
                </div>
                <div className="text-xs text-gray-500">
                  Total: {totalRepayment.toFixed(2)}
                </div>
              </div>
            </div>
          </div>

          {/* Risk Warning */}
          <div className="mb-6 p-4 rounded-xl bg-orange-500/10 border border-orange-500/30">
            <div className="flex gap-3">
              <AlertTriangle className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="text-sm font-semibold text-orange-300 mb-1">Liquidation Risk</h4>
                <p className="text-xs text-gray-400 mb-2">
                  Your collateral will be automatically liquidated if the collateral ratio drops below{' '}
                  <span className="text-orange-300 font-semibold">{LIQUIDATION_THRESHOLD * 100}%</span>.
                </p>
                <div className="flex items-start gap-2 text-xs text-gray-500">
                  <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
                  <span>
                    Liquidation Price: ${(parseFloat(loanAmount) * LIQUIDATION_THRESHOLD / requiredCollateral).toFixed(2)} per QUG
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 px-6 py-3 rounded-xl border-2 border-white/10 text-gray-300 font-semibold hover:bg-white/5 transition-all"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={isSubmitting || !hasSufficientCollateral || parseFloat(loanAmount) < 100}
              className={`flex-1 px-6 py-3 rounded-xl font-semibold transition-all flex items-center justify-center gap-2 ${
                isSubmitting || !hasSufficientCollateral || parseFloat(loanAmount) < 100
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:shadow-lg hover:shadow-purple-500/50'
              }`}
            >
              {isSubmitting ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <CreditCard className="w-5 h-5" />
                  Apply for Loan
                </>
              )}
            </button>
          </div>

          {/* Helper Text */}
          {!hasSufficientCollateral && (
            <p className="mt-3 text-xs text-center text-red-400">
              You need {(requiredCollateral - availableQUG).toFixed(2)} more {collateralType} to apply for this loan.
            </p>
          )}
        </motion.div>
      </motion.div>

      {/* Loan Pending Modal - Awaiting Quillon Bank CLI Approval */}
      {showApprovalModal && (
        <LoanPendingModal
          onClose={() => {
            setShowApprovalModal(false);
            onClose(); // Close both modals
          }}
          loanDetails={{
            amount: parseFloat(loanAmount),
            interestRate: estimatedInterestRate,
            termMonths: selectedTerm,
            monthlyPayment,
            collateralAmount: requiredCollateral,
            collateralType,
          }}
        />
      )}
    </AnimatePresence>
  );
};

export default LoanApplicationModal;
