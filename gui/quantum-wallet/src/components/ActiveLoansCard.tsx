import { motion } from 'framer-motion';
import { DollarSign, AlertCircle } from 'lucide-react';
import { useState, useEffect } from 'react';

interface ActiveLoan {
  loan_id: string;
  loan_amount: number;
  collateral_amount: number;
  collateral_type: string;
  interest_rate: number;
  term_months: number;
  monthly_payment: number;
  status: string;
  created_at: number;
}

interface ActiveLoansCardProps {
  onPayback?: (loanId: string) => void;
}

const ActiveLoansCard: React.FC<ActiveLoansCardProps> = ({ onPayback }) => {
  const [activeLoans, setActiveLoans] = useState<ActiveLoan[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadActiveLoans();
    // Refresh every 30 seconds
    const interval = setInterval(loadActiveLoans, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadActiveLoans = async () => {
    try {
      const response = await fetch('/api/v1/quillon-bank/lending/applications');
      const data = await response.json();

      if (data.success && Array.isArray(data.data?.applications)) {
        // Filter for approved loans only
        const approved = data.data.applications.filter(
          (loan: ActiveLoan) => loan.status === 'approved'
        );
        setActiveLoans(approved);
      }
    } catch (error) {
      console.error('Failed to load active loans:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleDateString();
  };

  const calculateHealthRatio = (loan: ActiveLoan) => {
    const QUG_PRICE = 3000.00;
    const loanValue = loan.loan_amount / 1e24;
    const collateralValue = loan.collateral_amount * QUG_PRICE;
    return (collateralValue / loanValue) * 100;
  };

  if (loading) {
    return (
      <div className="p-6 rounded-2xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/30">
        <div className="flex items-center gap-3 mb-4">
          <DollarSign className="w-6 h-6 text-purple-400" />
          <h3 className="text-xl font-bold text-white">Active Loans</h3>
        </div>
        <p className="text-gray-400 text-center py-8">Loading...</p>
      </div>
    );
  }

  if (activeLoans.length === 0) {
    return (
      <div className="p-6 rounded-2xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/30">
        <div className="flex items-center gap-3 mb-4">
          <DollarSign className="w-6 h-6 text-purple-400" />
          <h3 className="text-xl font-bold text-white">Active Loans</h3>
        </div>
        <div className="text-center py-8">
          <p className="text-gray-400 mb-2">No active loans</p>
          <p className="text-sm text-gray-500">Apply for a loan to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 rounded-2xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/30">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <DollarSign className="w-6 h-6 text-purple-400" />
          <h3 className="text-xl font-bold text-white">Active Loans</h3>
        </div>
        <span className="px-3 py-1 rounded-full bg-purple-500/20 text-purple-300 text-sm font-medium">
          {activeLoans.length} Active
        </span>
      </div>

      <div className="space-y-3">
        {activeLoans.map((loan) => {
          const healthRatio = calculateHealthRatio(loan);
          const isHealthy = healthRatio >= 150;
          const isWarning = healthRatio < 150 && healthRatio >= 120;
          const isDanger = healthRatio < 120;

          return (
            <motion.div
              key={loan.loan_id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-purple-500/50 transition-all"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="text-sm text-gray-400 mb-1">Loan Amount</div>
                  <div className="text-2xl font-bold text-white">
                    {(loan.loan_amount / 1e24).toFixed(2)} QUGUSD
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-400 mb-1">Collateral</div>
                  <div className="text-lg font-semibold text-orange-400">
                    {loan.collateral_amount.toFixed(2)} {loan.collateral_type}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3 mb-3">
                <div>
                  <div className="text-xs text-gray-500 mb-1">APR</div>
                  <div className="text-sm font-medium text-yellow-400">
                    {loan.interest_rate.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">Term</div>
                  <div className="text-sm font-medium text-white">
                    {loan.term_months}mo
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">Monthly</div>
                  <div className="text-sm font-medium text-purple-400">
                    ${loan.monthly_payment.toFixed(2)}
                  </div>
                </div>
              </div>

              {/* Health Ratio Indicator */}
              <div className="mb-3">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-gray-400">Collateral Health</span>
                  <span className={`font-semibold ${
                    isHealthy ? 'text-green-400' :
                    isWarning ? 'text-yellow-400' :
                    'text-red-400'
                  }`}>
                    {healthRatio.toFixed(0)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(healthRatio, 200) / 2}%` }}
                    className={`h-full rounded-full ${
                      isHealthy ? 'bg-gradient-to-r from-green-500 to-emerald-500' :
                      isWarning ? 'bg-gradient-to-r from-yellow-500 to-orange-500' :
                      'bg-gradient-to-r from-red-500 to-pink-500'
                    }`}
                  />
                </div>
                {isDanger && (
                  <div className="flex items-center gap-1 mt-2 text-xs text-red-400">
                    <AlertCircle className="w-3 h-3" />
                    <span>Warning: Risk of liquidation below 120%</span>
                  </div>
                )}
              </div>

              <div className="flex items-center justify-between">
                <div className="text-xs text-gray-500">
                  Started: {formatDate(loan.created_at)}
                </div>
                {onPayback && (
                  <button
                    onClick={() => onPayback(loan.loan_id)}
                    className="px-3 py-1 rounded-lg text-xs font-medium bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:shadow-lg hover:shadow-purple-500/50 transition-all"
                  >
                    Make Payment
                  </button>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};

export default ActiveLoansCard;
