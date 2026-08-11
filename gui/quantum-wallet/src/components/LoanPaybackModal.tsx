import { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, DollarSign, Calendar, Info, CreditCard, CheckCircle2, Loader2, AlertCircle } from 'lucide-react';

interface LoanPaybackModalProps {
  loanId: string;
  onClose: () => void;
}

interface LoanDetail {
  loan_id: string;
  borrower_address: string;
  loan_amount: string | number;     // base units (24 decimals)
  collateral_amount: number;        // display units (QUG)
  collateral_type: string;
  interest_rate: number;            // percent, e.g. 5.07
  term_months: number;
  monthly_payment: number;          // display units (QUGUSD)
  status: string;
  created_at: number;               // unix seconds
  amount_paid?: string | number;    // base units (may be absent on older backends → treat as 0)
}

const DEC = 1e24;

// Convert a display QUGUSD amount to a 24-decimal base-unit string (BigInt-precise,
// so we don't lose precision on the ~1e30 magnitudes JS numbers can't hold exactly).
const toBaseUnits = (display: number): string => {
  if (!isFinite(display) || display <= 0) return '0';
  const [intPart, fracPart = ''] = display.toFixed(8).split('.');
  const frac24 = (fracPart + '0'.repeat(24)).slice(0, 24);
  return (BigInt(intPart) * 10n ** 24n + BigInt(frac24)).toString();
};

const fmt = (n: number, dp = 2) =>
  n.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });

const LoanPaybackModal: React.FC<LoanPaybackModalProps> = ({ loanId, onClose }) => {
  const [loan, setLoan] = useState<LoanDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [amountInput, setAmountInput] = useState<string>('');
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; msg: string } | null>(null);

  // Fetch the specific loan from the applications list.
  const loadLoan = async () => {
    try {
      const res = await fetch('/api/v1/quillon-bank/lending/applications');
      const data = await res.json();
      const apps: LoanDetail[] = data?.data?.applications || [];
      const found = apps.find((l) => l.loan_id === loanId) || null;
      if (!found) setLoadError('Loan not found — it may have been fully repaid.');
      setLoan(found);
    } catch (e: any) {
      setLoadError(e?.message || 'Failed to load loan');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadLoan();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loanId]);

  // Derived financials.
  const calc = useMemo(() => {
    if (!loan) return null;
    const principal = Number(loan.loan_amount) / DEC;
    const interest = principal * (loan.interest_rate / 100) * (loan.term_months / 12);
    const totalOwed = principal + interest;
    const amountPaid = Number(loan.amount_paid ?? 0) / DEC;
    const remaining = Math.max(0, totalOwed - amountPaid);
    const monthly = loan.monthly_payment > 0 ? loan.monthly_payment : totalOwed / Math.max(1, loan.term_months);
    const paymentsMade = monthly > 0 ? Math.round(amountPaid / monthly) : 0;
    const progressPct = totalOwed > 0 ? Math.min(100, (amountPaid / totalOwed) * 100) : 0;

    // Next payment due = start date + (paymentsMade + 1) months (only if not fully paid).
    const start = new Date(loan.created_at * 1000);
    const next = new Date(start);
    next.setMonth(next.getMonth() + paymentsMade + 1);
    const fullyPaid = remaining <= 0.000001;

    return { principal, interest, totalOwed, amountPaid, remaining, monthly, paymentsMade, progressPct, start, next, fullyPaid };
  }, [loan]);

  // Default the input to the standard monthly payment (capped at remaining) once loaded.
  useEffect(() => {
    if (calc && amountInput === '' && !calc.fullyPaid) {
      const suggested = Math.min(calc.monthly, calc.remaining);
      setAmountInput(suggested > 0 ? suggested.toFixed(2) : '');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [calc]);

  const submitPayment = async () => {
    if (!loan || !calc) return;
    const amount = parseFloat(amountInput);
    if (!isFinite(amount) || amount <= 0) {
      setResult({ ok: false, msg: 'Enter a valid payment amount.' });
      return;
    }
    if (amount > calc.remaining + 0.01) {
      setResult({ ok: false, msg: `Amount exceeds remaining balance (${fmt(calc.remaining)} QUGUSD).` });
      return;
    }
    const walletAddress = localStorage.getItem('walletAddress') || '';
    if (!walletAddress) {
      setResult({ ok: false, msg: 'No wallet loaded — please unlock your wallet.' });
      return;
    }

    setSubmitting(true);
    setResult(null);
    try {
      const res = await fetch('/api/v1/quillon-bank/lending/payback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: walletAddress,
          loan_id: loan.loan_id,
          payment_amount: toBaseUnits(amount), // base-unit string (BigInt-precise)
        }),
      });
      const text = await res.text();
      let parsed: any;
      try { parsed = JSON.parse(text); } catch { parsed = text; }
      const okBody = res.ok && !(parsed && typeof parsed === 'object' && parsed.success === false);
      if (okBody) {
        setResult({ ok: true, msg: `Payment of ${fmt(amount)} QUGUSD applied.` });
        await loadLoan(); // refresh remaining / progress
        setAmountInput('');
      } else {
        const reason = typeof parsed === 'string' ? parsed : (parsed?.message || parsed?.error || `HTTP ${res.status}`);
        setResult({ ok: false, msg: reason });
      }
    } catch (e: any) {
      setResult({ ok: false, msg: e?.message || 'Network error' });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, y: 20 }}
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
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/5 transition-colors">
              <X className="w-6 h-6 text-gray-400" />
            </button>
          </div>

          {loading && (
            <div className="py-12 text-center text-gray-400">
              <Loader2 className="w-8 h-8 mx-auto mb-3 animate-spin text-purple-400" />
              Loading loan…
            </div>
          )}

          {!loading && loadError && (
            <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 flex gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-red-200">{loadError}</p>
            </div>
          )}

          {!loading && loan && calc && (
            <>
              {/* Summary grid */}
              <div className="grid grid-cols-2 gap-3 mb-5">
                <Stat label="Remaining Balance" value={`${fmt(calc.remaining)} QUGUSD`} accent="text-emerald-300" />
                <Stat label="Total Owed" value={`${fmt(calc.totalOwed)} QUGUSD`} />
                <Stat label="Monthly Payment" value={`${fmt(calc.monthly)} QUGUSD`} />
                <Stat label="Already Paid" value={`${fmt(calc.amountPaid)} QUGUSD`} />
                <Stat label="Collateral Locked" value={`${fmt(loan.collateral_amount)} ${loan.collateral_type}`} />
                <Stat
                  label="Next Payment Due"
                  value={calc.fullyPaid ? 'Paid in full' : calc.next.toLocaleDateString()}
                  accent={calc.fullyPaid ? 'text-emerald-300' : 'text-amber-300'}
                />
              </div>

              {/* Progress */}
              <div className="mb-6">
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                  <span>Repayment progress</span>
                  <span>{calc.progressPct.toFixed(1)}% · {calc.paymentsMade}/{loan.term_months} payments</span>
                </div>
                <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-emerald-500 to-green-400" style={{ width: `${calc.progressPct}%` }} />
                </div>
              </div>

              {calc.fullyPaid ? (
                <div className="mb-6 p-5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-center">
                  <CheckCircle2 className="w-10 h-10 mx-auto mb-2 text-emerald-400" />
                  <p className="text-emerald-200 font-semibold">This loan is fully repaid.</p>
                  <p className="text-xs text-gray-400 mt-1">Your {loan.collateral_type} collateral has been returned.</p>
                </div>
              ) : (
                <>
                  {/* Payment input */}
                  <label className="block text-sm text-gray-300 mb-2">Payment amount (QUGUSD)</label>
                  <div className="flex gap-2 mb-3">
                    <input
                      type="number" min="0" step="0.01" value={amountInput}
                      onChange={(e) => setAmountInput(e.target.value)}
                      placeholder="0.00"
                      className="flex-1 px-4 py-3 rounded-xl bg-black/40 border border-white/10 text-white font-mono focus:outline-none focus:border-purple-500/60"
                    />
                    <button
                      onClick={() => setAmountInput(Math.min(calc.monthly, calc.remaining).toFixed(2))}
                      className="px-3 py-2 rounded-xl text-xs font-semibold bg-white/5 border border-white/10 text-gray-200 hover:bg-white/10"
                    >Monthly</button>
                    <button
                      onClick={() => setAmountInput(calc.remaining.toFixed(2))}
                      className="px-3 py-2 rounded-xl text-xs font-semibold bg-purple-500/20 border border-purple-500/40 text-purple-200 hover:bg-purple-500/30"
                    >Pay Off</button>
                  </div>

                  {result && (
                    <div className={`mb-4 p-3 rounded-xl border flex gap-2 items-start ${result.ok ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-red-500/10 border-red-500/30'}`}>
                      {result.ok ? <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" /> : <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />}
                      <p className={`text-sm ${result.ok ? 'text-emerald-200' : 'text-red-200'}`}>{result.msg}</p>
                    </div>
                  )}

                  <div className="flex gap-3">
                    <button onClick={onClose} className="flex-1 px-6 py-3 rounded-xl border-2 border-white/10 text-gray-300 font-semibold hover:bg-white/5 transition-all">
                      Close
                    </button>
                    <button
                      onClick={submitPayment} disabled={submitting}
                      className="flex-1 px-6 py-3 rounded-xl bg-gradient-to-r from-green-500 to-emerald-500 text-white font-semibold hover:opacity-90 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                    >
                      {submitting ? <><Loader2 className="w-4 h-4 animate-spin" /> Processing…</> : <><CreditCard className="w-4 h-4" /> Make Payment</>}
                    </button>
                  </div>
                </>
              )}

              {/* Info footer */}
              <div className="mt-5 p-3 rounded-xl bg-blue-500/10 border border-blue-500/20 flex gap-2">
                <Info className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-gray-400">
                  Payments are made in QUGUSD from your wallet. Paying off the full remaining balance returns your locked {loan.collateral_type} collateral.
                  <span className="block mt-1 text-gray-500">Loan ID: <span className="font-mono text-purple-400">{loan.loan_id}</span></span>
                </p>
              </div>
            </>
          )}

          {!loading && !loan && !loadError && (
            <div className="flex gap-3">
              <button onClick={onClose} className="flex-1 px-6 py-3 rounded-xl border-2 border-white/10 text-gray-300 font-semibold hover:bg-white/5 transition-all">
                Close
              </button>
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

const Stat: React.FC<{ label: string; value: string; accent?: string }> = ({ label, value, accent }) => (
  <div className="p-3 rounded-xl bg-black/30 border border-white/5">
    <div className="text-[10px] uppercase tracking-wide text-gray-500 mb-0.5 flex items-center gap-1">
      <Calendar className="w-3 h-3 opacity-0" />{label}
    </div>
    <div className={`text-sm font-semibold ${accent || 'text-white'}`}>{value}</div>
  </div>
);

export default LoanPaybackModal;
