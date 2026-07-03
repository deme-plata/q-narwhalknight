/**
 * POSMode.tsx - Merchant Point-of-Sale Mode
 *
 * v1.0.0: Self-contained POS component for merchants to accept QR code payments.
 *
 * User flow:
 * 1. Merchant enters amount via large touch-friendly numpad
 * 2. Taps "Generate QR" to display full-screen QR code
 * 3. Customer scans QR with phone wallet
 * 4. SSE listener detects incoming payment in real time
 * 5. Green confirmation screen with amount + tx hash
 * 6. "New Payment" resets for next customer
 *
 * Standalone — no dependency on Payment Request API (#019).
 * QR data format: quillon:ADDRESS?amount=AMOUNT&memo=MEMO
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { QRCodeCanvas } from 'qrcode.react';
import {
  Check,
  Delete,
  X,
  ArrowLeft,
  Store,
  CreditCard,
  MessageSquare,
  ChevronDown,
  ChevronUp,
  Copy,
} from 'lucide-react';
import { TICKER_SYMBOL } from '../constants/ticker';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type POSStatus = 'entry' | 'waiting' | 'confirmed';

interface POSModeProps {
  walletAddress: string;
  serverUrl: string; // base URL such as '' (relative) or 'https://quillon.xyz'
}

interface ConfirmationData {
  amount: string;
  txHash: string;
  timestamp: Date;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a quillon: URI for the QR payload. */
function buildQRPayload(address: string, amount: string, memo: string): string {
  let uri = `quillon:${address}?amount=${amount}`;
  if (memo.trim()) {
    uri += `&memo=${encodeURIComponent(memo.trim())}`;
  }
  return uri;
}

/** Truncate a hex hash for display. */
function truncateHash(hash: string, head = 10, tail = 8): string {
  if (hash.length <= head + tail + 3) return hash;
  return `${hash.slice(0, head)}...${hash.slice(-tail)}`;
}

/** Format a Date to a locale-friendly string. */
function formatTimestamp(d: Date): string {
  return d.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Large touch-friendly numpad button. */
function NumpadButton({
  label,
  onClick,
  variant = 'default',
  span = 1,
}: {
  label: React.ReactNode;
  onClick: () => void;
  variant?: 'default' | 'action' | 'delete';
  span?: number;
}) {
  const bg: Record<string, string> = {
    default:
      'bg-slate-800/80 hover:bg-slate-700/90 active:bg-slate-600 border-slate-600/40',
    action:
      'bg-amber-600/90 hover:bg-amber-500 active:bg-amber-400 border-amber-500/50 text-white',
    delete:
      'bg-rose-900/60 hover:bg-rose-800/80 active:bg-rose-700 border-rose-700/40',
  };

  return (
    <motion.button
      whileTap={{ scale: 0.92 }}
      onClick={onClick}
      className={`
        flex items-center justify-center rounded-2xl border text-2xl font-semibold
        select-none transition-colors duration-100
        ${bg[variant]}
        ${span === 2 ? 'col-span-2' : ''}
      `}
      style={{ minHeight: 64, touchAction: 'manipulation' }}
    >
      {label}
    </motion.button>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function POSMode({ walletAddress, serverUrl }: POSModeProps) {
  // --- State ---------------------------------------------------------------
  const [status, setStatus] = useState<POSStatus>('entry');
  const [amount, setAmount] = useState('0');
  const [memo, setMemo] = useState('');
  const [memoExpanded, setMemoExpanded] = useState(false);
  const [confirmation, setConfirmation] = useState<ConfirmationData | null>(null);
  const [copied, setCopied] = useState(false);

  // Ref to track the balance at the moment we start waiting, so we can detect
  // an *increase* of at least the requested amount.
  const baselineBalanceRef = useRef<number | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const requestedAmountRef = useRef<number>(0);

  // --- Numpad logic --------------------------------------------------------

  const appendDigit = useCallback(
    (digit: string) => {
      setAmount((prev) => {
        // Limit to 12 digits total (incl. decimal point)
        if (prev.replace('.', '').length >= 12) return prev;
        // Leading zero rules
        if (prev === '0' && digit !== '.') return digit;
        // Only one decimal point
        if (digit === '.' && prev.includes('.')) return prev;
        // Max 8 decimal places
        const decIdx = prev.indexOf('.');
        if (decIdx !== -1 && prev.length - decIdx > 8) return prev;
        return prev + digit;
      });
    },
    [],
  );

  const deleteLastChar = useCallback(() => {
    setAmount((prev) => (prev.length <= 1 ? '0' : prev.slice(0, -1)));
  }, []);

  const clearAmount = useCallback(() => {
    setAmount('0');
  }, []);

  // --- QR generation -------------------------------------------------------

  const handleGenerateQR = useCallback(() => {
    const numericAmount = parseFloat(amount);
    if (!numericAmount || numericAmount <= 0) return;

    requestedAmountRef.current = numericAmount;
    baselineBalanceRef.current = null; // will be set by first SSE event
    setStatus('waiting');
  }, [amount]);

  // --- SSE listener --------------------------------------------------------

  useEffect(() => {
    if (status !== 'waiting') return;

    // Build the SSE URL. The project's api.ts uses /api/v1/events?wallet_address=...
    const base = serverUrl || '';
    const sseUrl = `${base}/api/v1/events?wallet_address=${encodeURIComponent(walletAddress)}`;

    console.log('[POS] Connecting SSE:', sseUrl);
    const es = new EventSource(sseUrl);
    eventSourceRef.current = es;

    es.onopen = () => {
      console.log('[POS] SSE connected');
    };

    es.onerror = (err) => {
      console.warn('[POS] SSE error, will auto-reconnect:', err);
    };

    // Listen for balance-updated events (server sends this type)
    const handleBalanceUpdate = (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        console.log('[POS] balance-updated event:', data);

        const newBalance: number =
          typeof data.new_balance === 'number'
            ? data.new_balance
            : parseFloat(data.new_balance ?? '0');

        // First event — record baseline
        if (baselineBalanceRef.current === null) {
          baselineBalanceRef.current = newBalance;
          console.log('[POS] Baseline balance set:', newBalance);
          return;
        }

        const increase = newBalance - baselineBalanceRef.current;
        console.log('[POS] Balance change:', {
          baseline: baselineBalanceRef.current,
          current: newBalance,
          increase,
          needed: requestedAmountRef.current,
        });

        if (increase >= requestedAmountRef.current * 0.999) {
          // Allow 0.1% tolerance for rounding
          setConfirmation({
            amount: requestedAmountRef.current.toString(),
            txHash: data.tx_hash || data.block_hash || '',
            timestamp: new Date(),
          });
          setStatus('confirmed');
        }
      } catch (err) {
        console.warn('[POS] Failed to parse SSE event:', err);
      }
    };

    es.addEventListener('balance-updated', handleBalanceUpdate);

    // Also listen for mining_reward in case the payment arrives as a direct
    // coinbase (unlikely in POS scenario but harmless to catch).
    es.addEventListener('mining_reward', handleBalanceUpdate);

    return () => {
      console.log('[POS] Closing SSE');
      es.removeEventListener('balance-updated', handleBalanceUpdate);
      es.removeEventListener('mining_reward', handleBalanceUpdate);
      es.close();
      eventSourceRef.current = null;
    };
  }, [status, walletAddress, serverUrl]);

  // --- Cancel / Reset ------------------------------------------------------

  const handleCancel = useCallback(() => {
    setStatus('entry');
    baselineBalanceRef.current = null;
  }, []);

  const handleNewPayment = useCallback(() => {
    setAmount('0');
    setMemo('');
    setMemoExpanded(false);
    setConfirmation(null);
    baselineBalanceRef.current = null;
    setStatus('entry');
  }, []);

  // --- Copy tx hash --------------------------------------------------------

  const handleCopyHash = useCallback(() => {
    if (!confirmation?.txHash) return;
    navigator.clipboard.writeText(confirmation.txHash).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [confirmation]);

  // --- Exit POS mode -------------------------------------------------------

  const handleExit = useCallback(() => {
    // Navigate away from /pos
    window.history.pushState(null, '', '/');
    window.location.reload();
  }, []);

  // --- Render helpers ------------------------------------------------------

  const numericAmount = parseFloat(amount) || 0;
  const qrPayload = buildQRPayload(walletAddress, amount, memo);

  // =========================================================================
  // ENTRY STATE — Amount input + numpad
  // =========================================================================

  if (status === 'entry') {
    return (
      <div className="fixed inset-0 z-[9999] flex flex-col bg-slate-950 text-white overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 pt-4 pb-2">
          <button
            onClick={handleExit}
            className="flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
            style={{ touchAction: 'manipulation' }}
          >
            <ArrowLeft className="w-4 h-4" />
            Exit POS
          </button>
          <div className="flex items-center gap-2 text-amber-400">
            <Store className="w-5 h-5" />
            <span className="font-semibold text-sm tracking-wide">MERCHANT POS</span>
          </div>
        </div>

        {/* Amount display */}
        <div className="flex-1 flex flex-col items-center justify-center px-6 pb-2 min-h-0">
          <p className="text-slate-400 text-sm mb-2 flex items-center gap-1">
            <CreditCard className="w-4 h-4" />
            Enter payment amount
          </p>
          <div className="text-center mb-4">
            <motion.p
              key={amount}
              initial={{ scale: 1.05, opacity: 0.7 }}
              animate={{ scale: 1, opacity: 1 }}
              className="font-mono font-bold tracking-tight leading-none"
              style={{
                fontSize: amount.length > 8 ? '2.5rem' : '3.5rem',
                background: 'linear-gradient(135deg, #fbbf24, #f59e0b, #d97706)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              {amount}
            </motion.p>
            <p className="text-amber-400/70 text-lg font-medium mt-1">{TICKER_SYMBOL}</p>
          </div>

          {/* Memo (collapsible) */}
          <div className="w-full max-w-xs mb-4">
            <button
              onClick={() => setMemoExpanded((v) => !v)}
              className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200 transition-colors mx-auto"
              style={{ touchAction: 'manipulation' }}
            >
              <MessageSquare className="w-3 h-3" />
              {memoExpanded ? 'Hide memo' : 'Add memo (optional)'}
              {memoExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
            <AnimatePresence>
              {memoExpanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden"
                >
                  <input
                    type="text"
                    maxLength={120}
                    placeholder="e.g. Table 5 - Coffee"
                    value={memo}
                    onChange={(e) => setMemo(e.target.value)}
                    className="mt-2 w-full rounded-xl border border-slate-600/50 bg-slate-800/60 px-4 py-3
                               text-sm text-white placeholder-slate-500 outline-none focus:border-amber-500/50
                               focus:ring-1 focus:ring-amber-500/30"
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Numpad */}
        <div className="px-3 pb-3 flex-shrink-0">
          <div className="grid grid-cols-3 gap-2 max-w-xs mx-auto mb-3">
            {['1', '2', '3', '4', '5', '6', '7', '8', '9'].map((d) => (
              <NumpadButton key={d} label={d} onClick={() => appendDigit(d)} />
            ))}
            <NumpadButton label="." onClick={() => appendDigit('.')} />
            <NumpadButton label="0" onClick={() => appendDigit('0')} />
            <NumpadButton
              label={<Delete className="w-6 h-6" />}
              onClick={deleteLastChar}
              variant="delete"
            />
          </div>

          {/* Clear + Generate QR buttons */}
          <div className="grid grid-cols-3 gap-2 max-w-xs mx-auto">
            <button
              onClick={clearAmount}
              className="rounded-2xl border border-slate-600/40 bg-slate-800/60 text-slate-300
                         text-sm font-medium py-3 active:bg-slate-700 transition-colors"
              style={{ touchAction: 'manipulation' }}
            >
              Clear
            </button>
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={handleGenerateQR}
              disabled={numericAmount <= 0}
              className={`col-span-2 rounded-2xl border text-lg font-bold py-3 transition-colors ${
                numericAmount > 0
                  ? 'bg-gradient-to-r from-amber-600 to-amber-500 border-amber-500/60 text-white shadow-lg shadow-amber-900/40'
                  : 'bg-slate-800/40 border-slate-700/30 text-slate-600 cursor-not-allowed'
              }`}
              style={{ touchAction: 'manipulation' }}
            >
              Generate QR
            </motion.button>
          </div>
        </div>
      </div>
    );
  }

  // =========================================================================
  // WAITING STATE — QR code displayed, listening for payment
  // =========================================================================

  if (status === 'waiting') {
    return (
      <div className="fixed inset-0 z-[9999] flex flex-col items-center justify-center bg-white text-slate-900 overflow-hidden">
        {/* Cancel button */}
        <button
          onClick={handleCancel}
          className="absolute top-4 right-4 p-2 rounded-full bg-slate-200 hover:bg-slate-300 active:bg-slate-400 transition-colors"
          style={{ touchAction: 'manipulation' }}
        >
          <X className="w-5 h-5 text-slate-600" />
        </button>

        {/* Amount */}
        <p className="text-slate-500 text-sm mb-1 font-medium">Payment requested</p>
        <p className="text-3xl font-bold text-slate-900 mb-4">
          {parseFloat(amount).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 8 })}{' '}
          <span className="text-amber-600">{TICKER_SYMBOL}</span>
        </p>

        {/* QR Code */}
        <motion.div
          initial={{ scale: 0.85, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', damping: 20, stiffness: 200 }}
          className="relative mb-4"
        >
          {/* Pulsing ring to indicate "waiting" */}
          <motion.div
            className="absolute -inset-4 rounded-3xl border-2 border-amber-400/50"
            animate={{ scale: [1, 1.04, 1], opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          />
          <div className="bg-white p-3 rounded-2xl shadow-xl border border-slate-200">
            <QRCodeCanvas
              id="pos-qr-canvas"
              value={qrPayload}
              size={Math.min(280, window.innerWidth - 80)}
              level="H"
              includeMargin
              bgColor="#ffffff"
              fgColor="#1e293b"
            />
          </div>
        </motion.div>

        {/* Memo */}
        {memo.trim() && (
          <p className="text-sm text-slate-500 mb-3 px-8 text-center break-words max-w-xs">
            {memo}
          </p>
        )}

        {/* Waiting indicator */}
        <motion.div
          className="flex items-center gap-2 text-amber-600 mt-2"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
        >
          <div className="w-2 h-2 rounded-full bg-amber-500" />
          <span className="text-sm font-medium">Waiting for payment...</span>
        </motion.div>

        {/* Scan instructions */}
        <p className="text-xs text-slate-400 mt-6 px-8 text-center">
          Customer: open your Quillon Graph wallet and scan this QR code
        </p>
      </div>
    );
  }

  // =========================================================================
  // CONFIRMED STATE — Payment received
  // =========================================================================

  if (status === 'confirmed' && confirmation) {
    return (
      <div className="fixed inset-0 z-[9999] flex flex-col items-center justify-center overflow-hidden"
        style={{ background: 'linear-gradient(135deg, #065f46 0%, #047857 30%, #059669 100%)' }}
      >
        {/* Checkmark */}
        <motion.div
          initial={{ scale: 0, rotate: -90 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ type: 'spring', damping: 12, stiffness: 200 }}
          className="w-24 h-24 rounded-full bg-white/20 backdrop-blur flex items-center justify-center mb-6
                     border-2 border-white/30 shadow-lg shadow-emerald-900/50"
        >
          <Check className="w-14 h-14 text-white" strokeWidth={3} />
        </motion.div>

        {/* Title */}
        <motion.p
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-2xl font-extrabold text-white tracking-wide mb-1"
        >
          PAYMENT RECEIVED
        </motion.p>

        {/* Amount */}
        <motion.p
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="text-4xl font-bold text-white mt-2 mb-1"
        >
          {parseFloat(confirmation.amount).toLocaleString(undefined, {
            minimumFractionDigits: 0,
            maximumFractionDigits: 8,
          })}{' '}
          <span className="text-emerald-200">{TICKER_SYMBOL}</span>
        </motion.p>

        {/* Tx hash */}
        {confirmation.txHash && (
          <motion.button
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
            onClick={handleCopyHash}
            className="flex items-center gap-2 mt-4 px-4 py-2 rounded-xl bg-white/10 border border-white/20
                       hover:bg-white/20 active:bg-white/30 transition-colors"
            style={{ touchAction: 'manipulation' }}
          >
            <span className="font-mono text-xs text-emerald-100">
              {truncateHash(confirmation.txHash)}
            </span>
            {copied ? (
              <Check className="w-4 h-4 text-emerald-200" />
            ) : (
              <Copy className="w-4 h-4 text-emerald-200/60" />
            )}
          </motion.button>
        )}

        {/* Timestamp */}
        <motion.p
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-sm text-emerald-200/70 mt-3"
        >
          {formatTimestamp(confirmation.timestamp)}
        </motion.p>

        {/* New Payment button */}
        <motion.button
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.7 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleNewPayment}
          className="mt-10 px-10 py-4 rounded-2xl bg-white text-emerald-800 text-lg font-bold
                     shadow-xl shadow-emerald-900/40 hover:bg-emerald-50 active:bg-emerald-100 transition-colors"
          style={{ touchAction: 'manipulation' }}
        >
          New Payment
        </motion.button>
      </div>
    );
  }

  // Fallback (should not reach here)
  return null;
}
