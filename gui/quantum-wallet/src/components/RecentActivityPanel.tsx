/**
 * RecentActivityPanel — full-feature transaction history sidebar.
 *
 * Replaces the inline "Recent Activity" block that used to live inside
 * Dashboard.tsx's right column. The previous version had only 5 type-pills
 * and no way to find a transaction by memo, address, or amount range, which
 * is why users said "filtering doesn't work and I can't find memo anywhere".
 *
 * What this gives you:
 *  - Free-text search across memo + from/to address + tx hash
 *  - Type filter (all/receive/send/mining/swap) — same as before, but reactive
 *  - Sort by date or amount, ascending or descending
 *  - Amount range (min / max in display QUG)
 *  - Date range (from / to)
 *  - Memo visible on every row when present (with overflow ellipsis + tooltip)
 *  - Status pill (confirmed / pending / finalized) derived from tx.status
 *  - Copy-to-clipboard with toast for tx hash + from + to
 *  - Pagination (20 per page, identical to old behaviour) — list virtualises
 *    cheaply since each page renders <=20 rows
 *  - LIVE indicator wired to the existing sseConnected prop so the parent
 *    SSE pipeline (sseManager) keeps driving real-time refreshes
 *
 * Memo data path: backend handlers.rs:5266 → api.ts UnifiedTransactionEntry.memo
 * → Dashboard fetchRecentTransactionsCore (Dashboard.tsx ~1595) → tx.memo here.
 */

import { useMemo, useState, useEffect, useCallback } from 'react';
import type { MouseEvent as ReactMouseEvent } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  Zap,
  ChevronLeft,
  ChevronRight,
  Search,
  X,
  SlidersHorizontal,
  ArrowDownUp,
  Copy,
  Check,
  MessageSquare,
  CircleCheck,
  CircleDot,
  Sparkles,
} from 'lucide-react';
import { TICKER_SYMBOL } from '../constants/ticker';

// Mirrors the shape used by Dashboard.tsx's Transaction interface.
// Keep this in sync with Dashboard.tsx:69-84.
export interface ActivityTransaction {
  id: string;
  type: 'receive' | 'send' | 'mining' | 'swap';
  amount: number;
  from?: string;
  to?: string;
  timestamp: string;
  txHash: string;
  tokenSymbol?: string;
  tokenAddress?: string;
  amountOut?: string;
  tokenIn?: string;
  tokenOut?: string;
  memo?: string;
  /** v3.5.8-beta: backend returns 'confirmed' for everything currently;
   *  treat missing as 'pending' so mining receipts pre-confirmation read sensibly. */
  status?: string;
  blockHeight?: number;
}

type FilterType = 'all' | 'receive' | 'send' | 'mining' | 'swap';
type SortField = 'date' | 'amount';
type SortOrder = 'asc' | 'desc';
type StatusKind = 'pending' | 'confirmed' | 'finalized';

interface RecentActivityPanelProps {
  transactions: ActivityTransaction[];
  sseConnected: boolean;
  onSelectTransaction: (tx: ActivityTransaction) => void;
  /** Optional QUG balance — kept for parity with old props, unused but accepted. */
  currentHeight?: number;
}

const ITEMS_PER_PAGE = 20;
/** Once the chain has produced this many blocks past a tx it counts as finalized. */
const FINALITY_DEPTH = 6;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function formatAmount(amount: number, hidden = false): string {
  if (hidden) return '••••••••';
  if (!isFinite(amount) || isNaN(amount)) return '0';
  const abs = Math.abs(amount);
  // Thin-space grouping per QUG style guide (U+202F). Intl 'fr-FR' uses
  // a narrow no-break space as default thousands separator.
  const fmt = (min: number, max: number) =>
    new Intl.NumberFormat('fr-FR', { minimumFractionDigits: min, maximumFractionDigits: max })
      .format(amount)
      .replace(/ /g, ' ');
  if (abs >= 1000) return fmt(2, 2);
  if (abs >= 1) return fmt(4, 4);
  if (abs >= 0.0001) return fmt(6, 6);
  return fmt(8, 8);
}

function relativeTime(timestamp: string): string {
  const diffMs = Date.now() - new Date(timestamp).getTime();
  const s = Math.floor(diffMs / 1000);
  if (s < 0) return 'just now';
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function shortAddr(addr?: string): string {
  if (!addr) return '';
  if (addr.length <= 14) return addr;
  return `${addr.slice(0, 6)}…${addr.slice(-4)}`;
}

function deriveStatus(tx: ActivityTransaction, networkHeight?: number): StatusKind {
  const raw = (tx.status || '').toLowerCase();
  if (raw === 'pending' || raw === 'unconfirmed') return 'pending';
  if (raw && raw !== 'confirmed' && raw !== 'finalized') return 'confirmed';
  if (networkHeight && tx.blockHeight && tx.blockHeight > 0) {
    const depth = networkHeight - tx.blockHeight;
    if (depth >= FINALITY_DEPTH) return 'finalized';
  }
  if (raw === 'finalized') return 'finalized';
  return raw === '' ? 'confirmed' : (raw as StatusKind);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

interface StatusPillProps {
  kind: StatusKind;
}

function StatusPill({ kind }: StatusPillProps) {
  const map: Record<StatusKind, { label: string; color: string; bg: string; Icon: typeof CircleCheck }> = {
    pending: { label: 'Pending', color: '#FBBF24', bg: 'rgba(251,191,36,0.12)', Icon: CircleDot },
    confirmed: { label: 'Confirmed', color: '#34D399', bg: 'rgba(52,211,153,0.12)', Icon: CircleCheck },
    finalized: { label: 'Finalized', color: '#A78BFA', bg: 'rgba(167,139,250,0.14)', Icon: Sparkles },
  };
  const { label, color, bg, Icon } = map[kind];
  return (
    <span
      className="inline-flex items-center gap-1 px-1.5 py-[1px] rounded-md text-[9px] font-semibold tracking-wide uppercase"
      style={{ background: bg, color }}
    >
      <Icon className="w-2.5 h-2.5" />
      {label}
    </span>
  );
}

interface CopyButtonProps {
  value: string;
  onCopied: (label: string) => void;
  label: string;
  title?: string;
}

function CopyButton({ value, onCopied, label, title }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);
  const handle = useCallback(
    (e: ReactMouseEvent) => {
      e.stopPropagation();
      if (!value) return;
      navigator.clipboard.writeText(value).then(
        () => {
          setCopied(true);
          onCopied(label);
          setTimeout(() => setCopied(false), 1200);
        },
        () => {
          /* swallow — clipboard denied */
        }
      );
    },
    [value, onCopied, label]
  );
  return (
    <motion.button
      whileTap={{ scale: 0.9 }}
      onClick={handle}
      title={title || `Copy ${label}`}
      className="p-0.5 rounded text-white/25 hover:text-amber-300 transition-colors"
      style={{ lineHeight: 0 }}
    >
      {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
    </motion.button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

export default function RecentActivityPanel({
  transactions,
  sseConnected,
  onSelectTransaction,
  currentHeight,
}: RecentActivityPanelProps) {
  // Filter / sort state
  const [filterType, setFilterType] = useState<FilterType>('all');
  const [sortField, setSortField] = useState<SortField>('date');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [search, setSearch] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [minAmount, setMinAmount] = useState('');
  const [maxAmount, setMaxAmount] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [toast, setToast] = useState<string | null>(null);

  // Reset to page 1 whenever filter input changes
  useEffect(() => {
    setCurrentPage(1);
  }, [filterType, sortField, sortOrder, search, minAmount, maxAmount, dateFrom, dateTo]);

  const showToast = useCallback((label: string) => {
    setToast(`${label} copied`);
    setTimeout(() => setToast(null), 1500);
  }, []);

  const filteredSorted = useMemo<ActivityTransaction[]>(() => {
    const min = parseFloat(minAmount);
    const max = parseFloat(maxAmount);
    const fromTs = dateFrom ? new Date(dateFrom).getTime() : Number.NEGATIVE_INFINITY;
    const toTs = dateTo ? new Date(dateTo).getTime() + 86_400_000 : Number.POSITIVE_INFINITY;
    const needle = search.trim().toLowerCase();

    const out = transactions.filter((tx) => {
      if (filterType !== 'all' && tx.type !== filterType) return false;
      if (Number.isFinite(min) && tx.amount < min) return false;
      if (Number.isFinite(max) && tx.amount > max) return false;
      const ts = new Date(tx.timestamp).getTime();
      if (ts < fromTs || ts > toTs) return false;
      if (needle) {
        const haystack = [
          tx.memo,
          tx.from,
          tx.to,
          tx.txHash,
          tx.id,
          tx.tokenSymbol,
          tx.tokenIn,
          tx.tokenOut,
        ]
          .filter(Boolean)
          .join(' ')
          .toLowerCase();
        if (!haystack.includes(needle)) return false;
      }
      return true;
    });

    out.sort((a, b) => {
      let cmp = 0;
      if (sortField === 'date') {
        cmp = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      } else {
        cmp = a.amount - b.amount;
      }
      return sortOrder === 'asc' ? cmp : -cmp;
    });
    return out;
  }, [transactions, filterType, sortField, sortOrder, search, minAmount, maxAmount, dateFrom, dateTo]);

  const totalPages = Math.max(1, Math.ceil(filteredSorted.length / ITEMS_PER_PAGE));
  const safePage = Math.min(currentPage, totalPages);
  const pageStart = (safePage - 1) * ITEMS_PER_PAGE;
  const paginated = filteredSorted.slice(pageStart, pageStart + ITEMS_PER_PAGE);

  const hasAnyFilter =
    filterType !== 'all' ||
    search.length > 0 ||
    minAmount.length > 0 ||
    maxAmount.length > 0 ||
    dateFrom.length > 0 ||
    dateTo.length > 0;

  const clearAllFilters = () => {
    setFilterType('all');
    setSearch('');
    setMinAmount('');
    setMaxAmount('');
    setDateFrom('');
    setDateTo('');
    setSortField('date');
    setSortOrder('desc');
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Render
  // ───────────────────────────────────────────────────────────────────────────

  return (
    <motion.div
      className="rounded-2xl flex flex-col overflow-hidden"
      style={{
        background: 'rgba(10, 12, 20, 0.85)',
        backdropFilter: 'blur(24px)',
        border: '1px solid rgba(212, 175, 55, 0.15)',
        boxShadow: '0 8px 32px rgba(0,0,0,0.4), 0 0 0 0.5px rgba(212,175,55,0.08) inset',
        position: 'sticky',
        top: '1rem',
        maxHeight: 'calc(100vh - 2rem)',
      }}
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.25, type: 'spring', stiffness: 260, damping: 22 }}
    >
      {/* Header */}
      <div className="px-4 pt-4 pb-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div
              className="w-6 h-6 rounded-md flex items-center justify-center"
              style={{ background: 'linear-gradient(135deg, rgba(212,175,55,0.25), rgba(255,215,0,0.15))' }}
            >
              <Zap className="w-3.5 h-3.5 text-amber-400" />
            </div>
            <span className="text-sm font-semibold text-white/90 tracking-tight">Activity</span>
            <span className="text-[11px] text-white/30 font-medium tabular-nums">
              {filteredSorted.length}
              {hasAnyFilter && transactions.length !== filteredSorted.length
                ? `/${transactions.length}`
                : ''}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <motion.div
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: sseConnected ? '#10B981' : '#4B5563' }}
              animate={sseConnected ? { scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] } : {}}
              transition={{ duration: 1.8, repeat: Infinity }}
            />
            <span className="text-[10px] font-medium" style={{ color: sseConnected ? '#34D399' : '#6B7280' }}>
              {sseConnected ? 'LIVE' : 'OFFLINE'}
            </span>
          </div>
        </div>

        {/* Search row */}
        <div className="flex items-center gap-1.5 mb-2">
          <div
            className="flex-1 flex items-center gap-1.5 px-2 py-1.5 rounded-lg"
            style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)' }}
          >
            <Search className="w-3 h-3 text-white/30" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search memo, address, hash…"
              className="flex-1 bg-transparent text-[11px] text-white/85 placeholder:text-white/25 outline-none"
              spellCheck={false}
              autoComplete="off"
            />
            {search && (
              <motion.button
                whileTap={{ scale: 0.9 }}
                onClick={() => setSearch('')}
                className="text-white/30 hover:text-white/60"
                aria-label="Clear search"
              >
                <X className="w-3 h-3" />
              </motion.button>
            )}
          </div>
          <motion.button
            onClick={() => setShowAdvanced((v) => !v)}
            whileTap={{ scale: 0.92 }}
            title="More filters"
            className="p-1.5 rounded-lg text-[11px]"
            style={{
              background: showAdvanced
                ? 'linear-gradient(135deg, rgba(212,175,55,0.35), rgba(245,158,11,0.25))'
                : 'rgba(255,255,255,0.04)',
              color: showAdvanced ? '#0a0c14' : 'rgba(255,255,255,0.4)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            <SlidersHorizontal className="w-3 h-3" />
          </motion.button>
        </div>

        {/* Filter pills */}
        <div className="flex items-center gap-1 flex-wrap">
          {(
            [
              { id: 'all', label: 'All' },
              { id: 'receive', label: '↓ In' },
              { id: 'send', label: '↑ Out' },
              { id: 'mining', label: '⛏ Mine' },
              { id: 'swap', label: '⇄ Swap' },
            ] as const
          ).map(({ id, label }) => (
            <motion.button
              key={id}
              onClick={() => setFilterType(id)}
              className="px-2 py-1 rounded-md text-[10px] font-semibold tracking-wide transition-all"
              style={
                filterType === id
                  ? { background: 'linear-gradient(135deg, #D4AF37, #F59E0B)', color: '#0a0c14' }
                  : {
                      background: 'rgba(255,255,255,0.04)',
                      color: 'rgba(255,255,255,0.35)',
                      border: '1px solid rgba(255,255,255,0.06)',
                    }
              }
              whileTap={{ scale: 0.93 }}
            >
              {label}
            </motion.button>
          ))}
          <motion.button
            onClick={() => {
              if (sortField === 'date') {
                setSortField('amount');
              } else {
                setSortField('date');
              }
            }}
            className="ml-auto px-1.5 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wide"
            style={{
              background: 'rgba(255,255,255,0.04)',
              color: 'rgba(255,255,255,0.45)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
            whileTap={{ scale: 0.93 }}
            title="Toggle sort field"
          >
            {sortField === 'date' ? 'Date' : 'Amount'}
          </motion.button>
          <motion.button
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            className="p-1 rounded-md text-[10px]"
            style={{
              background: 'rgba(255,255,255,0.04)',
              color: 'rgba(255,255,255,0.45)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
            whileTap={{ scale: 0.93 }}
            title={sortOrder === 'desc' ? 'Showing largest/newest first' : 'Showing smallest/oldest first'}
          >
            <ArrowDownUp className="w-3 h-3" style={{ transform: sortOrder === 'asc' ? 'scaleY(-1)' : 'none' }} />
          </motion.button>
        </div>

        {/* Advanced filters drawer */}
        <AnimatePresence initial={false}>
          {showAdvanced && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.18 }}
              className="overflow-hidden"
            >
              <div className="grid grid-cols-2 gap-1.5 pt-2.5">
                <label className="flex flex-col gap-1">
                  <span className="text-[9px] uppercase tracking-wider text-white/30">Min amount</span>
                  <input
                    type="number"
                    value={minAmount}
                    onChange={(e) => setMinAmount(e.target.value)}
                    placeholder="0"
                    step="any"
                    min="0"
                    className="px-2 py-1 rounded-md text-[11px] text-white/85 tabular-nums outline-none"
                    style={{
                      background: 'rgba(255,255,255,0.04)',
                      border: '1px solid rgba(255,255,255,0.06)',
                    }}
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-[9px] uppercase tracking-wider text-white/30">Max amount</span>
                  <input
                    type="number"
                    value={maxAmount}
                    onChange={(e) => setMaxAmount(e.target.value)}
                    placeholder="∞"
                    step="any"
                    min="0"
                    className="px-2 py-1 rounded-md text-[11px] text-white/85 tabular-nums outline-none"
                    style={{
                      background: 'rgba(255,255,255,0.04)',
                      border: '1px solid rgba(255,255,255,0.06)',
                    }}
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-[9px] uppercase tracking-wider text-white/30">From</span>
                  <input
                    type="date"
                    value={dateFrom}
                    onChange={(e) => setDateFrom(e.target.value)}
                    className="px-2 py-1 rounded-md text-[11px] text-white/85 outline-none"
                    style={{
                      background: 'rgba(255,255,255,0.04)',
                      border: '1px solid rgba(255,255,255,0.06)',
                      colorScheme: 'dark',
                    }}
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-[9px] uppercase tracking-wider text-white/30">To</span>
                  <input
                    type="date"
                    value={dateTo}
                    onChange={(e) => setDateTo(e.target.value)}
                    className="px-2 py-1 rounded-md text-[11px] text-white/85 outline-none"
                    style={{
                      background: 'rgba(255,255,255,0.04)',
                      border: '1px solid rgba(255,255,255,0.06)',
                      colorScheme: 'dark',
                    }}
                  />
                </label>
              </div>
              {hasAnyFilter && (
                <motion.button
                  whileTap={{ scale: 0.95 }}
                  onClick={clearAllFilters}
                  className="mt-2 w-full text-[10px] text-amber-300/70 hover:text-amber-300 py-1 rounded-md"
                  style={{ background: 'rgba(212,175,55,0.05)', border: '1px dashed rgba(212,175,55,0.2)' }}
                >
                  Clear all filters
                </motion.button>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Transaction list */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1" style={{ scrollbarWidth: 'none' }}>
        <AnimatePresence initial={false}>
          {paginated.length > 0 ? (
            paginated.map((tx) => {
              const isIn = tx.type === 'receive' || tx.type === 'mining';
              const accentColor =
                tx.type === 'receive'
                  ? '#22C55E'
                  : tx.type === 'mining'
                  ? '#F59E0B'
                  : tx.type === 'swap'
                  ? '#A78BFA'
                  : '#F43F5E';
              const iconBg =
                tx.type === 'receive'
                  ? 'rgba(34,197,94,0.12)'
                  : tx.type === 'mining'
                  ? 'rgba(245,158,11,0.12)'
                  : tx.type === 'swap'
                  ? 'rgba(167,139,250,0.12)'
                  : 'rgba(244,63,94,0.12)';
              const iconGlyph =
                tx.type === 'receive' ? '↓' : tx.type === 'mining' ? '⛏' : tx.type === 'swap' ? '⇄' : '↑';
              const label =
                tx.type === 'receive'
                  ? 'From'
                  : tx.type === 'mining'
                  ? 'Block reward'
                  : tx.type === 'swap'
                  ? `${tx.tokenIn || '?'} → ${tx.tokenOut || '?'}`
                  : 'To';
              const addr = tx.type === 'receive' ? tx.from : tx.type === 'send' ? tx.to : undefined;
              const status = deriveStatus(tx, currentHeight);
              const sign = isIn ? '+' : tx.type === 'swap' ? '' : '−';

              return (
                <motion.div
                  key={tx.id}
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.96 }}
                  transition={{ duration: 0.18 }}
                  className="flex flex-col gap-1 p-2.5 rounded-xl cursor-pointer group"
                  style={{ transition: 'background 0.15s' }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.04)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                  }}
                  onClick={() => onSelectTransaction(tx)}
                >
                  <div className="flex items-center gap-3">
                    {/* Icon */}
                    <div
                      className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-sm"
                      style={{ background: iconBg, color: accentColor }}
                    >
                      {iconGlyph}
                    </div>
                    {/* Middle */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[11px] font-semibold text-white/85 uppercase tracking-wide">
                          {tx.type === 'mining'
                            ? 'Mining'
                            : tx.type === 'swap'
                            ? 'Swap'
                            : tx.type === 'receive'
                            ? 'Received'
                            : 'Sent'}
                        </span>
                        <StatusPill kind={status} />
                        {addr && (
                          <span className="text-[10px] font-mono text-white/30 truncate flex items-center gap-1">
                            {label} {shortAddr(addr)}
                            <CopyButton value={addr} onCopied={showToast} label="Address" />
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <span className="text-[10px] text-white/30">{relativeTime(tx.timestamp)}</span>
                        <span className="text-[10px] text-white/15">·</span>
                        <span className="text-[10px] font-mono text-white/30">{tx.txHash?.slice(0, 8)}…</span>
                        <CopyButton value={tx.txHash} onCopied={showToast} label="Hash" />
                      </div>
                    </div>
                    {/* Amount */}
                    <div className="text-right flex-shrink-0">
                      <div className="text-sm font-bold tabular-nums" style={{ color: accentColor }}>
                        {sign}
                        {formatAmount(tx.amount)}
                      </div>
                      <div className="text-[10px] text-white/30">{tx.tokenSymbol || TICKER_SYMBOL}</div>
                    </div>
                  </div>
                  {/* Memo — full-width row so it never truncates aggressively */}
                  {tx.memo && (
                    <div
                      className="flex items-start gap-1.5 ml-11 mr-1 pl-2 pr-1.5 py-1 rounded-md"
                      style={{
                        background: 'rgba(212,175,55,0.05)',
                        border: '1px solid rgba(212,175,55,0.12)',
                      }}
                      title={tx.memo}
                    >
                      <MessageSquare className="w-3 h-3 text-amber-300/70 flex-shrink-0 mt-[1px]" />
                      <span className="text-[10.5px] italic text-amber-100/85 leading-tight break-words">
                        {tx.memo.length > 140 ? `${tx.memo.slice(0, 140)}…` : tx.memo}
                      </span>
                    </div>
                  )}
                </motion.div>
              );
            })
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center py-16 gap-3"
            >
              <div
                className="w-12 h-12 rounded-2xl flex items-center justify-center"
                style={{ background: 'rgba(212,175,55,0.06)', border: '1px solid rgba(212,175,55,0.1)' }}
              >
                <Activity className="w-5 h-5 text-amber-400/40" />
              </div>
              <div className="text-center">
                <p className="text-sm font-medium text-white/30">
                  {hasAnyFilter ? 'No transactions match your filters' : 'No activity yet'}
                </p>
                <p className="text-[11px] text-white/15 mt-1">
                  {hasAnyFilter ? 'Try clearing them to see all activity.' : 'Transactions will appear instantly'}
                </p>
                {hasAnyFilter && (
                  <motion.button
                    whileTap={{ scale: 0.95 }}
                    onClick={clearAllFilters}
                    className="mt-3 px-3 py-1 rounded-md text-[10px] text-amber-300/80 hover:text-amber-300"
                    style={{
                      background: 'rgba(212,175,55,0.06)',
                      border: '1px solid rgba(212,175,55,0.2)',
                    }}
                  >
                    Clear filters
                  </motion.button>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div
          className="px-4 py-3 flex items-center justify-between"
          style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}
        >
          <span className="text-[10px] text-white/25 tabular-nums">
            {safePage}/{totalPages}
          </span>
          <div className="flex items-center gap-1">
            <motion.button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={safePage === 1}
              className="w-6 h-6 rounded-md flex items-center justify-center disabled:opacity-20"
              style={{ background: 'rgba(255,255,255,0.06)' }}
              whileTap={{ scale: 0.9 }}
            >
              <ChevronLeft className="w-3 h-3 text-white/50" />
            </motion.button>
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              const p =
                totalPages <= 5
                  ? i + 1
                  : safePage <= 3
                  ? i + 1
                  : safePage >= totalPages - 2
                  ? totalPages - 4 + i
                  : safePage - 2 + i;
              return (
                <motion.button
                  key={p}
                  onClick={() => setCurrentPage(p)}
                  className="w-6 h-6 rounded-md text-[10px] font-bold"
                  style={
                    safePage === p
                      ? { background: 'linear-gradient(135deg,#D4AF37,#F59E0B)', color: '#0a0c14' }
                      : { background: 'rgba(255,255,255,0.04)', color: 'rgba(255,255,255,0.3)' }
                  }
                  whileTap={{ scale: 0.9 }}
                >
                  {p}
                </motion.button>
              );
            })}
            <motion.button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={safePage === totalPages}
              className="w-6 h-6 rounded-md flex items-center justify-center disabled:opacity-20"
              style={{ background: 'rgba(255,255,255,0.06)' }}
              whileTap={{ scale: 0.9 }}
            >
              <ChevronRight className="w-3 h-3 text-white/50" />
            </motion.button>
          </div>
        </div>
      )}

      {/* Copy toast */}
      <AnimatePresence>
        {toast && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            className="absolute bottom-3 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-lg text-[11px] font-medium pointer-events-none"
            style={{
              background: 'rgba(15,18,28,0.95)',
              border: '1px solid rgba(212,175,55,0.4)',
              color: '#FCD34D',
              boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
            }}
          >
            {toast}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
