import { useState, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X, BookOpen, Sparkles, TrendingUp, TrendingDown,
  ArrowRightLeft, Coins, Clock, Filter, Loader2,
  Activity, DollarSign, ChevronRight, ExternalLink,
  MessageCircle, User, Bot, Zap
} from "lucide-react";
import { qnkAPI, type UnifiedTransactionEntry } from "../services/api";

// ── Known AI Agent Wallets ──────────────────────────────────────────
const KNOWN_AGENTS: Record<string, { name: string; color: string; icon: string }> = {
  qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723: { name: "Viktor", color: "#d4af37", icon: "👑" },
  qnk7154929a6aa0c118791373ea21004aca6e494e6e031c36f780cd5acedf031ccb: { name: "Claude", color: "#f97316", icon: "🪨" },
  qnka3a92bba0a666947d286d777ea34fe351b3aeb8722fb6187d66ed45586c21f96: { name: "Codex", color: "#8b5cf6", icon: "📜" },
  qnk7f31f299f370a9f751d31a117961c934631887be9d5d14fa57ce867ee78b318b: { name: "DeepSeek", color: "#06b6d4", icon: "🐋" },
};

interface AgentJournalEntry {
  id: string;
  agent: string;
  agentColor: string;
  agentIcon: string;
  tx_type: string;
  amount: number;
  token_symbol: string;
  timestamp: number;
  direction: string;
  memo?: string;
  counterparty?: string;
}

interface AgentJournalModalProps {
  isOpen: boolean;
  onClose: () => void;
}

// ── Helper: format relative time ────────────────────────────────────
function timeAgo(ts: number): string {
  const diff = Math.floor((Date.now() / 1000) - ts);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatAmount(n: number): string {
  if (n >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (n >= 1) return n.toFixed(4);
  return n.toFixed(8);
}

// ── Direction badge ─────────────────────────────────────────────────
function DirectionBadge({ direction, tx_type }: { direction: string; tx_type: string }) {
  if (tx_type === "swap") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-violet-500/15 text-violet-300 border border-violet-500/30">
        <ArrowRightLeft className="w-3 h-3" /> SWAP
      </span>
    );
  }
  if (direction === "sent") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-red-500/15 text-red-300 border border-red-500/30">
        <TrendingUp className="w-3 h-3" /> OUT
      </span>
    );
  }
  if (direction === "received") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
        <TrendingDown className="w-3 h-3" /> IN
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-slate-500/15 text-slate-300 border border-slate-500/30">
      {tx_type}
    </span>
  );
}

export default function AgentJournalModal({ isOpen, onClose }: AgentJournalModalProps) {
  const [entries, setEntries] = useState<AgentJournalEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeAgents, setActiveAgents] = useState<Set<string>>(new Set(Object.keys(KNOWN_AGENTS)));
  const [hoveredEntry, setHoveredEntry] = useState<string | null>(null);

  // ── Fetch today's trades for all known agents ─────────────────────
  useEffect(() => {
    if (!isOpen) return;
    
    let cancelled = false;
    setLoading(true);
    setError(null);

    const fetchAll = async () => {
      const todayStart = new Date();
      todayStart.setHours(0, 0, 0, 0);
      const todayTs = Math.floor(todayStart.getTime() / 1000);

      const allEntries: AgentJournalEntry[] = [];
      const agentAddrs = Object.keys(KNOWN_AGENTS);

      for (const addr of agentAddrs) {
        try {
          const cleanAddr = addr.startsWith("qnk") ? addr.slice(3) : addr;
          const res = await qnkAPI.getWalletHistory(cleanAddr, 50);
          if (!res.success || !res.data) continue;

          const agent = KNOWN_AGENTS[addr];
          for (const tx of res.data) {
            if (tx.timestamp < todayTs) continue;
            
            const entry: AgentJournalEntry = {
              id: tx.id,
              agent: agent.name,
              agentColor: agent.color,
              agentIcon: agent.icon,
              tx_type: tx.tx_type || "transfer",
              amount: parseFloat(tx.amount || "0"),
              token_symbol: tx.token_symbol || "QUG",
              timestamp: tx.timestamp,
              direction: tx.direction || "unknown",
              memo: tx.memo,
              counterparty: tx.direction === "sent" ? tx.to : tx.from,
            };
            allEntries.push(entry);
          }
        } catch (e) {
          console.warn(`[AgentJournal] Failed to fetch for ${agentAddrs}:`, e);
        }
      }

      if (cancelled) return;

      // Sort newest first
      allEntries.sort((a, b) => b.timestamp - a.timestamp);
      setEntries(allEntries);
      setLoading(false);
    };

    fetchAll();
    return () => { cancelled = true; };
  }, [isOpen]);

  // ── Filter by active agents ────────────────────────────────────────
  const filteredEntries = useMemo(() => {
    return entries.filter(e => activeAgents.has(
      Object.entries(KNOWN_AGENTS).find(([, v]) => v.name === e.agent)?.[0] || ""
    ));
  }, [entries, activeAgents]);

  // ── Summary stats ──────────────────────────────────────────────────
  const stats = useMemo(() => {
    const total = filteredEntries.length;
    const volume = filteredEntries.reduce((sum, e) => sum + (isNaN(e.amount) ? 0 : e.amount), 0);
    const swaps = filteredEntries.filter(e => e.tx_type === "swap").length;
    const transfers = filteredEntries.filter(e => e.tx_type === "transfer").length;
    const agentCounts: Record<string, number> = {};
    filteredEntries.forEach(e => { agentCounts[e.agent] = (agentCounts[e.agent] || 0) + 1; });
    const mostActive = Object.entries(agentCounts).sort((a, b) => b[1] - a[1])[0];
    return { total, volume, swaps, transfers, mostActive };
  }, [filteredEntries]);

  const toggleAgent = (addr: string) => {
    setActiveAgents(prev => {
      const next = new Set(prev);
      if (next.has(addr)) next.delete(addr);
      else next.add(addr);
      return next;
    });
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-[150] flex justify-end"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          {/* Backdrop */}
          <motion.div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />

          {/* Slide-in panel */}
          <motion.div
            className="relative w-[480px] max-w-[92vw] h-full bg-slate-900/98 border-l border-amber-500/20 shadow-2xl overflow-hidden flex flex-col"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 28, stiffness: 300 }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* ── Header ──────────────────────────────────────────── */}
            <div className="shrink-0 px-5 py-4 border-b border-white/5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center"
                    style={{
                      background: "linear-gradient(135deg, rgba(212,175,55,0.25), rgba(255,215,0,0.1))",
                      border: "1px solid rgba(212,175,55,0.35)",
                    }}
                  >
                    <BookOpen className="w-5 h-5 text-amber-400" />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-amber-100">Agentic Money Journal</h2>
                    <p className="text-xs text-amber-400/60">
                      {new Date().toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" })}
                    </p>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="w-8 h-8 rounded-lg flex items-center justify-center hover:bg-white/5 transition-colors"
                >
                  <X className="w-4 h-4 text-slate-400" />
                </button>
              </div>

              {/* Agent filter toggles */}
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(KNOWN_AGENTS).map(([addr, agent]) => {
                  const isActive = activeAgents.has(addr);
                  const count = entries.filter(e => e.agent === agent.name).length;
                  return (
                    <motion.button
                      key={addr}
                      whileHover={{ scale: 1.04 }}
                      whileTap={{ scale: 0.96 }}
                      onClick={() => toggleAgent(addr)}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-all"
                      style={{
                        background: isActive
                          ? `${agent.color}20`
                          : "rgba(255,255,255,0.03)",
                        border: `1px solid ${isActive ? agent.color + "50" : "rgba(255,255,255,0.06)"}`,
                        color: isActive ? agent.color : "rgba(148,163,184,0.6)",
                        opacity: count === 0 ? 0.35 : 1,
                      }}
                    >
                      <span>{agent.icon}</span>
                      <span>{agent.name}</span>
                      <span className="text-[10px] opacity-60 ml-0.5">{count}</span>
                    </motion.button>
                  );
                })}
              </div>
            </div>

            {/* ── Summary strip ───────────────────────────────────── */}
            <div className="shrink-0 px-5 py-2.5 grid grid-cols-4 gap-2 border-b border-white/5">
              <div className="text-center">
                <div className="text-[10px] text-slate-500 uppercase tracking-wider">Trades</div>
                <div className="text-base font-bold text-amber-200">{stats.total}</div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-slate-500 uppercase tracking-wider">Volume</div>
                <div className="text-base font-bold text-amber-200">
                  {stats.volume >= 1000
                    ? `${(stats.volume / 1000).toFixed(1)}k`
                    : formatAmount(stats.volume)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-slate-500 uppercase tracking-wider">Swaps</div>
                <div className="text-base font-bold text-violet-300">{stats.swaps}</div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-slate-500 uppercase tracking-wider">Most Active</div>
                <div className="text-base font-bold text-amber-200 truncate">
                  {stats.mostActive?.[0] || "—"}
                </div>
              </div>
            </div>

            {/* ── Trade list ──────────────────────────────────────── */}
            <div className="flex-1 overflow-y-auto px-3 py-2">
              {loading ? (
                <div className="flex items-center justify-center py-16">
                  <Loader2 className="w-6 h-6 text-amber-400 animate-spin" />
                  <span className="ml-2 text-amber-300/70 text-sm">Loading journal...</span>
                </div>
              ) : error ? (
                <div className="text-center py-16 text-red-400/80 text-sm">{error}</div>
              ) : filteredEntries.length === 0 ? (
                <div className="text-center py-16">
                  <BookOpen className="w-10 h-10 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-500 text-sm">No agent trades today</p>
                  <p className="text-slate-600 text-xs mt-1">
                    AI agents haven&apos;t traded yet. Check back later.
                  </p>
                </div>
              ) : (
                <div className="space-y-1.5">
                  <AnimatePresence initial={false}>
                    {filteredEntries.map((entry) => (
                      <motion.div
                        key={entry.id}
                        layout
                        initial={{ opacity: 0, y: -8 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        onMouseEnter={() => setHoveredEntry(entry.id)}
                        onMouseLeave={() => setHoveredEntry(null)}
                        className="relative group rounded-xl px-3.5 py-3 transition-all cursor-default"
                        style={{
                          background: hoveredEntry === entry.id
                            ? `${entry.agentColor}10`
                            : "rgba(255,255,255,0.015)",
                          border: `1px solid ${hoveredEntry === entry.id ? entry.agentColor + "30" : "transparent"}`,
                        }}
                      >
                        {/* Left stripe */}
                        <div
                          className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full opacity-60"
                          style={{ background: entry.agentColor }}
                        />

                        <div className="flex items-center gap-3">
                          {/* Agent avatar */}
                          <div
                            className="w-8 h-8 rounded-lg flex items-center justify-center text-sm shrink-0"
                            style={{
                              background: `${entry.agentColor}20`,
                              border: `1px solid ${entry.agentColor}40`,
                            }}
                          >
                            {entry.agentIcon}
                          </div>

                          {/* Content */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-0.5">
                              <span className="text-sm font-semibold text-slate-200">
                                {entry.agent}
                              </span>
                              <DirectionBadge direction={entry.direction} tx_type={entry.tx_type} />
                              {entry.memo && (
                                <span className="text-[10px] text-amber-400/60 truncate max-w-[120px]" title={entry.memo}>
                                  <MessageCircle className="w-3 h-3 inline mr-0.5" />
                                  {entry.memo.slice(0, 30)}
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-2 text-[11px] text-slate-500">
                              <Clock className="w-3 h-3" />
                              <span>{formatTime(entry.timestamp)}</span>
                              <span>·</span>
                              <span>{timeAgo(entry.timestamp)}</span>
                            </div>
                          </div>

                          {/* Amount */}
                          <div className="text-right shrink-0">
                            <div className={`text-sm font-bold font-mono ${entry.direction === "received" ? "text-emerald-400" : entry.direction === "sent" ? "text-red-400" : entry.tx_type === "swap" ? "text-violet-400" : "text-slate-300"}`}>
                              {entry.direction === "received" ? "+" : entry.direction === "sent" ? "-" : ""}
                              {formatAmount(entry.amount)}
                            </div>
                            <div className="text-[10px] text-slate-500 font-medium">
                              {entry.token_symbol}
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              )}
            </div>

            {/* ── Footer ──────────────────────────────────────────── */}
            <div className="shrink-0 px-5 py-3 border-t border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-2 text-[11px] text-slate-500">
                <Sparkles className="w-3.5 h-3.5 text-amber-500/70" />
                <span>Powered by Agentic Money AI</span>
              </div>
              <div className="text-[11px] text-slate-500">
                Live · {new Date().toLocaleTimeString()}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
