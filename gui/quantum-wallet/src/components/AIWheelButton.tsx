import { useState, useRef, useEffect, useCallback, memo } from 'react';
import { createPortal } from 'react-dom';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import {
  Sparkles,
  Send,
  Wallet,
  ArrowUpDown,
  Pickaxe,
  Clock,
  Layers,
  ArrowUpRight,
  Minimize2,
  X,
  Bot,
  User,
  Copy,
  Check,
  Loader2,
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════════
// AIWheelButton — "Quillon Graph AI" floating assistant
// v10.2.1: FAB → Mini Chat → Expanded Modal with SSE streaming
// powered by Nemotron Cascade 2 via Ollama
// ═══════════════════════════════════════════════════════════════

// ── Types ──────────────────────────────────────────────────────

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

interface Command {
  id: string;
  label: string;
  icon: typeof Send;
  color: string;
  example: string;
}

type Mode = 'closed' | 'mini' | 'expanded';

// ── Commands ───────────────────────────────────────────────────

const COMMANDS: Command[] = [
  { id: 'send',    label: 'Send QUG',      icon: Send,        color: '#22C55E', example: 'Send 10 QUG to alice' },
  { id: 'balance', label: 'Check Balance',  icon: Wallet,      color: '#8B5CF6', example: "What's my balance?" },
  { id: 'swap',    label: 'Swap Tokens',    icon: ArrowUpDown, color: '#3B82F6', example: 'Swap 50 QUG for QUGUSD' },
  { id: 'mining',  label: 'Mining Stats',   icon: Pickaxe,     color: '#EF4444', example: 'Show my mining stats' },
  { id: 'history', label: 'TX History',     icon: Clock,       color: '#A855F7', example: 'Show my last 10 transactions' },
  { id: 'stake',   label: 'Stake',          icon: Layers,      color: '#10B981', example: 'Stake 100 QUG' },
];

// ── Smart Command Routing ────────────────────────────────────
// Fetches real blockchain data from QNK API and includes it as
// context so Nemotron can summarize actual user data instead of
// returning generic web search results.

type CommandType = 'mining' | 'balance' | 'history' | 'stake' | 'swap' | 'send' | 'network' | null;

/** Match user query to a command type for smart data fetching */
function detectCommandIntent(query: string): CommandType {
  const q = query.toLowerCase();
  if (/\b(min(ing|er?)|hashrate|hash\s*rate|block.*found|blocks_found)\b/.test(q)) return 'mining';
  if (/\b(balance|how\s*much|funds|coins?|wallet)\b/.test(q)) return 'balance';
  if (/\b(histor|transaction|tx|transfer|recent|last\s*\d+)\b/.test(q)) return 'history';
  if (/\b(stak(e|ing)|qcredit|credit|reward|tier)\b/.test(q)) return 'stake';
  if (/\b(swap|exchange|trade|convert|dex|price|oracle)\b/.test(q)) return 'swap';
  if (/\b(send|transfer.*to|pay)\b/.test(q)) return 'send';
  if (/\b(network|supply|emission|node|peer|height|status)\b/.test(q)) return 'network';
  return null;
}

/** Fetch real API data silently, return formatted context or empty string on failure */
async function fetchCommandContext(intent: CommandType): Promise<string> {
  const walletAddress = localStorage.getItem('walletAddress') || '';
  const results: string[] = [];

  const safeFetch = async (url: string, label: string): Promise<any> => {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  };

  switch (intent) {
    case 'mining': {
      const [stats, node] = await Promise.all([
        walletAddress ? safeFetch(`/api/v1/mining/stats/${encodeURIComponent(walletAddress)}`, 'Mining Stats') : null,
        safeFetch('/api/v1/node/status', 'Node Status'),
      ]);
      if (stats?.data || stats?.hashrate !== undefined) results.push(`Mining Stats: ${JSON.stringify(stats.data || stats)}`);
      if (node?.data || node?.height !== undefined) results.push(`Node Status: ${JSON.stringify(node.data || node)}`);
      break;
    }
    case 'balance': {
      const [node, supply] = await Promise.all([
        safeFetch('/api/v1/node/status', 'Node Status'),
        safeFetch('/api/v1/network/supply', 'Supply'),
      ]);
      if (node?.data || node?.height !== undefined) results.push(`Node Status: ${JSON.stringify(node.data || node)}`);
      if (supply?.data) results.push(`Network Supply: ${JSON.stringify(supply.data)}`);
      // Note: balance endpoint requires auth, so we read from the DOM/localStorage instead
      const cachedBalance = localStorage.getItem('walletBalance');
      if (cachedBalance) results.push(`Cached Wallet Balance: ${cachedBalance} QUG`);
      if (walletAddress) results.push(`Wallet Address: ${walletAddress}`);
      break;
    }
    case 'history': {
      const txHistory = walletAddress
        ? await safeFetch(`/api/v1/transactions/unified-history?wallet=${encodeURIComponent(walletAddress)}&limit=10`, 'TX History')
        : null;
      if (txHistory?.data) results.push(`Recent Transactions: ${JSON.stringify(txHistory.data)}`);
      else if (txHistory?.transactions) results.push(`Recent Transactions: ${JSON.stringify(txHistory.transactions)}`);
      break;
    }
    case 'stake': {
      const [status, tiers] = await Promise.all([
        safeFetch('/api/v1/qcredit/status', 'QCredit Status'),
        safeFetch('/api/v1/qcredit/tiers', 'QCredit Tiers'),
      ]);
      if (status?.data) results.push(`Staking Status: ${JSON.stringify(status.data)}`);
      if (tiers?.data) results.push(`Available Staking Tiers: ${JSON.stringify(tiers.data)}`);
      break;
    }
    case 'swap': {
      const [prices, tokens] = await Promise.all([
        safeFetch('/api/v1/oracle/prices', 'Prices'),
        safeFetch('/api/v1/dex/tokens', 'DEX Tokens'),
      ]);
      if (prices?.data) results.push(`Token Prices: ${JSON.stringify(prices.data)}`);
      if (tokens?.data) results.push(`Available Tokens: ${JSON.stringify(tokens.data)}`);
      break;
    }
    case 'network': {
      const [node, supply, emission] = await Promise.all([
        safeFetch('/api/v1/node/status', 'Node Status'),
        safeFetch('/api/v1/network/supply', 'Network Supply'),
        safeFetch('/api/v1/emission/stats', 'Emission Stats'),
      ]);
      if (node?.data || node?.height !== undefined) results.push(`Node Status: ${JSON.stringify(node.data || node)}`);
      if (supply?.data) results.push(`Network Supply: ${JSON.stringify(supply.data)}`);
      if (emission?.data) results.push(`Emission Stats: ${JSON.stringify(emission.data)}`);
      break;
    }
    case 'send':
      // Send is an action, not a data query — just provide context about the wallet
      if (walletAddress) results.push(`Your Wallet Address: ${walletAddress}`);
      results.push('Note: QUG transfers require the recipient address and a signed transaction. Guide the user through the send process.');
      break;
  }

  if (results.length === 0) return '';
  return results.join('\n');
}

/** Build an enriched query that includes real blockchain data as context */
function buildEnrichedQuery(originalQuery: string, context: string, intent: CommandType): string {
  const intentLabels: Record<string, string> = {
    mining: 'mining statistics',
    balance: 'wallet balance information',
    history: 'transaction history',
    stake: 'staking/QCredit information',
    swap: 'token swap/DEX information',
    send: 'sending QUG tokens',
    network: 'network status',
  };
  const label = intent ? intentLabels[intent] || intent : 'blockchain data';

  return [
    `The user is asking about their ${label} on the Quillon (QNK) blockchain.`,
    `Their original question: "${originalQuery}"`,
    '',
    '=== REAL-TIME BLOCKCHAIN DATA (from QNK node API) ===',
    context,
    '=== END OF BLOCKCHAIN DATA ===',
    '',
    'IMPORTANT: Use ONLY the real blockchain data above to answer. Do NOT use web search results.',
    'Format your response clearly with bullet points or tables where appropriate.',
    'If any data field is null or missing, mention that the data is currently unavailable.',
    'Use QUG as the token symbol. Amounts with many decimal places should be shown with 4-6 significant digits.',
  ].join('\n');
}

// ── Shared Styles ──────────────────────────────────────────────

const PANEL_BG = 'linear-gradient(180deg, rgba(15,23,42,0.98), rgba(30,41,59,0.98))';
const BORDER_SUBTLE = '1px solid rgba(212,175,55,0.2)';
const BORDER_ACTIVE = '2px solid rgba(212,175,55,0.4)';
const GLOW_GOLD = '0 0 20px rgba(212,175,55,0.2)';
const GOLD_GRADIENT = 'linear-gradient(135deg, #D4AF37, #FFD700, #D4AF37)';
const INPUT_BG = 'rgba(30,41,59,0.5)';

// ── MessageBubble ──────────────────────────────────────────────

const MessageBubble = memo(function MessageBubble({
  message,
  compact,
}: {
  message: Message;
  compact?: boolean;
}) {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const isUser = message.role === 'user';

  const handleCopy = useCallback((text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  }, []);

  return (
    <div className={`flex gap-2 ${isUser ? 'flex-row-reverse' : 'flex-row'} ${compact ? 'mb-2' : 'mb-4'}`}>
      {/* Avatar */}
      <div
        className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5"
        style={{
          background: isUser
            ? 'linear-gradient(135deg, #D4AF37, #FFD700)'
            : 'linear-gradient(135deg, #334155, #475569)',
          border: isUser ? '1px solid rgba(255,215,0,0.4)' : '1px solid rgba(100,116,139,0.4)',
        }}
      >
        {isUser ? (
          <User className="w-3.5 h-3.5" style={{ color: '#0F172A' }} />
        ) : (
          <Bot className="w-3.5 h-3.5 text-amber-400" />
        )}
      </div>

      {/* Bubble */}
      <div
        className={`relative group max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${
          compact ? 'text-xs' : 'text-sm'
        }`}
        style={{
          background: isUser ? 'rgba(212,175,55,0.12)' : 'rgba(51,65,85,0.4)',
          border: isUser ? '1px solid rgba(212,175,55,0.25)' : '1px solid rgba(100,116,139,0.2)',
          color: isUser ? '#FDE68A' : '#E2E8F0',
        }}
      >
        {isUser ? (
          <span>{message.content}</span>
        ) : (
          <div className="prose prose-invert prose-sm max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight, rehypeRaw]}
              components={{
                code: ({ inline, className, children, ...props }: any) => {
                  const match = /language-(\w+)/.exec(className || '');
                  const codeStr = String(children).replace(/\n$/, '');
                  const codeId = `${message.id}-${codeStr.slice(0, 16)}`;
                  return !inline && match ? (
                    <div className="relative my-3 group/code">
                      <div className="absolute top-0 right-0 flex items-center gap-1 px-2 py-1 text-xs rounded-bl-lg rounded-tr-lg"
                        style={{ background: 'rgba(15,23,42,0.8)', border: BORDER_SUBTLE }}>
                        <span className="text-amber-400 text-[10px]">{match[1]}</span>
                        <button onClick={() => handleCopy(codeStr, codeId)}
                          className="ml-1 p-0.5 rounded hover:bg-amber-500/20 transition-all">
                          {copiedId === codeId
                            ? <Check className="w-3 h-3 text-green-400" />
                            : <Copy className="w-3 h-3 text-amber-400/70" />}
                        </button>
                      </div>
                      <code className={`${className} block p-3 pt-7 rounded-lg overflow-x-auto text-xs`}
                        style={{ background: 'rgba(15,23,42,0.8)', border: BORDER_SUBTLE }} {...props}>
                        {children}
                      </code>
                    </div>
                  ) : (
                    <code className="px-1.5 py-0.5 rounded text-xs"
                      style={{ background: 'rgba(212,175,55,0.15)', border: '1px solid rgba(212,175,55,0.3)', color: '#FCD34D' }}
                      {...props}>
                      {children}
                    </code>
                  );
                },
                pre: ({ children }: any) => <div className="not-prose">{children}</div>,
                p: ({ children }: any) => <p className="mb-2 last:mb-0">{children}</p>,
                ul: ({ children }: any) => <ul className="list-disc list-inside mb-2 space-y-0.5">{children}</ul>,
                ol: ({ children }: any) => <ol className="list-decimal list-inside mb-2 space-y-0.5">{children}</ol>,
                li: ({ children }: any) => <li className="text-amber-50/90">{children}</li>,
                h1: ({ children }: any) => <h1 className="text-lg font-bold text-amber-400 mb-2 mt-3">{children}</h1>,
                h2: ({ children }: any) => <h2 className="text-base font-bold text-amber-400 mb-1.5 mt-2">{children}</h2>,
                h3: ({ children }: any) => <h3 className="text-sm font-bold text-amber-400 mb-1 mt-2">{children}</h3>,
                blockquote: ({ children }: any) => (
                  <blockquote className="border-l-3 border-amber-500/50 pl-3 italic text-amber-200/80 my-2">{children}</blockquote>
                ),
                a: ({ children, href }: any) => (
                  <a href={href} target="_blank" rel="noopener noreferrer" className="text-amber-400 hover:text-amber-300 underline">
                    {children}
                  </a>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Copy button for assistant messages (non-compact) */}
        {!isUser && !compact && (
          <button
            onClick={() => handleCopy(message.content, message.id)}
            className="absolute -bottom-1 right-2 opacity-0 group-hover:opacity-100 p-1 rounded transition-all"
            style={{ background: 'rgba(15,23,42,0.8)' }}
          >
            {copiedId === message.id
              ? <Check className="w-3 h-3 text-green-400" />
              : <Copy className="w-3 h-3 text-slate-400" />}
          </button>
        )}
      </div>
    </div>
  );
});

// ── CommandPalette ──────────────────────────────────────────────

function CommandPalette({ onCommand }: { onCommand: (example: string) => void }) {
  return (
    <div className="flex flex-col gap-2 p-3">
      <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-1 px-1">
        Quick Commands
      </div>
      {COMMANDS.map((cmd) => (
        <button
          key={cmd.id}
          onClick={() => onCommand(cmd.example)}
          className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-left transition-all hover:scale-[1.02]"
          style={{
            background: `${cmd.color}10`,
            border: `1px solid ${cmd.color}25`,
          }}
          onMouseEnter={(e) => {
            (e.currentTarget.style.background = `${cmd.color}20`);
            (e.currentTarget.style.borderColor = `${cmd.color}50`);
          }}
          onMouseLeave={(e) => {
            (e.currentTarget.style.background = `${cmd.color}10`);
            (e.currentTarget.style.borderColor = `${cmd.color}25`);
          }}
        >
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: `${cmd.color}20` }}
          >
            <cmd.icon className="w-4 h-4" style={{ color: cmd.color }} />
          </div>
          <div className="min-w-0">
            <div className="text-xs font-semibold text-slate-200 truncate">{cmd.label}</div>
            <div className="text-[10px] text-slate-500 truncate">{cmd.example}</div>
          </div>
        </button>
      ))}
    </div>
  );
}

// ── ChatInput ──────────────────────────────────────────────────

function ChatInput({
  value,
  onChange,
  onSend,
  loading,
  inputRef,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  loading: boolean;
  inputRef?: React.RefObject<HTMLInputElement | null>;
  placeholder?: string;
}) {
  return (
    <div className="flex items-center gap-2 p-3">
      <div
        className="flex-1 flex items-center gap-2 rounded-xl px-3 py-2.5"
        style={{ background: INPUT_BG, border: '1px solid rgba(212,175,55,0.15)' }}
      >
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); } }}
          placeholder={placeholder || 'Ask Quillon Graph AI...'}
          disabled={loading}
          className="flex-1 bg-transparent text-sm text-white placeholder-slate-500 outline-none disabled:opacity-50"
        />
      </div>
      <button
        onClick={onSend}
        disabled={!value.trim() || loading}
        className="w-10 h-10 rounded-xl flex items-center justify-center transition-all disabled:opacity-30 hover:scale-105 active:scale-95"
        style={{
          background: value.trim() && !loading ? GOLD_GRADIENT : 'rgba(51,65,85,0.5)',
          border: value.trim() && !loading ? '1px solid rgba(255,215,0,0.4)' : '1px solid rgba(100,116,139,0.2)',
        }}
      >
        {loading ? (
          <Loader2 className="w-4 h-4 text-amber-400 animate-spin" />
        ) : (
          <Send className="w-4 h-4" style={{ color: value.trim() ? '#0F172A' : '#64748B' }} />
        )}
      </button>
    </div>
  );
}

// ── MiniChatPanel ──────────────────────────────────────────────

function MiniChatPanel({
  messages,
  input,
  setInput,
  onSend,
  loading,
  onExpand,
  onClose,
  inputRef,
  scrollRef,
}: {
  messages: Message[];
  input: string;
  setInput: (v: string) => void;
  onSend: () => void;
  loading: boolean;
  onExpand: () => void;
  onClose: () => void;
  inputRef: React.RefObject<HTMLInputElement | null>;
  scrollRef: React.RefObject<HTMLDivElement | null>;
}) {
  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  // Close on outside click (delayed registration to avoid catching the FAB click itself)
  useEffect(() => {
    let active = true;
    const timer = setTimeout(() => {
      if (!active) return;
      const handler = (e: MouseEvent) => {
        const target = e.target as HTMLElement;
        if (!target.closest('[data-ai-panel]') && !target.closest('[data-ai-fab]')) {
          onClose();
        }
      };
      document.addEventListener('mousedown', handler);
      // Store handler for cleanup
      (window as any).__aiOutsideHandler = handler;
    }, 150);
    return () => {
      active = false;
      clearTimeout(timer);
      if ((window as any).__aiOutsideHandler) {
        document.removeEventListener('mousedown', (window as any).__aiOutsideHandler);
        (window as any).__aiOutsideHandler = null;
      }
    };
  }, [onClose]);

  const visibleMessages = messages.slice(-5);

  return createPortal(
    <motion.div
      data-ai-panel
      initial={{ opacity: 0, y: 16, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
      className="fixed bottom-24 right-6 z-[10001]"
      style={{ width: 'min(320px, calc(100vw - 48px))' }}
    >
      <div
        className="rounded-2xl overflow-hidden"
        style={{ background: PANEL_BG, border: BORDER_ACTIVE, boxShadow: `${GLOW_GOLD}, 0 20px 60px rgba(0,0,0,0.5)` }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-3.5 py-2.5"
          style={{ borderBottom: BORDER_SUBTLE }}>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-full flex items-center justify-center"
              style={{ background: GOLD_GRADIENT }}>
              <Bot className="w-3.5 h-3.5" style={{ color: '#0F172A' }} />
            </div>
            <span className="text-sm font-semibold text-amber-100">Quillon Graph AI</span>
            <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
          </div>
          <button onClick={onExpand}
            className="p-1.5 rounded-lg hover:bg-slate-700/50 transition-colors"
            title="Expand">
            <ArrowUpRight className="w-4 h-4 text-slate-400" />
          </button>
        </div>

        {/* Messages */}
        <div ref={scrollRef} className="overflow-y-auto px-3 py-2" style={{ maxHeight: '256px' }}>
          {visibleMessages.length === 0 ? (
            <div className="text-center py-6">
              <Sparkles className="w-6 h-6 text-amber-400/40 mx-auto mb-2" />
              <div className="text-xs text-slate-500">Ask me anything about Quillon</div>
            </div>
          ) : (
            visibleMessages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} compact />
            ))
          )}
          {loading && (
            <div className="flex items-center gap-2 mb-2">
              <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0"
                style={{ background: 'linear-gradient(135deg, #334155, #475569)' }}>
                <Bot className="w-3.5 h-3.5 text-amber-400" />
              </div>
              <div className="flex gap-1 px-3 py-2 rounded-2xl" style={{ background: 'rgba(51,65,85,0.4)', border: '1px solid rgba(100,116,139,0.2)' }}>
                <span className="w-1.5 h-1.5 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div style={{ borderTop: BORDER_SUBTLE }}>
          <ChatInput value={input} onChange={setInput} onSend={onSend} loading={loading}
            inputRef={inputRef} placeholder="Ask anything..." />
        </div>
      </div>
    </motion.div>,
    document.body,
  );
}

// ── ExpandedModal ──────────────────────────────────────────────

function ExpandedModal({
  messages,
  input,
  setInput,
  onSend,
  loading,
  onShrink,
  onClose,
  onCommand,
  inputRef,
  scrollRef,
}: {
  messages: Message[];
  input: string;
  setInput: (v: string) => void;
  onSend: () => void;
  loading: boolean;
  onShrink: () => void;
  onClose: () => void;
  onCommand: (example: string) => void;
  inputRef: React.RefObject<HTMLInputElement | null>;
  scrollRef: React.RefObject<HTMLDivElement | null>;
}) {
  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onShrink(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onShrink]);

  return createPortal(
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="fixed inset-0 z-[50000] flex items-center justify-center p-4 sm:p-6"
      style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)' }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <motion.div
        data-ai-panel
        initial={{ scale: 0.92, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.92, y: 20 }}
        transition={{ type: 'spring', stiffness: 350, damping: 28 }}
        className="flex flex-col rounded-3xl overflow-hidden w-full h-full sm:w-auto sm:h-auto"
        style={{
          maxWidth: 'min(90vw, 1100px)',
          maxHeight: 'min(90vh, 700px)',
          width: '100%',
          height: '100%',
          background: PANEL_BG,
          border: BORDER_ACTIVE,
          boxShadow: `${GLOW_GOLD}, 0 40px 100px rgba(0,0,0,0.6)`,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 flex-shrink-0"
          style={{ borderBottom: BORDER_SUBTLE }}>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full flex items-center justify-center"
              style={{ background: GOLD_GRADIENT }}>
              <Bot className="w-4 h-4" style={{ color: '#0F172A' }} />
            </div>
            <div>
              <div className="text-sm font-bold text-amber-100">Quillon Graph AI</div>
              <div className="text-[10px] text-slate-500">Powered by Nemotron Cascade 2</div>
            </div>
            <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse ml-1" />
          </div>
          <div className="flex items-center gap-1.5">
            <button onClick={onShrink}
              className="p-2 rounded-lg hover:bg-slate-700/50 transition-colors" title="Minimize">
              <Minimize2 className="w-4 h-4 text-slate-400" />
            </button>
            <button onClick={onClose}
              className="p-2 rounded-lg hover:bg-red-500/20 transition-colors" title="Close">
              <X className="w-4 h-4 text-slate-400 hover:text-red-400" />
            </button>
          </div>
        </div>

        {/* Body: Chat + Commands */}
        <div className="flex flex-1 min-h-0 overflow-hidden">
          {/* Chat Area */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Messages */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-4">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center py-12">
                  <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                    style={{ background: 'rgba(212,175,55,0.1)', border: '1px solid rgba(212,175,55,0.15)' }}>
                    <Sparkles className="w-8 h-8 text-amber-400/60" />
                  </div>
                  <div className="text-base font-semibold text-slate-300 mb-1">How can I help?</div>
                  <div className="text-xs text-slate-500 max-w-xs">
                    Ask about balances, transactions, mining, or use a quick command from the right panel.
                  </div>
                </div>
              ) : (
                messages.map((msg) => (
                  <MessageBubble key={msg.id} message={msg} />
                ))
              )}
              {loading && (
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0"
                    style={{ background: 'linear-gradient(135deg, #334155, #475569)' }}>
                    <Bot className="w-3.5 h-3.5 text-amber-400" />
                  </div>
                  <div className="flex gap-1.5 px-4 py-3 rounded-2xl" style={{ background: 'rgba(51,65,85,0.4)', border: '1px solid rgba(100,116,139,0.2)' }}>
                    <span className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              )}
            </div>

            {/* Input */}
            <div className="flex-shrink-0" style={{ borderTop: BORDER_SUBTLE }}>
              <ChatInput value={input} onChange={setInput} onSend={onSend} loading={loading}
                inputRef={inputRef} />
            </div>
          </div>

          {/* Commands Panel (hidden on mobile) */}
          <div className="hidden sm:flex flex-col w-56 flex-shrink-0 overflow-y-auto"
            style={{ borderLeft: BORDER_SUBTLE }}>
            <CommandPalette onCommand={onCommand} />
          </div>
        </div>
      </motion.div>
    </motion.div>,
    document.body,
  );
}

// ── Root: AIWheelButton ────────────────────────────────────────

let msgCounter = 0;
function nextId() { return `msg-${++msgCounter}-${Date.now()}`; }

export default function AIWheelButton() {
  const [mode, setMode] = useState<Mode>('closed');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  // Focus input when mode changes
  useEffect(() => {
    if (mode !== 'closed') {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [mode]);

  // SSE streaming call — supports direct chat, smart commands, and web search
  // context='' means direct chat (no web search), context with data means smart command,
  // context=undefined means web search mode (DuckDuckGo)
  const streamMessage = useCallback(async (query: string, context?: string) => {
    const userMsg: Message = { id: nextId(), role: 'user', content: query };
    const assistantId = nextId();
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;

    // Build request body — include context to skip web search
    const body: Record<string, any> = { query, stream: true };
    if (context !== undefined) {
      body.context = context; // '' = direct chat, non-empty = smart command data
    }

    try {
      const res = await fetch('/api/v1/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        setMessages((prev) => [
          ...prev,
          { id: assistantId, role: 'assistant', content: `**Error:** ${errData.error || `Server returned ${res.status}`}` },
        ]);
        setLoading(false);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        setMessages((prev) => [
          ...prev,
          { id: assistantId, role: 'assistant', content: '**Error:** No response stream.' },
        ]);
        setLoading(false);
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let cumulativeText = '';
      let created = false;
      let currentEventType = ''; // Local variable instead of window global

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith(':')) continue; // Skip empty lines and SSE comments/keepalives

          if (trimmed.startsWith('event: ')) {
            currentEventType = trimmed.slice(7);
            continue;
          }

          if (trimmed.startsWith('data: ')) {
            const data = trimmed.slice(6);
            const eventType = currentEventType || 'token';
            currentEventType = '';

            try {
              const parsed = JSON.parse(data);

              switch (eventType) {
                case 'search_results':
                  // Show sources found message while waiting for AI inference
                  if (parsed.results?.length) {
                    const sourceCount = parsed.results.length;
                    const sourceSummary = `*Searching ${sourceCount} sources...*\n\n`;
                    if (!created) {
                      created = true;
                      cumulativeText = sourceSummary;
                      setMessages((prev) => [
                        ...prev,
                        { id: assistantId, role: 'assistant', content: sourceSummary },
                      ]);
                    }
                  }
                  break;
                case 'token':
                  if (parsed.content) {
                    // On first real token, clear the "searching" placeholder
                    if (cumulativeText.startsWith('*Searching')) {
                      cumulativeText = '';
                    }
                    cumulativeText += parsed.content;
                    const snapshot = cumulativeText;
                    if (!created) {
                      created = true;
                      setMessages((prev) => [
                        ...prev,
                        { id: assistantId, role: 'assistant', content: snapshot },
                      ]);
                    } else {
                      setMessages((prev) =>
                        prev.map((m) => (m.id === assistantId ? { ...m, content: snapshot } : m)),
                      );
                    }
                  }
                  break;
                case 'done':
                  break;
                case 'error':
                  cumulativeText += `\n\n**Error:** ${parsed.message || 'Unknown error'}`;
                  if (!created) {
                    created = true;
                    setMessages((prev) => [
                      ...prev,
                      { id: assistantId, role: 'assistant', content: cumulativeText },
                    ]);
                  } else {
                    setMessages((prev) =>
                      prev.map((m) => (m.id === assistantId ? { ...m, content: cumulativeText } : m)),
                    );
                  }
                  break;
              }
            } catch {
              // Non-JSON data line, skip
            }
          }
        }
      }

      // If no tokens came through, show a fallback
      if (!created) {
        setMessages((prev) => [
          ...prev,
          { id: assistantId, role: 'assistant', content: cumulativeText || 'No response received. The AI model may still be loading — please try again.' },
        ]);
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        console.error('[Quillon Graph AI] Fetch error:', err);
        const msg = err.message?.includes('network')
          ? 'Network error — the AI model may be loading. Please wait a moment and try again.'
          : `Connection error: ${err.message || 'Unknown error'}`;
        setMessages((prev) => [
          ...prev,
          { id: assistantId, role: 'assistant', content: `**Error:** ${msg}` },
        ]);
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, []);

  // Smart send: detect command intent, fetch real data, or direct chat
  const smartSend = useCallback(async (query: string) => {
    const intent = detectCommandIntent(query);
    if (!intent) {
      // No command detected — direct chat mode (no web search, just talk to Nemotron)
      streamMessage(query, '');
      return;
    }

    // Fetch real blockchain data for this command
    const apiData = await fetchCommandContext(intent);
    if (apiData) {
      const enriched = buildEnrichedQuery(query, apiData, intent);
      streamMessage(query, enriched);
    } else {
      // API fetch failed — fall back to direct chat (still no web search)
      streamMessage(query, '');
    }
  }, [streamMessage]);

  const handleSend = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;
    setInput('');
    smartSend(trimmed);
  }, [input, loading, smartSend]);

  const handleCommand = useCallback((example: string) => {
    setInput('');
    smartSend(example);
  }, [smartSend]);

  const handleFabClick = useCallback(() => {
    if (mode === 'closed') {
      setMode('mini');
    } else {
      // Cancel in-flight stream
      abortRef.current?.abort();
      setMode('closed');
    }
  }, [mode]);

  const handleExpand = useCallback(() => setMode('expanded'), []);
  const handleShrink = useCallback(() => setMode('mini'), []);
  const handleClose = useCallback(() => {
    abortRef.current?.abort();
    setMode('closed');
  }, []);

  return (
    <>
      {/* Panels via portals — no AnimatePresence wrapper (incompatible with portal-returning children) */}
      {mode === 'mini' && (
        <MiniChatPanel
          messages={messages}
          input={input}
          setInput={setInput}
          onSend={handleSend}
          loading={loading}
          onExpand={handleExpand}
          onClose={handleClose}
          inputRef={inputRef}
          scrollRef={scrollRef}
        />
      )}
      {mode === 'expanded' && (
        <ExpandedModal
          messages={messages}
          input={input}
          setInput={setInput}
          onSend={handleSend}
          loading={loading}
          onShrink={handleShrink}
          onClose={handleClose}
          onCommand={handleCommand}
          inputRef={inputRef}
          scrollRef={scrollRef}
        />
      )}

      {/* FAB */}
      <div className="fixed bottom-6 right-6 z-[9990]" style={{ pointerEvents: 'auto' }}>
        <motion.button
          data-ai-fab
          whileHover={{ scale: 1.08 }}
          whileTap={{ scale: 0.92 }}
          onClick={handleFabClick}
          className="relative w-14 h-14 rounded-full flex items-center justify-center shadow-2xl"
          style={{
            background: mode !== 'closed'
              ? 'linear-gradient(135deg, #1e1b4b, #0f172a)'
              : GOLD_GRADIENT,
            border: mode !== 'closed'
              ? '2px solid rgba(239,68,68,0.4)'
              : '2px solid rgba(255,215,0,0.5)',
            boxShadow: mode !== 'closed'
              ? '0 0 20px rgba(239,68,68,0.2), 0 8px 32px rgba(0,0,0,0.4)'
              : '0 0 25px rgba(212,175,55,0.3), 0 8px 32px rgba(0,0,0,0.4)',
          }}
        >
          <motion.div
            animate={{ rotate: mode !== 'closed' ? 135 : 0 }}
            transition={{ duration: 0.3, type: 'spring', stiffness: 200 }}
          >
            {mode !== 'closed' ? (
              <X className="w-6 h-6 text-red-400" />
            ) : (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
              >
                <Sparkles className="w-6 h-6" style={{ color: '#0F172A' }} />
              </motion.div>
            )}
          </motion.div>

          {/* Pulse ring when closed */}
          {mode === 'closed' && (
            <motion.div
              className="absolute inset-0 rounded-full pointer-events-none"
              style={{ border: '2px solid rgba(212,175,55,0.4)' }}
              animate={{
                scale: [1, 1.5, 1.5],
                opacity: [0.6, 0, 0],
              }}
              transition={{
                duration: 2.5,
                repeat: Infinity,
                ease: 'easeOut',
              }}
            />
          )}
        </motion.button>
      </div>
    </>
  );
}
