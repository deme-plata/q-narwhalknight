import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import {
  Search,
  Globe,
  ExternalLink,
  Clock,
  Loader2,
  X,
  Sparkles,
  CalendarDays,
  History,
  ChevronRight,
  BookOpen,
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════════
// WebSearchScreen — AI-powered web search with GLM-4-Flash
// Streams AI-summarized answers with source citations via SSE
// ═══════════════════════════════════════════════════════════════

type RecencyFilter = 'any' | 'day' | 'week' | 'month';

interface SourceCard {
  title: string;
  url: string;
  snippet: string;
}

interface HistoryEntry {
  query: string;
  timestamp: number;
}

const RECENCY_OPTIONS: { value: RecencyFilter; label: string }[] = [
  { value: 'any', label: 'Any time' },
  { value: 'day', label: 'Past day' },
  { value: 'week', label: 'Past week' },
  { value: 'month', label: 'Past month' },
];

const QUICK_SUGGESTIONS = [
  'QUG price today',
  'quantum computing news',
  'latest crypto trends',
  'blockchain scalability 2026',
  'post-quantum cryptography',
];

function getDomain(url: string): string {
  try {
    return new URL(url).hostname.replace('www.', '');
  } catch {
    return url;
  }
}

function getFaviconUrl(url: string): string {
  try {
    const domain = new URL(url).origin;
    return `https://www.google.com/s2/favicons?domain=${encodeURIComponent(domain)}&sz=32`;
  } catch {
    return '';
  }
}

export default function WebSearchScreen() {
  const [query, setQuery] = useState('');
  const [streamedText, setStreamedText] = useState('');
  const [sources, setSources] = useState<SourceCard[]>([]);
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [recency, setRecency] = useState<RecencyFilter>('any');
  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try {
      const saved = localStorage.getItem('webSearchHistory');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [showHistory, setShowHistory] = useState(false);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const saveHistory = useCallback((q: string) => {
    const entry: HistoryEntry = { query: q, timestamp: Date.now() };
    const updated = [entry, ...history.filter(h => h.query !== q)].slice(0, 20);
    setHistory(updated);
    localStorage.setItem('webSearchHistory', JSON.stringify(updated));
  }, [history]);

  const handleSearch = useCallback(async (searchQuery?: string) => {
    const q = (searchQuery || query).trim();
    if (!q) return;

    // Abort any ongoing search
    if (abortController) {
      abortController.abort();
    }

    const controller = new AbortController();
    setAbortController(controller);
    setLoading(true);
    setSearching(true);
    setStreamedText('');
    setSources([]);
    setShowHistory(false);
    saveHistory(q);

    try {
      const res = await fetch('/api/v1/web-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          recency: recency === 'any' ? undefined : recency,
          stream: true,
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        setStreamedText(`**Search Error:** ${errData.error || 'Failed to fetch results.'}`);
        setLoading(false);
        setSearching(false);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        setStreamedText('**Error:** No response stream.');
        setLoading(false);
        setSearching(false);
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE lines
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete last line

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed.startsWith('event: ')) {
            // Store event type for next data line
            (window as any).__sseEventType = trimmed.slice(7);
            continue;
          }

          if (trimmed.startsWith('data: ')) {
            const data = trimmed.slice(6);
            const eventType = (window as any).__sseEventType || 'token';
            (window as any).__sseEventType = null;

            try {
              const parsed = JSON.parse(data);

              switch (eventType) {
                case 'token':
                  if (parsed.content) {
                    setStreamedText(prev => prev + parsed.content);
                    setLoading(false); // First token received
                  }
                  break;

                case 'search_results':
                  if (parsed.results && Array.isArray(parsed.results)) {
                    setSources(prev => {
                      const existing = new Set(prev.map(s => s.url));
                      const newResults = parsed.results.filter(
                        (r: SourceCard) => r.url && !existing.has(r.url)
                      );
                      return [...prev, ...newResults];
                    });
                    // Sources arrived — show them immediately, switch to "summarizing"
                    setLoading(false);
                    setSummarizing(true);
                  }
                  break;

                case 'done':
                  setSearching(false);
                  setSummarizing(false);
                  break;

                case 'error':
                  setStreamedText(prev =>
                    prev + `\n\n**Error:** ${parsed.message || 'Unknown error'}`
                  );
                  setSearching(false);
                  setSummarizing(false);
                  break;
              }
            } catch {
              // Non-JSON data, skip
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        console.error('[WebSearch] Error:', err);
        setStreamedText('**Error:** Failed to connect to search service. Please try again.');
      }
    } finally {
      setLoading(false);
      setSearching(false);
      setSummarizing(false);
      setAbortController(null);
    }
  }, [query, recency, abortController, saveHistory]);

  const handleStop = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      setLoading(false);
      setSearching(false);
      setSummarizing(false);
    }
  };

  const clearSearch = () => {
    setQuery('');
    setStreamedText('');
    setSources([]);
    setLoading(false);
    setSearching(false);
    setSummarizing(false);
    if (abortController) {
      abortController.abort();
      setAbortController(null);
    }
    inputRef.current?.focus();
  };

  const hasResults = streamedText.length > 0 || sources.length > 0;

  return (
    <div className="max-w-4xl mx-auto space-y-5 pb-8">
      {/* Header */}
      <div className="text-center mb-4">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-center gap-3 mb-2"
        >
          <div className="relative">
            <Globe className="w-8 h-8 text-amber-400" />
            <Sparkles className="w-3.5 h-3.5 text-amber-300 absolute -top-1 -right-1" />
          </div>
          <h2 className="text-2xl font-bold text-white">Web Search</h2>
        </motion.div>
        <p className="text-sm text-gray-500">
          AI-powered search with real-time web results
        </p>
      </div>

      {/* Search Bar */}
      <motion.form
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        onSubmit={(e) => { e.preventDefault(); handleSearch(); }}
        className="relative"
      >
        <div
          className="flex items-center gap-3 rounded-2xl px-5 py-4 backdrop-blur-2xl transition-all"
          style={{
            background: 'linear-gradient(135deg, rgba(15, 10, 35, 0.8), rgba(20, 15, 40, 0.8))',
            border: '1px solid rgba(245, 158, 11, 0.25)',
            boxShadow: '0 4px 24px rgba(0, 0, 0, 0.3), 0 0 20px rgba(245, 158, 11, 0.06)',
          }}
        >
          <Search className="w-5 h-5 text-amber-400 flex-shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => !hasResults && setShowHistory(true)}
            placeholder="Ask anything... e.g. &quot;latest crypto news&quot;"
            className="flex-1 bg-transparent text-white text-sm placeholder-gray-500 outline-none"
          />
          {query && (
            <button
              type="button"
              onClick={clearSearch}
              className="p-1 rounded-full hover:bg-white/10 transition-colors"
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
          )}
          {searching ? (
            <button
              type="button"
              onClick={handleStop}
              className="px-4 py-2 rounded-xl text-sm font-semibold transition-all"
              style={{
                background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1))',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                color: '#ef4444',
              }}
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-4 py-2 rounded-xl text-sm font-semibold transition-all disabled:opacity-40"
              style={{
                background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1))',
                border: '1px solid rgba(245, 158, 11, 0.3)',
                color: '#f59e0b',
              }}
            >
              Search
            </button>
          )}
        </div>

        {/* Recency Filter */}
        <div className="flex items-center gap-2 mt-3 px-2">
          <CalendarDays className="w-3.5 h-3.5 text-gray-600" />
          {RECENCY_OPTIONS.map(opt => (
            <button
              key={opt.value}
              type="button"
              onClick={() => setRecency(opt.value)}
              className="px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all"
              style={{
                background: recency === opt.value
                  ? 'rgba(245, 158, 11, 0.15)'
                  : 'rgba(255, 255, 255, 0.03)',
                border: recency === opt.value
                  ? '1px solid rgba(245, 158, 11, 0.3)'
                  : '1px solid rgba(255, 255, 255, 0.06)',
                color: recency === opt.value ? '#f59e0b' : '#6b7280',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </motion.form>

      {/* Loading State */}
      {loading && !streamedText && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center justify-center py-16"
        >
          <div className="text-center space-y-3">
            <div className="flex items-center justify-center gap-2">
              <Loader2 className="w-5 h-5 text-amber-400 animate-spin" />
              <span className="text-sm text-gray-400">Searching the web...</span>
            </div>
            <div className="flex gap-1 justify-center">
              {[0, 1, 2].map(i => (
                <motion.div
                  key={i}
                  className="w-1.5 h-1.5 rounded-full bg-amber-400/40"
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ repeat: Infinity, duration: 1.2, delay: i * 0.2 }}
                />
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Quick Suggestions (no results yet) */}
      {!hasResults && !loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="space-y-5 mt-6"
        >
          {/* Suggestions */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 justify-center">
              <Sparkles className="w-3.5 h-3.5 text-amber-400/60" />
              <span className="text-[11px] text-gray-600 font-semibold uppercase tracking-wider">
                Try searching
              </span>
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              {QUICK_SUGGESTIONS.map(suggestion => (
                <motion.button
                  key={suggestion}
                  whileHover={{ scale: 1.03, y: -1 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={() => { setQuery(suggestion); handleSearch(suggestion); }}
                  className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs transition-all"
                  style={{
                    background: 'linear-gradient(135deg, rgba(15, 10, 35, 0.6), rgba(20, 15, 40, 0.6))',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                    color: '#9ca3af',
                  }}
                >
                  <ChevronRight className="w-3 h-3 text-amber-400/50" />
                  {suggestion}
                </motion.button>
              ))}
            </div>
          </div>

          {/* Search History */}
          {(showHistory || !query) && history.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4"
            >
              <div className="flex items-center justify-between mb-2 px-1">
                <span className="text-[11px] text-gray-600 flex items-center gap-1.5 font-medium">
                  <History className="w-3 h-3" />
                  Recent Searches
                </span>
                <button
                  onClick={() => {
                    setHistory([]);
                    localStorage.removeItem('webSearchHistory');
                  }}
                  className="text-[10px] text-gray-700 hover:text-gray-400 transition-colors"
                >
                  Clear
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {history.slice(0, 8).map((h, i) => (
                  <motion.button
                    key={`${h.query}-${i}`}
                    whileHover={{ scale: 1.02 }}
                    onClick={() => { setQuery(h.query); handleSearch(h.query); }}
                    className="px-3 py-1.5 rounded-lg text-xs text-gray-500 hover:text-gray-300 transition-colors"
                    style={{
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid rgba(255, 255, 255, 0.06)',
                    }}
                  >
                    {h.query.length > 30 ? h.query.substring(0, 30) + '...' : h.query}
                  </motion.button>
                ))}
              </div>
            </motion.div>
          )}
        </motion.div>
      )}

      {/* Results Area */}
      <AnimatePresence>
        {hasResults && (
          <motion.div
            ref={resultRef}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            {/* Source Cards — shown FIRST, immediately when DDG results arrive */}
            {sources.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-2"
              >
                <div className="flex items-center gap-2 px-1">
                  <BookOpen className="w-3.5 h-3.5 text-gray-500" />
                  <span className="text-[11px] text-gray-500 font-semibold uppercase tracking-wider">
                    Sources ({sources.length})
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {sources.map((source, i) => (
                    <motion.a
                      key={`${source.url}-${i}`}
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      initial={{ opacity: 0, y: 5 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className="group flex gap-3 p-3 rounded-xl transition-all hover:border-amber-400/20"
                      style={{
                        background: 'linear-gradient(135deg, rgba(15, 10, 35, 0.5), rgba(20, 15, 40, 0.5))',
                        border: '1px solid rgba(255, 255, 255, 0.06)',
                      }}
                    >
                      <img
                        src={getFaviconUrl(source.url)}
                        alt=""
                        className="w-5 h-5 rounded mt-0.5 flex-shrink-0"
                        onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-1">
                          <h4 className="text-xs font-semibold text-white truncate group-hover:text-amber-300 transition-colors">
                            {source.title || getDomain(source.url)}
                          </h4>
                          <ExternalLink className="w-3 h-3 text-gray-600 group-hover:text-amber-400 flex-shrink-0 mt-0.5 transition-colors" />
                        </div>
                        <p className="text-[10px] text-amber-400/60 truncate mt-0.5">
                          {getDomain(source.url)}
                        </p>
                        {source.snippet && (
                          <p className="text-[11px] text-gray-500 mt-1 line-clamp-2 leading-relaxed">
                            {source.snippet}
                          </p>
                        )}
                      </div>
                    </motion.a>
                  ))}
                </div>
              </motion.div>
            )}

            {/* "Generating summary" indicator — shown after sources arrive, before AI tokens */}
            {summarizing && !streamedText && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-2 px-2 py-3"
              >
                <Loader2 className="w-4 h-4 text-amber-400 animate-spin" />
                <span className="text-sm text-gray-400">Generating AI summary...</span>
                <div className="flex gap-1 ml-1">
                  {[0, 1, 2].map(i => (
                    <motion.div
                      key={i}
                      className="w-1 h-1 rounded-full bg-amber-400/40"
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ repeat: Infinity, duration: 1.2, delay: i * 0.2 }}
                    />
                  ))}
                </div>
              </motion.div>
            )}

            {/* AI Response — streams in after sources are already visible */}
            {streamedText && (
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-xl p-5 backdrop-blur-xl"
                style={{
                  background: 'linear-gradient(135deg, rgba(15, 10, 35, 0.7), rgba(20, 15, 40, 0.7))',
                  border: '1px solid rgba(245, 158, 11, 0.15)',
                }}
              >
                <div className="flex items-center gap-2 mb-3">
                  <Sparkles className="w-4 h-4 text-amber-400" />
                  <span className="text-xs font-semibold text-amber-400/80 uppercase tracking-wider">
                    AI Summary
                  </span>
                  {searching && (
                    <Loader2 className="w-3 h-3 text-amber-400/60 animate-spin ml-1" />
                  )}
                </div>

                <div className="prose prose-sm prose-invert max-w-none
                  prose-p:text-gray-300 prose-p:leading-relaxed prose-p:text-sm
                  prose-headings:text-white prose-headings:font-semibold
                  prose-a:text-amber-400 prose-a:no-underline hover:prose-a:underline
                  prose-strong:text-white
                  prose-code:text-amber-300 prose-code:bg-amber-400/10 prose-code:px-1 prose-code:py-0.5 prose-code:rounded
                  prose-li:text-gray-300 prose-li:text-sm
                  prose-blockquote:border-amber-400/30 prose-blockquote:text-gray-400
                ">
                  <ReactMarkdown>{streamedText}</ReactMarkdown>
                </div>

                {/* Streaming cursor */}
                {searching && (
                  <motion.span
                    className="inline-block w-2 h-4 bg-amber-400/60 rounded-sm ml-0.5"
                    animate={{ opacity: [1, 0] }}
                    transition={{ repeat: Infinity, duration: 0.8 }}
                  />
                )}
              </motion.div>
            )}

            {/* New Search button */}
            {!searching && hasResults && (
              <div className="flex justify-center pt-2">
                <button
                  onClick={clearSearch}
                  className="text-xs text-gray-500 hover:text-amber-400 transition-colors flex items-center gap-1.5"
                >
                  <Search className="w-3 h-3" />
                  New search
                </button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
