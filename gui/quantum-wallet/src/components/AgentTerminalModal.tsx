// Agent Terminal modal — preview of the upcoming xterm.js + WASM MCP integration.
// For now: shows what's coming + a faux-terminal demo of the planned UX.
// Full impl tracked in feature backlog (Niveau 1: ~4-6 weeks per spec).

import { motion, AnimatePresence } from 'framer-motion';
import { X, Code, Terminal, Sparkles } from 'lucide-react';
import { useState, useEffect } from 'react';

interface AgentTerminalModalProps {
  isOpen: boolean;
  onClose: () => void;
  walletAddress?: string;
}

const DEMO_LINES = [
  { type: 'prompt', text: '$ q-agent wallet status' },
  { type: 'output', text: '🔑 qnk7154929a6aa0c118791373ea21004aca6e494e6e031c36f780cd5acedf031ccb' },
  { type: 'output', text: '   Balance: 352.4 QUG · Mining: 4.2 MH/s' },
  { type: 'prompt', text: '$ q-agent qcredit lock 1.0 platinum' },
  { type: 'output', text: '🔒 Locked 1.0 QUG into Platinum (180 days, 25% APY)' },
  { type: 'output', text: '   Position: pid-7f2c | Unlocks 2026-11-19' },
  { type: 'prompt', text: '$ q-agent twitter draft "AFL-1 just shipped"' },
  { type: 'output', text: '📝 Draft scored: engagement 0.81 · neg-risk 0.04 ✓' },
  { type: 'prompt', text: '$ q-agent qshare premium-ratio' },
  { type: 'output', text: '📊 NAV: 1.0234 QUG · Market: 1.487 QUG · ratio: 1.452 (below mint threshold)' },
];

export default function AgentTerminalModal({ isOpen, onClose, walletAddress }: AgentTerminalModalProps) {
  const [visibleLines, setVisibleLines] = useState(0);

  useEffect(() => {
    if (!isOpen) {
      setVisibleLines(0);
      return;
    }
    const id = setInterval(() => {
      setVisibleLines((n) => (n < DEMO_LINES.length ? n + 1 : n));
    }, 350);
    return () => clearInterval(id);
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="bg-slate-900 border border-cyan-500/30 rounded-2xl shadow-2xl w-[720px] max-w-[95vw] max-h-[90vh] overflow-hidden flex flex-col"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-cyan-500/20 bg-cyan-500/5">
              <div className="flex items-center gap-2">
                <Terminal className="w-5 h-5 text-cyan-400" />
                <h2 className="text-lg font-bold text-cyan-200">Agent Terminal</h2>
                <span className="text-xs px-2 py-0.5 bg-amber-500/20 text-amber-300 rounded">Coming Soon</span>
              </div>
              <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              <div className="flex items-start gap-3 p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
                <Sparkles className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-semibold text-cyan-200 mb-1">Claude Code MCP in your browser tab.</div>
                  <p className="text-xs text-slate-400">
                    Zero-install agent terminal. xterm.js + WASM-compiled MCP servers running locally in this browser.
                    Bring your Anthropic API key, connect to Quillon Graph, transact directly.
                  </p>
                </div>
              </div>

              <div className="font-mono text-xs bg-slate-950 rounded-lg p-4 border border-slate-800 overflow-x-auto">
                {DEMO_LINES.slice(0, visibleLines).map((line, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className={line.type === 'prompt' ? 'text-cyan-300' : 'text-slate-300'}
                  >
                    {line.text}
                  </motion.div>
                ))}
                {visibleLines >= DEMO_LINES.length && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-cyan-300 inline-flex items-center"
                  >
                    $ <span className="inline-block w-2 h-3.5 bg-cyan-400 ml-1 animate-pulse" />
                  </motion.div>
                )}
              </div>

              <div className="text-xs space-y-2 text-slate-400">
                <p>
                  <span className="font-semibold text-slate-200">What's planned:</span> xterm.js terminal UI, WASM-compiled
                  quillon-wallet-mcp + quillon-twitter-mcp, WebSocket connection to Agent Fiber Lane (PR #87), WebGPU mining
                  integration, Anthropic Claude inference via user-provided API key.
                </p>
                <p>
                  <span className="font-semibold text-slate-200">Timeline:</span> 4-6 weeks once PR #87 (AFL-1) and PR #34
                  (self-healing peer registry) merge. Spec at <code className="bg-slate-800 px-1 rounded">docs/agent-terminal-spec.md</code> (forthcoming).
                </p>
              </div>

              {walletAddress && (
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <div className="text-xs text-slate-500 mb-1">Will run as wallet:</div>
                  <div className="font-mono text-xs text-cyan-300 truncate">{walletAddress}</div>
                </div>
              )}

              <button
                disabled
                className="w-full py-3 bg-slate-800 text-slate-500 rounded-lg font-medium cursor-not-allowed flex items-center justify-center gap-2"
                title="Phase 2 — coming after PR #87 + #34 merge"
              >
                <Code className="w-4 h-4" />
                Launch Terminal (coming soon)
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
