import { useState, useRef, useEffect } from 'react';
import { Download, ChevronDown, Server, Pickaxe, Bot, Copy, Check, ExternalLink, ArrowRight } from 'lucide-react';

// v10.11.59: Global topbar download menu — node, miner, wallet MCP, with copyable
// wget commands + a link to the full downloads page. Self-contained.

const NODE_VERSION = 'v10.11.64';
const NODE_WGET = `wget https://quillon.xyz/downloads/q-api-server-${NODE_VERSION} && chmod +x q-api-server-${NODE_VERSION}`;
const MINER_WGET = 'wget https://quillon.xyz/downloads/q-miner-linux-x64 && chmod +x q-miner-linux-x64';
const MCP_INSTALL = 'curl -fsSL https://quillon.xyz/setup-ai.sh | bash';

function CopyRow({ cmd, accent }: { cmd: string; accent: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try { await navigator.clipboard.writeText(cmd); setCopied(true); setTimeout(() => setCopied(false), 1600); } catch { /* ignore */ }
  };
  return (
    <button
      onClick={copy}
      className={`w-full flex items-center justify-between gap-2 px-2 py-1.5 rounded-lg bg-black/40 border ${accent} hover:brightness-125 transition-all group`}
    >
      <code className="text-[10px] font-mono truncate text-slate-200/90">{cmd}</code>
      {copied
        ? <Check className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
        : <Copy className="w-3.5 h-3.5 text-slate-400 group-hover:text-white flex-shrink-0" />}
    </button>
  );
}

export default function DownloadMenu() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(o => !o)}
        title="Downloads — node, miner, AI wallet MCP"
        className="relative flex items-center gap-1.5 px-2 py-1 bg-amber-500/10 border border-amber-500/30 rounded-lg cursor-pointer hover:bg-amber-500/20 transition-colors"
      >
        <Download className="w-3.5 h-3.5 text-amber-400" />
        <span className="text-amber-300 text-xs font-medium">Download</span>
        <ChevronDown className={`w-3 h-3 text-amber-400/70 transition-transform ${open ? 'rotate-180' : ''}`} />
        <span className="absolute -top-1.5 -right-2 px-1.5 py-0.5 rounded-full text-[8px] font-extrabold leading-none bg-gradient-to-r from-cyan-400 to-emerald-400 text-emerald-950 shadow-[0_0_8px_#22d3ee] animate-pulse pointer-events-none select-none">v64</span>
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-80 z-[80] rounded-2xl border border-amber-500/25 bg-[#140f1e]/95 backdrop-blur-xl shadow-2xl shadow-black/50 overflow-hidden">
          <div className="px-3 py-2 border-b border-amber-500/15 flex items-center justify-between">
            <span className="text-amber-300/90 text-xs font-semibold tracking-wide">DOWNLOADS</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300">node {NODE_VERSION}</span>
          </div>

          {/* Node */}
          <div className="px-3 py-3 border-b border-amber-500/10">
            <div className="flex items-start gap-3 mb-2">
              <div className="mt-0.5 w-8 h-8 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center flex-shrink-0">
                <Server className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="min-w-0">
                <div className="text-slate-100 text-sm font-medium">Quillon Node {NODE_VERSION}</div>
                <div className="text-slate-400 text-[11px] leading-snug">Standalone q-api-server — run a full node + miner.</div>
              </div>
            </div>
            <CopyRow cmd={NODE_WGET} accent="border-emerald-500/25" />
          </div>

          {/* Miner */}
          <div className="px-3 py-3 border-b border-amber-500/10">
            <div className="flex items-start gap-3 mb-2">
              <div className="mt-0.5 w-8 h-8 rounded-lg bg-orange-500/15 border border-orange-500/30 flex items-center justify-center flex-shrink-0">
                <Pickaxe className="w-4 h-4 text-orange-400" />
              </div>
              <div className="min-w-0">
                <div className="text-slate-100 text-sm font-medium">q-miner (standalone)</div>
                <div className="text-slate-400 text-[11px] leading-snug">CPU/GPU miner — point it at quillon.xyz, earn QUG.</div>
              </div>
            </div>
            <CopyRow cmd={MINER_WGET} accent="border-orange-500/25" />
          </div>

          {/* MCP */}
          <div className="px-3 py-3">
            <div className="flex items-start gap-3 mb-2">
              <div className="mt-0.5 w-8 h-8 rounded-lg bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center flex-shrink-0">
                <Bot className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="min-w-0">
                <div className="text-slate-100 text-sm font-medium">Quillon Wallet MCP</div>
                <div className="text-slate-400 text-[11px] leading-snug">AI-agent wallet for Claude Code, Cursor, Codex, Qwen.</div>
              </div>
            </div>
            <CopyRow cmd={MCP_INSTALL} accent="border-cyan-500/25" />
          </div>

          {/* Full page link */}
          <a
            href="/downloads.html"
            className="flex items-center justify-center gap-2 px-3 py-2.5 border-t border-amber-500/15 bg-amber-500/5 hover:bg-amber-500/15 transition-colors text-amber-300 text-xs font-medium"
          >
            All downloads, checksums &amp; setup guide
            <ArrowRight className="w-3.5 h-3.5" />
          </a>
        </div>
      )}
    </div>
  );
}
