// MemoRender — shared component for transaction-memo display.
//
// What it does
// ------------
// Renders a memo string in a way that honors its content:
//   • Plain ASCII memos with no structure  → normal flowing text.
//   • Rich memos (multi-line, contains box-drawing chars, math symbols,
//     monospace structure) → monospaced, white-space: pre-wrap, with a
//     subtle "rich" badge so the viewer knows formatting is preserved.
//   • Always: autolink http(s) URLs; lightweight markdown for inline
//     emphasis (**bold**, *italic*, `code`); preserve newlines.
//
// Two modes
// ---------
//   • `mode="compact"`  — single-line truncated, italic, for list rows.
//   • `mode="full"`     — full memo in a styled card, multi-line capable.
//
// Why
// ---
// Memos like "v10.11.0 ships: integrity-root endpoint live; …" or the
// LaTeX-quality "╭───── ⟨ Claude → Viktor ⟩ ─────╮ … ∀ b ∈ Quillon …"
// were collapsing into a single break-words paragraph that destroyed
// alignment. This component is the upgrade — gives memos the parchment-
// quality rendering they deserve.

import { useMemo } from 'react';
import { MessageSquare, Sparkles } from 'lucide-react';

interface MemoRenderProps {
  memo: string;
  mode?: 'compact' | 'full';
  className?: string;
  /** In compact mode, truncate after this many chars (default 140). */
  maxCompactChars?: number;
}

// Detect whether a memo has "rich" intent. Heuristics:
//   - contains a newline                                  → rich
//   - contains box-drawing or block-element Unicode chars → rich
//   - contains a mathematical operator (set/logic/calc)   → rich
//   - has 2+ space runs that look like ascii-art alignment → rich
function isRich(memo: string): boolean {
  if (memo.includes('\n')) return true;
  // Box-drawing: U+2500-257F; Block elements: U+2580-259F
  if (/[─-▟]/.test(memo)) return true;
  // Math operators + arrows: U+2190-21FF (arrows), U+2200-22FF (math operators),
  // U+27F0-27FF (supplemental arrows-A), U+2900-297F (supplemental arrows-B)
  if (/[←-⇿∀-⋿⟰-⟿⤀-⥿]/.test(memo)) return true;
  // Multiple-space alignment (3+ spaces in a row)
  if (/ {3,}/.test(memo)) return true;
  return false;
}

// Very small markdown subset:
//   **bold**  → <strong>
//   *italic*  → <em>
//   `code`    → <code>
//   https://… → <a>
// All applied without regex catastrophe-backtracking via plain split logic.
function renderInline(text: string, keyPrefix: string): React.ReactNode[] {
  // Tokenize URL boundaries first.
  const urlRegex = /(https?:\/\/[^\s<>"']+)/g;
  const parts: React.ReactNode[] = [];
  let lastIdx = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = urlRegex.exec(text)) !== null) {
    if (m.index > lastIdx) {
      parts.push(<span key={`${keyPrefix}-t${i++}`}>{renderEmphasis(text.slice(lastIdx, m.index), `${keyPrefix}-e${i}`)}</span>);
    }
    parts.push(
      <a
        key={`${keyPrefix}-u${i++}`}
        href={m[0]}
        target="_blank"
        rel="noopener noreferrer"
        className="text-amber-300 underline decoration-amber-500/40 hover:decoration-amber-300 break-all"
      >
        {m[0]}
      </a>
    );
    lastIdx = m.index + m[0].length;
  }
  if (lastIdx < text.length) {
    parts.push(<span key={`${keyPrefix}-t${i++}`}>{renderEmphasis(text.slice(lastIdx), `${keyPrefix}-e${i}`)}</span>);
  }
  return parts;
}

// Recognize **bold**, *italic*, `code` inside a URL-free chunk.
function renderEmphasis(text: string, keyPrefix: string): React.ReactNode[] {
  // Tokenize by **, *, ` — keep them as boundary markers.
  // Pattern matches: **...** | *...* | `...` (non-greedy)
  const pattern = /(\*\*[^*\n]+\*\*|\*[^*\n]+\*|`[^`\n]+`)/g;
  const out: React.ReactNode[] = [];
  let lastIdx = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = pattern.exec(text)) !== null) {
    if (m.index > lastIdx) out.push(text.slice(lastIdx, m.index));
    const tok = m[0];
    if (tok.startsWith('**') && tok.endsWith('**')) {
      out.push(<strong key={`${keyPrefix}-b${i++}`} className="font-bold text-amber-50">{tok.slice(2, -2)}</strong>);
    } else if (tok.startsWith('`') && tok.endsWith('`')) {
      out.push(<code key={`${keyPrefix}-c${i++}`} className="font-mono text-[0.92em] bg-amber-500/10 px-1 rounded text-amber-200">{tok.slice(1, -1)}</code>);
    } else if (tok.startsWith('*') && tok.endsWith('*')) {
      out.push(<em key={`${keyPrefix}-i${i++}`} className="italic text-amber-200">{tok.slice(1, -1)}</em>);
    } else {
      out.push(tok);
    }
    lastIdx = m.index + tok.length;
  }
  if (lastIdx < text.length) out.push(text.slice(lastIdx));
  return out;
}

export default function MemoRender({ memo, mode = 'full', className = '', maxCompactChars = 140 }: MemoRenderProps) {
  const rich = useMemo(() => isRich(memo), [memo]);
  const lines = useMemo(() => memo.split('\n'), [memo]);

  if (mode === 'compact') {
    // Single-line, no preserved formatting — but if rich, hint it.
    const flat = memo.replace(/\n+/g, ' ⇢ ').replace(/\s+/g, ' ').trim();
    const display = flat.length > maxCompactChars ? `${flat.slice(0, maxCompactChars)}…` : flat;
    return (
      <div
        className={`flex items-start gap-1.5 pl-2 pr-1.5 py-1 rounded-md ${className}`}
        style={{
          background: 'rgba(212,175,55,0.05)',
          border: '1px solid rgba(212,175,55,0.12)',
        }}
        title={memo}
      >
        {rich ? (
          <Sparkles className="w-3 h-3 text-amber-300/80 flex-shrink-0 mt-[1px]" />
        ) : (
          <MessageSquare className="w-3 h-3 text-amber-300/70 flex-shrink-0 mt-[1px]" />
        )}
        <span className="text-[10.5px] italic text-amber-100/85 leading-tight break-words">{display}</span>
      </div>
    );
  }

  // Full mode — parchment card.
  return (
    <div
      className={`rounded-xl p-4 ${className}`}
      style={{
        background: rich
          ? 'linear-gradient(135deg, rgba(40,28,80,0.55), rgba(28,18,48,0.55))'
          : 'linear-gradient(135deg, rgba(212,175,55,0.10), rgba(255,215,0,0.06))',
        border: rich ? '1px solid rgba(168,134,255,0.35)' : '1px solid rgba(212,175,55,0.3)',
        boxShadow: rich ? 'inset 0 0 24px rgba(168,134,255,0.06)' : undefined,
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {rich ? (
            <Sparkles className="w-4 h-4 text-amber-300" />
          ) : (
            <MessageSquare className="w-4 h-4 text-amber-400" />
          )}
          <span className="text-sm text-amber-200">Message</span>
          {rich && (
            <span className="text-[9.5px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-amber-500/15 border border-amber-500/30 text-amber-200/80">
              rich
            </span>
          )}
        </div>
        <span className="text-[9.5px] text-amber-300/40 font-mono">{memo.length} chars</span>
      </div>
      {rich ? (
        // Monospace, preserve every space + newline; allow horizontal scroll on
        // very wide lines so box-drawing alignment isn't broken by wrapping.
        <pre
          className="text-[12px] text-amber-100 leading-[1.45] font-mono whitespace-pre overflow-x-auto"
          style={{ fontVariantLigatures: 'none', tabSize: 2 }}
        >
          {lines.map((line, i) => (
            <span key={i}>
              {renderInline(line, `l${i}`)}
              {i < lines.length - 1 ? '\n' : null}
            </span>
          ))}
        </pre>
      ) : (
        <p className="text-sm text-amber-100 leading-relaxed break-words">
          {renderInline(memo, 'l0')}
        </p>
      )}
    </div>
  );
}
