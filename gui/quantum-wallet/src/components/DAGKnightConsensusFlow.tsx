interface Props {
  currentHeight: number;
}

/**
 * DAGKnightConsensusFlow — lightweight, fully-responsive SVG visualization of the
 * DAG-Knight block DAG. Replaces the old fixed-aspect canvas visualizer (which no
 * longer scaled with the layout). Pure SVG + CSS keyframes: no canvas, no rAF loop,
 * no three.js — it scales cleanly to any width and matches the glassmorphic design
 * of the QNO Oracle card beside it.
 */

// Deterministic DAG layout (viewBox 800×240): columns = consensus rounds, flowing L→R.
const COLS_X = [70, 190, 315, 440, 565, 690];
const ROWS_Y = [60, 120, 180];

// Which rows are populated per column (organic, non-uniform DAG shape).
const COL_ROWS: number[][] = [
  [1],        // genesis-ish single node
  [0, 2],
  [0, 1, 2],
  [1, 2],
  [0, 1, 2],
  [0, 1],     // frontier / tip
];

const nodes = COL_ROWS.flatMap((rows, c) =>
  rows.map((r) => ({ id: `${c}-${r}`, x: COLS_X[c], y: ROWS_Y[r], col: c, row: r }))
);

// Edges: every node points to 1–2 parents in the previous column (DAG = many parents).
const edges: { x1: number; y1: number; x2: number; y2: number; delay: number }[] = [];
for (let c = 1; c < COL_ROWS.length; c++) {
  for (const r of COL_ROWS[c]) {
    const parents = COL_ROWS[c - 1];
    // nearest parent + occasionally a second (cross-link) for the DAG braid look
    const sorted = [...parents].sort((a, b) => Math.abs(a - r) - Math.abs(b - r));
    const picks = sorted.slice(0, sorted.length > 1 && (r + c) % 2 === 0 ? 2 : 1);
    for (const p of picks) {
      edges.push({
        x1: COLS_X[c - 1], y1: ROWS_Y[p],
        x2: COLS_X[c], y2: ROWS_Y[r],
        delay: c * 0.35,
      });
    }
  }
}

export default function DAGKnightConsensusFlow({ currentHeight }: Props) {
  return (
    <div
      className="backdrop-blur-xl rounded-3xl overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.9) 0%, rgba(25, 25, 40, 0.9) 100%)',
        border: '2px solid rgba(139, 92, 246, 0.3)',
        boxShadow: '0 0 30px rgba(139, 92, 246, 0.1)',
      }}
    >
      <div className="p-4 border-b border-purple-500/20 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-purple-100 flex items-center gap-2">
            <span className="text-xl">⬡</span>
            DAG-Knight Consensus
          </h3>
          <p className="text-sm text-purple-300/60 mt-1">
            Deterministic block-DAG ordering · live frontier
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-purple-500/10 border border-purple-500/30 shrink-0">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-400" />
          </span>
          <span className="text-xs font-mono text-cyan-200">
            #{(currentHeight || 0).toLocaleString()}
          </span>
        </div>
      </div>

      <div className="relative p-4">
        <svg viewBox="0 0 800 240" className="w-full" style={{ height: 'auto' }} preserveAspectRatio="xMidYMid meet">
          <defs>
            <linearGradient id="dkNode" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#a78bfa" />
              <stop offset="100%" stopColor="#22d3ee" />
            </linearGradient>
            <linearGradient id="dkWave" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity="0" />
              <stop offset="50%" stopColor="#22d3ee" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
            </linearGradient>
            <filter id="dkGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="b" />
              <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>

          {/* Edges — faint static link + an animated dash that flows toward the frontier */}
          {edges.map((e, i) => (
            <g key={`e-${i}`}>
              <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke="rgba(139,92,246,0.22)" strokeWidth="1.5" />
              <line
                x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
                stroke="url(#dkWave)" strokeWidth="2.5" strokeLinecap="round"
                strokeDasharray="14 60" className="dk-flow"
                style={{ animationDelay: `${e.delay}s` }}
              />
            </g>
          ))}

          {/* Nodes — pulsing blocks; the last column (tip) glows */}
          {nodes.map((n, i) => {
            const isTip = n.col === COL_ROWS.length - 1;
            return (
              <g key={n.id} className="dk-node" style={{ animationDelay: `${i * 0.18}s`, transformOrigin: `${n.x}px ${n.y}px` }}>
                <circle cx={n.x} cy={n.y} r={isTip ? 11 : 9} fill="url(#dkNode)" filter={isTip ? 'url(#dkGlow)' : undefined} />
                <circle cx={n.x} cy={n.y} r={isTip ? 11 : 9} fill="none" stroke="#e9d5ff" strokeOpacity="0.35" strokeWidth="1" />
              </g>
            );
          })}

          {/* Finalization sweep — a soft vertical line drifting L→R */}
          <line x1="0" y1="12" x2="0" y2="228" stroke="rgba(34,211,238,0.35)" strokeWidth="2" className="dk-sweep" />
        </svg>

        <style>{`
          @keyframes dkFlow { to { stroke-dashoffset: -74; } }
          .dk-flow { animation: dkFlow 1.6s linear infinite; }
          @keyframes dkNodePulse { 0%,100% { opacity: .8; transform: scale(1); } 50% { opacity: 1; transform: scale(1.12); } }
          .dk-node { animation: dkNodePulse 2.6s ease-in-out infinite; }
          @keyframes dkSweep { 0% { transform: translateX(20px); opacity: 0; } 15% { opacity: .8; } 85% { opacity: .8; } 100% { transform: translateX(780px); opacity: 0; } }
          .dk-sweep { animation: dkSweep 4.5s ease-in-out infinite; }
          @media (prefers-reduced-motion: reduce) { .dk-flow, .dk-node, .dk-sweep { animation: none; } }
        `}</style>
      </div>
    </div>
  );
}
