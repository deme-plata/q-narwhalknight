// BlockStreamBar — wicked-cool block-counter progress bar for GlobalTopBar.
//
// What it does
// ------------
// A horizontal "stream" of glowing cube glyphs flowing right-to-left, each
// cube representing a recent block. The newest block is at the right edge,
// pulses with a chromatic ring, and the height counter glitches forward
// when a new block lands.
//
// Visual layers (all pure SVG + CSS, no library, GPU-friendly):
//   1. Background void — radial-gradient deep indigo → near-black.
//   2. Constellation — tiny static stars at fixed positions (decorative).
//   3. DAG threads — three faint cubic-bezier curves drifting left, the
//      "lattice" that blocks float over.
//   4. Block cubes — `BLOCK_LANE_SIZE` cube glyphs, each translateX'd by
//      its age. Newer blocks: brighter + larger. The newest pulses.
//   5. Pulse ring — SVG circle around the newest cube, scales 1→2 on each
//      new-block event, fading.
//   6. Height readout — monospace numeric, color-shifts on increment via
//      a 600ms "glitch" CSS animation (chromatic-aberration tone).
//   7. Sparkline — a tiny instantaneous block-time deviation bar at the
//      bottom edge (shows the rolling 6-block tempo as a thin gradient
//      micro-graph).
//
// Polling
// -------
// Lightweight `fetch('/api/v1/status')` every 1500 ms. When height
// increases, push a new cube into the lane and fire the pulse animation.
// Falls back gracefully if the endpoint is unreachable (renders a muted
// "?" state with no animation).

import { useEffect, useRef, useState, useMemo } from 'react';

const POLL_INTERVAL_MS = 1500;
const BLOCK_LANE_SIZE = 14; // how many recent blocks we visualize in the cube stream
const SPARKLINE_WINDOW = 6;

interface BlockEvent {
  height: number;
  ts: number; // wall-clock ms when observed
}

interface BlockStreamBarProps {
  className?: string;
  compact?: boolean; // smaller variant for tight nav bars
}

export default function BlockStreamBar({ className = '', compact = false }: BlockStreamBarProps) {
  const [height, setHeight] = useState<number | null>(null);
  const [history, setHistory] = useState<BlockEvent[]>([]);
  const [glitchKey, setGlitchKey] = useState(0); // bumps to re-trigger CSS animation
  const [pulseKey, setPulseKey] = useState(0);
  const [reachable, setReachable] = useState(true);
  const lastSeenRef = useRef<number | null>(null);

  // Live polling of /api/v1/status — q-flux serves from same origin.
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await fetch('/api/v1/status', { signal: AbortSignal.timeout(2500) });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = await r.json();
        const h: number =
          j?.data?.upgrades?.current_height ??
          j?.data?.current_height ??
          j?.current_height ??
          0;
        if (cancelled) return;
        setReachable(true);
        if (lastSeenRef.current === null || h > lastSeenRef.current) {
          const now = Date.now();
          setHistory((prev) => {
            const next = [...prev, { height: h, ts: now }];
            return next.slice(-BLOCK_LANE_SIZE);
          });
          setHeight(h);
          setGlitchKey((k) => k + 1);
          setPulseKey((k) => k + 1);
          lastSeenRef.current = h;
        }
      } catch {
        if (!cancelled) setReachable(false);
      }
    };
    tick();
    const id = setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Rolling block-time tempo for sparkline. Smaller delta = denser, redshift.
  const sparkBars = useMemo(() => {
    const recent = history.slice(-SPARKLINE_WINDOW);
    if (recent.length < 2) return [] as number[];
    const deltas: number[] = [];
    for (let i = 1; i < recent.length; i++) {
      deltas.push(recent[i].ts - recent[i - 1].ts);
    }
    const max = Math.max(...deltas, 1);
    return deltas.map((d) => 1 - d / (max * 1.4));
  }, [history]);

  const width = compact ? 220 : 280;
  const lane_y = compact ? 16 : 18;
  const cube_size = compact ? 8 : 10;
  const cube_gap = compact ? 14 : 16;

  // Background stars (deterministic positions so they don't twinkle randomly each render).
  const stars = useMemo(() => {
    const seed = 7919; // prime
    const arr: { x: number; y: number; r: number }[] = [];
    let s = seed;
    for (let i = 0; i < 24; i++) {
      s = (s * 9301 + 49297) % 233280;
      const x = ((s / 233280) * width) | 0;
      s = (s * 9301 + 49297) % 233280;
      const y = ((s / 233280) * 32) | 0;
      s = (s * 9301 + 49297) % 233280;
      const r = (s / 233280) * 0.9 + 0.2;
      arr.push({ x, y, r });
    }
    return arr;
  }, [width]);

  return (
    <div
      className={`relative inline-flex items-center select-none ${className}`}
      title={
        reachable
          ? `Block height ${height ?? '—'} · live (1.5s poll)`
          : 'Status endpoint unreachable'
      }
    >
      <style>{`
        @keyframes bs-glitch {
          0%   { transform: translateX(0)    skewX(0deg);   filter: hue-rotate(0deg)   drop-shadow(0 0 4px #ffd76b); }
          20%  { transform: translateX(-1px) skewX(-1deg);  filter: hue-rotate(20deg)  drop-shadow(0 0 8px #ffeaa1); }
          40%  { transform: translateX(2px)  skewX(2deg);   filter: hue-rotate(-15deg) drop-shadow(0 0 10px #d8a8ff); }
          60%  { transform: translateX(-1px) skewX(0deg);   filter: hue-rotate(8deg)   drop-shadow(0 0 6px #9ed8ff); }
          100% { transform: translateX(0)    skewX(0deg);   filter: hue-rotate(0deg)   drop-shadow(0 0 3px #ffd76b); }
        }
        @keyframes bs-pulse-ring {
          0%   { r: 4;  opacity: 0.95; stroke-width: 1.2; }
          70%  { r: 12; opacity: 0;    stroke-width: 0.4; }
          100% { r: 12; opacity: 0;    stroke-width: 0.4; }
        }
        @keyframes bs-thread-drift {
          0%   { stroke-dashoffset: 0; }
          100% { stroke-dashoffset: -120; }
        }
        @keyframes bs-star-twinkle {
          0%, 100% { opacity: 0.35; }
          50%      { opacity: 0.85; }
        }
        .bs-glitch-text { animation: bs-glitch 0.55s ease-out 1; }
        .bs-pulse       { animation: bs-pulse-ring 1.2s ease-out 1; }
        .bs-thread      { animation: bs-thread-drift 6s linear infinite; }
      `}</style>

      <svg
        width={width}
        height={compact ? 28 : 34}
        viewBox={`0 0 ${width} ${compact ? 28 : 34}`}
        xmlns="http://www.w3.org/2000/svg"
        className="block"
        aria-label={`live block height ${height ?? 'unknown'}`}
      >
        <defs>
          <linearGradient id="bs-bg" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%"  stopColor="#0a0820" />
            <stop offset="60%" stopColor="#150a2e" />
            <stop offset="100%" stopColor="#080510" />
          </linearGradient>
          <linearGradient id="bs-cube" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0%"  stopColor="#ffd76b" />
            <stop offset="50%" stopColor="#ff9b6b" />
            <stop offset="100%" stopColor="#a978ff" />
          </linearGradient>
          <linearGradient id="bs-cube-old" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0%"  stopColor="#5a4f93" />
            <stop offset="100%" stopColor="#241c47" />
          </linearGradient>
          <linearGradient id="bs-thread" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%"   stopColor="rgba(168,134,255,0)" />
            <stop offset="50%"  stopColor="rgba(168,134,255,0.55)" />
            <stop offset="100%" stopColor="rgba(168,134,255,0)" />
          </linearGradient>
          <radialGradient id="bs-glow" cx="0.5" cy="0.5" r="0.6">
            <stop offset="0%"   stopColor="rgba(255,215,107,0.55)" />
            <stop offset="100%" stopColor="rgba(255,215,107,0)" />
          </radialGradient>
        </defs>

        {/* (1) Background void */}
        <rect x="0" y="0" width={width} height="100%" rx="8" fill="url(#bs-bg)" />

        {/* (2) Constellation */}
        {stars.map((s, i) => (
          <circle
            key={`s-${i}`}
            cx={s.x}
            cy={s.y}
            r={s.r}
            fill="#ffffff"
            opacity="0.55"
            style={{ animation: `bs-star-twinkle ${3 + (i % 4)}s ease-in-out infinite`, animationDelay: `${i * 0.17}s` }}
          />
        ))}

        {/* (3) Three drifting DAG threads */}
        <path
          d={`M 0 ${lane_y - 4} Q ${width * 0.3} ${lane_y - 12}, ${width * 0.55} ${lane_y - 5} T ${width} ${lane_y - 3}`}
          stroke="url(#bs-thread)"
          strokeWidth="0.8"
          strokeDasharray="3 5"
          fill="none"
          className="bs-thread"
        />
        <path
          d={`M 0 ${lane_y + 6} Q ${width * 0.4} ${lane_y - 2}, ${width * 0.7} ${lane_y + 8} T ${width} ${lane_y + 5}`}
          stroke="url(#bs-thread)"
          strokeWidth="0.7"
          strokeDasharray="2 6"
          strokeDashoffset="-40"
          fill="none"
          className="bs-thread"
        />
        <path
          d={`M 0 ${lane_y + 12} Q ${width * 0.5} ${lane_y + 4}, ${width * 0.8} ${lane_y + 14} T ${width} ${lane_y + 11}`}
          stroke="url(#bs-thread)"
          strokeWidth="0.6"
          strokeDasharray="4 4"
          strokeDashoffset="-80"
          fill="none"
          className="bs-thread"
        />

        {/* (4) Cube stream — newest at right edge */}
        {history.slice(-BLOCK_LANE_SIZE).map((b, idx, arr) => {
          const age = arr.length - 1 - idx; // 0 = newest
          const x = width - 12 - age * cube_gap;
          if (x < 0) return null;
          const isHead = age === 0;
          const size = isHead ? cube_size + 2 : Math.max(2, cube_size - age * 0.35);
          return (
            <g key={`cube-${b.height}-${b.ts}`} transform={`translate(${x - size / 2}, ${lane_y - size / 2})`}>
              {isHead && (
                <circle
                  cx={size / 2}
                  cy={size / 2}
                  r="14"
                  fill="url(#bs-glow)"
                />
              )}
              <rect
                width={size}
                height={size}
                rx={1}
                fill={isHead ? 'url(#bs-cube)' : 'url(#bs-cube-old)'}
                opacity={isHead ? 1 : Math.max(0.15, 1 - age * 0.08)}
                style={{
                  filter: isHead ? 'drop-shadow(0 0 4px #ffd76b)' : 'none',
                }}
              />
            </g>
          );
        })}

        {/* (5) Pulse ring around the newest cube */}
        {history.length > 0 && (
          <circle
            key={`pulse-${pulseKey}`}
            cx={width - 12}
            cy={lane_y}
            r="4"
            stroke="#ffd76b"
            fill="none"
            className="bs-pulse"
          />
        )}

        {/* (6) Height readout (centered/left) */}
        <g transform={`translate(8, ${compact ? 22 : 26})`}>
          <text
            key={`h-${glitchKey}`}
            fontFamily="ui-monospace, SFMono-Regular, monospace"
            fontSize={compact ? 11 : 13}
            fontWeight="700"
            fill={reachable ? '#ffd76b' : '#666'}
            className={reachable ? 'bs-glitch-text' : ''}
          >
            {reachable ? `#${(height ?? 0).toLocaleString()}` : '#?'}
          </text>
          <text
            x={compact ? 70 : 90}
            fontFamily="ui-monospace, SFMono-Regular, monospace"
            fontSize={compact ? 8 : 9}
            fill="rgba(168,134,255,0.65)"
          >
            {reachable ? 'BLOCK STREAM · LIVE' : 'LINK DOWN'}
          </text>
        </g>

        {/* (7) Tempo sparkline (bottom edge) */}
        {sparkBars.length > 0 && (
          <g transform={`translate(${width - sparkBars.length * 4 - 4}, ${(compact ? 28 : 34) - 3})`}>
            {sparkBars.map((v, i) => (
              <rect
                key={`spark-${i}`}
                x={i * 4}
                y={-Math.max(1, v * 3)}
                width="3"
                height={Math.max(1, v * 3)}
                fill={v > 0.6 ? '#ff9b6b' : v > 0.3 ? '#ffd76b' : '#a978ff'}
                opacity="0.75"
              />
            ))}
          </g>
        )}
      </svg>
    </div>
  );
}
