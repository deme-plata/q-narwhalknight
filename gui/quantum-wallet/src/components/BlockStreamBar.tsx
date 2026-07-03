// BlockStreamBar — Claude-Code-compaction-style block height ticker.
//
// One thin horizontal track. The fill animates from 0% to 100% over the
// expected block-time window (~1s on Quillon Graph), then snaps back to 0%
// when a new block lands. Above it: monospace height + a quiet "LIVE" /
// "OFFLINE" status pill. No cubes, no stars, no SVG decoration — just
// signal-and-fill, the way Claude Code shows compaction progress.

import { useEffect, useRef, useState } from 'react';

const POLL_INTERVAL_MS = 1000;          // /api/v1/status fetch cadence
const EXPECTED_BLOCK_TIME_MS = 1000;    // chain target — 1 bps
const FILL_TICK_MS = 50;                // 20 fps fill animation

interface BlockStreamBarProps {
  className?: string;
  compact?: boolean;
}

export default function BlockStreamBar({ className = '', compact = false }: BlockStreamBarProps) {
  const [height, setHeight] = useState<number | null>(null);
  const [reachable, setReachable] = useState(true);
  const [fill, setFill] = useState(0);          // 0..1
  const [flashKey, setFlashKey] = useState(0);  // bumps to retrigger the snap-back animation
  const lastBlockTsRef = useRef<number>(Date.now());
  const lastSeenHeightRef = useRef<number | null>(null);

  // Status fetcher
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const r = await fetch('/api/v1/status', { signal: AbortSignal.timeout(2500) });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = await r.json();
        const h: number =
          j?.data?.current_height ??
          j?.data?.upgrades?.current_height ??
          j?.current_height ?? 0;
        if (cancelled) return;
        setReachable(true);
        if (lastSeenHeightRef.current === null || h > lastSeenHeightRef.current) {
          lastSeenHeightRef.current = h;
          lastBlockTsRef.current = Date.now();
          setHeight(h);
          setFlashKey((k) => k + 1);
        }
      } catch {
        if (!cancelled) setReachable(false);
      }
    };
    tick();
    const id = setInterval(tick, POLL_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // Fill animation tick — ratio of (time since last block) / expected block time, clamped to [0,1]
  useEffect(() => {
    const id = setInterval(() => {
      const elapsed = Date.now() - lastBlockTsRef.current;
      setFill(Math.min(1, elapsed / EXPECTED_BLOCK_TIME_MS));
    }, FILL_TICK_MS);
    return () => clearInterval(id);
  }, []);

  const width = compact ? 180 : 220;
  const barHeight = compact ? 4 : 5;
  const heightLabel = reachable && height !== null
    ? `#${height.toLocaleString()}`
    : reachable ? '#—' : 'OFFLINE';

  return (
    <div
      className={className}
      style={{
        width,
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
        fontFamily: 'ui-monospace, SFMono-Regular, "JetBrains Mono", monospace',
      }}
      aria-label={`live block height ${height ?? 'unknown'}`}
    >
      {/* Label row — height + status */}
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
        <span
          key={flashKey}
          style={{
            fontSize: compact ? 11 : 12,
            fontWeight: 600,
            color: reachable ? '#e9e7ff' : '#6b6480',
            letterSpacing: 0.2,
            animation: reachable && height !== null ? 'bsb-flash 600ms ease-out 1' : undefined,
          }}
        >
          {heightLabel}
        </span>
        <span
          style={{
            fontSize: compact ? 9 : 10,
            color: reachable ? 'rgba(168,134,255,0.55)' : 'rgba(220,80,80,0.7)',
            textTransform: 'uppercase',
            letterSpacing: 1.2,
          }}
        >
          {reachable ? 'live' : 'link down'}
        </span>
      </div>

      {/* The bar itself */}
      <div
        style={{
          position: 'relative',
          width: '100%',
          height: barHeight,
          borderRadius: barHeight / 2,
          background: 'rgba(168,134,255,0.10)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            position: 'absolute',
            inset: 0,
            width: `${(reachable ? fill : 0) * 100}%`,
            background: reachable
              ? 'linear-gradient(90deg, rgba(168,134,255,0.55), rgba(255,215,107,0.85))'
              : 'rgba(220,80,80,0.4)',
            transition: 'width 50ms linear',
            borderRadius: barHeight / 2,
          }}
        />
        {/* Subtle shimmer band that follows the leading edge while < 100% */}
        {reachable && fill < 1 && (
          <div
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: `calc(${fill * 100}% - 12px)`,
              width: 24,
              background: 'linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.35), rgba(255,255,255,0))',
              filter: 'blur(0.5px)',
              pointerEvents: 'none',
            }}
          />
        )}
      </div>

      <style>{`
        @keyframes bsb-flash {
          0%   { color: #ffd76b; text-shadow: 0 0 8px rgba(255,215,107,0.55); }
          100% { color: #e9e7ff; text-shadow: none; }
        }
      `}</style>
    </div>
  );
}
