// BlockStreamBar — Claude-Code-compaction-style block height ticker.
//
// One thin horizontal track. The fill animates from 0% to 100% over the
// expected block-time window (~1s on Quillon Graph), then snaps back to 0%
// when a new block lands. Above it: monospace height + a quiet "LIVE" /
// "OFFLINE" status pill. No cubes, no stars, no SVG decoration — just
// signal-and-fill, the way Claude Code shows compaction progress.

import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import QuantumChamberCanvas from './QuantumChamberCanvas';

const POLL_INTERVAL_MS = 1000;          // /api/v1/status fetch cadence
const EXPECTED_BLOCK_TIME_MS = 1000;    // chain target — 1 bps
const FILL_TICK_MS = 50;                // 20 fps fill animation
const CHAMBER_OPEN_DELAY_MS = 160;      // don't fire on a mouse merely passing over
const CHAMBER_CLOSE_DELAY_MS = 220;     // survive the gap between bar and panel
const CHAMBER_W = 620;                  // panel width
const CHAMBER_CANVAS_H = 360;           // canvas height — the chamber is the point, give it room

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

  // ── Quantum Visualization Chamber on hover (2026-09-23) ──
  // The chamber canvas runs a rAF loop, so it is MOUNTED ONLY WHILE OPEN —
  // hovering a height ticker must not leave an animation running behind the UI.
  const [chamberOpen, setChamberOpen] = useState(false);
  const [anchor, setAnchor] = useState<{ x: number; y: number } | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const openTimer = useRef<number | undefined>(undefined);
  const closeTimer = useRef<number | undefined>(undefined);

  // The panel opens directly beneath the point the pointer actually touched the
  // bar — not beneath the bar's centre. pointerX is captured on enter and kept
  // fresh on move, so the origin is where the eye already is.
  const pointerX = useRef<number | null>(null);
  const trackPointer = (e: React.MouseEvent) => { pointerX.current = e.clientX; };

  const openChamber = (e?: React.MouseEvent) => {
    if (e) pointerX.current = e.clientX;
    window.clearTimeout(closeTimer.current);
    openTimer.current = window.setTimeout(() => {
      const r = rootRef.current?.getBoundingClientRect();
      if (r) setAnchor({ x: pointerX.current ?? r.left + r.width / 2, y: r.bottom + 10 });
      setChamberOpen(true);
    }, CHAMBER_OPEN_DELAY_MS);
  };
  const closeChamber = () => {
    window.clearTimeout(openTimer.current);
    closeTimer.current = window.setTimeout(() => setChamberOpen(false), CHAMBER_CLOSE_DELAY_MS);
  };
  useEffect(() => () => {
    window.clearTimeout(openTimer.current);
    window.clearTimeout(closeTimer.current);
  }, []);

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
      ref={rootRef}
      className={className}
      onMouseEnter={openChamber}
      onMouseMove={trackPointer}
      onMouseLeave={closeChamber}
      onFocus={() => openChamber()}
      onBlur={closeChamber}
      tabIndex={0}
      style={{
        cursor: 'help',
        outline: 'none',
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

      {/* ── The chamber, on hover ── fixed-position so no ancestor's overflow
           or stacking context can clip it out of a top bar. */}
      {chamberOpen && anchor && createPortal((
        <div
          onMouseEnter={() => window.clearTimeout(closeTimer.current)}
          onMouseLeave={closeChamber}
          style={{
            position: 'fixed',
            left: Math.min(Math.max(anchor.x - CHAMBER_W / 2, 12), Math.max(12, window.innerWidth - CHAMBER_W - 12)),
            top: anchor.y,
            width: CHAMBER_W,
            zIndex: 9999,
            borderRadius: 14,
            background: 'rgba(10,8,20,0.72)',
            border: '1px solid rgba(168,134,255,0.30)',
            boxShadow: '0 18px 60px rgba(0,0,0,0.55)',
            backdropFilter: 'blur(18px) saturate(1.25)',
            WebkitBackdropFilter: 'blur(18px) saturate(1.25)',
            animation: 'bsb-chamber-in 180ms ease-out',
          }}
        >
          {/* A tip that points back at the exact spot on the bar the hover began.
              Its x is the origin minus the panel's clamped left edge, so it stays
              correct even when the panel is pushed off the viewport edge. */}
          <div
            style={{
              position: 'absolute', top: -7,
              left: Math.min(Math.max(anchor.x - Math.min(Math.max(anchor.x - CHAMBER_W / 2, 12),
                     Math.max(12, window.innerWidth - CHAMBER_W - 12)) - 7, 14), CHAMBER_W - 28),
              width: 14, height: 14, transform: 'rotate(45deg)',
              background: 'rgba(10,8,20,0.72)',
              borderLeft: '1px solid rgba(168,134,255,0.34)',
              borderTop: '1px solid rgba(168,134,255,0.34)',
            }}
          />
          <div
            style={{
              display: 'flex', alignItems: 'baseline', justifyContent: 'space-between',
              padding: '9px 13px 7px', borderBottom: '1px solid rgba(168,134,255,0.16)',
            }}
          >
            <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: 1.1,
                           color: '#a886ff', textTransform: 'uppercase' }}>
              Quantum Visualization Chamber
            </span>
            <span style={{ fontSize: 11, color: '#e9e7ff' }}>{heightLabel}</span>
          </div>

          <div style={{ position: 'relative', height: CHAMBER_CANVAS_H }}>
            <QuantumChamberCanvas
              fractalOverlay
              photonWaterfall
              entanglementMoire
              rainbowBoxes
            />
          </div>

          {/* Plain-language legend. Each line describes what that effect actually
              draws, and the last line says what the whole thing is NOT. */}
          <div style={{ padding: '10px 14px 12px', borderTop: '1px solid rgba(168,134,255,0.13)' }}>
            {([
              ['#8b5cf6', 'Interference web',
               'Two waves crossing make a pattern that is in neither of them alone. Send light through two slits and it lands in stripes — this is that.'],
              ['#22d3ee', 'Photon rain',
               'Light arrives in countable lumps, not a smooth stream. Every falling streak is one of them.'],
              ['#ffd76b', 'Entangled pairs',
               'Two particles can share a single state. Measure one and the other\u2019s answer is settled, however far apart they are. The line between them is that shared state — not a signal travelling.'],
              ['#f0abfc', 'Wave and particle',
               'The same thing behaves as a spread-out wave or a hard little object depending on what you ask it. The shapes morph because neither picture is the whole answer.'],
            ] as const).map(([c, t, d]) => (
              <div key={t} style={{ display: 'flex', gap: 9, marginBottom: 7, alignItems: 'flex-start' }}>
                <span style={{ width: 7, height: 7, borderRadius: 7, background: c, marginTop: 6, flexShrink: 0 }} />
                <div>
                  <div style={{ fontSize: 11.5, fontWeight: 600, color: '#e9e7ff' }}>{t}</div>
                  <div style={{ fontSize: 11, lineHeight: 1.5, color: 'rgba(233,231,255,0.62)' }}>{d}</div>
                </div>
              </div>
            ))}
            <div style={{ fontSize: 10.5, lineHeight: 1.55, color: 'rgba(233,231,255,0.42)',
                          borderTop: '1px solid rgba(168,134,255,0.10)', paddingTop: 8, marginTop: 3 }}>
              None of this is measured from the chain. It is an illustration of the ideas the
              network is named after — the block height above is the only live number on screen.
              Settings › Visual turns the four effects on and off.
            </div>
          </div>
        </div>
      ), document.body)}

      <style>{`
        @keyframes bsb-chamber-in {
          from { opacity: 0; transform: translateY(-6px) scale(0.985); }
          to   { opacity: 1; transform: none; }
        }
        @keyframes bsb-flash {
          0%   { color: #ffd76b; text-shadow: 0 0 8px rgba(255,215,107,0.55); }
          100% { color: #e9e7ff; text-shadow: none; }
        }
      `}</style>
    </div>
  );
}
