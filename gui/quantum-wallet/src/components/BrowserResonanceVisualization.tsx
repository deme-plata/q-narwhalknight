/**
 * Browser Resonance Visualization - v3.5.8
 *
 * Visualizes browser peers as quantum resonating strings, inspired by
 * the Q-Resonance consensus algorithm's physics metaphors:
 *
 * - Each browser peer is a "vibrating string" with amplitude, frequency, and phase
 * - Amplitude (A) = √(connection strength) - visual intensity
 * - Frequency (ω) = block sync rate - oscillation speed
 * - Phase (φ) = temporal alignment - wave offset
 *
 * The visualization shows:
 * - Standing waves connecting browsers through the bootstrap relay
 * - Harmonic interference patterns when browsers are in sync
 * - Resonance rings that pulse with network activity
 */

import React, { useEffect, useRef, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { getKnownBrowserPeers, type KnownBrowserPeer } from '../libp2p/browserPeerDiscovery';

interface ResonanceState {
  amplitude: number;      // Visual intensity (0-1)
  frequency: number;      // Oscillation speed (Hz)
  phase: number;          // Wave offset (radians)
  energy: number;         // Total resonance energy
  harmonicMode: number;   // Which harmonic (1, 2, 3...)
}

interface BrowserResonanceProps {
  width?: number;
  height?: number;
  showLabels?: boolean;
  theme?: 'dark' | 'light' | 'quantum';
  currentHeight?: number;
}

// Calculate resonance state from browser peer data
function calculateResonanceState(peer: KnownBrowserPeer, currentHeight: number): ResonanceState {
  const timeSinceLastSeen = Date.now() - peer.lastSeen;
  const heightDiff = Math.abs(currentHeight - peer.blockHeight);

  // Amplitude: based on connection freshness (decays over time)
  const amplitude = Math.max(0.2, 1 - (timeSinceLastSeen / 120000)); // Decays over 2 min

  // Frequency: based on how in-sync the peer is (closer = higher frequency = more resonance)
  const syncScore = Math.max(0, 1 - (heightDiff / 100));
  const frequency = 0.5 + syncScore * 2; // 0.5 Hz to 2.5 Hz

  // Phase: based on peer ID hash (gives each peer unique wave offset)
  const peerIdHash = peer.peerId.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
  const phase = (peerIdHash % 628) / 100; // 0 to 2π

  // Harmonic mode: based on peer count (more peers = higher harmonics)
  const harmonicMode = Math.min(5, Math.floor(peer.peerCount / 2) + 1);

  // Energy: combination of all factors
  const energy = amplitude * frequency * (1 + syncScore);

  return { amplitude, frequency, phase, energy, harmonicMode };
}

// Generate wave path for a vibrating string between two points
function generateWavePath(
  x1: number, y1: number,
  x2: number, y2: number,
  amplitude: number,
  frequency: number,
  phase: number,
  time: number,
  harmonicMode: number
): string {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = Math.sqrt(dx * dx + dy * dy);
  const segments = Math.max(20, Math.floor(length / 5));

  const points: string[] = [];

  for (let i = 0; i <= segments; i++) {
    const t = i / segments;

    // Base position along the line
    const baseX = x1 + dx * t;
    const baseY = y1 + dy * t;

    // Standing wave: sin(nπx/L) * cos(ωt + φ)
    // Creates nodes at endpoints and antinodes in between
    const standingWave = Math.sin(harmonicMode * Math.PI * t) *
                         Math.cos(frequency * time * 2 * Math.PI + phase);

    // Perpendicular displacement
    const perpX = -dy / length;
    const perpY = dx / length;

    // Apply wave displacement (scaled by amplitude and distance from endpoints)
    const waveAmplitude = amplitude * 15 * Math.sin(Math.PI * t); // Fade at ends
    const displacement = standingWave * waveAmplitude;

    const x = baseX + perpX * displacement;
    const y = baseY + perpY * displacement;

    points.push(i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`);
  }

  return points.join(' ');
}

// Color based on resonance energy
function getResonanceColor(energy: number, theme: string): string {
  if (theme === 'quantum') {
    // Quantum theme: purple to cyan gradient based on energy
    const hue = 280 - energy * 80; // 280 (purple) to 200 (cyan)
    const saturation = 70 + energy * 30;
    const lightness = 50 + energy * 20;
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  }
  // Default: green to gold based on energy
  const hue = 120 + energy * 30; // 120 (green) to 150 (teal/gold)
  return `hsl(${hue}, 80%, ${50 + energy * 20}%)`;
}

export const BrowserResonanceVisualization: React.FC<BrowserResonanceProps> = ({
  width = 400,
  height = 400,
  showLabels = true,
  theme = 'quantum',
  currentHeight = 0,
}) => {
  const [time, setTime] = useState(0);
  const [browserPeers, setBrowserPeers] = useState<KnownBrowserPeer[]>([]);
  const animationRef = useRef<number | null>(null);
  const centerX = width / 2;
  const centerY = height / 2;

  // Animation loop
  useEffect(() => {
    const animate = () => {
      setTime(t => t + 0.016); // ~60fps
      animationRef.current = requestAnimationFrame(animate);
    };
    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, []);

  // Update browser peers
  useEffect(() => {
    const updatePeers = () => {
      setBrowserPeers(getKnownBrowserPeers());
    };
    updatePeers();
    const interval = setInterval(updatePeers, 2000);

    // Listen for peer discovery events
    const handleDiscovery = () => updatePeers();
    window.addEventListener('browser-peer-discovered', handleDiscovery);
    window.addEventListener('browser-peers-updated', handleDiscovery);

    return () => {
      clearInterval(interval);
      window.removeEventListener('browser-peer-discovered', handleDiscovery);
      window.removeEventListener('browser-peers-updated', handleDiscovery);
    };
  }, []);

  // Calculate peer positions and resonance states
  const peerData = useMemo(() => {
    const radius = Math.min(width, height) * 0.35;
    return browserPeers.map((peer, index) => {
      const angle = (index / Math.max(1, browserPeers.length)) * 2 * Math.PI - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;
      const resonance = calculateResonanceState(peer, currentHeight);
      return { peer, x, y, angle, resonance };
    });
  }, [browserPeers, width, height, centerX, centerY, currentHeight]);

  // Calculate total network resonance energy
  const totalEnergy = useMemo(() => {
    return peerData.reduce((sum, p) => sum + p.resonance.energy, 0);
  }, [peerData]);

  // Background gradient based on theme
  const bgGradient = theme === 'quantum'
    ? 'radial-gradient(ellipse at center, rgba(88, 28, 135, 0.3) 0%, rgba(15, 23, 42, 0.9) 70%)'
    : 'radial-gradient(ellipse at center, rgba(16, 185, 129, 0.2) 0%, rgba(15, 23, 42, 0.9) 70%)';

  return (
    <div
      className="relative overflow-hidden rounded-xl"
      style={{
        width,
        height,
        background: bgGradient,
        border: '1px solid rgba(139, 92, 246, 0.3)',
      }}
    >
      {/* Resonance energy indicator */}
      <div className="absolute top-2 left-2 text-xs font-mono">
        <div className="text-purple-400/80">
          <span className="text-purple-300">⚛️</span> Resonance Energy
        </div>
        <div className="text-lg font-bold text-purple-300">
          {totalEnergy.toFixed(2)} <span className="text-xs text-purple-400">eV</span>
        </div>
      </div>

      {/* Browser count */}
      <div className="absolute top-2 right-2 text-xs font-mono text-right">
        <div className="text-cyan-400/80">
          <span className="text-cyan-300">🌐</span> Browser Strings
        </div>
        <div className="text-lg font-bold text-cyan-300">
          {browserPeers.length}
        </div>
      </div>

      <svg width={width} height={height} className="absolute inset-0">
        <defs>
          {/* Glow filter for resonance effects */}
          <filter id="resonanceGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Gradient for center node */}
          <radialGradient id="bootstrapGradient" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(139, 92, 246, 0.8)" />
            <stop offset="100%" stopColor="rgba(88, 28, 135, 0.4)" />
          </radialGradient>

          {/* Animated gradient for waves */}
          <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(6, 182, 212, 0.8)">
              <animate attributeName="stop-color"
                values="rgba(6, 182, 212, 0.8);rgba(139, 92, 246, 0.8);rgba(6, 182, 212, 0.8)"
                dur="3s" repeatCount="indefinite" />
            </stop>
            <stop offset="100%" stopColor="rgba(139, 92, 246, 0.8)">
              <animate attributeName="stop-color"
                values="rgba(139, 92, 246, 0.8);rgba(6, 182, 212, 0.8);rgba(139, 92, 246, 0.8)"
                dur="3s" repeatCount="indefinite" />
            </stop>
          </linearGradient>
        </defs>

        {/* Resonance rings (pulsing based on total energy) */}
        {[1, 2, 3].map(ring => (
          <motion.circle
            key={ring}
            cx={centerX}
            cy={centerY}
            r={30 + ring * 40}
            fill="none"
            stroke={`rgba(139, 92, 246, ${0.15 / ring})`}
            strokeWidth={1}
            animate={{
              r: [30 + ring * 40, 35 + ring * 40 + totalEnergy * 5, 30 + ring * 40],
              opacity: [0.3, 0.5, 0.3],
            }}
            transition={{
              duration: 2 + ring * 0.5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        ))}

        {/* Vibrating string connections from each peer to center */}
        {peerData.map(({ peer, x, y, resonance }) => (
          <motion.path
            key={`wave-${peer.peerId}`}
            d={generateWavePath(
              centerX, centerY,
              x, y,
              resonance.amplitude,
              resonance.frequency,
              resonance.phase,
              time,
              resonance.harmonicMode
            )}
            fill="none"
            stroke={getResonanceColor(resonance.energy, theme)}
            strokeWidth={1.5 + resonance.amplitude}
            strokeLinecap="round"
            filter="url(#resonanceGlow)"
            opacity={0.7 + resonance.amplitude * 0.3}
          />
        ))}

        {/* Inter-peer resonance connections (when peers are in sync) */}
        {peerData.map((p1, i) =>
          peerData.slice(i + 1).map(p2 => {
            const heightDiff = Math.abs(p1.peer.blockHeight - p2.peer.blockHeight);
            const inSync = heightDiff < 10;
            if (!inSync) return null;

            const avgResonance = (p1.resonance.energy + p2.resonance.energy) / 2;
            return (
              <motion.path
                key={`sync-${p1.peer.peerId}-${p2.peer.peerId}`}
                d={generateWavePath(
                  p1.x, p1.y,
                  p2.x, p2.y,
                  avgResonance * 0.5,
                  (p1.resonance.frequency + p2.resonance.frequency) / 2,
                  0,
                  time,
                  2
                )}
                fill="none"
                stroke="rgba(6, 182, 212, 0.4)"
                strokeWidth={1}
                strokeDasharray="4,4"
                opacity={0.5}
              />
            );
          })
        )}

        {/* Bootstrap relay node (center) */}
        <motion.circle
          cx={centerX}
          cy={centerY}
          r={25}
          fill="url(#bootstrapGradient)"
          stroke="rgba(139, 92, 246, 0.8)"
          strokeWidth={2}
          filter="url(#resonanceGlow)"
          animate={{
            scale: [1, 1.05 + totalEnergy * 0.02, 1],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        <text
          x={centerX}
          y={centerY + 4}
          textAnchor="middle"
          fill="white"
          fontSize={10}
          fontWeight="bold"
        >
          RELAY
        </text>

        {/* Browser peer nodes */}
        <AnimatePresence>
          {peerData.map(({ peer, x, y, resonance }) => (
            <motion.g
              key={peer.peerId}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0 }}
              transition={{ duration: 0.5 }}
            >
              {/* Resonance aura */}
              <motion.circle
                cx={x}
                cy={y}
                r={20 + resonance.energy * 5}
                fill={getResonanceColor(resonance.energy, theme)}
                opacity={0.15 + resonance.amplitude * 0.1}
                animate={{
                  r: [20 + resonance.energy * 5, 25 + resonance.energy * 8, 20 + resonance.energy * 5],
                }}
                transition={{
                  duration: 1 / resonance.frequency,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
              />

              {/* Node circle */}
              <circle
                cx={x}
                cy={y}
                r={15}
                fill={peer.isTorBrowser ? 'rgba(168, 85, 247, 0.8)' : 'rgba(6, 182, 212, 0.8)'}
                stroke={getResonanceColor(resonance.energy, theme)}
                strokeWidth={2}
                filter="url(#resonanceGlow)"
              />

              {/* Browser icon */}
              <text
                x={x}
                y={y + 4}
                textAnchor="middle"
                fill="white"
                fontSize={12}
              >
                {peer.isTorBrowser ? '🧅' : '🌐'}
              </text>

              {/* Label */}
              {showLabels && (
                <text
                  x={x}
                  y={y + 30}
                  textAnchor="middle"
                  fill="rgba(255, 255, 255, 0.7)"
                  fontSize={9}
                  fontFamily="monospace"
                >
                  {peer.peerId.substring(0, 8)}...
                </text>
              )}

              {/* Block height indicator */}
              <text
                x={x}
                y={y - 22}
                textAnchor="middle"
                fill={getResonanceColor(resonance.energy, theme)}
                fontSize={8}
                fontFamily="monospace"
              >
                #{peer.blockHeight}
              </text>

              {/* Harmonic mode indicator */}
              <text
                x={x + 20}
                y={y - 10}
                textAnchor="start"
                fill="rgba(139, 92, 246, 0.6)"
                fontSize={7}
                fontFamily="monospace"
              >
                n={resonance.harmonicMode}
              </text>
            </motion.g>
          ))}
        </AnimatePresence>
      </svg>

      {/* Legend */}
      <div className="absolute bottom-2 left-2 right-2 flex justify-between text-[10px] text-white/50 font-mono">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-cyan-400"></div>
          <span>Browser</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-purple-400"></div>
          <span>Tor Browser</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-6 h-px bg-gradient-to-r from-cyan-400 to-purple-400" style={{
            clipPath: 'polygon(0% 50%, 10% 0%, 20% 100%, 30% 0%, 40% 100%, 50% 0%, 60% 100%, 70% 0%, 80% 100%, 90% 0%, 100% 50%)'
          }}></div>
          <span>Resonance</span>
        </div>
      </div>

      {/* Empty state */}
      {browserPeers.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-purple-300/60">
            <motion.div
              className="text-4xl mb-2"
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
            >
              ⚛️
            </motion.div>
            <div className="text-sm">Waiting for browser peers...</div>
            <div className="text-xs text-purple-400/40 mt-1">
              Quantum strings will appear when browsers connect
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BrowserResonanceVisualization;
