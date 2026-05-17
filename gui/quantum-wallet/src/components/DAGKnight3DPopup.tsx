// v3.3.9-beta: Fullscreen Professional Cyberpunk DAG-Knight Visualization
// Fixed: No more jarring regeneration - smooth continuous animation

import { useRef, useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface DAGKnight3DPopupProps {
  currentHeight: number;
  consensusRound: number;
  avgBlockTime: number;
  activePeers: number;
  visible: boolean;
  onClose?: () => void;
}

interface DAGVertex {
  id: string;
  x: number;
  y: number;
  round: number;
  hue: number;
  size: number;
  isAnchor: boolean;
  parents: string[];
  energy: number;
  targetX: number;
  targetY: number;
}

export default function DAGKnight3DPopup({
  currentHeight,
  consensusRound,
  avgBlockTime,
  activePeers,
  visible,
  onClose,
}: DAGKnight3DPopupProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>(0);
  const verticesRef = useRef<DAGVertex[]>([]);
  const [canvasSize, setCanvasSize] = useState({ width: 1200, height: 800 });
  const [vdfProgress, setVdfProgress] = useState(0);
  const [activeVertex, setActiveVertex] = useState<string | null>(null);
  const timeRef = useRef(0);
  const initializedRef = useRef(false);

  // Generate DAG structure once, then just animate
  const generateDAG = useCallback((width: number, height: number) => {
    const newVertices: DAGVertex[] = [];
    const rounds = 8;
    const topPadding = 80;
    const bottomPadding = 60;
    const availableHeight = height - topPadding - bottomPadding;
    const roundSpacing = availableHeight / (rounds - 1);

    // Color palette for rounds
    const roundColors = [340, 300, 260, 220, 180, 140, 80, 45];

    for (let r = 0; r < rounds; r++) {
      const verticesInRound = r === 0 ? 1 : (r === rounds - 1 ? 3 : 3 + Math.floor(Math.random() * 3));
      const y = topPadding + r * roundSpacing;

      for (let v = 0; v < verticesInRound; v++) {
        const totalWidth = width * 0.7;
        const spacing = totalWidth / (verticesInRound + 1);
        const baseX = (width - totalWidth) / 2 + spacing * (v + 1);
        const x = baseX + (Math.random() - 0.5) * 80;

        const parents: string[] = [];
        const prevRoundVertices = newVertices.filter(vtx => vtx.round === r - 1);

        if (prevRoundVertices.length > 0) {
          const numParents = Math.min(prevRoundVertices.length, 1 + Math.floor(Math.random() * 2));
          const shuffled = [...prevRoundVertices].sort(() => Math.random() - 0.5);
          for (let p = 0; p < numParents; p++) {
            parents.push(shuffled[p].id);
          }
        }

        const isAnchor = r === 0 || (r > 0 && v === 0 && Math.random() > 0.6);

        newVertices.push({
          id: `v${r}-${v}`,
          x,
          y,
          round: r,
          hue: roundColors[r] + (Math.random() - 0.5) * 20,
          size: isAnchor ? 20 : 14 + Math.random() * 6,
          isAnchor,
          parents,
          energy: 0.5 + Math.random() * 0.5,
          targetX: x,
          targetY: y,
        });
      }
    }

    return newVertices;
  }, []);

  // Handle window resize - fit within viewport with padding
  useEffect(() => {
    if (!visible) return;

    const updateSize = () => {
      // High resolution canvas - user can scroll if needed
      const width = Math.min(window.innerWidth - 320, 1200);
      const height = Math.min(window.innerHeight - 100, 850);
      setCanvasSize({ width, height });

      // Regenerate DAG with new size if not initialized or size changed significantly
      if (!initializedRef.current || verticesRef.current.length === 0) {
        verticesRef.current = generateDAG(width, height);
        initializedRef.current = true;
      } else {
        // Just update target positions for smooth transition
        const rounds = 8;
        const topPadding = 80;
        const bottomPadding = 60;
        const availableHeight = height - topPadding - bottomPadding;
        const roundSpacing = availableHeight / (rounds - 1);

        verticesRef.current.forEach(vertex => {
          const roundVertices = verticesRef.current.filter(v => v.round === vertex.round);
          const idx = roundVertices.indexOf(vertex);
          const totalWidth = width * 0.7;
          const spacing = totalWidth / (roundVertices.length + 1);
          const baseX = (width - totalWidth) / 2 + spacing * (idx + 1);

          vertex.targetX = baseX;
          vertex.targetY = topPadding + vertex.round * roundSpacing;
        });
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, [visible, generateDAG]);

  // Reset on close
  useEffect(() => {
    if (!visible) {
      initializedRef.current = false;
      verticesRef.current = [];
    }
  }, [visible]);

  // VDF animation - continuous, no regeneration
  useEffect(() => {
    if (!visible) return;

    const vdfInterval = setInterval(() => {
      setVdfProgress(prev => (prev + 1) % 100);
    }, 80);

    return () => clearInterval(vdfInterval);
  }, [visible]);

  // Main animation loop
  useEffect(() => {
    if (!visible || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasSize.width * dpr;
    canvas.height = canvasSize.height * dpr;
    ctx.scale(dpr, dpr);

    const animate = () => {
      timeRef.current += 0.016;
      const t = timeRef.current;
      const { width, height } = canvasSize;

      // Smooth position interpolation (vertices gently float)
      verticesRef.current.forEach(vertex => {
        // Add gentle floating motion
        const floatX = Math.sin(t * 0.5 + vertex.energy * 10) * 15;
        const floatY = Math.cos(t * 0.3 + vertex.energy * 8) * 10;

        // Smoothly interpolate to target + float
        vertex.x += ((vertex.targetX + floatX) - vertex.x) * 0.02;
        vertex.y += ((vertex.targetY + floatY) - vertex.y) * 0.02;
      });

      // Clear with dramatic gradient
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, '#000408');
      gradient.addColorStop(0.3, '#020812');
      gradient.addColorStop(0.7, '#030a18');
      gradient.addColorStop(1, '#010306');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      // Subtle animated grid
      ctx.strokeStyle = `rgba(0, 255, 255, ${0.02 + Math.sin(t * 0.5) * 0.01})`;
      ctx.lineWidth = 1;
      const gridSize = 50;
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Hexagonal background pattern
      ctx.strokeStyle = 'rgba(147, 51, 234, 0.04)';
      ctx.lineWidth = 1;
      const hexSize = 80;
      for (let row = -1; row < height / hexSize + 1; row++) {
        for (let col = -1; col < width / hexSize + 1; col++) {
          const hx = col * hexSize * 1.5 + (row % 2) * hexSize * 0.75;
          const hy = row * hexSize * 0.866;
          drawHexagon(ctx, hx, hy, hexSize * 0.4);
        }
      }

      // Round labels on the left
      const rounds = 8;
      const topPadding = 80;
      const bottomPadding = 60;
      const availableHeight = height - topPadding - bottomPadding;
      const roundSpacing = availableHeight / (rounds - 1);

      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      for (let r = 0; r < rounds; r++) {
        const ry = topPadding + r * roundSpacing;
        ctx.fillStyle = `hsla(${180 + r * 20}, 80%, 60%, 0.7)`;
        ctx.fillText(`R${r}`, 20, ry + 4);

        ctx.strokeStyle = `hsla(${180 + r * 20}, 80%, 50%, 0.1)`;
        ctx.setLineDash([5, 10]);
        ctx.beginPath();
        ctx.moveTo(50, ry);
        ctx.lineTo(width - 30, ry);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw edges
      verticesRef.current.forEach(vertex => {
        vertex.parents.forEach(parentId => {
          const parent = verticesRef.current.find(v => v.id === parentId);
          if (!parent) return;
          drawQuantumEdge(ctx, parent, vertex, t);
        });
      });

      // Draw data pulses
      verticesRef.current.forEach((vertex, idx) => {
        vertex.parents.forEach((parentId, pIdx) => {
          const parent = verticesRef.current.find(v => v.id === parentId);
          if (!parent) return;

          const pulseProgress = ((t * 0.4 + idx * 0.2 + pIdx * 0.15) % 1);
          const px = parent.x + (vertex.x - parent.x) * pulseProgress;
          const py = parent.y + (vertex.y - parent.y) * pulseProgress;

          const pulseGradient = ctx.createRadialGradient(px, py, 0, px, py, 12);
          pulseGradient.addColorStop(0, 'rgba(0, 255, 255, 0.9)');
          pulseGradient.addColorStop(0.4, 'rgba(0, 255, 255, 0.4)');
          pulseGradient.addColorStop(1, 'rgba(0, 255, 255, 0)');
          ctx.fillStyle = pulseGradient;
          ctx.beginPath();
          ctx.arc(px, py, 12, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = '#fff';
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, Math.PI * 2);
          ctx.fill();
        });
      });

      // Draw vertices
      verticesRef.current.forEach(vertex => {
        drawVertex(ctx, vertex, t, vertex.id === activeVertex);
      });

      // VDF progress ring at center
      const centerX = width / 2;
      const centerY = height / 2;
      const vdfRadius = Math.min(width, height) * 0.35;

      // VDF outer glow
      const vdfGlow = ctx.createRadialGradient(centerX, centerY, vdfRadius - 30, centerX, centerY, vdfRadius + 30);
      vdfGlow.addColorStop(0, 'rgba(147, 51, 234, 0)');
      vdfGlow.addColorStop(0.5, `rgba(147, 51, 234, ${0.03 + Math.sin(t * 2) * 0.02})`);
      vdfGlow.addColorStop(1, 'rgba(147, 51, 234, 0)');
      ctx.fillStyle = vdfGlow;
      ctx.beginPath();
      ctx.arc(centerX, centerY, vdfRadius + 30, 0, Math.PI * 2);
      ctx.fill();

      // VDF background ring
      ctx.strokeStyle = 'rgba(147, 51, 234, 0.15)';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(centerX, centerY, vdfRadius, 0, Math.PI * 2);
      ctx.stroke();

      // VDF progress arc
      ctx.strokeStyle = '#9333ea';
      ctx.lineWidth = 5;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.arc(centerX, centerY, vdfRadius, -Math.PI / 2, -Math.PI / 2 + (vdfProgress / 100) * Math.PI * 2);
      ctx.stroke();

      // VDF glow on progress
      ctx.shadowColor = '#9333ea';
      ctx.shadowBlur = 15;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Floating particles around VDF ring
      for (let i = 0; i < 40; i++) {
        const angle = (t * 0.15 + i * 0.157) % (Math.PI * 2);
        const radius = vdfRadius + Math.sin(t * 0.8 + i * 0.5) * 40;
        const px = centerX + Math.cos(angle) * radius;
        const py = centerY + Math.sin(angle) * radius;
        const alpha = 0.4 + Math.sin(t * 2 + i) * 0.2;
        const size = 2 + Math.sin(t + i * 0.3) * 1;

        ctx.fillStyle = i % 3 === 0 ? `rgba(0, 255, 255, ${alpha})` :
                        i % 3 === 1 ? `rgba(147, 51, 234, ${alpha})` :
                        `rgba(34, 197, 94, ${alpha})`;
        ctx.beginPath();
        ctx.arc(px, py, size, 0, Math.PI * 2);
        ctx.fill();
      }

      // Title
      ctx.font = 'bold 20px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.shadowColor = '#00ffff';
      ctx.shadowBlur = 20;
      ctx.fillStyle = '#00ffff';
      ctx.fillText('DAG-KNIGHT QUANTUM CONSENSUS', width / 2, 40);
      ctx.shadowBlur = 0;

      // Subtitle
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.fillText('Real-time Directed Acyclic Graph Visualization', width / 2, 60);

      // VDF percentage in center
      ctx.font = 'bold 16px "JetBrains Mono", monospace';
      ctx.fillStyle = '#9333ea';
      ctx.shadowColor = '#9333ea';
      ctx.shadowBlur = 10;
      ctx.fillText(`VDF ${vdfProgress}%`, centerX, centerY - vdfRadius - 20);
      ctx.shadowBlur = 0;

      // Stats bar at bottom
      const statsY = height - 25;
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.fillText(
        `Height: ${currentHeight.toLocaleString()}  •  Round: ${consensusRound}  •  Peers: ${activePeers}  •  Block Time: ${avgBlockTime.toFixed(1)}s`,
        width / 2,
        statsY
      );

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => cancelAnimationFrame(animationRef.current);
  }, [visible, canvasSize, currentHeight, consensusRound, activePeers, avgBlockTime, activeVertex, vdfProgress]);

  // Mouse interactions
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    let hoveredVertex: string | null = null;
    for (const vertex of verticesRef.current) {
      const dx = x - vertex.x;
      const dy = y - vertex.y;
      if (Math.sqrt(dx * dx + dy * dy) < vertex.size + 8) {
        hoveredVertex = vertex.id;
        break;
      }
    }
    setActiveVertex(hoveredVertex);
  };

  if (!visible) return null;

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          ref={containerRef}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="fixed inset-0 z-[9999] flex items-center justify-center overflow-auto py-4"
          style={{ backgroundColor: '#000' }}
        >
          {/* Close button - fixed position so always visible */}
          <button
            onClick={onClose}
            className="fixed top-4 right-4 z-[10000] w-12 h-12 rounded-xl bg-black/80 hover:bg-red-500/30 flex items-center justify-center transition-all border border-white/30 hover:border-red-500/50 group backdrop-blur-sm"
          >
            <span className="text-white group-hover:text-red-400 text-3xl leading-none font-bold">&times;</span>
          </button>

          {/* Live indicator - fixed position */}
          <div className="fixed top-4 left-4 z-[10000] flex items-center gap-3 px-4 py-2 rounded-xl bg-black/70 border border-cyan-500/30 backdrop-blur-sm">
            <div className="relative">
              <div className="w-3 h-3 rounded-full bg-cyan-400" />
              <div className="absolute inset-0 w-3 h-3 rounded-full bg-cyan-400 animate-ping" />
            </div>
            <span
              className="text-sm font-semibold tracking-wider"
              style={{
                color: '#00ffff',
                fontFamily: "'JetBrains Mono', monospace",
                textShadow: '0 0 10px rgba(0, 255, 255, 0.5)',
              }}
            >
              LIVE CONSENSUS
            </span>
          </div>

          {/* Main content container - canvas and info panel side by side */}
          <div className="flex items-start gap-6 relative">
            {/* Close button - top right of content */}
            <button
              onClick={onClose}
              className="absolute -top-2 -right-2 z-[10001] w-10 h-10 rounded-full bg-red-600 hover:bg-red-500 flex items-center justify-center transition-all shadow-lg shadow-red-500/50"
            >
              <span className="text-white text-2xl font-bold leading-none">×</span>
            </button>
            {/* Canvas */}
            <canvas
              ref={canvasRef}
              className="block cursor-crosshair rounded-xl border border-cyan-500/20"
              style={{ width: canvasSize.width, height: canvasSize.height }}
              onMouseMove={handleMouseMove}
            />

            {/* Educational Info Panel */}
            <div
              className="w-80 max-h-[750px] overflow-y-auto rounded-xl bg-black/80 border border-cyan-500/30 backdrop-blur-md flex-shrink-0"
              style={{ fontFamily: "'JetBrains Mono', monospace" }}
            >
              {/* Header */}
              <div className="px-4 py-3 border-b border-cyan-500/30 bg-gradient-to-r from-cyan-500/10 to-purple-500/10">
                <h3 className="text-sm font-bold text-cyan-400 tracking-wider">DAG-KNIGHT EXPLAINED</h3>
                <p className="text-[10px] text-gray-500 mt-1">Quantum-Enhanced Consensus</p>
              </div>

              {/* Rounds Section */}
              <div className="px-4 py-3 border-b border-white/10">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-bold text-purple-400">R0 - R7</span>
                  <span className="text-[10px] text-gray-500">Consensus Rounds</span>
                </div>
                <p className="text-[11px] text-gray-300 leading-relaxed">
                  Each <span className="text-cyan-400 font-semibold">round</span> represents a layer in the DAG structure.
                  <span className="text-purple-400 font-semibold"> R0</span> is the genesis round, with newer rounds building on previous ones below.
                </p>
              </div>

              {/* Anchor Vertex */}
              <div className="px-4 py-3 border-b border-white/10">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-4 h-4 rounded-full bg-gradient-to-br from-yellow-300 to-orange-500 shadow-lg shadow-yellow-500/30" />
                  <span className="text-xs font-bold text-yellow-400">Anchor Vertex</span>
                </div>
                <p className="text-[11px] text-gray-300 leading-relaxed">
                  Special vertices elected via <span className="text-purple-400 font-semibold">VDF (Verifiable Delay Function)</span>.
                  Anchors provide total ordering and <span className="text-cyan-400 font-semibold">finality</span>. Marked with golden spinning rings.
                </p>
              </div>

              {/* Standard Vertex */}
              <div className="px-4 py-3 border-b border-white/10">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3.5 h-3.5 rounded-full bg-gradient-to-br from-cyan-400 to-purple-500" />
                  <span className="text-xs font-bold text-cyan-400">Standard Vertex</span>
                </div>
                <p className="text-[11px] text-gray-300 leading-relaxed">
                  Contains batches of <span className="text-green-400 font-semibold">transactions</span> with references to multiple parents,
                  enabling <span className="text-cyan-400 font-semibold">parallel processing</span> and high throughput.
                </p>
              </div>

              {/* Quantum Edges */}
              <div className="px-4 py-3 border-b border-white/10">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-8 h-0.5 bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 rounded" />
                  <span className="text-xs font-bold text-pink-400">Quantum Edges</span>
                </div>
                <p className="text-[11px] text-gray-300 leading-relaxed">
                  Wavy lines show <span className="text-cyan-400 font-semibold">parent-child relationships</span>.
                  The wave pattern represents quantum interference in <span className="text-purple-400 font-semibold">post-quantum cryptography</span>.
                </p>
              </div>

              {/* Data Pulses */}
              <div className="px-4 py-3 border-b border-white/10">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3 h-3 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50 animate-pulse" />
                  <span className="text-xs font-bold text-cyan-400">Data Pulses</span>
                </div>
                <p className="text-[11px] text-gray-300 leading-relaxed">
                  Glowing orbs traveling along edges = <span className="text-green-400 font-semibold">data propagation</span> through the network during consensus.
                </p>
              </div>

              {/* VDF Ring */}
              <div className="px-4 py-3 border-b border-white/10">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-4 h-4 rounded-full border-2 border-purple-500" />
                  <span className="text-xs font-bold text-purple-400">VDF Progress Ring</span>
                </div>
                <p className="text-[11px] text-gray-300 leading-relaxed">
                  Central purple ring shows <span className="text-purple-400 font-semibold">VDF computation</span>.
                  Sequential computation ensures <span className="text-cyan-400 font-semibold">unpredictable, fair</span> anchor election.
                </p>
              </div>

              {/* Network Nodes vs DAG */}
              <div className="px-4 py-3 border-b border-white/10">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-4 h-4 rounded-sm bg-gradient-to-br from-blue-400 to-indigo-600" />
                  <span className="text-xs font-bold text-blue-400">Network Nodes</span>
                </div>
                <p className="text-[11px] text-gray-300 leading-relaxed">
                  <span className="text-blue-400 font-semibold">Physical peers</span> in the P2P network.
                  Each node broadcasts vertices via <span className="text-cyan-400 font-semibold">Gossipsub</span>.
                  The <span className="text-green-400 font-semibold">data pulses</span> represent messages flowing between nodes.
                </p>
                <div className="mt-2 p-2 rounded bg-blue-500/10 border border-blue-500/20">
                  <p className="text-[10px] text-blue-300">
                    <span className="font-bold">{activePeers} peers</span> currently connected, sharing vertices in real-time.
                  </p>
                </div>
              </div>

              {/* Data Exchange */}
              <div className="px-4 py-3 border-b border-white/10">
                <div className="flex items-center gap-2 mb-2">
                  <div className="flex gap-0.5">
                    <div className="w-1.5 h-3 bg-cyan-400 rounded-sm" />
                    <div className="w-1.5 h-3 bg-purple-400 rounded-sm" />
                    <div className="w-1.5 h-3 bg-green-400 rounded-sm" />
                  </div>
                  <span className="text-xs font-bold text-cyan-400">Data Exchange</span>
                </div>
                <p className="text-[11px] text-gray-300 leading-relaxed mb-2">
                  Nodes exchange several message types:
                </p>
                <ul className="text-[10px] text-gray-400 space-y-1">
                  <li className="flex items-center gap-2">
                    <span className="text-cyan-400">→</span> <span className="text-cyan-300">Vertices</span> - Transaction batches
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-purple-400">→</span> <span className="text-purple-300">Acks</span> - Reliable broadcast confirmations
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-green-400">→</span> <span className="text-green-300">Sync</span> - Catch-up requests
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-yellow-400">→</span> <span className="text-yellow-300">Heights</span> - Peer height announcements
                  </li>
                </ul>
              </div>

              {/* Why DAG? */}
              <div className="px-4 py-3">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-bold text-green-400">Why DAG-Knight?</span>
                </div>
                <ul className="text-[11px] text-gray-400 space-y-1">
                  <li className="flex items-center gap-2">
                    <span className="text-green-400">•</span> 50,000+ TPS
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-cyan-400">•</span> Sub-3s finality
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-purple-400">•</span> Post-quantum secure
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-yellow-400">•</span> ZK privacy
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* Stats cards - compact */}
          <div className="absolute bottom-3 left-1/2 transform -translate-x-1/2 z-50 flex gap-2">
            {[
              { label: 'HEIGHT', value: currentHeight.toLocaleString(), color: '#00ffff' },
              { label: 'ROUND', value: consensusRound, color: '#9333ea' },
              { label: 'BLOCK TIME', value: `${avgBlockTime.toFixed(1)}s`, color: '#22c55e' },
              { label: 'PEERS', value: activePeers, color: '#ec4899' },
            ].map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + i * 0.03 }}
                className="px-3 py-1.5 rounded-lg text-center backdrop-blur-sm"
                style={{
                  background: 'rgba(0, 0, 0, 0.6)',
                  border: `1px solid ${stat.color}30`,
                }}
              >
                <div
                  className="text-sm font-bold"
                  style={{
                    color: stat.color,
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                >
                  {stat.value}
                </div>
                <div className="text-[8px] text-gray-500 tracking-wider">{stat.label}</div>
              </motion.div>
            ))}
          </div>

        </motion.div>
      )}
    </AnimatePresence>
  );
}

// Helper: Draw hexagon
function drawHexagon(ctx: CanvasRenderingContext2D, x: number, y: number, size: number) {
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    const px = x + size * Math.cos(angle);
    const py = y + size * Math.sin(angle);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.stroke();
}

// Helper: Draw vertex
function drawVertex(
  ctx: CanvasRenderingContext2D,
  vertex: DAGVertex,
  time: number,
  isHovered: boolean
) {
  const { x, y, hue, size, isAnchor, energy } = vertex;
  const pulse = 1 + Math.sin(time * 2.5 + energy * 10) * 0.08;
  const actualSize = size * pulse;

  // Glow layers
  const glowLayers = isAnchor ? 5 : 3;
  for (let i = glowLayers; i > 0; i--) {
    const glowRadius = actualSize * (1 + i * 0.6);
    const glowGradient = ctx.createRadialGradient(x, y, 0, x, y, glowRadius);

    if (isAnchor) {
      glowGradient.addColorStop(0, `hsla(45, 100%, 60%, ${0.35 / i})`);
      glowGradient.addColorStop(0.5, `hsla(35, 100%, 50%, ${0.15 / i})`);
      glowGradient.addColorStop(1, 'transparent');
    } else {
      glowGradient.addColorStop(0, `hsla(${hue}, 80%, 60%, ${0.25 / i})`);
      glowGradient.addColorStop(0.6, `hsla(${hue}, 70%, 50%, ${0.1 / i})`);
      glowGradient.addColorStop(1, 'transparent');
    }

    ctx.fillStyle = glowGradient;
    ctx.beginPath();
    ctx.arc(x, y, glowRadius, 0, Math.PI * 2);
    ctx.fill();
  }

  // Anchor spinning ring
  if (isAnchor) {
    ctx.strokeStyle = `hsla(45, 100%, 65%, ${0.5 + Math.sin(time * 3) * 0.2})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, actualSize + 8, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = '#ffd700';
    ctx.lineWidth = 3;
    ctx.beginPath();
    const arcStart = time * 2;
    ctx.arc(x, y, actualSize + 14, arcStart, arcStart + Math.PI * 0.6);
    ctx.stroke();
  }

  // Main body
  const bodyGradient = ctx.createRadialGradient(x - actualSize * 0.3, y - actualSize * 0.3, 0, x, y, actualSize);
  if (isAnchor) {
    bodyGradient.addColorStop(0, '#fffacd');
    bodyGradient.addColorStop(0.3, '#ffd700');
    bodyGradient.addColorStop(1, '#b8860b');
  } else {
    bodyGradient.addColorStop(0, `hsl(${hue}, 50%, 85%)`);
    bodyGradient.addColorStop(0.4, `hsl(${hue}, 70%, 55%)`);
    bodyGradient.addColorStop(1, `hsl(${hue}, 80%, 30%)`);
  }

  ctx.fillStyle = bodyGradient;
  ctx.beginPath();
  ctx.arc(x, y, actualSize, 0, Math.PI * 2);
  ctx.fill();

  // Highlight
  ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
  ctx.beginPath();
  ctx.arc(x - actualSize * 0.3, y - actualSize * 0.3, actualSize * 0.3, 0, Math.PI * 2);
  ctx.fill();

  // Hover effect
  if (isHovered) {
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, actualSize + 5, 0, Math.PI * 2);
    ctx.stroke();

    // Tooltip
    const tooltipWidth = 100;
    const tooltipHeight = 50;
    const tx = x + 25;
    const ty = y - 30;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
    ctx.beginPath();
    ctx.roundRect(tx, ty, tooltipWidth, tooltipHeight, 8);
    ctx.fill();

    ctx.strokeStyle = `hsl(${hue}, 70%, 50%)`;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.font = 'bold 11px "JetBrains Mono", monospace';
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'left';
    ctx.fillText(vertex.id.toUpperCase(), tx + 10, ty + 20);

    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#888';
    ctx.fillText(`Round ${vertex.round}`, tx + 10, ty + 36);
    ctx.fillText(vertex.isAnchor ? '★ Anchor' : 'Vertex', tx + 10, ty + 36);
  }

  // Label
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.fillStyle = isAnchor ? 'rgba(255, 215, 0, 0.9)' : `hsla(${hue}, 70%, 75%, 0.8)`;
  ctx.textAlign = 'center';
  ctx.fillText(vertex.id.toUpperCase(), x, y + actualSize + 16);
}

// Helper: Draw quantum edge
function drawQuantumEdge(
  ctx: CanvasRenderingContext2D,
  from: DAGVertex,
  to: DAGVertex,
  time: number
) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist === 0) return;

  const perpX = -dy / dist;
  const perpY = dx / dist;

  ctx.beginPath();
  const steps = 50;

  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const baseX = from.x + dx * t;
    const baseY = from.y + dy * t;

    const envelope = Math.sin(t * Math.PI);
    const waveAmp = envelope * 12;
    const wave = Math.sin(time * 2 + t * Math.PI * 5) * waveAmp;

    const x = baseX + perpX * wave;
    const y = baseY + perpY * wave;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }

  const gradient = ctx.createLinearGradient(from.x, from.y, to.x, to.y);
  gradient.addColorStop(0, `hsla(${from.hue}, 70%, 50%, 0.6)`);
  gradient.addColorStop(0.5, `hsla(${(from.hue + to.hue) / 2}, 80%, 60%, 0.7)`);
  gradient.addColorStop(1, `hsla(${to.hue}, 70%, 50%, 0.6)`);

  ctx.strokeStyle = gradient;
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Glow
  ctx.lineWidth = 5;
  ctx.globalAlpha = 0.25;
  ctx.stroke();
  ctx.globalAlpha = 1;
}
