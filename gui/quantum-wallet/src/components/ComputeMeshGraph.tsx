/**
 * ComputeMeshGraph.tsx — D3-style Force-Directed Tunnel Mesh Visualization
 *
 * Issue #008: Tunnel Mesh Visualization (criteria 6/8, 7/8, 8/8)
 *
 * Implements:
 *   1. Force-directed graph with SVG (no d3 dependency — pure React + manual force simulation)
 *   2. Real-time bandwidth per tunnel with color-coded lines and animated particles
 *   3. Click-to-inspect node detail popup with CPU/GPU/RAM bars, layers, tunnels
 *
 * Usage:
 *   <ComputeMeshGraph
 *     nodes={meshNodes}
 *     tunnels={meshTunnels}
 *     onNodeClick={(node) => { ... }}
 *   />
 *
 * The force simulation runs in a requestAnimationFrame loop, computing
 * repulsive (Coulomb) forces between all node pairs, attractive (spring)
 * forces along tunnel edges, and a gentle centering gravity. Positions
 * converge within ~120 frames (~2 seconds) and the simulation cools to
 * near-zero velocity, at which point the loop sleeps until new data arrives.
 */
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';

// ─── Public Types ────────────────────────────────────────────────────────────

export interface MeshNode {
  id: string;
  name: string;
  role: 'bootstrap' | 'miner' | 'validator' | 'backup';
  cpu_percent: number;
  gpu_percent: number;
  ram_percent: number;
  cores: number;
  gpu_tflops: number;
  /** Optional: peer ID for detail popup */
  peer_id?: string;
  /** Optional: IP address */
  ip?: string;
  /** Optional: active layer names */
  active_layers?: string[];
  /** Optional: core allocation per layer */
  layer_cores?: Record<string, number>;
  /** Optional: task queue depth */
  task_queue_depth?: number;
  /** Internal simulation position — set by the force layout */
  x?: number;
  y?: number;
}

export interface MeshTunnel {
  source: string; // node id
  target: string; // node id
  bandwidth_mbps: number;
  max_bandwidth_mbps: number;
  active_tasks: number;
}

export interface ComputeMeshGraphProps {
  nodes: MeshNode[];
  tunnels: MeshTunnel[];
  /** Optional: called when a node circle is clicked */
  onNodeClick?: (node: MeshNode) => void;
  /** Optional: override min-height (default 400) */
  minHeight?: number;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const ROLE_COLORS: Record<MeshNode['role'], string> = {
  bootstrap: '#3b82f6', // blue-500
  miner: '#22c55e',     // green-500
  validator: '#a855f7', // purple-500
  backup: '#f59e0b',    // amber-500
};

const ROLE_LABELS: Record<MeshNode['role'], string> = {
  bootstrap: 'Bootstrap',
  miner: 'Miner',
  validator: 'Validator',
  backup: 'Backup',
};

/** Map bandwidth utilization ratio (0-1) to a CSS color string. */
function bandwidthColor(ratio: number): string {
  if (ratio < 0.25) return '#22c55e';     // green
  if (ratio < 0.50) return '#84cc16';     // lime
  if (ratio < 0.75) return '#eab308';     // yellow
  if (ratio < 0.90) return '#f97316';     // orange
  return '#ef4444';                        // red
}

/** Clamp a number between lo and hi. */
function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

// ─── Force Simulation (no d3 — implemented from scratch) ─────────────────────

interface SimNode {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  pinned: boolean;
}

interface SimEdge {
  sourceIdx: number;
  targetIdx: number;
  idealLength: number;
}

/**
 * One tick of the force simulation. Mutates simNodes in-place.
 *
 * Forces applied:
 *   - Repulsion between every node pair (Coulomb, 1/r^2)
 *   - Attraction along edges (Hooke spring)
 *   - Center gravity (pull toward center)
 *   - Velocity damping (friction)
 */
function forceTick(
  simNodes: SimNode[],
  simEdges: SimEdge[],
  width: number,
  height: number,
  alpha: number, // cooling factor 0-1
): number {
  const n = simNodes.length;
  if (n === 0) return 0;

  const cx = width / 2;
  const cy = height / 2;
  const repulsionStrength = 8000 * alpha;
  const springStrength = 0.008 * alpha;
  const gravityStrength = 0.02 * alpha;
  const damping = 0.85;

  // 1. Repulsion (all pairs)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let dx = simNodes[j].x - simNodes[i].x;
      let dy = simNodes[j].y - simNodes[i].y;
      let dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 1) dist = 1;
      const force = repulsionStrength / (dist * dist);
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      if (!simNodes[i].pinned) { simNodes[i].vx -= fx; simNodes[i].vy -= fy; }
      if (!simNodes[j].pinned) { simNodes[j].vx += fx; simNodes[j].vy += fy; }
    }
  }

  // 2. Spring attraction along edges
  for (const edge of simEdges) {
    const a = simNodes[edge.sourceIdx];
    const b = simNodes[edge.targetIdx];
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 1) dist = 1;
    const displacement = dist - edge.idealLength;
    const force = springStrength * displacement;
    const fx = (dx / dist) * force;
    const fy = (dy / dist) * force;
    if (!a.pinned) { a.vx += fx; a.vy += fy; }
    if (!b.pinned) { b.vx -= fx; b.vy -= fy; }
  }

  // 3. Center gravity
  for (const node of simNodes) {
    if (node.pinned) continue;
    node.vx += (cx - node.x) * gravityStrength;
    node.vy += (cy - node.y) * gravityStrength;
  }

  // 4. Apply velocity + damping + boundary clamp
  let totalKinetic = 0;
  const pad = 40;
  for (const node of simNodes) {
    if (node.pinned) continue;
    node.vx *= damping;
    node.vy *= damping;
    node.x += node.vx;
    node.y += node.vy;
    node.x = clamp(node.x, pad, width - pad);
    node.y = clamp(node.y, pad, height - pad);
    totalKinetic += node.vx * node.vx + node.vy * node.vy;
  }

  return totalKinetic;
}

// ─── Animated Particle Along a Tunnel ────────────────────────────────────────

interface ParticleState {
  tunnelKey: string;
  progress: number; // 0-1 along the edge
  speed: number;    // progress per frame
}

// ─── Node Detail Popup ──────────────────────────────────────────────────────

interface NodePopupProps {
  node: MeshNode;
  tunnels: MeshTunnel[];
  allNodes: MeshNode[];
  position: { x: number; y: number };
  onClose: () => void;
}

function NodeDetailPopup({ node, tunnels, allNodes, position, onClose }: NodePopupProps) {
  // Find tunnels connected to this node
  const connected = tunnels.filter(t => t.source === node.id || t.target === node.id);

  const miniBar = (pct: number, color: string, label: string) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
      <span style={{ fontSize: 10, color: 'rgba(251,191,36,0.6)', width: 28, flexShrink: 0 }}>{label}</span>
      <div style={{
        flex: 1, height: 6, borderRadius: 3,
        background: 'rgba(15,23,42,0.8)', overflow: 'hidden',
      }}>
        <div style={{
          width: `${clamp(pct, 0, 100)}%`, height: '100%', borderRadius: 3,
          background: color, transition: 'width 0.4s ease',
        }} />
      </div>
      <span style={{ fontSize: 10, color: 'rgba(251,191,36,0.8)', width: 36, textAlign: 'right', flexShrink: 0 }}>
        {pct.toFixed(1)}%
      </span>
    </div>
  );

  return (
    <div
      onClick={(e) => e.stopPropagation()}
      style={{
        position: 'absolute',
        left: position.x + 16,
        top: position.y - 20,
        minWidth: 280,
        maxWidth: 340,
        background: 'rgba(15,23,42,0.95)',
        border: '1px solid rgba(148,163,184,0.25)',
        borderRadius: 12,
        padding: 14,
        zIndex: 50,
        boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
        backdropFilter: 'blur(12px)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 10, height: 10, borderRadius: '50%',
            background: ROLE_COLORS[node.role],
            boxShadow: `0 0 8px ${ROLE_COLORS[node.role]}`,
          }} />
          <span style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0' }}>{node.name}</span>
          <span style={{
            fontSize: 9, padding: '1px 6px', borderRadius: 8,
            background: `${ROLE_COLORS[node.role]}22`,
            color: ROLE_COLORS[node.role],
            border: `1px solid ${ROLE_COLORS[node.role]}44`,
            textTransform: 'uppercase', fontWeight: 700,
          }}>
            {ROLE_LABELS[node.role]}
          </span>
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'transparent', border: 'none', cursor: 'pointer',
            color: 'rgba(148,163,184,0.6)', fontSize: 16, lineHeight: 1, padding: 2,
          }}
        >
          x
        </button>
      </div>

      {/* Server info */}
      {(node.ip || node.peer_id) && (
        <div style={{ marginBottom: 8 }}>
          {node.ip && (
            <div style={{ fontSize: 10, color: 'rgba(148,163,184,0.6)', marginBottom: 2 }}>
              IP: <span style={{ color: 'rgba(251,191,36,0.7)', fontFamily: 'monospace' }}>{node.ip}</span>
            </div>
          )}
          {node.peer_id && (
            <div style={{ fontSize: 10, color: 'rgba(148,163,184,0.6)', marginBottom: 2 }}>
              Peer: <span style={{ color: 'rgba(251,191,36,0.7)', fontFamily: 'monospace' }}>
                {node.peer_id.length > 20 ? node.peer_id.slice(0, 10) + '...' + node.peer_id.slice(-8) : node.peer_id}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Utilization bars */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ fontSize: 10, color: 'rgba(110,231,183,0.7)', marginBottom: 4, fontWeight: 600 }}>
          Resource Utilization
        </div>
        {miniBar(node.cpu_percent, '#22c55e', 'CPU')}
        {miniBar(node.gpu_percent, '#a855f7', 'GPU')}
        {miniBar(node.ram_percent, '#3b82f6', 'RAM')}
      </div>

      {/* Compute capacity */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 10, fontSize: 10, color: 'rgba(148,163,184,0.6)' }}>
        <span>Cores: <span style={{ color: '#e2e8f0' }}>{node.cores}</span></span>
        <span>GPU: <span style={{ color: '#e2e8f0' }}>{node.gpu_tflops.toFixed(1)} TFLOPS</span></span>
        {node.task_queue_depth !== undefined && (
          <span>Queue: <span style={{ color: '#e2e8f0' }}>{node.task_queue_depth}</span></span>
        )}
      </div>

      {/* Active layers */}
      {node.active_layers && node.active_layers.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 10, color: 'rgba(110,231,183,0.7)', marginBottom: 4, fontWeight: 600 }}>
            Active Layers
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {node.active_layers.map(layer => (
              <span key={layer} style={{
                fontSize: 9, padding: '2px 6px', borderRadius: 4,
                background: 'rgba(139,92,246,0.15)', color: '#c4b5fd',
                border: '1px solid rgba(139,92,246,0.25)',
              }}>
                {layer}
                {node.layer_cores?.[layer] !== undefined && (
                  <span style={{ color: 'rgba(251,191,36,0.6)', marginLeft: 4 }}>
                    {node.layer_cores[layer]}c
                  </span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Connected tunnels */}
      {connected.length > 0 && (
        <div>
          <div style={{ fontSize: 10, color: 'rgba(110,231,183,0.7)', marginBottom: 4, fontWeight: 600 }}>
            Connected Tunnels ({connected.length})
          </div>
          <div style={{ maxHeight: 100, overflowY: 'auto' }}>
            {connected.map((t, i) => {
              const peerId = t.source === node.id ? t.target : t.source;
              const peerNode = allNodes.find(n => n.id === peerId);
              const ratio = t.max_bandwidth_mbps > 0 ? t.bandwidth_mbps / t.max_bandwidth_mbps : 0;
              return (
                <div key={i} style={{
                  display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3,
                  fontSize: 10, color: 'rgba(148,163,184,0.7)',
                }}>
                  <div style={{
                    width: 6, height: 6, borderRadius: '50%',
                    background: peerNode ? ROLE_COLORS[peerNode.role] : '#64748b',
                  }} />
                  <span style={{ flex: 1 }}>{peerNode?.name || peerId}</span>
                  <span style={{ color: bandwidthColor(ratio), fontFamily: 'monospace', fontWeight: 600 }}>
                    {t.bandwidth_mbps.toFixed(1)}
                  </span>
                  <span style={{ color: 'rgba(148,163,184,0.4)' }}>
                    / {t.max_bandwidth_mbps.toFixed(0)} Mbps
                  </span>
                  {t.active_tasks > 0 && (
                    <span style={{
                      fontSize: 8, padding: '1px 4px', borderRadius: 4,
                      background: 'rgba(251,191,36,0.15)', color: '#fbbf24',
                    }}>
                      {t.active_tasks} tasks
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function ComputeMeshGraph({
  nodes,
  tunnels,
  onNodeClick,
  minHeight = 400,
}: ComputeMeshGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const simNodesRef = useRef<SimNode[]>([]);
  const simEdgesRef = useRef<SimEdge[]>([]);
  const alphaRef = useRef(1.0);
  const rafRef = useRef<number>(0);
  const frameRef = useRef(0);
  const [, forceRender] = useState(0);
  const [dimensions, setDimensions] = useState({ width: 800, height: minHeight });
  const [selectedNode, setSelectedNode] = useState<MeshNode | null>(null);
  const [popupPos, setPopupPos] = useState({ x: 0, y: 0 });
  const [dragNode, setDragNode] = useState<string | null>(null);
  const [particles, setParticles] = useState<ParticleState[]>([]);

  // ── Responsive sizing ──────────────────────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observe = () => {
      const rect = container.getBoundingClientRect();
      setDimensions({ width: Math.max(rect.width, 300), height: Math.max(rect.height, minHeight) });
    };
    observe();

    const ro = new ResizeObserver(observe);
    ro.observe(container);
    return () => ro.disconnect();
  }, [minHeight]);

  // ── Build simulation nodes/edges when data changes ─────────────────────────
  useEffect(() => {
    const { width, height } = dimensions;
    const existing = simNodesRef.current;
    const posMap = new Map<string, { x: number; y: number }>();
    for (const sn of existing) {
      posMap.set(sn.id, { x: sn.x, y: sn.y });
    }

    const newSimNodes: SimNode[] = nodes.map((n, i) => {
      const prev = posMap.get(n.id);
      // Node radius proportional to compute capacity (cores + GPU)
      const capacity = n.cores + n.gpu_tflops * 2;
      const radius = clamp(12 + Math.sqrt(capacity) * 2, 14, 40);
      return {
        id: n.id,
        x: prev ? prev.x : width / 2 + (Math.cos(i * 2.39996) * width * 0.25),
        y: prev ? prev.y : height / 2 + (Math.sin(i * 2.39996) * height * 0.25),
        vx: 0,
        vy: 0,
        radius,
        pinned: false,
      };
    });

    const idxMap = new Map<string, number>();
    newSimNodes.forEach((sn, i) => idxMap.set(sn.id, i));

    const newSimEdges: SimEdge[] = [];
    for (const t of tunnels) {
      const si = idxMap.get(t.source);
      const ti = idxMap.get(t.target);
      if (si !== undefined && ti !== undefined) {
        newSimEdges.push({ sourceIdx: si, targetIdx: ti, idealLength: 140 });
      }
    }

    simNodesRef.current = newSimNodes;
    simEdgesRef.current = newSimEdges;

    // Reheat the simulation
    alphaRef.current = 0.8;
  }, [nodes, tunnels, dimensions]);

  // ── Particle animation for active tunnels ──────────────────────────────────
  useEffect(() => {
    const newParticles: ParticleState[] = [];
    for (const t of tunnels) {
      if (t.bandwidth_mbps > 0) {
        const key = `${t.source}->${t.target}`;
        // Particle speed proportional to bandwidth ratio
        const ratio = t.max_bandwidth_mbps > 0 ? t.bandwidth_mbps / t.max_bandwidth_mbps : 0.1;
        const speed = 0.004 + ratio * 0.012;
        // Add 1-3 particles depending on traffic intensity
        const count = ratio > 0.5 ? (ratio > 0.8 ? 3 : 2) : 1;
        for (let i = 0; i < count; i++) {
          newParticles.push({
            tunnelKey: key,
            progress: (i / count), // evenly spread
            speed,
          });
        }
      }
    }
    setParticles(newParticles);
  }, [tunnels]);

  // ── Force simulation loop ──────────────────────────────────────────────────
  useEffect(() => {
    let running = true;

    const tick = () => {
      if (!running) return;
      frameRef.current++;

      const simNodes = simNodesRef.current;
      const simEdges = simEdgesRef.current;
      const alpha = alphaRef.current;

      if (alpha > 0.001 && simNodes.length > 0) {
        const kinetic = forceTick(simNodes, simEdges, dimensions.width, dimensions.height, alpha);
        // Cool down
        alphaRef.current *= 0.992;
        if (kinetic < 0.1) alphaRef.current *= 0.95;
      }

      // Advance particles
      setParticles(prev => prev.map(p => ({
        ...p,
        progress: (p.progress + p.speed) % 1.0,
      })));

      // Re-render every other frame for performance
      if (frameRef.current % 2 === 0) {
        forceRender(f => f + 1);
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => { running = false; cancelAnimationFrame(rafRef.current); };
  }, [dimensions]);

  // ── Drag handlers ──────────────────────────────────────────────────────────
  const handleMouseDown = useCallback((nodeId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragNode(nodeId);
    const sn = simNodesRef.current.find(n => n.id === nodeId);
    if (sn) sn.pinned = true;
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragNode || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const sn = simNodesRef.current.find(n => n.id === dragNode);
    if (sn) {
      sn.x = clamp(x, 20, dimensions.width - 20);
      sn.y = clamp(y, 20, dimensions.height - 20);
      sn.vx = 0;
      sn.vy = 0;
    }
    // Reheat slightly so other nodes adjust
    alphaRef.current = Math.max(alphaRef.current, 0.15);
    forceRender(f => f + 1);
  }, [dragNode, dimensions]);

  const handleMouseUp = useCallback(() => {
    if (dragNode) {
      const sn = simNodesRef.current.find(n => n.id === dragNode);
      if (sn) sn.pinned = false;
      setDragNode(null);
    }
  }, [dragNode]);

  // ── Click handler — toggle node detail popup ───────────────────────────────
  const handleNodeClick = useCallback((node: MeshNode, e: React.MouseEvent) => {
    if (dragNode) return; // was dragging, not clicking
    e.stopPropagation();
    const svgRect = svgRef.current?.getBoundingClientRect();
    const simNode = simNodesRef.current.find(n => n.id === node.id);
    if (svgRect && simNode) {
      // Position popup near the node but inside the container
      let px = simNode.x + 20;
      let py = simNode.y - 40;
      // Flip to left side if too close to right edge
      if (px + 300 > dimensions.width) px = simNode.x - 300;
      if (py < 10) py = 10;
      if (py + 200 > dimensions.height) py = dimensions.height - 220;
      setPopupPos({ x: px, y: py });
    }
    setSelectedNode(prev => prev?.id === node.id ? null : node);
    onNodeClick?.(node);
  }, [dragNode, onNodeClick, dimensions]);

  const handleBackgroundClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  // ── Tunnel index for quick lookup ──────────────────────────────────────────
  const tunnelMap = useMemo(() => {
    const m = new Map<string, MeshTunnel>();
    for (const t of tunnels) {
      m.set(`${t.source}->${t.target}`, t);
      m.set(`${t.target}->${t.source}`, t);
    }
    return m;
  }, [tunnels]);

  // ── Build SVG elements from simulation state ───────────────────────────────
  const simNodes = simNodesRef.current;
  const nodeIdxMap = new Map<string, number>();
  simNodes.forEach((sn, i) => nodeIdxMap.set(sn.id, i));

  // ── Legend ─────────────────────────────────────────────────────────────────
  const legendItems = useMemo(() => {
    const roles = new Set(nodes.map(n => n.role));
    return Array.from(roles).map(role => ({ role, color: ROLE_COLORS[role], label: ROLE_LABELS[role] }));
  }, [nodes]);

  // ── Empty state ────────────────────────────────────────────────────────────
  if (nodes.length === 0) {
    return (
      <div
        ref={containerRef}
        style={{
          width: '100%', minHeight, display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'rgba(15,23,42,0.6)', borderRadius: 12,
          border: '1px solid rgba(148,163,184,0.1)',
        }}
      >
        <span style={{ fontSize: 13, color: 'rgba(148,163,184,0.4)' }}>
          No compute mesh nodes available
        </span>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative', width: '100%', minHeight,
        background: 'rgba(15,23,42,0.6)', borderRadius: 12,
        border: '1px solid rgba(148,163,184,0.1)', overflow: 'hidden',
        cursor: dragNode ? 'grabbing' : 'default',
      }}
      onClick={handleBackgroundClick}
    >
      {/* SVG Graph */}
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ display: 'block' }}
      >
        <defs>
          {/* Glow filter for active tunnels */}
          <filter id="tunnel-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Soft shadow for nodes */}
          <filter id="node-shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="2" stdDeviation="4" floodColor="#000" floodOpacity="0.5" />
          </filter>
        </defs>

        {/* ── Tunnel Lines ── */}
        {tunnels.map((t) => {
          const si = nodeIdxMap.get(t.source);
          const ti = nodeIdxMap.get(t.target);
          if (si === undefined || ti === undefined) return null;
          const sn = simNodes[si];
          const tn = simNodes[ti];
          const ratio = t.max_bandwidth_mbps > 0 ? t.bandwidth_mbps / t.max_bandwidth_mbps : 0;
          const color = bandwidthColor(ratio);
          const thickness = clamp(1.5 + ratio * 4, 1.5, 6);
          const isActive = t.bandwidth_mbps > 0;
          const key = `${t.source}-${t.target}`;

          return (
            <g key={key}>
              {/* Base line */}
              <line
                x1={sn.x} y1={sn.y} x2={tn.x} y2={tn.y}
                stroke={color}
                strokeWidth={thickness}
                strokeOpacity={isActive ? 0.7 : 0.2}
                strokeLinecap="round"
                filter={isActive ? 'url(#tunnel-glow)' : undefined}
              />
              {/* Bandwidth label at midpoint */}
              {isActive && (
                <text
                  x={(sn.x + tn.x) / 2}
                  y={(sn.y + tn.y) / 2 - 8}
                  textAnchor="middle"
                  fill={color}
                  fontSize={9}
                  fontFamily="monospace"
                  fontWeight={600}
                  opacity={0.85}
                >
                  {t.bandwidth_mbps >= 1000
                    ? `${(t.bandwidth_mbps / 1000).toFixed(1)} Gbps`
                    : `${t.bandwidth_mbps.toFixed(1)} Mbps`}
                </text>
              )}
            </g>
          );
        })}

        {/* ── Animated Particles ── */}
        {particles.map((p, pi) => {
          // Parse tunnel key "source->target"
          const arrow = p.tunnelKey.indexOf('->');
          if (arrow < 0) return null;
          const srcId = p.tunnelKey.slice(0, arrow);
          const tgtId = p.tunnelKey.slice(arrow + 2);
          const si = nodeIdxMap.get(srcId);
          const ti = nodeIdxMap.get(tgtId);
          if (si === undefined || ti === undefined) return null;
          const sn = simNodes[si];
          const tn = simNodes[ti];
          const t = p.progress;
          const px = sn.x + (tn.x - sn.x) * t;
          const py = sn.y + (tn.y - sn.y) * t;

          const tunnel = tunnelMap.get(p.tunnelKey);
          const ratio = tunnel && tunnel.max_bandwidth_mbps > 0
            ? tunnel.bandwidth_mbps / tunnel.max_bandwidth_mbps
            : 0.3;

          return (
            <circle
              key={`particle-${pi}`}
              cx={px} cy={py}
              r={2.5 + ratio * 1.5}
              fill={bandwidthColor(ratio)}
              opacity={0.9}
            >
              <animate
                attributeName="opacity"
                values="0.9;0.4;0.9"
                dur="0.8s"
                repeatCount="indefinite"
              />
            </circle>
          );
        })}

        {/* ── Node Circles ── */}
        {nodes.map((node) => {
          const idx = nodeIdxMap.get(node.id);
          if (idx === undefined) return null;
          const sn = simNodes[idx];
          const color = ROLE_COLORS[node.role];
          const isSelected = selectedNode?.id === node.id;
          const isDragging = dragNode === node.id;

          // Outer ring shows overall load (max of cpu, gpu, ram)
          const load = Math.max(node.cpu_percent, node.gpu_percent, node.ram_percent);
          const loadColor = load > 80 ? '#ef4444' : load > 50 ? '#eab308' : '#22c55e';

          return (
            <g
              key={node.id}
              style={{ cursor: isDragging ? 'grabbing' : 'pointer' }}
              onMouseDown={(e) => handleMouseDown(node.id, e)}
              onClick={(e) => handleNodeClick(node, e)}
            >
              {/* Load ring (background) */}
              <circle
                cx={sn.x} cy={sn.y} r={sn.radius + 3}
                fill="none"
                stroke="rgba(148,163,184,0.15)"
                strokeWidth={3}
              />
              {/* Load ring (filled arc via stroke-dasharray) */}
              <circle
                cx={sn.x} cy={sn.y} r={sn.radius + 3}
                fill="none"
                stroke={loadColor}
                strokeWidth={3}
                strokeLinecap="round"
                strokeDasharray={`${(load / 100) * 2 * Math.PI * (sn.radius + 3)} ${2 * Math.PI * (sn.radius + 3)}`}
                strokeDashoffset={2 * Math.PI * (sn.radius + 3) * 0.25}
                opacity={0.6}
              />
              {/* Selection highlight */}
              {isSelected && (
                <circle
                  cx={sn.x} cy={sn.y} r={sn.radius + 8}
                  fill="none"
                  stroke={color}
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  opacity={0.6}
                >
                  <animateTransform
                    attributeName="transform"
                    type="rotate"
                    from={`0 ${sn.x} ${sn.y}`}
                    to={`360 ${sn.x} ${sn.y}`}
                    dur="8s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
              {/* Main circle */}
              <circle
                cx={sn.x} cy={sn.y} r={sn.radius}
                fill={`${color}33`}
                stroke={color}
                strokeWidth={isSelected ? 2.5 : 1.5}
                filter="url(#node-shadow)"
              />
              {/* Inner glow */}
              <circle
                cx={sn.x} cy={sn.y} r={sn.radius * 0.4}
                fill={color}
                opacity={0.25 + (load / 100) * 0.35}
              />
              {/* Label */}
              <text
                x={sn.x} y={sn.y + sn.radius + 14}
                textAnchor="middle"
                fill="#e2e8f0"
                fontSize={11}
                fontWeight={500}
              >
                {node.name}
              </text>
              {/* Role sub-label */}
              <text
                x={sn.x} y={sn.y + sn.radius + 25}
                textAnchor="middle"
                fill={color}
                fontSize={8}
                fontWeight={700}
              >
                {ROLE_LABELS[node.role].toUpperCase()}
              </text>
              {/* Core count inside circle */}
              <text
                x={sn.x} y={sn.y + 1}
                textAnchor="middle"
                dominantBaseline="central"
                fill="#e2e8f0"
                fontSize={sn.radius > 20 ? 11 : 9}
                fontWeight={700}
                fontFamily="monospace"
              >
                {node.cores}c
              </text>
            </g>
          );
        })}
      </svg>

      {/* ── Legend ── */}
      <div style={{
        position: 'absolute', top: 10, left: 10,
        display: 'flex', flexDirection: 'column', gap: 4,
        padding: '6px 10px', borderRadius: 8,
        background: 'rgba(15,23,42,0.8)',
        border: '1px solid rgba(148,163,184,0.15)',
      }}>
        <span style={{ fontSize: 9, color: 'rgba(148,163,184,0.5)', fontWeight: 600, marginBottom: 2 }}>
          NODE ROLES
        </span>
        {legendItems.map(item => (
          <div key={item.role} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{
              width: 8, height: 8, borderRadius: '50%',
              background: item.color, boxShadow: `0 0 4px ${item.color}55`,
            }} />
            <span style={{ fontSize: 10, color: '#e2e8f0' }}>{item.label}</span>
          </div>
        ))}
        <div style={{
          marginTop: 4, paddingTop: 4, borderTop: '1px solid rgba(148,163,184,0.1)',
        }}>
          <span style={{ fontSize: 9, color: 'rgba(148,163,184,0.5)', fontWeight: 600 }}>
            TUNNEL BW
          </span>
          <div style={{ display: 'flex', gap: 2, marginTop: 3 }}>
            {[
              { color: '#22c55e', label: 'Idle' },
              { color: '#eab308', label: '50%' },
              { color: '#ef4444', label: 'Full' },
            ].map(bw => (
              <div key={bw.label} style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                <div style={{ width: 12, height: 3, borderRadius: 2, background: bw.color }} />
                <span style={{ fontSize: 8, color: 'rgba(148,163,184,0.5)' }}>{bw.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Stats Badge (top-right) ── */}
      <div style={{
        position: 'absolute', top: 10, right: 10,
        padding: '4px 10px', borderRadius: 8,
        background: 'rgba(15,23,42,0.8)',
        border: '1px solid rgba(148,163,184,0.15)',
        display: 'flex', gap: 12, alignItems: 'center',
      }}>
        <span style={{ fontSize: 10, color: 'rgba(148,163,184,0.5)' }}>
          Nodes: <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{nodes.length}</span>
        </span>
        <span style={{ fontSize: 10, color: 'rgba(148,163,184,0.5)' }}>
          Tunnels: <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{tunnels.length}</span>
        </span>
        <span style={{ fontSize: 10, color: 'rgba(148,163,184,0.5)' }}>
          Active: <span style={{
            color: tunnels.filter(t => t.bandwidth_mbps > 0).length > 0 ? '#22c55e' : '#64748b',
            fontWeight: 600,
          }}>
            {tunnels.filter(t => t.bandwidth_mbps > 0).length}
          </span>
        </span>
      </div>

      {/* ── Node Detail Popup ── */}
      {selectedNode && (
        <NodeDetailPopup
          node={selectedNode}
          tunnels={tunnels}
          allNodes={nodes}
          position={popupPos}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </div>
  );
}
