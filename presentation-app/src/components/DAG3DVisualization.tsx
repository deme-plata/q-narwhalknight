import { useEffect, useRef, useState } from 'react';

interface DAGNode {
  id: string;
  round: number;
  x: number;
  y: number;
  z: number;
  velocity: { x: number; y: number; z: number };
  hue: number;
  radius: number;
  transactions: number;
  parents: string[];
}

interface DAGEdge {
  from: string;
  to: string;
  strength: number;
  phase: number;
}

export function DAG3DVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nodes, setNodes] = useState<DAGNode[]>([]);
  const [edges, setEdges] = useState<DAGEdge[]>([]);
  const rotationRef = useRef({ x: 0.3, y: 0 });
  const animationRef = useRef<number | undefined>(undefined);

  // Initialize DAG structure
  useEffect(() => {
    const initialNodes: DAGNode[] = [];
    const initialEdges: DAGEdge[] = [];

    // Genesis node (center)
    initialNodes.push({
      id: 'genesis',
      round: 0,
      x: 0,
      y: 0,
      z: 0,
      velocity: { x: 0, y: 0, z: 0 },
      hue: 280, // Magenta for genesis
      radius: 20,
      transactions: 0,
      parents: [],
    });

    // Create 5 rounds of DAG structure
    for (let round = 1; round <= 5; round++) {
      const nodesInRound = 3 + Math.floor(Math.random() * 2); // 3-4 nodes per round

      for (let i = 0; i < nodesInRound; i++) {
        const angle = (i / nodesInRound) * Math.PI * 2;
        const radius = 150 + round * 80;
        const height = round * 100 - 250;

        const nodeId = `r${round}-n${i}`;
        const txCount = Math.floor(Math.random() * 50) + 10;

        initialNodes.push({
          id: nodeId,
          round,
          x: Math.cos(angle) * radius,
          y: height,
          z: Math.sin(angle) * radius,
          velocity: { x: 0, y: 0, z: 0 },
          hue: (round * 60 + i * 30) % 360,
          radius: 12 + txCount / 10,
          transactions: txCount,
          parents: [],
        });

        // Connect to 2-3 parents from previous round
        const prevRoundNodes = initialNodes.filter(n => n.round === round - 1);
        const numParents = Math.min(2 + Math.floor(Math.random() * 2), prevRoundNodes.length);

        for (let p = 0; p < numParents; p++) {
          const parentIdx = Math.floor(Math.random() * prevRoundNodes.length);
          const parent = prevRoundNodes[parentIdx];

          if (!initialNodes[initialNodes.length - 1].parents.includes(parent.id)) {
            initialNodes[initialNodes.length - 1].parents.push(parent.id);

            initialEdges.push({
              from: parent.id,
              to: nodeId,
              strength: 0.3 + Math.random() * 0.7,
              phase: Math.random() * Math.PI * 2,
            });
          }
        }
      }
    }

    setNodes(initialNodes);
    setEdges(initialEdges);
  }, []);

  // Spring-force physics simulation
  useEffect(() => {
    if (nodes.length === 0) return;

    const applyForces = () => {
      setNodes(prevNodes => {
        const newNodes = [...prevNodes];
        const forces: { [key: string]: { x: number; y: number; z: number } } = {};

        // Initialize forces
        newNodes.forEach(node => {
          forces[node.id] = { x: 0, y: 0, z: 0 };
        });

        // Repulsion between all nodes
        for (let i = 0; i < newNodes.length; i++) {
          for (let j = i + 1; j < newNodes.length; j++) {
            const node1 = newNodes[i];
            const node2 = newNodes[j];

            const dx = node1.x - node2.x;
            const dy = node1.y - node2.y;
            const dz = node1.z - node2.z;
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

            if (distance > 0) {
              const repulsion = 5000 / (distance * distance);
              const fx = (dx / distance) * repulsion;
              const fy = (dy / distance) * repulsion;
              const fz = (dz / distance) * repulsion;

              forces[node1.id].x += fx;
              forces[node1.id].y += fy;
              forces[node1.id].z += fz;
              forces[node2.id].x -= fx;
              forces[node2.id].y -= fy;
              forces[node2.id].z -= fz;
            }
          }
        }

        // Attraction along edges
        edges.forEach(edge => {
          const fromNode = newNodes.find(n => n.id === edge.from);
          const toNode = newNodes.find(n => n.id === edge.to);

          if (fromNode && toNode) {
            const dx = toNode.x - fromNode.x;
            const dy = toNode.y - fromNode.y;
            const dz = toNode.z - fromNode.z;
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

            if (distance > 0) {
              const idealLength = 100;
              const spring = 0.02 * (distance - idealLength) * edge.strength;
              const fx = (dx / distance) * spring;
              const fy = (dy / distance) * spring;
              const fz = (dz / distance) * spring;

              forces[fromNode.id].x += fx;
              forces[fromNode.id].y += fy;
              forces[fromNode.id].z += fz;
              forces[toNode.id].x -= fx;
              forces[toNode.id].y -= fy;
              forces[toNode.id].z -= fz;
            }
          }
        });

        // Apply forces and update positions (skip genesis)
        newNodes.forEach(node => {
          if (node.id === 'genesis') return;

          const force = forces[node.id];
          const damping = 0.85;

          node.velocity.x = node.velocity.x * damping + force.x;
          node.velocity.y = node.velocity.y * damping + force.y;
          node.velocity.z = node.velocity.z * damping + force.z;

          node.x += node.velocity.x;
          node.y += node.velocity.y;
          node.z += node.velocity.z;

          // Keep in bounds
          const maxDist = 600;
          const dist = Math.sqrt(node.x * node.x + node.y * node.y + node.z * node.z);
          if (dist > maxDist) {
            node.x = (node.x / dist) * maxDist;
            node.y = (node.y / dist) * maxDist;
            node.z = (node.z / dist) * maxDist;
          }
        });

        return newNodes;
      });
    };

    // Run physics every 50ms
    const physicsInterval = setInterval(applyForces, 50);

    return () => clearInterval(physicsInterval);
  }, [nodes.length, edges]);

  // 3D rendering with rotation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;

    const render = () => {
      // Clear with dark background
      ctx.fillStyle = 'rgba(5, 7, 20, 0.95)';
      ctx.fillRect(0, 0, width, height);

      // Auto-rotate
      rotationRef.current.y += 0.003;

      // Project 3D to 2D
      const projected = nodes.map(node => {
        // Apply rotation
        const cosY = Math.cos(rotationRef.current.y);
        const sinY = Math.sin(rotationRef.current.y);
        const cosX = Math.cos(rotationRef.current.x);
        const sinX = Math.sin(rotationRef.current.x);

        // Rotate Y
        let x = node.x * cosY - node.z * sinY;
        let z = node.x * sinY + node.z * cosY;
        let y = node.y;

        // Rotate X
        const y2 = y * cosX - z * sinX;
        const z2 = y * sinX + z * cosX;

        // Perspective projection
        const perspective = 1200;
        const scale = perspective / (perspective + z2);
        const x2d = centerX + x * scale;
        const y2d = centerY + y2 * scale;

        return {
          node,
          x: x2d,
          y: y2d,
          z: z2,
          scale,
        };
      });

      // Sort by depth (back to front)
      projected.sort((a, b) => a.z - b.z);

      // Draw edges
      ctx.lineWidth = 2;
      edges.forEach(edge => {
        const from = projected.find(p => p.node.id === edge.from);
        const to = projected.find(p => p.node.id === edge.to);

        if (from && to && from.z < 500 && to.z < 500) {
          // Gradient based on edge strength
          const gradient = ctx.createLinearGradient(from.x, from.y, to.x, to.y);
          const hue = (edge.phase / (Math.PI * 2)) * 360;
          const alpha = Math.min(from.scale, to.scale) * edge.strength * 0.6;

          gradient.addColorStop(0, `hsla(${hue}, 80%, 60%, ${alpha})`);
          gradient.addColorStop(1, `hsla(${(hue + 60) % 360}, 80%, 60%, ${alpha})`);

          ctx.strokeStyle = gradient;
          ctx.lineWidth = edge.strength * 3;

          // Draw wavy edge for quantum interference
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);

          const steps = 20;
          for (let i = 1; i <= steps; i++) {
            const t = i / steps;
            const x = from.x + (to.x - from.x) * t;
            const y = from.y + (to.y - from.y) * t;

            // Add wave perturbation
            const wave = Math.sin(edge.phase + t * Math.PI * 4) * edge.strength * 5;
            const dx = to.y - from.y;
            const dy = -(to.x - from.x);
            const len = Math.sqrt(dx * dx + dy * dy);

            if (len > 0) {
              ctx.lineTo(x + (dx / len) * wave, y + (dy / len) * wave);
            } else {
              ctx.lineTo(x, y);
            }
          }

          ctx.stroke();
        }
      });

      // Draw nodes
      projected.forEach(({ node, x, y, z, scale }) => {
        if (z > 500) return; // Too far away

        const alpha = Math.min(scale, 1.0);
        const radius = node.radius * scale;

        // Entanglement halo
        const haloRadius = radius * 2.5;
        const haloGradient = ctx.createRadialGradient(x, y, radius, x, y, haloRadius);
        haloGradient.addColorStop(0, `hsla(${node.hue}, 70%, 50%, ${alpha * 0.3})`);
        haloGradient.addColorStop(1, `hsla(${node.hue}, 70%, 50%, 0)`);

        ctx.fillStyle = haloGradient;
        ctx.beginPath();
        ctx.arc(x, y, haloRadius, 0, Math.PI * 2);
        ctx.fill();

        // Core node
        const coreGradient = ctx.createRadialGradient(
          x - radius * 0.3, y - radius * 0.3, radius * 0.1,
          x, y, radius
        );
        coreGradient.addColorStop(0, `hsla(${node.hue}, 90%, 80%, ${alpha})`);
        coreGradient.addColorStop(1, `hsla(${node.hue}, 80%, 50%, ${alpha})`);

        ctx.fillStyle = coreGradient;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();

        // Border
        ctx.strokeStyle = `hsla(0, 0%, 100%, ${alpha * 0.8})`;
        ctx.lineWidth = 2 * scale;
        ctx.stroke();

        // Label
        if (scale > 0.6) {
          ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.9})`;
          ctx.font = `${Math.max(10, 12 * scale)}px 'Courier New', monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';

          if (node.id === 'genesis') {
            ctx.fillText('⚓', x, y);
          } else {
            ctx.fillText(`R${node.round}`, x, y);
          }
        }
      });

      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [nodes, edges]);

  return (
    <div style={{
      width: '100%',
      height: '600px',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      background: 'linear-gradient(135deg, #000714 0%, #0a0e27 100%)',
      borderRadius: '12px',
      border: '2px solid rgba(0, 255, 255, 0.3)',
      boxShadow: '0 0 30px rgba(0, 255, 255, 0.2)',
      position: 'relative',
      overflow: 'hidden',
    }}>
      <canvas
        ref={canvasRef}
        width={1600}
        height={600}
        style={{
          width: '100%',
          height: '100%',
          imageRendering: 'crisp-edges',
        }}
      />

      <div style={{
        position: 'absolute',
        bottom: '15px',
        left: '20px',
        color: 'rgba(0, 255, 255, 0.7)',
        fontSize: '14px',
        fontFamily: "'Courier New', monospace",
        textShadow: '0 0 10px rgba(0, 255, 255, 0.5)',
      }}>
        🔮 Quantum DAG-Knight Consensus Structure
      </div>

      <div style={{
        position: 'absolute',
        top: '15px',
        right: '20px',
        color: 'rgba(0, 255, 255, 0.7)',
        fontSize: '12px',
        fontFamily: "'Courier New', monospace",
        textAlign: 'right',
        lineHeight: '1.6',
      }}>
        <div>⚓ Genesis (R0)</div>
        <div>🔵 Vertices (R1-R5)</div>
        <div>🌊 Quantum Entanglement</div>
        <div>⚡ Spring-Force Layout</div>
      </div>
    </div>
  );
}
