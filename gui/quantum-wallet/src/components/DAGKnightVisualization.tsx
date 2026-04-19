import { useEffect, useRef, useState, useCallback } from 'react';
import { Activity, Zap, TrendingUp, GitBranch, X, Sparkles, Layers, Orbit, Box } from 'lucide-react';

interface DAGBlock {
  id: string; // block hash
  height: number;
  lane: number;
  timestamp: number;
  parents: string[];
  isBlueSet: boolean;
  x: number;
  y: number; // Use actual Y position for curved paths
  miner?: string;
  txCount: number;
  reward: number;
  prevHash?: string;
  totalDifficulty?: number;
  dagRound?: number;
  minerCount?: number;
  age?: number; // Age in milliseconds since creation (for entrance animation)
  producerId?: number;
  hashPrefix?: string; // First 8 chars of hash for coloring
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  color: string;
  size: number;
}

interface DAGKnightVisualizationProps {
  currentHeight: number;
}

// Hash function to generate consistent lane from block hash
const hashToLane = (hash: string, numLanes: number): number => {
  if (!hash || hash.length < 8) return 0;
  // Use first 8 chars of hash as hex, convert to number
  const hashNum = parseInt(hash.substring(0, 8), 16);
  return Math.abs(hashNum) % numLanes;
};

// Generate color from block hash for variety
const hashToColor = (hash: string): string => {
  if (!hash || hash.length < 6) return '#60a5fa';
  const r = parseInt(hash.substring(0, 2), 16);
  const g = parseInt(hash.substring(2, 4), 16);
  const b = parseInt(hash.substring(4, 6), 16);
  // Ensure colors are vibrant (shift towards brighter)
  const minBrightness = 100;
  return `rgb(${Math.max(r, minBrightness)}, ${Math.max(g, minBrightness)}, ${Math.max(b, minBrightness)})`;
};

export default function DAGKnightVisualization({ currentHeight }: DAGKnightVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  console.log('DAG Visualizer initialized at height:', currentHeight);
  const [blocks, setBlocks] = useState<DAGBlock[]>([]);
  const [selectedBlock, setSelectedBlock] = useState<DAGBlock | null>(null);
  const [stats, setStats] = useState({
    blueSetCount: 0,
    redSetCount: 0,
    totalBlocks: 0,
    blocksPerSecond: 0,
  });
  const [visualMode, setVisualMode] = useState<'flow' | 'quantum' | 'constellation' | 'matrix'>('quantum');
  const [isConnected, setIsConnected] = useState(false);

  const scrollOffset = useRef(0);
  const lastBlockTime = useRef(Date.now());
  const animationFrameId = useRef<number | undefined>(undefined);
  const laneOccupancy = useRef<Map<number, number>>(new Map()); // lane -> rightmost x position
  // v6.0.3: Use ref for particles to avoid infinite re-render loop in animation useEffect
  // Previously particles was state AND in the animation effect's deps, causing Error #185
  const particlesRef = useRef<Particle[]>([]);

  // Configuration - 4 lanes for better visual density
  const BLOCK_SIZE = 48; // Square blocks
  const BLOCK_RADIUS = 8; // Rounded corners
  const LANE_HEIGHT = 80;
  const NUM_LANES = 4;
  const SCROLL_SPEED = 150;
  const MIN_BLOCK_SPACING = 120;
  const CANVAS_HEIGHT = NUM_LANES * LANE_HEIGHT + 120;
  const OFFSCREEN_BUFFER = 800; // Keep blocks further off-screen for connection continuity
  const MAX_VISIBLE_BLOCKS = 80; // Increased from 50 to maintain more connections

  // Color palette for lanes
  const LANE_COLORS = [
    { primary: '#60a5fa', secondary: '#3b82f6', glow: 'rgba(96, 165, 250, 0.6)' },  // Blue
    { primary: '#a78bfa', secondary: '#8b5cf6', glow: 'rgba(167, 139, 250, 0.6)' }, // Purple
    { primary: '#34d399', secondary: '#10b981', glow: 'rgba(52, 211, 153, 0.6)' },  // Green
    { primary: '#f472b6', secondary: '#ec4899', glow: 'rgba(244, 114, 182, 0.6)' }, // Pink
  ];

  // Create quantum particles around a block — enhanced with firework burst
  const createParticles = useCallback((x: number, y: number, color: string, count: number = 15) => {
    const newParticles: Particle[] = [];
    // Main burst ring
    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 * i) / count + Math.random() * 0.3;
      const speed = 50 + Math.random() * 80;
      newParticles.push({
        x, y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 0,
        maxLife: 700 + Math.random() * 500,
        color,
        size: 2 + Math.random() * 4,
      });
    }
    // Secondary sparkle ring (smaller, faster, gold)
    for (let i = 0; i < 8; i++) {
      const angle = (Math.PI * 2 * i) / 8 + Math.random() * 0.8;
      const speed = 80 + Math.random() * 120;
      newParticles.push({
        x, y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 30,
        life: 0,
        maxLife: 400 + Math.random() * 200,
        color: '#FFD700',
        size: 1 + Math.random() * 2,
      });
    }
    // Quantum metadata trail sparks (tiny, long-lived, drift upward)
    for (let i = 0; i < 5; i++) {
      newParticles.push({
        x: x + (Math.random() - 0.5) * 20,
        y: y + (Math.random() - 0.5) * 10,
        vx: (Math.random() - 0.5) * 15,
        vy: -20 - Math.random() * 40,
        life: 0,
        maxLife: 1200 + Math.random() * 800,
        color: '#00E5FF',
        size: 1 + Math.random() * 1.5,
      });
    }
    particlesRef.current = [...particlesRef.current.slice(-150), ...newParticles];
  }, []);

  // Listen for new blocks via SSE
  useEffect(() => {
    console.log('🎨 DAG Visualization starting, connecting to SSE stream...');

    const eventSource = new EventSource('/api/v1/events');

    eventSource.onopen = () => {
      console.log('🎨 SSE connection opened');
      setIsConnected(true);
    };

    eventSource.addEventListener('new-block', (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('🎨 DAGKnight: Received new-block SSE event:', data);

        // Handle tagged enum format: {type: "NewBlock", data: {...}}
        const blockData = data.data || data;
        const blockHash = blockData.hash || `block-${blockData.height}`;

        const now = Date.now();
        const timeSinceLastBlock = (now - lastBlockTime.current) / 1000;

        // Use block hash for lane assignment (creates visual variety)
        const assignedLane = hashToLane(blockHash, NUM_LANES);
        const laneColor = LANE_COLORS[assignedLane];

        // Calculate X position with collision avoidance
        const canvasWidth = canvasRef.current?.width || 1200;
        const frontierX = scrollOffset.current + canvasWidth - 150;

        // Check if there's already a block in this lane recently
        const laneLastX = laneOccupancy.current.get(assignedLane) || 0;
        const minRequiredX = laneLastX + MIN_BLOCK_SPACING;

        // Position block at frontier or further right if lane is occupied
        const blockX = Math.max(frontierX, minRequiredX);

        // Add slight Y variation within the lane for organic feel
        const laneY = assignedLane * LANE_HEIGHT + 60;
        const yVariation = (Math.sin(blockData.height * 0.7) * 8);
        const blockY = laneY + yVariation;

        // Update lane occupancy tracking
        laneOccupancy.current.set(assignedLane, blockX);

        const newBlock: DAGBlock = {
          id: blockHash,
          height: blockData.height,
          lane: assignedLane,
          timestamp: now,
          parents: blockData.prev_hash ? [blockData.prev_hash] : (blockData.height > 0 ? [`block-${blockData.height - 1}`] : []),
          isBlueSet: true,
          x: blockX,
          y: blockY,
          miner: `Block Producer`,
          txCount: blockData.tx_count || blockData.solutions_count || 0,
          reward: blockData.block_reward || 0,
          prevHash: blockData.prev_hash,
          totalDifficulty: blockData.total_difficulty,
          dagRound: blockData.dag_round,
          minerCount: blockData.miner_count,
          age: 0,
          producerId: blockData.producer_id || 0,
          hashPrefix: blockHash.substring(0, 8),
        };

        setBlocks(prev => {
          const updated = [...prev, newBlock];
          // Keep more blocks for better connection continuity
          // Use larger off-screen buffer to maintain parent-child connections
          const filtered = updated.filter(b => b.x > scrollOffset.current - OFFSCREEN_BUFFER).slice(-MAX_VISIBLE_BLOCKS);

          // Update lane occupancy for visible blocks
          const visibleLaneMaxX = new Map<number, number>();
          filtered.forEach(block => {
            const currentMax = visibleLaneMaxX.get(block.lane) || 0;
            visibleLaneMaxX.set(block.lane, Math.max(currentMax, block.x));
          });
          laneOccupancy.current = visibleLaneMaxX;

          return filtered;
        });

        // Create particles for new block
        if (visualMode === 'quantum' || visualMode === 'constellation') {
          const blockScreenX = blockX - scrollOffset.current + BLOCK_SIZE / 2;
          const blockScreenY = blockY + BLOCK_SIZE / 2;
          createParticles(blockScreenX, blockScreenY, laneColor.primary, 20);
        }

        // Update stats
        setStats(prevStats => ({
          ...prevStats,
          totalBlocks: blockData.height,
          blueSetCount: blockData.height,
          redSetCount: 0,
          blocksPerSecond: timeSinceLastBlock > 0 ? Math.min(1 / timeSinceLastBlock, 10) : 0,
        }));

        lastBlockTime.current = now;
      } catch (error) {
        console.error('Error processing new-block event:', error);
      }
    });

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      setIsConnected(false);
    };

    return () => {
      console.log('Closing SSE connection');
      eventSource.close();
    };
  }, [visualMode, createParticles]);

  // Handle canvas click to select blocks
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const clickX = (event.clientX - rect.left) * scaleX;
    const clickY = (event.clientY - rect.top) * scaleY;

    // Find clicked block
    for (const block of blocks) {
      const blockX = block.x - scrollOffset.current;
      const blockY = block.y;

      if (
        clickX >= blockX &&
        clickX <= blockX + BLOCK_SIZE &&
        clickY >= blockY &&
        clickY <= blockY + BLOCK_SIZE
      ) {
        setSelectedBlock(block);
        return;
      }
    }

    setSelectedBlock(null);
  };

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let lastTime = Date.now();

    const animate = () => {
      const now = Date.now();
      const deltaTime = (now - lastTime) / 1000;
      lastTime = now;

      // Scroll the view
      scrollOffset.current += SCROLL_SPEED * deltaTime;

      // Update block ages
      blocks.forEach(block => {
        if (block.age !== undefined) {
          block.age += deltaTime * 1000;
        }
      });

      // Update particles (mutate ref directly, no setState to avoid re-render loop)
      particlesRef.current = particlesRef.current
          .map(p => ({
            ...p,
            x: p.x + p.vx * deltaTime,
            y: p.y + p.vy * deltaTime,
            life: p.life + deltaTime * 1000,
            vx: p.vx * 0.96,
            vy: p.vy * 0.96,
          }))
          .filter(p => p.life < p.maxLife);

      // Clear canvas with gradient background
      const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, '#0a0a1e');
      gradient.addColorStop(0.5, '#0d0d24');
      gradient.addColorStop(1, '#0a0a1e');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw animated background grid
      ctx.strokeStyle = 'rgba(139, 92, 246, 0.05)';
      ctx.lineWidth = 1;
      const gridOffset = (scrollOffset.current * 0.3) % 40;
      for (let x = -gridOffset; x < canvas.width + 40; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      // Draw lane guides with glow
      for (let i = 0; i < NUM_LANES; i++) {
        const laneY = i * LANE_HEIGHT + 60 + BLOCK_SIZE / 2;
        const laneColor = LANE_COLORS[i];

        // Glowing lane line
        ctx.strokeStyle = laneColor.glow.replace('0.6', '0.15');
        ctx.lineWidth = 2;
        ctx.setLineDash([10, 20]);
        ctx.beginPath();
        ctx.moveTo(0, laneY);
        ctx.lineTo(canvas.width, laneY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Lane label with glow
        ctx.shadowBlur = 10;
        ctx.shadowColor = laneColor.primary;
        ctx.fillStyle = laneColor.primary;
        ctx.font = 'bold 10px Inter, system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Lane ${i}`, 12, laneY - 25);
        ctx.shadowBlur = 0;
      }

      // Sort blocks by height for proper connection drawing
      const sortedBlocks = [...blocks].sort((a, b) => a.height - b.height);

      // Build a map of blocks by height for efficient lookup
      const blocksByHeight = new Map<number, DAGBlock>();
      sortedBlocks.forEach(block => {
        blocksByHeight.set(block.height, block);
      });

      // Draw connections between blocks (DAG edges)
      // Connect each block to its parent (previous height) AND add cross-lane DAG connections
      sortedBlocks.forEach((block) => {
        const blockCenterX = block.x - scrollOffset.current + BLOCK_SIZE / 2;
        const blockCenterY = block.y + BLOCK_SIZE / 2;

        // Skip if block is too far off-screen (but allow some buffer for connections)
        if (blockCenterX < -300 || blockCenterX > canvas.width + 300) return;

        const laneColor = LANE_COLORS[block.lane];

        // Find parent by height (block with height - 1)
        const parent = blocksByHeight.get(block.height - 1);

        if (parent) {
          const parentCenterX = parent.x - scrollOffset.current + BLOCK_SIZE / 2;
          const parentCenterY = parent.y + BLOCK_SIZE / 2;

          // Draw connection even if parent is off-screen (for continuity)
          // Calculate fade based on how far off-screen the connection goes
          const leftmostX = Math.min(parentCenterX, blockCenterX);
          const rightmostX = Math.max(parentCenterX, blockCenterX);

          // Fade out connections that are mostly off-screen
          let edgeFade = 1.0;
          if (leftmostX < 0) {
            edgeFade = Math.max(0.1, 1.0 + leftmostX / 300); // Gradual fade as it goes off left
          }
          if (rightmostX > canvas.width) {
            edgeFade = Math.min(edgeFade, Math.max(0.1, 1.0 - (rightmostX - canvas.width) / 300));
          }

          // Draw curved connection
          const isNewConnection = (block.age || 0) < 500;
          const baseAlpha = isNewConnection ? 0.4 + 0.4 * Math.min((block.age || 0) / 500, 1) : 0.7;
          const connectionAlpha = baseAlpha * edgeFade;

          // Only draw if connection has meaningful visibility
          if (connectionAlpha > 0.05) {
            // Gradient line
            const connGradient = ctx.createLinearGradient(parentCenterX, parentCenterY, blockCenterX, blockCenterY);
            connGradient.addColorStop(0, LANE_COLORS[parent.lane].glow.replace('0.6', String(connectionAlpha * 0.7)));
            connGradient.addColorStop(1, laneColor.glow.replace('0.6', String(connectionAlpha)));
            ctx.strokeStyle = connGradient;
            ctx.lineWidth = 2.5;

            // Always use curved path for DAG feel
            const midX = (parentCenterX + blockCenterX) / 2;
            const curveOffset = (block.lane - parent.lane) * 15; // Curve based on lane difference

            ctx.beginPath();
            ctx.moveTo(parentCenterX, parentCenterY);
            ctx.bezierCurveTo(
              midX - curveOffset, parentCenterY + curveOffset,
              midX + curveOffset, blockCenterY - curveOffset,
              blockCenterX, blockCenterY
            );
            ctx.stroke();

            // Glowing arrow head (only if visible on screen)
            if (blockCenterX > 0 && blockCenterX < canvas.width) {
              ctx.shadowBlur = 8 * edgeFade;
              ctx.shadowColor = laneColor.primary;
              const arrowSize = 7;
              ctx.fillStyle = laneColor.glow.replace('0.6', String(connectionAlpha));
              ctx.beginPath();
              ctx.moveTo(blockCenterX - BLOCK_SIZE/2 - 2, blockCenterY);
              ctx.lineTo(
                blockCenterX - BLOCK_SIZE/2 - arrowSize - 5,
                blockCenterY - arrowSize/2
              );
              ctx.lineTo(
                blockCenterX - BLOCK_SIZE/2 - arrowSize - 5,
                blockCenterY + arrowSize/2
              );
              ctx.closePath();
              ctx.fill();
              ctx.shadowBlur = 0;
            }
          }
        }

        // DAG-style: Also connect to grandparent (height - 2) with thinner line for DAG feel
        // Use deterministic selection based on block hash to avoid flickering
        const grandparent = blocksByHeight.get(block.height - 2);
        // Use hash to deterministically decide if grandparent connection should show (avoid random flickering)
        const showGrandparent = grandparent && block.hashPrefix &&
          (parseInt(block.hashPrefix.charAt(0), 16) % 3 !== 0); // ~67% of blocks show grandparent

        if (showGrandparent && grandparent) {
          const gpCenterX = grandparent.x - scrollOffset.current + BLOCK_SIZE / 2;
          const gpCenterY = grandparent.y + BLOCK_SIZE / 2;

          // Calculate edge fade for grandparent connection too
          const gpLeftX = Math.min(gpCenterX, blockCenterX);
          let gpFade = 1.0;
          if (gpLeftX < 0) {
            gpFade = Math.max(0.05, 1.0 + gpLeftX / 400);
          }

          if (gpFade > 0.05) {
            const gpAlpha = 0.25 * gpFade;
            ctx.strokeStyle = `rgba(139, 92, 246, ${gpAlpha})`;
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 6]);

            const midX = (gpCenterX + blockCenterX) / 2;
            ctx.beginPath();
            ctx.moveTo(gpCenterX, gpCenterY);
            ctx.bezierCurveTo(
              midX, gpCenterY,
              midX, blockCenterY,
              blockCenterX, blockCenterY
            );
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      });

      // Draw constellation connections (connect nearby blocks)
      if (visualMode === 'constellation') {
        blocks.forEach((block, i) => {
          const blockX = block.x - scrollOffset.current + BLOCK_SIZE / 2;
          const blockY = block.y + BLOCK_SIZE / 2;
          if (blockX < 0 || blockX > canvas.width) return;

          blocks.slice(i + 1).forEach(otherBlock => {
            const otherX = otherBlock.x - scrollOffset.current + BLOCK_SIZE / 2;
            const otherY = otherBlock.y + BLOCK_SIZE / 2;
            if (otherX < 0 || otherX > canvas.width) return;

            const dist = Math.sqrt((blockX - otherX) ** 2 + (blockY - otherY) ** 2);
            if (dist < 180 && dist > 50) {
              const alpha = 0.15 * (1 - dist / 180);
              ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(blockX, blockY);
              ctx.lineTo(otherX, otherY);
              ctx.stroke();
            }
          });
        });
      }

      // Draw particles — enhanced with trails and glow
      particlesRef.current.forEach(p => {
        const progress = p.life / p.maxLife;
        const alpha = Math.max(0, 1 - progress);
        const size = p.size * (1 - progress * 0.5);

        // Sparkle trail (fading tail behind particle)
        if (alpha > 0.2 && size > 1.5) {
          const tailLen = 3;
          for (let t = 1; t <= tailLen; t++) {
            const tailAlpha = alpha * (1 - t / tailLen) * 0.3;
            const tailX = p.x - p.vx * t * 0.003;
            const tailY = p.y - p.vy * t * 0.003;
            ctx.fillStyle = p.color.includes('rgb')
              ? p.color.replace('rgb', 'rgba').replace(')', `, ${tailAlpha})`)
              : `${p.color}${Math.floor(tailAlpha * 255).toString(16).padStart(2, '0')}`;
            ctx.beginPath();
            ctx.arc(tailX, tailY, size * (1 - t * 0.2), 0, Math.PI * 2);
            ctx.fill();
          }
        }

        // Outer glow
        ctx.shadowBlur = size > 2 ? 20 : 10;
        ctx.shadowColor = p.color;

        // Core particle with bright center
        const pGrad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, size * 1.5);
        pGrad.addColorStop(0, `rgba(255,255,255,${alpha * 0.9})`);
        pGrad.addColorStop(0.3, p.color.includes('rgb')
          ? p.color.replace('rgb', 'rgba').replace(')', `, ${alpha * 0.8})`)
          : p.color);
        pGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = pGrad;
        ctx.beginPath();
        ctx.arc(p.x, p.y, size * 1.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.shadowBlur = 0;
      });

      // Draw blocks
      blocks.forEach(block => {
        const x = block.x - scrollOffset.current;
        const y = block.y;

        // Skip if off-screen
        if (x < -BLOCK_SIZE || x > canvas.width + BLOCK_SIZE) return;

        const isSelected = selectedBlock?.id === block.id;
        const laneColor = LANE_COLORS[block.lane];

        // Entrance animation
        const age = block.age !== undefined ? block.age : 10000;
        const isNewBlock = age < 800;
        const animationProgress = Math.min(age / 800, 1);

        // Scale and glow effects
        const scale = isNewBlock ? 0.5 + (0.5 * animationProgress) : 1.0;
        const opacity = isNewBlock ? 0.5 + 0.5 * animationProgress : 1.0;
        const glowIntensity = isNewBlock ? 30 * (1 - animationProgress) + 10 : (isSelected ? 20 : 8);

        const scaledSize = BLOCK_SIZE * scale;
        const scaledX = x + (BLOCK_SIZE - scaledSize) / 2;
        const scaledY = y + (BLOCK_SIZE - scaledSize) / 2;

        // Glow effect
        ctx.shadowBlur = glowIntensity;
        ctx.shadowColor = laneColor.glow;

        // Block gradient fill
        const blockGradient = ctx.createLinearGradient(scaledX, scaledY, scaledX + scaledSize, scaledY + scaledSize);
        blockGradient.addColorStop(0, laneColor.primary);
        blockGradient.addColorStop(1, laneColor.secondary);
        ctx.fillStyle = blockGradient;
        ctx.globalAlpha = opacity;

        // Draw rounded rectangle
        ctx.beginPath();
        ctx.roundRect(scaledX, scaledY, scaledSize, scaledSize, BLOCK_RADIUS * scale);
        ctx.fill();

        // Border
        ctx.strokeStyle = isSelected ? '#ffffff' : laneColor.primary;
        ctx.lineWidth = isSelected ? 3 : 1.5;
        ctx.stroke();

        ctx.globalAlpha = 1;
        ctx.shadowBlur = 0;

        // Block content (only if not too small)
        if (scale > 0.7) {
          // Height number
          ctx.fillStyle = '#ffffff';
          ctx.font = `bold ${Math.round(12 * scale)}px Inter, system-ui, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(`${block.height}`, x + BLOCK_SIZE / 2, y + BLOCK_SIZE / 2 - 4);

          // TX count badge
          if (block.txCount > 0) {
            ctx.font = `${Math.round(8 * scale)}px Inter, system-ui, sans-serif`;
            ctx.fillStyle = 'rgba(16, 185, 129, 0.9)';
            ctx.fillText(`${block.txCount} tx`, x + BLOCK_SIZE / 2, y + BLOCK_SIZE / 2 + 12);
          }
        }

        // Matrix mode: draw falling characters
        if (visualMode === 'matrix' && Math.random() > 0.97) {
          const chars = '0123456789ABCDEF';
          const char = chars[Math.floor(Math.random() * chars.length)];
          ctx.font = '10px monospace';
          ctx.fillStyle = `rgba(16, 185, 129, ${0.3 + Math.random() * 0.5})`;
          ctx.fillText(char, x + Math.random() * BLOCK_SIZE, y + BLOCK_SIZE + 20 + Math.random() * 30);
        }
      });

      animationFrameId.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [blocks, selectedBlock, visualMode]);

  return (
    <div className="relative">
      {/* Visual Mode Controls */}
      <div className="absolute top-3 right-3 z-10 flex gap-1.5">
        {[
          { mode: 'flow' as const, icon: GitBranch, label: 'Flow', color: 'bg-blue-500' },
          { mode: 'quantum' as const, icon: Sparkles, label: 'Quantum', color: 'bg-purple-500' },
          { mode: 'constellation' as const, icon: Orbit, label: 'Constellation', color: 'bg-pink-500' },
          { mode: 'matrix' as const, icon: Box, label: 'Matrix', color: 'bg-green-500' },
        ].map(({ mode, icon: Icon, label, color }) => (
          <button
            key={mode}
            onClick={() => setVisualMode(mode)}
            className={`px-2.5 py-1.5 rounded-lg font-medium text-[10px] transition-all flex items-center gap-1 ${
              visualMode === mode
                ? `${color} text-white shadow-lg`
                : 'bg-slate-800/70 text-gray-300 hover:bg-slate-700/70'
            }`}
          >
            <Icon className="w-3 h-3" />
            {label}
          </button>
        ))}
      </div>

      {/* Stats Panel */}
      <div className="absolute top-3 left-3 z-10 flex flex-col gap-1.5">
        <div className={`px-3 py-1.5 rounded-lg backdrop-blur-sm border flex items-center gap-2 ${
          isConnected ? 'bg-emerald-500/20 border-emerald-500/30' : 'bg-red-500/20 border-red-500/30'
        }`}>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
          <span className="text-[10px] font-medium text-white">
            {isConnected ? 'Live' : 'Disconnected'}
          </span>
        </div>

        <div className="px-3 py-1.5 bg-slate-800/70 backdrop-blur-sm rounded-lg border border-blue-500/30 flex items-center gap-2">
          <Activity className="w-3 h-3 text-blue-400" />
          <span className="text-[10px] font-medium text-white">Height: {stats.totalBlocks.toLocaleString()}</span>
        </div>

        <div className="px-3 py-1.5 bg-slate-800/70 backdrop-blur-sm rounded-lg border border-emerald-500/30 flex items-center gap-2">
          <Zap className="w-3 h-3 text-emerald-400" />
          <span className="text-[10px] font-medium text-white">
            {stats.blocksPerSecond.toFixed(2)} blk/s
          </span>
        </div>

        <div className="px-3 py-1.5 bg-slate-800/70 backdrop-blur-sm rounded-lg border border-purple-500/30 flex items-center gap-2">
          <Layers className="w-3 h-3 text-purple-400" />
          <span className="text-[10px] font-medium text-white">DAG-Knight Consensus</span>
        </div>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={1200}
        height={CANVAS_HEIGHT}
        className="w-full h-auto bg-slate-900 rounded-xl border border-purple-500/20 cursor-pointer shadow-xl shadow-purple-500/5"
        onClick={handleCanvasClick}
      />

      {/* Selected Block Details */}
      {selectedBlock && (
        <div className="absolute bottom-3 right-3 z-10 max-w-xs">
          <div className="p-3 bg-slate-900/95 backdrop-blur-xl rounded-xl border border-blue-500/40 shadow-xl">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-bold text-white flex items-center gap-1.5">
                <Box className="w-3.5 h-3.5 text-blue-400" />
                Block #{selectedBlock.height}
              </h3>
              <button
                onClick={() => setSelectedBlock(null)}
                className="text-gray-400 hover:text-white p-0.5 hover:bg-slate-700 rounded"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="space-y-1.5 text-[10px]">
              <div className="flex justify-between">
                <span className="text-gray-400">Hash:</span>
                <span className="text-white font-mono">{selectedBlock.hashPrefix}...</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Transactions:</span>
                <span className="text-emerald-400 font-bold">{selectedBlock.txCount}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Reward:</span>
                <span className="text-blue-400 font-bold">{selectedBlock.reward.toFixed(4)} QUG</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Lane:</span>
                <span className="font-bold" style={{ color: LANE_COLORS[selectedBlock.lane]?.primary || '#60a5fa' }}>
                  {selectedBlock.lane}
                </span>
              </div>
              {selectedBlock.dagRound !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-400">DAG Round:</span>
                  <span className="text-purple-400 font-bold">{selectedBlock.dagRound}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
