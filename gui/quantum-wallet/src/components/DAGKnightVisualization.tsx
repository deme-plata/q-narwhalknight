import { useEffect, useRef, useState } from 'react';
import { Activity, Zap, TrendingUp, GitBranch, X, Sparkles, Layers, Orbit } from 'lucide-react';

interface DAGBlock {
  id: string; // block hash
  height: number;
  lane: number;
  timestamp: number;
  parents: string[];
  isBlueSet: boolean;
  x: number;
  miner?: string;
  txCount: number;
  reward: number;
  prevHash?: string;
  totalDifficulty?: number;
  dagRound?: number;
  minerCount?: number;
  age?: number; // Age in milliseconds since creation (for entrance animation)
  producerId?: number; // Phase 2: Producer ID for parallel visualization
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

export default function DAGKnightVisualization({ currentHeight }: DAGKnightVisualizationProps) {
  // Accept currentHeight prop (used for initial sync check)
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
  const [visualMode, setVisualMode] = useState<'normal' | 'quantum' | 'constellation' | 'heatmap'>('quantum');
  const [particles, setParticles] = useState<Particle[]>([]);

  const scrollOffset = useRef(0);
  const lastBlockTime = useRef(Date.now());
  const animationFrameId = useRef<number | undefined>(undefined);
  const laneAssignments = useRef<Map<number, number>>(new Map()); // height -> lane mapping
  const laneOccupancy = useRef<Map<number, number>>(new Map()); // lane -> rightmost x position

  // Phase 2 Constants: 8 parallel producers!
  const BLOCK_WIDTH = 60;
  const BLOCK_HEIGHT = 40;
  const LANE_HEIGHT = 70;
  const NUM_LANES = 8; // Phase 2: 8 producers = 8 lanes
  const SCROLL_SPEED = 200; // Faster for excitement
  const MIN_BLOCK_SPACING = 100; // Minimum horizontal spacing between blocks in same lane

  // Assign lane based on producer_id for true Phase 2 parallelism
  const assignLane = (height: number, producerId?: number): number => {
    if (laneAssignments.current.has(height)) {
      return laneAssignments.current.get(height)!;
    }

    // Phase 2: Use producer_id for true parallel block production visualization
    let lane: number;
    if (producerId !== undefined) {
      // True parallelism: each producer gets its own lane (0-7)
      lane = producerId % NUM_LANES;
    } else {
      // Fallback: distribute blocks across lanes for visual variety
      lane = (height * 7 + height % 3) % NUM_LANES;
    }

    laneAssignments.current.set(height, lane);
    return lane;
  };

  // Create quantum particles around a block
  const createParticles = (x: number, y: number, count: number = 20) => {
    const newParticles: Particle[] = [];
    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 * i) / count;
      const speed = 30 + Math.random() * 50;
      newParticles.push({
        x,
        y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 0,
        maxLife: 800 + Math.random() * 400,
        color: ['#60a5fa', '#a78bfa', '#ec4899', '#10b981'][Math.floor(Math.random() * 4)],
        size: 2 + Math.random() * 3,
      });
    }
    setParticles(prev => [...prev, ...newParticles]);
  };

  // Listen for new blocks via SSE
  useEffect(() => {
    console.log('🎨 DAG Visualization starting (Phase 2 Mode - 8 Producers), connecting to SSE stream...');

    const eventSource = new EventSource('/api/v1/events');

    eventSource.addEventListener('new-block', (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('🎨 DAGKnight: Received new-block SSE event:', data);

        // Handle tagged enum format: {type: "NewBlock", data: {...}}
        const blockData = data.data || data;
        console.log('🎨 DAGKnight: Extracted block data:', blockData);
        console.log('🎨 DAGKnight: producer_id from event:', blockData.producer_id);

        const now = Date.now();
        const timeSinceLastBlock = (now - lastBlockTime.current) / 1000;

        // Phase 2: Extract producer_id from block data
        const producerId = blockData.producer_id !== undefined ? blockData.producer_id : blockData.height % 8;
        console.log('🎨 DAGKnight: Using producerId:', producerId, '(from event or fallback)');

        // Assign lane for this block (Phase 2: based on producer_id)
        const assignedLane = assignLane(blockData.height, producerId);

        // Calculate X position with collision avoidance
        const canvasWidth = canvasRef.current?.width || 1200;
        const frontierX = scrollOffset.current + canvasWidth - 100;

        // Check if there's already a block in this lane recently
        const laneLastX = laneOccupancy.current.get(assignedLane) || 0;
        const minRequiredX = laneLastX + MIN_BLOCK_SPACING;

        // Position block at frontier or further right if lane is occupied
        const blockX = Math.max(frontierX, minRequiredX);

        // Update lane occupancy tracking
        laneOccupancy.current.set(assignedLane, blockX);

        const newBlock: DAGBlock = {
          id: blockData.hash || `block-${blockData.height}`,
          height: blockData.height,
          lane: assignedLane,
          timestamp: now,
          parents: blockData.prev_hash ? [blockData.prev_hash] : (blockData.height > 0 ? [`block-${blockData.height - 1}`] : []),
          isBlueSet: true, // All blocks are blue in Phase 2
          x: blockX,
          miner: `Producer #${producerId}`,
          txCount: blockData.tx_count || blockData.solutions_count || 0,
          reward: blockData.block_reward || 0,
          prevHash: blockData.prev_hash,
          totalDifficulty: blockData.total_difficulty,
          dagRound: blockData.dag_round,
          minerCount: blockData.miner_count,
          age: 0, // Track age for entrance animation
          producerId,
        };

        setBlocks(prev => {
          const updated = [...prev, newBlock];
          // Clean up old blocks that scrolled off-screen
          const filtered = updated.filter(b => b.x > scrollOffset.current - 300);

          // Clean up lane occupancy for off-screen blocks
          const visibleLaneMaxX = new Map<number, number>();
          filtered.forEach(block => {
            const currentMax = visibleLaneMaxX.get(block.lane) || 0;
            visibleLaneMaxX.set(block.lane, Math.max(currentMax, block.x));
          });
          laneOccupancy.current = visibleLaneMaxX;

          return filtered;
        });

        // Create quantum particles for new block (if in quantum mode)
        if (visualMode === 'quantum') {
          const canvas = canvasRef.current;
          if (canvas) {
            const blockScreenX = blockX - scrollOffset.current + BLOCK_WIDTH / 2;
            const blockScreenY = assignedLane * LANE_HEIGHT + 50 + BLOCK_HEIGHT / 2;
            createParticles(blockScreenX, blockScreenY, 30);
          }
        }

        // Update stats
        setStats(prevStats => ({
          ...prevStats,
          totalBlocks: blockData.height,
          blueSetCount: blockData.height,
          redSetCount: 0,
          blocksPerSecond: timeSinceLastBlock > 0 ? 1 / timeSinceLastBlock : 0,
        }));

        lastBlockTime.current = now;
        console.log('✨ Block added to visualization:', newBlock);
      } catch (error) {
        console.error('Error processing new-block event:', error);
      }
    });

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
    };

    return () => {
      console.log('Closing SSE connection');
      eventSource.close();
    };
  }, [visualMode]);

  // Handle canvas click to select blocks
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    // Find clicked block
    for (const block of blocks) {
      const blockX = block.x - scrollOffset.current;
      const blockY = block.lane * LANE_HEIGHT + 50;

      if (
        clickX >= blockX &&
        clickX <= blockX + BLOCK_WIDTH &&
        clickY >= blockY &&
        clickY <= blockY + BLOCK_HEIGHT
      ) {
        setSelectedBlock(block);
        return;
      }
    }

    // Click outside blocks - deselect
    setSelectedBlock(null);
  };

  // Animation loop for scrolling and effects
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

      // Update particles
      setParticles(prev =>
        prev
          .map(p => ({
            ...p,
            x: p.x + p.vx * deltaTime,
            y: p.y + p.vy * deltaTime,
            life: p.life + deltaTime * 1000,
            vx: p.vx * 0.98, // Slow down over time
            vy: p.vy * 0.98,
          }))
          .filter(p => p.life < p.maxLife)
      );

      // Clear canvas with gradient background
      const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, '#0a0a1a');
      gradient.addColorStop(1, '#1a0a2a');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw grid lines for lanes (Phase 2: 8 lanes)
      ctx.strokeStyle = 'rgba(139, 92, 246, 0.1)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= NUM_LANES; i++) {
        const y = i * LANE_HEIGHT + 50;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();

        // Lane labels (Producer #0 - Producer #7)
        if (i < NUM_LANES) {
          ctx.fillStyle = 'rgba(139, 92, 246, 0.4)';
          ctx.font = '10px monospace';
          ctx.fillText(`Producer #${i}`, 10, y + 20);
        }
      }

      // Constellation mode: draw all connections between nearby blocks
      if (visualMode === 'constellation') {
        blocks.forEach((block, i) => {
          blocks.slice(i + 1).forEach(otherBlock => {
            const blockX = block.x - scrollOffset.current + BLOCK_WIDTH / 2;
            const blockY = block.lane * LANE_HEIGHT + 50 + BLOCK_HEIGHT / 2;
            const otherX = otherBlock.x - scrollOffset.current + BLOCK_WIDTH / 2;
            const otherY = otherBlock.lane * LANE_HEIGHT + 50 + BLOCK_HEIGHT / 2;

            const dist = Math.sqrt((blockX - otherX) ** 2 + (blockY - otherY) ** 2);
            if (dist < 200) {
              ctx.strokeStyle = `rgba(139, 92, 246, ${0.3 * (1 - dist / 200)})`;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(blockX, blockY);
              ctx.lineTo(otherX, otherY);
              ctx.stroke();
            }
          });
        });
      }

      // Draw parent connection lines first (behind blocks)
      blocks.forEach(block => {
        const blockX = block.x - scrollOffset.current + BLOCK_WIDTH / 2;
        const blockY = block.lane * LANE_HEIGHT + 50 + BLOCK_HEIGHT / 2;

        block.parents.forEach(parentId => {
          const parent = blocks.find(b => b.id === parentId);
          if (!parent) return;

          const parentX = parent.x - scrollOffset.current + BLOCK_WIDTH / 2;
          const parentY = parent.lane * LANE_HEIGHT + 50 + BLOCK_HEIGHT / 2;

          // Rainbow gradient for connections in quantum mode
          if (visualMode === 'quantum') {
            const gradient = ctx.createLinearGradient(parentX, parentY, blockX, blockY);
            gradient.addColorStop(0, 'rgba(96, 165, 250, 0.6)');
            gradient.addColorStop(0.5, 'rgba(167, 139, 250, 0.6)');
            gradient.addColorStop(1, 'rgba(236, 72, 153, 0.6)');
            ctx.strokeStyle = gradient;
          } else {
            ctx.strokeStyle = block.isBlueSet
              ? 'rgba(59, 130, 246, 0.5)'
              : 'rgba(239, 68, 68, 0.4)';
          }
          ctx.lineWidth = 2;

          // Bezier curves for cross-lane connections
          if (Math.abs(block.lane - parent.lane) > 0) {
            const controlPoint1X = parentX + (blockX - parentX) * 0.3;
            const controlPoint1Y = parentY;
            const controlPoint2X = parentX + (blockX - parentX) * 0.7;
            const controlPoint2Y = blockY;

            ctx.beginPath();
            ctx.moveTo(parentX, parentY);
            ctx.bezierCurveTo(controlPoint1X, controlPoint1Y, controlPoint2X, controlPoint2Y, blockX, blockY);
            ctx.stroke();

            // Arrow head
            const arrowSize = 6;
            const angle = Math.atan2(blockY - controlPoint2Y, blockX - controlPoint2X);
            ctx.beginPath();
            ctx.moveTo(blockX, blockY);
            ctx.lineTo(
              blockX - arrowSize * Math.cos(angle - Math.PI / 6),
              blockY - arrowSize * Math.sin(angle - Math.PI / 6)
            );
            ctx.moveTo(blockX, blockY);
            ctx.lineTo(
              blockX - arrowSize * Math.cos(angle + Math.PI / 6),
              blockY - arrowSize * Math.sin(angle + Math.PI / 6)
            );
            ctx.stroke();
          } else {
            // Same lane connection
            ctx.beginPath();
            ctx.moveTo(parentX, parentY);
            ctx.lineTo(blockX, blockY);
            ctx.stroke();

            const arrowSize = 6;
            ctx.beginPath();
            ctx.moveTo(blockX - arrowSize, blockY - 3);
            ctx.lineTo(blockX, blockY);
            ctx.lineTo(blockX - arrowSize, blockY + 3);
            ctx.stroke();
          }
        });
      });

      // Draw particles (quantum mode)
      if (visualMode === 'quantum') {
        particles.forEach(p => {
          const alpha = 1 - (p.life / p.maxLife);
          ctx.fillStyle = p.color.replace(')', `, ${alpha})`).replace('rgb', 'rgba');
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
          ctx.fill();

          // Trailing effect
          ctx.shadowBlur = 10;
          ctx.shadowColor = p.color;
        });
      }

      // Draw blocks
      blocks.forEach(block => {
        const x = block.x - scrollOffset.current;
        const y = block.lane * LANE_HEIGHT + 50;

        // Skip if off-screen
        if (x < -BLOCK_WIDTH || x > canvas.width + BLOCK_WIDTH) return;

        const isSelected = selectedBlock?.id === block.id;

        // Entrance animation
        const age = block.age !== undefined ? block.age : 10000;
        const isNewBlock = age < 1000;
        const animationProgress = Math.min(age / 1000, 1);

        // Scale and pulse
        const scale = isNewBlock ? 0.7 + (0.3 * animationProgress) : 1.0;
        const opacity = isNewBlock ? animationProgress : 1.0;
        const pulse = age < 500 ? 1 + (0.3 * Math.sin((age / 500) * Math.PI * 4)) : 1.0;

        const scaledWidth = BLOCK_WIDTH * scale * pulse;
        const scaledHeight = BLOCK_HEIGHT * scale * pulse;
        const scaledX = x + (BLOCK_WIDTH - scaledWidth) / 2;
        const scaledY = y + (BLOCK_HEIGHT - scaledHeight) / 2;

        // Heatmap mode: color based on transaction count
        if (visualMode === 'heatmap') {
          const intensity = Math.min(block.txCount / 100, 1);
          const gradient = ctx.createLinearGradient(scaledX, scaledY, scaledX, scaledY + scaledHeight);
          gradient.addColorStop(0, `rgba(${255 * intensity}, ${255 * (1 - intensity)}, 100, ${opacity})`);
          gradient.addColorStop(1, `rgba(${200 * intensity}, ${200 * (1 - intensity)}, 80, ${0.8 * opacity})`);
          ctx.fillStyle = gradient;
          ctx.shadowColor = `rgba(${255 * intensity}, ${255 * (1 - intensity)}, 100, 0.8)`;
          ctx.shadowBlur = 15 + intensity * 20;
        } else {
          // Normal/quantum mode: gradient based on blue set
          const gradient = ctx.createLinearGradient(scaledX, scaledY, scaledX, scaledY + scaledHeight);
          if (block.isBlueSet) {
            gradient.addColorStop(0, isSelected ? `rgba(96, 165, 250, ${opacity})` : `rgba(59, 130, 246, ${0.9 * opacity})`);
            gradient.addColorStop(1, isSelected ? `rgba(59, 130, 246, ${0.9 * opacity})` : `rgba(37, 99, 235, ${0.7 * opacity})`);
            ctx.fillStyle = gradient;
            ctx.shadowColor = isNewBlock ? `rgba(96, 165, 250, ${0.9 * opacity})` : (isSelected ? 'rgba(96, 165, 250, 0.8)' : 'rgba(59, 130, 246, 0.5)');
          } else {
            gradient.addColorStop(0, isSelected ? `rgba(248, 113, 113, ${opacity})` : `rgba(239, 68, 68, ${0.9 * opacity})`);
            gradient.addColorStop(1, isSelected ? `rgba(239, 68, 68, ${0.9 * opacity})` : `rgba(220, 38, 38, ${0.7 * opacity})`);
            ctx.fillStyle = gradient;
            ctx.shadowColor = isNewBlock ? `rgba(248, 113, 113, ${0.9 * opacity})` : (isSelected ? 'rgba(248, 113, 113, 0.8)' : 'rgba(239, 68, 68, 0.5)');
          }
          ctx.shadowBlur = isNewBlock ? 30 * pulse : (isSelected ? 20 : 10);
        }

        // Draw block
        ctx.beginPath();
        ctx.roundRect(scaledX, scaledY, scaledWidth, scaledHeight, 6);
        ctx.fill();

        // Border
        ctx.strokeStyle = block.isBlueSet
          ? (isSelected ? `rgba(147, 197, 253, ${opacity})` : `rgba(96, 165, 250, ${0.8 * opacity})`)
          : (isSelected ? `rgba(252, 165, 165, ${opacity})` : `rgba(248, 113, 113, ${0.8 * opacity})`);
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.stroke();

        // Reset shadow
        ctx.shadowBlur = 0;

        // Block height text
        ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
        ctx.font = 'bold 11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`#${block.height}`, x + BLOCK_WIDTH / 2, y + BLOCK_HEIGHT / 2);

        // Transaction count (if > 0)
        if (block.txCount > 0) {
          ctx.font = '8px monospace';
          ctx.fillStyle = `rgba(16, 185, 129, ${opacity})`;
          ctx.fillText(`${block.txCount} tx`, x + BLOCK_WIDTH / 2, y + BLOCK_HEIGHT / 2 + 12);
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
  }, [blocks, selectedBlock, visualMode, particles]);

  return (
    <div className="relative">
      {/* Visual Mode Controls */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        <button
          onClick={() => setVisualMode('normal')}
          className={`px-3 py-2 rounded-lg font-semibold text-xs transition-all ${
            visualMode === 'normal'
              ? 'bg-quantum-cyan text-white shadow-lg shadow-quantum-cyan/50'
              : 'bg-quantum-indigo/30 text-gray-300 hover:bg-quantum-indigo/50'
          }`}
        >
          <GitBranch className="w-4 h-4 inline mr-1" />
          Normal
        </button>
        <button
          onClick={() => setVisualMode('quantum')}
          className={`px-3 py-2 rounded-lg font-semibold text-xs transition-all ${
            visualMode === 'quantum'
              ? 'bg-quantum-purple text-white shadow-lg shadow-quantum-purple/50'
              : 'bg-quantum-indigo/30 text-gray-300 hover:bg-quantum-indigo/50'
          }`}
        >
          <Sparkles className="w-4 h-4 inline mr-1" />
          Quantum
        </button>
        <button
          onClick={() => setVisualMode('constellation')}
          className={`px-3 py-2 rounded-lg font-semibold text-xs transition-all ${
            visualMode === 'constellation'
              ? 'bg-quantum-pink text-white shadow-lg shadow-quantum-pink/50'
              : 'bg-quantum-indigo/30 text-gray-300 hover:bg-quantum-indigo/50'
          }`}
        >
          <Orbit className="w-4 h-4 inline mr-1" />
          Constellation
        </button>
        <button
          onClick={() => setVisualMode('heatmap')}
          className={`px-3 py-2 rounded-lg font-semibold text-xs transition-all ${
            visualMode === 'heatmap'
              ? 'bg-quantum-orange text-white shadow-lg shadow-quantum-orange/50'
              : 'bg-quantum-indigo/30 text-gray-300 hover:bg-quantum-indigo/50'
          }`}
        >
          <Activity className="w-4 h-4 inline mr-1" />
          Heatmap
        </button>
      </div>

      {/* Stats Panel */}
      <div className="absolute top-4 left-4 z-10 space-y-2">
        <div className="px-4 py-2 bg-quantum-indigo/80 backdrop-blur-xl rounded-lg border border-quantum-cyan/30">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-quantum-cyan" />
            <span className="text-xs font-bold text-white">Total Blocks: {stats.totalBlocks}</span>
          </div>
        </div>
        <div className="px-4 py-2 bg-quantum-indigo/80 backdrop-blur-xl rounded-lg border border-quantum-green/30">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-quantum-green" />
            <span className="text-xs font-bold text-white">
              {stats.blocksPerSecond.toFixed(2)} blocks/sec
            </span>
          </div>
        </div>
        <div className="px-4 py-2 bg-quantum-indigo/80 backdrop-blur-xl rounded-lg border border-quantum-purple/30">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-quantum-purple" />
            <span className="text-xs font-bold text-white">8 Parallel Producers</span>
          </div>
        </div>
        <div className="px-4 py-2 bg-quantum-indigo/80 backdrop-blur-xl rounded-lg border border-quantum-pink/30">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-quantum-pink" />
            <span className="text-xs font-bold text-white">Phase 2 Active</span>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={1200}
        height={NUM_LANES * LANE_HEIGHT + 100}
        className="w-full h-auto bg-quantum-dark rounded-xl border border-quantum-purple/30 cursor-pointer"
        onClick={handleCanvasClick}
      />

      {/* Selected Block Details */}
      {selectedBlock && (
        <div className="absolute bottom-4 right-4 z-10 max-w-sm">
          <div className="p-4 bg-quantum-indigo/90 backdrop-blur-xl rounded-xl border border-quantum-cyan/50 shadow-lg">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <GitBranch className="w-4 h-4 text-quantum-cyan" />
                Block #{selectedBlock.height}
              </h3>
              <button
                onClick={() => setSelectedBlock(null)}
                className="text-gray-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">Miner:</span>
                <span className="text-white font-mono">{selectedBlock.miner}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Transactions:</span>
                <span className="text-quantum-green font-bold">{selectedBlock.txCount}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Reward:</span>
                <span className="text-quantum-cyan font-bold">{selectedBlock.reward.toFixed(8)} QUG</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Lane:</span>
                <span className="text-quantum-purple font-bold">{selectedBlock.lane}</span>
              </div>
              {selectedBlock.producerId !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-400">Producer:</span>
                  <span className="text-quantum-pink font-bold">#{selectedBlock.producerId}</span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-gray-400">Hash:</span>
                <span className="text-white font-mono text-[10px]">
                  {selectedBlock.id.substring(0, 16)}...
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
