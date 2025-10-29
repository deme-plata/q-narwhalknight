import { useEffect, useRef, useState } from 'react';
import { Activity, Zap, TrendingUp, GitBranch, X } from 'lucide-react';

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
}

interface DAGKnightVisualizationProps {
  currentHeight: number;
}

export default function DAGKnightVisualization({ }: DAGKnightVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [blocks, setBlocks] = useState<DAGBlock[]>([]);
  const [selectedBlock, setSelectedBlock] = useState<DAGBlock | null>(null);
  const [stats, setStats] = useState({
    blueSetCount: 0,
    redSetCount: 0,
    totalBlocks: 0,
    blocksPerSecond: 0,
  });
  const scrollOffset = useRef(0);
  const lastBlockTime = useRef(Date.now());
  const animationFrameId = useRef<number | undefined>(undefined);
  const laneAssignments = useRef<Map<number, number>>(new Map()); // height -> lane mapping
  const laneOccupancy = useRef<Map<number, number>>(new Map()); // lane -> rightmost x position

  // Constants for layout (Phase 2: Optimized for parallel block visualization)
  const BLOCK_WIDTH = 60;
  const BLOCK_HEIGHT = 40;
  const LANE_HEIGHT = 80;
  const NUM_LANES = 5;
  const SCROLL_SPEED = 150; // Phase 2: 3x faster scroll for exciting animation
  const MIN_BLOCK_SPACING = 100; // Minimum horizontal spacing between blocks in same lane

  // Assign lane based on producer_id for true parallelism, fallback to height distribution
  const assignLane = (height: number, producerId?: number): number => {
    if (laneAssignments.current.has(height)) {
      return laneAssignments.current.get(height)!;
    }

    // Phase 2: Use producer_id for true parallel block production visualization
    let lane: number;
    if (producerId !== undefined && producerId > 0) {
      // True parallelism: each producer gets its own lane
      lane = producerId % NUM_LANES;
    } else {
      // Fallback: distribute blocks across lanes for visual variety
      lane = (height * 7 + height % 3) % NUM_LANES;
    }

    laneAssignments.current.set(height, lane);
    return lane;
  };

  // Listen for new blocks via SSE
  useEffect(() => {
    console.log('DAG Visualization starting, connecting to SSE stream...');

    const eventSource = new EventSource('/api/v1/events');

    eventSource.addEventListener('new-block', (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received new-block event:', data);

        const blockData = data.data || data;
        const now = Date.now();
        const timeSinceLastBlock = (now - lastBlockTime.current) / 1000;

        // Assign lane for this block
        const assignedLane = assignLane(blockData.height, blockData.producer_id);

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
          lane: assignedLane, // Phase 2: Pass producer_id for parallel lanes
          timestamp: now,
          parents: blockData.prev_hash ? [blockData.prev_hash] : (blockData.height > 0 ? [`block-${blockData.height - 1}`] : []),
          isBlueSet: true, // All blocks are blue in single validator mode
          x: blockX, // Anti-overlap positioning
          miner: 'Validator',
          txCount: blockData.tx_count || blockData.solutions_count || 0,
          reward: blockData.block_reward || 0,
          prevHash: blockData.prev_hash,
          totalDifficulty: blockData.total_difficulty,
          dagRound: blockData.dag_round,
          minerCount: blockData.miner_count,
          age: 0, // Phase 2: Track age for entrance animation
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

        // Update stats
        setStats(prevStats => ({
          ...prevStats,
          totalBlocks: blockData.height,
          blueSetCount: blockData.height,
          redSetCount: 0,
          blocksPerSecond: timeSinceLastBlock > 0 ? 1 / timeSinceLastBlock : 0,
        }));

        lastBlockTime.current = now;
        console.log('Block added to visualization:', newBlock);
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
  }, []);

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

  // Animation loop for scrolling
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

      // Phase 2: Update block ages for entrance animation effects
      blocks.forEach(block => {
        if (block.age !== undefined) {
          block.age += deltaTime * 1000; // Convert to milliseconds
        }
      });

      // Clear canvas
      ctx.fillStyle = '#0a0a1a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw grid lines for lanes
      ctx.strokeStyle = 'rgba(139, 92, 246, 0.1)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= NUM_LANES; i++) {
        const y = i * LANE_HEIGHT + 50;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
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

          // Connection color based on blue/red sets
          const isBlueConnection = block.isBlueSet && parent.isBlueSet;
          ctx.strokeStyle = isBlueConnection
            ? 'rgba(59, 130, 246, 0.5)'  // Blue connection
            : 'rgba(239, 68, 68, 0.4)';   // Red connection
          ctx.lineWidth = 2;

          // Use Bezier curves for cross-lane connections to show parallelization clearly
          if (Math.abs(block.lane - parent.lane) > 0) {
            // Cross-lane connection - use smooth Bezier curve
            const controlPoint1X = parentX + (blockX - parentX) * 0.3;
            const controlPoint1Y = parentY;
            const controlPoint2X = parentX + (blockX - parentX) * 0.7;
            const controlPoint2Y = blockY;

            ctx.beginPath();
            ctx.moveTo(parentX, parentY);
            ctx.bezierCurveTo(controlPoint1X, controlPoint1Y, controlPoint2X, controlPoint2Y, blockX, blockY);
            ctx.stroke();

            // Add arrow head to show direction
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
            // Same lane connection - simple line
            ctx.beginPath();
            ctx.moveTo(parentX, parentY);
            ctx.lineTo(blockX, blockY);
            ctx.stroke();

            // Add simple arrow
            const arrowSize = 6;
            ctx.beginPath();
            ctx.moveTo(blockX - arrowSize, blockY - 3);
            ctx.lineTo(blockX, blockY);
            ctx.lineTo(blockX - arrowSize, blockY + 3);
            ctx.stroke();
          }
        });
      });

      // Draw blocks
      blocks.forEach(block => {
        const x = block.x - scrollOffset.current;
        const y = block.lane * LANE_HEIGHT + 50;

        // Skip if off-screen
        if (x < -BLOCK_WIDTH || x > canvas.width + BLOCK_WIDTH) return;

        // Highlight selected block
        const isSelected = selectedBlock?.id === block.id;

        // Phase 2: Entrance animation for new blocks (exciting!)
        const age = block.age !== undefined ? block.age : 10000; // Old blocks have no age tracking
        const isNewBlock = age < 1000; // New block animation for first 1 second
        const animationProgress = Math.min(age / 1000, 1); // 0 to 1 over 1 second

        // Scale effect: blocks pop in with a slight scale animation
        const scale = isNewBlock ? 0.7 + (0.3 * animationProgress) : 1.0;
        const opacity = isNewBlock ? animationProgress : 1.0;

        // Pulse effect for very new blocks (first 500ms)
        const pulse = age < 500 ? 1 + (0.3 * Math.sin((age / 500) * Math.PI * 4)) : 1.0;

        // Calculate scaled dimensions
        const scaledWidth = BLOCK_WIDTH * scale * pulse;
        const scaledHeight = BLOCK_HEIGHT * scale * pulse;
        const scaledX = x + (BLOCK_WIDTH - scaledWidth) / 2;
        const scaledY = y + (BLOCK_HEIGHT - scaledHeight) / 2;

        // Block color based on blue/red set with opacity
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

        // Exciting glow effect for new blocks!
        ctx.shadowBlur = isNewBlock ? 30 * pulse : (isSelected ? 20 : 10);

        // Draw rounded rectangle for block (using scaled dimensions for animation)
        ctx.beginPath();
        ctx.roundRect(scaledX, scaledY, scaledWidth, scaledHeight, 6);
        ctx.fill();

        // Draw border
        ctx.strokeStyle = block.isBlueSet
          ? (isSelected ? `rgba(147, 197, 253, ${opacity})` : `rgba(96, 165, 250, ${0.8 * opacity})`)
          : (isSelected ? `rgba(252, 165, 165, ${opacity})` : `rgba(248, 113, 113, ${0.8 * opacity})`);
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.stroke();

        // Reset shadow
        ctx.shadowBlur = 0;

        // Draw block height text (with opacity for entrance animation)
        ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
        ctx.font = `bold ${11 * scale}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`#${block.height}`, scaledX + scaledWidth / 2, scaledY + scaledHeight / 2);
      });

      // Draw frontier line (right edge where new blocks appear)
      const frontierX = canvas.width - 100;
      ctx.strokeStyle = 'rgba(34, 197, 94, 0.5)';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(frontierX, 0);
      ctx.lineTo(frontierX, canvas.height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw "LIVE" label at frontier
      ctx.fillStyle = 'rgba(34, 197, 94, 0.8)';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('LIVE', frontierX, 30);

      animationFrameId.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [blocks, selectedBlock]);

  return (
    <div className="relative w-full bg-gradient-to-br from-slate-900 via-purple-900/20 to-slate-900 rounded-xl overflow-hidden border border-purple-500/30">
      {/* Header */}
      <div className="absolute top-4 left-4 z-10 bg-slate-800/80 backdrop-blur-lg rounded-lg p-4 border border-purple-500/30">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-purple-500 rounded-full blur-md animate-pulse"></div>
            <GitBranch className="w-8 h-8 text-purple-400 relative z-10" />
          </div>
          <div>
            <div className="text-sm text-slate-400">DAG-Knight Consensus</div>
            <div className="text-xl font-bold text-white">Live BlockDAG Stream</div>
            <div className="text-xs text-green-400 mt-1">Real-time blockchain data</div>
          </div>
        </div>
      </div>

      {/* Stats Panel */}
      <div className="absolute top-4 right-4 z-10 bg-slate-800/80 backdrop-blur-lg rounded-lg p-4 border border-purple-500/30">
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <Activity className="w-5 h-5 text-blue-400" />
            <div>
              <div className="text-xs text-slate-400">Blue Set</div>
              <div className="text-lg font-bold text-blue-400">{stats.blueSetCount}</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Zap className="w-5 h-5 text-red-400" />
            <div>
              <div className="text-xs text-slate-400">Red Set</div>
              <div className="text-lg font-bold text-red-400">{stats.redSetCount}</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <TrendingUp className="w-5 h-5 text-green-400" />
            <div>
              <div className="text-xs text-slate-400">Throughput</div>
              <div className="text-lg font-bold text-green-400">{stats.blocksPerSecond.toFixed(1)} BPS</div>
            </div>
          </div>
        </div>
      </div>

      {/* Block Details Panel */}
      {selectedBlock && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20 bg-slate-800/95 backdrop-blur-lg rounded-lg p-4 border border-purple-500/50 min-w-[400px]">
          <div className="flex items-start justify-between mb-3">
            <div className="text-sm font-semibold text-white">Block Details</div>
            <button
              onClick={() => setSelectedBlock(null)}
              className="text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Height:</span>
              <span className="text-white font-mono">#{selectedBlock.height}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Hash:</span>
              <span className="text-white font-mono text-xs">{selectedBlock.id.substring(0, 16)}...</span>
            </div>
            {selectedBlock.prevHash && (
              <div className="flex justify-between">
                <span className="text-slate-400">Parent:</span>
                <span className="text-white font-mono text-xs">{selectedBlock.prevHash.substring(0, 16)}...</span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-slate-400">Set:</span>
              <span className={selectedBlock.isBlueSet ? 'text-blue-400' : 'text-red-400'}>
                {selectedBlock.isBlueSet ? 'Blue (Preferred)' : 'Red (Valid)'}
              </span>
            </div>
            {selectedBlock.dagRound !== undefined && (
              <div className="flex justify-between">
                <span className="text-slate-400">DAG Round:</span>
                <span className="text-white">{selectedBlock.dagRound}</span>
              </div>
            )}
            {selectedBlock.totalDifficulty !== undefined && (
              <div className="flex justify-between">
                <span className="text-slate-400">Difficulty:</span>
                <span className="text-white">{selectedBlock.totalDifficulty.toLocaleString()}</span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-slate-400">Transactions:</span>
              <span className="text-white">{selectedBlock.txCount}</span>
            </div>
            {selectedBlock.minerCount !== undefined && selectedBlock.minerCount > 0 && (
              <div className="flex justify-between">
                <span className="text-slate-400">Miners:</span>
                <span className="text-white">{selectedBlock.minerCount}</span>
              </div>
            )}
            {selectedBlock.reward > 0 && (
              <div className="flex justify-between">
                <span className="text-slate-400">Reward:</span>
                <span className="text-green-400">{(selectedBlock.reward / 100_000_000).toFixed(8)} QNK</span>
              </div>
            )}
            {selectedBlock.miner && (
              <div className="flex justify-between">
                <span className="text-slate-400">Miner:</span>
                <span className="text-white font-mono text-xs">{selectedBlock.miner.substring(0, 16)}...</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-10 bg-slate-800/80 backdrop-blur-lg rounded-lg p-4 border border-purple-500/30">
        <div className="text-sm font-semibold text-white mb-3">Legend</div>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-blue-500"></div>
            <span className="text-xs text-slate-300">Blue Set (Preferred Order)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-red-500"></div>
            <span className="text-xs text-slate-300">Red Set (Valid, Non-Preferred)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-blue-400"></div>
            <span className="text-xs text-slate-300">Parent References</span>
          </div>
        </div>
      </div>

      {/* Info Panel */}
      <div className="absolute bottom-4 right-4 z-10 bg-slate-800/80 backdrop-blur-lg rounded-lg p-4 border border-purple-500/30 max-w-sm">
        <div className="text-sm text-slate-300">
          <strong className="text-white">DAG-Knight Ordering</strong>
          <p className="text-xs mt-1 text-slate-400">
            Real-time blockchain visualization. Click blocks to view details. The{' '}
            <span className="text-blue-400">blue set</span> represents the preferred ordering backbone, while{' '}
            <span className="text-red-400">red blocks</span> are valid but not in the preferred chain. All blocks contribute to security and throughput.
          </p>
        </div>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={1200}
        height={500}
        className="w-full h-[500px] cursor-pointer"
        onClick={handleCanvasClick}
      />

      {/* Timeline indicator */}
      <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-slate-900/90 to-transparent flex items-center justify-center">
        <div className="text-xs text-slate-400 font-mono">
          ← Older Blocks | Timeline | Newer Blocks (Live Frontier) →
        </div>
      </div>
    </div>
  );
}
