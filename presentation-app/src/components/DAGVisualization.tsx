import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface Vertex {
  id: string;
  round: number;
  x: number;
  y: number;
  state: 'pending' | 'certified' | 'anchor' | 'finalized';
  parents: string[];
}

export function DAGVisualization() {
  const [vertices, setVertices] = useState<Vertex[]>([]);
  const [connections, setConnections] = useState<Array<[string, string]>>([]);

  useEffect(() => {
    // Generate DAG structure
    const dagVertices: Vertex[] = [
      // Round 1 (Genesis)
      { id: 'V1', round: 1, x: 20, y: 80, state: 'finalized', parents: [] },
      { id: 'V2', round: 1, x: 50, y: 80, state: 'finalized', parents: [] },
      { id: 'V3', round: 1, x: 80, y: 80, state: 'finalized', parents: [] },

      // Round 2
      { id: 'V4', round: 2, x: 20, y: 50, state: 'finalized', parents: ['V1', 'V2'] },
      { id: 'V5', round: 2, x: 50, y: 50, state: 'anchor', parents: ['V1', 'V2', 'V3'] },
      { id: 'V6', round: 2, x: 80, y: 50, state: 'finalized', parents: ['V2', 'V3'] },

      // Round 3
      { id: 'V7', round: 3, x: 20, y: 20, state: 'certified', parents: ['V4', 'V5'] },
      { id: 'V8', round: 3, x: 50, y: 20, state: 'certified', parents: ['V4', 'V5', 'V6'] },
      { id: 'V9', round: 3, x: 80, y: 20, state: 'pending', parents: ['V5', 'V6'] },
    ];

    setVertices(dagVertices);

    // Generate connections
    const conns: Array<[string, string]> = [];
    dagVertices.forEach((v) => {
      v.parents.forEach((parentId) => {
        conns.push([parentId, v.id]);
      });
    });
    setConnections(conns);
  }, []);

  const getVertexColor = (state: Vertex['state']) => {
    switch (state) {
      case 'pending':
        return '#00ffff'; // Cyan
      case 'certified':
        return '#00ff88'; // Green
      case 'anchor':
        return '#ff00ff'; // Magenta
      case 'finalized':
        return '#8892b0'; // Gray
    }
  };

  const getVertexPosition = (id: string): { x: number; y: number } => {
    const vertex = vertices.find((v) => v.id === id);
    return vertex ? { x: vertex.x, y: vertex.y } : { x: 0, y: 0 };
  };

  return (
    <div className="dag-visualization">
      <div className="dag-title">DAG-Knight Consensus Structure</div>

      <div className="dag-legend">
        <div className="legend-item">
          <div className="legend-dot" style={{ background: '#00ffff' }}></div>
          <span>Pending</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ background: '#00ff88' }}></div>
          <span>Certified</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ background: '#ff00ff' }}></div>
          <span>Anchor (V5)</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ background: '#8892b0' }}></div>
          <span>Finalized</span>
        </div>
      </div>

      <svg className="dag-canvas" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
        {/* Draw connections */}
        <g className="connections">
          {connections.map(([from, to], index) => {
            const fromPos = getVertexPosition(from);
            const toPos = getVertexPosition(to);
            const fromVertex = vertices.find((v) => v.id === from);
            const color =
              fromVertex?.state === 'anchor' || fromVertex?.state === 'finalized'
                ? '#00ff88'
                : '#495670';

            return (
              <motion.line
                key={`${from}-${to}-${index}`}
                x1={fromPos.x}
                y1={fromPos.y}
                x2={toPos.x}
                y2={toPos.y}
                stroke={color}
                strokeWidth="0.5"
                opacity="0.6"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 0.6 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              />
            );
          })}
        </g>

        {/* Draw vertices */}
        <g className="vertices">
          {vertices.map((vertex, index) => {
            const color = getVertexColor(vertex.state);
            const isAnchor = vertex.state === 'anchor';

            return (
              <g key={vertex.id}>
                <motion.circle
                  cx={vertex.x}
                  cy={vertex.y}
                  r={isAnchor ? 4 : 3}
                  fill={color}
                  stroke={isAnchor ? '#ff00ff' : '#ffffff'}
                  strokeWidth={isAnchor ? 0.5 : 0.3}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{
                    scale: 1,
                    opacity: 1,
                  }}
                  transition={{ duration: 0.3, delay: index * 0.15 }}
                  style={{
                    filter: isAnchor ? `drop-shadow(0 0 5px ${color})` : 'none',
                  }}
                />

                {isAnchor && (
                  <motion.circle
                    cx={vertex.x}
                    cy={vertex.y}
                    r={6}
                    fill="none"
                    stroke="#ff00ff"
                    strokeWidth="0.5"
                    opacity="0.5"
                    initial={{ scale: 0 }}
                    animate={{
                      scale: [1, 1.5, 1],
                      opacity: [0.5, 0.2, 0.5],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: 'easeInOut',
                    }}
                  />
                )}

                <text
                  x={vertex.x}
                  y={vertex.y + 6}
                  fill="#ffffff"
                  fontSize="4"
                  textAnchor="middle"
                  fontFamily="'Courier New', monospace"
                  fontWeight="bold"
                >
                  {vertex.id}
                </text>
              </g>
            );
          })}
        </g>

        {/* Round labels */}
        <g className="round-labels">
          <text x="2" y="85" fill="#8892b0" fontSize="3" fontFamily="'Courier New', monospace">
            Round 1 (Genesis)
          </text>
          <text x="2" y="55" fill="#8892b0" fontSize="3" fontFamily="'Courier New', monospace">
            Round 2 ← Anchor
          </text>
          <text x="2" y="25" fill="#8892b0" fontSize="3" fontFamily="'Courier New', monospace">
            Round 3 (Current)
          </text>
        </g>

        {/* Arrows indicating flow */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="10"
            refX="5"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 5 3, 0 6" fill="#00ff88" />
          </marker>
        </defs>
      </svg>

      <div className="dag-explanation">
        <div className="explanation-step">
          <div className="step-number">1</div>
          <div className="step-text">
            <strong>Build DAG:</strong> Vertices from 3 rounds with parent references
          </div>
        </div>
        <div className="explanation-step">
          <div className="step-number">2</div>
          <div className="step-text">
            <strong>Elect Anchor:</strong> V5 chosen via VDF (deterministic)
          </div>
        </div>
        <div className="explanation-step">
          <div className="step-number">3</div>
          <div className="step-text">
            <strong>Sort DAG:</strong> V1→V2→V3→V4→V5→V6 (topological order)
          </div>
        </div>
        <div className="explanation-step">
          <div className="step-number">4</div>
          <div className="step-text">
            <strong>Extract TXs:</strong> Apply transactions to state machine
          </div>
        </div>
      </div>

      <div className="dag-metrics">
        <div className="metric">
          <div className="metric-value">O(1)</div>
          <div className="metric-label">Message Complexity</div>
        </div>
        <div className="metric">
          <div className="metric-value">0</div>
          <div className="metric-label">Voting Rounds</div>
        </div>
        <div className="metric">
          <div className="metric-value">100%</div>
          <div className="metric-label">Deterministic</div>
        </div>
      </div>

      <style>{`
        .dag-visualization {
          width: 100%;
          padding: 30px;
          background: rgba(0, 0, 0, 0.4);
          border: 2px solid var(--cyan);
          border-radius: 12px;
          box-shadow: 0 0 30px rgba(0, 255, 255, 0.3);
        }

        .dag-title {
          font-size: 32px;
          color: var(--cyan);
          font-weight: bold;
          margin-bottom: 20px;
          text-align: center;
          text-shadow: 0 0 10px var(--cyan);
        }

        .dag-legend {
          display: flex;
          justify-content: center;
          gap: 25px;
          margin-bottom: 20px;
          flex-wrap: wrap;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 16px;
          color: var(--white);
        }

        .legend-dot {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          box-shadow: 0 0 10px currentColor;
        }

        .dag-canvas {
          width: 100%;
          height: 400px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 8px;
          margin-bottom: 20px;
        }

        .dag-explanation {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 15px;
          margin-bottom: 20px;
        }

        .explanation-step {
          display: flex;
          gap: 15px;
          align-items: flex-start;
          background: rgba(0, 255, 255, 0.05);
          padding: 15px;
          border-radius: 8px;
          border-left: 3px solid var(--cyan);
        }

        .step-number {
          width: 30px;
          height: 30px;
          background: var(--cyan);
          color: var(--bg-darker);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          font-size: 16px;
          flex-shrink: 0;
        }

        .step-text {
          font-size: 14px;
          color: var(--white);
          line-height: 1.4;
        }

        .step-text strong {
          color: var(--cyan);
        }

        .dag-metrics {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 15px;
        }

        .metric {
          background: rgba(0, 255, 255, 0.1);
          border: 2px solid var(--cyan);
          border-radius: 8px;
          padding: 15px;
          text-align: center;
        }

        .metric-value {
          font-size: 32px;
          color: var(--cyan);
          font-weight: bold;
          font-family: 'Courier New', monospace;
          text-shadow: 0 0 10px var(--cyan);
        }

        .metric-label {
          font-size: 14px;
          color: var(--gray);
          margin-top: 5px;
        }
      `}</style>
    </div>
  );
}
