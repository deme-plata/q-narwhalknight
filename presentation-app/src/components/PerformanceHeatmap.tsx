import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface DataPoint {
  time: number;
  latency: number;
  tps: number;
}

export function PerformanceHeatmap() {
  const [data, setData] = useState<DataPoint[]>([]);

  useEffect(() => {
    // Generate simulated performance data
    const generateData = () => {
      const points: DataPoint[] = [];
      for (let i = 0; i < 50; i++) {
        points.push({
          time: i,
          latency: 7 + Math.random() * 5, // 7-12ms
          tps: 1000000 + Math.random() * 250000, // 1M-1.25M TPS
        });
      }
      setData(points);
    };

    generateData();
    const interval = setInterval(generateData, 5000);

    return () => clearInterval(interval);
  }, []);

  const getLatencyColor = (latency: number) => {
    if (latency < 8) return '#00ff88'; // Green
    if (latency < 10) return '#ffff00'; // Yellow
    return '#ff0066'; // Red
  };

  const maxLatency = 15;
  const minTps = 900000;
  const maxTps = 1300000;

  return (
    <div className="performance-heatmap">
      <div className="heatmap-title">Real-Time Performance Distribution</div>

      <div className="metrics-grid">
        <div className="metric-box">
          <div className="metric-value">
            {data.length > 0
              ? (data.reduce((sum, d) => sum + d.latency, 0) / data.length).toFixed(1)
              : '0'}
            ms
          </div>
          <div className="metric-label">Average Latency</div>
        </div>

        <div className="metric-box">
          <div className="metric-value">
            {data.length > 0
              ? Math.max(...data.map(d => d.latency)).toFixed(1)
              : '0'}
            ms
          </div>
          <div className="metric-label">P99 Latency</div>
        </div>

        <div className="metric-box">
          <div className="metric-value">
            {data.length > 0
              ? (data.reduce((sum, d) => sum + d.tps, 0) / data.length / 1000000).toFixed(2)
              : '0'}
            M
          </div>
          <div className="metric-label">Sustained TPS</div>
        </div>

        <div className="metric-box">
          <div className="metric-value">
            {data.length > 0
              ? (Math.max(...data.map(d => d.tps)) / 1000000).toFixed(2)
              : '0'}
            M
          </div>
          <div className="metric-label">Peak TPS</div>
        </div>
      </div>

      <div className="heatmap-container">
        <div className="heatmap-y-axis">
          <div className="y-label">15ms</div>
          <div className="y-label">10ms</div>
          <div className="y-label">5ms</div>
          <div className="y-label">0ms</div>
        </div>

        <div className="heatmap-grid">
          {data.map((point, index) => {
            const height = ((maxLatency - point.latency) / maxLatency) * 100;
            const color = getLatencyColor(point.latency);
            const opacity = 0.3 + ((point.tps - minTps) / (maxTps - minTps)) * 0.7;

            return (
              <motion.div
                key={index}
                className="heatmap-bar"
                style={{
                  height: `${height}%`,
                  background: color,
                  opacity,
                  boxShadow: `0 0 10px ${color}`,
                }}
                initial={{ height: 0 }}
                animate={{ height: `${height}%` }}
                transition={{ duration: 0.5, delay: index * 0.02 }}
              />
            );
          })}
        </div>

        <div className="heatmap-x-axis">
          <div className="x-label">Time →</div>
        </div>
      </div>

      <div className="legend-container">
        <div className="legend-title">Latency Distribution</div>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-box" style={{ background: '#00ff88' }}></div>
            <span>&lt;8ms: Excellent</span>
          </div>
          <div className="legend-item">
            <div className="legend-box" style={{ background: '#ffff00' }}></div>
            <span>8-10ms: Good</span>
          </div>
          <div className="legend-item">
            <div className="legend-box" style={{ background: '#ff0066' }}></div>
            <span>&gt;10ms: Target</span>
          </div>
        </div>
      </div>

      <div className="stats-footer">
        <div className="stat-item">
          📊 1,000 validators • 4 regions • 10 Gbps network
        </div>
        <div className="stat-item">
          🔐 Phase 1: Dilithium5 + Kyber1024 active
        </div>
      </div>

      <style>{`
        .performance-heatmap {
          width: 100%;
          padding: 30px;
          background: rgba(0, 0, 0, 0.4);
          border: 2px solid var(--magenta);
          border-radius: 12px;
          box-shadow: 0 0 30px rgba(255, 0, 255, 0.3);
        }

        .heatmap-title {
          font-size: 32px;
          color: var(--magenta);
          font-weight: bold;
          margin-bottom: 25px;
          text-align: center;
          text-shadow: 0 0 10px var(--magenta);
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 20px;
          margin-bottom: 30px;
        }

        .metric-box {
          background: rgba(255, 0, 255, 0.1);
          border: 2px solid var(--magenta);
          border-radius: 8px;
          padding: 15px;
          text-align: center;
        }

        .metric-value {
          font-size: 36px;
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

        .heatmap-container {
          display: flex;
          align-items: stretch;
          gap: 15px;
          height: 300px;
          margin-bottom: 20px;
        }

        .heatmap-y-axis {
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          padding: 10px 0;
        }

        .y-label {
          font-size: 14px;
          color: var(--gray);
          font-family: 'Courier New', monospace;
        }

        .heatmap-grid {
          flex: 1;
          display: flex;
          align-items: flex-end;
          gap: 4px;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid var(--gray-dark);
          border-radius: 8px;
          padding: 10px;
        }

        .heatmap-bar {
          flex: 1;
          min-width: 8px;
          border-radius: 4px 4px 0 0;
          transition: all 0.3s ease;
        }

        .heatmap-bar:hover {
          transform: scaleY(1.05);
        }

        .heatmap-x-axis {
          display: flex;
          justify-content: center;
          padding-top: 10px;
        }

        .x-label {
          font-size: 14px;
          color: var(--gray);
          font-family: 'Courier New', monospace;
        }

        .legend-container {
          margin: 20px 0;
          padding: 15px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 8px;
        }

        .legend-title {
          font-size: 18px;
          color: var(--white);
          margin-bottom: 10px;
          font-weight: bold;
        }

        .legend-items {
          display: flex;
          gap: 25px;
          justify-content: center;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 16px;
          color: var(--white);
        }

        .legend-box {
          width: 20px;
          height: 20px;
          border-radius: 4px;
          box-shadow: 0 0 10px currentColor;
        }

        .stats-footer {
          display: flex;
          justify-content: space-between;
          gap: 20px;
          margin-top: 20px;
        }

        .stat-item {
          font-size: 16px;
          color: var(--gray);
          padding: 10px 15px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 6px;
          flex: 1;
          text-align: center;
        }
      `}</style>
    </div>
  );
}
