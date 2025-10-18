import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface SystemPerformance {
  name: string;
  tps: number;
  color: string;
  finality: string;
}

const systems: SystemPerformance[] = [
  { name: 'Bitcoin', tps: 7, color: '#F7931A', finality: '60+ min' },
  { name: 'Ethereum', tps: 30, color: '#627EEA', finality: '6+ min' },
  { name: 'Solana', tps: 65000, color: '#14F195', finality: '400ms' },
  { name: 'Aptos', tps: 160000, color: '#7FBA5A', finality: '1s' },
  { name: 'Sui', tps: 297000, color: '#6FBCF0', finality: '480ms' },
  { name: 'Quillon', tps: 1247832, color: '#00ffff', finality: '<10ms' },
];

export function ThroughputRaceChart() {
  const [animatedValues, setAnimatedValues] = useState<number[]>(
    systems.map(() => 0)
  );

  useEffect(() => {
    const duration = 3000; // 3 seconds animation
    const steps = 60;
    const interval = duration / steps;

    let currentStep = 0;
    const timer = setInterval(() => {
      currentStep++;
      const progress = currentStep / steps;

      setAnimatedValues(
        systems.map((sys) => sys.tps * easeOutCubic(progress))
      );

      if (currentStep >= steps) {
        clearInterval(timer);
      }
    }, interval);

    return () => clearInterval(timer);
  }, []);

  const maxTps = Math.max(...systems.map((s) => s.tps));

  const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);

  return (
    <div className="throughput-race-chart">
      <div className="chart-title">Throughput Comparison (TPS)</div>

      <div className="chart-bars">
        {systems.map((system, index) => {
          const percentage = (animatedValues[index] / maxTps) * 100;
          const isQuillon = system.name === 'Quillon';

          return (
            <div key={system.name} className="bar-row">
              <div className="bar-label">
                <span className="system-name">{system.name}</span>
                <span className="tps-value" style={{ color: system.color }}>
                  {animatedValues[index].toLocaleString('en-US', {
                    maximumFractionDigits: 0,
                  })}{' '}
                  TPS
                </span>
                <span className="finality-badge">
                  {system.finality}
                </span>
              </div>

              <div className="bar-container">
                <motion.div
                  className={`bar ${isQuillon ? 'bar-highlight' : ''}`}
                  style={{
                    width: `${percentage}%`,
                    background: isQuillon
                      ? `linear-gradient(90deg, ${system.color}, #ff00ff)`
                      : system.color,
                    boxShadow: isQuillon
                      ? `0 0 20px ${system.color}`
                      : 'none',
                  }}
                  initial={{ width: 0 }}
                  animate={{ width: `${percentage}%` }}
                  transition={{ duration: 3, ease: 'easeOut' }}
                >
                  {isQuillon && (
                    <div className="winner-badge">
                      🏆 FASTEST
                    </div>
                  )}
                </motion.div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="chart-legend">
        <div className="legend-item">
          <div className="legend-dot" style={{ background: '#00ffff' }}></div>
          <span>3-5x faster throughput, 48x faster finality</span>
        </div>
      </div>

      <style>{`
        .throughput-race-chart {
          width: 100%;
          padding: 30px;
          background: rgba(0, 0, 0, 0.4);
          border: 2px solid var(--cyan);
          border-radius: 12px;
          box-shadow: 0 0 30px rgba(0, 255, 255, 0.3);
        }

        .chart-title {
          font-size: 36px;
          color: var(--cyan);
          font-weight: bold;
          margin-bottom: 30px;
          text-align: center;
          text-shadow: 0 0 10px var(--cyan);
        }

        .chart-bars {
          display: flex;
          flex-direction: column;
          gap: 25px;
        }

        .bar-row {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .bar-label {
          display: flex;
          align-items: center;
          gap: 15px;
          font-size: 20px;
        }

        .system-name {
          color: var(--white);
          font-weight: bold;
          min-width: 120px;
        }

        .tps-value {
          font-weight: bold;
          min-width: 200px;
          font-family: 'Courier New', monospace;
        }

        .finality-badge {
          font-size: 16px;
          color: var(--gray);
          padding: 4px 12px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 4px;
        }

        .bar-container {
          height: 40px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          overflow: hidden;
          position: relative;
        }

        .bar {
          height: 100%;
          border-radius: 8px;
          position: relative;
          transition: width 0.3s ease;
          display: flex;
          align-items: center;
          justify-content: flex-end;
          padding-right: 15px;
        }

        .bar-highlight {
          animation: pulse 2s ease-in-out infinite;
        }

        .winner-badge {
          font-size: 16px;
          font-weight: bold;
          color: var(--white);
          text-shadow: 0 0 10px var(--cyan);
          animation: bounce 1s ease-in-out infinite;
        }

        .chart-legend {
          margin-top: 30px;
          display: flex;
          justify-content: center;
          gap: 30px;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 18px;
          color: var(--white);
        }

        .legend-dot {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          box-shadow: 0 0 10px currentColor;
        }

        @keyframes pulse {
          0%, 100% {
            box-shadow: 0 0 20px var(--cyan);
          }
          50% {
            box-shadow: 0 0 40px var(--cyan), 0 0 60px var(--magenta);
          }
        }

        @keyframes bounce {
          0%, 100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-5px);
          }
        }
      `}</style>
    </div>
  );
}
