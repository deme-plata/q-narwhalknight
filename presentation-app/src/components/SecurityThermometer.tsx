import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface Phase {
  id: number;
  name: string;
  level: number;
  color: string;
  description: string;
}

const phases: Phase[] = [
  { id: 0, name: 'Phase 0: Classical', level: 20, color: '#ff0066', description: 'Ed25519 - Quantum Vulnerable' },
  { id: 1, name: 'Phase 1: Post-Quantum', level: 60, color: '#ffff00', description: 'Dilithium5 + Kyber1024' },
  { id: 2, name: 'Phase 2: QRNG', level: 75, color: '#7FBA5A', description: 'Quantum Randomness' },
  { id: 3, name: 'Phase 3: QKD', level: 90, color: '#00ff88', description: 'Quantum Key Distribution' },
  { id: 4, name: 'Phase 4: Full Quantum', level: 100, color: '#00ffff', description: 'Complete Quantum Security' },
];

export function SecurityThermometer() {
  const [currentPhase] = useState(1); // Currently at Phase 1
  const [animatedLevel, setAnimatedLevel] = useState(0);

  useEffect(() => {
    const targetLevel = phases[currentPhase].level;
    const duration = 2000; // 2 seconds
    const steps = 60;
    const increment = targetLevel / steps;
    const interval = duration / steps;

    let current = 0;
    const timer = setInterval(() => {
      current += increment;
      if (current >= targetLevel) {
        setAnimatedLevel(targetLevel);
        clearInterval(timer);
      } else {
        setAnimatedLevel(current);
      }
    }, interval);

    return () => clearInterval(timer);
  }, [currentPhase]);

  const currentPhaseData = phases[currentPhase];

  return (
    <div className="security-thermometer">
      <div className="thermo-title">Quantum Security Level</div>

      <div className="thermo-container">
        <div className="thermo-scale">
          <div className="scale-marker" style={{ bottom: '100%' }}>
            <span>100%</span>
            <span className="scale-label">Full Quantum</span>
          </div>
          <div className="scale-marker" style={{ bottom: '75%' }}>
            <span>75%</span>
            <span className="scale-label">QRNG</span>
          </div>
          <div className="scale-marker" style={{ bottom: '50%' }}>
            <span>50%</span>
            <span className="scale-label">Transition</span>
          </div>
          <div className="scale-marker" style={{ bottom: '25%' }}>
            <span>25%</span>
            <span className="scale-label">Vulnerable</span>
          </div>
          <div className="scale-marker" style={{ bottom: '0%' }}>
            <span>0%</span>
            <span className="scale-label">Classical</span>
          </div>
        </div>

        <div className="thermo-tube">
          <div className="thermo-bg"></div>
          <motion.div
            className="thermo-fill"
            style={{
              height: `${animatedLevel}%`,
              background: `linear-gradient(to top, #ff0066, ${currentPhaseData.color})`,
              boxShadow: `0 0 30px ${currentPhaseData.color}`,
            }}
            initial={{ height: 0 }}
            animate={{ height: `${animatedLevel}%` }}
            transition={{ duration: 2, ease: 'easeOut' }}
          >
            <div className="thermo-bubble">
              <div className="bubble-inner"></div>
            </div>
          </motion.div>

          <div className="thermo-level-indicator" style={{ bottom: `${animatedLevel}%` }}>
            <div className="level-badge" style={{ background: currentPhaseData.color }}>
              {animatedLevel.toFixed(0)}%
            </div>
          </div>
        </div>

        <div className="phase-indicators">
          {phases.map((phase) => (
            <div
              key={phase.id}
              className={`phase-indicator ${phase.id === currentPhase ? 'active' : ''} ${
                phase.id < currentPhase ? 'completed' : ''
              }`}
              style={{ bottom: `${phase.level}%` }}
            >
              <div
                className="phase-dot"
                style={{
                  background: phase.id <= currentPhase ? phase.color : '#495670',
                  boxShadow:
                    phase.id === currentPhase
                      ? `0 0 20px ${phase.color}`
                      : 'none',
                }}
              ></div>
            </div>
          ))}
        </div>
      </div>

      <div className="phase-info">
        <div className="current-phase-box">
          <div className="phase-badge" style={{ background: currentPhaseData.color }}>
            CURRENT PHASE
          </div>
          <div className="phase-name">{currentPhaseData.name}</div>
          <div className="phase-desc">{currentPhaseData.description}</div>
        </div>

        <div className="phase-roadmap">
          {phases.map((phase, index) => (
            <div
              key={phase.id}
              className={`roadmap-item ${
                phase.id === currentPhase
                  ? 'current'
                  : phase.id < currentPhase
                  ? 'completed'
                  : 'upcoming'
              }`}
            >
              <div
                className="roadmap-dot"
                style={{
                  background: phase.id <= currentPhase ? phase.color : '#495670',
                }}
              ></div>
              <div className="roadmap-content">
                <div className="roadmap-name">{phase.name}</div>
                <div className="roadmap-level">{phase.level}% Security</div>
              </div>
              {index < phases.length - 1 && (
                <div
                  className="roadmap-line"
                  style={{
                    background:
                      phase.id < currentPhase
                        ? `linear-gradient(to bottom, ${phase.color}, ${phases[index + 1].color})`
                        : '#495670',
                  }}
                ></div>
              )}
            </div>
          ))}
        </div>
      </div>

      <style>{`
        .security-thermometer {
          width: 100%;
          padding: 30px;
          background: rgba(0, 0, 0, 0.4);
          border: 2px solid var(--green);
          border-radius: 12px;
          box-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
        }

        .thermo-title {
          font-size: 32px;
          color: var(--green);
          font-weight: bold;
          margin-bottom: 30px;
          text-align: center;
          text-shadow: 0 0 10px var(--green);
        }

        .thermo-container {
          display: flex;
          justify-content: center;
          align-items: stretch;
          gap: 30px;
          height: 400px;
          margin-bottom: 30px;
          position: relative;
        }

        .thermo-scale {
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          position: relative;
          width: 150px;
        }

        .scale-marker {
          display: flex;
          flex-direction: column;
          align-items: flex-end;
          gap: 5px;
          font-size: 16px;
          color: var(--white);
          font-family: 'Courier New', monospace;
        }

        .scale-label {
          font-size: 12px;
          color: var(--gray);
        }

        .thermo-tube {
          width: 80px;
          height: 100%;
          position: relative;
          border-radius: 40px;
          overflow: hidden;
        }

        .thermo-bg {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          border: 3px solid var(--gray-dark);
          border-radius: 40px;
        }

        .thermo-fill {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          border-radius: 0 0 37px 37px;
          transition: height 0.3s ease;
        }

        .thermo-bubble {
          position: absolute;
          top: -20px;
          left: 50%;
          transform: translateX(-50%);
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background: inherit;
          animation: bubble-pulse 2s ease-in-out infinite;
        }

        .bubble-inner {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: rgba(255, 255, 255, 0.5);
          animation: bubble-glow 1s ease-in-out infinite;
        }

        .thermo-level-indicator {
          position: absolute;
          right: -60px;
          transform: translateY(50%);
        }

        .level-badge {
          padding: 8px 16px;
          border-radius: 8px;
          font-size: 24px;
          font-weight: bold;
          color: var(--white);
          box-shadow: 0 0 20px currentColor;
          font-family: 'Courier New', monospace;
        }

        .phase-indicators {
          position: absolute;
          left: 50%;
          height: 100%;
        }

        .phase-indicator {
          position: absolute;
          left: 120px;
        }

        .phase-dot {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          border: 3px solid var(--white);
          transition: all 0.3s ease;
        }

        .phase-indicator.active .phase-dot {
          width: 30px;
          height: 30px;
          animation: pulse-phase 2s ease-in-out infinite;
        }

        .phase-info {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
        }

        .current-phase-box {
          background: rgba(0, 255, 136, 0.1);
          border: 2px solid var(--green);
          border-radius: 8px;
          padding: 20px;
          text-align: center;
        }

        .phase-badge {
          display: inline-block;
          padding: 6px 16px;
          border-radius: 6px;
          font-size: 14px;
          font-weight: bold;
          color: var(--white);
          margin-bottom: 15px;
        }

        .phase-name {
          font-size: 24px;
          color: var(--white);
          font-weight: bold;
          margin-bottom: 10px;
        }

        .phase-desc {
          font-size: 16px;
          color: var(--gray);
        }

        .phase-roadmap {
          display: flex;
          flex-direction: column;
          gap: 0;
        }

        .roadmap-item {
          display: flex;
          align-items: center;
          gap: 15px;
          position: relative;
          padding: 10px 0;
        }

        .roadmap-dot {
          width: 16px;
          height: 16px;
          border-radius: 50%;
          flex-shrink: 0;
          box-shadow: 0 0 10px currentColor;
        }

        .roadmap-item.current .roadmap-dot {
          width: 20px;
          height: 20px;
          animation: pulse-phase 2s ease-in-out infinite;
        }

        .roadmap-content {
          flex: 1;
        }

        .roadmap-name {
          font-size: 16px;
          color: var(--white);
          font-weight: bold;
        }

        .roadmap-level {
          font-size: 14px;
          color: var(--gray);
        }

        .roadmap-line {
          position: absolute;
          left: 7px;
          top: 30px;
          width: 2px;
          height: 40px;
        }

        @keyframes bubble-pulse {
          0%, 100% {
            transform: translateX(-50%) scale(1);
          }
          50% {
            transform: translateX(-50%) scale(1.1);
          }
        }

        @keyframes bubble-glow {
          0%, 100% {
            opacity: 0.5;
          }
          50% {
            opacity: 1;
          }
        }

        @keyframes pulse-phase {
          0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 10px currentColor;
          }
          50% {
            transform: scale(1.1);
            box-shadow: 0 0 20px currentColor;
          }
        }
      `}</style>
    </div>
  );
}
