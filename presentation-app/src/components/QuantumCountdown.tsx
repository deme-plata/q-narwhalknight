import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export function QuantumCountdown() {
  const [timeLeft, setTimeLeft] = useState({
    years: 0,
    days: 0,
    hours: 0,
    minutes: 0,
  });

  useEffect(() => {
    // Target date: January 1, 2030 (estimated quantum threat)
    const targetDate = new Date('2030-01-01T00:00:00Z');

    const updateCountdown = () => {
      const now = new Date();
      const diff = targetDate.getTime() - now.getTime();

      const years = Math.floor(diff / (1000 * 60 * 60 * 24 * 365));
      const days = Math.floor(
        (diff % (1000 * 60 * 60 * 24 * 365)) / (1000 * 60 * 60 * 24)
      );
      const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));

      setTimeLeft({ years, days, hours, minutes });
    };

    updateCountdown();
    const interval = setInterval(updateCountdown, 60000); // Update every minute

    return () => clearInterval(interval);
  }, []);

  const threatLevel = calculateThreatLevel(timeLeft.years);

  return (
    <div className="quantum-countdown">
      <div className="countdown-header">
        <div className="warning-icon">⚠️</div>
        <div className="countdown-title">Quantum Threat Countdown</div>
      </div>

      <div className="countdown-display">
        <div className="time-unit">
          <div className="time-value">{timeLeft.years}</div>
          <div className="time-label">Years</div>
        </div>
        <div className="time-separator">:</div>
        <div className="time-unit">
          <div className="time-value">{timeLeft.days}</div>
          <div className="time-label">Days</div>
        </div>
        <div className="time-separator">:</div>
        <div className="time-unit">
          <div className="time-value">{String(timeLeft.hours).padStart(2, '0')}</div>
          <div className="time-label">Hours</div>
        </div>
        <div className="time-separator">:</div>
        <div className="time-unit">
          <div className="time-value">{String(timeLeft.minutes).padStart(2, '0')}</div>
          <div className="time-label">Minutes</div>
        </div>
      </div>

      <div className="threat-meter">
        <div className="threat-label">Threat Level</div>
        <div className="threat-bar-container">
          <motion.div
            className="threat-bar"
            style={{
              width: `${threatLevel}%`,
              background: `linear-gradient(90deg, #00ff88, #ffff00, #ff0066)`,
            }}
            initial={{ width: 0 }}
            animate={{ width: `${threatLevel}%` }}
            transition={{ duration: 2, ease: 'easeOut' }}
          />
        </div>
        <div className="threat-percentage">{threatLevel}% Critical</div>
      </div>

      <div className="countdown-message">
        <div className="message-text">
          🔐 RSA-4096: Breakable in 10 minutes by quantum computer
        </div>
        <div className="message-text">
          ⏰ Classical: 500 supercomputers × 1,000 years
        </div>
        <div className="message-text urgent">
          🚨 Harvest-Now-Decrypt-Later attacks happening TODAY
        </div>
        <div className="message-text urgent">
          ✅ Deployment Deadline: 2028 (2 years before threat)
        </div>
      </div>

      <style>{`
        .quantum-countdown {
          width: 100%;
          padding: 30px;
          background: rgba(255, 0, 102, 0.1);
          border: 3px solid #ff0066;
          border-radius: 12px;
          box-shadow: 0 0 30px rgba(255, 0, 102, 0.5);
        }

        .countdown-header {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 15px;
          margin-bottom: 30px;
        }

        .warning-icon {
          font-size: 48px;
          animation: pulse-warning 1.5s ease-in-out infinite;
        }

        .countdown-title {
          font-size: 36px;
          color: #ff0066;
          font-weight: bold;
          text-shadow: 0 0 15px #ff0066;
        }

        .countdown-display {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 20px;
          margin-bottom: 30px;
        }

        .time-unit {
          display: flex;
          flex-direction: column;
          align-items: center;
          background: rgba(0, 0, 0, 0.5);
          padding: 20px 25px;
          border-radius: 12px;
          border: 2px solid #ff0066;
          box-shadow: 0 0 20px rgba(255, 0, 102, 0.3);
        }

        .time-value {
          font-size: 48px;
          color: #ff0066;
          font-weight: bold;
          font-family: 'Courier New', monospace;
          text-shadow: 0 0 15px #ff0066;
        }

        .time-label {
          font-size: 16px;
          color: var(--white);
          margin-top: 5px;
        }

        .time-separator {
          font-size: 48px;
          color: #ff0066;
          font-weight: bold;
          animation: blink 1s ease-in-out infinite;
        }

        .threat-meter {
          margin: 30px 0;
        }

        .threat-label {
          font-size: 24px;
          color: var(--white);
          margin-bottom: 10px;
          text-align: center;
        }

        .threat-bar-container {
          height: 30px;
          background: rgba(0, 0, 0, 0.5);
          border-radius: 15px;
          overflow: hidden;
          border: 2px solid #ff0066;
        }

        .threat-bar {
          height: 100%;
          border-radius: 15px;
          box-shadow: 0 0 20px currentColor;
        }

        .threat-percentage {
          font-size: 20px;
          color: #ff0066;
          margin-top: 10px;
          text-align: center;
          font-weight: bold;
        }

        .countdown-message {
          display: flex;
          flex-direction: column;
          gap: 10px;
          margin-top: 25px;
        }

        .message-text {
          font-size: 20px;
          color: var(--white);
          padding: 10px 20px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 8px;
          border-left: 4px solid #ffff00;
        }

        .message-text.urgent {
          border-left-color: #ff0066;
          animation: pulse-urgent 2s ease-in-out infinite;
        }

        @keyframes pulse-warning {
          0%, 100% {
            transform: scale(1);
            filter: brightness(1);
          }
          50% {
            transform: scale(1.1);
            filter: brightness(1.5);
          }
        }

        @keyframes blink {
          0%, 49%, 100% {
            opacity: 1;
          }
          50%, 99% {
            opacity: 0.3;
          }
        }

        @keyframes pulse-urgent {
          0%, 100% {
            background: rgba(255, 0, 102, 0.1);
          }
          50% {
            background: rgba(255, 0, 102, 0.3);
          }
        }
      `}</style>
    </div>
  );
}

function calculateThreatLevel(yearsLeft: number): number {
  // Assuming 2030 is the target date, and we're currently in 2025
  // 5 years left = 0% threat, 0 years = 100% threat
  const totalYears = 5;
  const elapsed = totalYears - yearsLeft;
  return Math.min(100, Math.max(0, (elapsed / totalYears) * 100));
}
