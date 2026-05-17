import React, { useEffect, useState } from 'react';
import './SwapSuccessModal.css';

interface SwapSuccessModalProps {
  isOpen: boolean;
  onClose: () => void;
  fromToken: string;
  toToken: string;
  fromAmount: number;
  toAmount: number;
  transactionHash?: string;
}

const SwapSuccessModal: React.FC<SwapSuccessModalProps> = ({
  isOpen,
  onClose,
  fromToken,
  toToken,
  fromAmount,
  toAmount,
  transactionHash,
}) => {
  const [showContent, setShowContent] = useState(false);
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; delay: number }>>([]);

  useEffect(() => {
    if (isOpen) {
      // Generate random particles for the celebration effect
      const newParticles = Array.from({ length: 50 }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 100,
        delay: Math.random() * 0.5,
      }));
      setParticles(newParticles);

      // Delay content appearance for dramatic effect
      setTimeout(() => setShowContent(true), 100);

      // Auto-close after 5 seconds
      const timer = setTimeout(() => {
        handleClose();
      }, 5000);

      return () => clearTimeout(timer);
    } else {
      setShowContent(false);
    }
  }, [isOpen]);

  const handleClose = () => {
    setShowContent(false);
    setTimeout(() => onClose(), 300);
  };

  if (!isOpen) return null;

  return (
    <div className="swap-success-overlay" onClick={handleClose}>
      <div className="swap-success-modal" onClick={(e) => e.stopPropagation()}>
        {/* Quantum particle effects */}
        <div className="particle-container">
          {particles.map((particle) => (
            <div
              key={particle.id}
              className="quantum-particle"
              style={{
                left: `${particle.x}%`,
                top: `${particle.y}%`,
                animationDelay: `${particle.delay}s`,
              }}
            />
          ))}
        </div>

        {/* Pulsing quantum rings */}
        <div className="quantum-rings">
          <div className="quantum-ring ring-1"></div>
          <div className="quantum-ring ring-2"></div>
          <div className="quantum-ring ring-3"></div>
        </div>

        {/* Success icon with animations */}
        <div className={`success-icon-wrapper ${showContent ? 'show' : ''}`}>
          <div className="success-icon-bg"></div>
          <svg className="success-checkmark" viewBox="0 0 52 52">
            <circle className="checkmark-circle" cx="26" cy="26" r="25" fill="none" />
            <path className="checkmark-check" fill="none" d="M14.1 27.2l7.1 7.2 16.7-16.8" />
          </svg>
        </div>

        {/* Content */}
        <div className={`success-content ${showContent ? 'show' : ''}`}>
          <h2 className="success-title">
            <span className="title-text">Swap Successful!</span>
            <div className="title-glow"></div>
          </h2>

          <p className="success-subtitle">
            Your quantum-secured transaction is complete
          </p>

          {/* Swap details with animated arrow */}
          <div className="swap-details">
            <div className="token-display from-token">
              <div className="token-amount">{(isFinite(fromAmount) && !isNaN(fromAmount) ? fromAmount : 0).toLocaleString(undefined, { maximumFractionDigits: 6 })}</div>
              <div className="token-symbol">{fromToken}</div>
              <div className="token-glow"></div>
            </div>

            <div className="swap-arrow-container">
              <div className="swap-arrow">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </div>
              <div className="arrow-trail trail-1"></div>
              <div className="arrow-trail trail-2"></div>
              <div className="arrow-trail trail-3"></div>
            </div>

            <div className="token-display to-token">
              <div className="token-amount">{(isFinite(toAmount) && !isNaN(toAmount) ? toAmount : 0).toLocaleString(undefined, { maximumFractionDigits: 6 })}</div>
              <div className="token-symbol">{toToken}</div>
              <div className="token-glow"></div>
            </div>
          </div>

          {/* Transaction hash */}
          {transactionHash && (
            <div className="transaction-hash">
              <div className="hash-label">Transaction Hash</div>
              <div className="hash-value">
                {transactionHash.substring(0, 10)}...{transactionHash.substring(transactionHash.length - 8)}
              </div>
            </div>
          )}

          {/* Quantum security badge */}
          <div className="security-badge">
            <div className="badge-icon">🔐</div>
            <div className="badge-text">
              <div className="badge-title">Quantum Secured</div>
              <div className="badge-subtitle">Post-quantum cryptography verified</div>
            </div>
            <div className="badge-shimmer"></div>
          </div>

          {/* Action buttons */}
          <div className="success-actions">
            <button className="action-button secondary" onClick={handleClose}>
              <span className="button-text">Close</span>
              <div className="button-glow"></div>
            </button>
            <button className="action-button primary" onClick={handleClose}>
              <span className="button-text">Make Another Swap</span>
              <div className="button-glow"></div>
            </button>
          </div>
        </div>

        {/* Closing X button */}
        <button className="close-button" onClick={handleClose}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>

        {/* Holographic overlay effect */}
        <div className="holographic-overlay"></div>
      </div>
    </div>
  );
};

export default SwapSuccessModal;
