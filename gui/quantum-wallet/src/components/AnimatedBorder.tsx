import React, { useState, useEffect } from 'react';
import './AnimatedBorder.css';

interface AnimatedBorderProps {
  children: React.ReactNode;
}

export type BorderTheme = 'purple' | 'red';

const AnimatedBorder: React.FC<AnimatedBorderProps> = ({
  children
}) => {
  const [currentTheme, setCurrentTheme] = useState<BorderTheme>('purple');
  const [nextTheme, setNextTheme] = useState<BorderTheme>('red');
  const [isTransitioning, setIsTransitioning] = useState(false);

  // Listen for transaction-sent events to flash red briefly
  useEffect(() => {
    const handleTransactionSent = () => {
      // Flash to red
      setIsTransitioning(true);
      setNextTheme('red');

      setTimeout(() => {
        setCurrentTheme('red');
        setIsTransitioning(false);

        // After 1.5 seconds, fade back to purple
        setTimeout(() => {
          setIsTransitioning(true);
          setNextTheme('purple');

          setTimeout(() => {
            setCurrentTheme('purple');
            setIsTransitioning(false);
          }, 1000);
        }, 1500);
      }, 1000);
    };

    window.addEventListener('transaction-sent', handleTransactionSent);
    return () => window.removeEventListener('transaction-sent', handleTransactionSent);
  }, []);

  return (
    <div className="animated-border-container">
      {/* Full border frame - current theme */}
      <div className={`border-frame border-frame-current ${isTransitioning ? 'fading-out' : ''}`}>
        <img src={`/borders/${currentTheme}/frame-full.png`} alt="" />
      </div>

      {/* Full border frame - next theme (for crossfade) */}
      <div className={`border-frame border-frame-next ${isTransitioning ? 'fading-in' : ''}`}>
        <img src={`/borders/${nextTheme}/frame-full.png`} alt="" />
      </div>

      {/* Glow effect overlay */}
      <div className={`border-glow ${currentTheme} ${isTransitioning ? 'transitioning' : ''}`} />

      {/* Content */}
      <div className="border-content">
        {children}
      </div>
    </div>
  );
};

export default AnimatedBorder;

// Helper function to trigger red flash when transaction is sent
export const flashBorderRed = () => {
  window.dispatchEvent(new CustomEvent('transaction-sent'));
};
