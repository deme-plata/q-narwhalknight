import React, { useState, useEffect } from 'react';
import './AnimatedBorder.css';

interface AnimatedBorderProps {
  children: React.ReactNode;
}

export type BorderTheme =
  | 'purple'
  | 'predator'
  | 'red'
  | 'inferno'
  | 'abyssal'
  | 'neon'
  | 'aurora'
  | 'hellfire'
  | 'alien'
  | 'sovereign'
  | 'frost'
  | 'cosmic'
  | 'metallic'
  | 'flowerpower'
  | 'cozywizard'
  | 'cyberpunk'
  | 'gothic'
  | 'ornate'
  | 'carbonfiber'
  | 'frameless';

export interface ThemeConfig {
  id: BorderTheme;
  name: string;
  accent: string;        // primary accent color
  accentAlt: string;     // secondary accent color
  textPrimary: string;   // heading/balance text
  textSecondary: string; // labels/muted text
  borderGlow: string;    // border/separator glow
  bgCard: string;        // card/panel background
  font: string;          // Google Font family
}

export const THEME_LIST: ThemeConfig[] = [
  {
    id: 'purple', name: 'Royal Purple', accent: '#9333ea', accentAlt: '#c084fc',
    textPrimary: '#f5e6ff', textSecondary: '#a78bfa', borderGlow: '#7c3aed',
    bgCard: 'rgba(88, 28, 135, 0.15)', font: '"Cinzel", serif',
  },
  {
    id: 'predator', name: 'Predator', accent: '#39ff14', accentAlt: '#00ff41',
    textPrimary: '#d4ffd4', textSecondary: '#7cfc00', borderGlow: '#00cc00',
    bgCard: 'rgba(0, 40, 0, 0.18)', font: '"Orbitron", sans-serif',
  },
  {
    id: 'inferno', name: 'Inferno', accent: '#ef4444', accentAlt: '#f97316',
    textPrimary: '#fff1e6', textSecondary: '#fb923c', borderGlow: '#dc2626',
    bgCard: 'rgba(153, 27, 27, 0.15)', font: '"Teko", sans-serif',
  },
  {
    id: 'abyssal', name: 'Abyssal Teal', accent: '#06b6d4', accentAlt: '#22d3ee',
    textPrimary: '#e6fcff', textSecondary: '#67e8f9', borderGlow: '#0891b2',
    bgCard: 'rgba(14, 83, 97, 0.15)', font: '"Exo 2", sans-serif',
  },
  {
    id: 'neon', name: 'Neon Cyber', accent: '#d946ef', accentAlt: '#06b6d4',
    textPrimary: '#fce7ff', textSecondary: '#e879f9', borderGlow: '#a21caf',
    bgCard: 'rgba(112, 26, 117, 0.15)', font: '"Audiowide", sans-serif',
  },
  {
    id: 'aurora', name: 'Aurora', accent: '#3b82f6', accentAlt: '#f59e0b',
    textPrimary: '#e6f0ff', textSecondary: '#93c5fd', borderGlow: '#2563eb',
    bgCard: 'rgba(30, 58, 138, 0.15)', font: '"Chakra Petch", sans-serif',
  },
  {
    id: 'hellfire', name: 'Hellfire', accent: '#f97316', accentAlt: '#ef4444',
    textPrimary: '#fff3e0', textSecondary: '#fdba74', borderGlow: '#ea580c',
    bgCard: 'rgba(124, 45, 18, 0.15)', font: '"Creepster", system-ui',
  },
  {
    id: 'alien', name: 'Alien Tech', accent: '#22c55e', accentAlt: '#4ade80',
    textPrimary: '#e6ffe6', textSecondary: '#86efac', borderGlow: '#16a34a',
    bgCard: 'rgba(20, 83, 45, 0.15)', font: '"Share Tech Mono", monospace',
  },
  {
    id: 'sovereign', name: 'Sovereign Gold', accent: '#d4a017', accentAlt: '#fbbf24',
    textPrimary: '#fff9e6', textSecondary: '#fcd34d', borderGlow: '#b45309',
    bgCard: 'rgba(120, 53, 15, 0.15)', font: '"Cinzel Decorative", serif',
  },
  {
    id: 'frost', name: 'Frost Crystal', accent: '#38bdf8', accentAlt: '#7dd3fc',
    textPrimary: '#e6f7ff', textSecondary: '#7dd3fc', borderGlow: '#0284c7',
    bgCard: 'rgba(12, 74, 110, 0.15)', font: '"Michroma", sans-serif',
  },
  {
    id: 'cosmic', name: 'Cosmic Void', accent: '#a855f7', accentAlt: '#c084fc',
    textPrimary: '#f3e8ff', textSecondary: '#c4b5fd', borderGlow: '#7c3aed',
    bgCard: 'rgba(76, 29, 149, 0.15)', font: '"Space Mono", monospace',
  },
  {
    id: 'metallic', name: 'Metallic Pro', accent: '#60a5fa', accentAlt: '#93c5fd',
    textPrimary: '#e0eaff', textSecondary: '#7dd3fc', borderGlow: '#2563eb',
    bgCard: 'rgba(30, 58, 138, 0.15)', font: '"Rajdhani", sans-serif',
  },
  {
    id: 'flowerpower', name: 'Flower Power', accent: '#f472b6', accentAlt: '#a78bfa',
    textPrimary: '#fef1f7', textSecondary: '#c4b5fd', borderGlow: '#e879a8',
    bgCard: 'rgba(120, 40, 90, 0.14)', font: '"Playfair Display", serif',
  },
  {
    id: 'cozywizard', name: 'Cozy Wizard', accent: '#d4a053', accentAlt: '#a8896c',
    textPrimary: '#f5efe6', textSecondary: '#c9b896', borderGlow: '#b8860b',
    bgCard: 'rgba(60, 40, 20, 0.22)', font: '"MedievalSharp", cursive',
  },
  {
    id: 'cyberpunk', name: 'Neon Cyberpunk', accent: '#ff2d95', accentAlt: '#00f0ff',
    textPrimary: '#e0f0ff', textSecondary: '#ff69b4', borderGlow: '#ff00ff',
    bgCard: 'rgba(80, 0, 60, 0.18)', font: '"Syncopate", sans-serif',
  },
  {
    id: 'gothic', name: 'Gothic Elegance', accent: '#c41e3a', accentAlt: '#c9b037',
    textPrimary: '#f5efe6', textSecondary: '#b8a07a', borderGlow: '#8b0a1e',
    bgCard: 'rgba(30, 8, 12, 0.24)', font: '"Cormorant Garamond", serif',
  },
  {
    id: 'ornate', name: 'Ornate', accent: '#c9a94e', accentAlt: '#3d3529',
    textPrimary: '#ede4d3', textSecondary: '#a89468', borderGlow: '#8a7339',
    bgCard: 'rgba(22, 19, 15, 0.38)', font: '"Uncial Antiqua", cursive',
  },
  {
    id: 'carbonfiber', name: 'Carbon Fiber', accent: '#dc2626', accentAlt: '#a0a0a8',
    textPrimary: '#e0dfe0', textSecondary: '#8a8a90', borderGlow: '#991b1b',
    bgCard: 'rgba(14, 14, 16, 0.4)', font: '"Barlow", sans-serif',
  },
  {
    id: 'frameless', name: 'Edge', accent: '#6366f1', accentAlt: '#818cf8',
    textPrimary: '#e0e7ff', textSecondary: '#a5b4fc', borderGlow: '#4f46e5',
    bgCard: 'rgba(30, 27, 75, 0.15)', font: '"Inter", system-ui, sans-serif',
  },
  {
    id: 'red', name: 'Blood Red', accent: '#dc2626', accentAlt: '#ef4444',
    textPrimary: '#fce4ec', textSecondary: '#f87171', borderGlow: '#991b1b',
    bgCard: 'rgba(127, 29, 29, 0.15)', font: '"Rajdhani", sans-serif',
  },
];

// Cache-busting version for frame images (increment when frames change)
export const FRAME_VERSION = 6;

function getStoredTheme(): BorderTheme {
  const stored = localStorage.getItem('borderTheme');
  if (stored && THEME_LIST.some(t => t.id === stored)) {
    return stored as BorderTheme;
  }
  return 'frameless';
}

/** Apply theme CSS variables + font to the document root */
function applyThemeVars(themeId: BorderTheme) {
  const theme = THEME_LIST.find(t => t.id === themeId);
  if (!theme) return;

  const root = document.documentElement;
  root.style.setProperty('--theme-accent', theme.accent);
  root.style.setProperty('--theme-accent-alt', theme.accentAlt);
  root.style.setProperty('--theme-text-primary', theme.textPrimary);
  root.style.setProperty('--theme-text-secondary', theme.textSecondary);
  root.style.setProperty('--theme-border-glow', theme.borderGlow);
  root.style.setProperty('--theme-bg-card', theme.bgCard);
  root.style.setProperty('--theme-font', theme.font);

  // Set data attribute for CSS selectors
  root.setAttribute('data-theme', themeId);
}

const AnimatedBorder: React.FC<AnimatedBorderProps> = ({
  children
}) => {
  const [currentTheme, setCurrentTheme] = useState<BorderTheme>(getStoredTheme);
  const [nextTheme, setNextTheme] = useState<BorderTheme>('red');
  const [isTransitioning, setIsTransitioning] = useState(false);

  // Apply theme vars on mount and when theme changes
  useEffect(() => {
    applyThemeVars(currentTheme);
  }, [currentTheme]);

  // Load Google Fonts for all themes on mount
  useEffect(() => {
    const fonts = [
      'Cinzel:wght@400;700',
      'Teko:wght@400;600;700',
      'Rajdhani:wght@400;600;700',
      'Orbitron:wght@400;600;700',
      'Exo+2:wght@400;600;700',
      'Audiowide',
      'Chakra+Petch:wght@400;600;700',
      'Creepster',
      'Share+Tech+Mono',
      'Cinzel+Decorative:wght@400;700',
      'Michroma',
      'Space+Mono:wght@400;700',
      'Playfair+Display:wght@400;600;700',
      'MedievalSharp',
      'Syncopate:wght@400;700',
      'Cormorant+Garamond:wght@400;600;700',
      'Uncial+Antiqua',
      'Barlow:wght@400;600;700',
    ];
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = `https://fonts.googleapis.com/css2?${fonts.map(f => `family=${f}`).join('&')}&display=swap`;
    document.head.appendChild(link);

    // Apply initial theme
    applyThemeVars(getStoredTheme());

    return () => { document.head.removeChild(link); };
  }, []);

  // Listen for theme changes from the theme chooser
  useEffect(() => {
    const handleThemeChange = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      const newTheme = detail?.theme as BorderTheme;
      if (newTheme && THEME_LIST.some(t => t.id === newTheme)) {
        setIsTransitioning(true);
        setNextTheme(newTheme);
        setTimeout(() => {
          setCurrentTheme(newTheme);
          setIsTransitioning(false);
        }, 1000);
      }
    };

    window.addEventListener('border-theme-change', handleThemeChange);
    return () => window.removeEventListener('border-theme-change', handleThemeChange);
  }, []);

  // Listen for transaction-sent events — pulse glow on current theme (no theme switch)
  useEffect(() => {
    const handleTransactionSent = () => {
      // Add a bright pulse effect to the current frame without switching themes
      const container = document.querySelector('.animated-border-container');
      if (container) {
        container.classList.add('transaction-pulse');
        setTimeout(() => {
          container.classList.remove('transaction-pulse');
        }, 2500);
      }
    };

    window.addEventListener('transaction-sent', handleTransactionSent);
    return () => window.removeEventListener('transaction-sent', handleTransactionSent);
  }, []);

  const themeAccent = THEME_LIST.find(t => t.id === currentTheme)?.accent || '#9333ea';

  return (
    <div className="animated-border-container">
      {/* Layer 1: Full frame BEHIND content (atmosphere/background) */}
      <div className={`border-frame border-frame-current ${isTransitioning ? 'fading-out' : ''}`}>
        <img
          src={`/borders/${currentTheme}/frame-full.png?v=${FRAME_VERSION}`}
          alt=""
          style={{ filter: `drop-shadow(0 0 20px ${themeAccent}66)` }}
        />
      </div>
      <div className={`border-frame border-frame-next ${isTransitioning ? 'fading-in' : ''}`}>
        <img src={`/borders/${nextTheme}/frame-full.png?v=${FRAME_VERSION}`} alt="" />
      </div>

      {/* Layer 2: Content (interactive UI) */}
      <div className="border-content">
        {children}
      </div>

      {/* Layer 3: Corner details ON TOP of content (masked to only show corners + edges) */}
      <div className={`border-frame-corners ${isTransitioning ? 'fading-out' : ''}`}>
        <img
          src={`/borders/${currentTheme}/frame-full.png?v=${FRAME_VERSION}`}
          alt=""
          style={{ filter: `drop-shadow(0 0 15px ${themeAccent}88)` }}
        />
      </div>

      <div className={`border-glow ${currentTheme} ${isTransitioning ? 'transitioning' : ''}`} />
    </div>
  );
};

export default AnimatedBorder;

export const flashBorderRed = () => {
  window.dispatchEvent(new CustomEvent('transaction-sent'));
};

export const changeBorderTheme = (theme: BorderTheme) => {
  localStorage.setItem('borderTheme', theme);
  window.dispatchEvent(new CustomEvent('border-theme-change', { detail: { theme } }));
};
