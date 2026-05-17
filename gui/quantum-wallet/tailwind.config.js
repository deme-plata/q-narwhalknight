/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        quantum: {
          dark: '#0A0B14',
          darker: '#050510',
          indigo: '#1A1B26',
          violet: '#2D1B69',
          purple: '#6B46C1',
          cyan: '#00D9FF',
          green: '#00FF88',
          pink: '#FF0080',
          yellow: '#FFD700',
          blue: '#0080FF',
        }
      },
      animation: {
        'quantum-pulse': 'quantum-pulse 2s ease-in-out infinite',
        'photon-flow': 'photon-flow 3s linear infinite',
        'entangle': 'entangle 4s ease-in-out infinite',
        'collapse': 'collapse 0.5s ease-out',
        'fractal-bloom': 'fractal-bloom 1s ease-out',
        'rainbow-shift': 'rainbow-shift 5s linear infinite',
      },
      keyframes: {
        'quantum-pulse': {
          '0%, 100%': { opacity: 0.4, transform: 'scale(1)' },
          '50%': { opacity: 1, transform: 'scale(1.05)' },
        },
        'photon-flow': {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        'entangle': {
          '0%, 100%': { transform: 'rotate(0deg) scale(1)' },
          '50%': { transform: 'rotate(180deg) scale(1.1)' },
        },
        'collapse': {
          '0%': { filter: 'blur(10px)', opacity: 0.3 },
          '100%': { filter: 'blur(0px)', opacity: 1 },
        },
        'fractal-bloom': {
          '0%': { transform: 'scale(0) rotate(0deg)' },
          '100%': { transform: 'scale(1) rotate(360deg)' },
        },
        'rainbow-shift': {
          '0%': { filter: 'hue-rotate(0deg)' },
          '100%': { filter: 'hue-rotate(360deg)' },
        }
      },
      backgroundImage: {
        'quantum-gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #00d9ff 75%, #00ff88 100%)',
        'photon-gradient': 'linear-gradient(180deg, transparent, rgba(0, 217, 255, 0.4), transparent)',
        'entanglement': 'radial-gradient(circle, rgba(107, 70, 193, 0.3) 0%, transparent 70%)',
      }
    },
  },
  plugins: [],
}