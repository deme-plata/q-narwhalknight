/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'cyber-dark': '#0a0e27',
        'cyber-darker': '#050714',
        'cyber-cyan': '#00ffff',
        'cyber-magenta': '#ff00ff',
        'cyber-green': '#00ff88',
        'cyber-yellow': '#ffff00',
      },
      boxShadow: {
        'glow-cyan': '0 0 10px rgba(0, 255, 255, 0.5)',
        'glow-magenta': '0 0 10px rgba(255, 0, 255, 0.5)',
        'glow-cyan-lg': '0 0 20px rgba(0, 255, 255, 0.7)',
        'glow-magenta-lg': '0 0 20px rgba(255, 0, 255, 0.7)',
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite',
      },
      keyframes: {
        glow: {
          '0%, 100%': { boxShadow: '0 0 15px rgba(0, 255, 255, 0.3)' },
          '50%': { boxShadow: '0 0 25px rgba(0, 255, 255, 0.6)' },
        },
      },
    },
  },
  plugins: [],
}
