/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'quantum-dark': '#0a0b0f',
        'quantum-indigo': '#1e1b4b',
        'quantum-purple': '#7c3aed',
        'quantum-cyan': '#06b6d4',
        'quantum-pink': '#ec4899',
        'quantum-green': '#10b981',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
