import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        quantum: {
          purple: '#9333ea',
          blue: '#3b82f6',
          cyan: '#06b6d4',
          pink: '#ec4899',
        },
      },
      backgroundImage: {
        'quantum-gradient': 'linear-gradient(135deg, #9333ea, #3b82f6, #06b6d4)',
      },
    },
  },
  plugins: [],
} satisfies Config
