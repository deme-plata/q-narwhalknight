import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: './dist-final',
    emptyOutDir: false, // IMPORTANT: Don't delete downloads folder!
    chunkSizeWarningLimit: 1000, // Warn on chunks >1MB
    rollupOptions: {
      output: {
        // Content hashes handle cache busting — no Date.now() needed
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]',
        manualChunks: {
          'vendor-react': ['react', 'react-dom'],
          'vendor-three': ['three', '@react-three/fiber', '@react-three/drei'],
          'vendor-motion': ['framer-motion'],
          'vendor-crypto': ['@noble/ed25519', '@noble/hashes', 'crystals-kyber', 'dilithium-crystals'],
          'vendor-p2p': [
            'libp2p',
            '@chainsafe/libp2p-gossipsub',
            '@chainsafe/libp2p-noise',
            '@chainsafe/libp2p-yamux',
            '@libp2p/bootstrap',
            '@libp2p/websockets',
            '@libp2p/kad-dht',
            '@libp2p/identify',
            '@libp2p/ping',
          ],
        },
      }
    }
  },
  esbuild: {
    // Console logs temporarily enabled for signaling debug (re-enable drop after fix)
    // drop: ['console', 'debugger'],
  },
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        secure: false,
        ws: true
      }
    }
  }
})
