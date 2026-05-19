import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5174,
    // Proxy /api/* to the local Quillon node during development
    proxy: {
      '/api': {
        target: 'https://quillon.xyz',
        changeOrigin: true,
        secure: true,
      },
    },
  },
  build: {
    target: 'esnext',
    outDir: 'dist',
  },
  // Treat .wgsl files as raw strings via ?raw query suffix
  assetsInclude: ['**/*.wgsl'],
});
