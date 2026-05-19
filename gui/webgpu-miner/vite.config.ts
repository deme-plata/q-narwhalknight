import { defineConfig } from 'vite';

export default defineConfig({
  // Deployed at https://quillon.xyz/webgpu-miner/ → vite needs the base
  // path so all asset URLs in dist/index.html are prefixed correctly.
  base: '/webgpu-miner/',
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
