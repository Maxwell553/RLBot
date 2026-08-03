/// <reference types="vitest/config" />
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const frontendDir = path.dirname(fileURLToPath(import.meta.url))

/**
 * Dev boots with ``frontend/node_modules`` symlinked to a /tmp npm install
 * (see scripts/dev.mjs) so optimizeDeps never crawls iCloud placeholders.
 */
export default defineConfig({
  plugins: [react(), tailwindcss()],
  cacheDir: process.env.VITE_CACHE_DIR || '/tmp/markettrainer-vite-cache',
  optimizeDeps: {
    noDiscovery: true,
    holdUntilCrawlEnd: false,
    include: [
      'react',
      'react/jsx-runtime',
      'react/jsx-dev-runtime',
      'react-dom',
      'react-dom/client',
      'react-router',
      'react-router-dom',
      'clsx',
      'tailwind-merge',
      'framer-motion',
      'lucide-react',
      'yaml',
    ],
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    preTransformRequests: false,
    // Warm only the boot path — heavy ops pages lazy-load on navigation.
    warmup: {
      clientFiles: [
        './src/main.tsx',
        './src/App.tsx',
        './src/pages/LandingPage.tsx',
        './src/index.css',
      ],
    },
    proxy: {
      // Optional: preflight / force-forward only. Page data comes from /data/*.json.
      '/api': {
        target: 'http://127.0.0.1:8787',
        changeOrigin: true,
        timeout: 30_000,
        proxyTimeout: 30_000,
      },
      // Same-origin workflow path — avoids browser CORS while :8790 boots.
      '/workflow-api': {
        target: 'http://127.0.0.1:8790',
        changeOrigin: true,
        rewrite: (p: string) => p.replace(/^\/workflow-api/, ''),
        timeout: 30_000,
        proxyTimeout: 30_000,
      },
    },
    watch: {
      // Function form catches Finder duplicates like "node_modules 2" (space in name).
      ignored: (watchPath: string) => {
        const p = watchPath.replace(/\\/g, '/')
        return (
          p.includes('/node_modules') ||
          p.includes('/.node_modules.icloud') ||
          p.includes('/Runs/') ||
          p.includes('/.venv/') ||
          p.includes('/execution/') ||
          p.includes('/.cache/') ||
          p.includes('/dist/')
        )
      },
    },
    fs: {
      // macOS resolves /tmp → /private/tmp; allow both so @fontsource woff2 loads
      // when frontend/node_modules is symlinked to the /tmp install.
      allow: [
        frontendDir,
        path.resolve(frontendDir, '..'),
        '/tmp/markettrainer-frontend',
        '/private/tmp/markettrainer-frontend',
        '/tmp/markettrainer-vite-cache',
        '/private/tmp/markettrainer-vite-cache',
      ],
      deny: ['**/Runs/**', '**/.venv/**', '**/execution/logs/**'],
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
