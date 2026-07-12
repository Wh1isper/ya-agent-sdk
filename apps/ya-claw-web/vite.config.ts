import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const clawProxyTarget =
  process.env.VITE_CLAW_PROXY_TARGET ?? 'http://127.0.0.1:9042'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          router: ['@tanstack/react-router'],
          query: ['@tanstack/react-query'],
          radix: [
            '@radix-ui/react-dialog',
            '@radix-ui/react-dropdown-menu',
            '@radix-ui/react-tabs',
          ],
          forms: ['react-hook-form'],
          icons: ['lucide-react'],
          markdown: ['react-markdown', 'remark-gfm'],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: clawProxyTarget,
        changeOrigin: true,
      },
      '/healthz': {
        target: clawProxyTarget,
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    maxWorkers: 4,
    setupFiles: './src/test/setup.ts',
  },
})
