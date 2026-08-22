import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')
  const apiProxy = {
    '/api': {
      target: env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000',
      changeOrigin: true,
    },
  }
  const previewApiProxy = {
    '/api': {
      ...apiProxy['/api'],
      ...(env.RKNODE_PREVIEW_ADMIN_TOKEN
        ? { headers: { Authorization: `Bearer ${env.RKNODE_PREVIEW_ADMIN_TOKEN}` } }
        : {}),
    },
  }
  return {
    plugins: [react()],
    server: {
      watch: {
        ignored: ['**/.venv/**', '**/var/**', '**/.trellis/**'],
      },
      proxy: apiProxy,
    },
    preview: {
      proxy: previewApiProxy,
    },
  }
})
