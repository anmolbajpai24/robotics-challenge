import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev mode: `npm run dev` proxies API calls to the FastAPI server on 8011,
// so the frontend and backend can run side by side with no CORS setup.
// Prod mode: `npm run build` emits static files into dashboard/static,
// which FastAPI serves directly (see INTEGRATION.md).
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../static',
    emptyOutDir: true
  },
  server: {
    proxy: {
      '/summary': 'http://127.0.0.1:8011',
      '/runs': 'http://127.0.0.1:8011',
      '/curves': 'http://127.0.0.1:8011',
      '/videos': 'http://127.0.0.1:8011',
      '/health': 'http://127.0.0.1:8011'
    }
  }
})
