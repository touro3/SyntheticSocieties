import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
  plugins: [vue()],
  base: '/',
  build: {
    outDir: path.resolve(__dirname, '../api/static'),
    emptyOutDir: true,
  },
  server: {
    port: 3000,
    proxy: {
      '/health':            'http://localhost:5050',
      '/ping':              'http://localhost:5050',
      '/simulate':          'http://localhost:5050',
      '/simulate-wizard':   'http://localhost:5050',
      '/status':            'http://localhost:5050',
      '/results':           'http://localhost:5050',
      '/replay':            'http://localhost:5050',
      '/experiments':       'http://localhost:5050',
      '/interview':         'http://localhost:5050',
      '/inject':            'http://localhost:5050',
      '/anchor':            'http://localhost:5050',
      '/report':            'http://localhost:5050',
      '/incomplete':        'http://localhost:5050',
      '/configs':           'http://localhost:5050',
      '/upload-ess-data':   'http://localhost:5050',
      '/design-simulation': 'http://localhost:5050',
      '/api':               'http://localhost:5050',
      '/human-eval':        'http://localhost:5050',
      '/human-game':        'http://localhost:5050',
      '/benchmark':         'http://localhost:5050',
    },
  },
})
