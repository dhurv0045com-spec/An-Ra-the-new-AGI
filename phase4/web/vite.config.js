import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../../ui', // Builds to root level An-Ra/ui directory
    emptyOutDir: true, // Empties the target directory first
  }
})
