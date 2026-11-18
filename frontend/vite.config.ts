import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icon.svg'],
      manifest: {
        name: 'Nepali Speech Collector',
        short_name: 'Nepali Speech',
        description:
          'Open Nepali speech dataset collection tool — record prompts, review recordings, track progress.',
        start_url: '.',
        display: 'standalone',
        background_color: '#FAF6ED',
        theme_color: '#5B3EA8',
        lang: 'ne',
        icons: [
          {
            src: 'icon.svg',
            sizes: '512x512',
            type: 'image/svg+xml',
            purpose: 'any',
          },
        ],
      },
    }),
  ],
});
