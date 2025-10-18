# Q-NarwhalKnight Technical Deep Dive - Setup Complete ✅

## 🎉 Your presentation is live and ready for recording!

### 🌐 Live URL
**https://technical-deepdive.quillon.xyz**

### ✅ What's Been Set Up

1. **Vite + React + TypeScript App**
   - 27 comprehensive slides covering Q-NarwhalKnight architecture
   - Auto-play functionality with per-slide timing
   - Cyberpunk/terminal aesthetic (neon cyan, magenta, green)
   - 1920x1080 resolution optimized for OBS Studio

2. **nginx Web Server**
   - Configured at `/etc/nginx/sites-available/technical-deepdive.quillon.xyz`
   - Serving from `/opt/orobit/shared/q-narwhalknight/presentation-app/dist`
   - Gzip compression enabled
   - Static asset caching (1 year)

3. **Let's Encrypt SSL Certificate**
   - Domain: technical-deepdive.quillon.xyz
   - Expires: 2026-01-07
   - Auto-renewal configured via certbot
   - HTTPS redirect enabled

4. **Production Build**
   - Built and deployed to `dist/` directory
   - Optimized assets with gzip
   - Total bundle: ~220KB (68KB gzipped)

## 🎥 How to Record with OBS Studio

### Quick Start

1. **Open OBS Studio**

2. **Add Browser Source**:
   - Click **+** in Sources
   - Choose **Browser**
   - Settings:
     - **URL**: `https://technical-deepdive.quillon.xyz`
     - **Width**: `1920`
     - **Height**: `1080`
     - **FPS**: `30`
     - **Custom CSS**: Leave empty

3. **Start Recording**:
   - Visit the site in the browser source
   - Press **P** to start auto-play
   - Click **Start Recording** in OBS
   - Narrate over the slides as they auto-advance
   - Total duration: ~23 minutes

4. **Controls During Recording**:
   - **P**: Pause/resume auto-play
   - **Space/→**: Next slide (when paused)
   - **←**: Previous slide (when paused)
   - **R**: Reset to beginning
   - **Esc**: Stop auto-play

### Recording Tips

- Practice your narration before recording
- The timer shows seconds remaining on each slide
- Pause (press P) if you need more time on a slide
- Yellow "Visual Cue" boxes suggest where to add diagrams
- You can add overlays in OBS or edit graphics in post-production

## 📋 Slide Overview (27 slides, ~23 minutes)

1. **Title** - Q-NarwhalKnight intro (5s)
2. **Problem** - Blockchain trilemma (45s)
3. **Solution** - Architecture overview (60s)
4. **Metrics** - Performance targets (30s)
5. **Narwhal** - Mempool component (45s)
6. **Code** - Narwhal implementation (60s)
7. **Vertex** - Data structure (45s)
8. **DAG-Knight** - Consensus algorithm (60s)
9. **Anchor** - Election mechanism (45s)
10. **Ordering** - DAG example (60s)
11. **Crypto-Agile** - 5-phase strategy (60s)
12. **Code** - Crypto implementation (45s)
13. **Quantum** - Why it matters (45s)
14. **libp2p** - Networking stack (60s)
15. **Code** - Gossipsub handler (60s)
16. **Topics** - Network topics (45s)
17. **API** - REST & streaming (60s)
18. **Streaming** - Architecture (45s)
19. **Visualization** - Rainbow-box (60s)
20. **Use Cases** - Visualization value (45s)
21. **Benchmarks** - Performance results (60s)
22. **Comparison** - vs other systems (60s)
23. **Resources** - Memory & storage (45s)
24. **Roadmap** - Future milestones (60s)
25. **Open Source** - Contribution (45s)
26. **Resources** - Learning materials (45s)
27. **Thank You** - Closing (10s)

## 🎨 Visual Style

The presentation features a cyberpunk/terminal aesthetic:

- **Dark Background**: Gradient from #050714 to #0a0e27
- **Neon Colors**:
  - Cyan (#00ffff) - Titles, borders
  - Magenta (#ff00ff) - Metadata, code blocks
  - Green (#00ff88) - Headers, inline code
  - Yellow (#ffff00) - Visual cue hints
- **Typography**: Courier New monospace font
- **Effects**:
  - Glowing neon text shadows
  - Subtle grid background
  - Animated title glow
  - Progress bar with gradient

## 🔧 Updating the Presentation

### To Make Changes:

1. Edit slides in `src/slides.ts`
2. Edit styling in `src/App.css`
3. Rebuild:
   ```bash
   cd /opt/orobit/shared/q-narwhalknight/presentation-app
   npm run build
   systemctl reload nginx
   ```

### To Adjust Slide Timings:

Edit the `duration` field in `src/slides.ts`:

```typescript
{
  id: 5,
  title: "Component 1: Narwhal Mempool",
  duration: 45,  // ← Change this (in seconds)
  content: [ ... ]
}
```

## 📁 File Locations

- **Source Code**: `/opt/orobit/shared/q-narwhalknight/presentation-app/src/`
- **Production Build**: `/opt/orobit/shared/q-narwhalknight/presentation-app/dist/`
- **nginx Config**: `/etc/nginx/sites-available/technical-deepdive.quillon.xyz`
- **SSL Certificate**: `/etc/letsencrypt/live/technical-deepdive.quillon.xyz/`
- **Content Source**: `/opt/orobit/shared/q-narwhalknight/Q_NARWHALKNIGHT_NODE_TECHNICAL_DEEP_DIVE.md`

## 🚀 Next Steps

1. **Open the site**: https://technical-deepdive.quillon.xyz
2. **Test the controls**: Use keyboard shortcuts to navigate
3. **Set up OBS**: Add browser source with the URL above
4. **Record your video**: Press P to start, narrate over slides
5. **Edit & publish**: Add any graphics/overlays in post-production

## 📊 Technical Details

**Frontend**:
- React 18
- TypeScript 5
- Vite 7
- Custom CSS (no framework)

**Backend**:
- nginx 1.22.1
- Let's Encrypt SSL
- HTTP/2 enabled
- Gzip compression

**Performance**:
- Initial load: ~220KB
- Gzipped: ~68KB
- First Contentful Paint: <1s
- Time to Interactive: <1.5s

## 🎯 What Makes This Great for Recording

✅ **Fixed 1920x1080 resolution** - Perfect for Full HD video
✅ **Auto-play with timing** - Hands-free recording
✅ **Visual cue hints** - Reminders for diagrams
✅ **High contrast colors** - Looks great on camera
✅ **Monospace font** - Technical aesthetic
✅ **Progress tracking** - Know how much is left
✅ **Keyboard controls** - Easy to pause/navigate

## 📞 Support

If you need to make changes or have questions:
- Edit `src/slides.ts` for content
- Edit `src/App.css` for styling
- Run `npm run build` to rebuild
- Reload nginx with `systemctl reload nginx`

---

## 🎬 You're all set! Start recording your Q-NarwhalKnight deep dive video! 🚀

**Live site**: https://technical-deepdive.quillon.xyz

Press **P** to play, and enjoy creating your technical content! 🎥✨
