# Q-NarwhalKnight Technical Deep Dive Presentation

Auto-playing presentation app for creating YouTube video content about Q-NarwhalKnight blockchain technology.

## 🎬 Live Demo

**URL**: https://technical-deepdive.quillon.xyz

## 🚀 Features

- **27 Comprehensive Slides** covering all aspects of Q-NarwhalKnight
- **Auto-play Mode** with configurable timing per slide
- **Cyberpunk/Terminal Aesthetic** with neon colors (cyan, magenta, green)
- **1920x1080 Resolution** optimized for OBS Studio recording
- **Keyboard Controls** for easy navigation
- **Progress Bar** showing overall presentation progress
- **Code Highlighting** for Rust examples
- **Visual Cue Reminders** for adding graphics/diagrams

## 📋 Content Covered

1. **Introduction** - Q-NarwhalKnight overview
2. **Problem Statement** - Blockchain trilemma
3. **Solution Overview** - DAG + Narwhal + DAG-Knight
4. **Performance Metrics** - 50k TPS, 2-3s finality
5. **Narwhal Mempool** - High-throughput batching
6. **DAG-Knight Consensus** - Zero-message BFT
7. **Crypto-Agile Framework** - 5-phase quantum transition
8. **libp2p Networking** - Modern P2P stack
9. **REST API & Streaming** - Real-time monitoring
10. **Quantum Visualization** - Rainbow-box technique
11. **Performance Benchmarks** - Test results
12. **Roadmap** - Future milestones
13. **Thank You** - Closing

## ⌨️ Keyboard Controls

- **Space / →**: Next slide (when paused)
- **←**: Previous slide (when paused)
- **P**: Play/Pause auto-play
- **R**: Reset to first slide
- **Esc**: Stop auto-play

## 🎥 Recording with OBS Studio

### Setup Instructions

1. **Add Browser Source** in OBS:
   - URL: `https://technical-deepdive.quillon.xyz`
   - Width: `1920`
   - Height: `1080`
   - Custom CSS (optional): None needed
   - FPS: `30`

2. **Recording Settings**:
   - Resolution: 1920x1080
   - FPS: 30 or 60
   - Format: MP4 (H.264)

3. **Workflow**:
   - Open presentation in browser source
   - Press **P** to start auto-play
   - Start OBS recording
   - Narrate over slides as they advance
   - Each slide has a timer showing remaining time
   - Total presentation time: ~23 minutes

4. **Tips**:
   - Practice narration before recording
   - Use the "Visual Cue" suggestions to add diagrams
   - Pause (Press P) if you need more time on a slide
   - Use keyboard shortcuts to navigate during pauses

## 🛠️ Development

### Local Development

```bash
cd presentation-app
npm install
npm run dev
```

Open http://localhost:5173

### Build for Production

```bash
npm run build
```

Output in `dist/` directory

### Deploy to Server

The app is already deployed to production at:
- **URL**: https://technical-deepdive.quillon.xyz
- **Server**: nginx with Let's Encrypt SSL
- **Location**: `/opt/orobit/shared/q-narwhalknight/presentation-app/dist`

To redeploy after changes:

```bash
npm run build
systemctl reload nginx
```

## 📁 Project Structure

```
presentation-app/
├── src/
│   ├── App.tsx          # Main presentation component
│   ├── App.css          # Cyberpunk styling
│   ├── slides.ts        # All 27 slides with content
│   ├── main.tsx         # React entry point
│   └── index.css        # Global styles
├── dist/                # Production build
├── package.json
└── README.md
```

## 🎨 Styling

The presentation uses a **cyberpunk/terminal aesthetic**:

- **Background**: Dark gradient (#050714 → #0a0e27)
- **Primary**: Cyan (#00ffff) - titles, borders
- **Secondary**: Magenta (#ff00ff) - metadata, code blocks
- **Accent**: Green (#00ff88) - headers, code
- **Warning**: Yellow (#ffff00) - visual cues
- **Text**: White (#ffffff) - content
- **Font**: Courier New (monospace)
- **Effects**: Neon glow, grid background, glowing titles

## 📊 Slide Timings

Total duration: ~23 minutes (adjustable per slide)

- Title slides: 5-10 seconds
- Content slides: 30-60 seconds
- Code slides: 60 seconds
- Complex concept slides: 45-60 seconds

Edit timings in `src/slides.ts` by changing the `duration` property.

## 🔧 Customization

### Adding New Slides

Edit `src/slides.ts`:

```typescript
{
  id: 28,
  title: "New Slide Title",
  duration: 45, // seconds
  content: [
    "Line 1 of content",
    "Line 2 of content",
    "",  // blank line
    "🎯 Header with emoji",
    "   • Indented bullet point"
  ],
  code: `optional code block`,
  language: "rust",
  visualCue: "Suggestion for visual graphics"
}
```

### Changing Colors

Edit CSS variables in `src/App.css`:

```css
:root {
  --cyan: #00ffff;
  --magenta: #ff00ff;
  --green: #00ff88;
  /* etc */
}
```

### Adjusting Layout

The presentation is locked to 1920x1080 for OBS recording. To change:

```css
.presentation {
  width: 1920px;
  height: 1080px;
}
```

## 📝 Content Source

All content is based on the technical deep dive manuscript:
`/opt/orobit/shared/q-narwhalknight/Q_NARWHALKNIGHT_NODE_TECHNICAL_DEEP_DIVE.md`

## 🌐 SSL Certificate

Let's Encrypt SSL certificate configured with auto-renewal:
- **Domain**: technical-deepdive.quillon.xyz
- **Expires**: 2026-01-07
- **Auto-renewal**: Enabled via certbot systemd timer

## 🎯 Production Notes

The presentation includes "Visual Cue" suggestions at the bottom of slides. These are reminders to add:
- Architecture diagrams
- Code flow animations
- Performance charts
- Comparison tables
- Network topology visualizations

You can add these as overlays in OBS or edit them into the video during post-production.

## 📧 Support

For issues or questions about the Q-NarwhalKnight project:
- GitHub: github.com/q-narwhalknight/core
- Email: info@q-narwhalknight.dev

---

**Built with**:
- ⚛️ React + TypeScript
- ⚡ Vite
- 🎨 Custom CSS (no framework)
- 🔐 HTTPS via Let's Encrypt
- 🌐 nginx web server

**Ready for recording!** 🎥
