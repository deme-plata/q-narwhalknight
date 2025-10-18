# GitHub Source Code Viewer - Implementation Plan 🚀

## Sophisticated Vite + React + TypeScript Viewer for Q-NarwhalKnight

A professional, feature-rich GitHub repository viewer that allows users to browse, search, and download the Quillon (Q-NarwhalKnight) source code.

---

## 🎯 Project Goals

### **Primary Objectives:**
1. **Browse Repository**: Navigate the Quillon codebase file tree
2. **View Source Code**: Syntax-highlighted code viewing with line numbers
3. **Download Files**: Individual files or entire repository as ZIP
4. **Search Functionality**: Search across codebase for keywords
5. **Documentation**: README, CLAUDE.md, and documentation files
6. **Statistics**: Repository stats (stars, forks, commits, contributors)
7. **Responsive Design**: Works on desktop, tablet, mobile
8. **Cyberpunk Theme**: Matches Quillon presentation aesthetic

### **Tech Stack:**
- **Framework**: Vite + React 18 + TypeScript
- **Styling**: TailwindCSS + cyberpunk custom theme
- **Code Highlighting**: Prism.js or Shiki
- **GitHub API**: REST API v3 (public access, no auth needed)
- **File Tree**: react-folder-tree or custom component
- **Icons**: Lucide React (modern, clean icons)
- **Deployment**: Nginx on same server as presentation

---

## 📋 Feature Breakdown

### **Core Features (Phase 1-2):**

#### **1. Repository Overview**
- GitHub stats (stars, forks, watchers, open issues)
- Last commit date and author
- Main language (Rust)
- License information (MIT/Apache)
- Quick links to GitHub, presentation, documentation

#### **2. File Tree Navigation**
- Hierarchical folder structure
- Expandable/collapsible folders
- File type icons (Rust, TypeScript, Markdown, etc.)
- Search/filter file tree
- Breadcrumb navigation
- "Pin" frequently accessed files

#### **3. Code Viewer**
- Syntax highlighting for 50+ languages
- Line numbers with click-to-copy
- Copy entire file button
- Download individual file
- Raw file view
- Responsive text sizing
- Search in file (Ctrl+F enhancement)

#### **4. File Operations**
- Download single file (as .rs, .ts, etc.)
- Download folder as ZIP
- Download entire repository as ZIP
- View file history (GitHub link)
- View blame/contributors (GitHub link)

#### **5. Search & Filter**
- Search across all files (file names)
- Advanced search (by extension, path, content)
- Filter by language
- Recent files history
- Bookmarked files

### **Advanced Features (Phase 3-4):**

#### **6. Documentation Viewer**
- README.md rendered with GitHub-flavored Markdown
- CLAUDE.md (development guide)
- Architecture diagrams
- Auto-generated docs from rustdoc
- API reference

#### **7. Dependency Graph**
- Cargo.toml visualization
- Crate dependency tree
- Inter-crate relationships
- External dependencies

#### **8. Statistics Dashboard**
- Lines of code by language
- Commit activity graph
- Top contributors
- Code churn analysis
- Repository health metrics

#### **9. Dark/Light Mode**
- Toggle between themes
- Cyberpunk (dark) as default
- Clean light mode option
- Persisted preference

#### **10. Mobile Optimization**
- Responsive file tree (drawer on mobile)
- Touch-friendly navigation
- Optimized code rendering
- Mobile-first design

---

## 🏗️ Phase-Based Implementation

### **Phase 1: Foundation (Week 1) - MVP**

**Goal**: Basic repository browser with file viewing

#### **Components to Build:**
1. **Layout Components**:
   - `App.tsx` - Main app shell
   - `Sidebar.tsx` - File tree navigation
   - `CodeViewer.tsx` - Syntax-highlighted code display
   - `Header.tsx` - Stats, search, download

2. **GitHub API Integration**:
   - `api/github.ts` - Fetch repository tree
   - `api/fetchFile.ts` - Fetch individual file content
   - `api/stats.ts` - Fetch repository statistics

3. **Utilities**:
   - `utils/syntax.ts` - Syntax highlighting setup
   - `utils/fileIcons.ts` - File type icon mapping
   - `utils/download.ts` - File download handlers

#### **User Stories:**
- ✅ As a user, I can see the repository file tree
- ✅ As a user, I can navigate through folders
- ✅ As a user, I can view a Rust source file with syntax highlighting
- ✅ As a user, I can copy code to clipboard
- ✅ As a user, I can download a file

#### **Deliverables:**
- Working file tree navigation
- Syntax-highlighted code viewer
- Basic download functionality
- Responsive layout (3-column: stats | tree | code)

---

### **Phase 2: Enhancement (Week 2) - Polish**

**Goal**: Advanced features and UX improvements

#### **New Components:**
1. **Search Components**:
   - `SearchBar.tsx` - Global file search
   - `FileSearchModal.tsx` - Advanced search modal
   - `SearchResults.tsx` - Search results display

2. **Documentation Components**:
   - `MarkdownViewer.tsx` - Render README, CLAUDE.md
   - `DocsSidebar.tsx` - Documentation navigation
   - `CodeBlock.tsx` - Enhanced code blocks in docs

3. **Download Components**:
   - `DownloadModal.tsx` - Download options (file, folder, ZIP)
   - `ZipGenerator.tsx` - Generate ZIP files client-side

#### **API Enhancements:**
- `api/search.ts` - Search across repository
- `api/markdown.ts` - Fetch and render Markdown
- `api/download.ts` - Generate ZIP archives

#### **User Stories:**
- ✅ As a user, I can search for files by name
- ✅ As a user, I can view README.md with formatting
- ✅ As a user, I can download entire repository as ZIP
- ✅ As a user, I can bookmark frequently accessed files
- ✅ As a user, I can view recent files

#### **Deliverables:**
- Fast file search with fuzzy matching
- Markdown rendering with syntax highlighting
- ZIP download for folders and entire repo
- Bookmark/recent files system
- Improved mobile responsiveness

---

### **Phase 3: Sophistication (Week 3) - Pro Features**

**Goal**: Advanced analysis and visualization

#### **New Components:**
1. **Visualization Components**:
   - `DependencyGraph.tsx` - Cargo dependency visualization
   - `StatsChart.tsx` - LOC, commit activity charts
   - `ContributorGraph.tsx` - Top contributors

2. **Analysis Components**:
   - `CodeMetrics.tsx` - Lines of code, complexity
   - `HealthDashboard.tsx` - Repository health score
   - `ActivityTimeline.tsx` - Commit timeline

3. **Navigation Components**:
   - `Breadcrumb.tsx` - Path breadcrumb navigation
   - `FileHistory.tsx` - Recently viewed files
   - `Minimap.tsx` - Code minimap (like VS Code)

#### **API Additions:**
- `api/commits.ts` - Fetch commit history
- `api/contributors.ts` - Fetch contributor data
- `api/analysis.ts` - Code metrics and analysis

#### **User Stories:**
- ✅ As a user, I can see dependency relationships
- ✅ As a user, I can view code statistics
- ✅ As a user, I can see commit history
- ✅ As a user, I can navigate with breadcrumbs
- ✅ As a user, I can see a code minimap

#### **Deliverables:**
- Interactive dependency graph (D3.js or vis.js)
- Code metrics dashboard
- Commit history timeline
- Breadcrumb navigation
- Code minimap for long files

---

### **Phase 4: Excellence (Week 4) - Final Polish**

**Goal**: Production-ready, feature-complete

#### **Final Components:**
1. **Settings & Preferences**:
   - `Settings.tsx` - User preferences panel
   - `ThemeToggle.tsx` - Dark/light mode toggle
   - `FontSizePicker.tsx` - Adjust code font size

2. **Integration Components**:
   - `PresentationLink.tsx` - Link to Quillon presentation
   - `DocumentationHub.tsx` - All documentation in one place
   - `QuickStart.tsx` - Getting started guide

3. **Performance Components**:
   - `VirtualList.tsx` - Virtualized file tree for performance
   - `CodeSplitting.tsx` - Lazy load large components
   - `CacheManager.tsx` - Cache GitHub API responses

#### **Polish & Optimization:**
- SEO optimization (meta tags, OpenGraph)
- Accessibility (ARIA labels, keyboard navigation)
- Loading states and error handling
- Performance optimization (code splitting, lazy loading)
- Analytics integration (optional)

#### **User Stories:**
- ✅ As a user, I can customize theme and font size
- ✅ As a user, I can navigate entirely with keyboard
- ✅ As a user, I can share direct links to files
- ✅ As a user, I experience fast load times (<2s)
- ✅ As a user, I can access from mobile with ease

#### **Deliverables:**
- Full settings panel with preferences
- Keyboard shortcuts (like VS Code)
- Performance optimization (90+ Lighthouse score)
- Complete documentation
- Production deployment

---

## 🎨 Design System

### **Cyberpunk Theme (Default):**

```css
:root {
  /* Background Colors */
  --bg-primary: #0a0e27;      /* Deep space blue */
  --bg-secondary: #050714;    /* Almost black */
  --bg-tertiary: #1a1f3a;     /* Elevated surfaces */

  /* Accent Colors */
  --cyan: #00ffff;            /* Primary accent */
  --magenta: #ff00ff;         /* Secondary accent */
  --green: #00ff88;           /* Success */
  --yellow: #ffff00;          /* Warning */
  --red: #ff0066;             /* Error */

  /* Text Colors */
  --text-primary: #ffffff;    /* Main text */
  --text-secondary: #8892b0;  /* Muted text */
  --text-tertiary: #495670;   /* Very muted */

  /* Glow Effects */
  --glow-cyan: 0 0 20px rgba(0, 255, 255, 0.5);
  --glow-magenta: 0 0 20px rgba(255, 0, 255, 0.5);
  --glow-green: 0 0 20px rgba(0, 255, 136, 0.5);
}
```

### **Component Styling:**

#### **Header:**
- Height: 70px
- Background: --bg-secondary with bottom border (--cyan)
- Quillon logo (left), stats (center), actions (right)
- Glassmorphism effect

#### **Sidebar (File Tree):**
- Width: 300px (collapsible to 50px)
- Background: --bg-tertiary
- File icons with colors
- Hover: --cyan glow
- Active file: --cyan background

#### **Code Viewer:**
- Background: --bg-primary
- Line numbers: --text-tertiary
- Code: Prism Cyberpunk theme
- Scrollbars: Custom styled (--cyan)
- Font: 'Fira Code', 'Cascadia Code', monospace

#### **Search Bar:**
- Floating modal (center screen)
- Backdrop blur
- Real-time fuzzy search results
- Keyboard navigation (↑↓, Enter)

---

## 🛠️ Technical Implementation

### **GitHub API Endpoints:**

```typescript
// Repository tree (entire file structure)
GET https://api.github.com/repos/deme-plata/q-narwhalknight/git/trees/main?recursive=1

// File content (raw)
GET https://api.github.com/repos/deme-plata/q-narwhalknight/contents/{path}

// Repository stats
GET https://api.github.com/repos/deme-plata/q-narwhalknight

// Commits
GET https://api.github.com/repos/deme-plata/q-narwhalknight/commits

// Contributors
GET https://api.github.com/repos/deme-plata/q-narwhalknight/contributors
```

### **File Structure:**

```
github-viewer/
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   ├── App.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── Footer.tsx
│   │   ├── code/
│   │   │   ├── CodeViewer.tsx
│   │   │   ├── CodeBlock.tsx
│   │   │   ├── LineNumbers.tsx
│   │   │   └── Minimap.tsx
│   │   ├── navigation/
│   │   │   ├── FileTree.tsx
│   │   │   ├── Breadcrumb.tsx
│   │   │   ├── SearchBar.tsx
│   │   │   └── FileHistory.tsx
│   │   ├── docs/
│   │   │   ├── MarkdownViewer.tsx
│   │   │   ├── DocsSidebar.tsx
│   │   │   └── TableOfContents.tsx
│   │   ├── stats/
│   │   │   ├── StatsOverview.tsx
│   │   │   ├── DependencyGraph.tsx
│   │   │   ├── CommitTimeline.tsx
│   │   │   └── ContributorList.tsx
│   │   └── ui/
│   │       ├── Button.tsx
│   │       ├── Modal.tsx
│   │       ├── Spinner.tsx
│   │       └── Toast.tsx
│   ├── api/
│   │   ├── github.ts
│   │   ├── fetchFile.ts
│   │   ├── search.ts
│   │   └── stats.ts
│   ├── utils/
│   │   ├── syntax.ts
│   │   ├── fileIcons.ts
│   │   ├── download.ts
│   │   └── cache.ts
│   ├── hooks/
│   │   ├── useFileTree.ts
│   │   ├── useCodeHighlight.ts
│   │   └── useGitHubAPI.ts
│   ├── types/
│   │   ├── github.ts
│   │   ├── file.ts
│   │   └── stats.ts
│   ├── App.tsx
│   ├── App.css
│   └── main.tsx
├── public/
│   ├── logos/
│   └── favicon.ico
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

### **Key Libraries:**

```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.24.0",
    "prismjs": "^1.29.0",
    "react-markdown": "^9.0.0",
    "lucide-react": "^0.400.0",
    "jszip": "^3.10.1",
    "d3": "^7.9.0",
    "fuse.js": "^7.0.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.5.0",
    "vite": "^7.1.0",
    "tailwindcss": "^3.4.0",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.38"
  }
}
```

---

## 📦 Deployment Strategy

### **URL Structure:**
- **Production**: https://code.quillon.xyz
- **Alternate**: https://github.quillon.xyz

### **Nginx Configuration:**

```nginx
server {
    listen 443 ssl http2;
    server_name code.quillon.xyz;

    ssl_certificate /etc/letsencrypt/live/code.quillon.xyz/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/code.quillon.xyz/privkey.pem;

    root /opt/orobit/shared/q-narwhalknight/github-viewer/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        # Proxy to GitHub API if needed
        proxy_pass https://api.github.com/;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### **CI/CD Pipeline:**

```bash
# Build script
#!/bin/bash
cd /opt/orobit/shared/q-narwhalknight/github-viewer
npm install
npm run build
sudo systemctl reload nginx
```

---

## 🎯 Success Metrics

### **Performance Targets:**
- **Initial Load**: <2 seconds (Lighthouse 90+)
- **Time to Interactive**: <3 seconds
- **Bundle Size**: <500 KB (gzipped)
- **API Response**: <1 second per request

### **User Experience:**
- **Mobile Responsive**: 100% usable on phones
- **Keyboard Navigation**: All features accessible via keyboard
- **Accessibility**: WCAG 2.1 AA compliant
- **Error Handling**: Graceful fallbacks for API failures

### **Feature Completeness:**
- **Browse**: 100% of repository tree accessible
- **View**: Syntax highlighting for 50+ languages
- **Download**: Files, folders, entire repo
- **Search**: File name, content, extension
- **Documentation**: All Markdown files rendered

---

## 🚀 Next Steps

### **Phase 1 Implementation (This Session):**

1. **Create project structure**:
   ```bash
   npm create vite@latest github-viewer -- --template react-ts
   cd github-viewer
   npm install
   ```

2. **Install dependencies**:
   ```bash
   npm install react-router-dom prismjs lucide-react jszip
   npm install -D tailwindcss autoprefixer postcss
   ```

3. **Setup Tailwind**:
   ```bash
   npx tailwindcss init -p
   ```

4. **Build core components**:
   - Layout (App, Header, Sidebar)
   - FileTree component
   - CodeViewer with syntax highlighting
   - GitHub API integration

5. **Test locally**:
   ```bash
   npm run dev
   ```

6. **Deploy to production**:
   ```bash
   npm run build
   # Copy dist to server
   ```

---

## 🎉 Expected Outcome

A sophisticated, production-ready GitHub source code viewer that:

✅ Looks professional with cyberpunk theme
✅ Provides excellent UX for browsing Quillon codebase
✅ Enables easy downloading of source code
✅ Offers advanced features (search, stats, docs)
✅ Works flawlessly on all devices
✅ Loads fast (<2s) and performs well
✅ Integrates seamlessly with Quillon brand

**The viewer will be a powerful tool for developers, researchers, and users to explore the Q-NarwhalKnight quantum consensus implementation!** 🚀🔮

---

## 📚 References

- **GitHub API Documentation**: https://docs.github.com/en/rest
- **Prism.js Themes**: https://prismjs.com/
- **React Folder Tree**: https://github.com/shunjizhan/react-folder-tree
- **JSZip**: https://stuk.github.io/jszip/
- **Fuse.js (Fuzzy Search)**: https://fusejs.io/

---

**Ready to implement Phase 1! Let's build the MVP.** 🛠️
