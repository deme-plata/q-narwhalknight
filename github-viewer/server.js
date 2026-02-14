import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFile } from 'fs/promises';
import { existsSync, statSync } from 'fs';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = 3002;

// Path to the Q-NarwhalKnight repository
const REPO_PATH = '/opt/orobit/shared/q-narwhalknight';

// Middleware
app.use(cors());
app.use(express.json());

// Get file tree using git ls-tree (respects .gitignore, only shows tracked files)
async function getGitFileTree() {
  try {
    const output = execSync('git ls-tree -r --name-only HEAD', {
      cwd: REPO_PATH,
      encoding: 'utf-8',
      maxBuffer: 50 * 1024 * 1024
    });

    const files = output.trim().split('\n').filter(Boolean);
    const result = [];

    // Track directories we've already added
    const dirs = new Set();

    for (const filePath of files) {
      // Add parent directories as tree entries
      const parts = filePath.split('/');
      for (let i = 1; i < parts.length; i++) {
        const dirPath = parts.slice(0, i).join('/');
        if (!dirs.has(dirPath)) {
          dirs.add(dirPath);
          result.push({ path: dirPath, type: 'tree' });
        }
      }

      // Add file entry - get size from filesystem if available
      let size = 0;
      try {
        const fullPath = join(REPO_PATH, filePath);
        if (existsSync(fullPath)) {
          size = statSync(fullPath).size;
        }
      } catch {}

      result.push({ path: filePath, type: 'blob', size });
    }

    return result;
  } catch (error) {
    console.error('Error running git ls-tree:', error.message);
    return [];
  }
}

// Endpoint: Get repository info
app.get('/api/repo', async (req, res) => {
  try {
    res.json({
      name: 'q-narwhalknight',
      full_name: 'local/q-narwhalknight',
      description: 'Q-NarwhalKnight - Quantum-Enhanced DAG-BFT Consensus System (Local Repository)',
      html_url: 'https://github.com/deme-plata/q-narwhalknight',
      stargazers_count: 0,
      forks_count: 0,
      watchers_count: 0,
      language: 'Rust',
      default_branch: 'clean-branch'
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Endpoint: Get file tree (uses git ls-tree, respects .gitignore)
app.get('/api/tree', async (req, res) => {
  try {
    const tree = await getGitFileTree();

    res.json({
      sha: 'local',
      url: 'local',
      tree,
      truncated: false
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Endpoint: Get file content
app.get('/api/contents/*', async (req, res) => {
  try {
    // Extract path from URL (everything after /api/contents/)
    const filePath = req.url.replace('/api/contents/', '');
    const fullPath = join(REPO_PATH, filePath);

    // Security check: ensure the path is within REPO_PATH
    if (!fullPath.startsWith(REPO_PATH)) {
      return res.status(403).json({ error: 'Access denied' });
    }

    if (!existsSync(fullPath)) {
      return res.status(404).json({ error: 'File not found' });
    }

    const stats = await stat(fullPath);

    if (stats.isDirectory()) {
      return res.status(400).json({ error: 'Path is a directory' });
    }

    // Read file as buffer to handle binary files
    const content = await readFile(fullPath);

    // Return in GitHub API format (base64 encoded)
    const base64Content = content.toString('base64');

    res.json({
      name: filePath.split('/').pop(),
      path: filePath,
      size: stats.size,
      content: base64Content,
      encoding: 'base64',
      download_url: `http://localhost:${PORT}/api/raw/${filePath}`
    });
  } catch (error) {
    if (error.code === 'ENOENT') {
      res.status(404).json({ error: 'File not found' });
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// Endpoint: Get raw file content (for downloads)
app.get('/api/raw/*', async (req, res) => {
  try {
    // Extract path from URL
    const filePath = req.url.replace('/api/raw/', '');
    const fullPath = join(REPO_PATH, filePath);

    // Security check
    if (!fullPath.startsWith(REPO_PATH)) {
      return res.status(403).send('Access denied');
    }

    if (!existsSync(fullPath)) {
      return res.status(404).send('File not found');
    }

    // Read as buffer and detect content type
    const content = await readFile(fullPath);
    const ext = fullPath.split('.').pop().toLowerCase();

    // Set appropriate content type
    const contentTypes = {
      'pdf': 'application/pdf',
      'png': 'image/png',
      'jpg': 'image/jpeg',
      'jpeg': 'image/jpeg',
      'gif': 'image/gif',
      'svg': 'image/svg+xml',
      'webp': 'image/webp',
      'mp4': 'video/mp4',
      'mp3': 'audio/mpeg',
      'zip': 'application/zip',
      'json': 'application/json',
      'md': 'text/markdown',
    };

    res.type(contentTypes[ext] || 'text/plain').send(content);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', repo_path: REPO_PATH });
});

app.listen(PORT, () => {
  console.log(`🚀 Local Git Server running on http://localhost:${PORT}`);
  console.log(`📁 Serving repository: ${REPO_PATH}`);
  console.log(`🔍 File tree endpoint: http://localhost:${PORT}/api/tree`);
});
