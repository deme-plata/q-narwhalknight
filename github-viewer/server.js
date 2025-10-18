import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readdir, stat, readFile } from 'fs/promises';
import { existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = 3002;

// Path to the Q-NarwhalKnight repository
const REPO_PATH = '/opt/orobit/shared/q-narwhalknight';

// Middleware
app.use(cors());
app.use(express.json());

// Helper function to recursively get all files and folders
async function getFileTree(dirPath, basePath = '') {
  const items = [];

  try {
    const entries = await readdir(dirPath, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = join(dirPath, entry.name);
      const relativePath = basePath ? join(basePath, entry.name) : entry.name;

      // Skip common directories we don't want to show
      if (entry.name === 'node_modules' ||
          entry.name === 'target' ||
          entry.name === '.git' ||
          entry.name === 'dist' ||
          entry.name === 'dist-final' ||
          entry.name.startsWith('.')) {
        continue;
      }

      if (entry.isDirectory()) {
        const children = await getFileTree(fullPath, relativePath);
        items.push({
          path: relativePath,
          type: 'tree',
          name: entry.name,
          children
        });
      } else {
        const stats = await stat(fullPath);
        items.push({
          path: relativePath,
          type: 'blob',
          name: entry.name,
          size: stats.size
        });
      }
    }
  } catch (error) {
    console.error(`Error reading directory ${dirPath}:`, error);
  }

  return items;
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

// Endpoint: Get file tree
app.get('/api/tree', async (req, res) => {
  try {
    const tree = await getFileTree(REPO_PATH);

    res.json({
      sha: 'local',
      url: 'local',
      tree: flattenTree(tree),
      truncated: false
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Helper to flatten tree structure to match GitHub API format
function flattenTree(items, result = []) {
  for (const item of items) {
    result.push({
      path: item.path,
      type: item.type,
      size: item.size
    });

    if (item.children && item.children.length > 0) {
      flattenTree(item.children, result);
    }
  }

  return result;
}

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
