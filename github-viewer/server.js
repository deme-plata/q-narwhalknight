import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFile, stat, writeFile, mkdir } from 'fs/promises';
import { existsSync, statSync } from 'fs';
import { execSync } from 'child_process';
import { randomUUID } from 'crypto';

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
      full_name: 'quillon/q-narwhalknight',
      description: 'Q-NarwhalKnight - Quantum-Enhanced DAG-BFT Consensus System',
      html_url: 'https://code.quillon.xyz',
      clone_url: 'https://code.quillon.xyz/repo.git',
      stargazers_count: 0,
      forks_count: 0,
      watchers_count: 0,
      language: 'Rust',
      default_branch: 'main'
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

// Endpoint: List all branches
app.get('/api/branches', async (req, res) => {
  try {
    const output = execSync('git branch -a --format="%(refname:short)\t%(objectname:short)\t%(committerdate:iso8601)\t%(subject)"', {
      cwd: REPO_PATH,
      encoding: 'utf-8',
      maxBuffer: 10 * 1024 * 1024
    });

    const branches = output.trim().split('\n').filter(Boolean).map(line => {
      const [name, sha, date, ...messageParts] = line.split('\t');
      return {
        name: name.replace('origin/', ''),
        sha,
        date,
        message: messageParts.join('\t'),
        remote: name.startsWith('origin/'),
      };
    });

    // Deduplicate (local + remote of same branch)
    const seen = new Set();
    const unique = branches.filter(b => {
      if (seen.has(b.name)) return false;
      seen.add(b.name);
      return true;
    });

    res.json(unique);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Endpoint: Get diff between a branch and main (for merge request preview)
app.get('/api/diff/:branch', async (req, res) => {
  try {
    const branch = req.params.branch.replace(/[^a-zA-Z0-9._\-\/]/g, '');
    const base = req.query.base || 'main';

    // Get commit log for the branch
    const logOutput = execSync(
      `git log ${base}..${branch} --oneline --format="%H\t%s\t%an\t%ai" 2>/dev/null || echo ""`,
      { cwd: REPO_PATH, encoding: 'utf-8', maxBuffer: 5 * 1024 * 1024 }
    );

    const commits = logOutput.trim().split('\n').filter(Boolean).map(line => {
      const [hash, subject, author, date] = line.split('\t');
      return { hash, subject, author, date };
    });

    // Get diff stats
    const statOutput = execSync(
      `git diff --stat ${base}...${branch} 2>/dev/null || echo ""`,
      { cwd: REPO_PATH, encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 }
    );

    // Get full diff (limited to 500KB to prevent huge responses)
    let diffOutput = '';
    try {
      diffOutput = execSync(
        `git diff ${base}...${branch} 2>/dev/null | head -c 512000`,
        { cwd: REPO_PATH, encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 }
      );
    } catch {}

    // Get changed files list
    const filesOutput = execSync(
      `git diff --name-status ${base}...${branch} 2>/dev/null || echo ""`,
      { cwd: REPO_PATH, encoding: 'utf-8', maxBuffer: 5 * 1024 * 1024 }
    );

    const files = filesOutput.trim().split('\n').filter(Boolean).map(line => {
      const [status, ...pathParts] = line.split('\t');
      return {
        status: status === 'A' ? 'added' : status === 'D' ? 'deleted' : status === 'M' ? 'modified' : status,
        path: pathParts.join('\t'),
      };
    });

    res.json({
      branch,
      base,
      commits,
      files,
      stats: statOutput.trim(),
      diff: diffOutput,
      total_commits: commits.length,
      total_files: files.length,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Endpoint: Get commit log for a branch
app.get('/api/commits/:branch', async (req, res) => {
  try {
    const branch = req.params.branch.replace(/[^a-zA-Z0-9._\-\/]/g, '');
    const limit = Math.min(parseInt(req.query.limit) || 50, 200);

    const output = execSync(
      `git log ${branch} -n ${limit} --format="%H\t%h\t%s\t%an\t%ae\t%ai"`,
      { cwd: REPO_PATH, encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 }
    );

    const commits = output.trim().split('\n').filter(Boolean).map(line => {
      const [hash, short_hash, subject, author_name, author_email, date] = line.split('\t');
      return { hash, short_hash, subject, author_name, author_email, date };
    });

    res.json(commits);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Endpoint: Search code (grep through tracked files)
app.get('/api/search', async (req, res) => {
  try {
    const query = req.query.q;
    if (!query || query.length < 2) {
      return res.status(400).json({ error: 'Query must be at least 2 characters' });
    }

    // Sanitize query for shell safety
    const safeQuery = query.replace(/['"\\$`!]/g, '');
    const limit = Math.min(parseInt(req.query.limit) || 50, 100);

    const output = execSync(
      `git grep -n -i --count "${safeQuery}" HEAD -- '*.rs' '*.tsx' '*.ts' '*.js' '*.toml' '*.md' 2>/dev/null | head -${limit}`,
      { cwd: REPO_PATH, encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 }
    );

    const results = output.trim().split('\n').filter(Boolean).map(line => {
      // Format: HEAD:path:count
      const match = line.match(/^HEAD:(.+):(\d+)$/);
      if (match) return { path: match[1], count: parseInt(match[2]) };
      return null;
    }).filter(Boolean);

    res.json({ query, results, total: results.length });
  } catch (error) {
    // git grep returns exit code 1 when no matches
    if (error.status === 1) {
      return res.json({ query: req.query.q, results: [], total: 0 });
    }
    res.status(500).json({ error: error.message });
  }
});

// Endpoint: Get contributor list
app.get('/api/contributors', async (req, res) => {
  try {
    const output = execSync(
      'git shortlog -sne HEAD',
      { cwd: REPO_PATH, encoding: 'utf-8', maxBuffer: 5 * 1024 * 1024 }
    );

    const contributors = output.trim().split('\n').filter(Boolean).map(line => {
      const match = line.trim().match(/^(\d+)\t(.+)\s<(.+)>$/);
      if (match) return { commits: parseInt(match[1]), name: match[2].trim(), email: match[3] };
      return null;
    }).filter(Boolean);

    res.json(contributors);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ============================================================================
// MCP (Model Context Protocol) SSE Transport for Claude Code Integration
// READ-ONLY access to the Q-NarwhalKnight repository.
// Contributions are saved as proposals in .contributions/ — NEVER applied to repo.
// No code alteration, no git push, no file writes to the repository.
// ============================================================================

// Contributions directory for patches submitted via MCP (OUTSIDE the git tree)
const CONTRIB_PATH = '/opt/orobit/contributions';

// Rate limiting: max requests per IP per minute
const rateLimitMap = new Map();
const RATE_LIMIT_WINDOW_MS = 60_000;
const RATE_LIMIT_MAX = 60; // 60 requests/minute per IP

function rateLimit(ip) {
  const now = Date.now();
  const entry = rateLimitMap.get(ip) || { count: 0, reset: now + RATE_LIMIT_WINDOW_MS };
  if (now > entry.reset) {
    entry.count = 0;
    entry.reset = now + RATE_LIMIT_WINDOW_MS;
  }
  entry.count++;
  rateLimitMap.set(ip, entry);
  return entry.count > RATE_LIMIT_MAX;
}

// Clean up rate limit entries every 5 minutes
setInterval(() => {
  const now = Date.now();
  for (const [ip, entry] of rateLimitMap) {
    if (now > entry.reset + RATE_LIMIT_WINDOW_MS) rateLimitMap.delete(ip);
  }
}, 300_000);

// Files/directories that must NEVER be readable via MCP
const BLOCKED_PATHS = [
  '.env', '.env.local', '.env.production',
  'node_signing_key', 'libp2p_identity',
  'encryption', '.keys', '.key',
  '.git/', 'node-data/',
  '.contributions/',
];

function isPathBlocked(filePath) {
  const lower = filePath.toLowerCase();
  return BLOCKED_PATHS.some(blocked => lower.includes(blocked));
}

// Validate a path is safe (no traversal, no sensitive files)
function validatePath(filePath) {
  // Normalize and strip leading slashes
  const cleaned = filePath.replace(/\\/g, '/').replace(/^\/+/, '');
  // Block directory traversal
  if (cleaned.includes('..') || cleaned.includes('\0')) return null;
  // Block absolute paths
  if (filePath.startsWith('/')) return null;
  // Block sensitive files
  if (isPathBlocked(cleaned)) return null;
  // Resolve and verify it stays within repo
  const full = join(REPO_PATH, cleaned);
  if (!full.startsWith(REPO_PATH + '/')) return null;
  return cleaned;
}

// MCP tool definitions for Claude Code integration
// ALL tools are READ-ONLY. submit_contribution saves proposals outside the repo.
const MCP_TOOLS = [
  {
    name: 'read_file',
    description: 'Read a git-tracked file from the Q-NarwhalKnight repository (READ-ONLY)',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path relative to repo root (e.g. "crates/q-types/src/lib.rs")' },
      },
      required: ['path'],
    },
  },
  {
    name: 'search_code',
    description: 'Search for code patterns across git-tracked files (READ-ONLY)',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query (regex supported)' },
        file_type: { type: 'string', description: 'File extension filter: rs, tsx, ts, toml, md' },
      },
      required: ['query'],
    },
  },
  {
    name: 'list_branches',
    description: 'List all branches in the repository (READ-ONLY)',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'view_diff',
    description: 'View the diff of a branch compared to main (READ-ONLY)',
    inputSchema: {
      type: 'object',
      properties: {
        branch: { type: 'string', description: 'Branch name to diff' },
        base: { type: 'string', description: 'Base branch (default: main)' },
      },
      required: ['branch'],
    },
  },
  {
    name: 'list_files',
    description: 'List all git-tracked files in the repository (READ-ONLY)',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'submit_contribution',
    description: 'Submit a code contribution proposal for maintainer review. This does NOT modify the repository — proposals are stored separately and reviewed by maintainers before any merge.',
    inputSchema: {
      type: 'object',
      properties: {
        title: { type: 'string', description: 'Short title for the contribution (max 200 chars)' },
        description: { type: 'string', description: 'Detailed description of the bug fix or feature (max 5000 chars)' },
        diff: { type: 'string', description: 'Unified diff format of proposed changes (max 50000 chars)' },
        author: { type: 'string', description: 'Author name or Claude Code user identifier (max 100 chars)' },
        severity: { type: 'string', enum: ['Critical', 'High', 'Medium', 'Low'], description: 'Bug severity if this is a bug fix' },
      },
      required: ['title', 'description', 'diff'],
    },
  },
  {
    name: 'list_contributions',
    description: 'List all submitted contribution proposals pending review (READ-ONLY)',
    inputSchema: { type: 'object', properties: {} },
  },
];

// Active SSE connections for MCP
const mcpSessions = new Map();

// Max concurrent MCP sessions (prevent resource exhaustion)
const MAX_MCP_SESSIONS = 20;

// MCP SSE endpoint - Claude Code connects here (READ-ONLY)
app.get('/mcp/sse', (req, res) => {
  const clientIp = req.headers['x-real-ip'] || req.ip;

  // Rate limit
  if (rateLimit(clientIp)) {
    return res.status(429).json({ error: 'Too many requests. Try again in 1 minute.' });
  }

  // Session limit
  if (mcpSessions.size >= MAX_MCP_SESSIONS) {
    return res.status(503).json({ error: 'Server busy. Too many active sessions.' });
  }

  const sessionId = randomUUID();

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*',
  });

  // Send endpoint URL for the client to POST messages to
  res.write(`event: endpoint\ndata: /mcp/message?sessionId=${sessionId}\n\n`);

  mcpSessions.set(sessionId, { res, ip: clientIp, created: Date.now() });

  req.on('close', () => {
    mcpSessions.delete(sessionId);
  });

  // Keep-alive ping every 30s, auto-expire after 1 hour
  const keepAlive = setInterval(() => {
    const session = mcpSessions.get(sessionId);
    if (!session) {
      clearInterval(keepAlive);
      return;
    }
    // Auto-expire after 1 hour
    if (Date.now() - session.created > 3600_000) {
      mcpSessions.delete(sessionId);
      try { res.end(); } catch {}
      clearInterval(keepAlive);
      return;
    }
    try { res.write(': ping\n\n'); } catch { clearInterval(keepAlive); }
  }, 30000);
});

// MCP message endpoint - receives JSON-RPC messages (READ-ONLY operations)
app.post('/mcp/message', express.json({ limit: '1mb' }), async (req, res) => {
  const clientIp = req.headers['x-real-ip'] || req.ip;

  // Rate limit
  if (rateLimit(clientIp)) {
    return res.status(429).json({ error: 'Too many requests' });
  }

  const sessionId = req.query.sessionId;
  const session = mcpSessions.get(sessionId);

  if (!session) {
    return res.status(404).json({ error: 'Session not found' });
  }

  const sseRes = session.res;

  const { jsonrpc, id, method, params } = req.body;

  let result;

  try {
    switch (method) {
      case 'initialize':
        result = {
          protocolVersion: '2024-11-05',
          capabilities: { tools: {} },
          serverInfo: {
            name: 'quillon-code',
            version: '1.0.0',
          },
        };
        break;

      case 'tools/list':
        result = { tools: MCP_TOOLS };
        break;

      case 'tools/call':
        result = await handleToolCall(params.name, params.arguments || {});
        break;

      case 'notifications/initialized':
        // Client acknowledges initialization - no response needed
        res.json({ jsonrpc: '2.0', id });
        return;

      default:
        result = { error: { code: -32601, message: `Method not found: ${method}` } };
    }
  } catch (error) {
    result = { content: [{ type: 'text', text: `Error: ${error.message}` }], isError: true };
  }

  const response = { jsonrpc: '2.0', id, result };

  // Send via SSE
  sseRes.write(`event: message\ndata: ${JSON.stringify(response)}\n\n`);

  // Also send as HTTP response
  res.json(response);
});

// Handle MCP tool calls — ALL READ-ONLY except submit_contribution (writes to isolated dir)
async function handleToolCall(toolName, args) {
  switch (toolName) {
    case 'read_file': {
      // SECURITY: Use `git show HEAD:path` to ONLY serve git-tracked content
      // This prevents reading .env, keys, untracked files, or anything outside the repo
      const safePath = validatePath(args.path || '');
      if (!safePath) {
        return { content: [{ type: 'text', text: 'Access denied: invalid or restricted path' }], isError: true };
      }
      try {
        const output = execSync(`git show "HEAD:${safePath}"`, {
          cwd: REPO_PATH, encoding: 'utf-8', maxBuffer: 5 * 1024 * 1024, timeout: 10000,
        });
        return { content: [{ type: 'text', text: output }] };
      } catch {
        return { content: [{ type: 'text', text: `File not found in git: ${safePath}` }], isError: true };
      }
    }

    case 'search_code': {
      // SECURITY: Strict sanitization of query and file_type
      const query = (args.query || '').replace(/['"\\$`!;|&(){}\[\]<>]/g, '').slice(0, 200);
      if (query.length < 2) {
        return { content: [{ type: 'text', text: 'Query too short (min 2 chars)' }], isError: true };
      }
      // Whitelist allowed file extensions
      const ALLOWED_TYPES = ['rs', 'tsx', 'ts', 'js', 'toml', 'md', 'json', 'css', 'html'];
      let typeFilter;
      if (args.file_type && ALLOWED_TYPES.includes(args.file_type.replace(/[^a-z]/g, ''))) {
        typeFilter = `-- '*.${args.file_type.replace(/[^a-z]/g, '')}'`;
      } else {
        typeFilter = "-- '*.rs' '*.tsx' '*.ts' '*.toml' '*.md'";
      }
      try {
        const output = execSync(
          `git grep -n -i -- "${query}" HEAD ${typeFilter} 2>/dev/null | head -100`,
          { cwd: REPO_PATH, encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024, timeout: 15000 }
        );
        return { content: [{ type: 'text', text: output || 'No matches found' }] };
      } catch {
        return { content: [{ type: 'text', text: 'No matches found' }] };
      }
    }

    case 'list_branches': {
      try {
        const output = execSync('git branch -a --format="%(refname:short) %(objectname:short) %(subject)"', {
          cwd: REPO_PATH, encoding: 'utf-8', maxBuffer: 5 * 1024 * 1024, timeout: 10000,
        });
        return { content: [{ type: 'text', text: output }] };
      } catch {
        return { content: [{ type: 'text', text: 'Error listing branches' }], isError: true };
      }
    }

    case 'view_diff': {
      const branch = (args.branch || '').replace(/[^a-zA-Z0-9._\-\/]/g, '').slice(0, 100);
      const base = (args.base || 'main').replace(/[^a-zA-Z0-9._\-\/]/g, '').slice(0, 100);
      if (!branch) {
        return { content: [{ type: 'text', text: 'Branch name required' }], isError: true };
      }
      try {
        const output = execSync(`git diff --stat "${base}...${branch}" 2>/dev/null`, {
          cwd: REPO_PATH, encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024, timeout: 15000,
        });
        return { content: [{ type: 'text', text: output || 'No differences or branch not found' }] };
      } catch {
        return { content: [{ type: 'text', text: 'Branch not found or no diff available' }] };
      }
    }

    case 'list_files': {
      try {
        const output = execSync('git ls-tree -r --name-only HEAD | head -500', {
          cwd: REPO_PATH, encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024, timeout: 10000,
        });
        return { content: [{ type: 'text', text: output }] };
      } catch {
        return { content: [{ type: 'text', text: 'Error listing files' }], isError: true };
      }
    }

    case 'submit_contribution': {
      // SECURITY: This saves a text-only proposal OUTSIDE the repository.
      // It NEVER writes to the repo, NEVER modifies git, NEVER executes patches.
      // Maintainers manually review proposals and apply them if approved.
      const title = (args.title || '').slice(0, 200);
      const description = (args.description || '').slice(0, 5000);
      const diff = (args.diff || '').slice(0, 50000);
      const author = (args.author || 'anonymous-claude-code-user').replace(/[^a-zA-Z0-9._\-@ ]/g, '').slice(0, 100);

      if (!title || !description) {
        return { content: [{ type: 'text', text: 'Title and description are required' }], isError: true };
      }

      const patchId = randomUUID().slice(0, 8);
      const timestamp = new Date().toISOString();

      // Ensure contributions directory exists (OUTSIDE repo)
      if (!existsSync(CONTRIB_PATH)) {
        await mkdir(CONTRIB_PATH, { recursive: true });
      }

      const contribution = {
        id: patchId,
        title,
        description,
        author,
        severity: args.severity || null,
        diff_length: diff.length,
        submitted_at: timestamp,
        status: 'pending_review',
      };

      // Save contribution metadata (JSON only, no executable content)
      const metaPath = join(CONTRIB_PATH, `${patchId}.json`);
      await writeFile(metaPath, JSON.stringify(contribution, null, 2));

      // Save the diff as a plain .patch file (text only)
      if (diff) {
        const patchPath = join(CONTRIB_PATH, `${patchId}.patch`);
        await writeFile(patchPath, diff);
      }

      console.log(`📬 Contribution proposal: ${patchId} - "${title}" by ${author}`);

      return {
        content: [{
          type: 'text',
          text: `Contribution proposal submitted!\n\nPatch ID: ${patchId}\nTitle: ${title}\nDiff size: ${diff.length} chars\nStatus: pending_review\n\nMaintainers will review your proposal. This does NOT modify the repository — it is stored as a proposal only. Thank you for contributing to Q-NarwhalKnight!`,
        }],
      };
    }

    case 'list_contributions': {
      if (!existsSync(CONTRIB_PATH)) {
        return { content: [{ type: 'text', text: 'No contributions submitted yet.' }] };
      }
      const { readdirSync } = await import('fs');
      const files = readdirSync(CONTRIB_PATH).filter(f => f.endsWith('.json'));
      if (files.length === 0) {
        return { content: [{ type: 'text', text: 'No contributions submitted yet.' }] };
      }
      const contributions = [];
      for (const file of files.slice(0, 50)) { // Cap at 50 results
        try {
          const data = JSON.parse(await readFile(join(CONTRIB_PATH, file), 'utf-8'));
          contributions.push(`[${data.status}] ${data.id}: ${data.title} (by ${data.author}, ${data.submitted_at})`);
        } catch {}
      }
      return { content: [{ type: 'text', text: contributions.join('\n') || 'No contributions found.' }] };
    }

    default:
      return { content: [{ type: 'text', text: `Unknown tool: ${toolName}` }], isError: true };
  }
}

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', repo_path: REPO_PATH, mcp: true });
});

app.listen(PORT, () => {
  console.log(`🚀 Quillon Code Server running on http://localhost:${PORT}`);
  console.log(`📁 Serving repository: ${REPO_PATH} (READ-ONLY)`);
  console.log(`🔒 Contributions stored at: ${CONTRIB_PATH} (isolated from repo)`);
  console.log(`🔍 REST Endpoints:`);
  console.log(`   GET /api/repo          - Repository info`);
  console.log(`   GET /api/tree          - File tree`);
  console.log(`   GET /api/contents/*    - File content`);
  console.log(`   GET /api/raw/*         - Raw file download`);
  console.log(`   GET /api/branches      - List branches`);
  console.log(`   GET /api/diff/:branch  - Branch diff (merge request preview)`);
  console.log(`   GET /api/commits/:branch - Commit log`);
  console.log(`   GET /api/search?q=     - Code search`);
  console.log(`   GET /api/contributors  - Contributor list`);
  console.log(`🤖 MCP Endpoints (Claude Code integration):`);
  console.log(`   GET /mcp/sse           - SSE transport (READ-ONLY tools)`);
  console.log(`   POST /mcp/message      - JSON-RPC message handler`);
});
