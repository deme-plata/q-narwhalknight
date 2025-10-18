import type { GitHubRepo, GitHubTree, GitHubFileContent, FileTreeNode } from '../types/github';

// Use local backend API instead of GitHub API
const API_BASE = import.meta.env.PROD
  ? '/api'  // Production: use nginx proxy
  : 'http://localhost:3002/api';  // Development: direct to backend

// Cache for API responses
const cache = new Map<string, { data: any; timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

function getCached<T>(key: string): T | null {
  const cached = cache.get(key);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data as T;
  }
  return null;
}

function setCache(key: string, data: any) {
  cache.set(key, { data, timestamp: Date.now() });
}

/**
 * Fetch repository information
 */
export async function fetchRepoInfo(): Promise<GitHubRepo> {
  const cacheKey = 'repo-info';
  const cached = getCached<GitHubRepo>(cacheKey);
  if (cached) return cached;

  const response = await fetch(`${API_BASE}/repo`);
  if (!response.ok) {
    throw new Error(`Failed to fetch repository info: ${response.statusText}`);
  }

  const data = await response.json();
  setCache(cacheKey, data);
  return data;
}

/**
 * Fetch entire repository tree
 */
export async function fetchRepositoryTree(): Promise<GitHubTree> {
  const cacheKey = 'repo-tree';
  const cached = getCached<GitHubTree>(cacheKey);
  if (cached) return cached;

  const response = await fetch(`${API_BASE}/tree`);

  if (!response.ok) {
    throw new Error(`Failed to fetch repository tree: ${response.statusText}`);
  }

  const data = await response.json();
  setCache(cacheKey, data);
  return data;
}

/**
 * Fetch file content
 */
export async function fetchFileContent(path: string): Promise<string> {
  const cacheKey = `file-${path}`;
  const cached = getCached<string>(cacheKey);
  if (cached) return cached;

  const response = await fetch(`${API_BASE}/contents/${path}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch file: ${response.statusText}`);
  }

  const data: GitHubFileContent = await response.json();

  if (data.content && data.encoding === 'base64') {
    const content = atob(data.content.replace(/\n/g, ''));
    setCache(cacheKey, content);
    return content;
  }

  throw new Error('File content not available');
}

/**
 * Build hierarchical file tree from flat GitHub tree
 */
export function buildFileTree(githubTree: GitHubTree): FileTreeNode {
  const root: FileTreeNode = {
    name: 'q-narwhalknight',
    path: '',
    type: 'folder',
    children: [],
  };

  // Sort items to ensure folders come before files
  const sortedItems = [...githubTree.tree].sort((a, b) => {
    if (a.type !== b.type) {
      return a.type === 'tree' ? -1 : 1;
    }
    return a.path.localeCompare(b.path);
  });

  for (const item of sortedItems) {
    if (item.type === 'blob') {
      insertFileNode(root, item.path, item.size);
    }
  }

  return root;
}

function insertFileNode(root: FileTreeNode, path: string, size?: number) {
  const parts = path.split('/');
  let current = root;

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    const isLastPart = i === parts.length - 1;

    if (!current.children) {
      current.children = [];
    }

    let existing = current.children.find(child => child.name === part);

    if (!existing) {
      existing = {
        name: part,
        path: parts.slice(0, i + 1).join('/'),
        type: isLastPart ? 'file' : 'folder',
        children: isLastPart ? undefined : [],
        size: isLastPart ? size : undefined,
      };
      current.children.push(existing);
    }

    if (!isLastPart) {
      current = existing;
    }
  }
}

/**
 * Get file extension
 */
export function getFileExtension(filename: string): string {
  const parts = filename.split('.');
  return parts.length > 1 ? parts[parts.length - 1].toLowerCase() : '';
}

/**
 * Get programming language from file extension
 */
export function getLanguageFromExtension(ext: string): string {
  const languageMap: Record<string, string> = {
    rs: 'rust',
    ts: 'typescript',
    tsx: 'typescript',
    js: 'javascript',
    jsx: 'javascript',
    md: 'markdown',
    json: 'json',
    toml: 'toml',
    yaml: 'yaml',
    yml: 'yaml',
    sh: 'bash',
    py: 'python',
    c: 'c',
    cpp: 'cpp',
    h: 'c',
    hpp: 'cpp',
    css: 'css',
    html: 'html',
    xml: 'xml',
    sql: 'sql',
    go: 'go',
    java: 'java',
    kt: 'kotlin',
    swift: 'swift',
    rb: 'ruby',
    php: 'php',
    cs: 'csharp',
    scala: 'scala',
  };

  return languageMap[ext] || 'text';
}

/**
 * Format file size
 */
export function formatFileSize(bytes: number | undefined): string {
  if (!bytes) return 'N/A';

  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

/**
 * Download file content
 */
export async function downloadFile(path: string, content: string) {
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = path.split('/').pop() || 'file.txt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
