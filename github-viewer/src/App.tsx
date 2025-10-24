import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { FileTree } from './components/FileTree';
import { CodeViewer } from './components/CodeViewer';
import {
  fetchRepoInfo,
  fetchRepositoryTree,
  fetchFileContent,
  buildFileTree,
} from './api/github';
import type { GitHubRepo, FileTreeNode } from './types/github';
import { Loader2, AlertCircle } from 'lucide-react';
import './App.css';

function App() {
  const [repoInfo, setRepoInfo] = useState<GitHubRepo | null>(null);
  const [fileTree, setFileTree] = useState<FileTreeNode | null>(null);
  const [selectedPath, setSelectedPath] = useState<string>('');
  const [fileContent, setFileContent] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [loadingFile, setLoadingFile] = useState(false);
  const [error, setError] = useState<string>('');

  // Load repository data on mount
  useEffect(() => {
    loadRepositoryData();
  }, []);

  async function loadRepositoryData() {
    try {
      setLoading(true);
      setError('');

      const [info, tree] = await Promise.all([
        fetchRepoInfo(),
        fetchRepositoryTree(),
      ]);

      setRepoInfo(info);
      setFileTree(buildFileTree(tree));

      // Auto-select README.md if it exists
      const readmePath = 'README.md';
      const hasReadme = tree.tree.some(item => item.path === readmePath);
      if (hasReadme) {
        handleFileSelect(readmePath);
      }
    } catch (err) {
      console.error('Failed to load repository:', err);
      setError('Failed to load repository data. Please try again later.');
    } finally {
      setLoading(false);
    }
  }

  async function handleFileSelect(path: string) {
    try {
      setLoadingFile(true);
      setSelectedPath(path);
      setError('');

      const content = await fetchFileContent(path);
      setFileContent(content);
    } catch (err) {
      console.error('Failed to load file:', err);
      setError(`Failed to load file: ${path}`);
      setFileContent('');
    } finally {
      setLoadingFile(false);
    }
  }

  function handleDownloadRepo() {
    window.open(
      'https://github.com/deme-plata/q-narwhalknight/archive/refs/heads/main.zip',
      '_blank'
    );
  }

  if (loading) {
    return (
      <div className="h-screen w-screen bg-[#0a0e27] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 size={48} className="text-cyan-400 animate-spin" />
          <p className="text-cyan-400 font-mono text-lg">Loading Quillon codebase...</p>
        </div>
      </div>
    );
  }

  if (error && !fileTree) {
    return (
      <div className="h-screen w-screen bg-[#0a0e27] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 max-w-md">
          <AlertCircle size={48} className="text-red-400" />
          <p className="text-red-400 font-mono text-center">{error}</p>
          <button
            onClick={loadRepositoryData}
            className="px-6 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg font-mono
                     hover:bg-cyan-500/30 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0a0e27] overflow-hidden">
      <Header
        repoInfo={repoInfo}
        fileTree={fileTree}
        onDownloadRepo={handleDownloadRepo}
        onFileSelect={handleFileSelect}
      />

      <div className="flex-1 flex overflow-hidden">
        {/* File Tree Sidebar */}
        <div className="w-80 bg-[#050714] border-r border-cyan-500/30 overflow-y-auto">
          <div className="p-4 border-b border-cyan-500/20">
            <h2 className="text-cyan-400 font-mono text-sm font-semibold flex items-center gap-2">
              📁 File Explorer
            </h2>
          </div>

          {fileTree && (
            <FileTree
              node={fileTree}
              onFileSelect={handleFileSelect}
              selectedPath={selectedPath}
            />
          )}
        </div>

        {/* Code Viewer */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {loadingFile ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="flex flex-col items-center gap-4">
                <Loader2 size={32} className="text-cyan-400 animate-spin" />
                <p className="text-cyan-400 font-mono text-sm">Loading file...</p>
              </div>
            </div>
          ) : fileContent && selectedPath ? (
            <CodeViewer
              content={fileContent}
              filename={selectedPath.split('/').pop() || ''}
              path={selectedPath}
            />
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center max-w-lg">
                <div className="text-6xl mb-4">🔮</div>
                <h2 className="text-2xl font-bold text-cyan-400 font-mono mb-2">
                  Quillon Source Code Viewer
                </h2>
                <p className="text-gray-400 font-mono text-sm mb-6">
                  Select a file from the explorer to view its contents
                </p>
                <div className="grid grid-cols-2 gap-4 text-left">
                  <div className="bg-cyan-500/10 p-4 rounded-lg border border-cyan-500/30">
                    <div className="text-cyan-400 font-mono text-xs mb-1">Core Consensus</div>
                    <div className="text-gray-300 font-mono text-xs">crates/q-dag-knight/</div>
                  </div>
                  <div className="bg-magenta-500/10 p-4 rounded-lg border border-magenta-500/30">
                    <div className="text-magenta-400 font-mono text-xs mb-1">API Server</div>
                    <div className="text-gray-300 font-mono text-xs">crates/q-api-server/</div>
                  </div>
                  <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/30">
                    <div className="text-green-400 font-mono text-xs mb-1">Crypto</div>
                    <div className="text-gray-300 font-mono text-xs">crates/q-quantum-crypto/</div>
                  </div>
                  <div className="bg-yellow-500/10 p-4 rounded-lg border border-yellow-500/30">
                    <div className="text-yellow-400 font-mono text-xs mb-1">Networking</div>
                    <div className="text-gray-300 font-mono text-xs">crates/q-network/</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {error && fileTree && (
            <div className="absolute bottom-4 right-4 bg-red-500/20 border border-red-500/50
                          px-4 py-2 rounded-lg text-red-400 font-mono text-sm flex items-center gap-2">
              <AlertCircle size={16} />
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Background Grid Effect */}
      <div
        className="fixed inset-0 pointer-events-none opacity-10"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0, 255, 255, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
        }}
      />
    </div>
  );
}

export default App;
