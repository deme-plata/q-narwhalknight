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
    // Copy git clone command to clipboard and show notification
    navigator.clipboard.writeText('git clone https://code.quillon.xyz/repo.git').then(() => {
      const el = document.createElement('div');
      el.className = 'fixed bottom-6 right-6 bg-cyan-500/20 border border-cyan-500/50 px-6 py-3 rounded-lg text-cyan-400 font-mono text-sm z-50 backdrop-blur-sm';
      el.innerHTML = '&#x2705; Clone command copied to clipboard!<br/><code class="text-cyan-300">git clone https://code.quillon.xyz/repo.git</code>';
      document.body.appendChild(el);
      setTimeout(() => el.remove(), 4000);
    });
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
              <div className="text-center max-w-2xl">
                <div className="text-6xl mb-4">🔮</div>
                <h2 className="text-3xl font-bold text-cyan-400 font-mono mb-1">
                  Q-NarwhalKnight
                </h2>
                <p className="text-gray-500 font-mono text-sm mb-6">
                  Quantum-Enhanced DAG-BFT Consensus &middot; v8.0.0-mainnet &middot; 80+ crates
                </p>

                {/* Quick Start Cards */}
                <div className="grid grid-cols-3 gap-3 text-left mb-6">
                  <div className="bg-cyan-500/10 p-4 rounded-lg border border-cyan-500/30 hover:border-cyan-500/60 transition-colors cursor-pointer group"
                       onClick={() => handleFileSelect('crates/q-dag-knight/src/ordering_rules.rs')}>
                    <div className="text-cyan-400 font-mono text-xs mb-1 group-hover:text-cyan-300">DAG-Knight Consensus</div>
                    <div className="text-gray-400 font-mono text-[10px]">crates/q-dag-knight/</div>
                  </div>
                  <div className="bg-purple-500/10 p-4 rounded-lg border border-purple-500/30 hover:border-purple-500/60 transition-colors cursor-pointer group"
                       onClick={() => handleFileSelect('crates/q-api-server/src/main.rs')}>
                    <div className="text-purple-400 font-mono text-xs mb-1 group-hover:text-purple-300">API Server</div>
                    <div className="text-gray-400 font-mono text-[10px]">crates/q-api-server/</div>
                  </div>
                  <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/30 hover:border-green-500/60 transition-colors cursor-pointer group"
                       onClick={() => handleFileSelect('crates/q-miner/src/main.rs')}>
                    <div className="text-green-400 font-mono text-xs mb-1 group-hover:text-green-300">Miner</div>
                    <div className="text-gray-400 font-mono text-[10px]">crates/q-miner/</div>
                  </div>
                  <div className="bg-orange-500/10 p-4 rounded-lg border border-orange-500/30 hover:border-orange-500/60 transition-colors cursor-pointer group"
                       onClick={() => handleFileSelect('crates/q-storage/src/lib.rs')}>
                    <div className="text-orange-400 font-mono text-xs mb-1 group-hover:text-orange-300">Storage (RocksDB)</div>
                    <div className="text-gray-400 font-mono text-[10px]">crates/q-storage/</div>
                  </div>
                  <div className="bg-yellow-500/10 p-4 rounded-lg border border-yellow-500/30 hover:border-yellow-500/60 transition-colors cursor-pointer group"
                       onClick={() => handleFileSelect('crates/q-network/src/unified_network_manager.rs')}>
                    <div className="text-yellow-400 font-mono text-xs mb-1 group-hover:text-yellow-300">P2P Network</div>
                    <div className="text-gray-400 font-mono text-[10px]">crates/q-network/</div>
                  </div>
                  <div className="bg-pink-500/10 p-4 rounded-lg border border-pink-500/30 hover:border-pink-500/60 transition-colors cursor-pointer group"
                       onClick={() => handleFileSelect('crates/q-crypto-advanced/src/lib.rs')}>
                    <div className="text-pink-400 font-mono text-xs mb-1 group-hover:text-pink-300">Post-Quantum Crypto</div>
                    <div className="text-gray-400 font-mono text-[10px]">crates/q-crypto-advanced/</div>
                  </div>
                </div>

                {/* MCP Integration Banner */}
                <div className="bg-gradient-to-r from-cyan-500/10 via-purple-500/10 to-cyan-500/10 border border-cyan-500/30 rounded-xl p-5 text-left">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-purple-500 rounded-lg flex items-center justify-center text-white text-sm font-bold">AI</div>
                    <div>
                      <div className="text-cyan-400 font-mono text-sm font-bold">Claude Code MCP Integration</div>
                      <div className="text-gray-500 font-mono text-[10px]">Browse, search, and contribute using AI</div>
                    </div>
                  </div>
                  <pre className="bg-[#0a0e27] rounded-lg p-3 text-[11px] font-mono overflow-x-auto border border-cyan-500/20">
                    <span className="text-gray-500">// Add to ~/.claude/settings.json</span>{'\n'}
                    <span className="text-cyan-300">{`"mcpServers": {`}</span>{'\n'}
                    <span className="text-green-300">{`  "quillon-code": {`}</span>{'\n'}
                    <span className="text-yellow-300">{`    "type": "sse",`}</span>{'\n'}
                    <span className="text-yellow-300">{`    "url": "https://code.quillon.xyz/mcp/sse"`}</span>{'\n'}
                    <span className="text-green-300">{`  }`}</span>{'\n'}
                    <span className="text-cyan-300">{`}`}</span>
                  </pre>
                  <div className="flex gap-2 mt-3 flex-wrap">
                    <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 rounded text-[10px] font-mono">read_file</span>
                    <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 rounded text-[10px] font-mono">search_code</span>
                    <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 rounded text-[10px] font-mono">list_branches</span>
                    <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 rounded text-[10px] font-mono">view_diff</span>
                    <span className="px-2 py-0.5 bg-green-500/20 text-green-400 rounded text-[10px] font-mono">submit_contribution</span>
                  </div>
                </div>

                <p className="text-gray-600 font-mono text-xs mt-4">
                  Select a file from the explorer or click a crate above to start browsing
                </p>
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
