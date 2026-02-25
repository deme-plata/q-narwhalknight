import { Download, BookOpen, Terminal, GitBranch, Cpu, Bot } from 'lucide-react';
import { useState, useEffect } from 'react';
import type { GitHubRepo, FileTreeNode } from '../types/github';
import { Documentation } from './Documentation';
import { Search } from './Search';

interface HeaderProps {
  repoInfo: GitHubRepo | null;
  fileTree: FileTreeNode | null;
  onDownloadRepo: () => void;
  onFileSelect: (path: string) => void;
}

interface BranchInfo {
  name: string;
  sha: string;
  date: string;
  message: string;
}

const API_BASE = import.meta.env.PROD ? '/api' : 'http://localhost:3002/api';

export function Header({ fileTree, onDownloadRepo, onFileSelect }: HeaderProps) {
  const [showDocs, setShowDocs] = useState(false);
  const [branches, setBranches] = useState<BranchInfo[]>([]);
  const [fileCount, setFileCount] = useState(0);

  useEffect(() => {
    // Fetch branch count
    fetch(`${API_BASE}/branches`)
      .then(r => r.json())
      .then((data: BranchInfo[]) => setBranches(data))
      .catch(() => {});

    // Count files from tree
    if (fileTree) {
      const count = (node: FileTreeNode): number => {
        if (node.type === 'file') return 1;
        return (node.children || []).reduce((sum, c) => sum + count(c), 0);
      };
      setFileCount(count(fileTree));
    }
  }, [fileTree]);

  return (
    <>
      {showDocs && <Documentation onClose={() => setShowDocs(false)} />}
      <header className="h-20 bg-[#050714] border-b-2 border-cyan-500 flex items-center justify-between px-8">
      {/* Left: Logo and Title */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-purple-500 rounded-lg flex items-center justify-center shadow-[0_0_20px_rgba(0,255,255,0.4)]">
            <span className="text-2xl font-bold text-white">Q</span>
          </div>
          <div>
            <h1 className="text-2xl font-bold text-cyan-400 font-mono tracking-wide">
              Quillon Source Code
            </h1>
            <p className="text-sm text-gray-400 font-mono flex items-center gap-2">
              v8.0.0-mainnet
              <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 rounded text-[10px] font-semibold">LIVE</span>
            </p>
          </div>
        </div>
      </div>

      {/* Center: Stats */}
      <div className="flex items-center gap-5">
        <div className="flex items-center gap-2 text-gray-300" title="Language">
          <Cpu size={16} className="text-orange-400" />
          <span className="font-mono text-sm text-orange-400">Rust</span>
        </div>

        <div className="flex items-center gap-2 text-gray-300" title="Tracked files">
          <Terminal size={16} className="text-cyan-400" />
          <span className="font-mono text-sm">{fileCount.toLocaleString()} files</span>
        </div>

        <div className="flex items-center gap-2 text-gray-300" title="Branches">
          <GitBranch size={16} className="text-purple-400" />
          <span className="font-mono text-sm">{branches.length} branches</span>
        </div>

        <div className="px-2.5 py-1 bg-cyan-500/15 text-cyan-400 rounded-full font-mono text-[11px] border border-cyan-500/30 flex items-center gap-1.5"
             title="Claude Code MCP integration available">
          <Bot size={13} />
          MCP Ready
        </div>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-3">
        <Search fileTree={fileTree} onFileSelect={onFileSelect} />

        <button
          onClick={() => setShowDocs(true)}
          className="flex items-center gap-2 px-4 py-2 bg-green-500/20 hover:bg-green-500/30
                   text-green-400 rounded-lg transition-all duration-200 font-mono text-sm
                   border border-green-500/30 hover:border-green-500/50
                   shadow-[0_0_15px_rgba(0,255,136,0.3)] hover:shadow-[0_0_25px_rgba(0,255,136,0.5)]"
        >
          <BookOpen size={18} />
          <span className="hidden xl:inline">Contribute</span>
        </button>

        <button
          onClick={onDownloadRepo}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-500/20 hover:bg-cyan-500/30
                   text-cyan-400 rounded-lg transition-all duration-200 font-mono text-sm
                   border border-cyan-500/30 hover:border-cyan-500/50
                   shadow-[0_0_15px_rgba(0,255,255,0.3)] hover:shadow-[0_0_25px_rgba(0,255,255,0.5)]"
        >
          <Download size={18} />
          Clone
        </button>

        <a
          href="https://quillon.xyz"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-4 py-2 bg-purple-500/20 hover:bg-purple-500/30
                   text-purple-400 rounded-lg transition-all duration-200 font-mono text-sm
                   border border-purple-500/30 hover:border-purple-500/50
                   shadow-[0_0_15px_rgba(168,85,247,0.3)] hover:shadow-[0_0_25px_rgba(168,85,247,0.5)]"
        >
          <span className="text-base">🌐</span>
          Network
        </a>
      </div>
    </header>
    </>
  );
}
