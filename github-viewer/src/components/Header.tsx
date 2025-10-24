import { Star, GitFork, Eye, Download, ExternalLink, BookOpen } from 'lucide-react';
import { useState } from 'react';
import type { GitHubRepo, FileTreeNode } from '../types/github';
import { Documentation } from './Documentation';
import { Search } from './Search';

interface HeaderProps {
  repoInfo: GitHubRepo | null;
  fileTree: FileTreeNode | null;
  onDownloadRepo: () => void;
  onFileSelect: (path: string) => void;
}

export function Header({ repoInfo, fileTree, onDownloadRepo, onFileSelect }: HeaderProps) {
  const [showDocs, setShowDocs] = useState(false);

  return (
    <>
      {showDocs && <Documentation onClose={() => setShowDocs(false)} />}
      <header className="h-20 bg-[#050714] border-b-2 border-cyan-500 flex items-center justify-between px-8">
      {/* Left: Logo and Title */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-magenta-500 rounded-lg flex items-center justify-center">
            <span className="text-2xl font-bold text-white">Q</span>
          </div>
          <div>
            <h1 className="text-2xl font-bold text-cyan-400 font-mono tracking-wide">
              Quillon Source Code
            </h1>
            <p className="text-sm text-gray-400 font-mono">
              Quantum-Enhanced DAG-BFT Consensus
            </p>
          </div>
        </div>
      </div>

      {/* Center: Stats */}
      {repoInfo && (
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-gray-300">
            <Star size={18} className="text-yellow-400" />
            <span className="font-mono text-sm">{repoInfo.stargazers_count}</span>
          </div>

          <div className="flex items-center gap-2 text-gray-300">
            <GitFork size={18} className="text-cyan-400" />
            <span className="font-mono text-sm">{repoInfo.forks_count}</span>
          </div>

          <div className="flex items-center gap-2 text-gray-300">
            <Eye size={18} className="text-magenta-400" />
            <span className="font-mono text-sm">{repoInfo.watchers_count}</span>
          </div>

          <div className="px-3 py-1 bg-green-500/20 text-green-400 rounded font-mono text-xs">
            {repoInfo.language}
          </div>
        </div>
      )}

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
          <span className="hidden xl:inline">How to Contribute</span>
        </button>

        <button
          onClick={onDownloadRepo}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-500/20 hover:bg-cyan-500/30
                   text-cyan-400 rounded-lg transition-all duration-200 font-mono text-sm
                   border border-cyan-500/30 hover:border-cyan-500/50
                   shadow-[0_0_15px_rgba(0,255,255,0.3)] hover:shadow-[0_0_25px_rgba(0,255,255,0.5)]"
        >
          <Download size={18} />
          Download ZIP
        </button>

        <a
          href="https://github.com/deme-plata/q-narwhalknight"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-4 py-2 bg-magenta-500/20 hover:bg-magenta-500/30
                   text-magenta-400 rounded-lg transition-all duration-200 font-mono text-sm
                   border border-magenta-500/30 hover:border-magenta-500/50
                   shadow-[0_0_15px_rgba(255,0,255,0.3)] hover:shadow-[0_0_25px_rgba(255,0,255,0.5)]"
        >
          <ExternalLink size={18} />
          GitHub
        </a>

        <a
          href="https://technical-deepdive.quillon.xyz"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-4 py-2 bg-yellow-500/20 hover:bg-yellow-500/30
                   text-yellow-400 rounded-lg transition-all duration-200 font-mono text-sm
                   border border-yellow-500/30 hover:border-yellow-500/50
                   shadow-[0_0_15px_rgba(255,255,0,0.3)] hover:shadow-[0_0_25px_rgba(255,255,0,0.5)]"
        >
          📊 Presentation
        </a>
      </div>
    </header>
    </>
  );
}
