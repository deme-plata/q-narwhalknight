import { useState } from 'react';
import { ChevronRight, ChevronDown, File, Folder, FolderOpen } from 'lucide-react';
import type { FileTreeNode } from '../types/github';
import { getFileExtension } from '../api/github';

interface FileTreeProps {
  node: FileTreeNode;
  onFileSelect: (path: string) => void;
  selectedPath?: string;
  level?: number;
}

export function FileTree({ node, onFileSelect, selectedPath, level = 0 }: FileTreeProps) {
  const [isExpanded, setIsExpanded] = useState(level < 2); // Auto-expand first 2 levels

  const isSelected = selectedPath === node.path;
  const hasChildren = node.children && node.children.length > 0;

  const handleClick = () => {
    if (node.type === 'folder') {
      setIsExpanded(!isExpanded);
    } else {
      onFileSelect(node.path);
    }
  };

  const getFileIcon = () => {
    if (node.type === 'folder') {
      return isExpanded ? <FolderOpen size={16} /> : <Folder size={16} />;
    }

    const ext = getFileExtension(node.name);
    const iconColor = getFileIconColor(ext);

    return <File size={16} style={{ color: iconColor }} />;
  };

  const getFileIconColor = (ext: string): string => {
    const colorMap: Record<string, string> = {
      rs: '#00ff88',      // Rust - green
      ts: '#00ffff',      // TypeScript - cyan
      tsx: '#00ffff',     // TypeScript - cyan
      js: '#ffff00',      // JavaScript - yellow
      jsx: '#ffff00',     // JavaScript - yellow
      md: '#ff00ff',      // Markdown - magenta
      json: '#8892b0',    // JSON - gray
      toml: '#8892b0',    // TOML - gray
      yaml: '#8892b0',    // YAML - gray
      yml: '#8892b0',     // YAML - gray
      sh: '#00ff88',      // Shell - green
      lock: '#495670',    // Lock files - dark gray
    };

    return colorMap[ext] || '#ffffff';
  };

  return (
    <div>
      <div
        className={`
          flex items-center gap-2 px-3 py-1.5 cursor-pointer
          transition-all duration-150
          ${isSelected ? 'bg-cyan-500/20 border-l-2 border-cyan-500' : 'hover:bg-white/5'}
          ${level > 0 ? 'ml-' + (level * 4) : ''}
        `}
        onClick={handleClick}
        style={{ paddingLeft: `${level * 16 + 12}px` }}
      >
        {hasChildren && (
          <span className="text-cyan-400">
            {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </span>
        )}
        {!hasChildren && <span className="w-4" />}

        <span className="flex-shrink-0">{getFileIcon()}</span>

        <span
          className={`
            text-sm font-mono truncate
            ${isSelected ? 'text-cyan-400 font-semibold' : 'text-gray-300'}
          `}
        >
          {node.name}
        </span>
      </div>

      {isExpanded && hasChildren && (
        <div>
          {node.children!.map((child) => (
            <FileTree
              key={child.path}
              node={child}
              onFileSelect={onFileSelect}
              selectedPath={selectedPath}
              level={level + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}
