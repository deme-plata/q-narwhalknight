import { useEffect, useState } from 'react';
import { Copy, Download, Check, ExternalLink, FileText } from 'lucide-react';
import Prism from 'prismjs';
import 'prismjs/themes/prism-tomorrow.css';
import 'prismjs/components/prism-rust';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-toml';
import 'prismjs/components/prism-yaml';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-python';
import { getLanguageFromExtension, getFileExtension, downloadFile } from '../api/github';

interface CodeViewerProps {
  content: string;
  filename: string;
  path: string;
}

// Helper to detect file type
function getFileType(ext: string): 'image' | 'pdf' | 'video' | 'audio' | 'code' | 'binary' {
  const imageExts = ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'bmp', 'ico'];
  const videoExts = ['mp4', 'webm', 'ogg', 'mov'];
  const audioExts = ['mp3', 'wav', 'ogg', 'm4a'];

  if (ext === 'pdf') return 'pdf';
  if (imageExts.includes(ext)) return 'image';
  if (videoExts.includes(ext)) return 'video';
  if (audioExts.includes(ext)) return 'audio';

  // Check if it's likely binary
  const binaryExts = ['zip', 'tar', 'gz', 'exe', 'dll', 'so', 'dylib', 'bin'];
  if (binaryExts.includes(ext)) return 'binary';

  return 'code';
}

export function CodeViewer({ content, filename, path }: CodeViewerProps) {
  const [copied, setCopied] = useState(false);
  const [highlightedCode, setHighlightedCode] = useState('');

  const ext = getFileExtension(filename);
  const language = getLanguageFromExtension(ext);
  const fileType = getFileType(ext);

  // Get the raw file URL for binary files
  const API_BASE = import.meta.env.PROD ? '/api' : 'http://localhost:3002/api';
  const rawUrl = `${API_BASE}/raw/${path}`;

  useEffect(() => {
    if (fileType === 'code') {
      try {
        // Prism highlighting with emoji preservation
        const highlighted = Prism.highlight(
          content,
          Prism.languages[language] || Prism.languages.text,
          language
        );
        setHighlightedCode(highlighted);
      } catch (error) {
        console.error('Syntax highlighting error:', error);
        // Fallback: escape HTML but preserve emojis
        const escaped = content
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');
        setHighlightedCode(escaped);
      }
    }
  }, [content, language, fileType]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    // For binary files, use the raw URL
    if (fileType !== 'code') {
      const a = document.createElement('a');
      a.href = rawUrl;
      a.download = filename;
      a.click();
    } else {
      downloadFile(path, content);
    }
  };

  const lines = content.split('\n');

  // Render based on file type
  const renderContent = () => {
    switch (fileType) {
      case 'image':
        return (
          <div className="flex items-center justify-center h-full bg-[#0a0e27] p-8">
            <img
              src={rawUrl}
              alt={filename}
              className="max-w-full max-h-full object-contain rounded-lg shadow-[0_0_20px_rgba(0,255,255,0.3)]"
            />
          </div>
        );

      case 'pdf':
        return (
          <div className="flex flex-col items-center justify-center h-full bg-[#0a0e27] p-8">
            <FileText size={64} className="text-cyan-400 mb-4" />
            <p className="text-gray-400 mb-4 font-mono">PDF Viewer</p>
            <iframe
              src={rawUrl}
              className="w-full h-full border-2 border-cyan-500/30 rounded-lg"
              title={filename}
            />
            <p className="text-gray-500 text-sm mt-4 font-mono">
              If PDF doesn't display, <a href={rawUrl} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300">click here to open</a>
            </p>
          </div>
        );

      case 'video':
        return (
          <div className="flex items-center justify-center h-full bg-[#0a0e27] p-8">
            <video
              controls
              className="max-w-full max-h-full rounded-lg shadow-[0_0_20px_rgba(0,255,255,0.3)]"
            >
              <source src={rawUrl} />
              Your browser doesn't support video playback.
            </video>
          </div>
        );

      case 'audio':
        return (
          <div className="flex flex-col items-center justify-center h-full bg-[#0a0e27] p-8">
            <div className="w-full max-w-2xl">
              <p className="text-cyan-400 font-mono mb-4">{filename}</p>
              <audio controls className="w-full">
                <source src={rawUrl} />
                Your browser doesn't support audio playback.
              </audio>
            </div>
          </div>
        );

      case 'binary':
        return (
          <div className="flex flex-col items-center justify-center h-full bg-[#0a0e27] p-8">
            <FileText size={64} className="text-gray-500 mb-4" />
            <p className="text-gray-400 font-mono mb-2">Binary File</p>
            <p className="text-gray-500 text-sm font-mono mb-4">This file cannot be displayed</p>
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-4 py-2 bg-cyan-500/20 hover:bg-cyan-500/30
                       text-cyan-400 rounded transition-colors duration-150 font-mono"
            >
              <Download size={20} />
              Download File
            </button>
          </div>
        );

      default: // code
        return (
          <div className="flex-1 overflow-auto">
            <div className="flex">
              {/* Line Numbers */}
              <div className="flex-shrink-0 px-4 py-4 bg-[#050714] border-r border-cyan-500/20 select-none">
                {lines.map((_, index) => (
                  <div
                    key={index}
                    className="text-gray-600 text-right font-mono text-sm leading-6"
                  >
                    {index + 1}
                  </div>
                ))}
              </div>

              {/* Code */}
              <div className="flex-1 px-6 py-4 overflow-x-auto">
                <pre className="text-sm leading-6 font-mono">
                  <code
                    className={`language-${language}`}
                    dangerouslySetInnerHTML={{ __html: highlightedCode }}
                  />
                </pre>
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0a0e27]">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 bg-[#050714] border-b border-cyan-500/30">
        <div className="flex items-center gap-3">
          <span className="text-cyan-400 font-mono text-sm font-semibold">{filename}</span>
          {fileType === 'code' && (
            <>
              <span className="text-gray-500 text-xs font-mono">{lines.length} lines</span>
              <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 text-xs font-mono rounded">
                {language}
              </span>
            </>
          )}
          {fileType !== 'code' && (
            <span className="px-2 py-0.5 bg-magenta-500/20 text-magenta-400 text-xs font-mono rounded">
              {fileType}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {fileType === 'code' && (
            <button
              onClick={handleCopy}
              className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/10 hover:bg-cyan-500/20
                       text-cyan-400 rounded transition-colors duration-150 text-sm font-mono"
            >
              {copied ? <Check size={16} /> : <Copy size={16} />}
              {copied ? 'Copied!' : 'Copy'}
            </button>
          )}

          <button
            onClick={handleDownload}
            className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/10 hover:bg-cyan-500/20
                     text-cyan-400 rounded transition-colors duration-150 text-sm font-mono"
          >
            <Download size={16} />
            Download
          </button>

          <a
            href={`https://github.com/deme-plata/q-narwhalknight/blob/main/${path}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-3 py-1.5 bg-magenta-500/10 hover:bg-magenta-500/20
                     text-magenta-400 rounded transition-colors duration-150 text-sm font-mono"
          >
            <ExternalLink size={16} />
            GitHub
          </a>
        </div>
      </div>

      {/* Content */}
      {renderContent()}
    </div>
  );
}
