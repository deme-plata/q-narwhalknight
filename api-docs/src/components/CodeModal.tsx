import { useState, useEffect } from 'react';
import { X, Copy, Check, Download, ExternalLink } from 'lucide-react';

interface CodeModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  fileUrl: string;
  language?: string;
}

export default function CodeModal({ isOpen, onClose, title, fileUrl, language = 'html' }: CodeModalProps) {
  const [code, setCode] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && !code) {
      fetchCode();
    }
  }, [isOpen]);

  const fetchCode = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(fileUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.statusText}`);
      }
      const text = await response.text();
      setCode(text);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load code');
      console.error('Error fetching code:', err);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadCode = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileUrl.split('/').pop() || 'code.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const openInNewTab = () => {
    window.open(fileUrl, '_blank');
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
         onClick={onClose}>
      <div className="relative w-full max-w-6xl max-h-[90vh] bg-quantum-dark border-2 border-quantum-purple/50 rounded-2xl shadow-2xl overflow-hidden"
           onClick={(e) => e.stopPropagation()}>

        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-quantum-purple/30 bg-quantum-indigo/20">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              {title}
            </h2>
            <p className="text-sm text-gray-400 mt-1">{fileUrl.split('/').pop()}</p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={copyToClipboard}
              className="p-2 rounded-lg bg-quantum-cyan/20 hover:bg-quantum-cyan/30 border border-quantum-cyan/30 transition-all"
              title="Copy to clipboard"
            >
              {copied ? <Check className="w-5 h-5 text-green-400" /> : <Copy className="w-5 h-5 text-quantum-cyan" />}
            </button>

            <button
              onClick={downloadCode}
              className="p-2 rounded-lg bg-quantum-purple/20 hover:bg-quantum-purple/30 border border-quantum-purple/30 transition-all"
              title="Download file"
            >
              <Download className="w-5 h-5 text-quantum-purple" />
            </button>

            <button
              onClick={openInNewTab}
              className="p-2 rounded-lg bg-quantum-pink/20 hover:bg-quantum-pink/30 border border-quantum-pink/30 transition-all"
              title="Open in new tab"
            >
              <ExternalLink className="w-5 h-5 text-quantum-pink" />
            </button>

            <button
              onClick={onClose}
              className="p-2 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 transition-all ml-2"
              title="Close"
            >
              <X className="w-5 h-5 text-red-400" />
            </button>
          </div>
        </div>

        {/* Code Content */}
        <div className="overflow-auto max-h-[calc(90vh-8rem)] p-6 bg-quantum-dark/50">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-quantum-cyan"></div>
              <p className="ml-4 text-gray-400">Loading code...</p>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <p className="text-red-400 text-lg mb-2">❌ {error}</p>
                <button
                  onClick={fetchCode}
                  className="px-4 py-2 bg-quantum-cyan/20 hover:bg-quantum-cyan/30 border border-quantum-cyan/30 rounded-lg transition-all"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : (
            <pre className="text-sm text-gray-300 font-mono whitespace-pre-wrap break-words">
              <code className={`language-${language}`}>{code}</code>
            </pre>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-quantum-purple/30 bg-quantum-indigo/10 flex items-center justify-between">
          <div className="text-sm text-gray-400">
            {code.split('\n').length} lines • {(code.length / 1024).toFixed(1)} KB
          </div>
          <div className="text-sm text-gray-500">
            Press <kbd className="px-2 py-1 bg-quantum-dark rounded border border-quantum-purple/30">Esc</kbd> to close
          </div>
        </div>
      </div>
    </div>
  );
}

// Keyboard shortcut to close on Esc
if (typeof window !== 'undefined') {
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      // This will be handled by the parent component
    }
  });
}
