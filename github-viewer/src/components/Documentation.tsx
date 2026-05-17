import { GitBranch, AlertCircle, GitPullRequest, Copy, Check, Bot, Terminal } from 'lucide-react';
import { useState } from 'react';

interface DocumentationProps {
  onClose: () => void;
}

export function Documentation({ onClose }: DocumentationProps) {
  const [copiedSection, setCopiedSection] = useState<string>('');

  const handleCopy = (text: string, section: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSection(section);
    setTimeout(() => setCopiedSection(''), 2000);
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#050714] border-2 border-cyan-500/50 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
        {/* Header */}
        <div className="sticky top-0 bg-[#050714] border-b border-cyan-500/30 p-6 flex items-center justify-between z-10">
          <div>
            <h2 className="text-3xl font-bold text-cyan-400 font-mono mb-2">
              Contributing to Quillon
            </h2>
            <p className="text-gray-400 font-mono text-sm">
              Clone, report bugs, contribute code, or use Claude Code with MCP
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-cyan-400 transition-colors text-2xl font-bold"
          >
            &times;
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-8">
          {/* Clone Repository Section */}
          <section>
            <div className="flex items-center gap-3 mb-4">
              <GitBranch className="text-cyan-400" size={24} />
              <h3 className="text-2xl font-bold text-cyan-400 font-mono">
                Clone Repository
              </h3>
            </div>

            <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4 space-y-3">
              <p className="text-gray-300 font-mono text-sm mb-3">
                Clone the Quillon repository to your local machine:
              </p>

              <div className="relative">
                <pre className="bg-[#0a0e27] border border-cyan-500/20 rounded-lg p-4 text-cyan-300 font-mono text-sm overflow-x-auto">
                  git clone https://code.quillon.xyz/repo.git
                </pre>
                <button
                  onClick={() => handleCopy('git clone https://code.quillon.xyz/repo.git', 'clone')}
                  className="absolute top-2 right-2 p-2 bg-cyan-500/20 hover:bg-cyan-500/30 rounded-lg transition-colors"
                  title="Copy to clipboard"
                >
                  {copiedSection === 'clone' ? (
                    <Check size={16} className="text-green-400" />
                  ) : (
                    <Copy size={16} className="text-cyan-400" />
                  )}
                </button>
              </div>

              <div className="space-y-2 mt-4">
                <p className="text-gray-400 font-mono text-xs">Build the project:</p>
                <div className="relative">
                  <pre className="bg-[#0a0e27] border border-cyan-500/20 rounded-lg p-3 text-cyan-300 font-mono text-sm overflow-x-auto">
                    cd q-narwhalknight && timeout 36000 cargo build --release
                  </pre>
                  <button
                    onClick={() => handleCopy('cd q-narwhalknight && timeout 36000 cargo build --release', 'build')}
                    className="absolute top-2 right-2 p-2 bg-cyan-500/20 hover:bg-cyan-500/30 rounded-lg transition-colors"
                    title="Copy to clipboard"
                  >
                    {copiedSection === 'build' ? (
                      <Check size={16} className="text-green-400" />
                    ) : (
                      <Copy size={16} className="text-cyan-400" />
                    )}
                  </button>
                </div>
                <p className="text-gray-500 font-mono text-[10px]">
                  First build may take 30+ minutes (post-quantum crypto crates). Use a long timeout.
                </p>
              </div>
            </div>
          </section>

          {/* Claude Code MCP Section */}
          <section>
            <div className="flex items-center gap-3 mb-4">
              <Bot className="text-purple-400" size={24} />
              <h3 className="text-2xl font-bold text-purple-400 font-mono">
                AI-Powered Contributing (Claude Code)
              </h3>
            </div>

            <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 space-y-4">
              <p className="text-gray-300 font-mono text-sm">
                Use Claude Code with MCP to browse, search, and contribute using AI:
              </p>

              <div className="space-y-3">
                <div>
                  <p className="text-purple-300 font-mono text-xs font-bold mb-2">1. Install Claude Code</p>
                  <div className="relative">
                    <pre className="bg-[#0a0e27] border border-purple-500/20 rounded-lg p-3 text-purple-300 font-mono text-sm">
                      npm install -g @anthropic-ai/claude-code
                    </pre>
                    <button
                      onClick={() => handleCopy('npm install -g @anthropic-ai/claude-code', 'install-claude')}
                      className="absolute top-2 right-2 p-1.5 bg-purple-500/20 hover:bg-purple-500/30 rounded-lg transition-colors"
                    >
                      {copiedSection === 'install-claude' ? <Check size={14} className="text-green-400" /> : <Copy size={14} className="text-purple-400" />}
                    </button>
                  </div>
                </div>

                <div>
                  <p className="text-purple-300 font-mono text-xs font-bold mb-2">2. Add MCP server config</p>
                  <div className="relative">
                    <pre className="bg-[#0a0e27] border border-purple-500/20 rounded-lg p-3 text-purple-300 font-mono text-xs overflow-x-auto">
{`// ~/.claude/settings.json
{
  "mcpServers": {
    "quillon-code": {
      "type": "sse",
      "url": "https://code.quillon.xyz/mcp/sse"
    }
  }
}`}
                    </pre>
                    <button
                      onClick={() => handleCopy('{\n  "mcpServers": {\n    "quillon-code": {\n      "type": "sse",\n      "url": "https://code.quillon.xyz/mcp/sse"\n    }\n  }\n}', 'mcp-config')}
                      className="absolute top-2 right-2 p-1.5 bg-purple-500/20 hover:bg-purple-500/30 rounded-lg transition-colors"
                    >
                      {copiedSection === 'mcp-config' ? <Check size={14} className="text-green-400" /> : <Copy size={14} className="text-purple-400" />}
                    </button>
                  </div>
                </div>

                <div>
                  <p className="text-purple-300 font-mono text-xs font-bold mb-2">3. Available MCP Tools</p>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { name: 'read_file', desc: 'Read any git-tracked file', color: 'cyan' },
                      { name: 'search_code', desc: 'Search patterns across codebase', color: 'cyan' },
                      { name: 'list_branches', desc: 'See all branches', color: 'cyan' },
                      { name: 'view_diff', desc: 'Preview branch diffs', color: 'cyan' },
                      { name: 'list_files', desc: 'List tracked files', color: 'cyan' },
                      { name: 'submit_contribution', desc: 'Submit a code proposal', color: 'green' },
                    ].map(tool => (
                      <div key={tool.name} className={`bg-${tool.color}-500/10 border border-${tool.color}-500/20 rounded-lg p-2`}>
                        <code className={`text-${tool.color}-400 text-[11px] font-bold`}>{tool.name}</code>
                        <div className="text-gray-500 text-[10px] mt-0.5">{tool.desc}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
                <p className="text-amber-300 font-mono text-xs">
                  <strong>All tools are READ-ONLY.</strong> Contributions are saved as proposals for maintainer review — no code is modified on the server.
                </p>
              </div>
            </div>
          </section>

          {/* Submit Issue Section */}
          <section>
            <div className="flex items-center gap-3 mb-4">
              <AlertCircle className="text-rose-400" size={24} />
              <h3 className="text-2xl font-bold text-rose-400 font-mono">
                Report a Bug
              </h3>
            </div>

            <div className="bg-rose-500/10 border border-rose-500/30 rounded-lg p-4 space-y-3">
              <p className="text-gray-300 font-mono text-sm mb-3">
                Found a bug? Submit via the Bounty Campaign and earn rewards:
              </p>

              <ol className="space-y-2 text-gray-300 font-mono text-sm list-decimal list-inside">
                <li>Visit <a href="https://quillon.xyz" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 underline">quillon.xyz</a> and connect your wallet</li>
                <li>Navigate to the Bounty Campaign section</li>
                <li>Click "Report Bug" and select severity level</li>
                <li>Describe the bug with steps to reproduce</li>
                <li>Include an issue URL from code.quillon.xyz if applicable</li>
              </ol>

              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 mt-4">
                <p className="text-amber-300 font-mono text-xs mb-2"><strong>Bounty Rewards:</strong></p>
                <div className="grid grid-cols-2 gap-2 text-gray-400 font-mono text-xs">
                  <div>🔴 Critical: 50+ points</div>
                  <div>🟠 High: 20+ points</div>
                  <div>🟡 Medium: 10+ points</div>
                  <div>🟢 Low: 5+ points</div>
                </div>
              </div>
            </div>
          </section>

          {/* Contribute Code Section */}
          <section>
            <div className="flex items-center gap-3 mb-4">
              <GitPullRequest className="text-green-400" size={24} />
              <h3 className="text-2xl font-bold text-green-400 font-mono">
                Contribute Code
              </h3>
            </div>

            <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 space-y-3">
              <p className="text-gray-300 font-mono text-sm mb-3">
                Two ways to contribute code to Q-NarwhalKnight:
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Manual Path */}
                <div className="bg-[#0a0e27] border border-green-500/20 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Terminal size={18} className="text-green-400" />
                    <h4 className="text-green-400 font-mono text-sm font-bold">Manual (git patch)</h4>
                  </div>
                  <ol className="space-y-2 text-gray-400 font-mono text-xs list-decimal list-inside">
                    <li>Clone the repository</li>
                    <li>Make your changes locally</li>
                    <li>Run tests: <code className="text-green-300">cargo test</code></li>
                    <li>Generate diff: <code className="text-green-300">git diff &gt; fix.patch</code></li>
                    <li>Submit patch via bounty dApp</li>
                  </ol>
                </div>

                {/* AI Path */}
                <div className="bg-[#0a0e27] border border-purple-500/20 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Bot size={18} className="text-purple-400" />
                    <h4 className="text-purple-400 font-mono text-sm font-bold">Claude Code (MCP)</h4>
                  </div>
                  <ol className="space-y-2 text-gray-400 font-mono text-xs list-decimal list-inside">
                    <li>Configure MCP server (see above)</li>
                    <li>Browse code with <code className="text-purple-300">read_file</code></li>
                    <li>Find issues with <code className="text-purple-300">search_code</code></li>
                    <li>Generate a fix diff</li>
                    <li>Submit via <code className="text-purple-300">submit_contribution</code></li>
                  </ol>
                </div>
              </div>

              <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3 mt-4">
                <p className="text-cyan-300 font-mono text-xs">
                  All contributions are reviewed by maintainers. Focus areas: consensus safety, balance integrity, P2P security, performance. Accepted contributions earn bounty points.
                </p>
              </div>
            </div>
          </section>

          {/* Resources */}
          <section>
            <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4">
              <h4 className="text-cyan-400 font-mono font-bold mb-3">Resources</h4>
              <ul className="space-y-2 text-gray-300 font-mono text-sm">
                <li>💻 <a href="https://code.quillon.xyz" className="text-cyan-400 hover:text-cyan-300 underline">code.quillon.xyz</a> — Source code browser</li>
                <li>🌐 <a href="https://quillon.xyz" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 underline">quillon.xyz</a> — Network dashboard & bounty campaign</li>
                <li>📊 <a href="https://technical-deepdive.quillon.xyz" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 underline">technical-deepdive.quillon.xyz</a> — Architecture presentation</li>
              </ul>
            </div>
          </section>
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-[#050714] border-t border-cyan-500/30 p-4 text-center">
          <p className="text-gray-500 font-mono text-xs">
            Q-NarwhalKnight v8.0.0-mainnet &middot; 80+ Rust crates &middot; Post-quantum consensus
          </p>
        </div>
      </div>
    </div>
  );
}
