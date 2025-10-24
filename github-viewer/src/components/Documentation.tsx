import { GitBranch, AlertCircle, GitPullRequest, Copy, Check } from 'lucide-react';
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
        <div className="sticky top-0 bg-[#050714] border-b border-cyan-500/30 p-6 flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold text-cyan-400 font-mono mb-2">
              Contributing to Quillon
            </h2>
            <p className="text-gray-400 font-mono text-sm">
              Clone, submit issues, and create pull requests
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-cyan-400 transition-colors text-2xl font-bold"
          >
            ×
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
                Clone the Quillon repository to your local machine (read-only):
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

              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3 mt-3">
                <p className="text-yellow-300 font-mono text-xs">
                  <strong>Note:</strong> This is a read-only clone. You cannot push directly to this repository.
                  To contribute, fork on GitHub and submit a pull request.
                </p>
              </div>

              <div className="space-y-2 mt-4">
                <p className="text-gray-400 font-mono text-xs">After cloning, navigate into the directory:</p>
                <div className="relative">
                  <pre className="bg-[#0a0e27] border border-cyan-500/20 rounded-lg p-3 text-cyan-300 font-mono text-sm">
                    cd q-narwhalknight
                  </pre>
                </div>

                <p className="text-gray-400 font-mono text-xs mt-3">Build the project:</p>
                <div className="relative">
                  <pre className="bg-[#0a0e27] border border-cyan-500/20 rounded-lg p-3 text-cyan-300 font-mono text-sm overflow-x-auto">
                    timeout 36000 cargo build --release --workspace
                  </pre>
                  <button
                    onClick={() => handleCopy('timeout 36000 cargo build --release --workspace', 'build')}
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
              </div>
            </div>
          </section>

          {/* Submit Issue Section */}
          <section>
            <div className="flex items-center gap-3 mb-4">
              <AlertCircle className="text-magenta-400" size={24} />
              <h3 className="text-2xl font-bold text-magenta-400 font-mono">
                Submit an Issue
              </h3>
            </div>

            <div className="bg-magenta-500/10 border border-magenta-500/30 rounded-lg p-4 space-y-3">
              <p className="text-gray-300 font-mono text-sm mb-3">
                Found a bug or have a feature request? Submit via our Bounty Campaign:
              </p>

              <ol className="space-y-3 text-gray-300 font-mono text-sm list-decimal list-inside">
                <li>
                  Visit{' '}
                  <a
                    href="https://bounty.quillon.xyz"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-cyan-400 hover:text-cyan-300 underline"
                  >
                    bounty.quillon.xyz
                  </a>
                </li>
                <li>Connect your wallet to authenticate</li>
                <li>Navigate to "Report Bug" section</li>
                <li>Select severity level (Critical, High, Medium, Low)</li>
                <li>Provide detailed description with steps to reproduce</li>
                <li>Submit and earn bounty rewards for verified bugs!</li>
              </ol>

              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 mt-4">
                <p className="text-amber-300 font-mono text-xs mb-2">
                  <strong>Bounty Rewards:</strong>
                </p>
                <ul className="space-y-1 text-gray-400 font-mono text-xs list-disc list-inside ml-2">
                  <li>🔴 Critical bugs: 50+ points</li>
                  <li>🟠 High severity: 20+ points</li>
                  <li>🟡 Medium severity: 10+ points</li>
                  <li>🟢 Low severity: 5+ points</li>
                  <li>Points convert to QNK tokens in mainnet</li>
                </ul>
              </div>

              <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3 mt-4">
                <p className="text-cyan-300 font-mono text-xs mb-2">
                  <strong>Good bug reports include:</strong>
                </p>
                <ul className="space-y-1 text-gray-400 font-mono text-xs list-disc list-inside ml-2">
                  <li>Clear, descriptive title</li>
                  <li>Steps to reproduce</li>
                  <li>Expected vs actual behavior</li>
                  <li>System information (OS, Rust version, etc.)</li>
                  <li>Relevant logs or error messages</li>
                  <li>Screenshots or videos if applicable</li>
                </ul>
              </div>
            </div>
          </section>

          {/* Submit PR Section */}
          <section>
            <div className="flex items-center gap-3 mb-4">
              <GitPullRequest className="text-green-400" size={24} />
              <h3 className="text-2xl font-bold text-green-400 font-mono">
                Contribute Code & Improvements
              </h3>
            </div>

            <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 space-y-3">
              <p className="text-gray-300 font-mono text-sm mb-3">
                Want to contribute code or improvements? Earn bounty rewards!
              </p>

              <ol className="space-y-3 text-gray-300 font-mono text-sm">
                <li>
                  <strong className="text-green-400">1. Clone the repository (read-only)</strong>
                  <div className="relative mt-2 ml-4">
                    <pre className="bg-[#0a0e27] border border-green-500/20 rounded-lg p-3 text-green-300 font-mono text-xs">
                      git clone https://code.quillon.xyz/repo.git
                    </pre>
                  </div>
                </li>

                <li>
                  <strong className="text-green-400">2. Make your changes</strong>
                  <div className="ml-4 mt-1 text-gray-400 text-xs">
                    Edit code, add features, fix bugs, optimize performance
                  </div>
                </li>

                <li>
                  <strong className="text-green-400">3. Test your changes thoroughly</strong>
                  <div className="relative mt-2 ml-4">
                    <pre className="bg-[#0a0e27] border border-green-500/20 rounded-lg p-3 text-green-300 font-mono text-xs overflow-x-auto">
                      cargo test --workspace{'\n'}
                      cargo clippy -- -D warnings{'\n'}
                      cargo fmt --check
                    </pre>
                    <button
                      onClick={() => handleCopy('cargo test --workspace\ncargo clippy -- -D warnings\ncargo fmt --check', 'test')}
                      className="absolute top-2 right-2 p-1.5 bg-green-500/20 hover:bg-green-500/30 rounded-lg transition-colors"
                      title="Copy to clipboard"
                    >
                      {copiedSection === 'test' ? (
                        <Check size={14} className="text-green-400" />
                      ) : (
                        <Copy size={14} className="text-green-400" />
                      )}
                    </button>
                  </div>
                </li>

                <li>
                  <strong className="text-green-400">4. Create a patch file</strong>
                  <div className="relative mt-2 ml-4">
                    <pre className="bg-[#0a0e27] border border-green-500/20 rounded-lg p-3 text-green-300 font-mono text-xs overflow-x-auto">
                      git diff {'>'} my-contribution.patch
                    </pre>
                  </div>
                </li>

                <li>
                  <strong className="text-green-400">5. Submit via Bounty Campaign</strong>
                  <div className="ml-4 mt-1 text-gray-400 text-xs space-y-1">
                    <div>Go to <a href="https://bounty.quillon.xyz" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 underline">bounty.quillon.xyz</a></div>
                    <div>Connect your wallet</div>
                    <div>Navigate to "Submit Contribution"</div>
                    <div>Upload your patch file and describe your changes</div>
                    <div>Submit and earn bounty rewards for accepted contributions!</div>
                  </div>
                </li>
              </ol>

              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 mt-4">
                <p className="text-amber-300 font-mono text-xs mb-2">
                  <strong>Contribution Rewards:</strong>
                </p>
                <ul className="space-y-1 text-gray-400 font-mono text-xs list-disc list-inside ml-2">
                  <li>🔥 New features: 30-100+ points</li>
                  <li>⚡ Performance improvements: 20-50 points</li>
                  <li>🔧 Bug fixes: 10-30 points</li>
                  <li>📚 Documentation: 5-15 points</li>
                  <li>✅ Tests & quality improvements: 5-20 points</li>
                </ul>
              </div>

              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3 mt-4">
                <p className="text-yellow-300 font-mono text-xs mb-2">
                  <strong>Contribution Guidelines:</strong>
                </p>
                <ul className="space-y-1 text-gray-400 font-mono text-xs list-disc list-inside ml-2">
                  <li>Keep contributions focused on a single feature or fix</li>
                  <li>Include tests for new functionality</li>
                  <li>Update documentation if needed</li>
                  <li>Follow the existing code style and conventions</li>
                  <li>Ensure all tests pass before submitting</li>
                  <li>Provide clear description of what and why</li>
                </ul>
              </div>
            </div>
          </section>

          {/* Additional Resources */}
          <section>
            <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4">
              <h4 className="text-cyan-400 font-mono font-bold mb-3">Additional Resources</h4>
              <ul className="space-y-2 text-gray-300 font-mono text-sm">
                <li>
                  💻{' '}
                  <a
                    href="https://code.quillon.xyz"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-cyan-400 hover:text-cyan-300 underline"
                  >
                    Source Code Browser
                  </a>{' '}
                  - Explore the complete codebase
                </li>
                <li>
                  🌐{' '}
                  <a
                    href="https://quillon.xyz"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-cyan-400 hover:text-cyan-300 underline"
                  >
                    Quillon Network
                  </a>{' '}
                  - Official website and testnet explorer
                </li>
                <li>
                  🎁{' '}
                  <a
                    href="https://bounty.quillon.xyz"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-cyan-400 hover:text-cyan-300 underline"
                  >
                    Bounty Campaign
                  </a>{' '}
                  - Submit bugs, earn rewards
                </li>
                <li>
                  📊{' '}
                  <a
                    href="https://technical-deepdive.quillon.xyz"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-cyan-400 hover:text-cyan-300 underline"
                  >
                    Technical Deep Dive
                  </a>{' '}
                  - Architecture presentation
                </li>
              </ul>
            </div>
          </section>
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-[#050714] border-t border-cyan-500/30 p-4 text-center">
          <p className="text-gray-400 font-mono text-sm">
            Questions or need help? Visit{' '}
            <a
              href="https://bounty.quillon.xyz"
              target="_blank"
              rel="noopener noreferrer"
              className="text-cyan-400 hover:text-cyan-300 underline"
            >
              bounty.quillon.xyz
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
