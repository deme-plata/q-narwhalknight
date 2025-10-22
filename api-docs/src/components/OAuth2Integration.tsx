import React, { useState } from 'react';
import { Shield, Lock, Key, Globe, Code, CheckCircle, AlertTriangle, Copy, Check, Eye } from 'lucide-react';
import CodeModal from './CodeModal';

export default function OAuth2Integration() {
  const [copiedSection, setCopiedSection] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedExample, setSelectedExample] = useState<{ title: string; url: string; language: string } | null>(null);

  const copyToClipboard = (text: string, section: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSection(section);
    setTimeout(() => setCopiedSection(null), 2000);
  };

  const openCodeModal = (title: string, url: string, language: string) => {
    setSelectedExample({ title, url, language });
    setModalOpen(true);
  };

  const closeModal = () => {
    setModalOpen(false);
    setSelectedExample(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-quantum-dark via-quantum-indigo/20 to-quantum-dark text-white p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Shield className="w-12 h-12 text-quantum-cyan" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent">
              OAuth2 Integration Guide
            </h1>
          </div>
          <p className="text-xl text-gray-300">
            Connect third-party applications to Quillon Wallet with quantum-secure authentication
          </p>
        </div>

        {/* Overview */}
        <section className="mb-12 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-6 flex items-center gap-3">
            <Globe className="w-8 h-8 text-quantum-cyan" />
            What is OAuth2?
          </h2>
          <p className="text-gray-300 mb-4 text-lg">
            OAuth2 allows third-party websites and applications to authenticate users via Quillon Wallet
            and access blockchain functionality with user consent - all without exposing private keys.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="bg-quantum-dark/50 p-6 rounded-xl border border-quantum-cyan/30">
              <Shield className="w-10 h-10 text-quantum-cyan mb-3" />
              <h3 className="font-bold mb-2">Secure Authentication</h3>
              <p className="text-sm text-gray-400">
                Industry-standard OAuth2 2.0 with PKCE protection against authorization code interception
              </p>
            </div>
            <div className="bg-quantum-dark/50 p-6 rounded-xl border border-quantum-purple/30">
              <Lock className="w-10 h-10 text-quantum-purple mb-3" />
              <h3 className="font-bold mb-2">Post-Quantum Ready</h3>
              <p className="text-sm text-gray-400">
                Optional Kyber1024 encryption for quantum-resistant token exchange
              </p>
            </div>
            <div className="bg-quantum-dark/50 p-6 rounded-xl border border-quantum-pink/30">
              <Key className="w-10 h-10 text-quantum-pink mb-3" />
              <h3 className="font-bold mb-2">Granular Permissions</h3>
              <p className="text-sm text-gray-400">
                Scope-based access control - users grant only the permissions your app needs
              </p>
            </div>
          </div>
        </section>

        {/* Quick Start */}
        <section className="mb-12 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-6">⚡ Quick Start</h2>

          <div className="space-y-6">
            {/* Step 1 */}
            <div className="border-l-4 border-quantum-cyan pl-6">
              <h3 className="text-2xl font-bold mb-3 flex items-center gap-2">
                <span className="bg-quantum-cyan text-quantum-dark rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold">
                  1
                </span>
                Register Your Application
              </h3>
              <p className="text-gray-300 mb-4">
                First, register your application with Quillon to obtain OAuth2 credentials:
              </p>
              <CodeBlock
                language="bash"
                code={`curl -X POST https://api.quillon.xyz/api/v1/oauth2/register \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "My Awesome App",
    "redirect_uris": ["https://myapp.com/callback"],
    "website": "https://myapp.com",
    "scopes": ["read:balance", "send:transaction"]
  }'

# Response:
{
  "success": true,
  "data": {
    "client_id": "abc123xyz...",
    "client_secret": "secret_def456...",
    "redirect_uris": ["https://myapp.com/callback"]
  }
}`}
                onCopy={(code) => copyToClipboard(code, 'register')}
                copied={copiedSection === 'register'}
              />
              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 mt-4 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                <div className="text-sm">
                  <strong className="text-yellow-400">Security Warning:</strong> Store your <code className="bg-black/30 px-2 py-1 rounded">client_secret</code> securely
                  on your server. Never expose it in client-side code or public repositories.
                </div>
              </div>
            </div>

            {/* Step 2 */}
            <div className="border-l-4 border-quantum-purple pl-6">
              <h3 className="text-2xl font-bold mb-3 flex items-center gap-2">
                <span className="bg-quantum-purple text-white rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold">
                  2
                </span>
                Install the SDK
              </h3>
              <p className="text-gray-300 mb-4">
                Install the Quillon OAuth2 SDK via NPM:
              </p>
              <CodeBlock
                language="bash"
                code="npm install @quillon/oauth2-sdk"
                onCopy={(code) => copyToClipboard(code, 'install')}
                copied={copiedSection === 'install'}
              />
            </div>

            {/* Step 3 */}
            <div className="border-l-4 border-quantum-pink pl-6">
              <h3 className="text-2xl font-bold mb-3 flex items-center gap-2">
                <span className="bg-quantum-pink text-white rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold">
                  3
                </span>
                Implement Authentication
              </h3>
              <p className="text-gray-300 mb-4">
                Initialize the SDK and start the OAuth2 flow:
              </p>
              <CodeBlock
                language="javascript"
                code={`import QullionOAuth2Client from '@quillon/oauth2-sdk';

const client = new QullionOAuth2Client({
  clientId: 'your-client-id',
  clientSecret: 'your-client-secret', // Server-side only!
  redirectUri: 'https://myapp.com/callback',
  scopes: ['read:balance', 'send:transaction']
});

// Start OAuth2 flow (redirects to Quillon Wallet)
await client.authorize();

// Handle callback (in your /callback page)
const tokenResponse = await client.handleCallback();

// Get user info
const userInfo = await client.getUserInfo();
console.log('Wallet:', userInfo.wallet_address);

// Get balance
const balance = await client.getBalance('QUG');
console.log('Balance:', balance / 100000000, 'QUG');`}
                onCopy={(code) => copyToClipboard(code, 'auth')}
                copied={copiedSection === 'auth'}
              />
            </div>
          </div>
        </section>

        {/* OAuth2 Flow */}
        <section className="mb-12 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-6">🔄 OAuth2 Authorization Flow</h2>

          <div className="bg-quantum-dark/50 p-6 rounded-xl font-mono text-sm space-y-4">
            <div className="flex items-start gap-4">
              <span className="bg-quantum-cyan text-quantum-dark px-3 py-1 rounded font-bold flex-shrink-0">STEP 1</span>
              <div>
                <div className="text-quantum-cyan font-bold">User clicks "Connect Wallet" on your app</div>
                <div className="text-gray-400 mt-1">Your app redirects to Quillon Wallet authorization page</div>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <span className="bg-quantum-purple text-white px-3 py-1 rounded font-bold flex-shrink-0">STEP 2</span>
              <div>
                <div className="text-quantum-purple font-bold">User reviews permissions and approves</div>
                <div className="text-gray-400 mt-1">Consent screen shows which permissions your app requests</div>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <span className="bg-quantum-pink text-white px-3 py-1 rounded font-bold flex-shrink-0">STEP 3</span>
              <div>
                <div className="text-quantum-pink font-bold">Quillon redirects back with authorization code</div>
                <div className="text-gray-400 mt-1">Redirect: https://myapp.com/callback?code=AUTH_CODE&state=CSRF_TOKEN</div>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <span className="bg-green-500 text-white px-3 py-1 rounded font-bold flex-shrink-0">STEP 4</span>
              <div>
                <div className="text-green-400 font-bold">Your app exchanges code for access token</div>
                <div className="text-gray-400 mt-1">POST /api/v1/oauth2/token with code + PKCE verifier</div>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <span className="bg-blue-500 text-white px-3 py-1 rounded font-bold flex-shrink-0">STEP 5</span>
              <div>
                <div className="text-blue-400 font-bold">Use access token to call Quillon APIs</div>
                <div className="text-gray-400 mt-1">Authorization: Bearer ACCESS_TOKEN</div>
              </div>
            </div>
          </div>
        </section>

        {/* Available Scopes */}
        <section className="mb-12 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-6">🔑 Available Scopes</h2>
          <p className="text-gray-300 mb-6">
            Request only the permissions your application needs:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ScopeCard
              scope="read:balance"
              name="Read Balance"
              description="View user's QUG, QUGUSD, and custom token balances"
              icon={<Shield className="w-6 h-6 text-quantum-cyan" />}
            />
            <ScopeCard
              scope="send:transaction"
              name="Send Transactions"
              description="Send QUG and tokens on behalf of the user"
              icon={<Key className="w-6 h-6 text-quantum-purple" />}
              warning="Requires user approval for each transaction"
            />
            <ScopeCard
              scope="read:transactions"
              name="Read Transaction History"
              description="View user's past transactions and activity"
              icon={<Globe className="w-6 h-6 text-quantum-pink" />}
            />
            <ScopeCard
              scope="manage:tokens"
              name="Manage Tokens"
              description="Create and manage custom tokens"
              icon={<Lock className="w-6 h-6 text-quantum-green" />}
              warning="High privilege - request only if necessary"
            />
          </div>
        </section>

        {/* SDK Reference */}
        <section className="mb-12 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-6">📚 SDK Reference</h2>

          <div className="space-y-6">
            <APIMethod
              method="authorize()"
              description="Starts the OAuth2 authorization flow. Redirects user to Quillon Wallet."
              example="await client.authorize();"
              returns="Promise<void>"
            />

            <APIMethod
              method="handleCallback()"
              description="Handles OAuth2 callback after user consent. Call this in your redirect URI page."
              example="const tokenResponse = await client.handleCallback();"
              returns="Promise<TokenResponse>"
            />

            <APIMethod
              method="getUserInfo()"
              description="Gets authenticated user information (wallet address, scopes)."
              example="const userInfo = await client.getUserInfo();"
              returns="Promise<{ wallet_address: string, scopes: string[] }>"
            />

            <APIMethod
              method="getBalance(token)"
              description="Gets user's token balance in base units (8 decimals)."
              example="const balance = await client.getBalance('QUG');"
              params={[{ name: 'token', type: 'string', description: 'Token symbol (e.g., "QUG", "QUGUSD")' }]}
              returns="Promise<number>"
            />

            <APIMethod
              method="sendTransaction(params)"
              description="Sends a transaction on behalf of the user."
              example={`await client.sendTransaction({
  to: 'wallet-address',
  amount: 100000000, // 1 QUG
  token: 'QUG'
});`}
              params={[
                { name: 'to', type: 'string', description: 'Recipient wallet address' },
                { name: 'amount', type: 'number', description: 'Amount in base units' },
                { name: 'token', type: 'string', description: 'Token symbol' }
              ]}
              returns="Promise<Transaction>"
            />

            <APIMethod
              method="getTransactionHistory(limit)"
              description="Gets user's transaction history."
              example="const transactions = await client.getTransactionHistory(20);"
              params={[{ name: 'limit', type: 'number', description: 'Maximum transactions to return' }]}
              returns="Promise<Transaction[]>"
            />

            <APIMethod
              method="isAuthenticated()"
              description="Checks if user is currently authenticated."
              example="if (client.isAuthenticated()) { ... }"
              returns="boolean"
            />

            <APIMethod
              method="revoke()"
              description="Revokes access token and logs out user."
              example="await client.revoke();"
              returns="Promise<void>"
            />
          </div>
        </section>

        {/* Security Best Practices */}
        <section className="mb-12 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-6 flex items-center gap-3">
            <Shield className="w-8 h-8 text-quantum-cyan" />
            Security Best Practices
          </h2>

          <div className="space-y-4">
            <SecurityItem
              icon={<CheckCircle className="w-6 h-6 text-green-500" />}
              title="Always use HTTPS in production"
              description="OAuth2 redirect URIs must use HTTPS to prevent token interception."
            />
            <SecurityItem
              icon={<CheckCircle className="w-6 h-6 text-green-500" />}
              title="Never expose client_secret in frontend code"
              description="Use server-side token exchange. The SDK handles PKCE for client-side security."
            />
            <SecurityItem
              icon={<CheckCircle className="w-6 h-6 text-green-500" />}
              title="Request minimal scopes"
              description="Only request the permissions your application actually needs."
            />
            <SecurityItem
              icon={<CheckCircle className="w-6 h-6 text-green-500" />}
              title="Validate state parameter"
              description="The SDK automatically validates state to prevent CSRF attacks."
            />
            <SecurityItem
              icon={<CheckCircle className="w-6 h-6 text-green-500" />}
              title="Handle token expiration"
              description="Use client.getAccessToken() which auto-refreshes expired tokens."
            />
          </div>
        </section>

        {/* Examples */}
        <section className="mb-12 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-6">💡 Complete Examples</h2>

          <div className="space-y-6">
            <ExampleCard
              title="Vanilla JavaScript"
              description="Simple HTML + JavaScript integration"
              link="/sdk/examples/basic-integration.html"
              onViewCode={() => openCodeModal('Vanilla JavaScript Example', '/sdk/examples/basic-integration.html', 'html')}
            />
            <ExampleCard
              title="React + TypeScript"
              description="Full React application with hooks and context"
              link="/sdk/examples/react-integration.tsx"
              onViewCode={() => openCodeModal('React + TypeScript Example', '/sdk/examples/react-integration.tsx', 'typescript')}
            />
            <ExampleCard
              title="Node.js Server"
              description="Server-side OAuth2 implementation with Express"
              link="/sdk/examples/nodejs-server.js"
              onViewCode={() => openCodeModal('Node.js Server Example', '/sdk/examples/nodejs-server.js', 'javascript')}
            />
            <ExampleCard
              title="Python + FastAPI"
              description="Python server-side OAuth2 implementation"
              link="/sdk/examples/python-oauth2-example.py"
              onViewCode={() => openCodeModal('Python + FastAPI Example', '/sdk/examples/python-oauth2-example.py', 'python')}
            />
            <ExampleCard
              title="Advanced HTML/JS"
              description="Advanced OAuth2 flow with complete UI"
              link="/sdk/examples/advanced-oauth2-example.html"
              onViewCode={() => openCodeModal('Advanced HTML/JS Example', '/sdk/examples/advanced-oauth2-example.html', 'html')}
            />
          </div>
        </section>

        {/* Code Modal */}
        {modalOpen && selectedExample && (
          <CodeModal
            isOpen={modalOpen}
            onClose={closeModal}
            title={selectedExample.title}
            fileUrl={selectedExample.url}
            language={selectedExample.language}
          />
        )}

        {/* Footer */}
        <div className="text-center text-gray-500 pt-12 border-t border-quantum-purple/30">
          <p className="mb-2">
            Need help? Join our{' '}
            <a href="https://discord.gg/jEhaYtAhfx" className="text-quantum-cyan hover:underline">
              Discord
            </a>{' '}
            or{' '}
            <a href="mailto:bitknight.dipper688@passmail.net" className="text-quantum-cyan hover:underline">
              contact support
            </a>
          </p>
          <p className="text-sm">
            Protected by Post-Quantum Cryptography (Kyber1024 + Dilithium5)
          </p>
        </div>
      </div>
    </div>
  );
}

// Helper Components

interface CodeBlockProps {
  language: string;
  code: string;
  onCopy: (code: string) => void;
  copied: boolean;
}

function CodeBlock({ code, onCopy, copied }: CodeBlockProps) {
  return (
    <div className="relative">
      <pre className="bg-black/50 border border-quantum-purple/30 rounded-lg p-4 overflow-x-auto">
        <code className="text-sm text-gray-300">{code}</code>
      </pre>
      <button
        onClick={() => onCopy(code)}
        className="absolute top-2 right-2 p-2 bg-quantum-purple/20 hover:bg-quantum-purple/30 border border-quantum-purple/30 rounded-lg transition-colors"
      >
        {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-gray-400" />}
      </button>
    </div>
  );
}

interface ScopeCardProps {
  scope: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  warning?: string;
}

function ScopeCard({ scope, name, description, icon, warning }: ScopeCardProps) {
  return (
    <div className="bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl p-6">
      <div className="flex items-start gap-4">
        <div className="bg-quantum-purple/20 rounded-lg p-3">{icon}</div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="font-bold text-lg">{name}</h3>
            <code className="text-xs bg-black/30 px-2 py-1 rounded text-quantum-cyan">{scope}</code>
          </div>
          <p className="text-sm text-gray-400 mb-2">{description}</p>
          {warning && (
            <div className="flex items-start gap-2 text-xs text-yellow-400 bg-yellow-500/10 border border-yellow-500/30 rounded p-2 mt-2">
              <AlertTriangle className="w-3 h-3 flex-shrink-0 mt-0.5" />
              <span>{warning}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface APIMethodProps {
  method: string;
  description: string;
  example: string;
  params?: Array<{ name: string; type: string; description: string }>;
  returns: string;
}

function APIMethod({ method, description, example, params, returns }: APIMethodProps) {
  return (
    <div className="bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl p-6">
      <h3 className="text-xl font-bold text-quantum-cyan mb-2">{method}</h3>
      <p className="text-gray-300 mb-4">{description}</p>

      {params && params.length > 0 && (
        <div className="mb-4">
          <h4 className="font-bold text-sm text-gray-400 mb-2">Parameters:</h4>
          <div className="space-y-2">
            {params.map((param) => (
              <div key={param.name} className="text-sm">
                <code className="bg-black/30 px-2 py-1 rounded text-quantum-purple">{param.name}</code>
                <span className="text-gray-500 mx-2">:</span>
                <code className="text-quantum-cyan">{param.type}</code>
                <span className="text-gray-400 ml-2">- {param.description}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mb-4">
        <h4 className="font-bold text-sm text-gray-400 mb-2">Returns:</h4>
        <code className="text-quantum-cyan">{returns}</code>
      </div>

      <div>
        <h4 className="font-bold text-sm text-gray-400 mb-2">Example:</h4>
        <pre className="bg-black/50 border border-quantum-purple/30 rounded p-3 overflow-x-auto">
          <code className="text-sm text-gray-300">{example}</code>
        </pre>
      </div>
    </div>
  );
}

interface SecurityItemProps {
  icon: React.ReactNode;
  title: string;
  description: string;
}

function SecurityItem({ icon, title, description }: SecurityItemProps) {
  return (
    <div className="flex items-start gap-4 bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl p-4">
      <div className="flex-shrink-0">{icon}</div>
      <div>
        <h3 className="font-bold mb-1">{title}</h3>
        <p className="text-sm text-gray-400">{description}</p>
      </div>
    </div>
  );
}

interface ExampleCardProps {
  title: string;
  description: string;
  link: string;
  onViewCode?: () => void;
}

function ExampleCard({ title, description, link, onViewCode }: ExampleCardProps) {
  return (
    <div className="bg-quantum-dark/50 border border-quantum-purple/30 rounded-xl p-6 hover:border-quantum-cyan/50 transition-colors flex items-center justify-between">
      <div className="flex-1">
        <h3 className="font-bold text-lg mb-2 text-quantum-cyan">{title}</h3>
        <p className="text-gray-400">{description}</p>
      </div>
      <div className="flex gap-3 ml-4">
        {onViewCode && (
          <button
            onClick={onViewCode}
            className="px-4 py-2 bg-quantum-purple/20 hover:bg-quantum-purple/30 border border-quantum-purple/50 rounded-lg transition-all flex items-center gap-2"
          >
            <Eye className="w-4 h-4" />
            View Code
          </button>
        )}
        <a
          href={link}
          target="_blank"
          rel="noopener noreferrer"
          className="px-4 py-2 bg-quantum-cyan/20 hover:bg-quantum-cyan/30 border border-quantum-cyan/50 rounded-lg transition-all flex items-center gap-2"
        >
          <Code className="w-4 h-4" />
          Open
        </a>
      </div>
    </div>
  );
}
