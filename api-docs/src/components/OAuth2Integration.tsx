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
                Register your application with Quillon to obtain OAuth2 credentials. The server auto-generates a <code className="bg-black/30 px-1 rounded">client_id</code> and <code className="bg-black/30 px-1 rounded">client_secret</code> if you don't provide them:
              </p>
              <CodeBlock
                language="bash"
                code={`curl -X POST https://quillon.xyz/api/v1/oauth2/register \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "My Awesome App",
    "redirect_uris": ["https://myapp.com/callback"],
    "website": "https://myapp.com"
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
                  on your server. For browser-only (public) clients, you can omit the secret and rely on PKCE alone.
                </div>
              </div>
            </div>

            {/* Step 2 */}
            <div className="border-l-4 border-quantum-purple pl-6">
              <h3 className="text-2xl font-bold mb-3 flex items-center gap-2">
                <span className="bg-quantum-purple text-white rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold">
                  2
                </span>
                Redirect to Authorize (with PKCE)
              </h3>
              <p className="text-gray-300 mb-4">
                Generate a PKCE code verifier + challenge, then redirect the user to the Quillon consent page:
              </p>
              <CodeBlock
                language="javascript"
                code={`// Generate PKCE code verifier (random 128-char hex string)
const codeVerifier = Array.from(crypto.getRandomValues(new Uint8Array(64)),
  b => b.toString(16).padStart(2, '0')).join('');

// SHA-256 hash → base64url encode for code_challenge
const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(codeVerifier));
const codeChallenge = btoa(String.fromCharCode(...new Uint8Array(hash)))
  .replace(/\\+/g, '-').replace(/\\//g, '_').replace(/=+$/, '');

// Save verifier for step 3
sessionStorage.setItem('pkce_verifier', codeVerifier);

// Redirect user to Quillon authorization
const params = new URLSearchParams({
  response_type: 'code',
  client_id: 'YOUR_CLIENT_ID',
  redirect_uri: 'https://myapp.com/callback',
  scope: 'read:balance read:profile',
  state: crypto.randomUUID(),
  code_challenge: codeChallenge,
  code_challenge_method: 'S256'
});

window.location.href = \`https://quillon.xyz/api/v1/oauth2/authorize?\${params}\`;`}
                onCopy={(code) => copyToClipboard(code, 'authorize')}
                copied={copiedSection === 'authorize'}
              />
            </div>

            {/* Step 3 */}
            <div className="border-l-4 border-quantum-pink pl-6">
              <h3 className="text-2xl font-bold mb-3 flex items-center gap-2">
                <span className="bg-quantum-pink text-white rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold">
                  3
                </span>
                Exchange Code for Token
              </h3>
              <p className="text-gray-300 mb-4">
                On your <code className="bg-black/30 px-1 rounded">/callback</code> page, exchange the authorization code for an access token:
              </p>
              <CodeBlock
                language="javascript"
                code={`// In your /callback route handler:
const urlParams = new URLSearchParams(window.location.search);
const code = urlParams.get('code');
const codeVerifier = sessionStorage.getItem('pkce_verifier');

const tokenRes = await fetch('https://quillon.xyz/api/v1/oauth2/token', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    grant_type: 'authorization_code',
    code,
    redirect_uri: 'https://myapp.com/callback',
    client_id: 'YOUR_CLIENT_ID',
    client_secret: 'YOUR_CLIENT_SECRET', // omit for public clients
    code_verifier: codeVerifier
  })
});

const { access_token } = await tokenRes.json();

// Now fetch user info (wallet address)
const userRes = await fetch('https://quillon.xyz/api/v1/oauth2/userinfo', {
  headers: { Authorization: \`Bearer \${access_token}\` }
});
const user = await userRes.json();
console.log('Wallet:', user.data.wallet_address);`}
                onCopy={(code) => copyToClipboard(code, 'token')}
                copied={copiedSection === 'token'}
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
              scope="read:profile"
              name="Read Profile"
              description="View user's wallet address and basic account info"
              icon={<Globe className="w-6 h-6 text-quantum-pink" />}
            />
            <ScopeCard
              scope="send:transaction"
              name="Send Transactions"
              description="Send QUG and tokens on behalf of the user (coming soon)"
              icon={<Key className="w-6 h-6 text-quantum-purple" />}
              warning="Requires user approval for each transaction"
            />
            <ScopeCard
              scope="read:transactions"
              name="Read Transaction History"
              description="View user's past transactions and activity (coming soon)"
              icon={<Lock className="w-6 h-6 text-quantum-green" />}
            />
          </div>
        </section>

        {/* REST API Reference */}
        <section className="mb-12 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-6">📚 OAuth2 REST API Reference</h2>
          <p className="text-gray-300 mb-6">
            All OAuth2 endpoints are on the main Quillon domain: <code className="bg-black/30 px-2 py-1 rounded text-quantum-cyan">https://quillon.xyz</code>
          </p>

          <div className="space-y-6">
            <APIMethod
              method="POST /api/v1/oauth2/register"
              description="Register a new OAuth2 client application. Returns client_id and client_secret (auto-generated if not provided)."
              example={`curl -X POST https://quillon.xyz/api/v1/oauth2/register \\
  -H "Content-Type: application/json" \\
  -d '{"name":"My App","redirect_uris":["https://myapp.com/callback"]}'`}
              returns='{ "success": true, "data": { "client_id": "...", "client_secret": "..." } }'
            />

            <APIMethod
              method="GET /api/v1/oauth2/authorize"
              description="Start the authorization flow. Redirect the user's browser here with query params. Shows the Quillon consent screen."
              example={`https://quillon.xyz/api/v1/oauth2/authorize?
  response_type=code&client_id=YOUR_ID&redirect_uri=https://myapp.com/callback
  &scope=read:balance+read:profile&state=RANDOM&code_challenge=SHA256_HASH
  &code_challenge_method=S256`}
              params={[
                { name: 'response_type', type: 'string', description: 'Must be "code"' },
                { name: 'client_id', type: 'string', description: 'Your registered client ID' },
                { name: 'redirect_uri', type: 'string', description: 'Must match a registered redirect URI' },
                { name: 'scope', type: 'string', description: 'Space-separated scopes (e.g., "read:balance read:profile")' },
                { name: 'state', type: 'string', description: 'CSRF protection token (random string)' },
                { name: 'code_challenge', type: 'string', description: 'PKCE challenge (base64url of SHA-256 hash of verifier)' },
                { name: 'code_challenge_method', type: 'string', description: 'Must be "S256"' },
              ]}
              returns="Redirects to redirect_uri with ?code=AUTH_CODE&state=STATE"
            />

            <APIMethod
              method="POST /api/v1/oauth2/token"
              description="Exchange authorization code for an access token. Validates PKCE code_verifier against the original challenge."
              example={`curl -X POST https://quillon.xyz/api/v1/oauth2/token \\
  -H "Content-Type: application/json" \\
  -d '{"grant_type":"authorization_code","code":"AUTH_CODE",
       "redirect_uri":"https://myapp.com/callback","client_id":"YOUR_ID",
       "client_secret":"YOUR_SECRET","code_verifier":"ORIGINAL_VERIFIER"}'`}
              params={[
                { name: 'grant_type', type: 'string', description: 'Must be "authorization_code"' },
                { name: 'code', type: 'string', description: 'The authorization code from the callback' },
                { name: 'redirect_uri', type: 'string', description: 'Must match the one used in /authorize' },
                { name: 'client_id', type: 'string', description: 'Your registered client ID' },
                { name: 'client_secret', type: 'string', description: 'Your client secret (optional for public clients using PKCE)' },
                { name: 'code_verifier', type: 'string', description: 'The original PKCE code verifier' },
              ]}
              returns='{ "access_token": "...", "token_type": "bearer", "expires_in": 3600 }'
            />

            <APIMethod
              method="GET /api/v1/oauth2/userinfo"
              description="Get the authenticated user's wallet address and granted scopes. Requires Bearer token."
              example={`curl https://quillon.xyz/api/v1/oauth2/userinfo \\
  -H "Authorization: Bearer ACCESS_TOKEN"`}
              returns='{ "data": { "wallet_address": "qnk...", "scopes": ["read:balance", "read:profile"] } }'
            />

            <APIMethod
              method="POST /api/v1/oauth2/revoke"
              description="Revoke an access token, ending the session."
              example={`curl -X POST https://quillon.xyz/api/v1/oauth2/revoke \\
  -H "Content-Type: application/json" \\
  -d '{"token":"ACCESS_TOKEN"}'`}
              returns='{ "success": true }'
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
              title="Use PKCE for browser-based apps"
              description="For public clients (SPAs), omit client_secret and use PKCE code_verifier for security."
            />
            <SecurityItem
              icon={<CheckCircle className="w-6 h-6 text-green-500" />}
              title="Request minimal scopes"
              description="Only request the permissions your application actually needs."
            />
            <SecurityItem
              icon={<CheckCircle className="w-6 h-6 text-green-500" />}
              title="Validate state parameter"
              description="Always compare the state returned in the callback with the one you sent to prevent CSRF attacks."
            />
            <SecurityItem
              icon={<CheckCircle className="w-6 h-6 text-green-500" />}
              title="Handle token expiration"
              description="Access tokens expire after 1 hour. Re-authenticate the user when the token expires."
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
