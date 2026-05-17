# Quillon Wallet OAuth2 SDK

Official JavaScript/TypeScript SDK for integrating Quillon Wallet authentication into third-party websites and applications.

## Features

- ✅ **OAuth2 2.0 Standard** - Full compliance with RFC 6749
- 🔐 **PKCE Security** - Protection against authorization code interception
- 🔑 **Post-Quantum Ready** - Supports Kyber1024 encryption
- 🎯 **Easy Integration** - Simple API, minimal setup
- 📦 **TypeScript Support** - Full type definitions included
- ⚡ **Token Management** - Automatic refresh and expiration handling
- 🌐 **Cross-Platform** - Works in browsers and Node.js
- 🔄 **DEX Support** - Built-in token swapping functionality
- 📡 **Real-time Updates** - Server-Sent Events (SSE) for live balance updates
- 💰 **Multi-Token Support** - Manage multiple tokens including QUGUSD stablecoin

## Installation

### NPM

```bash
npm install @quillon/oauth2-sdk
```

### CDN

```html
<script src="https://cdn.quillon.xyz/oauth2-sdk/v1/quillon-oauth2-sdk.js"></script>
```

### Manual

Download `quillon-oauth2-sdk.js` from this repository and include it in your project.

## Quick Start

### 1. Register Your Application

First, register your application with Quillon to get OAuth2 credentials:

```javascript
// POST https://api.quillon.xyz/api/v1/oauth2/register
const response = await fetch('https://api.quillon.xyz/api/v1/oauth2/register', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'My Awesome App',
    redirect_uris: ['https://myapp.com/callback'],
    website: 'https://myapp.com',
    scopes: ['read:balance', 'send:transaction']
  })
});

const { client_id, client_secret } = await response.json();
```

**Important:** Store `client_secret` securely on your server. Never expose it in client-side code.

### 2. Initialize the SDK

```javascript
import QullionOAuth2Client from '@quillon/oauth2-sdk';

const client = new QullionOAuth2Client({
  clientId: 'your-client-id',
  clientSecret: 'your-client-secret', // Server-side only!
  redirectUri: 'https://myapp.com/callback',
  scopes: ['read:balance', 'send:transaction', 'read:transactions']
});
```

### 3. Start Authorization Flow

```javascript
// Redirect user to Quillon Wallet for consent
await client.authorize();
```

### 4. Handle Callback

```javascript
// In your /callback page
try {
  const tokenResponse = await client.handleCallback();
  console.log('Access token:', tokenResponse.access_token);

  // User is now authenticated!
  const userInfo = await client.getUserInfo();
  console.log('Wallet address:', userInfo.wallet_address);
} catch (error) {
  console.error('Authentication failed:', error);
}
```

### 5. Use Authenticated APIs

```javascript
// Get balance
const balance = await client.getBalance('QUG');
console.log(`Balance: ${balance / 100000000} QUG`);

// Send transaction
await client.sendTransaction({
  to: 'recipient-wallet-address',
  amount: 1000000000, // 10 QUG (8 decimal places)
  token: 'QUG'
});

// Get transaction history
const transactions = await client.getTransactionHistory(20);
```

## API Reference

### Constructor

```typescript
new QullionOAuth2Client(config: QullionOAuth2Config)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `clientId` | string | ✅ | Your OAuth2 client ID |
| `clientSecret` | string | ⚠️ | Your OAuth2 client secret (server-side only) |
| `redirectUri` | string | ✅ | Registered redirect URI |
| `scopes` | string[] | ❌ | Requested scopes (default: `['read:balance']`) |
| `walletUrl` | string | ❌ | Wallet URL (default: `https://wallet.quillon.xyz`) |
| `apiUrl` | string | ❌ | API URL (default: `https://api.quillon.xyz`) |

### Available Scopes

| Scope | Description |
|-------|-------------|
| `read:balance` | View token balances |
| `send:transaction` | Send transactions on behalf of user |
| `read:transactions` | View transaction history |
| `manage:tokens` | Create and manage custom tokens |
| `swap:tokens` | Swap tokens on the DEX |

### Methods

#### `authorize()`

Starts the OAuth2 authorization flow. Redirects user to Quillon Wallet.

```typescript
await client.authorize();
```

#### `handleCallback()`

Handles the OAuth2 callback after user consent. Call this on your redirect URI page.

```typescript
const tokenResponse = await client.handleCallback();
// Returns: { access_token, token_type, expires_in, refresh_token, scope }
```

#### `getUserInfo()`

Gets authenticated user information.

```typescript
const userInfo = await client.getUserInfo();
// Returns: { wallet_address, scopes, client_id }
```

#### `getBalance(token)`

Gets user's token balance.

```typescript
const balance = await client.getBalance('QUG');
// Returns: number (in base units, 8 decimals)
```

#### `sendTransaction(params)`

Sends a transaction on behalf of the user.

```typescript
await client.sendTransaction({
  to: 'wallet-address',
  amount: 100000000, // Amount in base units
  token: 'QUG'
});
```

#### `getTransactionHistory(limit)`

Gets user's transaction history.

```typescript
const transactions = await client.getTransactionHistory(20);
// Returns: Transaction[]
```

#### `isAuthenticated()`

Checks if user is currently authenticated.

```typescript
if (client.isAuthenticated()) {
  console.log('User is logged in');
}
```

#### `revoke()`

Revokes access token and logs out user.

```typescript
await client.revoke();
```

#### `swapTokens(from, to, amount)`

Swaps tokens on the Quillon DEX.

```typescript
const result = await client.swapTokens('qug', 'qugusd', 10.5);
// Swaps 10.5 QUG for QUGUSD
console.log('Received:', result.data.amount_out / 100000000);
```

#### `getTokenBalances()`

Gets all token balances for the authenticated user.

```typescript
const tokens = await client.getTokenBalances();
// Returns: { qug: {...}, qugusd: {...}, ... }
```

#### `setupSSE(onBalanceUpdate, onTransaction)`

Sets up Server-Sent Events for real-time updates.

```typescript
client.setupSSE(
  () => console.log('Balance updated!'),
  (tx) => console.log('Transaction confirmed:', tx)
);
```

#### `authenticatedRequest(endpoint, options)`

Makes a custom authenticated API request.

```typescript
const response = await client.authenticatedRequest('/api/v1/custom/endpoint', {
  method: 'POST',
  body: JSON.stringify({ data: 'value' })
});
```

## Examples

### Vanilla JavaScript

See [examples/basic-integration.html](examples/basic-integration.html) for a complete HTML/JS example.

```html
<!DOCTYPE html>
<html>
<head>
  <script src="quillon-oauth2-sdk.js"></script>
</head>
<body>
  <button id="login">Connect Wallet</button>
  <div id="balance"></div>

  <script>
    const client = new QullionOAuth2Client({
      clientId: 'your-client-id',
      redirectUri: window.location.origin
    });

    document.getElementById('login').addEventListener('click', () => {
      client.authorize();
    });

    // Handle callback
    if (new URLSearchParams(window.location.search).has('code')) {
      client.handleCallback().then(() => {
        return client.getBalance('QUG');
      }).then(balance => {
        document.getElementById('balance').textContent =
          `Balance: ${balance / 100000000} QUG`;
      });
    }
  </script>
</body>
</html>
```

### React + TypeScript

See [examples/react-integration.tsx](examples/react-integration.tsx) for a complete React example with DEX and SSE support.

```tsx
import { QullionProvider, useQuillon, QullionLoginButton, QullionSwapForm } from './quillon-integration';

function App() {
  return (
    <QullionProvider
      clientId="your-client-id"
      redirectUri={window.location.origin + '/callback'}
      scopes={['read:balance', 'send:transaction', 'swap:tokens']}
    >
      <Dashboard />
    </QullionProvider>
  );
}

function Dashboard() {
  const { isAuthenticated, balance, tokens, swapTokens } = useQuillon();

  if (!isAuthenticated) {
    return <QullionLoginButton />;
  }

  return (
    <div>
      {/* Primary Balance */}
      <h1>Balance: {balance ? (balance / 100000000) : 'Loading...'} QUG</h1>

      {/* All Token Balances - Updates automatically via SSE */}
      {Object.entries(tokens).map(([key, token]) => (
        <div key={key}>
          {token.symbol}: {token.balance.toFixed(8)}
        </div>
      ))}

      {/* DEX Swap Component */}
      <QullionSwapForm />
    </div>
  );
}
```

### Node.js Server

```javascript
const QullionOAuth2Client = require('@quillon/oauth2-sdk');
const express = require('express');

const app = express();
const client = new QullionOAuth2Client({
  clientId: process.env.QUILLON_CLIENT_ID,
  clientSecret: process.env.QUILLON_CLIENT_SECRET,
  redirectUri: 'http://localhost:3000/callback'
});

app.get('/login', (req, res) => {
  client.authorize();
});

app.get('/callback', async (req, res) => {
  try {
    await client.handleCallback();
    const userInfo = await client.getUserInfo();
    res.json({ success: true, user: userInfo });
  } catch (error) {
    res.status(401).json({ error: error.message });
  }
});

app.listen(3000);
```

## Security Best Practices

### 1. Never Expose Client Secret

❌ **DON'T:**
```javascript
// In browser/frontend code
const client = new QullionOAuth2Client({
  clientSecret: 'abc123' // NEVER DO THIS!
});
```

✅ **DO:**
```javascript
// Use a server-side proxy for token exchange
// Frontend only handles authorization, backend handles token exchange
```

### 2. Use HTTPS in Production

Always use HTTPS for your redirect URIs and API requests in production.

### 3. Validate State Parameter

The SDK automatically validates the `state` parameter to prevent CSRF attacks.

### 4. Store Tokens Securely

Tokens are stored in `localStorage` by default. For sensitive applications, consider:

- HttpOnly cookies (server-side)
- Encrypted storage
- Session-only storage

### 5. Handle Token Expiration

The SDK automatically refreshes expired tokens. Always use `getAccessToken()` to ensure token validity:

```javascript
const token = await client.getAccessToken(); // Auto-refreshes if expired
```

## DEX (Token Swapping)

The SDK provides built-in support for swapping tokens on the Quillon DEX:

```javascript
// Swap 10 QUG for QUGUSD
const result = await client.swapTokens('qug', 'qugusd', 10);

console.log('Swap successful!');
console.log('Received:', result.data.amount_out / 100000000, 'QUGUSD');
console.log('Transaction ID:', result.data.transaction_id);
```

### Supported Tokens

- **QUG** - Quillon native token
- **QUGUSD** - Quillon USD stablecoin (algorithmically pegged to $1)
- Custom tokens created on the platform

### Swap Pricing

Token swaps use automated market maker (AMM) pricing with dynamic fees based on liquidity pools.

## Real-time Updates (SSE)

The SDK automatically connects to Server-Sent Events for real-time balance and transaction updates:

```javascript
// Setup is automatic when you use QullionProvider in React
// Or manually for vanilla JS:
const eventSource = new EventSource('https://api.quillon.xyz/api/v1/stream');

eventSource.addEventListener('balance-updated', (event) => {
  console.log('💰 Balance updated!');
  // Refresh your UI
});

eventSource.addEventListener('transaction-confirmed', (event) => {
  const data = JSON.parse(event.data);
  console.log('✅ Transaction confirmed:', data.transaction_id);
});
```

### SSE Events

| Event | Description |
|-------|-------------|
| `balance-updated` | User's balance has changed |
| `transaction-confirmed` | Transaction has been confirmed on-chain |
| `token-created` | New token has been minted |
| `swap-completed` | DEX swap has completed |

## Base Units Conversion

Quillon uses **8 decimal places** for all tokens:

```javascript
// 1 QUG = 100,000,000 base units
const displayBalance = baseUnits / 100000000;
const baseUnits = displayBalance * 100000000;

// Examples:
10.5 QUG = 1,050,000,000 base units
0.00000001 QUG = 1 base unit (smallest unit)
```

## Error Handling

```javascript
try {
  await client.authorize();
} catch (error) {
  if (error.message.includes('access_denied')) {
    console.log('User denied access');
  } else if (error.message.includes('State mismatch')) {
    console.log('Possible CSRF attack');
  } else {
    console.log('Authentication error:', error.message);
  }
}
```

## Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

Requires `crypto.subtle` API for PKCE support.

## Development

### Testing Locally

```bash
# Start a local server
python3 -m http.server 8000

# Open browser
open http://localhost:8000/examples/basic-integration.html
```

### Development Mode

```javascript
const client = new QullionOAuth2Client({
  clientId: 'dev-client-id',
  redirectUri: 'http://localhost:8000',
  walletUrl: 'http://localhost:5173', // Local wallet
  apiUrl: 'http://localhost:8080'     // Local API
});
```

## Support

- 📚 Documentation: https://api.quillon.xyz
- 🐛 Issues: https://github.com/deme-plata/q-narwhalknight/issues
- 💬 Discord: https://discord.gg/jEhaYtAhfx
- 📧 Email: bitknight.dipper688@passmail.net

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](../CONTRIBUTING.md) first.

---

Built with ❤️ by the Quillon Team | Powered by Post-Quantum Cryptography
