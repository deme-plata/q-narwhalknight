/**
 * Node.js + Express OAuth2 Integration Example
 *
 * This example shows how to integrate Quillon Wallet OAuth2
 * into a Node.js Express server.
 */

const express = require('express');
const session = require('express-session');
const QullionOAuth2Client = require('../quillon-oauth2-sdk');

const app = express();
const port = 3000;

// Session middleware for storing OAuth2 state
app.use(session({
  secret: 'your-secure-session-secret',
  resave: false,
  saveUninitialized: false,
  cookie: { secure: false } // Set to true in production with HTTPS
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Initialize OAuth2 client
// IMPORTANT: Store client secret in environment variables, NEVER in code!
const client = new QullionOAuth2Client({
  clientId: process.env.QUILLON_CLIENT_ID || 'your-client-id',
  clientSecret: process.env.QUILLON_CLIENT_SECRET || 'your-client-secret',
  redirectUri: 'http://localhost:3000/callback',
  scopes: ['read:balance', 'send:transaction', 'read:transactions'],
  apiUrl: process.env.QUILLON_API_URL || 'http://localhost:8080',
  walletUrl: process.env.QUILLON_WALLET_URL || 'http://localhost:5173'
});

// ============================================================================
// Routes
// ============================================================================

// Home page
app.get('/', (req, res) => {
  if (req.session.userInfo) {
    res.send(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>Quillon OAuth2 Demo</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
          }
          .card {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
          }
          button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            margin-right: 10px;
          }
          button:hover {
            opacity: 0.9;
          }
          .wallet-address {
            font-family: monospace;
            background: #e0e0e0;
            padding: 10px;
            border-radius: 4px;
            word-break: break-all;
          }
        </style>
      </head>
      <body>
        <h1>🎉 Welcome to Quillon App</h1>
        <div class="card">
          <h2>Connected Wallet</h2>
          <div class="wallet-address">${req.session.userInfo.wallet_address}</div>
        </div>
        <div class="card">
          <h2>Balance</h2>
          <p id="balance">Loading...</p>
          <button onclick="refreshBalance()">Refresh Balance</button>
        </div>
        <div class="card">
          <h2>Send Transaction</h2>
          <form action="/send" method="POST">
            <label>Recipient Address:</label><br>
            <input type="text" name="recipient" required style="width: 100%; padding: 8px; margin-bottom: 10px;"><br>
            <label>Amount (QUG):</label><br>
            <input type="number" name="amount" step="0.00000001" required style="width: 100%; padding: 8px; margin-bottom: 10px;"><br>
            <button type="submit">Send Transaction</button>
          </form>
        </div>
        <button onclick="window.location.href='/logout'">Logout</button>

        <script>
          async function refreshBalance() {
            const response = await fetch('/api/balance');
            const data = await response.json();
            document.getElementById('balance').textContent = data.balance + ' QUG';
          }
          refreshBalance();
        </script>
      </body>
      </html>
    `);
  } else {
    res.send(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>Quillon OAuth2 Demo</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            text-align: center;
          }
          button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
          }
          button:hover {
            opacity: 0.9;
          }
        </style>
      </head>
      <body>
        <h1>Welcome to Quillon OAuth2 Demo</h1>
        <p>Connect your Quillon Wallet to get started</p>
        <button onclick="window.location.href='/login'">Connect Quillon Wallet</button>
      </body>
      </html>
    `);
  }
});

// Start OAuth2 login flow
app.get('/login', async (req, res) => {
  try {
    // The SDK will redirect the user to Quillon Wallet
    await client.authorize();
  } catch (error) {
    res.status(500).send('Login error: ' + error.message);
  }
});

// OAuth2 callback
app.get('/callback', async (req, res) => {
  try {
    // Handle the OAuth2 callback
    const tokenResponse = await client.handleCallback();

    // Get user info
    const userInfo = await client.getUserInfo();

    // Store user info in session
    req.session.userInfo = userInfo;
    req.session.accessToken = tokenResponse.access_token;

    // Redirect to home
    res.redirect('/');
  } catch (error) {
    console.error('Callback error:', error);
    res.status(401).send('Authentication failed: ' + error.message);
  }
});

// API: Get balance
app.get('/api/balance', async (req, res) => {
  if (!req.session.userInfo) {
    return res.status(401).json({ error: 'Not authenticated' });
  }

  try {
    const balance = await client.getBalance('QUG');
    res.json({ balance: (balance / 100000000).toFixed(8) });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Send transaction
app.post('/send', async (req, res) => {
  if (!req.session.userInfo) {
    return res.status(401).send('Not authenticated');
  }

  const { recipient, amount } = req.body;

  try {
    const result = await client.sendTransaction({
      to: recipient,
      amount: Math.floor(parseFloat(amount) * 100000000), // Convert to base units
      token: 'QUG'
    });

    res.send(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>Transaction Sent</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            text-align: center;
          }
          .success {
            background: #d4edda;
            color: #155724;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
          }
          button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
          }
        </style>
      </head>
      <body>
        <div class="success">
          <h2>✅ Transaction Sent Successfully!</h2>
          <p>Sent ${amount} QUG to ${recipient}</p>
        </div>
        <button onclick="window.location.href='/'">Back to Dashboard</button>
      </body>
      </html>
    `);
  } catch (error) {
    res.status(500).send('Transaction failed: ' + error.message);
  }
});

// Logout
app.get('/logout', async (req, res) => {
  try {
    await client.revoke();
    req.session.destroy();
    res.redirect('/');
  } catch (error) {
    res.status(500).send('Logout error: ' + error.message);
  }
});

// ============================================================================
// Start server
// ============================================================================

app.listen(port, () => {
  console.log(`🚀 Quillon OAuth2 Demo Server running at http://localhost:${port}`);
  console.log('');
  console.log('Environment variables:');
  console.log('  QUILLON_CLIENT_ID:', process.env.QUILLON_CLIENT_ID || '(not set)');
  console.log('  QUILLON_CLIENT_SECRET:', process.env.QUILLON_CLIENT_SECRET ? '***' : '(not set)');
  console.log('  QUILLON_API_URL:', process.env.QUILLON_API_URL || 'http://localhost:8080 (default)');
  console.log('  QUILLON_WALLET_URL:', process.env.QUILLON_WALLET_URL || 'http://localhost:5173 (default)');
  console.log('');
  console.log('Visit http://localhost:3000 to test OAuth2 integration');
});
