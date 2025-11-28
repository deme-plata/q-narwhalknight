import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import ErrorBoundary from './components/ErrorBoundary.tsx'
import { PasswordModalProvider } from './contexts/PasswordModalContext.tsx'
import { SessionTimeoutProvider } from './contexts/SessionTimeoutContext.tsx'
import { LibP2PProvider } from './contexts/LibP2PContext.tsx'
import * as ed25519 from '@noble/ed25519'
import { sha512 } from '@noble/hashes/sha512'

console.log('🎯 main.tsx executing');

// Configure noble-ed25519 to use SHA-512 from @noble/hashes
// This is required for Ed25519 signature generation
ed25519.etc.sha512Sync = (...m: Uint8Array[]) => sha512(ed25519.etc.concatBytes(...m));
ed25519.etc.sha512Async = async (...m: Uint8Array[]) => sha512(ed25519.etc.concatBytes(...m));

console.log('✅ Ed25519 SHA-512 configured');

// Global error handlers
window.addEventListener('error', (event) => {
  console.error('❌ Global error caught:', event.error);
  console.error('Error message:', event.message);
  console.error('Error stack:', event.error?.stack);
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('❌ Unhandled promise rejection:', event.reason);
});

createRoot(document.getElementById('root')!).render(
  <ErrorBoundary>
    <SessionTimeoutProvider>
      <PasswordModalProvider>
        <LibP2PProvider>
          <App />
        </LibP2PProvider>
      </PasswordModalProvider>
    </SessionTimeoutProvider>
  </ErrorBoundary>
)
