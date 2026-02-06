// Q-NarwhalKnight Node Auto-Discovery Service
// v1.0.53: Automatically discovers the API server port when default port is unavailable

/**
 * Configuration for node discovery
 */
export interface NodeDiscoveryConfig {
  /** Base host to try (default: current window origin or localhost) */
  baseHost?: string;
  /** Starting port to try (default: 8080) */
  startPort?: number;
  /** Maximum number of ports to try (default: 10) */
  maxAttempts?: number;
  /** Timeout for each connection attempt in ms (default: 2000) */
  timeout?: number;
  /** Whether to use HTTPS (auto-detected from current page) */
  useHttps?: boolean;
}

/**
 * Result of node discovery
 */
export interface NodeDiscoveryResult {
  success: boolean;
  url: string | null;
  port: number | null;
  error?: string;
  attemptedPorts: number[];
}

/**
 * Cached discovery result to avoid repeated discovery attempts
 */
let cachedResult: NodeDiscoveryResult | null = null;
let lastDiscoveryTime: number = 0;
const CACHE_DURATION_MS = 60000; // Cache for 1 minute

/**
 * Check if a node is available at the given URL
 */
async function checkNodeHealth(url: string, timeout: number): Promise<boolean> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(`${url}/api/v1/health`, {
      method: 'GET',
      signal: controller.signal,
      headers: {
        'Accept': 'application/json',
      },
    });

    clearTimeout(timeoutId);

    if (response.ok) {
      // Verify it's actually our node by checking response format
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const data = await response.json();
        // Our health endpoint returns success: true
        return data.success === true || data.status === 'healthy' || data === 'OK';
      }
    }
    return false;
  } catch (error) {
    clearTimeout(timeoutId);
    return false;
  }
}

/**
 * Discover the Q-NarwhalKnight node API server
 *
 * Tries ports starting from startPort up to startPort + maxAttempts
 * Returns the first working endpoint found
 *
 * @param config - Discovery configuration
 * @returns Discovery result with URL if found
 */
export async function discoverNode(config: NodeDiscoveryConfig = {}): Promise<NodeDiscoveryResult> {
  // Check cache first
  const now = Date.now();
  if (cachedResult && (now - lastDiscoveryTime) < CACHE_DURATION_MS && cachedResult.success) {
    console.log('🔍 [Node Discovery] Using cached result:', cachedResult.url);
    return cachedResult;
  }

  // v3.4.15: Handle .onion Tor hidden service domains
  // When accessing via .onion, the API is proxied through nginx at port 80
  // Don't try port discovery - use relative URL path instead
  if (typeof window !== 'undefined' && window.location) {
    const currentHost = window.location.hostname;
    if (currentHost.endsWith('.onion')) {
      console.log('🧅 [Node Discovery] Detected Tor hidden service, using relative API path');
      const result: NodeDiscoveryResult = {
        success: true,
        url: '',  // Empty URL means use relative /api path
        port: 80,
        attemptedPorts: [],
      };
      cachedResult = result;
      lastDiscoveryTime = now;
      return result;
    }
  }

  // Check if user has manually set an API URL
  const manualUrl = localStorage.getItem('apiBaseURL');
  if (manualUrl) {
    console.log('🔍 [Node Discovery] Using manually configured URL:', manualUrl);
    // Verify the manual URL works
    const isHealthy = await checkNodeHealth(manualUrl, config.timeout || 2000);
    if (isHealthy) {
      const result: NodeDiscoveryResult = {
        success: true,
        url: manualUrl,
        port: parseInt(new URL(manualUrl).port) || (manualUrl.startsWith('https') ? 443 : 80),
        attemptedPorts: [],
      };
      cachedResult = result;
      lastDiscoveryTime = now;
      return result;
    }
    console.warn('⚠️ [Node Discovery] Manually configured URL is not responding, trying auto-discovery...');
  }

  const {
    startPort = 8080,
    maxAttempts = 10,
    timeout = 2000,
  } = config;

  // Determine base host and protocol
  let baseHost = config.baseHost;
  let useHttps = config.useHttps;

  if (!baseHost) {
    // Auto-detect from current page
    if (typeof window !== 'undefined' && window.location) {
      const currentHost = window.location.hostname;
      useHttps = useHttps ?? window.location.protocol === 'https:';

      // If we're on the production domain, use it
      if (currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
        baseHost = currentHost;
      } else {
        baseHost = 'localhost';
      }
    } else {
      baseHost = 'localhost';
      useHttps = useHttps ?? false;
    }
  }

  const attemptedPorts: number[] = [];

  console.log(`🔍 [Node Discovery] Starting discovery on ${baseHost}`);

  // v3.6.11-beta: CRITICAL FIX - When on HTTPS, try the Nginx-proxied endpoint FIRST
  // Port 8080 is HTTP-only (no SSL), so https://quillon.xyz:8080 will always fail
  // The correct approach: use https://quillon.xyz (port 443, via Nginx) for HTTPS sites
  if (useHttps && baseHost !== 'localhost' && baseHost !== '127.0.0.1') {
    // Try the standard HTTPS endpoint (no port = 443, proxied by Nginx)
    const httpsUrl = `https://${baseHost}`;
    console.log(`🔍 [Node Discovery] Trying HTTPS via Nginx: ${httpsUrl}...`);
    attemptedPorts.push(443);

    const isHealthy = await checkNodeHealth(httpsUrl, timeout);
    if (isHealthy) {
      console.log(`✅ [Node Discovery] Found node at ${httpsUrl} (via Nginx)`);
      const result: NodeDiscoveryResult = {
        success: true,
        url: httpsUrl,
        port: 443,
        attemptedPorts,
      };
      cachedResult = result;
      lastDiscoveryTime = now;
      localStorage.setItem('apiBaseURL', httpsUrl);
      localStorage.setItem('discoveredPort', '443');
      window.dispatchEvent(new CustomEvent('node-discovered', { detail: result }));
      return result;
    }
    console.warn('⚠️ [Node Discovery] HTTPS via Nginx failed, will NOT try port 8080 with HTTPS (no SSL there)');
  }

  // For HTTP or localhost, try ports as before
  const protocol = useHttps ? 'https' : 'http';
  // Only try port discovery for HTTP (not HTTPS on production domain)
  if (useHttps && baseHost !== 'localhost' && baseHost !== '127.0.0.1') {
    // Skip port discovery for HTTPS production - we already tried Nginx above
    console.log('🔍 [Node Discovery] Skipping port discovery for HTTPS production domain');
  } else {
    console.log(`🔍 [Node Discovery] Trying ports ${startPort}-${startPort + maxAttempts - 1}`);

    // Try ports in sequence (HTTP only for non-localhost, or any protocol for localhost)
    for (let i = 0; i < maxAttempts; i++) {
      const port = startPort + i;
      attemptedPorts.push(port);

      const url = `${protocol}://${baseHost}:${port}`;
      console.log(`🔍 [Node Discovery] Trying ${url}...`);

      const isHealthy = await checkNodeHealth(url, timeout);

      if (isHealthy) {
        console.log(`✅ [Node Discovery] Found node at ${url}`);

        const result: NodeDiscoveryResult = {
          success: true,
          url,
          port,
          attemptedPorts,
        };

        // Cache the result
        cachedResult = result;
        lastDiscoveryTime = now;

        // Save to localStorage for future use
        localStorage.setItem('apiBaseURL', url);
        localStorage.setItem('discoveredPort', port.toString());

        // Notify other components about the discovery
        window.dispatchEvent(new CustomEvent('node-discovered', { detail: result }));

        return result;
      }
    }
  }

  // No node found
  console.error(`❌ [Node Discovery] No node found on ports ${startPort}-${startPort + maxAttempts - 1}`);

  const result: NodeDiscoveryResult = {
    success: false,
    url: null,
    port: null,
    error: `No Q-NarwhalKnight node found on ports ${startPort}-${startPort + maxAttempts - 1}. Is the node running?`,
    attemptedPorts,
  };

  return result;
}

/**
 * Clear the discovery cache and force re-discovery
 */
export function clearDiscoveryCache(): void {
  cachedResult = null;
  lastDiscoveryTime = 0;
  localStorage.removeItem('apiBaseURL');
  localStorage.removeItem('discoveredPort');
  console.log('🗑️ [Node Discovery] Cache cleared');
}

/**
 * Get the currently discovered/configured node URL
 * Returns null if no node has been discovered
 */
export function getDiscoveredNodeUrl(): string | null {
  return localStorage.getItem('apiBaseURL');
}

/**
 * Get the currently discovered port
 */
export function getDiscoveredPort(): number | null {
  const port = localStorage.getItem('discoveredPort');
  return port ? parseInt(port) : null;
}

/**
 * React hook-compatible function to discover node on mount
 * Call this in useEffect or component initialization
 */
export async function initializeNodeConnection(): Promise<NodeDiscoveryResult> {
  console.log('🚀 [Node Discovery] Initializing node connection...');

  const result = await discoverNode();

  if (!result.success) {
    console.error('❌ [Node Discovery] Failed to connect to node:', result.error);
  }

  return result;
}

/**
 * Subscribe to node discovery events
 * @param callback - Function to call when a node is discovered
 * @returns Cleanup function to remove listener
 */
export function onNodeDiscovered(callback: (result: NodeDiscoveryResult) => void): () => void {
  const handler = (event: Event) => {
    const customEvent = event as CustomEvent<NodeDiscoveryResult>;
    callback(customEvent.detail);
  };

  window.addEventListener('node-discovered', handler);

  return () => {
    window.removeEventListener('node-discovered', handler);
  };
}

/**
 * Test connection to a specific URL
 * Useful for manual configuration validation
 */
export async function testConnection(url: string, timeout: number = 2000): Promise<boolean> {
  console.log(`🔍 [Node Discovery] Testing connection to ${url}...`);
  const isHealthy = await checkNodeHealth(url, timeout);
  console.log(`${isHealthy ? '✅' : '❌'} [Node Discovery] Connection test ${isHealthy ? 'passed' : 'failed'}`);
  return isHealthy;
}
