/**
 * Error monitoring and crash reporting via Sentry.
 *
 * Peer review: "The document does not mention analytics or crash reporting.
 * For a production app, integrate Sentry or Firebase Crashlytics."
 */

import * as Sentry from '@sentry/react-native';

const SENTRY_DSN = ''; // Set in production via EAS secrets

/**
 * Initialize Sentry crash reporting.
 * Call once in app/_layout.tsx before any rendering.
 */
export function initMonitoring(): void {
  if (!SENTRY_DSN) {
    console.log('[Monitoring] Sentry DSN not configured, skipping init');
    return;
  }

  Sentry.init({
    dsn: SENTRY_DSN,
    debug: __DEV__,
    enableAutoSessionTracking: true,
    sessionTrackingIntervalMillis: 30_000,
    tracesSampleRate: __DEV__ ? 1.0 : 0.2,
    // Never send sensitive data
    beforeBreadcrumb(breadcrumb) {
      // Strip wallet addresses from breadcrumbs
      if (breadcrumb.message?.includes('qnk')) {
        breadcrumb.message = breadcrumb.message.replace(/qnk[0-9a-f]{64}/g, 'qnk[REDACTED]');
      }
      return breadcrumb;
    },
    beforeSend(event) {
      // Never send mnemonic or private key data
      if (event.extra) {
        for (const key of Object.keys(event.extra)) {
          if (/mnemonic|seed|private|secret/i.test(key)) {
            delete event.extra[key];
          }
        }
      }
      return event;
    },
  });
}

/**
 * Set user context (wallet address only, no PII).
 */
export function setMonitoringUser(address: string): void {
  Sentry.setUser({ id: address.slice(0, 12) + '...' });
}

/**
 * Clear user context on logout.
 */
export function clearMonitoringUser(): void {
  Sentry.setUser(null);
}

/**
 * Capture a non-fatal error with context.
 */
export function captureError(error: Error, context?: Record<string, string>): void {
  if (context) {
    Sentry.setContext('app_context', context);
  }
  Sentry.captureException(error);
}

/**
 * Add a breadcrumb for debugging.
 */
export function addBreadcrumb(message: string, category: string = 'app'): void {
  Sentry.addBreadcrumb({ message, category, level: 'info' });
}
