import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('❌ Error caught by boundary:', error, errorInfo);
    this.setState({
      error,
      errorInfo,
    });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: '#0A0B14',
          color: 'white',
          padding: '20px',
          overflow: 'auto',
          fontFamily: 'monospace',
          zIndex: 99999,
        }}>
          <h1 style={{ color: '#ff0080', marginBottom: '20px' }}>
            ⚠️ Application Error
          </h1>
          <div style={{ background: '#1a1b26', padding: '15px', borderRadius: '8px', marginBottom: '20px' }}>
            <h2 style={{ color: '#00d9ff', marginBottom: '10px' }}>Error:</h2>
            <pre style={{ color: '#ff6b6b', whiteSpace: 'pre-wrap' }}>
              {this.state.error?.toString()}
            </pre>
          </div>
          {this.state.errorInfo && (
            <div style={{ background: '#1a1b26', padding: '15px', borderRadius: '8px' }}>
              <h2 style={{ color: '#00d9ff', marginBottom: '10px' }}>Stack Trace:</h2>
              <pre style={{ color: '#gray', whiteSpace: 'pre-wrap', fontSize: '12px' }}>
                {this.state.errorInfo.componentStack}
              </pre>
            </div>
          )}
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: '20px',
              padding: '10px 20px',
              background: '#6b46c1',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
            }}
          >
            Reload Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
