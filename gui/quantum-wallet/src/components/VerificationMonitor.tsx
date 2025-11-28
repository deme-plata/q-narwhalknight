import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Shield,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Activity,
  Cpu,
  Eye,
  Zap,
} from 'lucide-react';

interface VerificationEvent {
  type: string;
  timestamp_ms: number;
  // Event-specific fields (discriminated union)
  request_id?: string;
  worker_node_id?: string;
  merkle_root?: string;
  token_count?: number;
  token_indices?: number[];
  deadline_ms?: number;
  proofs_count?: number;
  result?: string;
  tokens_verified?: number;
  reason?: string;
  amount_qbc?: number;
  challenge_id?: string;
  claimed_capability?: string;
  benchmark_tokens?: number;
  tokens_generated?: number;
  time_ms?: number;
  tokens_per_second?: number;
  score?: number;
  old_status?: string;
  new_status?: string;
  failed_worker_id?: string;
  retry_worker_id?: string;
  failure_type?: string;
  attempt?: number;
}

interface VerificationStats {
  total_proofs_submitted: number;
  total_proofs_verified: number;
  total_proofs_invalid: number;
  total_slashing_events: number;
  total_slashed_qbc: number;
  total_benchmarks_issued: number;
  total_benchmarks_passed: number;
  total_benchmarks_failed: number;
  active_workers: number;
  healthy_workers: number;
  unhealthy_workers: number;
  average_verification_time_ms: number;
}

interface WorkerHealthStatus {
  worker_node_id: string;
  health: string;
  reputation: number;
  recent_failures: number;
  total_requests: number;
  success_rate: number;
}

export default function VerificationMonitor() {
  const [events, setEvents] = useState<VerificationEvent[]>([]);
  const [stats, setStats] = useState<VerificationStats | null>(null);
  const [workers, setWorkers] = useState<WorkerHealthStatus[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [showEvents, setShowEvents] = useState(true);
  const [showWorkers, setShowWorkers] = useState(true);
  const [showStats, setShowStats] = useState(true);

  // Connect to SSE stream
  useEffect(() => {
    const eventSource = new EventSource('/api/verification/stream');

    eventSource.onopen = () => {
      console.log('🔗 Verification stream connected');
      setIsConnected(true);
    };

    eventSource.addEventListener('verification', (e) => {
      try {
        const event: VerificationEvent = JSON.parse(e.data);
        setEvents((prev) => [event, ...prev].slice(0, 100)); // Keep last 100 events
      } catch (error) {
        console.error('Failed to parse verification event:', error);
      }
    });

    eventSource.onerror = () => {
      console.error('❌ Verification stream error');
      setIsConnected(false);
    };

    return () => {
      eventSource.close();
    };
  }, []);

  // Fetch stats periodically
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/verification/stats');
        const data = await response.json();
        setStats(data);
      } catch (error) {
        console.error('Failed to fetch verification stats:', error);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 5000); // Every 5 seconds

    return () => clearInterval(interval);
  }, []);

  // Fetch worker health periodically
  useEffect(() => {
    const fetchWorkers = async () => {
      try {
        const response = await fetch('/api/verification/worker-health');
        const data = await response.json();
        setWorkers(data.workers || []);
      } catch (error) {
        console.error('Failed to fetch worker health:', error);
      }
    };

    fetchWorkers();
    const interval = setInterval(fetchWorkers, 5000); // Every 5 seconds

    return () => clearInterval(interval);
  }, []);

  // Format timestamp
  const formatTime = (timestamp_ms: number) => {
    const date = new Date(timestamp_ms);
    return date.toLocaleTimeString();
  };

  // Get event icon and color
  const getEventDisplay = (event: VerificationEvent) => {
    switch (event.type) {
      case 'ProofSubmitted':
        return {
          icon: <Shield className="w-4 h-4" />,
          color: 'text-blue-400',
          bgColor: 'bg-blue-500/10',
          title: 'Proof Submitted',
          description: `Worker ${event.worker_node_id?.slice(0, 8)} submitted proof for ${event.token_count} tokens`,
        };

      case 'ChallengeIssued':
        return {
          icon: <Eye className="w-4 h-4" />,
          color: 'text-yellow-400',
          bgColor: 'bg-yellow-500/10',
          title: 'Challenge Issued',
          description: `Challenging ${event.token_indices?.length} tokens from worker ${event.worker_node_id?.slice(0, 8)}`,
        };

      case 'VerificationComplete':
        const isValid = event.result === 'valid';
        return {
          icon: isValid ? <CheckCircle2 className="w-4 h-4" /> : <XCircle className="w-4 h-4" />,
          color: isValid ? 'text-green-400' : 'text-red-400',
          bgColor: isValid ? 'bg-green-500/10' : 'bg-red-500/10',
          title: `Verification ${event.result?.toUpperCase()}`,
          description: `${event.tokens_verified} tokens verified for ${event.worker_node_id?.slice(0, 8)}`,
        };

      case 'WorkerSlashed':
        return {
          icon: <AlertTriangle className="w-4 h-4" />,
          color: 'text-red-500',
          bgColor: 'bg-red-500/20',
          title: 'Worker Slashed',
          description: `${event.worker_node_id?.slice(0, 8)} penalized ${event.amount_qbc} QBC - ${event.reason}`,
        };

      case 'BenchmarkChallengeIssued':
        return {
          icon: <Cpu className="w-4 h-4" />,
          color: 'text-purple-400',
          bgColor: 'bg-purple-500/10',
          title: 'Benchmark Challenge',
          description: `Testing ${event.worker_node_id?.slice(0, 8)} - ${event.claimed_capability}`,
        };

      case 'BenchmarkVerificationComplete':
        const passed = event.result === 'passed';
        return {
          icon: passed ? <CheckCircle2 className="w-4 h-4" /> : <XCircle className="w-4 h-4" />,
          color: passed ? 'text-green-400' : 'text-red-400',
          bgColor: passed ? 'bg-green-500/10' : 'bg-red-500/10',
          title: `Benchmark ${event.result?.toUpperCase()}`,
          description: `${event.worker_node_id?.slice(0, 8)} scored ${event.score?.toFixed(3) || 'N/A'}`,
        };

      case 'FailoverEvent':
        return {
          icon: <Zap className="w-4 h-4" />,
          color: 'text-orange-400',
          bgColor: 'bg-orange-500/10',
          title: 'Failover',
          description: `Request failover: ${event.failed_worker_id?.slice(0, 8)} → ${event.retry_worker_id?.slice(0, 8)}`,
        };

      default:
        return {
          icon: <Activity className="w-4 h-4" />,
          color: 'text-gray-400',
          bgColor: 'bg-gray-500/10',
          title: event.type,
          description: 'Unknown event type',
        };
    }
  };

  // Get worker health color
  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy':
        return 'text-green-400 bg-green-500/10';
      case 'degraded':
        return 'text-yellow-400 bg-yellow-500/10';
      case 'unhealthy':
        return 'text-red-400 bg-red-500/10';
      case 'testing':
        return 'text-blue-400 bg-blue-500/10';
      default:
        return 'text-gray-400 bg-gray-500/10';
    }
  };

  return (
    <div className="space-y-4">
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
          <span className="text-sm text-gray-400">
            {isConnected ? 'Live Monitoring Active' : 'Disconnected'}
          </span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowStats(!showStats)}
            className={`px-3 py-1 rounded-lg text-xs ${
              showStats ? 'bg-blue-500/20 text-blue-400' : 'bg-gray-700 text-gray-400'
            }`}
          >
            Stats
          </button>
          <button
            onClick={() => setShowWorkers(!showWorkers)}
            className={`px-3 py-1 rounded-lg text-xs ${
              showWorkers ? 'bg-purple-500/20 text-purple-400' : 'bg-gray-700 text-gray-400'
            }`}
          >
            Workers
          </button>
          <button
            onClick={() => setShowEvents(!showEvents)}
            className={`px-3 py-1 rounded-lg text-xs ${
              showEvents ? 'bg-green-500/20 text-green-400' : 'bg-gray-700 text-gray-400'
            }`}
          >
            Events
          </button>
        </div>
      </div>

      {/* Statistics Panel */}
      <AnimatePresence>
        {showStats && stats && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-4 border border-gray-700"
          >
            <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Verification Statistics
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-2xl font-bold text-green-400">{stats.total_proofs_verified}</div>
                <div className="text-xs text-gray-400">Proofs Verified</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-2xl font-bold text-red-400">{stats.total_proofs_invalid}</div>
                <div className="text-xs text-gray-400">Invalid Proofs</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-2xl font-bold text-yellow-400">{stats.total_slashed_qbc.toFixed(2)}</div>
                <div className="text-xs text-gray-400">QBC Slashed</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-2xl font-bold text-blue-400">{stats.healthy_workers}/{stats.active_workers}</div>
                <div className="text-xs text-gray-400">Healthy Workers</div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Worker Health Panel */}
      <AnimatePresence>
        {showWorkers && workers.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-4 border border-gray-700"
          >
            <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              Worker Health ({workers.length})
            </h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {workers.map((worker) => (
                <div
                  key={worker.worker_node_id}
                  className="bg-gray-800/50 rounded-lg p-3 flex items-center justify-between"
                >
                  <div className="flex items-center gap-3">
                    <div className={`px-2 py-1 rounded-lg text-xs font-medium ${getHealthColor(worker.health)}`}>
                      {worker.health}
                    </div>
                    <div>
                      <div className="text-sm font-mono text-gray-300">
                        {worker.worker_node_id.slice(0, 12)}...
                      </div>
                      <div className="text-xs text-gray-500">
                        {worker.total_requests} requests · {(worker.success_rate * 100).toFixed(1)}% success
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="text-right">
                      <div className="text-sm font-semibold text-gray-300">
                        {(worker.reputation * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-gray-500">reputation</div>
                    </div>
                    {worker.recent_failures > 0 && (
                      <div className="text-red-400 text-xs">
                        {worker.recent_failures} fails
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Live Events Feed */}
      <AnimatePresence>
        {showEvents && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-4 border border-gray-700"
          >
            <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Live Verification Events
            </h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              <AnimatePresence mode="popLayout">
                {events.map((event, index) => {
                  const display = getEventDisplay(event);
                  return (
                    <motion.div
                      key={`${event.timestamp_ms}-${index}`}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      className={`${display.bgColor} rounded-lg p-3 flex items-start gap-3 border border-gray-700/50`}
                    >
                      <div className={`${display.color} mt-0.5`}>{display.icon}</div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <span className={`text-sm font-medium ${display.color}`}>
                            {display.title}
                          </span>
                          <span className="text-xs text-gray-500">
                            {formatTime(event.timestamp_ms)}
                          </span>
                        </div>
                        <p className="text-xs text-gray-400 leading-relaxed">
                          {display.description}
                        </p>
                      </div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
              {events.length === 0 && (
                <div className="text-center text-gray-500 text-sm py-8">
                  No verification events yet. Waiting for activity...
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
