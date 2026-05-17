#!/usr/bin/env python3

import re

# Read the file
with open('crates/q-network/src/network_manager.rs', 'r') as f:
    content = f.read()

# Fix tracing macro calls with problematic structured logging
fixes = [
    # Pattern 1: Simple key-value pairs
    (r'"([^"]+)": "([^"]+),', r'"{}: {}", \2,'),
    
    # Pattern 2: Multiple structured fields
    (r'info!\(\s*"([^"]*)",\s*"connection_id": "connection_id,\s*"([^"]+)": "([^"]+),\s*\);', 
     r'info!("\1 - connection_id: {}, \2: {}", connection_id, \3);'),
     
    # Fix remaining malformed strings
    (r'"connection_id": "connection_id,', r'connection_id,'),
    (r'"([^"]+)": "([^"]+),', r'\2,'),
]

# Apply fixes
for pattern, replacement in fixes:
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# Special case fixes for specific patterns
content = content.replace(
    'info!(\n            "✅ P2P CONNECTION ESTABLISHED WITH FULL DEBUG",\n            connection_id,\n            hex::encode(validator_id),\n            p2p_debug.timing_metrics.total_connection_time_ms,\n            "connection_protocol" = ?p2p_debug.connection_protocol,\n            p2p_debug.performance_metrics.connection_quality_score,\n            p2p_debug.security_metrics.anonymity_score,\n        );',
    'info!(\n            "✅ P2P CONNECTION ESTABLISHED WITH FULL DEBUG - conn: {}, target: {}, time: {}ms, protocol: {:?}, perf: {:.2}, anon: {:.2}",\n            connection_id,\n            hex::encode(validator_id),\n            p2p_debug.timing_metrics.total_connection_time_ms,\n            p2p_debug.connection_protocol,\n            p2p_debug.performance_metrics.connection_quality_score,\n            p2p_debug.security_metrics.anonymity_score,\n        );'
)

content = content.replace(
    'info!(\n            "🔍 DNS RESOLUTION DEBUG",\n            connection_id,\n            onion_address,\n        );',
    'info!(\n            "🔍 DNS RESOLUTION DEBUG - connection_id: {}, target: {}",\n            connection_id,\n            onion_address,\n        );'
)

content = content.replace(
    'debug!(\n            "✅ DNS RESOLUTION COMPLETE",\n            connection_id,\n            resolution_time,\n            debug_info.queries_performed.len(),\n            debug_info.cache_hit,\n        );',
    'debug!(\n            "✅ DNS RESOLUTION COMPLETE - conn: {}, time: {}ms, queries: {}, cached: {}",\n            connection_id,\n            resolution_time,\n            debug_info.queries_performed.len(),\n            debug_info.cache_hit,\n        );'
)

content = content.replace(
    'info!(\n            "🧅 TOR CONNECTION DEBUG",\n            connection_id,\n            circuit_id,\n            peer_info.onion_address,\n        );',
    'info!(\n            "🧅 TOR CONNECTION DEBUG - conn: {}, circuit: {}, target: {}",\n            connection_id,\n            circuit_id,\n            peer_info.onion_address,\n        );'
)

content = content.replace(
    'info!(\n            "✅ TOR CIRCUIT ESTABLISHED",\n            connection_id,\n            circuit_id,\n            circuit_build_time,\n            tor_debug.bandwidth_allocation.allocated_bandwidth_kb_s,\n        );',
    'info!(\n            "✅ TOR CIRCUIT ESTABLISHED - conn: {}, circuit: {}, time: {}ms, bandwidth: {}kb/s",\n            connection_id,\n            circuit_id,\n            circuit_build_time,\n            tor_debug.bandwidth_allocation.allocated_bandwidth_kb_s,\n        );'
)

content = content.replace(
    'info!(\n            "👻 PHANTOM DNS CONNECTION DEBUG",\n            connection_id,\n            peer_info.onion_address,\n        );',
    'info!(\n            "👻 PHANTOM DNS CONNECTION DEBUG - conn: {}, target: {}",\n            connection_id,\n            peer_info.onion_address,\n        );'
)

content = content.replace(
    'debug!(\n            "✅ PHANTOM DNS ESTABLISHED",\n            connection_id,\n            phantom_debug.steganography_method,\n            phantom_debug.encoding_efficiency,\n            phantom_debug.detection_risk_score,\n        );',
    'debug!(\n            "✅ PHANTOM DNS ESTABLISHED - conn: {}, method: {}, efficiency: {:.2}, risk: {:.2}",\n            connection_id,\n            phantom_debug.steganography_method,\n            phantom_debug.encoding_efficiency,\n            phantom_debug.detection_risk_score,\n        );'
)

content = content.replace(
    'debug!(\n            "📊 CONNECTION PERFORMANCE MEASURED",\n            hex::encode(validator_id),\n            metrics.initial_latency_ms,\n            metrics.bandwidth_estimate_bps,\n            metrics.connection_quality_score,\n        );',
    'debug!(\n            "📊 CONNECTION PERFORMANCE MEASURED - validator: {}, latency: {}ms, bandwidth: {}bps, quality: {:.2}",\n            hex::encode(validator_id),\n            metrics.initial_latency_ms,\n            metrics.bandwidth_estimate_bps,\n            metrics.connection_quality_score,\n        );'
)

# Write the fixed content
with open('crates/q-network/src/network_manager.rs', 'w') as f:
    f.write(content)

print("Fixed tracing macros in network_manager.rs")