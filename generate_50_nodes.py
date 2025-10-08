#!/usr/bin/env python3
"""
Generate complete 50-node Docker Compose configuration for Q-NarwhalKnight massive scale testing
"""

def generate_validator_node(node_id):
    """Generate a validator node configuration"""
    return f"""
  validator-{node_id:02d}:
    build:
      context: .
      dockerfile: docker/Dockerfile.qnarwhal
    container_name: q-validator-{node_id:02d}
    environment:
      - RUST_LOG=info
      - Q_NODE_ID=validator-{node_id:02d}
      - Q_SERVER_ROLE=validator
      - Q_CONSENSUS_MODE=dag_knight
      - Q_TOR_PROXY=tor-proxy:9050
      - Q_DNS_PHANTOM_DISCOVERY=true
      - Q_VALIDATOR_INDEX={node_id}
    networks:
      - qnarwhal-net
    depends_on:
      - dns-phantom-hub
      - tor-proxy
    volumes:
      - ./target/release:/app/bin
      - ./docker/logs:/app/logs
    command: ["/app/bin/q-api-server", "--node-id", "validator-{node_id:02d}", "--consensus-validator", "--validator-index", "{node_id}"]"""

def generate_complete_docker_compose():
    """Generate the complete Docker Compose file with all 50 nodes"""
    
    # Base configuration (infrastructure services)
    base_config = """version: '3.8'

services:
  # Core infrastructure services
  tor-proxy:
    image: torproject/tor:latest
    container_name: q-tor-proxy
    ports:
      - "9050:9050"
      - "9051:9051"
    volumes:
      - ./docker/tor-config:/etc/tor
    command: ["tor", "-f", "/etc/tor/torrc"]
    networks:
      - qnarwhal-net
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "9050"]
      interval: 5s
      retries: 5

  dns-phantom-hub:
    build:
      context: .
      dockerfile: docker/Dockerfile.qnarwhal
    container_name: q-dns-phantom-hub
    ports:
      - "8053:53/udp"
      - "8080:8080"
    environment:
      - RUST_LOG=info
      - Q_NODE_ROLE=dns_phantom_hub
      - Q_DNS_STEGANOGRAPHIC=true
    networks:
      - qnarwhal-net
    depends_on:
      - tor-proxy
    volumes:
      - ./target/release:/app/bin
      - ./docker/logs:/app/logs
    command: ["/app/bin/q-api-server", "--dns-phantom-hub", "--port", "8080"]

  # Beta server - main coordinator
  beta-coordinator:
    build:
      context: .
      dockerfile: docker/Dockerfile.qnarwhal
    container_name: q-beta-coordinator
    ports:
      - "8081:8081"
      - "8082:8080"
    environment:
      - RUST_LOG=info
      - Q_NODE_ID=beta-coordinator
      - Q_SERVER_ROLE=beta
      - Q_P2P_PORT=8081
      - Q_API_PORT=8080
      - Q_TOR_PROXY=tor-proxy:9050
      - Q_DNS_PHANTOM_DISCOVERY=true
    networks:
      - qnarwhal-net
    depends_on:
      - tor-proxy
      - dns-phantom-hub
    volumes:
      - ./target/release:/app/bin
      - ./docker/logs:/app/logs
    command: ["/app/bin/q-api-server", "--node-id", "beta-coordinator", "--p2p-port", "8081", "--api-port", "8080"]"""
    
    # Alpha nodes (10 nodes)
    alpha_nodes = ""
    for i in range(1, 11):
        alpha_nodes += f"""
  alpha-node-{i:02d}:
    build:
      context: .
      dockerfile: docker/Dockerfile.qnarwhal
    container_name: q-alpha-node-{i:02d}
    environment:
      - RUST_LOG=info
      - Q_NODE_ID=alpha-node-{i:02d}
      - Q_SERVER_ROLE=alpha
      - Q_BETA_TARGET=beta-coordinator:8081
      - Q_TOR_PROXY=tor-proxy:9050
      - Q_DNS_PHANTOM_DISCOVERY=true
      - Q_ALPHA_INDEX={i}
    networks:
      - qnarwhal-net
    depends_on:
      - beta-coordinator
      - dns-phantom-hub
    volumes:
      - ./target/release:/app/bin
      - ./docker/logs:/app/logs
    command: ["/app/bin/q-api-server", "--node-id", "alpha-node-{i:02d}", "--target-beta", "beta-coordinator:8081"]"""
    
    # Validator nodes (39 nodes - 50 total minus infrastructure + alpha + beta)
    validator_nodes = ""
    for i in range(1, 40):
        validator_nodes += generate_validator_node(i)
    
    # Monitoring and testing services
    monitoring_config = """
  # Monitoring and metrics services
  prometheus:
    image: prom/prometheus:latest
    container_name: q-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - qnarwhal-net
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    container_name: q-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=qnarwhal123
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./docker/grafana-dashboards:/var/lib/grafana/dashboards
      - grafana-data:/var/lib/grafana
    networks:
      - qnarwhal-net
    depends_on:
      - prometheus

  # Massive scale load testing service
  massive-scale-tester:
    build:
      context: .
      dockerfile: docker/Dockerfile.qnarwhal
    container_name: q-massive-scale-tester
    environment:
      - RUST_LOG=info
      - Q_LOAD_TEST_MODE=massive_scale
      - Q_TARGET_TPS=50000
      - Q_TEST_DURATION=600
      - Q_VALIDATOR_COUNT=50
    networks:
      - qnarwhal-net
    depends_on:
      - beta-coordinator
      - alpha-node-01
      - validator-01
    volumes:
      - ./target/release:/app/bin
      - ./docker/logs:/app/logs
    command: ["/app/bin/massive_scale_test", "--validators", "50", "--duration", "600", "--real-dht", "--real-tor"]

  # Network monitor and connection tracker
  network-monitor:
    build:
      context: .
      dockerfile: docker/Dockerfile.qnarwhal
    container_name: q-network-monitor
    environment:
      - RUST_LOG=info
      - Q_NODE_ID=network-monitor
      - Q_SERVER_ROLE=monitor
    networks:
      - qnarwhal-net
    depends_on:
      - beta-coordinator
    volumes:
      - ./target/release:/app/bin
      - ./docker/logs:/app/logs
    command: ["sh", "-c", "while true; do echo '[MONITOR] Active connections:' && netstat -an | grep ESTABLISHED | wc -l && echo '[MONITOR] DNS discoveries:' && grep -c 'DNS.*anomaly' /app/logs/*.log || echo 0 && sleep 30; done"]

networks:
  qnarwhal-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  prometheus-data:
  grafana-data:"""
    
    return base_config + alpha_nodes + validator_nodes + monitoring_config

def main():
    """Generate and save the complete Docker Compose configuration"""
    print("🚀 Generating 50-node Q-NarwhalKnight Docker Compose configuration...")
    
    compose_config = generate_complete_docker_compose()
    
    output_file = "/opt/orobit/shared/q-narwhalknight/docker-compose-full-50-nodes.yml"
    with open(output_file, 'w') as f:
        f.write(compose_config)
    
    print(f"✅ Generated complete 50-node configuration: {output_file}")
    print("🎯 Configuration includes:")
    print("   - 1 Tor proxy service")
    print("   - 1 DNS-Phantom hub")
    print("   - 1 Beta coordinator")
    print("   - 10 Alpha nodes")
    print("   - 39 Validator nodes")
    print("   - Prometheus + Grafana monitoring")
    print("   - Massive scale load tester")
    print("   - Network connection monitor")
    print("   = 54 total containers for comprehensive testing")
    print("")
    print("🚀 To deploy: docker-compose -f docker-compose-full-50-nodes.yml up -d")

if __name__ == "__main__":
    main()