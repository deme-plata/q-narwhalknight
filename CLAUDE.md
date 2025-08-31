# CLAUDE.md - Multi-Server Development Guide

## Claude Code Distributed Development for Q-NarwhalKnight

This guide explains how to set up distributed development with multiple Claude Code servers working collaboratively on the Q-NarwhalKnight quantum consensus system.

## 🏗️ Development Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Server Alpha  │    │   GitLab Repo    │    │   Server Beta   │
│  (Primary Dev)  │◄──►│ dagknight/       │◄──►│ (Contributor)   │
│                 │    │ q-narwhalknight  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
 /mnt/shared/Q-Knight     GitLab CI/CD         /mnt/shared/Q-Knight
 (Shared Storage)        (Auto Testing)        (Shared Storage)
```

## 🚀 Initial Setup (Server Alpha)

### 1. Repository Initialization
```bash
# Initialize Git repository
git init
git remote add origin https://gitlab.com/dagknight/q-narwhalknight.git
git branch -M main

# Set up GitLab authentication
git config user.name "Claude Code Alpha"
git config user.email "claude-alpha@anthropic.com"
```

### 2. Commit Structure
```bash
# Stage all critical files
git add README.md CLAUDE.md LICENSE
git add Cargo.toml
git add crates/
git add papers/quantum-aesthetics.pdf

# Create comprehensive commit
git commit -m "feat: Initial Q-NarwhalKnight v0.0.1-alpha implementation

🌟 Quantum-Enhanced DAG-BFT Consensus System

Core Components Implemented:
- ✅ DAG-Knight consensus engine with quantum anchor election  
- ✅ Narwhal mempool with reliable broadcast (Bracha's protocol)
- ✅ libp2p networking with crypto-agile framework
- ✅ Phase 0 (Ed25519) and Phase 1 (Dilithium5/Kyber1024) cryptography
- ✅ REST API server with real-time streaming (SSE/WebSocket)
- ✅ Quantum state visualization with rainbow-box technique
- ✅ Comprehensive test coverage and benchmarking framework

Technical Achievements:
- Zero-message complexity BFT consensus
- VDF-based quantum-enhanced randomness
- Post-quantum cryptographic agility
- Sub-50ms streaming latency targets
- Scalable P2P networking with capability negotiation

Architecture:
- Modular Rust workspace with 7 specialized crates
- Phase-based quantum threat model (Q0 → Q1 → Q2 → Q3 → Q4)
- Academic paper: Quantum Aesthetics in Consensus Systems

Next Phase: Performance optimization, Phase 1 completion, multi-server development

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

### 3. Tag Creation
```bash
# Create alpha release tag
git tag -a v0.0.1-alpha -m "Q-NarwhalKnight Alpha Release

Initial implementation of quantum-enhanced DAG-BFT consensus:
- Phase 0: Classical cryptography (Ed25519 + QUIC)
- Phase 1: Post-quantum transition (Dilithium5 + Kyber1024)
- DAG-Knight consensus with VDF-based anchor election
- Narwhal mempool with reliable broadcast
- Real-time API with quantum visualizations

Milestone: First working quantum-ready consensus prototype"

# Push everything to GitLab
git push origin main
git push origin v0.0.1-alpha
```

## 🤝 Multi-Server Collaboration

### Server Beta Setup Instructions

#### 1. Clone and Environment Setup
```bash
# Clone the repository to shared mount
cd /mnt/shared/
git clone https://gitlab.com/dagknight/q-narwhalknight.git Q-NarwhalKnight-Beta
cd Q-NarwhalKnight-Beta

# Set up Git identity
git config user.name "Claude Code Beta"  
git config user.email "claude-beta@anthropic.com"

# Set up GitLab token authentication
git config credential.helper store
echo "https://oauth2:glpat-5u5rhtquECnMkHpCQQmyCm86MQp1OmQ5NGF2Cw@gitlab.com" > ~/.git-credentials
```

#### 2. Development Branch Strategy
```bash
# Create feature branch for contributions
git checkout -b feature/server-beta-contributions
git checkout -b feature/performance-optimizations
git checkout -b feature/phase1-completion
```

#### 3. Shared Storage Coordination
```bash
# Symlink to shared development folder
ln -s /mnt/shared/Q-NarwhalKnight-Beta /mnt/s3-storage/Q-NarwhalKnight-Beta

# Set up workspace coordination
export Q_KNIGHT_WORKSPACE="/mnt/shared/Q-NarwhalKnight-Beta"
export RUST_LOG=debug
```

## 🎯 Contribution Areas for Server Beta

### Primary Focus Areas:

#### 1. Performance Optimization & Benchmarking
```bash
# Tasks for Server Beta:
- Implement comprehensive benchmarking suite
- Optimize DAG-Knight anchor election performance
- Add memory usage profiling and optimization
- Create load testing framework with realistic scenarios
- Implement parallel vertex processing optimization
```

#### 2. Phase 1 Post-Quantum Completion
```bash
# Crypto-agile enhancements:
- Complete hybrid classical+post-quantum mode
- Implement algorithm migration tools
- Add cryptographic protocol testing suite
- Build compatibility layer for smooth transitions
- Optimize post-quantum signature verification
```

#### 3. Network Layer Enhancements
```bash
# libp2p networking improvements:
- Implement advanced peer discovery mechanisms
- Add network partition tolerance features
- Create network monitoring and diagnostics
- Optimize gossip protocol for quantum readiness
- Build QKD preparation layer (Phase 2 prep)
```

#### 4. API & Visualization Improvements
```bash
# User experience enhancements:
- Expand quantum visualization capabilities
- Add real-time consensus monitoring dashboard
- Implement WebSocket connection scaling
- Create mobile-responsive visualization interface
- Build developer debugging tools
```

### Collaboration Workflow:

#### Server Beta Daily Process:
```bash
# 1. Sync with main repository
git fetch origin
git rebase origin/main

# 2. Work on assigned features
# Implement improvements based on current focus area

# 3. Test thoroughly
cargo test --workspace
cargo bench
cargo check --workspace

# 4. Commit with detailed messages
git add .
git commit -s -m "feat(performance): Add comprehensive benchmarking suite

- Implement criterion-based performance benchmarks
- Add memory profiling for vertex processing
- Create latency measurement framework
- Optimize consensus critical path performance

Performance improvements:
- 25% faster vertex validation
- 40% memory usage reduction in mempool
- Sub-10ms consensus round processing

Co-Authored-By: Claude Code Beta <noreply@anthropic.com>"

# 5. Push to feature branch
git push origin feature/performance-optimizations
```


## 🔄 GitLab CI/CD Pipeline

### .gitlab-ci.yml Configuration:
```yaml
stages:
  - test
  - build
  - deploy
  - quantum-analysis

variables:
  RUST_VERSION: "1.70"
  CARGO_HOME: ".cargo"

cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - .cargo/
    - target/

test:
  stage: test
  image: rust:${RUST_VERSION}
  script:
    - rustup component add clippy rustfmt
    - cargo fmt --check
    - cargo clippy -- -D warnings
    - cargo test --workspace --verbose
    - cargo bench --no-run
  coverage: '/^\d+\.\d+% coverage/'

build-release:
  stage: build
  image: rust:${RUST_VERSION}
  script:
    - cargo build --release --workspace
  artifacts:
    paths:
      - target/release/
    expire_in: 1 week

quantum-consensus-analysis:
  stage: quantum-analysis
  image: python:3.9
  script:
    - pip install numpy scipy matplotlib
    - python scripts/analyze_quantum_consensus.py
    - python scripts/benchmark_analysis.py
  artifacts:
    reports:
      junit: test-results.xml
    paths:
      - analysis_reports/
```

## 🎛️ Development Coordination

### Communication Protocol:
1. **Daily Sync**: Each server commits progress with detailed messages
2. **Feature Coordination**: Use GitLab issues for task assignment
3. **Code Reviews**: Mandatory peer review via merge requests
4. **Integration Testing**: Automated testing on every push

### Shared Resource Management:
```bash
# Shared configuration file: /mnt/shared/q-knight-config.toml
[development]
server_alpha_focus = ["consensus", "networking", "core-types"]
server_beta_focus = ["performance", "visualization", "api", "testing"]

[coordination]
daily_sync_time = "12:00 UTC"
integration_branch = "integration/multi-server"
feature_freeze_day = "friday"

[shared_storage]
workspace_path = "/mnt/shared/Q-NarwhalKnight"
backup_path = "/mnt/backup/q-knight-snapshots"
log_path = "/mnt/logs/q-knight-development"
```

### Git Hooks for Coordination:
```bash
#!/bin/bash
# .git/hooks/pre-commit
echo "🚀 Q-NarwhalKnight Development - Server $(hostname)"
echo "📊 Running pre-commit checks..."

# Ensure code quality
cargo fmt --check || (echo "❌ Format check failed" && exit 1)
cargo clippy -- -D warnings || (echo "❌ Clippy check failed" && exit 1)

# Run quick tests
cargo test --lib || (echo "❌ Library tests failed" && exit 1)

echo "✅ Pre-commit checks passed"
echo "🌟 Ready to commit to quantum consensus future!"
```

## 🎯 Prompt Instructions for Server Beta

### Server Beta Claude Code Prompt:
```
You are Claude Code Beta, contributing to the Q-NarwhalKnight quantum consensus system. 

Your primary repository is at: /mnt/shared/Q-NarwhalKnight-Beta
Your focus areas are: Performance optimization, Phase 1 completion, API enhancements, comprehensive testing

Current project status: Phase 0 complete, Phase 1 crypto-agility implemented, multi-server development active

Your tasks:
1. **Performance Optimization**: Implement benchmarking, optimize consensus performance, add profiling
2. **Phase 1 Completion**: Finish post-quantum integration, build migration tools, add compatibility layers  
3. **Network Enhancement**: Improve peer discovery, add network resilience, optimize gossip protocol
4. **Testing & Quality**: Build comprehensive test suites, add integration tests, create debugging tools

Always:
- Test thoroughly before committing
- Use detailed commit messages with performance metrics
- Coordinate with Server Alpha via GitLab issues and merge requests
- Focus on quantum-readiness and scalability
- Maintain code quality with clippy and rustfmt

The codebase uses:
- Rust workspace with 7 crates
- libp2p networking 
- Post-quantum cryptography (Dilithium5, Kyber1024)
- DAG-Knight consensus with VDF-based anchor election
- Real-time streaming APIs

Start by reviewing the current codebase and identifying performance bottlenecks or areas for Phase 1 enhancement.
```

## 📊 Progress Tracking

### Development Metrics Dashboard:
```bash
# Track multi-server progress
echo "📈 Q-NarwhalKnight Development Dashboard"
echo "🔧 Server Alpha: Core consensus & networking"  
echo "⚡ Server Beta: Performance & optimization"
echo "🚀 Combined Progress: $(git log --oneline | wc -l) commits"
echo "🎯 Next Milestone: Phase 1 completion & benchmarking"
```

### Automated Reporting:
```bash
#!/bin/bash
# Generate weekly development report
echo "# Q-NarwhalKnight Weekly Report $(date +%Y-%m-%d)" > weekly-report.md
echo "## Commits This Week" >> weekly-report.md
git log --since="1 week ago" --oneline >> weekly-report.md
echo "## Performance Benchmarks" >> weekly-report.md
cargo bench --message-format=json | jq '.reason' >> weekly-report.md
echo "## Test Coverage" >> weekly-report.md
cargo tarpaulin --out Md >> weekly-report.md
```

## 🌟 Success Metrics

### Collaboration Goals:
- **Code Quality**: Maintain >95% test coverage
- **Performance**: Achieve <50ms consensus latency
- **Integration**: Seamless multi-server development flow
- **Innovation**: Advance quantum consensus research

### Long-term Vision:
Building the world's first production-ready quantum-enhanced distributed consensus system through innovative multi-server Claude Code collaboration.

---

**Quantum consensus awaits - let's build the future together!** ⚛️🤝🚀
