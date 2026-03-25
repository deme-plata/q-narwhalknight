"""Configuration management for q-grover."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class NodeConfig:
    url: str = "https://quillon.xyz"
    fallback_urls: list[str] = field(default_factory=list)
    timeout: int = 30
    poll_interval: int = 2


@dataclass
class MinerConfig:
    address: str = ""
    worker_name: str = "q-grover-worker-1"
    classical_workers: int = 4
    max_nonces_per_challenge: int = 0


@dataclass
class OriginPilotConfig:
    api_url: str = ""
    api_token: str = ""
    qpu_name: str = ""
    qpu_preference: str = ""  # Preferred QPU (e.g., "wuyuan-72"); empty = auto-select
    job_timeout: int = 300    # Max seconds to wait for QPU job result


@dataclass
class QuantumConfig:
    backend: str = "simulator"
    prefix_qubits: int = 20
    shots: int = 1024
    grover_iterations: int = 0
    noise_adaptive: bool = True
    optimization_level: int = 1
    origin_pilot: OriginPilotConfig = field(default_factory=OriginPilotConfig)


@dataclass
class MEVConfig:
    enabled: bool = True
    commit_reveal: bool = True
    oracle_integrity_check: bool = True


@dataclass
class MetricsConfig:
    enabled: bool = True
    log_interval: int = 30
    track_advantage: bool = True


@dataclass
class LoggingConfig:
    level: str = "info"
    json: bool = False


@dataclass
class Config:
    node: NodeConfig = field(default_factory=NodeConfig)
    miner: MinerConfig = field(default_factory=MinerConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    mev: MEVConfig = field(default_factory=MEVConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _apply_dict(target, data: dict) -> None:
    """Apply dictionary values to a dataclass instance."""
    for key, value in data.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _apply_dict(current, value)
        else:
            setattr(target, key, value)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from TOML file.

    Search order:
    1. Explicit path argument
    2. Q_GROVER_CONFIG environment variable
    3. ./q-grover.toml
    4. ~/.config/q-grover/config.toml
    5. Default values
    """
    cfg = Config()

    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    if env := os.environ.get("Q_GROVER_CONFIG"):
        candidates.append(Path(env))
    candidates.append(Path("q-grover.toml"))
    candidates.append(Path.home() / ".config" / "q-grover" / "config.toml")

    for candidate in candidates:
        if candidate.is_file():
            with open(candidate, "rb") as f:
                data = tomllib.load(f)
            _apply_dict(cfg, data)
            break

    # Environment variable overrides
    if addr := os.environ.get("Q_GROVER_ADDRESS"):
        cfg.miner.address = addr
    if url := os.environ.get("Q_GROVER_NODE_URL"):
        cfg.node.url = url
    if backend := os.environ.get("Q_GROVER_BACKEND"):
        cfg.quantum.backend = backend
    if qubits := os.environ.get("Q_GROVER_QUBITS"):
        cfg.quantum.prefix_qubits = int(qubits)
    if level := os.environ.get("Q_GROVER_LOG_LEVEL"):
        cfg.logging.level = level

    return cfg


def validate_config(cfg: Config) -> list[str]:
    """Validate configuration, return list of error messages."""
    errors = []
    if not cfg.miner.address:
        errors.append("miner.address is required (set in config or Q_GROVER_ADDRESS env)")
    elif not cfg.miner.address.startswith("qnk") or len(cfg.miner.address) != 67:
        errors.append(
            f"miner.address must be 'qnk' + 64 hex chars (67 total), got {len(cfg.miner.address)} chars"
        )
    if cfg.quantum.prefix_qubits < 4 or cfg.quantum.prefix_qubits > 30:
        errors.append("quantum.prefix_qubits must be between 4 and 30")
    if cfg.quantum.shots < 1:
        errors.append("quantum.shots must be >= 1")
    if cfg.quantum.backend not in ("simulator", "origin_pilot", "local_qpu"):
        errors.append(f"quantum.backend must be simulator/origin_pilot/local_qpu, got {cfg.quantum.backend}")
    if cfg.quantum.backend == "origin_pilot":
        if not cfg.quantum.origin_pilot.api_url:
            errors.append(
                "quantum.origin_pilot.api_url is required when backend is 'origin_pilot' "
                "(set in config or use simulator backend)"
            )
        if not cfg.quantum.origin_pilot.api_token:
            errors.append(
                "quantum.origin_pilot.api_token is required when backend is 'origin_pilot'"
            )
    return errors
