"""Command-line interface for q-grover quantum miner."""

from __future__ import annotations

import sys

import click
import structlog

from . import __version__
from .config import load_config, validate_config
from .vdf import compute_hashrate


def setup_logging(level: str = "info", json: bool = False) -> None:
    """Configure structured logging."""
    processors: list = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    if json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


@click.group()
@click.version_option(__version__, prog_name="q-grover")
def main():
    """q-grover: Quantum Grover miner for Q-NarwhalKnight.

    Mine QUG tokens using Grover's algorithm on QPanda / Origin Pilot OS.
    """


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.option("--node", default=None, help="QNK node URL")
@click.option("--address", default=None, help="Miner wallet address (qnk...)")
@click.option("--qubits", default=None, type=int, help="Prefix qubits for Grover")
@click.option("--backend", default=None, help="Quantum backend (simulator/origin_pilot/local_qpu)")
@click.option("--classical", is_flag=True, help="Force classical-only mode (no quantum)")
@click.option("--log-level", default=None, help="Log level (debug/info/warning/error)")
def mine(config_path, node, address, qubits, backend, classical, log_level):
    """Start quantum mining.

    Connects to a QNK node and mines blocks using Grover's algorithm.

    \b
    Examples:
      q-grover mine --address qnk1234...
      q-grover mine --node https://quillon.xyz --qubits 16
      q-grover mine --classical  # Force classical mode
    """
    cfg = load_config(config_path)

    # CLI overrides
    if node:
        cfg.node.url = node
    if address:
        cfg.miner.address = address
    if qubits:
        cfg.quantum.prefix_qubits = qubits
    if backend:
        cfg.quantum.backend = backend
    if classical:
        cfg.quantum.backend = "simulator"
        cfg.quantum.prefix_qubits = 0  # Disable Grover
    if log_level:
        cfg.logging.level = log_level

    setup_logging(cfg.logging.level, cfg.logging.json)
    logger = structlog.get_logger()

    # Validate
    errors = validate_config(cfg)
    if errors:
        for err in errors:
            logger.error("config_error", message=err)
        sys.exit(1)

    # Print banner
    _print_banner(cfg)

    # Start mining
    from .miner import QuantumMiner
    miner = QuantumMiner(cfg)
    miner.start()


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.option("--duration", default=5.0, type=float, help="Benchmark duration (seconds)")
def benchmark(config_path, duration):
    """Benchmark classical hash rate.

    Measures how many BLAKE3 VDF computations per second
    your hardware can perform. Useful for comparing against
    quantum speedup.
    """
    setup_logging("info")
    logger = structlog.get_logger()

    logger.info("benchmark_starting", duration=f"{duration}s")

    # Use a dummy challenge for benchmarking
    dummy_challenge = bytes(32)
    rate = compute_hashrate(dummy_challenge, duration)

    logger.info(
        "benchmark_result",
        hashrate=f"{rate:.1f} H/s",
        hashrate_kh=f"{rate / 1000:.3f} KH/s",
    )

    # Estimate quantum advantage
    from .difficulty import quantum_advantage_ratio
    for qubits in [12, 16, 20, 24]:
        advantage = quantum_advantage_ratio(qubits)
        effective_rate = rate * advantage
        logger.info(
            f"quantum_{qubits}q",
            advantage=f"{advantage:.0f}x",
            effective_hashrate=f"{effective_rate:.0f} H/s",
            effective_kh=f"{effective_rate / 1000:.1f} KH/s",
        )


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config file")
@click.option("--node", default=None, help="QNK node URL")
def status(config_path, node):
    """Check QNK node status and current mining challenge."""
    cfg = load_config(config_path)
    if node:
        cfg.node.url = node

    setup_logging("info")
    logger = structlog.get_logger()

    from .api_client import QNKClient
    from .difficulty import leading_zero_bits, difficulty_from_target, search_space_size

    client = QNKClient(base_url=cfg.node.url)
    try:
        challenge = client.get_challenge()
        target_bytes = challenge.target_bytes
        zero_bits = leading_zero_bits(target_bytes)
        difficulty = difficulty_from_target(target_bytes)
        search_size = search_space_size(target_bytes)

        logger.info(
            "node_status",
            url=cfg.node.url,
            version=challenge.server_version,
            height=challenge.block_height,
            reward=f"{challenge.block_reward} QUG",
            difficulty=f"{difficulty:.0f}",
            leading_zeros=zero_bits,
            search_space=f"2^{zero_bits} = {search_size:,}",
            vdf_iterations=challenge.vdf_iterations,
            network_hashrate=challenge.network_hashrate_hs,
            miners=challenge.connected_miners,
        )
    except Exception as e:
        logger.error("status_failed", error=str(e))
        sys.exit(1)
    finally:
        client.close()


def _print_banner(cfg) -> None:
    """Print startup banner."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel(
            f"[bold cyan]q-grover[/] v{__version__}\n"
            f"Quantum Grover Miner for Q-NarwhalKnight\n\n"
            f"Node:    {cfg.node.url}\n"
            f"Address: {cfg.miner.address[:16]}...{cfg.miner.address[-8:]}\n"
            f"Backend: {cfg.quantum.backend}\n"
            f"Qubits:  {cfg.quantum.prefix_qubits}",
            title="[bold]Quantum Mining[/]",
            border_style="cyan",
        ))
    except ImportError:
        print(f"q-grover v{__version__} — Quantum Grover Miner")
        print(f"Node: {cfg.node.url}")
        print(f"Address: {cfg.miner.address[:16]}...")
        print()


if __name__ == "__main__":
    main()
