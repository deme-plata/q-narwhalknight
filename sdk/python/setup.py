#!/usr/bin/env python3
"""
Q-NarwhalKnight Privacy-as-a-Service Python SDK
Production-ready Bitcoin integration with proper UTXO management
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="q-narwhalknight-paas",
    version="4.0.0",
    author="Q-NarwhalKnight Team",
    author_email="developers@q-narwhalknight.io",
    description="Production-ready Privacy-as-a-Service SDK for Bitcoin and other blockchains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/q-narwhalknight/sdk",
    project_urls={
        "Documentation": "https://quillon.xyz/docs",
        "Source": "https://github.com/q-narwhalknight/sdk",
        "Tracker": "https://github.com/q-narwhalknight/sdk/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "cryptography>=41.0.0",
        "python-bitcoinlib>=0.12.0",
        "coincurve>=18.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "mypy>=1.5.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "q-paas=q_paas.cli:main",
        ],
    },
    keywords="privacy cryptocurrency bitcoin ethereum solana blockchain mixing tor differential-privacy",
    license="MIT",
)
