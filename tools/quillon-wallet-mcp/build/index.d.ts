#!/usr/bin/env node
/**
 * Quillon Wallet & Mining MCP Server
 *
 * Exposes wallet creation, balance checking, and mining setup to Claude Code.
 * No more 9-step security briefings — just "create a wallet" or "start mining".
 *
 * Tools:
 *   create_wallet    — Generate a new wallet, return address + mnemonic
 *   get_balance      — Check balance of any qnk address
 *   import_wallet    — Recover wallet from mnemonic phrase
 *   list_wallets     — List all wallets on this node
 *   send_qug         — Send QUG from one address to another
 *   setup_miner      — Download and configure the miner on Linux
 *   start_mining     — Start mining to a wallet address
 *   mining_status    — Check mining stats (hashrate, rewards, blocks)
 *   network_status   — Current network height, peers, block rate
 */
export {};
