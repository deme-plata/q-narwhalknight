#!/usr/bin/env python3
"""
MemeVerse Token Deployment Script for QUG DEX
Deploys 13 meme tokens with reflection + staking, then creates liquidity pools + nitro boosts.
Run this ON EPSILON (localhost:8080) for AIOC auth to work.
"""
import json
import time
import hashlib
import hmac
import struct
import urllib.request
import urllib.error
import sys

API_HOST = "http://localhost:8080"
MASTER_WALLET = "qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723"
AIOC_SECRET = "ef0ba4ad8fac73460ae74750b2d5d3e95a8f03444544d1574e93c2cf1cea727f"

# ============================================================================
# TOKEN DEFINITIONS
# name, symbol, supply, decimals, description, icon_url
# ============================================================================
TOKENS = [
    ("BORK", "BORK", "1000000000", 24,
     "Community-driven meme coin inspired by bork culture. Reflection rewards holders on every transaction.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f436.png"),
    ("YOLO", "YOLO", "42069000000", 8,
     "You Only Live Once — for the bold risk-takers. High supply, high energy, high rewards.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f680.png"),
    ("FOMO", "FOMO", "777777777", 24,
     "Fear of Missing Out — the token that keeps you ahead. Lucky 7s supply with reflection rewards.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f4b0.png"),
    ("STONK", "STONK", "100000000000", 8,
     "Stonks only go up! Inspired by the legendary meme. Massive supply, penny-stock vibes.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f4c8.png"),
    ("HODL", "HODL", "21000000", 24,
     "Diamond hands only. Bitcoin-inspired scarcity with the highest staking rewards.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f48e.png"),
    ("MOON", "MOON", "50000000000", 8,
     "To the Moon! Community-driven growth with reflection rewards for all holders.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f319.png"),
    ("DOGE2", "DOGE2", "420690000000", 8,
     "Next evolution of meme coins. Maximum supply for maximum fun. Such wow.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f415.png"),
    ("LOLZ", "LOLZ", "888888888", 24,
     "Laughter is the best currency. Lucky 8s supply with community-driven humor.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f602.png"),
    ("DERP", "DERP", "69420000000", 8,
     "Embracing the quirky and unexpected. Classic meme numbers, classic meme energy.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f92a.png"),
    ("WOJAK", "WOJAK", "500000000", 24,
     "The feels token. Premium mid-cap for those who know the highs and lows.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f614.png"),
    ("CHONK", "CHONK", "999999999999", 8,
     "Big, bold, and lovable. Nearly 1 trillion supply — chonky by design.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f431.png"),
    ("SMOL", "SMOL", "10000000", 24,
     "Small but mighty. Tiny supply makes every SMOL precious. Diamond hands rewarded.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f90f.png"),
    ("CHAD", "CHAD", "100000000", 24,
     "Alpha energy. Confidence, strength, bold moves. Premium scarcity for winners.",
     "https://cdn.jsdelivr.net/npm/emoji-datasource-apple@15.1.2/img/apple/64/1f4aa.png"),
]

# Pool configs: symbol, token_amount_display, qug_amount_display
POOLS = [
    ("BORK", 10000000, 100),
    ("YOLO", 420690000, 100),
    ("FOMO", 7777777, 77),
    ("STONK", 1000000000, 100),
    ("HODL", 210000, 210),
    ("MOON", 500000000, 50),
    ("DOGE2", 4206900000, 69),
    ("LOLZ", 8888888, 88),
    ("DERP", 694200000, 42),
    ("WOJAK", 5000000, 150),
    ("CHONK", 9999999999, 50),
    ("SMOL", 100000, 500),
    ("CHAD", 1000000, 300),
]

# ============================================================================
# AIOC HMAC Authentication
# ============================================================================
def generate_aioc_header(wallet_address: str) -> str:
    ts = int(time.time())
    # HMAC message: wallet_address string bytes + timestamp i64 LE bytes
    msg = wallet_address.encode('utf-8') + struct.pack('<q', ts)
    h = hmac.new(AIOC_SECRET.encode('utf-8'), msg, hashlib.sha256).hexdigest()
    return json.dumps({
        "service": "aioc",
        "wallet_address": wallet_address,
        "timestamp": ts,
        "hmac": h
    })

def api_call(path: str, body: dict, auth: bool = False) -> dict:
    url = f"{API_HOST}{path}"
    data = json.dumps(body).encode('utf-8')
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["X-AIOC-Service-Auth"] = generate_aioc_header(MASTER_WALLET)
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            return json.loads(body)
        except:
            return {"success": False, "error": f"HTTP {e.code}: {body[:200]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# Deploy a single token
# ============================================================================
def deploy_token(name: str, symbol: str, supply: str, decimals: int) -> str:
    body = {
        "contract_type": "advanced_token",
        "owner": MASTER_WALLET,
        "parameters": {
            "name": name,
            "symbol": symbol,
            "initial_supply": supply,
            "decimals": decimals,
            "reflection": True,
            "staking": True,
            "mintable": True,
            "burnable": True,
            "governance": False,
            "pausable": True
        }
    }
    resp = api_call("/api/v1/contracts/deploy", body, auth=True)
    if resp.get("success") and resp.get("data"):
        addr = resp["data"].get("contract_address", "")
        return addr
    else:
        print(f"  ERROR: {resp.get('error', 'unknown')}")
        return ""

# ============================================================================
# Set social profile (icon + description)
# ============================================================================
def set_social_profile(contract_addr: str, description: str, icon_url: str):
    body = {
        "logo_url": icon_url,
        "description": description,
        "website": "https://quillon.xyz",
        "twitter": "QuillonGraph",
        "owner_address": MASTER_WALLET
    }
    api_call(f"/api/v1/contracts/{contract_addr}/social", body)

# ============================================================================
# Remove liquidity from a pool
# ============================================================================
def remove_pool(pool_id: str) -> bool:
    body = {
        "pool_id": pool_id,
        "provider": MASTER_WALLET,
        "percentage": 100
    }
    resp = api_call("/api/v1/liquidity/remove", body)
    if resp.get("success"):
        return True
    else:
        print(f"  REMOVE ERROR: {resp.get('error', 'unknown')}")
        return False

# ============================================================================
# Create liquidity pool
# ============================================================================
def create_pool(symbol: str, token_addr: str, token_amount_base: str, qug_amount_base: str) -> str:
    body = {
        "token0": "QUG",
        "token1": token_addr,
        "amount0": qug_amount_base,
        "amount1": token_amount_base,
        "provider": MASTER_WALLET
    }
    resp = api_call("/api/v1/liquidity/add", body)
    if resp.get("success") and resp.get("data"):
        pool_id = resp["data"].get("pool_id", "unknown")
        return pool_id
    elif resp.get("pool_id"):
        return resp["pool_id"]
    else:
        print(f"  POOL ERROR: {resp.get('error', json.dumps(resp)[:200])}")
        return ""

# ============================================================================
# Nitro boost for top bar visibility
# ============================================================================
def nitro_boost(token_addr: str, symbol: str):
    body = {
        "token_id": token_addr,
        "points": 500,
        "wallet_address": MASTER_WALLET
    }
    resp = api_call("/api/v1/nitro/boost", body)
    if resp.get("success"):
        return True
    else:
        print(f"  BOOST WARN: {resp.get('error', 'unknown')}")
        return False

# ============================================================================
# Main
# ============================================================================
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "help"

    if mode == "help":
        print("MemeVerse Token Deployment Tool")
        print()
        print("Usage: python3 memeverse-deploy.py {test|deploy|pools|boost|full}")
        print()
        print("  test    — Deploy 1 test token + pool")
        print("  deploy  — Deploy all 13 meme tokens")
        print("  pools   — Create liquidity pools")
        print("  boost   — Nitro boost all tokens for top bar")
        print("  full    — Deploy + pools + boost")
        print("  status  — Show deployed token addresses")
        return

    # Load existing addresses
    addresses = {}
    try:
        with open("/home/orobit/memeverse-addresses.json", "r") as f:
            addresses = json.load(f)
    except:
        pass

    def save_addresses():
        with open("/home/orobit/memeverse-addresses.json", "w") as f:
            json.dump(addresses, f, indent=2)
        print(f"\n  Saved {len(addresses)} addresses to /home/orobit/memeverse-addresses.json")

    # ---- TEST ----
    if mode == "test":
        print("=" * 50)
        print("  MemeVerse TEST (1 token)")
        print("=" * 50)

        # Connectivity
        print("\n[TEST] Checking API...")
        try:
            req = urllib.request.Request(f"{API_HOST}/api/v1/status")
            resp = urllib.request.urlopen(req, timeout=5)
            print("[OK] API reachable")
        except Exception as e:
            print(f"[FAIL] API unreachable: {e}")
            return

        print("\n[TEST] Deploying TestMoon (TMOON)...")
        addr = deploy_token("TestMoon", "TMOON", "1000", 8)
        if addr:
            print(f"[OK] TMOON deployed: {addr}")
        else:
            print("[FAIL] Deploy failed")
            return

        print("\n[TEST] Creating pool TMOON/QUG...")
        token_base = str(100 * 10**8)  # 100 tokens
        qug_base = str(1 * 10**24)     # 1 QUG
        pool = create_pool("TMOON", addr, token_base, qug_base)
        if pool:
            print(f"[OK] Pool created: {pool}")

        print("\n[OK] TEST COMPLETE")
        return

    # ---- DEPLOY ----
    if mode in ("deploy", "full"):
        print("=" * 50)
        print("  MemeVerse Token Deployment (13 tokens)")
        print("=" * 50)

        deployed = 0
        failed = 0

        for name, symbol, supply, decimals, description, icon_url in TOKENS:
            # Skip if already deployed
            if symbol in addresses and addresses[symbol]:
                print(f"\n[SKIP] {symbol} already deployed: {addresses[symbol][:20]}...")
                deployed += 1
                continue

            print(f"\n[DEPLOY] {name} ({symbol}) — supply: {supply}, decimals: {decimals}")
            addr = deploy_token(name, symbol, supply, decimals)

            if addr:
                addresses[symbol] = addr
                print(f"[OK] {symbol} deployed: {addr}")

                # Set social profile
                time.sleep(1)
                set_social_profile(addr, description, icon_url)
                print(f"[OK] {symbol} social profile set")
                deployed += 1
            else:
                failed += 1

            time.sleep(2)

        save_addresses()
        print(f"\nDeployed: {deployed} | Failed: {failed}")

    # ---- POOLS ----
    if mode in ("pools", "full"):
        if mode == "full":
            print("\n[PAUSE] Waiting 5s before creating pools...")
            time.sleep(5)

        print("\n" + "=" * 50)
        print("  MemeVerse Pool Creation (13 pools)")
        print("=" * 50)

        created = 0
        failed = 0

        # Find decimals for each token
        token_decimals = {t[1]: t[3] for t in TOKENS}

        for symbol, token_display, qug_display in POOLS:
            addr = addresses.get(symbol, "")
            if not addr:
                print(f"\n[SKIP] {symbol}: no deployed address")
                failed += 1
                continue

            # AMM operates in 24-decimal space for ALL tokens
            # Always use 10^24 regardless of token's native decimals
            token_base = str(int(token_display * 10**24))
            qug_base = str(int(qug_display * 10**24))

            print(f"\n[POOL] {symbol}/QUG — {token_display:,} {symbol} + {qug_display} QUG")
            pool_id = create_pool(symbol, addr, token_base, qug_base)

            if pool_id:
                print(f"[OK] {symbol}/QUG pool: {pool_id}")
                created += 1
            else:
                failed += 1

            time.sleep(2)

        print(f"\nPools created: {created} | Failed: {failed}")

    # ---- BOOST ----
    if mode in ("boost", "full"):
        if mode == "full":
            print("\n[PAUSE] Waiting 3s before boosting...")
            time.sleep(3)

        print("\n" + "=" * 50)
        print("  MemeVerse Nitro Boost (Top Bar)")
        print("=" * 50)

        boosted = 0
        for symbol, addr in addresses.items():
            if not addr or symbol.startswith("T"):  # Skip test tokens
                continue
            print(f"\n[BOOST] {symbol}...")
            if nitro_boost(addr, symbol):
                print(f"[OK] {symbol} boosted — will appear in top bar")
                boosted += 1
            time.sleep(1)

        print(f"\nBoosted: {boosted} tokens")

    # ---- FIX-POOLS: Remove broken 8-decimal pools and recreate with 24-decimal amounts ----
    if mode == "fix-pools":
        print("=" * 50)
        print("  Fix 8-decimal token pools (remove + recreate)")
        print("=" * 50)

        # Broken pools (8-decimal tokens that were created with wrong base units)
        BROKEN_POOLS = {
            "YOLO": "pool-e08e980bc33b4932f2ab039026d4a07d",
            "STONK": "pool-458b4c1a557146d86297e3becf141230",
            "MOON": "pool-3d577c3ea6db637707a51d3b5050c90a",
            "DOGE2": "pool-c4349e12b9632dee3b33482b726a2a7c",
            "DERP": "pool-007db6b6b8cf7230809d29e5a0d9010c",
            "CHONK": "pool-3db56e6cce4cd726d900f784512087b0",
        }

        # Also remove test pools
        TEST_POOLS = {
            "TBORK": "pool-b08a42d23b65fef47e22c9fe3c42cd95",
        }

        print("\n--- Removing broken pools ---")
        for symbol, pool_id in {**BROKEN_POOLS, **TEST_POOLS}.items():
            print(f"\n[REMOVE] {symbol}/QUG pool: {pool_id}")
            if remove_pool(pool_id):
                print(f"[OK] {symbol}/QUG pool removed")
            else:
                print(f"[WARN] {symbol}/QUG removal failed (may already be empty)")
            time.sleep(1)

        print("\n--- Recreating pools with correct 24-decimal amounts ---")
        # Pool configs for 8-decimal tokens (amounts in display units)
        FIXED_POOLS = [
            ("YOLO", 420690000, 100),
            ("STONK", 1000000000, 100),
            ("MOON", 500000000, 50),
            ("DOGE2", 4206900000, 69),
            ("DERP", 694200000, 42),
            ("CHONK", 9999999999, 50),
        ]

        created = 0
        for symbol, token_display, qug_display in FIXED_POOLS:
            addr = addresses.get(symbol, "")
            if not addr:
                print(f"\n[SKIP] {symbol}: no deployed address")
                continue

            # ALL pool amounts must be in 24-decimal base units
            token_base = str(int(token_display * 10**24))
            qug_base = str(int(qug_display * 10**24))

            print(f"\n[POOL] {symbol}/QUG — {token_display:,} {symbol} + {qug_display} QUG (24-decimal)")
            pool_id = create_pool(symbol, addr, token_base, qug_base)
            if pool_id:
                print(f"[OK] {symbol}/QUG pool: {pool_id}")
                created += 1
            time.sleep(2)

        print(f"\nFixed pools: {created}")

    # ---- STATUS ----
    if mode == "status":
        print("MemeVerse Token Addresses:")
        print("-" * 80)
        for symbol, addr in sorted(addresses.items()):
            print(f"  {symbol:8s} {addr}")

if __name__ == "__main__":
    main()
