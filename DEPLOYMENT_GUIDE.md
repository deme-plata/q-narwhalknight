# Q-NarwhalKnight PaaS SDK - Deployment Guide

## 🚀 Publishing SDKs to Public Registries

All three SDKs are built and ready for publication!

---

## ✅ Pre-Deployment Status

- ✅ **Python SDK**: Built successfully (8.6 KB wheel + 8.3 KB source)
- ✅ **JavaScript SDK**: Packaged successfully (8.3 KB tarball)
- ✅ **Rust SDK**: Package manifest valid

---

## 📦 1. PyPI (Python SDK)

### Quick Publish

```bash
cd /opt/orobit/shared/q-narwhalknight/sdk/python

# Install tools
pip install twine

# Build
python3 setup.py sdist bdist_wheel

# Publish
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_API_TOKEN
twine upload dist/*
```

**Result**: `pip install q-narwhalknight-paas` works worldwide

---

## 📦 2. npm (JavaScript SDK)

### Quick Publish

```bash
cd /opt/orobit/shared/q-narwhalknight/sdk/javascript

# Login
npm login

# Publish
npm publish --access public
```

**Result**: `npm install @q-narwhalknight/paas-sdk` works worldwide

---

## 📦 3. crates.io (Rust SDK)

### Quick Publish

```bash
cd /opt/orobit/shared/q-narwhalknight/sdk/rust

# Login
cargo login

# Publish (commit first or use --allow-dirty)
cargo publish --allow-dirty
```

**Result**: `cargo add q-paas-sdk` works worldwide

---

## 🔐 Get API Tokens

- **PyPI**: https://pypi.org/manage/account/token/
- **npm**: `npm login` (uses your account)
- **crates.io**: https://crates.io/settings/tokens

---

## ✅ Verification Commands

```bash
# Test PyPI package
pip install q-narwhalknight-paas
python -c "from q_paas import QNarwhalKnightPaaSClient; print('Works!')"

# Test npm package
npm install @q-narwhalknight/paas-sdk
node -e "const sdk = require('@q-narwhalknight/paas-sdk'); console.log('Works!')"

# Test crates.io package
cargo new test && cd test
cargo add q-paas-sdk --features full
cargo check
```

---

## 📊 After Publication

Add badges to README:

```markdown
[![PyPI](https://badge.fury.io/py/q-narwhalknight-paas.svg)](https://pypi.org/project/q-narwhalknight-paas/)
[![npm](https://badge.fury.io/js/@q-narwhalknight%2Fpaas-sdk.svg)](https://www.npmjs.com/package/@q-narwhalknight/paas-sdk)
[![Crates.io](https://img.shields.io/crates/v/q-paas-sdk.svg)](https://crates.io/crates/q-paas-sdk)
```

Users can now install with simple commands globally! 🚀
