#!/usr/bin/env node
// drift-check.mjs — flux-api drift-guard (A), checker half.
// Compares THREE views of the Quillon API surface and flags drift:
//   1. manifest.json        — the declared contract
//   2. ../src/index.ts      — what the WORKING MCP actually calls
//   3. the live server      — what /api/v1 actually answers (optional, --live)
// Exit code is non-zero on drift, so it can gate CI. Reads only; mutates nothing.
//
// Usage:  node drift-check.mjs           # manifest vs MCP source
//         node drift-check.mjs --live    # also probe the live server
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const manifest = JSON.parse(fs.readFileSync(path.join(HERE, 'manifest.json'), 'utf8'))
const base = manifest.base_url || 'https://quillon.xyz/api/v1'

const norm = p => p
  .replace(/\$\{[^}]*\}/g, '{param}')   // template literals → {param}
  .replace(/\{[^}]+\}/g, '{param}')     // any {x} → {param}
  .replace(/[\\"'`\s].*$/, '')          // cut at first quote/space/newline noise
  .replace(/\/+$/, '')                  // trailing slash

// 1. manifest paths (relative to /api/v1) → absolute form for comparison
const manifestSet = new Set(manifest.endpoints.map(e => norm('/api/v1' + e.path)))

// 2. extract every /api/.. and /capi/.. path the MCP source references
const src = fs.readFileSync(path.join(HERE, '..', 'src', 'index.ts'), 'utf8')
const found = new Set()
for (const m of src.matchAll(/(\/(?:api|capi)\/[A-Za-z0-9/_{}.$-]+)/g)) {
  const n = norm(m[1])
  // keep only real endpoints: at least /api/v1/<segment>
  if (/^\/api\/v1\/[a-z]/i.test(n) || /^\/capi\//.test(n)) found.add(n)
}

const inMcpNotManifest = [...found].filter(p => !manifestSet.has(p) && !/^\/capi\//.test(p)).sort()
const inManifestNotMcp = [...manifestSet].filter(p => !found.has(p)).sort()

let drift = 0
console.log(`\n=== flux-api drift-guard — Quillon Wallet MCP ===`)
console.log(`manifest endpoints: ${manifestSet.size}   |   MCP-referenced /api paths: ${found.size}`)

if (inMcpNotManifest.length) {
  drift += inMcpNotManifest.length
  console.log(`\n⚠  MCP calls these, but manifest does NOT cover them (add to manifest):`)
  inMcpNotManifest.forEach(p => console.log(`   - ${p}`))
} else console.log(`\n✓ Every MCP-called endpoint is covered by the manifest.`)

if (inManifestNotMcp.length) {
  console.log(`\nℹ  In manifest but not seen in MCP source (stale or called dynamically):`)
  inManifestNotMcp.forEach(p => console.log(`   - ${p}`))
}

if (process.argv.includes('--live')) {
  console.log(`\n=== live server probe (${base}) ===`)
  for (const ep of manifest.endpoints) {
    if (ep.path.includes('{')) continue              // skip path-param endpoints
    if ((ep.method || 'GET') !== 'GET') continue      // only safe to probe GETs
    const url = base + ep.path
    try {
      const r = await fetch(url, { method: 'GET' })
      const bad = r.status >= 500 || r.status === 404
      if (bad) { drift++; console.log(`   ✗ ${r.status}  ${ep.path}`) }
      else console.log(`   ✓ ${r.status}  ${ep.path}`)
    } catch (e) {
      drift++; console.log(`   ✗ ERR  ${ep.path}  (${e.message})`)
    }
  }
}

console.log(`\n${drift ? `❌ DRIFT: ${drift} issue(s)` : '✅ No drift'}\n`)
process.exit(drift ? 1 : 0)
