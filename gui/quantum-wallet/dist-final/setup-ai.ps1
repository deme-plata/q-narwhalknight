# Quillon Graph — AI Wallet & Mining Setup (PowerShell)
# Run from PowerShell on Windows:
#   irm https://quillon.xyz/setup-ai.ps1 | iex
#
# Auto-installs Node.js if missing, then configures Quillon MCP for Cursor
# and/or Claude Code (whichever it finds). No bash, no WSL, no Git Bash needed.

$ErrorActionPreference = 'Stop'

Write-Host ""
Write-Host "  Quillon Graph - AI Wallet & Mining Setup (Windows)"
Write-Host "  ==================================================="
Write-Host ""

# ---------------------------------------------------------------------------
# 1. Node.js — install if missing
# ---------------------------------------------------------------------------
function Get-NodeVersion {
    try { return (& node --version 2>$null) } catch { return $null }
}

$nodeVersion = Get-NodeVersion
if (-not $nodeVersion) {
    Write-Host "  Node.js not found. Installing via winget..."
    # winget is built-in on Windows 10 2004+ / Windows 11
    $wingetOk = $false
    try {
        & winget install --id OpenJS.NodeJS.LTS --silent --accept-source-agreements --accept-package-agreements
        $wingetOk = $true
    } catch {
        Write-Host "  winget failed: $_"
    }

    if (-not $wingetOk) {
        # Fallback: direct MSI download
        $arch = if ([Environment]::Is64BitOperatingSystem) { "x64" } else { "x86" }
        $msiUrl = "https://nodejs.org/dist/v22.11.0/node-v22.11.0-$arch.msi"
        $msiPath = "$env:TEMP\nodejs-installer.msi"
        Write-Host "  Downloading Node.js installer from $msiUrl..."
        Invoke-WebRequest -Uri $msiUrl -OutFile $msiPath -UseBasicParsing
        Write-Host "  Running installer (silent, may take a minute)..."
        Start-Process msiexec.exe -ArgumentList "/i `"$msiPath`" /qn /norestart" -Wait
        Remove-Item $msiPath -ErrorAction SilentlyContinue
    }

    # PATH won't be picked up by the current shell — refresh from registry
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

    $nodeVersion = Get-NodeVersion
    if (-not $nodeVersion) {
        Write-Host ""
        Write-Host "  Node.js installation didn't expose 'node' in this shell."
        Write-Host "  Close this PowerShell window, open a NEW one, and re-run:"
        Write-Host "    irm https://quillon.xyz/setup-ai.ps1 | iex"
        exit 1
    }
}
Write-Host "  [+] Node.js $nodeVersion"

# ---------------------------------------------------------------------------
# 2. Detect AI clients (Cursor, Claude Code)
# ---------------------------------------------------------------------------
$hasCursor = (Test-Path "$env:USERPROFILE\.cursor") -or (Get-Command cursor -ErrorAction SilentlyContinue)
$hasClaude = (Get-Command claude -ErrorAction SilentlyContinue) -ne $null

if (-not $hasCursor -and -not $hasClaude) {
    Write-Host ""
    Write-Host "  No supported AI client found."
    Write-Host "  Install ONE of:"
    Write-Host "    Cursor:       https://cursor.sh"
    Write-Host "    Claude Code:  npm install -g @anthropic-ai/claude-code"
    Write-Host "  Then re-run this script."
    exit 1
}
if ($hasCursor) { Write-Host "  [+] Cursor found" }
if ($hasClaude) { Write-Host "  [+] Claude Code found" }

# ---------------------------------------------------------------------------
# 3. Install Quillon MCP server
# ---------------------------------------------------------------------------
$installDir = Join-Path $env:USERPROFILE ".quillon\mcp"
New-Item -ItemType Directory -Force -Path $installDir | Out-Null

Write-Host "  Downloading Quillon AI tools..."
$tarUrl  = "https://quillon.xyz/downloads/quillon-wallet-mcp.tar.gz"
$tarPath = "$env:TEMP\quillon-mcp.tar.gz"
$gotTar  = $false
try {
    Invoke-WebRequest -Uri $tarUrl -OutFile $tarPath -UseBasicParsing
    $gotTar = $true
} catch {
    Write-Host "  Tarball not available; using inline fallback."
}

if ($gotTar) {
    # tar.exe is built into Windows 10 1803+
    & tar -xzf $tarPath -C $installDir
    Remove-Item $tarPath -ErrorAction SilentlyContinue
    Push-Location $installDir
    & npm install --production 2>$null | Out-Null
    Pop-Location
} else {
    New-Item -ItemType Directory -Force -Path "$installDir\build" | Out-Null
    Set-Content -Path "$installDir\package.json" -Value '{"name":"quillon-wallet-mcp","version":"1.0.0","type":"module","main":"build/index.js","dependencies":{"@modelcontextprotocol/sdk":"^1.12.1"}}'
    Push-Location $installDir
    & npm install --production 2>$null | Out-Null
    Pop-Location
    try {
        Invoke-WebRequest -Uri "https://quillon.xyz/downloads/quillon-mcp-index.js" -OutFile "$installDir\build\index.js" -UseBasicParsing
    } catch {
        Write-Host "  Could not download MCP server. Check https://quillon.xyz/downloads/"
        exit 1
    }
}
Write-Host "  [+] Quillon AI tools installed at $installDir"

# ---------------------------------------------------------------------------
# 4. Configure clients
# ---------------------------------------------------------------------------
$mcpIndexFwd = ($installDir + "\build\index.js").Replace('\', '/')

function Update-McpConfig($path) {
    $dir = Split-Path $path -Parent
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $cfg = @{}
    if (Test-Path $path) {
        try { $cfg = Get-Content $path -Raw | ConvertFrom-Json -AsHashtable } catch { $cfg = @{} }
    }
    if (-not $cfg.mcpServers) { $cfg.mcpServers = @{} }
    $cfg.mcpServers['quillon-wallet'] = @{
        command = "node"
        args = @($mcpIndexFwd)
        env = @{ QUILLON_API_URL = "https://quillon.xyz/api/v1" }
    }
    $cfg | ConvertTo-Json -Depth 6 | Set-Content -Path $path -Encoding UTF8
}

if ($hasCursor) {
    $cursorMcp = "$env:USERPROFILE\.cursor\mcp.json"
    Update-McpConfig $cursorMcp
    Write-Host "  [+] Cursor configured at $cursorMcp"
    Write-Host "      -> Reload the Cursor window (Ctrl+Shift+P -> Reload Window)"
}
if ($hasClaude) {
    $claudeSettings = "$env:USERPROFILE\.claude\settings.json"
    Update-McpConfig $claudeSettings
    Write-Host "  [+] Claude Code configured at $claudeSettings"
}

Write-Host ""
Write-Host "  ==================================================="
Write-Host "         Setup Complete!"
Write-Host "  ==================================================="
Write-Host ""
if ($hasCursor) {
    Write-Host "  In Cursor (Agent mode, after reload):"
    Write-Host "    'What Quillon tools are available?'"
    Write-Host "    'Show my QUG balance'"
    Write-Host "    'Check node sync status'"
    Write-Host ""
}
if ($hasClaude) {
    Write-Host "  In Claude Code:"
    Write-Host "    'Create a wallet'"
    Write-Host "    'Start mining'"
    Write-Host "    'What's the network status?'"
    Write-Host ""
}
Write-Host "  quillon.xyz | Post-Quantum Electronic Cash"
Write-Host ""
