# Q-NarwhalKnight Node - Windows Installation Script
# Version: 0.1.0-alpha
# Description: One-click installer for Q-NarwhalKnight quantum consensus node
# Usage: iwr -useb https://quillon.xyz/downloads/install-windows.ps1 | iex

$ErrorActionPreference = "Stop"

# Configuration
$InstallDir = "$env:ProgramFiles\Q-NarwhalKnight"
$DataDir = "$env:LOCALAPPDATA\Q-NarwhalKnight"
$BinaryName = "q-api-server.exe"
$DownloadUrl = "https://quillon.xyz/downloads/q-api-server-windows-x64.exe"
$ServiceName = "QNarwhalKnight"

# ASCII Art Banner
Write-Host @"
  ___    _   _                 _           _ _  __      _       _     _
 / _ \  | \ | | __ _ _ ____      _____  _| | |/ /_ __ (_) __ _| |__ | |_
| | | | |  \| |/ _` | '__\ \ /\ / / _ \| | | ' /| '_ \| |/ _` | '_ \| __|
| |_| | | |\  | (_| | |   \ V  V |  __/| | | . \| | | | | (_| | | | | |_
 \__\_\ |_| \_|\__,_|_|    \_/\_/ \___||_|_|_|\_|_| |_|_|\__, |_| |_|\__|
                                                          |___/
   Quantum Consensus Network - Node Installer v0.1.0-alpha

"@ -ForegroundColor Cyan

# Check if running as Administrator
$IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $IsAdmin) {
    Write-Host "⚠️  Not running as Administrator" -ForegroundColor Yellow
    Write-Host "   Installer will attempt to run without admin privileges" -ForegroundColor Yellow
    Write-Host "   For Windows Service installation, please run as Administrator" -ForegroundColor Yellow
    $InstallDir = "$env:LOCALAPPDATA\Programs\Q-NarwhalKnight"
}

# Detect system information
Write-Host "`n🔍 Detecting system information..." -ForegroundColor Cyan
$OSVersion = [System.Environment]::OSVersion.Version
$Architecture = [System.Environment]::Is64BitOperatingSystem

Write-Host "   OS Version: " -NoNewline
Write-Host "$($OSVersion.Major).$($OSVersion.Minor)" -ForegroundColor Green
Write-Host "   Architecture: " -NoNewline
Write-Host $(if ($Architecture) { "x64" } else { "x86" }) -ForegroundColor Green

# Check if x64
if (-not $Architecture) {
    Write-Host "`n❌ This installer only supports Windows x64" -ForegroundColor Red
    exit 1
}

# Check Windows version (10+ required)
if ($OSVersion.Major -lt 10) {
    Write-Host "`n❌ Windows 10 or higher is required" -ForegroundColor Red
    Write-Host "   Your version: Windows $($OSVersion.Major).$($OSVersion.Minor)" -ForegroundColor Yellow
    exit 1
}

# Create directories
Write-Host "`n📁 Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
Write-Host "   Install directory: " -NoNewline
Write-Host $InstallDir -ForegroundColor Green
Write-Host "   Data directory: " -NoNewline
Write-Host $DataDir -ForegroundColor Green

# Download binary
Write-Host "`n📥 Downloading Q-NarwhalKnight node binary..." -ForegroundColor Cyan
$TempFile = Join-Path $env:TEMP "q-api-server-temp.exe"

try {
    $ProgressPreference = 'Continue'
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $TempFile -UseBasicParsing
    Write-Host "✅ Download complete ($([math]::Round((Get-Item $TempFile).Length / 1MB, 2)) MB)" -ForegroundColor Green
} catch {
    Write-Host "`n❌ Download failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Verify download
if (-not (Test-Path $TempFile) -or (Get-Item $TempFile).Length -eq 0) {
    Write-Host "`n❌ Download failed or file is empty" -ForegroundColor Red
    exit 1
}

# Install binary
Write-Host "`n📦 Installing binary..." -ForegroundColor Cyan
$BinaryPath = Join-Path $InstallDir $BinaryName
Move-Item -Path $TempFile -Destination $BinaryPath -Force
Write-Host "✅ Binary installed to $BinaryPath" -ForegroundColor Green

# Add to PATH if not already there
$CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($CurrentPath -notlike "*$InstallDir*") {
    Write-Host "`n📝 Adding to PATH..." -ForegroundColor Cyan
    [Environment]::SetEnvironmentVariable("Path", "$CurrentPath;$InstallDir", "User")
    Write-Host "✅ Added $InstallDir to user PATH" -ForegroundColor Green
    Write-Host "   Restart your terminal for PATH changes to take effect" -ForegroundColor Yellow
}

# Configure Windows Firewall
Write-Host "`n🔥 Configuring Windows Firewall..." -ForegroundColor Cyan
try {
    if ($IsAdmin) {
        New-NetFirewallRule -DisplayName "Q-NarwhalKnight Node" -Direction Inbound -Program $BinaryPath -Action Allow -Protocol TCP -LocalPort 8080 -ErrorAction SilentlyContinue | Out-Null
        Write-Host "✅ Firewall rule added for port 8080" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Skipping firewall configuration (requires admin)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⚠️  Firewall configuration failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Create Windows Service (if admin)
if ($IsAdmin) {
    Write-Host "`n⚙️  Creating Windows Service..." -ForegroundColor Cyan

    # Check if service already exists
    $ExistingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($ExistingService) {
        Write-Host "   ⚠️  Service already exists, updating..." -ForegroundColor Yellow
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
        sc.exe delete $ServiceName | Out-Null
    }

    # Create service
    $ServiceCmd = "`"$BinaryPath`" --port 8080"
    New-Service -Name $ServiceName `
        -BinaryPathName $ServiceCmd `
        -DisplayName "Q-NarwhalKnight Quantum Consensus Node" `
        -Description "Q-NarwhalKnight quantum-resistant consensus node with Phase 1 post-quantum cryptography" `
        -StartupType Manual `
        -ErrorAction SilentlyContinue | Out-Null

    Write-Host "✅ Windows Service created" -ForegroundColor Green
    Write-Host "   Service Name: $ServiceName" -ForegroundColor Gray
}

# Create configuration file
Write-Host "`n⚙️  Creating configuration file..." -ForegroundColor Cyan
$ConfigPath = Join-Path $DataDir "config.toml"
$ConfigContent = @"
# Q-NarwhalKnight Node Configuration
# Edit this file to customize your node

[node]
port = 8080
node_id = "auto-generated"

[network]
max_peers = 50
enable_tor = true
enable_bitcoin_bridge = true
enable_dns_phantom = true

[consensus]
validator_mode = false  # Set to true to participate in consensus
enable_mining = false   # Set to true to enable GPU mining

[database]
path = "$($DataDir.Replace('\', '/'))/data"

[logging]
level = "info"  # Options: trace, debug, info, warn, error
"@

Set-Content -Path $ConfigPath -Value $ConfigContent
Write-Host "✅ Configuration created at $ConfigPath" -ForegroundColor Green

# Create desktop shortcut
Write-Host "`n🔗 Creating desktop shortcut..." -ForegroundColor Cyan
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Q-NarwhalKnight Node.lnk")
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-NoExit -Command `"cd '$DataDir'; & '$BinaryPath' --port 8080`""
$Shortcut.WorkingDirectory = $DataDir
$Shortcut.IconLocation = $BinaryPath
$Shortcut.Description = "Q-NarwhalKnight Quantum Consensus Node"
$Shortcut.Save()
Write-Host "✅ Desktop shortcut created" -ForegroundColor Green

# Display completion message
Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Green

if ($IsAdmin) {
    Write-Host "🚀 Quick Start (Windows Service):" -ForegroundColor Cyan
    Write-Host "   Start-Service $ServiceName              - Start the node" -ForegroundColor Yellow
    Write-Host "   Get-Service $ServiceName                - Check status" -ForegroundColor Yellow
    Write-Host "   Stop-Service $ServiceName               - Stop the node" -ForegroundColor Yellow
} else {
    Write-Host "🚀 Quick Start (Manual):" -ForegroundColor Cyan
    Write-Host "   & `"$BinaryPath`" --port 8080" -ForegroundColor Yellow
    Write-Host "`n   Or double-click the desktop shortcut!" -ForegroundColor Cyan
}

Write-Host "`n📝 Configuration:" -ForegroundColor Cyan
Write-Host "   notepad `"$ConfigPath`"" -ForegroundColor Yellow

Write-Host "`n🌐 Web Interface:" -ForegroundColor Cyan
Write-Host "   http://localhost:8080" -ForegroundColor Yellow

Write-Host "`n📊 API Documentation:" -ForegroundColor Cyan
Write-Host "   http://localhost:8080/api/v1/health" -ForegroundColor Yellow

Write-Host "`n💎 Wallet:" -ForegroundColor Cyan
Write-Host "   https://quillon.xyz" -ForegroundColor Yellow

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Magenta
Write-Host "⚛️  Join the quantum consensus revolution!" -ForegroundColor Magenta
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Magenta
