# Q-NarwhalKnight Easy Mining Script (Windows)
# Version: 1.1.0

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$WalletAddress,

    [Parameter(Position=1)]
    [int]$Threads = 0,  # 0 = auto-detect

    [Parameter(Position=2)]
    [int]$Intensity = 10
)

$ErrorActionPreference = "Stop"

# Display banner
Write-Host @"

██████╗     ███╗   ██╗ █████╗ ██████╗ ██╗    ██╗██╗  ██╗ █████╗ ██╗
██╔═══██╗    ████╗  ██║██╔══██╗██╔══██╗██║    ██║██║  ██║██╔══██╗██║
██║   ██║    ██╔██╗ ██║███████║██████╔╝██║ █╗ ██║███████║███████║██║
██║▄▄ ██║    ██║╚██╗██║██╔══██║██╔══██╗██║███╗██║██╔══██║██╔══██║██║
╚██████╔╝    ██║ ╚████║██║  ██║██║  ██║╚███╔███╔╝██║  ██║██║  ██║███████╗
 ╚══▀▀═╝     ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
                    MINER v1.1.0 - OPTIMIZED FOR MAXIMUM PERFORMANCE

"@ -ForegroundColor Cyan

# Default configuration
$DEFAULT_SERVER = "http://185.182.185.227:8080/"

Write-Host "🚀 Starting Q-NarwhalKnight Miner" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  • Wallet:    $WalletAddress"
Write-Host "  • Server:    $DEFAULT_SERVER"
Write-Host "  • Threads:   $Threads (0 = auto-detect)"
Write-Host "  • Intensity: $Intensity (1-10)"
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
Write-Host ""

# Detect CPU info
try {
    $cpu = Get-WmiObject Win32_Processor
    Write-Host "💻 CPU Detected:" -ForegroundColor Yellow
    Write-Host "  • Model: $($cpu.Name)"
    Write-Host "  • Cores: $($cpu.NumberOfCores)"
    Write-Host "  • Threads: $($cpu.NumberOfLogicalProcessors)"
    Write-Host ""
} catch {
    Write-Host "⚠️  Could not detect CPU info" -ForegroundColor Yellow
}

# Performance recommendations
if ($Intensity -eq 10) {
    Write-Host "⚡ Running at MAXIMUM intensity (99% CPU usage)" -ForegroundColor Yellow
    Write-Host "   Close other applications for best performance" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "🔥 Starting miner... Press Ctrl+C to stop" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
Write-Host ""

# Start the miner
& ".\q-miner.exe" --mode solo --server $DEFAULT_SERVER --wallet $WalletAddress --threads $Threads --intensity $Intensity
