# Q-NarwhalKnight Windows Connection Test Script
# Test connectivity to Linux server at 185.182.185.227

Write-Host "🌐 Q-NarwhalKnight Connection Test" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

$LinuxServerIP = "185.182.185.227"
$APIPort = 8080
$P2PPort = 8081

Write-Host "📡 Testing connection to Linux server..." -ForegroundColor Yellow
Write-Host "   IP Address: $LinuxServerIP" -ForegroundColor Gray
Write-Host ""

# Test API port
Write-Host "🔍 Testing API port ($APIPort)..." -ForegroundColor Yellow
try {
    $result = Test-NetConnection -ComputerName $LinuxServerIP -Port $APIPort -WarningAction SilentlyContinue
    if ($result.TcpTestSucceeded) {
        Write-Host "   ✅ API port $APIPort is OPEN" -ForegroundColor Green

        # Try to fetch status
        Write-Host "   📊 Fetching server status..." -ForegroundColor Gray
        try {
            $status = Invoke-RestMethod -Uri "http://${LinuxServerIP}:${APIPort}/status" -TimeoutSec 5 -ErrorAction Stop
            Write-Host "   ✅ Successfully connected to API!" -ForegroundColor Green
            Write-Host "   Node ID: $($status.node_id)" -ForegroundColor Gray
        } catch {
            Write-Host "   ⚠️  Port is open but API request failed: $_" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   ❌ API port $APIPort is CLOSED" -ForegroundColor Red
    }
} catch {
    Write-Host "   ❌ Connection test failed: $_" -ForegroundColor Red
}

Write-Host ""

# Test P2P port
Write-Host "🔍 Testing P2P port ($P2PPort)..." -ForegroundColor Yellow
try {
    $result = Test-NetConnection -ComputerName $LinuxServerIP -Port $P2PPort -WarningAction SilentlyContinue
    if ($result.TcpTestSucceeded) {
        Write-Host "   ✅ P2P port $P2PPort is OPEN" -ForegroundColor Green
    } else {
        Write-Host "   ❌ P2P port $P2PPort is CLOSED" -ForegroundColor Red
        Write-Host "   💡 You may need to open this port on the Linux server firewall" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ❌ Connection test failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "📝 Summary:" -ForegroundColor Cyan
Write-Host "   - If both ports are open, you're ready to connect!" -ForegroundColor Gray
Write-Host "   - If P2P port is closed, run on Linux server:" -ForegroundColor Gray
Write-Host "     sudo ufw allow 8081/tcp" -ForegroundColor Gray
Write-Host "     # or" -ForegroundColor Gray
Write-Host "     sudo iptables -A INPUT -p tcp --dport 8081 -j ACCEPT" -ForegroundColor Gray
Write-Host ""

Write-Host "🚀 To connect your Windows node:" -ForegroundColor Cyan
Write-Host "   Currently, mDNS doesn't work across networks." -ForegroundColor Yellow
Write-Host "   You'll need bootstrap peer support to manually specify:" -ForegroundColor Yellow
Write-Host "   /ip4/$LinuxServerIP/tcp/$P2PPort/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG" -ForegroundColor Gray
Write-Host ""
Write-Host "   This feature needs to be implemented in the code." -ForegroundColor Yellow
Write-Host ""
