# Q-NarwhalKnight Windows Node Startup Script
# Connects to Linux server at 185.182.185.227

Write-Host "🌟 Q-NarwhalKnight Windows Node" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Set bootstrap peer to Linux server
$env:Q_BOOTSTRAP_PEERS = "/ip4/185.182.185.227/tcp/8081/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG"

Write-Host "📡 Bootstrap Peer Configured:" -ForegroundColor Yellow
Write-Host "   $env:Q_BOOTSTRAP_PEERS" -ForegroundColor Gray
Write-Host ""

Write-Host "🚀 Starting Windows node on port 8096..." -ForegroundColor Green
Write-Host ""

# Start the node
.\q-api-server.exe --port 8096
