@echo off
REM Q-NarwhalKnight Windows Node Startup Script
REM Connects to Linux server at 185.182.185.227

echo.
echo ========================================
echo   Q-NarwhalKnight Windows Node
echo ========================================
echo.

REM Set bootstrap peer to Linux server
set Q_BOOTSTRAP_PEERS=/ip4/185.182.185.227/tcp/8081/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG

echo Bootstrap Peer: %Q_BOOTSTRAP_PEERS%
echo.
echo Starting Windows node on port 8096...
echo.

REM Start the node
q-api-server.exe --port 8096

pause
