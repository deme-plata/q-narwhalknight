#!/bin/bash

# Q-NarwhalKnight Quantum Wallet Deployment Script
# Server: quantum.bitcoinoro.xyz (185.182.185.227)

set -e

echo "🌟 Deploying Q-NarwhalKnight Quantum Wallet..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
DOMAIN="quantum.bitcoinoro.xyz"
SERVER_IP="185.182.185.227"
PROJECT_DIR="/mnt/s3-storage/Q-NarwhalKnight/gui/quantum-wallet"
NGINX_SITES="/etc/nginx/sites-available"
NGINX_ENABLED="/etc/nginx/sites-enabled"

echo -e "${BLUE}📦 Installing dependencies...${NC}"
npm install

echo -e "${BLUE}🏗️ Building quantum wallet...${NC}"
npm run build

echo -e "${BLUE}🔧 Setting up nginx configuration...${NC}"
sudo cp nginx.conf ${NGINX_SITES}/quantum-wallet

# Enable the site
sudo ln -sf ${NGINX_SITES}/quantum-wallet ${NGINX_ENABLED}/

# Remove default nginx site if it exists
sudo rm -f ${NGINX_ENABLED}/default

echo -e "${BLUE}🔐 Setting up SSL certificate...${NC}"
sudo certbot --nginx -d ${DOMAIN} --non-interactive --agree-tos --email admin@bitcoinoro.xyz

echo -e "${BLUE}🔥 Configuring firewall...${NC}"
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 8080  # Q-NarwhalKnight node
sudo ufw allow 9090  # Prometheus metrics
sudo ufw --force enable

echo -e "${BLUE}🔄 Restarting nginx...${NC}"
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

echo -e "${BLUE}📊 Setting up log rotation...${NC}"
sudo tee /etc/logrotate.d/quantum-wallet > /dev/null << EOF
/var/log/nginx/quantum-wallet-*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    postrotate
        sudo systemctl reload nginx
    endscript
}
EOF

echo -e "${BLUE}🎯 Setting up systemd service for auto-deployment...${NC}"
sudo tee /etc/systemd/system/quantum-wallet-deploy.service > /dev/null << EOF
[Unit]
Description=Q-NarwhalKnight Quantum Wallet Auto Deploy
After=network.target

[Service]
Type=oneshot
User=root
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PROJECT_DIR}/deploy.sh
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable quantum-wallet-deploy.service

echo -e "${BLUE}📈 Setting up monitoring...${NC}"
# Create monitoring script
sudo tee /usr/local/bin/quantum-wallet-monitor.sh > /dev/null << 'EOF'
#!/bin/bash
# Monitor quantum wallet health

DOMAIN="quantum.bitcoinoro.xyz"
LOG_FILE="/var/log/quantum-wallet-monitor.log"

check_health() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local status=$(curl -s -o /dev/null -w "%{http_code}" https://${DOMAIN}/health)
    
    if [ "$status" = "200" ]; then
        echo "[$timestamp] ✅ Quantum Wallet: HEALTHY" >> $LOG_FILE
    else
        echo "[$timestamp] ❌ Quantum Wallet: UNHEALTHY (HTTP $status)" >> $LOG_FILE
        # Send alert (implement your notification system here)
    fi
}

check_health
EOF

sudo chmod +x /usr/local/bin/quantum-wallet-monitor.sh

# Add to crontab for monitoring every 5 minutes
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/quantum-wallet-monitor.sh") | sudo crontab -

echo -e "${GREEN}✅ Quantum Wallet deployment complete!${NC}"
echo -e "${YELLOW}🌟 Wallet is now live at: https://${DOMAIN}${NC}"
echo ""
echo -e "${BLUE}📊 Service Status:${NC}"
sudo systemctl status nginx --no-pager -l
echo ""
echo -e "${BLUE}🔍 SSL Certificate Status:${NC}"
sudo certbot certificates
echo ""
echo -e "${BLUE}🎯 Next Steps:${NC}"
echo "1. Visit https://${DOMAIN} to access your quantum wallet"
echo "2. Start Q-NarwhalKnight node on port 8080 for full functionality"  
echo "3. Monitor logs: sudo tail -f /var/log/nginx/quantum-wallet-*.log"
echo "4. Check health: curl https://${DOMAIN}/health"
echo ""
echo -e "${GREEN}🚀 Quantum consensus awaits! Welcome to the future of finance.${NC}"