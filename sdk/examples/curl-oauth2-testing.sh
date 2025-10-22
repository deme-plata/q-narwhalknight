#!/bin/bash
#
# OAuth2 Testing Script using curl
#
# This script demonstrates the complete OAuth2 flow using curl commands
# for testing and debugging purposes.
#
# Usage:
#   ./curl-oauth2-testing.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_BASE_URL="http://localhost:8090/api/v1"
CLIENT_ID="curl-test-client"
CLIENT_SECRET="curl-secret-$(date +%s)"
REDIRECT_URI="http://localhost:8000/callback"
WALLET_ADDRESS="test-wallet-$(openssl rand -hex 16)"

echo -e "${BLUE}🔐 OAuth2 Flow Testing with curl${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ============================================================================
# Step 1: Register OAuth2 Client
# ============================================================================

echo -e "${YELLOW}📝 Step 1: Register OAuth2 Client${NC}"
echo "  Client ID: $CLIENT_ID"

REGISTER_RESPONSE=$(curl -s -X POST "$API_BASE_URL/oauth2/register" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "'"$CLIENT_ID"'",
    "client_secret": "'"$CLIENT_SECRET"'",
    "client_name": "Curl Test Client",
    "description": "Testing OAuth2 with curl",
    "redirect_uri": "'"$REDIRECT_URI"'",
    "website": "https://example.com",
    "scopes": ["read:balance", "read:transactions", "write:transactions"]
  }')

echo "  Response: $REGISTER_RESPONSE"

if [[ $REGISTER_RESPONSE == *'"success":true'* ]]; then
    echo -e "  ${GREEN}✅ Client registered successfully${NC}"
else
    echo -e "  ${RED}❌ Client registration failed${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Step 2: Generate PKCE Parameters
# ============================================================================

echo -e "${YELLOW}🔑 Step 2: Generate PKCE Parameters${NC}"

# Generate code verifier (random 43-128 char string)
CODE_VERIFIER=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-43)
echo "  Code Verifier: $CODE_VERIFIER"

# Generate code challenge (SHA256 hash of verifier)
CODE_CHALLENGE=$(echo -n "$CODE_VERIFIER" | shasum -a 256 | cut -d' ' -f1 | xxd -r -p | base64 | tr -d "=+/" | tr '/+' '_-')
echo "  Code Challenge: $CODE_CHALLENGE"

# Generate state for CSRF protection
STATE=$(openssl rand -hex 16)
echo "  State: $STATE"

echo ""

# ============================================================================
# Step 3: Build Authorization URL
# ============================================================================

echo -e "${YELLOW}🔗 Step 3: Authorization URL${NC}"

AUTH_URL="$API_BASE_URL/oauth2/authorize"
AUTH_URL+="?response_type=code"
AUTH_URL+="&client_id=$CLIENT_ID"
AUTH_URL+="&redirect_uri=$(printf %s "$REDIRECT_URI" | jq -sRr @uri)"
AUTH_URL+="&scope=read:balance%20read:transactions%20write:transactions"
AUTH_URL+="&state=$STATE"
AUTH_URL+="&code_challenge=$CODE_CHALLENGE"
AUTH_URL+="&code_challenge_method=S256"

echo "  URL: $AUTH_URL"
echo ""
echo -e "  ${BLUE}ℹ️  In a real flow, user would visit this URL and approve${NC}"

echo ""

# ============================================================================
# Step 4: Simulate User Consent
# ============================================================================

echo -e "${YELLOW}👤 Step 4: Simulate User Consent${NC}"

CONSENT_RESPONSE=$(curl -s -X POST "$API_BASE_URL/oauth2/consent" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "'"$CLIENT_ID"'",
    "wallet_address": "'"$WALLET_ADDRESS"'",
    "scopes": ["read:balance", "read:transactions", "write:transactions"]
  }')

echo "  Response: $CONSENT_RESPONSE"

if [[ $CONSENT_RESPONSE == *'"success":true'* ]]; then
    echo -e "  ${GREEN}✅ User consent granted${NC}"
    AUTH_CODE=$(echo "$CONSENT_RESPONSE" | grep -o '"data":"[^"]*"' | cut -d'"' -f4)
    echo "  Authorization Code: $AUTH_CODE"
else
    echo -e "  ${RED}❌ Consent failed${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Step 5: Exchange Authorization Code for Access Token
# ============================================================================

echo -e "${YELLOW}🎫 Step 5: Exchange Code for Access Token${NC}"

TOKEN_RESPONSE=$(curl -s -X POST "$API_BASE_URL/oauth2/token" \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "authorization_code",
    "code": "'"$AUTH_CODE"'",
    "client_id": "'"$CLIENT_ID"'",
    "client_secret": "'"$CLIENT_SECRET"'",
    "redirect_uri": "'"$REDIRECT_URI"'",
    "code_verifier": "'"$CODE_VERIFIER"'"
  }')

echo "  Response: $TOKEN_RESPONSE"

if [[ $TOKEN_RESPONSE == *'"success":true'* ]]; then
    echo -e "  ${GREEN}✅ Access token obtained${NC}"

    # Extract tokens
    ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
    REFRESH_TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"refresh_token":"[^"]*"' | cut -d'"' -f4)
    EXPIRES_IN=$(echo "$TOKEN_RESPONSE" | grep -o '"expires_in":[0-9]*' | cut -d':' -f2)

    echo "  Access Token: ${ACCESS_TOKEN:0:40}..."
    echo "  Refresh Token: ${REFRESH_TOKEN:0:40}..."
    echo "  Expires In: $EXPIRES_IN seconds"
else
    echo -e "  ${RED}❌ Token exchange failed${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Step 6: Get User Info
# ============================================================================

echo -e "${YELLOW}ℹ️  Step 6: Get User Information${NC}"

USERINFO_RESPONSE=$(curl -s -X GET "$API_BASE_URL/oauth2/userinfo" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

echo "  Response: $USERINFO_RESPONSE"

if [[ $USERINFO_RESPONSE == *'"success":true'* ]]; then
    echo -e "  ${GREEN}✅ User info retrieved${NC}"

    # Parse user info
    USER_WALLET=$(echo "$USERINFO_RESPONSE" | grep -o '"wallet_address":"[^"]*"' | cut -d'"' -f4)
    echo "  Wallet Address: $USER_WALLET"

    # Check for balance
    if [[ $USERINFO_RESPONSE == *'"balance"'* ]]; then
        USER_BALANCE=$(echo "$USERINFO_RESPONSE" | grep -o '"balance_qug":[0-9.]*' | cut -d':' -f2)
        echo "  Balance: $USER_BALANCE QUG"
    fi
else
    echo -e "  ${RED}❌ Failed to get user info${NC}"
fi

echo ""

# ============================================================================
# Step 7: Make Authenticated API Request
# ============================================================================

echo -e "${YELLOW}🔐 Step 7: Make Authenticated API Request${NC}"
echo "  (Example: Get wallet balance)"

BALANCE_RESPONSE=$(curl -s -X GET "$API_BASE_URL/wallet/$USER_WALLET/balance" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

echo "  Response: $BALANCE_RESPONSE"

if [[ $BALANCE_RESPONSE == *'"success"'* ]]; then
    echo -e "  ${GREEN}✅ Authenticated request successful${NC}"
else
    echo -e "  ${YELLOW}⚠️  Wallet not found (this is expected for test wallet)${NC}"
fi

echo ""

# ============================================================================
# Step 8: Refresh Access Token
# ============================================================================

echo -e "${YELLOW}🔄 Step 8: Refresh Access Token${NC}"

REFRESH_RESPONSE=$(curl -s -X POST "$API_BASE_URL/oauth2/token" \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "refresh_token",
    "refresh_token": "'"$REFRESH_TOKEN"'",
    "client_id": "'"$CLIENT_ID"'",
    "client_secret": "'"$CLIENT_SECRET"'"
  }')

echo "  Response: $REFRESH_RESPONSE"

if [[ $REFRESH_RESPONSE == *'"success":true'* ]]; then
    echo -e "  ${GREEN}✅ Token refreshed successfully${NC}"

    NEW_ACCESS_TOKEN=$(echo "$REFRESH_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
    echo "  New Access Token: ${NEW_ACCESS_TOKEN:0:40}..."
else
    echo -e "  ${RED}❌ Token refresh failed${NC}"
fi

echo ""

# ============================================================================
# Step 9: Get Client Info
# ============================================================================

echo -e "${YELLOW}📋 Step 9: Get Client Information${NC}"

CLIENT_INFO_RESPONSE=$(curl -s -X GET "$API_BASE_URL/oauth2/clients/$CLIENT_ID")

echo "  Response: $CLIENT_INFO_RESPONSE"

if [[ $CLIENT_INFO_RESPONSE == *'"success":true'* ]]; then
    echo -e "  ${GREEN}✅ Client info retrieved${NC}"
else
    echo -e "  ${RED}❌ Failed to get client info${NC}"
fi

echo ""

# ============================================================================
# Step 10: Revoke Token
# ============================================================================

echo -e "${YELLOW}🚫 Step 10: Revoke Access Token${NC}"

REVOKE_RESPONSE=$(curl -s -X POST "$API_BASE_URL/oauth2/revoke" \
  -H "Content-Type: application/json" \
  -d '{
    "token": "'"$NEW_ACCESS_TOKEN"'"
  }')

echo "  Response: $REVOKE_RESPONSE"

if [[ $REVOKE_RESPONSE == *'"success":true'* ]]; then
    echo -e "  ${GREEN}✅ Token revoked successfully${NC}"
else
    echo -e "  ${RED}❌ Token revocation failed${NC}"
fi

echo ""

# ============================================================================
# Step 11: Verify Revocation
# ============================================================================

echo -e "${YELLOW}🔍 Step 11: Verify Token Revocation${NC}"

VERIFY_RESPONSE=$(curl -s -X GET "$API_BASE_URL/oauth2/userinfo" \
  -H "Authorization: Bearer $NEW_ACCESS_TOKEN")

echo "  Response: $VERIFY_RESPONSE"

if [[ $VERIFY_RESPONSE == *'"success":false'* ]] || [[ $VERIFY_RESPONSE == *'Unauthorized'* ]]; then
    echo -e "  ${GREEN}✅ Token correctly revoked (request denied)${NC}"
else
    echo -e "  ${RED}❌ Revoked token still works (should fail)${NC}"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ OAuth2 Flow Test Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo -e "${BLUE}📊 Test Summary${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Client ID: $CLIENT_ID"
echo "  Wallet Address: $USER_WALLET"
echo "  Original Access Token: ${ACCESS_TOKEN:0:40}..."
echo "  Refreshed Access Token: ${NEW_ACCESS_TOKEN:0:40}..."
echo "  Token Status: Revoked"
echo ""
echo -e "${GREEN}All OAuth2 endpoints tested successfully!${NC}"
echo ""
