#!/usr/bin/env python3
"""
Quillon Wallet OAuth2 Integration Example (Python + FastAPI)

This example demonstrates how to integrate Quillon Wallet OAuth2
authentication in a Python web application using FastAPI.

Requirements:
    pip install fastapi uvicorn httpx pydantic

Usage:
    python python-oauth2-example.py

Then visit http://localhost:8000 in your browser.
"""

import hashlib
import secrets
import base64
from typing import Optional
from datetime import datetime, timedelta

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel


# ============================================================================
# OAuth2 Configuration
# ============================================================================

class OAuth2Config:
    CLIENT_ID = "python-demo-app"
    CLIENT_SECRET = "python-secret-key-67890"
    REDIRECT_URI = "http://localhost:8000/callback"
    API_BASE_URL = "http://localhost:8090/api/v1"
    SCOPES = ["read:balance", "read:transactions", "write:transactions"]

    # OAuth2 endpoints
    AUTHORIZE_URL = f"{API_BASE_URL}/oauth2/authorize"
    TOKEN_URL = f"{API_BASE_URL}/oauth2/token"
    USERINFO_URL = f"{API_BASE_URL}/oauth2/userinfo"
    REVOKE_URL = f"{API_BASE_URL}/oauth2/revoke"


# ============================================================================
# Data Models
# ============================================================================

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None
    scope: str


class UserInfo(BaseModel):
    sub: str
    wallet_address: str
    balance: Optional[int] = None
    balance_qug: Optional[float] = None


# ============================================================================
# OAuth2 Client
# ============================================================================

class QuillonOAuth2Client:
    """OAuth2 client for Quillon Wallet integration"""

    def __init__(self):
        self.config = OAuth2Config()
        self.token: Optional[TokenResponse] = None
        self.code_verifier: Optional[str] = None

    def generate_pkce_pair(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge"""
        # Generate random code verifier (43-128 characters)
        self.code_verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')

        # Generate code challenge (SHA256 hash of verifier)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(self.code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')

        return self.code_verifier, code_challenge

    def get_authorization_url(self, state: str) -> str:
        """Build OAuth2 authorization URL with PKCE"""
        code_verifier, code_challenge = self.generate_pkce_pair()

        params = {
            "response_type": "code",
            "client_id": self.config.CLIENT_ID,
            "redirect_uri": self.config.REDIRECT_URI,
            "scope": " ".join(self.config.SCOPES),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256"
        }

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.config.AUTHORIZE_URL}?{query_string}"

    async def exchange_code_for_token(self, code: str) -> TokenResponse:
        """Exchange authorization code for access token"""
        if not self.code_verifier:
            raise ValueError("No code verifier found")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.config.TOKEN_URL,
                json={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": self.config.CLIENT_ID,
                    "client_secret": self.config.CLIENT_SECRET,
                    "redirect_uri": self.config.REDIRECT_URI,
                    "code_verifier": self.code_verifier
                }
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Token exchange failed: {response.text}"
                )

            data = response.json()
            if not data.get("success"):
                raise HTTPException(
                    status_code=400,
                    detail=data.get("error", "Token exchange failed")
                )

            self.token = TokenResponse(**data["data"])
            return self.token

    async def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.config.TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self.config.CLIENT_ID,
                    "client_secret": self.config.CLIENT_SECRET
                }
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Token refresh failed"
                )

            data = response.json()
            if not data.get("success"):
                raise HTTPException(
                    status_code=400,
                    detail=data.get("error", "Token refresh failed")
                )

            self.token = TokenResponse(**data["data"])
            return self.token

    async def get_user_info(self, access_token: str) -> UserInfo:
        """Fetch user information using access token"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.config.USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"}
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to fetch user info"
                )

            data = response.json()
            if not data.get("success"):
                raise HTTPException(
                    status_code=400,
                    detail=data.get("error", "Failed to fetch user info")
                )

            return UserInfo(**data["data"])

    async def revoke_token(self, token: str) -> bool:
        """Revoke access token"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.config.REVOKE_URL,
                json={"token": token}
            )

            return response.status_code == 200


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Quillon OAuth2 Python Example")
oauth_client = QuillonOAuth2Client()

# In-memory session storage (use Redis in production)
sessions = {}


@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with login button"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quillon OAuth2 Python Example</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #333; }
            .btn {
                display: inline-block;
                padding: 12px 24px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 6px;
                font-weight: bold;
            }
            .btn:hover { background: #5568d3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔐 Quillon Wallet OAuth2</h1>
            <h2>Python + FastAPI Example</h2>
            <p>This example demonstrates OAuth2 integration with Quillon Wallet.</p>
            <a href="/login" class="btn">🚀 Login with Quillon Wallet</a>

            <h3 style="margin-top: 40px;">Features:</h3>
            <ul>
                <li>✅ OAuth2 2.0 with PKCE support</li>
                <li>✅ Secure token management</li>
                <li>✅ User info retrieval</li>
                <li>✅ Token refresh mechanism</li>
                <li>✅ Wallet balance access</li>
            </ul>
        </div>
    </body>
    </html>
    """


@app.get("/login")
async def login():
    """Initiate OAuth2 authorization flow"""
    # Generate random state for CSRF protection
    state = secrets.token_urlsafe(32)
    sessions[state] = {"created_at": datetime.now()}

    # Redirect to authorization URL
    auth_url = oauth_client.get_authorization_url(state)
    return RedirectResponse(url=auth_url)


@app.get("/callback")
async def callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    """OAuth2 callback handler"""
    if error:
        return HTMLResponse(
            f"<h1>❌ Authorization Error</h1><p>{error}</p>",
            status_code=400
        )

    if not code or not state:
        return HTMLResponse(
            "<h1>❌ Missing Parameters</h1>",
            status_code=400
        )

    # Verify state (CSRF protection)
    if state not in sessions:
        return HTMLResponse(
            "<h1>❌ Invalid State</h1>",
            status_code=400
        )

    try:
        # Exchange authorization code for access token
        token = await oauth_client.exchange_code_for_token(code)

        # Get user information
        user_info = await oauth_client.get_user_info(token.access_token)

        # Store token in session
        sessions[state]["token"] = token
        sessions[state]["user_info"] = user_info

        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Success - Quillon OAuth2</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .success {{ color: #10b981; }}
                .info {{ background: #f0f9ff; padding: 16px; border-radius: 6px; margin: 16px 0; }}
                code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="success">✅ Authentication Successful!</h1>

                <div class="info">
                    <h3>👤 User Information:</h3>
                    <p><strong>Wallet Address:</strong> <code>{user_info.wallet_address}</code></p>
                    <p><strong>Balance:</strong> {user_info.balance_qug:.8f} QUG</p>
                    <p><strong>Scopes:</strong> {token.scope}</p>
                </div>

                <div class="info">
                    <h3>🔑 Access Token:</h3>
                    <p><code>{token.access_token[:40]}...</code></p>
                    <p><strong>Expires in:</strong> {token.expires_in} seconds</p>
                </div>

                <h3>🎯 Next Steps:</h3>
                <ul>
                    <li>Use the access token for API requests</li>
                    <li>Implement token refresh before expiry</li>
                    <li>Store tokens securely (encrypted)</li>
                </ul>

                <p><a href="/api/userinfo?state={state}">🔍 View User Info API</a></p>
            </div>
        </body>
        </html>
        """)

    except Exception as e:
        return HTMLResponse(
            f"<h1>❌ Error</h1><p>{str(e)}</p>",
            status_code=500
        )


@app.get("/api/userinfo")
async def api_userinfo(state: str):
    """API endpoint to get user info"""
    if state not in sessions or "token" not in sessions[state]:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = sessions[state]["token"]
    user_info = await oauth_client.get_user_info(token.access_token)

    return JSONResponse({
        "success": True,
        "data": user_info.dict()
    })


@app.post("/api/logout")
async def logout(state: str):
    """Revoke token and logout"""
    if state in sessions and "token" in sessions[state]:
        token = sessions[state]["token"]
        await oauth_client.revoke_token(token.access_token)
        del sessions[state]

    return JSONResponse({"success": True, "message": "Logged out successfully"})


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Quillon OAuth2 Python Example")
    print("📍 Visit: http://localhost:8000")
    print("⚙️  Client ID:", OAuth2Config.CLIENT_ID)
    print("🔗 Redirect URI:", OAuth2Config.REDIRECT_URI)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)
