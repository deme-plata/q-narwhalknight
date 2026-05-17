use std::collections::HashMap;
use std::sync::mpsc;

/// Port for the local OAuth2 callback server.
pub const OAUTH_PORT: u16 = 17655;

/// Parsed OAuth2 authorization request from a third-party website.
#[derive(Debug, Clone)]
pub struct OAuthAuthorizeRequest {
    pub client_id: String,
    pub redirect_uri: String,
    pub scope: String,
    pub code_challenge: String,
    pub code_challenge_method: String,
    pub state: String,
    pub response_type: String,
}

/// Response from the wallet UI (approve or deny).
#[derive(Debug, Clone)]
pub struct OAuthConsentResponse {
    pub approved: bool,
    /// Auth code returned by the backend (only set when approved).
    pub auth_code: Option<String>,
}

/// Starts the local OAuth2 HTTP server in a background thread.
///
/// Returns:
/// - A receiver that emits `OAuthAuthorizeRequest` whenever a third-party
///   website sends an authorization request.
/// - A sender the main thread uses to communicate the user's consent decision back.
/// - A receiver for auth codes arriving at `/callback` (from the server's browser consent flow).
pub fn start_oauth_server() -> (
    mpsc::Receiver<OAuthAuthorizeRequest>,
    mpsc::Sender<OAuthConsentResponse>,
    mpsc::Receiver<String>,
) {
    let (req_tx, req_rx) = mpsc::channel::<OAuthAuthorizeRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<OAuthConsentResponse>();
    let (callback_code_tx, callback_code_rx) = mpsc::channel::<String>();

    std::thread::Builder::new()
        .name("oauth-server".into())
        .spawn(move || {
            run_server(req_tx, resp_rx, callback_code_tx);
        })
        .expect("Failed to start OAuth server thread");

    (req_rx, resp_tx, callback_code_rx)
}

fn cors_headers() -> Vec<tiny_http::Header> {
    vec![
        tiny_http::Header::from_bytes(
            b"Access-Control-Allow-Origin",
            b"*",
        )
        .unwrap(),
        tiny_http::Header::from_bytes(
            b"Access-Control-Allow-Methods",
            b"GET, OPTIONS",
        )
        .unwrap(),
        tiny_http::Header::from_bytes(
            b"Access-Control-Allow-Headers",
            b"Content-Type",
        )
        .unwrap(),
    ]
}

fn run_server(
    req_tx: mpsc::Sender<OAuthAuthorizeRequest>,
    resp_rx: mpsc::Receiver<OAuthConsentResponse>,
    callback_code_tx: mpsc::Sender<String>,
) {
    let server = match tiny_http::Server::http(format!("127.0.0.1:{}", OAUTH_PORT)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[OAuth] Failed to bind port {}: {}", OAUTH_PORT, e);
            return;
        }
    };

    println!("[OAuth] Local server listening on 127.0.0.1:{}", OAUTH_PORT);

    for request in server.incoming_requests() {
        let url = request.url().to_string();
        let method = request.method().as_str().to_uppercase();

        // Handle CORS preflight
        if method == "OPTIONS" {
            let mut resp = tiny_http::Response::from_string("")
                .with_status_code(204);
            for h in cors_headers() {
                resp = resp.with_header(h);
            }
            let _ = request.respond(resp);
            continue;
        }

        // Discovery endpoint — lets local apps find the wallet's OAuth server
        if url == "/.well-known/oauth-authorization-server" || url == "/.well-known/openid-configuration" {
            let discovery = serde_json::json!({
                "issuer": format!("http://127.0.0.1:{}", OAUTH_PORT),
                "authorization_endpoint": format!("http://127.0.0.1:{}/authorize", OAUTH_PORT),
                "response_types_supported": ["code"],
                "code_challenge_methods_supported": ["S256"],
                "grant_types_supported": ["authorization_code"],
                "scopes_supported": ["read:balance", "read:history", "read:tokens", "send:transaction"],
            });
            let body = serde_json::to_string_pretty(&discovery).unwrap();
            let mut resp = tiny_http::Response::from_string(body)
                .with_status_code(200)
                .with_header(
                    tiny_http::Header::from_bytes(b"Content-Type", b"application/json")
                        .unwrap(),
                );
            for h in cors_headers() {
                resp = resp.with_header(h);
            }
            let _ = request.respond(resp);
            continue;
        }

        // Health check
        if url == "/health" {
            let mut resp = tiny_http::Response::from_string(r#"{"status":"ok"}"#)
                .with_status_code(200)
                .with_header(
                    tiny_http::Header::from_bytes(b"Content-Type", b"application/json")
                        .unwrap(),
                );
            for h in cors_headers() {
                resp = resp.with_header(h);
            }
            let _ = request.respond(resp);
            continue;
        }

        // Handle /callback — receives auth code after server consent, shows success page
        // and forwards the code to the main thread for token exchange
        if url.starts_with("/callback") {
            let params = parse_query_string(&url);
            let code = params.get("code").cloned().unwrap_or_default();
            let error = params.get("error").cloned().unwrap_or_default();

            // Forward auth code to main thread for PKCE token exchange
            if !code.is_empty() {
                let _ = callback_code_tx.send(code.clone());
            }

            let html = if !code.is_empty() {
                format!(
                    r#"<!DOCTYPE html>
<html><head><title>Authorized</title>
<style>body{{font-family:system-ui;background:#0a0e1a;color:#e0e0e0;display:flex;justify-content:center;align-items:center;height:100vh;margin:0}}
.card{{background:#151b2e;border:1px solid #00e5ff33;border-radius:16px;padding:40px;text-align:center;max-width:420px}}
h1{{color:#00e5ff;margin:0 0 12px}}p{{color:#a0a0a0;margin:8px 0}}
.code{{background:#0a0e1a;border:1px solid #00e5ff22;border-radius:8px;padding:12px;font-family:monospace;font-size:12px;word-break:break-all;color:#4caf50}}</style>
</head><body><div class="card">
<h1>Authorized</h1>
<p>Your wallet has been connected successfully.</p>
<p style="color:#4caf50">You can close this tab.</p>
<div class="code">{}</div>
</div></body></html>"#,
                    code
                )
            } else {
                format!(
                    r#"<!DOCTYPE html>
<html><head><title>Authorization Failed</title>
<style>body{{font-family:system-ui;background:#0a0e1a;color:#e0e0e0;display:flex;justify-content:center;align-items:center;height:100vh;margin:0}}
.card{{background:#151b2e;border:1px solid #ff444433;border-radius:16px;padding:40px;text-align:center;max-width:420px}}
h1{{color:#ff4444;margin:0 0 12px}}p{{color:#a0a0a0}}</style>
</head><body><div class="card">
<h1>Authorization Failed</h1>
<p>{}</p>
</div></body></html>"#,
                    if error.is_empty() { "Unknown error" } else { &error }
                )
            };

            let mut resp = tiny_http::Response::from_string(html)
                .with_status_code(200)
                .with_header(
                    tiny_http::Header::from_bytes(b"Content-Type", b"text/html; charset=utf-8")
                        .unwrap(),
                );
            for h in cors_headers() {
                resp = resp.with_header(h);
            }
            let _ = request.respond(resp);
            continue;
        }

        // Only handle GET /authorize
        if !url.starts_with("/authorize") {
            let resp = tiny_http::Response::from_string("Not Found")
                .with_status_code(404);
            let _ = request.respond(resp);
            continue;
        }

        // Parse query parameters
        let params = parse_query_string(&url);

        let client_id = match params.get("client_id") {
            Some(v) => v.clone(),
            None => {
                let resp = tiny_http::Response::from_string(
                    "Missing required parameter: client_id",
                )
                .with_status_code(400);
                let _ = request.respond(resp);
                continue;
            }
        };

        let redirect_uri = match params.get("redirect_uri") {
            Some(v) => v.clone(),
            None => {
                let resp = tiny_http::Response::from_string(
                    "Missing required parameter: redirect_uri",
                )
                .with_status_code(400);
                let _ = request.respond(resp);
                continue;
            }
        };

        let auth_request = OAuthAuthorizeRequest {
            client_id,
            redirect_uri: redirect_uri.clone(),
            scope: params
                .get("scope")
                .cloned()
                .unwrap_or_else(|| "read:balance".to_string()),
            code_challenge: params
                .get("code_challenge")
                .cloned()
                .unwrap_or_default(),
            code_challenge_method: params
                .get("code_challenge_method")
                .cloned()
                .unwrap_or_else(|| "S256".to_string()),
            state: params.get("state").cloned().unwrap_or_default(),
            response_type: params
                .get("response_type")
                .cloned()
                .unwrap_or_else(|| "code".to_string()),
        };

        let state_param = auth_request.state.clone();

        // Send to main thread to show consent screen
        if req_tx.send(auth_request).is_err() {
            let resp = tiny_http::Response::from_string("Wallet unavailable")
                .with_status_code(503);
            let _ = request.respond(resp);
            continue;
        }

        // Wait for user's consent decision (blocks this thread, which is fine -
        // only one OAuth request at a time)
        let consent = match resp_rx.recv() {
            Ok(c) => c,
            Err(_) => {
                let resp = tiny_http::Response::from_string("Wallet closed")
                    .with_status_code(503);
                let _ = request.respond(resp);
                continue;
            }
        };

        // Build redirect URL
        let location = if consent.approved {
            if let Some(code) = consent.auth_code {
                format!(
                    "{}?code={}&state={}",
                    redirect_uri,
                    urlencoding::encode(&code),
                    urlencoding::encode(&state_param)
                )
            } else {
                format!(
                    "{}?error=server_error&state={}",
                    redirect_uri,
                    urlencoding::encode(&state_param)
                )
            }
        } else {
            format!(
                "{}?error=access_denied&state={}",
                redirect_uri,
                urlencoding::encode(&state_param)
            )
        };

        // Respond with HTML redirect (302 + meta-refresh fallback)
        let html = format!(
            r#"<!DOCTYPE html>
<html><head>
<meta http-equiv="refresh" content="0;url={location}">
<title>Redirecting...</title>
</head><body>
<p>Redirecting to application...</p>
<script>window.location.href="{location}";</script>
</body></html>"#,
            location = location
        );

        let mut resp = tiny_http::Response::from_string(html)
            .with_status_code(302)
            .with_header(
                tiny_http::Header::from_bytes(b"Location", location.as_bytes())
                    .unwrap(),
            )
            .with_header(
                tiny_http::Header::from_bytes(b"Content-Type", b"text/html; charset=utf-8")
                    .unwrap(),
            );
        for h in cors_headers() {
            resp = resp.with_header(h);
        }

        let _ = request.respond(resp);
    }
}

/// Parse query parameters from a URL path like "/authorize?key=val&key2=val2".
fn parse_query_string(url: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Some(query) = url.split('?').nth(1) {
        for pair in query.split('&') {
            let mut parts = pair.splitn(2, '=');
            if let (Some(key), Some(val)) = (parts.next(), parts.next()) {
                let decoded_val =
                    urlencoding::decode(val).unwrap_or_else(|_| val.into());
                map.insert(key.to_string(), decoded_val.into_owned());
            }
        }
    }
    map
}
