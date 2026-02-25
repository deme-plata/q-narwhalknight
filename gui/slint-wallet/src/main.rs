#[allow(dead_code)]
mod api_client;
mod miner;
#[allow(dead_code)]
mod models;
mod oauth_server;
mod updater;
#[allow(dead_code)]
mod wallet;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use api_client::ApiClient;
use miner::MinerState;
use wallet::Wallet;

slint::include_modules!();

fn main() {
    // Parse optional CLI args
    let args: Vec<String> = std::env::args().collect();
    let _default_url = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "http://localhost:8080".to_string());

    let app = match AppWindow::new() {
        Ok(app) => app,
        Err(e) => {
            eprintln!("[WARN] Failed to create window with default renderer: {e}");
            eprintln!("[INFO] Retrying with software renderer (no GPU required)...");
            std::env::set_var("SLINT_BACKEND", "winit-software");
            AppWindow::new().expect("Failed to create window even with software renderer")
        }
    };
    let app_weak = app.as_weak();

    // Shared state
    let wallet: Arc<std::sync::Mutex<Option<Arc<Wallet>>>> =
        Arc::new(std::sync::Mutex::new(None));
    let api_client: Arc<std::sync::Mutex<Option<Arc<ApiClient>>>> =
        Arc::new(std::sync::Mutex::new(None));
    let miner_state = Arc::new(MinerState::new());
    // v8.0.1: SSE listener cancel flag — set to true to stop a running SSE listener
    let sse_running = Arc::new(AtomicBool::new(false));
    // v8.0.7: Store mnemonic in memory for transaction signing
    let stored_mnemonic: Arc<std::sync::Mutex<Option<String>>> =
        Arc::new(std::sync::Mutex::new(None));
    // Token balance map: symbol → display balance string (e.g. "1234.5678")
    let token_balances_map: Arc<std::sync::Mutex<std::collections::HashMap<String, String>>> =
        Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
    // Auto-updater state (initialized lazily once server URL is known after login)
    let updater_state: Arc<std::sync::Mutex<Option<Arc<updater::Updater>>>> =
        Arc::new(std::sync::Mutex::new(None));
    // Tick counter for update polling (check every 12th tick = ~60s at 5s interval)
    let update_tick: Arc<std::sync::atomic::AtomicU32> =
        Arc::new(std::sync::atomic::AtomicU32::new(0));

    // Tokio runtime for async API calls
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");
    let rt_handle = rt.handle().clone();

    // ── Start local OAuth2 server ──
    let (oauth_req_rx, oauth_resp_tx, callback_code_rx) = oauth_server::start_oauth_server();
    let oauth_resp_tx = Arc::new(std::sync::Mutex::new(oauth_resp_tx));

    // Shared state for PKCE flow (code_verifier stored here after browser open)
    let pkce_verifier: Arc<std::sync::Mutex<Option<String>>> =
        Arc::new(std::sync::Mutex::new(None));

    // Set default server URL
    app.set_wallet_address(slint::SharedString::from(""));

    // ── Create Wallet callback ──
    {
        let app_weak = app_weak.clone();
        app.on_create_wallet(move || {
            let app = app_weak.upgrade().unwrap();
            match Wallet::create() {
                Ok((_wallet, mnemonic)) => {
                    app.set_generated_mnemonic(slint::SharedString::from(&mnemonic));
                    app.set_show_create_view(true);
                    app.set_login_error(slint::SharedString::from(""));
                }
                Err(e) => {
                    app.set_login_error(slint::SharedString::from(format!(
                        "Failed to create wallet: {}",
                        e
                    )));
                }
            }
        });
    }

    // ── Confirm Created callback ──
    {
        let app_weak = app_weak.clone();
        let wallet = wallet.clone();
        let api_client = api_client.clone();
        let rt_handle = rt_handle.clone();
        let sse_running = sse_running.clone();
        let stored_mnemonic = stored_mnemonic.clone();
        app.on_confirm_created(move || {
            let app = app_weak.upgrade().unwrap();
            let mnemonic = app.get_generated_mnemonic().to_string();
            let server_url = get_server_url(&app);

            match Wallet::from_mnemonic(&mnemonic) {
                Ok(w) => {
                    let w = Arc::new(w);
                    let client = Arc::new(ApiClient::new(&server_url, w.clone()));
                    let addr = w.address().to_string();
                    app.set_wallet_address(slint::SharedString::from(&addr));
                    app.set_qr_code_image(generate_qr_image(&addr));
                    *wallet.lock().unwrap() = Some(w.clone());
                    *api_client.lock().unwrap() = Some(client.clone());
                    // v8.0.7: Store mnemonic for transaction signing
                    *stored_mnemonic.lock().unwrap() = Some(mnemonic.clone());
                    app.set_active_screen(1);
                    app.set_show_create_view(false);

                    // v8.0.1: Start SSE for real-time balance updates
                    // v8.2.7: Pass wallet auth for authenticated SSE events
                    let addr_hex = if addr.starts_with("qnk") { addr[3..].to_string() } else { addr.clone() };
                    start_sse_listener(&rt_handle, app.as_weak(), server_url, addr_hex, sse_running.clone(), SseAuth::Wallet(w));

                    // Register as OAuth2 client with the connected node (fire-and-forget)
                    rt_handle.spawn(async move {
                        let _ = client.register_oauth2_client().await;
                    });
                }
                Err(e) => {
                    app.set_login_error(slint::SharedString::from(format!("{}", e)));
                }
            }
        });
    }

    // ── Import Wallet callback ──
    {
        let app_weak = app_weak.clone();
        let wallet = wallet.clone();
        let api_client = api_client.clone();
        let rt_handle = rt_handle.clone();
        let sse_running = sse_running.clone();
        let stored_mnemonic = stored_mnemonic.clone();
        app.on_import_wallet(move |mnemonic| {
            let app = app_weak.upgrade().unwrap();
            let server_url = get_server_url(&app);

            // Normalize mnemonic: trim, collapse whitespace, lowercase
            let mnemonic_clean: String = mnemonic
                .trim()
                .split_whitespace()
                .collect::<Vec<&str>>()
                .join(" ")
                .to_lowercase();

            if mnemonic_clean.is_empty() {
                app.set_login_error(slint::SharedString::from("Please enter your recovery phrase"));
                return;
            }

            match Wallet::from_mnemonic(&mnemonic_clean) {
                Ok(w) => {
                    let w = Arc::new(w);
                    let client = Arc::new(ApiClient::new(&server_url, w.clone()));
                    let addr = w.address().to_string();
                    app.set_wallet_address(slint::SharedString::from(&addr));
                    app.set_qr_code_image(generate_qr_image(&addr));
                    *wallet.lock().unwrap() = Some(w.clone());
                    *api_client.lock().unwrap() = Some(client.clone());
                    // v8.0.7: Store mnemonic for transaction signing
                    *stored_mnemonic.lock().unwrap() = Some(mnemonic_clean.clone());
                    app.set_active_screen(1);
                    app.set_login_error(slint::SharedString::from(""));

                    // v8.0.1: Start SSE for real-time balance updates
                    // v8.2.7: Pass wallet auth for authenticated SSE events
                    let addr_hex = if addr.starts_with("qnk") { addr[3..].to_string() } else { addr.clone() };
                    start_sse_listener(&rt_handle, app.as_weak(), server_url, addr_hex, sse_running.clone(), SseAuth::Wallet(w.clone()));

                    // Register as OAuth2 client with the connected node (fire-and-forget)
                    rt_handle.spawn(async move {
                        let _ = client.register_oauth2_client().await;
                    });
                }
                Err(e) => {
                    app.set_login_error(slint::SharedString::from(format!(
                        "Invalid recovery phrase: {}",
                        e
                    )));
                }
            }
        });
    }

    // ── Send Transaction callback ──
    // v8.1.7: OAuth2 users auto-sign via server vault (no mnemonic needed).
    // If server returns "vault_key_missing", show one-time seed phrase prompt.
    {
        let app_weak = app_weak.clone();
        let api_client = api_client.clone();
        let rt_handle = rt_handle.clone();
        let stored_mnemonic = stored_mnemonic.clone();
        app.on_send_transaction(move |recipient, amount, memo, seed_phrase| {
            let app = app_weak.upgrade().unwrap();
            let client_lock = api_client.lock().unwrap();
            let Some(client) = client_lock.as_ref().cloned() else {
                app.set_send_status(slint::SharedString::from("Wallet not connected"));
                return;
            };
            let is_bearer_mode = !client.has_wallet();
            drop(client_lock);

            // v8.1.7: Get mnemonic — from stored, from UI field, or None (OAuth2 vault auto-sign)
            let seed_str = seed_phrase.to_string();
            let mnemonic = stored_mnemonic.lock().unwrap().clone().or_else(|| {
                if seed_str.trim().is_empty() { None } else { Some(seed_str.clone()) }
            });

            // v8.1.7: OAuth2 Bearer users can send WITHOUT mnemonic (server vault signs).
            // Only block if wallet-auth mode (local key required) and no mnemonic.
            if mnemonic.is_none() && !is_bearer_mode {
                app.set_send_status(slint::SharedString::from(
                    "Enter your 24-word seed phrase above to sign the transaction.",
                ));
                return;
            }

            let recipient = recipient.to_string();
            let amount = amount.to_string();
            let memo_str = memo.to_string();
            let memo_opt = if memo_str.is_empty() {
                None
            } else {
                Some(memo_str)
            };

            // Validate inputs
            if !recipient.starts_with("qnk") || recipient.len() != 67 {
                app.set_send_status(slint::SharedString::from(
                    "Invalid address (must start with qnk, 67 chars)",
                ));
                return;
            }
            if amount.parse::<f64>().unwrap_or(-1.0) <= 0.0 {
                app.set_send_status(slint::SharedString::from("Invalid amount"));
                return;
            }

            // Show sending state immediately
            app.set_sending(true);
            app.set_send_status(slint::SharedString::from(""));

            let amount_confirm = amount.clone();
            let recipient_confirm = recipient.clone();
            let stored_mn = stored_mnemonic.clone();
            let seed_to_store = if !seed_str.trim().is_empty() {
                Some(seed_str)
            } else {
                None
            };

            let weak = app.as_weak();
            rt_handle.spawn(async move {
                let result = client.send_transaction(&recipient, &amount, memo_opt, mnemonic).await;
                let _ = slint::invoke_from_event_loop(move || {
                    let app = weak.upgrade().unwrap();
                    app.set_sending(false);
                    match result {
                        Ok(_) => {
                            // v8.0.8: Store seed phrase for OAuth2 users (remembered for session)
                            if let Some(seed) = seed_to_store {
                                *stored_mn.lock().unwrap() = Some(seed);
                                app.set_needs_seed_phrase(false);
                            }
                            // Show confirm modal
                            app.set_confirm_amount(slint::SharedString::from(&amount_confirm));
                            app.set_confirm_recipient(slint::SharedString::from(
                                &recipient_confirm,
                            ));
                            app.set_show_send_confirm(true);
                            app.set_contact_saved(false);
                            app.set_send_status(slint::SharedString::from(""));
                        }
                        Err(e) => {
                            let err_msg = e.to_string();
                            // v8.1.7: Handle vault_key_missing — show one-time seed phrase prompt
                            if err_msg.contains("vault_key_missing") {
                                app.set_needs_seed_phrase(true);
                                app.set_send_status(slint::SharedString::from(
                                    "Enter your seed phrase once to enable automatic signing.",
                                ));
                            } else if err_msg.contains("vault_key_corrupt") {
                                app.set_needs_seed_phrase(true);
                                app.set_send_status(slint::SharedString::from(
                                    "Vault key needs reset. Enter your seed phrase to re-initialize.",
                                ));
                            } else {
                                app.set_send_status(slint::SharedString::from(format!(
                                    "Error: {}",
                                    e
                                )));
                            }
                        }
                    }
                });
            });
        });
    }

    // ── Refresh History callback ──
    {
        let app_weak = app_weak.clone();
        let api_client = api_client.clone();
        let wallet_ref = wallet.clone();
        let rt_handle = rt_handle.clone();
        app.on_refresh_history(move || {
            let client_lock = api_client.lock().unwrap();
            let Some(client) = client_lock.as_ref().cloned() else {
                return;
            };
            drop(client_lock);

            // Get address from wallet (if available) or from api client (Bearer mode)
            let my_address = wallet_ref
                .lock()
                .unwrap()
                .as_ref()
                .map(|w| w.address().to_string())
                .unwrap_or_else(|| client.address().to_string());

            let weak = app_weak.clone();
            rt_handle.spawn(async move {
                let weak2 = weak.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    let app = weak2.upgrade().unwrap();
                    app.set_history_loading(true);
                });

                let records = match client.get_history().await {
                    Ok(r) => {
                        println!("[History] Loaded {} transactions", r.len());
                        r
                    }
                    Err(e) => {
                        eprintln!("[History] Fetch failed: {}", e);
                        vec![]
                    }
                };
                let _ = slint::invoke_from_event_loop(move || {
                    let app = weak.upgrade().unwrap();
                    // Strip qnk prefix for comparison
                    let my_addr_hex = if my_address.starts_with("qnk") {
                        &my_address[3..]
                    } else {
                        &my_address
                    };
                    let tx_model: Vec<TxRecord> = records
                        .iter()
                        .map(|r| {
                            let is_sent = r.from == my_addr_hex || r.from == my_address;
                            // v8.0.8: Convert raw 24-decimal amount to display
                            let display_amount = if r.amount > 1e15 {
                                r.amount / 1e24
                            } else {
                                r.amount
                            };
                            TxRecord {
                                direction: slint::SharedString::from(if is_sent {
                                    "Sent"
                                } else {
                                    "Received"
                                }),
                                amount: slint::SharedString::from(format!(
                                    "{:.4} QUG",
                                    display_amount
                                )),
                                counterparty: slint::SharedString::from(
                                    if is_sent { &r.to } else { &r.from },
                                ),
                                timestamp: slint::SharedString::from(&r.timestamp),
                                tx_hash: slint::SharedString::from(&r.id),
                                fee: slint::SharedString::from("0.001"),
                                token: slint::SharedString::from("QUG"),
                            }
                        })
                        .collect();
                    let model = std::rc::Rc::new(slint::VecModel::from(tx_model));
                    app.set_transactions(model.into());
                    app.set_history_loading(false);
                });
            });
        });
    }

    // ── Load Address Book callback ──
    {
        let app_weak = app_weak.clone();
        let api_client = api_client.clone();
        let rt_handle = rt_handle.clone();
        app.on_load_address_book(move || {
            let client_lock = api_client.lock().unwrap();
            let Some(client) = client_lock.as_ref().cloned() else {
                return;
            };
            drop(client_lock);

            let weak = app_weak.clone();
            rt_handle.spawn(async move {
                match client.get_address_book().await {
                    Ok(entries) => {
                        println!("[AddressBook] Loaded {} contacts", entries.len());
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(app) = weak.upgrade() {
                                let contacts: Vec<AddressEntry> = entries
                                    .iter()
                                    .map(|e| AddressEntry {
                                        label: slint::SharedString::from(&e.label),
                                        address: slint::SharedString::from(&e.address),
                                    })
                                    .collect();
                                let model =
                                    std::rc::Rc::new(slint::VecModel::from(contacts));
                                app.set_address_book(model.into());
                            }
                        });
                    }
                    Err(e) => {
                        eprintln!("[AddressBook] Load failed: {}", e);
                    }
                }
            });
        });
    }

    // ── Save Contact callback ──
    {
        let app_weak = app_weak.clone();
        let api_client = api_client.clone();
        let rt_handle = rt_handle.clone();
        app.on_save_contact(move |address, label| {
            let client_lock = api_client.lock().unwrap();
            let Some(client) = client_lock.as_ref().cloned() else {
                return;
            };
            drop(client_lock);

            let address_str = address.to_string();
            let label_str = label.to_string();
            if label_str.trim().is_empty() || address_str.trim().is_empty() {
                return;
            }

            let weak = app_weak.clone();
            rt_handle.spawn(async move {
                match client.save_address(&address_str, &label_str).await {
                    Ok(_) => {
                        println!(
                            "[AddressBook] Saved contact: {} -> {}",
                            label_str, address_str
                        );
                        // Refresh address book after save
                        if let Ok(entries) = client.get_address_book().await {
                            let _ = slint::invoke_from_event_loop(move || {
                                if let Some(app) = weak.upgrade() {
                                    let contacts: Vec<AddressEntry> = entries
                                        .iter()
                                        .map(|e| AddressEntry {
                                            label: slint::SharedString::from(&e.label),
                                            address: slint::SharedString::from(&e.address),
                                        })
                                        .collect();
                                    let model =
                                        std::rc::Rc::new(slint::VecModel::from(contacts));
                                    app.set_address_book(model.into());
                                    app.set_contact_saved(true);
                                }
                            });
                        } else {
                            // Still mark as saved even if refresh fails
                            let _ = slint::invoke_from_event_loop(move || {
                                if let Some(app) = weak.upgrade() {
                                    app.set_contact_saved(true);
                                }
                            });
                        }
                    }
                    Err(e) => {
                        eprintln!("[AddressBook] Save failed: {}", e);
                    }
                }
            });
        });
    }

    // ── Toggle Mining callback ──
    {
        let app_weak = app_weak.clone();
        let miner_state = miner_state.clone();
        let api_client = api_client.clone();
        let wallet_ref = wallet.clone();
        let rt_handle = rt_handle.clone();
        app.on_toggle_mining(move || {
            let app = app_weak.upgrade().unwrap();

            if miner_state.running.load(Ordering::SeqCst) {
                miner::stop_mining(&miner_state);
                app.set_mining(false);
            } else {
                let client_lock = api_client.lock().unwrap();
                let Some(client) = client_lock.as_ref().cloned() else {
                    return;
                };
                drop(client_lock);

                // Get address from wallet (import/create) or from API client (OAuth2 Bearer)
                let address = wallet_ref
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|w| w.address().to_string())
                    .unwrap_or_else(|| client.address().to_string());

                if address.is_empty() {
                    eprintln!("[Miner] No wallet address available for mining");
                    return;
                }

                let pool = app.get_pool_mode();
                println!("[Miner] Starting mining with address: {} (pool={})", address, pool);
                miner::start_mining(
                    miner_state.clone(),
                    client,
                    address,
                    rt_handle.clone(),
                    pool,
                );
                app.set_mining(true);
            }
        });
    }

    // ── Copy Address callback (with "Copied!" feedback) ──
    {
        let wallet_ref = wallet.clone();
        let app_weak = app_weak.clone();
        app.on_copy_address(move || {
            if let Some(w) = wallet_ref.lock().unwrap().as_ref() {
                let address = w.address().to_string();
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    let _ = clipboard.set_text(&address);
                }
            }
            // Show "Copied!" feedback, auto-reset after 2 seconds
            let weak = app_weak.clone();
            if let Some(app) = weak.upgrade() {
                app.set_copy_feedback(true);
            }
            let weak2 = weak.clone();
            let timer = slint::Timer::default();
            timer.start(slint::TimerMode::SingleShot, std::time::Duration::from_secs(2), move || {
                if let Some(app) = weak2.upgrade() {
                    app.set_copy_feedback(false);
                }
            });
            std::mem::forget(timer); // prevent timer from being dropped
        });
    }

    // ── Send Token callback (from dashboard token card) ──
    {
        let app_weak = app_weak.clone();
        let balances_map = token_balances_map.clone();
        app.on_send_token(move |symbol| {
            if let Some(app) = app_weak.upgrade() {
                app.set_selected_token(symbol.clone());
                // Update balance display for the selected token
                let sym = symbol.to_string();
                let bal = balances_map.lock().unwrap().get(&sym).cloned().unwrap_or_else(|| "0.0000".to_string());
                app.set_send_balance_display(slint::SharedString::from(format!("{} {}", bal, sym)));
                app.set_active_screen(2);
                app.invoke_load_address_book();
            }
        });
    }

    // ── Select Token callback (from send screen dropdown) ──
    {
        let app_weak = app_weak.clone();
        let balances_map = token_balances_map.clone();
        app.on_select_token(move |symbol| {
            if let Some(app) = app_weak.upgrade() {
                let sym = symbol.to_string();
                let bal = balances_map.lock().unwrap().get(&sym).cloned().unwrap_or_else(|| "0.0000".to_string());
                app.set_send_balance_display(slint::SharedString::from(format!("{} {}", bal, sym)));
            }
        });
    }

    // ── Copy Text callback (for history detail modal) ──
    {
        app.on_copy_text(move |text| {
            let text_str = text.to_string();
            if let Ok(mut clipboard) = arboard::Clipboard::new() {
                let _ = clipboard.set_text(&text_str);
            }
        });
    }

    // ── Open OAuth2 Browser callback ──
    // Opens browser to the NODE's consent page (e.g. quillon.xyz/oauth/consent)
    // After user approves, node redirects to localhost:17655/callback with auth code
    {
        let app_weak = app_weak.clone();
        let pkce_verifier = pkce_verifier.clone();
        app.on_open_oauth_browser(move || {
            let app = app_weak.upgrade().unwrap();
            let server_url = app.get_server_url().to_string();
            let server_url = server_url.trim_end_matches('/');

            // Generate PKCE code_verifier (RFC 7636: 43-128 chars, unreserved chars)
            use rand::Rng;
            let verifier: String = rand::thread_rng()
                .sample_iter(&rand::distributions::Alphanumeric)
                .take(64)
                .map(char::from)
                .collect();

            // Compute code_challenge = BASE64URL(SHA256(code_verifier))
            use sha2::Digest;
            let hash = sha2::Sha256::digest(verifier.as_bytes());
            let challenge = base64::Engine::encode(
                &base64::engine::general_purpose::URL_SAFE_NO_PAD,
                hash,
            );

            // Store verifier for token exchange later
            *pkce_verifier.lock().unwrap() = Some(verifier);

            // redirect_uri points back to the wallet's local callback server
            let redirect_uri = format!("http://127.0.0.1:{}/callback", oauth_server::OAUTH_PORT);

            // Open browser to the NODE's OAuth2 authorize endpoint with PKCE
            let authorize_url = format!(
                "{}/api/v1/oauth2/authorize?client_id=slint-wallet&redirect_uri={}&scope={}&response_type=code&state=slint-self-connect&code_challenge={}&code_challenge_method=S256",
                server_url,
                urlencoding::encode(&redirect_uri),
                urlencoding::encode("read:balance read:history read:tokens send:transaction"),
                urlencoding::encode(&challenge),
            );

            println!("[OAuth] Opening browser: {}", authorize_url);
            open_browser(&authorize_url);
        });
    }

    // ── OAuth callbacks and polling ──
    // Shared pending request state for approve/deny/poll
    let pending_oauth: Arc<std::sync::Mutex<Option<oauth_server::OAuthAuthorizeRequest>>> =
        Arc::new(std::sync::Mutex::new(None));

    // OAuth Approve
    {
        let app_weak = app_weak.clone();
        let api_client = api_client.clone();
        let oauth_resp_tx = oauth_resp_tx.clone();
        let rt_handle = rt_handle.clone();
        let pending_oauth = pending_oauth.clone();

        app.on_oauth_approve(move || {
            let app = app_weak.upgrade().unwrap();
            let client_lock = api_client.lock().unwrap();
            let Some(client) = client_lock.as_ref().cloned() else {
                let _ = oauth_resp_tx.lock().unwrap().send(
                    oauth_server::OAuthConsentResponse {
                        approved: false,
                        auth_code: None,
                    },
                );
                app.set_active_screen(1);
                return;
            };
            drop(client_lock);

            let Some(req) = pending_oauth.lock().unwrap().take() else {
                app.set_active_screen(1);
                return;
            };

            let tx = oauth_resp_tx.clone();
            let weak = app.as_weak();
            let scopes: Vec<String> =
                req.scope.split_whitespace().map(|s| s.to_string()).collect();
            let redirect_uri = req.redirect_uri.clone();
            let code_challenge = if req.code_challenge.is_empty() {
                None
            } else {
                Some(req.code_challenge.clone())
            };
            let code_challenge_method = if req.code_challenge_method.is_empty() {
                None
            } else {
                Some(req.code_challenge_method.clone())
            };

            rt_handle.spawn(async move {
                let result = client
                    .authorize_oauth2(
                        &req.client_id,
                        &scopes,
                        &redirect_uri,
                        code_challenge.as_deref(),
                        code_challenge_method.as_deref(),
                    )
                    .await;

                let response = match result {
                    Ok(code) => oauth_server::OAuthConsentResponse {
                        approved: true,
                        auth_code: Some(code),
                    },
                    Err(e) => {
                        eprintln!("[OAuth] Backend consent error: {}", e);
                        oauth_server::OAuthConsentResponse {
                            approved: false,
                            auth_code: None,
                        }
                    }
                };

                let _ = tx.lock().unwrap().send(response);
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(app) = weak.upgrade() {
                        app.set_active_screen(1);
                    }
                });
            });
        });
    }

    // OAuth Deny
    {
        let app_weak = app_weak.clone();
        let oauth_resp_tx = oauth_resp_tx.clone();
        let pending_oauth = pending_oauth.clone();
        app.on_oauth_deny(move || {
            let _ = pending_oauth.lock().unwrap().take();
            let _ = oauth_resp_tx.lock().unwrap().send(
                oauth_server::OAuthConsentResponse {
                    approved: false,
                    auth_code: None,
                },
            );
            if let Some(app) = app_weak.upgrade() {
                app.set_active_screen(1);
            }
        });
    }

    // OAuth request polling timer — checks for incoming authorize requests AND callback codes
    {
        let app_weak = app_weak.clone();
        let _oauth_resp_tx = oauth_resp_tx.clone();
        let pending_oauth = pending_oauth.clone();
        let pkce_verifier = pkce_verifier.clone();
        let api_client = api_client.clone();
        let rt_handle = rt_handle.clone();
        let sse_running = sse_running.clone();
        let oauth_timer = slint::Timer::default();
        oauth_timer.start(
            slint::TimerMode::Repeated,
            std::time::Duration::from_millis(250),
            move || {
                let app = match app_weak.upgrade() {
                    Some(a) => a,
                    None => return,
                };

                // Check for callback auth codes (from browser consent → /callback redirect)
                if let Ok(code) = callback_code_rx.try_recv() {
                    let verifier = pkce_verifier.lock().unwrap().take();
                    if let Some(verifier) = verifier {
                        let server_url = get_server_url(&app);
                        let redirect_uri = format!(
                            "http://127.0.0.1:{}/callback",
                            oauth_server::OAUTH_PORT
                        );
                        println!("[OAuth] Received callback code, exchanging for token...");

                        let weak = app.as_weak();
                        let weak_err = weak.clone();
                        let api_client_oauth = api_client.clone();
                        let sse_running_sse = sse_running.clone();
                        rt_handle.spawn(async move {
                            match ApiClient::exchange_oauth2_token(
                                &server_url,
                                &code,
                                &redirect_uri,
                                &verifier,
                            )
                            .await
                            {
                                Ok(token_resp) => {
                                    println!(
                                        "[OAuth] Token exchange successful, scope: {}",
                                        token_resp.scope
                                    );
                                    let access_token = token_resp.access_token.clone();
                                    // v8.2.7: Clone token for SSE auth before it moves into ApiClient
                                    let sse_token = access_token.clone();

                                    // Get wallet address from userinfo endpoint
                                    let addr = match ApiClient::get_userinfo_with_token(
                                        &server_url,
                                        &access_token,
                                    )
                                    .await
                                    {
                                        Ok(info) => {
                                            if !info.wallet_address.is_empty() {
                                                info.wallet_address
                                            } else {
                                                info.sub
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("[OAuth] Userinfo error (non-fatal): {}", e);
                                            "OAuth User".to_string()
                                        }
                                    };

                                    // v8.0.1: Create Bearer-auth API client for OAuth2 users
                                    // Supports balance, history, send transactions via Bearer token
                                    let bearer_client = Arc::new(ApiClient::from_bearer(
                                        &server_url,
                                        access_token,
                                        addr.clone(),
                                    ));
                                    *api_client_oauth.lock().unwrap() = Some(bearer_client);

                                    // v8.0.1: Start SSE for real-time balance updates
                                    // v8.2.7: Pass bearer token for authenticated SSE events
                                    {
                                        let addr_hex = if addr.starts_with("qnk") { addr[3..].to_string() } else { addr.clone() };
                                        let rt = tokio::runtime::Handle::current();
                                        start_sse_listener(&rt, weak.clone(), server_url, addr_hex, sse_running_sse, SseAuth::Bearer(sse_token));
                                    }

                                    let _ = slint::invoke_from_event_loop(move || {
                                        let app = match weak.upgrade() {
                                            Some(a) => a,
                                            None => return,
                                        };
                                        app.set_wallet_address(slint::SharedString::from(&addr));
                                        app.set_qr_code_image(generate_qr_image(&addr));
                                        app.set_login_error(slint::SharedString::from(""));
                                        // v8.1.7: OAuth2 uses server vault for auto-signing — no seed phrase needed
                                        // If vault key is missing, server returns "vault_key_missing" and
                                        // the send handler will show a one-time prompt then.
                                        app.set_needs_seed_phrase(false);
                                        app.set_active_screen(1);
                                        println!(
                                            "[OAuth] Login complete via browser consent for: {}",
                                            addr
                                        );
                                    });
                                }
                                Err(e) => {
                                    eprintln!("[OAuth] Token exchange failed: {}", e);
                                    let _ = slint::invoke_from_event_loop(move || {
                                        if let Some(app) = weak_err.upgrade() {
                                            app.set_login_error(slint::SharedString::from(
                                                format!("OAuth login failed: {}", e),
                                            ));
                                        }
                                    });
                                }
                            }
                        });
                    } else {
                        eprintln!("[OAuth] Received callback code but no PKCE verifier stored");
                    }
                }

                // Check for third-party authorize requests (consent screen flow)
                if let Ok(req) = oauth_req_rx.try_recv() {
                    // Build scope list for the consent screen
                    let scope_list: Vec<OAuthScope> = req
                        .scope
                        .split_whitespace()
                        .map(|s| {
                            let desc = match s {
                                "read:balance" => "View your wallet balance",
                                "send:transaction" => "Send transactions on your behalf",
                                "read:history" => "View your transaction history",
                                "read:tokens" => "View your token balances",
                                _ => "Custom permission",
                            };
                            OAuthScope {
                                name: slint::SharedString::from(s),
                                description: slint::SharedString::from(desc),
                                checked: true,
                            }
                        })
                        .collect();

                    app.set_oauth_client_name(slint::SharedString::from(&req.client_id));
                    app.set_oauth_client_id(slint::SharedString::from(&req.client_id));
                    app.set_oauth_redirect_uri(slint::SharedString::from(&req.redirect_uri));
                    let model = std::rc::Rc::new(slint::VecModel::from(scope_list));
                    app.set_oauth_scopes(model.into());

                    // Store request for approve/deny callbacks
                    *pending_oauth.lock().unwrap() = Some(req);

                    app.set_active_screen(6);
                }
            },
        );
        std::mem::forget(oauth_timer);
    }

    // ── Auto-update callbacks ──
    {
        let updater_ref = updater_state.clone();
        let rt_handle_upd = rt_handle.clone();
        let app_weak_upd = app_weak.clone();
        app.on_start_update_download(move || {
            let updater_lock = updater_ref.lock().unwrap();
            let Some(upd) = updater_lock.as_ref().cloned() else { return };
            drop(updater_lock);

            // Get the version from current UI state
            let version = {
                let app = app_weak_upd.upgrade().unwrap();
                app.get_update_version().to_string()
            };

            let weak = app_weak_upd.clone();
            let upd2 = upd.clone();
            rt_handle_upd.spawn(async move {
                // Set UI to downloading immediately
                let _ = slint::invoke_from_event_loop({
                    let weak = weak.clone();
                    move || {
                        if let Some(app) = weak.upgrade() {
                            app.set_update_state(2);
                            app.set_update_progress(0.0);
                            app.set_update_status(slint::SharedString::from("starting..."));
                        }
                    }
                });

                // Subscribe to progress updates
                let mut rx = upd2.subscribe();
                let weak_progress = weak.clone();
                let progress_task = tokio::spawn(async move {
                    while rx.changed().await.is_ok() {
                        let state = rx.borrow().clone();
                        let weak = weak_progress.clone();
                        match state {
                            updater::UpdateState::Downloading { bytes_downloaded, bytes_total, .. } => {
                                let progress = if bytes_total > 0 {
                                    bytes_downloaded as f32 / bytes_total as f32
                                } else {
                                    0.0
                                };
                                let status = if bytes_total > 0 {
                                    format!(
                                        "{:.1}/{:.1} MB",
                                        bytes_downloaded as f64 / 1_048_576.0,
                                        bytes_total as f64 / 1_048_576.0
                                    )
                                } else {
                                    format!("{:.1} MB", bytes_downloaded as f64 / 1_048_576.0)
                                };
                                let _ = slint::invoke_from_event_loop(move || {
                                    if let Some(app) = weak.upgrade() {
                                        app.set_update_progress(progress);
                                        app.set_update_status(slint::SharedString::from(&status));
                                    }
                                });
                            }
                            updater::UpdateState::ReadyToRestart { .. } => {
                                let _ = slint::invoke_from_event_loop(move || {
                                    if let Some(app) = weak.upgrade() {
                                        app.set_update_state(3);
                                    }
                                });
                                break;
                            }
                            updater::UpdateState::Error { message, .. } => {
                                let _ = slint::invoke_from_event_loop(move || {
                                    if let Some(app) = weak.upgrade() {
                                        app.set_update_state(4);
                                        app.set_update_error(slint::SharedString::from(&message));
                                    }
                                });
                                break;
                            }
                            _ => {}
                        }
                    }
                });

                // Start download
                if let Err(e) = upd2.download_update(&version).await {
                    eprintln!("[Updater] Download error: {}", e);
                    let weak = weak.clone();
                    let msg = e.to_string();
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(app) = weak.upgrade() {
                            app.set_update_state(4);
                            app.set_update_error(slint::SharedString::from(&msg));
                        }
                    });
                }
                progress_task.abort();
            });
        });
    }

    {
        let updater_ref = updater_state.clone();
        app.on_restart_for_update(move || {
            let updater_lock = updater_ref.lock().unwrap();
            if let Some(upd) = updater_lock.as_ref() {
                if let Err(e) = upd.apply_update() {
                    eprintln!("[Updater] Apply error: {}", e);
                    return;
                }
            }
            drop(updater_lock);
            if let Err(e) = updater::Updater::restart() {
                eprintln!("[Updater] Restart error: {}", e);
            }
        });
    }

    {
        let updater_ref = updater_state.clone();
        let app_weak_err = app_weak.clone();
        app.on_dismiss_update_error(move || {
            let updater_lock = updater_ref.lock().unwrap();
            if let Some(upd) = updater_lock.as_ref() {
                let ver = {
                    let app = app_weak_err.upgrade().unwrap();
                    app.get_update_version().to_string()
                };
                upd.reset_to_available(&ver);
            }
            drop(updater_lock);
            if let Some(app) = app_weak_err.upgrade() {
                app.set_update_state(1);
                app.set_update_error(slint::SharedString::from(""));
            }
        });
    }

    // ── Background polling timer ──
    // Poll status every 5s, balances every 10s, miner stats continuously
    {
        let app_weak = app_weak.clone();
        let api_client = api_client.clone();
        let _wallet_ref = wallet.clone();
        let miner_state = miner_state.clone();
        let rt_handle = rt_handle.clone();
        let updater_ref = updater_state.clone();
        let update_tick = update_tick.clone();

        let poll_timer = slint::Timer::default();
        poll_timer.start(
            slint::TimerMode::Repeated,
            std::time::Duration::from_secs(5),
            move || {
                let app = match app_weak.upgrade() {
                    Some(a) => a,
                    None => return,
                };

                // Only poll when logged in
                if app.get_active_screen() == 0 {
                    return;
                }

                // ── Auto-update check: first at ~60s, then every 4 hours ──
                let tick = update_tick.fetch_add(1, Ordering::Relaxed);
                let should_check_update = tick == 12 || (tick > 12 && (tick - 12) % 2880 == 0);
                if should_check_update {
                    // Lazily initialize updater with current server URL
                    {
                        let mut lock = updater_ref.lock().unwrap();
                        if lock.is_none() {
                            let url = get_server_url(&app);
                            if !url.is_empty() {
                                *lock = Some(Arc::new(updater::Updater::new(&url)));
                            }
                        }
                    }

                    let upd_lock = updater_ref.lock().unwrap();
                    if let Some(upd) = upd_lock.as_ref().cloned() {
                        // Only check if we're idle or haven't checked yet
                        if upd.state().is_idle() {
                            let weak_upd = app.as_weak();
                            rt_handle.spawn(async move {
                                match upd.check_for_update().await {
                                    Ok(Some(version)) => {
                                        let _ = slint::invoke_from_event_loop(move || {
                                            if let Some(app) = weak_upd.upgrade() {
                                                app.set_update_state(1);
                                                app.set_update_version(slint::SharedString::from(&version));
                                            }
                                        });
                                    }
                                    Ok(None) => {} // Up to date
                                    Err(e) => {
                                        eprintln!("[Updater] Check failed: {}", e);
                                    }
                                }
                            });
                        }
                    }
                }

                // Update miner stats from atomics (fast, no async)
                if miner_state.running.load(Ordering::Relaxed) {
                    let hr = miner_state.hashrate.load(Ordering::Relaxed);
                    app.set_hashrate(slint::SharedString::from(format_hashrate(hr)));
                    app.set_blocks_found(slint::SharedString::from(format!(
                        "{}",
                        miner_state.blocks_found.load(Ordering::Relaxed)
                    )));
                    // Show miner status and thread count
                    if let Ok(status) = miner_state.last_status.lock() {
                        app.set_miner_status(slint::SharedString::from(status.as_str()));
                    }
                    app.set_thread_count(slint::SharedString::from(format!(
                        "{}",
                        miner_state.active_threads.load(Ordering::Relaxed)
                    )));
                }

                // Async API polls
                let client_lock = api_client.lock().unwrap();
                let Some(client) = client_lock.as_ref().cloned() else {
                    return;
                };
                drop(client_lock);

                let weak = app.as_weak();
                let client2 = client.clone();
                let weak2 = weak.clone();

                // Fetch status
                rt_handle.spawn(async move {
                    match client.get_status().await {
                        Ok(status) => {
                            let _ = slint::invoke_from_event_loop(move || {
                                if let Some(app) = weak.upgrade() {
                                    app.set_connected(true);
                                    app.set_current_height(slint::SharedString::from(format!(
                                        "{}",
                                        status.current_height
                                    )));
                                    app.set_network_height(slint::SharedString::from(format!(
                                        "{}",
                                        status.highest_network_height.max(status.current_height)
                                    )));
                                    let pct = if status.highest_network_height > 0 {
                                        (status.current_height as f64 / status.highest_network_height as f64
                                            * 100.0)
                                            .min(100.0)
                                    } else {
                                        100.0
                                    };
                                    app.set_sync_percent(pct as f32);
                                }
                            });
                        }
                        Err(_) => {
                            let _ = slint::invoke_from_event_loop(move || {
                                if let Some(app) = weak2.upgrade() {
                                    app.set_connected(false);
                                }
                            });
                        }
                    }
                });

                // Fetch mining challenge info (for difficulty & reward display)
                {
                    let weak_ch = app.as_weak();
                    let client_ch = client2.clone();
                    rt_handle.spawn(async move {
                        if let Ok(ch) = client_ch.get_mining_challenge().await {
                            let diff_display = if ch.difficulty_target.len() >= 8 {
                                format!("0x{}...", &ch.difficulty_target[..8])
                            } else {
                                ch.difficulty_target.clone()
                            };
                            let reward_display = format!("{:.4}", ch.block_reward);
                            let _ = slint::invoke_from_event_loop(move || {
                                if let Some(app) = weak_ch.upgrade() {
                                    app.set_difficulty(slint::SharedString::from(&diff_display));
                                    app.set_reward_per_block(slint::SharedString::from(&reward_display));
                                }
                            });
                        }
                    });
                }

                // v8.2.8: Unified balance + token fetch
                // The /wallet/tokens endpoint returns QUG balance AND all tokens in one call.
                // This is the PRIMARY source for the main QUG balance display.
                // The dedicated /wallets/{addr}/balance endpoint is a secondary fallback.
                let weak_tok = app.as_weak();
                let client3 = api_client.lock().unwrap().as_ref().cloned();
                let balances_map_tok = token_balances_map.clone();
                if let Some(client3) = client3 {
                    let weak_bal_fallback = weak_tok.clone();
                    let balances_map_fallback = balances_map_tok.clone();

                    rt_handle.spawn(async move {
                        match client3.get_token_balances().await {
                            Ok(tokens) => {
                                // v8.2.8: Extract QUG balance from tokens for main display
                                let qug_from_tokens = tokens.iter()
                                    .find(|t| t.symbol == "QUG")
                                    .and_then(|t| t.balance.parse::<f64>().ok())
                                    .unwrap_or(0.0);

                                // Store ALL token balances in shared map (for send screen lookup)
                                {
                                    let mut map = balances_map_tok.lock().unwrap();
                                    for t in &tokens {
                                        let bal_f64: f64 = t.balance.parse().unwrap_or(0.0);
                                        map.insert(t.symbol.clone(), format!("{:.4}", bal_f64));
                                    }
                                }

                                let _ = slint::invoke_from_event_loop(move || {
                                    let Some(app) = weak_tok.upgrade() else { return };

                                    // v8.2.8: Update main QUG balance from token endpoint (primary source)
                                    if qug_from_tokens > 0.0 || app.get_qug_balance().is_empty() || app.get_qug_balance() == "0.0000" {
                                        let balance_display = format!("{:.4}", qug_from_tokens);
                                        let value_usd = format!("${:.2}", qug_from_tokens * 3000.0);
                                        app.set_qug_balance(slint::SharedString::from(&balance_display));
                                        app.set_qug_value_usd(slint::SharedString::from(&value_usd));
                                        if app.get_selected_token() == "QUG" {
                                            app.set_send_balance_display(slint::SharedString::from(
                                                format!("{} QUG", balance_display),
                                            ));
                                        }
                                    }

                                    let token_list: Vec<TokenInfo> = tokens
                                        .iter()
                                        .map(|t| {
                                            let bal_f64: f64 =
                                                t.balance.parse().unwrap_or(0.0);
                                            TokenInfo {
                                                name: slint::SharedString::from(&t.name),
                                                symbol: slint::SharedString::from(
                                                    &t.symbol,
                                                ),
                                                balance: slint::SharedString::from(
                                                    format!("{:.4}", bal_f64),
                                                ),
                                                value_usd: slint::SharedString::from(
                                                    format!("${:.2}", t.usd_value),
                                                ),
                                            }
                                        })
                                        .collect();
                                    // Build available tokens list for send screen selector
                                    let mut avail: Vec<slint::SharedString> = vec![slint::SharedString::from("QUG")];
                                    for t in &token_list {
                                        if t.symbol != "QUG" {
                                            avail.push(t.symbol.clone());
                                        }
                                    }
                                    let avail_model = std::rc::Rc::new(slint::VecModel::from(avail));
                                    app.set_available_tokens(avail_model.into());

                                    // Update send balance display for currently selected token
                                    let sel = app.get_selected_token().to_string();
                                    if sel != "QUG" {
                                        if let Some(t) = token_list.iter().find(|t| t.symbol == sel.as_str()) {
                                            app.set_send_balance_display(slint::SharedString::from(
                                                format!("{} {}", t.balance, sel),
                                            ));
                                        }
                                    }

                                    let model =
                                        std::rc::Rc::new(slint::VecModel::from(token_list));
                                    app.set_tokens(model.into());
                                });
                            }
                            Err(e) => {
                                eprintln!("[Tokens] Fetch failed: {}", e);
                                // v8.2.8: Token endpoint failed — fall back to dedicated balance endpoint
                                let client_fb = client3;
                                let weak_fb = weak_bal_fallback;
                                let map_fb = balances_map_fallback;
                                tokio::spawn(async move {
                                    match client_fb.get_balance().await {
                                        Ok(bal) => {
                                            let balance_display = format!("{:.4}", bal.balance_qnk);
                                            let value_usd = format!("${:.2}", bal.balance_qnk * 3000.0);
                                            map_fb.lock().unwrap().insert("QUG".to_string(), balance_display.clone());
                                            let _ = slint::invoke_from_event_loop(move || {
                                                if let Some(app) = weak_fb.upgrade() {
                                                    app.set_qug_balance(slint::SharedString::from(&balance_display));
                                                    app.set_qug_value_usd(slint::SharedString::from(&value_usd));
                                                    if app.get_selected_token() == "QUG" {
                                                        app.set_send_balance_display(slint::SharedString::from(
                                                            format!("{} QUG", balance_display),
                                                        ));
                                                    }
                                                }
                                            });
                                        }
                                        Err(e2) => {
                                            eprintln!("[Balance] Fallback also failed: {}", e2);
                                        }
                                    }
                                });
                            }
                        }
                    });
                }
            },
        );

        // Keep timer alive by leaking it (it lives for the app lifetime)
        std::mem::forget(poll_timer);
    }

    if let Err(e) = app.run() {
        eprintln!("[ERROR] Application failed: {e}");
        eprintln!("[HINT] If this is an OpenGL error, try setting SLINT_BACKEND=winit-software");
        std::process::exit(1);
    }
}

fn get_server_url(app: &AppWindow) -> String {
    let url = app.get_server_url().to_string();
    if url.is_empty() {
        let args: Vec<String> = std::env::args().collect();
        args.get(1)
            .cloned()
            .unwrap_or_else(|| "https://quillon.xyz".to_string())
    } else {
        url
    }
}

fn truncate_address(addr: &str) -> String {
    if addr.len() > 16 {
        format!("{}...{}", &addr[..10], &addr[addr.len() - 6..])
    } else {
        addr.to_string()
    }
}

fn generate_qr_image(data: &str) -> slint::Image {
    use qrcode::QrCode;

    let code = QrCode::new(data.as_bytes()).unwrap();
    let qr_width = code.width();

    // Scale QR modules to fill ~240px image
    let scale = std::cmp::max(1, 240 / qr_width);
    let img_size = qr_width * scale;

    let mut buffer =
        slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(img_size as u32, img_size as u32);
    let pixels = buffer.make_mut_bytes();

    for y in 0..img_size {
        for x in 0..img_size {
            let qr_x = x / scale;
            let qr_y = y / scale;
            let dark = code[(qr_x, qr_y)] == qrcode::Color::Dark;
            let idx = (y * img_size + x) * 4;
            let val = if dark { 0u8 } else { 255u8 };
            pixels[idx] = val;     // R
            pixels[idx + 1] = val; // G
            pixels[idx + 2] = val; // B
            pixels[idx + 3] = 255; // A
        }
    }

    slint::Image::from_rgba8(buffer)
}

fn format_hashrate(h: u64) -> String {
    if h >= 1_000_000 {
        format!("{:.1}M", h as f64 / 1_000_000.0)
    } else if h >= 1_000 {
        format!("{:.1}K", h as f64 / 1_000.0)
    } else {
        format!("{}", h)
    }
}

/// Authentication info for SSE connections.
/// v8.2.7: SSE now authenticates to receive wallet-specific balance events.
enum SseAuth {
    /// Ed25519 wallet signature (generates fresh X-Wallet-Auth on each reconnect)
    Wallet(Arc<Wallet>),
    /// OAuth2 Bearer token
    Bearer(String),
}

/// v8.0.1: Start SSE listener for real-time balance updates.
/// Connects to /api/v1/events?wallet_address={hex} and pushes balance changes to UI instantly.
/// v8.2.7: Now authenticates SSE requests + heartbeat timeout for dead connection detection.
fn start_sse_listener(
    rt_handle: &tokio::runtime::Handle,
    app_weak: slint::Weak<AppWindow>,
    server_url: String,
    wallet_address_hex: String,
    running: Arc<AtomicBool>,
    auth: SseAuth,
) {
    // Stop any existing SSE listener
    running.store(true, Ordering::SeqCst);

    let running_clone = running.clone();
    rt_handle.spawn(async move {
        // v8.2.7: Use connect_timeout but no overall timeout (SSE is long-lived)
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .build()
            .unwrap_or_default();

        let sse_url = format!(
            "{}/api/v1/events?wallet_address={}",
            server_url.trim_end_matches('/'),
            wallet_address_hex
        );

        let mut backoff_secs = 3u64;
        const MAX_BACKOFF: u64 = 60;
        // v8.2.7: Heartbeat timeout — if no data in 120s, assume connection is dead
        // (nginx proxy_read_timeout is 300s, so 120s gives margin before nginx kills it)
        const HEARTBEAT_TIMEOUT_SECS: u64 = 120;

        println!("[SSE] Connecting to: {}", sse_url);

        loop {
            if !running_clone.load(Ordering::SeqCst) {
                println!("[SSE] Listener stopped");
                return;
            }

            // v8.2.7: Generate fresh auth header on each connection attempt
            let mut req = client
                .get(&sse_url)
                .header("Accept", "text/event-stream")
                .header("Cache-Control", "no-cache");

            req = match &auth {
                SseAuth::Wallet(wallet) => {
                    req.header("X-Wallet-Auth", wallet.auth_header("/api/v1/events"))
                }
                SseAuth::Bearer(token) => {
                    req.header("Authorization", format!("Bearer {}", token))
                }
            };

            let resp = match req.send().await {
                Ok(r) => {
                    backoff_secs = 3; // Reset backoff on successful connection
                    r
                }
                Err(e) => {
                    eprintln!("[SSE] Connection failed: {}, retrying in {}s...", e, backoff_secs);
                    tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
                    backoff_secs = (backoff_secs * 2).min(MAX_BACKOFF);
                    continue;
                }
            };

            // v8.2.7: Check for auth errors — server may return 401/403
            if resp.status().as_u16() == 401 || resp.status().as_u16() == 403 {
                eprintln!("[SSE] Auth rejected ({}), retrying in {}s...", resp.status(), backoff_secs);
                tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(MAX_BACKOFF);
                continue;
            }

            println!("[SSE] Connected (status {}), streaming events...", resp.status());

            // Read SSE stream line by line
            let mut event_type = String::new();
            let mut event_data = String::new();

            use futures_util::StreamExt;
            let mut stream = resp.bytes_stream();

            let mut buf = String::new();
            loop {
                if !running_clone.load(Ordering::SeqCst) {
                    return;
                }

                // v8.2.7: Wrap stream.next() in a timeout to detect dead connections
                let chunk_result = match tokio::time::timeout(
                    std::time::Duration::from_secs(HEARTBEAT_TIMEOUT_SECS),
                    stream.next(),
                )
                .await
                {
                    Ok(Some(result)) => result,
                    Ok(None) => {
                        // Stream ended normally
                        break;
                    }
                    Err(_) => {
                        // Heartbeat timeout — no data in 120s, connection is likely dead
                        eprintln!("[SSE] No data in {}s, reconnecting...", HEARTBEAT_TIMEOUT_SECS);
                        break;
                    }
                };

                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("[SSE] Stream error: {}", e);
                        break; // Connection error, reconnect
                    }
                };

                buf.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete lines
                while let Some(newline_pos) = buf.find('\n') {
                    let line = buf[..newline_pos].trim_end_matches('\r').to_string();
                    buf = buf[newline_pos + 1..].to_string();

                    if line.starts_with("event:") {
                        event_type = line[6..].trim().to_string();
                    } else if line.starts_with("data:") {
                        event_data = line[5..].trim().to_string();
                    } else if line.is_empty() && !event_type.is_empty() && !event_data.is_empty() {
                        // Complete SSE event received
                        handle_sse_event(&app_weak, &event_type, &event_data);
                        event_type.clear();
                        event_data.clear();
                    }
                }
            }

            // Stream ended, reconnect after delay
            eprintln!("[SSE] Stream ended, reconnecting in {}s...", backoff_secs);
            tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
        }
    });
}

/// Handle a single SSE event, updating the UI if relevant.
/// Server sends adjacently-tagged enum: {"type":"BalanceUpdated","data":{...}}
fn handle_sse_event(
    app_weak: &slint::Weak<AppWindow>,
    event_type: &str,
    event_data: &str,
) {
    match event_type {
        "balance-updated" => {
            // Server sends {"type":"BalanceUpdated","data":{"wallet_address":...,"new_balance":...}}
            // Extract the inner "data" object first, then parse as SseBalanceUpdate
            let balance_update = serde_json::from_str::<serde_json::Value>(event_data)
                .ok()
                .and_then(|v| {
                    // Try unwrapping {"type":...,"data":{...}} envelope
                    if let Some(inner) = v.get("data") {
                        serde_json::from_value::<models::SseBalanceUpdate>(inner.clone()).ok()
                    } else {
                        // Fallback: try parsing as flat SseBalanceUpdate (no envelope)
                        serde_json::from_value::<models::SseBalanceUpdate>(v).ok()
                    }
                });

            if let Some(update) = balance_update {
                println!("[SSE] Balance update: {:.4} QUG", update.new_balance);
                let balance_display = format!("{:.4}", update.new_balance);
                let value_usd = format!("${:.2}", update.new_balance * 3000.0);
                let weak = app_weak.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(app) = weak.upgrade() {
                        app.set_qug_balance(slint::SharedString::from(&balance_display));
                        app.set_qug_value_usd(slint::SharedString::from(&value_usd));
                        app.set_send_balance_display(slint::SharedString::from(
                            format!("{} QUG", balance_display),
                        ));
                    }
                });
            } else {
                eprintln!("[SSE] Failed to parse balance-updated event: {}", event_data);
            }
        }
        "token-balance-updated" => {
            // Token balance updates are handled by the polling timer
            // (SSE gives us the signal, polling fetches the full list)
        }
        _ => {} // Ignore other event types
    }
}

/// Open a URL in the system's default browser.
fn open_browser(url: &str) {
    #[cfg(target_os = "windows")]
    {
        // rundll32 avoids cmd.exe mangling & characters in URLs
        let _ = std::process::Command::new("rundll32")
            .args(["url.dll,FileProtocolHandler", url])
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(url).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}
