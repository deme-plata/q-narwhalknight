#[allow(dead_code)]
mod api_client;
mod miner;
#[allow(dead_code)]
mod models;
#[allow(dead_code)]
mod wallet;

use std::sync::atomic::Ordering;
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

    let app = AppWindow::new().unwrap();
    let app_weak = app.as_weak();

    // Shared state
    let wallet: Arc<std::sync::Mutex<Option<Arc<Wallet>>>> =
        Arc::new(std::sync::Mutex::new(None));
    let api_client: Arc<std::sync::Mutex<Option<Arc<ApiClient>>>> =
        Arc::new(std::sync::Mutex::new(None));
    let miner_state = Arc::new(MinerState::new());

    // Tokio runtime for async API calls
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");
    let rt_handle = rt.handle().clone();

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
                    *wallet.lock().unwrap() = Some(w);
                    *api_client.lock().unwrap() = Some(client);
                    app.set_active_screen(1);
                    app.set_show_create_view(false);
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
        app.on_import_wallet(move |mnemonic| {
            let app = app_weak.upgrade().unwrap();
            let server_url = get_server_url(&app);

            match Wallet::from_mnemonic(mnemonic.as_str()) {
                Ok(w) => {
                    let w = Arc::new(w);
                    let client = Arc::new(ApiClient::new(&server_url, w.clone()));
                    let addr = w.address().to_string();
                    app.set_wallet_address(slint::SharedString::from(&addr));
                    app.set_qr_code_image(generate_qr_image(&addr));
                    *wallet.lock().unwrap() = Some(w);
                    *api_client.lock().unwrap() = Some(client);
                    app.set_active_screen(1);
                    app.set_login_error(slint::SharedString::from(""));
                }
                Err(e) => {
                    app.set_login_error(slint::SharedString::from(format!("{}", e)));
                }
            }
        });
    }

    // ── Send Transaction callback ──
    {
        let app_weak = app_weak.clone();
        let api_client = api_client.clone();
        let rt_handle = rt_handle.clone();
        app.on_send_transaction(move |recipient, amount, memo| {
            let app = app_weak.upgrade().unwrap();
            let client_lock = api_client.lock().unwrap();
            let Some(client) = client_lock.as_ref().cloned() else {
                app.set_send_status(slint::SharedString::from("Wallet not connected"));
                return;
            };
            drop(client_lock);

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

            let weak = app.as_weak();
            rt_handle.spawn(async move {
                let result = client.send_transaction(&recipient, &amount, memo_opt).await;
                let _ = slint::invoke_from_event_loop(move || {
                    let app = weak.upgrade().unwrap();
                    match result {
                        Ok(_) => {
                            app.set_send_status(slint::SharedString::from(
                                "Transaction sent!",
                            ));
                        }
                        Err(e) => {
                            app.set_send_status(slint::SharedString::from(format!(
                                "Error: {}",
                                e
                            )));
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

            let my_address = wallet_ref
                .lock()
                .unwrap()
                .as_ref()
                .map(|w| w.address().to_string())
                .unwrap_or_default();

            let weak = app_weak.clone();
            rt_handle.spawn(async move {
                let weak2 = weak.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    let app = weak2.upgrade().unwrap();
                    app.set_history_loading(true);
                });

                let records = client.get_history().await.unwrap_or_default();
                let _ = slint::invoke_from_event_loop(move || {
                    let app = weak.upgrade().unwrap();
                    let tx_model: Vec<TxRecord> = records
                        .iter()
                        .map(|r| {
                            let is_sent = r.from == my_address;
                            TxRecord {
                                direction: slint::SharedString::from(if is_sent {
                                    "Sent"
                                } else {
                                    "Received"
                                }),
                                amount: slint::SharedString::from(format!(
                                    "{:.4} QUG",
                                    r.amount
                                )),
                                counterparty: slint::SharedString::from(truncate_address(
                                    if is_sent { &r.to } else { &r.from },
                                )),
                                timestamp: slint::SharedString::from(&r.timestamp),
                                tx_hash: slint::SharedString::from(&r.id),
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

                let address = wallet_ref
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|w| w.address().to_string())
                    .unwrap_or_default();

                miner::start_mining(
                    miner_state.clone(),
                    client,
                    address,
                    rt_handle.clone(),
                );
                app.set_mining(true);
            }
        });
    }

    // ── Copy Address callback ──
    {
        let wallet_ref = wallet.clone();
        app.on_copy_address(move || {
            if let Some(w) = wallet_ref.lock().unwrap().as_ref() {
                let address = w.address().to_string();
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    let _ = clipboard.set_text(&address);
                }
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

                // Update miner stats from atomics (fast, no async)
                if miner_state.running.load(Ordering::Relaxed) {
                    let hr = miner_state.hashrate.load(Ordering::Relaxed);
                    app.set_hashrate(slint::SharedString::from(format_hashrate(hr)));
                    app.set_blocks_found(slint::SharedString::from(format!(
                        "{}",
                        miner_state.blocks_found.load(Ordering::Relaxed)
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
                    if let Ok(status) = client.get_status().await {
                        let _ = slint::invoke_from_event_loop(move || {
                            let app = weak.upgrade().unwrap();
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
                        });
                    } else {
                        let _ = slint::invoke_from_event_loop(move || {
                            let app = weak2.upgrade().unwrap();
                            app.set_connected(false);
                        });
                    }
                });

                // Fetch balances
                let weak_bal = app.as_weak();
                rt_handle.spawn(async move {
                    if let Ok(bal) = client2.get_balance().await {
                        let balance_display = format!("{:.4}", bal.balance_qnk);
                        let value_usd =
                            format!("${:.2}", bal.balance_qnk * 42.5); // Use QUG price estimate

                        let _ = slint::invoke_from_event_loop(move || {
                            let app = weak_bal.upgrade().unwrap();
                            app.set_qug_balance(slint::SharedString::from(&balance_display));
                            app.set_qug_value_usd(slint::SharedString::from(&value_usd));
                            app.set_send_balance_display(slint::SharedString::from(
                                format!("{} QUG", balance_display),
                            ));
                        });
                    }
                });

                // Fetch token balances
                let weak_tok = app.as_weak();
                let client3 = api_client.lock().unwrap().as_ref().cloned();
                if let Some(client3) = client3 {
                    rt_handle.spawn(async move {
                        if let Ok(tokens) = client3.get_token_balances().await {
                            let _ = slint::invoke_from_event_loop(move || {
                                let app = weak_tok.upgrade().unwrap();
                                let token_list: Vec<TokenInfo> = tokens
                                    .tokens
                                    .iter()
                                    .map(|t| {
                                        let bal_f64: f64 =
                                            t.balance.parse().unwrap_or(0.0);
                                        let divisor = 10f64
                                            .powi(2 * t.decimals as i32);
                                        let display_bal = bal_f64 / divisor;
                                        let usd = display_bal * t.price_usd;
                                        TokenInfo {
                                            name: slint::SharedString::from(&t.name),
                                            symbol: slint::SharedString::from(&t.symbol),
                                            balance: slint::SharedString::from(format!(
                                                "{:.4}",
                                                display_bal
                                            )),
                                            value_usd: slint::SharedString::from(format!(
                                                "${:.2}",
                                                usd
                                            )),
                                        }
                                    })
                                    .collect();
                                let model =
                                    std::rc::Rc::new(slint::VecModel::from(token_list));
                                app.set_tokens(model.into());
                            });
                        }
                    });
                }
            },
        );

        // Keep timer alive by leaking it (it lives for the app lifetime)
        std::mem::forget(poll_timer);
    }

    app.run().unwrap();
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
