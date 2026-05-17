//! Cross-platform desktop notifications.
//!
//! Linux / macOS: backed by `notify-rust` (libnotify / NSUserNotification).
//! Windows: backed by `tauri-winrt-notification` (WinRT toast → Action Center).
//!
//! The API is intentionally minimal — `notify(title, body)` and an enum for
//! the notification category. Callers fire-and-forget; errors are logged and
//! swallowed so a missing notification daemon never breaks wallet flow.

#[derive(Debug, Clone, Copy)]
pub enum Category {
    MiningReward,
    IncomingTx,
    SendConfirmation,
    Info,
}

impl Category {
    fn summary_prefix(self) -> &'static str {
        match self {
            Category::MiningReward => "Mining reward",
            Category::IncomingTx => "Incoming transaction",
            Category::SendConfirmation => "Transaction sent",
            Category::Info => "Quillon Wallet",
        }
    }
}

/// Fire a desktop notification. Non-blocking, errors are logged.
pub fn notify(category: Category, body: &str) {
    let title = category.summary_prefix();
    if let Err(e) = notify_impl(title, body) {
        eprintln!("[notifications] WARN: {} → {}: {}", title, body, e);
    }
}

#[cfg(not(target_os = "windows"))]
fn notify_impl(title: &str, body: &str) -> Result<(), String> {
    use notify_rust::Notification;
    Notification::new()
        .summary(title)
        .body(body)
        .appname("Quillon Wallet")
        .icon("quillon-wallet") // matches the Icon= entry in the .desktop file
        .show()
        .map(|_| ())
        .map_err(|e| format!("notify-rust: {e}"))
}

#[cfg(target_os = "windows")]
fn notify_impl(title: &str, body: &str) -> Result<(), String> {
    use tauri_winrt_notification::Toast;
    Toast::new(Toast::POWERSHELL_APP_ID)
        .title(title)
        .text1(body)
        .show()
        .map_err(|e| format!("winrt-notification: {e}"))
}
