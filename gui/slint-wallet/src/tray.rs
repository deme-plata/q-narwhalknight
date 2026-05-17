//! Cross-platform system tray icon (Linux, macOS, Windows).
//!
//! Provides a basic tray menu: Show Wallet / Quit. Clicking the icon focuses
//! the wallet window. Closing the window minimizes to tray instead of exiting.
//!
//! Integrates with Slint's event loop via `slint::invoke_from_event_loop` —
//! tray events arrive on tray-icon's internal channel (a separate thread) and
//! we forward them to the UI thread.
//!
//! TODO: add Pause/Resume Mining menu items once we have a stable handle on
//! the miner state. Add per-token balance preview in the tray tooltip.

use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
    Icon, TrayIcon, TrayIconBuilder, TrayIconEvent,
};

const ICON_BYTES: &[u8] = include_bytes!("../ui/icons/quillon-logo.png");

/// Keep this alive for the lifetime of the wallet. Dropping it removes the
/// tray icon.
pub struct TrayHandle {
    _icon: TrayIcon,
}

/// Install the tray icon and wire its events to the Slint app window.
///
/// On success, returns a `TrayHandle` that the caller must keep alive until
/// exit. On failure (no system tray available, icon decode failure, etc.)
/// returns `None` and the wallet falls back to "no tray" mode.
pub fn install(weak_app: slint::Weak<crate::AppWindow>) -> Option<TrayHandle> {
    let icon = match load_icon() {
        Ok(i) => i,
        Err(e) => {
            eprintln!("[tray] WARN: failed to load icon: {e}");
            return None;
        }
    };

    let menu = Menu::new();
    let show_item = MenuItem::new("Show Wallet", true, None);
    let quit_item = MenuItem::new("Quit", true, None);
    if let Err(e) = menu.append_items(&[
        &show_item,
        &PredefinedMenuItem::separator(),
        &quit_item,
    ]) {
        eprintln!("[tray] WARN: failed to build menu: {e}");
        return None;
    }

    let show_id = show_item.id().clone();
    let quit_id = quit_item.id().clone();

    let tray = match TrayIconBuilder::new()
        .with_tooltip("Quillon Wallet")
        .with_icon(icon)
        .with_menu(Box::new(menu))
        .build()
    {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[tray] WARN: failed to build tray icon: {e}");
            return None;
        }
    };

    // Spawn a forwarder thread that reads tray events and posts them to the
    // Slint UI thread. tray-icon's events arrive on global channels.
    let menu_rx = MenuEvent::receiver();
    let tray_rx = TrayIconEvent::receiver();
    let weak_for_menu = weak_app.clone();
    let weak_for_click = weak_app;

    std::thread::spawn(move || loop {
        // Block on either channel; we just poll both with a short timeout.
        if let Ok(ev) = menu_rx.recv_timeout(std::time::Duration::from_millis(200)) {
            let weak = weak_for_menu.clone();
            let show_id = show_id.clone();
            let quit_id = quit_id.clone();
            let id = ev.id.clone();
            let _ = slint::invoke_from_event_loop(move || {
                if id == show_id {
                    if let Some(app) = weak.upgrade() {
                        show_window(&app);
                    }
                } else if id == quit_id {
                    let _ = slint::quit_event_loop();
                }
            });
        }
        if let Ok(ev) = tray_rx.recv_timeout(std::time::Duration::from_millis(50)) {
            // Left-click on the icon brings the wallet to front. Other events
            // (right-click, double-click) are handled by the OS-native menu.
            if let TrayIconEvent::Click { button, button_state, .. } = ev {
                use tray_icon::{MouseButton, MouseButtonState};
                if button == MouseButton::Left && button_state == MouseButtonState::Up {
                    let weak = weak_for_click.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(app) = weak.upgrade() {
                            show_window(&app);
                        }
                    });
                }
            }
        }
    });

    Some(TrayHandle { _icon: tray })
}

fn load_icon() -> Result<Icon, String> {
    let img = image::load_from_memory(ICON_BYTES).map_err(|e| format!("decode: {e}"))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Icon::from_rgba(rgba.into_raw(), w, h).map_err(|e| format!("from_rgba: {e}"))
}

fn show_window(app: &crate::AppWindow) {
    // slint's show() also raises and focuses the window on all backends.
    let _ = app.show();
}
