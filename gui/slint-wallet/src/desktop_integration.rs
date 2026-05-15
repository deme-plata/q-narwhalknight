//! Linux desktop integration: install application menu entry, icon, and autostart.
//!
//! On first launch (and idempotently thereafter) writes:
//!   ~/.local/share/applications/quillon-wallet.desktop  — app menu launcher
//!   ~/.local/share/icons/hicolor/256x256/apps/quillon-wallet.png  — icon
//!   ~/.config/autostart/quillon-wallet.desktop  — auto-launch on login
//!
//! macOS and Windows are no-ops for now (platform-specific bundle/registry work
//! belongs in a separate module when those targets are wired up).
//!
//! TODO(tray): autostart currently launches the full wallet window. Once the
//! system-tray feature lands, the autostart Exec should add `--minimized` and
//! the wallet should honor that flag by starting hidden in the tray.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

const APP_ID: &str = "quillon-wallet";
const APP_NAME: &str = "Quillon Wallet";
const APP_COMMENT: &str = "Quillon (Q-NarwhalKnight) post-quantum wallet";
const ICON_BYTES: &[u8] = include_bytes!("../ui/icons/quillon-logo.png");

#[cfg(not(target_os = "linux"))]
pub fn install_desktop_integration() {
    // Non-Linux platforms: no-op for now.
}

#[cfg(target_os = "linux")]
pub fn install_desktop_integration() {
    if let Err(e) = install_inner() {
        eprintln!("[desktop-integration] WARN: {}", e);
    }
}

#[cfg(target_os = "linux")]
fn install_inner() -> Result<(), String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let exe = std::env::current_exe()
        .map_err(|e| format!("current_exe: {e}"))?
        .to_string_lossy()
        .into_owned();

    let icon_dir: PathBuf = [&home, ".local", "share", "icons", "hicolor", "256x256", "apps"]
        .iter()
        .collect();
    let app_dir: PathBuf = [&home, ".local", "share", "applications"].iter().collect();
    let autostart_dir: PathBuf = [&home, ".config", "autostart"].iter().collect();

    fs::create_dir_all(&icon_dir).map_err(|e| format!("mkdir icon_dir: {e}"))?;
    fs::create_dir_all(&app_dir).map_err(|e| format!("mkdir app_dir: {e}"))?;
    fs::create_dir_all(&autostart_dir).map_err(|e| format!("mkdir autostart_dir: {e}"))?;

    // Icon
    let icon_path = icon_dir.join(format!("{APP_ID}.png"));
    write_atomic(&icon_path, ICON_BYTES).map_err(|e| format!("write icon: {e}"))?;

    // Application menu entry
    let menu_entry = format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name={name}\n\
         GenericName=Wallet\n\
         Comment={comment}\n\
         Exec={exe} %U\n\
         Icon={icon_name}\n\
         Terminal=false\n\
         Categories=Office;Finance;Network;\n\
         StartupNotify=true\n\
         StartupWMClass={wmclass}\n\
         Keywords=quillon;wallet;crypto;qug;quantum;\n",
        name = APP_NAME,
        comment = APP_COMMENT,
        exe = exe,
        icon_name = APP_ID,
        wmclass = APP_ID,
    );
    let menu_path = app_dir.join(format!("{APP_ID}.desktop"));
    write_atomic(&menu_path, menu_entry.as_bytes())
        .map_err(|e| format!("write menu .desktop: {e}"))?;

    // Autostart entry (passes --autostart so future tray code can decide to start hidden)
    let autostart_entry = format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name={name}\n\
         Comment={comment}\n\
         Exec={exe} --autostart\n\
         Icon={icon_name}\n\
         Terminal=false\n\
         Hidden=false\n\
         NoDisplay=false\n\
         X-GNOME-Autostart-enabled=true\n",
        name = APP_NAME,
        comment = APP_COMMENT,
        exe = exe,
        icon_name = APP_ID,
    );
    let autostart_path = autostart_dir.join(format!("{APP_ID}.desktop"));
    write_atomic(&autostart_path, autostart_entry.as_bytes())
        .map_err(|e| format!("write autostart .desktop: {e}"))?;

    eprintln!("[desktop-integration] OK — wrote {menu_path:?}, {icon_path:?}, {autostart_path:?}");
    Ok(())
}

#[cfg(target_os = "linux")]
fn write_atomic(path: &PathBuf, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)
}
