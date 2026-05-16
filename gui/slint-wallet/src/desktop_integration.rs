//! Cross-platform desktop integration.
//!
//! Linux: writes ~/.local/share/applications/quillon-wallet.desktop (app menu
//! launcher with `quillon://` protocol handler), the icon at
//! ~/.local/share/icons/hicolor/256x256/apps/quillon-wallet.png, and an
//! autostart entry at ~/.config/autostart/quillon-wallet.desktop.
//!
//! Windows: writes a Start Menu .lnk, a Desktop .lnk, the Run-key autostart
//! registry value, and the HKCU\Software\Classes\quillon protocol handler
//! registry tree.
//!
//! macOS: no-op for now (proper .app bundling belongs in build/packaging, not
//! at-runtime install).
//!
//! Every operation is idempotent — running the wallet repeatedly does not
//! accumulate state. Errors are logged to stderr and non-fatal.
//!
//! TODO(tray): autostart currently launches the full wallet window. Once the
//! tray feature gains minimize-to-tray on startup, the autostart Exec should
//! add `--minimized` and the wallet should honor that flag.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

const APP_ID: &str = "quillon-wallet";
const APP_NAME: &str = "Quillon Wallet";
const APP_COMMENT: &str = "Quillon (Q-NarwhalKnight) post-quantum wallet";
const ICON_BYTES: &[u8] = include_bytes!("../ui/icons/quillon-logo.png");

pub fn install_desktop_integration() {
    #[cfg(target_os = "linux")]
    if let Err(e) = install_linux() {
        eprintln!("[desktop-integration] WARN linux: {}", e);
    }
    #[cfg(target_os = "windows")]
    if let Err(e) = install_windows() {
        eprintln!("[desktop-integration] WARN windows: {}", e);
    }
    // macOS: see module docs.
}

#[cfg(target_os = "linux")]
fn install_linux() -> Result<(), String> {
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

    // Application menu entry (also registers as quillon:// protocol handler via MimeType)
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
         MimeType=x-scheme-handler/quillon;\n\
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

    // Best-effort: register the .desktop as the default handler for x-scheme-handler/quillon.
    // Silently ignore failures — most desktops will pick up MimeType on first menu rescan anyway.
    let _ = std::process::Command::new("xdg-mime")
        .args(["default", &format!("{APP_ID}.desktop"), "x-scheme-handler/quillon"])
        .status();
    let _ = std::process::Command::new("update-desktop-database")
        .arg(&app_dir)
        .status();

    eprintln!("[desktop-integration] OK linux — wrote {menu_path:?}, {icon_path:?}, {autostart_path:?}");
    Ok(())
}

#[cfg(target_os = "windows")]
fn install_windows() -> Result<(), String> {
    use mslnk::ShellLink;
    use winreg::enums::*;
    use winreg::RegKey;

    let exe = std::env::current_exe()
        .map_err(|e| format!("current_exe: {e}"))?;
    let exe_str = exe.to_string_lossy().into_owned();
    let appdata = std::env::var("APPDATA").map_err(|_| "APPDATA not set".to_string())?;
    let userprofile = std::env::var("USERPROFILE").map_err(|_| "USERPROFILE not set".to_string())?;

    // Start Menu shortcut
    let start_menu_dir: PathBuf = [&appdata, "Microsoft", "Windows", "Start Menu", "Programs"]
        .iter().collect();
    fs::create_dir_all(&start_menu_dir).map_err(|e| format!("mkdir start_menu: {e}"))?;
    let start_lnk = start_menu_dir.join(format!("{APP_NAME}.lnk"));
    let mut sl = ShellLink::new(&exe_str).map_err(|e| format!("ShellLink::new: {e}"))?;
    sl.set_name(Some(APP_NAME.to_string()));
    sl.set_icon_location(Some(exe_str.clone()));
    sl.create_lnk(&start_lnk).map_err(|e| format!("create_lnk start: {e}"))?;

    // Desktop shortcut
    let desktop_dir = PathBuf::from(&userprofile).join("Desktop");
    if desktop_dir.is_dir() {
        let desktop_lnk = desktop_dir.join(format!("{APP_NAME}.lnk"));
        let mut sl2 = ShellLink::new(&exe_str).map_err(|e| format!("ShellLink::new desktop: {e}"))?;
        sl2.set_name(Some(APP_NAME.to_string()));
        sl2.set_icon_location(Some(exe_str.clone()));
        sl2.create_lnk(&desktop_lnk).map_err(|e| format!("create_lnk desktop: {e}"))?;
    }

    // Autostart: HKCU\Software\Microsoft\Windows\CurrentVersion\Run
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let (run, _) = hkcu
        .create_subkey(r"Software\Microsoft\Windows\CurrentVersion\Run")
        .map_err(|e| format!("open Run key: {e}"))?;
    run.set_value(APP_NAME, &format!("\"{exe_str}\" --autostart"))
        .map_err(|e| format!("set Run value: {e}"))?;

    // Protocol handler: HKCU\Software\Classes\quillon
    let (proto, _) = hkcu
        .create_subkey(r"Software\Classes\quillon")
        .map_err(|e| format!("create quillon key: {e}"))?;
    proto.set_value("", &"URL:Quillon Protocol").map_err(|e| format!("set proto default: {e}"))?;
    proto.set_value("URL Protocol", &"").map_err(|e| format!("set URL Protocol: {e}"))?;
    let (icon, _) = proto
        .create_subkey("DefaultIcon")
        .map_err(|e| format!("create DefaultIcon: {e}"))?;
    icon.set_value("", &format!("\"{exe_str}\",0")).map_err(|e| format!("set icon: {e}"))?;
    let (cmd, _) = proto
        .create_subkey(r"shell\open\command")
        .map_err(|e| format!("create open\\command: {e}"))?;
    cmd.set_value("", &format!("\"{exe_str}\" \"%1\""))
        .map_err(|e| format!("set command: {e}"))?;

    eprintln!("[desktop-integration] OK windows — Start Menu, Run autostart, quillon:// handler installed");
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
