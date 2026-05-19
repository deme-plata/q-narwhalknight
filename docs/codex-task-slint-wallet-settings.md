# Slint wallet settings tab — feature brief for Codex

**Date**: 2026-05-19
**Status**: Brief, awaiting implementation
**Target version**: slint-wallet v10.10.0+
**Estimated effort**: 1-2 days

## Three concrete asks

1. New **Settings** tab in the slint wallet sidebar/nav, alongside the existing dashboard / dex / history / miner / chat tabs
2. **Autostart toggle** — user can disable the "launch on system boot" behavior currently always-on via `desktop_integration.rs`
3. **View private key / mnemonic** — PIN/password-gated reveal of the wallet's recovery phrase, with explicit destructive-action warning UX

## What already exists

- `gui/slint-wallet/src/desktop_integration.rs:55-107` — writes `~/.config/autostart/quillon-wallet.desktop` for Linux, `.lnk` files + Run-key for Windows. Currently runs unconditionally; never user-toggleable.
- `gui/slint-wallet/ui/app.slint` — main AppWindow with callbacks for create-wallet, import-wallet, send-transaction, toggle-mining, copy-address, etc. (see callbacks at lines 91-146).
- `gui/slint-wallet/src/wallet.rs` — wallet state + mnemonic storage. The mnemonic is currently stored in plaintext in the wallet keystore (verify and fix if needed).
- The slint app is set up as a Rust app with Slint UI — the pattern for adding a tab is to add a slint file + wire callbacks in main.rs.

## Required deliverables

### Deliverable 1: settings.slint

New file `gui/slint-wallet/ui/settings.slint`. A new Settings tab card with three sections:

```
┌─ Settings ────────────────────────────────────┐
│                                                │
│  Startup                                       │
│  [✓] Launch on system boot                    │
│      Currently enabled — disable to stop      │
│      auto-startup at login.                   │
│                                                │
│  Security                                      │
│  [Change PIN code]                            │
│      Last changed: 12 days ago                 │
│                                                │
│  [View recovery phrase]                       │
│      Reveals your 12-word mnemonic.           │
│      ⚠️ Anyone with this phrase controls       │
│      your wallet. Never share it.             │
│                                                │
│  Advanced                                      │
│  [Export wallet for backup]                   │
│  [Wipe wallet and start over]                 │
│                                                │
└────────────────────────────────────────────────┘
```

### Deliverable 2: Slint callbacks (wire in app.slint)

```slint
callback toggle-autostart(bool);
callback set-pin(string);                      // new PIN
callback verify-pin(string) -> bool;            // returns true if PIN matches stored hash
callback reveal-mnemonic(string) -> string;     // pass PIN, returns mnemonic on success
callback export-wallet-encrypted() -> string;   // returns encrypted blob
callback wipe-wallet();                          // destructive — requires double-confirm in UI

in property <bool> autostart-enabled: true;
in property <int>  days-since-pin-change: 0;
```

### Deliverable 3: Rust handlers

In `gui/slint-wallet/src/main.rs`:

```rust
ui.on_toggle_autostart(move |enabled| {
    if enabled {
        desktop_integration::install_autostart().ok();
    } else {
        desktop_integration::remove_autostart().ok();
    }
    config::set_autostart_enabled(enabled).ok();
});

ui.on_set_pin(move |new_pin| {
    let hash = argon2_hash(&new_pin); // use argon2 crate
    config::set_pin_hash(&hash).ok();
    config::touch_pin_changed_at().ok();
});

ui.on_verify_pin(move |pin| {
    let stored = config::get_pin_hash().unwrap_or_default();
    argon2_verify(&pin, &stored)
});

ui.on_reveal_mnemonic(move |pin| {
    if !verify_pin_internal(&pin) { return SharedString::default(); }
    wallet::read_mnemonic_after_pin_check().unwrap_or_default().into()
});
```

### Deliverable 4: Config persistence

Extend `gui/slint-wallet/src/config.rs` (create if missing) with:

```rust
// Persisted to ~/.config/quillon-wallet/settings.toml
pub struct WalletSettings {
    pub autostart_enabled: bool,        // default true (matches current behavior)
    pub pin_hash: Option<String>,        // argon2 hash, None = no PIN set
    pub pin_changed_at: Option<DateTime<Utc>>,
    pub theme: String,                   // "dark"|"light"|"system" (future)
}
```

### Deliverable 5: desktop_integration.rs `remove_autostart`

Currently only has `install_autostart`. Add the inverse — removes the `.desktop` file on Linux, removes Run-key entries on Windows, removes LaunchAgent on macOS.

## Acceptance criteria

1. Settings tab appears in slint-wallet nav and renders all three sections
2. Toggling autostart actually installs/removes the OS autostart entry, verifiable by checking `~/.config/autostart/quillon-wallet.desktop` exists or not
3. PIN is set, then "View recovery phrase" requires correct PIN entry; wrong PIN shows "incorrect PIN" with rate-limit (3 attempts before 1-minute lock)
4. Mnemonic is displayed in modal-style overlay with copy-to-clipboard button and a "Done — hide" button. After 30s of inactivity, the mnemonic display auto-hides.
5. PIN is hashed via argon2 (NOT plaintext, NOT bcrypt, NOT SHA-only). Argon2id with default tuning is fine.
6. All settings persist across wallet restarts (TOML at `~/.config/quillon-wallet/settings.toml`).

## Security considerations

- **PIN is never logged.** Add `#[derive(Zeroize)]` or use `secrecy::Secret<String>` to prevent it from sitting in memory longer than necessary.
- **Mnemonic display state** must not be persisted anywhere; only held in memory while modal is open.
- **PIN reset path**: if user forgets PIN, the only recovery is to wipe wallet and re-import from mnemonic. Document this in the UI ("Forgot PIN? Wipe wallet and re-import from your recovery phrase.")
- **Argon2 parameters**: t=2, m=64MB, p=1 is enough for desktop-class PIN protection. Adjust higher if testing shows < 100ms is too fast.
- **Rate-limit on PIN attempts**: 3 attempts → 1 minute lock → 5 attempts → 5 minute lock → 10 attempts → wipe-or-wait-1-hour prompt. Persist attempt counter to settings.toml so it survives restart.

## Files to read first

- `gui/slint-wallet/ui/app.slint` — existing AppWindow with callbacks (lines 91-146)
- `gui/slint-wallet/src/desktop_integration.rs` — autostart install/remove logic (currently only install)
- `gui/slint-wallet/src/wallet.rs` — wallet/mnemonic storage
- `gui/slint-wallet/src/main.rs` — callback registration patterns

## Suggested PR title

`feat(slint-wallet): Settings tab — autostart toggle + PIN lock + mnemonic reveal with rate-limit`

---

*Drafted by Claude Opus 4.7. Same pattern as the dragon-ball briefs and the AFL-1 spec — should land in a single focused PR.*
