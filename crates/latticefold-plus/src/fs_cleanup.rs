use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Best-effort fast deletion for huge intermediate dirs: rename then delete asynchronously.
///
/// - Constant-time on the hot path (rename).
/// - Uses an external `rm -rf` on Unix so cleanup can continue even if the caller exits.
pub fn fast_remove_dir_best_effort(dir: &Path) {
    if !dir.exists() {
        return;
    }
    let pid = std::process::id();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let trash = dir.with_extension(format!("trash.{pid}.{ts}"));
    if std::fs::rename(dir, &trash).is_ok() {
        #[cfg(unix)]
        {
            if std::process::Command::new("rm")
                .arg("-rf")
                .arg(&trash)
                .spawn()
                .is_err()
            {
                std::thread::spawn(move || {
                    let _ = std::fs::remove_dir_all(trash);
                });
            }
        }
        #[cfg(not(unix))]
        {
            std::thread::spawn(move || {
                let _ = std::fs::remove_dir_all(trash);
            });
        }
    } else {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Best-effort deletion, but moves the directory to a **global temp trash** first.
///
/// This is useful when you want `du -sh <parent>` to drop immediately: renaming a subdir to
/// `<subdir>.trash.*` keeps it inside the parent. This variant renames to a sibling under
/// `std::env::temp_dir()` so it is no longer counted under the original parent directory.
pub fn fast_remove_dir_best_effort_to_tmp(dir: &Path) {
    if !dir.exists() {
        return;
    }
    let pid = std::process::id();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let base = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("dir");
    let mut trash = std::env::temp_dir();
    trash.push(format!("{base}.trash.{pid}.{ts}"));

    // If rename fails (e.g. across filesystems), fall back to in-place best-effort.
    if std::fs::rename(dir, &trash).is_err() {
        fast_remove_dir_best_effort(dir);
        return;
    }

    #[cfg(unix)]
    {
        if std::process::Command::new("rm")
            .arg("-rf")
            .arg(&trash)
            .spawn()
            .is_err()
        {
            std::thread::spawn(move || {
                let _ = std::fs::remove_dir_all(trash);
            });
        }
    }
    #[cfg(not(unix))]
    {
        std::thread::spawn(move || {
            let _ = std::fs::remove_dir_all(trash);
        });
    }
}

