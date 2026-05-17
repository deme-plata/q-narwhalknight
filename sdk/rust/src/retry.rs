//! Retry logic with exponential backoff

use crate::error::Result;
use std::time::Duration;
use tokio::time::sleep;

/// Retry a function with exponential backoff
pub async fn retry_with_backoff<F, T, E>(
    mut f: F,
    max_retries: u32,
    initial_delay: Duration,
) -> std::result::Result<T, E>
where
    F: FnMut() -> std::result::Result<T, E>,
{
    let mut attempt = 0;
    let mut delay = initial_delay;

    loop {
        match f() {
            Ok(result) => return Ok(result),
            Err(e) => {
                attempt += 1;
                if attempt >= max_retries {
                    return Err(e);
                }
                sleep(delay).await;
                delay *= 2; // Exponential backoff
            }
        }
    }
}
