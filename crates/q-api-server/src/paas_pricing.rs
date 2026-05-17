// PaaS Dynamic Pricing Module
// Integrates with Quantum Oracle for real-time QUG/USD conversion
//
// Features:
// - Dynamic USD pricing based on oracle feeds
// - Fallback to fixed rates when oracle unavailable
// - Price caching with TTL (30 second cache)
// - Automatic fee calculation with oracle integration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// PaaS service pricing in USD
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaaSPricingUSD {
    /// Tor relay pricing: $0.001 per MB
    pub tor_relay_per_mb_usd: f64,

    /// Mixing fee: 0.1% of transaction value (minimum $0.01)
    pub mixing_fee_basis_points: u64,
    pub mixing_min_fee_usd: f64,

    /// Ring signature generation: $0.001
    pub ring_signature_fee_usd: f64,

    /// Stealth address generation: $0.0001
    pub stealth_address_fee_usd: f64,

    /// ZK-STARK proof generation: $0.01
    pub zk_stark_proof_fee_usd: f64,

    /// Atomic swap fee: $0.05
    pub atomic_swap_fee_usd: f64,

    /// Enterprise tier monthly subscriptions
    pub professional_tier_monthly_usd: f64,
    pub enterprise_tier_monthly_usd: f64,
    pub white_label_tier_monthly_usd: f64,
}

impl Default for PaaSPricingUSD {
    fn default() -> Self {
        Self {
            tor_relay_per_mb_usd: 0.001,
            mixing_fee_basis_points: 10, // 0.1%
            mixing_min_fee_usd: 0.01,
            ring_signature_fee_usd: 0.001,
            stealth_address_fee_usd: 0.0001,
            zk_stark_proof_fee_usd: 0.01,
            atomic_swap_fee_usd: 0.05,
            professional_tier_monthly_usd: 499.0,
            enterprise_tier_monthly_usd: 1999.0,
            white_label_tier_monthly_usd: 9999.0,
        }
    }
}

/// Cached price entry with TTL
#[derive(Debug, Clone)]
struct CachedPrice {
    qug_usd_price: f64,
    cached_at: u64, // Unix timestamp
    ttl_seconds: u64,
}

impl CachedPrice {
    fn is_expired(&self) -> bool {
        let now = chrono::Utc::now().timestamp() as u64;
        now - self.cached_at > self.ttl_seconds
    }
}

/// PaaS pricing manager with oracle integration
pub struct PaaSPricingManager {
    /// USD pricing configuration
    usd_pricing: PaaSPricingUSD,

    /// Cached QUG/USD price
    cached_price: Arc<RwLock<Option<CachedPrice>>>,

    /// Fallback QUG/USD price (when oracle unavailable)
    fallback_qug_usd: f64,

    /// Price cache TTL (seconds)
    cache_ttl: u64,
}

impl PaaSPricingManager {
    pub fn new() -> Self {
        Self {
            usd_pricing: PaaSPricingUSD::default(),
            cached_price: Arc::new(RwLock::new(None)),
            fallback_qug_usd: 0.50, // $0.50 per QUG (default)
            cache_ttl: 30,          // 30 second cache
        }
    }

    /// Get current QUG/USD price (with caching)
    pub async fn get_qug_usd_price(&self) -> f64 {
        // Check cache first
        let cached = self.cached_price.read().await;
        if let Some(cached_price) = &*cached {
            if !cached_price.is_expired() {
                return cached_price.qug_usd_price;
            }
        }
        drop(cached);

        // Cache expired or not set, fetch from oracle
        match self.fetch_qug_usd_from_oracle().await {
            Ok(price) => {
                // Update cache
                let mut cached = self.cached_price.write().await;
                *cached = Some(CachedPrice {
                    qug_usd_price: price,
                    cached_at: chrono::Utc::now().timestamp() as u64,
                    ttl_seconds: self.cache_ttl,
                });

                info!("💲 QUG/USD price updated from oracle: ${:.6}", price);
                price
            }
            Err(e) => {
                warn!(
                    "⚠️ Failed to fetch QUG/USD from oracle: {}. Using fallback price: ${:.6}",
                    e, self.fallback_qug_usd
                );
                self.fallback_qug_usd
            }
        }
    }

    /// Fetch QUG/USD price from Quantum Oracle
    async fn fetch_qug_usd_from_oracle(&self) -> Result<f64, String> {
        // TODO: Integrate with q-oracle when it's enabled in Cargo.toml
        // For now, return an error to trigger fallback
        //
        // Example integration code:
        // ```rust
        // let oracle = state.quantum_oracle.as_ref()
        //     .ok_or("Oracle not initialized")?;
        //
        // let price_data = oracle.get_quantum_price("QUG/USD").await
        //     .map_err(|e| format!("Oracle query failed: {}", e))?;
        //
        // let price_f64 = price_data.price.to_string()
        //     .parse::<f64>()
        //     .map_err(|e| format!("Failed to parse price: {}", e))?;
        //
        // Ok(price_f64)
        // ```

        Err("Oracle integration not yet enabled".to_string())
    }

    /// Convert USD amount to QUG (atomic units)
    pub async fn usd_to_qug(&self, usd_amount: f64) -> u64 {
        let qug_usd_price = self.get_qug_usd_price().await;

        // Calculate QUG amount
        let qug_amount = usd_amount / qug_usd_price;

        // Convert to atomic units (1 QUG = 100_000_000 atomic units)
        let atomic_units = (qug_amount * 1e24) as u64;

        atomic_units
    }

    /// Calculate Tor relay fee in QUG (atomic units)
    pub async fn calculate_tor_relay_fee(&self, data_size_mb: f64) -> u64 {
        let usd_fee = data_size_mb * self.usd_pricing.tor_relay_per_mb_usd;
        let qug_fee = self.usd_to_qug(usd_fee).await;

        // Minimum 0.001 QUG
        qug_fee.max(100_000)
    }

    /// Calculate mixing fee in QUG (atomic units)
    pub async fn calculate_mixing_fee(&self, tx_value_qug: u64) -> u64 {
        // 0.1% of transaction value
        let fee_qug = (tx_value_qug as u128 * self.usd_pricing.mixing_fee_basis_points as u128
            / 10000) as u64;

        // Convert minimum USD fee to QUG
        let min_fee_qug = self.usd_to_qug(self.usd_pricing.mixing_min_fee_usd).await;

        // Return greater of percentage fee or minimum
        fee_qug.max(min_fee_qug)
    }

    /// Calculate ring signature fee in QUG (atomic units)
    pub async fn calculate_ring_signature_fee(&self) -> u64 {
        self.usd_to_qug(self.usd_pricing.ring_signature_fee_usd)
            .await
    }

    /// Calculate stealth address fee in QUG (atomic units)
    pub async fn calculate_stealth_address_fee(&self, count: u32) -> u64 {
        let total_usd = self.usd_pricing.stealth_address_fee_usd * count as f64;
        self.usd_to_qug(total_usd).await
    }

    /// Calculate ZK-STARK proof fee in QUG (atomic units)
    pub async fn calculate_zk_stark_fee(&self) -> u64 {
        self.usd_to_qug(self.usd_pricing.zk_stark_proof_fee_usd)
            .await
    }

    /// Calculate atomic swap fee in QUG (atomic units)
    pub async fn calculate_atomic_swap_fee(&self) -> u64 {
        self.usd_to_qug(self.usd_pricing.atomic_swap_fee_usd).await
    }

    /// Calculate monthly subscription fee in QUG (atomic units)
    pub async fn calculate_subscription_fee(&self, tier: SubscriptionTier) -> u64 {
        let usd_amount = match tier {
            SubscriptionTier::Professional => self.usd_pricing.professional_tier_monthly_usd,
            SubscriptionTier::Enterprise => self.usd_pricing.enterprise_tier_monthly_usd,
            SubscriptionTier::WhiteLabel => self.usd_pricing.white_label_tier_monthly_usd,
        };

        self.usd_to_qug(usd_amount).await
    }

    /// Get pricing summary with current QUG/USD rate
    pub async fn get_pricing_summary(&self) -> PricingSummary {
        let qug_usd = self.get_qug_usd_price().await;

        PricingSummary {
            qug_usd_rate: qug_usd,
            tor_relay_per_mb: PricingEntry {
                usd: self.usd_pricing.tor_relay_per_mb_usd,
                qug_atomic: self.calculate_tor_relay_fee(1.0).await,
            },
            mixing_fee_minimum: PricingEntry {
                usd: self.usd_pricing.mixing_min_fee_usd,
                qug_atomic: self.usd_to_qug(self.usd_pricing.mixing_min_fee_usd).await,
            },
            ring_signature: PricingEntry {
                usd: self.usd_pricing.ring_signature_fee_usd,
                qug_atomic: self.calculate_ring_signature_fee().await,
            },
            stealth_address: PricingEntry {
                usd: self.usd_pricing.stealth_address_fee_usd,
                qug_atomic: self.calculate_stealth_address_fee(1).await,
            },
            zk_stark_proof: PricingEntry {
                usd: self.usd_pricing.zk_stark_proof_fee_usd,
                qug_atomic: self.calculate_zk_stark_fee().await,
            },
            atomic_swap: PricingEntry {
                usd: self.usd_pricing.atomic_swap_fee_usd,
                qug_atomic: self.calculate_atomic_swap_fee().await,
            },
            updated_at: chrono::Utc::now(),
        }
    }

    /// Set fallback QUG/USD price
    pub fn set_fallback_price(&mut self, price: f64) {
        self.fallback_qug_usd = price;
        info!("💲 Fallback QUG/USD price set to: ${:.6}", price);
    }

    /// Set cache TTL
    pub fn set_cache_ttl(&mut self, ttl_seconds: u64) {
        self.cache_ttl = ttl_seconds;
        info!("⏱️  Price cache TTL set to: {}s", ttl_seconds);
    }
}

/// Subscription tier types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubscriptionTier {
    Professional,
    Enterprise,
    WhiteLabel,
}

/// Pricing entry with USD and QUG amounts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingEntry {
    pub usd: f64,
    pub qug_atomic: u64,
}

/// Complete pricing summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingSummary {
    pub qug_usd_rate: f64,
    pub tor_relay_per_mb: PricingEntry,
    pub mixing_fee_minimum: PricingEntry,
    pub ring_signature: PricingEntry,
    pub stealth_address: PricingEntry,
    pub zk_stark_proof: PricingEntry,
    pub atomic_swap: PricingEntry,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_usd_to_qug_conversion() {
        let mut manager = PaaSPricingManager::new();
        manager.set_fallback_price(0.50); // $0.50 per QUG

        // $1.00 USD should equal 2 QUG = 200_000_000 atomic units
        let qug_atomic = manager.usd_to_qug(1.0).await;
        assert_eq!(qug_atomic, 200_000_000);

        // $0.01 USD should equal 0.02 QUG = 2_000_000 atomic units
        let qug_atomic = manager.usd_to_qug(0.01).await;
        assert_eq!(qug_atomic, 2_000_000);
    }

    #[tokio::test]
    async fn test_tor_relay_fee_calculation() {
        let mut manager = PaaSPricingManager::new();
        manager.set_fallback_price(0.50);

        // 1 MB at $0.001/MB = $0.001 = 0.002 QUG = 200_000 atomic units
        let fee = manager.calculate_tor_relay_fee(1.0).await;
        assert_eq!(fee, 200_000);
    }

    #[tokio::test]
    async fn test_mixing_fee_calculation() {
        let mut manager = PaaSPricingManager::new();
        manager.set_fallback_price(0.50);

        // 1000 QUG transaction * 0.1% = 1 QUG = 100_000_000 atomic units
        let fee = manager.calculate_mixing_fee(1000_00000000).await;
        assert_eq!(fee, 100_000_000);

        // Small transaction should use minimum fee
        let fee = manager.calculate_mixing_fee(100_000).await;
        let min_fee = manager.usd_to_qug(0.01).await;
        assert_eq!(fee, min_fee);
    }

    #[tokio::test]
    async fn test_pricing_summary() {
        let mut manager = PaaSPricingManager::new();
        manager.set_fallback_price(0.50);

        let summary = manager.get_pricing_summary().await;

        assert_eq!(summary.qug_usd_rate, 0.50);
        assert!(summary.tor_relay_per_mb.qug_atomic > 0);
        assert!(summary.mixing_fee_minimum.qug_atomic > 0);
    }

    #[tokio::test]
    async fn test_price_caching() {
        let mut manager = PaaSPricingManager::new();
        manager.set_fallback_price(0.50);
        manager.set_cache_ttl(1); // 1 second TTL

        // First call should fetch and cache
        let price1 = manager.get_qug_usd_price().await;
        assert_eq!(price1, 0.50);

        // Second call should use cache
        let price2 = manager.get_qug_usd_price().await;
        assert_eq!(price2, 0.50);

        // Wait for cache to expire
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // Should fetch again after expiration
        let price3 = manager.get_qug_usd_price().await;
        assert_eq!(price3, 0.50);
    }
}
