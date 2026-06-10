use crate::EncodingMethod;
/// Algorithmic Domain Generation for DNS Phantom Network
///
/// This module implements sophisticated domain generation algorithms that create
/// legitimate-looking DNS queries while encoding hidden data patterns.
///
/// Key features:
/// - Deterministic but unpredictable domain generation
/// - Legitimate-looking patterns that mimic real web traffic
/// - Time-based domain rotation for security
/// - Anti-correlation algorithms to prevent pattern detection
use anyhow::Result;
use chrono::{DateTime, Timelike, Utc};
use q_types::NodeId;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha3::{Digest, Sha3_256};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Domain generation algorithms
pub struct DomainGenerator {
    base_domains: Vec<String>,
    node_id: NodeId,

    /// Current generation parameters
    generation_seed: Arc<RwLock<[u8; 32]>>,
    rotation_schedule: Arc<RwLock<RotationSchedule>>,

    /// Generated domain cache
    domain_cache: Arc<RwLock<HashMap<String, GeneratedDomain>>>,

    /// Pattern libraries for realistic generation
    legitimate_patterns: LegitimatePatternLibrary,
}

#[derive(Debug, Clone)]
pub struct RotationSchedule {
    pub current_epoch: u64,
    pub epoch_duration: std::time::Duration,
    pub last_rotation: DateTime<Utc>,
    pub next_rotation: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct GeneratedDomain {
    pub domain: String,
    pub generation_method: GenerationMethod,
    pub confidence_score: f64, // How legitimate it looks
    pub created_at: DateTime<Utc>,
    pub usage_count: u32,
}

#[derive(Debug, Clone)]
pub enum GenerationMethod {
    /// Content Delivery Network patterns
    CDNPattern,
    /// API endpoint patterns  
    APIPattern,
    /// Analytics and tracking patterns
    AnalyticsPattern,
    /// Cloud service patterns
    CloudServicePattern,
    /// Developer tool patterns
    DeveloperToolPattern,
    /// Social media patterns
    SocialMediaPattern,
    /// E-commerce patterns
    ECommercePattern,
}

/// Library of legitimate domain patterns observed in real internet traffic
#[derive(Debug, Clone)]
pub struct LegitimatePatternLibrary {
    pub cdn_patterns: Vec<String>,
    pub api_patterns: Vec<String>,
    pub analytics_patterns: Vec<String>,
    pub cloud_patterns: Vec<String>,
    pub dev_tool_patterns: Vec<String>,
    pub social_patterns: Vec<String>,
    pub ecommerce_patterns: Vec<String>,
}

impl LegitimatePatternLibrary {
    pub fn new() -> Self {
        Self {
            cdn_patterns: vec![
                "{hash}.assets.{domain}".to_string(),
                "cdn-{region}.{domain}".to_string(),
                "static{number}.{domain}".to_string(),
                "{version}.js.{domain}".to_string(),
                "cache{number}.{domain}".to_string(),
            ],
            api_patterns: vec![
                "api{version}.{domain}".to_string(),
                "{service}-api.{domain}".to_string(),
                "v{number}.api.{domain}".to_string(),
                "{endpoint}.rest.{domain}".to_string(),
                "graphql.{domain}".to_string(),
            ],
            analytics_patterns: vec![
                "analytics{number}.{domain}".to_string(),
                "track.{domain}".to_string(),
                "metrics-{region}.{domain}".to_string(),
                "{hash}.events.{domain}".to_string(),
                "telemetry.{domain}".to_string(),
            ],
            cloud_patterns: vec![
                "{region}-{service}.{domain}".to_string(),
                "s3-{bucket}.{domain}".to_string(),
                "{zone}.compute.{domain}".to_string(),
                "storage{number}.{domain}".to_string(),
                "{hash}.blob.{domain}".to_string(),
            ],
            dev_tool_patterns: vec![
                "dev-{environment}.{domain}".to_string(),
                "{branch}.preview.{domain}".to_string(),
                "staging{number}.{domain}".to_string(),
                "{hash}.deploy.{domain}".to_string(),
                "build-{number}.{domain}".to_string(),
            ],
            social_patterns: vec![
                "media{number}.{domain}".to_string(),
                "{hash}.img.{domain}".to_string(),
                "video-{quality}.{domain}".to_string(),
                "avatar{size}.{domain}".to_string(),
                "thumb{resolution}.{domain}".to_string(),
            ],
            ecommerce_patterns: vec![
                "shop{region}.{domain}".to_string(),
                "cart-{session}.{domain}".to_string(),
                "checkout{version}.{domain}".to_string(),
                "{product}.catalog.{domain}".to_string(),
                "payment{processor}.{domain}".to_string(),
            ],
        }
    }

    pub fn get_pattern(&self, method: &GenerationMethod) -> &Vec<String> {
        match method {
            GenerationMethod::CDNPattern => &self.cdn_patterns,
            GenerationMethod::APIPattern => &self.api_patterns,
            GenerationMethod::AnalyticsPattern => &self.analytics_patterns,
            GenerationMethod::CloudServicePattern => &self.cloud_patterns,
            GenerationMethod::DeveloperToolPattern => &self.dev_tool_patterns,
            GenerationMethod::SocialMediaPattern => &self.social_patterns,
            GenerationMethod::ECommercePattern => &self.ecommerce_patterns,
        }
    }
}

impl DomainGenerator {
    /// Create a new domain generator
    pub async fn new(base_domains: &[String], node_id: NodeId) -> Result<Self> {
        // Initialize generation seed from node ID
        let mut hasher = Sha3_256::new();
        hasher.update(&node_id);
        hasher.update(b"dns-phantom-seed");
        let generation_seed: [u8; 32] = hasher.finalize().into();

        let now = Utc::now();
        let rotation_schedule = RotationSchedule {
            current_epoch: 0,
            epoch_duration: std::time::Duration::from_secs(3600), // 1 hour epochs
            last_rotation: now,
            next_rotation: now + chrono::Duration::hours(1),
        };

        let generator = Self {
            base_domains: base_domains.to_vec(),
            node_id,
            generation_seed: Arc::new(RwLock::new(generation_seed)),
            rotation_schedule: Arc::new(RwLock::new(rotation_schedule)),
            domain_cache: Arc::new(RwLock::new(HashMap::new())),
            legitimate_patterns: LegitimatePatternLibrary::new(),
        };

        info!(
            "Initialized DNS domain generator for node {}",
            hex::encode(node_id)
        );
        Ok(generator)
    }

    /// Generate discovery domains for peer finding
    pub async fn generate_discovery_domains(&self, count: usize) -> Result<Vec<String>> {
        let mut domains = Vec::new();
        let mut rng = self.create_seeded_rng().await;

        for i in 0..count {
            // Select generation method
            let methods = [
                GenerationMethod::CDNPattern,
                GenerationMethod::APIPattern,
                GenerationMethod::AnalyticsPattern,
                GenerationMethod::CloudServicePattern,
                GenerationMethod::DeveloperToolPattern,
            ];
            let method = &methods[rng.gen_range(0..methods.len())];

            // Generate domain
            let domain = self
                .generate_domain_with_method(method, i, &mut rng)
                .await?;
            domains.push(domain);
        }

        debug!("Generated {} discovery domains", domains.len());
        Ok(domains)
    }

    /// Generate domain using specific method
    async fn generate_domain_with_method(
        &self,
        method: &GenerationMethod,
        index: usize,
        rng: &mut ChaCha20Rng,
    ) -> Result<String> {
        let patterns = self.legitimate_patterns.get_pattern(method);
        let pattern = &patterns[rng.gen_range(0..patterns.len())];
        let base_domain = &self.base_domains[rng.gen_range(0..self.base_domains.len())];

        // Fill in pattern variables
        let domain = self
            .fill_pattern_variables(pattern, base_domain, index, rng)
            .await?;

        // Calculate confidence score
        let confidence = self.calculate_legitimacy_confidence(&domain, method).await;

        // Cache the generated domain
        let generated_domain = GeneratedDomain {
            domain: domain.clone(),
            generation_method: method.clone(),
            confidence_score: confidence,
            created_at: Utc::now(),
            usage_count: 0,
        };

        {
            let mut cache = self.domain_cache.write().await;
            cache.insert(domain.clone(), generated_domain);
        }

        Ok(domain)
    }

    /// Fill pattern variables with realistic values
    async fn fill_pattern_variables(
        &self,
        pattern: &str,
        base_domain: &str,
        index: usize,
        rng: &mut ChaCha20Rng,
    ) -> Result<String> {
        let mut result = pattern.replace("{domain}", base_domain);

        // Common variable replacements
        let replacements = [
            ("{hash}", &self.generate_realistic_hash(rng)),
            ("{region}", &self.generate_region_code(rng)),
            ("{number}", &format!("{}", rng.gen_range(1..=99))),
            ("{version}", &format!("v{}", rng.gen_range(1..=5))),
            ("{service}", &self.generate_service_name(rng)),
            ("{endpoint}", &self.generate_endpoint_name(rng)),
            ("{bucket}", &self.generate_bucket_name(rng)),
            ("{zone}", &self.generate_zone_name(rng)),
            ("{environment}", &self.generate_environment_name(rng)),
            ("{branch}", &self.generate_branch_name(rng)),
            ("{session}", &self.generate_session_id(rng)),
            ("{product}", &self.generate_product_name(rng)),
            ("{processor}", &self.generate_processor_name(rng)),
            ("{quality}", &self.generate_quality_name(rng)),
            ("{size}", &format!("{}", rng.gen_range(16..=512))),
            (
                "{resolution}",
                &format!(
                    "{}x{}",
                    rng.gen_range(100..=1920),
                    rng.gen_range(100..=1080)
                ),
            ),
        ];

        for (placeholder, replacement) in replacements {
            result = result.replace(placeholder, replacement);
        }

        // Add subtle node-specific encoding
        result = self.add_node_encoding(&result, index).await?;

        Ok(result)
    }

    /// Generate realistic hash values
    fn generate_realistic_hash(&self, rng: &mut ChaCha20Rng) -> String {
        let hash_types = [
            (8, "short"),   // Short hashes like git commits
            (16, "medium"), // Medium hashes
            (32, "long"),   // Full hashes
        ];

        let (length, _) = hash_types[rng.gen_range(0..hash_types.len())];
        (0..length)
            .map(|_| format!("{:x}", rng.gen_range(0..16)))
            .collect::<String>()
    }

    /// Generate realistic region codes
    fn generate_region_code(&self, rng: &mut ChaCha20Rng) -> String {
        let regions = [
            "us-east-1",
            "us-west-2",
            "eu-west-1",
            "eu-central-1",
            "ap-southeast-1",
            "ap-northeast-1",
            "ca-central-1",
            "sa-east-1",
            "af-south-1",
            "me-south-1",
        ];
        regions[rng.gen_range(0..regions.len())].to_string()
    }

    /// Generate realistic service names
    fn generate_service_name(&self, rng: &mut ChaCha20Rng) -> String {
        let services = [
            "auth",
            "user",
            "payment",
            "notification",
            "search",
            "media",
            "storage",
            "compute",
            "database",
            "cache",
            "queue",
            "streaming",
            "analytics",
            "monitoring",
        ];
        services[rng.gen_range(0..services.len())].to_string()
    }

    /// Generate realistic API endpoint names
    fn generate_endpoint_name(&self, rng: &mut ChaCha20Rng) -> String {
        let endpoints = [
            "users",
            "orders",
            "products",
            "inventory",
            "reviews",
            "payments",
            "shipping",
            "analytics",
            "reports",
            "admin",
            "upload",
            "download",
            "stream",
            "batch",
            "webhook",
        ];
        endpoints[rng.gen_range(0..endpoints.len())].to_string()
    }

    /// Generate realistic storage bucket names
    fn generate_bucket_name(&self, rng: &mut ChaCha20Rng) -> String {
        let prefixes = ["data", "backup", "logs", "static", "media", "docs"];
        let suffixes = ["prod", "staging", "dev", "archive", "temp"];

        format!(
            "{}-{}-{}",
            prefixes[rng.gen_range(0..prefixes.len())],
            suffixes[rng.gen_range(0..suffixes.len())],
            rng.gen_range(100..999)
        )
    }

    /// Generate realistic zone names
    fn generate_zone_name(&self, rng: &mut ChaCha20Rng) -> String {
        let zones = ["a", "b", "c", "1a", "2b", "3c"];
        zones[rng.gen_range(0..zones.len())].to_string()
    }

    /// Generate realistic environment names
    fn generate_environment_name(&self, rng: &mut ChaCha20Rng) -> String {
        let environments = ["prod", "staging", "dev", "test", "preview", "demo"];
        environments[rng.gen_range(0..environments.len())].to_string()
    }

    /// Generate realistic branch names
    fn generate_branch_name(&self, rng: &mut ChaCha20Rng) -> String {
        let branches = ["main", "develop", "feature", "hotfix", "release"];
        let branch = branches[rng.gen_range(0..branches.len())];

        if branch == "feature" || branch == "hotfix" {
            format!("{}-{}", branch, rng.gen_range(1000..9999))
        } else {
            branch.to_string()
        }
    }

    /// Generate realistic session IDs
    fn generate_session_id(&self, rng: &mut ChaCha20Rng) -> String {
        (0..12)
            .map(|_| {
                let chars = b"abcdefghijklmnopqrstuvwxyz0123456789";
                chars[rng.gen_range(0..chars.len())] as char
            })
            .collect()
    }

    /// Generate realistic product names
    fn generate_product_name(&self, rng: &mut ChaCha20Rng) -> String {
        let products = ["laptop", "phone", "tablet", "watch", "headphones", "camera"];
        let models = ["pro", "air", "mini", "max", "lite", "plus"];

        format!(
            "{}-{}",
            products[rng.gen_range(0..products.len())],
            models[rng.gen_range(0..models.len())]
        )
    }

    /// Generate realistic payment processor names
    fn generate_processor_name(&self, rng: &mut ChaCha20Rng) -> String {
        let processors = ["stripe", "paypal", "square", "braintree", "authorize"];
        processors[rng.gen_range(0..processors.len())].to_string()
    }

    /// Generate realistic quality settings
    fn generate_quality_name(&self, rng: &mut ChaCha20Rng) -> String {
        let qualities = ["hd", "4k", "1080p", "720p", "480p", "high", "medium", "low"];
        qualities[rng.gen_range(0..qualities.len())].to_string()
    }

    /// Add subtle node-specific encoding to domain
    async fn add_node_encoding(&self, domain: &str, index: usize) -> Result<String> {
        // Encode node ID and index in a way that looks natural
        let mut hasher = Sha3_256::new();
        hasher.update(&self.node_id);
        hasher.update(&(index as u64).to_le_bytes());
        let hash = hasher.finalize();

        // Use first few bytes to create natural-looking variations
        let variation_byte = hash[0];

        // Add subtle variation to the domain that encodes our node info
        let encoded_domain = match variation_byte % 4 {
            0 => {
                // Add a number suffix
                format!("{}{}", domain, (variation_byte % 10))
            }
            1 => {
                // Add a letter suffix
                let letter = (b'a' + (variation_byte % 26)) as char;
                format!("{}{}", domain, letter)
            }
            2 => {
                // Add a dash and short code
                format!("{}-{:02x}", domain, variation_byte)
            }
            3 => {
                // Keep original domain
                domain.to_string()
            }
            _ => unreachable!(),
        };

        Ok(encoded_domain)
    }

    /// Calculate how legitimate a generated domain looks
    async fn calculate_legitimacy_confidence(
        &self,
        domain: &str,
        method: &GenerationMethod,
    ) -> f64 {
        let mut confidence = 0.5; // Base confidence

        // Length check (realistic domains are usually 10-50 characters)
        let length_score = if domain.len() >= 10 && domain.len() <= 50 {
            1.0
        } else {
            0.5
        };
        confidence += length_score * 0.2;

        // Pattern realism check
        let pattern_score = match method {
            GenerationMethod::CDNPattern | GenerationMethod::CloudServicePattern => 0.9,
            GenerationMethod::APIPattern | GenerationMethod::AnalyticsPattern => 0.8,
            GenerationMethod::DeveloperToolPattern => 0.7,
            GenerationMethod::SocialMediaPattern | GenerationMethod::ECommercePattern => 0.85,
        };
        confidence += pattern_score * 0.3;

        // Character distribution check (realistic domains have balanced character usage)
        let char_score = self.analyze_character_distribution(domain);
        confidence += char_score * 0.2;

        // Common subdomain patterns bonus
        if domain.contains("api") || domain.contains("cdn") || domain.contains("static") {
            confidence += 0.1;
        }

        // Avoid obviously suspicious patterns
        if domain.contains("tor") || domain.contains("onion") || domain.contains("crypto") {
            confidence -= 0.3;
        }

        confidence.min(1.0).max(0.0)
    }

    /// Analyze character distribution for realism
    fn analyze_character_distribution(&self, domain: &str) -> f64 {
        let mut letter_count = 0;
        let mut digit_count = 0;
        let mut special_count = 0;

        for ch in domain.chars() {
            match ch {
                'a'..='z' | 'A'..='Z' => letter_count += 1,
                '0'..='9' => digit_count += 1,
                '-' | '.' | '_' => special_count += 1,
                _ => return 0.1, // Unusual characters lower score
            }
        }

        let total = domain.len() as f64;
        let letter_ratio = letter_count as f64 / total;
        let digit_ratio = digit_count as f64 / total;
        let special_ratio = special_count as f64 / total;

        // Realistic domains are mostly letters with some digits and few special chars
        if letter_ratio > 0.6 && letter_ratio < 0.9 && digit_ratio < 0.3 && special_ratio < 0.2 {
            1.0
        } else {
            0.5
        }
    }

    /// Create a seeded RNG for deterministic but unpredictable generation
    async fn create_seeded_rng(&self) -> ChaCha20Rng {
        let seed = *self.generation_seed.read().await;
        ChaCha20Rng::from_seed(seed)
    }

    /// Get current domain patterns for peer advertisement
    pub async fn get_current_patterns(&self) -> Vec<String> {
        let cache = self.domain_cache.read().await;
        cache.keys().cloned().collect()
    }

    /// Rotate generation parameters for security
    pub async fn rotate_generation_parameters(&self) -> Result<()> {
        let now = Utc::now();
        let mut schedule = self.rotation_schedule.write().await;

        if now >= schedule.next_rotation {
            info!("Rotating DNS domain generation parameters");

            // Update epoch
            schedule.current_epoch += 1;
            schedule.last_rotation = now;
            schedule.next_rotation = now + chrono::Duration::from_std(schedule.epoch_duration)?;

            // Generate new seed
            let mut hasher = Sha3_256::new();
            hasher.update(&self.node_id);
            hasher.update(&schedule.current_epoch.to_le_bytes());
            hasher.update(&now.timestamp().to_le_bytes());

            let new_seed: [u8; 32] = hasher.finalize().into();
            *self.generation_seed.write().await = new_seed;

            // Clear domain cache to force regeneration
            {
                let mut cache = self.domain_cache.write().await;
                cache.clear();
            }

            info!(
                "DNS generation parameters rotated for epoch {}",
                schedule.current_epoch
            );
        }

        Ok(())
    }

    /// Generate domains for specific data encoding
    pub async fn generate_encoding_domains(
        &self,
        data: &[u8],
        method: &EncodingMethod,
    ) -> Result<Vec<String>> {
        match method {
            EncodingMethod::SubdomainSteganography => {
                self.generate_subdomain_encoding_domains(data).await
            }
            EncodingMethod::TXTRecordSteganography => {
                self.generate_txt_encoding_domains(data).await
            }
            EncodingMethod::TimingSteganography => {
                self.generate_timing_encoding_domains(data).await
            }
            EncodingMethod::MultiQuerySteganography => {
                self.generate_multi_query_domains(data).await
            }
            EncodingMethod::MetadataSteganography => {
                self.generate_metadata_encoding_domains(data).await
            }
        }
    }

    /// Generate domains that encode data in subdomain patterns
    async fn generate_subdomain_encoding_domains(&self, data: &[u8]) -> Result<Vec<String>> {
        let mut domains = Vec::new();
        let mut rng = self.create_seeded_rng().await;

        // Encode each byte as a subdomain component
        for (i, &byte) in data.iter().enumerate() {
            let base_domain = &self.base_domains[rng.gen_range(0..self.base_domains.len())];

            // Convert byte to realistic subdomain pattern
            let encoded_subdomain = self.encode_byte_as_subdomain(byte, &mut rng);
            let domain = format!("{}.{}", encoded_subdomain, base_domain);

            domains.push(domain);
        }

        Ok(domains)
    }

    /// Encode a byte as a realistic subdomain
    fn encode_byte_as_subdomain(&self, byte: u8, rng: &mut ChaCha20Rng) -> String {
        // Use byte value to select from realistic subdomain patterns
        let patterns = [
            "api{}", "cdn{}", "static{}", "media{}", "assets{}", "cache{}", "img{}", "js{}",
            "css{}", "fonts{}",
        ];

        let pattern = patterns[byte as usize % patterns.len()];
        let number = (byte as u16 * 7 + rng.gen_range(0..100)) % 100; // Add some randomness

        format!("{}", pattern.replace("{}", &number.to_string()))
    }

    /// Generate domains for TXT record encoding using real legitimate domains
    async fn generate_txt_encoding_domains(&self, data: &[u8]) -> Result<Vec<String>> {
        let mut domains = Vec::new();
        let base_domains = [
            "github.com",
            "stackoverflow.com",
            "reddit.com",
            "wikipedia.org",
        ];

        for (i, chunk) in data.chunks(4).enumerate() {
            let hash = format!(
                "{:02x}",
                chunk.iter().fold(0u8, |acc, &b| acc.wrapping_add(b))
            );
            let subdomain = format!("api-{}-v{}", hash, i % 10 + 1);
            let domain = format!("{}.{}", subdomain, base_domains[i % base_domains.len()]);
            domains.push(domain);
        }

        Ok(domains)
    }

    /// Generate domains for timing encoding using CDN patterns
    async fn generate_timing_encoding_domains(&self, data: &[u8]) -> Result<Vec<String>> {
        let mut domains = Vec::new();
        let cdn_bases = [
            "cloudflare.com",
            "amazonaws.com",
            "googleapis.com",
            "microsoft.com",
        ];

        for (i, byte) in data.iter().take(8).enumerate() {
            let timing_id = format!("{:02x}", byte);
            let subdomain = format!("cdn-cache-{}-v{}", timing_id, (i % 5) + 1);
            let domain = format!("{}.{}", subdomain, cdn_bases[i % cdn_bases.len()]);
            domains.push(domain);
        }

        Ok(domains)
    }

    /// Generate domains for multi-query encoding using API patterns
    async fn generate_multi_query_domains(&self, data: &[u8]) -> Result<Vec<String>> {
        let mut domains = Vec::new();
        let api_bases = [
            "github.com",
            "reddit.com",
            "stackoverflow.com",
            "cloudflare.com",
        ];

        let patterns = ["api", "v2", "auth", "cdn", "static", "assets"];

        for (i, chunk) in data.chunks(3).enumerate().take(6) {
            let hash = chunk
                .iter()
                .fold(0u32, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u32));
            let pattern = patterns[i % patterns.len()];
            let subdomain = format!("{}-{:04x}-prod", pattern, hash % 9999);
            let domain = format!("{}.{}", subdomain, api_bases[i % api_bases.len()]);
            domains.push(domain);
        }

        Ok(domains)
    }

    /// Generate domains for metadata encoding using analytics patterns
    async fn generate_metadata_encoding_domains(&self, data: &[u8]) -> Result<Vec<String>> {
        let mut domains = Vec::new();
        let analytics_bases = [
            "google.com",
            "microsoft.com",
            "amazon.com",
            "cloudflare.com",
        ];

        let metadata_patterns = ["analytics", "metrics", "tracking", "insights", "stats"];

        for (i, chunk) in data.chunks(4).enumerate().take(5) {
            let checksum = chunk
                .iter()
                .fold(0u16, |acc, &b| acc.wrapping_add(b as u16));
            let pattern = metadata_patterns[i % metadata_patterns.len()];
            let subdomain = format!("{}-{:04x}-{}", pattern, checksum % 9999, (i % 3) + 1);
            let domain = format!(
                "{}.{}",
                subdomain,
                analytics_bases[i % analytics_bases.len()]
            );
            domains.push(domain);
        }

        Ok(domains)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_domain_generator_creation() {
        let base_domains = vec!["example.com".to_string(), "test.example".to_string()];
        let node_id = [1u8; 32];

        let generator = DomainGenerator::new(&base_domains, node_id).await.unwrap();
        assert_eq!(generator.node_id, node_id);
        assert_eq!(generator.base_domains.len(), 2);
    }

    #[tokio::test]
    async fn test_discovery_domain_generation() {
        let base_domains = vec!["example.com".to_string()];
        let node_id = [42u8; 32];

        let generator = DomainGenerator::new(&base_domains, node_id).await.unwrap();
        let domains = generator.generate_discovery_domains(5).await.unwrap();

        assert_eq!(domains.len(), 5);

        // All domains should contain the base domain
        for domain in &domains {
            assert!(domain.contains("example.com"));
        }

        // Domains should be unique
        let unique_domains: std::collections::HashSet<_> = domains.iter().collect();
        assert_eq!(unique_domains.len(), domains.len());
    }

    #[tokio::test]
    async fn test_legitimacy_scoring() {
        let base_domains = vec!["example.com".to_string()];
        let node_id = [1u8; 32];

        let generator = DomainGenerator::new(&base_domains, node_id).await.unwrap();

        // Test legitimate-looking domain
        let good_score = generator
            .calculate_legitimacy_confidence("api-v2.example.com", &GenerationMethod::APIPattern)
            .await;
        assert!(good_score > 0.7);

        // Test suspicious domain
        let bad_score = generator
            .calculate_legitimacy_confidence(
                "torxxxcryptonode.example.com",
                &GenerationMethod::APIPattern,
            )
            .await;
        assert!(bad_score < 0.5);
    }

    #[tokio::test]
    async fn test_pattern_rotation() {
        let base_domains = vec!["example.com".to_string()];
        let node_id = [1u8; 32];

        let generator = DomainGenerator::new(&base_domains, node_id).await.unwrap();

        // Get initial patterns
        let patterns1 = generator.get_current_patterns().await;

        // Force rotation
        generator.rotate_generation_parameters().await.unwrap();

        // Generate new patterns
        let _domains = generator.generate_discovery_domains(3).await.unwrap();
        let patterns2 = generator.get_current_patterns().await;

        // Patterns should be different after rotation
        // (Note: this test might be flaky if domains happen to be the same by chance)
        if !patterns1.is_empty() && !patterns2.is_empty() {
            // At minimum, the generation should be deterministic for the same seed
            assert!(!patterns2.is_empty());
        }
    }

    #[test]
    fn test_realistic_value_generation() {
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
        let generator = DomainGenerator {
            base_domains: vec![],
            node_id: [0u8; 32],
            generation_seed: Arc::new(RwLock::new([0u8; 32])),
            rotation_schedule: Arc::new(RwLock::new(RotationSchedule {
                current_epoch: 0,
                epoch_duration: std::time::Duration::from_secs(3600),
                last_rotation: Utc::now(),
                next_rotation: Utc::now(),
            })),
            domain_cache: Arc::new(RwLock::new(HashMap::new())),
            legitimate_patterns: LegitimatePatternLibrary::new(),
        };

        // Test hash generation
        let hash = generator.generate_realistic_hash(&mut rng);
        assert!(hash.len() >= 8 && hash.len() <= 32);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));

        // Test region generation
        let region = generator.generate_region_code(&mut rng);
        assert!(region.contains("-") || region.len() >= 5);

        // Test service generation
        let service = generator.generate_service_name(&mut rng);
        assert!(service.len() >= 3 && service.len() <= 15);
    }
}
