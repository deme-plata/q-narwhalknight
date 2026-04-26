/// Water Robot Swarm — Proof-of-Biosynthesis consensus.
///
/// Each DropletNode is a DNA-programmed water droplet that evaluates
/// proposed swaps. A swap is approved only if >66% of total DNA mass votes YES.
///
/// The six species (from the FCC-ee water robot research doc):
///   Quantum Jellyfish   — Market maker, LP provider
///   Entangled Dolphin   — Arbitrageur
///   Tunneling Octopus   — DCA executor (primary for this bot)
///   Wave-Particle Whale — Large order splitter
///   Superposition Seahorse — Price oracle
///   Nano Quantumonas    — High-frequency micro-trader

use crate::dna::DnaChain;
use crate::resonance::ResonanceState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Species {
    QuantumJellyfish,
    EntangledDolphin,
    TunnelingOctopus,
    WaveParticleWhale,
    SuperpositionSeahorse,
    NanoQuantumonas,
}

impl Species {
    pub fn emoji(&self) -> &'static str {
        match self {
            Species::QuantumJellyfish    => "🪼",
            Species::EntangledDolphin    => "🐬",
            Species::TunnelingOctopus    => "🐙",
            Species::WaveParticleWhale   => "🐋",
            Species::SuperpositionSeahorse => "🦭",
            Species::NanoQuantumonas     => "🦠",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropletNode {
    pub id: usize,
    pub species: Species,
    /// DNA blockchain — encodes trade history
    pub dna: DnaChain,
    /// Position (price level this droplet "inhabits")
    pub price_anchor: f64,
    /// Energy: fraction of swap budget this droplet manages (0-1)
    pub energy: f64,
    /// Size in nanoliters (grows with successful trades, splits at 100nL)
    pub size_nl: f64,
}

impl DropletNode {
    pub fn new(id: usize, species: Species, price_anchor: f64) -> Self {
        // Seed DNA mass with some random bases so new nodes still have mass
        let mut dna = DnaChain::new();
        // Give each droplet a small initial sequence proportional to id (diversity)
        for i in 0..(id * 3 + 2) as u32 {
            dna.record_swap(price_anchor * (1.0 + i as f64 * 0.001), 0.001, 0.0);
        }
        DropletNode {
            id,
            species,
            dna,
            price_anchor,
            energy: 1.0 / 6.0, // equal share among 6 droplets
            size_nl: 10.0 + id as f64 * 5.0,
        }
    }

    /// Each droplet evaluates a proposed swap and returns YES or NO.
    /// Evaluation criteria depend on species role.
    pub fn evaluate(
        &self,
        resonance: &ResonanceState,
        proposed_amount_display: f64,
        current_price: f64,
        fcc_threshold: f64,
    ) -> bool {
        match self.species {
            Species::TunnelingOctopus => {
                // DCA executor: vote YES if resonance threshold met
                resonance.efficiency >= fcc_threshold
            }
            Species::EntangledDolphin => {
                // Arbitrageur: vote YES if price deviates >0.5% from anchor
                let deviation = (current_price - self.price_anchor).abs() / self.price_anchor;
                deviation > 0.005 && resonance.efficiency >= fcc_threshold * 0.8
            }
            Species::QuantumJellyfish => {
                // Market maker: prefer balanced pools, always cautious
                resonance.ratio > 0.9 && resonance.ratio < 1.1
            }
            Species::WaveParticleWhale => {
                // Only approve if amount is ≤ 5% of reserve (prevent self-slippage)
                let reserve_estimate = current_price * 10_000.0; // rough
                proposed_amount_display / reserve_estimate < 0.05
                    && resonance.efficiency >= fcc_threshold
            }
            Species::SuperpositionSeahorse => {
                // Oracle: vote YES when price is within 2% of its known anchor
                let deviation = (current_price - self.price_anchor).abs() / self.price_anchor;
                deviation < 0.02
            }
            Species::NanoQuantumonas => {
                // HFT: almost always YES for small amounts
                proposed_amount_display < 1.0 && resonance.efficiency >= 0.3
            }
        }
    }

    /// Record the outcome of a swap into this droplet's DNA chain.
    pub fn record_outcome(&mut self, price: f64, amount: f64, profit_pct: f64, rng_seed: u64) {
        self.dna.record_swap(price, amount, profit_pct);
        self.dna.maybe_mutate(profit_pct, rng_seed);

        // Binary fission: at 100 nL, the droplet splits (reset size, double energy weight)
        self.size_nl += amount * 0.01;
        if self.size_nl >= 100.0 {
            self.size_nl = 10.0; // fission resets size
        }

        // Update price anchor with exponential moving average
        let alpha = 0.05;
        self.price_anchor = alpha * price + (1.0 - alpha) * self.price_anchor;
    }
}

/// The full swarm of water robot droplets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterRobotSwarm {
    pub droplets: Vec<DropletNode>,
}

/// Result of a swarm vote.
#[derive(Debug, Clone)]
pub struct VoteResult {
    pub approved: bool,
    pub yes_mass_pg: f64,
    pub total_mass_pg: f64,
    pub yes_fraction: f64,
    pub voters: Vec<(Species, bool)>,
}

impl WaterRobotSwarm {
    /// Create the canonical 6-species swarm.
    pub fn new(initial_price: f64) -> Self {
        let droplets = vec![
            DropletNode::new(0, Species::QuantumJellyfish,      initial_price),
            DropletNode::new(1, Species::EntangledDolphin,      initial_price),
            DropletNode::new(2, Species::TunnelingOctopus,      initial_price),
            DropletNode::new(3, Species::WaveParticleWhale,     initial_price),
            DropletNode::new(4, Species::SuperpositionSeahorse, initial_price),
            DropletNode::new(5, Species::NanoQuantumonas,       initial_price),
        ];
        WaterRobotSwarm { droplets }
    }

    /// Proof-of-Biosynthesis vote: approve swap if >66% of total DNA mass votes YES.
    pub fn vote(
        &self,
        resonance: &ResonanceState,
        amount_display: f64,
        current_price: f64,
        fcc_threshold: f64,
    ) -> VoteResult {
        let mut total_mass = 0.0_f64;
        let mut yes_mass = 0.0_f64;
        let mut voters = Vec::new();

        for droplet in &self.droplets {
            let mass = droplet.dna.mass_picograms();
            let vote = droplet.evaluate(resonance, amount_display, current_price, fcc_threshold);
            total_mass += mass;
            if vote {
                yes_mass += mass;
            }
            voters.push((droplet.species.clone(), vote));
        }

        let yes_fraction = if total_mass > 0.0 { yes_mass / total_mass } else { 0.0 };
        let approved = yes_fraction > 0.66; // DNA-mass supermajority

        VoteResult { approved, yes_mass_pg: yes_mass, total_mass_pg: total_mass, yes_fraction, voters }
    }

    /// Record swap outcome across all voting droplets.
    pub fn record_outcome(&mut self, price: f64, amount: f64, profit_pct: f64) {
        let seed = (price * 1e6) as u64 ^ (amount * 1e8) as u64;
        for droplet in &mut self.droplets {
            droplet.record_outcome(price, amount, profit_pct, seed ^ droplet.id as u64);
        }
    }

    /// Total swarm DNA mass in picograms.
    pub fn total_mass_pg(&self) -> f64 {
        self.droplets.iter().map(|d| d.dna.mass_picograms()).sum()
    }
}

impl VoteResult {
    pub fn summary_line(&self) -> String {
        let detail: Vec<String> = self.voters.iter().map(|(s, v)| {
            format!("{}{}", s.emoji(), if *v { "✓" } else { "✗" })
        }).collect();
        format!(
            "[{:.0}% YES | {:.1}pg/{:.1}pg] {}",
            self.yes_fraction * 100.0,
            self.yes_mass_pg,
            self.total_mass_pg,
            detail.join(" "),
        )
    }
}
