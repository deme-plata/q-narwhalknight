//! Biosafety Framework
//!
//! Safety constraints and verification for biological synthesis operations.
//! Implements containment requirements, prohibited molecule detection,
//! and kill switch requirements.

use crate::types::*;
use crate::BioDSLError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Biosafety controller for synthesis operations
pub struct BiosafetyController {
    /// Prohibited molecules (never synthesize)
    prohibited_molecules: HashSet<String>,
    /// Controlled substances requiring license
    controlled_substances: HashMap<String, ControlLevel>,
    /// Quantity limits per molecule
    quantity_limits: HashMap<String, QuantityLimit>,
    /// Required safety features per category
    safety_requirements: HashMap<SynthesisCategory, Vec<SafetyRequirement>>,
    /// Current synthesis quantities (for tracking)
    current_quantities: HashMap<String, f64>,
}

impl BiosafetyController {
    pub fn new() -> Self {
        let mut controller = Self {
            prohibited_molecules: HashSet::new(),
            controlled_substances: HashMap::new(),
            quantity_limits: HashMap::new(),
            safety_requirements: HashMap::new(),
            current_quantities: HashMap::new(),
        };

        controller.load_default_restrictions();
        controller
    }

    fn load_default_restrictions(&mut self) {
        // Prohibited molecules (select agents, toxins, etc.)
        // These are examples - real implementation would include comprehensive lists
        self.prohibited_molecules.insert("ricin".to_string());
        self.prohibited_molecules.insert("abrin".to_string());
        self.prohibited_molecules.insert("botulinum_toxin".to_string());
        self.prohibited_molecules.insert("anthrax_toxin".to_string());
        self.prohibited_molecules.insert("VX".to_string());
        self.prohibited_molecules.insert("sarin".to_string());
        self.prohibited_molecules.insert("novichok".to_string());

        // Controlled substances
        self.controlled_substances.insert(
            "THC".to_string(),
            ControlLevel::ScheduleI,
        );
        self.controlled_substances.insert(
            "psilocybin".to_string(),
            ControlLevel::ScheduleI,
        );
        self.controlled_substances.insert(
            "DMT".to_string(),
            ControlLevel::ScheduleI,
        );
        self.controlled_substances.insert(
            "LSD".to_string(),
            ControlLevel::ScheduleI,
        );
        self.controlled_substances.insert(
            "mescaline".to_string(),
            ControlLevel::ScheduleI,
        );
        self.controlled_substances.insert(
            "morphine".to_string(),
            ControlLevel::ScheduleII,
        );
        self.controlled_substances.insert(
            "fentanyl".to_string(),
            ControlLevel::ScheduleII,
        );
        self.controlled_substances.insert(
            "amphetamine".to_string(),
            ControlLevel::ScheduleII,
        );
        self.controlled_substances.insert(
            "ketamine".to_string(),
            ControlLevel::ScheduleIII,
        );
        self.controlled_substances.insert(
            "codeine".to_string(),
            ControlLevel::ScheduleIII,
        );

        // CBD is generally unrestricted (varies by jurisdiction)
        // Not adding to controlled list

        // Quantity limits (research quantities)
        self.quantity_limits.insert(
            "THC".to_string(),
            QuantityLimit {
                max_per_synthesis_mg: 100.0,
                max_per_day_mg: 500.0,
                requires_logging: true,
            },
        );

        self.quantity_limits.insert(
            "psilocybin".to_string(),
            QuantityLimit {
                max_per_synthesis_mg: 50.0,
                max_per_day_mg: 200.0,
                requires_logging: true,
            },
        );

        // Safety requirements for genetic circuits
        self.safety_requirements.insert(
            SynthesisCategory::GeneticCircuit,
            vec![
                SafetyRequirement::KillSwitch,
                SafetyRequirement::AuxotrophicMarker,
            ],
        );

        self.safety_requirements.insert(
            SynthesisCategory::ModifiedOrganism,
            vec![
                SafetyRequirement::KillSwitch,
                SafetyRequirement::AuxotrophicMarker,
                SafetyRequirement::GeneticFirewall,
                SafetyRequirement::GenerationLimit(100),
            ],
        );
    }

    /// Check if synthesis is allowed
    pub fn check_synthesis(
        &self,
        molecule_name: &str,
        quantity_mg: f64,
        license: Option<&License>,
    ) -> Result<SynthesisApproval, BiosafetyError> {
        let name_lower = molecule_name.to_lowercase();

        // Check prohibited list
        if self.prohibited_molecules.contains(&name_lower) {
            return Err(BiosafetyError::ProhibitedMolecule {
                name: molecule_name.to_string(),
                reason: "Listed as prohibited select agent or toxin".to_string(),
            });
        }

        // Check controlled substances
        if let Some(control_level) = self.controlled_substances.get(&name_lower) {
            match license {
                Some(lic) if lic.permits_control_level(control_level) => {
                    // License valid
                }
                Some(lic) => {
                    return Err(BiosafetyError::InsufficientLicense {
                        molecule: molecule_name.to_string(),
                        required_level: *control_level,
                        provided_level: lic.level,
                    });
                }
                None => {
                    return Err(BiosafetyError::LicenseRequired {
                        molecule: molecule_name.to_string(),
                        control_level: *control_level,
                    });
                }
            }
        }

        // Check quantity limits
        if let Some(limit) = self.quantity_limits.get(&name_lower) {
            if quantity_mg > limit.max_per_synthesis_mg {
                return Err(BiosafetyError::QuantityExceeded {
                    molecule: molecule_name.to_string(),
                    requested: quantity_mg,
                    limit: limit.max_per_synthesis_mg,
                });
            }

            // Check daily limit
            let current = self.current_quantities.get(&name_lower).unwrap_or(&0.0);
            if current + quantity_mg > limit.max_per_day_mg {
                return Err(BiosafetyError::DailyLimitExceeded {
                    molecule: molecule_name.to_string(),
                    requested: quantity_mg,
                    already_today: *current,
                    daily_limit: limit.max_per_day_mg,
                });
            }
        }

        // Determine required safety features
        let mut requirements = Vec::new();
        if self.controlled_substances.contains_key(&name_lower) {
            requirements.push(SafetyRequirement::LoggingRequired);
            requirements.push(SafetyRequirement::SecureStorage);
        }

        Ok(SynthesisApproval {
            approved: true,
            molecule: molecule_name.to_string(),
            quantity_mg,
            requirements,
            warnings: Vec::new(),
        })
    }

    /// Check genetic circuit safety
    pub fn check_genetic_circuit(
        &self,
        has_kill_switch: bool,
        has_auxotrophy: bool,
        generation_limit: Option<u32>,
    ) -> Result<CircuitApproval, BiosafetyError> {
        let mut missing_features = Vec::new();

        if !has_kill_switch {
            missing_features.push("kill switch".to_string());
        }

        if !has_auxotrophy {
            missing_features.push("auxotrophic marker".to_string());
        }

        if generation_limit.is_none() || generation_limit.unwrap() > 1000 {
            missing_features.push("reasonable generation limit (<=1000)".to_string());
        }

        if !missing_features.is_empty() {
            return Err(BiosafetyError::MissingSafetyFeatures {
                missing: missing_features,
            });
        }

        Ok(CircuitApproval {
            approved: true,
            containment_level: ContainmentLevel::BSL1,
            additional_requirements: Vec::new(),
        })
    }

    /// Record synthesis for tracking
    pub fn record_synthesis(&mut self, molecule_name: &str, quantity_mg: f64) {
        let name_lower = molecule_name.to_lowercase();
        *self.current_quantities.entry(name_lower).or_insert(0.0) += quantity_mg;
    }

    /// Reset daily quantities (call at start of day)
    pub fn reset_daily_quantities(&mut self) {
        self.current_quantities.clear();
    }

    /// Add custom prohibited molecule
    pub fn add_prohibited(&mut self, name: &str, reason: &str) {
        tracing::warn!("Adding {} to prohibited list: {}", name, reason);
        self.prohibited_molecules.insert(name.to_lowercase());
    }

    /// Check if molecule is prohibited
    pub fn is_prohibited(&self, name: &str) -> bool {
        self.prohibited_molecules.contains(&name.to_lowercase())
    }

    /// Get control level for molecule
    pub fn get_control_level(&self, name: &str) -> Option<&ControlLevel> {
        self.controlled_substances.get(&name.to_lowercase())
    }
}

impl Default for BiosafetyController {
    fn default() -> Self {
        Self::new()
    }
}

/// Control level for substances
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlLevel {
    /// DEA Schedule I - High potential for abuse, no accepted medical use
    ScheduleI,
    /// DEA Schedule II - High potential for abuse, accepted medical use
    ScheduleII,
    /// DEA Schedule III - Moderate potential for abuse
    ScheduleIII,
    /// DEA Schedule IV - Low potential for abuse
    ScheduleIV,
    /// DEA Schedule V - Lowest potential for abuse
    ScheduleV,
    /// Not scheduled but requires prescription
    PrescriptionOnly,
    /// Research use only
    ResearchOnly,
}

/// Quantity limit for a molecule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantityLimit {
    pub max_per_synthesis_mg: f64,
    pub max_per_day_mg: f64,
    pub requires_logging: bool,
}

/// Synthesis category for safety requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SynthesisCategory {
    SmallMolecule,
    Protein,
    GeneticCircuit,
    ModifiedOrganism,
    Virus,
    Toxin,
}

/// Safety requirement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SafetyRequirement {
    KillSwitch,
    AuxotrophicMarker,
    GeneticFirewall,
    GenerationLimit(u32),
    ContainmentLevel(ContainmentLevel),
    LoggingRequired,
    SecureStorage,
    DualUseReview,
}

/// Containment level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContainmentLevel {
    BSL1,
    BSL2,
    BSL3,
    BSL4,
}

/// License for controlled substance handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct License {
    pub holder: String,
    pub license_number: String,
    pub level: ControlLevel,
    pub expiration: u64, // Unix timestamp
    pub allowed_substances: Vec<String>,
}

impl License {
    pub fn permits_control_level(&self, required: &ControlLevel) -> bool {
        // Schedule I license permits all lower schedules
        match (self.level, required) {
            (ControlLevel::ScheduleI, _) => true,
            (ControlLevel::ScheduleII, ControlLevel::ScheduleII) => true,
            (ControlLevel::ScheduleII, ControlLevel::ScheduleIII) => true,
            (ControlLevel::ScheduleII, ControlLevel::ScheduleIV) => true,
            (ControlLevel::ScheduleII, ControlLevel::ScheduleV) => true,
            (ControlLevel::ScheduleIII, ControlLevel::ScheduleIII) => true,
            (ControlLevel::ScheduleIII, ControlLevel::ScheduleIV) => true,
            (ControlLevel::ScheduleIII, ControlLevel::ScheduleV) => true,
            (ControlLevel::ScheduleIV, ControlLevel::ScheduleIV) => true,
            (ControlLevel::ScheduleIV, ControlLevel::ScheduleV) => true,
            (ControlLevel::ScheduleV, ControlLevel::ScheduleV) => true,
            (ControlLevel::ResearchOnly, ControlLevel::ResearchOnly) => true,
            _ => false,
        }
    }
}

/// Approval result for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisApproval {
    pub approved: bool,
    pub molecule: String,
    pub quantity_mg: f64,
    pub requirements: Vec<SafetyRequirement>,
    pub warnings: Vec<String>,
}

/// Approval result for genetic circuits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitApproval {
    pub approved: bool,
    pub containment_level: ContainmentLevel,
    pub additional_requirements: Vec<SafetyRequirement>,
}

/// Biosafety error
#[derive(Debug, thiserror::Error)]
pub enum BiosafetyError {
    #[error("Prohibited molecule: {name} - {reason}")]
    ProhibitedMolecule { name: String, reason: String },

    #[error("License required for {molecule} (control level: {control_level:?})")]
    LicenseRequired {
        molecule: String,
        control_level: ControlLevel,
    },

    #[error("Insufficient license for {molecule}: requires {required_level:?}, provided {provided_level:?}")]
    InsufficientLicense {
        molecule: String,
        required_level: ControlLevel,
        provided_level: ControlLevel,
    },

    #[error("Quantity exceeded for {molecule}: requested {requested}mg, limit {limit}mg")]
    QuantityExceeded {
        molecule: String,
        requested: f64,
        limit: f64,
    },

    #[error("Daily limit exceeded for {molecule}: requested {requested}mg, already {already_today}mg today, limit {daily_limit}mg")]
    DailyLimitExceeded {
        molecule: String,
        requested: f64,
        already_today: f64,
        daily_limit: f64,
    },

    #[error("Missing required safety features: {missing:?}")]
    MissingSafetyFeatures { missing: Vec<String> },
}

/// Kill switch implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitch {
    pub switch_type: KillSwitchType,
    pub toxin_gene: String,
    pub trigger_condition: String,
    pub response_time_minutes: u32,
    pub tested: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KillSwitchType {
    /// Toxin-antitoxin system
    ToxinAntitoxin { toxin: String, antitoxin: String },
    /// Temperature sensitive (dies outside range)
    TemperatureSensitive { min_c: f64, max_c: f64 },
    /// Nutrient dependent (dies without specific nutrient)
    NutrientDependent { nutrient: String },
    /// Light activated
    LightActivated { wavelength_nm: u32 },
    /// Chemical inducible
    ChemicalInducible { inducer: String },
}

impl KillSwitch {
    /// Create temperature-sensitive kill switch
    pub fn temperature_sensitive(max_c: f64) -> Self {
        Self {
            switch_type: KillSwitchType::TemperatureSensitive {
                min_c: 15.0,
                max_c,
            },
            toxin_gene: "ccdB".to_string(),
            trigger_condition: format!("Temperature > {}°C", max_c),
            response_time_minutes: 30,
            tested: false,
        }
    }

    /// Create nutrient-dependent kill switch
    pub fn nutrient_dependent(nutrient: &str) -> Self {
        Self {
            switch_type: KillSwitchType::NutrientDependent {
                nutrient: nutrient.to_string(),
            },
            toxin_gene: "relE".to_string(),
            trigger_condition: format!("Absence of {}", nutrient),
            response_time_minutes: 60,
            tested: false,
        }
    }

    /// Create toxin-antitoxin kill switch
    pub fn toxin_antitoxin(toxin: &str, antitoxin: &str) -> Self {
        Self {
            switch_type: KillSwitchType::ToxinAntitoxin {
                toxin: toxin.to_string(),
                antitoxin: antitoxin.to_string(),
            },
            toxin_gene: toxin.to_string(),
            trigger_condition: format!("{} degradation without {} replenishment", antitoxin, antitoxin),
            response_time_minutes: 45,
            tested: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prohibited_molecule() {
        let controller = BiosafetyController::new();

        let result = controller.check_synthesis("ricin", 1.0, None);
        assert!(result.is_err());

        if let Err(BiosafetyError::ProhibitedMolecule { name, .. }) = result {
            assert_eq!(name, "ricin");
        } else {
            panic!("Expected ProhibitedMolecule error");
        }
    }

    #[test]
    fn test_controlled_substance_no_license() {
        let controller = BiosafetyController::new();

        let result = controller.check_synthesis("THC", 10.0, None);
        assert!(result.is_err());

        if let Err(BiosafetyError::LicenseRequired { molecule, .. }) = result {
            assert_eq!(molecule, "THC");
        } else {
            panic!("Expected LicenseRequired error");
        }
    }

    #[test]
    fn test_controlled_substance_with_license() {
        let controller = BiosafetyController::new();

        let license = License {
            holder: "Research Lab".to_string(),
            license_number: "DEA-12345".to_string(),
            level: ControlLevel::ScheduleI,
            expiration: u64::MAX,
            allowed_substances: vec!["THC".to_string()],
        };

        let result = controller.check_synthesis("THC", 10.0, Some(&license));
        assert!(result.is_ok());

        let approval = result.unwrap();
        assert!(approval.approved);
        assert_eq!(approval.quantity_mg, 10.0);
    }

    #[test]
    fn test_quantity_limit() {
        let controller = BiosafetyController::new();

        let license = License {
            holder: "Research Lab".to_string(),
            license_number: "DEA-12345".to_string(),
            level: ControlLevel::ScheduleI,
            expiration: u64::MAX,
            allowed_substances: vec!["THC".to_string()],
        };

        // Exceed per-synthesis limit
        let result = controller.check_synthesis("THC", 200.0, Some(&license));
        assert!(result.is_err());

        if let Err(BiosafetyError::QuantityExceeded { requested, limit, .. }) = result {
            assert_eq!(requested, 200.0);
            assert_eq!(limit, 100.0);
        } else {
            panic!("Expected QuantityExceeded error");
        }
    }

    #[test]
    fn test_genetic_circuit_safety() {
        let controller = BiosafetyController::new();

        // Missing safety features
        let result = controller.check_genetic_circuit(false, false, None);
        assert!(result.is_err());

        // All features present
        let result = controller.check_genetic_circuit(true, true, Some(100));
        assert!(result.is_ok());
    }

    #[test]
    fn test_unrestricted_molecule() {
        let controller = BiosafetyController::new();

        // CBD is not restricted
        let result = controller.check_synthesis("CBD", 1000.0, None);
        assert!(result.is_ok());

        // Caffeine is unrestricted
        let result = controller.check_synthesis("caffeine", 500.0, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_kill_switch_creation() {
        let ks = KillSwitch::temperature_sensitive(30.0);
        assert!(matches!(
            ks.switch_type,
            KillSwitchType::TemperatureSensitive { max_c: 30.0, .. }
        ));

        let ks = KillSwitch::nutrient_dependent("tryptophan");
        assert!(matches!(
            ks.switch_type,
            KillSwitchType::NutrientDependent { .. }
        ));
    }
}
