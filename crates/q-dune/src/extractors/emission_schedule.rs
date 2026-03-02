use crate::csv_formatter::EmissionScheduleCsv;
use q_storage::emission_controller;
use tracing::debug;

/// Generate the full 64-era halving emission schedule.
/// This is static data — pushed once to Dune.
pub fn extract_emission_schedule(current_era: u64) -> String {
    let mut csv = EmissionScheduleCsv::new();
    let mut cumulative: u128 = 0;

    // Genesis year: 2026
    let genesis_year: u64 = 2026;

    for era in 0..64u64 {
        let annual = emission_controller::annual_emission(era);
        // Each era = 4 years of emission
        let era_total = annual.saturating_mul(4);
        cumulative = cumulative.saturating_add(era_total);

        let halving_factor = 1.0_f64 / (1u64 << era) as f64;
        let start_year = genesis_year + era * 4;

        csv.add_row(
            era,
            start_year,
            annual,
            cumulative,
            halving_factor,
            era == current_era,
        );
    }

    debug!("[Dune] Generated 64-era emission schedule (current era: {})", current_era);
    csv.finish()
}
