/// Display utilities for terminal output

use colored::*;
use prettytable::{Cell, Row, Table, format};

pub fn print_header(title: &str) {
    println!("\n{}", format!("┌─────────────────────────────────────────────┐").cyan());
    println!("{}", format!("│ {:<43} │", title).cyan());
    println!("{}", format!("├─────────────────────────────────────────────┤").cyan());
}

pub fn print_footer() {
    println!("{}", format!("└─────────────────────────────────────────────┘").cyan());
}

pub fn print_success(message: &str) {
    println!("{}", format!("✅ {}", message).green());
}

pub fn print_error(message: &str) {
    println!("{}", format!("❌ {}", message).red());
}

pub fn print_warning(message: &str) {
    println!("{}", format!("⚠️  {}", message).yellow());
}

pub fn print_info(message: &str) {
    println!("{}", format!("ℹ️  {}", message).blue());
}

pub fn print_kv(key: &str, value: &str) {
    println!("{}", format!("│ {:<25} {:<15} │", key, value).cyan());
}

pub fn create_table(headers: Vec<&str>) -> Table {
    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_BOX_CHARS);

    let header_cells: Vec<Cell> = headers
        .iter()
        .map(|h| Cell::new(h).style_spec("Fb"))
        .collect();

    table.add_row(Row::new(header_cells));
    table
}

pub fn format_amount(amount: u64) -> String {
    let s = amount.to_string();
    let mut result = String::new();
    let mut count = 0;

    for c in s.chars().rev() {
        if count > 0 && count % 3 == 0 {
            result.push(',');
        }
        result.push(c);
        count += 1;
    }

    result.chars().rev().collect()
}

pub fn format_percentage(value: f64) -> String {
    format!("{:.2}%", value * 100.0)
}

pub fn format_currency(amount: u64) -> String {
    format!("${}", format_amount(amount))
}

pub fn print_status_indicator(status: &str, is_ok: bool) {
    let indicator = if is_ok { "✓" } else { "✗" };
    let color = if is_ok { "green" } else { "red" };

    println!("{}", format!("│ {}: {} {}", status, indicator, if is_ok { "OK" } else { "ERROR" }).color(color));
}