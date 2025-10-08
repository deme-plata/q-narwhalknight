//! Graph generation and LaTeX export for k-parameter research
//! Generates publication-quality plots and LaTeX tables

use plotters::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::Write;

pub mod surface3d;
pub use surface3d::{Surface3D, ColorMap, Annotation3D};

/// Data series for plotting
#[derive(Debug, Clone, Serialize)]
pub struct DataSeries {
    pub name: String,
    pub x_values: Vec<f64>,
    pub y_values: Vec<f64>,
    pub color: String,
    pub error_bars: Option<Vec<f64>>, // NEW: y-error bars (±)
    pub fill_between: Option<(Vec<f64>, Vec<f64>)>, // NEW: uncertainty bands (lower, upper)
    pub line_style: LineStyle, // NEW: solid, dashed, dotted
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
}

impl DataSeries {
    pub fn new(name: &str, x_values: Vec<f64>, y_values: Vec<f64>) -> Self {
        Self {
            name: name.to_string(),
            x_values,
            y_values,
            color: "blue".to_string(),
            error_bars: None,
            fill_between: None,
            line_style: LineStyle::Solid,
        }
    }

    pub fn with_color(mut self, color: &str) -> Self {
        self.color = color.to_string();
        self
    }

    pub fn with_error_bars(mut self, errors: Vec<f64>) -> Self {
        self.error_bars = Some(errors);
        self
    }

    pub fn with_uncertainty_band(mut self, lower: Vec<f64>, upper: Vec<f64>) -> Self {
        self.fill_between = Some((lower, upper));
        self
    }

    pub fn with_line_style(mut self, style: LineStyle) -> Self {
        self.line_style = style;
        self
    }
}

/// Scale type for axes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleType {
    Linear,
    Logarithmic,
}

/// Graph plotter for k-parameter analysis
pub struct GraphPlotter {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub x_scale: ScaleType, // NEW: linear or log scale
    pub y_scale: ScaleType, // NEW: linear or log scale
}

impl GraphPlotter {
    pub fn new(title: &str, x_label: &str, y_label: &str) -> Self {
        Self {
            width: 1200,
            height: 800,
            title: title.to_string(),
            x_label: x_label.to_string(),
            y_label: y_label.to_string(),
            x_scale: ScaleType::Linear,
            y_scale: ScaleType::Linear,
        }
    }

    pub fn with_log_x(mut self) -> Self {
        self.x_scale = ScaleType::Logarithmic;
        self
    }

    pub fn with_log_y(mut self) -> Self {
        self.y_scale = ScaleType::Logarithmic;
        self
    }

    /// Plot multiple data series to PNG file
    pub fn plot_to_file(&self, series: &[DataSeries], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(output_path, (self.width, self.height)).into_drawing_area();
        root.fill(&WHITE)?;

        // Calculate plot ranges
        let x_min = series.iter()
            .flat_map(|s| s.x_values.iter())
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let x_max = series.iter()
            .flat_map(|s| s.x_values.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = series.iter()
            .flat_map(|s| s.y_values.iter())
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let y_max = series.iter()
            .flat_map(|s| s.y_values.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.title, ("sans-serif", 40).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart.configure_mesh()
            .x_desc(&self.x_label)
            .y_desc(&self.y_label)
            .draw()?;

        // Plot each series
        for data_series in series {
            let color = match data_series.color.as_str() {
                "red" => &RED,
                "green" => &GREEN,
                "blue" => &BLUE,
                "black" => &BLACK,
                "magenta" => &MAGENTA,
                "cyan" => &CYAN,
                _ => &BLUE,
            };

            let points: Vec<(f64, f64)> = data_series.x_values.iter()
                .zip(data_series.y_values.iter())
                .map(|(x, y)| (*x, *y))
                .collect();

            chart.draw_series(LineSeries::new(points, color))?
                .label(&data_series.name)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }
}

/// LaTeX table generator
pub struct LaTeXTableGenerator {
    pub caption: String,
    pub label: String,
}

impl LaTeXTableGenerator {
    pub fn new(caption: &str, label: &str) -> Self {
        Self {
            caption: caption.to_string(),
            label: label.to_string(),
        }
    }

    /// Generate LaTeX table from data
    pub fn generate_table(
        &self,
        headers: &[&str],
        rows: &[Vec<String>],
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(output_path)?;

        // Write table header
        writeln!(file, "\\begin{{table}}[htbp]")?;
        writeln!(file, "  \\centering")?;
        writeln!(file, "  \\caption{{{}}}", self.caption)?;
        writeln!(file, "  \\label{{{}}}", self.label)?;

        // Write tabular environment
        let col_spec = "c".repeat(headers.len());
        writeln!(file, "  \\begin{{tabular}}{{|{}|}}", col_spec)?;
        writeln!(file, "    \\hline")?;

        // Write headers
        let header_row = headers.join(" & ");
        writeln!(file, "    {} \\\\", header_row)?;
        writeln!(file, "    \\hline")?;

        // Write data rows
        for row in rows {
            let row_str = row.join(" & ");
            writeln!(file, "    {} \\\\", row_str)?;
        }

        writeln!(file, "    \\hline")?;
        writeln!(file, "  \\end{{tabular}}")?;
        writeln!(file, "\\end{{table}}")?;

        Ok(())
    }
}

/// LaTeX figure generator
pub struct LaTeXFigureGenerator {
    pub caption: String,
    pub label: String,
}

impl LaTeXFigureGenerator {
    pub fn new(caption: &str, label: &str) -> Self {
        Self {
            caption: caption.to_string(),
            label: label.to_string(),
        }
    }

    /// Generate LaTeX figure environment
    pub fn generate_figure(
        &self,
        image_path: &str,
        width: &str,
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(output_path)?;

        writeln!(file, "\\begin{{figure}}[htbp]")?;
        writeln!(file, "  \\centering")?;
        writeln!(file, "  \\includegraphics[width={}]{{{}}}",  width, image_path)?;
        writeln!(file, "  \\caption{{{}}}", self.caption)?;
        writeln!(file, "  \\label{{{}}}", self.label)?;
        writeln!(file, "\\end{{figure}}")?;

        Ok(())
    }
}

/// Export complete LaTeX document section with graphs and tables
pub fn export_latex_section(
    section_title: &str,
    graphs: &[(&str, &str, &str)], // (image_path, caption, label)
    tables: &[(Vec<&str>, Vec<Vec<String>>, &str, &str)], // (headers, rows, caption, label)
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(output_path)?;

    writeln!(file, "\\section{{{}}}", section_title)?;
    writeln!(file)?;

    // Add figures
    for (img_path, caption, label) in graphs {
        let fig_gen = LaTeXFigureGenerator::new(caption, label);
        writeln!(file, "\\begin{{figure}}[htbp]")?;
        writeln!(file, "  \\centering")?;
        writeln!(file, "  \\includegraphics[width=0.8\\textwidth]{{{}}}", img_path)?;
        writeln!(file, "  \\caption{{{}}}", caption)?;
        writeln!(file, "  \\label{{{}}}", label)?;
        writeln!(file, "\\end{{figure}}")?;
        writeln!(file)?;
    }

    // Add tables
    for (headers, rows, caption, label) in tables {
        let col_spec = "c".repeat(headers.len());
        writeln!(file, "\\begin{{table}}[htbp]")?;
        writeln!(file, "  \\centering")?;
        writeln!(file, "  \\caption{{{}}}", caption)?;
        writeln!(file, "  \\label{{{}}}", label)?;
        writeln!(file, "  \\begin{{tabular}}{{|{}|}}", col_spec)?;
        writeln!(file, "    \\hline")?;

        let header_row = headers.join(" & ");
        writeln!(file, "    {} \\\\", header_row)?;
        writeln!(file, "    \\hline")?;

        for row in rows {
            let row_str = row.join(" & ");
            writeln!(file, "    {} \\\\", row_str)?;
        }

        writeln!(file, "    \\hline")?;
        writeln!(file, "  \\end{{tabular}}")?;
        writeln!(file, "\\end{{table}}")?;
        writeln!(file)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_series() {
        let series = DataSeries::new("Test", vec![1.0, 2.0, 3.0], vec![1.0, 4.0, 9.0])
            .with_color("red");

        assert_eq!(series.name, "Test");
        assert_eq!(series.color, "red");
        assert_eq!(series.x_values.len(), 3);
    }

    #[test]
    fn test_latex_table_generator() {
        let gen = LaTeXTableGenerator::new("Test Caption", "tab:test");
        assert_eq!(gen.caption, "Test Caption");
    }

    #[test]
    fn test_latex_figure_generator() {
        let gen = LaTeXFigureGenerator::new("Figure Caption", "fig:test");
        assert_eq!(gen.label, "fig:test");
    }
}
