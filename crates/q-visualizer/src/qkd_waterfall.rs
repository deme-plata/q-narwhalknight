/// QKD Photon Waterfall Visualization
/// Real-time ASCII waterfall display of quantum key distribution photon detection events

use anyhow::Result;
use std::collections::VecDeque;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};

/// QKD photon detection event
#[derive(Debug, Clone)]
pub struct PhotonEvent {
    pub timestamp: DateTime<Utc>,
    pub count: u64,
    pub qber: f64,       // Quantum Bit Error Rate
    pub key_rate: f64,   // Key generation rate in bits/sec
    pub detector_id: u8,
    pub polarization: PhotonPolarization,
}

#[derive(Debug, Clone, Copy)]
pub enum PhotonPolarization {
    Horizontal,
    Vertical,
    Diagonal,
    AntiDiagonal,
}

/// QKD Waterfall visualization engine
#[derive(Clone)]
pub struct QKDWaterfall {
    photon_buffer: RwLock<VecDeque<PhotonEvent>>,
    waterfall_lines: RwLock<VecDeque<String>>,
    canvas_width: usize,
    canvas_height: usize,
    total_photons: RwLock<u64>,
    current_key_rate: RwLock<f64>,
    current_qber: RwLock<f64>,
}

impl QKDWaterfall {
    pub fn new() -> Result<Self> {
        Ok(Self {
            photon_buffer: RwLock::new(VecDeque::with_capacity(1000)),
            waterfall_lines: RwLock::new(VecDeque::with_capacity(50)),
            canvas_width: 80,
            canvas_height: 25,
            total_photons: RwLock::new(0),
            current_key_rate: RwLock::new(0.0),
            current_qber: RwLock::new(0.0),
        })
    }

    /// Add new photon detection events
    pub async fn add_photons(&self, count: u64, qber: f64) -> Result<()> {
        let event = PhotonEvent {
            timestamp: Utc::now(),
            count,
            qber,
            key_rate: self.calculate_key_rate(count, qber).await,
            detector_id: 0, // Single detector for now
            polarization: PhotonPolarization::Horizontal, // Simplified
        };

        // Add to buffer
        {
            let mut buffer = self.photon_buffer.write().await;
            buffer.push_back(event.clone());
            
            // Keep buffer size manageable
            while buffer.len() > 1000 {
                buffer.pop_front();
            }
        }

        // Update counters
        {
            let mut total = self.total_photons.write().await;
            *total += count;
        }

        {
            let mut rate = self.current_key_rate.write().await;
            *rate = event.key_rate;
        }

        {
            let mut error_rate = self.current_qber.write().await;
            *error_rate = qber;
        }

        // Generate new waterfall line
        self.generate_waterfall_line(&event).await?;

        Ok(())
    }

    /// Generate ASCII waterfall line from photon event
    async fn generate_waterfall_line(&self, event: &PhotonEvent) -> Result<()> {
        let mut line = String::with_capacity(self.canvas_width);
        
        // Map photon count to character intensity
        let intensity_chars = ['·', '◦', '°', '∘', '○', '●', '⬤', '⬢', '⬡', '◉'];
        let max_count_per_char = 100; // Adjust based on expected photon rates
        
        for i in 0..self.canvas_width {
            // Create quantum interference pattern
            let phase = (i as f64 * 2.0 * std::f64::consts::PI / self.canvas_width as f64) + 
                       (event.timestamp.timestamp_millis() as f64 / 1000.0);
            
            // Photon detection probability (quantum mechanical)
            let detection_prob = 0.5 * (1.0 + (phase + event.qber * std::f64::consts::PI).sin());
            
            // Scale by actual photon count
            let local_intensity = (detection_prob * event.count as f64 / max_count_per_char as f64).min(1.0);
            
            // Select character based on intensity
            let char_index = (local_intensity * (intensity_chars.len() - 1) as f64) as usize;
            line.push(intensity_chars[char_index]);
        }

        // Add to waterfall display
        {
            let mut lines = self.waterfall_lines.write().await;
            lines.push_back(line);
            
            // Maintain display height
            while lines.len() > self.canvas_height {
                lines.pop_front();
            }
        }

        Ok(())
    }

    /// Calculate key generation rate from photon count and QBER
    async fn calculate_key_rate(&self, photon_count: u64, qber: f64) -> f64 {
        // Simplified key rate calculation
        // Real QKD systems use more complex formulas involving error correction
        let raw_key_rate = photon_count as f64 * 0.5; // 50% detection efficiency
        let error_correction_overhead = 1.0 + 2.0 * qber; // Simplified
        
        (raw_key_rate / error_correction_overhead).max(0.0)
    }

    /// Render current waterfall display
    pub async fn render_ascii_waterfall(&self) -> String {
        let mut display = String::new();
        
        // Header with current metrics
        let key_rate = *self.current_key_rate.read().await;
        let qber = *self.current_qber.read().await;
        let total_photons = *self.total_photons.read().await;
        
        display.push_str(&format!(
            "[QKD]  {:.2} kbit/s key rate  |  QBER {:.1}%  |  {} photons total\n",
            key_rate / 1000.0, qber * 100.0, total_photons
        ));
        
        display.push_str("\nPhoton Waterfall (Live):\n");
        
        // Render waterfall lines
        let lines = self.waterfall_lines.read().await;
        for line in lines.iter() {
            display.push_str(line);
            display.push('\n');
        }
        
        // Add round indicator if in consensus mode
        if let Some(current_line) = lines.back() {
            let progress = (key_rate / 10000.0).min(1.0); // Scale to 0-1
            let progress_chars = (progress * self.canvas_width as f64) as usize;
            
            display.push_str("\nRound Progress: ");
            for i in 0..self.canvas_width {
                if i < progress_chars {
                    display.push('█');
                } else {
                    display.push('░');
                }
            }
            display.push_str(&format!(" {:.0}%", progress * 100.0));
        }
        
        display
    }

    /// Start background animation loop
    pub async fn start_animation(&mut self) -> Result<()> {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));
        
        loop {
            interval.tick().await;
            
            // Generate simulated photon events for demonstration
            if rand::random::<f64>() > 0.7 { // 30% chance each tick
                let count = rand::random::<u64>() % 500 + 50; // 50-550 photons
                let qber = rand::random::<f64>() * 0.05; // 0-5% error rate
                
                self.add_photons(count, qber).await?;
            }
            
            // Print current display (in real system, this would update a terminal UI)
            if rand::random::<f64>() > 0.95 { // Occasional display update
                let display = self.render_ascii_waterfall().await;
                tracing::debug!("QKD Waterfall:\n{}", display);
            }
        }
    }

    /// Get total photons processed
    pub async fn total_photons(&self) -> u64 {
        *self.total_photons.read().await
    }

    /// Generate SVG waterfall for web interface
    pub async fn render_svg_waterfall(&self) -> Result<String> {
        let lines = self.waterfall_lines.read().await;
        let key_rate = *self.current_key_rate.read().await;
        let qber = *self.current_qber.read().await;
        
        let width = 800;
        let height = 500;
        let char_width = width / self.canvas_width;
        let line_height = height / self.canvas_height;
        
        let mut svg = format!(
            r#"<svg width="{}" height="{}" viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">"#,
            width, height, width, height
        );
        
        // Background
        svg.push_str(r#"<rect width="100%" height="100%" fill="#000020"/>"#);
        
        // Title
        svg.push_str(&format!(
            r#"<text x="10" y="25" font-family="monospace" font-size="16" fill="cyan">
                QKD Photon Waterfall - {:.1} kbit/s, QBER {:.2}%
            </text>"#,
            key_rate / 1000.0, qber * 100.0
        ));
        
        // Render waterfall
        for (line_idx, line) in lines.iter().enumerate() {
            let y = line_idx * line_height + 50;
            
            for (char_idx, ch) in line.chars().enumerate() {
                let x = char_idx * char_width;
                
                // Map character to color intensity
                let intensity = match ch {
                    '·' => 0.1,
                    '◦' => 0.2,
                    '°' => 0.3,
                    '∘' => 0.4,
                    '○' => 0.5,
                    '●' => 0.6,
                    '⬤' => 0.7,
                    '⬢' => 0.8,
                    '⬡' => 0.9,
                    '◉' => 1.0,
                    _ => 0.1,
                };
                
                // Color transitions from blue (low) to white (high intensity)
                let color = if intensity < 0.5 {
                    format!("rgb({}, {}, {})", 
                           (intensity * 100.0) as u32,
                           (intensity * 150.0) as u32,
                           255)
                } else {
                    format!("rgb({}, {}, {})",
                           255,
                           255,
                           255)
                };
                
                svg.push_str(&format!(
                    r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" opacity="{:.2}"/>"#,
                    x, y, char_width, line_height, color, intensity
                ));
            }
        }
        
        // Add quantum interference overlay
        for i in 0..50 {
            let x = (i * width / 50) as f64;
            let phase = (i as f64 * 2.0 * std::f64::consts::PI / 50.0) + 
                       (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                        .unwrap().as_millis() as f64 / 1000.0);
            let amplitude = 20.0 * phase.sin();
            
            svg.push_str(&format!(
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="rgba(0,255,255,0.3)" stroke-width="1"/>"#,
                x, height as f64 / 2.0 + amplitude,
                x + (width / 50) as f64, height as f64 / 2.0 + amplitude
            ));
        }
        
        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Get waterfall statistics for monitoring
    pub async fn get_statistics(&self) -> QKDStatistics {
        let buffer = self.photon_buffer.read().await;
        let total_photons = *self.total_photons.read().await;
        let current_key_rate = *self.current_key_rate.read().await;
        let current_qber = *self.current_qber.read().await;
        
        // Calculate averages over recent events
        let recent_events: Vec<_> = buffer.iter().rev().take(100).collect();
        let avg_key_rate = if !recent_events.is_empty() {
            recent_events.iter().map(|e| e.key_rate).sum::<f64>() / recent_events.len() as f64
        } else {
            0.0
        };
        
        let avg_qber = if !recent_events.is_empty() {
            recent_events.iter().map(|e| e.qber).sum::<f64>() / recent_events.len() as f64
        } else {
            0.0
        };
        
        QKDStatistics {
            total_photons,
            current_key_rate,
            current_qber,
            average_key_rate: avg_key_rate,
            average_qber: avg_qber,
            events_processed: buffer.len() as u64,
            waterfall_lines: self.waterfall_lines.read().await.len() as u32,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct QKDStatistics {
    pub total_photons: u64,
    pub current_key_rate: f64,
    pub current_qber: f64,
    pub average_key_rate: f64,
    pub average_qber: f64,
    pub events_processed: u64,
    pub waterfall_lines: u32,
}

impl PhotonPolarization {
    pub fn to_char(&self) -> char {
        match self {
            PhotonPolarization::Horizontal => '─',
            PhotonPolarization::Vertical => '│',
            PhotonPolarization::Diagonal => '╱',
            PhotonPolarization::AntiDiagonal => '╲',
        }
    }
    
    pub fn to_color(&self) -> &'static str {
        match self {
            PhotonPolarization::Horizontal => "red",
            PhotonPolarization::Vertical => "blue", 
            PhotonPolarization::Diagonal => "green",
            PhotonPolarization::AntiDiagonal => "yellow",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_waterfall_creation() {
        let waterfall = QKDWaterfall::new().unwrap();
        let stats = waterfall.get_statistics().await;
        
        assert_eq!(stats.total_photons, 0);
        assert_eq!(stats.events_processed, 0);
    }

    #[tokio::test]
    async fn test_add_photons() {
        let waterfall = QKDWaterfall::new().unwrap();
        
        waterfall.add_photons(100, 0.05).await.unwrap();
        
        let stats = waterfall.get_statistics().await;
        assert_eq!(stats.total_photons, 100);
        assert_eq!(stats.events_processed, 1);
        assert!(stats.current_qber > 0.0);
    }

    #[tokio::test]
    async fn test_ascii_rendering() {
        let waterfall = QKDWaterfall::new().unwrap();
        
        waterfall.add_photons(200, 0.02).await.unwrap();
        waterfall.add_photons(150, 0.03).await.unwrap();
        
        let ascii = waterfall.render_ascii_waterfall().await;
        assert!(ascii.contains("[QKD]"));
        assert!(ascii.contains("Photon Waterfall"));
        assert!(ascii.len() > 100); // Should have substantial content
    }

    #[tokio::test]
    async fn test_svg_rendering() {
        let waterfall = QKDWaterfall::new().unwrap();
        
        waterfall.add_photons(300, 0.01).await.unwrap();
        
        let svg = waterfall.render_svg_waterfall().await.unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("QKD Photon Waterfall"));
        assert!(svg.contains("</svg>"));
    }

    #[tokio::test]
    async fn test_key_rate_calculation() {
        let waterfall = QKDWaterfall::new().unwrap();
        
        // Test with different QBER values
        let rate_1 = waterfall.calculate_key_rate(1000, 0.01).await;
        let rate_2 = waterfall.calculate_key_rate(1000, 0.05).await;
        
        assert!(rate_1 > rate_2); // Lower QBER should give higher key rate
        assert!(rate_1 > 0.0);
        assert!(rate_2 > 0.0);
    }

    #[test]
    fn test_polarization_conversion() {
        let pol = PhotonPolarization::Horizontal;
        assert_eq!(pol.to_char(), '─');
        assert_eq!(pol.to_color(), "red");
        
        let pol = PhotonPolarization::Vertical;
        assert_eq!(pol.to_char(), '│');
        assert_eq!(pol.to_color(), "blue");
    }
}