use std::sync::Arc;

pub mod narwhal_bullshark;
pub mod pbft;

pub struct Knight {
    pub dag: Arc<dyn std::any::Any + Send + Sync>, // Use a trait object temporarily
}

impl Knight {
    pub fn new(dag: Arc<dyn std::any::Any + Send + Sync>) -> Self {
        Self { dag }
    }

    pub fn get_current_k(&self) -> usize {
        2 // Placeholder
    }
}
