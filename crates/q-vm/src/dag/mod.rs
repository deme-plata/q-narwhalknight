//! DAG module for consensus - now integrates with existing DAG-Knight implementation
pub use crate::dag_integration::VMIntegratedDAG;

// Re-export for backward compatibility
#[derive(Debug)]
pub struct DAG {
    // This is now a lightweight wrapper around VMIntegratedDAG
    // Kept for backward compatibility with existing code
}

impl DAG {
    pub fn new() -> Self {
        Self {}
    }

    /// Create a new DAG with VM integration
    /// This is the recommended way to create a DAG instance
    pub async fn new_with_vm_integration(
        node_id: [u8; 32],
        f: usize,
        state_db_path: &str,
    ) -> anyhow::Result<VMIntegratedDAG> {
        VMIntegratedDAG::new(node_id, f, state_db_path).await
    }
}
