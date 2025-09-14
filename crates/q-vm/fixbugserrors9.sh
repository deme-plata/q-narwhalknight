#!/bin/bash

echo "Fixing the last import issue..."

# Update consensus/mod.rs to use the correct import
cat > src/consensus/mod.rs << 'EOF'
use dagknight_vm::DAG;  // Import directly from the crate
use std::sync::Arc;

pub mod narwhal_bullshark;
pub mod pbft;

pub struct Knight {
    pub dag: Arc<DAG>,
}

impl Knight {
    pub fn new(dag: Arc<DAG>) -> Self {
        Self { dag }
    }

    pub fn get_current_k(&self) -> usize {
        2 // Placeholder
    }
}
EOF

echo "The last import issue has been fixed!"
echo "You can now try to compile the project with 'cargo build'"
