//! Nodal property pool for multimaterial contact.

use nalgebra::DMatrix;
use std::collections::HashMap;

/// Stores per-node, per-material properties for multimaterial contact.
pub struct NodalProperties {
    properties: HashMap<String, DMatrix<f64>>,
}

impl NodalProperties {
    pub fn new() -> Self {
        Self {
            properties: HashMap::new(),
        }
    }

    /// Create a new property matrix.
    pub fn create_property(&mut self, name: &str, rows: usize, columns: usize) -> bool {
        self.properties
            .insert(name.to_string(), DMatrix::zeros(rows, columns));
        true
    }

    /// Reset all properties to zero.
    pub fn initialise(&mut self) {
        for matrix in self.properties.values_mut() {
            matrix.fill(0.0);
        }
    }

    /// Get a property value.
    pub fn property(
        &self,
        name: &str,
        node_id: usize,
        mat_id: usize,
        nprops: usize,
    ) -> Option<DMatrix<f64>> {
        self.properties.get(name).map(|mat| {
            let row = node_id * nprops + mat_id;
            let cols = mat.ncols();
            DMatrix::from_fn(1, cols, |_, c| {
                if row < mat.nrows() {
                    mat[(row, c)]
                } else {
                    0.0
                }
            })
        })
    }

    /// Update (accumulate) a property value.
    pub fn update_property(
        &mut self,
        name: &str,
        node_id: usize,
        mat_id: usize,
        value: &DMatrix<f64>,
        nprops: usize,
    ) {
        if let Some(mat) = self.properties.get_mut(name) {
            let row = node_id * nprops + mat_id;
            let cols = mat.ncols().min(value.ncols());
            for c in 0..cols {
                if row < mat.nrows() {
                    mat[(row, c)] += value[(0, c)];
                }
            }
        }
    }
}

impl Default for NodalProperties {
    fn default() -> Self {
        Self::new()
    }
}
