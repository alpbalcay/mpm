//! Abstract material trait definition.

use mpm_core::{DenseMap, Index, Vector6d};

/// Read-only access to particle data needed by material models.
pub trait ParticleData {
    fn strain_rate(&self) -> Vector6d;
    fn pressure(&self) -> f64;
    fn previous_stress(&self) -> Vector6d;
}

/// Trait for constitutive material models.
///
/// All materials implement this trait to provide stress computation
/// given incremental strain and current state.
pub trait Material<const DIM: usize>: Send + Sync {
    /// Return the material ID.
    fn id(&self) -> Index;

    /// Return the material density.
    fn density(&self) -> f64;

    /// Compute updated stress given current stress, strain increment, and state.
    fn compute_stress(
        &self,
        stress: &Vector6d,
        dstrain: &Vector6d,
        particle: &dyn ParticleData,
        state_vars: &mut DenseMap,
    ) -> Vector6d;

    /// Return the thermodynamic pressure for a volumetric strain increment.
    fn thermodynamic_pressure(&self, dvolumetric_strain: f64) -> f64 {
        let _ = dvolumetric_strain;
        0.0
    }

    /// Initialise state variables for this material.
    fn initialise_state_variables(&self) -> DenseMap;

    /// Return the list of state variable names.
    fn state_variables(&self) -> Vec<String>;
}
