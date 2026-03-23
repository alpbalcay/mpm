//! HDF5 particle data structure for checkpoint/restart.

use mpm_core::Index;
use serde::{Deserialize, Serialize};

/// HDF5 particle data for serialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Hdf5Particle {
    pub id: Index,
    pub mass: f64,
    pub volume: f64,
    pub pressure: f64,
    pub coord_x: f64,
    pub coord_y: f64,
    pub coord_z: f64,
    pub displacement_x: f64,
    pub displacement_y: f64,
    pub displacement_z: f64,
    pub nsize_x: f64,
    pub nsize_y: f64,
    pub nsize_z: f64,
    pub velocity_x: f64,
    pub velocity_y: f64,
    pub velocity_z: f64,
    pub stress_xx: f64,
    pub stress_yy: f64,
    pub stress_zz: f64,
    pub tau_xy: f64,
    pub tau_yz: f64,
    pub tau_xz: f64,
    pub strain_xx: f64,
    pub strain_yy: f64,
    pub strain_zz: f64,
    pub gamma_xy: f64,
    pub gamma_yz: f64,
    pub gamma_xz: f64,
    pub epsilon_v: f64,
    pub cell_id: Index,
    pub status: bool,
    pub material_id: u32,
    pub nstate_vars: u32,
    pub svars: [f64; 20],
}

impl Hdf5Particle {
    pub fn new() -> Self {
        Self::default()
    }
}
