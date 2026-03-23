//! Core type aliases and constants for the MPM library.

use nalgebra as na;
use std::collections::HashMap;

/// Global index type for nodes, cells, particles.
pub type Index = u64;

/// Dense map for state variables (string -> double).
pub type DenseMap = HashMap<String, f64>;

/// Fixed-size vector of dimension `DIM`.
pub type VectorDim<const DIM: usize> = na::SVector<f64, DIM>;

/// Fixed-size square matrix of dimension `DIM x DIM`.
pub type MatrixDim<const DIM: usize> = na::SMatrix<f64, DIM, DIM>;

/// 6-component Voigt stress/strain vector.
pub type Vector6d = na::SVector<f64, 6>;

/// 6x6 constitutive/stiffness matrix.
pub type Matrix6x6 = na::SMatrix<f64, 6, 6>;

/// Number of Voigt stress/strain components.
pub const VOIGT_SIZE: usize = 6;

/// Number of phases for multiphase problems.
pub const NPHASES: usize = 1;

/// Particle phase identifiers.
pub mod particle_phase {
    pub const SOLID: usize = 0;
    pub const LIQUID: usize = 1;
}

/// Element degree enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ElementDegree {
    Linear = 1,
    Quadratic = 2,
}

/// Shape function type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ShapefnType {
    NormalMpm = 1,
    Gimp = 2,
    Cpdi = 3,
}

/// Damping type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DampingType {
    None,
    Cundall,
}

/// Return a zero value for common types.
pub fn zero_vector<const DIM: usize>() -> VectorDim<DIM> {
    VectorDim::<DIM>::zeros()
}

/// Return a zero matrix.
pub fn zero_matrix<const DIM: usize>() -> MatrixDim<DIM> {
    MatrixDim::<DIM>::zeros()
}

/// Return a zero Voigt vector.
pub fn zero_voigt() -> Vector6d {
    Vector6d::zeros()
}
