//! MPM solvers: explicit time integration with USF/USL schemes.

pub mod scheme;
pub mod mpm_explicit;

pub use mpm_explicit::MpmExplicit;
