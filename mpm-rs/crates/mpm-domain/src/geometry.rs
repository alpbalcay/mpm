//! Geometric utility functions.

use mpm_core::VectorDim;
use nalgebra::SMatrix;

/// Compute the angle between two vectors in radians.
pub fn angle_between_vectors<const DIM: usize>(
    a: &VectorDim<DIM>,
    b: &VectorDim<DIM>,
) -> f64 {
    let dot = a.dot(b);
    let na = a.norm();
    let nb = b.norm();
    if na < 1.0e-16 || nb < 1.0e-16 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0).acos()
}

/// Compute rotation matrix from Euler angles.
pub fn rotation_matrix_2d(alpha: f64) -> SMatrix<f64, 2, 2> {
    SMatrix::<f64, 2, 2>::new(
        alpha.cos(), -alpha.sin(),
        alpha.sin(),  alpha.cos(),
    )
}

/// Compute 3D rotation matrix from Euler angles (ZYZ convention).
pub fn rotation_matrix_3d(alpha: f64, beta: f64, gamma: f64) -> SMatrix<f64, 3, 3> {
    let ca = alpha.cos();
    let sa = alpha.sin();
    let cb = beta.cos();
    let sb = beta.sin();
    let cg = gamma.cos();
    let sg = gamma.sin();

    SMatrix::<f64, 3, 3>::new(
        ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg,  ca * sb,
        sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg,  sa * sb,
        -sb * cg,                sb * sg,                  cb,
    )
}
