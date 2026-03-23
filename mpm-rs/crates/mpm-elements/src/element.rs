//! Abstract element trait definition.

use mpm_core::{ElementDegree, ShapefnType, VectorDim};
use nalgebra::{DMatrix, DVector, SMatrix};

/// Trait for finite element shape functions and related computations.
///
/// All element types implement this trait, providing shape functions,
/// their gradients, Jacobian computations, and B-matrices for
/// strain-displacement relationships.
pub trait Element<const DIM: usize>: Send + Sync {
    /// Number of shape functions (nodes) in this element.
    fn nfunctions(&self) -> usize;

    /// Evaluate shape functions at natural coordinate `xi`.
    fn shapefn(
        &self,
        xi: &VectorDim<DIM>,
        particle_size: &VectorDim<DIM>,
        deformation_gradient: &VectorDim<DIM>,
    ) -> DVector<f64>;

    /// Evaluate shape functions using local (element-level) coordinates.
    fn shapefn_local(
        &self,
        xi: &VectorDim<DIM>,
        particle_size: &VectorDim<DIM>,
        deformation_gradient: &VectorDim<DIM>,
    ) -> DVector<f64>;

    /// Evaluate gradients of shape functions at natural coordinate `xi`.
    ///
    /// Returns a matrix of shape (nfunctions, DIM) where row i contains
    /// dN_i/dxi_j for each natural coordinate direction j.
    fn grad_shapefn(
        &self,
        xi: &VectorDim<DIM>,
        particle_size: &VectorDim<DIM>,
        deformation_gradient: &VectorDim<DIM>,
    ) -> DMatrix<f64>;

    /// Compute the Jacobian matrix at natural coordinate `xi`.
    fn jacobian(
        &self,
        xi: &VectorDim<DIM>,
        nodal_coordinates: &DMatrix<f64>,
        particle_size: &VectorDim<DIM>,
        deformation_gradient: &VectorDim<DIM>,
    ) -> SMatrix<f64, DIM, DIM>;

    /// Compute the Jacobian using local coordinates.
    fn jacobian_local(
        &self,
        xi: &VectorDim<DIM>,
        nodal_coordinates: &DMatrix<f64>,
        particle_size: &VectorDim<DIM>,
        deformation_gradient: &VectorDim<DIM>,
    ) -> SMatrix<f64, DIM, DIM>;

    /// Compute shape function gradients in physical coordinates (dN/dx).
    fn dn_dx(
        &self,
        xi: &VectorDim<DIM>,
        nodal_coordinates: &DMatrix<f64>,
        particle_size: &VectorDim<DIM>,
        deformation_gradient: &VectorDim<DIM>,
    ) -> DMatrix<f64>;

    /// Compute the B-matrix (strain-displacement) for each node.
    ///
    /// Returns a vector of matrices, one per shape function.
    fn bmatrix(
        &self,
        xi: &VectorDim<DIM>,
        nodal_coordinates: &DMatrix<f64>,
        particle_size: &VectorDim<DIM>,
        deformation_gradient: &VectorDim<DIM>,
    ) -> Vec<DMatrix<f64>>;

    /// Compute the Ni*Nj mass matrix using quadrature points.
    fn ni_nj_matrix(&self, xi_s: &[VectorDim<DIM>]) -> DMatrix<f64>;

    /// Compute the Laplacian matrix using quadrature points.
    fn laplace_matrix(
        &self,
        xi_s: &[VectorDim<DIM>],
        nodal_coordinates: &DMatrix<f64>,
    ) -> DMatrix<f64>;

    /// Return the polynomial degree of the element.
    fn degree(&self) -> ElementDegree;

    /// Return the shape function type.
    fn shapefn_type(&self) -> ShapefnType;

    /// Return the unit cell coordinates as a matrix (nnodes x DIM).
    fn unit_cell_coordinates(&self) -> DMatrix<f64>;

    /// Return the side node indices as a matrix.
    fn sides_indices(&self) -> DMatrix<i32>;

    /// Return the corner node indices.
    fn corner_indices(&self) -> DVector<i32>;

    /// Return the inhedron test indices.
    fn inhedron_indices(&self) -> DMatrix<i32>;

    /// Return the node indices for a given face.
    fn face_indices(&self, face_id: usize) -> DVector<i32>;

    /// Return the number of faces.
    fn nfaces(&self) -> usize;

    /// Return the unit element length (e.g., 2.0 for [-1,1] reference domain).
    fn unit_element_length(&self) -> f64;

    /// Create a quadrature rule with `nquadratures` points.
    fn quadrature(
        &self,
        nquadratures: usize,
    ) -> Box<dyn crate::quadrature::Quadrature<DIM>>;

    /// Compute the volume (area in 2D) of the element given nodal coordinates.
    fn compute_volume(&self, nodal_coordinates: &DMatrix<f64>) -> f64;

    /// Return whether analytical natural coordinates are available.
    fn isvalid_natural_coordinates_analytical(&self) -> bool;

    /// Compute natural coordinates analytically from a physical point.
    fn natural_coordinates_analytical(
        &self,
        point: &VectorDim<DIM>,
        nodal_coordinates: &DMatrix<f64>,
    ) -> VectorDim<DIM>;
}
