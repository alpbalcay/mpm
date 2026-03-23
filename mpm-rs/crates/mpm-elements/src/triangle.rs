//! 2D Triangle element with 3 or 6 nodes.

use crate::element::Element;
use crate::quadrature::{Quadrature, TriangleQuadrature};
use mpm_core::{ElementDegree, ShapefnType, VectorDim};
use nalgebra::{DMatrix, DVector, SMatrix};

/// 2D Triangle element.
pub struct TriangleElement {
    nnodes: usize,
}

impl TriangleElement {
    pub fn new(nnodes: usize) -> Self {
        assert!(
            matches!(nnodes, 3 | 6),
            "TriangleElement supports 3 or 6 nodes"
        );
        Self { nnodes }
    }
}

impl Element<2> for TriangleElement {
    fn nfunctions(&self) -> usize {
        self.nnodes
    }

    fn shapefn(
        &self,
        xi: &VectorDim<2>,
        _particle_size: &VectorDim<2>,
        _deformation_gradient: &VectorDim<2>,
    ) -> DVector<f64> {
        let r = xi[0];
        let s = xi[1];

        match self.nnodes {
            3 => DVector::from_vec(vec![
                1.0 - r - s,
                r,
                s,
            ]),
            6 => DVector::from_vec(vec![
                (1.0 - r - s) * (1.0 - 2.0 * r - 2.0 * s),
                r * (2.0 * r - 1.0),
                s * (2.0 * s - 1.0),
                4.0 * r * (1.0 - r - s),
                4.0 * r * s,
                4.0 * s * (1.0 - r - s),
            ]),
            _ => unreachable!(),
        }
    }

    fn shapefn_local(
        &self,
        xi: &VectorDim<2>,
        particle_size: &VectorDim<2>,
        deformation_gradient: &VectorDim<2>,
    ) -> DVector<f64> {
        self.shapefn(xi, particle_size, deformation_gradient)
    }

    fn grad_shapefn(
        &self,
        xi: &VectorDim<2>,
        _particle_size: &VectorDim<2>,
        _deformation_gradient: &VectorDim<2>,
    ) -> DMatrix<f64> {
        let r = xi[0];
        let s = xi[1];

        match self.nnodes {
            3 => DMatrix::from_row_slice(
                3,
                2,
                &[
                    -1.0, -1.0,
                     1.0,  0.0,
                     0.0,  1.0,
                ],
            ),
            6 => DMatrix::from_row_slice(
                6,
                2,
                &[
                    // dN0/dr, dN0/ds
                    -3.0 + 4.0 * r + 4.0 * s, -3.0 + 4.0 * r + 4.0 * s,
                    // dN1/dr, dN1/ds
                    4.0 * r - 1.0, 0.0,
                    // dN2/dr, dN2/ds
                    0.0, 4.0 * s - 1.0,
                    // dN3/dr, dN3/ds
                    4.0 - 8.0 * r - 4.0 * s, -4.0 * r,
                    // dN4/dr, dN4/ds
                    4.0 * s, 4.0 * r,
                    // dN5/dr, dN5/ds
                    -4.0 * s, 4.0 - 4.0 * r - 8.0 * s,
                ],
            ),
            _ => unreachable!(),
        }
    }

    fn jacobian(
        &self,
        xi: &VectorDim<2>,
        nodal_coordinates: &DMatrix<f64>,
        particle_size: &VectorDim<2>,
        deformation_gradient: &VectorDim<2>,
    ) -> SMatrix<f64, 2, 2> {
        let grad = self.grad_shapefn(xi, particle_size, deformation_gradient);
        let j = grad.transpose() * nodal_coordinates;
        SMatrix::<f64, 2, 2>::from_fn(|i, j_idx| j[(i, j_idx)])
    }

    fn jacobian_local(
        &self,
        xi: &VectorDim<2>,
        nodal_coordinates: &DMatrix<f64>,
        particle_size: &VectorDim<2>,
        deformation_gradient: &VectorDim<2>,
    ) -> SMatrix<f64, 2, 2> {
        self.jacobian(xi, nodal_coordinates, particle_size, deformation_gradient)
    }

    fn dn_dx(
        &self,
        xi: &VectorDim<2>,
        nodal_coordinates: &DMatrix<f64>,
        particle_size: &VectorDim<2>,
        deformation_gradient: &VectorDim<2>,
    ) -> DMatrix<f64> {
        let jac = self.jacobian(xi, nodal_coordinates, particle_size, deformation_gradient);
        let jac_inv = jac.try_inverse().expect("Jacobian is singular");
        let grad = self.grad_shapefn(xi, particle_size, deformation_gradient);
        let jit = jac_inv.transpose();
        let jit_dyn = DMatrix::from_fn(2, 2, |i, j| jit[(i, j)]);
        &grad * &jit_dyn
    }

    fn bmatrix(
        &self,
        xi: &VectorDim<2>,
        nodal_coordinates: &DMatrix<f64>,
        particle_size: &VectorDim<2>,
        deformation_gradient: &VectorDim<2>,
    ) -> Vec<DMatrix<f64>> {
        let dn = self.dn_dx(xi, nodal_coordinates, particle_size, deformation_gradient);
        let nfn = self.nfunctions();
        let mut bmatrices = Vec::with_capacity(nfn);

        for i in 0..nfn {
            let mut b = DMatrix::zeros(6, 2);
            b[(0, 0)] = dn[(i, 0)];
            b[(1, 1)] = dn[(i, 1)];
            b[(3, 0)] = dn[(i, 1)];
            b[(3, 1)] = dn[(i, 0)];
            bmatrices.push(b);
        }
        bmatrices
    }

    fn ni_nj_matrix(&self, xi_s: &[VectorDim<2>]) -> DMatrix<f64> {
        let nfn = self.nfunctions();
        let mut result = DMatrix::zeros(nfn, nfn);
        let zero_vec = VectorDim::<2>::zeros();

        for xi in xi_s {
            let sf = self.shapefn(xi, &zero_vec, &zero_vec);
            result += &sf * sf.transpose();
        }
        result
    }

    fn laplace_matrix(
        &self,
        xi_s: &[VectorDim<2>],
        nodal_coordinates: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let nfn = self.nfunctions();
        let mut result = DMatrix::zeros(nfn, nfn);
        let zero_vec = VectorDim::<2>::zeros();

        for xi in xi_s {
            let dn = self.dn_dx(xi, nodal_coordinates, &zero_vec, &zero_vec);
            result += &dn * dn.transpose();
        }
        result
    }

    fn degree(&self) -> ElementDegree {
        match self.nnodes {
            3 => ElementDegree::Linear,
            6 => ElementDegree::Quadratic,
            _ => unreachable!(),
        }
    }

    fn shapefn_type(&self) -> ShapefnType {
        ShapefnType::NormalMpm
    }

    fn unit_cell_coordinates(&self) -> DMatrix<f64> {
        match self.nnodes {
            3 => DMatrix::from_row_slice(
                3,
                2,
                &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ),
            6 => DMatrix::from_row_slice(
                6,
                2,
                &[
                    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.5, 0.0,
                    0.5,
                ],
            ),
            _ => unreachable!(),
        }
    }

    fn sides_indices(&self) -> DMatrix<i32> {
        DMatrix::from_row_slice(3, 2, &[0, 1, 1, 2, 2, 0])
    }

    fn corner_indices(&self) -> DVector<i32> {
        DVector::from_vec(vec![0, 1, 2])
    }

    fn inhedron_indices(&self) -> DMatrix<i32> {
        DMatrix::from_row_slice(3, 3, &[1, 0, 2, 2, 1, 0, 0, 2, 1])
    }

    fn face_indices(&self, face_id: usize) -> DVector<i32> {
        match face_id {
            0 => DVector::from_vec(vec![0, 1]),
            1 => DVector::from_vec(vec![1, 2]),
            2 => DVector::from_vec(vec![2, 0]),
            _ => panic!("Invalid face_id for triangle: {}", face_id),
        }
    }

    fn nfaces(&self) -> usize {
        3
    }

    fn unit_element_length(&self) -> f64 {
        1.0
    }

    fn quadrature(&self, nquadratures: usize) -> Box<dyn Quadrature<2>> {
        Box::new(TriangleQuadrature::new(nquadratures))
    }

    fn compute_volume(&self, nodal_coordinates: &DMatrix<f64>) -> f64 {
        let x0 = nodal_coordinates[(0, 0)];
        let y0 = nodal_coordinates[(0, 1)];
        let x1 = nodal_coordinates[(1, 0)];
        let y1 = nodal_coordinates[(1, 1)];
        let x2 = nodal_coordinates[(2, 0)];
        let y2 = nodal_coordinates[(2, 1)];
        0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs()
    }

    fn isvalid_natural_coordinates_analytical(&self) -> bool {
        self.nnodes == 3
    }

    fn natural_coordinates_analytical(
        &self,
        point: &VectorDim<2>,
        nodal_coordinates: &DMatrix<f64>,
    ) -> VectorDim<2> {
        if self.nnodes != 3 {
            panic!("Analytical coordinates only for 3-node triangle");
        }

        let x0 = nodal_coordinates[(0, 0)];
        let y0 = nodal_coordinates[(0, 1)];
        let x1 = nodal_coordinates[(1, 0)];
        let y1 = nodal_coordinates[(1, 1)];
        let x2 = nodal_coordinates[(2, 0)];
        let y2 = nodal_coordinates[(2, 1)];

        let det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);

        let mut xi = VectorDim::<2>::zeros();
        xi[0] = ((y1 - y2) * (point[0] - x2) + (x2 - x1) * (point[1] - y2)) / det;
        xi[1] = ((y2 - y0) * (point[0] - x2) + (x0 - x2) * (point[1] - y2)) / det;

        // Convert barycentric to natural: xi = (L1, L2) where L0 = 1 - L1 - L2
        // For triangle element: xi[0] = L1, xi[1] = L2
        xi
    }
}
