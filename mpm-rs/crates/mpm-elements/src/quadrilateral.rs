//! 2D Quadrilateral element with 4, 8, or 9 nodes.

use crate::element::Element;
use crate::quadrature::{Quadrature, QuadrilateralQuadrature};
use mpm_core::{ElementDegree, ShapefnType, VectorDim};
use nalgebra::{DMatrix, DVector, SMatrix};

/// 2D Quadrilateral element parameterized by number of nodes.
pub struct QuadrilateralElement {
    nnodes: usize,
}

impl QuadrilateralElement {
    pub fn new(nnodes: usize) -> Self {
        assert!(
            matches!(nnodes, 4 | 8 | 9),
            "QuadrilateralElement supports 4, 8, or 9 nodes"
        );
        Self { nnodes }
    }
}

impl Element<2> for QuadrilateralElement {
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
            4 => DVector::from_vec(vec![
                0.25 * (1.0 - r) * (1.0 - s),
                0.25 * (1.0 + r) * (1.0 - s),
                0.25 * (1.0 + r) * (1.0 + s),
                0.25 * (1.0 - r) * (1.0 + s),
            ]),
            8 => {
                let mut sf = DVector::zeros(8);
                // Corner nodes
                sf[0] = 0.25 * (1.0 - r) * (1.0 - s) * (-r - s - 1.0);
                sf[1] = 0.25 * (1.0 + r) * (1.0 - s) * (r - s - 1.0);
                sf[2] = 0.25 * (1.0 + r) * (1.0 + s) * (r + s - 1.0);
                sf[3] = 0.25 * (1.0 - r) * (1.0 + s) * (-r + s - 1.0);
                // Mid-side nodes
                sf[4] = 0.5 * (1.0 - r * r) * (1.0 - s);
                sf[5] = 0.5 * (1.0 + r) * (1.0 - s * s);
                sf[6] = 0.5 * (1.0 - r * r) * (1.0 + s);
                sf[7] = 0.5 * (1.0 - r) * (1.0 - s * s);
                sf
            }
            9 => {
                let mut sf = DVector::zeros(9);
                // Corner nodes
                sf[0] = 0.25 * r * (r - 1.0) * s * (s - 1.0);
                sf[1] = 0.25 * r * (r + 1.0) * s * (s - 1.0);
                sf[2] = 0.25 * r * (r + 1.0) * s * (s + 1.0);
                sf[3] = 0.25 * r * (r - 1.0) * s * (s + 1.0);
                // Mid-side nodes
                sf[4] = 0.5 * (1.0 - r * r) * s * (s - 1.0);
                sf[5] = 0.5 * r * (r + 1.0) * (1.0 - s * s);
                sf[6] = 0.5 * (1.0 - r * r) * s * (s + 1.0);
                sf[7] = 0.5 * r * (r - 1.0) * (1.0 - s * s);
                // Center node
                sf[8] = (1.0 - r * r) * (1.0 - s * s);
                sf
            }
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
            4 => DMatrix::from_row_slice(
                4,
                2,
                &[
                    -0.25 * (1.0 - s), -0.25 * (1.0 - r),
                     0.25 * (1.0 - s), -0.25 * (1.0 + r),
                     0.25 * (1.0 + s),  0.25 * (1.0 + r),
                    -0.25 * (1.0 + s),  0.25 * (1.0 - r),
                ],
            ),
            8 => {
                let mut grad = DMatrix::zeros(8, 2);
                // dN/dr
                grad[(0, 0)] = -0.25 * (1.0 - s) * (-2.0 * r - s);
                grad[(1, 0)] = 0.25 * (1.0 - s) * (2.0 * r - s);
                grad[(2, 0)] = 0.25 * (1.0 + s) * (2.0 * r + s);
                grad[(3, 0)] = -0.25 * (1.0 + s) * (-2.0 * r + s);
                grad[(4, 0)] = -r * (1.0 - s);
                grad[(5, 0)] = 0.5 * (1.0 - s * s);
                grad[(6, 0)] = -r * (1.0 + s);
                grad[(7, 0)] = -0.5 * (1.0 - s * s);
                // dN/ds
                grad[(0, 1)] = -0.25 * (1.0 - r) * (-r - 2.0 * s);
                grad[(1, 1)] = -0.25 * (1.0 + r) * (r - 2.0 * s);
                grad[(2, 1)] = 0.25 * (1.0 + r) * (r + 2.0 * s);
                grad[(3, 1)] = 0.25 * (1.0 - r) * (-r + 2.0 * s);
                grad[(4, 1)] = -0.5 * (1.0 - r * r);
                grad[(5, 1)] = -(1.0 + r) * s;
                grad[(6, 1)] = 0.5 * (1.0 - r * r);
                grad[(7, 1)] = -(1.0 - r) * s;
                grad
            }
            9 => {
                let mut grad = DMatrix::zeros(9, 2);
                // dN/dr
                grad[(0, 0)] = 0.25 * (2.0 * r - 1.0) * s * (s - 1.0);
                grad[(1, 0)] = 0.25 * (2.0 * r + 1.0) * s * (s - 1.0);
                grad[(2, 0)] = 0.25 * (2.0 * r + 1.0) * s * (s + 1.0);
                grad[(3, 0)] = 0.25 * (2.0 * r - 1.0) * s * (s + 1.0);
                grad[(4, 0)] = -r * s * (s - 1.0);
                grad[(5, 0)] = 0.5 * (2.0 * r + 1.0) * (1.0 - s * s);
                grad[(6, 0)] = -r * s * (s + 1.0);
                grad[(7, 0)] = 0.5 * (2.0 * r - 1.0) * (1.0 - s * s);
                grad[(8, 0)] = -2.0 * r * (1.0 - s * s);
                // dN/ds
                grad[(0, 1)] = 0.25 * r * (r - 1.0) * (2.0 * s - 1.0);
                grad[(1, 1)] = 0.25 * r * (r + 1.0) * (2.0 * s - 1.0);
                grad[(2, 1)] = 0.25 * r * (r + 1.0) * (2.0 * s + 1.0);
                grad[(3, 1)] = 0.25 * r * (r - 1.0) * (2.0 * s + 1.0);
                grad[(4, 1)] = 0.5 * (1.0 - r * r) * (2.0 * s - 1.0);
                grad[(5, 1)] = -r * (r + 1.0) * s;
                grad[(6, 1)] = 0.5 * (1.0 - r * r) * (2.0 * s + 1.0);
                grad[(7, 1)] = -r * (r - 1.0) * s;
                grad[(8, 1)] = -2.0 * (1.0 - r * r) * s;
                grad
            }
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
        // J = grad^T * nodal_coordinates
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
        let jac_inv = jac
            .try_inverse()
            .expect("Jacobian is singular");
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
            // B matrix for 2D (3 strain components: exx, eyy, exy)
            // In Voigt notation with 6 components
            let mut b = DMatrix::zeros(6, 2);
            b[(0, 0)] = dn[(i, 0)]; // dN/dx -> exx
            b[(1, 1)] = dn[(i, 1)]; // dN/dy -> eyy
            b[(3, 0)] = dn[(i, 1)]; // dN/dy -> exy
            b[(3, 1)] = dn[(i, 0)]; // dN/dx -> exy
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
            4 => ElementDegree::Linear,
            8 | 9 => ElementDegree::Quadratic,
            _ => unreachable!(),
        }
    }

    fn shapefn_type(&self) -> ShapefnType {
        ShapefnType::NormalMpm
    }

    fn unit_cell_coordinates(&self) -> DMatrix<f64> {
        match self.nnodes {
            4 => DMatrix::from_row_slice(
                4,
                2,
                &[-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
            ),
            8 => DMatrix::from_row_slice(
                8,
                2,
                &[
                    -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, -1.0,
                    1.0, 0.0, 0.0, 1.0, -1.0, 0.0,
                ],
            ),
            9 => DMatrix::from_row_slice(
                9,
                2,
                &[
                    -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, -1.0,
                    1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,
                ],
            ),
            _ => unreachable!(),
        }
    }

    fn sides_indices(&self) -> DMatrix<i32> {
        // 4 sides, each with 2 node indices
        DMatrix::from_row_slice(
            4,
            2,
            &[0, 1, 1, 2, 2, 3, 3, 0],
        )
    }

    fn corner_indices(&self) -> DVector<i32> {
        DVector::from_vec(vec![0, 1, 2, 3])
    }

    fn inhedron_indices(&self) -> DMatrix<i32> {
        DMatrix::from_row_slice(
            4,
            3,
            &[
                1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2,
            ],
        )
    }

    fn face_indices(&self, face_id: usize) -> DVector<i32> {
        match face_id {
            0 => DVector::from_vec(vec![0, 1]),
            1 => DVector::from_vec(vec![1, 2]),
            2 => DVector::from_vec(vec![2, 3]),
            3 => DVector::from_vec(vec![3, 0]),
            _ => panic!("Invalid face_id for quadrilateral: {}", face_id),
        }
    }

    fn nfaces(&self) -> usize {
        4
    }

    fn unit_element_length(&self) -> f64 {
        2.0
    }

    fn quadrature(&self, nquadratures: usize) -> Box<dyn Quadrature<2>> {
        Box::new(QuadrilateralQuadrature::new(nquadratures))
    }

    fn compute_volume(&self, nodal_coordinates: &DMatrix<f64>) -> f64 {
        // Use the shoelace formula for a quadrilateral
        let npts = nodal_coordinates.nrows();
        let mut area = 0.0;
        for i in 0..npts {
            let j = (i + 1) % npts;
            area += nodal_coordinates[(i, 0)] * nodal_coordinates[(j, 1)]
                - nodal_coordinates[(j, 0)] * nodal_coordinates[(i, 1)];
        }
        (area / 2.0).abs()
    }

    fn isvalid_natural_coordinates_analytical(&self) -> bool {
        self.nnodes == 4
    }

    fn natural_coordinates_analytical(
        &self,
        point: &VectorDim<2>,
        nodal_coordinates: &DMatrix<f64>,
    ) -> VectorDim<2> {
        if self.nnodes != 4 {
            panic!("Analytical natural coordinates only for 4-node quad");
        }

        // Bilinear inverse mapping for 4-node quad
        let x = point[0];
        let y = point[1];

        let x0 = nodal_coordinates[(0, 0)];
        let y0 = nodal_coordinates[(0, 1)];
        let x1 = nodal_coordinates[(1, 0)];
        let y1 = nodal_coordinates[(1, 1)];
        let x2 = nodal_coordinates[(2, 0)];
        let y2 = nodal_coordinates[(2, 1)];
        let x3 = nodal_coordinates[(3, 0)];
        let y3 = nodal_coordinates[(3, 1)];

        // Use Newton-Raphson iteration
        let mut xi = VectorDim::<2>::zeros();
        for _ in 0..20 {
            let sf = self.shapefn(&xi, &VectorDim::<2>::zeros(), &VectorDim::<2>::zeros());
            let residual_x = sf[0] * x0 + sf[1] * x1 + sf[2] * x2 + sf[3] * x3 - x;
            let residual_y = sf[0] * y0 + sf[1] * y1 + sf[2] * y2 + sf[3] * y3 - y;

            if residual_x.abs() < 1e-12 && residual_y.abs() < 1e-12 {
                break;
            }

            let grad = self.grad_shapefn(&xi, &VectorDim::<2>::zeros(), &VectorDim::<2>::zeros());
            let j00 = grad[(0, 0)] * x0 + grad[(1, 0)] * x1 + grad[(2, 0)] * x2 + grad[(3, 0)] * x3;
            let j01 = grad[(0, 1)] * x0 + grad[(1, 1)] * x1 + grad[(2, 1)] * x2 + grad[(3, 1)] * x3;
            let j10 = grad[(0, 0)] * y0 + grad[(1, 0)] * y1 + grad[(2, 0)] * y2 + grad[(3, 0)] * y3;
            let j11 = grad[(0, 1)] * y0 + grad[(1, 1)] * y1 + grad[(2, 1)] * y2 + grad[(3, 1)] * y3;

            let det = j00 * j11 - j01 * j10;
            xi[0] -= (j11 * residual_x - j01 * residual_y) / det;
            xi[1] -= (-j10 * residual_x + j00 * residual_y) / det;
        }
        xi
    }
}
