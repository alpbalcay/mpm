//! 3D Hexahedron element with 8 or 20 nodes.

use crate::element::Element;
use crate::quadrature::{HexahedronQuadrature, Quadrature};
use mpm_core::{ElementDegree, ShapefnType, VectorDim};
use nalgebra::{DMatrix, DVector, SMatrix};

/// 3D Hexahedron element.
pub struct HexahedronElement {
    nnodes: usize,
}

impl HexahedronElement {
    pub fn new(nnodes: usize) -> Self {
        assert!(
            matches!(nnodes, 8 | 20),
            "HexahedronElement supports 8 or 20 nodes"
        );
        Self { nnodes }
    }
}

impl Element<3> for HexahedronElement {
    fn nfunctions(&self) -> usize {
        self.nnodes
    }

    fn shapefn(
        &self,
        xi: &VectorDim<3>,
        _particle_size: &VectorDim<3>,
        _deformation_gradient: &VectorDim<3>,
    ) -> DVector<f64> {
        let r = xi[0];
        let s = xi[1];
        let t = xi[2];

        match self.nnodes {
            8 => DVector::from_vec(vec![
                0.125 * (1.0 - r) * (1.0 - s) * (1.0 - t),
                0.125 * (1.0 + r) * (1.0 - s) * (1.0 - t),
                0.125 * (1.0 + r) * (1.0 + s) * (1.0 - t),
                0.125 * (1.0 - r) * (1.0 + s) * (1.0 - t),
                0.125 * (1.0 - r) * (1.0 - s) * (1.0 + t),
                0.125 * (1.0 + r) * (1.0 - s) * (1.0 + t),
                0.125 * (1.0 + r) * (1.0 + s) * (1.0 + t),
                0.125 * (1.0 - r) * (1.0 + s) * (1.0 + t),
            ]),
            20 => {
                let mut sf = DVector::zeros(20);
                // Corner nodes (0-7)
                let corners: [(f64, f64, f64); 8] = [
                    (-1.0, -1.0, -1.0),
                    (1.0, -1.0, -1.0),
                    (1.0, 1.0, -1.0),
                    (-1.0, 1.0, -1.0),
                    (-1.0, -1.0, 1.0),
                    (1.0, -1.0, 1.0),
                    (1.0, 1.0, 1.0),
                    (-1.0, 1.0, 1.0),
                ];

                for (i, &(ri, si, ti)) in corners.iter().enumerate() {
                    sf[i] = 0.125 * (1.0 + r * ri) * (1.0 + s * si) * (1.0 + t * ti)
                        * (r * ri + s * si + t * ti - 2.0);
                }

                // Mid-edge nodes (8-19)
                // Edges along r (s, t constant)
                sf[8] = 0.25 * (1.0 - r * r) * (1.0 - s) * (1.0 - t);
                sf[10] = 0.25 * (1.0 - r * r) * (1.0 + s) * (1.0 - t);
                sf[16] = 0.25 * (1.0 - r * r) * (1.0 - s) * (1.0 + t);
                sf[18] = 0.25 * (1.0 - r * r) * (1.0 + s) * (1.0 + t);

                // Edges along s (r, t constant)
                sf[9] = 0.25 * (1.0 + r) * (1.0 - s * s) * (1.0 - t);
                sf[11] = 0.25 * (1.0 - r) * (1.0 - s * s) * (1.0 - t);
                sf[17] = 0.25 * (1.0 + r) * (1.0 - s * s) * (1.0 + t);
                sf[19] = 0.25 * (1.0 - r) * (1.0 - s * s) * (1.0 + t);

                // Edges along t (r, s constant)
                sf[12] = 0.25 * (1.0 - r) * (1.0 - s) * (1.0 - t * t);
                sf[13] = 0.25 * (1.0 + r) * (1.0 - s) * (1.0 - t * t);
                sf[14] = 0.25 * (1.0 + r) * (1.0 + s) * (1.0 - t * t);
                sf[15] = 0.25 * (1.0 - r) * (1.0 + s) * (1.0 - t * t);
                sf
            }
            _ => unreachable!(),
        }
    }

    fn shapefn_local(
        &self,
        xi: &VectorDim<3>,
        particle_size: &VectorDim<3>,
        deformation_gradient: &VectorDim<3>,
    ) -> DVector<f64> {
        self.shapefn(xi, particle_size, deformation_gradient)
    }

    fn grad_shapefn(
        &self,
        xi: &VectorDim<3>,
        _particle_size: &VectorDim<3>,
        _deformation_gradient: &VectorDim<3>,
    ) -> DMatrix<f64> {
        let r = xi[0];
        let s = xi[1];
        let t = xi[2];

        match self.nnodes {
            8 => {
                let mut grad = DMatrix::zeros(8, 3);
                let signs: [(f64, f64, f64); 8] = [
                    (-1.0, -1.0, -1.0),
                    (1.0, -1.0, -1.0),
                    (1.0, 1.0, -1.0),
                    (-1.0, 1.0, -1.0),
                    (-1.0, -1.0, 1.0),
                    (1.0, -1.0, 1.0),
                    (1.0, 1.0, 1.0),
                    (-1.0, 1.0, 1.0),
                ];

                for (i, &(ri, si, ti)) in signs.iter().enumerate() {
                    grad[(i, 0)] = 0.125 * ri * (1.0 + s * si) * (1.0 + t * ti);
                    grad[(i, 1)] = 0.125 * (1.0 + r * ri) * si * (1.0 + t * ti);
                    grad[(i, 2)] = 0.125 * (1.0 + r * ri) * (1.0 + s * si) * ti;
                }
                grad
            }
            20 => {
                let mut grad = DMatrix::zeros(20, 3);

                let corners: [(f64, f64, f64); 8] = [
                    (-1.0, -1.0, -1.0),
                    (1.0, -1.0, -1.0),
                    (1.0, 1.0, -1.0),
                    (-1.0, 1.0, -1.0),
                    (-1.0, -1.0, 1.0),
                    (1.0, -1.0, 1.0),
                    (1.0, 1.0, 1.0),
                    (-1.0, 1.0, 1.0),
                ];

                for (i, &(ri, si, ti)) in corners.iter().enumerate() {
                    let fr = 1.0 + r * ri;
                    let fs = 1.0 + s * si;
                    let ft = 1.0 + t * ti;
                    let sum = r * ri + s * si + t * ti - 2.0;

                    grad[(i, 0)] = 0.125 * ri * fs * ft * sum + 0.125 * fr * fs * ft * ri;
                    grad[(i, 1)] = 0.125 * fr * si * ft * sum + 0.125 * fr * fs * ft * si;
                    grad[(i, 2)] = 0.125 * fr * fs * ti * sum + 0.125 * fr * fs * ft * ti;
                }

                // Mid-edge nodes along r
                grad[(8, 0)] = -0.5 * r * (1.0 - s) * (1.0 - t);
                grad[(8, 1)] = -0.25 * (1.0 - r * r) * (1.0 - t);
                grad[(8, 2)] = -0.25 * (1.0 - r * r) * (1.0 - s);

                grad[(10, 0)] = -0.5 * r * (1.0 + s) * (1.0 - t);
                grad[(10, 1)] = 0.25 * (1.0 - r * r) * (1.0 - t);
                grad[(10, 2)] = -0.25 * (1.0 - r * r) * (1.0 + s);

                grad[(16, 0)] = -0.5 * r * (1.0 - s) * (1.0 + t);
                grad[(16, 1)] = -0.25 * (1.0 - r * r) * (1.0 + t);
                grad[(16, 2)] = 0.25 * (1.0 - r * r) * (1.0 - s);

                grad[(18, 0)] = -0.5 * r * (1.0 + s) * (1.0 + t);
                grad[(18, 1)] = 0.25 * (1.0 - r * r) * (1.0 + t);
                grad[(18, 2)] = 0.25 * (1.0 - r * r) * (1.0 + s);

                // Mid-edge nodes along s
                grad[(9, 0)] = 0.25 * (1.0 - s * s) * (1.0 - t);
                grad[(9, 1)] = -0.5 * s * (1.0 + r) * (1.0 - t);
                grad[(9, 2)] = -0.25 * (1.0 + r) * (1.0 - s * s);

                grad[(11, 0)] = -0.25 * (1.0 - s * s) * (1.0 - t);
                grad[(11, 1)] = -0.5 * s * (1.0 - r) * (1.0 - t);
                grad[(11, 2)] = -0.25 * (1.0 - r) * (1.0 - s * s);

                grad[(17, 0)] = 0.25 * (1.0 - s * s) * (1.0 + t);
                grad[(17, 1)] = -0.5 * s * (1.0 + r) * (1.0 + t);
                grad[(17, 2)] = 0.25 * (1.0 + r) * (1.0 - s * s);

                grad[(19, 0)] = -0.25 * (1.0 - s * s) * (1.0 + t);
                grad[(19, 1)] = -0.5 * s * (1.0 - r) * (1.0 + t);
                grad[(19, 2)] = 0.25 * (1.0 - r) * (1.0 - s * s);

                // Mid-edge nodes along t
                grad[(12, 0)] = -0.25 * (1.0 - s) * (1.0 - t * t);
                grad[(12, 1)] = -0.25 * (1.0 - r) * (1.0 - t * t);
                grad[(12, 2)] = -0.5 * t * (1.0 - r) * (1.0 - s);

                grad[(13, 0)] = 0.25 * (1.0 - s) * (1.0 - t * t);
                grad[(13, 1)] = -0.25 * (1.0 + r) * (1.0 - t * t);
                grad[(13, 2)] = -0.5 * t * (1.0 + r) * (1.0 - s);

                grad[(14, 0)] = 0.25 * (1.0 + s) * (1.0 - t * t);
                grad[(14, 1)] = 0.25 * (1.0 + r) * (1.0 - t * t);
                grad[(14, 2)] = -0.5 * t * (1.0 + r) * (1.0 + s);

                grad[(15, 0)] = -0.25 * (1.0 + s) * (1.0 - t * t);
                grad[(15, 1)] = 0.25 * (1.0 - r) * (1.0 - t * t);
                grad[(15, 2)] = -0.5 * t * (1.0 - r) * (1.0 + s);

                grad
            }
            _ => unreachable!(),
        }
    }

    fn jacobian(
        &self,
        xi: &VectorDim<3>,
        nodal_coordinates: &DMatrix<f64>,
        particle_size: &VectorDim<3>,
        deformation_gradient: &VectorDim<3>,
    ) -> SMatrix<f64, 3, 3> {
        let grad = self.grad_shapefn(xi, particle_size, deformation_gradient);
        let j = grad.transpose() * nodal_coordinates;
        SMatrix::<f64, 3, 3>::from_fn(|i, j_idx| j[(i, j_idx)])
    }

    fn jacobian_local(
        &self,
        xi: &VectorDim<3>,
        nodal_coordinates: &DMatrix<f64>,
        particle_size: &VectorDim<3>,
        deformation_gradient: &VectorDim<3>,
    ) -> SMatrix<f64, 3, 3> {
        self.jacobian(xi, nodal_coordinates, particle_size, deformation_gradient)
    }

    fn dn_dx(
        &self,
        xi: &VectorDim<3>,
        nodal_coordinates: &DMatrix<f64>,
        particle_size: &VectorDim<3>,
        deformation_gradient: &VectorDim<3>,
    ) -> DMatrix<f64> {
        let jac = self.jacobian(xi, nodal_coordinates, particle_size, deformation_gradient);
        let jac_inv = jac.try_inverse().expect("Jacobian is singular");
        let grad = self.grad_shapefn(xi, particle_size, deformation_gradient);
        let jit = jac_inv.transpose();
        let jit_dyn = DMatrix::from_fn(3, 3, |i, j| jit[(i, j)]);
        &grad * &jit_dyn
    }

    fn bmatrix(
        &self,
        xi: &VectorDim<3>,
        nodal_coordinates: &DMatrix<f64>,
        particle_size: &VectorDim<3>,
        deformation_gradient: &VectorDim<3>,
    ) -> Vec<DMatrix<f64>> {
        let dn = self.dn_dx(xi, nodal_coordinates, particle_size, deformation_gradient);
        let nfn = self.nfunctions();
        let mut bmatrices = Vec::with_capacity(nfn);

        for i in 0..nfn {
            let mut b = DMatrix::zeros(6, 3);
            b[(0, 0)] = dn[(i, 0)]; // exx
            b[(1, 1)] = dn[(i, 1)]; // eyy
            b[(2, 2)] = dn[(i, 2)]; // ezz
            b[(3, 0)] = dn[(i, 1)]; // exy
            b[(3, 1)] = dn[(i, 0)];
            b[(4, 1)] = dn[(i, 2)]; // eyz
            b[(4, 2)] = dn[(i, 1)];
            b[(5, 0)] = dn[(i, 2)]; // exz
            b[(5, 2)] = dn[(i, 0)];
            bmatrices.push(b);
        }
        bmatrices
    }

    fn ni_nj_matrix(&self, xi_s: &[VectorDim<3>]) -> DMatrix<f64> {
        let nfn = self.nfunctions();
        let mut result = DMatrix::zeros(nfn, nfn);
        let zero_vec = VectorDim::<3>::zeros();

        for xi in xi_s {
            let sf = self.shapefn(xi, &zero_vec, &zero_vec);
            result += &sf * sf.transpose();
        }
        result
    }

    fn laplace_matrix(
        &self,
        xi_s: &[VectorDim<3>],
        nodal_coordinates: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let nfn = self.nfunctions();
        let mut result = DMatrix::zeros(nfn, nfn);
        let zero_vec = VectorDim::<3>::zeros();

        for xi in xi_s {
            let dn = self.dn_dx(xi, nodal_coordinates, &zero_vec, &zero_vec);
            result += &dn * dn.transpose();
        }
        result
    }

    fn degree(&self) -> ElementDegree {
        match self.nnodes {
            8 => ElementDegree::Linear,
            20 => ElementDegree::Quadratic,
            _ => unreachable!(),
        }
    }

    fn shapefn_type(&self) -> ShapefnType {
        ShapefnType::NormalMpm
    }

    fn unit_cell_coordinates(&self) -> DMatrix<f64> {
        match self.nnodes {
            8 => DMatrix::from_row_slice(
                8,
                3,
                &[
                    -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
                    1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
                    1.0, -1.0, 1.0, 1.0,
                ],
            ),
            20 => {
                let mut coords = DMatrix::zeros(20, 3);
                // Corner nodes
                let corner_coords: [(f64, f64, f64); 8] = [
                    (-1.0, -1.0, -1.0),
                    (1.0, -1.0, -1.0),
                    (1.0, 1.0, -1.0),
                    (-1.0, 1.0, -1.0),
                    (-1.0, -1.0, 1.0),
                    (1.0, -1.0, 1.0),
                    (1.0, 1.0, 1.0),
                    (-1.0, 1.0, 1.0),
                ];
                for (i, &(x, y, z)) in corner_coords.iter().enumerate() {
                    coords[(i, 0)] = x;
                    coords[(i, 1)] = y;
                    coords[(i, 2)] = z;
                }
                // Mid-edge nodes
                let mid_coords: [(f64, f64, f64); 12] = [
                    (0.0, -1.0, -1.0),
                    (1.0, 0.0, -1.0),
                    (0.0, 1.0, -1.0),
                    (-1.0, 0.0, -1.0),
                    (-1.0, -1.0, 0.0),
                    (1.0, -1.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (-1.0, 1.0, 0.0),
                    (0.0, -1.0, 1.0),
                    (1.0, 0.0, 1.0),
                    (0.0, 1.0, 1.0),
                    (-1.0, 0.0, 1.0),
                ];
                for (i, &(x, y, z)) in mid_coords.iter().enumerate() {
                    coords[(8 + i, 0)] = x;
                    coords[(8 + i, 1)] = y;
                    coords[(8 + i, 2)] = z;
                }
                coords
            }
            _ => unreachable!(),
        }
    }

    fn sides_indices(&self) -> DMatrix<i32> {
        DMatrix::from_row_slice(
            12,
            2,
            &[
                0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5,
                2, 6, 3, 7,
            ],
        )
    }

    fn corner_indices(&self) -> DVector<i32> {
        DVector::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7])
    }

    fn inhedron_indices(&self) -> DMatrix<i32> {
        DMatrix::from_row_slice(
            6,
            4,
            &[
                0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6,
                3, 0, 4, 7,
            ],
        )
    }

    fn face_indices(&self, face_id: usize) -> DVector<i32> {
        match face_id {
            0 => DVector::from_vec(vec![0, 3, 2, 1]),
            1 => DVector::from_vec(vec![4, 5, 6, 7]),
            2 => DVector::from_vec(vec![0, 1, 5, 4]),
            3 => DVector::from_vec(vec![1, 2, 6, 5]),
            4 => DVector::from_vec(vec![2, 3, 7, 6]),
            5 => DVector::from_vec(vec![3, 0, 4, 7]),
            _ => panic!("Invalid face_id for hexahedron: {}", face_id),
        }
    }

    fn nfaces(&self) -> usize {
        6
    }

    fn unit_element_length(&self) -> f64 {
        2.0
    }

    fn quadrature(&self, nquadratures: usize) -> Box<dyn Quadrature<3>> {
        Box::new(HexahedronQuadrature::new(nquadratures))
    }

    fn compute_volume(&self, nodal_coordinates: &DMatrix<f64>) -> f64 {
        let zero_vec = VectorDim::<3>::zeros();
        // Use 8-point Gauss quadrature to integrate the determinant of the Jacobian
        let quad = HexahedronQuadrature::new(8);
        let points = quad.points();
        let weights = quad.weights();
        let mut volume = 0.0;

        for i in 0..8 {
            let xi = VectorDim::<3>::new(points[(i, 0)], points[(i, 1)], points[(i, 2)]);
            let jac = self.jacobian(&xi, nodal_coordinates, &zero_vec, &zero_vec);
            volume += jac.determinant().abs() * weights[i];
        }
        volume
    }

    fn isvalid_natural_coordinates_analytical(&self) -> bool {
        false
    }

    fn natural_coordinates_analytical(
        &self,
        _point: &VectorDim<3>,
        _nodal_coordinates: &DMatrix<f64>,
    ) -> VectorDim<3> {
        panic!("Analytical coordinates not available for hexahedron")
    }
}
