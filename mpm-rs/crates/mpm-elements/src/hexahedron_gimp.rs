//! 3D Hexahedron GIMP element (64 nodes).

use crate::element::Element;
use crate::quadrature::{HexahedronQuadrature, Quadrature};
use crate::quadrilateral_gimp::QuadrilateralGimpElement;
use mpm_core::{ElementDegree, ShapefnType, VectorDim};
use nalgebra::{DMatrix, DVector, SMatrix};

/// 3D GIMP Hexahedron element with 64 nodes.
pub struct HexahedronGimpElement;

impl HexahedronGimpElement {
    pub fn new() -> Self {
        Self
    }
}

impl Element<3> for HexahedronGimpElement {
    fn nfunctions(&self) -> usize {
        64
    }

    fn shapefn(
        &self,
        xi: &VectorDim<3>,
        particle_size: &VectorDim<3>,
        _deformation_gradient: &VectorDim<3>,
    ) -> DVector<f64> {
        let lp = [
            particle_size[0] * 0.5,
            particle_size[1] * 0.5,
            particle_size[2] * 0.5,
        ];

        let positions = [-3.0, -1.0, 1.0, 3.0];
        let mut sf = DVector::zeros(64);
        let mut idx = 0;

        for &nz in &positions {
            for &ny in &positions {
                for &nx in &positions {
                    let sx = QuadrilateralGimpElement::gimp_shapefn_1d(xi[0] - nx, lp[0]);
                    let sy = QuadrilateralGimpElement::gimp_shapefn_1d(xi[1] - ny, lp[1]);
                    let sz = QuadrilateralGimpElement::gimp_shapefn_1d(xi[2] - nz, lp[2]);
                    sf[idx] = sx * sy * sz;
                    idx += 1;
                }
            }
        }
        sf
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
        particle_size: &VectorDim<3>,
        _deformation_gradient: &VectorDim<3>,
    ) -> DMatrix<f64> {
        let lp = [
            particle_size[0] * 0.5,
            particle_size[1] * 0.5,
            particle_size[2] * 0.5,
        ];

        let positions = [-3.0, -1.0, 1.0, 3.0];
        let mut grad = DMatrix::zeros(64, 3);
        let mut idx = 0;

        for &nz in &positions {
            for &ny in &positions {
                for &nx in &positions {
                    let dx = xi[0] - nx;
                    let dy = xi[1] - ny;
                    let dz = xi[2] - nz;

                    let sx = QuadrilateralGimpElement::gimp_shapefn_1d(dx, lp[0]);
                    let sy = QuadrilateralGimpElement::gimp_shapefn_1d(dy, lp[1]);
                    let sz = QuadrilateralGimpElement::gimp_shapefn_1d(dz, lp[2]);

                    let gx = QuadrilateralGimpElement::gimp_grad_shapefn_1d(dx, lp[0]);
                    let gy = QuadrilateralGimpElement::gimp_grad_shapefn_1d(dy, lp[1]);
                    let gz = QuadrilateralGimpElement::gimp_grad_shapefn_1d(dz, lp[2]);

                    grad[(idx, 0)] = gx * sy * sz;
                    grad[(idx, 1)] = sx * gy * sz;
                    grad[(idx, 2)] = sx * sy * gz;
                    idx += 1;
                }
            }
        }
        grad
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
            b[(0, 0)] = dn[(i, 0)];
            b[(1, 1)] = dn[(i, 1)];
            b[(2, 2)] = dn[(i, 2)];
            b[(3, 0)] = dn[(i, 1)];
            b[(3, 1)] = dn[(i, 0)];
            b[(4, 1)] = dn[(i, 2)];
            b[(4, 2)] = dn[(i, 1)];
            b[(5, 0)] = dn[(i, 2)];
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
        ElementDegree::Linear
    }

    fn shapefn_type(&self) -> ShapefnType {
        ShapefnType::Gimp
    }

    fn unit_cell_coordinates(&self) -> DMatrix<f64> {
        let positions = [-3.0, -1.0, 1.0, 3.0];
        let mut coords = DMatrix::zeros(64, 3);
        let mut idx = 0;
        for &z in &positions {
            for &y in &positions {
                for &x in &positions {
                    coords[(idx, 0)] = x;
                    coords[(idx, 1)] = y;
                    coords[(idx, 2)] = z;
                    idx += 1;
                }
            }
        }
        coords
    }

    fn sides_indices(&self) -> DMatrix<i32> {
        DMatrix::from_row_slice(
            12, 2,
            &[0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7],
        )
    }

    fn corner_indices(&self) -> DVector<i32> {
        DVector::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7])
    }

    fn inhedron_indices(&self) -> DMatrix<i32> {
        DMatrix::from_row_slice(
            6, 4,
            &[0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7],
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
            _ => panic!("Invalid face_id: {}", face_id),
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
        panic!("Analytical coordinates not available for HexahedronGIMP")
    }
}
