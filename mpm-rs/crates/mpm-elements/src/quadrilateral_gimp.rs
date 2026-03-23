//! 2D Quadrilateral GIMP element (16 nodes).

use crate::element::Element;
use crate::quadrature::{Quadrature, QuadrilateralQuadrature};
use mpm_core::{ElementDegree, ShapefnType, VectorDim};
use nalgebra::{DMatrix, DVector, SMatrix};

/// 2D GIMP (Generalized Interpolation Material Point) element.
///
/// Uses 16 nodes arranged in a 4x4 grid of the parent element and its
/// neighbors. GIMP shape functions depend on particle size to avoid
/// cell-crossing instability.
pub struct QuadrilateralGimpElement;

impl QuadrilateralGimpElement {
    pub fn new() -> Self {
        Self
    }

    /// Compute 1D GIMP shape function value.
    pub fn gimp_shapefn_1d(npni: f64, lp: f64) -> f64 {
        let h = 2.0; // element length in natural coords
        if npni <= -(h + lp) {
            0.0
        } else if npni <= -(h - lp) {
            let v = h + lp + npni;
            v * v / (4.0 * h * lp)
        } else if npni <= -lp {
            1.0 + npni / h
        } else if npni <= lp {
            1.0 - (npni * npni + lp * lp) / (2.0 * h * lp)
        } else if npni <= h - lp {
            1.0 - npni / h
        } else if npni <= h + lp {
            let v = h + lp - npni;
            v * v / (4.0 * h * lp)
        } else {
            0.0
        }
    }

    /// Compute 1D GIMP shape function gradient.
    pub fn gimp_grad_shapefn_1d(npni: f64, lp: f64) -> f64 {
        let h = 2.0;
        if npni <= -(h + lp) {
            0.0
        } else if npni <= -(h - lp) {
            (h + lp + npni) / (2.0 * h * lp)
        } else if npni <= -lp {
            1.0 / h
        } else if npni <= lp {
            -npni / (h * lp)
        } else if npni <= h - lp {
            -1.0 / h
        } else if npni <= h + lp {
            -(h + lp - npni) / (2.0 * h * lp)
        } else {
            0.0
        }
    }
}

impl Element<2> for QuadrilateralGimpElement {
    fn nfunctions(&self) -> usize {
        16
    }

    fn shapefn(
        &self,
        xi: &VectorDim<2>,
        particle_size: &VectorDim<2>,
        _deformation_gradient: &VectorDim<2>,
    ) -> DVector<f64> {
        let lp_x = particle_size[0] * 0.5;
        let lp_y = particle_size[1] * 0.5;

        // 16 nodes in 4x4 grid: natural coords at -3, -1, 1, 3
        let node_coords: [(f64, f64); 16] = [
            (-3.0, -3.0), (-1.0, -3.0), (1.0, -3.0), (3.0, -3.0),
            (-3.0, -1.0), (-1.0, -1.0), (1.0, -1.0), (3.0, -1.0),
            (-3.0,  1.0), (-1.0,  1.0), (1.0,  1.0), (3.0,  1.0),
            (-3.0,  3.0), (-1.0,  3.0), (1.0,  3.0), (3.0,  3.0),
        ];

        let mut sf = DVector::zeros(16);
        for (i, &(nx, ny)) in node_coords.iter().enumerate() {
            let npni_x = xi[0] - nx;
            let npni_y = xi[1] - ny;
            sf[i] = Self::gimp_shapefn_1d(npni_x, lp_x) * Self::gimp_shapefn_1d(npni_y, lp_y);
        }
        sf
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
        particle_size: &VectorDim<2>,
        _deformation_gradient: &VectorDim<2>,
    ) -> DMatrix<f64> {
        let lp_x = particle_size[0] * 0.5;
        let lp_y = particle_size[1] * 0.5;

        let node_coords: [(f64, f64); 16] = [
            (-3.0, -3.0), (-1.0, -3.0), (1.0, -3.0), (3.0, -3.0),
            (-3.0, -1.0), (-1.0, -1.0), (1.0, -1.0), (3.0, -1.0),
            (-3.0,  1.0), (-1.0,  1.0), (1.0,  1.0), (3.0,  1.0),
            (-3.0,  3.0), (-1.0,  3.0), (1.0,  3.0), (3.0,  3.0),
        ];

        let mut grad = DMatrix::zeros(16, 2);
        for (i, &(nx, ny)) in node_coords.iter().enumerate() {
            let npni_x = xi[0] - nx;
            let npni_y = xi[1] - ny;
            grad[(i, 0)] = Self::gimp_grad_shapefn_1d(npni_x, lp_x)
                * Self::gimp_shapefn_1d(npni_y, lp_y);
            grad[(i, 1)] = Self::gimp_shapefn_1d(npni_x, lp_x)
                * Self::gimp_grad_shapefn_1d(npni_y, lp_y);
        }
        grad
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
        ElementDegree::Linear
    }

    fn shapefn_type(&self) -> ShapefnType {
        ShapefnType::Gimp
    }

    fn unit_cell_coordinates(&self) -> DMatrix<f64> {
        let mut coords = DMatrix::zeros(16, 2);
        let positions = [-3.0, -1.0, 1.0, 3.0];
        let mut idx = 0;
        for &y in &positions {
            for &x in &positions {
                coords[(idx, 0)] = x;
                coords[(idx, 1)] = y;
                idx += 1;
            }
        }
        coords
    }

    fn sides_indices(&self) -> DMatrix<i32> {
        // Return base quad sides
        DMatrix::from_row_slice(4, 2, &[0, 1, 1, 2, 2, 3, 3, 0])
    }

    fn corner_indices(&self) -> DVector<i32> {
        DVector::from_vec(vec![0, 1, 2, 3])
    }

    fn inhedron_indices(&self) -> DMatrix<i32> {
        DMatrix::from_row_slice(4, 3, &[1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2])
    }

    fn face_indices(&self, face_id: usize) -> DVector<i32> {
        match face_id {
            0 => DVector::from_vec(vec![0, 1]),
            1 => DVector::from_vec(vec![1, 2]),
            2 => DVector::from_vec(vec![2, 3]),
            3 => DVector::from_vec(vec![3, 0]),
            _ => panic!("Invalid face_id: {}", face_id),
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
        // Use only the 4 corner nodes of the base element
        let npts = 4.min(nodal_coordinates.nrows());
        let mut area = 0.0;
        for i in 0..npts {
            let j = (i + 1) % npts;
            area += nodal_coordinates[(i, 0)] * nodal_coordinates[(j, 1)]
                - nodal_coordinates[(j, 0)] * nodal_coordinates[(i, 1)];
        }
        (area / 2.0).abs()
    }

    fn isvalid_natural_coordinates_analytical(&self) -> bool {
        false
    }

    fn natural_coordinates_analytical(
        &self,
        _point: &VectorDim<2>,
        _nodal_coordinates: &DMatrix<f64>,
    ) -> VectorDim<2> {
        panic!("Analytical solution for QuadGIMP has not been implemented")
    }
}
