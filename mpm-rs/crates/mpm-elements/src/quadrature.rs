//! Quadrature (numerical integration) rules.

use mpm_core::VectorDim;
use nalgebra::{DMatrix, DVector};

/// Trait for Gauss quadrature rules.
pub trait Quadrature<const DIM: usize>: Send + Sync {
    /// Return quadrature points as a matrix (npoints x DIM).
    fn points(&self) -> DMatrix<f64>;

    /// Return quadrature weights as a vector (npoints).
    fn weights(&self) -> DVector<f64>;

    /// Return the number of quadrature points.
    fn npoints(&self) -> usize;
}

/// 2D quadrilateral quadrature rule.
pub struct QuadrilateralQuadrature {
    nquadratures: usize,
}

impl QuadrilateralQuadrature {
    pub fn new(nquadratures: usize) -> Self {
        assert!(
            matches!(nquadratures, 1 | 4 | 9 | 16),
            "QuadrilateralQuadrature supports 1, 4, 9, or 16 points"
        );
        Self { nquadratures }
    }
}

impl Quadrature<2> for QuadrilateralQuadrature {
    fn npoints(&self) -> usize {
        self.nquadratures
    }

    fn points(&self) -> DMatrix<f64> {
        match self.nquadratures {
            1 => DMatrix::from_row_slice(1, 2, &[0.0, 0.0]),
            4 => {
                let a = 1.0 / 3.0_f64.sqrt();
                DMatrix::from_row_slice(
                    4,
                    2,
                    &[-a, -a, a, -a, a, a, -a, a],
                )
            }
            9 => {
                let a = (3.0 / 5.0_f64).sqrt();
                DMatrix::from_row_slice(
                    9,
                    2,
                    &[
                        -a, -a, a, -a, a, a, -a, a, 0.0, -a, a, 0.0, 0.0, a,
                        -a, 0.0, 0.0, 0.0,
                    ],
                )
            }
            16 => {
                let a = ((3.0 / 7.0) - (2.0 / 7.0) * (6.0 / 5.0_f64).sqrt()).sqrt();
                let b = ((3.0 / 7.0) + (2.0 / 7.0) * (6.0 / 5.0_f64).sqrt()).sqrt();
                let pts = [-b, -a, a, b];
                let mut data = Vec::with_capacity(32);
                for &py in &pts {
                    for &px in &pts {
                        data.push(px);
                        data.push(py);
                    }
                }
                DMatrix::from_row_slice(16, 2, &data)
            }
            _ => unreachable!(),
        }
    }

    fn weights(&self) -> DVector<f64> {
        match self.nquadratures {
            1 => DVector::from_element(1, 4.0),
            4 => DVector::from_element(4, 1.0),
            9 => {
                let w1 = 25.0 / 81.0;
                let w2 = 40.0 / 81.0;
                let w3 = 64.0 / 81.0;
                DVector::from_vec(vec![w1, w1, w1, w1, w2, w2, w2, w2, w3])
            }
            16 => {
                let w1 = (18.0 - 30.0_f64.sqrt()) / 36.0;
                let w2 = (18.0 + 30.0_f64.sqrt()) / 36.0;
                let w = [w1, w2, w2, w1];
                let mut weights = Vec::with_capacity(16);
                for &wy in &w {
                    for &wx in &w {
                        weights.push(wx * wy);
                    }
                }
                DVector::from_vec(weights)
            }
            _ => unreachable!(),
        }
    }
}

/// 2D triangle quadrature rule.
pub struct TriangleQuadrature {
    nquadratures: usize,
}

impl TriangleQuadrature {
    pub fn new(nquadratures: usize) -> Self {
        assert!(
            matches!(nquadratures, 1 | 3),
            "TriangleQuadrature supports 1 or 3 points"
        );
        Self { nquadratures }
    }
}

impl Quadrature<2> for TriangleQuadrature {
    fn npoints(&self) -> usize {
        self.nquadratures
    }

    fn points(&self) -> DMatrix<f64> {
        match self.nquadratures {
            1 => DMatrix::from_row_slice(1, 2, &[1.0 / 3.0, 1.0 / 3.0]),
            3 => DMatrix::from_row_slice(
                3,
                2,
                &[
                    1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0,
                    2.0 / 3.0,
                ],
            ),
            _ => unreachable!(),
        }
    }

    fn weights(&self) -> DVector<f64> {
        match self.nquadratures {
            1 => DVector::from_element(1, 0.5),
            3 => DVector::from_vec(vec![1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]),
            _ => unreachable!(),
        }
    }
}

/// 3D hexahedron quadrature rule.
pub struct HexahedronQuadrature {
    nquadratures: usize,
}

impl HexahedronQuadrature {
    pub fn new(nquadratures: usize) -> Self {
        assert!(
            matches!(nquadratures, 1 | 8 | 27 | 64),
            "HexahedronQuadrature supports 1, 8, 27, or 64 points"
        );
        Self { nquadratures }
    }
}

impl Quadrature<3> for HexahedronQuadrature {
    fn npoints(&self) -> usize {
        self.nquadratures
    }

    fn points(&self) -> DMatrix<f64> {
        match self.nquadratures {
            1 => DMatrix::from_row_slice(1, 3, &[0.0, 0.0, 0.0]),
            8 => {
                let a = 1.0 / 3.0_f64.sqrt();
                let mut data = Vec::with_capacity(24);
                for &z in &[-a, a] {
                    for &y in &[-a, a] {
                        for &x in &[-a, a] {
                            data.push(x);
                            data.push(y);
                            data.push(z);
                        }
                    }
                }
                DMatrix::from_row_slice(8, 3, &data)
            }
            27 => {
                let a = (3.0 / 5.0_f64).sqrt();
                let pts = [-a, 0.0, a];
                let mut data = Vec::with_capacity(81);
                for &z in &pts {
                    for &y in &pts {
                        for &x in &pts {
                            data.push(x);
                            data.push(y);
                            data.push(z);
                        }
                    }
                }
                DMatrix::from_row_slice(27, 3, &data)
            }
            64 => {
                let a = ((3.0 / 7.0) - (2.0 / 7.0) * (6.0 / 5.0_f64).sqrt()).sqrt();
                let b = ((3.0 / 7.0) + (2.0 / 7.0) * (6.0 / 5.0_f64).sqrt()).sqrt();
                let pts = [-b, -a, a, b];
                let mut data = Vec::with_capacity(192);
                for &z in &pts {
                    for &y in &pts {
                        for &x in &pts {
                            data.push(x);
                            data.push(y);
                            data.push(z);
                        }
                    }
                }
                DMatrix::from_row_slice(64, 3, &data)
            }
            _ => unreachable!(),
        }
    }

    fn weights(&self) -> DVector<f64> {
        match self.nquadratures {
            1 => DVector::from_element(1, 8.0),
            8 => DVector::from_element(8, 1.0),
            27 => {
                let w1 = 5.0 / 9.0;
                let w2 = 8.0 / 9.0;
                let ws = [w1, w2, w1];
                let mut weights = Vec::with_capacity(27);
                for &wz in &ws {
                    for &wy in &ws {
                        for &wx in &ws {
                            weights.push(wx * wy * wz);
                        }
                    }
                }
                DVector::from_vec(weights)
            }
            64 => {
                let w1 = (18.0 - 30.0_f64.sqrt()) / 36.0;
                let w2 = (18.0 + 30.0_f64.sqrt()) / 36.0;
                let ws = [w1, w2, w2, w1];
                let mut weights = Vec::with_capacity(64);
                for &wz in &ws {
                    for &wy in &ws {
                        for &wx in &ws {
                            weights.push(wx * wy * wz);
                        }
                    }
                }
                DVector::from_vec(weights)
            }
            _ => unreachable!(),
        }
    }
}
