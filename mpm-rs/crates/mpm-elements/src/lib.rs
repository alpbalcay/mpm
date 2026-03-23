//! Finite elements and quadrature rules for the MPM library.
//!
//! Provides element shape functions, gradient computations, and
//! Gauss quadrature rules for 2D and 3D elements.

pub mod element;
pub mod quadrature;

pub mod quadrilateral;
pub mod triangle;
pub mod hexahedron;
pub mod quadrilateral_gimp;
pub mod hexahedron_gimp;

pub use element::Element;
pub use quadrature::Quadrature;
