//! Linear elastic constitutive model.

use crate::material::{Material, ParticleData};
use mpm_core::{DenseMap, Index, Matrix6x6, Vector6d};

/// Isotropic linear elastic material.
pub struct LinearElastic {
    id: Index,
    density: f64,
    youngs_modulus: f64,
    poisson_ratio: f64,
    bulk_modulus: f64,
    shear_modulus: f64,
    de: Matrix6x6,
}

impl LinearElastic {
    pub fn new(id: Index, properties: &serde_json::Value) -> Result<Self, String> {
        let density = properties
            .get("density")
            .and_then(|v| v.as_f64())
            .ok_or("Missing 'density'")?;
        let youngs_modulus = properties
            .get("youngs_modulus")
            .and_then(|v| v.as_f64())
            .ok_or("Missing 'youngs_modulus'")?;
        let poisson_ratio = properties
            .get("poisson_ratio")
            .and_then(|v| v.as_f64())
            .ok_or("Missing 'poisson_ratio'")?;

        let bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio));
        let shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio));

        let de = compute_elastic_tensor(bulk_modulus, shear_modulus);

        Ok(Self {
            id,
            density,
            youngs_modulus,
            poisson_ratio,
            bulk_modulus,
            shear_modulus,
            de,
        })
    }
}

fn compute_elastic_tensor(bulk: f64, shear: f64) -> Matrix6x6 {
    let a1 = bulk + 4.0 * shear / 3.0;
    let a2 = bulk - 2.0 * shear / 3.0;

    let mut de = Matrix6x6::zeros();
    de[(0, 0)] = a1;
    de[(0, 1)] = a2;
    de[(0, 2)] = a2;
    de[(1, 0)] = a2;
    de[(1, 1)] = a1;
    de[(1, 2)] = a2;
    de[(2, 0)] = a2;
    de[(2, 1)] = a2;
    de[(2, 2)] = a1;
    de[(3, 3)] = shear;
    de[(4, 4)] = shear;
    de[(5, 5)] = shear;
    de
}

impl Material<2> for LinearElastic {
    fn id(&self) -> Index {
        self.id
    }

    fn density(&self) -> f64 {
        self.density
    }

    fn compute_stress(
        &self,
        stress: &Vector6d,
        dstrain: &Vector6d,
        _particle: &dyn ParticleData,
        _state_vars: &mut DenseMap,
    ) -> Vector6d {
        stress + self.de * dstrain
    }

    fn initialise_state_variables(&self) -> DenseMap {
        DenseMap::new()
    }

    fn state_variables(&self) -> Vec<String> {
        vec![]
    }
}

impl Material<3> for LinearElastic {
    fn id(&self) -> Index {
        self.id
    }

    fn density(&self) -> f64 {
        self.density
    }

    fn compute_stress(
        &self,
        stress: &Vector6d,
        dstrain: &Vector6d,
        _particle: &dyn ParticleData,
        _state_vars: &mut DenseMap,
    ) -> Vector6d {
        stress + self.de * dstrain
    }

    fn initialise_state_variables(&self) -> DenseMap {
        DenseMap::new()
    }

    fn state_variables(&self) -> Vec<String> {
        vec![]
    }
}
