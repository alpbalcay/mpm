//! Newtonian fluid constitutive model.

use crate::material::{Material, ParticleData};
use mpm_core::{DenseMap, Index, Vector6d};

/// Newtonian viscous fluid material.
pub struct Newtonian {
    id: Index,
    density: f64,
    bulk_modulus: f64,
    dynamic_viscosity: f64,
    incompressible: bool,
}

impl Newtonian {
    pub fn new(id: Index, properties: &serde_json::Value) -> Result<Self, String> {
        let density = properties
            .get("density")
            .and_then(|v| v.as_f64())
            .ok_or("Missing 'density'")?;
        let bulk_modulus = properties
            .get("bulk_modulus")
            .and_then(|v| v.as_f64())
            .ok_or("Missing 'bulk_modulus'")?;
        let dynamic_viscosity = properties
            .get("dynamic_viscosity")
            .and_then(|v| v.as_f64())
            .ok_or("Missing 'dynamic_viscosity'")?;
        let incompressible = properties
            .get("incompressible")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Self {
            id,
            density,
            bulk_modulus,
            dynamic_viscosity,
            incompressible,
        })
    }

    fn compute_stress_impl(
        &self,
        _stress: &Vector6d,
        _dstrain: &Vector6d,
        particle: &dyn ParticleData,
        state_vars: &mut DenseMap,
        dim: usize,
    ) -> Vector6d {
        let strain_rate = particle.strain_rate();

        // Volumetric strain rate
        let dvol = if dim == 2 {
            strain_rate[0] + strain_rate[1]
        } else {
            strain_rate[0] + strain_rate[1] + strain_rate[2]
        };

        // Update pressure
        let pressure = state_vars.get("pressure").copied().unwrap_or(0.0);
        let compressibility = if self.incompressible { 0.0 } else { 1.0 };
        let dp = -self.bulk_modulus * dvol * compressibility;
        let new_pressure = pressure + dp;
        state_vars.insert("pressure".to_string(), new_pressure);

        let mu = self.dynamic_viscosity;
        let sigma_vol = -new_pressure - (2.0 * mu / 3.0) * dvol;

        let mut result = Vector6d::zeros();
        result[0] = sigma_vol + 2.0 * mu * strain_rate[0];
        result[1] = sigma_vol + 2.0 * mu * strain_rate[1];
        if dim == 3 {
            result[2] = sigma_vol + 2.0 * mu * strain_rate[2];
        } else {
            result[2] = -new_pressure;
        }
        result[3] = mu * strain_rate[3];
        if dim == 3 {
            result[4] = mu * strain_rate[4];
            result[5] = mu * strain_rate[5];
        }
        result
    }
}

impl Material<2> for Newtonian {
    fn id(&self) -> Index { self.id }
    fn density(&self) -> f64 { self.density }

    fn compute_stress(
        &self,
        stress: &Vector6d,
        dstrain: &Vector6d,
        particle: &dyn ParticleData,
        state_vars: &mut DenseMap,
    ) -> Vector6d {
        self.compute_stress_impl(stress, dstrain, particle, state_vars, 2)
    }

    fn thermodynamic_pressure(&self, dvol: f64) -> f64 {
        -self.bulk_modulus * dvol
    }

    fn initialise_state_variables(&self) -> DenseMap {
        let mut vars = DenseMap::new();
        vars.insert("pressure".to_string(), 0.0);
        vars
    }

    fn state_variables(&self) -> Vec<String> {
        vec!["pressure".to_string()]
    }
}

impl Material<3> for Newtonian {
    fn id(&self) -> Index { self.id }
    fn density(&self) -> f64 { self.density }

    fn compute_stress(
        &self,
        stress: &Vector6d,
        dstrain: &Vector6d,
        particle: &dyn ParticleData,
        state_vars: &mut DenseMap,
    ) -> Vector6d {
        self.compute_stress_impl(stress, dstrain, particle, state_vars, 3)
    }

    fn thermodynamic_pressure(&self, dvol: f64) -> f64 {
        -self.bulk_modulus * dvol
    }

    fn initialise_state_variables(&self) -> DenseMap {
        let mut vars = DenseMap::new();
        vars.insert("pressure".to_string(), 0.0);
        vars
    }

    fn state_variables(&self) -> Vec<String> {
        vec!["pressure".to_string()]
    }
}
