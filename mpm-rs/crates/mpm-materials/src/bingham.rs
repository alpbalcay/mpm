//! Bingham viscoplastic fluid constitutive model.

use crate::material::{Material, ParticleData};
use mpm_core::{DenseMap, Index, Vector6d};

/// Bingham viscoplastic fluid material.
pub struct Bingham {
    id: Index,
    density: f64,
    bulk_modulus: f64,
    tau0: f64,
    mu: f64,
    critical_shear_rate: f64,
    incompressible: bool,
}

impl Bingham {
    pub fn new(id: Index, properties: &serde_json::Value) -> Result<Self, String> {
        let density = properties.get("density").and_then(|v| v.as_f64())
            .ok_or("Missing 'density'")?;
        let youngs_modulus = properties.get("youngs_modulus").and_then(|v| v.as_f64())
            .ok_or("Missing 'youngs_modulus'")?;
        let poisson_ratio = properties.get("poisson_ratio").and_then(|v| v.as_f64())
            .ok_or("Missing 'poisson_ratio'")?;
        let tau0 = properties.get("tau0").and_then(|v| v.as_f64())
            .ok_or("Missing 'tau0'")?;
        let mu = properties.get("mu").and_then(|v| v.as_f64())
            .ok_or("Missing 'mu'")?;
        let critical_shear_rate = properties.get("critical_shear_rate").and_then(|v| v.as_f64())
            .ok_or("Missing 'critical_shear_rate'")?;
        let incompressible = properties.get("incompressible").and_then(|v| v.as_bool())
            .unwrap_or(false);

        let bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio));

        Ok(Self {
            id,
            density,
            bulk_modulus,
            tau0,
            mu,
            critical_shear_rate,
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

        // Rate of deformation D = strain_rate / 2 for shear components
        let mut d = strain_rate;
        d[3] *= 0.5;
        d[4] *= 0.5;
        d[5] *= 0.5;

        // Volumetric strain rate
        let dvol = if dim == 2 {
            strain_rate[0] + strain_rate[1]
        } else {
            strain_rate[0] + strain_rate[1] + strain_rate[2]
        };

        // Shear rate: gamma_dot = sqrt(2 * D:D)
        let d_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            + 2.0 * (d[3] * d[3] + d[4] * d[4] + d[5] * d[5]);
        let gamma_dot_sq = 2.0 * d_sq;

        // Apparent viscosity
        let eta_app = if gamma_dot_sq > self.critical_shear_rate * self.critical_shear_rate {
            let gamma_dot = gamma_dot_sq.sqrt();
            2.0 * (self.tau0 / gamma_dot + self.mu)
        } else {
            0.0
        };

        // Deviatoric stress: tau = eta_app * D
        let mut tau = Vector6d::zeros();
        for i in 0..6 {
            tau[i] = eta_app * d[i];
        }

        // Von Mises yield check
        let tau_dev_sq = 0.5
            * (tau[0] * tau[0] + tau[1] * tau[1] + tau[2] * tau[2]
                + 2.0 * (tau[3] * tau[3] + tau[4] * tau[4] + tau[5] * tau[5]));
        if tau_dev_sq < self.tau0 * self.tau0 {
            tau = Vector6d::zeros();
        }

        // Update pressure
        let pressure = state_vars.get("pressure").copied().unwrap_or(0.0);
        let compressibility = if self.incompressible { 0.0 } else { 1.0 };
        let new_pressure = pressure + self.bulk_modulus * dvol * compressibility;
        state_vars.insert("pressure".to_string(), new_pressure);

        // Final stress: sigma = -p * I + tau
        let mut result = tau;
        result[0] -= new_pressure;
        result[1] -= new_pressure;
        if dim == 3 {
            result[2] -= new_pressure;
        } else {
            result[2] = -new_pressure;
        }
        result
    }
}

impl Material<2> for Bingham {
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

    fn initialise_state_variables(&self) -> DenseMap {
        let mut vars = DenseMap::new();
        vars.insert("pressure".to_string(), 0.0);
        vars
    }

    fn state_variables(&self) -> Vec<String> {
        vec!["pressure".to_string()]
    }
}

impl Material<3> for Bingham {
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

    fn initialise_state_variables(&self) -> DenseMap {
        let mut vars = DenseMap::new();
        vars.insert("pressure".to_string(), 0.0);
        vars
    }

    fn state_variables(&self) -> Vec<String> {
        vec!["pressure".to_string()]
    }
}
