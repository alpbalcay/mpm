//! Mohr-Coulomb elastoplastic constitutive model with tension cutoff.

use crate::material::{Material, ParticleData};
use crate::material_utility;
use mpm_core::{DenseMap, Index, Matrix6x6, Vector6d};

/// Yield state for Mohr-Coulomb model.
#[derive(Debug, Clone, Copy, PartialEq)]
enum YieldState {
    Elastic,
    Tensile,
    Shear,
}

/// Mohr-Coulomb material with optional softening.
pub struct MohrCoulomb {
    id: Index,
    density: f64,
    youngs_modulus: f64,
    poisson_ratio: f64,
    bulk_modulus: f64,
    shear_modulus: f64,
    phi_peak: f64,
    psi_peak: f64,
    cohesion_peak: f64,
    phi_residual: f64,
    psi_residual: f64,
    cohesion_residual: f64,
    pdstrain_peak: f64,
    pdstrain_residual: f64,
    tension_cutoff: f64,
    softening: bool,
    de: Matrix6x6,
}

impl MohrCoulomb {
    pub fn new(id: Index, properties: &serde_json::Value) -> Result<Self, String> {
        let density = properties.get("density").and_then(|v| v.as_f64())
            .ok_or("Missing 'density'")?;
        let youngs_modulus = properties.get("youngs_modulus").and_then(|v| v.as_f64())
            .ok_or("Missing 'youngs_modulus'")?;
        let poisson_ratio = properties.get("poisson_ratio").and_then(|v| v.as_f64())
            .ok_or("Missing 'poisson_ratio'")?;
        let friction = properties.get("friction").and_then(|v| v.as_f64())
            .ok_or("Missing 'friction'")?;
        let dilation = properties.get("dilation").and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let cohesion = properties.get("cohesion").and_then(|v| v.as_f64())
            .ok_or("Missing 'cohesion'")?;
        let tension_cutoff = properties.get("tension_cutoff").and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let softening = properties.get("softening").and_then(|v| v.as_bool())
            .unwrap_or(false);

        let phi_peak = friction.to_radians();
        let psi_peak = dilation.to_radians();

        let phi_residual = properties.get("residual_friction").and_then(|v| v.as_f64())
            .unwrap_or(friction).to_radians();
        let psi_residual = properties.get("residual_dilation").and_then(|v| v.as_f64())
            .unwrap_or(dilation).to_radians();
        let cohesion_residual = properties.get("residual_cohesion").and_then(|v| v.as_f64())
            .unwrap_or(cohesion);
        let pdstrain_peak = properties.get("peak_pdstrain").and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let pdstrain_residual = properties.get("residual_pdstrain").and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio));
        let shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio));

        let a1 = bulk_modulus + 4.0 * shear_modulus / 3.0;
        let a2 = bulk_modulus - 2.0 * shear_modulus / 3.0;

        let mut de = Matrix6x6::zeros();
        de[(0, 0)] = a1; de[(0, 1)] = a2; de[(0, 2)] = a2;
        de[(1, 0)] = a2; de[(1, 1)] = a1; de[(1, 2)] = a2;
        de[(2, 0)] = a2; de[(2, 1)] = a2; de[(2, 2)] = a1;
        de[(3, 3)] = shear_modulus;
        de[(4, 4)] = shear_modulus;
        de[(5, 5)] = shear_modulus;

        Ok(Self {
            id,
            density,
            youngs_modulus,
            poisson_ratio,
            bulk_modulus,
            shear_modulus,
            phi_peak,
            psi_peak,
            cohesion_peak: cohesion,
            phi_residual,
            psi_residual,
            cohesion_residual,
            pdstrain_peak,
            pdstrain_residual,
            tension_cutoff,
            softening,
            de,
        })
    }

    fn compute_stress_impl(
        &self,
        stress: &Vector6d,
        dstrain: &Vector6d,
        _particle: &dyn ParticleData,
        state_vars: &mut DenseMap,
    ) -> Vector6d {
        let current_pdstrain = state_vars.get("pdstrain").copied().unwrap_or(0.0);

        // Softening: update yield parameters
        let (phi, psi, cohesion) = if self.softening && current_pdstrain > self.pdstrain_peak {
            if current_pdstrain < self.pdstrain_residual {
                let frac = (current_pdstrain - self.pdstrain_peak)
                    / (self.pdstrain_residual - self.pdstrain_peak);
                let phi = self.phi_peak + frac * (self.phi_residual - self.phi_peak);
                let psi = self.psi_peak + frac * (self.psi_residual - self.psi_peak);
                let c = self.cohesion_peak + frac * (self.cohesion_residual - self.cohesion_peak);
                (phi, psi, c)
            } else {
                (self.phi_residual, self.psi_residual, self.cohesion_residual)
            }
        } else {
            (self.phi_peak, self.psi_peak, self.cohesion_peak)
        };

        state_vars.insert("phi".to_string(), phi);
        state_vars.insert("psi".to_string(), psi);
        state_vars.insert("cohesion".to_string(), cohesion);

        // Elastic trial stress
        let trial_stress = stress + self.de * dstrain;

        // Compute invariants
        let epsilon = material_utility::mean_p(&trial_stress) * 3.0_f64.sqrt();
        let rho = (2.0 * material_utility::j2(&trial_stress)).sqrt();
        let theta = material_utility::lode_angle(&trial_stress);

        state_vars.insert("epsilon".to_string(), epsilon);
        state_vars.insert("rho".to_string(), rho);
        state_vars.insert("theta".to_string(), theta);

        // Check yield
        let yield_state = self.compute_yield_state(epsilon, rho, theta, phi, cohesion);

        if yield_state == YieldState::Elastic {
            return trial_stress;
        }

        // Plastic return mapping (simplified)
        let mut updated_stress = trial_stress;
        let mut pdstrain_acc = current_pdstrain;
        let tol = 1.0e-10;

        for _ in 0..100 {
            let eps = material_utility::mean_p(&updated_stress) * 3.0_f64.sqrt();
            let rho_val = (2.0 * material_utility::j2(&updated_stress)).sqrt();
            let theta_val = material_utility::lode_angle(&updated_stress);

            let f = match yield_state {
                YieldState::Tensile => {
                    (2.0_f64 / 3.0).sqrt() * theta_val.cos() * rho_val + eps / 3.0_f64.sqrt()
                        - self.tension_cutoff
                }
                YieldState::Shear => {
                    let sin_phi = phi.sin();
                    let cos_phi = phi.cos();
                    1.5_f64.sqrt() * rho_val
                        * ((theta_val + std::f64::consts::FRAC_PI_3).sin()
                            / (3.0_f64.sqrt() * cos_phi)
                            + (theta_val + std::f64::consts::FRAC_PI_3).cos() * sin_phi
                                / 3.0)
                        + eps * sin_phi / 3.0_f64.sqrt()
                        - cohesion
                }
                YieldState::Elastic => unreachable!(),
            };

            if f.abs() < tol {
                break;
            }

            // Simple radial return
            let scale = 0.9;
            let q_val = material_utility::q(&updated_stress);
            if q_val > tol {
                let p = material_utility::mean_p(&updated_stress);
                let dev = material_utility::deviatoric_stress(&updated_stress);
                let new_q = (q_val - f * scale).max(0.0);
                let ratio = if q_val > 1.0e-16 { new_q / q_val } else { 0.0 };
                updated_stress[0] = p + ratio * dev[0];
                updated_stress[1] = p + ratio * dev[1];
                updated_stress[2] = p + ratio * dev[2];
                updated_stress[3] = ratio * dev[3];
                updated_stress[4] = ratio * dev[4];
                updated_stress[5] = ratio * dev[5];
            }

            pdstrain_acc += f.abs() * 0.01;
        }

        state_vars.insert("pdstrain".to_string(), pdstrain_acc);
        updated_stress
    }

    fn compute_yield_state(
        &self,
        epsilon: f64,
        rho: f64,
        theta: f64,
        phi: f64,
        cohesion: f64,
    ) -> YieldState {
        // Tension yield
        let ft = (2.0_f64 / 3.0).sqrt() * theta.cos() * rho + epsilon / 3.0_f64.sqrt()
            - self.tension_cutoff;

        // Shear yield
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let fs = 1.5_f64.sqrt() * rho
            * ((theta + std::f64::consts::FRAC_PI_3).sin() / (3.0_f64.sqrt() * cos_phi)
                + (theta + std::f64::consts::FRAC_PI_3).cos() * sin_phi / 3.0)
            + epsilon * sin_phi / 3.0_f64.sqrt()
            - cohesion;

        if ft > 1.0e-10 && ft >= fs {
            YieldState::Tensile
        } else if fs > 1.0e-10 {
            YieldState::Shear
        } else {
            YieldState::Elastic
        }
    }
}

impl Material<2> for MohrCoulomb {
    fn id(&self) -> Index { self.id }
    fn density(&self) -> f64 { self.density }

    fn compute_stress(
        &self, stress: &Vector6d, dstrain: &Vector6d,
        particle: &dyn ParticleData, state_vars: &mut DenseMap,
    ) -> Vector6d {
        self.compute_stress_impl(stress, dstrain, particle, state_vars)
    }

    fn initialise_state_variables(&self) -> DenseMap {
        let mut vars = DenseMap::new();
        vars.insert("phi".to_string(), self.phi_peak);
        vars.insert("psi".to_string(), self.psi_peak);
        vars.insert("cohesion".to_string(), self.cohesion_peak);
        vars.insert("epsilon".to_string(), 0.0);
        vars.insert("rho".to_string(), 0.0);
        vars.insert("theta".to_string(), 0.0);
        vars.insert("pdstrain".to_string(), 0.0);
        vars
    }

    fn state_variables(&self) -> Vec<String> {
        vec!["phi", "psi", "cohesion", "epsilon", "rho", "theta", "pdstrain"]
            .into_iter().map(String::from).collect()
    }
}

impl Material<3> for MohrCoulomb {
    fn id(&self) -> Index { self.id }
    fn density(&self) -> f64 { self.density }

    fn compute_stress(
        &self, stress: &Vector6d, dstrain: &Vector6d,
        particle: &dyn ParticleData, state_vars: &mut DenseMap,
    ) -> Vector6d {
        self.compute_stress_impl(stress, dstrain, particle, state_vars)
    }

    fn initialise_state_variables(&self) -> DenseMap {
        let mut vars = DenseMap::new();
        vars.insert("phi".to_string(), self.phi_peak);
        vars.insert("psi".to_string(), self.psi_peak);
        vars.insert("cohesion".to_string(), self.cohesion_peak);
        vars.insert("epsilon".to_string(), 0.0);
        vars.insert("rho".to_string(), 0.0);
        vars.insert("theta".to_string(), 0.0);
        vars.insert("pdstrain".to_string(), 0.0);
        vars
    }

    fn state_variables(&self) -> Vec<String> {
        vec!["phi", "psi", "cohesion", "epsilon", "rho", "theta", "pdstrain"]
            .into_iter().map(String::from).collect()
    }
}
