//! NorSand constitutive model for sand.

use crate::material::{Material, ParticleData};
use crate::material_utility;
use mpm_core::{DenseMap, Index, Matrix6x6, Vector6d};

/// NorSand critical state sand model.
pub struct NorSand {
    id: Index,
    density: f64,
    poisson_ratio: f64,
    reference_pressure: f64,
    friction_cs: f64,
    n_coupling: f64,
    lambda: f64,
    kappa: f64,
    gamma: f64,
    chi: f64,
    hardening_modulus: f64,
    void_ratio_initial: f64,
    p_image_initial: f64,
    m_tc: f64,
    m_te: f64,
    chi_image: f64,
    bond_model: bool,
    p_cohesion_initial: f64,
    p_dilation_initial: f64,
    m_cohesion: f64,
    m_dilation: f64,
    m_modulus: f64,
}

impl NorSand {
    pub fn new(id: Index, properties: &serde_json::Value) -> Result<Self, String> {
        let density = properties.get("density").and_then(|v| v.as_f64())
            .ok_or("Missing 'density'")?;
        let poisson_ratio = properties.get("poisson_ratio").and_then(|v| v.as_f64())
            .ok_or("Missing 'poisson_ratio'")?;
        let reference_pressure = properties.get("reference_pressure").and_then(|v| v.as_f64())
            .ok_or("Missing 'reference_pressure'")?;
        let friction_cs_deg = properties.get("friction_cs").and_then(|v| v.as_f64())
            .ok_or("Missing 'friction_cs'")?;
        let n_coupling = properties.get("N").and_then(|v| v.as_f64())
            .ok_or("Missing 'N'")?;
        let lambda = properties.get("lambda").and_then(|v| v.as_f64())
            .ok_or("Missing 'lambda'")?;
        let kappa = properties.get("kappa").and_then(|v| v.as_f64())
            .ok_or("Missing 'kappa'")?;
        let gamma = properties.get("gamma").and_then(|v| v.as_f64())
            .ok_or("Missing 'gamma'")?;
        let chi = properties.get("chi").and_then(|v| v.as_f64())
            .ok_or("Missing 'chi'")?;
        let hardening_modulus = properties.get("hardening_modulus").and_then(|v| v.as_f64())
            .ok_or("Missing 'hardening_modulus'")?;
        let void_ratio_initial = properties.get("void_ratio_initial").and_then(|v| v.as_f64())
            .ok_or("Missing 'void_ratio_initial'")?;
        let p_image_initial = properties.get("p_image_initial").and_then(|v| v.as_f64())
            .ok_or("Missing 'p_image_initial'")?;

        let friction_cs = friction_cs_deg.to_radians();
        let sin_phi = friction_cs.sin();
        let m_tc = 6.0 * sin_phi / (3.0 - sin_phi);
        let m_te = 6.0 * sin_phi / (3.0 + sin_phi);
        let chi_image = chi / (1.0 - chi * lambda / m_tc);

        let bond_model = properties.get("bond_model").and_then(|v| v.as_bool())
            .unwrap_or(false);
        let p_cohesion_initial = properties.get("p_cohesion_initial").and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let p_dilation_initial = properties.get("p_dilation_initial").and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let m_cohesion = properties.get("m_cohesion").and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let m_dilation = properties.get("m_dilation").and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let m_modulus = properties.get("m_modulus").and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        Ok(Self {
            id, density, poisson_ratio, reference_pressure, friction_cs,
            n_coupling, lambda, kappa, gamma, chi, hardening_modulus,
            void_ratio_initial, p_image_initial, m_tc, m_te, chi_image,
            bond_model, p_cohesion_initial, p_dilation_initial,
            m_cohesion, m_dilation, m_modulus,
        })
    }

    fn compute_stress_impl(
        &self,
        stress: &Vector6d,
        dstrain: &Vector6d,
        _particle: &dyn ParticleData,
        state_vars: &mut DenseMap,
    ) -> Vector6d {
        // Use compression-positive convention
        let stress_neg = -stress;
        let dstrain_neg = -dstrain;

        let e = state_vars.get("void_ratio").copied().unwrap_or(self.void_ratio_initial);
        let p_image = state_vars.get("p_image").copied().unwrap_or(self.p_image_initial);
        let p_cohesion = state_vars.get("p_cohesion").copied().unwrap_or(self.p_cohesion_initial);
        let p_dilation = state_vars.get("p_dilation").copied().unwrap_or(self.p_dilation_initial);

        // Pressure-dependent elastic moduli
        let p = material_utility::mean_p(&stress_neg).abs().max(1.0);
        let bulk_modulus = ((1.0 + e) / self.kappa * p)
            + self.m_modulus * (p_cohesion + p_dilation);
        let shear_modulus = 3.0 * bulk_modulus * (1.0 - 2.0 * self.poisson_ratio)
            / (2.0 * (1.0 + self.poisson_ratio));

        // Elastic tensor
        let a1 = bulk_modulus + 4.0 * shear_modulus / 3.0;
        let a2 = bulk_modulus - 2.0 * shear_modulus / 3.0;
        let mut de = Matrix6x6::zeros();
        de[(0, 0)] = a1; de[(0, 1)] = a2; de[(0, 2)] = a2;
        de[(1, 0)] = a2; de[(1, 1)] = a1; de[(1, 2)] = a2;
        de[(2, 0)] = a2; de[(2, 1)] = a2; de[(2, 2)] = a1;
        de[(3, 3)] = shear_modulus;
        de[(4, 4)] = shear_modulus;
        de[(5, 5)] = shear_modulus;

        // Trial stress
        let trial_stress = stress_neg + de * dstrain_neg;

        let p_trial = material_utility::mean_p(&trial_stress).max(1.0e-10);
        let q_trial = material_utility::q(&trial_stress);
        let theta = material_utility::lode_angle(&trial_stress);

        // M_theta from Lode angle
        let m_theta = self.m_tc
            - (self.m_tc.powi(3)) / (3.0 + self.m_tc)
                * (3.0 * theta).cos();
        let m_theta = m_theta.max(self.m_te);

        state_vars.insert("M_theta".to_string(), m_theta);

        // Image parameters
        let e_image = self.gamma - self.lambda * (p_image / self.reference_pressure).ln();
        let psi_image = e - e_image;
        state_vars.insert("e_image".to_string(), e_image);
        state_vars.insert("psi_image".to_string(), psi_image);

        // M_image
        let m_image = m_theta - self.n_coupling * self.chi_image * psi_image / m_theta;
        let m_image_tc = self.m_tc - self.n_coupling * self.chi_image * psi_image / self.m_tc;

        state_vars.insert("M_image".to_string(), m_image);
        state_vars.insert("M_image_tc".to_string(), m_image_tc);

        // Yield function
        let p_eff = p_trial + p_cohesion;
        let p_i_eff = p_image + p_cohesion + p_dilation;

        let f = if p_eff > 0.0 && p_i_eff > 0.0 {
            q_trial / p_eff - m_image + m_image * (p_eff / p_i_eff).ln()
        } else {
            -1.0 // elastic
        };

        if f <= 0.0 {
            // Elastic
            let dvol = dstrain_neg[0] + dstrain_neg[1] + dstrain_neg[2];
            let new_e = e - dvol * (1.0 + e);
            state_vars.insert("void_ratio".to_string(), new_e);
            state_vars.insert("p_image".to_string(), p_image);
            return -trial_stress;
        }

        // Plastic: use standard continuum tangent
        // Simplified: scale deviatoric stress back to yield surface
        let mut p_val = p_trial;
        let mut q_val = q_trial;

        for _ in 0..50 {
            let p_eff = p_val + p_cohesion;
            let p_i_eff = p_image + p_cohesion + p_dilation;

            let f = if p_eff > 0.0 && p_i_eff > 0.0 {
                q_val / p_eff - m_image + m_image * (p_eff / p_i_eff).ln()
            } else {
                break;
            };

            if f.abs() < 1.0e-10 {
                break;
            }

            // Scale q
            let target_q = p_eff * (m_image - m_image * (p_eff / p_i_eff).ln());
            q_val = target_q.max(0.0);
            break;
        }

        // Reconstruct stress
        let dev = material_utility::deviatoric_stress(&trial_stress);
        let q_t = material_utility::q(&trial_stress);
        let ratio = if q_t > 1.0e-16 { q_val / q_t } else { 0.0 };

        let mut result = Vector6d::zeros();
        for i in 0..6 {
            result[i] = if i < 3 { p_val } else { 0.0 } + ratio * dev[i];
        }

        // Update state variables
        let dvol = dstrain_neg[0] + dstrain_neg[1] + dstrain_neg[2];
        let new_e = e - dvol * (1.0 + e);
        state_vars.insert("void_ratio".to_string(), new_e);
        state_vars.insert("p_image".to_string(), p_image);

        -result
    }
}

impl Material<2> for NorSand {
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
        vars.insert("M_theta".to_string(), self.m_tc);
        vars.insert("M_image".to_string(), self.m_tc);
        vars.insert("M_image_tc".to_string(), self.m_tc);
        vars.insert("void_ratio".to_string(), self.void_ratio_initial);
        vars.insert("e_image".to_string(), self.void_ratio_initial);
        vars.insert("psi_image".to_string(), 0.0);
        vars.insert("p_image".to_string(), self.p_image_initial);
        vars.insert("pdstrain".to_string(), 0.0);
        for i in 0..6 {
            vars.insert(format!("plastic_strain{}", i), 0.0);
        }
        if self.bond_model {
            vars.insert("p_cohesion".to_string(), self.p_cohesion_initial);
            vars.insert("p_dilation".to_string(), self.p_dilation_initial);
        }
        vars
    }

    fn state_variables(&self) -> Vec<String> {
        let mut vars: Vec<String> = vec![
            "M_theta", "M_image", "M_image_tc",
            "void_ratio", "e_image", "psi_image", "p_image", "pdstrain",
        ].into_iter().map(String::from).collect();
        for i in 0..6 {
            vars.push(format!("plastic_strain{}", i));
        }
        if self.bond_model {
            vars.push("p_cohesion".to_string());
            vars.push("p_dilation".to_string());
        }
        vars
    }
}

impl Material<3> for NorSand {
    fn id(&self) -> Index { self.id }
    fn density(&self) -> f64 { self.density }

    fn compute_stress(
        &self, stress: &Vector6d, dstrain: &Vector6d,
        particle: &dyn ParticleData, state_vars: &mut DenseMap,
    ) -> Vector6d {
        self.compute_stress_impl(stress, dstrain, particle, state_vars)
    }

    fn initialise_state_variables(&self) -> DenseMap {
        <Self as Material<2>>::initialise_state_variables(self)
    }

    fn state_variables(&self) -> Vec<String> {
        <Self as Material<2>>::state_variables(self)
    }
}
