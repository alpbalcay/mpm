//! Modified Cam Clay constitutive model with optional subloading and bonding.

use crate::material::{Material, ParticleData};
use crate::material_utility;
use mpm_core::{DenseMap, Index, Matrix6x6, Vector6d};

/// Modified Cam Clay material model.
pub struct ModifiedCamClay {
    id: Index,
    density: f64,
    youngs_modulus: f64,
    poisson_ratio: f64,
    p_ref: f64,
    e_ref: f64,
    ocr: f64,
    pc0: f64,
    m: f64,
    lambda: f64,
    kappa: f64,
    three_invariants: bool,
    subloading: bool,
    subloading_u: f64,
    bonding: bool,
    s_h: f64,
    mc_a: f64,
    mc_b: f64,
    mc_c: f64,
    mc_d: f64,
    m_degradation: f64,
    m_shear: f64,
}

impl ModifiedCamClay {
    pub fn new(id: Index, properties: &serde_json::Value) -> Result<Self, String> {
        let density = properties.get("density").and_then(|v| v.as_f64())
            .ok_or("Missing 'density'")?;
        let youngs_modulus = properties.get("youngs_modulus").and_then(|v| v.as_f64())
            .ok_or("Missing 'youngs_modulus'")?;
        let poisson_ratio = properties.get("poisson_ratio").and_then(|v| v.as_f64())
            .ok_or("Missing 'poisson_ratio'")?;
        let p_ref = properties.get("p_ref").and_then(|v| v.as_f64())
            .ok_or("Missing 'p_ref'")?;
        let e_ref = properties.get("e_ref").and_then(|v| v.as_f64())
            .ok_or("Missing 'e_ref'")?;
        let ocr = properties.get("ocr").and_then(|v| v.as_f64())
            .ok_or("Missing 'ocr'")?;
        let pc0 = properties.get("pc0").and_then(|v| v.as_f64())
            .ok_or("Missing 'pc0'")?;
        let m = properties.get("m").and_then(|v| v.as_f64())
            .ok_or("Missing 'm'")?;
        let lambda = properties.get("lambda").and_then(|v| v.as_f64())
            .ok_or("Missing 'lambda'")?;
        let kappa = properties.get("kappa").and_then(|v| v.as_f64())
            .ok_or("Missing 'kappa'")?;
        let three_invariants = properties.get("three_invariants").and_then(|v| v.as_bool())
            .unwrap_or(false);
        let subloading = properties.get("subloading").and_then(|v| v.as_bool())
            .unwrap_or(false);
        let subloading_u = properties.get("subloading_u").and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let bonding = properties.get("bonding").and_then(|v| v.as_bool())
            .unwrap_or(false);
        let s_h = properties.get("s_h").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let mc_a = properties.get("mc_a").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let mc_b = properties.get("mc_b").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let mc_c = properties.get("mc_c").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let mc_d = properties.get("mc_d").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let m_degradation = properties.get("m_degradation").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let m_shear = properties.get("m_shear").and_then(|v| v.as_f64()).unwrap_or(0.0);

        Ok(Self {
            id, density, youngs_modulus, poisson_ratio,
            p_ref, e_ref, ocr, pc0, m, lambda, kappa,
            three_invariants, subloading, subloading_u,
            bonding, s_h, mc_a, mc_b, mc_c, mc_d,
            m_degradation, m_shear,
        })
    }

    fn compute_stress_impl(
        &self,
        stress: &Vector6d,
        dstrain: &Vector6d,
        _particle: &dyn ParticleData,
        state_vars: &mut DenseMap,
    ) -> Vector6d {
        let e = state_vars.get("void_ratio").copied().unwrap_or(self.e_ref);
        let pc = state_vars.get("pc").copied().unwrap_or(self.pc0);

        // Pressure-dependent elastic moduli
        let p = material_utility::mean_p(stress).abs().max(1.0);
        let bulk_modulus = (1.0 + e) / self.kappa * p;
        let shear_modulus = 3.0 * bulk_modulus * (1.0 - 2.0 * self.poisson_ratio)
            / (2.0 * (1.0 + self.poisson_ratio));

        state_vars.insert("bulk_modulus".to_string(), bulk_modulus);
        state_vars.insert("shear_modulus".to_string(), shear_modulus);

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
        let trial_stress = stress + de * dstrain;

        let p_trial = -material_utility::mean_p(&trial_stress);
        let q_trial = material_utility::q(&trial_stress);

        let m_theta = self.m;
        state_vars.insert("M_theta".to_string(), m_theta);

        // Yield function: F = (q/M)^2 + p*(p - pc)
        let p_t = p_trial.max(1.0e-10);
        let f_trial = (q_trial / m_theta).powi(2) + p_t * (p_t - pc);

        state_vars.insert("f_function".to_string(), f_trial);
        state_vars.insert("p".to_string(), p_t);
        state_vars.insert("q".to_string(), q_trial);

        if f_trial <= 0.0 {
            // Elastic: update void ratio
            let dvol = dstrain[0] + dstrain[1] + dstrain[2];
            let new_e = e + dvol * (1.0 + e);
            state_vars.insert("void_ratio".to_string(), new_e);
            return trial_stress;
        }

        // Plastic: closest point projection
        let mut delta_phi = 0.0;
        let mut p_val = p_t;
        let mut q_val = q_trial;
        let mut pc_val = pc;
        let tol = 1.0e-10;

        for _ in 0..100 {
            let f = (q_val / m_theta).powi(2) + p_val * (p_val - pc_val);
            if f.abs() < tol {
                break;
            }

            // dF/d(delta_phi)
            let dp_ddphi = -2.0 * bulk_modulus * (2.0 * p_val - pc_val);
            let dq_ddphi = -6.0 * shear_modulus * q_val / (m_theta * m_theta);

            // pc evolution
            let dpc_ddphi = pc_val * (1.0 + e) / (self.lambda - self.kappa)
                * (2.0 * p_val - pc_val);

            let df_ddphi = 2.0 * q_val / (m_theta * m_theta) * dq_ddphi
                + dp_ddphi * (2.0 * p_val - pc_val)
                + p_val * (-dpc_ddphi);

            if df_ddphi.abs() < 1.0e-20 {
                break;
            }

            delta_phi -= f / df_ddphi;
            delta_phi = delta_phi.max(0.0);

            p_val = (p_t + bulk_modulus * delta_phi * pc_val)
                / (1.0 + 2.0 * bulk_modulus * delta_phi);
            q_val = q_trial / (1.0 + 6.0 * shear_modulus * delta_phi / (m_theta * m_theta));

            // Update pc
            let dpv = delta_phi * (2.0 * p_val - pc_val);
            pc_val = pc * ((1.0 + e) / (self.lambda - self.kappa) * dpv).exp();
        }

        state_vars.insert("pc".to_string(), pc_val);
        state_vars.insert("delta_phi".to_string(), delta_phi);
        state_vars.insert("p".to_string(), p_val);
        state_vars.insert("q".to_string(), q_val);

        // Reconstruct stress from p, q and deviatoric direction
        let dev_trial = material_utility::deviatoric_stress(&trial_stress);
        let q_t_check = material_utility::q(&trial_stress);

        let mut result = Vector6d::zeros();
        if q_t_check > 1.0e-16 {
            let ratio = q_val / q_t_check;
            for i in 0..6 {
                result[i] = -p_val * if i < 3 { 1.0 } else { 0.0 } + ratio * dev_trial[i];
            }
        } else {
            result[0] = -p_val;
            result[1] = -p_val;
            result[2] = -p_val;
        }

        // Update void ratio
        let dvol = dstrain[0] + dstrain[1] + dstrain[2];
        let new_e = e + dvol * (1.0 + e);
        state_vars.insert("void_ratio".to_string(), new_e);

        result
    }
}

impl Material<2> for ModifiedCamClay {
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
        let e0 = self.e_ref - self.lambda * (self.pc0 / self.p_ref).ln()
            + self.kappa * (self.pc0 / self.p_ref).ln();
        vars.insert("bulk_modulus".to_string(), self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio)));
        vars.insert("shear_modulus".to_string(), self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio)));
        vars.insert("p".to_string(), 0.0);
        vars.insert("q".to_string(), 0.0);
        vars.insert("theta".to_string(), 0.0);
        vars.insert("pc".to_string(), self.pc0);
        vars.insert("void_ratio".to_string(), e0);
        vars.insert("delta_phi".to_string(), 0.0);
        vars.insert("M_theta".to_string(), self.m);
        vars.insert("f_function".to_string(), 0.0);
        vars.insert("dpvstrain".to_string(), 0.0);
        vars.insert("dpdstrain".to_string(), 0.0);
        vars.insert("pvstrain".to_string(), 0.0);
        vars.insert("pdstrain".to_string(), 0.0);
        if self.bonding {
            vars.insert("chi".to_string(), 1.0);
            vars.insert("pcd".to_string(), 0.0);
            vars.insert("pcc".to_string(), 0.0);
        }
        if self.subloading {
            vars.insert("subloading_r".to_string(), self.ocr);
        }
        vars
    }

    fn state_variables(&self) -> Vec<String> {
        let mut vars = vec![
            "bulk_modulus", "shear_modulus", "p", "q", "theta", "pc",
            "void_ratio", "delta_phi", "M_theta", "f_function",
            "dpvstrain", "dpdstrain", "pvstrain", "pdstrain",
        ];
        if self.bonding {
            vars.extend_from_slice(&["chi", "pcd", "pcc"]);
        }
        if self.subloading {
            vars.push("subloading_r");
        }
        vars.into_iter().map(String::from).collect()
    }
}

impl Material<3> for ModifiedCamClay {
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
