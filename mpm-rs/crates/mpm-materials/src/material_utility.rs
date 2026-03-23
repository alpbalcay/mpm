//! Material utility functions for stress invariants and derivatives.

use mpm_core::Vector6d;

/// Compute mean stress p (tension positive convention).
pub fn mean_p(stress: &Vector6d) -> f64 {
    (stress[0] + stress[1] + stress[2]) / 3.0
}

/// Compute deviatoric stress components.
pub fn deviatoric_stress(stress: &Vector6d) -> Vector6d {
    let p = mean_p(stress);
    let mut dev = *stress;
    dev[0] -= p;
    dev[1] -= p;
    dev[2] -= p;
    dev
}

/// Compute J2 invariant.
pub fn j2(stress: &Vector6d) -> f64 {
    let s = deviatoric_stress(stress);
    0.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
        + s[3] * s[3]
        + s[4] * s[4]
        + s[5] * s[5]
}

/// Compute J3 invariant.
pub fn j3(stress: &Vector6d) -> f64 {
    let s = deviatoric_stress(stress);
    s[0] * (s[1] * s[2] - s[4] * s[4])
        - s[3] * (s[3] * s[2] - s[4] * s[5])
        + s[5] * (s[3] * s[4] - s[1] * s[5])
}

/// Compute deviatoric stress q = sqrt(3 * J2).
pub fn q(stress: &Vector6d) -> f64 {
    (3.0 * j2(stress)).sqrt()
}

/// Compute Lode angle theta (cosine convention).
pub fn lode_angle(stress: &Vector6d) -> f64 {
    let j2_val = j2(stress);
    if j2_val < 1.0e-16 {
        return 0.0;
    }
    let j3_val = j3(stress);
    let ratio = (3.0 * 3.0_f64.sqrt() * j3_val) / (2.0 * j2_val.powf(1.5));
    let ratio_clamped = ratio.clamp(-1.0, 1.0);
    ratio_clamped.acos() / 3.0
}

/// Compute equivalent plastic deviatoric strain.
pub fn pdstrain(plastic_strain: &Vector6d) -> f64 {
    let dev = deviatoric_stress(plastic_strain);
    (2.0 / 3.0
        * (dev[0] * dev[0]
            + dev[1] * dev[1]
            + dev[2] * dev[2]
            + 2.0 * (dev[3] * dev[3] + dev[4] * dev[4] + dev[5] * dev[5])))
    .sqrt()
}

/// Compute dp/dsigma = [1/3, 1/3, 1/3, 0, 0, 0]^T.
pub fn dp_dsigma() -> Vector6d {
    Vector6d::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0)
}

/// Compute dJ2/dsigma.
pub fn dj2_dsigma(stress: &Vector6d) -> Vector6d {
    let s = deviatoric_stress(stress);
    Vector6d::new(s[0], s[1], s[2], 2.0 * s[3], 2.0 * s[4], 2.0 * s[5])
}

/// Compute dq/dsigma.
pub fn dq_dsigma(stress: &Vector6d) -> Vector6d {
    let q_val = q(stress);
    if q_val < 1.0e-16 {
        return Vector6d::zeros();
    }
    let dj2 = dj2_dsigma(stress);
    dj2 * (3.0 / (2.0 * q_val))
}

/// Compute dJ3/dsigma.
pub fn dj3_dsigma(stress: &Vector6d) -> Vector6d {
    let s = deviatoric_stress(stress);
    let j2_val = j2(stress);

    let mut result = Vector6d::zeros();
    result[0] = s[1] * s[2] - s[4] * s[4] - j2_val / 3.0;
    result[1] = s[0] * s[2] - s[5] * s[5] - j2_val / 3.0;
    result[2] = s[0] * s[1] - s[3] * s[3] - j2_val / 3.0;
    result[3] = 2.0 * (s[4] * s[5] - s[3] * s[2]);
    result[4] = 2.0 * (s[3] * s[5] - s[4] * s[0]);
    result[5] = 2.0 * (s[3] * s[4] - s[5] * s[1]);
    result
}

/// Compute dtheta/dsigma.
pub fn dtheta_dsigma(stress: &Vector6d) -> Vector6d {
    let j2_val = j2(stress);
    if j2_val < 1.0e-16 {
        return Vector6d::zeros();
    }
    let j3_val = j3(stress);
    let dj2 = dj2_dsigma(stress);
    let dj3 = dj3_dsigma(stress);

    let r = 3.0 * 3.0_f64.sqrt();
    let arg = r * j3_val / (2.0 * j2_val.powf(1.5));
    let arg_clamped = arg.clamp(-1.0 + 1.0e-10, 1.0 - 1.0e-10);
    let denom = -3.0 * (1.0 - arg_clamped * arg_clamped).sqrt();

    if denom.abs() < 1.0e-16 {
        return Vector6d::zeros();
    }

    let factor = r / (2.0 * denom);
    let term1 = &dj3 / j2_val.powf(1.5);
    let term2 = &dj2 * (1.5 * j3_val / j2_val.powf(2.5));

    (term1 - term2) * factor
}
