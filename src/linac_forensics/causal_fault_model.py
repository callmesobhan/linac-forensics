"""Physics-Informed Electro-Mechanical Fault Library & Bayesian BIC Attribution Engine.

Research Software — NOT A MEDICAL DEVICE.

Separates and quantifies five canonical radiotherapy linear accelerator failure modes
from 20–50 Hz trajectory telemetry using matched-filter regression, orthogonal
multi-channel standardization, and Bayesian Information Criterion (BIC/AIC) model selection:

1. MLC Leaf Motor Friction / Viscous Drag: Δx_i(t) ≈ -τ_i * v_i(t).
2. MLC Coulomb / Stribeck Stiction & Leadscrew Backlash: Δx_i(t) ≈ -β_i * tanh(v_i(t) / v_0).
3. Systematic Bank / DLG Calibration Offset: constant aperture widening/narrowing Δgap = c.
4. Gantry Gravity-Sag Harmonic: periodic encoder/head sag Δy(θ) = A sin(θ) + B cos(θ).
5. Dose-Rate / Ion-Chamber Calibration Drift: multiplicative MU shift ΔMU / MU = δ.
"""

from __future__ import annotations

import numpy as np

from linac_forensics.ir import (
    CausalFaultDiagnosis,
    FaultHypothesisResult,
    MLCFaultSignature,
    TrajectoryLogTensor,
)
from linac_forensics.log_parser import validate_trajectory_tensor
from linac_forensics.provenance import build_provenance, sha256_json_payload

# Nominal clinical telemetry noise floors for multi-channel standardization
_SIGMA_MLC_MM = 0.05
_SIGMA_SAG_MM = 0.05
_SIGMA_MU_REL = 0.002  # 0.2%
_STRIBECK_V0_MM_S = 5.0


def _gaussian_ic(rss: float, n_obs: int, k_params: int) -> tuple[float, float]:
    """Compute Gaussian log-likelihood AIC and BIC with safe numerical floor."""
    mean_sq = max(rss / max(n_obs, 1), 1e-12)
    base = n_obs * float(np.log(mean_sq))
    aic = base + 2.0 * k_params
    bic = base + k_params * float(np.log(max(n_obs, 2)))
    return aic, bic


def diagnose_causal_faults(tensor: TrajectoryLogTensor) -> CausalFaultDiagnosis:
    """Perform physics-informed matched-filter & Bayesian BIC fault attribution."""
    validate_trajectory_tensor(tensor)

    n_snap = tensor.n_snapshots
    n_leaves = tensor.n_leaves_per_bank
    dt_s = tensor.sampling_interval_ms / 1000.0

    bank_a_exp = np.asarray(tensor.bank_a_expected_mm, dtype=np.float64)
    bank_a_act = np.asarray(tensor.bank_a_actual_mm, dtype=np.float64)
    bank_b_exp = np.asarray(tensor.bank_b_expected_mm, dtype=np.float64)
    bank_b_act = np.asarray(tensor.bank_b_actual_mm, dtype=np.float64)

    gantry_exp = np.asarray(tensor.gantry_expected_deg, dtype=np.float64)
    theta_rad = np.deg2rad(gantry_exp)

    jaw_y1_exp = np.asarray(tensor.jaw_expected_mm["Y1"], dtype=np.float64)
    jaw_y1_act = np.asarray(tensor.jaw_actual_mm["Y1"], dtype=np.float64)
    jaw_y2_exp = np.asarray(tensor.jaw_expected_mm["Y2"], dtype=np.float64)
    jaw_y2_act = np.asarray(tensor.jaw_actual_mm["Y2"], dtype=np.float64)

    mu_exp = np.asarray(tensor.mu_expected, dtype=np.float64)
    mu_act = np.asarray(tensor.mu_actual, dtype=np.float64)
    beam_on = np.asarray(tensor.beam_hold, dtype=np.int32) == 0

    # Identify active leaves (those that open or move during the arc)
    exp_gap = bank_a_exp - bank_b_exp
    active_leaf_mask = (
        (np.max(np.abs(exp_gap), axis=0) > 1.0)
        | (np.ptp(bank_a_exp, axis=0) > 0.5)
        | (np.ptp(bank_b_exp, axis=0) > 0.5)
    )
    active_indices = np.where(active_leaf_mask)[0]
    if active_indices.size == 0:
        active_indices = np.arange(n_leaves)

    err_a = bank_a_act - bank_a_exp
    err_b = bank_b_act - bank_b_exp
    vel_a = np.gradient(bank_a_exp, dt_s, axis=0)
    vel_b = np.gradient(bank_b_exp, dt_s, axis=0)

    # ------------------------------------------------------------------
    # 1 & 2. Per-leaf joint Viscous Drag (-tau * v) + Coulomb Stiction (-beta * tanh(v/v0))
    #        Model: err(t) = c_0 - tau * v(t) - beta * tanh(v(t) / v_0)
    # ------------------------------------------------------------------
    all_leaf_sigs: list[MLCFaultSignature] = []
    flagged_viscous: list[MLCFaultSignature] = []
    flagged_stiction: list[MLCFaultSignature] = []
    flagged_leaves: list[MLCFaultSignature] = []

    best_visc_pred_a = np.zeros_like(err_a)
    best_visc_pred_b = np.zeros_like(err_b)
    best_stic_pred_a = np.zeros_like(err_a)
    best_stic_pred_b = np.zeros_like(err_b)
    max_visc_snr = 0.0
    max_stic_snr = 0.0

    for bank_label, err_mat, vel_mat in (("A", err_a, vel_a), ("B", err_b, vel_b)):
        for l_idx in active_indices:
            e = err_mat[:, l_idx]
            v = vel_mat[:, l_idx]
            v_c = v - np.mean(v)
            e_c = e - np.mean(e)
            v_ss = float(np.sum(v_c**2))
            if v_ss > 1e-6:
                st_basis = np.tanh(v / _STRIBECK_V0_MM_S)
                X = np.column_stack([np.ones(n_snap), -v, -st_basis])
                beta_vec, *_ = np.linalg.lstsq(X, e, rcond=None)
                c0, tau_s, stic_mm = float(beta_vec[0]), float(beta_vec[1]), float(beta_vec[2])
                res = e - X @ beta_vec
                s2 = float(np.sum(res**2) / max(n_snap - 3, 1))
                cov = max(s2, 1e-12) * np.linalg.pinv(X.T @ X)
                se_tau_s = float(np.sqrt(max(cov[1, 1], 1e-12)))
                se_stic_mm = float(np.sqrt(max(cov[2, 2], 1e-12)))
                denom_r = float(np.sqrt(v_ss * max(np.sum(e_c**2), 1e-18)))
                r_corr = float(np.clip(np.sum(-v_c * e_c) / max(denom_r, 1e-12), -1.0, 1.0))
            else:
                c0, tau_s, stic_mm = float(np.mean(e)), 0.0, 0.0
                se_tau_s, se_stic_mm, r_corr = 1e-3, 1e-3, 0.0
                st_basis = np.zeros_like(v)

            tau_ms = tau_s * 1000.0
            se_ms = se_tau_s * 1000.0
            visc_snr = abs(tau_ms) / max(se_ms, 0.25)
            stic_snr = abs(stic_mm) / max(se_stic_mm, 0.01)
            rms_mm = float(np.sqrt(np.mean(e**2)))
            max_mm = float(np.max(np.abs(e)))

            visc_hit = bool(
                (abs(tau_ms) >= 8.0) and (visc_snr >= 4.0) and (abs(r_corr) >= 0.45) and (rms_mm >= 0.08)
            )
            stic_hit = bool((abs(stic_mm) >= 0.15) and (stic_snr >= 5.0) and (rms_mm >= 0.08))
            is_flagged = bool(visc_hit or stic_hit)

            sig = MLCFaultSignature(
                bank=bank_label,  # type: ignore[arg-type]
                leaf_index=int(l_idx + 1),
                rms_error_mm=round(rms_mm, 5),
                max_error_mm=round(max_mm, 5),
                velocity_lag_ms=round(tau_ms, 3),
                velocity_lag_ci95_ms=(
                    round(tau_ms - 1.96 * se_ms, 3),
                    round(tau_ms + 1.96 * se_ms, 3),
                ),
                velocity_correlation_r=round(r_corr, 4),
                stiction_amplitude_mm=round(stic_mm, 4),
                stiction_ci95_mm=(
                    round(stic_mm - 1.96 * se_stic_mm, 4),
                    round(stic_mm + 1.96 * se_stic_mm, 4),
                ),
                systematic_offset_mm=round(c0, 5),
                is_flagged=is_flagged,
            )
            all_leaf_sigs.append(sig)
            if is_flagged:
                flagged_leaves.append(sig)
            if visc_hit:
                flagged_viscous.append(sig)
                max_visc_snr = max(max_visc_snr, visc_snr)
                if bank_label == "A":
                    best_visc_pred_a[:, l_idx] = -tau_s * v
                else:
                    best_visc_pred_b[:, l_idx] = -tau_s * v
            if stic_hit:
                flagged_stiction.append(sig)
                max_stic_snr = max(max_stic_snr, stic_snr)
                if bank_label == "A":
                    best_stic_pred_a[:, l_idx] = -stic_mm * st_basis
                else:
                    best_stic_pred_b[:, l_idx] = -stic_mm * st_basis

    flagged_leaves.sort(
        key=lambda s: max(abs(s.velocity_lag_ms) / 10.0, abs(s.stiction_amplitude_mm) * 5.0),
        reverse=True,
    )
    if not flagged_viscous and all_leaf_sigs:
        top_v = max(all_leaf_sigs, key=lambda s: abs(s.velocity_lag_ms))
        ci_half = max((top_v.velocity_lag_ci95_ms[1] - top_v.velocity_lag_ci95_ms[0]) / 3.92, 0.25)
        max_visc_snr = abs(top_v.velocity_lag_ms) / ci_half
    if not flagged_stiction and all_leaf_sigs:
        top_s = max(all_leaf_sigs, key=lambda s: abs(s.stiction_amplitude_mm))
        ci_half_s = max((top_s.stiction_ci95_mm[1] - top_s.stiction_ci95_mm[0]) / 3.92, 0.01)
        max_stic_snr = abs(top_s.stiction_amplitude_mm) / ci_half_s

    # ------------------------------------------------------------------
    # 3. Systematic Bank / DLG Calibration Offset (Δgap = c across clean active pairs)
    # ------------------------------------------------------------------
    flagged_leaf_0 = {s.leaf_index - 1 for s in flagged_leaves}
    clean_active = np.array(
        [idx for idx in active_indices if idx not in flagged_leaf_0], dtype=np.int32
    )
    if clean_active.size == 0:
        clean_active = active_indices

    act_gap = bank_a_act - bank_b_act
    delta_gap = (act_gap - exp_gap)[:, clean_active]
    dlg_est = float(np.mean(delta_gap))
    dlg_se = float(np.std(delta_gap, ddof=1) / np.sqrt(max(delta_gap.size, 1)))
    dlg_se_eff = max(dlg_se, 0.002)
    dlg_snr = abs(dlg_est) / dlg_se_eff
    dlg_detected = bool((abs(dlg_est) >= 0.08) and (dlg_snr >= 4.0))

    dlg_pred_a = np.zeros_like(err_a)
    dlg_pred_b = np.zeros_like(err_b)
    dlg_pred_a[:, active_indices] = +0.5 * dlg_est
    dlg_pred_b[:, active_indices] = -0.5 * dlg_est

    # ------------------------------------------------------------------
    # 4. Gantry Gravity-Sag Harmonic: Δy(θ) = A sin(θ) + B cos(θ)
    # ------------------------------------------------------------------
    sag_obs = 0.5 * ((jaw_y1_act - jaw_y1_exp) + (jaw_y2_act - jaw_y2_exp))
    X_sag = np.column_stack([np.sin(theta_rad), np.cos(theta_rad)])
    beta_sag, *_ = np.linalg.lstsq(X_sag, sag_obs, rcond=None)
    sag_a_hat, sag_b_hat = float(beta_sag[0]), float(beta_sag[1])
    sag_pred = X_sag @ beta_sag
    sag_res = sag_obs - sag_pred
    sag_s2 = float(np.sum(sag_res**2) / max(n_snap - 2, 1))
    cov_sag = max(sag_s2, 1e-10) * np.linalg.pinv(X_sag.T @ X_sag)
    se_sag_a = float(np.sqrt(max(cov_sag[0, 0], 1e-12)))
    se_sag_b = float(np.sqrt(max(cov_sag[1, 1], 1e-12)))
    sag_amp = float(np.hypot(sag_a_hat, sag_b_hat))
    sag_phase_deg = float(np.degrees(np.arctan2(sag_b_hat, sag_a_hat))) if sag_amp > 1e-9 else 0.0
    se_amp = max(float(np.sqrt(0.5 * (se_sag_a**2 + se_sag_b**2))), 0.005)
    sag_snr = sag_amp / se_amp
    sag_detected = bool((sag_amp >= 0.10) and (sag_snr >= 4.0))

    # ------------------------------------------------------------------
    # 5. Dose-Rate / Ion-Chamber Calibration Drift: ΔMU_act = (1 + δ) ΔMU_exp
    # ------------------------------------------------------------------
    dmu_exp = np.diff(mu_exp)
    dmu_act = np.diff(mu_act)
    valid_mu = (dmu_exp > 1e-5) & beam_on[1:]
    if np.any(valid_mu):
        rel_mu_err = (dmu_act[valid_mu] - dmu_exp[valid_mu]) / dmu_exp[valid_mu]
        weights = dmu_exp[valid_mu]
        mu_delta = float(np.sum(weights * rel_mu_err) / np.sum(weights))
        mu_var = float(np.sum(weights * (rel_mu_err - mu_delta) ** 2) / np.sum(weights))
        mu_se = float(np.sqrt(max(mu_var, 1e-12) / max(int(np.sum(valid_mu)), 1)))
    else:
        mu_delta, mu_se = 0.0, 1e-4

    mu_drift_pct = mu_delta * 100.0
    mu_se_pct = max(mu_se * 100.0, 0.02)
    mu_snr = abs(mu_drift_pct) / mu_se_pct
    mu_detected = bool((abs(mu_drift_pct) >= 0.25) and (mu_snr >= 4.0))

    # ------------------------------------------------------------------
    # 6. Unified Multi-Channel Standardized Residual Vector & BIC Selection
    # ------------------------------------------------------------------
    y_mlc_a = (err_a[:, active_indices] / _SIGMA_MLC_MM).ravel()
    y_mlc_b = (err_b[:, active_indices] / _SIGMA_MLC_MM).ravel()
    y_sag = sag_obs / _SIGMA_SAG_MM
    if np.any(valid_mu):
        y_mu = ((dmu_act[valid_mu] - dmu_exp[valid_mu]) / dmu_exp[valid_mu]) / _SIGMA_MU_REL
    else:
        y_mu = np.zeros(1, dtype=np.float64)

    y_all = np.concatenate([y_mlc_a, y_mlc_b, y_sag, y_mu])
    n_obs = int(y_all.size)

    def _pack_pred(
        pa: np.ndarray | None = None,
        pb: np.ndarray | None = None,
        psag: np.ndarray | None = None,
        pmu: float = 0.0,
    ) -> np.ndarray:
        z_a = (
            (pa[:, active_indices] / _SIGMA_MLC_MM).ravel()
            if pa is not None
            else np.zeros_like(y_mlc_a)
        )
        z_b = (
            (pb[:, active_indices] / _SIGMA_MLC_MM).ravel()
            if pb is not None
            else np.zeros_like(y_mlc_b)
        )
        z_s = (psag / _SIGMA_SAG_MM) if psag is not None else np.zeros_like(y_sag)
        z_m = np.full_like(y_mu, pmu / _SIGMA_MU_REL)
        return np.concatenate([z_a, z_b, z_s, z_m])

    rss_null = float(np.sum(y_all**2))
    aic_null, bic_null = _gaussian_ic(rss_null, n_obs, 0)

    pred_fric = _pack_pred(pa=best_visc_pred_a, pb=best_visc_pred_b)
    rss_fric = float(np.sum((y_all - pred_fric) ** 2))
    k_fric = max(len(flagged_viscous), 1)
    aic_fric, bic_fric = _gaussian_ic(rss_fric, n_obs, k_fric)

    pred_stic = _pack_pred(pa=best_stic_pred_a, pb=best_stic_pred_b)
    rss_stic = float(np.sum((y_all - pred_stic) ** 2))
    k_stic = max(len(flagged_stiction), 1)
    aic_stic, bic_stic = _gaussian_ic(rss_stic, n_obs, k_stic)

    pred_dlg = _pack_pred(pa=dlg_pred_a, pb=dlg_pred_b)
    rss_dlg = float(np.sum((y_all - pred_dlg) ** 2))
    aic_dlg, bic_dlg = _gaussian_ic(rss_dlg, n_obs, 1)

    pred_sag_v = _pack_pred(psag=sag_pred)
    rss_sag = float(np.sum((y_all - pred_sag_v) ** 2))
    aic_sag, bic_sag = _gaussian_ic(rss_sag, n_obs, 2)

    pred_mu_v = _pack_pred(pmu=mu_delta)
    rss_mu = float(np.sum((y_all - pred_mu_v) ** 2))
    aic_mu, bic_mu = _gaussian_ic(rss_mu, n_obs, 1)

    joint_pa = (
        (best_visc_pred_a if flagged_viscous else 0.0)
        + (best_stic_pred_a if flagged_stiction else 0.0)
        + (dlg_pred_a if dlg_detected else 0.0)
    )
    joint_pb = (
        (best_visc_pred_b if flagged_viscous else 0.0)
        + (best_stic_pred_b if flagged_stiction else 0.0)
        + (dlg_pred_b if dlg_detected else 0.0)
    )
    joint_psag = sag_pred if sag_detected else np.zeros_like(sag_obs)
    joint_pmu = mu_delta if mu_detected else 0.0
    k_joint = (
        len(flagged_viscous)
        + len(flagged_stiction)
        + (1 if dlg_detected else 0)
        + (2 if sag_detected else 0)
        + (1 if mu_detected else 0)
    )
    pred_joint = _pack_pred(
        pa=joint_pa if isinstance(joint_pa, np.ndarray) else None,
        pb=joint_pb if isinstance(joint_pb, np.ndarray) else None,
        psag=joint_psag,
        pmu=joint_pmu,
    )
    rss_joint = float(np.sum((y_all - pred_joint) ** 2))
    aic_joint, bic_joint = _gaussian_ic(rss_joint, n_obs, max(k_joint, 1))

    top_visc = flagged_viscous[0] if flagged_viscous else None
    fric_params: dict[str, float | int | str] = (
        {
            "bank": top_visc.bank,
            "leaf_index": top_visc.leaf_index,
            "tau_ms": top_visc.velocity_lag_ms,
            "n_flagged_leaves": len(flagged_viscous),
        }
        if top_visc
        else {"n_flagged_leaves": 0, "tau_ms": 0.0}
    )
    fric_ci = {"tau_ms": top_visc.velocity_lag_ci95_ms} if top_visc else {}

    top_stic = flagged_stiction[0] if flagged_stiction else None
    stic_params: dict[str, float | int | str] = (
        {
            "bank": top_stic.bank,
            "leaf_index": top_stic.leaf_index,
            "stiction_mm": top_stic.stiction_amplitude_mm,
            "n_flagged_leaves": len(flagged_stiction),
        }
        if top_stic
        else {"n_flagged_leaves": 0, "stiction_mm": 0.0}
    )
    stic_ci = {"stiction_mm": top_stic.stiction_ci95_mm} if top_stic else {}

    active_faults: list[str] = []
    if flagged_viscous:
        active_faults.append("MLC_LEAF_MOTOR_FRICTION")
    if flagged_stiction:
        active_faults.append("MLC_COULOMB_STICTION")
    if dlg_detected:
        active_faults.append("SYSTEMATIC_DLG_OFFSET")
    if sag_detected:
        active_faults.append("GANTRY_GRAVITY_SAG_HARMONIC")
    if mu_detected:
        active_faults.append("DOSE_RATE_ION_CHAMBER_DRIFT")

    raw_hypotheses: list[
        tuple[
            str,
            str,
            dict[str, float | int | str],
            dict[str, tuple[float, float]],
            float,
            int,
            float,
            float,
            float,
            bool,
        ]
    ] = [
        (
            "NULL_NOMINAL",
            "Nominal electro-mechanical delivery (zero systematic faults).",
            {},
            {},
            rss_null,
            0,
            aic_null,
            bic_null,
            0.0,
            len(active_faults) == 0,
        ),
        (
            "MLC_LEAF_MOTOR_FRICTION",
            "Localized MLC leaf motor viscous drag / T-nut friction: Δx_i(t) ≈ -τ_i * v_i(t).",
            fric_params,
            fric_ci,
            rss_fric,
            k_fric,
            aic_fric,
            bic_fric,
            max_visc_snr,
            bool(flagged_viscous),
        ),
        (
            "MLC_COULOMB_STICTION",
            "Nonlinear Coulomb/Stribeck stiction & leadscrew backlash: Δx_i(t) ≈ -β_i * tanh(v_i/v_0).",
            stic_params,
            stic_ci,
            rss_stic,
            k_stic,
            aic_stic,
            bic_stic,
            max_stic_snr,
            bool(flagged_stiction),
        ),
        (
            "SYSTEMATIC_DLG_OFFSET",
            "Systematic Bank A/B Dosimetric Leaf Gap (DLG) calibration offset: Δgap = c.",
            {"dlg_offset_mm": round(dlg_est, 4)},
            {
                "dlg_offset_mm": (
                    round(dlg_est - 1.96 * dlg_se_eff, 4),
                    round(dlg_est + 1.96 * dlg_se_eff, 4),
                )
            },
            rss_dlg,
            1,
            aic_dlg,
            bic_dlg,
            dlg_snr,
            dlg_detected,
        ),
        (
            "GANTRY_GRAVITY_SAG_HARMONIC",
            "Periodic gantry head/carriage gravity sag harmonic: Δy(θ) = A sin(θ) + B cos(θ).",
            {
                "sin_coeff_mm": round(sag_a_hat, 4),
                "cos_coeff_mm": round(sag_b_hat, 4),
                "amplitude_mm": round(sag_amp, 4),
                "phase_deg": round(sag_phase_deg, 2),
            },
            {
                "sin_coeff_mm": (
                    round(sag_a_hat - 1.96 * se_sag_a, 4),
                    round(sag_a_hat + 1.96 * se_sag_a, 4),
                ),
                "cos_coeff_mm": (
                    round(sag_b_hat - 1.96 * se_sag_b, 4),
                    round(sag_b_hat + 1.96 * se_sag_b, 4),
                ),
            },
            rss_sag,
            2,
            aic_sag,
            bic_sag,
            sag_snr,
            sag_detected,
        ),
        (
            "DOSE_RATE_ION_CHAMBER_DRIFT",
            "Multiplicative monitor ion-chamber / dose-rate calibration drift: ΔMU / MU = δ.",
            {"mu_drift_pct": round(mu_drift_pct, 4)},
            {
                "mu_drift_pct": (
                    round(mu_drift_pct - 1.96 * mu_se_pct, 4),
                    round(mu_drift_pct + 1.96 * mu_se_pct, 4),
                )
            },
            rss_mu,
            1,
            aic_mu,
            bic_mu,
            mu_snr,
            mu_detected,
        ),
    ]

    if len(active_faults) >= 2:
        raw_hypotheses.append(
            (
                "JOINT_MULTI_FAULT",
                "Composite multi-subsystem fault model combining all statistically active causes.",
                {
                    "n_active_faults": len(active_faults),
                    "active_faults": ",".join(active_faults),
                },
                {},
                rss_joint,
                k_joint,
                aic_joint,
                bic_joint,
                float(np.sqrt(max(0.0, (rss_null - rss_joint) / max(k_joint, 1)))),
                True,
            )
        )

    bics = np.array([h[7] for h in raw_hypotheses], dtype=np.float64)
    min_bic = float(np.min(bics))
    log_w = np.clip(-0.5 * (bics - min_bic), -700.0, 0.0)
    weights = np.exp(log_w)
    weights /= np.sum(weights)

    hypotheses: list[FaultHypothesisResult] = []
    for idx, h in enumerate(raw_hypotheses):
        hypotheses.append(
            FaultHypothesisResult(
                fault_type=h[0],  # type: ignore[arg-type]
                description=h[1],
                parameters=h[2],
                parameter_ci95=h[3],
                rss=round(h[4], 6),
                n_params=h[5],
                aic=round(h[6], 3),
                bic=round(h[7], 3),
                delta_bic_vs_null=round(bic_null - h[7], 3),
                posterior_weight=round(float(weights[idx]), 6),
                matched_filter_snr=round(h[8], 3),
                detected=h[9],
            )
        )

    hypotheses.sort(key=lambda item: item.bic)

    if len(active_faults) == 0:
        primary_cause = "NULL_NOMINAL"
        residual_rms_mm = float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        [err_a[:, active_indices].ravel(), err_b[:, active_indices].ravel(), sag_obs]
                    )
                    ** 2
                )
            )
        )
        var_explained = 100.0 if rss_null < 1e-9 else 0.0
        service_rec = (
            "No systematic electro-mechanical faults detected; continue routine AAPM TG-142/MPC QA."
        )
    else:
        primary_cause = "JOINT_MULTI_FAULT" if len(active_faults) >= 2 else active_faults[0]
        res_mlc_a = err_a[:, active_indices] - (
            joint_pa[:, active_indices] if isinstance(joint_pa, np.ndarray) else 0.0
        )
        res_mlc_b = err_b[:, active_indices] - (
            joint_pb[:, active_indices] if isinstance(joint_pb, np.ndarray) else 0.0
        )
        res_sag_mm = sag_obs - joint_psag
        residual_rms_mm = float(
            np.sqrt(
                np.mean(np.concatenate([res_mlc_a.ravel(), res_mlc_b.ravel(), res_sag_mm]) ** 2)
            )
        )
        var_explained = float(
            np.clip(100.0 * (1.0 - rss_joint / max(rss_null, 1e-12)), 0.0, 100.0)
        )
        actions: list[str] = []
        if flagged_viscous:
            leaf_desc = ", ".join(
                f"Bank {s.bank} Leaf #{s.leaf_index} (τ={s.velocity_lag_ms:.1f} ms)"
                for s in flagged_viscous[:3]
            )
            actions.append(f"Inspect/replace MLC motor & T-nut assembly for {leaf_desc}")
        if flagged_stiction:
            st_desc = ", ".join(
                f"Bank {s.bank} Leaf #{s.leaf_index} (β={s.stiction_amplitude_mm:.2f} mm)"
                for s in flagged_stiction[:3]
            )
            actions.append(
                f"Clean/re-seat MLC leadscrew & check backlash deadband on {st_desc}"
            )
        if dlg_detected:
            actions.append(
                f"Re-initialize MLC banks & recalibrate DLG (observed offset {dlg_est:+.3f} mm)"
            )
        if sag_detected:
            actions.append(
                f"Perform Winston-Lutz / gantry bearing & carriage flex check (sag amp {sag_amp:.2f} mm @ {sag_phase_deg:.0f}°)"
            )
        if mu_detected:
            actions.append(
                f"Verify TG-51 / monitor ion-chamber gain calibration (drift {mu_drift_pct:+.2f}%)"
            )
        service_rec = "; ".join(actions) + "."

    provenance = build_provenance(
        record_id=f"causal:{tensor.log_id}",
        provenance_tag=tensor.provenance.provenance_tag,
        dataset_source=tensor.provenance.dataset_source,
        sha256_inputs={
            **tensor.provenance.sha256_inputs,
            "diagnosis_summary": sha256_json_payload(
                {"primary": primary_cause, "active": active_faults}
            ),
        },
        notes=f"BIC causal attribution across {n_obs} standardized observations (viscous + stiction + DLG + sag + MU).",
    )

    return CausalFaultDiagnosis(
        log_id=tensor.log_id,
        machine_id=tensor.machine_id,
        primary_root_cause=primary_cause,
        active_faults=active_faults,
        hypotheses=hypotheses,
        flagged_leaves=flagged_leaves,
        estimated_dlg_offset_mm=round(dlg_est, 4),
        estimated_dlg_ci95_mm=(
            round(dlg_est - 1.96 * dlg_se_eff, 4),
            round(dlg_est + 1.96 * dlg_se_eff, 4),
        ),
        estimated_gantry_sag_amp_mm=round(sag_amp, 4),
        estimated_gantry_sag_phase_deg=round(sag_phase_deg, 2),
        estimated_mu_drift_pct=round(mu_drift_pct, 4),
        estimated_mu_drift_ci95_pct=(
            round(mu_drift_pct - 1.96 * mu_se_pct, 4),
            round(mu_drift_pct + 1.96 * mu_se_pct, 4),
        ),
        residual_rms_mm=round(residual_rms_mm, 5),
        variance_explained_pct=round(var_explained, 3),
        service_recommendation=service_rec,
        provenance=provenance,
    )
