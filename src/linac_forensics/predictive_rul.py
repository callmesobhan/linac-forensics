"""Kalman State-Space RUL Forecasting & Multi-Fraction Cumulative 3D Dose Accumulation.

Research Software — NOT A MEDICAL DEVICE.

Provides:
1. Local-Linear-Trend Kalman Filter over multi-day AAPM TG-142 / MPC QA trends
   estimating instantaneous drift velocity, multi-step predictive distributions
   (mean + 95% credible intervals), breach probabilities, and probabilistic
   Remaining Useful Life (RUL) before TG-142 action-limit crossing.
2. Multi-Fraction Cumulative 3D Dose Accumulation across a series of delivered
   fraction trajectory logs, decomposing systematic vs. random dosimetric errors
   and computing compensatory remaining-fraction adaptive MU scaling.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

from linac_forensics.anatomy import PatientAnatomyVolume
from linac_forensics.ir import (
    KalmanRULForecast,
    MultiFractionAccumulationReport,
    TG142TrendSeries,
    TrajectoryLogTensor,
)
from linac_forensics.perturbative_dose import (
    PerturbativeReplayArrays,
    compute_perturbative_dose_replay,
)
from linac_forensics.provenance import build_provenance, sha256_json_payload


def forecast_tg142_kalman_rul(
    trend: TG142TrendSeries,
    metric_name: str = "mlc_max_rms_mm",
    action_limit: float = 0.70,
    observation_cutoff_day: int | None = None,
    horizon_days: int = 12,
    process_noise_level: float = 0.015,
    process_noise_slope: float = 0.008,
    measurement_noise: float = 0.025,
) -> KalmanRULForecast:
    """Run a Local-Linear-Trend Kalman Filter to forecast TG-142 drift and Remaining Useful Life.

    State vector: `s_t = [x_t, v_t]^T` (level & drift velocity per day)
    Transition:   `s_{t+1} = [[1, 1], [0, 1]] s_t + w_t`, `w_t ~ N(0, Q)`
    Observation:  `y_t = [1, 0] s_t + e_t`, `e_t ~ N(0, R)`
    """
    series_map = {
        "mlc_max_rms_mm": trend.mlc_max_rms_mm,
        "dlg_offset_mm": [abs(v) for v in trend.dlg_offset_mm],
        "gantry_sag_amp_mm": trend.gantry_sag_amp_mm,
        "output_drift_pct": [abs(v) for v in trend.output_drift_pct],
    }
    if metric_name not in series_map:
        raise ValueError(f"Unsupported TG-142 metric '{metric_name}'; choose from {sorted(series_map)}")

    values = np.asarray(series_map[metric_name], dtype=np.float64)
    days = np.asarray(trend.days, dtype=np.int32)
    cutoff = observation_cutoff_day if observation_cutoff_day is not None else int(days[-1])
    mask = days <= cutoff
    obs_y = values[mask]
    obs_d = days[mask]
    if obs_y.size < 3:
        raise ValueError("Kalman RUL forecast requires at least 3 daily observations.")

    F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    Q = np.diag([process_noise_level**2, process_noise_slope**2])
    R = np.array([[measurement_noise**2]], dtype=np.float64)

    # Initialize state from first two observations
    x = np.array([obs_y[0], obs_y[1] - obs_y[0]], dtype=np.float64)
    P = np.diag([measurement_noise**2, 0.05**2])

    for idx in range(1, obs_y.size):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        # Update
        innov = obs_y[idx] - float((H @ x_pred)[0])
        S = float((H @ P_pred @ H.T + R)[0, 0])
        K = (P_pred @ H.T) / max(S, 1e-12)
        x = x_pred + K.ravel() * innov
        P = (np.eye(2) - K @ H) @ P_pred

    # Multi-step ahead forecast over `horizon_days`
    f_days: list[int] = []
    f_mean: list[float] = []
    f_lo: list[float] = []
    f_hi: list[float] = []
    breach_probs: list[float] = []

    x_h = x.copy()
    P_h = P.copy()
    last_day = int(obs_d[-1])

    for step in range(1, horizon_days + 1):
        x_h = F @ x_h
        P_h = F @ P_h @ F.T + Q
        mu = float(x_h[0])
        var_y = float(P_h[0, 0] + measurement_noise**2)
        sd = float(np.sqrt(max(var_y, 1e-10)))
        f_days.append(last_day + step)
        f_mean.append(round(mu, 4))
        f_lo.append(round(mu - 1.96 * sd, 4))
        f_hi.append(round(mu + 1.96 * sd, 4))
        # Probability of exceeding action limit at step h
        p_breach = float(1.0 - norm.cdf((action_limit - mu) / sd))
        breach_probs.append(p_breach)

    curr_level = float(x[0])
    drift_rate = float(x[1])
    slope_sd = float(np.sqrt(max(P[1, 1], 1e-12)))

    rul_mean: float | None = None
    rul_ci95: tuple[float, float] | None = None
    if curr_level >= action_limit:
        rul_mean = 0.0
        rul_ci95 = (0.0, 0.0)
    elif drift_rate > 1e-4:
        remaining = action_limit - curr_level
        rul_mean = round(remaining / drift_rate, 2)
        # Monte Carlo propagation of correlated (level, slope) posterior for RUL 95% CI
        rng = np.random.default_rng(42)
        draws = rng.multivariate_normal(x, P, size=4000)
        pos_slopes = draws[:, 1] > 1e-4
        if np.any(pos_slopes):
            ruls = np.clip((action_limit - draws[pos_slopes, 0]) / draws[pos_slopes, 1], 0.0, 180.0)
            lo_r, hi_r = np.percentile(ruls, [2.5, 97.5])
            rul_ci95 = (round(float(lo_r), 2), round(float(hi_r), 2))
        else:
            rul_ci95 = (round(max(0.0, remaining / (drift_rate + 1.96 * slope_sd)), 2), 90.0)

    max_p_breach = float(max(breach_probs)) if breach_probs else 0.0
    if rul_mean is not None and rul_mean <= 7.0:
        rec = (
            f"PROACTIVE SERVICE ALERT: Predicted breach of {metric_name} ({action_limit}) in "
            f"{rul_mean:.1f} days (95% CI {rul_ci95}; {horizon_days}-day breach prob={max_p_breach * 100:.0f}%). "
            "Schedule preventive MLC/DLG service prior to clinical lockout."
        )
    elif max_p_breach >= 0.30:
        rec = (
            f"ELEVATED DRIFT WATCH: {horizon_days}-day breach probability {max_p_breach * 100:.0f}%; "
            "increase daily MPC & trajectory-log audit cadence."
        )
    else:
        rec = (
            f"STABLE TREND: Estimated drift {drift_rate:+.4f}/day; "
            f"{horizon_days}-day breach probability {max_p_breach * 100:.1f}%. Continue routine TG-142 QA."
        )

    prov = build_provenance(
        record_id=f"kalman_rul:{trend.machine_id}:{metric_name}:d{last_day}",
        provenance_tag=trend.provenance.provenance_tag,
        dataset_source=trend.provenance.dataset_source,
        sha256_inputs={
            **trend.provenance.sha256_inputs,
            "rul_config": sha256_json_payload(
                {
                    "metric": metric_name,
                    "cutoff_day": last_day,
                    "action_limit": action_limit,
                    "horizon": horizon_days,
                }
            ),
        },
        notes=f"Local-linear-trend Kalman RUL forecast from Day {last_day} (+{horizon_days}d).",
    )

    return KalmanRULForecast(
        machine_id=trend.machine_id,
        metric_name=metric_name,
        action_limit=action_limit,
        observation_cutoff_day=last_day,
        horizon_days=horizon_days,
        forecast_days=f_days,
        forecast_mean=f_mean,
        forecast_ci95_low=f_lo,
        forecast_ci95_high=f_hi,
        estimated_drift_rate_per_day=round(drift_rate, 5),
        rul_mean_days=rul_mean,
        rul_ci95_days=rul_ci95,
        breach_probability_within_horizon=round(max_p_breach, 4),
        maintenance_recommendation=rec,
        provenance=prov,
    )


def accumulate_multifraction_course(
    fraction_logs: Sequence[TrajectoryLogTensor],
    anatomy: PatientAnatomyVolume,
    total_prescribed_fractions: int = 35,
    n_sectors: int = 8,
) -> MultiFractionAccumulationReport:
    """Accumulate 3D perturbative dose across multiple delivered fractions.

    Separates systematic vs. random delivery deviations and computes a compensatory
    remaining-fraction MU scaling factor to restore target prescription if needed.
    """
    if len(fraction_logs) == 0:
        raise ValueError("At least one delivered fraction log is required for accumulation.")

    n_fx = len(fraction_logs)
    total_fx = max(total_prescribed_fractions, n_fx)
    planned_3d = anatomy.planned_dose_gy.astype(np.float64, copy=False)
    cum_delta_3d = np.zeros_like(planned_3d, dtype=np.float64)
    per_fx_ptv_shifts: list[float] = []

    for log in fraction_logs:
        rep = compute_perturbative_dose_replay(log, anatomy, n_sectors=n_sectors, return_arrays=True)
        assert isinstance(rep, PerturbativeReplayArrays)
        cum_delta_3d += rep.delta_dose_3d / float(n_fx)
        ptv_shift = next(
            (s for s in rep.impact.structure_shifts if s.roi_name == "PTV70"),
            rep.impact.structure_shifts[0],
        )
        per_fx_ptv_shifts.append(ptv_shift.delta_d95_pct)

    cum_deliv_3d = np.where(
        anatomy.possible_dose_mask,
        np.maximum(0.0, planned_3d + cum_delta_3d),
        0.0,
    )

    ptv_name = "PTV70" if "PTV70" in anatomy.roi_masks else next(iter(anatomy.roi_masks))
    ptv_mask = anatomy.roi_masks[ptv_name]
    d95_plan = float(np.percentile(planned_3d[ptv_mask], 5.0))
    d95_cum = float(np.percentile(cum_deliv_3d[ptv_mask], 5.0))
    cum_ptv_delta_pct = 100.0 * (d95_cum - d95_plan) / max(d95_plan, 1.0)

    # Systematic (mean) vs random (std) fraction-to-fraction decomposition
    sys_pct = float(np.mean(per_fx_ptv_shifts))
    rand_rms_pct = float(np.std(per_fx_ptv_shifts, ddof=1)) if n_fx >= 2 else 0.0

    # Worst cumulative OAR D0.03cc shift
    voxel_vol_cc = anatomy.voxel_volume_cc
    best_oar_name = "SpinalCord"
    worst_oar_delta = -1e9
    for roi_name, mask in anatomy.roi_masks.items():
        if roi_name.startswith("PTV") or not np.any(mask):
            continue
        k = max(1, int(round(0.03 / max(voxel_vol_cc, 1e-6))))
        dp_s = np.sort(planned_3d[mask])[::-1]
        dd_s = np.sort(cum_deliv_3d[mask])[::-1]
        idx = min(k - 1, dp_s.size - 1)
        delta_gy = float(dd_s[idx] - dp_s[idx])
        if delta_gy > worst_oar_delta:
            worst_oar_delta = delta_gy
            best_oar_name = roi_name
    if worst_oar_delta < -1e8:
        worst_oar_delta = 0.0

    # Compensatory MU scaling for remaining fractions:
    # n_fx * (1 + cum_ptv_delta_pct/100) + n_rem * s_rem = total_fx
    n_rem = max(total_fx - n_fx, 1)
    course_deficit_fx = -(cum_ptv_delta_pct / 100.0) * float(n_fx)
    mu_scale_rem = float(np.clip(1.0 + course_deficit_fx / float(n_rem), 0.92, 1.08))
    adaptive_triggered = bool(abs(cum_ptv_delta_pct) >= 2.0 or worst_oar_delta >= 1.5)

    first_log = fraction_logs[0]
    prov = build_provenance(
        record_id=f"multifx:{first_log.patient_id}:{n_fx}fx",
        provenance_tag="COMPOSITE_OPENKBP_PLUS_SYNTHETIC_LOG",
        dataset_source=f"{anatomy.anatomy_source} + {n_fx} delivered fraction logs",
        sha256_inputs={
            **anatomy.sha256_inputs,
            "fraction_ids": sha256_json_payload([lg.log_id for lg in fraction_logs]),
        },
        notes=f"Cumulative 3D dose accumulation over {n_fx}/{total_fx} fractions.",
    )

    return MultiFractionAccumulationReport(
        patient_id=first_log.patient_id,
        plan_id=first_log.plan_id,
        anatomy_source=anatomy.anatomy_source,
        fractions_delivered=n_fx,
        total_prescribed_fractions=total_fx,
        cumulative_ptv_delta_d95_pct=round(cum_ptv_delta_pct, 4),
        systematic_component_pct=round(sys_pct, 4),
        random_component_rms_pct=round(rand_rms_pct, 4),
        cumulative_critical_oar_delta_d003cc_gy=round(worst_oar_delta, 4),
        critical_oar_name=best_oar_name,
        recommended_remaining_fx_mu_scale=round(mu_scale_rem, 5),
        adaptive_replan_triggered=adaptive_triggered,
        provenance=prov,
    )
