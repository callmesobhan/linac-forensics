"""Three-Way Closed-Loop EPID Portal Dosimetry Forensics (TPS vs Trajectory Log vs Measured EPID).

Research Software — NOT A MEDICAL DEVICE.

Solves the fundamental blind spot of trajectory-log-only QA: encoder-blind silent
hardware faults (e.g., slipped MLC optical encoder belt, detached leaf-tip screw,
miscalibrated jaw zero-point, or beam-steering tilt) where the trajectory log reports
`actual == expected` while the measured amorphous-silicon (a-Si) EPID portal image
exposes the true radiation delivery deviation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter

from linac_forensics.ir import EPIDThreeWayForensicsReport, TrajectoryLogTensor
from linac_forensics.log_parser import validate_trajectory_tensor
from linac_forensics.perturbative_dose import (
    _evaluate_aperture_fluence_batch,
    compute_gamma_pass_rate_2d,
)
from linac_forensics.provenance import build_provenance, sha256_json_payload


@dataclass(frozen=True)
class EPIDThreeWayArrays:
    """2D grids returned alongside the typed EPIDThreeWayForensicsReport."""

    u_mm: np.ndarray
    v_mm: np.ndarray
    fluence_planned: np.ndarray
    fluence_from_log: np.ndarray
    epid_raw_measured: np.ndarray
    fluence_from_epid: np.ndarray
    delta_epid_minus_log_pct: np.ndarray
    report: EPIDThreeWayForensicsReport


def _reconstruct_planned_and_log_fluence(
    tensor: TrajectoryLogTensor,
    dlg_nominal_mm: float = 1.4,
    leaf_width_mm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute integrated 2D planned and trajectory-log fluence maps on an 81x81 (2 mm) grid."""
    validate_trajectory_tensor(tensor)
    u_mm = np.linspace(-80.0, 80.0, 81, dtype=np.float64)
    v_mm = np.linspace(-80.0, 80.0, 81, dtype=np.float64)

    mu_exp = np.asarray(tensor.mu_expected, dtype=np.float64)
    mu_act = np.asarray(tensor.mu_actual, dtype=np.float64)
    beam_on = np.asarray(tensor.beam_hold, dtype=np.int32) == 0
    dmu_exp = np.maximum(0.0, np.diff(mu_exp, prepend=mu_exp[0])) * beam_on
    dmu_act = np.maximum(0.0, np.diff(mu_act, prepend=mu_act[0])) * beam_on

    jaw_y1_exp = np.asarray(tensor.jaw_expected_mm["Y1"], dtype=np.float64)
    jaw_y2_exp = np.asarray(tensor.jaw_expected_mm["Y2"], dtype=np.float64)
    jaw_y1_act = np.asarray(tensor.jaw_actual_mm["Y1"], dtype=np.float64)
    jaw_y2_act = np.asarray(tensor.jaw_actual_mm["Y2"], dtype=np.float64)
    sag_y_exp = np.zeros_like(jaw_y1_exp)
    sag_y_act = 0.5 * ((jaw_y1_act - jaw_y1_exp) + (jaw_y2_act - jaw_y2_exp))

    f_plan = _evaluate_aperture_fluence_batch(
        np.asarray(tensor.bank_a_expected_mm, dtype=np.float64),
        np.asarray(tensor.bank_b_expected_mm, dtype=np.float64),
        dmu_exp,
        np.asarray(tensor.jaw_expected_mm["X1"], dtype=np.float64),
        np.asarray(tensor.jaw_expected_mm["X2"], dtype=np.float64),
        jaw_y1_exp,
        jaw_y2_exp,
        sag_y_exp,
        u_mm,
        v_mm,
        leaf_width_mm=leaf_width_mm,
        dlg_nominal_mm=dlg_nominal_mm,
    )
    f_log = _evaluate_aperture_fluence_batch(
        np.asarray(tensor.bank_a_actual_mm, dtype=np.float64),
        np.asarray(tensor.bank_b_actual_mm, dtype=np.float64),
        dmu_act,
        np.asarray(tensor.jaw_actual_mm["X1"], dtype=np.float64),
        np.asarray(tensor.jaw_actual_mm["X2"], dtype=np.float64),
        jaw_y1_act,
        jaw_y2_act,
        sag_y_act,
        u_mm,
        v_mm,
        leaf_width_mm=leaf_width_mm,
        dlg_nominal_mm=dlg_nominal_mm,
    )
    return u_mm, v_mm, f_plan, f_log


def simulate_measured_epid_image(
    tensor: TrajectoryLogTensor,
    silent_dlg_bias_mm: float = 0.0,
    silent_leaf_slip_mm: float = 0.0,
    silent_leaf_index: int = 34,
    beam_tilt_x_pct_per_10cm: float = 0.0,
    beam_tilt_y_pct_per_10cm: float = 0.0,
    epid_backscatter_sigma_mm: float = 3.0,
    noise_rel_sigma: float = 0.002,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-simulate an a-Si EPID portal image with optional encoder-blind silent faults.

    Injects physical radiation-field deviations (silent encoder slip, unrecorded DLG offset,
    or steering symmetry tilt) that do NOT appear in `tensor`, then convolves with an
    a-Si optical/backscatter PSF kernel and adds realistic detector noise.
    """
    rng = np.random.default_rng(seed)
    u_mm = np.linspace(-80.0, 80.0, 81, dtype=np.float64)
    v_mm = np.linspace(-80.0, 80.0, 81, dtype=np.float64)

    ba_true = np.asarray(tensor.bank_a_actual_mm, dtype=np.float64).copy()
    bb_true = np.asarray(tensor.bank_b_actual_mm, dtype=np.float64).copy()
    active = (np.max(np.abs(ba_true - bb_true), axis=0) > 1.0)
    if silent_dlg_bias_mm != 0.0:
        ba_true[:, active] += 0.5 * silent_dlg_bias_mm
        bb_true[:, active] -= 0.5 * silent_dlg_bias_mm
    if silent_leaf_slip_mm != 0.0:
        idx0 = silent_leaf_index - 1
        if 0 <= idx0 < bb_true.shape[1]:
            bb_true[:, idx0] -= silent_leaf_slip_mm

    mu_act = np.asarray(tensor.mu_actual, dtype=np.float64)
    beam_on = np.asarray(tensor.beam_hold, dtype=np.int32) == 0
    dmu_act = np.maximum(0.0, np.diff(mu_act, prepend=mu_act[0])) * beam_on
    sag_y = 0.5 * (
        (np.asarray(tensor.jaw_actual_mm["Y1"]) - np.asarray(tensor.jaw_expected_mm["Y1"]))
        + (np.asarray(tensor.jaw_actual_mm["Y2"]) - np.asarray(tensor.jaw_expected_mm["Y2"]))
    )

    f_true = _evaluate_aperture_fluence_batch(
        ba_true,
        bb_true,
        dmu_act,
        np.asarray(tensor.jaw_actual_mm["X1"], dtype=np.float64),
        np.asarray(tensor.jaw_actual_mm["X2"], dtype=np.float64),
        np.asarray(tensor.jaw_actual_mm["Y1"], dtype=np.float64),
        np.asarray(tensor.jaw_actual_mm["Y2"], dtype=np.float64),
        sag_y,
        u_mm,
        v_mm,
    )

    # Apply optional beam steering symmetry tilt (linear gradient across crossplane/inplane)
    vv, uu = np.meshgrid(v_mm, u_mm, indexing="ij")
    tilt_factor = (
        1.0
        + (beam_tilt_x_pct_per_10cm / 100.0) * (uu / 100.0)
        + (beam_tilt_y_pct_per_10cm / 100.0) * (vv / 100.0)
    )
    f_tilted = np.maximum(0.0, f_true * tilt_factor)

    # Convolve with a-Si EPID phosphor/backscatter PSF (85% primary + 15% broad backscatter)
    du = float(u_mm[1] - u_mm[0])
    sig_px = epid_backscatter_sigma_mm / max(du, 1e-6)
    epid_blur = 0.85 * f_tilted + 0.15 * gaussian_filter(f_tilted, sigma=sig_px)
    if noise_rel_sigma > 0.0:
        peak = max(float(np.max(epid_blur)), 1e-6)
        epid_blur = np.maximum(0.0, epid_blur + rng.normal(0.0, noise_rel_sigma * peak, epid_blur.shape))

    return u_mm, v_mm, epid_blur


def deconvolve_epid_to_fluence(
    epid_image: np.ndarray,
    spacing_mm: float = 2.0,
    backscatter_sigma_mm: float = 3.0,
    n_iterations: int = 8,
) -> np.ndarray:
    """Fast Richardson–Lucy / Neumann-series backscatter deconvolution of a-Si EPID image."""
    sig_px = backscatter_sigma_mm / max(spacing_mm, 1e-6)
    est = epid_image.astype(np.float64, copy=True)
    # Exact Neumann fixed-point inversion for (0.85 I + 0.15 G) f = y
    for _ in range(n_iterations):
        forward = 0.85 * est + 0.15 * gaussian_filter(est, sigma=sig_px)
        est = np.maximum(0.0, est + (epid_image - forward))
    return est


def analyze_three_way_epid_forensics(
    tensor: TrajectoryLogTensor,
    measured_epid_image: np.ndarray,
    spacing_mm: float = 2.0,
    backscatter_sigma_mm: float = 3.0,
    return_arrays: bool = False,
) -> EPIDThreeWayForensicsReport | EPIDThreeWayArrays:
    """Perform closed-loop three-way comparison: Planned TPS vs Trajectory Log vs Measured EPID."""
    u_mm, v_mm, f_plan, f_log = _reconstruct_planned_and_log_fluence(tensor)
    f_epid = deconvolve_epid_to_fluence(
        measured_epid_image,
        spacing_mm=spacing_mm,
        backscatter_sigma_mm=backscatter_sigma_mm,
    )

    peak_plan = max(float(np.max(f_plan)), 1e-6)
    active_mask = f_plan >= 0.15 * peak_plan

    g_plan_log = compute_gamma_pass_rate_2d(f_plan, f_log, (spacing_mm, spacing_mm), 2.0, 1.0)
    g_plan_epid = compute_gamma_pass_rate_2d(f_plan, f_epid, (spacing_mm, spacing_mm), 2.0, 1.0)
    g_log_epid = compute_gamma_pass_rate_2d(f_log, f_epid, (spacing_mm, spacing_mm), 2.0, 1.0)

    diff_pct = 100.0 * (f_epid - f_log) / peak_plan
    rms_diff = float(np.sqrt(np.mean(diff_pct[active_mask] ** 2))) if np.any(active_mask) else 0.0
    max_diff = float(np.max(np.abs(diff_pct[active_mask]))) if np.any(active_mask) else 0.0

    # Estimate multiplicative beam steering / panel symmetry tilt across high-signal core field
    vv, uu = np.meshgrid(v_mm, u_mm, indexing="ij")
    core_mask = f_log >= 0.35 * peak_plan
    if np.any(core_mask):
        u_a = uu[core_mask] / 100.0
        v_a = vv[core_mask] / 100.0
        y_a = 100.0 * (f_epid[core_mask] - f_log[core_mask]) / np.maximum(f_log[core_mask], 1e-6)
        X_tilt = np.column_stack([np.ones_like(u_a), u_a, v_a])
        coef, *_ = np.linalg.lstsq(X_tilt, y_a, rcond=None)
        crossplane_tilt = float(coef[1])
        inplane_tilt = float(coef[2])
        tilt_pred = X_tilt @ coef
        tilt_r2 = 1.0 - float(np.sum((y_a - tilt_pred) ** 2)) / max(
            float(np.sum((y_a - np.mean(y_a)) ** 2)), 1e-9
        )
    else:
        crossplane_tilt, inplane_tilt, tilt_r2 = 0.0, 0.0, 0.0

    # Three-way causal classification (TG-142 symmetry tolerance: 2.0%/10cm)
    is_pure_tilt = (abs(crossplane_tilt) >= 2.0 or abs(inplane_tilt) >= 2.0) and (tilt_r2 >= 0.70)
    log_epid_discordant = (
        (g_log_epid < 95.0) or (rms_diff >= 1.2) or (max_diff >= 3.5) or is_pure_tilt
    )
    plan_log_discordant = g_plan_log < 95.0

    classification: Literal[
        "CONCORDANT_NOMINAL",
        "LOG_VERIFIED_MECHANICAL_FAULT",
        "ENCODER_BLIND_SILENT_FAULT",
        "EPID_CALIBRATION_OR_PANEL_DRIFT",
    ]
    if not log_epid_discordant and not plan_log_discordant:
        classification = "CONCORDANT_NOMINAL"
        encoder_blind = False
        summary = "Planned TPS, 50 Hz trajectory log, and deconvolved EPID portal image are concordant."
    elif not log_epid_discordant and plan_log_discordant:
        classification = "LOG_VERIFIED_MECHANICAL_FAULT"
        encoder_blind = False
        summary = (
            f"EPID confirms trajectory-log mechanical fault (Log vs EPID γ={g_log_epid:.1f}%, "
            f"Plan vs Log γ={g_plan_log:.1f}%)."
        )
    elif is_pure_tilt:
        classification = "EPID_CALIBRATION_OR_PANEL_DRIFT"
        encoder_blind = True
        summary = (
            f"Linear symmetry tilt detected across EPID panel (crossplane={crossplane_tilt:+.2f}%/10cm, "
            f"inplane={inplane_tilt:+.2f}%/10cm, R²={tilt_r2:.2f}); check beam steering & EPID flood field."
        )
    else:
        classification = "ENCODER_BLIND_SILENT_FAULT"
        encoder_blind = True
        summary = (
            f"CRITICAL ENCODER-BLIND SILENT FAULT: Trajectory log agrees with plan (γ={g_plan_log:.1f}%) "
            f"while measured EPID diverges (Log vs EPID γ={g_log_epid:.1f}%, max |ΔΦ|={max_diff:.1f}%). "
            "Inspect MLC optical encoder coupling, leaf-tip screws, and physical jaw/DLG zero calibration."
        )

    prov = build_provenance(
        record_id=f"epid3way:{tensor.log_id}",
        provenance_tag="EPID_THREE_WAY_FORENSICS",
        dataset_source=f"Three-Way EPID Forensics ({tensor.provenance.dataset_source})",
        sha256_inputs={
            **tensor.provenance.sha256_inputs,
            "epid_summary": sha256_json_payload(
                {"classification": classification, "g_log_epid": round(g_log_epid, 3)}
            ),
        },
        notes=f"Three-way TPS vs Log vs EPID verification (classification={classification}).",
    )

    report = EPIDThreeWayForensicsReport(
        log_id=tensor.log_id,
        machine_id=tensor.machine_id,
        classification=classification,
        encoder_blind_fault_detected=encoder_blind,
        gamma_plan_vs_log_2pct_1mm=round(g_plan_log, 3),
        gamma_plan_vs_epid_2pct_1mm=round(g_plan_epid, 3),
        gamma_log_vs_epid_2pct_1mm=round(g_log_epid, 3),
        rms_epid_minus_log_pct=round(rms_diff, 4),
        max_epid_minus_log_pct=round(max_diff, 4),
        inplane_symmetry_tilt_pct=round(inplane_tilt, 3),
        crossplane_symmetry_tilt_pct=round(crossplane_tilt, 3),
        inferred_silent_fault_summary=summary,
        provenance=prov,
    )

    if return_arrays:
        return EPIDThreeWayArrays(
            u_mm=u_mm,
            v_mm=v_mm,
            fluence_planned=f_plan,
            fluence_from_log=f_log,
            epid_raw_measured=measured_epid_image,
            fluence_from_epid=f_epid,
            delta_epid_minus_log_pct=diff_pct,
            report=report,
        )
    return report
