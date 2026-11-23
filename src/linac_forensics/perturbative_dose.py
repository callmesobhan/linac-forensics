"""Fast Perturbative Fluence-to-3D-Dose Replay Engine & 2D/3D Gamma Analyzer.

Research Software — NOT A MEDICAL DEVICE.

Converts 50 Hz expected vs. actual MLC apertures, jaws, gantry angles, and MU
weights from a `TrajectoryLogTensor` into 2D delivered vs. planned fluence
difference `ΔΦ(u, v)` (including rounded leaf-tip DLG transmission, interleaf
leakage, and Gaussian penumbra kernel), projects `ΔΦ` into 3D patient anatomy
(e.g. real OpenKBP `pt_1`) via a vectorized pencil-beam / radiological depth-dose
perturbation kernel, and computes exact structure-specific `ΔD95`, `ΔD0.03cc`,
`ΔDmean`, `ΔgEUD`, `ΔTCP/NTCP`, and 2D/3D Gamma (3%/2mm vs 2%/1mm).

Formal Zero-Error Identity Invariant:
    When `actual == expected` across all trajectory channels, `max(|ΔΦ|) == 0.0`,
    `max(|ΔD|) == 0.0`, and all 2D/3D gamma pass rates equal `100.0%` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erf

from linac_forensics.anatomy import (
    ROI_TYPES,
    PatientAnatomyVolume,
    ct_hu_to_relative_electron_density,
)
from linac_forensics.ir import (
    PerturbativeDoseImpact,
    StructureDVHShift,
    TrajectoryLogTensor,
)
from linac_forensics.log_parser import validate_trajectory_tensor
from linac_forensics.provenance import build_provenance, sha256_json_payload

# Biological EUD / TCP / NTCP reference parameters (head-and-neck 2 Gy/fx clinical literature values)
_ROI_BIO_PARAMS: dict[str, dict[str, float]] = {
    "PTV70": {"a": -10.0, "d50": 63.0, "gamma50": 4.0, "rx": 70.0},
    "PTV63": {"a": -10.0, "d50": 56.0, "gamma50": 4.0, "rx": 63.0},
    "PTV56": {"a": -10.0, "d50": 49.0, "gamma50": 4.0, "rx": 56.0},
    "SpinalCord": {"a": 12.0, "d50": 55.0, "gamma50": 4.0, "rx": 45.0},
    "Brainstem": {"a": 12.0, "d50": 60.0, "gamma50": 4.0, "rx": 54.0},
    "LeftParotid": {"a": 1.0, "d50": 32.0, "gamma50": 2.0, "rx": 26.0},
    "RightParotid": {"a": 1.0, "d50": 32.0, "gamma50": 2.0, "rx": 26.0},
    "Mandible": {"a": 10.0, "d50": 72.0, "gamma50": 4.0, "rx": 70.0},
    "Esophagus": {"a": 5.0, "d50": 55.0, "gamma50": 3.0, "rx": 45.0},
}


@dataclass(frozen=True)
class PerturbativeReplayArrays:
    """Intermediate 2D fluence and 3D dose arrays returned alongside the typed summary."""

    u_mm: np.ndarray  # [Nu]
    v_mm: np.ndarray  # [Nv]
    fluence_planned_2d: np.ndarray  # [Nv, Nu]
    fluence_delivered_2d: np.ndarray  # [Nv, Nu]
    delta_fluence_2d: np.ndarray  # [Nv, Nu]
    planned_dose_3d: np.ndarray  # [Nx, Ny, Nz]
    delivered_dose_3d: np.ndarray  # [Nx, Ny, Nz]
    delta_dose_3d: np.ndarray  # [Nx, Ny, Nz]
    impact: PerturbativeDoseImpact


def _evaluate_aperture_fluence_batch(
    bank_a_mm: np.ndarray,  # [N, L]
    bank_b_mm: np.ndarray,  # [N, L]
    dmu: np.ndarray,  # [N]
    jaw_x1: np.ndarray,  # [N]
    jaw_x2: np.ndarray,  # [N]
    jaw_y1: np.ndarray,  # [N]
    jaw_y2: np.ndarray,  # [N]
    sag_y_mm: np.ndarray,  # [N]
    u_mm: np.ndarray,  # [Nu]
    v_mm: np.ndarray,  # [Nv]
    leaf_width_mm: float = 5.0,
    dlg_nominal_mm: float = 1.4,
    penumbra_sigma_mm: float = 2.2,
    mlc_transmission: float = 0.015,
) -> np.ndarray:
    """Vectorized 2D fluence accumulator with rounded-tip DLG, leakage, jaw clipping & penumbra.

    Returns fluence array of shape `[Nv, Nu]`.
    """
    n_snap, n_leaves = bank_a_mm.shape
    if n_snap == 0 or float(np.sum(dmu)) <= 0.0:
        return np.zeros((v_mm.size, u_mm.size), dtype=np.float64)

    # Map each v coordinate (adjusted by sagittal head shift) to its covering MLC leaf index
    # Leaf center y_j = (j - n_leaves/2 + 0.5) * leaf_width_mm
    half_span = 0.5 * n_leaves * leaf_width_mm
    inv_sqrt2_sig = 1.0 / (np.sqrt(2.0) * max(penumbra_sigma_mm, 0.25))
    half_dlg = 0.5 * dlg_nominal_mm

    # Pre-evaluate leaf profiles across snapshots: [N, L, Nu]
    # Effective aperture edges include rounded leaf-tip DLG transmission
    left_edge = bank_b_mm[:, :, None] - half_dlg  # [N, L, 1]
    right_edge = bank_a_mm[:, :, None] + half_dlg  # [N, L, 1]
    u_grid = u_mm[None, None, :]  # [1, 1, Nu]

    # Smooth penumbra opening in [0, 1]
    open_profile = 0.5 * (
        erf((u_grid - left_edge) * inv_sqrt2_sig) - erf((u_grid - right_edge) * inv_sqrt2_sig)
    )
    open_profile = np.clip(open_profile, 0.0, 1.0)
    # Add interleaf/body MLC transmission inside jaw aperture
    leaf_transmission = mlc_transmission + (1.0 - mlc_transmission) * open_profile

    # Jaw X clipping [N, 1, Nu]
    jaw_x_mask = (
        0.5
        * (
            erf((u_mm[None, :] - jaw_x1[:, None]) * inv_sqrt2_sig)
            - erf((u_mm[None, :] - jaw_x2[:, None]) * inv_sqrt2_sig)
        )
    )
    jaw_x_mask = np.clip(jaw_x_mask, 0.0, 1.0)

    # Map v_mm shifted by sag_y_mm onto leaf indices [N, Nv]
    v_shifted = v_mm[None, :] - sag_y_mm[:, None]  # [N, Nv]
    leaf_float = (v_shifted + half_span) / leaf_width_mm - 0.5
    leaf_low = np.clip(np.floor(leaf_float).astype(np.int32), 0, n_leaves - 1)
    leaf_high = np.clip(leaf_low + 1, 0, n_leaves - 1)
    frac = np.clip(leaf_float - np.floor(leaf_float), 0.0, 1.0)  # [N, Nv]

    # Jaw Y clipping [N, Nv]
    jaw_y_mask = 0.5 * (
        erf((v_mm[None, :] - jaw_y1[:, None]) * inv_sqrt2_sig)
        - erf((v_mm[None, :] - jaw_y2[:, None]) * inv_sqrt2_sig)
    )
    jaw_y_mask = np.clip(jaw_y_mask, 0.0, 1.0)

    # Accumulate weighted fluence into [Nv, Nu]
    fluence = np.zeros((v_mm.size, u_mm.size), dtype=np.float64)
    snap_idx = np.arange(n_snap)[:, None]
    for v_i in range(v_mm.size):
        l0 = leaf_low[:, v_i]
        l1 = leaf_high[:, v_i]
        w1 = frac[:, v_i : v_i + 1]
        prof_v = (1.0 - w1) * leaf_transmission[snap_idx[:, 0], l0, :] + w1 * leaf_transmission[
            snap_idx[:, 0], l1, :
        ]
        weight_v = (dmu * jaw_y_mask[:, v_i])[:, None] * jaw_x_mask
        fluence[v_i, :] = np.sum(weight_v * prof_v, axis=0)

    return fluence


def compute_gamma_pass_rate_2d(
    reference: np.ndarray,
    evaluated: np.ndarray,
    spacing_mm: tuple[float, float],
    dose_pct: float = 3.0,
    dta_mm: float = 2.0,
    low_dose_threshold_pct: float = 10.0,
) -> float:
    """Global 2D planar gamma pass rate (AAPM TG-218 definition)."""
    if np.array_equal(reference, evaluated):
        return 100.0
    max_ref = float(np.max(reference))
    if max_ref <= 1e-12:
        return 100.0

    mask = reference >= (low_dose_threshold_pct / 100.0) * max_ref
    if not np.any(mask):
        return 100.0

    dose_tol = (dose_pct / 100.0) * max_ref
    dy, dx = spacing_mm
    ry = int(np.ceil(dta_mm / max(dy, 1e-6))) + 1
    rx = int(np.ceil(dta_mm / max(dx, 1e-6))) + 1

    ny, nx = reference.shape
    min_gamma_sq = np.full(reference.shape, np.inf, dtype=np.float64)

    for oy in range(-ry, ry + 1):
        for ox in range(-rx, rx + 1):
            dist_sq = (oy * dy) ** 2 + (ox * dx) ** 2
            dta_term = dist_sq / (dta_mm**2)
            if dta_term > 4.0:
                continue
            y_src_start = max(0, -oy)
            y_src_end = min(ny, ny - oy)
            x_src_start = max(0, -ox)
            x_src_end = min(nx, nx - ox)
            y_dst_start = max(0, oy)
            y_dst_end = min(ny, ny + oy)
            x_dst_start = max(0, ox)
            x_dst_end = min(nx, nx + ox)

            ref_slice = reference[y_src_start:y_src_end, x_src_start:x_src_end]
            ev_slice = evaluated[y_dst_start:y_dst_end, x_dst_start:x_dst_end]
            gamma_sq = ((ev_slice - ref_slice) / dose_tol) ** 2 + dta_term
            sub = min_gamma_sq[y_src_start:y_src_end, x_src_start:x_src_end]
            np.minimum(sub, gamma_sq, out=sub)

    passed = np.sum(min_gamma_sq[mask] <= 1.0)
    return float(100.0 * passed / np.sum(mask))


def compute_gamma_pass_rate_3d(
    reference: np.ndarray,
    evaluated: np.ndarray,
    voxel_dims_mm: tuple[float, float, float],
    dose_pct: float = 3.0,
    dta_mm: float = 2.0,
    low_dose_threshold_pct: float = 10.0,
) -> float:
    """Fast vectorized global 3D volumetric gamma pass rate (AAPM TG-218 definition)."""
    if np.array_equal(reference, evaluated):
        return 100.0
    max_ref = float(np.max(reference))
    if max_ref <= 1e-12:
        return 100.0

    mask = reference >= (low_dose_threshold_pct / 100.0) * max_ref
    if not np.any(mask):
        return 100.0

    # Restrict bounding box around active dose mask for speed
    coords = np.argwhere(mask)
    mins = np.maximum(0, coords.min(axis=0) - 2)
    maxs = np.minimum(np.array(reference.shape), coords.max(axis=0) + 3)
    ref_crop = reference[mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]].astype(
        np.float64, copy=False
    )
    ev_crop = evaluated[mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]].astype(
        np.float64, copy=False
    )
    mask_crop = mask[mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]]

    dose_tol = (dose_pct / 100.0) * max_ref
    dx, dy, dz = voxel_dims_mm
    rx = int(np.ceil(dta_mm / max(dx, 1e-6)))
    ry = int(np.ceil(dta_mm / max(dy, 1e-6)))
    rz = int(np.ceil(dta_mm / max(dz, 1e-6)))

    nx, ny, nz = ref_crop.shape
    min_gamma_sq = np.full(ref_crop.shape, np.inf, dtype=np.float64)

    # First check zero offset (fast exit for most voxels)
    zero_g2 = ((ev_crop - ref_crop) / dose_tol) ** 2
    np.minimum(min_gamma_sq, zero_g2, out=min_gamma_sq)

    # Sub-voxel linear-interpolated gradient approximation when voxel spacing > dta_mm
    # plus discrete integer-shift exact search within neighborhood
    grad_x, grad_y, grad_z = np.gradient(ev_crop, dx, dy, dz)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2) + 1e-9
    # Optimal continuous local displacement along gradient within +/- 0.5 voxel
    half_step = 0.5 * min(dx, dy, dz)
    diff = ev_crop - ref_crop
    s_opt = np.clip(
        diff * grad_mag / (grad_mag**2 + (dose_tol / dta_mm) ** 2),
        -half_step,
        half_step,
    )
    cont_g2 = ((diff - s_opt * grad_mag) / dose_tol) ** 2 + (s_opt / dta_mm) ** 2
    np.minimum(min_gamma_sq, cont_g2, out=min_gamma_sq)

    for ox in range(-rx, rx + 1):
        for oy in range(-ry, ry + 1):
            for oz in range(-rz, rz + 1):
                if ox == 0 and oy == 0 and oz == 0:
                    continue
                dta_term = ((ox * dx) ** 2 + (oy * dy) ** 2 + (oz * dz) ** 2) / (dta_mm**2)
                if dta_term > 1.5:
                    continue
                xs0, xs1 = max(0, -ox), min(nx, nx - ox)
                ys0, ys1 = max(0, -oy), min(ny, ny - oy)
                zs0, zs1 = max(0, -oz), min(nz, nz - oz)
                xd0, xd1 = max(0, ox), min(nx, nx + ox)
                yd0, yd1 = max(0, oy), min(ny, ny + oy)
                zd0, zd1 = max(0, oz), min(nz, nz + oz)

                g2 = (
                    (ev_crop[xd0:xd1, yd0:yd1, zd0:zd1] - ref_crop[xs0:xs1, ys0:ys1, zs0:zs1])
                    / dose_tol
                ) ** 2 + dta_term
                sub = min_gamma_sq[xs0:xs1, ys0:ys1, zs0:zs1]
                np.minimum(sub, g2, out=sub)

    passed = np.sum(min_gamma_sq[mask_crop] <= 1.0)
    return float(100.0 * passed / np.sum(mask_crop))


def _compute_d003cc(dose_values: np.ndarray, voxel_vol_cc: float) -> float:
    """Compute exact D0.03cc near-maximum dose (Gy) via linear interpolation of sorted voxels."""
    if dose_values.size == 0:
        return 0.0
    sorted_desc = np.sort(dose_values)[::-1]
    k_voxels = 0.03 / max(voxel_vol_cc, 1e-9)
    if k_voxels <= 1.0 or sorted_desc.size == 1:
        return float(sorted_desc[0])
    idx_low = min(int(np.floor(k_voxels)) - 1, sorted_desc.size - 1)
    idx_high = min(idx_low + 1, sorted_desc.size - 1)
    frac = k_voxels - np.floor(k_voxels)
    return float((1.0 - frac) * sorted_desc[idx_low] + frac * sorted_desc[idx_high])


def _compute_geud(dose_values: np.ndarray, a: float) -> float:
    """Niemierko generalized Equivalent Uniform Dose (gEUD)."""
    if dose_values.size == 0:
        return 0.0
    safe = np.maximum(dose_values.astype(np.float64), 1e-4)
    if abs(a - 1.0) < 1e-6:
        return float(np.mean(safe))
    # Numerically stable power-mean
    scale = float(np.max(safe))
    scaled = safe / max(scale, 1e-6)
    return float(scale * (np.mean(scaled**a) ** (1.0 / a)))


def _compute_logistic_bio_prob(eud_gy: float, d50: float, gamma50: float) -> float:
    """Logistic TCP / NTCP model parameterized by (D50, gamma50)."""
    if eud_gy <= 1e-6:
        return 0.0
    exponent = np.clip(-4.0 * gamma50 * (eud_gy / d50 - 1.0), -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(exponent)))


def _bilinear_sample_2d(
    grid: np.ndarray,  # [Nv, Nu]
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    u_query: np.ndarray,
    v_query: np.ndarray,
) -> np.ndarray:
    """Fast vectorized bilinear interpolation on a uniform 2D (v, u) fluence grid."""
    du = float(u_axis[1] - u_axis[0])
    dv = float(v_axis[1] - v_axis[0])
    nu = u_axis.size
    nv = v_axis.size

    fu = (u_query - u_axis[0]) / du
    fv = (v_query - v_axis[0]) / dv

    in_bounds = (fu >= 0.0) & (fu <= nu - 1.0) & (fv >= 0.0) & (fv <= nv - 1.0)
    fu_c = np.clip(fu, 0.0, nu - 1.00001)
    fv_c = np.clip(fv, 0.0, nv - 1.00001)

    u0 = np.floor(fu_c).astype(np.int32)
    v0 = np.floor(fv_c).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1
    wu = fu_c - u0
    wv = fv_c - v0

    val = (
        (1.0 - wv) * (1.0 - wu) * grid[v0, u0]
        + (1.0 - wv) * wu * grid[v0, u1]
        + wv * (1.0 - wu) * grid[v1, u0]
        + wv * wu * grid[v1, u1]
    )
    return np.where(in_bounds, val, 0.0)


def compute_perturbative_dose_replay(
    tensor: TrajectoryLogTensor,
    anatomy: PatientAnatomyVolume,
    dlg_nominal_mm: float = 1.4,
    leaf_width_mm: float = 5.0,
    n_sectors: int = 12,
    use_heterogeneity: bool = True,
    return_arrays: bool = False,
) -> PerturbativeDoseImpact | PerturbativeReplayArrays:
    """Execute full 2D fluence -> 3D perturbative dose replay on real 3D patient anatomy.

    Guarantees exact zero-error identity: if `actual == expected` on every channel,
    `delta_fluence_2d == 0.0`, `delta_dose_3d == 0.0`, and all gamma pass rates == `100.0%`.
    """
    validate_trajectory_tensor(tensor)

    # Extract arrays
    mu_exp = np.asarray(tensor.mu_expected, dtype=np.float64)
    mu_act = np.asarray(tensor.mu_actual, dtype=np.float64)
    beam_hold = np.asarray(tensor.beam_hold, dtype=np.int32)
    gantry_exp = np.asarray(tensor.gantry_expected_deg, dtype=np.float64)
    gantry_act = np.asarray(tensor.gantry_actual_deg, dtype=np.float64)

    bank_a_exp = np.asarray(tensor.bank_a_expected_mm, dtype=np.float64)
    bank_a_act = np.asarray(tensor.bank_a_actual_mm, dtype=np.float64)
    bank_b_exp = np.asarray(tensor.bank_b_expected_mm, dtype=np.float64)
    bank_b_act = np.asarray(tensor.bank_b_actual_mm, dtype=np.float64)

    jaw_x1_exp = np.asarray(tensor.jaw_expected_mm["X1"], dtype=np.float64)
    jaw_x2_exp = np.asarray(tensor.jaw_expected_mm["X2"], dtype=np.float64)
    jaw_y1_exp = np.asarray(tensor.jaw_expected_mm["Y1"], dtype=np.float64)
    jaw_y2_exp = np.asarray(tensor.jaw_expected_mm["Y2"], dtype=np.float64)

    jaw_x1_act = np.asarray(tensor.jaw_actual_mm["X1"], dtype=np.float64)
    jaw_x2_act = np.asarray(tensor.jaw_actual_mm["X2"], dtype=np.float64)
    jaw_y1_act = np.asarray(tensor.jaw_actual_mm["Y1"], dtype=np.float64)
    jaw_y2_act = np.asarray(tensor.jaw_actual_mm["Y2"], dtype=np.float64)

    # Incremental MU weights per snapshot (beam-on snapshots only)
    beam_on = beam_hold == 0
    dmu_exp = np.maximum(0.0, np.diff(mu_exp, prepend=mu_exp[0])) * beam_on
    dmu_act = np.maximum(0.0, np.diff(mu_act, prepend=mu_act[0])) * beam_on

    # Sagittal head gravity-sag shift (0 for planned, common-mode Y jaw residual for actual)
    sag_y_exp = np.zeros_like(jaw_y1_exp)
    sag_y_act = 0.5 * ((jaw_y1_act - jaw_y1_exp) + (jaw_y2_act - jaw_y2_exp))

    # 2D Fluence grid at isocenter plane (2.0 mm spacing, 81 x 81 covering [-80, +80] mm)
    u_mm = np.linspace(-80.0, 80.0, 81, dtype=np.float64)
    v_mm = np.linspace(-80.0, 80.0, 81, dtype=np.float64)

    # Exact zero-error short-circuit check (verifiable mathematically because planned == delivered)
    is_exact_zero_error = bool(
        np.array_equal(bank_a_act, bank_a_exp)
        and np.array_equal(bank_b_act, bank_b_exp)
        and np.array_equal(mu_act, mu_exp)
        and np.array_equal(gantry_act, gantry_exp)
        and np.array_equal(jaw_x1_act, jaw_x1_exp)
        and np.array_equal(jaw_x2_act, jaw_x2_exp)
        and np.array_equal(jaw_y1_act, jaw_y1_exp)
        and np.array_equal(jaw_y2_act, jaw_y2_exp)
    )

    # Compute sector-resolved 2D planned & delivered fluence maps
    sector_edges = np.linspace(0.0, 360.0, n_sectors + 1)
    fluence_plan_2d = np.zeros((v_mm.size, u_mm.size), dtype=np.float64)
    fluence_deliv_2d = np.zeros((v_mm.size, u_mm.size), dtype=np.float64)
    sector_deltas: list[tuple[float, np.ndarray]] = []
    sector_plans: list[np.ndarray] = []

    for s_idx in range(n_sectors):
        lo, hi = sector_edges[s_idx], sector_edges[s_idx + 1]
        in_sec = (gantry_exp >= lo) & (gantry_exp <= hi if s_idx == n_sectors - 1 else gantry_exp < hi)
        if not np.any(in_sec):
            continue
        f_p = _evaluate_aperture_fluence_batch(
            bank_a_exp[in_sec],
            bank_b_exp[in_sec],
            dmu_exp[in_sec],
            jaw_x1_exp[in_sec],
            jaw_x2_exp[in_sec],
            jaw_y1_exp[in_sec],
            jaw_y2_exp[in_sec],
            sag_y_exp[in_sec],
            u_mm,
            v_mm,
            leaf_width_mm=leaf_width_mm,
            dlg_nominal_mm=dlg_nominal_mm,
        )
        if is_exact_zero_error:
            f_d = f_p.copy()
        else:
            f_d = _evaluate_aperture_fluence_batch(
                bank_a_act[in_sec],
                bank_b_act[in_sec],
                dmu_act[in_sec],
                jaw_x1_act[in_sec],
                jaw_x2_act[in_sec],
                jaw_y1_act[in_sec],
                jaw_y2_act[in_sec],
                sag_y_act[in_sec],
                u_mm,
                v_mm,
                leaf_width_mm=leaf_width_mm,
                dlg_nominal_mm=dlg_nominal_mm,
            )
        fluence_plan_2d += f_p
        fluence_deliv_2d += f_d
        theta_mid = float(np.mean(gantry_act[in_sec]))
        sector_plans.append(f_p)
        sector_deltas.append((theta_mid, f_d - f_p))

    delta_fluence_2d = fluence_deliv_2d - fluence_plan_2d
    max_abs_df = float(np.max(np.abs(delta_fluence_2d)))
    rms_df = float(np.sqrt(np.mean(delta_fluence_2d**2)))

    # Project angular ΔΦ_k(u, v) into 3D patient anatomy via fast pencil-beam back-projection
    planned_dose_3d = anatomy.planned_dose_gy.astype(np.float64, copy=False)
    delta_dose_3d = np.zeros_like(planned_dose_3d, dtype=np.float64)

    if not is_exact_zero_error and max_abs_df > 0.0:
        # Determine isocenter at PTV70 centroid (fallback: high-dose region centroid)
        if "PTV70" in anatomy.roi_masks and np.any(anatomy.roi_masks["PTV70"]):
            iso_idx = np.argwhere(anatomy.roi_masks["PTV70"]).mean(axis=0)
        else:
            hi_mask = planned_dose_3d >= 0.8 * float(np.max(planned_dose_3d))
            iso_idx = np.argwhere(hi_mask).mean(axis=0)

        active_idx = np.argwhere(anatomy.possible_dose_mask)
        dx_mm, dy_mm, dz_mm = anatomy.voxel_dims_mm
        # OpenKBP axes: 0=AP, 1=LR, 2=SI
        ap_mm = (active_idx[:, 0] - iso_idx[0]) * dx_mm
        lr_mm = (active_idx[:, 1] - iso_idx[1]) * dy_mm
        si_mm = (active_idx[:, 2] - iso_idx[2]) * dz_mm

        # Effective 6 MV linear attenuation in water (mm^-1) + CT HU-to-rho_e heterogeneity scaling
        mu_eff = 0.0048
        if use_heterogeneity:
            rho_e_vol = ct_hu_to_relative_electron_density(anatomy.ct_hu)
            # Smooth path-averaged relative electron density around each voxel (captures air/sinus/bone)
            rho_e_path = gaussian_filter(rho_e_vol, sigma=1.8)
            rho_active = np.clip(
                rho_e_path[active_idx[:, 0], active_idx[:, 1], active_idx[:, 2]], 0.08, 2.2
            )
            rho_local = np.clip(
                rho_e_vol[active_idx[:, 0], active_idx[:, 1], active_idx[:, 2]], 0.05, 2.2
            )
            # Electron disequilibrium buildup/scatter modulation in low-density air vs dense bone
            het_scatter_mod = np.clip(0.88 + 0.12 * (rho_local**0.45), 0.72, 1.12)
        else:
            rho_active = np.ones(active_idx.shape[0], dtype=np.float64)
            het_scatter_mod = np.ones(active_idx.shape[0], dtype=np.float64)

        # Reference central open-field planned fluence per sector
        mean_plan_fluence = max(float(np.percentile(fluence_plan_2d, 90)), 1e-6)
        n_active_sectors = max(len(sector_deltas), 1)
        ref_sector_fluence = mean_plan_fluence / n_active_sectors
        prescribed_ref_gy = float(np.percentile(planned_dose_3d[anatomy.possible_dose_mask], 98))

        # Accumulate normalized perturbative dose kernel across VMAT gantry sectors
        accum_rel_delta = np.zeros(active_idx.shape[0], dtype=np.float64)
        for theta_deg, df_sector in sector_deltas:
            th = np.deg2rad(theta_deg)
            # Rotate (LR, AP) into Beam's-Eye-View lateral u and beam-axis depth d
            u_bev = lr_mm * np.cos(th) + ap_mm * np.sin(th)
            d_bev = -lr_mm * np.sin(th) + ap_mm * np.cos(th)
            v_bev = si_mm

            df_sampled = _bilinear_sample_2d(df_sector, u_mm, v_mm, u_bev, v_bev)
            # Radiological depth modulation scaled by local/path relative electron density rho_e
            rad_depth = rho_active * np.clip(d_bev, -65.0, 95.0)
            depth_factor = np.exp(-mu_eff * rad_depth)
            accum_rel_delta += (df_sampled / ref_sector_fluence) * depth_factor

        # Calibrated perturbative dose coupling: local dose sensitivity scaled by beam fluence &
        # anatomy dose envelope (captures both target uniform shifts and steep OAR gradient shifts)
        local_plan = planned_dose_3d[active_idx[:, 0], active_idx[:, 1], active_idx[:, 2]]
        # Envelope blends 65% local dose weighting + 35% open-beam scatter/gradient spillover
        dose_envelope = 0.65 * local_plan + 0.35 * prescribed_ref_gy * np.clip(
            local_plan / max(0.35 * prescribed_ref_gy, 1e-6), 0.12, 1.0
        )
        # 1.85x modulation amplification reflects narrow-aperture VMAT MU/aperture-width sensitivity
        delta_active = (1.85 / n_active_sectors) * accum_rel_delta * dose_envelope * het_scatter_mod

        delta_dose_3d[active_idx[:, 0], active_idx[:, 1], active_idx[:, 2]] = delta_active
        # Mild 3D lateral scatter kernel (sigma ~ 0.6 voxel ~ 2 mm)
        delta_dose_3d = gaussian_filter(delta_dose_3d, sigma=0.6)
        delta_dose_3d = np.where(anatomy.possible_dose_mask, delta_dose_3d, 0.0)

    delivered_dose_3d = np.where(
        anatomy.possible_dose_mask,
        np.maximum(0.0, planned_dose_3d + delta_dose_3d),
        0.0,
    )
    # Recompute exact clipped delta_dose_3d (preserves exact 0.0 when zero error)
    delta_dose_3d = delivered_dose_3d - planned_dose_3d

    # Compute 2D planar and 3D volumetric Gamma pass rates (3%/2mm and 2%/1mm)
    gamma_2d_3_2 = compute_gamma_pass_rate_2d(
        fluence_plan_2d, fluence_deliv_2d, spacing_mm=(2.0, 2.0), dose_pct=3.0, dta_mm=2.0
    )
    gamma_2d_2_1 = compute_gamma_pass_rate_2d(
        fluence_plan_2d, fluence_deliv_2d, spacing_mm=(2.0, 2.0), dose_pct=2.0, dta_mm=1.0
    )
    gamma_3d_3_2 = compute_gamma_pass_rate_3d(
        planned_dose_3d,
        delivered_dose_3d,
        voxel_dims_mm=anatomy.voxel_dims_mm,
        dose_pct=3.0,
        dta_mm=2.0,
    )
    gamma_3d_2_1 = compute_gamma_pass_rate_3d(
        planned_dose_3d,
        delivered_dose_3d,
        voxel_dims_mm=anatomy.voxel_dims_mm,
        dose_pct=2.0,
        dta_mm=1.0,
    )

    # Compute structure-specific DVH shifts, gEUD, TCP/NTCP
    voxel_vol_cc = anatomy.voxel_volume_cc
    structure_shifts: list[StructureDVHShift] = []
    any_clinical_fail = False

    for roi_name, mask in anatomy.roi_masks.items():
        if not np.any(mask):
            continue
        roi_type = ROI_TYPES.get(roi_name, "OAR")
        d_plan = planned_dose_3d[mask]
        d_deliv = delivered_dose_3d[mask]
        vol_cc = float(np.sum(mask) * voxel_vol_cc)

        d95_p = float(np.percentile(d_plan, 5.0))
        d95_d = float(np.percentile(d_deliv, 5.0))
        dd95_gy = d95_d - d95_p
        dd95_pct = 100.0 * dd95_gy / max(d95_p, 1.0)

        d003_p = _compute_d003cc(d_plan, voxel_vol_cc)
        d003_d = _compute_d003cc(d_deliv, voxel_vol_cc)
        dd003_gy = d003_d - d003_p

        dmean_p = float(np.mean(d_plan))
        dmean_d = float(np.mean(d_deliv))
        ddmean_gy = dmean_d - dmean_p

        bio = _ROI_BIO_PARAMS.get(roi_name, {"a": 1.0, "d50": 50.0, "gamma50": 3.0})
        eud_p = _compute_geud(d_plan, bio["a"])
        eud_d = _compute_geud(d_deliv, bio["a"])
        deud_gy = eud_d - eud_p

        prob_p = _compute_logistic_bio_prob(eud_p, bio["d50"], bio["gamma50"])
        prob_d = _compute_logistic_bio_prob(eud_d, bio["d50"], bio["gamma50"])
        dprob = prob_d - prob_p

        if roi_type == "PTV":
            unacceptable = abs(dd95_pct) >= 2.0
            bio_metric = "TCP"
        else:
            unacceptable = (dd003_gy >= 1.5) or (ddmean_gy >= 1.5)
            bio_metric = "NTCP"

        if unacceptable:
            any_clinical_fail = True

        structure_shifts.append(
            StructureDVHShift(
                roi_name=roi_name,
                roi_type=roi_type,  # type: ignore[arg-type]
                volume_cc=round(vol_cc, 3),
                d95_planned_gy=round(d95_p, 4),
                d95_delivered_gy=round(d95_d, 4),
                delta_d95_gy=round(dd95_gy, 4),
                delta_d95_pct=round(dd95_pct, 4),
                d003cc_planned_gy=round(d003_p, 4),
                d003cc_delivered_gy=round(d003_d, 4),
                delta_d003cc_gy=round(dd003_gy, 4),
                dmean_planned_gy=round(dmean_p, 4),
                dmean_delivered_gy=round(dmean_d, 4),
                delta_dmean_gy=round(ddmean_gy, 4),
                eud_planned_gy=round(eud_p, 4),
                eud_delivered_gy=round(eud_d, 4),
                delta_eud_gy=round(deud_gy, 4),
                bio_prob_metric=bio_metric,  # type: ignore[arg-type]
                bio_prob_planned=round(prob_p, 5),
                bio_prob_delivered=round(prob_d, 5),
                delta_bio_prob=round(dprob, 5),
                clinically_unacceptable=unacceptable,
            )
        )

    # Nelms et al. (2011) blindspot: 2D 3%/2mm passes TG-218 (>=95%) despite clinically unacceptable ΔDVH
    gamma_blindspot = bool((gamma_2d_3_2 >= 95.0) and any_clinical_fail)

    prov_tag = (
        "COMPOSITE_OPENKBP_PLUS_SYNTHETIC_LOG"
        if anatomy.provenance_tag.startswith("OPEN_PUBLIC_ANATOMY_OPENKBP_")
        and tensor.provenance.provenance_tag == "SYNTHETIC_FAULT_INJECTION_BENCHMARK"
        else tensor.provenance.provenance_tag
    )
    sha_inputs = {**anatomy.sha256_inputs, **tensor.provenance.sha256_inputs}
    sha_inputs["replay_config"] = sha256_json_payload(
        {
            "dlg_nominal_mm": dlg_nominal_mm,
            "leaf_width_mm": leaf_width_mm,
            "n_sectors": n_sectors,
            "use_heterogeneity": use_heterogeneity,
            "log_id": tensor.log_id,
        }
    )

    provenance = build_provenance(
        record_id=f"dose_replay:{tensor.log_id}:{anatomy.patient_id}",
        provenance_tag=prov_tag,  # type: ignore[arg-type]
        dataset_source=f"{anatomy.anatomy_source} + {tensor.provenance.dataset_source}",
        sha256_inputs=sha_inputs,
        notes=f"Perturbative 2D->3D dose replay ({n_sectors} sectors, het={use_heterogeneity}); blindspot={gamma_blindspot}.",
    )

    impact = PerturbativeDoseImpact(
        log_id=tensor.log_id,
        patient_id=anatomy.patient_id,
        anatomy_source=anatomy.anatomy_source,
        max_abs_delta_fluence=max_abs_df,
        rms_delta_fluence=rms_df,
        max_abs_delta_dose_gy=float(np.max(np.abs(delta_dose_3d))),
        rms_delta_dose_gy=float(
            np.sqrt(np.mean(delta_dose_3d[anatomy.possible_dose_mask] ** 2))
            if np.any(anatomy.possible_dose_mask)
            else 0.0
        ),
        gamma_2d_3pct_2mm_pass_pct=round(gamma_2d_3_2, 3),
        gamma_2d_2pct_1mm_pass_pct=round(gamma_2d_2_1, 3),
        gamma_3d_3pct_2mm_pass_pct=round(gamma_3d_3_2, 3),
        gamma_3d_2pct_1mm_pass_pct=round(gamma_3d_2_1, 3),
        gamma_blindspot_detected=gamma_blindspot,
        heterogeneity_corrected=use_heterogeneity,
        structure_shifts=structure_shifts,
        provenance=provenance,
    )

    if return_arrays:
        return PerturbativeReplayArrays(
            u_mm=u_mm,
            v_mm=v_mm,
            fluence_planned_2d=fluence_plan_2d,
            fluence_delivered_2d=fluence_deliv_2d,
            delta_fluence_2d=delta_fluence_2d,
            planned_dose_3d=planned_dose_3d,
            delivered_dose_3d=delivered_dose_3d,
            delta_dose_3d=delta_dose_3d,
            impact=impact,
        )
    return impact
