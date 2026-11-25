#!/usr/bin/env python3
"""Generate 8 publication-grade 300-DPI PNG + PDF figures for linac-forensics.

Research Software — NOT A MEDICAL DEVICE.

Visual Identity: Aerospace / Linac Control-Room Telemetry & Oscilloscope
Instrumentation — brushed instrument-aluminum canvas (#EAEDF0), crisp white
plot panels (#FFFFFF) inside full 4-sided dark-graphite precision bezels
(#2B303A, 1.05pt), inward-pointing engineering ticks, dual-tier oscilloscope
grids (#CFD6DE / #E6EBF0), dark-graphite top telemetry header banners (#1E242B)
with industrial amber accent bars (#F77F00), monospaced instrumentation labels
(DejaVu Sans Mono) paired with crisp technical sans-serif titles (DejaVu Sans),
dark-graphite rectangular channel tags (CH-A, CH-B, CH-C), and a colorblind-safe
industrial telemetry palette.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Rectangle
from matplotlib.transforms import offset_copy

from linac_forensics.anatomy import load_openkbp_cohort, load_openkbp_patient
from linac_forensics.causal_fault_model import diagnose_causal_faults
from linac_forensics.epid_forensics import (
    EPIDThreeWayArrays,
    analyze_three_way_epid_forensics,
    simulate_measured_epid_image,
)
from linac_forensics.harness.agent_orchestrator import (
    TG142_LIMITS,
    generate_tg142_fleet_trends,
    run_fleet_triage_investigation,
)
from linac_forensics.log_parser import generate_synthetic_vmat_log
from linac_forensics.perturbative_dose import (
    PerturbativeReplayArrays,
    compute_perturbative_dose_replay,
)
from linac_forensics.predictive_rul import forecast_tg142_kalman_rul
from linac_forensics.robustness_certificate import (
    STATIC_TG142_LIMITS,
    compute_fleet_routing_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENKBP_PT1 = REPO_ROOT / "data" / "openkbp" / "pt_1"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

# Colorblind-safe Industrial Telemetry Palette
TEL = {
    "navy": "#1D3557",        # Primary Telemetry Signal (Gunmetal Navy)
    "red": "#D62828",         # Fault / Interlock / Violation (Industrial Signal Red)
    "amber": "#E87A00",       # Mechanical / Lag / Caution (Safety Ochre-Amber)
    "teal": "#1B8A7A",        # Nominal / Pass / Healthy (Instrumentation Teal)
    "plum": "#6D597A",        # Secondary / Auxiliary (Slate Plum)
    "graphite": "#4A5568",    # Neutral / Baseline (Graphite)
    # Instrument chrome & surfaces
    "canvas": "#EAEDF0",      # Brushed instrument-aluminum light gray
    "panel": "#FFFFFF",       # Crisp white oscilloscope plot panel
    "banner": "#1E242B",      # Dark-graphite telemetry banner & channel tags
    "bezel": "#2B303A",       # 4-sided precision instrument bezel
    "accent": "#F77F00",      # Industrial amber accent bar
    "grid_maj": "#CFD6DE",    # Oscilloscope major grid
    "grid_min": "#E6EBF0",    # Oscilloscope minor grid
    "callout_bg": "#F4F6F8",  # Recessed instrument readout box
    "ink": "#1E242B",         # Deep instrument ink
    "red_dark": "#8E1919",    # High-contrast dark signal red text
    "amber_dark": "#9A4F00",  # High-contrast dark amber text
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans Mono",
            "font.size": 8.4,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": TEL["ink"],
            "axes.labelsize": 8.4,
            "axes.labelweight": "bold",
            "axes.labelcolor": TEL["bezel"],
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "xtick.color": TEL["bezel"],
            "ytick.color": TEL["bezel"],
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.size": 2.4,
            "ytick.minor.size": 2.4,
            "xtick.minor.width": 0.65,
            "ytick.minor.width": 0.65,
            "axes.spines.top": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.spines.right": True,
            "axes.edgecolor": TEL["bezel"],
            "axes.linewidth": 1.05,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": TEL["grid_maj"],
            "grid.alpha": 0.9,
            "grid.linestyle": "-",
            "grid.linewidth": 0.65,
            "legend.fontsize": 7.2,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": TEL["bezel"],
            "legend.fancybox": False,
            "figure.facecolor": TEL["canvas"],
            "axes.facecolor": TEL["panel"],
            "savefig.facecolor": TEL["canvas"],
            "savefig.edgecolor": "none",
            "savefig.dpi": 300,
        }
    )


def _add_telemetry_banner(fig: plt.Figure, title: str, readout_tag: str) -> None:
    """Add dark-graphite control-room telemetry banner with industrial amber accent."""
    # Main dark-graphite top banner strip
    fig.patches.append(
        Rectangle(
            (0.0, 0.905),
            1.0,
            0.095,
            transform=fig.transFigure,
            facecolor=TEL["banner"],
            edgecolor="none",
            zorder=1,
        )
    )
    # Industrial amber lower accent rule
    fig.patches.append(
        Rectangle(
            (0.0, 0.896),
            1.0,
            0.009,
            transform=fig.transFigure,
            facecolor=TEL["accent"],
            edgecolor="none",
            zorder=2,
        )
    )
    # Left industrial amber indicator tab
    fig.patches.append(
        Rectangle(
            (0.0, 0.905),
            0.007,
            0.095,
            transform=fig.transFigure,
            facecolor=TEL["accent"],
            edgecolor="none",
            zorder=3,
        )
    )
    fig.text(
        0.018,
        0.953,
        title,
        ha="left",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        color="#FFFFFF",
        zorder=4,
    )
    fig.text(
        0.985,
        0.953,
        readout_tag,
        ha="right",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        fontfamily="DejaVu Sans Mono",
        color=TEL["accent"],
        bbox={
            "boxstyle": "square,pad=0.30",
            "facecolor": "#2B303A",
            "edgecolor": TEL["accent"],
            "linewidth": 0.9,
        },
        zorder=4,
    )


def _set_panel_header(ax: plt.Axes, channel: str, title: str, title_fontsize: float = 9.4) -> None:
    """Place dark-graphite rectangular channel tag (CH-A, CH-B, ...) beside crisp sans header."""
    ax.text(
        0.0,
        1.036,
        channel,
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        fontfamily="DejaVu Sans Mono",
        color="#FFFFFF",
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "square,pad=0.26",
            "facecolor": TEL["banner"],
            "edgecolor": TEL["accent"],
            "linewidth": 0.85,
        },
        clip_on=False,
    )
    fig = ax.get_figure()
    assert fig is not None
    title_trans = offset_copy(ax.transAxes, fig=fig, x=37, y=0, units="points")
    ax.text(
        0.0,
        1.034,
        title,
        transform=title_trans,
        fontsize=title_fontsize,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        color=TEL["ink"],
        va="bottom",
        ha="left",
        clip_on=False,
    )


def _style_scope_axis(
    ax: plt.Axes,
    *,
    grid: bool = True,
    minor_x: bool = True,
    minor_y: bool = True,
    dark_ticks: bool = False,
) -> None:
    """Apply 4-sided precision bezel, inward engineering ticks, and oscilloscope grid."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(TEL["bezel"])
        spine.set_linewidth(1.05)

    tick_col = "#E6EBF0" if dark_ticks else TEL["bezel"]
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        color=tick_col,
        labelcolor=TEL["bezel"],
        length=4.5,
        width=0.9,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=minor_x,
        bottom=minor_x,
        left=minor_y,
        right=minor_y,
        color=tick_col,
        length=2.4,
        width=0.65,
    )
    if not minor_x:
        ax.xaxis.set_minor_locator(plt.NullLocator())
    if not minor_y:
        ax.yaxis.set_minor_locator(plt.NullLocator())

    if grid:
        ax.set_axisbelow(True)
        ax.grid(True, which="major", color=TEL["grid_maj"], linewidth=0.68, linestyle="-", alpha=0.92)
        if minor_x or minor_y:
            ax.grid(True, which="minor", color=TEL["grid_min"], linewidth=0.45, linestyle="-", alpha=0.90)
    else:
        ax.grid(False, which="both")


def _style_colorbar(cbar: matplotlib.colorbar.Colorbar, label: str) -> None:
    """Style colorbar with dark-graphite precision bezel, inward ticks, and monospace label."""
    cbar.outline.set_edgecolor(TEL["bezel"])
    cbar.outline.set_linewidth(1.05)
    cbar.ax.tick_params(
        direction="in",
        color=TEL["bezel"],
        labelcolor=TEL["bezel"],
        labelsize=7.4,
        length=3.5,
        width=0.8,
    )
    cbar.set_label(label, fontsize=8.0, fontweight="bold", fontfamily="DejaVu Sans Mono", color=TEL["bezel"])


def _style_legend(ax: plt.Axes, *args: object, **kwargs: object) -> matplotlib.legend.Legend:
    """Create crisp square instrument-box legend with dark-graphite bezel."""
    leg = ax.legend(*args, **kwargs)
    frame = leg.get_frame()
    frame.set_facecolor("#FFFFFF")
    frame.set_edgecolor(TEL["bezel"])
    frame.set_linewidth(0.85)
    frame.set_alpha(0.95)
    return leg


def _save_fig(fig: plt.Figure, stem: str, footer: str) -> None:
    fig.patches.append(
        Rectangle(
            (0.015, 0.045),
            0.97,
            0.0022,
            transform=fig.transFigure,
            facecolor=TEL["grid_maj"],
            edgecolor="none",
            zorder=2,
        )
    )
    fig.text(
        0.5,
        0.014,
        footer,
        ha="center",
        va="bottom",
        fontsize=6.9,
        fontfamily="DejaVu Sans Mono",
        color=TEL["graphite"],
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, facecolor=TEL["canvas"])
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", facecolor=TEL["canvas"])
    plt.close(fig)


def make_fig1_trajectory_telemetry() -> None:
    """Fig 1: 50 Hz MLC leaf RMS heatmap, velocity-vs-lag scatter, and gravity-sag harmonic."""
    log = generate_synthetic_vmat_log(
        log_id="FIG1_FORENSICS",
        n_snapshots=360,
        failing_bank="B",
        failing_leaf_index=34,
        leaf_drag_ms=42.0,
        dlg_offset_mm=+0.45,
        gantry_sag_sin_mm=0.90,
        gantry_sag_cos_mm=-0.45,
        mu_drift_pct=+1.20,
        noise_sigma_mm=0.03,
        seed=42,
    )
    diag = diagnose_causal_faults(log)

    dt_s = log.sampling_interval_ms / 1000.0
    gantry = np.asarray(log.gantry_expected_deg)
    ba_exp = np.asarray(log.bank_a_expected_mm)
    ba_act = np.asarray(log.bank_a_actual_mm)
    bb_exp = np.asarray(log.bank_b_expected_mm)
    bb_act = np.asarray(log.bank_b_actual_mm)
    err_b = bb_act - bb_exp
    vel_b = np.gradient(bb_exp, dt_s, axis=0)
    err_a = ba_act - ba_exp

    # Active central leaves 18..43 (1-based)
    leaf_slice = slice(17, 43)
    leaf_nums = np.arange(18, 44)
    # Absolute leaf error [n_active, n_snapshots] for Bank B
    abs_err_b = np.abs(err_b[:, leaf_slice]).T

    fig = plt.figure(figsize=(15.2, 5.15))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.25, 1.05, 1.15], wspace=0.33, left=0.055, right=0.97, top=0.80, bottom=0.165
    )

    # Panel A: 50 Hz Bank B MLC Leaf Error Heatmap across VMAT arc
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(
        abs_err_b,
        aspect="auto",
        origin="lower",
        extent=[float(gantry[0]), float(gantry[-1]), float(leaf_nums[0]) - 0.5, float(leaf_nums[-1]) + 0.5],
        cmap="afmhot",
        vmin=0.0,
        vmax=max(1.2, float(np.percentile(abs_err_b, 99.5))),
    )
    ax1.axhline(34, color=TEL["accent"], lw=1.35, ls="--", alpha=0.95)
    ax1.annotate(
        "BANK B LEAF #34\n(drag anomaly)",
        xy=(190, 34),
        xytext=(195, 39.2),
        color="#FFFFFF",
        fontsize=7.6,
        fontweight="bold",
        bbox={
            "boxstyle": "square,pad=0.28",
            "facecolor": TEL["banner"],
            "edgecolor": TEL["accent"],
            "linewidth": 1.0,
            "alpha": 0.94,
        },
        arrowprops={"arrowstyle": "->", "color": TEL["accent"], "lw": 1.35},
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    _style_colorbar(cbar, "|Δx| Leaf Error (mm)")
    _style_scope_axis(ax1, grid=False, minor_x=False, minor_y=False, dark_ticks=True)
    _set_panel_header(ax1, "CH-A", "50 Hz Bank B MLC Error Heatmap")
    ax1.set_xlabel("Gantry Angle θ (deg)")
    ax1.set_ylabel("Active MLC Leaf Index (1-based)")

    # Panel B: Velocity vs Position Error Scatter (Leaf #34 vs Nominal Leaf #30)
    ax2 = fig.add_subplot(gs[0, 1])
    v34, e34 = vel_b[:, 33], err_b[:, 33]
    v30, e30 = vel_b[:, 29], err_b[:, 29]
    ax2.scatter(v30, e30, s=15, alpha=0.52, color=TEL["teal"], label="Healthy Leaf #30 (τ ≈ 0 ms)")
    ax2.scatter(v34, e34, s=17, alpha=0.72, color=TEL["red"], label="Failing Leaf #34 (τ = 42.0 ms)")

    v_line = np.linspace(float(v34.min()), float(v34.max()), 100)
    top_leaf = diag.flagged_leaves[0]
    e_fit = top_leaf.systematic_offset_mm - (top_leaf.velocity_lag_ms / 1000.0) * v_line
    ax2.plot(
        v_line,
        e_fit,
        color=TEL["navy"],
        lw=2.1,
        ls="-",
        label=f"Matched filter: τ̂={top_leaf.velocity_lag_ms:.1f} ms (r={top_leaf.velocity_correlation_r:.2f})",
    )
    ax2.axhline(
        -0.5 * diag.estimated_dlg_offset_mm,
        color=TEL["amber"],
        lw=1.45,
        ls="--",
        label=f"Bank DLG offset ({-0.5 * diag.estimated_dlg_offset_mm:+.2f} mm)",
    )
    _style_scope_axis(ax2, grid=True, minor_x=True, minor_y=True)
    _set_panel_header(ax2, "CH-B", "Velocity-Proportional Viscous Drag")
    ax2.set_xlabel("Planned Leaf Velocity v(t) (mm/s)")
    ax2.set_ylabel("Position Residual Δx(t) (mm)")
    _style_legend(ax2, loc="upper right", fontsize=6.8)

    # Panel C: Gantry Gravity-Sag Harmonic Residual Δy(θ)
    ax3 = fig.add_subplot(gs[0, 2])
    sag_obs = 0.5 * (
        (np.asarray(log.jaw_actual_mm["Y1"]) - np.asarray(log.jaw_expected_mm["Y1"]))
        + (np.asarray(log.jaw_actual_mm["Y2"]) - np.asarray(log.jaw_expected_mm["Y2"]))
    )
    th_rad = np.deg2rad(gantry)
    sag_hyp = next(h for h in diag.hypotheses if h.fault_type == "GANTRY_GRAVITY_SAG_HARMONIC")
    a_hat = float(sag_hyp.parameters["sin_coeff_mm"])
    b_hat = float(sag_hyp.parameters["cos_coeff_mm"])
    sag_fit = a_hat * np.sin(th_rad) + b_hat * np.cos(th_rad)
    sag_res = sag_obs - sag_fit

    ax3.scatter(gantry[::2], sag_obs[::2], s=15, alpha=0.48, color=TEL["navy"], label="50 Hz Head/Jaw Y Residual")
    ax3.plot(
        gantry,
        sag_fit,
        color=TEL["red"],
        lw=2.2,
        label=f"Harmonic fit: R̂={diag.estimated_gantry_sag_amp_mm:.2f} mm, φ̂={diag.estimated_gantry_sag_phase_deg:.0f}°",
    )
    ax3.plot(
        gantry,
        sag_res,
        color=TEL["graphite"],
        lw=1.05,
        alpha=0.80,
        label=f"Post-fit residual (RMS={np.std(sag_res):.3f} mm)",
    )
    ax3.axhline(1.0, color=TEL["amber"], lw=1.35, ls="--", label="TG-142 Sag Action (±1.0 mm)")
    ax3.axhline(-1.0, color=TEL["amber"], lw=1.35, ls="--")
    _style_scope_axis(ax3, grid=True, minor_x=True, minor_y=True)
    _set_panel_header(ax3, "CH-C", "Gantry Gravity-Sag Harmonic Δy(θ)")
    ax3.set_xlabel("Gantry Angle θ (deg)")
    ax3.set_ylabel("Sagittal Head Shift Δy (mm)")
    ax3.set_xlim(0, 360)
    ax3.set_ylim(-1.48, 1.48)
    _style_legend(ax3, loc="lower left", fontsize=6.7)

    bank_a_rms = float(np.sqrt(np.mean(err_a[:, leaf_slice] ** 2)))
    _add_telemetry_banner(
        fig,
        "HIGH-FREQUENCY (50 HZ) TRAJECTORY LOG FORENSICS — MULTI-FAULT ATTRIBUTION",
        f"TEL-01 // BANK A RMS {bank_a_rms:.2f} mm",
    )
    _save_fig(
        fig,
        "fig1_trajectory_telemetry_forensics",
        "PROVENANCE: SYNTHETIC_FAULT_INJECTION_BENCHMARK (Varian TrueBeam v5.0 50 Hz VMAT) | Research Software — NOT A MEDICAL DEVICE",
    )


def make_fig2_fluence_to_3d_dose() -> None:
    """Fig 2: 2D fluence error ΔΦ, axial OpenKBP pt_1 CT + 3D ΔDose overlay, and DVH shift."""
    anatomy = load_openkbp_patient(OPENKBP_PT1, verify_manifest=True)
    log = generate_synthetic_vmat_log(
        log_id="FIG2_OPENKBP_REPLAY",
        n_snapshots=240,
        failing_bank="B",
        failing_leaf_index=34,
        leaf_drag_ms=42.0,
        dlg_offset_mm=+0.75,
        gantry_sag_sin_mm=0.90,
        gantry_sag_cos_mm=0.25,
        mu_drift_pct=+1.20,
        noise_sigma_mm=0.02,
        seed=42,
    )
    res = compute_perturbative_dose_replay(log, anatomy, n_sectors=12, return_arrays=True)
    assert isinstance(res, PerturbativeReplayArrays)

    fig = plt.figure(figsize=(15.4, 5.2))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.05, 1.12, 1.28], wspace=0.31, left=0.055, right=0.97, top=0.80, bottom=0.165
    )

    # Panel A: 2D Fluence Error ΔΦ(u, v)
    ax1 = fig.add_subplot(gs[0, 0])
    df_norm = 100.0 * res.delta_fluence_2d / max(float(np.max(res.fluence_planned_2d)), 1e-6)
    vlim_f = max(4.0, float(np.percentile(np.abs(df_norm), 99.5)))
    im1 = ax1.imshow(
        df_norm,
        origin="lower",
        extent=[float(res.u_mm[0]), float(res.u_mm[-1]), float(res.v_mm[0]), float(res.v_mm[-1])],
        cmap="RdBu_r",
        vmin=-vlim_f,
        vmax=vlim_f,
        aspect="equal",
    )
    c1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    _style_colorbar(c1, "ΔΦ / Φ_max (%)")
    _style_scope_axis(ax1, grid=False, minor_x=False, minor_y=False)
    _set_panel_header(ax1, "CH-A", "2D Fluence Error ΔΦ(u, v)", title_fontsize=9.1)
    ax1.set_xlabel("Cross-Plane u (mm)")
    ax1.set_ylabel("In-Plane v (mm)")

    # Panel B: Axial OpenKBP pt_1 CT + 3D ΔDose(x, y, z) + ROI Contours
    ax2 = fig.add_subplot(gs[0, 1])
    ptv_slices = np.sum(anatomy.roi_masks["PTV70"], axis=(0, 1))
    cord_slices = np.sum(anatomy.roi_masks["SpinalCord"], axis=(0, 1))
    lpar_slices = np.sum(anatomy.roi_masks["LeftParotid"], axis=(0, 1))
    overlap_3 = (ptv_slices > 0) & (cord_slices > 0) & (lpar_slices > 0)
    if np.any(overlap_3):
        z_idx = int(np.argmax(ptv_slices * overlap_3))
    else:
        overlap_score = ptv_slices * (cord_slices > 0)
        z_idx = int(np.argmax(overlap_score)) if np.any(overlap_score > 0) else 66

    ct_sl = anatomy.ct_hu[:, :, z_idx]
    dd_sl = res.delta_dose_3d[:, :, z_idx]
    mask_sl = anatomy.possible_dose_mask[:, :, z_idx]

    nz_r, nz_c = np.where(mask_sl)
    r0, r1 = max(0, int(nz_r.min()) - 6), min(128, int(nz_r.max()) + 7)
    c0, c1 = max(0, int(nz_c.min()) - 6), min(128, int(nz_c.max()) + 7)

    ct_crop = np.clip(ct_sl[r0:r1, c0:c1], -300, 500)
    dd_crop = np.ma.masked_where(~mask_sl[r0:r1, c0:c1] | (np.abs(dd_sl[r0:r1, c0:c1]) < 0.25), dd_sl[r0:r1, c0:c1])
    vlim_d = max(2.5, float(np.percentile(np.abs(dd_sl[mask_sl]), 99.5)))

    ax2.imshow(ct_crop, cmap="gray", origin="upper", aspect="equal")
    im2 = ax2.imshow(dd_crop, cmap="RdBu_r", vmin=-vlim_d, vmax=vlim_d, alpha=0.72, origin="upper", aspect="equal")
    contour_specs = [
        ("PTV70", TEL["accent"], 1.9),
        ("SpinalCord", TEL["teal"], 1.9),
        ("LeftParotid", TEL["red"], 1.6),
        ("RightParotid", TEL["plum"], 1.6),
    ]
    legend_handles = []
    for roi, col, lw in contour_specs:
        roi_sl = anatomy.roi_masks[roi][r0:r1, c0:c1, z_idx]
        if np.any(roi_sl):
            ax2.contour(roi_sl.astype(float), levels=[0.5], colors=[col], linewidths=lw)
            legend_handles.append(Line2D([0], [0], color=col, lw=lw, label=roi))

    c2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    _style_colorbar(c2, "3D ΔDose (Gy)")
    _style_scope_axis(ax2, grid=False, minor_x=False, minor_y=False)
    _set_panel_header(ax2, "CH-B", f"Axial CT + 3D ΔDose (z={z_idx})", title_fontsize=9.1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    _style_legend(ax2, handles=legend_handles, loc="lower left", fontsize=6.6, ncol=2)

    # Panel C: Planned vs Delivered Cumulative DVH
    ax3 = fig.add_subplot(gs[0, 2])
    dose_bins = np.linspace(0.0, 80.0, 400)
    dvh_rois = [
        ("PTV70", TEL["red"]),
        ("PTV56", TEL["amber"]),
        ("LeftParotid", TEL["teal"]),
        ("Brainstem", TEL["plum"]),
        ("SpinalCord", TEL["navy"]),
    ]
    shifts_map = {s.roi_name: s for s in res.impact.structure_shifts}
    for roi, col in dvh_rois:
        m = anatomy.roi_masks[roi]
        dp = res.planned_dose_3d[m]
        dd = res.delivered_dose_3d[m]
        vol_p = [100.0 * float(np.mean(dp >= d)) for d in dose_bins]
        vol_d = [100.0 * float(np.mean(dd >= d)) for d in dose_bins]
        ax3.plot(dose_bins, vol_p, color=col, lw=1.95, ls="-", label=roi)
        ax3.plot(dose_bins, vol_d, color=col, lw=1.7, ls="--", alpha=0.88)

    ptv_s = shifts_map["PTV70"]
    cord_s = shifts_map["SpinalCord"]
    callout = (
        f"SOLID: Planned | DASHED: Delivered\n"
        f"PTV70 ΔD95: {ptv_s.delta_d95_gy:+.2f} Gy ({ptv_s.delta_d95_pct:+.2f}%)\n"
        f"SpinalCord ΔD0.03cc: {cord_s.delta_d003cc_gy:+.2f} Gy\n"
        f"2D γ(3%/2mm): {res.impact.gamma_2d_3pct_2mm_pass_pct:.1f}% | "
        f"3D γ(2%/1mm): {res.impact.gamma_3d_2pct_1mm_pass_pct:.1f}%"
    )
    ax3.text(
        0.03,
        0.04,
        callout,
        transform=ax3.transAxes,
        fontsize=6.7,
        va="bottom",
        ha="left",
        color=TEL["ink"],
        bbox={
            "boxstyle": "square,pad=0.32",
            "facecolor": TEL["callout_bg"],
            "edgecolor": TEL["bezel"],
            "linewidth": 0.9,
            "alpha": 0.96,
        },
    )
    _style_scope_axis(ax3, grid=True, minor_x=True, minor_y=True)
    _set_panel_header(ax3, "CH-C", "3D Cumulative DVH (Plan vs Deliv.)", title_fontsize=9.1)
    ax3.set_xlabel("Dose (Gy)")
    ax3.set_ylabel("Volume (%)")
    ax3.set_xlim(0, 79)
    ax3.set_ylim(0, 103)
    _style_legend(ax3, loc="upper right", fontsize=7.0)

    _add_telemetry_banner(
        fig,
        "PERTURBATIVE FLUENCE-TO-3D-DOSE REPLAY ON REAL OPENKBP HEAD-AND-NECK ANATOMY",
        "TEL-02 // OPENKBP PT_1",
    )
    _save_fig(
        fig,
        "fig2_fluence_to_3d_delta_dose",
        "PROVENANCE: OPEN_PUBLIC_ANATOMY_OPENKBP_PT1 (Babier et al. 2021, CC BY 4.0) + SYNTHETIC_FAULT_INJECTION_BENCHMARK | Research Software — NOT A MEDICAL DEVICE",
    )


def make_fig3_gamma_blindspot(summary: dict) -> None:
    """Fig 3: Nelms et al. (2011) 2D 3%/2mm Gamma Blindspot vs True 3D ΔDVH Failures."""
    cases = summary["gamma_blindspot_benchmark"]["cases"]
    g2_32 = np.array([c["gamma_2d_3pct_2mm"] for c in cases])
    g2_21 = np.array([c["gamma_2d_2pct_1mm"] for c in cases])
    abs_d95 = np.array([abs(c["ptv70_delta_d95_pct"]) for c in cases])
    unacc = np.array([c["clinically_unacceptable"] for c in cases])
    blind = np.array([c["gamma_blindspot"] for c in cases])

    pure = [c for i, c in enumerate(cases) if i % 2 == 0]
    dlg_vals = np.array([c["dlg_offset_mm"] for c in pure])
    p_g2_32 = np.array([c["gamma_2d_3pct_2mm"] for c in pure])
    p_g3_32 = np.array([c["gamma_3d_3pct_2mm"] for c in pure])
    p_g3_21 = np.array([c["gamma_3d_2pct_1mm"] for c in pure])
    p_abs_d95 = np.array([abs(c["ptv70_delta_d95_pct"]) for c in pure])

    fig = plt.figure(figsize=(14.4, 5.2))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.1, 1.15], wspace=0.30, left=0.065, right=0.925, top=0.80, bottom=0.165
    )

    # Panel A: 2D Gamma vs |PTV70 ΔD95| Blindspot Quadrant
    ax1 = fig.add_subplot(gs[0, 0])
    y_max = max(4.8, float(abs_d95.max()) + 0.65)
    x_min = min(92.0, float(g2_32.min()) - 1.5)
    ax1.add_patch(
        Rectangle(
            (95.0, 2.0),
            100.8 - 95.0,
            y_max - 2.0,
            facecolor=TEL["red"],
            alpha=0.11,
            edgecolor="none",
        )
    )
    ax1.axvline(95.0, color=TEL["ink"], lw=1.35, ls="--", label="TG-218 Universal Tolerance (95%)")
    ax1.axhline(2.0, color=TEL["red"], lw=1.35, ls=":", label="Clinical |ΔD95(PTV70)| Limit (2.0%)")

    ok_mask = ~unacc
    caught_mask = unacc & ~blind
    ax1.scatter(
        g2_32[ok_mask],
        abs_d95[ok_mask],
        s=54,
        color=TEL["teal"],
        edgecolor="#FFFFFF",
        lw=0.7,
        label=f"Clinically Acceptable (n={int(ok_mask.sum())})",
        zorder=4,
    )
    if np.any(caught_mask):
        ax1.scatter(
            g2_32[caught_mask],
            abs_d95[caught_mask],
            s=58,
            marker="s",
            color=TEL["navy"],
            edgecolor="#FFFFFF",
            lw=0.7,
            label=f"Caught by 2D γ (n={int(caught_mask.sum())})",
            zorder=4,
        )
    ax1.scatter(
        g2_32[blind],
        abs_d95[blind],
        s=76,
        marker="^",
        color=TEL["red"],
        edgecolor=TEL["ink"],
        lw=0.75,
        label=f"2D γ Blindspot / False Neg. (n={int(blind.sum())})",
        zorder=5,
    )

    ax1.text(
        97.8,
        y_max - 0.25,
        f"NELMS BLINDSPOT ZONE\n{int(blind.sum())}/{int(unacc.sum())} failing plans pass 2D 3%/2mm\n(PTV |ΔD95| ≥ 2% or OAR ΔD ≥ 1.5 Gy)",
        ha="center",
        va="top",
        fontsize=6.8,
        fontweight="bold",
        color=TEL["red_dark"],
        bbox={
            "boxstyle": "square,pad=0.30",
            "facecolor": "#FFFFFF",
            "edgecolor": TEL["red"],
            "linewidth": 1.0,
            "alpha": 0.95,
        },
    )
    _style_scope_axis(ax1, grid=True, minor_x=True, minor_y=True)
    _set_panel_header(ax1, "CH-A", "2D Planar Gamma (3%/2mm) vs 3D PTV |ΔD95|")
    ax1.set_xlabel("2D Planar Gamma Pass Rate — 3%/2mm (%)")
    ax1.set_ylabel("True 3D PTV70 |ΔD95| Error (%)")
    ax1.set_xlim(x_min, 100.8)
    ax1.set_ylim(-0.1, y_max)
    _style_legend(ax1, loc="lower left", fontsize=6.9)

    # Panel B: Pass-Rate Sensitivity vs DLG Drift (Dual Axis with |ΔD95|)
    ax2 = fig.add_subplot(gs[0, 1])
    l1, = ax2.plot(dlg_vals, p_g2_32, "o-", color=TEL["amber"], lw=2.0, ms=4.5, label="2D γ (3%/2mm) — Blind to DLG")
    l2, = ax2.plot(dlg_vals, p_g3_32, "s--", color=TEL["navy"], lw=1.85, ms=4.2, label="3D γ (3%/2mm)")
    l3, = ax2.plot(dlg_vals, p_g3_21, "^-", color=TEL["plum"], lw=2.1, ms=5.0, label="3D γ (2%/1mm) — Sensitive")
    ax2.axhline(95.0, color=TEL["graphite"], lw=1.2, ls=":", label="95% TG-218 Pass Threshold")
    _style_scope_axis(ax2, grid=True, minor_x=True, minor_y=True)
    ax2.tick_params(which="both", right=False)
    ax2.set_xlabel("Systematic DLG Calibration Offset (mm)")
    ax2.set_ylabel("Gamma Pass Rate (%)", color=TEL["navy"])
    ax2.set_ylim(44, 104)
    ax2.set_yticks([50, 60, 70, 80, 90, 95, 100])
    ax2.set_xlim(-1.12, 1.12)

    ax2_r = ax2.twinx()
    for spine in ax2_r.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(TEL["bezel"])
        spine.set_linewidth(1.05)
    l4, = ax2_r.plot(dlg_vals, p_abs_d95, "D-", color=TEL["red"], lw=2.0, ms=4.5, label="3D PTV70 |ΔD95| (%) [right]")
    ax2_r.axhline(2.0, color=TEL["red"], lw=1.0, ls="-.", alpha=0.65)
    ax2_r.set_ylabel("3D PTV70 |ΔD95| (%)", color=TEL["red"])
    ax2_r.tick_params(which="major", direction="in", right=True, color=TEL["red"], labelcolor=TEL["red"], length=4.5, width=0.9)
    ax2_r.tick_params(which="minor", direction="in", right=True, color=TEL["red"], length=2.4, width=0.65)
    ax2_r.set_ylim(-1.6, 4.2)
    ax2_r.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ax2_r.grid(False)

    lines = [l1, l2, l3, l4]
    _style_legend(ax2, lines, [ln.get_label() for ln in lines], loc="lower center", fontsize=6.9)
    _set_panel_header(ax2, "CH-B", "Why 2D 3%/2mm Masks Systematic DLG Errors")

    mean_g2_21 = float(np.mean(g2_21[blind])) if np.any(blind) else 100.0
    _add_telemetry_banner(
        fig,
        "PLANAR GAMMA BLINDSPOT DEMONSTRATION (NELMS ET AL. 2011) VS TRUE 3D ΔDVH",
        f"TEL-03 // MEAN 2D γ(2%/1mm) {mean_g2_21:.1f}%",
    )
    _save_fig(
        fig,
        "fig3_gamma_blindspot_vs_dvh",
        "PROVENANCE: OPEN_PUBLIC_ANATOMY_OPENKBP_PT1 + 32 Synthetic VMAT Fault Injections (AAPM TG-218) | Research Software — NOT A MEDICAL DEVICE",
    )


def make_fig4_causal_attribution(summary: dict) -> None:
    """Fig 4: Bayesian BIC Attribution Heatmap & Exact Multi-Fault Parameter Recovery."""
    suite = summary["causal_attribution_suite"]
    fault_cases = suite[1:]
    model_keys = [
        "MLC_LEAF_MOTOR_FRICTION",
        "SYSTEMATIC_DLG_OFFSET",
        "GANTRY_GRAVITY_SAG_HARMONIC",
        "DOSE_RATE_ION_CHAMBER_DRIFT",
        "JOINT_MULTI_FAULT",
    ]
    model_short = ["Leaf Drag\n(τ_i)", "DLG Offset\n(Δgap)", "Gravity Sag\n(A sin+B cos)", "MU Drift\n(ΔMU/MU)", "Joint\nMulti-Fault"]
    scen_labels = [c["scenario"] for c in fault_cases]

    mat = np.zeros((len(fault_cases), len(model_keys)))
    weights = np.zeros_like(mat)
    for r, c in enumerate(fault_cases):
        dbics = c["hypotheses_delta_bic"]
        pws = c["hypotheses_posterior_weight"]
        for col_i, mk in enumerate(model_keys):
            val = float(dbics.get(mk, 0.0))
            mat[r, col_i] = np.log10(max(1.0, val)) if val > 0.0 else 0.0
            weights[r, col_i] = float(pws.get(mk, 0.0))

    fig = plt.figure(figsize=(15.0, 5.2))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.25, 1.05], wspace=0.38, left=0.17, right=0.97, top=0.80, bottom=0.18
    )

    # Panel A: Bayesian BIC Attribution Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=max(4.5, float(mat.max())))
    for r in range(mat.shape[0]):
        for c_i in range(mat.shape[1]):
            dbic = float(fault_cases[r]["hypotheses_delta_bic"].get(model_keys[c_i], 0.0))
            pw = weights[r, c_i]
            txt_col = "#FFFFFF" if mat[r, c_i] > 2.5 else TEL["ink"]
            if model_keys[c_i] == "JOINT_MULTI_FAULT" and r < 4:
                label = "n/a"
                txt_col = TEL["graphite"]
            else:
                star = " ★" if pw > 0.5 else ""
                label = f"{dbic:,.0f}{star}" if dbic > 0 else "0"
            ax1.text(
                c_i,
                r,
                label,
                ha="center",
                va="center",
                fontsize=7.6,
                fontweight="bold" if pw > 0.5 else "normal",
                color=txt_col,
            )

    ax1.set_xticks(range(len(model_keys)))
    ax1.set_xticklabels(model_short, fontsize=7.6)
    ax1.set_yticks(range(len(scen_labels)))
    ax1.set_yticklabels(scen_labels, fontsize=7.8)
    _style_scope_axis(ax1, grid=False, minor_x=False, minor_y=False)
    # Subtle instrument cell dividers
    ax1.set_xticks(np.arange(-0.5, len(model_keys), 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(scen_labels), 1), minor=True)
    ax1.grid(True, which="minor", color=TEL["canvas"], linewidth=0.9)
    ax1.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    _set_panel_header(ax1, "CH-A", "Bayesian ΔBIC Evidence (★ = Posterior Winner)", title_fontsize=9.2)
    cb = fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.03)
    _style_colorbar(cb, "log₁₀(max(1, ΔBIC))")

    # Panel B: Parameter Recovery Fidelity on Joint 4-Fault Delivery
    ax2 = fig.add_subplot(gs[0, 1])
    joint = suite[-1]
    true_vals = [
        42.0,
        0.55,
        float(np.hypot(0.85, 0.45)),
        1.60,
    ]
    rec_vals = [
        float(joint["flagged_leaf"]["tau_ms"]),
        float(joint["estimated_dlg_mm"]),
        float(joint["estimated_sag_amp_mm"]),
        float(joint["estimated_mu_drift_pct"]),
    ]
    ci_half = [
        0.5 * (joint["flagged_leaf"]["ci95_ms"][1] - joint["flagged_leaf"]["ci95_ms"][0]),
        0.5 * (joint["estimated_dlg_ci95_mm"][1] - joint["estimated_dlg_ci95_mm"][0]),
        0.03,
        0.5 * (joint["estimated_mu_ci95_pct"][1] - joint["estimated_mu_ci95_pct"][0]),
    ]
    ratios = [100.0 * r / t for r, t in zip(rec_vals, true_vals, strict=True)]
    ratio_err = [100.0 * e / t for e, t in zip(ci_half, true_vals, strict=True)]
    param_labels = [
        f"Leaf #34 Drag\n({rec_vals[0]:.1f}/{true_vals[0]:.1f} ms)",
        f"DLG Offset\n({rec_vals[1]:+.3f}/{true_vals[1]:+.2f} mm)",
        f"Gantry Sag\n({rec_vals[2]:.3f}/{true_vals[2]:.3f} mm)",
        f"MU Drift\n({rec_vals[3]:+.2f}/{true_vals[3]:+.2f}%)",
    ]
    y_pos = np.arange(len(param_labels))
    bar_cols = [TEL["red"], TEL["navy"], TEL["teal"], TEL["amber"]]
    ax2.barh(
        y_pos,
        ratios,
        xerr=ratio_err,
        height=0.54,
        color=bar_cols,
        alpha=0.88,
        edgecolor=TEL["ink"],
        lw=0.9,
        capsize=4,
        error_kw={"ecolor": TEL["ink"], "lw": 1.0},
    )
    ax2.axvline(100.0, color=TEL["ink"], lw=1.5, ls="--", label="Exact Recovery (100%)")
    ax2.axvspan(95.0, 105.0, color=TEL["graphite"], alpha=0.14, label="±5% Accuracy Band")
    for idx, val in enumerate(ratios):
        ax2.text(
            val + max(ratio_err[idx], 0.8) + 1.0,
            idx,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=8.0,
            fontweight="bold",
            color=TEL["ink"],
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(param_labels, fontsize=7.6)
    ax2.set_ylim(4.2, -0.55)
    ax2.set_xlim(80, 115)
    _style_scope_axis(ax2, grid=True, minor_x=True, minor_y=False)
    _set_panel_header(
        ax2, "CH-B", f"Joint 4-Fault Recovery (Var. Expl. {joint['variance_explained_pct']:.1f}%)", title_fontsize=9.2
    )
    ax2.set_xlabel("Recovered / Injected Parameter Ratio (%)")
    _style_legend(ax2, loc="lower left", ncol=2, fontsize=6.9)

    _add_telemetry_banner(
        fig,
        "CAUSAL ELECTRO-MECHANICAL FAULT ATTRIBUTION — MATCHED-FILTER & BIC SELECTION",
        "TEL-04 // 5-MODE CAUSAL BIC",
    )
    _save_fig(
        fig,
        "fig4_causal_fault_attribution",
        "PROVENANCE: SYNTHETIC_FAULT_INJECTION_BENCHMARK (6 controlled ground-truth scenarios, σ_noise = 0.025 mm) | Research Software — NOT A MEDICAL DEVICE",
    )


def make_fig5_tg142_fleet_triage() -> None:
    """Fig 5: 30-Day Multi-Linac TG-142/MPC Drift Control Chart + Clinical Patient Triage Matrix."""
    anatomy = load_openkbp_patient(OPENKBP_PT1, verify_manifest=True)
    ledger = run_fleet_triage_investigation(anatomy=anatomy, machine_id="TrueBeam-01", incident_id="INC-TB01-WK47")

    fig = plt.figure(figsize=(15.0, 5.25))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.16, 1.14], wspace=0.28, left=0.06, right=0.97, top=0.80, bottom=0.165
    )

    # Panel A: 30-Day Multi-Linac TG-142 / MPC Control Chart
    ax1 = fig.add_subplot(gs[0, 0])
    styles = [
        ("TrueBeam-01", TEL["red"], "o-", 2.2, "TrueBeam-01 (Failing MLC/DLG)"),
        ("TrueBeam-02", TEL["teal"], "s-", 1.75, "TrueBeam-02 (Stable Nominal)"),
        ("TrueBeam-03", TEL["navy"], "^--", 1.75, "TrueBeam-03 (Mild Output Drift)"),
    ]
    tb01 = ledger.tg142_trends[0]
    for trend, (_, col, fmt, lw, lbl) in zip(ledger.tg142_trends, styles, strict=True):
        ax1.plot(trend.days, trend.mlc_max_rms_mm, fmt, color=col, lw=lw, ms=4.2, label=lbl)

    ew_lim = TG142_LIMITS["mlc_rms_early_warning_mm"]
    act_lim = TG142_LIMITS["mlc_rms_action_mm"]
    ax1.axhspan(ew_lim, act_lim, color=TEL["amber"], alpha=0.13, label="Early Predictive Warning Zone")
    ax1.axhline(ew_lim, color=TEL["amber"], lw=1.35, ls="--")
    ax1.axhline(act_lim, color=TEL["red"], lw=1.55, ls="-.", label=f"AAPM TG-142 Action ({act_lim:.2f} mm)")

    if tb01.early_warning_day is not None:
        ax1.axvline(tb01.early_warning_day, color=TEL["amber"], lw=1.3, ls=":")
        ax1.annotate(
            f"EARLY TRIGGER\nDay {tb01.early_warning_day}",
            xy=(tb01.early_warning_day, ew_lim),
            xytext=(tb01.early_warning_day - 7.0, 0.56),
            fontsize=7.3,
            fontweight="bold",
            color=TEL["amber_dark"],
            bbox={
                "boxstyle": "square,pad=0.26",
                "facecolor": "#FFFFFF",
                "edgecolor": TEL["amber"],
                "linewidth": 1.0,
                "alpha": 0.95,
            },
            arrowprops={"arrowstyle": "->", "color": TEL["amber"], "lw": 1.3},
        )
    if tb01.action_level_breach_day is not None:
        ax1.axvline(tb01.action_level_breach_day, color=TEL["red"], lw=1.3, ls=":")
        ax1.annotate(
            f"TG-142 BREACH\nDay {tb01.action_level_breach_day}",
            xy=(tb01.action_level_breach_day, act_lim),
            xytext=(tb01.action_level_breach_day - 7.0, 0.88),
            fontsize=7.3,
            fontweight="bold",
            color=TEL["red_dark"],
            bbox={
                "boxstyle": "square,pad=0.26",
                "facecolor": "#FFFFFF",
                "edgecolor": TEL["red"],
                "linewidth": 1.0,
                "alpha": 0.95,
            },
            arrowprops={"arrowstyle": "->", "color": TEL["red"], "lw": 1.3},
        )

    _style_scope_axis(ax1, grid=True, minor_x=True, minor_y=True)
    _set_panel_header(ax1, "CH-A", "30-Day Multi-Linac TG-142 / MPC Drift Chart")
    ax1.set_xlabel("Treatment Calendar Day")
    ax1.set_ylabel("Max MLC Leaf RMS Error (mm)")
    ax1.set_xlim(1, 30.5)
    ax1.set_ylim(0.0, 1.12)
    _style_legend(ax1, loc="upper left", fontsize=6.8)

    # Panel B: Prioritized Clinical Patient-Impact Triage Risk Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    prio_colors = {
        "CRITICAL": TEL["red"],
        "HIGH": TEL["amber"],
        "MODERATE": TEL["navy"],
        "ROUTINE": TEL["teal"],
    }
    ax2.axvline(2.0, color=TEL["red"], lw=1.25, ls=":", alpha=0.85)
    ax2.axhline(1.5, color=TEL["red"], lw=1.25, ls=":", alpha=0.85)
    ax2.add_patch(Rectangle((2.0, -1.6), 1.7, 4.35, facecolor=TEL["red"], alpha=0.07))
    ax2.add_patch(Rectangle((0.0, 1.5), 2.0, 1.25, facecolor=TEL["red"], alpha=0.07))

    label_offsets = {
        "PT-HN-002": (-12, 12),
        "PT-HN-001": (-16, 12),
        "PT-LG-005": (16, -2),
        "PT-PR-004": (-14, -18),
        "PT-SBRT-003": (-14, 14),
        "PT-BR-006": (14, -14),
    }
    for rec in ledger.triage_queue:
        x_val = abs(rec.delta_d95_ptv_pct)
        y_val = rec.delta_d003cc_oar_gy
        sz = 55 + 40 * rec.fractions_affected
        col = prio_colors[rec.triage_priority]
        marker = "*" if rec.gamma_blindspot else "o"
        ax2.scatter(
            x_val,
            y_val,
            s=sz * (1.7 if rec.gamma_blindspot else 1.0),
            c=col,
            marker=marker,
            edgecolor=TEL["ink"],
            lw=0.9,
            zorder=5,
        )
        dx, dy = label_offsets.get(rec.patient_id, (10, 10))
        ax2.annotate(
            f"{rec.patient_id} ({rec.clinical_risk_score:.0f})",
            xy=(x_val, y_val),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.0,
            fontweight="bold",
            color=TEL["ink"],
            ha="left" if dx >= 0 else "right",
            va="center",
            arrowprops={"arrowstyle": "-", "color": TEL["graphite"], "lw": 0.85},
            bbox={
                "boxstyle": "square,pad=0.22",
                "facecolor": "#FFFFFF",
                "edgecolor": col,
                "linewidth": 0.95,
                "alpha": 0.95,
            },
        )

    handles = [
        Patch(facecolor=TEL["red"], edgecolor=TEL["ink"], label="CRITICAL"),
        Patch(facecolor=TEL["amber"], edgecolor=TEL["ink"], label="HIGH"),
        Patch(facecolor=TEL["navy"], edgecolor=TEL["ink"], label="MODERATE"),
        Patch(facecolor=TEL["teal"], edgecolor=TEL["ink"], label="ROUTINE"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=TEL["red"],
            markeredgecolor=TEL["ink"],
            markersize=10,
            label="★ = 2D γ Blindspot",
        ),
    ]
    _style_scope_axis(ax2, grid=True, minor_x=True, minor_y=True)
    _set_panel_header(ax2, "CH-B", "Autonomous Fleet Physicist — Triage Risk Matrix")
    _style_legend(ax2, handles=handles, loc="lower right", fontsize=6.9, ncol=2)
    ax2.set_xlabel("PTV70 |ΔD95| Shift (%)")
    ax2.set_ylabel("Critical OAR ΔD0.03cc (Gy)")
    ax2.set_xlim(0.0, 3.6)
    ax2.set_ylim(-1.6, 2.7)

    _add_telemetry_banner(
        fig,
        "CLOSED-LOOP FLEET SURVEILLANCE & PATIENT-IMPACT TRIAGE",
        f"TEL-05 // CRITIC GATE PASS {ledger.critic_gate.checks_passed}/{ledger.critic_gate.checks_run}",
    )
    _save_fig(
        fig,
        "fig5_tg142_fleet_triage_timeline",
        "PROVENANCE: SYNTHETIC_TG142_FLEET_DRIFT + OPEN_PUBLIC_ANATOMY_OPENKBP_PT1 | Research Software — NOT A MEDICAL DEVICE",
    )


def make_fig6_sobol_sensitivity(summary: dict) -> None:
    """Fig 6: Saltelli-Jansen Global Sobol' Sensitivity (S_1 vs S_T) for PTV70 & SpinalCord."""
    reports = summary["sobol_sensitivity"]["reports"]
    r_ptv = reports["PTV70_delta_D95_pct"]
    r_cord = reports["SpinalCord_delta_D003cc_Gy"]

    pretty_names = {
        "mlc_leaf_lag_ms": "MLC Viscous Lag\n(0–60 ms)",
        "dlg_offset_mm": "DLG Drift\n(±0.8 mm)",
        "gantry_sag_amp_mm": "Gantry Sag\n(0–2.0 mm)",
        "mu_output_drift_pct": "MU Output Drift\n(±3.0%)",
    }

    fig = plt.figure(figsize=(14.4, 5.15))
    gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.065, right=0.97, top=0.80, bottom=0.195)

    panels = [
        ("CH-A", r_ptv, "PTV70 ΔD95 (%) Sensitivity"),
        ("CH-B", r_cord, "SpinalCord ΔD0.03cc (Gy) Sensitivity"),
    ]
    for idx, (channel, rep, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        params = rep["parameter_names"]
        x = np.arange(len(params))
        w = 0.34

        s1 = np.array([rep["first_order_s1"][p] for p in params])
        st = np.array([rep["total_order_st"][p] for p in params])
        s1_err = np.array(
            [
                [s1[i] - rep["s1_ci95"][p][0] for i, p in enumerate(params)],
                [rep["s1_ci95"][p][1] - s1[i] for i, p in enumerate(params)],
            ]
        )
        st_err = np.array(
            [
                [st[i] - rep["st_ci95"][p][0] for i, p in enumerate(params)],
                [rep["st_ci95"][p][1] - st[i] for i, p in enumerate(params)],
            ]
        )
        s1_err = np.maximum(0.0, s1_err)
        st_err = np.maximum(0.0, st_err)

        b1 = ax.bar(
            x - w / 2,
            s1,
            width=w,
            yerr=s1_err,
            capsize=3.5,
            color=TEL["navy"],
            edgecolor=TEL["ink"],
            lw=0.85,
            error_kw={"ecolor": TEL["ink"], "lw": 0.95},
            label=r"First-order $S_1$ (main effect)",
        )
        b2 = ax.bar(
            x + w / 2,
            st,
            width=w,
            yerr=st_err,
            capsize=3.5,
            color=TEL["amber"],
            edgecolor=TEL["ink"],
            lw=0.85,
            error_kw={"ecolor": TEL["ink"], "lw": 0.95},
            label=r"Total-order $S_T$ (incl. interactions)",
        )

        for bars, errs in ((b1, s1_err[1]), (b2, st_err[1])):
            for bar, e_hi in zip(bars, errs, strict=True):
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + e_hi + 0.018,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.4,
                    fontweight="bold",
                    color=TEL["ink"],
                )

        ax.set_xticks(x)
        ax.set_xticklabels([pretty_names[p] for p in params], fontsize=7.6)
        ax.set_ylabel("Sobol' Variance Sensitivity Index")
        ax.set_ylim(0.0, min(1.08, max(0.75, float(st.max()) + 0.18)))
        _style_scope_axis(ax, grid=True, minor_x=False, minor_y=True)
        _set_panel_header(ax, channel, f"{title} (N={rep['total_evaluations']:,})")
        _style_legend(ax, loc="upper left", fontsize=7.2)

    _add_telemetry_banner(
        fig,
        "GLOBAL SALTELLI–JANSEN QMC SOBOL' SENSITIVITY OF 3D DOSE ERRORS TO LINAC TOLERANCES",
        "TEL-06 // QMC SOBOL N=3,072",
    )
    _save_fig(
        fig,
        "fig6_sobol_delivery_sensitivity",
        "PROVENANCE: Saltelli (2010) QMC Sobol' calibrated on OPEN_PUBLIC_ANATOMY_OPENKBP_PT1 (95% bootstrap CIs) | Research Software — NOT A MEDICAL DEVICE",
    )


def make_fig7_epid_stiction_kalman_rul() -> None:
    """Fig 7: Three-Way EPID Silent Fault Map, Viscous vs Coulomb Stiction, and Kalman RUL Forecast."""
    fig = plt.figure(figsize=(15.4, 5.2))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.10, 1.08, 1.22], wspace=0.33, left=0.055, right=0.97, top=0.80, bottom=0.165
    )

    # Panel A: Three-Way EPID minus Trajectory-Log Discrepancy catching Encoder-Blind Silent Fault
    ax1 = fig.add_subplot(gs[0, 0])
    log_nom = generate_synthetic_vmat_log(log_id="FIG7_SILENT", n_snapshots=180)
    _, _, epid_img = simulate_measured_epid_image(
        log_nom, silent_dlg_bias_mm=+1.6, silent_leaf_slip_mm=3.5, silent_leaf_index=34, noise_rel_sigma=0.0015
    )
    arr = analyze_three_way_epid_forensics(log_nom, epid_img, return_arrays=True)
    assert isinstance(arr, EPIDThreeWayArrays)

    vlim = max(5.0, float(np.percentile(np.abs(arr.delta_epid_minus_log_pct), 99.5)))
    im1 = ax1.imshow(
        arr.delta_epid_minus_log_pct,
        origin="lower",
        extent=[float(arr.u_mm[0]), float(arr.u_mm[-1]), float(arr.v_mm[0]), float(arr.v_mm[-1])],
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        aspect="equal",
    )
    c1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    _style_colorbar(c1, "EPID − Log Fluence (%)")
    ax1.annotate(
        "ENCODER-BLIND SLIP\nLeaf #34 (+1.6 mm DLG)",
        xy=(-18, 17),
        xytext=(-75, 47),
        fontsize=6.7,
        fontweight="bold",
        color=TEL["red_dark"],
        bbox={
            "boxstyle": "square,pad=0.26",
            "facecolor": "#FFFFFF",
            "edgecolor": TEL["red"],
            "linewidth": 1.0,
            "alpha": 0.95,
        },
        arrowprops={"arrowstyle": "->", "color": TEL["red"], "lw": 1.3},
    )
    ax1.text(
        0.03,
        0.03,
        f"Plan vs Log γ: {arr.report.gamma_plan_vs_log_2pct_1mm:.0f}% (blind)\n"
        f"Log vs EPID γ: {arr.report.gamma_log_vs_epid_2pct_1mm:.1f}% (caught)",
        transform=ax1.transAxes,
        fontsize=6.7,
        va="bottom",
        color=TEL["ink"],
        bbox={
            "boxstyle": "square,pad=0.26",
            "facecolor": TEL["callout_bg"],
            "edgecolor": TEL["bezel"],
            "linewidth": 0.9,
            "alpha": 0.95,
        },
    )
    _style_scope_axis(ax1, grid=False, minor_x=False, minor_y=False)
    _set_panel_header(ax1, "CH-A", "Three-Way EPID Silent Fault Map", title_fontsize=9.0)
    ax1.set_xlabel("Cross-Plane u (mm)")
    ax1.set_ylabel("In-Plane v (mm)")

    # Panel B: Nonlinear Coulomb/Stribeck Stiction vs Linear Viscous Drag Phase Plane
    ax2 = fig.add_subplot(gs[0, 1])
    log_v = generate_synthetic_vmat_log(
        log_id="FIG7_VISC", n_snapshots=240, failing_bank="B", failing_leaf_index=34, leaf_drag_ms=32.0, noise_sigma_mm=0.025, seed=10
    )
    log_s = generate_synthetic_vmat_log(
        log_id="FIG7_STIC", n_snapshots=240, failing_bank="B", failing_leaf_index=34, leaf_stiction_mm=0.75, noise_sigma_mm=0.025, seed=11
    )
    dt_s = log_v.sampling_interval_ms / 1000.0
    v_vel = np.gradient(np.asarray(log_v.bank_b_expected_mm)[:, 33], dt_s)
    e_visc = np.asarray(log_v.bank_b_actual_mm)[:, 33] - np.asarray(log_v.bank_b_expected_mm)[:, 33]
    e_stic = np.asarray(log_s.bank_b_actual_mm)[:, 33] - np.asarray(log_s.bank_b_expected_mm)[:, 33]
    d_stic = diagnose_causal_faults(log_s)
    beta_hat = d_stic.flagged_leaves[0].stiction_amplitude_mm

    ax2.scatter(v_vel, e_visc, s=15, alpha=0.50, color=TEL["navy"], label="Viscous Drag (τ=32 ms, linear)")
    ax2.scatter(v_vel, e_stic, s=17, alpha=0.70, color=TEL["red"], label=f"Coulomb Stiction (β̂={beta_hat:.2f} mm)")
    v_grid = np.linspace(float(v_vel.min()), float(v_vel.max()), 200)
    ax2.plot(v_grid, -beta_hat * np.tanh(v_grid / 5.0), color=TEL["ink"], lw=2.1, label="Stribeck fit: −β tanh(v/v₀)")
    ax2.axhline(0.0, color=TEL["graphite"], lw=0.9, ls=":")
    ax2.axvline(0.0, color=TEL["graphite"], lw=0.9, ls=":")
    _style_scope_axis(ax2, grid=True, minor_x=True, minor_y=True)
    _set_panel_header(ax2, "CH-B", "Viscous Drag vs Coulomb Stiction", title_fontsize=9.0)
    ax2.set_xlabel("Planned Leaf Velocity v(t) (mm/s)")
    ax2.set_ylabel("Position Residual Δx(t) (mm)")
    _style_legend(ax2, loc="upper right", fontsize=6.7)

    # Panel C: Kalman State-Space RUL Forecast on TrueBeam-01
    ax3 = fig.add_subplot(gs[0, 2])
    trends = generate_tg142_fleet_trends()
    tb01 = trends[0]
    cutoff = 20
    fc = forecast_tg142_kalman_rul(
        tb01, metric_name="mlc_max_rms_mm", action_limit=0.70, observation_cutoff_day=cutoff, horizon_days=10
    )

    days_all = np.asarray(tb01.days)
    vals_all = np.asarray(tb01.mlc_max_rms_mm)
    obs_m = days_all <= cutoff
    fut_m = days_all > cutoff

    ax3.plot(days_all[obs_m], vals_all[obs_m], "o-", color=TEL["navy"], lw=1.85, ms=4.0, label="Observed MPC (Days 1–20)")
    ax3.plot(days_all[fut_m], vals_all[fut_m], "s:", color=TEL["graphite"], lw=1.45, ms=4.0, alpha=0.88, label="Held-out Actual (Days 21–30)")

    f_days = [cutoff, *fc.forecast_days]
    f_mean = [float(vals_all[obs_m][-1]), *fc.forecast_mean]
    f_lo = [float(vals_all[obs_m][-1]), *fc.forecast_ci95_low]
    f_hi = [float(vals_all[obs_m][-1]), *fc.forecast_ci95_high]

    ax3.plot(f_days, f_mean, "-", color=TEL["red"], lw=2.2, label=f"Kalman Forecast (RUL={fc.rul_mean_days:.1f} d)")
    ax3.fill_between(f_days, f_lo, f_hi, color=TEL["red"], alpha=0.16, label="95% Credible Envelope")
    ax3.axhline(0.70, color=TEL["amber"], lw=1.55, ls="--", label="TG-142 Action (0.70 mm)")
    ax3.axvline(cutoff, color=TEL["ink"], lw=1.15, ls=":")
    ax3.text(
        0.04,
        0.54,
        f"CUTOFF: Day {cutoff}\nRUL: {fc.rul_mean_days:.1f} d (95% CI {fc.rul_ci95_days[0]:.1f}–{fc.rul_ci95_days[1]:.1f} d)\n"
        f"10d Breach Prob: {fc.breach_probability_within_horizon * 100:.0f}%",
        transform=ax3.transAxes,
        fontsize=6.7,
        color=TEL["ink"],
        bbox={
            "boxstyle": "square,pad=0.28",
            "facecolor": "#FFFFFF",
            "edgecolor": TEL["red"],
            "linewidth": 0.95,
            "alpha": 0.95,
        },
    )
    _style_scope_axis(ax3, grid=True, minor_x=True, minor_y=True)
    _set_panel_header(ax3, "CH-C", "Kalman State-Space RUL Forecast", title_fontsize=9.0)
    ax3.set_xlabel("Treatment Calendar Day")
    ax3.set_ylabel("Max MLC Leaf RMS Error (mm)")
    ax3.set_xlim(1, 30.5)
    ax3.set_ylim(0.0, 1.28)
    _style_legend(ax3, loc="upper left", fontsize=6.6)

    _add_telemetry_banner(
        fig,
        "CLOSED-LOOP EXTENSIONS: THREE-WAY EPID SILENT FAULTS, STICTION & KALMAN RUL",
        "TEL-07 // EPID + STICTION + RUL",
    )
    _save_fig(
        fig,
        "fig7_epid_threeway_and_kalman_rul",
        "PROVENANCE: EPID_THREE_WAY_FORENSICS + SYNTHETIC_FAULT_INJECTION_BENCHMARK + SYNTHETIC_TG142_FLEET_DRIFT | Research Software — NOT A MEDICAL DEVICE",
    )


def make_fig8_delivery_robustness_and_routing() -> None:
    """Fig 8: Inverse Plan Tolerance Budgets, Fleet-Safe Routing Matrix & Adversarial Ellipsoid."""
    cohort = load_openkbp_cohort(REPO_ROOT / "data" / "openkbp", verify_manifest=True)
    routing = compute_fleet_routing_matrix(cohort)

    fig = plt.figure(figsize=(15.6, 5.35))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.12, 1.15, 1.20], wspace=0.35, left=0.105, right=0.975, top=0.80, bottom=0.175
    )

    # Panel A: Plan-Specific Inverse DLG Tolerance Budget vs Static TG-142
    ax1 = fig.add_subplot(gs[0, 0])
    budgets = routing.budgets
    p_labels = [f"{b.patient_id}\n({b.treatment_site.split('(')[0].strip()})" for b in budgets]
    dlg_lims = [b.max_safe_tolerances["dlg_offset_mm"] for b in budgets]
    tg_dlg = STATIC_TG142_LIMITS["dlg_offset_mm"]
    tb01_dlg = abs(routing.fingerprints[0].mean_vector["dlg_offset_mm"])

    y_pos = np.arange(len(budgets))
    bar_cols = [TEL["red"] if v < tg_dlg else TEL["teal"] for v in dlg_lims]
    ax1.barh(y_pos, dlg_lims, height=0.58, color=bar_cols, alpha=0.88, edgecolor=TEL["ink"], lw=0.85)
    ax1.axvline(tg_dlg, color=TEL["ink"], lw=1.6, ls="--", label=f"Static TG-142 ({tg_dlg:.2f} mm)")
    ax1.axvline(tb01_dlg, color=TEL["amber"], lw=1.6, ls=":", label=f"Live TrueBeam-01 ({tb01_dlg:.2f} mm)")
    for idx, val in enumerate(dlg_lims):
        ax1.text(val + 0.05, idx, f"{val:.2f} mm", va="center", ha="left", fontsize=7.4, fontweight="bold", color=TEL["ink"])

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(p_labels, fontsize=7.2)
    ax1.invert_yaxis()
    ax1.set_xlim(0.0, max(dlg_lims) + 0.65)
    _style_scope_axis(ax1, grid=True, minor_x=True, minor_y=False)
    _set_panel_header(ax1, "CH-A", "Inverse Plan DLG Budgets vs TG-142", title_fontsize=8.9)
    ax1.set_xlabel("Max Safe |ΔDLG| Budget (mm)")
    _style_legend(ax1, loc="upper right", fontsize=6.7)

    # Panel B: Fleet-Wide Linac-to-Patient Safe-Routing Matrix (All 3 Linacs Passing TG-142)
    ax2 = fig.add_subplot(gs[0, 1])
    mids = routing.machine_ids
    pids = routing.patient_ids
    mat = np.array([[routing.safety_margin_matrix[p][m] for m in mids] for p in pids], dtype=np.float64)
    recs_by_p = {r.patient_id: r for r in routing.routing_recommendations}

    # Diverging colorblind-safe RdBu centered exactly at the 1.0x safety boundary
    norm = TwoSlopeNorm(vmin=0.1, vcenter=1.0, vmax=2.6)
    im = ax2.imshow(np.clip(mat, 0.1, 2.6), cmap="RdBu", norm=norm, aspect="auto")
    for r_i, pid in enumerate(pids):
        rec = recs_by_p[pid]
        for c_i, mid in enumerate(mids):
            val = mat[r_i, c_i]
            is_best = rec.recommended_machine_id == mid
            tag = ""
            if val < 1.0:
                tag = "\nSTEALTH"
            elif is_best and rec.transfer_required:
                tag = "\n★ MOVE"
            elif is_best:
                tag = "\n★ KEEP"
            txt_col = "#FFFFFF" if (val < 0.55 or val > 1.85) else TEL["ink"]
            ax2.text(
                c_i,
                r_i,
                f"{val:.2f}×{tag}",
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold" if (val < 1.0 or is_best) else "normal",
                color=txt_col,
            )

    ax2.set_xticks(range(len(mids)))
    ax2.set_xticklabels([f"{m}\n(PASS TG-142)" for m in mids], fontsize=7.1)
    ax2.set_yticks(range(len(pids)))
    ax2.set_yticklabels(pids, fontsize=7.4)
    _style_scope_axis(ax2, grid=False, minor_x=False, minor_y=False)
    ax2.set_xticks(np.arange(-0.5, len(mids), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(pids), 1), minor=True)
    ax2.grid(True, which="minor", color=TEL["canvas"], linewidth=0.9)
    ax2.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    _set_panel_header(ax2, "CH-B", f"Fleet Routing ({routing.n_stealth_tg142_risks} Stealth Risks)", title_fontsize=8.9)
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    _style_colorbar(cb, "Worst-Case 95% Safety Margin (×)")

    # Panel C: 2D Failure Polytope vs TG-142 Box vs 95% Linac Ellipsoid (ΔDLG vs MU Drift)
    ax3 = fig.add_subplot(gs[0, 2])
    fp1, fp2 = routing.fingerprints[0], routing.fingerprints[1]
    b_hn = routing.budgets[0]  # PT-HN-001
    b_pr = routing.budgets[4]  # PT-PR-005

    ax3.add_patch(
        Rectangle(
            (-0.50, -2.0),
            1.00,
            4.00,
            facecolor=TEL["canvas"],
            edgecolor=TEL["graphite"],
            lw=1.4,
            ls="--",
            label="Static TG-142 Pass Box",
        )
    )

    dlg_x = np.linspace(-0.62, 0.72, 200)
    for b_obj, col, lbl in (
        (b_hn, TEL["red"], "H&N SIB Clinical Limit (|ΔD95|=2%)"),
        (b_pr, TEL["teal"], "Prostate Clinical Limit (|ΔD95|=2%)"),
    ):
        jd = b_obj.ptv_jacobian["dlg_offset_mm"]
        jm = max(abs(b_obj.ptv_jacobian["mu_drift_pct"]), 1e-3) * np.sign(
            b_obj.ptv_jacobian["mu_drift_pct"] or 1.0
        )
        mu_hi = (b_obj.ptv_d95_limit_pct - jd * dlg_x) / jm
        ax3.plot(dlg_x, mu_hi, color=col, lw=2.05, ls="-", label=lbl)

    scale_95 = 2.2414
    for fp, col, lbl in (
        (fp1, TEL["amber"], "TrueBeam-01 95% Ellipsoid"),
        (fp2, TEL["navy"], "TrueBeam-02 95% Ellipsoid"),
    ):
        cx, cy = fp.mean_vector["dlg_offset_mm"], fp.mean_vector["mu_drift_pct"]
        rx = scale_95 * fp.std_vector["dlg_offset_mm"]
        ry = scale_95 * fp.std_vector["mu_drift_pct"]
        ax3.add_patch(
            Ellipse(
                (cx, cy),
                width=2.0 * rx,
                height=2.0 * ry,
                facecolor=col,
                alpha=0.25,
                edgecolor=col,
                lw=1.8,
                label=lbl,
            )
        )
        ax3.scatter([cx], [cy], color=col, edgecolor=TEL["ink"], s=36, zorder=6)

    cert_hn_tb01 = next(
        c for c in routing.certificates if c.patient_id == "PT-HN-001" and c.machine_id == "TrueBeam-01"
    )
    ax_adv = cert_hn_tb01.adversarial_fault_vector["dlg_offset_mm"]
    ay_adv = cert_hn_tb01.adversarial_fault_vector["mu_drift_pct"]
    ax3.scatter(
        [ax_adv],
        [ay_adv],
        marker="*",
        s=150,
        color=TEL["red"],
        edgecolor=TEL["ink"],
        lw=0.85,
        zorder=8,
        label="Adversarial θ*_adv (Breach)",
    )
    ax3.annotate(
        "PASSES TG-142,\nBREACHES H&N!",
        xy=(ax_adv, ay_adv),
        xytext=(-0.48, 1.40),
        fontsize=6.9,
        fontweight="bold",
        color=TEL["red_dark"],
        bbox={
            "boxstyle": "square,pad=0.26",
            "facecolor": "#FFFFFF",
            "edgecolor": TEL["red"],
            "linewidth": 1.0,
            "alpha": 0.95,
        },
        arrowprops={"arrowstyle": "->", "color": TEL["red"], "lw": 1.3, "connectionstyle": "arc3,rad=-0.12"},
    )

    ax3.set_xlim(-0.56, 0.68)
    ax3.set_ylim(-2.25, 3.35)
    _style_scope_axis(ax3, grid=True, minor_x=True, minor_y=True)
    _set_panel_header(ax3, "CH-C", "Ellipsoidal Delivery Certificate", title_fontsize=8.9)
    ax3.set_xlabel("DLG Calibration Offset ΔDLG (mm)")
    ax3.set_ylabel("Ion-Chamber MU Drift (%)")
    _style_legend(ax3, loc="lower left", fontsize=6.4)

    _add_telemetry_banner(
        fig,
        "PLAN-SPECIFIC INVERSE TOLERANCE BUDGETS & FLEET-SAFE LINAC ROUTING",
        "TEL-08 // TG-100 · TG-142 · TG-218",
    )
    _save_fig(
        fig,
        "fig8_delivery_robustness_and_routing",
        "PROVENANCE: DELIVERY_ROBUSTNESS_CERTIFICATE on OPEN_PUBLIC_ANATOMY_OPENKBP (pt_1–pt_3) × 3 Live Linac Fingerprints | Research Software — NOT A MEDICAL DEVICE",
    )


def main() -> None:
    _configure_style()
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run scripts/run_benchmarks.py first to create {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    make_fig1_trajectory_telemetry()
    make_fig2_fluence_to_3d_dose()
    make_fig3_gamma_blindspot(summary)
    make_fig4_causal_attribution(summary)
    make_fig5_tg142_fleet_triage()
    make_fig6_sobol_sensitivity(summary)
    make_fig7_epid_stiction_kalman_rul()
    make_fig8_delivery_robustness_and_routing()
    print(f"Generated all 8 publication-grade 300-DPI PNG + PDF figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
