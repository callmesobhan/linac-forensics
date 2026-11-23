"""Tests for Physics-Informed Electro-Mechanical Fault Attribution & Bayesian BIC Model Selection."""

from __future__ import annotations

import numpy as np

from linac_forensics.causal_fault_model import diagnose_causal_faults
from linac_forensics.log_parser import generate_synthetic_vmat_log


def test_nominal_zero_fault_selects_null_model() -> None:
    """Zero-fault log must select NULL_NOMINAL with zero active faults."""
    log = generate_synthetic_vmat_log(log_id="NOMINAL_CLEAN", n_snapshots=180, noise_sigma_mm=0.01)
    diag = diagnose_causal_faults(log)
    assert diag.primary_root_cause == "NULL_NOMINAL"
    assert diag.active_faults == []
    assert diag.flagged_leaves == []
    assert abs(diag.estimated_dlg_offset_mm) < 0.05


def test_single_fault_mlc_leaf_motor_friction_recovery() -> None:
    """Isolate exact Bank B Leaf #34 viscous drag tau=42 ms within tight error bounds."""
    log = generate_synthetic_vmat_log(
        log_id="FAULT_LEAF34",
        n_snapshots=240,
        failing_bank="B",
        failing_leaf_index=34,
        leaf_drag_ms=42.0,
        noise_sigma_mm=0.02,
        seed=7,
    )
    diag = diagnose_causal_faults(log)
    assert diag.primary_root_cause == "MLC_LEAF_MOTOR_FRICTION"
    assert diag.active_faults == ["MLC_LEAF_MOTOR_FRICTION"]
    assert len(diag.flagged_leaves) == 1
    leaf = diag.flagged_leaves[0]
    assert leaf.bank == "B"
    assert leaf.leaf_index == 34
    assert abs(leaf.velocity_lag_ms - 42.0) < 2.0
    assert leaf.velocity_lag_ci95_ms[0] <= 42.0 <= leaf.velocity_lag_ci95_ms[1]
    assert leaf.velocity_correlation_r > 0.95


def test_single_fault_systematic_dlg_offset_recovery() -> None:
    """Recover systematic DLG offset (+0.50 mm) without misattributing to leaf friction."""
    log = generate_synthetic_vmat_log(
        log_id="FAULT_DLG",
        n_snapshots=240,
        dlg_offset_mm=+0.50,
        noise_sigma_mm=0.025,
        seed=11,
    )
    diag = diagnose_causal_faults(log)
    assert diag.primary_root_cause == "SYSTEMATIC_DLG_OFFSET"
    assert diag.active_faults == ["SYSTEMATIC_DLG_OFFSET"]
    assert abs(diag.estimated_dlg_offset_mm - 0.50) < 0.02
    assert diag.estimated_dlg_ci95_mm[0] <= 0.50 <= diag.estimated_dlg_ci95_mm[1]
    assert diag.flagged_leaves == []


def test_single_fault_gantry_gravity_sag_harmonic_recovery() -> None:
    """Recover exact gantry gravity-sag harmonic amplitude and phase across 360° VMAT arc."""
    a_true, b_true = 0.85, -0.45
    amp_true = float(np.hypot(a_true, b_true))
    phase_true = float(np.degrees(np.arctan2(b_true, a_true)))

    log = generate_synthetic_vmat_log(
        log_id="FAULT_SAG",
        n_snapshots=240,
        gantry_sag_sin_mm=a_true,
        gantry_sag_cos_mm=b_true,
        noise_sigma_mm=0.02,
        seed=19,
    )
    diag = diagnose_causal_faults(log)
    assert diag.primary_root_cause == "GANTRY_GRAVITY_SAG_HARMONIC"
    assert diag.active_faults == ["GANTRY_GRAVITY_SAG_HARMONIC"]
    assert abs(diag.estimated_gantry_sag_amp_mm - amp_true) < 0.04
    assert abs(diag.estimated_gantry_sag_phase_deg - phase_true) < 3.0


def test_single_fault_mu_ion_chamber_drift_recovery() -> None:
    """Recover multiplicative MU output calibration drift (-2.40%)."""
    log = generate_synthetic_vmat_log(
        log_id="FAULT_MU",
        n_snapshots=240,
        mu_drift_pct=-2.40,
        noise_sigma_mm=0.015,
        seed=23,
    )
    diag = diagnose_causal_faults(log)
    assert diag.primary_root_cause == "DOSE_RATE_ION_CHAMBER_DRIFT"
    assert diag.active_faults == ["DOSE_RATE_ION_CHAMBER_DRIFT"]
    assert abs(diag.estimated_mu_drift_pct - (-2.40)) < 0.08
    assert diag.estimated_mu_drift_ci95_pct[0] <= -2.40 <= diag.estimated_mu_drift_ci95_pct[1]


def test_joint_multi_fault_simultaneous_attribution() -> None:
    """Decompose 4 simultaneous faults cleanly via BIC model selection."""
    log = generate_synthetic_vmat_log(
        log_id="FAULT_JOINT_4X",
        n_snapshots=300,
        failing_bank="A",
        failing_leaf_index=27,
        leaf_drag_ms=38.0,
        dlg_offset_mm=-0.42,
        gantry_sag_sin_mm=0.75,
        gantry_sag_cos_mm=0.35,
        mu_drift_pct=+1.95,
        noise_sigma_mm=0.025,
        seed=31,
    )
    diag = diagnose_causal_faults(log)
    assert diag.primary_root_cause == "JOINT_MULTI_FAULT"
    assert set(diag.active_faults) == {
        "MLC_LEAF_MOTOR_FRICTION",
        "SYSTEMATIC_DLG_OFFSET",
        "GANTRY_GRAVITY_SAG_HARMONIC",
        "DOSE_RATE_ION_CHAMBER_DRIFT",
    }
    assert diag.flagged_leaves[0].bank == "A"
    assert diag.flagged_leaves[0].leaf_index == 27
    assert abs(diag.flagged_leaves[0].velocity_lag_ms - 38.0) < 2.0
    assert abs(diag.estimated_dlg_offset_mm - (-0.42)) < 0.03
    assert abs(diag.estimated_gantry_sag_amp_mm - float(np.hypot(0.75, 0.35))) < 0.04
    assert abs(diag.estimated_mu_drift_pct - 1.95) < 0.10
    assert diag.variance_explained_pct > 95.0

    # BIC posterior weights must sum to 1.0 and rank JOINT_MULTI_FAULT first
    total_w = sum(h.posterior_weight for h in diag.hypotheses)
    assert abs(total_w - 1.0) < 1e-4
    assert diag.hypotheses[0].fault_type == "JOINT_MULTI_FAULT"
