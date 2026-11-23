"""Tests for the Fast Perturbative Fluence-to-3D-Dose Replay Engine and OpenKBP pt_1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linac_forensics.anatomy import (
    PatientAnatomyVolume,
    build_synthetic_phantom_anatomy,
    load_openkbp_patient,
)
from linac_forensics.log_parser import generate_synthetic_vmat_log
from linac_forensics.perturbative_dose import (
    PerturbativeReplayArrays,
    compute_perturbative_dose_replay,
)

OPENKBP_PT1_DIR = Path(__file__).resolve().parents[1] / "data" / "openkbp" / "pt_1"


@pytest.fixture(scope="module")
def openkbp_pt1() -> PatientAnatomyVolume:
    assert OPENKBP_PT1_DIR.exists(), "OpenKBP pt_1 data directory must be staged."
    return load_openkbp_patient(OPENKBP_PT1_DIR, verify_manifest=True)


def test_openkbp_pt1_manifest_and_anatomy_integrity(openkbp_pt1: PatientAnatomyVolume) -> None:
    """Verify real OpenKBP pt_1 loads cleanly with SHA-256 verification and 8 anatomical ROIs."""
    assert openkbp_pt1.provenance_tag == "OPEN_PUBLIC_ANATOMY_OPENKBP_PT1"
    assert openkbp_pt1.ct_hu.shape == (128, 128, 128)
    assert openkbp_pt1.planned_dose_gy.shape == (128, 128, 128)
    assert len(openkbp_pt1.sha256_inputs) == 12
    for roi in ("PTV70", "SpinalCord", "Brainstem", "LeftParotid", "RightParotid", "Mandible"):
        assert roi in openkbp_pt1.roi_masks
        assert int(openkbp_pt1.roi_masks[roi].sum()) > 50


def test_zero_error_identity_invariant_exact(openkbp_pt1: PatientAnatomyVolume) -> None:
    """Formal invariant: actual == expected yields exact 0.0 ΔΦ, 0.0 ΔDose, and 100.0% gamma."""
    zero_log = generate_synthetic_vmat_log(log_id="ZERO_IDENTITY", n_snapshots=120)
    res = compute_perturbative_dose_replay(zero_log, openkbp_pt1, n_sectors=8, return_arrays=True)
    assert isinstance(res, PerturbativeReplayArrays)

    assert res.impact.max_abs_delta_fluence == 0.0
    assert res.impact.rms_delta_fluence == 0.0
    assert res.impact.max_abs_delta_dose_gy == 0.0
    assert res.impact.rms_delta_dose_gy == 0.0
    assert res.impact.gamma_2d_3pct_2mm_pass_pct == 100.0
    assert res.impact.gamma_2d_2pct_1mm_pass_pct == 100.0
    assert res.impact.gamma_3d_3pct_2mm_pass_pct == 100.0
    assert res.impact.gamma_3d_2pct_1mm_pass_pct == 100.0
    assert res.impact.gamma_blindspot_detected is False
    assert float(np.max(np.abs(res.delta_fluence_2d))) == 0.0
    assert float(np.max(np.abs(res.delta_dose_3d))) == 0.0

    for shift in res.impact.structure_shifts:
        assert shift.delta_d95_gy == 0.0
        assert shift.delta_d003cc_gy == 0.0
        assert shift.delta_eud_gy == 0.0
        assert shift.delta_bio_prob == 0.0
        assert shift.clinically_unacceptable is False


def test_dlg_sign_symmetry_and_clinical_direction(openkbp_pt1: PatientAnatomyVolume) -> None:
    """Widening DLG (+0.8 mm) overdoses PTV/OARs; narrowing DLG (-0.8 mm) underdoses PTV."""
    log_pos = generate_synthetic_vmat_log(log_id="DLG_POS", n_snapshots=120, dlg_offset_mm=+0.80)
    log_neg = generate_synthetic_vmat_log(log_id="DLG_NEG", n_snapshots=120, dlg_offset_mm=-0.80)

    imp_pos = compute_perturbative_dose_replay(log_pos, openkbp_pt1, n_sectors=8)
    imp_neg = compute_perturbative_dose_replay(log_neg, openkbp_pt1, n_sectors=8)
    assert not isinstance(imp_pos, PerturbativeReplayArrays)
    assert not isinstance(imp_neg, PerturbativeReplayArrays)

    ptv_pos = next(s for s in imp_pos.structure_shifts if s.roi_name == "PTV70")
    ptv_neg = next(s for s in imp_neg.structure_shifts if s.roi_name == "PTV70")
    cord_pos = next(s for s in imp_pos.structure_shifts if s.roi_name == "SpinalCord")

    assert ptv_pos.delta_d95_pct > +1.8
    assert ptv_neg.delta_d95_pct < -1.8
    assert cord_pos.delta_d003cc_gy > +0.8


def test_nelms_2d_gamma_blindspot_detection(openkbp_pt1: PatientAnatomyVolume) -> None:
    """Reproduce Nelms et al. (2011): 2D 3%/2mm passes (>=95%) while 3D ΔDVH fails clinically."""
    log_blind = generate_synthetic_vmat_log(
        log_id="NELMS_BLINDSPOT",
        n_snapshots=150,
        dlg_offset_mm=-0.85,
        leaf_drag_ms=35.0,
        noise_sigma_mm=0.015,
    )
    imp = compute_perturbative_dose_replay(log_blind, openkbp_pt1, n_sectors=10)
    assert not isinstance(imp, PerturbativeReplayArrays)

    ptv70 = next(s for s in imp.structure_shifts if s.roi_name == "PTV70")
    assert imp.gamma_2d_3pct_2mm_pass_pct >= 95.0
    assert abs(ptv70.delta_d95_pct) >= 2.0
    assert imp.gamma_blindspot_detected is True
    assert imp.gamma_3d_2pct_1mm_pass_pct < imp.gamma_2d_3pct_2mm_pass_pct
    assert imp.provenance.provenance_tag == "COMPOSITE_OPENKBP_PLUS_SYNTHETIC_LOG"


def test_mu_drift_monotonic_dose_response(openkbp_pt1: PatientAnatomyVolume) -> None:
    """Verify positive MU output drift monotonically increases delivered PTV D95."""
    log_lo = generate_synthetic_vmat_log(log_id="MU_1PCT", n_snapshots=100, mu_drift_pct=+1.0)
    log_hi = generate_synthetic_vmat_log(log_id="MU_3PCT", n_snapshots=100, mu_drift_pct=+3.0)

    imp_lo = compute_perturbative_dose_replay(log_lo, openkbp_pt1, n_sectors=8)
    imp_hi = compute_perturbative_dose_replay(log_hi, openkbp_pt1, n_sectors=8)
    assert not isinstance(imp_lo, PerturbativeReplayArrays)
    assert not isinstance(imp_hi, PerturbativeReplayArrays)

    ptv_lo = next(s for s in imp_lo.structure_shifts if s.roi_name == "PTV70")
    ptv_hi = next(s for s in imp_hi.structure_shifts if s.roi_name == "PTV70")
    assert ptv_hi.delta_d95_gy > ptv_lo.delta_d95_gy > 0.0


def test_synthetic_phantom_replay_consistency() -> None:
    """Verify lightweight synthetic phantom fallback works end-to-end with return_arrays=True."""
    phantom = build_synthetic_phantom_anatomy(shape=(32, 32, 32))
    log = generate_synthetic_vmat_log(log_id="PHANTOM_TEST", n_snapshots=60, dlg_offset_mm=+0.5)
    out = compute_perturbative_dose_replay(log, phantom, n_sectors=6, return_arrays=True)
    assert isinstance(out, PerturbativeReplayArrays)
    assert out.delivered_dose_3d.shape == (32, 32, 32)
    assert np.all(np.isfinite(out.delivered_dose_3d))
    assert np.all(out.delivered_dose_3d >= 0.0)
