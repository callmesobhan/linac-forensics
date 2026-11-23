"""Tests for v0.2.0 upgrades: CT heterogeneity, Dynalog .dlg, Coulomb stiction, EPID 3-way, Kalman RUL, multi-fraction, multi-patient OpenKBP, and MCP resources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linac_forensics.anatomy import (
    ct_hu_to_relative_electron_density,
    load_openkbp_cohort,
)
from linac_forensics.causal_fault_model import diagnose_causal_faults
from linac_forensics.epid_forensics import (
    EPIDThreeWayArrays,
    analyze_three_way_epid_forensics,
    simulate_measured_epid_image,
)
from linac_forensics.harness.agent_orchestrator import (
    generate_tg142_fleet_trends,
    run_fleet_triage_investigation,
)
from linac_forensics.harness.mcp_server import handle_jsonrpc_request
from linac_forensics.ir import AbstentionCode, AbstentionError, PerturbativeDoseImpact
from linac_forensics.log_parser import (
    generate_synthetic_vmat_log,
    parse_varian_dynalog_dlg,
    write_varian_dynalog_dlg,
)
from linac_forensics.perturbative_dose import compute_perturbative_dose_replay
from linac_forensics.predictive_rul import (
    accumulate_multifraction_course,
    forecast_tg142_kalman_rul,
)

OPENKBP_ROOT = Path(__file__).resolve().parents[1] / "data" / "openkbp"


def test_openkbp_three_patient_cohort_and_hu_electron_density() -> None:
    """Verify real OpenKBP pt_1, pt_2, pt_3 SHA-256 loading and HU-to-rho_e calibration."""
    cohort = load_openkbp_cohort(OPENKBP_ROOT, verify_manifest=True)
    assert set(cohort.keys()) == {"pt_1", "pt_2", "pt_3"}
    assert "Esophagus" in cohort["pt_2"].roi_masks

    rho = ct_hu_to_relative_electron_density(np.array([-1000.0, -700.0, 0.0, 400.0, 1200.0]))
    assert rho[0] < 0.01
    assert 0.25 < rho[1] < 0.35
    assert abs(rho[2] - 1.0) < 1e-6
    assert 1.2 < rho[3] < 1.3
    assert 1.65 < rho[4] < 1.75


def test_heterogeneity_correction_preserves_zero_identity_and_alters_nonzero_dose() -> None:
    """CT heterogeneity keeps exact 0.0 zero-error identity while shifting heterogeneous faulted dose."""
    cohort = load_openkbp_cohort(OPENKBP_ROOT, ["pt_1"], verify_manifest=True)
    anat = cohort["pt_1"]

    zero_log = generate_synthetic_vmat_log(log_id="HET_ZERO", n_snapshots=90)
    imp_z = compute_perturbative_dose_replay(zero_log, anat, n_sectors=8, use_heterogeneity=True)
    assert isinstance(imp_z, PerturbativeDoseImpact)
    assert imp_z.heterogeneity_corrected is True
    assert imp_z.max_abs_delta_dose_gy == 0.0
    assert imp_z.gamma_3d_3pct_2mm_pass_pct == 100.0

    fault_log = generate_synthetic_vmat_log(log_id="HET_DLG", n_snapshots=90, dlg_offset_mm=+0.75)
    imp_het = compute_perturbative_dose_replay(fault_log, anat, n_sectors=8, use_heterogeneity=True)
    imp_hom = compute_perturbative_dose_replay(fault_log, anat, n_sectors=8, use_heterogeneity=False)
    assert isinstance(imp_het, PerturbativeDoseImpact)
    assert isinstance(imp_hom, PerturbativeDoseImpact)
    assert abs(imp_het.rms_delta_dose_gy - imp_hom.rms_delta_dose_gy) > 1e-4


def test_varian_legacy_dynalog_dlg_roundtrip_and_abstention(tmp_path: Path) -> None:
    """Verify Varian Clinac/Trilogy Dynalog (.dlg) Bank A/B export, parse, and corruption gates."""
    orig = generate_synthetic_vmat_log(
        log_id="CLINAC_DLG",
        machine_id="Clinac-21EX",
        n_snapshots=60,
        sampling_interval_ms=50.0,
        dlg_offset_mm=+0.40,
        leaf_drag_ms=35.0,
        gantry_sag_sin_mm=0.60,
    )
    pa = tmp_path / "A_20261123.dlg"
    pb = tmp_path / "B_20261123.dlg"
    write_varian_dynalog_dlg(orig, pa, pb)
    parsed = parse_varian_dynalog_dlg(pa, pb)

    assert parsed.header_version == "DYNALOG-B"
    assert parsed.n_snapshots == 60
    np.testing.assert_allclose(parsed.bank_a_actual_mm, orig.bank_a_actual_mm, atol=0.02)
    np.testing.assert_allclose(parsed.bank_b_actual_mm, orig.bank_b_actual_mm, atol=0.02)
    assert "bank_a_dlg" in parsed.provenance.sha256_inputs

    # Mismatched snapshot count triggers TIMESTAMP_DESYNC
    pb_short = tmp_path / "B_short.dlg"
    pb_short.write_text("\n".join(pb.read_text().splitlines()[:-5]) + "\n", encoding="utf-8")
    with pytest.raises(AbstentionError) as exc:
        parse_varian_dynalog_dlg(pa, pb_short)
    assert exc.value.code == AbstentionCode.TIMESTAMP_DESYNC


def test_nonlinear_coulomb_stiction_vs_viscous_drag_separation() -> None:
    """Verify BIC separates pure Coulomb stiction from viscous drag and detects both jointly."""
    # 1. Pure Coulomb stiction (tau=0, beta=0.70 mm)
    log_s = generate_synthetic_vmat_log(
        log_id="PURE_STICTION",
        n_snapshots=300,
        failing_bank="B",
        failing_leaf_index=32,
        leaf_drag_ms=0.0,
        leaf_stiction_mm=0.70,
        noise_sigma_mm=0.02,
        seed=51,
    )
    diag_s = diagnose_causal_faults(log_s)
    assert diag_s.primary_root_cause == "MLC_COULOMB_STICTION"
    assert diag_s.active_faults == ["MLC_COULOMB_STICTION"]
    assert len(diag_s.flagged_leaves) == 1
    assert diag_s.flagged_leaves[0].leaf_index == 32
    assert abs(diag_s.flagged_leaves[0].stiction_amplitude_mm - 0.70) < 0.05

    # 2. Combined viscous drag + Coulomb stiction
    log_both = generate_synthetic_vmat_log(
        log_id="VISC_PLUS_STIC",
        n_snapshots=300,
        failing_bank="B",
        failing_leaf_index=34,
        leaf_drag_ms=36.0,
        leaf_stiction_mm=0.55,
        noise_sigma_mm=0.02,
        seed=52,
    )
    diag_both = diagnose_causal_faults(log_both)
    assert diag_both.primary_root_cause == "JOINT_MULTI_FAULT"
    assert set(diag_both.active_faults) == {"MLC_LEAF_MOTOR_FRICTION", "MLC_COULOMB_STICTION"}
    leaf = diag_both.flagged_leaves[0]
    assert abs(leaf.velocity_lag_ms - 36.0) < 2.5
    assert abs(leaf.stiction_amplitude_mm - 0.55) < 0.06


def test_epid_three_way_forensics_all_four_classifications() -> None:
    """Verify TPS vs Log vs EPID catches encoder-blind silent faults & steering tilts."""
    # 1. Concordant nominal
    log_nom = generate_synthetic_vmat_log(log_id="EPID_NOM", n_snapshots=120)
    _, _, epid_nom = simulate_measured_epid_image(log_nom, noise_rel_sigma=0.001)
    r_nom = analyze_three_way_epid_forensics(log_nom, epid_nom)
    assert not isinstance(r_nom, EPIDThreeWayArrays)
    assert r_nom.classification == "CONCORDANT_NOMINAL"
    assert r_nom.encoder_blind_fault_detected is False

    # 2. Log-verified mechanical fault (large DLG error recorded in log AND seen on EPID)
    log_mech = generate_synthetic_vmat_log(
        log_id="EPID_MECH", n_snapshots=120, dlg_offset_mm=+1.4, leaf_drag_ms=55.0
    )
    _, _, epid_mech = simulate_measured_epid_image(log_mech, noise_rel_sigma=0.001)
    r_mech = analyze_three_way_epid_forensics(log_mech, epid_mech)
    assert not isinstance(r_mech, EPIDThreeWayArrays)
    assert r_mech.classification == "LOG_VERIFIED_MECHANICAL_FAULT"
    assert r_mech.encoder_blind_fault_detected is False

    # 3. CRITICAL: Encoder-blind silent fault (log says nominal, EPID sees +1.6 mm slipped DLG + leaf slip)
    _, _, epid_silent = simulate_measured_epid_image(
        log_nom, silent_dlg_bias_mm=+1.6, silent_leaf_slip_mm=3.5, noise_rel_sigma=0.001
    )
    r_silent = analyze_three_way_epid_forensics(log_nom, epid_silent, return_arrays=True)
    assert isinstance(r_silent, EPIDThreeWayArrays)
    assert r_silent.report.classification == "ENCODER_BLIND_SILENT_FAULT"
    assert r_silent.report.encoder_blind_fault_detected is True
    assert r_silent.report.gamma_plan_vs_log_2pct_1mm == 100.0
    assert r_silent.report.gamma_log_vs_epid_2pct_1mm < 95.0

    # 4. Beam steering / EPID panel symmetry tilt
    _, _, epid_tilt = simulate_measured_epid_image(
        log_nom, beam_tilt_x_pct_per_10cm=+6.0, noise_rel_sigma=0.001
    )
    r_tilt = analyze_three_way_epid_forensics(log_nom, epid_tilt)
    assert not isinstance(r_tilt, EPIDThreeWayArrays)
    assert r_tilt.classification == "EPID_CALIBRATION_OR_PANEL_DRIFT"
    assert r_tilt.crossplane_symmetry_tilt_pct > 2.5


def test_kalman_rul_forecast_and_multifraction_accumulation() -> None:
    """Verify Kalman RUL anticipates TrueBeam-01 breach and multi-fraction separates sys vs random."""
    trends = generate_tg142_fleet_trends()
    tb01, tb02 = trends[0], trends[1]

    f_tb01 = forecast_tg142_kalman_rul(
        tb01, metric_name="mlc_max_rms_mm", action_limit=0.70, observation_cutoff_day=20, horizon_days=10
    )
    assert f_tb01.rul_mean_days is not None
    assert 1.5 <= f_tb01.rul_mean_days <= 12.0
    assert f_tb01.breach_probability_within_horizon > 0.50
    assert f_tb01.rul_ci95_days is not None and f_tb01.rul_ci95_days[0] <= f_tb01.rul_ci95_days[1]

    f_tb02 = forecast_tg142_kalman_rul(
        tb02, metric_name="mlc_max_rms_mm", action_limit=0.70, observation_cutoff_day=20, horizon_days=10
    )
    assert f_tb02.breach_probability_within_horizon < 0.05

    # Multi-fraction accumulation over 5 fractions on real OpenKBP pt_2
    cohort = load_openkbp_cohort(OPENKBP_ROOT, ["pt_2"], verify_manifest=True)
    pt2 = cohort["pt_2"]
    fx_logs = [
        generate_synthetic_vmat_log(
            log_id=f"PT2_FX{i}",
            patient_id="pt_2",
            n_snapshots=90,
            dlg_offset_mm=-0.85 + 0.06 * (i - 2),
            seed=80 + i,
        )
        for i in range(5)
    ]
    acc = accumulate_multifraction_course(fx_logs, pt2, total_prescribed_fractions=35, n_sectors=8)
    assert acc.fractions_delivered == 5
    assert acc.cumulative_ptv_delta_d95_pct < -1.8
    assert acc.random_component_rms_pct > 0.0
    assert acc.recommended_remaining_fx_mu_scale > 1.0


def test_mcp_new_tools_and_resources_endpoints() -> None:
    """Verify MCP v0.2.0 EPID 3-way tool, Kalman RUL tool, and resources/list + resources/read."""
    epid_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 101,
            "method": "tools/call",
            "params": {
                "name": "linac_analyze_epid_three_way",
                "arguments": {"silent_dlg_bias_mm": 1.6, "silent_leaf_slip_mm": 3.0},
            },
        }
    )
    assert epid_resp is not None and epid_resp["result"]["isError"] is False
    assert epid_resp["result"]["structuredContent"]["encoder_blind_fault_detected"] is True

    rul_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 102,
            "method": "tools/call",
            "params": {
                "name": "linac_forecast_tg142_rul",
                "arguments": {"machine_id": "TrueBeam-01", "observation_cutoff_day": 20},
            },
        }
    )
    assert rul_resp is not None and rul_resp["result"]["isError"] is False
    assert rul_resp["result"]["structuredContent"]["breach_probability_within_horizon"] > 0.5

    res_list = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 103, "method": "resources/list"})
    assert res_list is not None
    uris = {r["uri"] for r in res_list["result"]["resources"]}
    assert "linac://manifest/openkbp_pt1" in uris

    res_read = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 104,
            "method": "resources/read",
            "params": {"uri": "linac://manifest/openkbp_pt1"},
        }
    )
    assert res_read is not None
    assert "OpenKBP" in res_read["result"]["contents"][0]["text"]

    # Missing resource returns -32002
    missing = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 105,
            "method": "resources/read",
            "params": {"uri": "linac://does/not/exist"},
        }
    )
    assert missing is not None and missing["error"]["code"] == -32002


def test_multipatient_fleet_triage_with_kalman_rul() -> None:
    """Verify fleet triage runs across all 3 real OpenKBP patients and populates Kalman RUL."""
    cohort = load_openkbp_cohort(OPENKBP_ROOT, verify_manifest=True)
    ledger = run_fleet_triage_investigation(
        anatomy=cohort["pt_1"],
        anatomy_cohort=cohort,
        machine_id="TrueBeam-01",
        incident_id="INC-COHORT-V02",
    )
    assert ledger.critic_gate.passed is True
    assert ledger.rul_forecast is not None
    assert ledger.rul_forecast.rul_mean_days is not None
