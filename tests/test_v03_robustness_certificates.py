"""Tests for v0.3.0 Delivery Robustness Certificates, Inverse Tolerance Budgeting, Fleet-Safe Routing & Leaf-Outage Counterfactuals."""

from __future__ import annotations

from pathlib import Path

from linac_forensics.anatomy import load_openkbp_cohort
from linac_forensics.harness.agent_orchestrator import generate_tg142_fleet_trends
from linac_forensics.harness.mcp_server import handle_jsonrpc_request
from linac_forensics.log_parser import generate_synthetic_vmat_log
from linac_forensics.predictive_rul import forecast_tg142_kalman_rul
from linac_forensics.robustness_certificate import (
    STATIC_TG142_LIMITS,
    build_fleet_fingerprints,
    certify_delivery_robustness,
    compute_fleet_routing_matrix,
    compute_plan_tolerance_budget,
    extract_linac_fingerprint,
    simulate_counterfactual_leaf_outage,
)

OPENKBP_ROOT = Path(__file__).resolve().parents[1] / "data" / "openkbp"


def test_extract_linac_fingerprint_with_kalman_inflation() -> None:
    """Verify live fingerprint extraction & Kalman drift uncertainty inflation."""
    log = generate_synthetic_vmat_log(
        log_id="FP_TEST",
        machine_id="TrueBeam-01",
        n_snapshots=180,
        failing_bank="B",
        failing_leaf_index=34,
        leaf_drag_ms=24.0,
        dlg_offset_mm=+0.34,
        gantry_sag_sin_mm=0.48,
        mu_drift_pct=+0.65,
        noise_sigma_mm=0.02,
    )
    fp_base = extract_linac_fingerprint(log)
    assert fp_base.tg142_status == "PASSING_TG142"
    assert fp_base.failing_bank == "B"
    assert fp_base.failing_leaf_index == 34

    tb01_trend = generate_tg142_fleet_trends()[0]
    rul = forecast_tg142_kalman_rul(tb01_trend, observation_cutoff_day=20)
    fp_infl = extract_linac_fingerprint(log, rul_forecast=rul, predictive_horizon_days=5)
    assert fp_infl.std_vector["dlg_offset_mm"] > fp_base.std_vector["dlg_offset_mm"]
    assert fp_infl.std_vector["mlc_drag_ms"] > fp_base.std_vector["mlc_drag_ms"]


def test_plan_inverse_tolerance_budget_scales_with_modulation_complexity() -> None:
    """High-MCI H&N/SBRT plans demand tighter-than-TG-142 DLG budgets; open Breast has wide slack."""
    cohort = load_openkbp_cohort(OPENKBP_ROOT, ["pt_1"], verify_manifest=True)
    pt1 = cohort["pt_1"]

    budget_sbrt = compute_plan_tolerance_budget(
        pt1, "PT-SBRT", "VMAT_SBRT", "Spine SBRT", modulation_complexity_index=1.70
    )
    budget_breast = compute_plan_tolerance_budget(
        pt1, "PT-BR", "VMAT_Breast", "Whole Breast", modulation_complexity_index=0.42
    )

    assert "dlg_offset_mm" in budget_sbrt.tighter_than_tg142
    assert budget_sbrt.max_safe_tolerances["dlg_offset_mm"] < STATIC_TG142_LIMITS["dlg_offset_mm"]
    assert (
        budget_breast.max_safe_tolerances["dlg_offset_mm"]
        > budget_sbrt.max_safe_tolerances["dlg_offset_mm"] * 3.0
    )
    assert budget_breast.tighter_than_tg142 == []


def test_adversarial_ellipsoidal_certificate_detects_tg142_stealth_risk() -> None:
    """A linac passing static TG-142 still triggers TG-142 STEALTH RISK for high-MCI H&N SIB."""
    cohort = load_openkbp_cohort(OPENKBP_ROOT, ["pt_1"], verify_manifest=True)
    fps = build_fleet_fingerprints()
    tb01, tb02 = fps[0], fps[1]
    assert tb01.tg142_status == "PASSING_TG142"
    assert tb02.tg142_status == "PASSING_TG142"

    budget_hn = compute_plan_tolerance_budget(
        cohort["pt_1"], "PT-HN-001", "VMAT_HN_SIB", "H&N SIB", modulation_complexity_index=1.55
    )
    cert_tb01 = certify_delivery_robustness(budget_hn, tb01)
    cert_tb02 = certify_delivery_robustness(budget_hn, tb02)

    assert cert_tb01.verdict == "UNSAFE_REROUTE_REQUIRED"
    assert cert_tb01.tg142_stealth_risk is True
    assert cert_tb01.safety_margin < 1.0

    assert cert_tb02.verdict == "CERTIFIED_SAFE"
    assert cert_tb02.tg142_stealth_risk is False
    assert cert_tb02.safety_margin > 1.25
    assert abs(cert_tb01.worst_case_ptv_delta_d95_pct) >= abs(cert_tb01.nominal_ptv_delta_d95_pct)


def test_fleet_safe_routing_matrix_transfers_complex_and_retains_simple_plans() -> None:
    """Fleet routing transfers high-MCI stealth-risk patients to TrueBeam-02 while retaining low-MCI on TrueBeam-01."""
    cohort = load_openkbp_cohort(OPENKBP_ROOT, verify_manifest=True)
    routing = compute_fleet_routing_matrix(cohort)

    assert len(routing.machine_ids) == 3
    assert len(routing.patient_ids) == 6
    assert routing.n_stealth_tg142_risks >= 3

    recs = {r.patient_id: r for r in routing.routing_recommendations}
    assert recs["PT-HN-001"].transfer_required is True
    assert recs["PT-HN-001"].recommended_machine_id == "TrueBeam-02"
    assert recs["PT-HN-001"].margin_gain_ratio > 2.5

    assert recs["PT-PR-005"].transfer_required is False
    assert recs["PT-PR-005"].recommended_machine_id == "TrueBeam-01"
    assert recs["PT-BR-006"].transfer_required is False
    assert recs["PT-BR-006"].recommended_machine_id == "TrueBeam-01"


def test_counterfactual_leaf_outage_parked_vs_adjacent_compensation() -> None:
    """Parked closed Leaf #34 forces treatment hold; adjacent-leaf compensation restores safety."""
    cohort = load_openkbp_cohort(OPENKBP_ROOT, ["pt_1"], verify_manifest=True)
    pt1 = cohort["pt_1"]

    rep_parked = simulate_counterfactual_leaf_outage(
        pt1, outage_bank="B", outage_leaf_index=34, compensation_mode="PARKED_CLOSED"
    )
    rep_adj = simulate_counterfactual_leaf_outage(
        pt1, outage_bank="B", outage_leaf_index=34, compensation_mode="ADJACENT_LEAF_AVERAGE"
    )
    assert rep_parked.verdict == "HOLD_UNTIL_LEAF_REPAIRED"
    assert abs(rep_parked.ptv_delta_d95_pct) > 5.0
    assert rep_adj.verdict == "SAFE_TO_TREAT_WITH_OUTAGE"
    assert abs(rep_adj.ptv_delta_d95_pct) < 1.5


def test_mcp_v03_robustness_and_leaf_outage_tools() -> None:
    """Verify MCP tools #9 and #10 return provenance-tagged structured JSON."""
    r1 = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 201,
            "method": "tools/call",
            "params": {"name": "linac_certify_plan_robustness", "arguments": {}},
        }
    )
    assert r1 is not None and r1["result"]["isError"] is False
    assert r1["result"]["structuredContent"]["n_stealth_tg142_risks"] >= 3

    r2 = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 202,
            "method": "tools/call",
            "params": {
                "name": "linac_simulate_leaf_outage",
                "arguments": {"outage_bank": "B", "outage_leaf_index": 34, "compensation_mode": "PARKED_CLOSED"},
            },
        }
    )
    assert r2 is not None and r2["result"]["isError"] is False
    assert r2["result"]["structuredContent"]["verdict"] == "HOLD_UNTIL_LEAF_REPAIRED"
