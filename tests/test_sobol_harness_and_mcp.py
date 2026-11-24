"""Tests for Saltelli-Jansen Sobol' sensitivity, Autonomous Fleet Harness, and MCP Server."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from linac_forensics.anatomy import build_synthetic_phantom_anatomy, load_openkbp_patient
from linac_forensics.harness.agent_orchestrator import run_fleet_triage_investigation
from linac_forensics.harness.mcp_server import handle_jsonrpc_request
from linac_forensics.log_parser import generate_synthetic_vmat_log, write_varian_trajectory_bin
from linac_forensics.sobol_sensitivity import (
    compute_saltelli_jansen_indices,
    run_delivery_sobol_sensitivity,
)

OPENKBP_PT1_DIR = Path(__file__).resolve().parents[1] / "data" / "openkbp" / "pt_1"


def test_sobol_analytical_linear_benchmark() -> None:
    """Verify exact Saltelli-Jansen recovery on Y = 3*X1 + 1*X2 + 0*X3 on [0, 1]^3."""

    def linear_model(x: np.ndarray) -> np.ndarray:
        return 3.0 * x[:, 0] + 1.0 * x[:, 1] + 0.0 * x[:, 2]

    rep = compute_saltelli_jansen_indices(
        evaluator=linear_model,
        parameter_bounds={"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)},
        output_metric="analytical_linear",
        n_base_samples=512,
        seed=42,
    )
    # Exact analytical Sobol' indices: S1(x1)=9/10=0.90, S1(x2)=1/10=0.10, S1(x3)=0.0
    assert abs(rep.first_order_s1["x1"] - 0.90) < 0.03
    assert abs(rep.first_order_s1["x2"] - 0.10) < 0.03
    assert rep.total_order_st["x3"] < 0.01
    for p in rep.parameter_names:
        assert 0.0 <= rep.first_order_s1[p] <= 1.0
        assert 0.0 <= rep.total_order_st[p] <= 1.0
        assert rep.total_order_st[p] + 0.03 >= rep.first_order_s1[p]


def test_delivery_sobol_sensitivity_on_openkbp_pt1() -> None:
    """Verify global Sobol' sensitivity bounds on real OpenKBP pt_1 3D anatomy."""
    anat = load_openkbp_patient(OPENKBP_PT1_DIR, verify_manifest=True)
    reports = run_delivery_sobol_sensitivity(anat, n_base_samples=128, seed=42)
    assert set(reports.keys()) == {"PTV70_delta_D95_pct", "SpinalCord_delta_D003cc_Gy"}

    for rep in reports.values():
        assert rep.total_evaluations == 128 * (4 + 2)
        s1_sum = sum(rep.first_order_s1.values())
        st_sum = sum(rep.total_order_st.values())
        assert 0.5 <= s1_sum <= 1.15
        assert st_sum >= s1_sum - 0.05
        for name in rep.parameter_names:
            lo_s1, hi_s1 = rep.s1_ci95[name]
            lo_st, hi_st = rep.st_ci95[name]
            assert 0.0 <= lo_s1 <= hi_s1 <= 1.0
            assert 0.0 <= lo_st <= hi_st <= 1.0


def test_autonomous_fleet_triage_harness_and_critic_gate(tmp_path: Path) -> None:
    """Verify full Autonomous Fleet Physicist harness, Critic-Verifier Gate, and ledger exports."""
    anat = load_openkbp_patient(OPENKBP_PT1_DIR, verify_manifest=True)
    ledger = run_fleet_triage_investigation(
        anatomy=anat,
        machine_id="TrueBeam-01",
        incident_id="INC-TEST-01",
        output_dir=tmp_path,
    )
    assert ledger.critic_gate.passed is True
    assert ledger.critic_gate.checks_passed == ledger.critic_gate.checks_run
    assert len(ledger.triage_queue) == 6
    # Verify monotonic risk sorting
    scores = [r.clinical_risk_score for r in ledger.triage_queue]
    assert scores == sorted(scores, reverse=True)
    # Verify Bank B Leaf #34 & DLG isolated in work order
    subsys_text = " ".join(ledger.service_work_order.isolated_subsystems)
    assert "Leaf #34" in subsys_text
    assert "DLG" in subsys_text
    # Verify exported JSON + Markdown files exist
    assert (tmp_path / "INC-TEST-01_ledger.json").exists()
    assert (tmp_path / "INC-TEST-01_report.md").exists()
    assert "NOT A MEDICAL DEVICE" in ledger.markdown_report


def test_mcp_protocol_handshake_and_tools_list() -> None:
    """Verify MCP JSON-RPC 2.0 initialize, ping, notification, and 6-tool listing."""
    init = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init is not None
    assert init["result"]["serverInfo"]["name"] == "linac-forensics-mcp"
    assert "NOT A MEDICAL DEVICE" in init["result"]["serverInfo"]["notice"]

    assert handle_jsonrpc_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert handle_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "ping"})["result"] == {}

    listed = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    assert listed is not None
    names = {t["name"] for t in listed["result"]["tools"]}
    assert names == {
        "linac_parse_trajectory_log",
        "linac_compute_perturbative_dvh",
        "linac_diagnose_causal_fault",
        "linac_compare_gamma_vs_dvh",
        "linac_triage_patient_fleet",
        "linac_run_sobol_sensitivity",
        "linac_analyze_epid_three_way",
        "linac_forecast_tg142_rul",
        "linac_certify_plan_robustness",
        "linac_simulate_leaf_outage",
    }


def test_mcp_parse_tool_valid_and_abstention(tmp_path: Path) -> None:
    """Verify MCP linac_parse_trajectory_log on valid .bin and abstention on corrupted .bin."""
    log = generate_synthetic_vmat_log(log_id="MCP_BIN", n_snapshots=40)
    bin_path = tmp_path / "mcp_valid.bin"
    write_varian_trajectory_bin(log, bin_path, header_version="5.0")

    ok = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "linac_parse_trajectory_log", "arguments": {"file_path": str(bin_path)}},
        }
    )
    assert ok is not None
    assert ok["result"]["isError"] is False
    assert ok["result"]["structuredContent"]["n_snapshots"] == 40

    # Corrupted binary returns structured AbstentionError without crashing server
    bad_path = tmp_path / "corrupt.bin"
    bad_path.write_bytes(b"BADHEADER")
    abstained = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {"name": "linac_parse_trajectory_log", "arguments": {"file_path": str(bad_path)}},
        }
    )
    assert abstained is not None
    assert abstained["result"]["isError"] is True
    assert abstained["result"]["structuredContent"]["abstained"] is True
    assert abstained["result"]["structuredContent"]["code"] == "CORRUPTED_BINARY_LOG"


def test_mcp_diagnosis_dvh_and_gamma_compare_tools() -> None:
    """Verify MCP causal diagnosis, perturbative DVH, and gamma-vs-DVH comparison tools."""
    diag_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {
                "name": "linac_diagnose_causal_fault",
                "arguments": {
                    "synthetic_faults": {
                        "failing_bank": "B",
                        "failing_leaf_index": 34,
                        "leaf_drag_ms": 42.0,
                        "dlg_offset_mm": 0.45,
                    }
                },
            },
        }
    )
    assert diag_resp is not None and diag_resp["result"]["isError"] is False
    sc = diag_resp["result"]["structuredContent"]
    assert sc["primary_root_cause"] == "JOINT_MULTI_FAULT"

    dvh_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {
                "name": "linac_compute_perturbative_dvh",
                "arguments": {"synthetic_faults": {"dlg_offset_mm": -0.80, "leaf_drag_ms": 30.0}},
            },
        }
    )
    assert dvh_resp is not None and dvh_resp["result"]["isError"] is False
    assert dvh_resp["result"]["structuredContent"]["gamma_2d_3pct_2mm_pass_pct"] >= 95.0

    cmp_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": {
                "name": "linac_compare_gamma_vs_dvh",
                "arguments": {"dlg_offsets_mm": [-0.85, 0.0, 0.85]},
            },
        }
    )
    assert cmp_resp is not None and cmp_resp["result"]["isError"] is False
    assert cmp_resp["result"]["structuredContent"]["n_cases"] == 3
    assert cmp_resp["result"]["structuredContent"]["n_blindspots"] >= 1


def test_mcp_fleet_triage_sobol_and_error_handling() -> None:
    """Verify MCP fleet triage, Sobol' sensitivity, and unknown method/tool error codes."""
    triage_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 30,
            "method": "tools/call",
            "params": {"name": "linac_triage_patient_fleet", "arguments": {"machine_id": "TrueBeam-01"}},
        }
    )
    assert triage_resp is not None and triage_resp["result"]["isError"] is False
    assert triage_resp["result"]["structuredContent"]["critic_gate"]["passed"] is True

    sobol_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 31,
            "method": "tools/call",
            "params": {"name": "linac_run_sobol_sensitivity", "arguments": {"n_base_samples": 64}},
        }
    )
    assert sobol_resp is not None and sobol_resp["result"]["isError"] is False
    assert "PTV70_delta_D95_pct" in sobol_resp["result"]["structuredContent"]

    bad_tool = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 32,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
    )
    assert bad_tool is not None and bad_tool["error"]["code"] == -32601

    bad_method = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 33, "method": "foo/bar"})
    assert bad_method is not None and bad_method["error"]["code"] == -32601


def test_synthetic_phantom_builder_sanity() -> None:
    """Ensure synthetic phantom helper produces valid non-empty masks and positive voxel volume."""
    p = build_synthetic_phantom_anatomy((24, 24, 24))
    assert p.voxel_volume_cc > 0.0
    assert int(p.roi_masks["PTV70"].sum()) > 0
    assert int(p.roi_masks["SpinalCord"].sum()) > 0


def test_mcp_stdio_main_loop(monkeypatch: object) -> None:
    """Exercise newline-delimited JSON-RPC 2.0 stdio transport including parse error handling."""
    import io
    import json

    from linac_forensics.harness import mcp_server

    stdin_data = (
        '{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'
        '{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
        "not-valid-json\n"
        '{"jsonrpc":"2.0","id":2,"method":"ping"}\n'
    )
    fake_in = io.StringIO(stdin_data)
    fake_out = io.StringIO()
    monkeypatch.setattr(mcp_server.sys, "stdin", fake_in)  # type: ignore[attr-defined]
    monkeypatch.setattr(mcp_server.sys, "stdout", fake_out)  # type: ignore[attr-defined]
    mcp_server.main()

    lines = [json.loads(ln) for ln in fake_out.getvalue().strip().splitlines()]
    assert len(lines) == 3
    assert lines[0]["id"] == 1
    assert lines[1]["error"]["code"] == -32700
    assert lines[2]["id"] == 2

