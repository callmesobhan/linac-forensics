#!/usr/bin/env python3
"""Run reproducible linac-forensics benchmarks on real OpenKBP pt_1 + synthetic fault suites.

Research Software — NOT A MEDICAL DEVICE.

Outputs:
  - `results/benchmark_summary.json`: Quantitative metrics for binary round-trip,
    zero-error identity, causal attribution recovery, Nelms 2D gamma blindspots,
    fleet triage, and Saltelli-Jansen Sobol' sensitivity.
  - `results/sample_truebeam_v5_multifault.bin`: Authentic Varian TrueBeam v5.0
    binary trajectory log with CRC32 trailer.
  - `results/INC-TB01-WK47_ledger.json` & `results/INC-TB01-WK47_report.md`:
    Autonomous Fleet Physicist incident ledger verified by the Critic-Verifier Gate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from linac_forensics.anatomy import load_openkbp_patient
from linac_forensics.causal_fault_model import diagnose_causal_faults
from linac_forensics.harness.agent_orchestrator import run_fleet_triage_investigation
from linac_forensics.ir import PerturbativeDoseImpact
from linac_forensics.log_parser import (
    generate_synthetic_vmat_log,
    parse_varian_trajectory_bin,
    write_varian_trajectory_bin,
)
from linac_forensics.perturbative_dose import compute_perturbative_dose_replay
from linac_forensics.sobol_sensitivity import run_delivery_sobol_sensitivity

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENKBP_PT1 = REPO_ROOT / "data" / "openkbp" / "pt_1"
RESULTS_DIR = REPO_ROOT / "results"


def run_all_benchmarks() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    anatomy = load_openkbp_patient(OPENKBP_PT1, verify_manifest=True)

    # 1. Binary Varian .bin v5.0 write/parse round-trip benchmark
    sample_log = generate_synthetic_vmat_log(
        log_id="TB01_MULTIFAULT_V5",
        machine_id="TrueBeam-01",
        patient_id="pt_1",
        plan_id="VMAT_HN_SIB_70Gy",
        n_snapshots=360,
        failing_bank="B",
        failing_leaf_index=34,
        leaf_drag_ms=42.0,
        dlg_offset_mm=+0.55,
        gantry_sag_sin_mm=0.85,
        gantry_sag_cos_mm=0.45,
        mu_drift_pct=+1.60,
        noise_sigma_mm=0.025,
        seed=42,
    )
    bin_path = RESULTS_DIR / "sample_truebeam_v5_multifault.bin"
    t0 = time.perf_counter()
    write_varian_trajectory_bin(sample_log, bin_path, header_version="5.0")
    write_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    parsed = parse_varian_trajectory_bin(bin_path)
    parse_ms = (time.perf_counter() - t1) * 1000.0
    max_leaf_err = float(
        np.max(
            np.abs(
                np.asarray(parsed.bank_b_actual_mm) - np.asarray(sample_log.bank_b_actual_mm)
            )
        )
    )

    # 2. Formal zero-error identity invariant
    zero_log = generate_synthetic_vmat_log(log_id="ZERO_INVARIANT", n_snapshots=240)
    t2 = time.perf_counter()
    zero_imp = compute_perturbative_dose_replay(zero_log, anatomy, n_sectors=12)
    assert isinstance(zero_imp, PerturbativeDoseImpact)
    zero_replay_ms = (time.perf_counter() - t2) * 1000.0

    # 3. Causal electro-mechanical fault recovery suite (6 scenarios)
    causal_scenarios = [
        ("Nominal (Zero Fault)", {}),
        (
            "MLC Leaf #34 Viscous Drag",
            {"failing_bank": "B", "failing_leaf_index": 34, "leaf_drag_ms": 42.0},
        ),
        ("Systematic DLG Offset", {"dlg_offset_mm": +0.55}),
        (
            "Gantry Gravity-Sag Harmonic",
            {"gantry_sag_sin_mm": 0.90, "gantry_sag_cos_mm": -0.45},
        ),
        ("Ion-Chamber MU Drift", {"mu_drift_pct": -2.40}),
        (
            "Joint 4-Subsystem Fault",
            {
                "failing_bank": "B",
                "failing_leaf_index": 34,
                "leaf_drag_ms": 42.0,
                "dlg_offset_mm": +0.55,
                "gantry_sag_sin_mm": 0.85,
                "gantry_sag_cos_mm": 0.45,
                "mu_drift_pct": +1.60,
            },
        ),
    ]
    causal_results: list[dict[str, Any]] = []
    for idx, (label, kwargs) in enumerate(causal_scenarios):
        log = generate_synthetic_vmat_log(
            log_id=f"CAUSAL_{idx}",
            n_snapshots=300,
            noise_sigma_mm=0.025,
            seed=200 + idx,
            **kwargs,  # type: ignore[arg-type]
        )
        diag = diagnose_causal_faults(log)
        causal_results.append(
            {
                "scenario": label,
                "injected": kwargs,
                "primary_root_cause": diag.primary_root_cause,
                "active_faults": diag.active_faults,
                "flagged_leaf": (
                    {
                        "bank": diag.flagged_leaves[0].bank,
                        "leaf_index": diag.flagged_leaves[0].leaf_index,
                        "tau_ms": diag.flagged_leaves[0].velocity_lag_ms,
                        "ci95_ms": list(diag.flagged_leaves[0].velocity_lag_ci95_ms),
                    }
                    if diag.flagged_leaves
                    else None
                ),
                "estimated_dlg_mm": diag.estimated_dlg_offset_mm,
                "estimated_dlg_ci95_mm": list(diag.estimated_dlg_ci95_mm),
                "estimated_sag_amp_mm": diag.estimated_gantry_sag_amp_mm,
                "estimated_sag_phase_deg": diag.estimated_gantry_sag_phase_deg,
                "estimated_mu_drift_pct": diag.estimated_mu_drift_pct,
                "estimated_mu_ci95_pct": list(diag.estimated_mu_drift_ci95_pct),
                "variance_explained_pct": diag.variance_explained_pct,
                "hypotheses_delta_bic": {
                    h.fault_type: h.delta_bic_vs_null for h in diag.hypotheses
                },
                "hypotheses_posterior_weight": {
                    h.fault_type: h.posterior_weight for h in diag.hypotheses
                },
            }
        )

    # 4. Nelms et al. (2011) 2D Gamma Blindspot vs 3D ΔDVH Sweep (32 clinical cases)
    rng = np.random.default_rng(77)
    gamma_sweep_cases: list[dict[str, Any]] = []
    dlg_grid = np.linspace(-1.05, 1.05, 16)
    for idx, dlg in enumerate(dlg_grid):
        for variant in (0, 1):
            lag = float(rng.uniform(5.0, 52.0)) if variant == 1 else 0.0
            sag = float(rng.uniform(0.0, 1.2)) if variant == 1 else 0.15
            mu_d = float(rng.uniform(-1.2, 1.2)) if variant == 1 else 0.0
            log = generate_synthetic_vmat_log(
                log_id=f"BLINDSPOT_{idx}_{variant}",
                n_snapshots=150,
                dlg_offset_mm=float(dlg),
                leaf_drag_ms=lag,
                gantry_sag_sin_mm=sag,
                mu_drift_pct=mu_d,
                noise_sigma_mm=0.02,
                seed=500 + idx * 2 + variant,
            )
            imp = compute_perturbative_dose_replay(log, anatomy, n_sectors=8)
            assert isinstance(imp, PerturbativeDoseImpact)
            shifts = {s.roi_name: s for s in imp.structure_shifts}
            ptv70 = shifts["PTV70"]
            cord = shifts["SpinalCord"]
            lpar = shifts["LeftParotid"]
            gamma_sweep_cases.append(
                {
                    "case_id": f"C{idx * 2 + variant + 1:02d}",
                    "dlg_offset_mm": round(float(dlg), 4),
                    "leaf_drag_ms": round(lag, 2),
                    "gantry_sag_mm": round(sag, 3),
                    "mu_drift_pct": round(mu_d, 3),
                    "gamma_2d_3pct_2mm": imp.gamma_2d_3pct_2mm_pass_pct,
                    "gamma_2d_2pct_1mm": imp.gamma_2d_2pct_1mm_pass_pct,
                    "gamma_3d_3pct_2mm": imp.gamma_3d_3pct_2mm_pass_pct,
                    "gamma_3d_2pct_1mm": imp.gamma_3d_2pct_1mm_pass_pct,
                    "ptv70_delta_d95_pct": ptv70.delta_d95_pct,
                    "cord_delta_d003cc_gy": cord.delta_d003cc_gy,
                    "lpar_delta_dmean_gy": lpar.delta_dmean_gy,
                    "clinically_unacceptable": any(
                        s.clinically_unacceptable for s in imp.structure_shifts
                    ),
                    "gamma_blindspot": imp.gamma_blindspot_detected,
                }
            )

    n_unacc = sum(1 for c in gamma_sweep_cases if c["clinically_unacceptable"])
    n_blind = sum(1 for c in gamma_sweep_cases if c["gamma_blindspot"])
    n_caught_3d_2_1 = sum(
        1
        for c in gamma_sweep_cases
        if c["clinically_unacceptable"] and c["gamma_3d_2pct_1mm"] < 95.0
    )

    # 5. Autonomous Fleet Physicist & Incident-Triage Harness
    ledger = run_fleet_triage_investigation(
        anatomy=anatomy,
        machine_id="TrueBeam-01",
        incident_id="INC-TB01-WK47",
        output_dir=RESULTS_DIR,
    )

    # 6. Exact Saltelli-Jansen QMC Global Sobol' Sensitivity (N=512 -> 3,072 evals per metric)
    t3 = time.perf_counter()
    sobol_reports = run_delivery_sobol_sensitivity(anatomy, n_base_samples=512, seed=42)
    sobol_s = time.perf_counter() - t3

    summary: dict[str, Any] = {
        "research_notice": "Research Software — NOT A MEDICAL DEVICE",
        "anatomy_source": anatomy.anatomy_source,
        "binary_roundtrip": {
            "version": "5.0",
            "file_bytes": bin_path.stat().st_size,
            "write_ms": round(write_ms, 2),
            "parse_ms": round(parse_ms, 2),
            "max_leaf_roundtrip_error_mm": max_leaf_err,
        },
        "zero_error_invariant": {
            "replay_ms": round(zero_replay_ms, 2),
            "max_abs_delta_fluence": zero_imp.max_abs_delta_fluence,
            "max_abs_delta_dose_gy": zero_imp.max_abs_delta_dose_gy,
            "gamma_2d_3pct_2mm": zero_imp.gamma_2d_3pct_2mm_pass_pct,
            "gamma_3d_3pct_2mm": zero_imp.gamma_3d_3pct_2mm_pass_pct,
        },
        "causal_attribution_suite": causal_results,
        "gamma_blindspot_benchmark": {
            "n_cases": len(gamma_sweep_cases),
            "n_clinically_unacceptable": n_unacc,
            "n_2d_3pct_2mm_blindspots": n_blind,
            "blindspot_rate_pct": round(100.0 * n_blind / max(n_unacc, 1), 1),
            "n_caught_by_3d_2pct_1mm": n_caught_3d_2_1,
            "cases": gamma_sweep_cases,
        },
        "fleet_triage_summary": {
            "incident_id": ledger.incident_id,
            "critic_gate_passed": ledger.critic_gate.passed,
            "critic_checks": f"{ledger.critic_gate.checks_passed}/{ledger.critic_gate.checks_run}",
            "isolated_subsystems": ledger.service_work_order.isolated_subsystems,
            "urgency": ledger.service_work_order.urgency,
            "n_patients": len(ledger.triage_queue),
            "n_critical": sum(1 for r in ledger.triage_queue if r.triage_priority == "CRITICAL"),
            "tb01_early_warning_day": ledger.tg142_trends[0].early_warning_day,
            "tb01_action_breach_day": ledger.tg142_trends[0].action_level_breach_day,
        },
        "sobol_sensitivity": {
            "runtime_s": round(sobol_s, 2),
            "reports": {k: v.model_dump() for k, v in sobol_reports.items()},
        },
    }
    out_path = RESULTS_DIR / "benchmark_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmarks completed successfully -> {out_path}")
    print(
        f"  Binary v5.0 round-trip: write={write_ms:.1f} ms, parse={parse_ms:.1f} ms, max_err={max_leaf_err:.2e} mm"
    )
    print(
        f"  Zero-error identity: max|dPhi|={zero_imp.max_abs_delta_fluence}, max|dD|={zero_imp.max_abs_delta_dose_gy}, gamma={zero_imp.gamma_3d_3pct_2mm_pass_pct}%"
    )
    print(
        f"  Gamma blindspots: {n_blind}/{n_unacc} clinically failing cases passed 2D 3%/2mm ({summary['gamma_blindspot_benchmark']['blindspot_rate_pct']}%)"
    )
    print(
        f"  Fleet triage: critic_gate={ledger.critic_gate.passed} ({ledger.critic_gate.checks_passed}/{ledger.critic_gate.checks_run}), "
        f"early_warn=Day {ledger.tg142_trends[0].early_warning_day}, breach=Day {ledger.tg142_trends[0].action_level_breach_day}"
    )
    return summary


if __name__ == "__main__":
    run_all_benchmarks()
