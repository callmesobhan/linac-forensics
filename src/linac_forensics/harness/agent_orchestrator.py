"""Autonomous Fleet Physicist & Incident-Triage Agent Harness with Critic-Verifier Gate.

Research Software — NOT A MEDICAL DEVICE.

Orchestrates an end-to-end closed-loop radiotherapy machine physics investigation:
1. Ingests a multi-patient treatment week of 50 Hz trajectory logs + 30-day multi-linac
   AAPM TG-142 / Varian MPC trend series.
2. Formulates competing physical fault hypotheses, invokes the core matched-filter /
   Bayesian BIC attribution engine to falsify non-matching causes numerically, and
   isolates the exact failing hardware subsystem(s).
3. Runs 3D perturbative dose replays across all treated patients to assemble a
   prioritized Clinical Patient-Impact Triage Queue + Service Engineering Work Order.
4. Audits every artifact through an independent Critic-Verifier Gate before emitting
   both Markdown and JSON forensic ledgers.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from linac_forensics.anatomy import PatientAnatomyVolume
from linac_forensics.causal_fault_model import diagnose_causal_faults
from linac_forensics.ir import (
    CausalFaultDiagnosis,
    PatientTriageRecord,
    PerturbativeDoseImpact,
    ProvenanceRecord,
    TG142TrendSeries,
    TrajectoryLogTensor,
)
from linac_forensics.log_parser import generate_synthetic_vmat_log
from linac_forensics.perturbative_dose import compute_perturbative_dose_replay
from linac_forensics.provenance import build_provenance, sha256_json_payload

# AAPM TG-142 tolerance & early predictive warning thresholds (IMRT/VMAT tier)
TG142_LIMITS = {
    "dlg_early_warning_mm": 0.25,
    "dlg_action_mm": 0.50,
    "mlc_rms_early_warning_mm": 0.35,
    "mlc_rms_action_mm": 0.70,
    "gantry_sag_action_mm": 1.00,
    "output_early_warning_pct": 1.50,
    "output_action_pct": 2.00,
}


class HypothesisAuditEntry(BaseModel):
    """Numeric falsification record for a single competing root-cause hypothesis."""

    model_config = ConfigDict(extra="forbid")

    hypothesis: str
    status: Literal["SUPPORTED", "FALSIFIED"]
    evidence_summary: str
    mean_delta_bic_vs_null: float
    mean_snr: float


class ServiceWorkOrder(BaseModel):
    """Structured biomedical / field-service engineering work order."""

    model_config = ConfigDict(extra="forbid")

    work_order_id: str
    machine_id: str
    urgency: Literal["IMMEDIATE_LOCKOUT", "URGENT_SAME_DAY", "SCHEDULED_PM", "MONITOR"]
    isolated_subsystems: list[str]
    quantitative_findings: dict[str, float | str]
    recommended_procedures: list[str]
    verification_tests_post_repair: list[str]


class CriticVerifierGateResult(BaseModel):
    """Independent Critic-Verifier audit checks over the orchestrator outputs."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks_run: int
    checks_passed: int
    findings: list[str]
    gate_sha256: str


class FleetIncidentLedger(BaseModel):
    """Complete closed-loop forensic incident report (JSON + Markdown exportable)."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str
    machine_id: str
    tg142_trends: list[TG142TrendSeries]
    hypothesis_audit: list[HypothesisAuditEntry]
    representative_diagnosis: CausalFaultDiagnosis
    triage_queue: list[PatientTriageRecord]
    service_work_order: ServiceWorkOrder
    critic_gate: CriticVerifierGateResult
    provenance: ProvenanceRecord
    markdown_report: str = Field(default="")


def generate_tg142_fleet_trends(
    n_days: int = 30,
    start_date_iso: str = "2026-10-25",
    seed: int = 42,
) -> list[TG142TrendSeries]:
    """Generate a 30-day 3-linac TG-142/MPC telemetry drift benchmark.

    - `TrueBeam-01` (Failing): progressive Bank B Leaf #34 viscous drag + systematic DLG
      drift triggering early predictive warning around Day 16 and TG-142 action breach
      around Day 23.
    - `TrueBeam-02` (Stable): well within all AAPM TG-142 tolerances.
    - `TrueBeam-03` (Mild Output Drift): slight linear ion-chamber drift approaching
      early warning late in the month.
    """
    rng = np.random.default_rng(seed)
    start = date.fromisoformat(start_date_iso)
    days = list(range(1, n_days + 1))
    dates = [(start + timedelta(days=d - 1)).isoformat() for d in days]
    d_arr = np.arange(n_days, dtype=np.float64)

    trends: list[TG142TrendSeries] = []
    configs = [
        ("TrueBeam-01", "degrading_mlc_dlg"),
        ("TrueBeam-02", "nominal_stable"),
        ("TrueBeam-03", "mild_output_drift"),
    ]
    for machine_id, mode in configs:
        if mode == "degrading_mlc_dlg":
            # Progressive degradation after day 10
            ramp = np.maximum(0.0, d_arr - 9.0) / max(n_days - 10.0, 1.0)
            dlg = 0.04 + 0.62 * (ramp**1.35) + rng.normal(0.0, 0.025, n_days)
            mlc_rms = 0.11 + 0.88 * (ramp**1.45) + rng.normal(0.0, 0.025, n_days)
            sag = 0.24 + 0.35 * ramp + rng.normal(0.0, 0.02, n_days)
            out_drift = 0.15 + 0.55 * ramp + rng.normal(0.0, 0.12, n_days)
            epid = 0.45 + 0.75 * ramp + rng.normal(0.0, 0.06, n_days)
        elif mode == "nominal_stable":
            dlg = 0.03 + rng.normal(0.0, 0.025, n_days)
            mlc_rms = 0.10 + rng.normal(0.0, 0.02, n_days)
            sag = 0.22 + rng.normal(0.0, 0.02, n_days)
            out_drift = rng.normal(0.05, 0.15, n_days)
            epid = 0.42 + rng.normal(0.0, 0.05, n_days)
        else:
            ramp = d_arr / max(n_days - 1.0, 1.0)
            dlg = 0.06 + 0.05 * ramp + rng.normal(0.0, 0.025, n_days)
            mlc_rms = 0.13 + 0.06 * ramp + rng.normal(0.0, 0.02, n_days)
            sag = 0.28 + 0.10 * ramp + rng.normal(0.0, 0.02, n_days)
            out_drift = 0.10 + 1.55 * ramp + rng.normal(0.0, 0.12, n_days)
            epid = 0.48 + 0.22 * ramp + rng.normal(0.0, 0.05, n_days)

        mlc_rms = np.clip(mlc_rms, 0.02, 3.0)
        sag = np.clip(sag, 0.02, 3.0)
        epid = np.clip(epid, 0.1, 5.0)

        # Detect early warning (first day any metric exceeds early warning threshold)
        early_day: int | None = None
        action_day: int | None = None
        for idx in range(n_days):
            if early_day is None and (
                abs(dlg[idx]) >= TG142_LIMITS["dlg_early_warning_mm"]
                or mlc_rms[idx] >= TG142_LIMITS["mlc_rms_early_warning_mm"]
                or abs(out_drift[idx]) >= TG142_LIMITS["output_early_warning_pct"]
            ):
                early_day = days[idx]
            if action_day is None and (
                abs(dlg[idx]) >= TG142_LIMITS["dlg_action_mm"]
                or mlc_rms[idx] >= TG142_LIMITS["mlc_rms_action_mm"]
                or abs(out_drift[idx]) >= TG142_LIMITS["output_action_pct"]
            ):
                action_day = days[idx]

        prov = build_provenance(
            record_id=f"tg142:{machine_id}",
            provenance_tag="SYNTHETIC_TG142_FLEET_DRIFT",
            dataset_source="linac-forensics 30-day AAPM TG-142 / MPC Drift Benchmark",
            sha256_inputs={"config": sha256_json_payload({"machine": machine_id, "mode": mode, "seed": seed})},
            notes=f"30-day synthetic TG-142/MPC QA series ({mode}).",
        )
        trends.append(
            TG142TrendSeries(
                machine_id=machine_id,
                days=days,
                dates_iso=dates,
                dlg_offset_mm=[round(float(v), 4) for v in dlg],
                mlc_max_rms_mm=[round(float(v), 4) for v in mlc_rms],
                gantry_sag_amp_mm=[round(float(v), 4) for v in sag],
                output_drift_pct=[round(float(v), 4) for v in out_drift],
                epid_uniformity_pct=[round(float(v), 4) for v in epid],
                early_warning_day=early_day,
                action_level_breach_day=action_day,
                provenance=prov,
            )
        )
    return trends


def generate_treatment_week_cohort(
    machine_id: str = "TrueBeam-01",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build a 6-patient treatment week cohort on a linac experiencing coupled MLC drag + DLG drift."""
    cohort_specs = [
        # (patient_id, plan_id, site, fractions, dlg_mm, leaf_lag_ms, sag_sin, mu_drift_pct, severity_mod)
        ("PT-HN-001", "VMAT_HN_SIB_70Gy", "Head & Neck SIB (OpenKBP pt_1)", 5, +0.62, 42.0, 0.25, +0.4, 1.0),
        ("PT-HN-002", "VMAT_NPC_ReIrrad", "Nasopharynx Re-Irradiation", 5, +0.68, 44.0, 0.30, +0.5, 1.15),
        ("PT-SBRT-003", "VMAT_Spine_SBRT", "Thoracic Spine SBRT", 3, -0.78, 45.0, 0.20, -0.3, 1.25),
        ("PT-PR-004", "VMAT_Prostate_78Gy", "Prostate + Seminal Vesicles", 5, +0.42, 38.0, 0.18, +0.2, 0.65),
        ("PT-LG-005", "VMAT_Lung_60Gy", "Stage III NSCLC", 4, +0.48, 40.0, 0.22, +0.3, 0.75),
        ("PT-BR-006", "VMAT_Breast_40Gy", "Whole Breast Hypofrac", 5, +0.22, 24.0, 0.12, +0.1, 0.35),
    ]
    records: list[dict[str, Any]] = []
    for idx, (pid, plan_id, site, fx, dlg, lag, sag, mu_d, _) in enumerate(cohort_specs):
        log = generate_synthetic_vmat_log(
            log_id=f"{machine_id}_{pid}_WK1",
            machine_id=machine_id,
            patient_id=pid,
            plan_id=plan_id,
            n_snapshots=180,
            failing_bank="B",
            failing_leaf_index=34,
            leaf_drag_ms=lag,
            dlg_offset_mm=dlg,
            gantry_sag_sin_mm=sag,
            gantry_sag_cos_mm=0.05,
            mu_drift_pct=mu_d,
            noise_sigma_mm=0.025,
            seed=seed + idx * 7,
        )
        records.append(
            {
                "patient_id": pid,
                "plan_id": plan_id,
                "treatment_site": site,
                "fractions_affected": fx,
                "log": log,
            }
        )
    return records


def _score_patient_triage(
    patient_meta: dict[str, Any],
    impact: PerturbativeDoseImpact,
    machine_id: str,
) -> PatientTriageRecord:
    """Convert 3D perturbative ΔDVH results into a prioritized clinical triage record."""
    shifts = {s.roi_name: s for s in impact.structure_shifts}
    ptv = shifts.get("PTV70") or next(
        (s for s in impact.structure_shifts if s.roi_type == "PTV"),
        impact.structure_shifts[0],
    )
    oars = [s for s in impact.structure_shifts if s.roi_type == "OAR"]
    crit_oar = max(oars, key=lambda s: s.delta_d003cc_gy) if oars else ptv

    # Clinical risk score in [0, 100]: combines |PTV ΔD95%|, OAR ΔD0.03cc (Gy), fractions affected
    ptv_term = min(50.0, 16.0 * abs(ptv.delta_d95_pct))
    oar_term = min(40.0, 15.0 * max(0.0, crit_oar.delta_d003cc_gy))
    fx_term = min(10.0, 2.0 * float(patient_meta["fractions_affected"]))
    risk_score = float(np.clip(ptv_term + oar_term + fx_term, 0.0, 100.0))

    if risk_score >= 65.0 or abs(ptv.delta_d95_pct) >= 2.5 or crit_oar.delta_d003cc_gy >= 1.8:
        priority: Literal["CRITICAL", "HIGH", "MODERATE", "ROUTINE"] = "CRITICAL"
        action = (
            "Hold next fraction pending QMP 3D dose accumulation review; evaluate adaptive replanning "
            "or compensatory fraction adjustment."
        )
    elif risk_score >= 42.0 or abs(ptv.delta_d95_pct) >= 1.6 or crit_oar.delta_d003cc_gy >= 1.0:
        priority = "HIGH"
        action = (
            "Priority QMP physicist chart audit & patient-specific 3D QA verification prior to next fraction."
        )
    elif risk_score >= 22.0:
        priority = "MODERATE"
        action = "Log cumulative delivery deviation in patient chart; verify on post-repair fraction."
    else:
        priority = "ROUTINE"
        action = "Within clinical dosimetric tolerance; no treatment interruption required."

    return PatientTriageRecord(
        patient_id=str(patient_meta["patient_id"]),
        plan_id=str(patient_meta["plan_id"]),
        machine_id=machine_id,
        treatment_site=str(patient_meta["treatment_site"]),
        fractions_affected=int(patient_meta["fractions_affected"]),
        gamma_2d_3pct_2mm_pass_pct=impact.gamma_2d_3pct_2mm_pass_pct,
        delta_d95_ptv_pct=round(ptv.delta_d95_pct, 3),
        delta_d003cc_oar_gy=round(crit_oar.delta_d003cc_gy, 3),
        critical_oar_name=crit_oar.roi_name,
        clinical_risk_score=round(risk_score, 2),
        triage_priority=priority,
        gamma_blindspot=impact.gamma_blindspot_detected,
        recommended_action=action,
    )


def _run_critic_verifier_gate(
    hypothesis_audit: list[HypothesisAuditEntry],
    diagnosis: CausalFaultDiagnosis,
    triage_queue: list[PatientTriageRecord],
    work_order: ServiceWorkOrder,
    tg142_trends: list[TG142TrendSeries],
) -> CriticVerifierGateResult:
    """Independent Critic-Verifier Gate auditing physics consistency, provenance, and ordering."""
    findings: list[str] = []
    checks_run = 0
    checks_passed = 0

    def _check(condition: bool, ok_msg: str, fail_msg: str) -> None:
        nonlocal checks_run, checks_passed
        checks_run += 1
        if condition:
            checks_passed += 1
            findings.append(f"PASS: {ok_msg}")
        else:
            findings.append(f"FAIL: {fail_msg}")

    # 1. Provenance integrity & research disclaimer present
    _check(
        bool(diagnosis.provenance.sha256_inputs)
        and "NOT A MEDICAL DEVICE" in diagnosis.provenance.research_notice,
        "Provenance SHA-256 inputs and Research Software notice verified.",
        "Missing SHA-256 provenance inputs or Research Software disclaimer.",
    )

    # 2. Hypothesis falsification matches numeric causal diagnosis
    supported = {h.hypothesis for h in hypothesis_audit if h.status == "SUPPORTED"}
    _check(
        supported.issuperset(set(diagnosis.active_faults)),
        f"Hypothesis audit consistently supports all active faults ({sorted(supported)}).",
        f"Hypothesis audit mismatch: supported={supported}, active={diagnosis.active_faults}.",
    )

    # 3. Triage queue strictly sorted descending by clinical_risk_score
    scores = [r.clinical_risk_score for r in triage_queue]
    _check(
        scores == sorted(scores, reverse=True) and len(triage_queue) > 0,
        f"Patient triage queue ({len(triage_queue)} patients) strictly prioritized by risk score.",
        "Patient triage queue is empty or not monotonically sorted by clinical risk score.",
    )

    # 4. Work order isolates hardware subsystems whenever faults are active
    _check(
        (len(diagnosis.active_faults) == 0) or (len(work_order.isolated_subsystems) > 0),
        f"Service work order isolates {len(work_order.isolated_subsystems)} failing subsystem(s).",
        "Active faults detected without isolated hardware subsystems in Service Work Order.",
    )

    # 5. TG-142 early warning precedes or equals action breach on degrading machines
    tg_valid = all(
        (t.action_level_breach_day is None)
        or (t.early_warning_day is not None and t.early_warning_day <= t.action_level_breach_day)
        for t in tg142_trends
    )
    _check(
        tg_valid,
        "TG-142 early predictive warning precedes or coincides with every action-level breach.",
        "TG-142 early warning logic violated (breach occurred without prior early warning).",
    )

    # 6. Variance explained consistency check
    _check(
        (len(diagnosis.active_faults) == 0) or (diagnosis.variance_explained_pct >= 50.0),
        f"Causal fault model explains {diagnosis.variance_explained_pct:.1f}% of telemetry error variance.",
        f"Causal fault model variance explained too low ({diagnosis.variance_explained_pct:.1f}%).",
    )

    gate_payload = {
        "checks_run": checks_run,
        "checks_passed": checks_passed,
        "findings": findings,
        "work_order_id": work_order.work_order_id,
    }
    return CriticVerifierGateResult(
        passed=(checks_passed == checks_run),
        checks_run=checks_run,
        checks_passed=checks_passed,
        findings=findings,
        gate_sha256=sha256_json_payload(gate_payload),
    )


def _render_markdown_ledger(ledger: FleetIncidentLedger) -> str:
    """Render a clean, physicist-ready Markdown incident & triage ledger."""
    wo = ledger.service_work_order
    lines = [
        f"# Linac Forensics Autonomous Incident Ledger — `{ledger.incident_id}`",
        "",
        "> **Research Software — NOT A MEDICAL DEVICE.** For physics research and QA benchmarking only.",
        "",
        f"- **Target Linac:** `{ledger.machine_id}`",
        f"- **Primary Root Cause:** `{ledger.representative_diagnosis.primary_root_cause}`",
        f"- **Isolated Hardware Subsystems:** {', '.join(wo.isolated_subsystems) or 'None'}",
        f"- **Service Urgency:** `{wo.urgency}`",
        f"- **Critic-Verifier Gate:** `{'PASSED' if ledger.critic_gate.passed else 'FAILED'}` "
        f"({ledger.critic_gate.checks_passed}/{ledger.critic_gate.checks_run} checks)",
        "",
        "## 1. Competing Fault Hypothesis Falsification (BIC / Matched-Filter)",
        "",
        "| Hypothesis | Status | Mean ΔBIC vs Null | Mean SNR | Numerical Evidence |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for h in ledger.hypothesis_audit:
        lines.append(
            f"| `{h.hypothesis}` | **{h.status}** | {h.mean_delta_bic_vs_null:+.1f} | "
            f"{h.mean_snr:.1f} | {h.evidence_summary} |"
        )

    lines.extend(
        [
            "",
            "## 2. Prioritized Clinical Patient-Impact Triage Queue",
            "",
            "| Priority | Patient | Site | Fx | 2D γ (3%/2mm) | PTV ΔD95 (%) | Crit OAR ΔD0.03cc (Gy) | Blindspot | Risk |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | :---: | ---: |",
        ]
    )
    for r in ledger.triage_queue:
        lines.append(
            f"| **{r.triage_priority}** | `{r.patient_id}` | {r.treatment_site} | {r.fractions_affected} | "
            f"{r.gamma_2d_3pct_2mm_pass_pct:.1f}% | {r.delta_d95_ptv_pct:+.2f}% | "
            f"{r.delta_d003cc_oar_gy:+.2f} ({r.critical_oar_name}) | "
            f"{'YES' if r.gamma_blindspot else 'no'} | {r.clinical_risk_score:.1f} |"
        )

    lines.extend(
        [
            "",
            f"## 3. Service Engineering Work Order (`{wo.work_order_id}`)",
            "",
            "**Recommended Service Procedures:**",
        ]
    )
    for proc in wo.recommended_procedures:
        lines.append(f"- {proc}")
    lines.append("")
    lines.append("**Post-Repair Physics Verification Tests (AAPM TG-142 / TG-218):**")
    for test in wo.verification_tests_post_repair:
        lines.append(f"- {test}")

    lines.extend(["", "## 4. Critic-Verifier Gate Audit", ""])
    for f in ledger.critic_gate.findings:
        lines.append(f"- {f}")
    lines.append(f"- Gate SHA-256: `{ledger.critic_gate.gate_sha256}`")
    return "\n".join(lines) + "\n"


def run_fleet_triage_investigation(
    anatomy: PatientAnatomyVolume,
    cohort: list[dict[str, Any]] | None = None,
    tg142_trends: list[TG142TrendSeries] | None = None,
    machine_id: str = "TrueBeam-01",
    incident_id: str = "INC-TB01-WK47",
    output_dir: str | Path | None = None,
) -> FleetIncidentLedger:
    """Execute the full Autonomous Fleet Physicist & Incident-Triage Agent Harness."""
    trends = tg142_trends if tg142_trends is not None else generate_tg142_fleet_trends()
    records = cohort if cohort is not None else generate_treatment_week_cohort(machine_id=machine_id)

    diagnoses: list[CausalFaultDiagnosis] = []
    triage_records: list[PatientTriageRecord] = []

    for rec in records:
        log: TrajectoryLogTensor = rec["log"]
        diag = diagnose_causal_faults(log)
        diagnoses.append(diag)

        impact = compute_perturbative_dose_replay(log, anatomy, n_sectors=8)
        assert isinstance(impact, PerturbativeDoseImpact)
        triage_records.append(_score_patient_triage(rec, impact, machine_id=log.machine_id))

    # Sort patient triage queue strictly descending by clinical risk score
    triage_records.sort(key=lambda r: r.clinical_risk_score, reverse=True)

    # Aggregate hypothesis falsification across the treatment week
    canonical_faults = [
        "MLC_LEAF_MOTOR_FRICTION",
        "SYSTEMATIC_DLG_OFFSET",
        "GANTRY_GRAVITY_SAG_HARMONIC",
        "DOSE_RATE_ION_CHAMBER_DRIFT",
    ]
    hyp_audit: list[HypothesisAuditEntry] = []
    for f_name in canonical_faults:
        matched_hyps = [
            h for d in diagnoses for h in d.hypotheses if h.fault_type == f_name
        ]
        n_det = sum(1 for h in matched_hyps if h.detected)
        mean_dbic = float(np.mean([h.delta_bic_vs_null for h in matched_hyps])) if matched_hyps else 0.0
        mean_snr = float(np.mean([h.matched_filter_snr for h in matched_hyps])) if matched_hyps else 0.0
        supported = n_det >= max(1, len(diagnoses) // 2)
        if supported:
            ev = f"Detected in {n_det}/{len(diagnoses)} logs (mean ΔBIC={mean_dbic:+.1f}, SNR={mean_snr:.1f})."
        else:
            ev = f"Falsified numerically ({n_det}/{len(diagnoses)} detections; mean ΔBIC={mean_dbic:+.1f})."
        hyp_audit.append(
            HypothesisAuditEntry(
                hypothesis=f_name,
                status="SUPPORTED" if supported else "FALSIFIED",
                evidence_summary=ev,
                mean_delta_bic_vs_null=round(mean_dbic, 2),
                mean_snr=round(mean_snr, 2),
            )
        )

    # Pick most informative / representative diagnosis (highest variance explained with active faults)
    rep_diag = max(diagnoses, key=lambda d: (len(d.active_faults), d.variance_explained_pct))

    isolated_subsystems: list[str] = []
    procedures: list[str] = []
    if rep_diag.flagged_leaves:
        top = rep_diag.flagged_leaves[0]
        isolated_subsystems.append(
            f"MLC Bank {top.bank} Leaf #{top.leaf_index} Motor / T-Nut (τ={top.velocity_lag_ms:.1f} ms)"
        )
        procedures.append(
            f"Replace MLC drive motor and lubricate/replace T-nut on Bank {top.bank} Leaf #{top.leaf_index}; "
            "run Varian Picket Fence & leaf-speed stress test."
        )
    if "SYSTEMATIC_DLG_OFFSET" in rep_diag.active_faults:
        isolated_subsystems.append(
            f"MLC Bank Calibration / DLG ({rep_diag.estimated_dlg_offset_mm:+.3f} mm)"
        )
        procedures.append(
            "Re-initialize MLC banks, recalibrate gap zero-offset, and measure sweeping-gap DLG with Farmer/microDiamond chamber."
        )
    if "GANTRY_GRAVITY_SAG_HARMONIC" in rep_diag.active_faults:
        isolated_subsystems.append(
            f"Gantry Head / Carriage Flex ({rep_diag.estimated_gantry_sag_amp_mm:.2f} mm harmonic)"
        )
        procedures.append("Inspect gantry bearings, MLC carriage rails, and perform 8-angle Winston-Lutz test.")
    if "DOSE_RATE_ION_CHAMBER_DRIFT" in rep_diag.active_faults:
        isolated_subsystems.append(
            f"Monitor Ion-Chamber Gain ({rep_diag.estimated_mu_drift_pct:+.2f}%)"
        )
        procedures.append("Perform TG-51 reference dosimetry and recalibrate primary/secondary monitor chamber gain.")

    if not procedures:
        procedures.append("No hardware fault isolated; maintain standard TG-142 daily/monthly QA schedule.")

    critical_count = sum(1 for r in triage_records if r.triage_priority == "CRITICAL")
    if critical_count >= 1:
        urgency: Literal["IMMEDIATE_LOCKOUT", "URGENT_SAME_DAY", "SCHEDULED_PM", "MONITOR"] = (
            "IMMEDIATE_LOCKOUT"
        )
    elif rep_diag.active_faults:
        urgency = "URGENT_SAME_DAY"
    else:
        urgency = "MONITOR"

    work_order = ServiceWorkOrder(
        work_order_id=f"WO-{incident_id}",
        machine_id=machine_id,
        urgency=urgency,
        isolated_subsystems=isolated_subsystems,
        quantitative_findings={
            "primary_root_cause": rep_diag.primary_root_cause,
            "estimated_dlg_offset_mm": rep_diag.estimated_dlg_offset_mm,
            "estimated_gantry_sag_amp_mm": rep_diag.estimated_gantry_sag_amp_mm,
            "estimated_mu_drift_pct": rep_diag.estimated_mu_drift_pct,
            "variance_explained_pct": rep_diag.variance_explained_pct,
            "critical_patients_flagged": critical_count,
        },
        recommended_procedures=procedures,
        verification_tests_post_repair=[
            "AAPM TG-142 Dynamic MLC Picket Fence (< 0.5 mm max error)",
            "Sweeping-gap Dosimetric Leaf Gap (DLG) verification (±0.15 mm baseline)",
            "AAPM TG-218 3D Perturbative DVH & 2%/1mm Gamma End-to-End VMAT Verification",
            "Varian Machine Performance Check (MPC) Beam & Geometry Baseline Update",
        ],
    )

    critic_gate = _run_critic_verifier_gate(
        hypothesis_audit=hyp_audit,
        diagnosis=rep_diag,
        triage_queue=triage_records,
        work_order=work_order,
        tg142_trends=trends,
    )

    provenance = build_provenance(
        record_id=f"incident:{incident_id}",
        provenance_tag="COMPOSITE_OPENKBP_PLUS_SYNTHETIC_LOG",
        dataset_source=f"{anatomy.anatomy_source} + Multi-Patient Synthetic Fleet Telemetry",
        sha256_inputs={
            **anatomy.sha256_inputs,
            "critic_gate": critic_gate.gate_sha256,
        },
        notes=f"Autonomous Fleet Physicist triage across {len(triage_records)} patients & {len(trends)} linacs.",
    )

    ledger = FleetIncidentLedger(
        incident_id=incident_id,
        machine_id=machine_id,
        tg142_trends=trends,
        hypothesis_audit=hyp_audit,
        representative_diagnosis=rep_diag,
        triage_queue=triage_records,
        service_work_order=work_order,
        critic_gate=critic_gate,
        provenance=provenance,
    )
    ledger.markdown_report = _render_markdown_ledger(ledger)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{incident_id}_ledger.json").write_text(
            json.dumps(ledger.model_dump(exclude={"markdown_report"}), indent=2) + "\n",
            encoding="utf-8",
        )
        (out / f"{incident_id}_report.md").write_text(ledger.markdown_report, encoding="utf-8")

    return ledger
