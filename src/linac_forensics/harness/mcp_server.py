"""JSON-RPC 2.0 Stdio Model Context Protocol (MCP) Server for linac-forensics.

Research Software — NOT A MEDICAL DEVICE.

Exposes 6 strictly typed, provenance-gated medical physics forensics tools over
newline-delimited JSON-RPC 2.0 stdio transport:
  1. `linac_parse_trajectory_log`
  2. `linac_compute_perturbative_dvh`
  3. `linac_diagnose_causal_fault`
  4. `linac_compare_gamma_vs_dvh`
  5. `linac_triage_patient_fleet`
  6. `linac_run_sobol_sensitivity`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from linac_forensics import __version__
from linac_forensics.anatomy import (
    PatientAnatomyVolume,
    build_synthetic_phantom_anatomy,
    load_openkbp_patient,
)
from linac_forensics.causal_fault_model import diagnose_causal_faults
from linac_forensics.harness.agent_orchestrator import run_fleet_triage_investigation
from linac_forensics.ir import AbstentionError, PerturbativeDoseImpact, TrajectoryLogTensor
from linac_forensics.log_parser import (
    generate_synthetic_vmat_log,
    parse_telemetry_csv,
    parse_telemetry_json,
    parse_varian_trajectory_bin,
)
from linac_forensics.perturbative_dose import compute_perturbative_dose_replay
from linac_forensics.sobol_sensitivity import run_delivery_sobol_sensitivity

MCP_PROTOCOL_VERSION = "2024-11-05"

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "linac_parse_trajectory_log",
        "description": (
            "Parse and validate a Varian TrueBeam binary (.bin v3.0/v4.0/v5.0), JSON, or CSV "
            "trajectory log, enforcing monotonicity, MU conservation, and IEC 61217 gates."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to .bin, .json, or .csv log."},
                "format": {"type": "string", "enum": ["auto", "bin", "json", "csv"], "default": "auto"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "linac_compute_perturbative_dvh",
        "description": (
            "Run fast 2D fluence -> 3D perturbative dose replay on OpenKBP pt_1 or phantom anatomy "
            "and compute exact structure-specific ΔD95, ΔD0.03cc, ΔgEUD, ΔTCP/NTCP, and 2D/3D Gamma."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Optional trajectory log file path (.bin/.json/.csv)."},
                "anatomy_dir": {"type": "string", "description": "Optional OpenKBP patient directory (defaults to phantom if omitted)."},
                "synthetic_faults": {
                    "type": "object",
                    "description": "Optional synthetic fault injection parameters when file_path is omitted.",
                },
            },
        },
    },
    {
        "name": "linac_diagnose_causal_fault",
        "description": (
            "Perform physics-informed matched-filter & Bayesian BIC causal attribution separating "
            "MLC leaf viscous drag, systematic DLG offset, gantry gravity sag, and MU drift."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Optional path to .bin/.json/.csv trajectory log."},
                "synthetic_faults": {"type": "object", "description": "Optional synthetic fault config."},
            },
        },
    },
    {
        "name": "linac_compare_gamma_vs_dvh",
        "description": (
            "Sweep DLG/MLC fault magnitudes and quantify Nelms et al. (2011) 2D 3%/2mm gamma "
            "blindspots vs true 3D patient ΔDVH failures."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "anatomy_dir": {"type": "string", "description": "Optional OpenKBP patient directory."},
                "dlg_offsets_mm": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of DLG offsets (mm) to evaluate.",
                },
                "leaf_drag_ms": {"type": "number", "default": 30.0},
            },
        },
    },
    {
        "name": "linac_triage_patient_fleet",
        "description": (
            "Run the Autonomous Fleet Physicist & Incident-Triage Harness across a multi-patient "
            "treatment week and 30-day TG-142/MPC fleet trend series with Critic-Verifier Gate."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "anatomy_dir": {"type": "string", "description": "Optional OpenKBP patient directory."},
                "machine_id": {"type": "string", "default": "TrueBeam-01"},
                "incident_id": {"type": "string", "default": "INC-TB01-MCP"},
            },
        },
    },
    {
        "name": "linac_run_sobol_sensitivity",
        "description": (
            "Compute exact Saltelli–Jansen QMC global Sobol' sensitivity indices (S_1, S_T, 95% CIs) "
            "for PTV70 ΔD95 and SpinalCord ΔD0.03cc across electro-mechanical linac tolerances."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "anatomy_dir": {"type": "string", "description": "Optional OpenKBP patient directory."},
                "n_base_samples": {"type": "integer", "default": 128, "minimum": 16, "maximum": 2048},
                "seed": {"type": "integer", "default": 42},
            },
        },
    },
]


def _load_log_from_args(args: dict[str, Any]) -> TrajectoryLogTensor:
    file_path = args.get("file_path")
    if file_path:
        path = Path(file_path)
        fmt = str(args.get("format", "auto")).lower()
        if fmt == "auto":
            suffix = path.suffix.lower()
            fmt = "bin" if suffix == ".bin" else ("csv" if suffix == ".csv" else "json")
        if fmt == "bin":
            return parse_varian_trajectory_bin(path)
        if fmt == "csv":
            return parse_telemetry_csv(path)
        return parse_telemetry_json(path)

    sf = args.get("synthetic_faults") or {}
    return generate_synthetic_vmat_log(
        log_id=str(sf.get("log_id", "MCP_SYNTH_LOG")),
        n_snapshots=int(sf.get("n_snapshots", 180)),
        failing_bank=str(sf.get("failing_bank", "B")),  # type: ignore[arg-type]
        failing_leaf_index=int(sf.get("failing_leaf_index", 34)),
        leaf_drag_ms=float(sf.get("leaf_drag_ms", 0.0)),
        dlg_offset_mm=float(sf.get("dlg_offset_mm", 0.0)),
        gantry_sag_sin_mm=float(sf.get("gantry_sag_sin_mm", 0.0)),
        gantry_sag_cos_mm=float(sf.get("gantry_sag_cos_mm", 0.0)),
        mu_drift_pct=float(sf.get("mu_drift_pct", 0.0)),
        noise_sigma_mm=float(sf.get("noise_sigma_mm", 0.02)),
        seed=int(sf.get("seed", 42)),
    )


def _load_anatomy_from_args(args: dict[str, Any]) -> PatientAnatomyVolume:
    anatomy_dir = args.get("anatomy_dir")
    if anatomy_dir:
        return load_openkbp_patient(anatomy_dir)
    default_openkbp = Path(__file__).resolve().parents[3] / "data" / "openkbp" / "pt_1"
    if default_openkbp.exists():
        return load_openkbp_patient(default_openkbp)
    return build_synthetic_phantom_anatomy()


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a typed, provenance-gated linac-forensics MCP tool."""
    if name == "linac_parse_trajectory_log":
        tensor = _load_log_from_args(arguments)
        return {
            "log_id": tensor.log_id,
            "header_version": tensor.header_version,
            "machine_id": tensor.machine_id,
            "patient_id": tensor.patient_id,
            "plan_id": tensor.plan_id,
            "sampling_interval_ms": tensor.sampling_interval_ms,
            "n_snapshots": tensor.n_snapshots,
            "n_leaves_per_bank": tensor.n_leaves_per_bank,
            "coordinate_system": tensor.coordinate_system,
            "mu_expected_total": round(float(tensor.mu_expected[-1]), 4),
            "mu_actual_total": round(float(tensor.mu_actual[-1]), 4),
            "provenance": tensor.provenance.model_dump(),
        }

    if name == "linac_compute_perturbative_dvh":
        tensor = _load_log_from_args(arguments)
        anatomy = _load_anatomy_from_args(arguments)
        impact = compute_perturbative_dose_replay(tensor, anatomy, n_sectors=12)
        assert isinstance(impact, PerturbativeDoseImpact)
        return impact.model_dump()

    if name == "linac_diagnose_causal_fault":
        tensor = _load_log_from_args(arguments)
        diagnosis = diagnose_causal_faults(tensor)
        return diagnosis.model_dump()

    if name == "linac_compare_gamma_vs_dvh":
        anatomy = _load_anatomy_from_args(arguments)
        offsets = arguments.get("dlg_offsets_mm") or [-0.85, -0.60, -0.30, 0.0, 0.30, 0.60, 0.85]
        lag_ms = float(arguments.get("leaf_drag_ms", 30.0))
        cases: list[dict[str, Any]] = []
        for idx, dlg in enumerate(offsets):
            log = generate_synthetic_vmat_log(
                log_id=f"GAMMA_VS_DVH_{idx}",
                n_snapshots=150,
                dlg_offset_mm=float(dlg),
                leaf_drag_ms=lag_ms if abs(float(dlg)) > 1e-6 else 0.0,
                noise_sigma_mm=0.015 if abs(float(dlg)) > 1e-6 else 0.0,
                seed=100 + idx,
            )
            imp = compute_perturbative_dose_replay(log, anatomy, n_sectors=8)
            assert isinstance(imp, PerturbativeDoseImpact)
            shifts = {s.roi_name: s for s in imp.structure_shifts}
            ptv = shifts.get("PTV70") or imp.structure_shifts[0]
            cord = shifts.get("SpinalCord") or imp.structure_shifts[-1]
            cases.append(
                {
                    "dlg_offset_mm": float(dlg),
                    "gamma_2d_3pct_2mm_pass_pct": imp.gamma_2d_3pct_2mm_pass_pct,
                    "gamma_3d_2pct_1mm_pass_pct": imp.gamma_3d_2pct_1mm_pass_pct,
                    "ptv_delta_d95_pct": ptv.delta_d95_pct,
                    "cord_delta_d003cc_gy": cord.delta_d003cc_gy,
                    "gamma_blindspot_detected": imp.gamma_blindspot_detected,
                }
            )
        return {
            "anatomy_source": anatomy.anatomy_source,
            "n_cases": len(cases),
            "n_blindspots": sum(1 for c in cases if c["gamma_blindspot_detected"]),
            "cases": cases,
            "research_notice": "Research Software — NOT A MEDICAL DEVICE",
        }

    if name == "linac_triage_patient_fleet":
        anatomy = _load_anatomy_from_args(arguments)
        machine_id = str(arguments.get("machine_id", "TrueBeam-01"))
        incident_id = str(arguments.get("incident_id", "INC-TB01-MCP"))
        ledger = run_fleet_triage_investigation(
            anatomy=anatomy,
            machine_id=machine_id,
            incident_id=incident_id,
        )
        return ledger.model_dump(exclude={"markdown_report"})

    if name == "linac_run_sobol_sensitivity":
        anatomy = _load_anatomy_from_args(arguments)
        n_base = int(arguments.get("n_base_samples", 128))
        seed = int(arguments.get("seed", 42))
        reports = run_delivery_sobol_sensitivity(anatomy, n_base_samples=n_base, seed=seed)
        return {k: v.model_dump() for k, v in reports.items()}

    raise KeyError(f"Unknown MCP tool: {name}")


def handle_jsonrpc_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC 2.0 request and return the response dict (or None for notifications)."""
    req_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params") or {}

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "linac-forensics-mcp",
                    "version": __version__,
                    "notice": "Research Software — NOT A MEDICAL DEVICE",
                },
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOL_SPECS}}

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments") or {}
        try:
            payload = dispatch_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                    "structuredContent": payload,
                    "isError": False,
                },
            }
        except AbstentionError as exc:
            err_payload = exc.to_dict()
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(err_payload, indent=2)}],
                    "structuredContent": err_payload,
                    "isError": True,
                },
            }
        except KeyError as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": str(exc)},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": f"Tool execution error: {exc}"},
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    """Run the JSON-RPC 2.0 MCP server over stdio."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_jsonrpc_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
