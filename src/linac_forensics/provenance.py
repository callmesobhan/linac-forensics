"""Cryptographic provenance tracking and hard physics/telemetry abstention gates.

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from linac_forensics import __version__
from linac_forensics.ir import AbstentionCode, AbstentionError, ProvenanceRecord


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return hex SHA-256 digest of a file on disk."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_payload(payload: Any) -> str:
    """Return canonical deterministic SHA-256 digest for any JSON-serializable object."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return sha256_bytes(canonical)


def build_provenance(
    record_id: str,
    provenance_tag: Literal[
        "OPEN_PUBLIC_ANATOMY_OPENKBP_PT1",
        "OPEN_PUBLIC_ANATOMY_OPENKBP_PT2",
        "OPEN_PUBLIC_ANATOMY_OPENKBP_PT3",
        "OPEN_PUBLIC_ANATOMY_OPENKBP_COHORT",
        "SYNTHETIC_FAULT_INJECTION_BENCHMARK",
        "SYNTHETIC_TG142_FLEET_DRIFT",
        "VENDOR_TRAJECTORY_LOG",
        "EPID_THREE_WAY_FORENSICS",
        "COMPOSITE_OPENKBP_PLUS_SYNTHETIC_LOG",
    ],
    dataset_source: str,
    sha256_inputs: dict[str, str] | None = None,
    notes: str = "",
) -> ProvenanceRecord:
    """Construct a standardized ProvenanceRecord."""
    return ProvenanceRecord(
        record_id=record_id,
        provenance_tag=provenance_tag,
        dataset_source=dataset_source,
        sha256_inputs=sha256_inputs or {},
        engine_version=__version__,
        notes=notes,
    )


def validate_timestamps_monotonic(
    timestamps_ms: np.ndarray,
    sampling_interval_ms: float,
    max_jitter_ratio: float = 0.25,
) -> None:
    """Gate: enforce strictly monotonic 20–50 Hz telemetry timestamps without severe desync."""
    if timestamps_ms.ndim != 1 or timestamps_ms.size < 2:
        raise AbstentionError(
            AbstentionCode.TIMESTAMP_DESYNC,
            "Trajectory log requires at least 2 one-dimensional timestamp samples.",
            {"n_samples": int(timestamps_ms.size)},
        )
    if not np.all(np.isfinite(timestamps_ms)):
        raise AbstentionError(
            AbstentionCode.TIMESTAMP_DESYNC,
            "Non-finite (NaN/Inf) timestamps encountered in trajectory telemetry.",
        )
    diffs = np.diff(timestamps_ms)
    if np.any(diffs <= 0.0):
        bad_idx = int(np.argmax(diffs <= 0.0))
        raise AbstentionError(
            AbstentionCode.TIMESTAMP_DESYNC,
            f"Non-monotonic timestamp step detected at snapshot {bad_idx} -> {bad_idx + 1}.",
            {"bad_index": bad_idx, "dt_ms": float(diffs[bad_idx])},
        )
    rel_jitter = np.abs(diffs - sampling_interval_ms) / max(sampling_interval_ms, 1e-6)
    max_jitter = float(np.max(rel_jitter))
    if max_jitter > max_jitter_ratio:
        bad_idx = int(np.argmax(rel_jitter))
        raise AbstentionError(
            AbstentionCode.TIMESTAMP_DESYNC,
            (
                f"Clock jitter/dropout exceeds {max_jitter_ratio * 100:.0f}% tolerance "
                f"(observed {max_jitter * 100:.1f}% at snapshot {bad_idx})."
            ),
            {
                "bad_index": bad_idx,
                "observed_dt_ms": float(diffs[bad_idx]),
                "expected_dt_ms": float(sampling_interval_ms),
                "max_relative_jitter": max_jitter,
            },
        )


def validate_mu_conservation(
    mu_expected: np.ndarray,
    mu_actual: np.ndarray,
    beam_hold: np.ndarray,
    max_relative_discrepancy: float = 0.25,
    min_completion_fraction: float = 0.90,
) -> None:
    """Gate: enforce MU monotonicity, physical conservation, and complete delivery (no interlock abort)."""
    if (
        not np.all(np.isfinite(mu_expected))
        or not np.all(np.isfinite(mu_actual))
        or mu_expected.size != mu_actual.size
    ):
        raise AbstentionError(
            AbstentionCode.MU_CONSERVATION_VIOLATION,
            "Non-finite or length-mismatched MU channels.",
        )

    if np.any(mu_expected < -1e-6) or np.any(mu_actual < -1e-6):
        raise AbstentionError(
            AbstentionCode.MU_CONSERVATION_VIOLATION,
            "Negative cumulative monitor units (MU) detected.",
            {
                "min_mu_expected": float(np.min(mu_expected)),
                "min_mu_actual": float(np.min(mu_actual)),
            },
        )

    dmu_exp = np.diff(mu_expected)
    dmu_act = np.diff(mu_actual)
    if np.any(dmu_exp < -1e-4) or np.any(dmu_act < -1e-4):
        raise AbstentionError(
            AbstentionCode.MU_CONSERVATION_VIOLATION,
            "Cumulative MU decreased between consecutive snapshots (violates dose conservation).",
            {
                "min_dmu_expected": float(np.min(dmu_exp)),
                "min_dmu_actual": float(np.min(dmu_act)),
            },
        )

    final_exp = float(mu_expected[-1])
    final_act = float(mu_actual[-1])
    if final_exp <= 0.0:
        raise AbstentionError(
            AbstentionCode.MU_CONSERVATION_VIOLATION,
            "Expected total MU must be strictly positive.",
            {"final_mu_expected": final_exp},
        )

    # Check interlock / incomplete delivery before conservation drift check
    completion_ratio = final_act / final_exp
    interlock_fraction = float(np.mean(beam_hold >= 2))
    trailing_hold = bool(beam_hold.size >= 5 and np.all(beam_hold[-5:] >= 1))
    if completion_ratio < min_completion_fraction and (interlock_fraction > 0.0 or trailing_hold):
        raise AbstentionError(
            AbstentionCode.INTERLOCK_INCOMPLETE_DELIVERY,
            (
                f"Treatment delivery aborted under beam interlock/hold at "
                f"{completion_ratio * 100:.1f}% of prescribed MU ({final_act:.2f}/{final_exp:.2f} MU)."
            ),
            {
                "final_mu_expected": final_exp,
                "final_mu_actual": final_act,
                "completion_ratio": completion_ratio,
                "interlock_fraction": interlock_fraction,
            },
        )

    if interlock_fraction > 0.20:
        raise AbstentionError(
            AbstentionCode.INTERLOCK_INCOMPLETE_DELIVERY,
            f"Persistent beam interlock state across {interlock_fraction * 100:.1f}% of snapshots.",
            {"interlock_fraction": interlock_fraction},
        )

    rel_err = abs(final_act - final_exp) / final_exp
    if rel_err > max_relative_discrepancy:
        raise AbstentionError(
            AbstentionCode.MU_CONSERVATION_VIOLATION,
            (
                f"Total delivered MU diverges from planned MU by {rel_err * 100:.1f}% "
                f"(exceeds {max_relative_discrepancy * 100:.0f}% gross conservation gate)."
            ),
            {
                "final_mu_expected": final_exp,
                "final_mu_actual": final_act,
                "relative_error": rel_err,
            },
        )


def validate_iec61217_coordinates(
    coordinate_system: str,
    gantry_expected_deg: np.ndarray,
    gantry_actual_deg: np.ndarray,
    collimator_expected_deg: np.ndarray,
    collimator_actual_deg: np.ndarray,
    bank_a_expected_mm: np.ndarray,
    bank_a_actual_mm: np.ndarray,
    bank_b_expected_mm: np.ndarray,
    bank_b_actual_mm: np.ndarray,
    max_leaf_travel_mm: float = 200.0,
) -> None:
    """Gate: enforce IEC 61217 coordinate mapping, angular bounds, and physical MLC non-collision."""
    if coordinate_system.upper() != "IEC61217":
        raise AbstentionError(
            AbstentionCode.UNMAPPED_IEC61217_COORDINATES,
            f"Unsupported coordinate system '{coordinate_system}'; IEC61217 is mandatory.",
            {"coordinate_system": coordinate_system},
        )

    for label, arr in [
        ("gantry_expected_deg", gantry_expected_deg),
        ("gantry_actual_deg", gantry_actual_deg),
        ("collimator_expected_deg", collimator_expected_deg),
        ("collimator_actual_deg", collimator_actual_deg),
    ]:
        if not np.all(np.isfinite(arr)) or np.any(arr < -0.5) or np.any(arr > 360.5):
            raise AbstentionError(
                AbstentionCode.UNMAPPED_IEC61217_COORDINATES,
                f"Angle channel '{label}' outside IEC 61217 [0, 360] deg domain.",
                {"channel": label, "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr))},
            )

    for label, arr in [
        ("bank_a_expected_mm", bank_a_expected_mm),
        ("bank_a_actual_mm", bank_a_actual_mm),
        ("bank_b_expected_mm", bank_b_expected_mm),
        ("bank_b_actual_mm", bank_b_actual_mm),
    ]:
        if not np.all(np.isfinite(arr)):
            raise AbstentionError(
                AbstentionCode.UNMAPPED_IEC61217_COORDINATES,
                f"Non-finite MLC position in '{label}'.",
                {"channel": label},
            )
        if float(np.max(np.abs(arr))) > max_leaf_travel_mm:
            raise AbstentionError(
                AbstentionCode.UNMAPPED_IEC61217_COORDINATES,
                f"MLC leaf position in '{label}' exceeds ±{max_leaf_travel_mm:.0f} mm physical travel.",
                {"channel": label, "max_abs_mm": float(np.max(np.abs(arr)))},
            )

    # In IEC 61217 signed coordinates: Bank B is left (-X), Bank A is right (+X), so gap = A - B >= -3.0 mm
    # (allowing up to 3.0 mm physical calibration/fault overlap tolerance before flagging unmapped sign flip).
    min_exp_gap = float(np.min(bank_a_expected_mm - bank_b_expected_mm))
    min_act_gap = float(np.min(bank_a_actual_mm - bank_b_actual_mm))
    if min_exp_gap < -3.0 or min_act_gap < -5.0:
        raise AbstentionError(
            AbstentionCode.UNMAPPED_IEC61217_COORDINATES,
            (
                "Physical MLC bank collision or inverted Varian-to-IEC sign convention detected "
                f"(min expected gap={min_exp_gap:.2f} mm, min actual gap={min_act_gap:.2f} mm)."
            ),
            {"min_expected_gap_mm": min_exp_gap, "min_actual_gap_mm": min_act_gap},
        )
