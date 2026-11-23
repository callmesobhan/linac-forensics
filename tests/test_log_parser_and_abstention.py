"""Tests for Varian TrueBeam .bin parser/writer, JSON/CSV ingestion, and hard abstention gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from linac_forensics.ir import AbstentionCode, AbstentionError, ControlPointPlan
from linac_forensics.log_parser import (
    generate_synthetic_vmat_log,
    parse_telemetry_csv,
    parse_telemetry_json,
    parse_varian_trajectory_bin,
    validate_trajectory_tensor,
    write_varian_trajectory_bin,
)


@pytest.mark.parametrize("version", ["3.0", "4.0", "5.0"])
def test_varian_bin_roundtrip_versions(tmp_path: Path, version: str) -> None:
    """Verify exact float32 round-trip fidelity for Varian .bin v3.0, v4.0, and v5.0."""
    orig = generate_synthetic_vmat_log(
        log_id=f"ROUNDTRIP_V{version}",
        n_snapshots=80,
        header_version=version,  # type: ignore[arg-type]
        failing_bank="B",
        failing_leaf_index=34,
        leaf_drag_ms=42.0,
        dlg_offset_mm=0.45,
        gantry_sag_sin_mm=0.8,
        mu_drift_pct=1.5,
        noise_sigma_mm=0.02,
    )
    bin_path = tmp_path / f"sample_v{version}.bin"
    write_varian_trajectory_bin(orig, bin_path, header_version=version)  # type: ignore[arg-type]
    parsed = parse_varian_trajectory_bin(bin_path)

    assert parsed.header_version == version
    assert parsed.n_snapshots == orig.n_snapshots
    assert parsed.n_leaves_per_bank == orig.n_leaves_per_bank
    assert parsed.machine_id == orig.machine_id
    assert parsed.patient_id == orig.patient_id
    assert parsed.coordinate_system == "IEC61217"

    np.testing.assert_allclose(parsed.gantry_actual_deg, orig.gantry_actual_deg, atol=1e-4)
    np.testing.assert_allclose(parsed.mu_actual, orig.mu_actual, atol=1e-3)
    np.testing.assert_allclose(parsed.bank_a_actual_mm, orig.bank_a_actual_mm, atol=1e-3)
    np.testing.assert_allclose(parsed.bank_b_actual_mm, orig.bank_b_actual_mm, atol=1e-3)
    np.testing.assert_allclose(parsed.jaw_actual_mm["Y1"], orig.jaw_actual_mm["Y1"], atol=1e-3)
    assert "trajectory_bin" in parsed.provenance.sha256_inputs


def test_json_and_csv_vendor_neutral_ingestion(tmp_path: Path) -> None:
    """Verify clean JSON and CSV vendor-neutral telemetry ingestion."""
    orig = generate_synthetic_vmat_log(
        log_id="NEUTRAL_01", n_snapshots=25, n_leaves_per_bank=8, dlg_offset_mm=0.3
    )
    json_path = tmp_path / "telemetry.json"
    json_path.write_text(json.dumps(orig.model_dump()), encoding="utf-8")
    parsed_json = parse_telemetry_json(json_path)
    assert parsed_json.n_snapshots == 25
    np.testing.assert_allclose(parsed_json.bank_a_actual_mm, orig.bank_a_actual_mm, atol=1e-6)

    # Export minimal CSV with 4 leaves per bank
    csv_path = tmp_path / "telemetry.csv"
    n_l = 4
    headers = [
        "timestamp_ms",
        "gantry_exp",
        "gantry_act",
        "collimator_exp",
        "collimator_act",
        "mu_exp",
        "mu_act",
        "beam_hold",
    ]
    for i in range(1, n_l + 1):
        headers.extend([f"A{i}_exp", f"A{i}_act", f"B{i}_exp", f"B{i}_act"])

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for t in range(15):
            row: list[float | int] = [
                t * 20.0,
                10.0 + t * 5.0,
                10.0 + t * 5.0,
                0.0,
                0.0,
                float(t * 2.0),
                float(t * 2.02),
                0,
            ]
            for _ in range(n_l):
                row.extend([20.0, 20.2, -20.0, -20.2])
            writer.writerow(row)

    parsed_csv = parse_telemetry_csv(csv_path, log_id="CSV_TEST")
    assert parsed_csv.header_version == "CSV-1.0"
    assert parsed_csv.n_leaves_per_bank == 4
    assert parsed_csv.n_snapshots == 15


def test_corrupted_binary_log_gates(tmp_path: Path) -> None:
    """Verify CORRUPTED_BINARY_LOG abstention on bad signature, truncation, and CRC32 corruption."""
    orig = generate_synthetic_vmat_log(log_id="CRC_TEST", n_snapshots=30, header_version="5.0")
    bin_path = tmp_path / "valid_v5.bin"
    write_varian_trajectory_bin(orig, bin_path, header_version="5.0")
    raw = bytearray(bin_path.read_bytes())

    # 1. Corrupt CRC32 (flip one byte in snapshot payload)
    bad_crc = tmp_path / "bad_crc.bin"
    corrupted = bytearray(raw)
    corrupted[1800] ^= 0xFF
    bad_crc.write_bytes(bytes(corrupted))
    with pytest.raises(AbstentionError) as exc1:
        parse_varian_trajectory_bin(bad_crc)
    assert exc1.value.code == AbstentionCode.CORRUPTED_BINARY_LOG

    # 2. Truncated binary
    trunc = tmp_path / "trunc.bin"
    trunc.write_bytes(bytes(raw[:-100]))
    with pytest.raises(AbstentionError) as exc2:
        parse_varian_trajectory_bin(trunc)
    assert exc2.value.code == AbstentionCode.CORRUPTED_BINARY_LOG

    # 3. Invalid magic signature
    bad_sig = tmp_path / "bad_sig.bin"
    sig_bytes = bytearray(raw)
    sig_bytes[0:5] = b"NOPE!"
    bad_sig.write_bytes(bytes(sig_bytes))
    with pytest.raises(AbstentionError) as exc3:
        parse_varian_trajectory_bin(bad_sig)
    assert exc3.value.code == AbstentionCode.CORRUPTED_BINARY_LOG


def test_timestamp_desync_abstention() -> None:
    """Verify TIMESTAMP_DESYNC gate on non-monotonic timestamps and excessive jitter."""
    log = generate_synthetic_vmat_log(n_snapshots=30)
    bad_ts = list(log.timestamps_ms)
    bad_ts[10] = bad_ts[9] - 1.0
    broken = log.model_copy(update={"timestamps_ms": bad_ts})
    with pytest.raises(AbstentionError) as exc:
        validate_trajectory_tensor(broken)
    assert exc.value.code == AbstentionCode.TIMESTAMP_DESYNC

    jitter_ts = list(log.timestamps_ms)
    jitter_ts[15:] = [v + 40.0 for v in jitter_ts[15:]]
    broken2 = log.model_copy(update={"timestamps_ms": jitter_ts})
    with pytest.raises(AbstentionError) as exc2:
        validate_trajectory_tensor(broken2)
    assert exc2.value.code == AbstentionCode.TIMESTAMP_DESYNC


def test_mu_conservation_and_interlock_abstention() -> None:
    """Verify MU_CONSERVATION_VIOLATION and INTERLOCK_INCOMPLETE_DELIVERY gates."""
    log = generate_synthetic_vmat_log(n_snapshots=40)

    # 1. Decreasing cumulative MU -> MU_CONSERVATION_VIOLATION
    bad_mu = list(log.mu_actual)
    bad_mu[20] = bad_mu[19] - 2.0
    with pytest.raises(AbstentionError) as exc1:
        validate_trajectory_tensor(log.model_copy(update={"mu_actual": bad_mu}))
    assert exc1.value.code == AbstentionCode.MU_CONSERVATION_VIOLATION

    # 2. Gross 40% over-delivery -> MU_CONSERVATION_VIOLATION
    over_mu = [m * 1.40 for m in log.mu_expected]
    with pytest.raises(AbstentionError) as exc2:
        validate_trajectory_tensor(log.model_copy(update={"mu_actual": over_mu}))
    assert exc2.value.code == AbstentionCode.MU_CONSERVATION_VIOLATION

    # 3. Interlock abort halfway through delivery -> INTERLOCK_INCOMPLETE_DELIVERY
    abort_mu = [m if i < 20 else log.mu_expected[19] for i, m in enumerate(log.mu_expected)]
    abort_hold = [0 if i < 20 else 2 for i in range(40)]
    with pytest.raises(AbstentionError) as exc3:
        validate_trajectory_tensor(
            log.model_copy(update={"mu_actual": abort_mu, "beam_hold": abort_hold})
        )
    assert exc3.value.code == AbstentionCode.INTERLOCK_INCOMPLETE_DELIVERY


def test_iec61217_coordinate_integrity_abstention() -> None:
    """Verify UNMAPPED_IEC61217_COORDINATES gate on bad coordinate system, angle, or bank flip."""
    log = generate_synthetic_vmat_log(n_snapshots=30)

    with pytest.raises(AbstentionError) as exc1:
        validate_trajectory_tensor(log.model_copy(update={"coordinate_system": "VARIAN_NATIVE"}))
    assert exc1.value.code == AbstentionCode.UNMAPPED_IEC61217_COORDINATES

    bad_gantry = list(log.gantry_actual_deg)
    bad_gantry[5] = 410.0
    with pytest.raises(AbstentionError) as exc2:
        validate_trajectory_tensor(log.model_copy(update={"gantry_actual_deg": bad_gantry}))
    assert exc2.value.code == AbstentionCode.UNMAPPED_IEC61217_COORDINATES

    # Inverted sign convention (Bank A < Bank B by 40 mm) -> collision abstention
    with pytest.raises(AbstentionError) as exc3:
        validate_trajectory_tensor(
            log.model_copy(
                update={
                    "bank_a_actual_mm": log.bank_b_expected_mm,
                    "bank_b_actual_mm": log.bank_a_expected_mm,
                }
            )
        )
    assert exc3.value.code == AbstentionCode.UNMAPPED_IEC61217_COORDINATES


def test_control_point_plan_ir_validation() -> None:
    """Verify ControlPointPlan Pydantic v2 IR validation."""
    plan = ControlPointPlan(
        plan_id="PLAN_01",
        patient_id="pt_1",
        total_mu=320.0,
        n_leaves_per_bank=2,
        leaf_widths_mm=[5.0, 5.0],
        gantry_angles_deg=[0.0, 180.0],
        mu_weights=[0.0, 1.0],
        bank_a_planned_mm=[[20.0, 20.0], [18.0, 22.0]],
        bank_b_planned_mm=[[-20.0, -20.0], [-22.0, -18.0]],
    )
    assert plan.prescribed_dose_gy == 70.0
    with pytest.raises(ValueError):
        ControlPointPlan(
            plan_id="BAD",
            patient_id="pt_1",
            total_mu=100.0,
            n_leaves_per_bank=2,
            leaf_widths_mm=[5.0, -1.0],
            gantry_angles_deg=[0.0, 180.0],
            mu_weights=[0.0, 1.0],
            bank_a_planned_mm=[[10.0, 10.0], [10.0, 10.0]],
            bank_b_planned_mm=[[-10.0, -10.0], [-10.0, -10.0]],
        )
