"""Native Varian TrueBeam Trajectory Log (.bin v3.0/v4.0/v5.0) & JSON/CSV telemetry parser/writer.

Research Software — NOT A MEDICAL DEVICE.

Implements binary encoding/decoding matching the Varian TrueBeam Trajectory Log
specification (1024-byte header, 560-byte subbeam records, 20–50 Hz expected/actual
float32 snapshot records, IEC 61217 axis scaling, optional v5.0 CRC32 trailer) along
with vendor-neutral JSON/CSV ingestion and a controlled synthetic VMAT fault-injection
generator tagged explicitly in the provenance ledger.
"""

from __future__ import annotations

import csv
import json
import struct
import zlib
from pathlib import Path
from typing import Literal

import numpy as np

from linac_forensics.ir import (
    AbstentionCode,
    AbstentionError,
    TrajectoryLogTensor,
)
from linac_forensics.provenance import (
    build_provenance,
    sha256_bytes,
    sha256_file,
    validate_iec61217_coordinates,
    validate_mu_conservation,
    validate_timestamps_monotonic,
)

HEADER_SIGNATURE = b"VOSTL"
HEADER_SIZE = 1024
SUBBEAM_SIZE = 560
SUPPORTED_BIN_VERSIONS = {"3.0", "4.0", "5.0"}

# Varian TrueBeam standard axis enumerations
AXIS_COLL_RTN = 0
AXIS_GANTRY_RTN = 1
AXIS_Y1 = 2
AXIS_Y2 = 3
AXIS_X1 = 4
AXIS_X2 = 5
AXIS_MU = 40
AXIS_BEAM_HOLD = 41
AXIS_MLC = 50
STANDARD_AXES = [
    AXIS_COLL_RTN,
    AXIS_GANTRY_RTN,
    AXIS_Y1,
    AXIS_Y2,
    AXIS_X1,
    AXIS_X2,
    AXIS_MU,
    AXIS_BEAM_HOLD,
    AXIS_MLC,
]


def _pack_fixed_str(text: str, length: int) -> bytes:
    raw = text.encode("utf-8")[:length]
    return raw + b"\x00" * (length - len(raw))


def _unpack_fixed_str(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()


def validate_trajectory_tensor(tensor: TrajectoryLogTensor) -> TrajectoryLogTensor:
    """Run all hard physics & coordinate integrity abstention gates on a TrajectoryLogTensor."""
    ts = np.asarray(tensor.timestamps_ms, dtype=np.float64)
    validate_timestamps_monotonic(ts, tensor.sampling_interval_ms)

    mu_exp = np.asarray(tensor.mu_expected, dtype=np.float64)
    mu_act = np.asarray(tensor.mu_actual, dtype=np.float64)
    beam_hold = np.asarray(tensor.beam_hold, dtype=np.int32)
    validate_mu_conservation(mu_exp, mu_act, beam_hold)

    validate_iec61217_coordinates(
        coordinate_system=tensor.coordinate_system,
        gantry_expected_deg=np.asarray(tensor.gantry_expected_deg, dtype=np.float64),
        gantry_actual_deg=np.asarray(tensor.gantry_actual_deg, dtype=np.float64),
        collimator_expected_deg=np.asarray(tensor.collimator_expected_deg, dtype=np.float64),
        collimator_actual_deg=np.asarray(tensor.collimator_actual_deg, dtype=np.float64),
        bank_a_expected_mm=np.asarray(tensor.bank_a_expected_mm, dtype=np.float64),
        bank_a_actual_mm=np.asarray(tensor.bank_a_actual_mm, dtype=np.float64),
        bank_b_expected_mm=np.asarray(tensor.bank_b_expected_mm, dtype=np.float64),
        bank_b_actual_mm=np.asarray(tensor.bank_b_actual_mm, dtype=np.float64),
    )
    return tensor


def write_varian_trajectory_bin(
    tensor: TrajectoryLogTensor,
    output_path: str | Path,
    header_version: Literal["3.0", "4.0", "5.0"] | None = None,
) -> Path:
    """Serialize a TrajectoryLogTensor into Varian TrueBeam binary `.bin` format (v3.0/v4.0/v5.0)."""
    version = header_version or (
        tensor.header_version if tensor.header_version in SUPPORTED_BIN_VERSIONS else "5.0"
    )
    if version not in SUPPORTED_BIN_VERSIONS:
        raise ValueError(f"Unsupported Varian binary version: {version}")

    n_snap = tensor.n_snapshots
    n_leaves = tensor.n_leaves_per_bank
    # MLC axis packs 2 carriages (A, B) + n_leaves (Bank A) + n_leaves (Bank B)
    mlc_samples = 2 + 2 * n_leaves
    samples_per_axis = [1, 1, 1, 1, 1, 1, 1, 1, mlc_samples]
    num_axes = len(STANDARD_AXES)
    axis_scale = 2 if tensor.coordinate_system.upper() == "IEC61217" else 1
    num_subbeams = 1
    is_truncated = 0
    mlc_model = 2 if n_leaves == 60 else 3
    crc_enabled = 1 if version == "5.0" else 0

    header = bytearray()
    header += _pack_fixed_str("VOSTL", 16)
    header += _pack_fixed_str(version, 16)
    header += struct.pack("<iii", HEADER_SIZE, int(round(tensor.sampling_interval_ms)), num_axes)
    header += struct.pack(f"<{num_axes}i", *STANDARD_AXES)
    header += struct.pack(f"<{num_axes}i", *samples_per_axis)
    header += struct.pack("<iiiii", axis_scale, num_subbeams, is_truncated, n_snap, mlc_model)
    # Extended metadata inside reserved header area
    header += _pack_fixed_str(tensor.machine_id, 32)
    header += _pack_fixed_str(tensor.patient_id, 32)
    header += _pack_fixed_str(tensor.plan_id, 48)
    header += _pack_fixed_str(tensor.log_id, 48)
    header += _pack_fixed_str(tensor.provenance.provenance_tag, 48)
    header += struct.pack("<i", crc_enabled)

    if len(header) > HEADER_SIZE:
        raise RuntimeError(f"Header overflow: {len(header)} > {HEADER_SIZE}")
    header += b"\x00" * (HEADER_SIZE - len(header))

    # Subbeam record (560 bytes)
    total_mu = float(tensor.mu_expected[-1])
    rad_time_s = float(tensor.timestamps_ms[-1] - tensor.timestamps_ms[0]) / 1000.0
    subbeam = struct.pack("<fffi", 0.0, total_mu, rad_time_s, 0)
    subbeam += _pack_fixed_str(tensor.plan_id, 512)
    subbeam += b"\x00" * (SUBBEAM_SIZE - len(subbeam))

    # Build snapshot matrix [n_snap, total_samples * 2] float32 (linear axes in cm, angles in deg)
    total_samples = sum(samples_per_axis)
    snap_data = np.zeros((n_snap, total_samples * 2), dtype=np.float32)

    # Scalar axes (indices 0..15)
    snap_data[:, 0] = np.asarray(tensor.collimator_expected_deg, dtype=np.float32)
    snap_data[:, 1] = np.asarray(tensor.collimator_actual_deg, dtype=np.float32)
    snap_data[:, 2] = np.asarray(tensor.gantry_expected_deg, dtype=np.float32)
    snap_data[:, 3] = np.asarray(tensor.gantry_actual_deg, dtype=np.float32)
    # Jaws stored in cm on disk (mm / 10.0)
    snap_data[:, 4] = np.asarray(tensor.jaw_expected_mm["Y1"], dtype=np.float32) / 10.0
    snap_data[:, 5] = np.asarray(tensor.jaw_actual_mm["Y1"], dtype=np.float32) / 10.0
    snap_data[:, 6] = np.asarray(tensor.jaw_expected_mm["Y2"], dtype=np.float32) / 10.0
    snap_data[:, 7] = np.asarray(tensor.jaw_actual_mm["Y2"], dtype=np.float32) / 10.0
    snap_data[:, 8] = np.asarray(tensor.jaw_expected_mm["X1"], dtype=np.float32) / 10.0
    snap_data[:, 9] = np.asarray(tensor.jaw_actual_mm["X1"], dtype=np.float32) / 10.0
    snap_data[:, 10] = np.asarray(tensor.jaw_expected_mm["X2"], dtype=np.float32) / 10.0
    snap_data[:, 11] = np.asarray(tensor.jaw_actual_mm["X2"], dtype=np.float32) / 10.0
    snap_data[:, 12] = np.asarray(tensor.mu_expected, dtype=np.float32)
    snap_data[:, 13] = np.asarray(tensor.mu_actual, dtype=np.float32)
    snap_data[:, 14] = np.asarray(tensor.beam_hold, dtype=np.float32)
    snap_data[:, 15] = np.asarray(tensor.beam_hold, dtype=np.float32)

    # MLC carriages (mean bank position in cm) + leaves in cm
    bank_a_exp = np.asarray(tensor.bank_a_expected_mm, dtype=np.float32) / 10.0
    bank_a_act = np.asarray(tensor.bank_a_actual_mm, dtype=np.float32) / 10.0
    bank_b_exp = np.asarray(tensor.bank_b_expected_mm, dtype=np.float32) / 10.0
    bank_b_act = np.asarray(tensor.bank_b_actual_mm, dtype=np.float32) / 10.0

    # Carriage A & B expected/actual
    snap_data[:, 16] = np.mean(bank_a_exp, axis=1)
    snap_data[:, 17] = np.mean(bank_a_act, axis=1)
    snap_data[:, 18] = np.mean(bank_b_exp, axis=1)
    snap_data[:, 19] = np.mean(bank_b_act, axis=1)

    # Bank A leaves (20 .. 20 + 2*n_leaves)
    offset = 20
    snap_data[:, offset : offset + 2 * n_leaves : 2] = bank_a_exp
    snap_data[:, offset + 1 : offset + 2 * n_leaves : 2] = bank_a_act
    # Bank B leaves
    offset_b = offset + 2 * n_leaves
    snap_data[:, offset_b : offset_b + 2 * n_leaves : 2] = bank_b_exp
    snap_data[:, offset_b + 1 : offset_b + 2 * n_leaves : 2] = bank_b_act

    payload = bytes(header) + subbeam + snap_data.astype("<f4").tobytes()
    if crc_enabled:
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        payload += struct.pack("<I", crc)

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return dest


def parse_varian_trajectory_bin(
    bin_path: str | Path,
    validate: bool = True,
) -> TrajectoryLogTensor:
    """Parse a Varian TrueBeam binary `.bin` trajectory log (header v3.0/v4.0/v5.0)."""
    path = Path(bin_path)
    if not path.exists():
        raise AbstentionError(
            AbstentionCode.CORRUPTED_BINARY_LOG,
            f"Trajectory log binary file does not exist: {path}",
            {"path": str(path)},
        )

    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise AbstentionError(
            AbstentionCode.CORRUPTED_BINARY_LOG,
            f"Binary log too short ({len(raw)} bytes < {HEADER_SIZE}-byte Varian header).",
            {"bytes": len(raw)},
        )

    signature = _unpack_fixed_str(raw[0:16])
    if signature != "VOSTL":
        raise AbstentionError(
            AbstentionCode.CORRUPTED_BINARY_LOG,
            f"Invalid Varian binary signature '{signature}' (expected 'VOSTL').",
            {"signature": signature},
        )

    version = _unpack_fixed_str(raw[16:32])
    if version not in SUPPORTED_BIN_VERSIONS:
        raise AbstentionError(
            AbstentionCode.CORRUPTED_BINARY_LOG,
            f"Unsupported Varian header version '{version}' (expected 3.0, 4.0, or 5.0).",
            {"version": version},
        )

    header_size, sampling_interval_ms, num_axes = struct.unpack_from("<iii", raw, 32)
    if header_size != HEADER_SIZE or num_axes != len(STANDARD_AXES) or sampling_interval_ms <= 0:
        raise AbstentionError(
            AbstentionCode.CORRUPTED_BINARY_LOG,
            "Corrupted header dimensions or invalid sampling interval.",
            {
                "header_size": header_size,
                "num_axes": num_axes,
                "sampling_interval_ms": sampling_interval_ms,
            },
        )

    pos = 44
    axes = list(struct.unpack_from(f"<{num_axes}i", raw, pos))
    pos += 4 * num_axes
    samples_per_axis = list(struct.unpack_from(f"<{num_axes}i", raw, pos))
    pos += 4 * num_axes
    axis_scale, num_subbeams, is_truncated, n_snap, _mlc_model = struct.unpack_from(
        "<iiiii", raw, pos
    )
    pos += 20

    if axes != STANDARD_AXES:
        raise AbstentionError(
            AbstentionCode.CORRUPTED_BINARY_LOG,
            f"Unexpected Varian axis table: {axes}",
            {"axes": axes},
        )

    if axis_scale != 2:
        raise AbstentionError(
            AbstentionCode.UNMAPPED_IEC61217_COORDINATES,
            f"Unsupported Varian axis_scale={axis_scale}; IEC 61217 (axis_scale=2) is required.",
            {"axis_scale": axis_scale},
        )

    if is_truncated != 0:
        raise AbstentionError(
            AbstentionCode.INTERLOCK_INCOMPLETE_DELIVERY,
            "Varian trajectory log header marks delivery as truncated/interlocked.",
            {"is_truncated": is_truncated},
        )

    machine_id = _unpack_fixed_str(raw[pos : pos + 32]) or "TrueBeam-01"
    pos += 32
    patient_id = _unpack_fixed_str(raw[pos : pos + 32]) or "unknown_patient"
    pos += 32
    plan_id = _unpack_fixed_str(raw[pos : pos + 48]) or "unknown_plan"
    pos += 48
    log_id = _unpack_fixed_str(raw[pos : pos + 48]) or path.stem
    pos += 48
    prov_tag_raw = _unpack_fixed_str(raw[pos : pos + 48])
    pos += 48
    (crc_enabled,) = struct.unpack_from("<i", raw, pos)

    mlc_samples = samples_per_axis[-1]
    if mlc_samples < 4 or (mlc_samples - 2) % 2 != 0:
        raise AbstentionError(
            AbstentionCode.CORRUPTED_BINARY_LOG,
            f"Invalid MLC sample count per snapshot: {mlc_samples}",
            {"mlc_samples": mlc_samples},
        )
    n_leaves = (mlc_samples - 2) // 2

    total_samples = sum(samples_per_axis)
    data_start = header_size + num_subbeams * SUBBEAM_SIZE
    snapshot_bytes = n_snap * total_samples * 2 * 4
    expected_len = data_start + snapshot_bytes + (4 if crc_enabled else 0)

    if len(raw) != expected_len:
        raise AbstentionError(
            AbstentionCode.CORRUPTED_BINARY_LOG,
            f"Binary payload length mismatch: got {len(raw)} bytes, expected {expected_len}.",
            {"actual_bytes": len(raw), "expected_bytes": expected_len},
        )

    if crc_enabled:
        expected_crc = struct.unpack_from("<I", raw, len(raw) - 4)[0]
        actual_crc = zlib.crc32(raw[:-4]) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise AbstentionError(
                AbstentionCode.CORRUPTED_BINARY_LOG,
                f"CRC32 checksum mismatch in v5.0 binary log ({actual_crc:#010x} != {expected_crc:#010x}).",
                {"actual_crc32": f"{actual_crc:#010x}", "expected_crc32": f"{expected_crc:#010x}"},
            )

    snap = np.frombuffer(
        raw, dtype="<f4", count=n_snap * total_samples * 2, offset=data_start
    ).reshape((n_snap, total_samples * 2))

    timestamps_ms = (np.arange(n_snap, dtype=np.float64) * float(sampling_interval_ms)).tolist()
    offset_a = 20
    offset_b = offset_a + 2 * n_leaves

    valid_tags = {
        "OPEN_PUBLIC_ANATOMY_OPENKBP_PT1",
        "OPEN_PUBLIC_ANATOMY_OPENKBP_PT2",
        "OPEN_PUBLIC_ANATOMY_OPENKBP_PT3",
        "OPEN_PUBLIC_ANATOMY_OPENKBP_COHORT",
        "SYNTHETIC_FAULT_INJECTION_BENCHMARK",
        "SYNTHETIC_TG142_FLEET_DRIFT",
        "VENDOR_TRAJECTORY_LOG",
        "EPID_THREE_WAY_FORENSICS",
        "COMPOSITE_OPENKBP_PLUS_SYNTHETIC_LOG",
    }
    prov_tag = prov_tag_raw if prov_tag_raw in valid_tags else "VENDOR_TRAJECTORY_LOG"

    provenance = build_provenance(
        record_id=f"bin:{log_id}",
        provenance_tag=prov_tag,  # type: ignore[arg-type]
        dataset_source=f"Varian TrueBeam Trajectory Log v{version} ({path.name})",
        sha256_inputs={"trajectory_bin": sha256_bytes(raw)},
        notes=f"Parsed {n_snap} snapshots at {1000.0 / sampling_interval_ms:.1f} Hz ({n_leaves} leaf pairs).",
    )

    tensor = TrajectoryLogTensor(
        log_id=log_id,
        header_version=version,  # type: ignore[arg-type]
        machine_id=machine_id,
        patient_id=patient_id,
        plan_id=plan_id,
        sampling_interval_ms=float(sampling_interval_ms),
        n_snapshots=int(n_snap),
        n_leaves_per_bank=int(n_leaves),
        coordinate_system="IEC61217",
        timestamps_ms=timestamps_ms,
        collimator_expected_deg=snap[:, 0].astype(float).tolist(),
        collimator_actual_deg=snap[:, 1].astype(float).tolist(),
        gantry_expected_deg=snap[:, 2].astype(float).tolist(),
        gantry_actual_deg=snap[:, 3].astype(float).tolist(),
        jaw_expected_mm={
            "Y1": (snap[:, 4] * 10.0).astype(float).tolist(),
            "Y2": (snap[:, 6] * 10.0).astype(float).tolist(),
            "X1": (snap[:, 8] * 10.0).astype(float).tolist(),
            "X2": (snap[:, 10] * 10.0).astype(float).tolist(),
        },
        jaw_actual_mm={
            "Y1": (snap[:, 5] * 10.0).astype(float).tolist(),
            "Y2": (snap[:, 7] * 10.0).astype(float).tolist(),
            "X1": (snap[:, 9] * 10.0).astype(float).tolist(),
            "X2": (snap[:, 11] * 10.0).astype(float).tolist(),
        },
        mu_expected=snap[:, 12].astype(float).tolist(),
        mu_actual=snap[:, 13].astype(float).tolist(),
        beam_hold=np.rint(snap[:, 15]).astype(int).tolist(),
        bank_a_expected_mm=(snap[:, offset_a : offset_a + 2 * n_leaves : 2] * 10.0)
        .astype(float)
        .tolist(),
        bank_a_actual_mm=(snap[:, offset_a + 1 : offset_a + 2 * n_leaves : 2] * 10.0)
        .astype(float)
        .tolist(),
        bank_b_expected_mm=(snap[:, offset_b : offset_b + 2 * n_leaves : 2] * 10.0)
        .astype(float)
        .tolist(),
        bank_b_actual_mm=(snap[:, offset_b + 1 : offset_b + 2 * n_leaves : 2] * 10.0)
        .astype(float)
        .tolist(),
        provenance=provenance,
    )
    return validate_trajectory_tensor(tensor) if validate else tensor


def parse_telemetry_json(json_path: str | Path, validate: bool = True) -> TrajectoryLogTensor:
    """Ingest vendor-neutral JSON trajectory telemetry into a validated TrajectoryLogTensor."""
    path = Path(json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "provenance" not in payload:
        payload["provenance"] = build_provenance(
            record_id=f"json:{payload.get('log_id', path.stem)}",
            provenance_tag="VENDOR_TRAJECTORY_LOG",
            dataset_source=f"Vendor-neutral JSON telemetry ({path.name})",
            sha256_inputs={"telemetry_json": sha256_file(path)},
        ).model_dump()
    payload.setdefault("header_version", "JSON-1.0")
    tensor = TrajectoryLogTensor.model_validate(payload)
    return validate_trajectory_tensor(tensor) if validate else tensor


def parse_telemetry_csv(
    csv_path: str | Path,
    log_id: str | None = None,
    machine_id: str = "Linac-CSV",
    patient_id: str = "pt_1",
    plan_id: str = "CSV_Plan",
    validate: bool = True,
) -> TrajectoryLogTensor:
    """Ingest flat vendor-neutral CSV telemetry (columns: timestamp_ms, gantry_exp, gantry_act, ...).

    Expects leaf columns named `A1_exp`, `A1_act`, ..., `B1_exp`, `B1_act`, ... in mm.
    """
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if len(rows) < 2:
        raise AbstentionError(
            AbstentionCode.TIMESTAMP_DESYNC,
            "CSV telemetry must contain at least 2 snapshot rows.",
            {"rows": len(rows)},
        )

    leaf_indices = sorted(
        {
            int(col[1:].split("_")[0])
            for col in fieldnames
            if col.startswith("A") and col.endswith("_exp") and col[1:].split("_")[0].isdigit()
        }
    )
    if not leaf_indices:
        raise AbstentionError(
            AbstentionCode.CORRUPTED_BINARY_LOG,
            "CSV telemetry missing required MLC leaf columns (e.g., A1_exp, A1_act, B1_exp, B1_act).",
        )

    ts = [float(r["timestamp_ms"]) for r in rows]
    dt = float(np.median(np.diff(ts)))
    g_exp = [float(r["gantry_exp"]) for r in rows]
    g_act = [float(r["gantry_act"]) for r in rows]
    c_exp = [float(r.get("collimator_exp", 0.0)) for r in rows]
    c_act = [float(r.get("collimator_act", 0.0)) for r in rows]
    mu_exp = [float(r["mu_exp"]) for r in rows]
    mu_act = [float(r["mu_act"]) for r in rows]
    hold = [int(float(r.get("beam_hold", 0))) for r in rows]

    jaw_exp = {
        k: [float(r.get(f"jaw_{k}_exp", -75.0 if k.endswith("1") else 75.0)) for r in rows]
        for k in ("X1", "X2", "Y1", "Y2")
    }
    jaw_act = {
        k: [float(r.get(f"jaw_{k}_act", jaw_exp[k][idx])) for idx, r in enumerate(rows)]
        for k in ("X1", "X2", "Y1", "Y2")
    }

    bank_a_exp = [[float(r[f"A{i}_exp"]) for i in leaf_indices] for r in rows]
    bank_a_act = [[float(r[f"A{i}_act"]) for i in leaf_indices] for r in rows]
    bank_b_exp = [[float(r[f"B{i}_exp"]) for i in leaf_indices] for r in rows]
    bank_b_act = [[float(r[f"B{i}_act"]) for i in leaf_indices] for r in rows]

    lid = log_id or path.stem
    provenance = build_provenance(
        record_id=f"csv:{lid}",
        provenance_tag="VENDOR_TRAJECTORY_LOG",
        dataset_source=f"Vendor-neutral CSV telemetry ({path.name})",
        sha256_inputs={"telemetry_csv": sha256_file(path)},
    )

    tensor = TrajectoryLogTensor(
        log_id=lid,
        header_version="CSV-1.0",
        machine_id=machine_id,
        patient_id=patient_id,
        plan_id=plan_id,
        sampling_interval_ms=dt,
        n_snapshots=len(rows),
        n_leaves_per_bank=len(leaf_indices),
        coordinate_system="IEC61217",
        timestamps_ms=ts,
        gantry_expected_deg=g_exp,
        gantry_actual_deg=g_act,
        collimator_expected_deg=c_exp,
        collimator_actual_deg=c_act,
        jaw_expected_mm=jaw_exp,
        jaw_actual_mm=jaw_act,
        mu_expected=mu_exp,
        mu_actual=mu_act,
        beam_hold=hold,
        bank_a_expected_mm=bank_a_exp,
        bank_a_actual_mm=bank_a_act,
        bank_b_expected_mm=bank_b_exp,
        bank_b_actual_mm=bank_b_act,
        provenance=provenance,
    )
    return validate_trajectory_tensor(tensor) if validate else tensor


def generate_synthetic_vmat_log(
    log_id: str = "TB01_VMAT_ARC1_SYNTH",
    machine_id: str = "TrueBeam-01",
    patient_id: str = "pt_1",
    plan_id: str = "VMAT_HN_SIB_70Gy",
    n_snapshots: int = 360,
    sampling_interval_ms: float = 20.0,
    n_leaves_per_bank: int = 60,
    total_mu: float = 320.0,
    header_version: Literal["3.0", "4.0", "5.0"] = "5.0",
    # Injectable electro-mechanical physical faults (all 0.0 -> exact zero-error identity)
    failing_bank: Literal["A", "B"] = "B",
    failing_leaf_index: int = 34,  # 1-based clinical leaf index
    leaf_drag_ms: float = 0.0,
    leaf_stiction_mm: float = 0.0,
    dlg_offset_mm: float = 0.0,
    gantry_sag_sin_mm: float = 0.0,
    gantry_sag_cos_mm: float = 0.0,
    mu_drift_pct: float = 0.0,
    noise_sigma_mm: float = 0.0,
    seed: int = 42,
) -> TrajectoryLogTensor:
    """Generate a realistic 50 Hz 360° VMAT arc trajectory log with exact injectable faults.

    All fault parameters default to 0.0, yielding an exact zero-error identity tensor
    (`actual == expected` bit-for-bit).
    """
    rng = np.random.default_rng(seed)
    t_ms = np.arange(n_snapshots, dtype=np.float64) * sampling_interval_ms
    dt_s = sampling_interval_ms / 1000.0

    # Smooth full 360° VMAT arc (181° -> 179° CW mapped cleanly in [0, 360])
    gantry_exp = np.linspace(1.0, 359.0, n_snapshots, dtype=np.float64)
    theta_rad = np.deg2rad(gantry_exp)
    collimator_exp = np.full(n_snapshots, 15.0, dtype=np.float64)

    # Modulated VMAT MU delivery profile (smooth S-curve with variable dose rate)
    mu_rate = 1.0 + 0.18 * np.sin(2.0 * theta_rad) - 0.10 * np.cos(3.0 * theta_rad)
    mu_cum = np.cumsum(mu_rate)
    mu_exp = total_mu * (mu_cum - mu_cum[0]) / max(mu_cum[-1] - mu_cum[0], 1e-9)

    # Jaws framing H&N target volume (mm in IEC 61217)
    jaw_x1_exp = np.full(n_snapshots, -75.0)
    jaw_x2_exp = np.full(n_snapshots, 75.0)
    jaw_y1_exp = np.full(n_snapshots, -85.0)
    jaw_y2_exp = np.full(n_snapshots, 85.0)

    # Active central leaves (1-based indices 18..43 => 0-based 17..42 cover +/- 65 mm at 5 mm width)
    bank_a_exp = np.zeros((n_snapshots, n_leaves_per_bank), dtype=np.float64)
    bank_b_exp = np.zeros((n_snapshots, n_leaves_per_bank), dtype=np.float64)

    active_start = max(0, n_leaves_per_bank // 2 - 13)
    active_end = min(n_leaves_per_bank, n_leaves_per_bank // 2 + 13)

    for idx in range(n_leaves_per_bank):
        if active_start <= idx < active_end:
            rel_y = (idx - n_leaves_per_bank / 2.0 + 0.5) / 13.0  # [-1, +1]
            # Realistic dynamic sliding-window VMAT modulation across the arc
            phase_shift = 0.28 * rel_y
            center_x = (
                14.0 * np.sin(2.0 * theta_rad + phase_shift)
                + 9.0 * np.cos(4.0 * theta_rad - 1.5 * phase_shift)
                + 5.0 * np.sin(6.0 * theta_rad + 0.6 * idx)
            )
            half_width = (
                22.0 * np.sqrt(max(0.12, 1.0 - 0.72 * rel_y**2))
                + 5.5 * np.cos(3.0 * theta_rad + 0.2 * idx)
            )
            half_width = np.clip(half_width, 6.0, 42.0)
            bank_a_exp[:, idx] = center_x + half_width
            bank_b_exp[:, idx] = center_x - half_width
        else:
            # Parked closed under jaws (zero gap at x = 0)
            bank_a_exp[:, idx] = 0.0
            bank_b_exp[:, idx] = 0.0

    # Initialize actual == expected (exact zero-error baseline)
    gantry_act = gantry_exp.copy()
    collimator_act = collimator_exp.copy()
    jaw_x1_act = jaw_x1_exp.copy()
    jaw_x2_act = jaw_x2_exp.copy()
    jaw_y1_act = jaw_y1_exp.copy()
    jaw_y2_act = jaw_y2_exp.copy()
    mu_act = mu_exp.copy()
    bank_a_act = bank_a_exp.copy()
    bank_b_act = bank_b_exp.copy()
    beam_hold = np.zeros(n_snapshots, dtype=np.int32)

    # 1. Inject systematic DLG / bank calibration offset (widens aperture by +dlg_offset_mm)
    if dlg_offset_mm != 0.0:
        bank_a_act[:, active_start:active_end] += 0.5 * dlg_offset_mm
        bank_b_act[:, active_start:active_end] -= 0.5 * dlg_offset_mm

    # 2. Inject localized leaf viscous drag: Δx_i(t) = -τ_i * v_i(t)
    #    and/or nonlinear Coulomb/Stribeck stiction: Δx_i(t) = -β_i * tanh(v_i(t) / v_0)
    if leaf_drag_ms != 0.0 or leaf_stiction_mm != 0.0:
        leaf_0 = failing_leaf_index - 1
        if not (0 <= leaf_0 < n_leaves_per_bank):
            raise ValueError(f"failing_leaf_index {failing_leaf_index} out of range 1..{n_leaves_per_bank}")
        tau_s = leaf_drag_ms / 1000.0
        v0_stiction = 5.0
        if failing_bank == "A":
            vel = np.gradient(bank_a_exp[:, leaf_0], dt_s)
            bank_a_act[:, leaf_0] += -tau_s * vel - leaf_stiction_mm * np.tanh(vel / v0_stiction)
        else:
            vel = np.gradient(bank_b_exp[:, leaf_0], dt_s)
            bank_b_act[:, leaf_0] += -tau_s * vel - leaf_stiction_mm * np.tanh(vel / v0_stiction)

    # 3. Inject gantry gravity-sag harmonic: Δy(θ) = A sin(θ) + B cos(θ)
    if gantry_sag_sin_mm != 0.0 or gantry_sag_cos_mm != 0.0:
        sag_y = gantry_sag_sin_mm * np.sin(theta_rad) + gantry_sag_cos_mm * np.cos(theta_rad)
        jaw_y1_act = jaw_y1_exp + sag_y
        jaw_y2_act = jaw_y2_exp + sag_y
        # Small coupled gantry encoder flex (0.12 deg/mm)
        gantry_act = np.clip(gantry_exp + 0.12 * sag_y, 0.0, 360.0)

    # 4. Inject multiplicative ion-chamber / dose-rate MU drift
    if mu_drift_pct != 0.0:
        mu_act = mu_exp * (1.0 + mu_drift_pct / 100.0)

    # 5. Optional realistic encoder measurement noise (mm)
    if noise_sigma_mm > 0.0:
        bank_a_act[:, active_start:active_end] += rng.normal(
            0.0, noise_sigma_mm, size=(n_snapshots, active_end - active_start)
        )
        bank_b_act[:, active_start:active_end] += rng.normal(
            0.0, noise_sigma_mm, size=(n_snapshots, active_end - active_start)
        )
        sag_noise = rng.normal(0.0, noise_sigma_mm, size=n_snapshots)
        jaw_y1_act += sag_noise
        jaw_y2_act += sag_noise
        # Preserve MU monotonicity while adding tiny 0.03% ion-chamber noise
        dmu = np.diff(mu_act, prepend=0.0)
        dmu_noisy = np.maximum(0.0, dmu * (1.0 + rng.normal(0.0, 0.0003, size=n_snapshots)))
        dmu_noisy[0] = 0.0
        mu_act = np.cumsum(dmu_noisy)

    fault_meta = {
        "failing_bank": failing_bank,
        "failing_leaf_index": failing_leaf_index,
        "leaf_drag_ms": leaf_drag_ms,
        "leaf_stiction_mm": leaf_stiction_mm,
        "dlg_offset_mm": dlg_offset_mm,
        "gantry_sag_sin_mm": gantry_sag_sin_mm,
        "gantry_sag_cos_mm": gantry_sag_cos_mm,
        "mu_drift_pct": mu_drift_pct,
        "noise_sigma_mm": noise_sigma_mm,
        "seed": seed,
    }
    provenance = build_provenance(
        record_id=f"synth:{log_id}",
        provenance_tag="SYNTHETIC_FAULT_INJECTION_BENCHMARK",
        dataset_source="linac-forensics Synthetic VMAT Trajectory Fault-Injection Suite",
        sha256_inputs={"fault_config": sha256_bytes(json.dumps(fault_meta, sort_keys=True).encode())},
        notes=f"Controlled synthetic VMAT arc ({n_snapshots} snapshots @ {1000.0 / sampling_interval_ms:.0f} Hz); faults={fault_meta}",
    )

    tensor = TrajectoryLogTensor(
        log_id=log_id,
        header_version=header_version,
        machine_id=machine_id,
        patient_id=patient_id,
        plan_id=plan_id,
        sampling_interval_ms=sampling_interval_ms,
        n_snapshots=n_snapshots,
        n_leaves_per_bank=n_leaves_per_bank,
        coordinate_system="IEC61217",
        timestamps_ms=t_ms.tolist(),
        gantry_expected_deg=gantry_exp.tolist(),
        gantry_actual_deg=gantry_act.tolist(),
        collimator_expected_deg=collimator_exp.tolist(),
        collimator_actual_deg=collimator_act.tolist(),
        jaw_expected_mm={
            "X1": jaw_x1_exp.tolist(),
            "X2": jaw_x2_exp.tolist(),
            "Y1": jaw_y1_exp.tolist(),
            "Y2": jaw_y2_exp.tolist(),
        },
        jaw_actual_mm={
            "X1": jaw_x1_act.tolist(),
            "X2": jaw_x2_act.tolist(),
            "Y1": jaw_y1_act.tolist(),
            "Y2": jaw_y2_act.tolist(),
        },
        mu_expected=mu_exp.tolist(),
        mu_actual=mu_act.tolist(),
        beam_hold=beam_hold.tolist(),
        bank_a_expected_mm=bank_a_exp.tolist(),
        bank_a_actual_mm=bank_a_act.tolist(),
        bank_b_expected_mm=bank_b_exp.tolist(),
        bank_b_actual_mm=bank_b_act.tolist(),
        provenance=provenance,
    )
    return validate_trajectory_tensor(tensor)


def write_varian_dynalog_dlg(
    tensor: TrajectoryLogTensor,
    bank_a_path: str | Path,
    bank_b_path: str | Path,
) -> tuple[Path, Path]:
    """Export a TrajectoryLogTensor as a paired Varian Clinac/Trilogy Dynalog (.dlg) Bank A/B pair."""
    a_out = Path(bank_a_path)
    b_out = Path(bank_b_path)
    a_out.parent.mkdir(parents=True, exist_ok=True)
    b_out.parent.mkdir(parents=True, exist_ok=True)

    total_mu = max(float(tensor.mu_expected[-1]), 1e-6)
    n_leaves = tensor.n_leaves_per_bank

    for bank_char, dest, exp_mat, act_mat in (
        ("A", a_out, tensor.bank_a_expected_mm, tensor.bank_a_actual_mm),
        ("B", b_out, tensor.bank_b_expected_mm, tensor.bank_b_actual_mm),
    ):
        lines = [
            "B",
            f"{tensor.patient_id},{tensor.plan_id},{tensor.machine_id},{total_mu:.4f}",
            f"{tensor.log_id}_BANK_{bank_char}",
            "0.0",
            f"{n_leaves}",
            f"{tensor.sampling_interval_ms:.1f}",
        ]
        for t in range(tensor.n_snapshots):
            df_exp = int(round(25000.0 * float(tensor.mu_expected[t]) / total_mu))
            df_act = int(round(25000.0 * float(tensor.mu_actual[t]) / total_mu))
            hold = int(tensor.beam_hold[t])
            g_exp = int(round(float(tensor.gantry_expected_deg[t]) * 10.0))
            g_act = int(round(float(tensor.gantry_actual_deg[t]) * 10.0))
            c_exp = int(round(float(tensor.collimator_expected_deg[t]) * 10.0))
            y1_exp = int(round(float(tensor.jaw_expected_mm["Y1"][t]) * 10.0))
            y1_act = int(round(float(tensor.jaw_actual_mm["Y1"][t]) * 10.0))
            y2_exp = int(round(float(tensor.jaw_expected_mm["Y2"][t]) * 10.0))
            y2_act = int(round(float(tensor.jaw_actual_mm["Y2"][t]) * 10.0))
            x1_exp = int(round(float(tensor.jaw_expected_mm["X1"][t]) * 10.0))
            x2_exp = int(round(float(tensor.jaw_expected_mm["X2"][t]) * 10.0))
            row = [
                df_exp,
                df_act,
                hold,
                g_exp,
                g_act,
                c_exp,
                y1_exp,
                y1_act,
                y2_exp,
                y2_act,
                x1_exp,
                x2_exp,
            ]
            for l_idx in range(n_leaves):
                row.append(int(round(float(exp_mat[t][l_idx]) * 100.0)))
                row.append(int(round(float(act_mat[t][l_idx]) * 100.0)))
            lines.append(",".join(str(v) for v in row))
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return a_out, b_out


def parse_varian_dynalog_dlg(
    bank_a_path: str | Path,
    bank_b_path: str | Path,
    validate: bool = True,
) -> TrajectoryLogTensor:
    """Parse a paired Varian Clinac/Trilogy Dynalog (.dlg) Bank A & Bank B file pair."""
    pa = Path(bank_a_path)
    pb = Path(bank_b_path)
    if not pa.exists() or not pb.exists():
        raise AbstentionError(
            AbstentionCode.CORRUPTED_BINARY_LOG,
            "Both Bank A and Bank B Dynalog (.dlg) files must exist.",
            {"bank_a": str(pa), "bank_b": str(pb)},
        )

    lines_a = [ln.strip() for ln in pa.read_text(encoding="utf-8").splitlines() if ln.strip()]
    lines_b = [ln.strip() for ln in pb.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines_a) < 8 or len(lines_b) < 8 or lines_a[0] != "B" or lines_b[0] != "B":
        raise AbstentionError(
            AbstentionCode.CORRUPTED_BINARY_LOG,
            "Invalid or truncated Varian Dynalog Version B header.",
        )

    meta = lines_a[1].split(",")
    patient_id = meta[0] if len(meta) > 0 and meta[0] else "dynalog_pt"
    plan_id = meta[1] if len(meta) > 1 and meta[1] else "dynalog_plan"
    machine_id = meta[2] if len(meta) > 2 and meta[2] else "Clinac-21EX"
    total_mu = float(meta[3]) if len(meta) > 3 else 300.0
    n_leaves = int(lines_a[4])
    dt_ms = float(lines_a[5])

    rows_a = lines_a[6:]
    rows_b = lines_b[6:]
    if len(rows_a) != len(rows_b):
        raise AbstentionError(
            AbstentionCode.TIMESTAMP_DESYNC,
            f"Dynalog Bank A ({len(rows_a)}) and Bank B ({len(rows_b)}) snapshot count mismatch.",
        )

    n_snap = len(rows_a)
    ts = [i * dt_ms for i in range(n_snap)]
    g_exp, g_act, c_exp, mu_exp, mu_act, hold = [], [], [], [], [], []
    jaw_exp: dict[str, list[float]] = {"X1": [], "X2": [], "Y1": [], "Y2": []}
    jaw_act: dict[str, list[float]] = {"X1": [], "X2": [], "Y1": [], "Y2": []}
    ba_exp, ba_act, bb_exp, bb_act = [], [], [], []

    for ra, rb in zip(rows_a, rows_b, strict=True):
        fa = [int(x) for x in ra.split(",")]
        fb = [int(x) for x in rb.split(",")]
        expected_cols = 12 + 2 * n_leaves
        if len(fa) != expected_cols or len(fb) != expected_cols:
            raise AbstentionError(
                AbstentionCode.CORRUPTED_BINARY_LOG,
                f"Dynalog column count mismatch (expected {expected_cols}).",
            )
        mu_exp.append(total_mu * fa[0] / 25000.0)
        mu_act.append(total_mu * fa[1] / 25000.0)
        hold.append(fa[2])
        g_exp.append(fa[3] / 10.0)
        g_act.append(fa[4] / 10.0)
        c_exp.append(fa[5] / 10.0)
        jaw_exp["Y1"].append(fa[6] / 10.0)
        jaw_act["Y1"].append(fa[7] / 10.0)
        jaw_exp["Y2"].append(fa[8] / 10.0)
        jaw_act["Y2"].append(fa[9] / 10.0)
        jaw_exp["X1"].append(fa[10] / 10.0)
        jaw_act["X1"].append(fa[10] / 10.0)
        jaw_exp["X2"].append(fa[11] / 10.0)
        jaw_act["X2"].append(fa[11] / 10.0)
        ba_exp.append([fa[12 + 2 * i] / 100.0 for i in range(n_leaves)])
        ba_act.append([fa[13 + 2 * i] / 100.0 for i in range(n_leaves)])
        bb_exp.append([fb[12 + 2 * i] / 100.0 for i in range(n_leaves)])
        bb_act.append([fb[13 + 2 * i] / 100.0 for i in range(n_leaves)])

    lid = lines_a[2].replace("_BANK_A", "") or pa.stem
    prov = build_provenance(
        record_id=f"dlg:{lid}",
        provenance_tag="VENDOR_TRAJECTORY_LOG",
        dataset_source=f"Varian Legacy Dynalog Version B ({pa.name}, {pb.name})",
        sha256_inputs={"bank_a_dlg": sha256_file(pa), "bank_b_dlg": sha256_file(pb)},
        notes=f"Parsed {n_snap} Dynalog snapshots ({n_leaves} leaves/bank).",
    )
    tensor = TrajectoryLogTensor(
        log_id=lid,
        header_version="DYNALOG-B",
        machine_id=machine_id,
        patient_id=patient_id,
        plan_id=plan_id,
        sampling_interval_ms=dt_ms,
        n_snapshots=n_snap,
        n_leaves_per_bank=n_leaves,
        coordinate_system="IEC61217",
        timestamps_ms=ts,
        gantry_expected_deg=g_exp,
        gantry_actual_deg=g_act,
        collimator_expected_deg=c_exp,
        collimator_actual_deg=c_exp,
        jaw_expected_mm=jaw_exp,
        jaw_actual_mm=jaw_act,
        mu_expected=mu_exp,
        mu_actual=mu_act,
        beam_hold=hold,
        bank_a_expected_mm=ba_exp,
        bank_a_actual_mm=ba_act,
        bank_b_expected_mm=bb_exp,
        bank_b_actual_mm=bb_act,
        provenance=prov,
    )
    return validate_trajectory_tensor(tensor) if validate else tensor
