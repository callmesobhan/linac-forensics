"""OpenKBP 3D patient anatomy loader with SHA-256 manifest verification.

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from linac_forensics.provenance import sha256_file

OPENKBP_GRID_SHAPE = (128, 128, 128)

ROI_TYPES: dict[str, str] = {
    "PTV70": "PTV",
    "PTV63": "PTV",
    "PTV56": "PTV",
    "SpinalCord": "OAR",
    "Brainstem": "OAR",
    "LeftParotid": "OAR",
    "RightParotid": "OAR",
    "Mandible": "OAR",
}


@dataclass(frozen=True)
class PatientAnatomyVolume:
    """Loaded 3D patient CT, reference planned dose, and binary ROI masks."""

    patient_id: str
    anatomy_source: str
    provenance_tag: str
    ct_hu: np.ndarray  # float32 [128, 128, 128]
    planned_dose_gy: np.ndarray  # float32 [128, 128, 128]
    possible_dose_mask: np.ndarray  # bool [128, 128, 128]
    voxel_dims_mm: tuple[float, float, float]  # (dx, dy, dz) in mm
    roi_masks: dict[str, np.ndarray]  # bool [128, 128, 128]
    sha256_inputs: dict[str, str]

    @property
    def voxel_volume_cc(self) -> float:
        dx, dy, dz = self.voxel_dims_mm
        return float((dx * dy * dz) / 1000.0)


def _load_sparse_value_csv(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    """Load OpenKBP sparse (flat_index, data) CSV into a dense float32 3D array."""
    arr = np.zeros(int(np.prod(shape)), dtype=np.float32)
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if raw.size == 0:
        return arr.reshape(shape)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    indices = raw[:, 0].astype(np.int64)
    values = raw[:, 1].astype(np.float32)
    arr[indices] = values
    return arr.reshape(shape)


def _load_sparse_mask_csv(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    """Load OpenKBP sparse mask CSV (single column of flat indices) into dense bool array."""
    arr = np.zeros(int(np.prod(shape)), dtype=bool)
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()[1:] if line.strip()]
    if not lines:
        return arr.reshape(shape)
    indices = np.fromiter((int(line.split(",")[0]) for line in lines), dtype=np.int64)
    arr[indices] = True
    return arr.reshape(shape)


def load_openkbp_patient(
    patient_dir: str | Path,
    verify_manifest: bool = True,
) -> PatientAnatomyVolume:
    """Load an OpenKBP patient directory (e.g., `data/openkbp/pt_1`) and verify SHA-256 hashes."""
    pdir = Path(patient_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"OpenKBP patient directory not found: {pdir}")

    sha_map: dict[str, str] = {}
    manifest_path = pdir / "MANIFEST.json"
    if verify_manifest and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for fname, meta in manifest.get("files", {}).items():
            fpath = pdir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing OpenKBP file listed in MANIFEST.json: {fpath}")
            digest = sha256_file(fpath)
            expected = meta.get("sha256")
            if expected and digest != expected:
                raise ValueError(
                    f"OpenKBP SHA-256 verification failed for {fname}: {digest} != {expected}"
                )
            sha_map[f"openkbp/{fname}"] = digest

    voxel_dims_raw = np.loadtxt(pdir / "voxel_dimensions.csv", dtype=np.float64).ravel()
    voxel_dims_mm = (
        float(voxel_dims_raw[0]),
        float(voxel_dims_raw[1]),
        float(voxel_dims_raw[2]),
    )

    ct_hu = _load_sparse_value_csv(pdir / "ct.csv", OPENKBP_GRID_SHAPE)
    planned_dose_gy = _load_sparse_value_csv(pdir / "dose.csv", OPENKBP_GRID_SHAPE)
    possible_dose_mask = _load_sparse_mask_csv(
        pdir / "possible_dose_mask.csv", OPENKBP_GRID_SHAPE
    )

    roi_masks: dict[str, np.ndarray] = {}
    for roi_name in ROI_TYPES:
        roi_path = pdir / f"{roi_name}.csv"
        if roi_path.exists():
            roi_masks[roi_name] = _load_sparse_mask_csv(roi_path, OPENKBP_GRID_SHAPE)

    return PatientAnatomyVolume(
        patient_id=pdir.name,
        anatomy_source=f"OpenKBP/{pdir.name} (CC BY 4.0, Babier et al. 2021)",
        provenance_tag="OPEN_PUBLIC_ANATOMY_OPENKBP_PT1",
        ct_hu=ct_hu,
        planned_dose_gy=planned_dose_gy,
        possible_dose_mask=possible_dose_mask,
        voxel_dims_mm=voxel_dims_mm,
        roi_masks=roi_masks,
        sha256_inputs=sha_map,
    )


def build_synthetic_phantom_anatomy(shape: tuple[int, int, int] = (48, 48, 48)) -> PatientAnatomyVolume:
    """Fast lightweight 3D head-and-neck-like phantom for rapid unit tests."""
    nx, ny, nz = shape
    zz, yy, xx = np.meshgrid(
        np.linspace(-64.0, 64.0, nz),
        np.linspace(-64.0, 64.0, ny),
        np.linspace(-64.0, 64.0, nx),
        indexing="ij",
    )
    # Body cylinder
    body = (xx**2 + yy**2) <= 52.0**2
    ct_hu = np.where(body, 20.0, -1000.0).astype(np.float32)

    # PTV70 ellipsoid at center-anterior
    ptv70 = ((xx / 22.0) ** 2 + ((yy + 6.0) / 18.0) ** 2 + (zz / 24.0) ** 2) <= 1.0
    # SpinalCord posterior to PTV70
    cord = ((xx / 7.5) ** 2 + ((yy - 22.0) / 7.5) ** 2 + (zz / 36.0) ** 2) <= 1.0
    # Left & Right Parotids lateral to PTV70
    l_parotid = (((xx - 28.0) / 10.0) ** 2 + ((yy - 4.0) / 12.0) ** 2 + (zz / 16.0) ** 2) <= 1.0
    r_parotid = (((xx + 28.0) / 10.0) ** 2 + ((yy - 4.0) / 12.0) ** 2 + (zz / 16.0) ** 2) <= 1.0

    # Smooth realistic planned dose distribution (70 Gy in PTV70, steep falloff toward cord ~42 Gy max)
    dist_ptv = np.sqrt((xx / 24.0) ** 2 + ((yy + 6.0) / 20.0) ** 2 + (zz / 26.0) ** 2)
    dose = 71.5 / (1.0 + np.exp(4.8 * (dist_ptv - 1.05)))
    # Extra cord sparing indentation
    dist_cord = np.sqrt((xx / 9.5) ** 2 + ((yy - 22.0) / 9.5) ** 2)
    dose = dose * (1.0 - 0.35 * np.exp(-0.5 * dist_cord**2))
    dose = np.where(body, dose, 0.0).astype(np.float32)

    voxel_mm = (128.0 / nx, 128.0 / ny, 128.0 / nz)
    return PatientAnatomyVolume(
        patient_id="synthetic_hn_phantom",
        anatomy_source="Synthetic Head-and-Neck Verification Phantom",
        provenance_tag="SYNTHETIC_FAULT_INJECTION_BENCHMARK",
        ct_hu=ct_hu,
        planned_dose_gy=dose,
        possible_dose_mask=body,
        voxel_dims_mm=voxel_mm,
        roi_masks={
            "PTV70": ptv70,
            "SpinalCord": cord,
            "LeftParotid": l_parotid,
            "RightParotid": r_parotid,
        },
        sha256_inputs={"phantom": "deterministic_synthetic_v1"},
    )
