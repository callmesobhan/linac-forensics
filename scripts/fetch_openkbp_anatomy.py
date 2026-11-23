#!/usr/bin/env python3
"""Fetch and cryptographically verify OpenKBP pt_1 anatomy (CC BY 4.0).

Research Software — NOT A MEDICAL DEVICE.

Downloads the open-access OpenKBP `pt_1` 3D CT, planned dose, voxel dimensions,
and organ-at-risk / target structure masks from the official upstream GitHub
repository (`ababier/open-kbp`), copies the verbatim upstream `LICENSE`, writes
`ATTRIBUTION.md`, and emits a SHA-256 `MANIFEST.json`.

Supports `--source-dir` for reproducible offline verification against a local
verified copy of the upstream files (still enforcing exact SHA-256 checksums).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

UPSTREAM_REPO = "https://github.com/ababier/open-kbp"
RAW_BASE = "https://raw.githubusercontent.com/ababier/open-kbp/master"
PT1_PREFIX = "provided-data/train-pats/pt_1"

EXPECTED_FILES: dict[str, str] = {
    "ct.csv": "ed31b888e0fd1751ed8ec20c04c8f1575ff35811eb6497e1f0bc3a8b7ea8e68d",
    "dose.csv": "e850b853c7d6285992818cd0098a04bc4c22e81d582cb637452cabbf5c733f33",
    "possible_dose_mask.csv": "f6412f61bd0efbd067be6d294eab61675a0d4942f5aee282cd6d07035dce0e70",
    "voxel_dimensions.csv": "ffa866c090e5b655e8067820b1dcc7ae16e89d0cc593ff8e8f3dda7f5290a246",
    "Brainstem.csv": "acdf8de278d98967842d3e4540b2c821e711568be6a1c0fbcd7437a3dc4a1c45",
    "SpinalCord.csv": "fdaf5e9680d8195de73bccfb6d0712ca155c91ee698535cff29551defe072acd",
    "RightParotid.csv": "8eb279734796947e67e865fdfabdb20115b8190ad5a7e503b41f31c9c68b0bc9",
    "LeftParotid.csv": "f1f54e7d07b35d08715407580efa4ca136ed3deb108e184117bb42f7691f0755",
    "Mandible.csv": "632cbdc97d25ecad48b169eee141838498e75b5f4731050b3b0a447b0b3287e2",
    "PTV70.csv": "c7e5df95d5696ade43b54e67b80f8eafa0b0ee5d4bc1609f3b0d61e40ac2b97c",
    "PTV63.csv": "fabd944952a0d75c05c8c4d49da214381c018216f5134bf171b508c3f314f397",
    "PTV56.csv": "1c76ce104f9931df1dee422c6fe2270abbf865082bacbccbda96a92883de2873",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "linac-forensics/0.1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def fetch_openkbp_pt1(output_root: Path, source_dir: Path | None = None) -> Path:
    openkbp_root = output_root / "openkbp"
    pt1_dir = openkbp_root / "pt_1"
    pt1_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verbatim upstream LICENSE
    license_dest = openkbp_root / "LICENSE"
    if source_dir is not None:
        candidate_licenses = [
            source_dir / "LICENSE",
            source_dir / "UPSTREAM_LICENSE_OPENKBP.txt",
            source_dir.parent / "UPSTREAM_LICENSE_OPENKBP.txt",
        ]
        found = next((p for p in candidate_licenses if p.exists()), None)
        if found is None:
            raise FileNotFoundError(f"Upstream OpenKBP LICENSE not found near {source_dir}")
        shutil.copy2(found, license_dest)
    else:
        download_url(f"{RAW_BASE}/LICENSE", license_dest)

    # 2. Fetch or copy + SHA-256 verify all 12 pt_1 CSV files
    manifest_files: dict[str, dict[str, str | int]] = {}
    for filename, expected_sha in EXPECTED_FILES.items():
        url = f"{RAW_BASE}/{PT1_PREFIX}/{filename}"
        dest = pt1_dir / filename
        if source_dir is not None:
            src_candidate = source_dir / "pt_1" / filename
            if not src_candidate.exists():
                src_candidate = source_dir / filename
            if not src_candidate.exists():
                raise FileNotFoundError(f"Missing {filename} in {source_dir}")
            shutil.copy2(src_candidate, dest)
        else:
            download_url(url, dest)

        actual_sha = sha256_file(dest)
        if actual_sha != expected_sha:
            dest.unlink(missing_ok=True)
            raise ValueError(
                f"SHA-256 mismatch for {filename}: expected {expected_sha}, got {actual_sha}"
            )

        manifest_files[filename] = {
            "url": url,
            "bytes": dest.stat().st_size,
            "sha256": actual_sha,
        }

    # 3. Write ATTRIBUTION.md
    attribution_text = """# OpenKBP Dataset Attribution (`pt_1`)

This directory contains verbatim patient anatomy (`pt_1`: 3D CT, planned reference dose,
voxel dimensions, and OAR/PTV structure masks) from the **OpenKBP (Open-Access
Knowledge-Based Planning) 2020 AAPM Grand Challenge** dataset.

- **Upstream Repository**: https://github.com/ababier/open-kbp
- **Dataset License**: Creative Commons Attribution 4.0 International (**CC BY 4.0**)
- **Upstream Code License**: MIT License (verbatim copy in `LICENSE`)
- **Primary Publication**:
  Babier A, Zhang B, Mahmood R, Moore KL, Purdie TG, McNiven AL, Chan TCY.
  *OpenKBP: The open-access knowledge-based planning grand challenge and dataset.*
  Medical Physics. 2021;48(9):5549–5561. https://doi.org/10.1002/mp.14845

## Usage in `linac-forensics`

`pt_1` provides real 3D head-and-neck radiotherapy anatomy and OAR/PTV masks onto
which `linac-forensics` projects 2D perturbative fluence differences (`ΔΦ -> ΔDose`)
computed from synthetic Varian TrueBeam `.bin` trajectory-log fault-injection suites.
Real anatomy and synthetic delivery logs are always tagged separately in the engine's
cryptographic `ProvenanceRecord`.
"""
    (openkbp_root / "ATTRIBUTION.md").write_text(attribution_text, encoding="utf-8")

    # 4. Write MANIFEST.json
    manifest = {
        "dataset": "OpenKBP (Open-Access Knowledge-Based Planning Grand Challenge Dataset)",
        "provenance_tag": "OPEN_PUBLIC_ANATOMY_OPENKBP_PT1",
        "citation": (
            "Babier A, Zhang B, Mahmood R, Moore KL, Purdie TG, McNiven AL, Chan TCY. "
            "OpenKBP: The open-access knowledge-based planning grand challenge and dataset. "
            "Medical Physics. 2021;48(9):5549-5561. doi:10.1002/mp.14845"
        ),
        "repository": UPSTREAM_REPO,
        "license": "CC BY 4.0 (dataset) / MIT (upstream repository code)",
        "patient_id": "pt_1",
        "split": "train-pats",
        "verified_utc": datetime.now(UTC).isoformat(),
        "upstream_license_sha256": sha256_file(license_dest),
        "files": manifest_files,
    }
    manifest_path = pt1_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify OpenKBP pt_1 anatomy.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Destination data directory (default: <repo>/data).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional local OpenKBP mirror directory (still verified against upstream SHA-256).",
    )
    args = parser.parse_args()
    manifest_path = fetch_openkbp_pt1(args.output_dir, args.source_dir)
    print(f"Verified OpenKBP pt_1 staged successfully. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
