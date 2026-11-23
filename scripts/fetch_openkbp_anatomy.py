#!/usr/bin/env python3
"""Fetch and cryptographically verify OpenKBP pt_1, pt_2, pt_3 anatomies (CC BY 4.0).

Research Software — NOT A MEDICAL DEVICE.

Downloads open-access OpenKBP head-and-neck 3D CT, planned reference dose, voxel
dimensions, and OAR/PTV structure masks from `ababier/open-kbp`, copies the
verbatim upstream `LICENSE`, writes `ATTRIBUTION.md`, and emits SHA-256
`MANIFEST.json` files for each patient.
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

EXPECTED_PATIENTS: dict[str, dict[str, str]] = {
    "pt_1": {
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
    },
    "pt_2": {
        "ct.csv": "188a92faef4a96cc95be66037412b22dd9331e77282ad9bff7a559c54b1ba6c2",
        "dose.csv": "60dd44ef54759851c6d2592b88e07868af12a643710d33a610ef780afe5a18c5",
        "possible_dose_mask.csv": "545cc7c80e70f69d03b24399c916fdfc484e9559303e671600e43a7585189f67",
        "voxel_dimensions.csv": "d4fab7ca14982220a613cb45cd40f09ea920d0f0c1191f41bb0647663804c5ef",
        "Brainstem.csv": "ccbc0a607203706a66a1d6d57071f52496d8883c6a18ef638d48a00788305911",
        "SpinalCord.csv": "549625c9e4578c3ce4b0e58a03ae0121cc4b7b82b3f459bf8a838f947dfa3231",
        "RightParotid.csv": "d4b75163afa480a5adfdb423d6c92b3a8d85582d35f050b0ab8764ef37577ece",
        "LeftParotid.csv": "75ce575df672f9fac5de044dbf65c8bcd17fb67ff2a7056644163273587174ac",
        "Esophagus.csv": "3a0fb9d4673dce92f175528f330b0ac81bef72366fb605ca4adb0fbadaa44e5b",
        "Mandible.csv": "5ab261197c65a1bc1d7c666b2eaebebd3fa53be61fa81808bb944179b10b59b2",
        "PTV70.csv": "ac6b5df96a2e36116f887345de5ffd45007271ce664e2e3c368e583b9a3f5b35",
        "PTV63.csv": "7cd5c7a02003a21cef8947837c9414f6749a5ea8c02c0730de7eaad7bb0bb8fd",
        "PTV56.csv": "ee81d8e4b676b5c1dd1d78f1ac1bfa2623c679b38aa3786270bef968a7e5289a",
    },
    "pt_3": {
        "ct.csv": "7e6964b0c201c73aef4a1930d8e987652f29e4cbe2899a7e3ccf3f5c361bd0b2",
        "dose.csv": "a7ec3f253053617de5995c80fa533a4f752a334eb9e70f326630d79eed5b907d",
        "possible_dose_mask.csv": "627c0efe0029118e0a04d3e3b080e40692b76455dc48a5d170c7f97b9d2dfcac",
        "voxel_dimensions.csv": "0549346d527a7f1c918b34e583a4af36763881a60551dad07de51230383a7c8d",
        "Brainstem.csv": "3779d4955d250b1857c50995b9add5f2f2af40844a2dc60b5696582b733ad45f",
        "SpinalCord.csv": "a4c78f07f24c020826f9a4323b0c8d871d512f66688078fe8f4444b472930686",
        "RightParotid.csv": "8158c9d6ca6c3c8d56169e881f03a2d79c71c363a2948b819bf78fd5f0beda3c",
        "LeftParotid.csv": "2b02db9081752f6ef456fa31ca9ea258a529ec3d69fcc8030215bf5820a02982",
        "PTV70.csv": "054c8f1dfbff9b0578b3827cd046072ec0ed0d299771d7c375f1c57abaab6e5e",
        "PTV63.csv": "932035609831319bdf0b6387ac09bbade5e250bcea73d7eaf160e8da9f92318d",
        "PTV56.csv": "14e12b298718e811081cc844336b0acfcb48fe047cee7550b7b0f90be4e039c1",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "linac-forensics/0.2.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def fetch_openkbp_patients(
    output_root: Path,
    patients: list[str] | None = None,
    source_dir: Path | None = None,
) -> list[Path]:
    selected = patients or ["pt_1", "pt_2", "pt_3"]
    openkbp_root = output_root / "openkbp"
    openkbp_root.mkdir(parents=True, exist_ok=True)

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

    license_sha = sha256_file(license_dest)
    manifests: list[Path] = []

    for patient_id in selected:
        if patient_id not in EXPECTED_PATIENTS:
            raise ValueError(f"Unsupported OpenKBP patient_id '{patient_id}'")
        pt_dir = openkbp_root / patient_id
        pt_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"provided-data/train-pats/{patient_id}"

        manifest_files: dict[str, dict[str, str | int]] = {}
        for filename, expected_sha in EXPECTED_PATIENTS[patient_id].items():
            url = f"{RAW_BASE}/{prefix}/{filename}"
            dest = pt_dir / filename
            if source_dir is not None:
                src_candidate = source_dir / patient_id / filename
                if not src_candidate.exists():
                    src_candidate = source_dir / filename
                if not src_candidate.exists():
                    raise FileNotFoundError(f"Missing {patient_id}/{filename} in {source_dir}")
                shutil.copy2(src_candidate, dest)
            else:
                download_url(url, dest)

            actual_sha = sha256_file(dest)
            if actual_sha != expected_sha:
                dest.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA-256 mismatch for {patient_id}/{filename}: "
                    f"expected {expected_sha}, got {actual_sha}"
                )

            manifest_files[filename] = {
                "url": url,
                "bytes": dest.stat().st_size,
                "sha256": actual_sha,
            }

        manifest = {
            "dataset": "OpenKBP (Open-Access Knowledge-Based Planning Grand Challenge Dataset)",
            "provenance_tag": "OPEN_PUBLIC_ANATOMY_OPENKBP_PT1"
            if patient_id == "pt_1"
            else f"OPEN_PUBLIC_ANATOMY_OPENKBP_{patient_id.upper()}",
            "citation": (
                "Babier A, Zhang B, Mahmood R, Moore KL, Purdie TG, McNiven AL, Chan TCY. "
                "OpenKBP: The open-access knowledge-based planning grand challenge and dataset. "
                "Medical Physics. 2021;48(9):5549-5561. doi:10.1002/mp.14845"
            ),
            "repository": UPSTREAM_REPO,
            "license": "CC BY 4.0 (dataset) / MIT (upstream repository code)",
            "patient_id": patient_id,
            "split": "train-pats",
            "verified_utc": datetime.now(UTC).isoformat(),
            "upstream_license_sha256": license_sha,
            "files": manifest_files,
        }
        manifest_path = pt_dir / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests.append(manifest_path)

    attribution_text = """# OpenKBP Multi-Patient Dataset Attribution (`pt_1`, `pt_2`, `pt_3`)

This directory contains verbatim head-and-neck patient anatomies (`pt_1`, `pt_2`, `pt_3`:
3D CT, planned reference dose, voxel dimensions, and OAR/PTV structure masks) from the
**OpenKBP (Open-Access Knowledge-Based Planning) 2020 AAPM Grand Challenge** dataset.

- **Upstream Repository**: https://github.com/ababier/open-kbp
- **Dataset License**: Creative Commons Attribution 4.0 International (**CC BY 4.0**)
- **Upstream Code License**: MIT License (verbatim copy in `LICENSE`)
- **Primary Publication**:
  Babier A, Zhang B, Mahmood R, Moore KL, Purdie TG, McNiven AL, Chan TCY.
  *OpenKBP: The open-access knowledge-based planning grand challenge and dataset.*
  Medical Physics. 2021;48(9):5549–5561. https://doi.org/10.1002/mp.14845

## Usage in `linac-forensics`

`pt_1`, `pt_2`, and `pt_3` provide distinct real 3D head-and-neck radiotherapy anatomies
and CT HU electron-density volumes onto which `linac-forensics` projects 2D perturbative
fluence differences (`ΔΦ -> ΔDose`) computed from synthetic trajectory-log fault-injection
suites. Real anatomy and synthetic delivery logs are always tagged separately in the
engine's cryptographic `ProvenanceRecord`.
"""
    (openkbp_root / "ATTRIBUTION.md").write_text(attribution_text, encoding="utf-8")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify OpenKBP pt_1–pt_3 anatomies.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Destination data directory (default: <repo>/data).",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=["pt_1", "pt_2", "pt_3"],
        help="OpenKBP patient IDs to stage (default: pt_1 pt_2 pt_3).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional local OpenKBP mirror directory (still verified against upstream SHA-256).",
    )
    args = parser.parse_args()
    manifests = fetch_openkbp_patients(args.output_dir, args.patients, args.source_dir)
    for m in manifests:
        print(f"Verified OpenKBP staged: {m}")


if __name__ == "__main__":
    main()
