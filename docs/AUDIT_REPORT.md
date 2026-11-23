# Independent Technical & Scientific Audit Report — `linac-forensics`

> **Research Software — NOT A MEDICAL DEVICE.**
> **Audit Date:** 2026-11-23 | **Engine Version:** `0.1.0` | **Python Runtime:** CPython `3.12.16` via `uv`

---

## 1. Executive Audit Verdict

| Audit Dimension | Requirement | Verified Outcome | Status |
| --- | --- | --- | :---: |
| **Test Suite & Warnings** | `>= 25` tests, `0` warnings (`-W error`) | **30 passed, 0 warnings** | PASS |
| **Code Coverage** | `>= 75%` total line coverage | **95.4% total coverage** (all modules `>= 89%`) | PASS |
| **Static Lint (`ruff`)** | `0` errors across `src/`, `tests/`, `scripts/` | **0 errors** | PASS |
| **Binary `.bin` Round-Trip** | Varian TrueBeam v3.0 / v4.0 / v5.0 + CRC32 | Write `1.72 ms`, parse `3.57 ms`, max leaf error `5.72e-6 mm` | PASS |
| **Zero-Error Identity Invariant** | `actual == expected` $\implies \Delta\Phi=0, \Delta D=0, \gamma=100\%$ | Exact `0.0`, `0.0`, `100.0%` (2D & 3D) | PASS |
| **Causal Fault Attribution** | Single- & 4-fault BIC recovery + 95% CIs | `6/6` scenarios correctly classified; `99.8%–100.0%` joint recovery (`99.6%` var. explained) | PASS |
| **Hard Abstention Gates** | All 5 physical/binary safety codes tested | `TIMESTAMP_DESYNC`, `MU_CONSERVATION_VIOLATION`, `UNMAPPED_IEC61217_COORDINATES`, `CORRUPTED_BINARY_LOG`, `INTERLOCK_INCOMPLETE_DELIVERY` verified | PASS |
| **Real OpenKBP `pt_1` Provenance** | CC BY 4.0 + verbatim `LICENSE` + SHA-256 `MANIFEST.json` | All 12 CSVs SHA-256 verified against upstream `ababier/open-kbp` | PASS |
| **Visual Figure Audit** | 6 × 300-DPI PNG + PDF visually inspected | All 6 PNGs inspected; 4 layout overlaps detected on Pass 1, fixed, and re-verified clean on Pass 2 | PASS |
| **MCP JSON-RPC 2.0 Server** | 6 typed tools + stdio handshake + structured abstention | All 6 tools + protocol methods + error codes verified | PASS |
| **Autonomous Fleet Critic Gate** | Independent verifier over triage & work order | **6/6 checks PASSED**, SHA-256 signed ledger emitted | PASS |

---

## 2. Physics Invariant & Quantitative Verification

### 2.1 Binary Trajectory Log Fidelity & Abstention Safety
- Verified native binary serialization and deserialization across Varian TrueBeam header versions **v3.0**, **v4.0**, and **v5.0** (1024-byte header, 560-byte subbeam table, 50 Hz `float32` snapshot arrays, 120 Bank A/B leaves + 2 carriages, and v5.0 CRC32 trailer).
- Maximum floating-point round-trip deviation across 360 snapshots × 120 leaves: **`5.72 × 10⁻⁶ mm`** (exact IEEE-754 `float32` precision floor).
- Verified deterministic hard `AbstentionError` triggers on corrupted CRC32 bytes, truncated payloads, invalid `"VOSTL"` signatures, non-monotonic timestamps, clock jitter dropouts, decreasing/grossly non-conserved MU, interlock-aborted deliveries, non-IEC coordinate systems, out-of-range gantry angles, and inverted leaf-bank collisions.

### 2.2 Perturbative Fluence-to-3D-Dose Replay & Nelms et al. (2011) Blindspot
- Verified on real **OpenKBP `pt_1`** (`128 × 128 × 128`, voxel dimensions `3.906 × 3.906 × 2.5 mm`, 8 anatomical ROIs):
  - **Zero-error identity**: `max(|ΔΦ|) = 0.0`, `max(|ΔD|) = 0.0 Gy`, `2D γ(3%/2mm) = 100.0%`, `3D γ(3%/2mm) = 100.0%`, `ΔD95 = 0.0`, `ΔD0.03cc = 0.0`, `ΔTCP/NTCP = 0.0`.
  - **Nelms et al. (2011) 2D Gamma Blindspot Benchmark (32 faulted VMAT deliveries)**:
    - `16/32` deliveries produced clinically unacceptable 3D DVH shifts (`|PTV70 ΔD95| >= 2.0%` or `OAR ΔD >= 1.5 Gy`).
    - **`16/16` (100%)** of those clinically failing plans still passed conventional **2D planar 3%/2mm Gamma (`>= 95.0%`)**, because 2 mm DTA absorbs sub-millimeter systematic DLG aperture errors (`±0.6–1.0 mm`).
    - **`16/16` (100%)** were caught immediately by **3D volumetric 2%/1mm Gamma (`< 95%`)** and direct **3D perturbative `ΔDVH`**.

### 2.3 Causal Electro-Mechanical Fault Attribution (Bayesian BIC)
- Across 6 controlled scenarios (`σ_noise = 0.025 mm`):
  - **Nominal**: `NULL_NOMINAL` selected (`w = 0.972`).
  - **MLC Leaf #34 Viscous Drag (`42.0 ms`)**: isolated Bank B Leaf #34, `τ̂ = 42.06 ms` (`95% CI [41.99, 42.12]`), `ΔBIC = +73,938`, `w = 1.000`.
  - **Systematic DLG Offset (`+0.55 mm`)**: `Δgap̂ = +0.5496 mm` (`95% CI [0.5457, 0.5535]`), `ΔBIC = +77,171`, `w = 1.000`.
  - **Gantry Gravity-Sag Harmonic (`R = 1.006 mm, φ = -26.6°`)**: `R̂ = 1.0078 mm, φ̂ = -26.3°`, `ΔBIC = +45,461`, `w = 1.000`.
  - **Ion-Chamber MU Drift (`-2.40%`)**: `δ̂ = -2.400%` (`95% CI [-2.439, -2.361]`), `ΔBIC = +40,162`, `w = 1.000`.
  - **Joint 4-Subsystem Simultaneous Fault**: `JOINT_MULTI_FAULT` won (`ΔBIC = +88,024`, `w = 1.000`, **`99.57%` variance explained**), recovering all 4 parameters within `0.2%` of ground truth.

### 2.4 Global Saltelli–Jansen QMC Sobol' Sensitivity (`N = 3,072` evals/metric)
- Analytical validation on linear benchmark `Y = 3X₁ + X₂ + 0X₃` recovered `S₁ = (0.90, 0.10, 0.00)` within `< 0.03`.
- On real **OpenKBP `pt_1`**:
  - **PTV70 `ΔD95 (%)`**: MU Output Drift (`S₁ = 0.471`, `S_T = 0.472`), MLC Viscous Lag (`S₁ = 0.331`, `S_T = 0.333`), DLG Drift (`S₁ = 0.181`, `S_T = 0.182`), Gantry Sag (`S₁ = 0.015`, `S_T = 0.015`).
  - **SpinalCord `ΔD0.03cc (Gy)`**: MU Output Drift (`S₁ = 0.409`, `S_T = 0.416`), **Gantry Gravity Sag (`S₁ = 0.308`, `S_T = 0.315`)**, DLG Drift (`S₁ = 0.214`, `S_T = 0.224`), MLC Viscous Lag (`S₁ = 0.055`, `S_T = 0.064`).

---

## 3. Visual Figure Audit Log

All 6 figures (`300 DPI` PNG + vector PDF) were visually inspected via direct raster image review:
- **Pass 1 defects detected**:
  1. `fig2` Panel B: bold panel label `B` collided with `"OpenKBP"` in the subplot title; slice `z=69` omitted Parotid contours.
  2. `fig3` Panel B: orange `PTV70 |ΔD95|` curve intersected the bottom-center legend box.
  3. `fig4` Panel B: bottom-left legend overlapped the `MU Drift` horizontal bar.
  4. `fig5` Panel B: `PT-HN-001` and `PT-LG-005` callout boxes touched; vertical tolerance shading stopped at `y = -0.5`.
- **Pass 2 re-audit**: All 4 defects fixed (including selecting anatomical slice `z = 54` showing PTV70, SpinalCord, LeftParotid, and RightParotid simultaneously in `fig2`, clearing legend whitespace in `fig3`/`fig4`, adding non-overlapping leader-line badges in `fig5`, and eliminating matplotlib font warnings). All 6 figures re-verified clean with zero overlaps.

---

## 4. Known Scope Boundaries & Disclaimers

1. The fast perturbative 3D dose kernel is a linearized pencil-beam / radiological depth-attenuation perturbation over a clinically computed reference dose grid (`OpenKBP pt_1`), not a full Monte Carlo or collapsed-cone convolution/superposition TPS engine.
2. Trajectory logs and multi-linac TG-142 timelines are controlled synthetic physics benchmarks explicitly tagged in `ProvenanceRecord` (`SYNTHETIC_FAULT_INJECTION_BENCHMARK` / `SYNTHETIC_TG142_FLEET_DRIFT`) and kept strictly distinct from real OpenKBP `pt_1` anatomy (`OPEN_PUBLIC_ANATOMY_OPENKBP_PT1`).
3. **Not a medical device**: never use for clinical QA sign-off, patient treatment modification, or linac calibration.
