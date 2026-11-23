# Independent Technical & Scientific Audit Report — `linac-forensics` (v0.2.0)

> **Research Software — NOT A MEDICAL DEVICE.**
> **Audit Date:** 2026-11-23 | **Engine Version:** `0.2.0` | **Python Runtime:** CPython `3.12.16` via `uv`

---

## 1. Executive Audit Verdict

| Audit Dimension | Requirement | Verified Outcome | Status |
| --- | --- | --- | :---: |
| **Test Suite & Warnings** | `>= 25` tests, `0` warnings (`-W error`) | **38 passed, 0 warnings** | PASS |
| **Code Coverage** | `>= 75%` total line coverage | **95.71% total coverage** (all modules `>= 89.2%`) | PASS |
| **Static Lint (`ruff`)** | `0` errors across `src/`, `tests/`, `scripts/` | **0 errors** | PASS |
| **Binary `.bin` & Dynalog `.dlg`** | Varian TrueBeam v3/v4/v5 + Legacy Dynalog B | `.bin` v5 max error `5.72e-6 mm`; `.dlg` A/B exact `0.01 mm` count round-trip | PASS |
| **Zero-Error Identity (w/ Heterogeneity)** | `actual == expected` $\implies \Delta\Phi=0, \Delta D=0, \gamma=100\%$ | Exact `0.0`, `0.0`, `100.0%` (2D & 3D, CT HU-to-$\rho_e$ enabled) | PASS |
| **Causal Attribution (5 Fault Modes)** | Viscous drag, Coulomb stiction, DLG, sag, MU + Joint | `7/7` scenarios exact; stiction `0.7015 / 0.70 mm`; joint 4-fault `99.8%–100.0%` | PASS |
| **EPID Three-Way Silent Faults** | Catch encoder-blind faults missed by `.bin` logs | Plan-vs-Log `100.0%` (blind) vs Log-vs-EPID `55.4%` (`ENCODER_BLIND_SILENT_FAULT`) | PASS |
| **Kalman State-Space RUL Forecast** | Predictive RUL + 95% CI prior to TG-142 breach | Cutoff Day 20 predicts breach in `6.8 d` (actual Day 26, `0.8 d` error; `72%` 10d prob) | PASS |
| **Multi-Patient OpenKBP Cohort** | CC BY 4.0 + verbatim `LICENSE` + SHA-256 `MANIFEST.json` | `pt_1`, `pt_2`, `pt_3` (36 CSVs total) SHA-256 verified against `ababier/open-kbp` | PASS |
| **Visual Figure Audit** | 7 × 300-DPI PNG + PDF visually inspected | All 7 PNGs inspected via `read_file`; zero overlaps or defects | PASS |
| **MCP JSON-RPC 2.0 Server** | 8 typed tools + `resources/list` & `resources/read` | All 8 tools, resource URIs, abstention codes & clean pipe exit verified | PASS |
| **Autonomous Fleet Critic Gate** | Independent verifier over triage, RUL & work order | **6/6 checks PASSED**, SHA-256 signed ledger emitted | PASS |

---

## 2. Physics Invariant & Quantitative Verification Highlights

- **CT HU-to-$\rho_e$ Heterogeneity-Aware Perturbative 3D Replay & Nelms Blindspot**:
  - Zero-error identity verified exact (`0.0` $\Delta\Phi$, `0.0 Gy` $\Delta D$, `100.0%` 2D/3D gamma) with CT heterogeneity enabled.
  - Across 32 faulted deliveries on `OpenKBP pt_1`, **`17/17` (100%)** of clinically unacceptable deliveries (`|PTV70 ΔD95| ≥ 2%` or `OAR ΔD ≥ 1.5 Gy`) passed conventional 2D `3%/2mm` gamma (`≥95%`), whereas **3D `2%/1mm` gamma** and **3D `ΔDVH`** caught all `17/17`.
- **Three-Way Closed-Loop EPID Forensics**:
  - Verified all 4 clinical classifications (`CONCORDANT_NOMINAL`, `LOG_VERIFIED_MECHANICAL_FAULT`, `ENCODER_BLIND_SILENT_FAULT`, and `EPID_CALIBRATION_OR_PANEL_DRIFT` via multiplicative core-field TG-142 symmetry tilt regression).
- **Nonlinear Coulomb/Stribeck Stiction vs. Viscous Drag**:
  - 3-column orthogonal basis `[1, -v, -tanh(v/5)]` cleanly separates pure Coulomb stiction (`0.7015 mm` recovered vs `0.70 mm` injected, `95.5%` variance explained, BIC winner `MLC_COULOMB_STICTION`) from linear viscous drag (`42.06 ms` vs `42.0 ms`, `MLC_LEAF_MOTOR_FRICTION`) and isolates both simultaneously under `JOINT_MULTI_FAULT`.
- **Kalman RUL & Multi-Fraction Cumulative 3D Dose Accumulation**:
  - Local-linear-trend Kalman filter on `TrueBeam-01` (cutoff Day 20) predicts TG-142 `0.70 mm` MLC RMS breach in **`6.8 days`** (`95% CI [3.63, 23.88] d`, actual breach Day 26).
  - Multi-fraction accumulation on real `OpenKBP pt_2` (5/35 fractions) decomposes `-2.64%` cumulative `PTV70 ΔD95` into `-2.64%` systematic + `0.30%` random RMS and computes a compensatory remaining-fraction MU scaling factor (`1.00441`).

---

## 3. Visual Figure Audit Log
- **Figures 1–6**: Audited across Pass 1 and Pass 2 (4 Pass-1 layout collisions resolved and re-verified clean; `fig2` and `fig3` re-audited after enabling CT HU-to-$\rho_e$ heterogeneity).
- **Figure 7 (`fig7_epid_threeway_and_kalman_rul.png`)**: Audited directly via `read_file` — crisp Panel A EPID silent-fault anomaly streak, textbook Panel B viscous vs. Stribeck stiction separation, and Panel C Kalman RUL 95% credible envelope enclosing held-out Days 21–30 with zero text or legend collisions.

---

## 4. Known Scope Boundaries & Disclaimers
1. The heterogeneity-aware perturbative 3D dose kernel is a linearized pencil-beam / radiological-depth perturbation over a clinically computed reference dose grid (`OpenKBP pt_1–pt_3`), not a full Monte Carlo TPS engine.
2. Trajectory logs, EPID synthetic injections, and 30-day TG-142 timelines are controlled synthetic physics benchmarks explicitly tagged in `ProvenanceRecord`, separated from real OpenKBP CC BY 4.0 anatomies.
3. **Not a medical device**: never use for clinical QA sign-off, patient treatment modification, or linac calibration.
