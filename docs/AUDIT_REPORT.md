# Independent Technical & Scientific Audit Report — `linac-forensics` (v0.3.0)

> **Non-Commercial Research & Personal Use Only — NOT A MEDICAL DEVICE.**
> **Audit Date:** 2026-11-24 | **Engine Version:** `0.3.0` | **Python Runtime:** CPython `3.12.16` via `uv`

---

## 1. Executive Audit Verdict

| Audit Dimension | Requirement | Verified Outcome | Status |
| --- | --- | --- | :---: |
| **Test Suite & Warnings** | `>= 25` tests, `0` warnings (`-W error`) | **44 passed, 0 warnings** | PASS |
| **Code Coverage** | `>= 75%` total line coverage | **95.86% total coverage** (all modules `>= 89.2%`) | PASS |
| **Static Lint (`ruff`)** | `0` errors across `src/`, `tests/`, `scripts/` | **0 errors** | PASS |
| **License & Governance** | Non-commercial research w/ mandatory citation + personal use | Custom Non-Commercial License in `LICENSE` + `CITATION.cff` + `NOT A MEDICAL DEVICE` | PASS |
| **Binary `.bin` & Dynalog `.dlg`** | Varian TrueBeam v3/v4/v5 + Legacy Dynalog B | `.bin` v5 max error `5.72e-6 mm`; `.dlg` A/B exact `0.01 mm` round-trip | PASS |
| **Zero-Error Identity (w/ Heterogeneity)** | `actual == expected` $\implies \Delta\Phi=0, \Delta D=0, \gamma=100\%$ | Exact `0.0`, `0.0`, `100.0%` (2D & 3D, CT HU-to-$\rho_e$ enabled) | PASS |
| **5-Mode Causal Attribution (BIC)** | Viscous drag, Coulomb stiction, DLG, sag, MU + Joint | Exact classification; stiction `0.7015 / 0.70 mm`; joint 4-fault `99.8%–100.0%` | PASS |
| **EPID Three-Way Silent Faults** | Catch encoder-blind faults missed by `.bin` logs | Plan-vs-Log `100.0%` (blind) vs Log-vs-EPID `55.4%` (`ENCODER_BLIND_SILENT_FAULT`) | PASS |
| **Kalman State-Space RUL Forecast** | Predictive RUL + 95% CI prior to TG-142 breach | Cutoff Day 20 predicts breach in `6.8 d` (actual Day 26, `0.8 d` error; `72%` 10d prob) | PASS |
| **Inverse Tolerance & Fleet Routing (v0.3)** | Plan budgets + 95% adversarial certs + leaf outage | **4 TG-142 Stealth Risks** detected on passing linacs; `4.4×–4.9×` safety gain via safe transfer | PASS |
| **Multi-Patient OpenKBP Cohort** | CC BY 4.0 + verbatim `LICENSE` + SHA-256 `MANIFEST.json` | `pt_1`, `pt_2`, `pt_3` (36 CSVs total) SHA-256 verified against `ababier/open-kbp` | PASS |
| **Visual Figure Audit** | 8 × 300-DPI PNG + PDF visually inspected | All 8 PNGs visually audited via `read_file`; zero overlaps or defects | PASS |
| **MCP JSON-RPC 2.0 Server** | 10 typed tools + `resources/list` & `resources/read` | All 10 tools, resource URIs, abstention codes & clean pipe exit verified | PASS |
| **Autonomous Fleet Critic Gate** | Independent verifier over triage, RUL & work order | **6/6 checks PASSED**, SHA-256 signed ledger emitted | PASS |

---

## 2. v0.3.0 Flagship Inverse Robustness & Fleet-Safe Routing Verification

- **Plan-Specific Inverse DLG Tolerance Budgets (Real OpenKBP `pt_1–pt_3`)**:
  - `PT-SBRT-002` (Spine SBRT): `|ΔDLG| ≤ 0.17 mm`, `τ_MLC ≤ 22.8 ms` (**tighter than static TG-142** `0.50 mm` / `35 ms`).
  - `PT-HN-001` (H&N SIB 70 Gy): `|ΔDLG| ≤ 0.21 mm` (**2.4× tighter than static TG-142**).
  - `PT-PR-005` (Prostate VMAT): `|ΔDLG| ≤ 0.98 mm` (wide slack; safe on drifting linac).
  - `PT-BR-006` (Whole Breast): `|ΔDLG| ≤ 2.02 mm` (4× wider than static TG-142).
- **TG-142 Stealth Risk Detection & Safe Transfer Matrix**:
  - All 3 live linacs (`TrueBeam-01`, `TrueBeam-02`, `TrueBeam-03`) pass every static AAPM TG-142 action limit (`PASSING_TG142`).
  - Under 95% Bonferroni-exact ellipsoidal adversarial certification, `TrueBeam-01` (`DLG=+0.34 mm`, `Leaf #34 drag=24 ms`, `sag=0.52 mm`) triggers **4 TG-142 Stealth Risks** (`margin = 0.34×–0.65×`), automatically recommending transfer to `TrueBeam-02` (`margin = 1.61×–2.89×`, **`4.4×–4.9×` safety gain**) while retaining Prostate (`1.47×`) and Whole Breast (`1.34×`) on `TrueBeam-01`.
- **Counterfactual Leaf-Outage 3D Replay (`OpenKBP pt_1`, Bank B Leaf #34)**:
  - `PARKED_CLOSED`: `PTV70 ΔD95 = -56.48%`, `3D γ(2%/1mm) = 93.7%` → **`HOLD_UNTIL_LEAF_REPAIRED`**.
  - `ADJACENT_LEAF_AVERAGE`: `PTV70 ΔD95 = +0.00%`, `Mandible ΔD0.03cc = +0.01 Gy`, `3D γ(2%/1mm) = 100.0%` → **`SAFE_TO_TREAT_WITH_OUTAGE`**.

---

## 3. Visual Figure Audit Log (Figures 1–8)
- **Figures 1–7**: Visually audited and verified zero-overlap in v0.1–v0.2.
- **Figure 8 (`fig8_delivery_robustness_and_routing.png`)**:
  - **Pass 1 audit caught**: Panel A `lower right` legend overlapped the `2.02 mm` bar label; Panel C green Prostate boundary sat near the upper-right margin and `"Passes TG-142, Breaches H&N!"` touched the crimson H&N line.
  - **Pass 2 re-audit**: Legend moved cleanly to upper-right whitespace in Panel A; Panel C axes expanded to `[-0.56, 0.68] × [-2.25, 3.35]` with curved leader arrow from inside the open TG-142 box directly to $\theta^*_{\text{adv}}$. All 3 panels verified 100% collision-free.

---

## 4. License & Medical Device Disclaimer
- Licensed strictly under the **linac-forensics Non-Commercial Research & Personal Use License** ([LICENSE](../LICENSE)) — **commercial use is prohibited** without prior written commercial authorization; academic/research use requires citing [CITATION.cff](../CITATION.cff).
- **Not a medical device**: never use for clinical QA sign-off, patient treatment modification, or linac calibration.
