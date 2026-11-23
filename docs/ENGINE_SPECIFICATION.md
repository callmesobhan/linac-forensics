# `linac-forensics` Physics Engine & Architecture Specification (v0.2.0)

> **Research Software — NOT A MEDICAL DEVICE.** `linac-forensics` is designed exclusively for medical physics research, algorithmic QA benchmarking, and closed-loop telemetry forensics education.

---

## 1. System Architecture Overview

`linac-forensics` couples five strictly typed scientific computing layers:

1. **Multi-Format Telemetry & Binary Ingestion Layer (`ir.py`, `log_parser.py`, `provenance.py`)**:
   - Native binary encoder/decoder for **Varian TrueBeam Trajectory Logs (`.bin`, header v3.0 / v4.0 / v5.0)** with 1024-byte header, 560-byte subbeam tables, 20–50 Hz expected/actual `float32` snapshots (gantry, collimator, jaws X1/X2/Y1/Y2, cumulative MU, beam-hold state, 2 carriages, and 120 Bank A/B MLC leaves), IEC 61217 scaling, and v5.0 CRC32 verification.
   - Paired **Varian Legacy Dynalog (`.dlg` Version B)** Bank A/B parser & writer (`parse_varian_dynalog_dlg`, `write_varian_dynalog_dlg`) plus vendor-neutral JSON and CSV telemetry parsers mapped onto the identical Pydantic v2 `TrajectoryLogTensor` IR.
   - Five non-bypassable **`AbstentionError`** integrity gates:
     - `TIMESTAMP_DESYNC`, `MU_CONSERVATION_VIOLATION`, `INTERLOCK_INCOMPLETE_DELIVERY`, `UNMAPPED_IEC61217_COORDINATES`, `CORRUPTED_BINARY_LOG`.

2. **CT-Heterogeneity-Aware Perturbative Fluence-to-3D-Dose Replay Engine (`perturbative_dose.py`, `anatomy.py`)**:
   - Computes angular sector-resolved 2D planned fluence $\Phi_{\text{plan}}(u, v)$ and delivered fluence $\Phi_{\text{deliv}}(u, v)$ across all beam-on snapshots (`beam_hold == 0`) with rounded leaf-tip DLG transmission, interleaf leakage $T_{\text{MLC}} = 0.015$, Gaussian penumbra $\sigma_{\text{pen}} = 2.2\text{ mm}$, jaw clipping, and sagittal head shift $\Delta y_{\text{sag}}(\theta)$.
   - Converts real 3D patient CT Hounsfield Units (**OpenKBP `pt_1`, `pt_2`, `pt_3`**, CC BY 4.0, SHA-256 verified) to relative electron density $\rho_e(x,y,z)$ via piecewise-linear clinical HU-to-$\rho_e$ calibration and projects angular fluence differences $\Delta\Phi_k(u,v)$ via vectorized BEV rotation, density-weighted radiological depth attenuation $\exp(-\mu_{\text{eff}} \bar{\rho}_e d_{\text{BEV}})$, and density-modulated scatter:
     $$D_{\text{deliv}}(x, y, z) = \max\left(0,\; D_{\text{plan}}(x, y, z) + \Delta D(x, y, z)\right)$$
   - **Formal Zero-Error Identity Invariant**: Whenever `actual == expected` on every telemetry channel, $\Delta\Phi \equiv 0$, $\Delta D \equiv 0$, $\Delta\text{DVH} \equiv 0$, and 2D/3D gamma pass rates equal $100.0\%$ exactly.

3. **Three-Way Closed-Loop EPID Portal Dosimetry Forensics (`epid_forensics.py`)**:
   - Forward-models and deconvolves a-Si EPID portal images (Neumann-series optical/backscatter PSF inversion) to compare **Planned TPS ($\Phi_{\text{plan}}$)** vs. **Trajectory Log ($\Phi_{\text{log}}$)** vs. **Measured EPID ($\Phi_{\text{EPID}}$)**.
   - Automatically classifies `CONCORDANT_NOMINAL`, `LOG_VERIFIED_MECHANICAL_FAULT`, `EPID_CALIBRATION_OR_PANEL_DRIFT` (AAPM TG-142 symmetry tilt $\ge 2.0\%/10\text{ cm}$), and **`ENCODER_BLIND_SILENT_FAULT`** (slipped encoders or unrecorded jaw/DLG offsets invisible to trajectory logs alone).

4. **Physics-Informed Electro-Mechanical Causal Attribution Engine (`causal_fault_model.py`)**:
   - Jointly models five orthogonalized physical failure signatures across active telemetry channels:
     1. **MLC Leaf Motor Viscous Drag**: $-\tau_{b,i} v_{b,i}(t)$ per active leaf.
     2. **MLC Coulomb / Stribeck Stiction & Leadscrew Backlash**: $-\beta_{b,i} \tanh(v_{b,i}(t)/v_0)$ ($v_0 = 5\text{ mm/s}$).
     3. **Systematic Bank / DLG Calibration Offset**: $\Delta\text{gap} = c$ across non-flagged active leaf pairs.
     4. **Gantry Gravity-Sag Harmonic**: $\Delta y(\theta) = A \sin\theta + B \cos\theta$.
     5. **Monitor Ion-Chamber / Dose-Rate Calibration Drift**: $\Delta\text{MU}_{\text{act}}(t) = (1 + \delta)\Delta\text{MU}_{\text{exp}}(t)$.
   - Evaluates `NULL_NOMINAL`, all 5 single-fault models, and `JOINT_MULTI_FAULT` via standardized multi-channel Gaussian **AIC / BIC** ($\Delta\text{BIC}$ vs null, Schwarz posterior weights, matched-filter SNR, and 95% CIs).

5. **Kalman State-Space RUL, Multi-Fraction Accumulation, Sobol' & MCP (`predictive_rul.py`, `sobol_sensitivity.py`, `harness/`)**:
   - **Local-Linear-Trend Kalman Filter (`forecast_tg142_kalman_rul`)**: estimates drift velocity, multi-step 95% credible envelopes, horizon breach probability, and probabilistic Remaining Useful Life (RUL) before TG-142 action limits.
   - **Multi-Fraction Cumulative 3D Dose Accumulation (`accumulate_multifraction_course`)**: separates systematic vs. random delivery deviations across multiple fractions and computes compensatory remaining-course adaptive MU scaling.
   - **Exact Saltelli–Jansen QMC Sobol' (`sobol_sensitivity.py`)**: global variance decomposition ($S_1$, $S_T$, 95% bootstrap CIs) calibrated on real OpenKBP anatomy.
   - **JSON-RPC 2.0 Stdio MCP Server (`harness/mcp_server.py`)**: exposes **8 typed provenance-gated tools** plus live read-only **`resources/list` & `resources/read`** endpoints.

---

## 2. Coordinate Conventions (IEC 61217 & OpenKBP)

- **Linear Accelerator Collimation (IEC 61217)**: Bank A resides on $+X$, Bank B on $-X$, aperture opening $g_i = x_{A,i} - x_{B,i}$ (mm). `.bin` stores linear axes in `cm` (converted to `mm`); `.dlg` stores leaf positions in `0.01 mm` integer encoder counts.
- **OpenKBP `pt_1`, `pt_2`, `pt_3` Patient Volumes (`128 × 128 × 128`)**: Axis 0 = AP, Axis 1 = LR, Axis 2 = SI; isocenter is placed automatically at the `PTV70` centroid for BEV coordinate rotation.
