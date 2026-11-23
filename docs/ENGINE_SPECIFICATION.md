# `linac-forensics` Physics Engine & Architecture Specification

> **Research Software — NOT A MEDICAL DEVICE.** `linac-forensics` is designed exclusively for medical physics research, algorithmic QA benchmarking, and closed-loop telemetry forensics education.

---

## 1. System Architecture Overview

`linac-forensics` couples four strictly typed scientific computing layers:

1. **Telemetry & Binary Ingestion Layer (`ir.py`, `log_parser.py`, `provenance.py`)**:
   - Native binary encoder/decoder for **Varian TrueBeam Trajectory Logs (`.bin`, header v3.0 / v4.0 / v5.0)** with 1024-byte header, 560-byte subbeam tables, 20–50 Hz expected/actual `float32` snapshots (gantry, collimator, jaws X1/X2/Y1/Y2, cumulative MU, beam-hold state, 2 carriages, and 120 Bank A/B MLC leaves), IEC 61217 scaling, and v5.0 CRC32 verification.
   - Vendor-neutral JSON and CSV telemetry parsers mapped onto the identical Pydantic v2 `TrajectoryLogTensor` IR.
   - Five non-bypassable **`AbstentionError`** integrity gates:
     - `TIMESTAMP_DESYNC` (non-monotonicity, negative $\Delta t$, or severe clock jitter)
     - `MU_CONSERVATION_VIOLATION` (negative/decreasing MU or $>25\%$ gross conservation discrepancy)
     - `INTERLOCK_INCOMPLETE_DELIVERY` (truncated header flag, premature abort under beam hold, or $>20\%$ interlock snapshots)
     - `UNMAPPED_IEC61217_COORDINATES` (non-IEC coordinate frame, out-of-domain angles, leaf travel $>200\text{ mm}$, or inverted bank collision)
     - `CORRUPTED_BINARY_LOG` (signature mismatch, unsupported version, payload truncation, or CRC32 mismatch)

2. **Fast Perturbative Fluence-to-3D-Dose Replay Engine (`perturbative_dose.py`, `anatomy.py`)**:
   - Computes angular sector-resolved 2D planned fluence $\Phi_{\text{plan}}(u, v)$ and delivered fluence $\Phi_{\text{deliv}}(u, v)$ across all beam-on snapshots (`beam_hold == 0`) with rounded leaf-tip Dosimetric Leaf Gap (DLG) transmission, interleaf leakage $T_{\text{MLC}} = 0.015$, Gaussian penumbra $\sigma_{\text{pen}} = 2.2\text{ mm}$, jaw clipping, and sagittal head shift $\Delta y_{\text{sag}}(\theta)$.
   - Projects angular fluence differences $\Delta\Phi_k(u, v) = \Phi_{\text{deliv},k} - \Phi_{\text{plan},k}$ into real 3D patient anatomy (**OpenKBP `pt_1`**, 128 × 128 × 128 CT + planned clinical dose + 8 OAR/PTV masks, verified via SHA-256 `MANIFEST.json`) via vectorized Beam's-Eye-View (BEV) rotation and 6 MV radiological depth attenuation ($\mu_{\text{eff}} = 0.0048\text{ mm}^{-1}$) + 3D lateral scatter kernel:
     $$D_{\text{deliv}}(x, y, z) = \max\left(0,\; D_{\text{plan}}(x, y, z) + \Delta D(x, y, z)\right)$$
   - **Formal Zero-Error Identity Invariant**: Whenever `actual == expected` on every telemetry channel, $\Delta\Phi \equiv 0$, $\Delta D \equiv 0$, $\Delta\text{DVH} \equiv 0$, and 2D/3D gamma pass rates equal $100.0\%$ to exact machine epsilon.
   - Evaluates exact volumetric ROI metrics: $D_{95}$ (Gy and %), $D_{0.03\text{cc}}$ (using exact OpenKBP voxel volume $0.03814\text{ cc}$), $D_{\text{mean}}$, Niemierko generalized Equivalent Uniform Dose ($\text{gEUD}$), logistic TCP/NTCP, and AAPM TG-218 global **2D and 3D Gamma** (`3%/2mm` vs `2%/1mm`, 10% low-dose threshold).
   - Explicitly detects the **Nelms et al. (2011) 2D Gamma Blindspot**: `gamma_2d_3pct_2mm >= 95.0%` coinciding with clinically unacceptable 3D DVH shifts ($|\Delta D_{95}(\text{PTV})| \ge 2.0\%$ or $\Delta D_{0.03\text{cc}}(\text{OAR}) \ge 1.5\text{ Gy}$).

3. **Physics-Informed Electro-Mechanical Causal Attribution Engine (`causal_fault_model.py`)**:
   - Jointly models four orthogonalized physical failure signatures across active telemetry channels:
     1. **MLC Leaf Motor Friction / T-Nut Drag**: $\Delta x_{b,i}(t) = c_{b,i} - \tau_{b,i} v_{b,i}(t)$ per active leaf $i$ on bank $b \in \{A, B\}$.
     2. **Systematic Bank / DLG Calibration Offset**: $\Delta\text{gap} = c$ across non-flagged active leaf pairs ($\pm c/2$ symmetric bank bias).
     3. **Gantry Gravity-Sag Harmonic**: $\Delta y(\theta) = A \sin\theta + B \cos\theta$, recovering amplitude $R = \sqrt{A^2 + B^2}$ and phase $\phi = \operatorname{atan2}(B, A)$.
     4. **Monitor Ion-Chamber / Dose-Rate Calibration Drift**: $\Delta\text{MU}_{\text{act}}(t) = (1 + \delta)\Delta\text{MU}_{\text{exp}}(t)$ via MU-weighted least squares.
   - Performs unified multi-channel standardized Gaussian likelihood comparison across `NULL_NOMINAL`, the 4 single-fault models, and `JOINT_MULTI_FAULT` via **AIC / BIC** ($\Delta\text{BIC}$ vs null, Schwarz posterior weights, matched-filter SNR, and 95% confidence intervals).

4. **Global Sobol' Sensitivity & Frontier Agent Harness (`sobol_sensitivity.py`, `harness/agent_orchestrator.py`, `harness/mcp_server.py`)**:
   - Exact **Saltelli–Jansen QMC Sobol'** variance decomposition ($S_1$, $S_T$, $S_T - S_1$ interactions, 95% bootstrap CIs) calibrated directly to real 3D OpenKBP `pt_1` dose responses.
   - **Autonomous Fleet Physicist & Incident-Triage Harness**: ingests multi-patient treatment-week logs + 30-day multi-linac TG-142/MPC drift trends, falsifies competing root causes numerically, ranks patients by clinical dose impact, drafts a Biomedical Service Work Order, and verifies all outputs via an independent 6-check **Critic-Verifier Gate** emitting signed JSON + Markdown ledgers.
   - **JSON-RPC 2.0 Stdio MCP Server**: exposes 6 typed, provenance-gated tools (`linac_parse_trajectory_log`, `linac_compute_perturbative_dvh`, `linac_diagnose_causal_fault`, `linac_compare_gamma_vs_dvh`, `linac_triage_patient_fleet`, `linac_run_sobol_sensitivity`) with structured abstention propagation.

---

## 2. Coordinate Conventions (IEC 61217 & OpenKBP)

- **Linear Accelerator Collimation (IEC 61217)**:
  - Bank A resides on the $+X$ side; Bank B resides on the $-X$ side; aperture opening is $g_i = x_{A,i} - x_{B,i}$ (mm).
  - Binary `.bin` files store linear axes in `cm` (converted to `mm` on parse) and angular axes (`Gantry`, `Collimator`) in `[0, 360]` degrees.
- **OpenKBP `pt_1` Patient Volume (`128 × 128 × 128`, voxel size `3.906 × 3.906 × 2.5 mm`)**:
  - Axis 0: Anterior–Posterior (AP)
  - Axis 1: Left–Right (LR)
  - Axis 2: Superior–Inferior (SI)
  - Isocenter is placed automatically at the `PTV70` centroid for BEV coordinate rotation.
