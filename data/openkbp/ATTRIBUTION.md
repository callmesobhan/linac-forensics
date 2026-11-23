# OpenKBP Multi-Patient Dataset Attribution (`pt_1`, `pt_2`, `pt_3`)

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
