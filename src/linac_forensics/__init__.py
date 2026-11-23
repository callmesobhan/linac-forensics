"""linac-forensics: Closed-Loop Linac Telemetry, EPID & Perturbative 3D Delivery-Forensics Engine.

Research Software — NOT A MEDICAL DEVICE.
"""

__version__ = "0.1.0"

from linac_forensics.anatomy import (
    PatientAnatomyVolume,
    build_synthetic_phantom_anatomy,
    load_openkbp_patient,
)
from linac_forensics.causal_fault_model import diagnose_causal_faults
from linac_forensics.ir import (
    AbstentionCode,
    AbstentionError,
    CausalFaultDiagnosis,
    ControlPointPlan,
    MLCFaultSignature,
    PatientTriageRecord,
    PerturbativeDoseImpact,
    ProvenanceRecord,
    SobolSensitivityReport,
    TG142TrendSeries,
    TrajectoryLogTensor,
)
from linac_forensics.log_parser import (
    generate_synthetic_vmat_log,
    parse_telemetry_csv,
    parse_telemetry_json,
    parse_varian_trajectory_bin,
    write_varian_trajectory_bin,
)
from linac_forensics.perturbative_dose import compute_perturbative_dose_replay
from linac_forensics.sobol_sensitivity import run_delivery_sobol_sensitivity

__all__ = [
    "__version__",
    "AbstentionCode",
    "AbstentionError",
    "CausalFaultDiagnosis",
    "ControlPointPlan",
    "MLCFaultSignature",
    "PatientAnatomyVolume",
    "PatientTriageRecord",
    "PerturbativeDoseImpact",
    "ProvenanceRecord",
    "SobolSensitivityReport",
    "TG142TrendSeries",
    "TrajectoryLogTensor",
    "build_synthetic_phantom_anatomy",
    "compute_perturbative_dose_replay",
    "diagnose_causal_faults",
    "generate_synthetic_vmat_log",
    "load_openkbp_patient",
    "parse_telemetry_csv",
    "parse_telemetry_json",
    "parse_varian_trajectory_bin",
    "run_delivery_sobol_sensitivity",
    "write_varian_trajectory_bin",
]
