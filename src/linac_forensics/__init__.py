"""linac-forensics: Closed-Loop Linac Telemetry, EPID & Perturbative 3D Delivery-Forensics Engine.

Research Software — NOT A MEDICAL DEVICE.
"""

__version__ = "0.2.0"

from linac_forensics.anatomy import (
    PatientAnatomyVolume,
    build_synthetic_phantom_anatomy,
    ct_hu_to_relative_electron_density,
    load_openkbp_cohort,
    load_openkbp_patient,
)
from linac_forensics.causal_fault_model import diagnose_causal_faults
from linac_forensics.epid_forensics import (
    analyze_three_way_epid_forensics,
    deconvolve_epid_to_fluence,
    simulate_measured_epid_image,
)
from linac_forensics.ir import (
    AbstentionCode,
    AbstentionError,
    CausalFaultDiagnosis,
    ControlPointPlan,
    EPIDThreeWayForensicsReport,
    KalmanRULForecast,
    MLCFaultSignature,
    MultiFractionAccumulationReport,
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
    parse_varian_dynalog_dlg,
    parse_varian_trajectory_bin,
    write_varian_dynalog_dlg,
    write_varian_trajectory_bin,
)
from linac_forensics.perturbative_dose import compute_perturbative_dose_replay
from linac_forensics.predictive_rul import (
    accumulate_multifraction_course,
    forecast_tg142_kalman_rul,
)
from linac_forensics.sobol_sensitivity import run_delivery_sobol_sensitivity

__all__ = [
    "__version__",
    "AbstentionCode",
    "AbstentionError",
    "CausalFaultDiagnosis",
    "ControlPointPlan",
    "EPIDThreeWayForensicsReport",
    "KalmanRULForecast",
    "MLCFaultSignature",
    "MultiFractionAccumulationReport",
    "PatientAnatomyVolume",
    "PatientTriageRecord",
    "PerturbativeDoseImpact",
    "ProvenanceRecord",
    "SobolSensitivityReport",
    "TG142TrendSeries",
    "TrajectoryLogTensor",
    "accumulate_multifraction_course",
    "analyze_three_way_epid_forensics",
    "build_synthetic_phantom_anatomy",
    "compute_perturbative_dose_replay",
    "ct_hu_to_relative_electron_density",
    "deconvolve_epid_to_fluence",
    "diagnose_causal_faults",
    "forecast_tg142_kalman_rul",
    "generate_synthetic_vmat_log",
    "load_openkbp_cohort",
    "load_openkbp_patient",
    "parse_telemetry_csv",
    "parse_telemetry_json",
    "parse_varian_dynalog_dlg",
    "parse_varian_trajectory_bin",
    "run_delivery_sobol_sensitivity",
    "simulate_measured_epid_image",
    "write_varian_dynalog_dlg",
    "write_varian_trajectory_bin",
]
