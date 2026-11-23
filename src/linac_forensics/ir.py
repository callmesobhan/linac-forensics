"""Strictly typed Pydantic v2 Physics Intermediate Representation (IR) for linac-forensics.

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AbstentionCode(StrEnum):
    """Hard safety and physics integrity abstention gate codes."""

    TIMESTAMP_DESYNC = "TIMESTAMP_DESYNC"
    MU_CONSERVATION_VIOLATION = "MU_CONSERVATION_VIOLATION"
    UNMAPPED_IEC61217_COORDINATES = "UNMAPPED_IEC61217_COORDINATES"
    CORRUPTED_BINARY_LOG = "CORRUPTED_BINARY_LOG"
    INTERLOCK_INCOMPLETE_DELIVERY = "INTERLOCK_INCOMPLETE_DELIVERY"


class AbstentionError(ValueError):
    """Raised when a hard physics, coordinate, or telemetry integrity gate fails."""

    def __init__(
        self,
        code: AbstentionCode,
        message: str,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.evidence = evidence or {}
        super().__init__(f"[{code.value}] {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstained": True,
            "code": self.code.value,
            "message": self.message,
            "evidence": self.evidence,
        }


class ProvenanceRecord(BaseModel):
    """Cryptographic provenance record separating real public data from synthetic benchmarks."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(..., min_length=1)
    provenance_tag: Literal[
        "OPEN_PUBLIC_ANATOMY_OPENKBP_PT1",
        "OPEN_PUBLIC_ANATOMY_OPENKBP_PT2",
        "OPEN_PUBLIC_ANATOMY_OPENKBP_PT3",
        "OPEN_PUBLIC_ANATOMY_OPENKBP_COHORT",
        "SYNTHETIC_FAULT_INJECTION_BENCHMARK",
        "SYNTHETIC_TG142_FLEET_DRIFT",
        "VENDOR_TRAJECTORY_LOG",
        "EPID_THREE_WAY_FORENSICS",
        "COMPOSITE_OPENKBP_PLUS_SYNTHETIC_LOG",
    ]
    dataset_source: str = Field(..., min_length=1)
    sha256_inputs: dict[str, str] = Field(default_factory=dict)
    engine_version: str = Field(default="0.2.0")
    research_notice: str = Field(default="Research Software — NOT A MEDICAL DEVICE")
    created_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    notes: str = Field(default="")


class ControlPointPlan(BaseModel):
    """Simplified DICOM-RT VMAT/IMRT control-point plan representation in IEC 61217."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(..., min_length=1)
    patient_id: str = Field(..., min_length=1)
    machine_id: str = Field(default="TrueBeam-01")
    beam_energy_mv: float = Field(default=6.0, gt=0.0)
    prescribed_dose_gy: float = Field(default=70.0, gt=0.0)
    num_fractions: int = Field(default=35, ge=1)
    total_mu: float = Field(..., gt=0.0)
    n_leaves_per_bank: int = Field(default=60, ge=1, le=120)
    leaf_widths_mm: list[float] = Field(..., min_length=1)
    dlg_nominal_mm: float = Field(default=1.4, ge=0.0, le=5.0)
    gantry_angles_deg: list[float] = Field(..., min_length=2)
    collimator_angle_deg: float = Field(default=15.0, ge=0.0, le=360.0)
    jaw_x1_mm: float = Field(default=-75.0)
    jaw_x2_mm: float = Field(default=75.0)
    jaw_y1_mm: float = Field(default=-85.0)
    jaw_y2_mm: float = Field(default=85.0)
    mu_weights: list[float] = Field(..., min_length=2)
    bank_a_planned_mm: list[list[float]] = Field(
        ..., description="Shape [n_cp, n_leaves], IEC 61217 right (+X) bank positions in mm."
    )
    bank_b_planned_mm: list[list[float]] = Field(
        ..., description="Shape [n_cp, n_leaves], IEC 61217 left (-X) bank positions in mm."
    )

    @field_validator("leaf_widths_mm")
    @classmethod
    def _validate_positive_leaf_widths(cls, v: list[float]) -> list[float]:
        if any(w <= 0.0 for w in v):
            raise ValueError("All leaf widths must be strictly positive (mm).")
        return v


class TrajectoryLogTensor(BaseModel):
    """High-frequency (20–50 Hz) linac trajectory log tensor in IEC 61217 coordinates."""

    model_config = ConfigDict(extra="forbid")

    log_id: str = Field(..., min_length=1)
    header_version: Literal["3.0", "4.0", "5.0", "DYNALOG-B", "JSON-1.0", "CSV-1.0"] = "5.0"
    machine_id: str = Field(default="TrueBeam-01")
    patient_id: str = Field(default="pt_1")
    plan_id: str = Field(default="VMAT_HN_70Gy")
    sampling_interval_ms: float = Field(default=20.0, gt=0.0)
    n_snapshots: int = Field(..., ge=2)
    n_leaves_per_bank: int = Field(default=60, ge=1, le=120)
    coordinate_system: str = Field(default="IEC61217")
    timestamps_ms: list[float]
    gantry_expected_deg: list[float]
    gantry_actual_deg: list[float]
    collimator_expected_deg: list[float]
    collimator_actual_deg: list[float]
    jaw_expected_mm: dict[str, list[float]]
    jaw_actual_mm: dict[str, list[float]]
    mu_expected: list[float]
    mu_actual: list[float]
    beam_hold: list[int] = Field(..., description="0=beam on, 1=hold, 2=transition/interlock")
    bank_a_expected_mm: list[list[float]] = Field(
        ..., description="Shape [n_snapshots, n_leaves], +X bank in mm."
    )
    bank_a_actual_mm: list[list[float]]
    bank_b_expected_mm: list[list[float]] = Field(
        ..., description="Shape [n_snapshots, n_leaves], -X bank in mm."
    )
    bank_b_actual_mm: list[list[float]]
    provenance: ProvenanceRecord


class MLCFaultSignature(BaseModel):
    """Per-leaf electro-mechanical error signature extracted from trajectory logs."""

    model_config = ConfigDict(extra="forbid")

    bank: Literal["A", "B"]
    leaf_index: int = Field(..., ge=1, description="1-based clinical leaf index (1..60).")
    rms_error_mm: float = Field(..., ge=0.0)
    max_error_mm: float = Field(..., ge=0.0)
    velocity_lag_ms: float = Field(..., description="Estimated viscous drag time constant tau (ms).")
    velocity_lag_ci95_ms: tuple[float, float]
    velocity_correlation_r: float = Field(..., ge=-1.0, le=1.0)
    stiction_amplitude_mm: float = Field(
        default=0.0, description="Estimated Coulomb/Stribeck direction-reversal stiction beta (mm)."
    )
    stiction_ci95_mm: tuple[float, float] = Field(default=(0.0, 0.0))
    systematic_offset_mm: float
    is_flagged: bool


class StructureDVHShift(BaseModel):
    """Exact structure-specific planned vs delivered 3D DVH and biological impact."""

    model_config = ConfigDict(extra="forbid")

    roi_name: str
    roi_type: Literal["PTV", "OAR"]
    volume_cc: float = Field(..., ge=0.0)
    d95_planned_gy: float
    d95_delivered_gy: float
    delta_d95_gy: float
    delta_d95_pct: float
    d003cc_planned_gy: float
    d003cc_delivered_gy: float
    delta_d003cc_gy: float
    dmean_planned_gy: float
    dmean_delivered_gy: float
    delta_dmean_gy: float
    eud_planned_gy: float
    eud_delivered_gy: float
    delta_eud_gy: float
    bio_prob_metric: Literal["TCP", "NTCP"]
    bio_prob_planned: float = Field(..., ge=0.0, le=1.0)
    bio_prob_delivered: float = Field(..., ge=0.0, le=1.0)
    delta_bio_prob: float
    clinically_unacceptable: bool


class PerturbativeDoseImpact(BaseModel):
    """Output of fast perturbative fluence-to-3D-dose replay."""

    model_config = ConfigDict(extra="forbid")

    log_id: str
    patient_id: str
    anatomy_source: str
    max_abs_delta_fluence: float = Field(..., ge=0.0)
    rms_delta_fluence: float = Field(..., ge=0.0)
    max_abs_delta_dose_gy: float = Field(..., ge=0.0)
    rms_delta_dose_gy: float = Field(..., ge=0.0)
    gamma_2d_3pct_2mm_pass_pct: float = Field(..., ge=0.0, le=100.0)
    gamma_2d_2pct_1mm_pass_pct: float = Field(..., ge=0.0, le=100.0)
    gamma_3d_3pct_2mm_pass_pct: float = Field(..., ge=0.0, le=100.0)
    gamma_3d_2pct_1mm_pass_pct: float = Field(..., ge=0.0, le=100.0)
    gamma_blindspot_detected: bool
    heterogeneity_corrected: bool = Field(default=True)
    structure_shifts: list[StructureDVHShift]
    provenance: ProvenanceRecord


class FaultHypothesisResult(BaseModel):
    """Matched-filter / Bayesian BIC evaluation for one competing physical fault hypothesis."""

    model_config = ConfigDict(extra="forbid")

    fault_type: Literal[
        "NULL_NOMINAL",
        "MLC_LEAF_MOTOR_FRICTION",
        "MLC_COULOMB_STICTION",
        "SYSTEMATIC_DLG_OFFSET",
        "GANTRY_GRAVITY_SAG_HARMONIC",
        "DOSE_RATE_ION_CHAMBER_DRIFT",
        "JOINT_MULTI_FAULT",
    ]
    description: str
    parameters: dict[str, float | int | str]
    parameter_ci95: dict[str, tuple[float, float]] = Field(default_factory=dict)
    rss: float = Field(..., ge=0.0)
    n_params: int = Field(..., ge=0)
    aic: float
    bic: float
    delta_bic_vs_null: float
    posterior_weight: float = Field(..., ge=0.0, le=1.0)
    matched_filter_snr: float = Field(..., ge=0.0)
    detected: bool


class CausalFaultDiagnosis(BaseModel):
    """Ranked causal electro-mechanical fault attribution with uncertainty bounds."""

    model_config = ConfigDict(extra="forbid")

    log_id: str
    machine_id: str
    primary_root_cause: str
    active_faults: list[str]
    hypotheses: list[FaultHypothesisResult]
    flagged_leaves: list[MLCFaultSignature]
    estimated_dlg_offset_mm: float
    estimated_dlg_ci95_mm: tuple[float, float]
    estimated_gantry_sag_amp_mm: float
    estimated_gantry_sag_phase_deg: float
    estimated_mu_drift_pct: float
    estimated_mu_drift_ci95_pct: tuple[float, float]
    residual_rms_mm: float = Field(..., ge=0.0)
    variance_explained_pct: float = Field(..., ge=0.0, le=100.0)
    service_recommendation: str
    provenance: ProvenanceRecord


class TG142TrendSeries(BaseModel):
    """Multi-day AAPM TG-142 / Varian MPC machine QA trend telemetry."""

    model_config = ConfigDict(extra="forbid")

    machine_id: str
    days: list[int]
    dates_iso: list[str]
    dlg_offset_mm: list[float]
    mlc_max_rms_mm: list[float]
    gantry_sag_amp_mm: list[float]
    output_drift_pct: list[float]
    epid_uniformity_pct: list[float]
    early_warning_day: int | None = None
    action_level_breach_day: int | None = None
    provenance: ProvenanceRecord


class PatientTriageRecord(BaseModel):
    """Prioritized patient-level clinical dose impact record for incident triage."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str
    plan_id: str
    machine_id: str
    treatment_site: str
    fractions_affected: int = Field(..., ge=1)
    gamma_2d_3pct_2mm_pass_pct: float = Field(..., ge=0.0, le=100.0)
    delta_d95_ptv_pct: float
    delta_d003cc_oar_gy: float
    critical_oar_name: str
    clinical_risk_score: float = Field(..., ge=0.0, le=100.0)
    triage_priority: Literal["CRITICAL", "HIGH", "MODERATE", "ROUTINE"]
    gamma_blindspot: bool
    recommended_action: str


class SobolSensitivityReport(BaseModel):
    """Exact Saltelli-Jansen global Sobol' sensitivity indices with bootstrap 95% CIs."""

    model_config = ConfigDict(extra="forbid")

    output_metric: str
    n_base_samples: int = Field(..., ge=8)
    total_evaluations: int = Field(..., ge=16)
    parameter_names: list[str]
    parameter_bounds: dict[str, tuple[float, float]]
    first_order_s1: dict[str, float]
    total_order_st: dict[str, float]
    s1_ci95: dict[str, tuple[float, float]]
    st_ci95: dict[str, tuple[float, float]]
    interaction_index: dict[str, float]
    provenance: ProvenanceRecord


class EPIDThreeWayForensicsReport(BaseModel):
    """Closed-loop three-way comparison: Planned TPS vs Trajectory Log vs Measured EPID."""

    model_config = ConfigDict(extra="forbid")

    log_id: str
    machine_id: str
    classification: Literal[
        "CONCORDANT_NOMINAL",
        "LOG_VERIFIED_MECHANICAL_FAULT",
        "ENCODER_BLIND_SILENT_FAULT",
        "EPID_CALIBRATION_OR_PANEL_DRIFT",
    ]
    encoder_blind_fault_detected: bool
    gamma_plan_vs_log_2pct_1mm: float = Field(..., ge=0.0, le=100.0)
    gamma_plan_vs_epid_2pct_1mm: float = Field(..., ge=0.0, le=100.0)
    gamma_log_vs_epid_2pct_1mm: float = Field(..., ge=0.0, le=100.0)
    rms_epid_minus_log_pct: float = Field(..., ge=0.0)
    max_epid_minus_log_pct: float = Field(..., ge=0.0)
    inplane_symmetry_tilt_pct: float
    crossplane_symmetry_tilt_pct: float
    inferred_silent_fault_summary: str
    provenance: ProvenanceRecord


class KalmanRULForecast(BaseModel):
    """Kalman state-space trend & Remaining-Useful-Life (RUL) forecast before TG-142 breach."""

    model_config = ConfigDict(extra="forbid")

    machine_id: str
    metric_name: str
    action_limit: float
    observation_cutoff_day: int
    horizon_days: int
    forecast_days: list[int]
    forecast_mean: list[float]
    forecast_ci95_low: list[float]
    forecast_ci95_high: list[float]
    estimated_drift_rate_per_day: float
    rul_mean_days: float | None = None
    rul_ci95_days: tuple[float, float] | None = None
    breach_probability_within_horizon: float = Field(..., ge=0.0, le=1.0)
    maintenance_recommendation: str
    provenance: ProvenanceRecord


class MultiFractionAccumulationReport(BaseModel):
    """Multi-fraction cumulative 3D dose accumulation separating systematic vs random error."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str
    plan_id: str
    anatomy_source: str
    fractions_delivered: int = Field(..., ge=1)
    total_prescribed_fractions: int = Field(..., ge=1)
    cumulative_ptv_delta_d95_pct: float
    systematic_component_pct: float
    random_component_rms_pct: float
    cumulative_critical_oar_delta_d003cc_gy: float
    critical_oar_name: str
    recommended_remaining_fx_mu_scale: float
    adaptive_replan_triggered: bool
    provenance: ProvenanceRecord
