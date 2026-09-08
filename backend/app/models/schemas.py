"""
Core data models shared across the Causal Drift Sentinel backend.

These models are the contract between layers:
  Lineage Ingestion -> Drift Detection -> Causal Isolation -> LLM Reasoning -> Write-Back

Keeping them centralized means every layer speaks the same structured
language, which is what lets the LLM reasoning layer stay grounded in
real evidence instead of free-form guessing.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _NoProtectedNamespace(BaseModel):
    """Base class silencing pydantic's 'model_' protected-namespace warning,
    since our domain naturally uses `model_urn` (an ML model's URN)."""

    model_config = ConfigDict(protected_namespaces=())


# ---------------------------------------------------------------------------
# Lineage graph
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    DATASET = "dataset"
    FEATURE = "feature"
    MODEL = "model"
    DEPLOYMENT = "deployment"


class LineageNode(_NoProtectedNamespace):
    urn: str = Field(..., description="DataHub URN, e.g. urn:li:dataset:(...)")
    name: str
    node_type: NodeType
    platform: Optional[str] = None
    description: Optional[str] = None
    schema_fields: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class LineageEdge(_NoProtectedNamespace):
    upstream_urn: str
    downstream_urn: str
    relationship: str = Field(default="derives_from")


class LineageGraph(_NoProtectedNamespace):
    nodes: list[LineageNode]
    edges: list[LineageEdge]
    root_model_urn: str = Field(..., description="The model/deployment under investigation")


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

class DriftMethod(str, Enum):
    KS_TEST = "ks_test"
    PSI = "psi"
    EMBEDDING_COSINE = "embedding_cosine_drift"


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    # Not a drift measurement at all — baseline or current sample size was
    # below the statistical reliability floor (see settings.MIN_SAMPLE_
    # SIZE_FOR_DRIFT_TEST). Distinct from NONE (measured, genuinely no
    # drift) so a data outage or a not-yet-populated window is never
    # silently conflated with "the model is healthy", and distinct from
    # CRITICAL so degenerate input (e.g. an empty array, which previously
    # produced NaN statistics that fell through every comparison to the
    # CRITICAL catch-all) is never misread as the most severe possible
    # drift finding.
    INSUFFICIENT_DATA = "insufficient_data"


class FeatureDriftResult(_NoProtectedNamespace):
    node_urn: str
    feature_name: str
    method: DriftMethod
    statistic: float
    p_value: Optional[float] = None
    psi_score: Optional[float] = None
    severity: DriftSeverity
    baseline_window: str = Field(description="e.g. 'training' or '2026-06-01/2026-06-07'")
    current_window: str = Field(description="e.g. '2026-07-18/2026-07-25'")
    sample_size_baseline: int
    sample_size_current: int


class PredictionDriftResult(_NoProtectedNamespace):
    model_urn: str
    method: DriftMethod
    statistic: float
    p_value: Optional[float] = None
    severity: DriftSeverity
    detected_at: datetime


# ---------------------------------------------------------------------------
# Causal root-cause isolation
# ---------------------------------------------------------------------------

class CausalCandidate(_NoProtectedNamespace):
    node_urn: str
    node_name: str
    hops_from_model: int
    drift_result: FeatureDriftResult
    is_genuine_cause: bool
    fdr_adjusted_p_value: float = Field(
        default=1.0,
        description=(
            "Benjamini-Hochberg FDR-corrected p-value for this node's drift test, "
            "computed jointly across ALL ancestor nodes tested in the same run "
            "(not this node's raw KS-test p-value in isolation). A node only "
            "proceeds to the intervention check if this clears the FDR threshold — "
            "protects against false 'root causes' from pure multiple-testing noise."
        ),
    )
    survived_fdr_correction: bool = Field(
        default=False,
        description="Whether this node's drift remained significant after "
        "correcting for testing multiple ancestor nodes simultaneously.",
    )
    intervention_delta: float = Field(
        description=(
            "Mean change in downstream drift signal when this node's contribution "
            "is held constant vs. observed, across bootstrap resamples. Larger "
            "magnitude = stronger causal evidence that this node is driving the "
            "downstream drift, rather than merely having changed around the same time."
        )
    )
    intervention_delta_lower_ci: float = Field(
        default=0.0,
        description="5th percentile of intervention_delta across bootstrap resamples. "
        "is_genuine_cause requires THIS (not the mean) to clear the threshold — "
        "a robust effect, not a lucky draw.",
    )
    intervention_delta_upper_ci: float = Field(
        default=0.0,
        description="95th percentile of intervention_delta across bootstrap resamples.",
    )
    confounded_with: list[str] = Field(
        default_factory=list,
        description="Other upstream URNs this node's drift is correlated with, "
        "making causal attribution ambiguous unless disentangled.",
    )


class RootCauseTrace(_NoProtectedNamespace):
    model_urn: str
    prediction_drift: PredictionDriftResult
    candidates_examined: list[CausalCandidate]
    isolated_root_causes: list[CausalCandidate] = Field(
        description="Subset of candidates_examined judged to be genuine causes, ranked by intervention_delta"
    )
    graph_path: list[str] = Field(description="URNs from root cause to model, in order")
    fdr_uncorrected_false_positive_risk: float = Field(
        default=0.0,
        description=(
            "P(at least one false-positive candidate) if ancestor drift tests had "
            "been evaluated at raw alpha=0.05 without FDR correction, given how "
            "many ancestor nodes were tested this run. Surfaced so the report can "
            "state exactly how much noise the correction step removed."
        ),
    )


# ---------------------------------------------------------------------------
# LLM reasoning output
# ---------------------------------------------------------------------------

class SuggestedFix(_NoProtectedNamespace):
    action: str
    target_urn: str
    rationale: str


class RootCauseReport(_NoProtectedNamespace):
    model_urn: str
    generated_at: datetime
    summary: str
    detailed_explanation: str
    root_causes: list[str] = Field(description="Human-readable names of isolated root cause nodes")
    confidence: str = Field(description="'low' | 'moderate' | 'high', based on intervention_delta magnitude & confounding")
    suggested_fixes: list[SuggestedFix]
    raw_trace: RootCauseTrace


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------

class WriteBackResult(_NoProtectedNamespace):
    datahub_incident_urn: Optional[str] = None
    github_issue_url: Optional[str] = None
    github_pr_url: Optional[str] = None
    status: str


# ---------------------------------------------------------------------------
# Remediation planning & execution (Planner / Executor agents)
# ---------------------------------------------------------------------------

class RemediationActionType(str, Enum):
    TRIGGER_RETRAIN = "trigger_retrain"
    ROLLBACK_MODEL_VERSION = "rollback_model_version"
    QUARANTINE_DATA_SOURCE = "quarantine_data_source"
    OPEN_INCIDENT_TICKET = "open_incident_ticket"
    NO_ACTION = "no_action"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RemediationPlan(_NoProtectedNamespace):
    """Output of the Planner agent: a concrete, executable decision (not just
    a suggestion) paired with a confidence score and a risk assessment that
    determines whether the Executor may act autonomously or must escalate
    to a human first."""

    model_urn: str
    action_type: RemediationActionType
    target_urn: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0, description="Planner's confidence in this action, 0-1")
    risk_level: RiskLevel
    requires_human_approval: bool = Field(
        description="True when confidence is low or risk_level is high — the "
        "Executor must pause and wait for approval rather than act."
    )


class ActionOutcome(_NoProtectedNamespace):
    """Result of the Executor actually carrying out a RemediationPlan,
    plus the Verifier's check of whether it worked."""

    plan: RemediationPlan
    executed: bool
    execution_detail: str
    reversible: bool
    rollback_reference: Optional[str] = Field(
        default=None, description="Opaque reference the Executor can use to undo this action"
    )
    verified: Optional[bool] = Field(
        default=None, description="None = not yet checked; True/False = post-action verification result"
    )
    verification_detail: Optional[str] = None
    follow_up: Optional["ActionOutcome"] = Field(
        default=None,
        description=(
            "If verification found the original action did NOT resolve the drift "
            "(verified=False), the automatic corrective action the Coordinator took "
            "in response: a rollback (if the original action was reversible) or an "
            "auto-opened incident ticket (if not). None if verification passed, "
            "wasn't run, or this outcome IS itself a follow-up (bounded to one level "
            "— no recursive follow-up chains)."
        ),
    )


class IncidentRecord(_NoProtectedNamespace):
    """A single closed-loop run, persisted to memory so future Planner
    decisions can be informed by what was tried before and whether it worked."""

    incident_id: str
    model_urn: str
    created_at: datetime
    report: RootCauseReport
    plan: Optional[RemediationPlan] = None
    outcome: Optional[ActionOutcome] = None
    writeback: Optional[WriteBackResult] = None


class SentinelRunResult(_NoProtectedNamespace):
    """Full result of one Coordinator-orchestrated run: detect -> diagnose ->
    plan -> (execute -> verify | escalate) -> report -> remember."""

    incident_id: str
    model_urn: str
    trace: RootCauseTrace
    report: Optional[RootCauseReport] = None
    plan: Optional[RemediationPlan] = None
    outcome: Optional[ActionOutcome] = None
    writeback: Optional[WriteBackResult] = None
    escalated: bool = False
    similar_past_incidents: list[str] = Field(
        default_factory=list, description="incident_ids of similar past runs pulled from memory"
    )
