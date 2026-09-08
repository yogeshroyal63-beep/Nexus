"""
Centralized configuration for Nexus.
AWS-native: Bedrock for LLM reasoning, DynamoDB for memory,
Strands Agents SDK for the autonomous agent loop.
"""
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- AWS / Bedrock -------------------------------------------------------
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    # Bedrock model for the Planner + reasoning layer
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    # --- Groq (Explainer / report generation) --------------------------------
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_MAX_TOKENS: int = 2000

    # --- GitHub integration --------------------------------------------------
    GITHUB_TOKEN: str = ""
    GITHUB_REPO: str = ""   # "owner/repo"
    WRITEBACK_ENABLED: bool = False

    # --- SNS webhook authentication --------------------------------------------
    # Shared-secret check for /api/sns/drift-check, NOT full AWS SNS message
    # signature verification. See routes.py's sns_drift_check docstring for
    # the honest trade-off this represents. Empty by default so the demo
    # works out of the box without extra setup — set this in any deployment
    # where the endpoint URL becomes publicly reachable, since without it
    # the endpoint is both unauthenticated AND (before this fix) unrate-
    # limited, letting anyone who finds the URL trigger the full agentic
    # pipeline (real LLM calls, real GitHub issue creation if write-back is
    # enabled) on demand.
    SNS_WEBHOOK_SECRET: str = ""

    # --- Remediation thresholds ----------------------------------------------
    AUTO_EXECUTE_MIN_CONFIDENCE: float = 0.75
    AUTO_EXECUTE_ENABLED: bool = True

    # --- Memory (DynamoDB or local JSON fallback) ----------------------------
    MEMORY_BACKEND: str = "local"        # "local" | "dynamodb"
    MEMORY_LOCAL_PATH: str = "data/nexus_incidents.json"
    DYNAMODB_TABLE: str = "nexus-incidents"
    CLAIM_STALE_AFTER_SECONDS: int = 300  # 5 min — see memory.py claim_incident_for_approval

    # --- Drift thresholds ----------------------------------------------------
    KS_PVALUE_ALERT_THRESHOLD: float = 0.05
    PSI_LOW_THRESHOLD: float = 0.1
    PSI_MODERATE_THRESHOLD: float = 0.2
    PSI_HIGH_THRESHOLD: float = 0.3
    EMBEDDING_COSINE_DRIFT_THRESHOLD: float = 0.15
    INTERVENTION_DELTA_MIN: float = 0.05
    MIN_SAMPLE_SIZE_FOR_DRIFT_TEST: int = 20  # below this, KS/PSI are statistically unreliable

    # --- App -----------------------------------------------------------------
    USE_MOCK_DATAHUB: bool = True
    ENV: str = "development"
    ALLOWED_ORIGINS: str = "*"


settings = Settings()
