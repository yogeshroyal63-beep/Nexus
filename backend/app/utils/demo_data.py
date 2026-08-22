"""
Synthetic demo data generator (spec Section 4: "Data for the Demo").

Generates baseline ("training") and current ("last 7 days") samples for
every node in the mock fraud-detection lineage graph, with a scripted
drift injection into exactly one upstream table so the "Replay a failure"
demo mode can reliably reproduce the agent catching it and tracing it back
correctly (spec Section 4, "Replay a failure" demo mode).

Scenario: `raw_user_profiles.account_age_days` distribution silently shifts
(e.g. a new signup cohort or a broken join sends through mostly very-new
accounts). This changes `feature_user_risk_score`, which changes
`fraud_model_v3`'s prediction distribution. `raw_transactions` and
`raw_device_signals` are held stationary so the demo can show the causal
engine correctly ignoring the coincidentally-normal nodes and pinning the
one that actually moved.
"""
from __future__ import annotations

import numpy as np

from app.causal.isolator import FeatureSample

RNG = np.random.default_rng(7)

N_BASELINE = 2000
N_CURRENT = 1200

# Node URNs (must match app.lineage.datahub_client.MockDataHubClient)
RAW_TRANSACTIONS = "urn:li:dataset:(demo,raw_transactions,PROD)"
RAW_USER_PROFILES = "urn:li:dataset:(demo,raw_user_profiles,PROD)"
RAW_DEVICE_SIGNALS = "urn:li:dataset:(demo,raw_device_signals,PROD)"
FEATURE_TXN_VELOCITY = "urn:li:mlFeatureTable:(demo,feature_txn_velocity)"
FEATURE_USER_RISK_SCORE = "urn:li:mlFeatureTable:(demo,feature_user_risk_score)"
FEATURE_DEVICE_TRUST = "urn:li:mlFeatureTable:(demo,feature_device_trust)"
MODEL_URN = "urn:li:mlModel:(demo,fraud_model_v3,PROD)"
DEPLOYMENT_URN = "urn:li:mlModelDeployment:(demo,fraud_model_v3_prod)"


def _sample(mean: float, std: float, n: int, low: float = 0.0) -> np.ndarray:
    return np.clip(RNG.normal(mean, std, n), low, None)


def generate_demo_samples(inject_drift: bool = True) -> dict[str, FeatureSample]:
    """
    Returns baseline/current samples keyed by node URN, matching the
    FeatureSample shape the causal isolator expects.

    inject_drift=True reproduces the scripted failure scenario described
    above. inject_drift=False generates a fully healthy pipeline (useful
    for a "no incident" control run in the demo UI).
    """
    samples: dict[str, FeatureSample] = {}

    # --- raw_transactions: stationary in both scenarios ------------------------
    txn_amount_baseline = _sample(85, 40, N_BASELINE, low=1)
    txn_amount_current = _sample(85, 40, N_CURRENT, low=1)
    samples[RAW_TRANSACTIONS] = FeatureSample(
        node_urn=RAW_TRANSACTIONS, feature_name="amount",
        baseline=txn_amount_baseline, current=txn_amount_current,
    )

    # --- raw_device_signals: stationary in both scenarios -----------------------
    ip_risk_baseline = _sample(0.12, 0.08, N_BASELINE, low=0)
    ip_risk_current = _sample(0.12, 0.08, N_CURRENT, low=0)
    samples[RAW_DEVICE_SIGNALS] = FeatureSample(
        node_urn=RAW_DEVICE_SIGNALS, feature_name="ip_risk_score",
        baseline=ip_risk_baseline, current=ip_risk_current,
    )

    # --- raw_user_profiles: THE INJECTED DRIFT SOURCE ---------------------------
    account_age_baseline = _sample(420, 260, N_BASELINE, low=0)
    if inject_drift:
        # Silent cohort shift: mostly brand-new accounts show up now.
        account_age_current = _sample(35, 25, N_CURRENT, low=0)
    else:
        account_age_current = _sample(420, 260, N_CURRENT, low=0)
    samples[RAW_USER_PROFILES] = FeatureSample(
        node_urn=RAW_USER_PROFILES, feature_name="account_age_days",
        baseline=account_age_baseline, current=account_age_current,
    )

    # --- feature_txn_velocity: derived from raw_transactions (stationary) -------
    txn_velocity_baseline = txn_amount_baseline * RNG.normal(1.0, 0.05, N_BASELINE) / 20
    txn_velocity_current = txn_amount_current * RNG.normal(1.0, 0.05, N_CURRENT) / 20
    samples[FEATURE_TXN_VELOCITY] = FeatureSample(
        node_urn=FEATURE_TXN_VELOCITY, feature_name="txn_count_1h",
        baseline=txn_velocity_baseline, current=txn_velocity_current,
    )

    # --- feature_device_trust: derived from raw_device_signals (stationary) -----
    device_trust_baseline = 1 - ip_risk_baseline + RNG.normal(0, 0.03, N_BASELINE)
    device_trust_current = 1 - ip_risk_current + RNG.normal(0, 0.03, N_CURRENT)
    samples[FEATURE_DEVICE_TRUST] = FeatureSample(
        node_urn=FEATURE_DEVICE_TRUST, feature_name="device_trust_score",
        baseline=device_trust_baseline, current=device_trust_current,
    )

    # --- feature_user_risk_score: derived from raw_user_profiles (DRIFTS) -------
    # Lower account age -> higher composite risk score (newer accounts = riskier).
    user_risk_baseline = np.clip(1 - (account_age_baseline / account_age_baseline.max()), 0, 1)
    user_risk_current = np.clip(1 - (account_age_current / account_age_baseline.max()), 0, 1)
    samples[FEATURE_USER_RISK_SCORE] = FeatureSample(
        node_urn=FEATURE_USER_RISK_SCORE, feature_name="user_risk_score",
        baseline=user_risk_baseline, current=user_risk_current,
    )

    return samples


def generate_prediction_samples(feature_samples: dict[str, FeatureSample]) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize the model's own prediction (fraud probability) distribution
    as a function of the three feature nodes, so prediction_output_drift()
    has something realistic to test.
    """
    velocity_b = feature_samples[FEATURE_TXN_VELOCITY].baseline
    risk_b = feature_samples[FEATURE_USER_RISK_SCORE].baseline
    trust_b = feature_samples[FEATURE_DEVICE_TRUST].baseline
    n_b = min(len(velocity_b), len(risk_b), len(trust_b))
    logit_b = 0.4 * risk_b[:n_b] + 0.3 * (velocity_b[:n_b] / (velocity_b.max() + 1e-6)) - 0.5 * trust_b[:n_b]
    pred_baseline = 1 / (1 + np.exp(-4 * (logit_b - logit_b.mean())))

    velocity_c = feature_samples[FEATURE_TXN_VELOCITY].current
    risk_c = feature_samples[FEATURE_USER_RISK_SCORE].current
    trust_c = feature_samples[FEATURE_DEVICE_TRUST].current
    n_c = min(len(velocity_c), len(risk_c), len(trust_c))
    logit_c = 0.4 * risk_c[:n_c] + 0.3 * (velocity_c[:n_c] / (velocity_b.max() + 1e-6)) - 0.5 * trust_c[:n_c]
    pred_current = 1 / (1 + np.exp(-4 * (logit_c - logit_b.mean())))

    return pred_baseline, pred_current
