from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


# ----------------------------
# Utilities
# ----------------------------
def robust_zscore(x: pd.Series) -> pd.Series:
    """
    Robust z-score using median and MAD (more stable for heavy tails).
    """
    x = pd.to_numeric(x, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - med) / (1.4826 * mad)


def clip_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0)


def rank_pct(score: pd.Series, ascending: bool = False) -> pd.Series:
    """
    Percentile rank in [0,1]. Higher should mean riskier by default.
    """
    return score.rank(pct=True, ascending=True)


def isolation_forest_scores(
    X: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42
) -> tuple[np.ndarray, IsolationForest, RobustScaler]:
    """
    Fits IsolationForest and returns anomaly scores in [0,1] where higher = more anomalous.
    """
    X = clip_inf_nan(X)

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(Xs)

    # decision_function: higher = more normal; invert to make higher = more anomalous
    normality = model.decision_function(Xs)
    raw = -normality

    mn, mx = float(np.min(raw)), float(np.max(raw))
    if mx - mn < 1e-12:
        scores = np.zeros_like(raw)
    else:
        scores = (raw - mn) / (mx - mn)

    return scores, model, scaler


def peer_group_indicators(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str]
) -> pd.DataFrame:
    """
    Adds robust z-scores for each metric within peer group
    (e.g. provider_specialty + provider_state + month).
    """
    out = df.copy()
    for m in metric_cols:
        out[f"{m}_peer_rz"] = (
            out.groupby(group_cols, dropna=False)[m]
               .transform(robust_zscore)
               .fillna(0.0)
        )
    return out


# ----------------------------
# Provider Risk Scoring
# ----------------------------
def build_provider_risk(
    prov_feat: pd.DataFrame,
    use_peer_groups: bool = True,
    contamination: float = 0.03
) -> pd.DataFrame:
    """
    Provider-month risk scoring:
    - Peer robust z-scores (specialty+state+month)
    - Isolation Forest multivariate anomaly score
    - Composite risk score (weighted)
    """
    df = prov_feat.copy()

    id_cols = ["provider_npi", "provider_specialty", "provider_state", "month"]

    base_features = [
        "claims",
        "members",
        "claims_per_member",
        "allowed_sum",
        "allowed_per_claim",
        "allowed_mean",
        "allowed_p95",
        "paid_to_allowed_ratio",
        "units_per_claim",
        "weekend_pct",
        "em_high_rate",
        "ed_rate",
        "imaging_rate",
        "lab_rate",
        "distinct_icd10",
        "distinct_cpt",
        "icd10_entropy",
        "rare_icd_rate",
        "claims_spike_vs_roll3",
        "allowed_spike_vs_roll3",
        "em_high_shift_vs_roll3",
    ]
    base_features = [c for c in base_features if c in df.columns]

    df_num = df[base_features].apply(pd.to_numeric, errors="coerce")
    df_num = clip_inf_nan(df_num)

    # Peer group explainability layer
    if use_peer_groups:
        peer_cols = ["provider_specialty", "provider_state", "month"]
        peer_metrics = [
            c for c in ["claims", "claims_per_member", "allowed_per_claim", "weekend_pct", "em_high_rate", "rare_icd_rate"]
            if c in df.columns
        ]

        df = peer_group_indicators(df, peer_cols, peer_metrics)

        rz_cols = [f"{m}_peer_rz" for m in peer_metrics]
        if rz_cols:
            # Risk increases for positive tail; use max as simple explainable driver
            df["peer_risk_raw"] = df[rz_cols].clip(lower=0).max(axis=1)
        else:
            df["peer_risk_raw"] = 0.0

        df["peer_risk"] = rank_pct(df["peer_risk_raw"], ascending=False)
    else:
        df["peer_risk_raw"] = 0.0
        df["peer_risk"] = 0.0

    # Multivariate model layer
    iso_scores, _, _ = isolation_forest_scores(df_num, contamination=contamination)
    df["iforest_score"] = iso_scores
    df["iforest_risk"] = rank_pct(df["iforest_score"], ascending=False)

    # Composite score (tunable)
    df["risk_score"] = (0.55 * df["peer_risk"]) + (0.45 * df["iforest_risk"])
    df["risk_percentile"] = rank_pct(df["risk_score"], ascending=False)

    # Flags for dashboards
    df["flag_peer_high"] = (df["peer_risk"] >= 0.97).astype(int)
    df["flag_model_high"] = (df["iforest_risk"] >= 0.97).astype(int)
    df["flag_overall_high"] = (df["risk_percentile"] >= 0.98).astype(int)

    cols_out = (
        id_cols
        + [
            "risk_score",
            "risk_percentile",
            "peer_risk_raw",
            "peer_risk",
            "iforest_score",
            "iforest_risk",
            "flag_peer_high",
            "flag_model_high",
            "flag_overall_high",
        ]
        + base_features
        + [c for c in df.columns if c.endswith("_peer_rz")]
    )
    cols_out = [c for c in cols_out if c in df.columns]

    return (
        df[cols_out]
        .sort_values(["risk_score"], ascending=False)
        .reset_index(drop=True)
    )


# ----------------------------
# Member Risk Scoring
# ----------------------------
def build_member_risk(
    mem_feat: pd.DataFrame,
    contamination: float = 0.03
) -> pd.DataFrame:
    """
    Member-month risk scoring:
    - Isolation Forest on utilization & hopping behavior
    - Explainable flags for SIU triage
    """
    df = mem_feat.copy()

    id_cols = ["member_id", "member_state", "month"]

    features = [
        "claims",
        "providers",
        "specialties",
        "allowed_sum",
        "allowed_mean",
        "weekend_pct",
        "distinct_icd10",
        "distinct_cpt",
        "claims_per_provider",
        "provider_hopping_index",
        "claims_spike_vs_roll3",
        "allowed_spike_vs_roll3",
    ]
    features = [c for c in features if c in df.columns]

    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = clip_inf_nan(X)

    iso_scores, _, _ = isolation_forest_scores(X, contamination=contamination)
    df["iforest_score"] = iso_scores
    df["iforest_risk"] = rank_pct(df["iforest_score"], ascending=False)

    # Explainable flags
    df["hop_flag"] = (pd.to_numeric(df.get("providers", 0), errors="coerce").fillna(0) >= 6).astype(int)
    df["high_util_flag"] = (pd.to_numeric(df.get("claims", 0), errors="coerce").fillna(0) >= 10).astype(int)

    # Composite score for member-month
    util_mix = pd.to_numeric(df.get("providers", 0), errors="coerce").fillna(0) + pd.to_numeric(df.get("claims", 0), errors="coerce").fillna(0)
    util_rank = rank_pct(util_mix, ascending=False)

    df["risk_score"] = 0.75 * df["iforest_risk"] + 0.25 * util_rank
    df["risk_percentile"] = rank_pct(df["risk_score"], ascending=False)

    # Overall high-risk flag (top ~1.5%)
    df["flag_overall_high"] = (df["risk_percentile"] >= 0.985).astype(int)

    cols_out = id_cols + [
        "risk_score",
        "risk_percentile",
        "iforest_score",
        "iforest_risk",
        "hop_flag",
        "high_util_flag",
        "flag_overall_high",
    ] + features
    cols_out = [c for c in cols_out if c in df.columns]

    return (
        df[cols_out]
        .sort_values(["risk_score"], ascending=False)
        .reset_index(drop=True)
    )
