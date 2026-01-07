from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    d = d.replace(0, np.nan)
    return (n / d).fillna(0.0)


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _month_str(dt: pd.Series) -> pd.Series:
    # "YYYY-MM"
    return dt.dt.to_period("M").astype(str)


def _rolling_features(df: pd.DataFrame, group_cols: list[str], month_col: str, value_col: str, windows=(3, 6)) -> pd.DataFrame:
    """
    Adds rolling sum/mean/std for value_col over windows (months) within group.
    Assumes one row per group/month.
    """
    out = df.copy()
    out = out.sort_values(group_cols + [month_col])

    # create an ordered month index per group by using period start time
    out["_month_dt"] = pd.to_datetime(out[month_col] + "-01")

    for w in windows:
        out[f"{value_col}_roll{w}_sum"] = (
            out.groupby(group_cols, dropna=False)[value_col]
               .transform(lambda s: s.rolling(w, min_periods=1).sum())
        )
        out[f"{value_col}_roll{w}_mean"] = (
            out.groupby(group_cols, dropna=False)[value_col]
               .transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
        out[f"{value_col}_roll{w}_std"] = (
            out.groupby(group_cols, dropna=False)[value_col]
               .transform(lambda s: s.rolling(w, min_periods=2).std())
               .fillna(0.0)
        )

    out = out.drop(columns=["_month_dt"], errors="ignore")
    return out


# -----------------------------
# Main Feature Builders
# -----------------------------
def build_provider_month_features(claims: pd.DataFrame) -> pd.DataFrame:
    """
    Provider-month feature table designed for SIU / Integrity anomaly detection.

    Input expects columns:
    - claim_id, member_id, provider_npi, provider_specialty, provider_state
    - service_date (datetime), icd10_code, cpt_code
    - allowed_amount, paid_amount, units
    - is_weekend (0/1)
    """
    df = claims.copy()

    # Normalize types
    df["service_date"] = pd.to_datetime(df["service_date"])
    df["month"] = _month_str(df["service_date"])
    df["allowed_amount"] = pd.to_numeric(df["allowed_amount"], errors="coerce").fillna(0.0)
    df["paid_amount"] = pd.to_numeric(df["paid_amount"], errors="coerce").fillna(0.0)
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(1.0)
    df["is_weekend"] = pd.to_numeric(df.get("is_weekend", 0), errors="coerce").fillna(0).astype(int)

    # Define some healthcare-utilization signal groups
    df["is_em_low"] = df["cpt_code"].isin(["99213", "99203"]).astype(int)
    df["is_em_high"] = df["cpt_code"].isin(["99214", "99204"]).astype(int)
    df["is_ed"] = df["cpt_code"].isin(["99283", "99285"]).astype(int)
    df["is_imaging"] = df["cpt_code"].isin(["73721", "70450", "71046", "72100"]).astype(int)
    df["is_lab"] = df["cpt_code"].isin(["80053", "83036", "81002", "36415"]).astype(int)

    group_cols = ["provider_npi", "provider_specialty", "provider_state", "month"]

    base = (
        df.groupby(group_cols, dropna=False)
          .agg(
              claims=("claim_id", "nunique"),
              members=("member_id", "nunique"),
              allowed_sum=("allowed_amount", "sum"),
              paid_sum=("paid_amount", "sum"),
              allowed_mean=("allowed_amount", "mean"),
              allowed_p95=("allowed_amount", lambda s: float(np.percentile(s, 95)) if len(s) else 0.0),
              units_sum=("units", "sum"),
              weekend_pct=("is_weekend", "mean"),
              em_low_cnt=("is_em_low", "sum"),
              em_high_cnt=("is_em_high", "sum"),
              ed_cnt=("is_ed", "sum"),
              imaging_cnt=("is_imaging", "sum"),
              lab_cnt=("is_lab", "sum"),
              distinct_icd10=("icd10_code", "nunique"),
              distinct_cpt=("cpt_code", "nunique"),
          )
          .reset_index()
    )

    # Ratios & utilization intensity
    base["claims_per_member"] = _safe_div(base["claims"], base["members"])
    base["allowed_per_claim"] = _safe_div(base["allowed_sum"], base["claims"])
    base["paid_to_allowed_ratio"] = _safe_div(base["paid_sum"], base["allowed_sum"])
    base["units_per_claim"] = _safe_div(base["units_sum"], base["claims"])

    # Coding mix ratios
    base["em_high_rate"] = _safe_div(base["em_high_cnt"], (base["em_low_cnt"] + base["em_high_cnt"]))
    base["ed_rate"] = _safe_div(base["ed_cnt"], base["claims"])
    base["imaging_rate"] = _safe_div(base["imaging_cnt"], base["claims"])
    base["lab_rate"] = _safe_div(base["lab_cnt"], base["claims"])

    # ICD-10 entropy (coding diversity / concentration)
    # Compute from claim-level ICD within provider-month
    icd_counts = (
        df.groupby(group_cols + ["icd10_code"], dropna=False)["claim_id"]
          .nunique()
          .reset_index(name="icd_claims")
    )

    ent = (
        icd_counts.groupby(group_cols, dropna=False)["icd_claims"]
        .apply(lambda s: _entropy_from_counts(s.to_numpy()))
        .reset_index(name="icd10_entropy")
    )

    # Rare ICD rate: bottom 10% ICD frequencies globally (proxy for "rare coding")
    icd_global = df["icd10_code"].value_counts(dropna=False)
    if len(icd_global) > 0:
        threshold = np.percentile(icd_global.values, 10)
        rare_set = set(icd_global[icd_global <= threshold].index.tolist())
    else:
        rare_set = set()

    df["is_rare_icd"] = df["icd10_code"].isin(list(rare_set)).astype(int)

    rare = (
        df.groupby(group_cols, dropna=False)
          .agg(rare_icd_rate=("is_rare_icd", "mean"))
          .reset_index()
    )

    # Merge
    feat = base.merge(ent, on=group_cols, how="left").merge(rare, on=group_cols, how="left")
    feat["icd10_entropy"] = feat["icd10_entropy"].fillna(0.0)
    feat["rare_icd_rate"] = feat["rare_icd_rate"].fillna(0.0)

    # Rolling windows on key metrics
    feat = _rolling_features(feat, ["provider_npi"], "month", "claims", windows=(3, 6))
    feat = _rolling_features(feat, ["provider_npi"], "month", "allowed_sum", windows=(3, 6))
    feat = _rolling_features(feat, ["provider_npi"], "month", "em_high_rate", windows=(3, 6))

    # Simple stability/shift signals
    feat["claims_spike_vs_roll3"] = _safe_div(feat["claims"], feat["claims_roll3_mean"])
    feat["allowed_spike_vs_roll3"] = _safe_div(feat["allowed_sum"], feat["allowed_sum_roll3_mean"])
    feat["em_high_shift_vs_roll3"] = _safe_div(feat["em_high_rate"], (feat["em_high_rate_roll3_mean"] + 1e-9))

    return feat


def build_member_month_features(claims: pd.DataFrame) -> pd.DataFrame:
    """
    Member-month feature table for abnormal utilization & provider hopping.
    """
    df = claims.copy()
    df["service_date"] = pd.to_datetime(df["service_date"])
    df["month"] = _month_str(df["service_date"])
    df["allowed_amount"] = pd.to_numeric(df["allowed_amount"], errors="coerce").fillna(0.0)
    df["paid_amount"] = pd.to_numeric(df["paid_amount"], errors="coerce").fillna(0.0)
    df["is_weekend"] = pd.to_numeric(df.get("is_weekend", 0), errors="coerce").fillna(0).astype(int)

    group_cols = ["member_id", "member_state", "month"]

    base = (
        df.groupby(group_cols, dropna=False)
          .agg(
              claims=("claim_id", "nunique"),
              providers=("provider_npi", "nunique"),
              specialties=("provider_specialty", "nunique"),
              allowed_sum=("allowed_amount", "sum"),
              paid_sum=("paid_amount", "sum"),
              allowed_mean=("allowed_amount", "mean"),
              weekend_pct=("is_weekend", "mean"),
              distinct_icd10=("icd10_code", "nunique"),
              distinct_cpt=("cpt_code", "nunique"),
          )
          .reset_index()
    )

    base["paid_to_allowed_ratio"] = _safe_div(base["paid_sum"], base["allowed_sum"])
    base["claims_per_provider"] = _safe_div(base["claims"], base["providers"])
    base["provider_hopping_index"] = _safe_div(base["providers"], base["claims"])  # higher means more hopping

    # Rolling windows on member utilization
    base = _rolling_features(base, ["member_id"], "month", "claims", windows=(3, 6))
    base = _rolling_features(base, ["member_id"], "month", "allowed_sum", windows=(3, 6))

    base["claims_spike_vs_roll3"] = _safe_div(base["claims"], base["claims_roll3_mean"])
    base["allowed_spike_vs_roll3"] = _safe_div(base["allowed_sum"], base["allowed_sum_roll3_mean"])

    return base
