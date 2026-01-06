# ---------------------------------------------
# 01_claims_profiling.py
# Healthcare Claims — Profiling & Baselines
# ---------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/raw/claims.csv"
OUT_DIR = "outputs/profiling"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# 1) Load Data
# -------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["service_date"])

print("Shape:", df.shape)
print(df.head(3))

# -------------------------
# 2) Basic Data Quality
# -------------------------
dq = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "null_count": [int(df[c].isna().sum()) for c in df.columns],
    "null_pct": [round(float(df[c].isna().mean() * 100), 3) for c in df.columns],
    "nunique": [int(df[c].nunique(dropna=True)) for c in df.columns],
})
dq_path = os.path.join(OUT_DIR, "dq_summary.csv")
dq.to_csv(dq_path, index=False)
print("\n✅ Data Quality summary saved:", dq_path)

# Key checks
assert df["claim_id"].isna().sum() == 0, "claim_id has nulls"
assert df["member_id"].isna().sum() == 0, "member_id has nulls"
assert df["provider_npi"].isna().sum() == 0, "provider_npi has nulls"

# -------------------------
# 3) Basic Distribution Checks
# -------------------------
df["allowed_amount"] = pd.to_numeric(df["allowed_amount"], errors="coerce")
df["paid_amount"] = pd.to_numeric(df["paid_amount"], errors="coerce")
df["units"] = pd.to_numeric(df["units"], errors="coerce")

# Outlier sanity checks for costs
cost_summary = df[["allowed_amount", "paid_amount", "units"]].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).T
cost_path = os.path.join(OUT_DIR, "cost_summary.csv")
cost_summary.to_csv(cost_path)
print("✅ Cost summary saved:", cost_path)

# Plot allowed_amount distribution (log scale helpful)
plt.figure()
df["allowed_amount"].clip(lower=1).apply(np.log10).hist(bins=50)
plt.title("Allowed Amount Distribution (log10)")
plt.xlabel("log10(allowed_amount)")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "allowed_amount_log_hist.png"), dpi=160)
plt.close()

# -------------------------
# 4) Time Coverage
# -------------------------
min_date = df["service_date"].min()
max_date = df["service_date"].max()
print(f"\nService date range: {min_date.date()} → {max_date.date()}")

claims_by_month = (
    df.assign(month=df["service_date"].dt.to_period("M").astype(str))
      .groupby("month")["claim_id"].nunique()
      .reset_index(name="claims")
)

claims_by_month.to_csv(os.path.join(OUT_DIR, "claims_by_month.csv"), index=False)

plt.figure()
plt.plot(claims_by_month["month"], claims_by_month["claims"])
plt.xticks(rotation=60, ha="right")
plt.title("Claims Volume by Month")
plt.xlabel("month")
plt.ylabel("unique claims")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "claims_by_month.png"), dpi=160)
plt.close()

# -------------------------
# 5) ICD-10 + CPT Frequency
# -------------------------
top_icd = df["icd10_code"].value_counts().head(15).reset_index()
top_icd.columns = ["icd10_code", "claim_count"]
top_icd.to_csv(os.path.join(OUT_DIR, "top_icd10.csv"), index=False)

top_cpt = df["cpt_code"].value_counts().head(15).reset_index()
top_cpt.columns = ["cpt_code", "claim_count"]
top_cpt.to_csv(os.path.join(OUT_DIR, "top_cpt.csv"), index=False)

# -------------------------
# 6) Baselines: Provider Utilization
# -------------------------
# Provider-month utilization & cost
prov_month = (
    df.assign(month=df["service_date"].dt.to_period("M").astype(str))
      .groupby(["provider_npi", "provider_specialty", "provider_state", "month"])
      .agg(
          claims=("claim_id", "nunique"),
          members=("member_id", "nunique"),
          allowed=("allowed_amount", "sum"),
          paid=("paid_amount", "sum"),
          avg_allowed=("allowed_amount", "mean"),
          p95_allowed=("allowed_amount", lambda s: float(np.percentile(s, 95))),
          weekend_pct=("is_weekend", "mean"),
      )
      .reset_index()
)

prov_month.to_csv(os.path.join(OUT_DIR, "provider_month_baseline.csv"), index=False)
print("✅ Provider-month baseline saved")

# -------------------------
# 7) Peer Group Benchmarks (Specialty + State)
# -------------------------
peer_stats = (
    prov_month.groupby(["provider_specialty", "provider_state", "month"])
    .agg(
        peer_claims_mean=("claims", "mean"),
        peer_claims_p95=("claims", lambda s: float(np.percentile(s, 95))),
        peer_avg_allowed_mean=("avg_allowed", "mean"),
        peer_weekend_pct_mean=("weekend_pct", "mean"),
    )
    .reset_index()
)

peer_stats.to_csv(os.path.join(OUT_DIR, "peer_group_stats.csv"), index=False)
print("✅ Peer group stats saved")

# Join peer stats back to provider-month
prov_scored = prov_month.merge(peer_stats, on=["provider_specialty", "provider_state", "month"], how="left")

# Basic anomaly indicators:
# - claims_z: provider claims vs peer mean/std within peer group-month
grp = prov_scored.groupby(["provider_specialty", "provider_state", "month"])
prov_scored["peer_claims_std"] = grp["claims"].transform("std").replace(0, np.nan)
prov_scored["claims_z"] = (prov_scored["claims"] - prov_scored["peer_claims_mean"]) / prov_scored["peer_claims_std"]
prov_scored["claims_over_p95"] = (prov_scored["claims"] > prov_scored["peer_claims_p95"]).astype(int)

# Cost deviation indicator
prov_scored["avg_allowed_over_peer"] = (prov_scored["avg_allowed"] > prov_scored["peer_avg_allowed_mean"] * 1.35).astype(int)

# Weekend billing indicator
prov_scored["weekend_over_peer"] = (prov_scored["weekend_pct"] > prov_scored["peer_weekend_pct_mean"] * 1.75).astype(int)

prov_scored.to_csv(os.path.join(OUT_DIR, "provider_month_with_indicators.csv"), index=False)

# Top suspicious provider-month rows
top_flags = (
    prov_scored.sort_values(
        ["claims_over_p95", "avg_allowed_over_peer", "weekend_over_peer", "claims_z"],
        ascending=[False, False, False, False]
    )
    .head(50)
)

top_flags.to_csv(os.path.join(OUT_DIR, "top_provider_month_flags.csv"), index=False)
print("✅ Top provider-month flags saved:", os.path.join(OUT_DIR, "top_provider_month_flags.csv"))

# -------------------------
# 8) Member Baselines (Hopping / High Utilization)
# -------------------------
mem_month = (
    df.assign(month=df["service_date"].dt.to_period("M").astype(str))
      .groupby(["member_id", "member_state", "month"])
      .agg(
          claims=("claim_id", "nunique"),
          providers=("provider_npi", "nunique"),
          allowed=("allowed_amount", "sum"),
          avg_allowed=("allowed_amount", "mean"),
      )
      .reset_index()
)

mem_month["provider_hopping_flag"] = (mem_month["providers"] >= 6).astype(int)
mem_month["high_util_flag"] = (mem_month["claims"] >= 10).astype(int)

mem_month.to_csv(os.path.join(OUT_DIR, "member_month_baseline.csv"), index=False)

top_members = (
    mem_month.sort_values(["provider_hopping_flag", "providers", "claims"], ascending=[False, False, False])
    .head(50)
)
top_members.to_csv(os.path.join(OUT_DIR, "top_member_month_flags.csv"), index=False)
print("✅ Top member-month flags saved:", os.path.join(OUT_DIR, "top_member_month_flags.csv"))

# -------------------------
# 9) Quick Summary for README / Case Study
# -------------------------
summary = {
    "rows": int(df.shape[0]),
    "unique_members": int(df["member_id"].nunique()),
    "unique_providers": int(df["provider_npi"].nunique()),
    "date_min": str(min_date.date()),
    "date_max": str(max_date.date()),
    "top_specialties": df["provider_specialty"].value_counts().head(5).to_dict(),
    "top_states_member": df["member_state"].value_counts().head(5).to_dict(),
    "top_states_provider": df["provider_state"].value_counts().head(5).to_dict(),
}

summary_path = os.path.join(OUT_DIR, "profiling_summary.json")
import json
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("✅ Profiling summary saved:", summary_path)
print("\nDone.")
