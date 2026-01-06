import os
import pandas as pd


RAW_CLAIMS = "data/raw/claims.csv"
PROV_RISK = "outputs/anomaly/provider_month_risk_scored.csv"
MEM_RISK = "outputs/anomaly/member_month_risk_scored.csv"

OUT_DIR = "data/sample"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # Load scored provider-months and keep top high-risk provider-months
    prov = pd.read_csv(PROV_RISK)
    top_prov = prov.head(25)[["provider_npi", "month"]].drop_duplicates()

    # Load scored member-months and keep top high-risk member-months
    mem = pd.read_csv(MEM_RISK)
    top_mem = mem.head(50)[["member_id", "month"]].drop_duplicates()

    # Load raw claims and create month column
    claims = pd.read_csv(RAW_CLAIMS, parse_dates=["service_date"])
    claims["month"] = claims["service_date"].dt.to_period("M").astype(str)

    # Filter claims for those top provider-months OR member-months
    claims_top_prov = claims.merge(top_prov, on=["provider_npi", "month"], how="inner")
    claims_top_mem = claims.merge(top_mem, on=["member_id", "month"], how="inner")

    sample_claims = pd.concat([claims_top_prov, claims_top_mem], ignore_index=True).drop_duplicates(subset=["claim_id"])

    # Cap size for GitHub friendliness
    sample_claims = sample_claims.sample(n=min(20000, len(sample_claims)), random_state=42)

    # Save outputs
    sample_claims_path = os.path.join(OUT_DIR, "claims_sample.csv")
    top_prov_path = os.path.join(OUT_DIR, "provider_month_top_200_sample.csv")
    top_mem_path = os.path.join(OUT_DIR, "member_month_top_200_sample.csv")

    sample_claims.to_csv(sample_claims_path, index=False)
    prov.head(200).to_csv(top_prov_path, index=False)
    mem.head(200).to_csv(top_mem_path, index=False)

    print("✅ Sample datasets created:")
    print(f"- {sample_claims_path} ({len(sample_claims):,} rows)")
    print(f"- {top_prov_path}")
    print(f"- {top_mem_path}")


if __name__ == "__main__":
    main()
