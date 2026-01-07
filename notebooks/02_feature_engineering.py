import os
import pandas as pd


from src.feature_store import build_provider_month_features, build_member_month_features

DATA_PATH = "data/raw/claims.csv"
OUT_DIR = "data/features"
os.makedirs(OUT_DIR, exist_ok=True)

# Load claims
claims = pd.read_csv(DATA_PATH, parse_dates=["service_date"])

# Build features
prov_feat = build_provider_month_features(claims)
mem_feat = build_member_month_features(claims)

# Write outputs
prov_path = os.path.join(OUT_DIR, "provider_month_features.csv")
mem_path = os.path.join(OUT_DIR, "member_month_features.csv")

prov_feat.to_csv(prov_path, index=False)
mem_feat.to_csv(mem_path, index=False)

print("✅ Features generated")
print(f"- {prov_path} ({prov_feat.shape[0]:,} rows, {prov_feat.shape[1]:,} cols)")
print(f"- {mem_path}  ({mem_feat.shape[0]:,} rows, {mem_feat.shape[1]:,} cols)")
