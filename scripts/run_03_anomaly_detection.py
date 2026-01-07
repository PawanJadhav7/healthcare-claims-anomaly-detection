import os
import json
import pandas as pd

from src.anomaly_models import build_provider_risk, build_member_risk

PROV_PATH = "data/features/provider_month_features.csv"
MEM_PATH = "data/features/member_month_features.csv"
OUT_DIR = "outputs/anomaly"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    prov = pd.read_csv(PROV_PATH)
    mem = pd.read_csv(MEM_PATH)

    # Provider risk
    prov_scored = build_provider_risk(prov, use_peer_groups=True, contamination=0.03)
    prov_out = os.path.join(OUT_DIR, "provider_month_risk_scored.csv")
    prov_top = os.path.join(OUT_DIR, "provider_month_top_200.csv")

    prov_scored.to_csv(prov_out, index=False)
    prov_scored.head(200).to_csv(prov_top, index=False)

    # Member risk
    mem_scored = build_member_risk(mem, contamination=0.03)
    mem_out = os.path.join(OUT_DIR, "member_month_risk_scored.csv")
    mem_top = os.path.join(OUT_DIR, "member_month_top_200.csv")

    mem_scored.to_csv(mem_out, index=False)
    mem_scored.head(200).to_csv(mem_top, index=False)

    print("✅ Anomaly scoring complete")
    print(f"- {prov_out} ({prov_scored.shape[0]:,} rows)")
    print(f"- {prov_top}")
    print(f"- {mem_out} ({mem_scored.shape[0]:,} rows)")
    print(f"- {mem_top}")

    # Dashboard-ready summary (resilient if flags missing)
    prov_high = int((prov_scored["flag_overall_high"] == 1).sum()) if "flag_overall_high" in prov_scored.columns else 0
    mem_high = int((mem_scored["flag_overall_high"] == 1).sum()) if "flag_overall_high" in mem_scored.columns else 0

    summary = {
        "providers_scored_rows": int(prov_scored.shape[0]),
        "members_scored_rows": int(mem_scored.shape[0]),
        "top_provider_risk_percentile": float(prov_scored["risk_percentile"].iloc[0]) if len(prov_scored) else None,
        "top_member_risk_percentile": float(mem_scored["risk_percentile"].iloc[0]) if len(mem_scored) else None,
        "high_risk_provider_months": prov_high,
        "high_risk_member_months": mem_high,
        "provider_output": prov_out,
        "member_output": mem_out,
    }

    summary_path = os.path.join(OUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Summary:", summary_path)


if __name__ == "__main__":
    main()
