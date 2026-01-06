# Dashboard Specification — Healthcare Claims Anomaly Detection (FWA)

This spec defines the BI layer for SIU / Program Integrity triage.

Primary datasets:
- `outputs/anomaly/provider_month_risk_scored.csv`
- `outputs/anomaly/member_month_risk_scored.csv`
Optional drilldown dataset:
- `data/raw/claims.csv` (join on provider_npi+month and member_id+month)

---

## Page 1 — SIU Triage: Providers (Primary)

### Purpose
Rank and triage **provider-month** anomalies with explainability.

### Filters / Slicers
- Month (`month`) — default latest
- Specialty (`provider_specialty`)
- Provider state (`provider_state`)
- High risk only toggle (`flag_overall_high`)
- Risk threshold parameter (optional): `risk_score >= X`

### KPIs (Cards)
- Provider-months scored = COUNTROWS
- High risk provider-months = SUM(`flag_overall_high`)
- % flagged = flagged / total
- Avg risk_score (optional)

### Visual 1: Ranked Provider Table (core triage)
Columns:
- `risk_score`, `risk_percentile`
- `provider_npi`, `provider_specialty`, `provider_state`, `month`
- `peer_risk`, `iforest_risk`
- Flags: `flag_peer_high`, `flag_model_high`, `flag_overall_high`
- Key drivers:
  - `claims`, `members`, `claims_per_member`
  - `allowed_sum`, `allowed_per_claim`, `allowed_p95`
  - `weekend_pct`
  - `em_high_rate`, `ed_rate`, `imaging_rate`, `lab_rate`
  - `rare_icd_rate`, `icd10_entropy`
  - `claims_spike_vs_roll3`, `allowed_spike_vs_roll3`, `em_high_shift_vs_roll3`

Default sort: `risk_score` descending  
Row selection should filter other visuals.

### Visual 2: Explainability — Peer Drivers
Bar chart of peer robust z-scores for selected provider row:
- `claims_peer_rz`
- `claims_per_member_peer_rz`
- `allowed_per_claim_peer_rz`
- `weekend_pct_peer_rz`
- `em_high_rate_peer_rz`
- `rare_icd_rate_peer_rz`

Rule: show only positive values (clip at 0) to keep the narrative clear.

### Visual 3: Peer Scatter (Context)
Scatter plot (filtered by same month + specialty + state):
- X: `claims_per_member`
- Y: `allowed_per_claim`
- Size: `claims`
- Color: `risk_score`
- Tooltip: provider_npi, peer_risk, iforest_risk, weekend_pct, em_high_rate, rare_icd_rate

### Visual 4: Trend for Selected Provider (Month over Month)
Line chart:
- X: month
- Y1: `claims`
- Y2: `allowed_sum` (secondary axis)
Overlay baselines:
- `claims_roll3_mean`
- `allowed_sum_roll3_mean`
Optional: show spike ratios `claims_spike_vs_roll3`, `allowed_spike_vs_roll3`

---

## Page 2 — Member Utilization & Hopping (Secondary)

### Purpose
Identify members with abnormal utilization or provider hopping patterns.

### Filters / Slicers
- Month (`month`)
- Member state (`member_state`)
- High risk only (`flag_overall_high`)
- Hopping only (`hop_flag`) / High util only (`high_util_flag`)

### KPIs
- Member-months scored
- High risk member-months
- % hop_flag
- % high_util_flag

### Visual 1: Ranked Member Table
Columns:
- `risk_score`, `risk_percentile`
- `member_id`, `member_state`, `month`
- `providers`, `claims`, `provider_hopping_index`
- `allowed_sum`, `allowed_mean`
- Flags: `hop_flag`, `high_util_flag`, `flag_overall_high`

### Visual 2: Scatter — Hopping vs Utilization
- X: `providers`
- Y: `claims`
- Color: `risk_score`
- Tooltip: allowed_sum, provider_hopping_index, distinct_icd10, distinct_cpt

### Visual 3: Member Trend
Line chart by month for selected member:
- claims, allowed_sum
- rolling means: claims_roll3_mean, allowed_sum_roll3_mean

---

## Page 3 — Claims Drilldown (Optional, “Wow” page)

### Purpose
Investigators drill into claim lines for a selected provider-month or member-month.

### Data Model
Create derived month in claims: `month = FORMAT(service_date, "YYYY-MM")`

Relationships:
- claims.provider_npi + month → provider_month_risk_scored.provider_npi + month
- claims.member_id + month → member_month_risk_scored.member_id + month

### Visuals
1) Claims table filtered by selection:
- service_date, member_id, provider_npi, icd10_code, cpt_code, allowed_amount, place_of_service, is_weekend

2) ICD-10 mix:
- bar chart top ICD codes (count of claims)
- show rare_icd_rate + icd10_entropy in tooltip

3) CPT mix:
- bar chart CPT counts
- highlight E/M mix 99213 vs 99214 (upcoding narrative)
- imaging vs ED patterns

---

## Recommended Screenshots for Portfolio
- Provider triage table (Top 20)
- Explainability peer z-score bars
- Peer scatter plot
- Provider trend chart showing spike
- Member hopping scatter

Add screenshots to `docs/screens/` and link them in README.
