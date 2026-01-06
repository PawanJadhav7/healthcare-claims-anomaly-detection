# Data Dictionary — Healthcare Claims Anomaly Detection (FWA)

This project produces two main **risk-scored datasets** used for SIU/Integrity triage:

- `outputs/anomaly/provider_month_risk_scored.csv`
- `outputs/anomaly/member_month_risk_scored.csv`

It also uses:
- `data/raw/claims.csv` (synthetic/de-identified claims)
- `data/features/*.csv` (feature store outputs)

---

## 1) Raw Claims (`data/raw/claims.csv`)

| Column | Type | Description |
|---|---|---|
| claim_id | string | Unique claim identifier (synthetic) |
| member_id | string | Member identifier (synthetic) |
| provider_npi | string | NPI-like identifier (not a real NPI) |
| provider_specialty | string | Provider specialty code (e.g., FAMILY, IM, ER, ORTHO, RAD) |
| provider_state | string | Provider state (e.g., MA, NY, NJ) |
| member_state | string | Member state |
| service_date | date | Date of service |
| icd10_code | string | ICD-10-CM diagnosis code |
| icd10_desc | string | Diagnosis description |
| cpt_code | string | CPT/HCPCS procedure code |
| cpt_desc | string | CPT description |
| place_of_service | string | POS code (11 office, 22 outpatient, 23 ER) |
| allowed_amount | float | Allowed amount (proxy for cost) |
| paid_amount | float | Paid amount (proxy for reimbursement) |
| units | int | Units billed |
| is_weekend | int | 1 if weekend, else 0 |
| synthetic_label | int | 0 normal, 1 provider-anomaly, 2 member-anomaly (evaluation only) |

---

## 2) Provider Feature Store (`data/features/provider_month_features.csv`)

**Grain:** one row per **provider_npi + month** (also includes specialty/state columns).

### Identifiers
| Column | Type | Description |
|---|---|---|
| provider_npi | string | Provider identifier |
| provider_specialty | string | Specialty |
| provider_state | string | State |
| month | string | `YYYY-MM` |

### Volume & Utilization
| Column | Type | Description |
|---|---|---|
| claims | int | Unique claims in the month |
| members | int | Unique members seen in the month |
| claims_per_member | float | `claims / members` (utilization intensity) |
| units_sum | float | Total units billed |
| units_per_claim | float | `units_sum / claims` |

### Cost
| Column | Type | Description |
|---|---|---|
| allowed_sum | float | Total allowed amount |
| paid_sum | float | Total paid amount |
| allowed_mean | float | Mean allowed per claim |
| allowed_p95 | float | 95th percentile allowed per claim |
| allowed_per_claim | float | `allowed_sum / claims` |
| paid_to_allowed_ratio | float | `paid_sum / allowed_sum` |

### Coding Mix / Behavioral Signals
| Column | Type | Description |
|---|---|---|
| weekend_pct | float | % of claims on weekends |
| em_low_cnt | int | Count of low E/M codes (e.g., 99213, 99203) |
| em_high_cnt | int | Count of higher E/M codes (e.g., 99214, 99204) |
| em_high_rate | float | `em_high / (em_low + em_high)` (upcoding-like indicator) |
| ed_cnt | int | ED visits count |
| ed_rate | float | ED rate vs claims |
| imaging_cnt | int | Imaging claim count |
| imaging_rate | float | Imaging rate vs claims |
| lab_cnt | int | Lab claim count |
| lab_rate | float | Lab rate vs claims |
| distinct_icd10 | int | Unique diagnosis codes |
| distinct_cpt | int | Unique procedure codes |
| icd10_entropy | float | ICD-10 coding diversity (lower can imply concentration) |
| rare_icd_rate | float | % of claims using globally rare ICD codes |

### Rolling / Trend Features
| Column | Type | Description |
|---|---|---|
| claims_roll3_mean | float | 3-month rolling mean of claims |
| claims_roll6_mean | float | 6-month rolling mean of claims |
| allowed_sum_roll3_mean | float | 3-month rolling mean of allowed_sum |
| em_high_rate_roll3_mean | float | 3-month rolling mean of em_high_rate |
| claims_spike_vs_roll3 | float | `claims / claims_roll3_mean` |
| allowed_spike_vs_roll3 | float | `allowed_sum / allowed_sum_roll3_mean` |
| em_high_shift_vs_roll3 | float | `em_high_rate / em_high_rate_roll3_mean` |

---

## 3) Member Feature Store (`data/features/member_month_features.csv`)

**Grain:** one row per **member_id + month**.

| Column | Type | Description |
|---|---|---|
| member_id | string | Member identifier |
| member_state | string | Member state |
| month | string | `YYYY-MM` |
| claims | int | Claims per month |
| providers | int | Unique providers visited |
| specialties | int | Unique specialties visited |
| allowed_sum | float | Total allowed |
| allowed_mean | float | Mean allowed per claim |
| weekend_pct | float | Weekend utilization |
| distinct_icd10 | int | Unique ICD-10 codes |
| distinct_cpt | int | Unique CPT codes |
| claims_per_provider | float | `claims / providers` |
| provider_hopping_index | float | `providers / claims` (higher → more hopping) |
| claims_roll3_mean | float | Rolling utilization baseline |
| allowed_sum_roll3_mean | float | Rolling cost baseline |
| claims_spike_vs_roll3 | float | Utilization spike ratio |
| allowed_spike_vs_roll3 | float | Cost spike ratio |

---

## 4) Provider Risk Output (`outputs/anomaly/provider_month_risk_scored.csv`)

**Grain:** provider-month

### Risk / Explainability
| Column | Type | Description |
|---|---|---|
| peer_risk_raw | float | Max positive peer robust z-score across selected metrics |
| peer_risk | float | Percentile rank of peer_risk_raw |
| iforest_score | float | Isolation Forest anomaly score (0–1) |
| iforest_risk | float | Percentile rank of iforest_score |
| risk_score | float | Composite risk = `0.55*peer_risk + 0.45*iforest_risk` |
| risk_percentile | float | Percentile rank of risk_score |
| flag_peer_high | int | 1 if peer_risk ≥ threshold |
| flag_model_high | int | 1 if iforest_risk ≥ threshold |
| flag_overall_high | int | 1 if overall risk ≥ threshold |

### Peer Robust Z-Score Fields
Columns ending in `_peer_rz` represent robust z-scores within the peer group:
- peer group = `provider_specialty + provider_state + month`
- higher positive values indicate “above peer norms” for that metric

---

## 5) Member Risk Output (`outputs/anomaly/member_month_risk_scored.csv`)

**Grain:** member-month

| Column | Type | Description |
|---|---|---|
| iforest_score | float | Isolation Forest anomaly score (0–1) |
| iforest_risk | float | Percentile rank |
| risk_score | float | Composite risk |
| risk_percentile | float | Percentile rank |
| hop_flag | int | 1 if providers ≥ threshold |
| high_util_flag | int | 1 if claims ≥ threshold |
| flag_overall_high | int | 1 if overall risk ≥ threshold |

---

## Interpretation Notes (SIU-Friendly)

- Use `risk_score` to rank cases for investigation.
- Use `_peer_rz` fields to explain *why* a provider is risky vs similar peers.
- Use `claims_spike_vs_roll3` and `allowed_spike_vs_roll3` to highlight sudden shifts.
- Use `em_high_rate` + `em_high_shift_vs_roll3` to support an upcoding narrative.
- Use `rare_icd_rate` + `icd10_entropy` to flag abnormal diagnosis patterns.
