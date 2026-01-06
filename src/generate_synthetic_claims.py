#!/usr/bin/env python3
"""
Synthetic Healthcare Claims Generator (De-identified)

Generates:
- Members, providers (NPI-like IDs), specialties, geography
- Claims with ICD-10-CM + CPT, costs, place of service, dates
- Embedded anomaly patterns (FWA-like signals) for portfolio modeling:
  A) High-frequency office visits
  B) Upcoding-like higher-cost E/M mix
  C) Rare diagnosis concentration
  D) Member "provider hopping"

Outputs:
- data/raw/claims.csv
- data/raw/members.csv
- data/raw/providers.csv
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd

# ----------------------------
# Config / Seed
# ----------------------------
RNG = random.Random(42)
NP_RNG = np.random.default_rng(42)

# Basic code dictionaries (small but realistic enough for portfolio)
ICD10_POOL = [
    # Common primary care + chronic
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("I10", "Essential (primary) hypertension"),
    ("E78.5", "Hyperlipidemia, unspecified"),
    ("J06.9", "Acute upper respiratory infection, unspecified"),
    ("J45.909", "Asthma, uncomplicated"),
    ("M54.5", "Low back pain"),
    ("F41.9", "Anxiety disorder, unspecified"),
    ("K21.9", "GERD without esophagitis"),
    ("N39.0", "UTI, site not specified"),
    ("R07.9", "Chest pain, unspecified"),
    ("R51.9", "Headache, unspecified"),
    # Musculoskeletal / imaging drivers
    ("M17.9", "Osteoarthritis of knee, unspecified"),
    ("M25.561", "Pain in right knee"),
    ("M25.562", "Pain in left knee"),
    # “Rarer” codes to create anomalies
    ("G35", "Multiple sclerosis"),
    ("C50.919", "Malignant neoplasm of breast, unspecified"),
    ("D69.6", "Thrombocytopenia, unspecified"),
    ("Q90.9", "Down syndrome, unspecified"),
]

# CPT codes: office visits, imaging, labs, ED
CPT_POOL = [
    ("99213", "Office/outpatient visit, est.", 90),
    ("99214", "Office/outpatient visit, est. (higher)", 140),
    ("99203", "Office/outpatient visit, new", 130),
    ("99204", "Office/outpatient visit, new (higher)", 200),
    ("93000", "ECG", 50),
    ("80053", "Comprehensive metabolic panel", 25),
    ("83036", "Hemoglobin A1c", 30),
    ("81002", "Urinalysis", 10),
    ("71046", "Chest X-ray", 60),
    ("72100", "Lumbar spine X-ray", 70),
    ("73721", "MRI lower extremity (no contrast)", 600),
    ("70450", "CT head/brain (no contrast)", 350),
    ("36415", "Venipuncture", 5),
    ("99283", "ED visit", 300),
    ("99285", "ED visit (high)", 650),
]

PLACE_OF_SERVICE = [
    ("11", "Office"),
    ("22", "Outpatient Hospital"),
    ("23", "Emergency Room"),
]

SPECIALTIES = [
    ("FAMILY", "Family Medicine"),
    ("IM", "Internal Medicine"),
    ("ER", "Emergency Medicine"),
    ("ORTHO", "Orthopedics"),
    ("RAD", "Radiology"),
]

STATES = ["MA", "NY", "NJ", "PA", "CT"]

# Specialty → typical CPT mix weights (rough, for realism)
SPECIALTY_CPT_WEIGHTS = {
    "FAMILY": {"99213": 0.55, "99214": 0.20, "80053": 0.10, "83036": 0.08, "81002": 0.05, "36415": 0.02},
    "IM": {"99213": 0.50, "99214": 0.25, "80053": 0.10, "83036": 0.10, "93000": 0.05},
    "ER": {"99283": 0.70, "99285": 0.30},
    "ORTHO": {"99213": 0.30, "99214": 0.20, "72100": 0.20, "73721": 0.30},
    "RAD": {"71046": 0.30, "70450": 0.30, "73721": 0.40},
}

# ICD distribution base weights (common > rare)
ICD_WEIGHTS = {
    "E11.9": 0.11,
    "I10": 0.12,
    "E78.5": 0.08,
    "J06.9": 0.10,
    "J45.909": 0.05,
    "M54.5": 0.08,
    "F41.9": 0.06,
    "K21.9": 0.06,
    "N39.0": 0.06,
    "R07.9": 0.05,
    "R51.9": 0.05,
    "M17.9": 0.06,
    "M25.561": 0.04,
    "M25.562": 0.04,
    "G35": 0.01,
    "C50.919": 0.01,
    "D69.6": 0.01,
    "Q90.9": 0.01,
}

ICD_DESC = {c: d for c, d in ICD10_POOL}
CPT_DESC = {c: d for c, d, _ in CPT_POOL}
CPT_BASE_ALLOWED = {c: amt for c, _, amt in CPT_POOL}

def weighted_choice(items, weights_dict):
    weights = np.array([weights_dict[i] for i in items], dtype=float)
    weights = weights / weights.sum()
    return NP_RNG.choice(items, p=weights)

def make_npi_like(i: int) -> str:
    # 10-digit NPI-like string (not a real NPI)
    return f"{(1000000000 + i):010d}"

def make_member_id(i: int) -> str:
    return f"M{(100000 + i):06d}"

@dataclass
class AnomalyConfig:
    high_freq_provider_frac: float = 0.02
    upcoding_provider_frac: float = 0.02
    rare_dx_provider_frac: float = 0.015
    hopper_member_frac: float = 0.03

def generate_members(n_members: int) -> pd.DataFrame:
    ages = NP_RNG.integers(0, 90, size=n_members)
    genders = NP_RNG.choice(["F", "M"], size=n_members, p=[0.52, 0.48])
    states = NP_RNG.choice(STATES, size=n_members, p=[0.22, 0.25, 0.20, 0.18, 0.15])
    risk = NP_RNG.choice(["LOW", "MED", "HIGH"], size=n_members, p=[0.55, 0.35, 0.10])

    df = pd.DataFrame({
        "member_id": [make_member_id(i) for i in range(n_members)],
        "age": ages,
        "gender": genders,
        "state": states,
        "risk_tier": risk,
    })
    return df

def generate_providers(n_providers: int) -> pd.DataFrame:
    specialty_codes = [s[0] for s in SPECIALTIES]
    specialties = NP_RNG.choice(specialty_codes, size=n_providers, p=[0.38, 0.28, 0.10, 0.12, 0.12])
    states = NP_RNG.choice(STATES, size=n_providers, p=[0.22, 0.25, 0.20, 0.18, 0.15])

    df = pd.DataFrame({
        "provider_npi": [make_npi_like(i) for i in range(n_providers)],
        "specialty": specialties,
        "state": states
    })
    return df

def choose_cpt_for_specialty(spec: str, upcoding: bool = False) -> str:
    weights = dict(SPECIALTY_CPT_WEIGHTS[spec])

    # Upcoding-like behavior: shift weight from 99213/99203 to 99214/99204 and high ED
    if upcoding:
        if "99213" in weights and "99214" in weights:
            weights["99214"] = min(0.70, weights["99214"] + 0.25)
            weights["99213"] = max(0.05, weights["99213"] - 0.25)
        if "99203" in weights and "99204" in weights:
            weights["99204"] = min(0.60, weights["99204"] + 0.20)
            weights["99203"] = max(0.05, weights["99203"] - 0.20)
        if spec == "ER" and "99285" in weights:
            weights["99285"] = min(0.60, weights["99285"] + 0.20)
            weights["99283"] = max(0.20, weights["99283"] - 0.20)

        # Renormalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

    codes = list(weights.keys())
    probs = np.array([weights[c] for c in codes], dtype=float)
    probs = probs / probs.sum()
    return NP_RNG.choice(codes, p=probs)

def allowed_amount_for_cpt(cpt: str, spec: str) -> float:
    base = CPT_BASE_ALLOWED.get(cpt, 80)
    # Specialty/state variation and noise
    mult = 1.0
    if spec in ["ORTHO", "RAD"]:
        mult *= 1.15
    noise = NP_RNG.normal(1.0, 0.12)
    amt = max(5.0, base * mult * noise)
    return float(round(amt, 2))

def pick_pos(cpt: str, spec: str) -> str:
    # Rough mapping
    if spec == "ER" or cpt in ["99283", "99285"]:
        return "23"
    if spec in ["RAD"] and cpt in ["70450", "73721"]:
        return NP_RNG.choice(["22", "11"], p=[0.8, 0.2])
    return NP_RNG.choice(["11", "22"], p=[0.85, 0.15])

def generate_claims(
    members: pd.DataFrame,
    providers: pd.DataFrame,
    n_claims: int,
    start_date: str,
    end_date: str,
    anomaly_cfg: AnomalyConfig
) -> pd.DataFrame:
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    days = (end - start).days
    if days <= 0:
        raise ValueError("end_date must be after start_date")

    provider_list = providers["provider_npi"].tolist()
    member_list = members["member_id"].tolist()

    # Select anomaly cohorts
    n_prov = len(provider_list)
    n_mem = len(member_list)

    high_freq_providers = set(NP_RNG.choice(provider_list, size=max(1, int(n_prov * anomaly_cfg.high_freq_provider_frac)), replace=False))
    upcoding_providers = set(NP_RNG.choice(provider_list, size=max(1, int(n_prov * anomaly_cfg.upcoding_provider_frac)), replace=False))
    rare_dx_providers = set(NP_RNG.choice(provider_list, size=max(1, int(n_prov * anomaly_cfg.rare_dx_provider_frac)), replace=False))
    hopper_members = set(NP_RNG.choice(member_list, size=max(1, int(n_mem * anomaly_cfg.hopper_member_frac)), replace=False))

    # Helper maps
    prov_spec = dict(zip(providers["provider_npi"], providers["specialty"]))
    prov_state = dict(zip(providers["provider_npi"], providers["state"]))
    mem_state = dict(zip(members["member_id"], members["state"]))
    mem_risk = dict(zip(members["member_id"], members["risk_tier"]))

    # Base ICD list and probabilities
    icd_codes = list(ICD_WEIGHTS.keys())
    icd_probs = np.array([ICD_WEIGHTS[c] for c in icd_codes], dtype=float)
    icd_probs = icd_probs / icd_probs.sum()

    rows = []
    for i in range(n_claims):
        claim_id = f"C{(100000000 + i):09d}"

        # pick member
        member_id = RNG.choice(member_list)

        # pick provider: hopper members see more provider variety
        if member_id in hopper_members:
            provider_npi = RNG.choice(provider_list)
        else:
            # bias to same-state providers
            same_state = providers[providers["state"] == mem_state[member_id]]["provider_npi"].tolist()
            provider_npi = RNG.choice(same_state) if same_state and RNG.random() < 0.75 else RNG.choice(provider_list)

        spec = prov_spec[provider_npi]

        # date: uniform, but high-frequency providers have tighter clustering (more repeat visits)
        base_day = int(NP_RNG.integers(0, days))
        service_date = start + timedelta(days=base_day)
        if provider_npi in high_freq_providers and RNG.random() < 0.35:
            # force extra visits around same period
            service_date = service_date + timedelta(days=int(NP_RNG.integers(-3, 4)))

        # ICD selection
        if provider_npi in rare_dx_providers and RNG.random() < 0.55:
            # overuse rare dx codes
            rare_codes = ["G35", "C50.919", "D69.6", "Q90.9"]
            icd10 = RNG.choice(rare_codes)
        else:
            icd10 = NP_RNG.choice(icd_codes, p=icd_probs)

        # CPT selection (upcoding)
        upcoding = provider_npi in upcoding_providers
        cpt = choose_cpt_for_specialty(spec, upcoding=upcoding)

        pos = pick_pos(cpt, spec)
        allowed = allowed_amount_for_cpt(cpt, spec)

        # Risk adjustment: high-risk members slightly higher allowed (more labs/imaging)
        rt = mem_risk[member_id]
        if rt == "HIGH":
            allowed *= float(NP_RNG.normal(1.12, 0.05))
        elif rt == "LOW":
            allowed *= float(NP_RNG.normal(0.98, 0.04))
        allowed = float(round(max(5.0, allowed), 2))

        # Units: imaging usually 1; office visits 1; labs sometimes multiple
        if cpt in ["80053", "83036", "81002", "36415"]:
            units = int(NP_RNG.choice([1, 1, 1, 2, 2, 3], p=[0.45, 0.20, 0.15, 0.10, 0.06, 0.04]))
        else:
            units = 1

        paid = float(round(allowed * float(NP_RNG.normal(0.92, 0.06)), 2))
        paid = float(max(0.0, paid))

        rows.append({
            "claim_id": claim_id,
            "member_id": member_id,
            "provider_npi": provider_npi,
            "provider_specialty": spec,
            "provider_state": prov_state[provider_npi],
            "member_state": mem_state[member_id],
            "service_date": service_date.date().isoformat(),
            "icd10_code": icd10,
            "icd10_desc": ICD_DESC.get(icd10, ""),
            "cpt_code": cpt,
            "cpt_desc": CPT_DESC.get(cpt, ""),
            "place_of_service": pos,
            "allowed_amount": allowed,
            "paid_amount": paid,
            "units": units,
        })

    df = pd.DataFrame(rows)

    # Add derived time fields helpful for analysis
    df["service_date"] = pd.to_datetime(df["service_date"])
    df["service_year"] = df["service_date"].dt.year
    df["service_month"] = df["service_date"].dt.month
    df["service_dow"] = df["service_date"].dt.dayofweek  # 0=Mon
    df["is_weekend"] = df["service_dow"].isin([5, 6]).astype(int)

    # Add a hidden label column (optional) for evaluation in notebooks
    # 0=normal, 1=anomaly-provider, 2=anomaly-member
    df["synthetic_label"] = 0
    df.loc[df["provider_npi"].isin(list(high_freq_providers | upcoding_providers | rare_dx_providers)), "synthetic_label"] = 1
    df.loc[df["member_id"].isin(list(hopper_members)), "synthetic_label"] = 2

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=int, default=25000)
    parser.add_argument("--providers", type=int, default=1200)
    parser.add_argument("--claims", type=int, default=450000)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--outdir", type=str, default="data/raw")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    members = generate_members(args.members)
    providers = generate_providers(args.providers)

    claims = generate_claims(
        members=members,
        providers=providers,
        n_claims=args.claims,
        start_date=args.start,
        end_date=args.end,
        anomaly_cfg=AnomalyConfig()
    )

    members_path = os.path.join(args.outdir, "members.csv")
    providers_path = os.path.join(args.outdir, "providers.csv")
    claims_path = os.path.join(args.outdir, "claims.csv")

    members.to_csv(members_path, index=False)
    providers.to_csv(providers_path, index=False)
    claims.to_csv(claims_path, index=False)

    print("✅ Synthetic data generated")
    print(f"- {members_path}  ({len(members):,} members)")
    print(f"- {providers_path} ({len(providers):,} providers)")
    print(f"- {claims_path}    ({len(claims):,} claims)")
    print("\nNotes:")
    print("- provider_npi values are NPI-like but not real NPIs.")
    print("- synthetic_label is included for evaluation only (remove for 'production' realism).")

if __name__ == "__main__":
    main()
