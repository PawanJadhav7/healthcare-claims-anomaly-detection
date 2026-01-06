# 🏥 Healthcare Claims Anomaly Detection  
### Fraud, Waste & Abuse (FWA) Analytics using ICD-10-CM & Utilization Signals

---

## 📌 Overview

Healthcare payers process millions of claims every day. Traditional rule-based fraud detection systems often generate **high false positives**, overwhelming Special Investigation Units (SIU) and delaying legitimate claims.

This project implements a **production-style, end-to-end anomaly detection pipeline** to identify potential **Fraud, Waste, and Abuse (FWA)** in healthcare claims using:

- ICD-10-CM diagnosis patterns  
- Utilization and cost signals  
- Peer group benchmarking (specialty + state + month)  
- Explainable anomaly scoring  
- BI-ready outputs for SIU triage  

The solution is designed to be **scalable, explainable, and analytics-driven**, closely mirroring real-world healthcare payer integrity systems.

---

## 🎯 Business Objectives

- Detect abnormal provider and member behavior
- Reduce false positives from static rule-based systems
- Prioritize SIU case reviews using risk-based ranking
- Provide explainability suitable for audit and compliance

---

## 👥 Target Users

- Special Investigation Units (SIU)
- Program Integrity & Compliance Teams
- Healthcare Analytics Teams
- Medical Directors & Auditors

---

## 📊 Key Results (Synthetic Data)

- ~14K provider-months and ~230K member-months analyzed
- Top ~2% of provider-months flagged as high risk
- Clear separation of normal vs anomalous utilization patterns
- Explainable drivers (peer comparison + trends) for every flag

---

## 🏗️ Architecture

```mermaid
flowchart LR
  A[Claims Source Data] --> B[Ingestion & QA]
  B --> C[Raw Zone]
  C --> D[Profiling & Baselines]
  D --> E[Feature Store]
  E --> F[Anomaly Detection]
  F --> G[Risk Scoring Outputs]
  G --> H[BI Dashboards]

  subgraph Explainability
    X1[Peer Group Benchmarks] --> F
    X2[Explainable Flags] --> H
  end
