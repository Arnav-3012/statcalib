# 🩺 StatCalib — Statistical Calibration Audit Framework for Medical AI

---

## What is This?

StatCalib is a **statistical audit framework** that takes a pre-trained medical AI model and answers one question with formal mathematical proof:

> *When this AI says "87% confident" — is it actually right 87% of the time?*

We audited **DenseNet-121** on the **NIH ChestX-ray14 dataset** (112,120 chest X-rays, 14 disease labels), focused on **Pleural Effusion**. Short answer: No. The AI is lying about its confidence. And we proved it.

---

## Why This Matters

Companies like **Qure.ai**, **5C Network**, and **Niramai** deploy AI models across thousands of Indian hospitals every day. Doctors receive confidence scores like "95% sure this patient has Effusion" and make treatment decisions based on them.

Research shows these scores are almost always **miscalibrated** — a model saying 95% confident may only be right 63% of the time. That is 32 wrong diagnoses per 100 patients being acted on with false certainty. This problem is explicitly listed as unresolved in TRIPOD+AI (2024) — the global standard for clinical AI reporting.

---

## Key Results

| Method | ECE Before | ECE After | Reduction | SSDI Unit |
|--------|-----------|-----------|-----------|-----------|
| Original DenseNet-121 | 0.1692 | — | baseline | — |
| Platt Scaling | 0.1692 | 0.0165 | 90.2% | Unit 4 + 6 |
| Isotonic Regression | 0.1692 | 0.0110 | 93.5% | Unit 7 |
| Ridge-Platt Scaling | 0.1692 | 0.0170 | 89.9% | Unit 5 |

ECE = Expected Calibration Error. Lower is better. 0 = perfect.

**Statistical Proof**
- Hosmer-Lemeshow test: H = 799.15, p < 0.001 — formally rejected "model is well-calibrated"
- Wilcoxon Signed-Rank test: p ≈ 0.00 for all three methods — improvement is statistically real
- Bootstrap 95% CIs: [0.1583, 0.1801] original vs [0.0071, 0.0225] best — non-overlapping = proof
- AUC-ROC: 0.7839 before AND after — calibration did not hurt discriminative ability

---

## Repository Structure

```
statcalib/
├── app.py                              # Streamlit home page
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── pages/
│   ├── 1_The_Problem.py                # The clinical problem
│   ├── 2_Effusion_Deep_Dive.py         # Complete audit pipeline
│   ├── 3_All_14_Diseases.py            # All NIH pathologies
│   └── 4_Conformal_Prediction.py       # Prediction sets
├── notebooks/
│   ├── 01_smoke_test.ipynb
│   ├── 02_extract_scores.ipynb
│   ├── 03_exploration.ipynb
│   ├── 04_calibration_measurement.ipynb
│   ├── 05_hypothesis_testing.ipynb
│   ├── 06_calibration_methods.ipynb
│   ├── 07_proof_and_comparison.ipynb
│   ├── 08_subgroup_analysis.ipynb
│   ├── 10_conformal_prediction.ipynb
│   └── Multi-DiseaseCalibrationAudit.ipynb
└── data/
    ├── scores.csv
    ├── robust_calibration_results.csv
    ├── s_test.npy
    ├── y_test.npy
    ├── s_platt.npy
    ├── s_iso.npy
    ├── s_ridge.npy
    └── subgroup_results.json
```

---

## SSDI Syllabus Coverage

| SSDI Unit | Topic | How Used |
|-----------|-------|----------|
| Unit 1 | MLE, Bootstrap CIs | Platt Scaling fitting + 95% CIs on ECE |
| Unit 2 | Z-test, Bonferroni | Bin-level overconfidence testing (10 tests) |
| Unit 3 | Chi-Square / GOF | Hosmer-Lemeshow IS a chi-square GOF test |
| Unit 4 | Logistic Regression | Platt Scaling is logistic regression on scores |
| Unit 5 | Ridge Regularization | Ridge-Platt adds L2 penalty, lambda by CV |
| Unit 6 | GLMs | Platt = Binomial GLM with logit link function |
| Unit 7 | Nonparametric | Isotonic Regression (PAV) + Wilcoxon test |

---

## Running Locally

```bash
git clone https://github.com/yourusername/statcalib.git
cd statcalib
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| scipy.stats | chi2, wilcoxon, norm — all hypothesis tests |
| sklearn | LogisticRegression, IsotonicRegression |
| numpy | Bootstrap resampling, ECE computation |
| pandas | Data handling, results tables |
| plotly | Interactive charts in Streamlit |
| streamlit | Dashboard and multi-page app |
| torchxrayvision | DenseNet-121 model loading (notebooks only) |

---

## About

**Arnavv**
MBATech Data Science, Semester IV
NMIMS MPSTME, Mumbai

*SSDI (Statistical Structures in Data and Inference) — Course Project*

---

## License

MIT License — free to use with attribution.
