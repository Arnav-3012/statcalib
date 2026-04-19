# statcalib

🔍 What is This?
StatCalib is a statistical audit framework that takes a pre-trained medical AI model and answers one question with formal mathematical proof:

When this AI says "87% confident" — is it actually right 87% of the time?

We audited DenseNet-121 (a 121-layer deep learning model) on the NIH ChestX-ray14 dataset (112,120 chest X-rays, 14 disease labels). We focused on Pleural Effusion — fluid around the lungs — as the primary pathology.
Short answer: No. The AI is lying about its confidence. And we proved it.

🌍 Why This Matters
Companies like Qure.ai, 5C Network, and Niramai deploy AI models across thousands of Indian hospitals every day. Doctors receive confidence scores like "95% sure this patient has Effusion" and make treatment decisions based on them.
Research shows these scores are almost always miscalibrated — a model saying 95% confident may only be right 63% of the time. That's 32 wrong diagnoses per 100 patients being acted on with false certainty.
This problem is explicitly listed as unresolved in TRIPOD+AI (2024) — the global standard for clinical AI reporting.

📊 Key Results
MethodECE BeforeECE AfterReductionSSDI UnitOriginal DenseNet-1210.1692—baseline—Platt Scaling0.16920.016590.2%Unit 4 + 6Isotonic Regression0.16920.011093.5%Unit 7Ridge-Platt Scaling0.16920.017089.9%Unit 5
ECE = Expected Calibration Error. Lower is better. 0 = perfect.
Statistical Proof

Hosmer-Lemeshow test: H = 799.15, p < 0.001 → formally rejected "model is well-calibrated"
Wilcoxon Signed-Rank test: p ≈ 0.00 for all three methods → improvement is statistically real
Bootstrap 95% CIs: [0.1583, 0.1801] (original) vs [0.0071, 0.0225] (best) → non-overlapping = proof
AUC-ROC: 0.7839 before AND after → calibration did not hurt discriminative ability


🗂️ Repository Structure
statcalib/
│
├── app.py                          # Streamlit home page
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── pages/                          # Streamlit multi-page app
│   ├── 1_The_Problem.py            # The clinical problem + weather analogy
│   ├── 2_Effusion_Deep_Dive.py     # Complete audit pipeline for Effusion
│   ├── 3_All_14_Diseases.py        # Systematic audit — all NIH pathologies
│   └── 4_Conformal_Prediction.py  # Prediction sets with coverage guarantees
│
├── notebooks/                      # Analysis notebooks (run in order)
│   ├── 01_smoke_test.ipynb         # Model loading verification
│   ├── 02_extract_scores.ipynb     # Score extraction (10,000 samples)
│   ├── 03_exploration.ipynb        # Exploratory data analysis
│   ├── 04_calibration_measurement.ipynb  # ECE + Hosmer-Lemeshow test
│   ├── 05_hypothesis_testing.ipynb # Z-tests + Bonferroni correction
│   ├── 06_calibration_methods.ipynb      # Platt, Isotonic, Ridge-Platt
│   ├── 07_proof_and_comparison.ipynb     # Wilcoxon + bootstrap CIs
│   ├── 08_subgroup_analysis.ipynb  # Age/sex fairness analysis
│   ├── 10_conformal_prediction.ipynb     # Conformal prediction
│   └── Multi-DiseaseCalibrationAudit.ipynb  # All 14 diseases
│
└── data/                           # Data files (scores + calibrated arrays)
    ├── scores.csv                  # 10,000 samples: image_id, label, score
    ├── robust_calibration_results.csv  # Master results — all 14 diseases
    ├── s_test.npy                  # Original test scores
    ├── y_test.npy                  # Ground truth labels
    ├── s_platt.npy                 # Platt-calibrated scores
    ├── s_iso.npy                   # Isotonic-calibrated scores
    ├── s_ridge.npy                 # Ridge-Platt-calibrated scores
    └── subgroup_results.json       # Age/sex subgroup ECE results

🧮 How Every SSDI Topic Is Used
This project was built for the SSDI (Statistical Structures in Data and Inference) course at NMIMS MPSTME. Every statistical method in the syllabus maps directly to a real step in the audit:
SSDI UnitTopicHow Used in StatCalibUnit 1MLE, Bootstrap CIsPlatt Scaling fitting (MLE) + 95% CIs on ECEUnit 2Z-test, BonferroniBin-level overconfidence testing (10 tests)Unit 3Chi-Square / GOFHosmer-Lemeshow test IS a chi-square GOF testUnit 4Logistic RegressionPlatt Scaling is logistic regression on scoresUnit 5Ridge RegularizationRidge-Platt adds L2 penalty, λ selected by CVUnit 6GLMsPlatt = Binomial GLM with logit link functionUnit 7NonparametricIsotonic Regression (PAV) + Wilcoxon test

🚀 Running Locally
1. Clone the repo
bashgit clone https://github.com/yourusername/statcalib.git
cd statcalib
2. Install dependencies
bashpip install -r requirements.txt
3. Run the Streamlit app
bashstreamlit run app.py
4. Or run notebooks (in order)
bashcd notebooks
jupyter notebook
Start from 01_smoke_test.ipynb and run sequentially.

Note: The .npy files and scores.csv in data/ are pre-computed. If you want to regenerate from scratch, you need the NIH ChestX-ray14 dataset and torchxrayvision installed (pip install torchxrayvision).


📦 Tech Stack
LibraryVersionPurposescipy.stats≥1.11chi2, wilcoxon, norm — all hypothesis testssklearn≥1.3LogisticRegression, IsotonicRegression, metricsnumpy≥1.24Bootstrap resampling, ECE computationpandas≥2.0Data handling, results tablesmatplotlib≥3.7Reliability diagrams, all static plotsplotly≥5.18Interactive charts in Streamlitstreamlit≥1.32Dashboard and multi-page apptorchxrayvisionlatestDenseNet-121 model loading (notebooks only)torch≥2.0Tensor ops during score extraction

🔬 Novel Contributions

Formal hypothesis-testing-based calibration audit — not just visual inspection. Hosmer-Lemeshow + Z-tests + Wilcoxon + Bootstrap CIs all applied together.
First systematic calibration audit across all 14 NIH ChestX-ray14 pathologies — most published papers audit only one disease. We show miscalibration is a structural property of DenseNet-121, not disease-specific.
Ridge-regularized Platt Scaling for imbalanced medical data — L2 penalty stabilizes calibration fitting when positive class prevalence is < 10%.
Subgroup fairness analysis — ECE computed separately for age groups (< 40, 40–60, > 60) and sex (M/F). Elderly patients show ECE = 0.2154 vs 0.1494 for under-40 — a patient safety finding.


📚 Background

Dataset: NIH ChestX-ray14 — 112,120 frontal X-rays, 30,805 unique patients, 14 thoracic disease labels extracted via NLP from radiology reports
Model: DenseNet-121 pre-trained via torchxrayvision — audit only, no retraining
Target pathology: Pleural Effusion (AUC = 0.87, highest in dataset)
Why Effusion: AUC 0.87 vs Pneumonia's 0.63 — stronger discrimination = stronger calibration story
Calibration ≠ Accuracy: A key finding — post-hoc calibration corrects confidence scores without changing the model's ranking ability (AUC preserved at 0.7839)


📖 References

Hosmer, D.W. & Lemeshow, S. (1980). A goodness-of-fit test for the multiple logistic regression model.
Platt, J. (1999). Probabilistic outputs for support vector machines.
Zadrozny, B. & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates.
Guo, C. et al. (2017). On calibration of modern neural networks. ICML.
Collins, G.S. et al. (2024). TRIPOD+AI: reporting guideline for clinical prediction models using AI. BMJ.
Wang, X. et al. (2017). ChestX-ray8: Hospital-scale chest X-ray database. CVPR.


👤 About
Arnav
BTech Data Science
NMIMS MPSTME, Mumbai
SSDI (Statistical Structures in Data and Inference) — Course Project

📄 License
MIT License — free to use, modify, and distribute with attribution.
