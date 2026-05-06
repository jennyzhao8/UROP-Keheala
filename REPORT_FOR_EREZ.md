## Recent analyses

### 1. Clinic-level error vs. outcome scatter

**Description:** Plots the per-clinic DQA error rate (any Type I, Type II, or missing-in-TIBU record) against the per-clinic unsuccessful outcome rate (paper outcome ∈ {D, F, LTFU}). Clinics with fewer than 5 audited records are excluded. Markers are colour-coded by province and shaped by urban/rural status (× = urban, ○ = rural).

**Code:** `generate_clinic_error_vs_outcome_figure()` in `analysis_dqa.py` (line 1877), called from `main()`.

**Output:** `output/fig_clinic_error_vs_outcome.pdf`

---

### 2. Logistic regression table — predictors of DQA errors

**Description:**  Fits four logistic regressions predicting each error type (any error, Type I, Type II, missing-in-TIBU) from individual patient characteristics (age, sex, bacteriological confirmation, disease classification), clinic characteristics (urban/rural, clinic size), record age, and province fixed effects. Firth-penalized estimation is used for rare-outcome models where complete separation occurs. Results are saved as both a formatted LaTeX table and a standalone PDF.

**Code:** `generate_logistic_regression_table()` in `analysis_dqa.py` (line 1716), called from `main()`.

**Outputs:**
- `output/tblSI_logit_errors.tex` — LaTeX table included in `tibu_dqa.tex`
- `output/logit_predictors_of_dqa_errors.pdf` — standalone figure version