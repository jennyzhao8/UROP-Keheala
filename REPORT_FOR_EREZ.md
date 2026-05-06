# TIBU Data Quality Audit — Keheala Study 2

Supplementary data-quality analysis for Keheala Study 2, an RCT of a digital TB-adherence platform in Kenya. The analysis compares treatment outcomes in Kenya's national TIBU electronic registry against a contemporaneous paper-registry audit (ground truth). Output is a supplementary-information PDF (`tibu_dqa.pdf`).

## Repository overview

| Path | Purpose |
|------|---------|
| `analysis_dqa.py` | Single Python module — all analysis and figure generation |
| `tibu_dqa.tex` | LaTeX document that `\input`s generated table fragments |
| `tibu_dqa.pdf` | Canonical build artifact (updated on every build) |
| `newdata/` | Input CSVs (TIBU export, paper-audit, clinic metadata) |
| `output/` | Generated figures (PDF) and table fragments (`.tex`) — do not edit by hand |
| `tests/` | pytest suite covering data cleaning and error classification |

## Running the pipeline

```bash
python analysis_dqa.py                                        # regenerate all tables + figures
cp tibu_dqa.tex output/ && cd output \
  && pdflatex tibu_dqa.tex && pdflatex tibu_dqa.tex           # build PDF (run twice for refs)
cp output/tibu_dqa.pdf tibu_dqa.pdf                           # promote to repo root
```

## Recent analyses

### 1. Clinic-level error vs. outcome scatter

**What it does:** Plots the per-clinic DQA error rate (any Type I, Type II, or missing-in-TIBU record) against the per-clinic unsuccessful outcome rate (paper outcome ∈ {D, F, LTFU}). Clinics with fewer than 5 audited records are excluded. Markers are colour-coded by province and shaped by urban/rural status (× = urban, ○ = rural).

**Code:** `generate_clinic_error_vs_outcome_figure()` in `analysis_dqa.py` (line 1877), called from `main()`.

**Output:** `output/fig_clinic_error_vs_outcome.pdf`

---

### 2. Logistic regression table — predictors of DQA errors

**What it does:** Fits four logistic regressions predicting each error type (any error, Type I, Type II, missing-in-TIBU) from individual patient characteristics (age, sex, bacteriological confirmation, disease classification), clinic characteristics (urban/rural, clinic size), record age, and province fixed effects. Firth-penalized estimation is used for rare-outcome models where complete separation occurs. Results are saved as both a formatted LaTeX table and a standalone PDF.

**Code:** `generate_logistic_regression_table()` in `analysis_dqa.py` (line 1716), called from `main()`.

**Outputs:**
- `output/tblSI_logit_errors.tex` — LaTeX table included in `tibu_dqa.tex`
- `output/logit_predictors_of_dqa_errors.pdf` — standalone figure version

---

## Error taxonomy (brief)

| Error type | TIBU outcome | Paper outcome | Meaning |
|------------|-------------|---------------|---------|
| Type I (false positive) | C / TC (success) | D / F / LTFU (failure) | TIBU falsely claims positive |
| Type II (false negative) | D / F / LTFU | C / TC | TIBU falsely claims negative |
| False missing | blank or NC | any classifiable outcome | Missing from digital record |

Transfer-out (`TO`) records are excluded throughout. `NC` (not completed) is treated as blank, not a medical outcome. Figures use consistent colours: **red = Type I, green = Type II, blue = false missing**.
