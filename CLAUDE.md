# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Data Quality Assurance (DQA) analysis for Keheala Study 2, a TB adherence RCT in Kenya. The codebase compares treatment outcomes recorded in Kenya's national electronic TB registry (TIBU) against a contemporaneous paper-registry audit, which is treated as ground truth. Output is a supplementary-information PDF for the trial paper.

Scope: 119,653 TIBU records (registered April 2018 – December 2019; outcomes through late 2021), 3,645 audited paper records, 3,370 matched to TIBU, 3,111 classifiable after excluding transfer-out.

## Pipeline

Single-command pipeline: `python analysis_dqa.py` reads CSVs in `newdata/`, writes LaTeX table fragments and PDF figures to `output/`, then `tibu_dqa.tex` at the repo root `\input`s those fragments.

```bash
python analysis_dqa.py                             # regenerate all tables + figures
cp tibu_dqa.tex output/ && cd output \
  && pdflatex tibu_dqa.tex && pdflatex tibu_dqa.tex  # build PDF (run twice for refs/toc)
cp output/tibu_dqa.pdf tibu_dqa.pdf                # promote to canonical artifact
```

The LaTeX fragments are written to `output/` and `tibu_dqa.tex` uses unqualified `\input{tblSI_...}` paths, so the build runs inside `output/`. The single canonical build artifact is `tibu_dqa.pdf` at the repo root; it is tracked in git and overwritten on every build. The document was renamed from `main.*` to `tibu_dqa.*` in April 2026 to reflect that the audit is TIBU-specific and only contextually associated with the Keheala trial. Older dated PDFs (e.g. `20260423.pdf`) were removed at the same time — use `git log -- tibu_dqa.pdf` (or the earlier `main.pdf`) to recover historical builds.

## Architecture

`analysis_dqa.py` is a single module. `main()` orchestrates all outputs; each `generate_*` function is independently runnable given the two base DataFrames from `load_study_data()` and `clean_dqa_data()`. A shared `_build_error_df(main_df, dqa_df)` helper performs the main↔DQA merge and the error-classification that most figure functions need.

**Two-source merge is the backbone.** Nearly every `generate_*` function starts with:
```python
merged = pd.merge(main_df, dqa_df[["scrn", "to_paper"]], on="scrn", how="inner")
```
`main_df` is the TIBU-side study-clinic data; `dqa_df` is the cleaned paper-registry audit. `scrn` is the patient identifier. The script also loads `TIBU_firstnm_deidentified.csv` separately for the nationwide "All TIBU" column of Table 1a/1b.

**Data files in `newdata/`:**
- `study2_cleaned.csv` — study-clinic MITT sample (TIBU side of the DQA).
- `TIBU_firstnm_deidentified.csv` — full nationwide TIBU (~119k rows), used for the "All" column in patient-characteristics tables.
- `DQA_combined.csv` — digitized paper-registry audit.
- `clinic_summary_deidentified.csv` — clinic-level metadata (urban/rural, province, tibu_patients).
- `Urine_Test_Results.csv` — present but not currently used.

Dates in `TIBU_firstnm_deidentified.csv` are stored as **Stata serial days** (integers, epoch 1960-01-01). Study-clinic dates in `study2_cleaned.csv` are pre-parsed as the `*_formatted` columns.

## Error taxonomy — critical subtleties

The classification is central; several details are non-obvious and easy to break:

- **"Good" outcomes:** `C` (cured), `TC` (treatment completed).
- **"Bad" outcomes (both TIBU and paper):** `D` (died), `F` (treatment failed), `LTFU` (lost to follow-up).
- **`NC`** (not completed) is **treated as a blank placeholder**, not a medical outcome. A record with TIBU = `NC` and a classifiable paper outcome counts as false-missing.
- **`TO`** (transfer out; includes the recoded `MT4` drug-resistance referral) is **excluded** entirely from the DQA — transfer-out is not an auditable final outcome.
- **Type I error** = false positive in the digital record: TIBU records success (`C`/`TC`), paper records failure (`D`/`F`/`LTFU`). Semantically: TIBU falsely claims a positive outcome.
- **Type II error** = false negative in the digital record: TIBU records failure, paper records success. Semantically: TIBU falsely claims a negative outcome.
- **False missing** = TIBU is blank or `NC`, paper has a classifiable outcome. Semantically: missing from the digital record.
- **Denominator**: records with classifiable paper outcome (paper is the ground truth; records without it are uncheckable). Records where TIBU has an outcome but paper is blank are dropped as uncheckable.
- In `clean_dqa_data()`, paper outcomes `LFTU`→`LTFU`, `TF`/`CATIV`→`F`, `NF`→`""`, and duplicates are resolved via a Stata-style dedup (kept for parity with `_study2_analysis.do`).

### Historical note

An earlier iteration of the code (and its prose) inverted the false-missing semantics — the variable was computed as TIBU-has-outcome-paper-blank. That produced tables whose "Incomplete data" column was the opposite of the Table 2 crosstab legend. The current implementation matches the crosstab legend: false missing = missing from the digital side.

### Error type variable names

`_build_error_df` produces three columns that map directly to the Table 2 convention (red = Type I):

- `type1` — Type I events (TIBU success, paper failure — digital false positive).
- `type2` — Type II events (TIBU failure, paper success — digital false negative).
- `missing_tibu` — TIBU blank, paper has a classifiable outcome.

### "Age of record" and the TIBU snapshot

Two distinct age concepts, both measured against a **single global snapshot date of 2022-01-01** (inferred from `max(tibu_date) = 2021-12-31`):

- `age_reg` = snapshot − registration date (how old the record is at snapshot).
- `age_out` = snapshot − TIBU outcome date (reporting lag).

The snapshot is a constant, not per-record. The earlier per-source-file inference (using `max(outcome_date)` within each `source_file` in `TIBU_firstnm_deidentified`) was abandoned because `study2_cleaned.csv` was evidently enriched by a later consolidated pull whose date isn't captured in any source_file — 1,147 of 3,023 records had `tibu_date > source_file_proxy`, which would have produced negative lags or inconsistent reference points.

`age_days` is retained as an alias for `age_reg` only for backward compatibility with the older Table 5 definition, which used treatment duration (registration → TIBU outcome); that table has been rebuilt to show both new age definitions.

## Output conventions

- **Do not hand-edit files in `output/`** — they are regenerated by `analysis_dqa.py` on every run. Modify the generating function instead.
- `output/_old_output/` contains stale artifacts from earlier iterations; ignore unless asked.
- Figures use three consistent colors: **red = false positive (Type I)**, **green = false negative (Type II)**, **blue = false missing**. Matches the Table 2 crosstab legend. Preserve this mapping.
- Smoothed curves (Figs 2, 3, 5) use lowess with `frac=0.35` and percentile bootstrap 95% CIs (500 resamples) via the `_smoothed_curve_with_ci` helper.

## Known issues flagged for Jenny (not fixed)

1. **Clinic-size median bug** in `generate_error_by_clinic`: `median_size = df["tibu_patients"].median()` is computed on a patient-level merged df and returns 2 because many rows have null `tibu_patients`. Should be computed at the clinic level.
2. **`load_study_data` inflates the dataframe**: `main_df.merge(tibu[["scrn","source_file"]], on="scrn")` duplicates rows for patients who appear in multiple source_files (17,160 → 17,740). Affects every downstream aggregation that doesn't dedup.
3. **Variable name swap** in `_build_error_df` (see above).
4. **Paper-audit dates** include absurd values (e.g. year 0202, 2029); `_build_error_df` censors year < 2010 or > 2025, but `clean_dqa_data` doesn't — Jenny may want to move the censor upstream.
5. **Sloppy bits in Jenny's `main()`**: stray `1` statement, a duplicate guard, and `generate_patient_characteristics_table` called twice. Cosmetic; would clean up in a pass-through.

## Collaboration

This is a shared workspace with student Jenny Zhao (github.com/jennyzhao8/UROP-Keheala). The local directory is her repo clone. Pull before starting work; she pushes frequent "Add files via upload" commits in addition to normal ones.
