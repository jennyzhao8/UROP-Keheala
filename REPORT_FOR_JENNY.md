# DQA analysis — update from Erez, April 2026

Hi Jenny — I spent a session going through the DQA analysis and making several substantive changes. The current PDF is `20260423.pdf` at the repo root. This note walks through what changed and why.

> **Heads-up:** there are a few to-dos I left for you at the bottom of this note (the "For you to pick up" section). They didn't block the SI build but they're worth a pass before the next round.

The big picture: we found that what's currently happening in 2020 Q1–Q2 is a **COVID-era reporting breakdown** at two specific clinics — false-missing rates spike to ~15% for records whose treatment outcomes fell in the pandemic's first two quarters, concentrated at two clinics that apparently stopped updating TIBU while paper entries continued. Getting to a figure that showed this clearly required several upstream fixes to the classification logic.

## Substantive changes to the analysis

### 1. Reversed the false-missing semantics

The biggest fix. In the prior code:

```python
df["false_miss"] = df["uo_paper"].isna().astype(int)   # paper is blank
```

This computed false-missing as **TIBU has an outcome, paper is blank** (~96 records). But the Overview prose and the Table 2 crosstab legend both describe "incomplete data" as the *opposite* — **TIBU blank, paper has an outcome**. The two were inconsistent.

The new classification aligns with the crosstab legend:

```python
df = merged.dropna(subset=["uo_paper"]).copy()         # paper must be ground truth
df["false_miss"] = df["uo_tibu"].isna().astype(int)    # TIBU is blank
```

Implications:
- The **denominator** is now records with a classifiable *paper* outcome (paper is the ground truth; records where paper is blank can't be audited).
- Records where TIBU has an outcome but paper is blank (previously ~96 "false-missing") are now dropped as uncheckable, not counted as any kind of error.
- False-missing now reflects a real digital-side gap (TIBU didn't record what paper did), which is the clinically interesting DQA finding.

### 2. Treated `NC` as a placeholder, not a medical outcome

Previously, `NC` ("not completed") was grouped with `D`, `F`, `LTFU` as a "bad" TIBU outcome. But NC is an interim status, not a medical result, so a record with TIBU = `NC` and paper = `C`/`TC` shouldn't count as a Type II disagreement — it should count as TIBU failing to finalise.

New grouping in `_build_error_df`:
```python
bad  = ["D", "F", "LTFU"]   # NC removed
good = ["C", "TC"]
```

This reclassifies 76 records (previously Type II or concordant) as false-missing. Together with the 40 records where TIBU is genuinely blank, the new false-missing count is 116.

### 3. Single global TIBU snapshot date

Previously the analysis didn't have a notion of "when the TIBU data was pulled", which is what you need to compute record age or reporting lag. I initially tried inferring per-source-file snapshot dates from `max(outcome_date)` within each `.xlsx` in `TIBU_firstnm_deidentified.csv`. That broke for 1,147 records because `study2_cleaned.csv` was evidently enriched by a later consolidated pull whose date isn't captured in any source_file — those records had TIBU outcomes *after* the latest source_file's max outcome date.

The simpler fix, which we agreed to: treat the whole extract as a single snapshot at **1 January 2022**, inferred from `max(tibu_date) = 2021-12-31` across the full sample (the latest outcomes cluster in late December 2021). A specific documented pull date from the data team would be better, but this is a defensible lower bound.

### 4. `outcome_date` falls back to paper date for false-missing records

Since false-missing records have TIBU blank, they have no TIBU outcome date. But by definition they *do* have a paper outcome with a date. So for the reporting-lag axis:

```python
df["outcome_date"] = df["tibu_date"].fillna(df["paper_date"])
df["age_out"] = (df["snapshot_date"] - df["outcome_date"]).dt.days
```

This lets false-missing records appear on Figures 2 (over time) and 4 (reporting lag) alongside the other categories.

### 5. Fixed the Type I vs Type II label swap

This is a separate bug in the original code. The variable names in `_build_error_df` are semantically inverted:

- `false_neg` holds `(TIBU success, paper failure)` → **Type I** (false positive in the digital record)
- `false_pos` holds `(TIBU failure, paper success)` → **Type II** (false negative in the digital record)

So throughout the code, `sub["false_pos"]` was being labelled as "False positive" in tables and figures — but it was actually the Type II count. Tables 3 and 4 (clinic / patient characteristics), the over-time figure, and `tblSI_DQAtype12error.tex` all had the two columns swapped relative to the Table 2 crosstab, where red = Type I = false positive.

Rather than rename the variables (which would touch every caller), I added a local remap inside each affected helper:

```python
fp_n, fn_n, fm_n = int(sub["false_neg"].sum()), int(sub["false_pos"].sum()), int(sub["false_miss"].sum())
```

Every output now uses the Table 2 convention: **red = Type I = false positive = `false_neg` variable**. I left a note in CLAUDE.md about this; eventually it'd be cleanest to rename the variables themselves, but that's a bigger change.

### 6. Replaced binned error-bar figures with smoothed lowess + bootstrap CI

The previous age-of-record figures were 60-day binned error-bars, restricted to N≥10 or N≥100 per bin. They were noisy and the caption claimed a 3-quarter rolling mean that wasn't actually applied. More fundamentally, the x-axis was labelled "age of record" but was actually *treatment duration* (registration → TIBU outcome date), so the figures conflated treatment-duration outliers (MDR, retreatment, early death) with record age at snapshot.

The new figures (Figs 2, 3, 4):
- Use `_smoothed_curve_with_ci()` (statsmodels lowess with `frac=0.35`) plus 500-resample percentile bootstrap for CI bands.
- Plot against two distinct axes: `age_reg` (days from registration to snapshot) and `age_out` (days from outcome to snapshot).
- Figure 4 inverts the x-axis so calendar time reads left-to-right, matching Figure 2.
- Figure 5 is a new ECDF view of reporting lag by error category.

### 7. Section order flipped: over-time now leads

Since the COVID spike in false-missing is the dominant signal, §2.5 now starts with the over-time figure (Figure 2). The three age-of-record views are now §2.6 and framed as alternative representations of the same data.

### 8. Added an explanation paragraph in-line

The paragraph after Figure 2 walks the reader through the COVID-era finding: 63 of 116 false-missing records have outcomes in 2020 Q1–Q2; 50 of 116 are at two clinics with false-missing rates of 12.7% and 15.4% vs. 3.8% study-wide; Type I and Type II are both low and flat.

### 9. SI prose fix

The Overview prose previously said "patients registered between January 2017 and December 2021" — the actual range is April 2018 to December 2019. Updated.

## Code housekeeping

A few refactors to prevent the same drift from happening in multiple places:

- **Refactored `generate_error_by_clinic`, `generate_error_by_patient_characteristics`, and `generate_clinic_characteristics_table`** to call `_build_error_df` rather than reimplementing the classification inline. Previously they each had their own `uo_tibu` / `uo_paper` computations, which meant the NC-as-blank change would have had to be made in three places separately.
- **Removed `generate_missing_timing_check`**. It was a diagnostic I'd added earlier under the old false-missing semantics; it no longer applies.
- **Cleaned up `main()`**: removed a stray `1` statement, a duplicate `if dqa_df_cleaned is None: return`, a duplicate call to `generate_patient_characteristics_table`, and a trailing debug print.
- **Removed `print(">>> SCRIPT STARTED <<<")`** at the top of the file.
- **Regrouped imports and path constants**; expanded the module docstring to describe the classification convention.

Net: 1,315 → 1,258 lines.

I also updated `CLAUDE.md` at the repo root with the new error taxonomy, the snapshot-date convention, and a note about the variable-name swap for future collaborators.

## For you to pick up

These are things I noticed but didn't fix (some were out of scope; some I wanted to leave for you to decide how to handle):

1. **Clinic-size median bug in `generate_error_by_clinic`** (~line 697). The footnote in Table 3 reads "at least 2 patients registered in TIBU (median)", which is clearly wrong. The cause:
   ```python
   median_size = df["tibu_patients"].median()
   ```
   is computed on the patient-level merged dataframe, where `tibu_patients` is null for many rows. It should be computed at the clinic level (one row per clinic). Worth fixing — the "Large clinic" / "Small clinic" split in Table 3 is currently meaningless.

2. **`load_study_data` inflates `main_df`**:
   ```python
   tibu = pd.read_csv(TIBU_DEIDENTIFIED, dtype=str)[["anon_scrn_tibu", "source_file"]]
   df = df.merge(tibu, on="scrn", how="left")
   ```
   TIBU has 3,757 patients appearing in multiple source_files (up to 9 each), so this merge duplicates rows. `main_df` goes from 17,160 → 17,740 rows. Every downstream aggregation that doesn't dedup picks up those duplicates. Easiest fix: aggregate `source_file` per scrn before the merge (e.g. keep only the latest), or drop the join entirely since the analysis doesn't actually depend on `source_file` anymore.

3. **Variable-name swap in `_build_error_df`**. We worked around it with local remapping in every output function, but `false_neg` stores Type I events and `false_pos` stores Type II. Renaming them to `type1` / `type2` (or `fp_digital` / `fn_digital`) would remove a persistent source of confusion. Tag all callers when renaming.

4. **Paper-audit date censoring is asymmetric**. `clean_dqa_data` doesn't censor absurd `treatmentoutcomedate` values (we saw min `0202-03-30` and max `2029-12-04` in the raw file). `_build_error_df` now censors year < 2010 or > 2025 for all date columns as a guard, but moving that censor upstream into `clean_dqa_data` would prevent garbage dates from propagating through any future analysis that reads the cleaned DQA frame directly.

5. **Minor prose mismatch** in §2.3: the sentence reads "reports false-negative, false-positive, and false-missing rates" but Table 4's column order is FP / FN / Incomplete data. Cosmetic — swap the word order or the column order, your choice.

## How to reproduce

```bash
python analysis_dqa.py
cp main.tex output/ && cd output && pdflatex main.tex && pdflatex main.tex
```

The `output/` directory is regenerated on every run — don't hand-edit fragments there. `main.tex` is the single source for the SI structure and prose.

Happy to walk through any of this on our next call. Shoot me questions whenever.

— Erez
