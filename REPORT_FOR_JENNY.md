# DQA analysis — update from Erez, April 2026

Hi Jenny — this doc now covers two back-to-back sessions of edits (April 22 and April 23–24). The current PDF is `tibu_dqa.pdf` at the repo root. Session 1 (items 1–9 below) fixed several bugs in the classification logic and reworked the age-of-record figures. Session 2 (the second big section) is a larger structural pass: I retitled the document, rewrote the Overview for outside readers, commented out a couple of subsections that became less central, and added a new §5 that reinterprets the Type I error rate.

> **Heads-up:** to-dos for you are in the "For you to pick up" section at the bottom.

## Two headline findings

1. **COVID-era reporting breakdown at two specific clinics.** Missing-in-TIBU rates spike to ~15% for records with treatment outcomes in 2020 Q1–Q2, concentrated at two clinics (rates 12.7% and 15.4% vs. 3.8% study-wide) that apparently stopped updating TIBU while paper entries continued. 63 of 116 missing-in-TIBU records fall in those two quarters; 50 are at those two clinics.
2. **About 2/3 of Type I ("false positive") errors are likely audit artifacts, not TIBU errors.** When TIBU outcome dates are compared with the audit-transcribed paper outcome dates, Type I records show TIBU *later* than paper in 16 of 24 cases with parseable pairs. The most plausible story: paper was updated after the audit transcribed it (e.g., a patient marked LTFU at month 3 who later returned and completed treatment), and TIBU now reflects the revised outcome. On this reading the headline 1.0% Type I rate overstates true TIBU misclassification; the residual is closer to ~0.3%. **Type II and missing-in-TIBU rates are *not* affected by this caveat** — those look like real TIBU lapses.

---

## Session 1 changes (Apr 22)

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

This lets false-missing records appear on the over-time and reporting-lag figures alongside the other categories.

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

Replaced with smoothed curves using `_smoothed_curve_with_ci()` (statsmodels lowess with `frac=0.35`) plus 500-resample percentile bootstrap for CI bands. Two distinct axes: `age_reg` (days from registration to snapshot) and `age_out` (days from outcome to snapshot). Added a new ECDF view of reporting lag by error category.

### 7. Section order flipped: over-time now leads

Since the COVID spike in missing-in-TIBU is the dominant signal, the over-time figure (Fig 2) now leads §4.5. The three age-of-record views followed as supplementary framings of the same data (session 2 subsequently commented these out — see below).

### 8. Added an explanation paragraph in-line

The paragraph after Figure 2 walks the reader through the COVID-era finding: 63 of 116 missing-in-TIBU records have outcomes in 2020 Q1–Q2; 50 of 116 are at two clinics with rates of 12.7% and 15.4% vs. 3.8% study-wide; Type I and Type II are both low and flat.

### 9. SI prose fix

The Overview prose previously said "patients registered between January 2017 and December 2021" — the actual range is April 2018 to December 2019. Updated.

### Code housekeeping (session 1)

- **Refactored `generate_error_by_clinic`, `generate_error_by_patient_characteristics`, and `generate_clinic_characteristics_table`** to call `_build_error_df` rather than reimplementing the classification inline. Prevents the same NC/false_miss drift from happening silently in three places.
- **Removed `generate_missing_timing_check`** (no longer applicable under the new semantics).
- **Cleaned up `main()`**: removed a stray `1` statement, a duplicate guard, a duplicate call to `generate_patient_characteristics_table`, and a trailing debug print.
- **Removed `print(">>> SCRIPT STARTED <<<")`** at the top.
- **Regrouped imports and path constants**; expanded the module docstring.

---

## Session 2 changes (Apr 23–24)

### Document restructure
- Renamed `main.tex` → `tibu_dqa.tex`, and the canonical PDF is now `tibu_dqa.pdf` at the repo root. Single tracked artifact; no more dated versioning. The four historical dated PDFs (`20260401.pdf` etc.) were removed — they're recoverable via `git log -- tibu_dqa.pdf` or `git log -- main.pdf`.
- Retitled "Data Quality Assurance Analysis / Keheala Study 2" → "TIBU Data Quality Audit". The audit is TIBU-specific and was conducted in the Keheala context but stands independently of the trial results.
- Switched body font to Times New Roman (via `mathptmx`). Computer Modern felt busy for a dense document with lots of tables.
- Un-starred Overview so it's §1 in the ToC.

### Overview rewrite
Rewritten for a wider audience (TB clinicians, trial collaborators, external reviewers):
- Short framing of Keheala + cite of the medRxiv preprint.
- Justified why paper is the ground truth (paper → TIBU transcription, so discrepancies live on the digital side).
- Added clinical context on TB treatment duration (~6 months drug-sensitive, 18–24 months MDR), so the late-2021 outcome window makes sense.
- Defined Type I / Type II / missing-in-TIBU with clinical significance (Type I *overstates* program success; Type II *understates*; missing-in-TIBU = digital surveillance gap, distinct from LTFU).
- Dropped references to MITT (not relevant to a TIBU audit).
- Added a **Key Findings** framed box at the end of the Overview, calling out both the COVID-era breakdown and the Type I reinterpretation.
- Audit-window prose corrected: the earlier "Nov 2018 – Sep 2019" didn't match the data (paper outcome dates extend into mid-2020). Now says "multiple waves beginning in late 2018; the latest outcome dates captured by the audit fall in mid-2020".

### New §3 Record Matching Procedure (placeholder)
We need details from the Keheala data team on how SCRNs were matched, what the 275 unmatched audit records look like, etc. Section is a short placeholder flagging this is forthcoming.

### §4 Outcome Errors (renamed)
Section renamed to "Outcome Errors" for clarity. Two subsections commented out behind `\iffalse ... \fi` blocks — the underlying analysis still runs and the LaTeX fragments are still produced in `output/`, so removing the `\iffalse` brings them back:
- §4.4 Record age and reporting lag by error type (Table 5).
- §4.6 Supplementary views — age_reg, age_out, and ECDF figures.

Motivation: once the §5 analysis showed that most Type I records are probably audit artifacts, the age-axis framing in §4.4/§4.6 became less informative as a "what's wrong with the data" story.

### New §5: Incorporating Date Fields to Assess Outcome Errors
The key analytical addition.
- **Rationale**: we can't treat paper dates as ground truth for date accuracy — the audit team didn't prioritize date entry, and 12.6% of paper outcome dates are blank plus another 0.8% unparseable (typos like "30/2/2020", year slips "0202"/"2029"). So we don't attempt a full date audit. But the TIBU-vs-paper outcome-date comparison is informative for interpreting the outcome-error counts from §4.
- **Figure**: scatter of TIBU outcome date vs. paper outcome date for DQA-matched records, axes clipped to 2018–2022, with year-offset diagonals visible. No-error points plot as faint grey background so Type I (red) and Type II (green) pop on top.
- **Finding**: Type I records show TIBU *later* than paper in ~2/3 of cases (16 of 24) — consistent with paper having been updated after the audit transcribed it, with TIBU reflecting the revised outcome. Type II shows TIBU *earlier* in ~3/4 of cases (14 of 19) — still looks like real TIBU error (premature closure of a patient who later returned and completed).
- **Conclusion**: Type I rate 1.0% probably overstates true TIBU misclassification; the residual is closer to ~0.3%. Type II (0.7%) and missing-in-TIBU (3.8%) are real TIBU lapses.

### Terminology cleanup
- "Incomplete data" / "False missing" → **"Missing in TIBU"** everywhere (Python column headers, figure legends, prose).
- Column-order prose mismatch fixed: §4.2 and §4.3 prose now reads "false-positive, false-negative, and missing-in-TIBU rates" to match table column order.

### Data quirks documented but not pursued
Mostly in commented-out `\iffalse` blocks at the bottom of §5 for future reference:
- **Year-offset typos**: ~145 matched records have TIBU/paper dates off by exactly 1 or 2 years, overwhelmingly (145 of 146) in the no-error category — the typo doesn't flip the outcome code, just corrupts the date.
- **TIBU internal consistency**: ~6% of records have outcome-before-registration, 3% have outcome-before-treatment-start, 8% have implied duration < 30 days. For the DQA-matched subset, the paper date can triangulate which TIBU field is wrong ~half the time. Analysis + three-row table still generated in `output/` (`tblSI_date_consistency.tex`, `tblSI_date_triangulation.tex`); not included in the live doc.
- **Workflow quirk**: `dateofregistration` is on/after `dateoftreatmentstarted` in ~90% of records, typical gap a week or two — probably a TIBU data-entry convention (case formally registered after treatment initiation), not a data-quality issue.

---

## For you to pick up

Items 1–4 are carried over from session 1 (still open). Items 5–7 are new from session 2.

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

3. **Variable-name swap in `_build_error_df`**. We work around it with local remapping in every output function, but `false_neg` stores Type I events and `false_pos` stores Type II. Renaming them to `type1` / `type2` (or `fp_digital` / `fn_digital`) would remove a persistent source of confusion. Tag all callers when renaming.

4. **Paper-audit date censoring is asymmetric**. `clean_dqa_data` doesn't censor absurd `treatmentoutcomedate` values (we saw min `0202-03-30` and max `2029-12-04` in the raw file). `_build_error_df` censors year < 2010 or > 2025 for all date columns as a guard, but moving that censor upstream into `clean_dqa_data` would prevent garbage dates from propagating through any future analysis that reads the cleaned DQA frame directly. Lower priority now that §5 frames paper dates as not-ground-truth.

5. **Record matching procedure** — the §3 placeholder needs details from the Keheala data team: what fields were used to match SCRNs, how near-duplicates were resolved, what the 275 unmatched audit records look like. Would be good to know for the paper.

6. **Decision: uncomment §4.4 / §4.6?** — I commented out the record-age / reporting-lag subsections (Table 5 and the age_reg/age_out/ECDF figures) because once §5 reframed the Type I rate, those views felt less central. Easy to bring back by removing the `\iffalse ... \fi` wrappers if you disagree.

7. **Potential future analysis: replay §4.5 using paper outcome dates** — the over-time figure currently uses TIBU outcome dates (with paper fallback for missing-in-TIBU). Replaying with paper outcome dates throughout would separate "date-driven" from "outcome-driven" components of the COVID-era spike. We didn't do this because we can't fully trust paper dates, but it might still be informative as a sensitivity check.

8. **Sensitivity analysis: tighten the Type I / Type II definitions using outcome-date direction.** The §5 reinterpretation — "most Type I records are audit artifacts where paper was updated after the audit transcribed it" — could be operationalized directly in the classification. Specifically: only flag a matched record as a TIBU error if (a) the TIBU and paper outcomes disagree **and** (b) the paper outcome date is *later* than the TIBU outcome date (i.e., paper was updated after TIBU was entered and therefore is the more recent reading on which to ground-truth TIBU). Records where TIBU's outcome date is the later one get reclassified as concordant on the grounds that TIBU likely reflects a post-audit paper update. Under this rule, Type I drops from 24 to ~5 matched records (≈0.2% rate), Type II drops from 19 to ~14 (≈0.5%), and missing-in-TIBU is unchanged at 3.8%. Would be a clean one-table sensitivity analysis next to the headline rates — i.e., regenerate the §4.1 crosstab and the Table 3/4 error rates under both the raw and the date-gated definitions, and show both.

---

## How to reproduce

```bash
python analysis_dqa.py
cp tibu_dqa.tex output/ && cd output \
  && pdflatex tibu_dqa.tex && pdflatex tibu_dqa.tex \
  && cp tibu_dqa.pdf ..
```

The `output/` directory is regenerated on every run — don't hand-edit fragments there. `tibu_dqa.tex` at the repo root is the single source for document structure and prose.

Happy to walk through any of this on our next call. Shoot me questions whenever.

— Erez
