# DQA analysis — update from Erez

*April 29, 2026*

Hi Jenny — picking up from where you left off. The previous report (Apr 22--24) is now archived as `REPORT_FOR_JENNY_2026-04-24.md`; this one covers one session on Apr 29. Nothing on the to-do side this round — you'd already knocked out items 1--4 and 7 of the previous list, so we used this session to close out item 8 (the date-direction sensitivity), tidy terminology, and give the document a slightly more modern look.

## Session 3 changes (Apr 29)

### 1. Date-gated sensitivity classification (item 8)

The §5 finding from the previous round — TIBU outcome date is later than paper in ~70% of false-positive records, suggesting paper was updated post-audit — is now operationalised as a sister classification rather than only described in prose:

- Inside `_build_error_df`, three new columns: `type1_gated`, `type2_gated`, `missing_tibu_gated`. A discordant matched record is reclassified as concordant when `paper_date < tibu_date` (TIBU strictly later, so paper plausibly captured an interim audit-time state). Records with either date missing/unparseable preserve their headline classification — conservative default since the rule can't adjudicate.
- The headline columns (`type1`, `type2`, `missing_tibu`) and every output that uses them are byte-identical to before. No existing analysis is disturbed.
- Numerical effect: false positives 31 → 15 (1.05% → 0.51%); false negatives 21 → 17 (0.71% → 0.58%); missing-in-TIBU unchanged at 112 (3.79%).

### 2. New §5 sensitivity table (Table 6)

`generate_dqa_sensitivity_table()` produces `tblSI_sensitivity.tex` — a compact side-by-side of the headline vs.\ gated counts/rates with a footnote explaining the rule. Inserted into §5 right after the date-asymmetry prose, framed by a paragraph that introduces the gated definition explicitly.

### 3. New Appendix B — gated breakdowns

`generate_error_by_clinic`, `generate_error_by_patient_characteristics`, and `generate_error_over_time_figure` now accept a `gated=False/True` kwarg. When `True`, they use the gated columns and write outputs with a `_gated` suffix. Appendix B reproduces Tables 3 and 4 and Figure 2 under the gated definition.

I deliberately did not produce a "gated crosstab" — the Table 2 cells are outcome-pair counts that don't change under the gate; the gate only changes which cells get *labelled* as errors, and Table 6 already gives those labelled marginals. If you'd rather see a literal recreation of Table 2 with annotated cell breakdowns, easy to add.

### 4. New Appendix A — paper-date sensitivity check

The paper-date over-time sensitivity figure (formerly inline at §4.4 as Figure 3) was relocated to its own short appendix section, ahead of the gated breakdowns so the appendix order matches the order in which they're first referenced in the body.

### 5. Terminology consistency: "false positive / false negative" throughout

The Type I / Type II language is dense for non-statistics readers. The doc now introduces both terms once, parenthetically, in the §1 Overview bullets ("**False positive** in the digital record (a Type I error): ..."), and uses just "false positive" / "false negative" everywhere else — prose, table column headers, figure legends, Table 2 colour-coding legend, Table 5 row labels, Fig 3 scatter legend. The variable names in Python (`type1`, `type2`) are unchanged; this is purely a presentation change.

### 6. §5 preamble on audit fallibility

Added a short paragraph at the top of §5 noting that the audit itself can err — through transcription mistakes, by matching a paper record to the wrong patient, or by transcribing an out-of-date entry — and framing the date-direction analysis that follows as a probe of the third failure mode.

### 7. Numbers and §5 prose

The prose said "16 of 24" false-positive and "14 of 19" false-negative records had TIBU later/earlier than paper; the actual table values were 16 of 23 and 13 of 18 (the older numbers predated your `load_study_data` dedup). Corrected to match. The Key Findings box similarly predicted a "~0.3%" residual false-positive rate; the actual gated rate is 0.5%, so I updated it to match Table 6.

### 8. Aesthetic refresh of the .tex preamble

The document was readable but a touch dated:

- Document class 12pt → 11pt; margin 1.0in → 1.1in.
- Body font switched from `mathptmx` (Times) to **Latin Modern Sans** (`lmodern` + `\renewcommand{\familydefault}{\sfdefault}`). This is a clean modern sans-serif. We tried `helvet`/Helvetica and `tgheros` (TeX Gyre Heros) first — neither's TFM files are present in this TeX Live install, so LMSS is the closest available. If you have a richer texlive (e.g. `texlive-fonts-extra`), swapping `\usepackage{lmodern}` for `\usepackage[scaled=0.95]{helvet}` would give a true Helvetica look.
- Added `microtype` (subtle character protrusion / font expansion).
- Switched to `parskip` paragraphs (no first-line indent, baseline-spaced gaps).
- `titlesec` for slimmer section/subsection headings — bold but only modestly larger than body, no colour.
- `caption` with bold "Figure X:" / "Table X:" labels at 4pt skip.
- `hyperref` `colorlinks=true` with teal link colour (no red boxes around references).
- `\linespread{1.07}` for slightly more comfortable leading.
- Added a divider page that just says "Appendix" between the body and the appendices.

Tables (which are auto-generated) were not restyled — happy to convert them to `booktabs` rules in a separate pass if you want a more uniform look.

### 9. Housekeeping

- `output/main.pdf` (147 KB stale leftover from before the rename to `tibu_dqa.pdf`) staged for deletion. The .gitignore already excludes `output/tibu_dqa.pdf`, so only the canonical `tibu_dqa.pdf` at the repo root is tracked.
- The build pipeline is unchanged.

## How to reproduce

```bash
python analysis_dqa.py
cp tibu_dqa.tex output/ && cd output \
  && pdflatex tibu_dqa.tex && pdflatex tibu_dqa.tex \
  && cp tibu_dqa.pdf ..
```

— Erez
