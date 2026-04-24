"""
Keheala Study 2 — DQA Analysis
==============================

Compares treatment outcomes recorded in TIBU (Kenya's national electronic TB
registry) against a contemporaneous paper-registry audit, which is treated as
ground truth. Produces LaTeX fragments and PDF figures in ``output/`` that
``main.tex`` at the repo root assembles into the SI document.

Classification (see CLAUDE.md for the full taxonomy):
    Type I  (false positive in digital): TIBU success, paper failure
    Type II (false negative in digital): TIBU failure, paper success
    False missing                      : TIBU blank/NC, paper has outcome

NC is treated as a blank placeholder; TO (transfer out, including MT4) is
excluded entirely. The denominator is records with a classifiable paper
outcome — records where paper is blank are uncheckable and dropped.

Stata logic derived from ``_study2_analysis.do`` (lines 623-673).
"""

import os
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

ROOT_DIR           = os.path.abspath(os.path.dirname(__file__))
DEIDENTIFIED_DIR   = os.path.join(ROOT_DIR, "newdata")
INPUT_DATA_CLEANED = os.path.join(DEIDENTIFIED_DIR, "study2_cleaned.csv")
DQA_DATA_DIR       = DEIDENTIFIED_DIR
TIBU_DEIDENTIFIED  = os.path.join(DEIDENTIFIED_DIR, "TIBU_firstnm_deidentified.csv")
OUTPUT_DIR         = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# SHARED FUNCTIONS (Data Loading & Cleaning)
# -----------------------------------------------------------------------------

def clean_dqa_data():
    """
    Loads and cleans DQA CSVs based on Stata logic.
    Returns a DataFrame containing cleaned Paper Registry data.
    """
    print("\n--- Loading DQA Data ---")

    # Try combined file first (Level 2 de-identified), then per-county files
    combined_path = os.path.join(DQA_DATA_DIR, "DQA_combined.csv")
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path, dtype=str)
        print(f"Loaded DQA_combined.csv: {len(df)} records")
    else:
        counties = ["Kakamega", "Kiambu", "Kisumu", "Machakos", "Mombasa", "Nairobi", "Turkana"]
        dfs = []
        for county in counties:
            path = os.path.join(DQA_DATA_DIR, f"DQA_{county}.csv")
            if not os.path.exists(path):
                print(f"Warning: {path} not found")
                continue
            df_c = pd.read_csv(path, dtype=str)
            dfs.append(df_c)

        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)

    print(f"Total Raw DQA Records: {len(df)}")
    
    # Normalize column names (remove spaces, lowercase)
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]
    
    # Handle de-identified column names
    if "anon_scrn" in df.columns and "scrn" not in df.columns:
        df = df.rename(columns={"anon_scrn": "scrn"})

    # Handle scn/patientid (Critical for Turkana)
    if "scrn" not in df.columns and "patientid" in df.columns:
        df = df.rename(columns={"patientid": "scrn"})
    elif "scrn" in df.columns and "patientid" in df.columns:
        df["scrn"] = df["scrn"].fillna(df["patientid"])
    elif "scrn" not in df.columns:
        print("Warning: SCRN column missing from some DQA data")

    # Outcome Cleaning Function
    def clean_outcome(x):
        if pd.isna(x): return x
        x = x.strip()
        if x == "LFTU": return "LTFU"
        if x in ["TF", "CATIV"]: return "F"
        if x == "NF": return "" # Stata: replace with empty string
        return x

    df["treatmentoutcome"] = df["treatmentoutcome"].apply(clean_outcome)
    
    # Drop specific excluded outcomes
    drop_outcomes = ["MT4", "TO", "N/A", "NTB"]
    df = df[~df["treatmentoutcome"].isin(drop_outcomes)]
    
    # Drop missing identifiers
    df = df.dropna(subset=["scrn"])
    
    # Dedup Logic (Stata replication)
    df = df.sort_values(by="scrn")
    df["dup"] = df.duplicated(subset=["scrn"], keep=False)
    df["prev_outcome"] = df.groupby("scrn")["treatmentoutcome"].shift(1)
    
    mask_drop = (
        df["dup"] & 
        (
            (df["treatmentoutcome"] == "") | 
            df["treatmentoutcome"].isna() | 
            (df["treatmentoutcome"] == df["prev_outcome"])
        )
    )
    df = df[~mask_drop]
    
    # Heuristic dedup for any remaining (Stata uses m:1 merge, requiring uniqueness)
    df = df.drop_duplicates(subset=["scrn"], keep="last")
    
    # Rename outcome for clarity
    df = df.rename(columns={
        "treatmentoutcome": "to_paper",
        "treatmentoutcomedate": "date_paper"
    })    
    return df

def load_study_data():
    df = pd.read_csv(INPUT_DATA_CLEANED, low_memory=False)
    if "anon_scrn" in df.columns:
        df = df.rename(columns={"anon_scrn": "scrn"})
    df["treatmentoutcome"] = df["treatmentoutcome"].replace("MT4", "TO")

    tibu = pd.read_csv(TIBU_DEIDENTIFIED, dtype=str)[["anon_scrn_tibu", "source_file"]]
    tibu = tibu.rename(columns={"anon_scrn_tibu": "scrn", "source_file": "tibu_source_file"})
    tibu["scrn"] = tibu["scrn"].astype(str)
    df["scrn"] = df["scrn"].astype(str)

    df = df.merge(tibu, on="scrn", how="left")
    return df

# -----------------------------------------------------------------------------
# ANALYSIS 1: SENSITIVITY TABLES (SI14a, SI14b, SI14c)
# -----------------------------------------------------------------------------

def calculate_sensitivity_stats(df):
    """Calculates N and Unsuccessful Outcome rates for sensitivity analysis."""
    # Filter for MITT
    df_mitt = df[df["MITT"] == 1].copy()
    
    stats = {}
    groups = ["Control Group", "Keheala Group", "SBCC Group", "SMS Reminder Group"]
    
    for g in groups:
        sub = df_mitt[df_mitt["treatment_group"] == g]
        n_mitt = len(sub)
        sub_valid = sub.dropna(subset=["unsuccessful_outcome"])
        n_valid = len(sub_valid)
        count = sub_valid["unsuccessful_outcome"].sum()
        prop = sub_valid["unsuccessful_outcome"].mean()
        
        stats[g] = {
            "N_valid": n_valid,
            "prop": prop
        }
    return stats

def generate_sensitivity_table(label, filename, n_control, n_treatment, p_control, p_treatment, benchmarks=None):
    """Generates a sensitivity analysis table."""
    print(f"\n--- Generating {label} ({filename}) ---")
    
    base_diff = p_control - p_treatment
    
    alphas = [0, 0.02, 0.04, 0.06, 0.08, 0.10]
    betas = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    
    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{c|*{6}{c}}")
    latex_lines.append(r"\hline\hline \\[-8pt]")
    latex_lines.append(r"Type I & \multicolumn{6}{c}{Type II Error Rate (\%)} \\")
    cols_str = " & ".join([f"{int(b*100)}" for b in betas])
    latex_lines.append(fr"Error Rate (\%)   & {cols_str} \\ \hline")
    
    for alpha in alphas:
        row_line1 = [f"{int(alpha*100)}"]
        row_line2 = [""]
        for beta in betas:
            # Formula: Diff_attenuated = Diff_obs * (1 - alpha - beta)
            diff_corr = base_diff * (1 - alpha - beta)
            factor = (1 - alpha - beta)
            p_c_new = p_control * factor
            p_t_new = p_treatment * factor 
            count_c = int(p_c_new * n_control)
            count_t = int(p_t_new * n_treatment)
            stat, pval = proportions_ztest([count_c, count_t], [n_control, n_treatment], alternative='two-sided')
            
            val_display = diff_corr * 100
            row_line1.append(f"{val_display:.2f}") 
            p_str = "(p<.0001)" if pval < 0.0001 else f"(p={pval:.4f})"
            row_line2.append(p_str)

        latex_lines.append(" & ".join(row_line1) + r" \\")
        latex_lines.append(" & ".join(row_line2) + r" \\")
        if alpha != alphas[-1]:
            latex_lines.append(r"[1em]")
            
    latex_lines.append(r"\hline \hline")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"}")
    
    out_file = os.path.join(OUTPUT_DIR, filename)
    with open(out_file, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved {out_file}")

# -----------------------------------------------------------------------------
# ANALYSIS 2: DQA CROSSTAB (tblSI_DQAcrosstab.tex)
# -----------------------------------------------------------------------------

def generate_crosstab_table(main_df, dqa_df):
    """Generates the DQA Crosstab."""
    print("\n--- Generating DQA Crosstab (tblSI_DQAcrosstab.tex) ---")
    
    if "scrn" not in main_df.columns:
        print("Error: 'scrn' missing from main data")
        return

    # Merge
    merged = pd.merge(main_df, dqa_df[["scrn", "to_paper"]], on="scrn", how="inner")
    print(f"Matched Records (Crosstab): {len(merged)}")
    
    # Recode NaNs to "Blank"
    merged["to_tibu_clean"] = merged["treatmentoutcome"].fillna("Blank")
    merged["to_paper_clean"] = merged["to_paper"].fillna("Blank")
    merged.loc[merged["to_paper_clean"] == "", "to_paper_clean"] = "Blank"
    merged.loc[merged["to_tibu_clean"] == "", "to_tibu_clean"] = "Blank"
    
    ct = pd.crosstab(
        merged["to_tibu_clean"],
        merged["to_paper_clean"],
        margins=True,
        margins_name="Total"
    )

    # Verify zero cells
    print("\nZero cells in crosstab (TIBU row x Paper col):")
    for r in ct.index:
        if r == "Total": continue
        for c in ct.columns:
            if c == "Total": continue
            if ct.loc[r, c] == 0:
                print(f"  TIBU={r}, Paper={c}: 0")

    good_t = {"C", "TC"}
    bad_t  = {"D", "F", "LTFU", "NC", "Blank"}
    good_p = {"C", "TC"}
    bad_p  = {"D", "F", "LTFU"}

    col_order = ["Blank", "C", "D", "F", "LTFU", "TC", "Total"]
    row_order = ["Blank", "C", "D", "F", "LTFU", "N/A", "NC", "TC", "TO", "Total"]

    data_cols = [c for c in col_order if c != "Total"]  # 6 cols, each split into N + (%)
    n_data_cols = len(data_cols)  # 6
    total_cols = 1 + n_data_cols * 2 + 1  # row label + 6*2 + Total = 14

    col_spec = "l|" + "|".join(["rr"] * n_data_cols) + "|r"

    latex_lines = []
    latex_lines.append(r"\newsavebox{\DQACrosstabBox}")
    latex_lines.append(r"\savebox{\DQACrosstabBox}{\scriptsize{")
    latex_lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append(r"\hline \hline \\[-8pt]")
    latex_lines.append(
        rf"Outcome in TIBU & \multicolumn{{{n_data_cols * 2}}}{{c}}{{Treatment Outcome in Paper Registry}} & Total\\"
    )
    col_headers = " & ".join(
        rf"\multicolumn{{2}}{{c|}}{{{c}}}" for c in data_cols
    )
    latex_lines.append(rf"& {col_headers} & \\ ")
    sub_headers = " & ".join([r"$N$ & (\%)"] * n_data_cols)
    latex_lines.append(rf"& {sub_headers} & \\ \hline \\[-8pt]")

    for row_idx in row_order:
        if row_idx not in ct.index: continue
        if row_idx == "Total":
            latex_lines.append(r"\hline \\[-8pt]")
        row_data = ct.loc[row_idx]
        cells = []
        row_total = int(row_data.get("Total", 0))
        for col in col_order:
            raw = int(row_data.get(col, 0))
            if col == "Total":
                cells.append(f"{raw:,}")
            elif row_idx == "Total":
                cells.append(f"{raw:,}")
                cells.append("")
            else:
                pct = (raw / row_total * 100) if row_total > 0 else 0
                n_str = f"{raw:,}"
                p_str = rf"({pct:.1f}\%)"
                if row_idx == "Blank" and col != "Blank":
                    color = "blue"
                elif row_idx in good_t and col in bad_p:
                    color = "red"
                elif row_idx in bad_t and col in good_p:
                    color = "green!60!black"
                else:
                    color = None
                if color:
                    n_str = rf"\textcolor{{{color}}}{{{n_str}}}"
                    p_str = rf"\textcolor{{{color}}}{{{p_str}}}"
                cells.append(n_str)
                cells.append(p_str)
        latex_lines.append(f"{row_idx} & " + " & ".join(cells) + r" \\")

    latex_lines.append(r"\hline \hline")
    latex_lines.append(r"\end{tabular}}}")
    latex_lines.append(r"\begin{minipage}{\wd\DQACrosstabBox}")
    latex_lines.append(r"\usebox{\DQACrosstabBox}")
    latex_lines.append(r"\par\vspace{4pt}\noindent")
    latex_lines.append(
        r"\textit{Note:} Percentages are row shares. \\"
        r"\textcolor{red}{Red} = Type I error (TIBU: success; paper: failure). \\"
        r"\textcolor{green!60!black}{Green} = Type II error (TIBU: failure; paper: success). \\"
        r"\textcolor{blue}{Blue} = Missing in TIBU (TIBU: no outcome; paper: outcome present)."
    )
    latex_lines.append(r"\end{minipage}")
    
    out_file = os.path.join(OUTPUT_DIR, "tblSI_DQAcrosstab.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved {out_file}")

# -----------------------------------------------------------------------------
# ANALYSIS 3: TYPE 1 & 2 ERRORS (tblSI_DQAtype12error.tex)
# -----------------------------------------------------------------------------

def generate_error_table(main_df, dqa_df):
    """Generates the Type 1 & 2 Error Rate table."""
    print("\n--- Generating DQA Error Table (tblSI_DQAtype12error.tex) ---")
    
    merged = pd.merge(main_df, dqa_df[["scrn", "to_paper"]], on="scrn", how="inner")
    
    # Define TIBU Outcomes (0=Success, 1=Unsuccess)
    merged["uo_tibu"] = np.nan
    bad_tibu = ["D", "F", "LTFU"]  # NC is a placeholder, treated as blank elsewhere
    good_tibu = ["C", "TC"]
    merged.loc[merged["treatmentoutcome"].isin(bad_tibu), "uo_tibu"] = 1
    merged.loc[merged["treatmentoutcome"].isin(good_tibu), "uo_tibu"] = 0
    
    # Define Paper Outcomes (0=Success, 1=Unsuccess)
    merged["uo_paper"] = np.nan
    # Note: Cleaned values are standard (C, TC, D, F, LTFU)
    merged.loc[merged["to_paper"].isin(["D", "F", "LTFU", "NC"]), "uo_paper"] = 1
    merged.loc[merged["to_paper"].isin(["C", "TC"]), "uo_paper"] = 0
    
    # Filter for valid TIBU outcomes
    df_valid = merged.dropna(subset=["uo_tibu"]).copy()
    
    # Mismatch Logic: Mismatch if disagreement OR Paper is Missing
    df_valid["mismatch"] = 0
    mask_disagree = (df_valid["uo_paper"].notna()) & (df_valid["uo_tibu"] != df_valid["uo_paper"])
    df_valid.loc[mask_disagree, "mismatch"] = 1
    df_valid.loc[df_valid["uo_paper"].isna(), "mismatch"] = 1
    
    # Filter: Exclude NC cases to match benchmark definition
    df_final = df_valid[df_valid["treatmentoutcome"] != "NC"].copy()
    
    group_map = {
        "Control Group": "Control",
        "SMS Reminder Group": "SMS",
        "SBCC Group": "Platform",
        "Keheala Group": "Keheala"
    }
    df_final["group_label"] = df_final["treatment_group"].map(group_map)
    groups = ["Control", "SMS", "Platform", "Keheala"]

    def error_rates(sub):
        if len(sub) == 0: return 0, 0, 0, 0
        n = len(sub)
        fn = ((sub["uo_tibu"] == 0) & (sub["uo_paper"] == 1)).sum()
        fp = ((sub["uo_tibu"] == 1) & (sub["uo_paper"] == 0)).sum()
        fm = sub["uo_paper"].isna().sum()
        tot = sub["mismatch"].sum()
        return fn/n*100, fp/n*100, fm/n*100, tot/n*100

    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{lcccc}")
    latex_lines.append(r"\hline \hline \\[-8pt]")
    latex_lines.append(
        r"& \cellcolor{red!20} False positive (\%)"
        r"& \cellcolor{green!20} False negative (\%)"
        r"& \cellcolor{blue!20} False missing (\%)"
        r"& Total (\%) \\"
    )
    latex_lines.append(r"\hline \\[-8pt]")

    for g in groups:
        sub = df_final[df_final["group_label"] == g]
        if sub.empty: continue
        fn, fp, fm, tot = error_rates(sub)
        latex_lines.append(
            rf"{g} & \cellcolor{{red!20}}{fn:.1f} & \cellcolor{{green!20}}{fp:.1f} & \cellcolor{{blue!20}}{fm:.1f} & {tot:.1f} \\"
        )

    latex_lines.append(r"\hline \\[-8pt]")

    # Total Row
    sub_all = df_final[df_final["treatment_group"].isin(group_map.keys())]
    fn_all, fp_all, fm_all, tot_all = error_rates(sub_all)
    latex_lines.append(
        rf"Total & \cellcolor{{red!20}}{fn_all:.1f} & \cellcolor{{green!20}}{fp_all:.1f} & \cellcolor{{blue!20}}{fm_all:.1f} & {tot_all:.1f} \\"
    )
    latex_lines.append(r"\hline \hline")
    latex_lines.append(r"\multicolumn{5}{l}{\scriptsize{\textit{Color coding:}}} \\[-4pt]")
    latex_lines.append(rf"\cellcolor{{red!20}} & \multicolumn{{4}}{{l}}{{\scriptsize{{False positive / Type I (TIBU: success; paper: fail): {fn_all:.1f}\%}}}} \\[-4pt]")
    latex_lines.append(rf"\cellcolor{{green!20}} & \multicolumn{{4}}{{l}}{{\scriptsize{{False negative / Type II (TIBU: fail; paper: success): {fp_all:.1f}\%}}}} \\[-4pt]")
    latex_lines.append(rf"\cellcolor{{blue!20}} & \multicolumn{{4}}{{l}}{{\scriptsize{{False missing (paper absent): {fm_all:.1f}\%}}}} \\")
    latex_lines.append(r"\end{tabular}}")

    out_file = os.path.join(OUTPUT_DIR, "tblSI_DQAtype12error.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved {out_file}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def generate_patient_characteristics_table(main_df, dqa_df):
    """
    Table 1a: Patient and disease characteristics.
    Table 1b: Outcomes.
    Outputs tblSI_patient_characteristics.tex and tblSI_outcomes.tex
    """
    print("\n--- Generating Patient Characteristics Tables (1a, 1b) ---")

    # --- Column 1: All TIBU during study period ---
    tibu_full = pd.read_csv(TIBU_DEIDENTIFIED)
    tibu_full["treatmentoutcome"] = tibu_full["treatmentoutcome"].replace("MT4", "TO")

    tibu_full["male_tibu"] = tibu_full["sexmf"].isin(["M", "Male"]).astype(float)

    def parse_age_years(val):
        if pd.isna(val): return np.nan
        val = str(val).strip()
        if val.endswith("Y"):
            try: return float(val[:-1])
            except: return np.nan
        return np.nan

    tibu_full["age_years"] = tibu_full["ageonregistration"].apply(parse_age_years)
    tibu_full["hiv_pos_tibu"] = tibu_full["hivstatus"].eq("Pos").astype(float)
    tibu_full["extrapulmonary_tibu"] = tibu_full["typeoftbpep"].eq("EP").astype(float)
    tibu_full["retreatment_tibu"] = tibu_full["typeofpatient"].isin(["R", "TLF"]).astype(float)
    tibu_full["drugresistant_tibu"] = tibu_full["resistancepattern"].notna().astype(float)
    smear_pos = tibu_full["sputumsmearexamination0thmon"].str.strip() == "Pos"
    gx_pos = tibu_full["genexpert"].str.strip().str.startswith("MTB detected")
    tibu_full["bact_confirmed_tibu"] = (smear_pos | gx_pos).astype(float)
    bad_outcomes = ["D", "F", "LTFU"]
    tibu_full["unsuccessful_tibu"] = tibu_full["treatmentoutcome"].isin(bad_outcomes).astype(float)

    # --- Column 2: Study clinics, TIBU (MITT sample) ---
    df_study = main_df[main_df["MITT"] == 1].copy()

    # --- Column 3: Study clinics, paper records (DQA-matched patients' TIBU characteristics) ---
    df_paper = pd.merge(main_df, dqa_df[["scrn"]], on="scrn", how="inner")

    N_tibu  = len(tibu_full)
    N_study = len(df_study)
    N_paper = len(df_paper)

    def pct_tibu(series):
        v = pd.to_numeric(series, errors="coerce")
        return f"{v.mean()*100:.1f}"

    def pct(series):
        v = pd.to_numeric(series, errors="coerce")
        return f"{v.mean()*100:.1f}"

    def npct_tibu(series):
        v = pd.to_numeric(series, errors="coerce")
        n = int(v.sum())
        p = v.mean() * 100
        return rf"{n:,} & ({p:.1f}\%)"

    def npct(series):
        v = pd.to_numeric(series, errors="coerce")
        n = int(v.sum())
        p = v.mean() * 100
        return rf"{n:,} & ({p:.1f}\%)"

    def mean_val(series):
        v = pd.to_numeric(series, errors="coerce")
        return f"{v.mean():.1f}"

    def header_lines():
        return [
            r"Characteristic"
            r" & \multicolumn{4}{c}{Digital}"
            r" & \multicolumn{2}{c}{Paper} \\",
            rf" & \multicolumn{{2}}{{c}}{{\shortstack{{All \\ (N={N_tibu:,})}}}}"
            rf" & \multicolumn{{2}}{{c}}{{\shortstack{{Study clinics \\ (N={N_study:,})}}}}"
            rf" & \multicolumn{{2}}{{c}}{{\shortstack{{Study clinics \\ (N={N_paper:,})}}}} \\",
            r" & $N$ & (\%) & $N$ & (\%) & $N$ & (\%) \\ \hline \\[-8pt]",
        ]

    # --- Table 1a: Patient and disease characteristics ---
    lines_1a = []
    lines_1a.append(r"\scriptsize{")
    lines_1a.append(r"\begin{tabular}{lrr|rr|rr}")
    lines_1a.append(r"\hline\hline \\[-8pt]")
    lines_1a.extend(header_lines())

    lines_1a.append(r"\textit{Individual characteristics} & & & & & & \\")
    lines_1a.append(
        rf"\quad Male & {npct_tibu(tibu_full['male_tibu'])} & {npct(df_study['male'])} & {npct(df_paper['male'])} \\"
    )
    lines_1a.append(
        rf"\quad Mean age (years) & {mean_val(tibu_full['age_years'])} & & {mean_val(df_study['age_in_years'])} & & {mean_val(df_paper['age_in_years'])} & \\"
    )
    lines_1a.append(r"\rule{0pt}{14pt}\textit{Disease characteristics} & & & & & & \\")
    lines_1a.append(
        rf"\quad Bacteriologically confirmed & {npct_tibu(tibu_full['bact_confirmed_tibu'])} & {npct(df_study['bacteriologically_confirmed'])} & {npct(df_paper['bacteriologically_confirmed'])} \\"
    )
    lines_1a.append(
        rf"\quad Drug resistant & {npct_tibu(tibu_full['drugresistant_tibu'])} & {npct(df_study['drugresistant'])} & {npct(df_paper['drugresistant'])} \\"
    )
    lines_1a.append(
        rf"\quad Extrapulmonary TB & {npct_tibu(tibu_full['extrapulmonary_tibu'])} & {npct(df_study['extrapulmonary'])} & {npct(df_paper['extrapulmonary'])} \\"
    )
    lines_1a.append(
        rf"\quad HIV coinfection & {npct_tibu(tibu_full['hiv_pos_tibu'])} & {npct(df_study['hiv_positive'])} & {npct(df_paper['hiv_positive'])} \\"
    )
    lines_1a.append(
        rf"\quad Retreatment & {npct_tibu(tibu_full['retreatment_tibu'])} & {npct(df_study['retreatment'])} & {npct(df_paper['retreatment'])} \\"
    )

    lines_1a.append(r"\hline\hline")
    lines_1a.append(r"\end{tabular}}")

    out_1a = os.path.join(OUTPUT_DIR, "tblSI_patient_characteristics.tex")
    with open(out_1a, "w") as f:
        f.write("\n".join(lines_1a))
    print(f"Saved {out_1a}")

    # --- Table 1b: Outcomes ---
    lines_1b = []
    lines_1b.append(r"\scriptsize{")
    lines_1b.append(r"\begin{tabular}{lrr|rr|rr}")
    lines_1b.append(r"\hline\hline \\[-8pt]")
    lines_1b.extend(header_lines())

    lines_1b.append(r"\textit{Outcomes} & & & & & & \\")
    for code, label in [
        ("C",    "Cured"),
        ("TC",   "Treatment completed"),
        ("D",    "Died"),
        ("F",    "Treatment failed"),
        ("LTFU", "Lost to follow-up"),
        ("NC",   "Not completed"),
        ("TO",   "Transfer out"),
        ("NTB",  "Not TB"),
        (None,   "Missing"),
    ]:
        if code is None:
            t_n = tibu_full['treatmentoutcome'].isna().sum()
            t_pct = tibu_full['treatmentoutcome'].isna().mean()*100
            s_n = df_study['treatmentoutcome'].isna().sum()
            s_pct = df_study['treatmentoutcome'].isna().mean()*100
            p_n = df_paper['treatmentoutcome'].isna().sum()
            p_pct = df_paper['treatmentoutcome'].isna().mean()*100
        else:
            t_n = (tibu_full['treatmentoutcome'] == code).sum()
            t_pct = (tibu_full['treatmentoutcome'] == code).mean()*100
            s_n = (df_study['treatmentoutcome'] == code).sum()
            s_pct = (df_study['treatmentoutcome'] == code).mean()*100
            p_n = (df_paper['treatmentoutcome'] == code).sum()
            p_pct = (df_paper['treatmentoutcome'] == code).mean()*100
        t = rf"{t_n:,} & ({t_pct:.1f}\%)"
        s = rf"{s_n:,} & ({s_pct:.1f}\%)"
        p = rf"{p_n:,} & ({p_pct:.1f}\%)"
        lines_1b.append(rf"\quad {label} & {t} & {s} & {p} \\")

    lines_1b.append(r"\hline\hline")
    lines_1b.append(r"\end{tabular}}")

    out_1b = os.path.join(OUTPUT_DIR, "tblSI_outcomes.tex")
    with open(out_1b, "w") as f:
        f.write("\n".join(lines_1b))
    print(f"Saved {out_1b}")

def generate_outcome_corrections(main_df, dqa_df):
    """
    Fig 2: Most common outcome corrections (TIBU -> Paper).
    Saves both outcome_corrections.csv and tblSI_outcome_corrections.tex
    """
    print("\n--- Generating Outcome Corrections (Fig 2) ---")

    merged = pd.merge(main_df, dqa_df[["scrn", "to_paper"]], on="scrn", how="inner")
    df = merged.copy()
    df["tibu"] = df["treatmentoutcome"].fillna("Blank")
    df["paper"] = df["to_paper"].fillna("Blank")

    corrections = df[df["tibu"] != df["paper"]]
    correction_counts = (
        corrections.groupby(["tibu", "paper"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # Save CSV only (tex replaced by bolding in crosstab)
    correction_counts.to_csv(os.path.join(OUTPUT_DIR, "outcome_corrections.csv"), index=False)
    print(f"Saved outcome_corrections.csv")


def generate_error_by_clinic(main_df, dqa_df):
    """
    Fig 3: Where are errors most severe?
    Saves error_rates_by_clinic.csv, error_rates_by_clinic_n10.csv,
    and tblSI_error_by_clinic_char.tex
    """
    print("\n--- Generating Error by Clinic (Fig 3) ---")

    clinic_summary = pd.read_csv(os.path.join(DEIDENTIFIED_DIR, "clinic_summary_deidentified.csv"))
    clinic_summary["clinic_id"] = pd.to_numeric(clinic_summary["clinic_id"], errors="coerce")

    df = _build_error_df(main_df, dqa_df)
    df["clinic_id_num"] = pd.to_numeric(df["clinic_id"], errors="coerce")
    df = df.merge(
        clinic_summary[["clinic_id", "urban", "tibu_patients", "province"]],
        left_on="clinic_id_num", right_on="clinic_id", how="left"
    )
    df["mismatch"] = (df["false_neg"] | df["false_pos"] | df["false_miss"]).astype(int)

    # --- CSV: per-clinic stats ---
    clinic_n_tibu = main_df.groupby("clinic_id")["scrn"].count().reset_index(name="n_tibu")
    clinic_stats = (
        df.groupby("clinic_id_num")
        .agg(n_study=("scrn","count"), mismatch_rate=("mismatch","mean"),
             false_neg_rate=("false_neg","mean"), false_pos_rate=("false_pos","mean"),
             false_miss_rate=("false_miss","mean"))
        .reset_index()
        .rename(columns={"clinic_id_num": "clinic_id"})
    )
    clinic_stats = clinic_stats.merge(clinic_n_tibu, on="clinic_id", how="left")
    clinic_stats = clinic_stats[["clinic_id","n_tibu","n_study","mismatch_rate","false_neg_rate","false_pos_rate","false_miss_rate"]]

    clinic_stats.to_csv(os.path.join(OUTPUT_DIR, "error_rates_by_clinic.csv"), index=False)
    filtered = clinic_stats[clinic_stats["n_study"] >= 10].sort_values("mismatch_rate", ascending=False)
    filtered.to_csv(os.path.join(OUTPUT_DIR, "error_rates_by_clinic_n10.csv"), index=False)

    total_patients = clinic_stats["n_study"].sum()
    filtered_patients = filtered["n_study"].sum()
    print(f"Clinics total: {clinic_stats.shape[0]}, with n>=10: {filtered.shape[0]}")
    print(f"Matched patients: {total_patients}, in n>=10 clinics: {filtered_patients} ({filtered_patients/total_patients:.1%})")
    print("\nTop 10 clinics by mismatch rate (n>=10):")
    print(filtered.head(10))

    # --- Tex: by urban/rural, province, and clinic size ---
    median_size = df["tibu_patients"].median()
    df["large_clinic"] = (df["tibu_patients"] >= median_size).astype(float)

    total_n = len(df)

    def stats(sub):
        n = len(sub)
        n_pct = n / total_n * 100 if total_n else 0
        # `false_neg` and `false_pos` in _build_error_df are semantically swapped.
        # Rebind so fp_* = false positive (Type I) and fn_* = false negative (Type II).
        fp_n, fn_n, fm_n = int(sub["false_neg"].sum()), int(sub["false_pos"].sum()), int(sub["false_miss"].sum())
        fp_p, fn_p, fm_p = sub["false_neg"].mean()*100, sub["false_pos"].mean()*100, sub["false_miss"].mean()*100
        return n, n_pct, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p

    def fmt(count, pct):
        return rf"{count:,} ({pct:.1f}\%)"

    latex_lines = []
    latex_lines.append(r"\newsavebox{\ClinicErrBox}")
    latex_lines.append(r"\savebox{\ClinicErrBox}{\scriptsize{")
    latex_lines.append(r"\begin{tabular}{l|rr|rr|rr|rr}")
    latex_lines.append(r"\hline\hline \\[-8pt]")
    latex_lines.append(
        r"& \multicolumn{2}{c|}{}"
        r" & \multicolumn{2}{c|}{False positive}"
        r" & \multicolumn{2}{c|}{False negative}"
        r" & \multicolumn{2}{c}{Missing in TIBU} \\"
    )
    latex_lines.append(r"& $N$ & (\%) & $N$ & (\%) & $N$ & (\%) & $N$ & (\%) \\")
    latex_lines.append(r"\hline \\[-8pt]")

    # By province
    latex_lines.append(r"\textit{By province} & & & & & & & & \\")
    for prov in sorted(df["province"].dropna().unique()):
        n, n_pct, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df[df["province"] == prov])
        latex_lines.append(rf"\quad {prov} & {n:,} & ({n_pct:.1f}\%) & {fp_n:,} & ({fp_p:.1f}\%) & {fn_n:,} & ({fn_p:.1f}\%) & {fm_n:,} & ({fm_p:.1f}\%) \\")

    # By urban/rural
    latex_lines.append(r"\rule{0pt}{14pt}\textit{By urban/rural} & & & & & & & & \\")
    for label, mask in [("Urban", df["urban"] == 1), ("Rural", df["urban"] == 0)]:
        n, n_pct, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df[mask])
        latex_lines.append(rf"\quad {label} & {n:,} & ({n_pct:.1f}\%) & {fp_n:,} & ({fp_p:.1f}\%) & {fn_n:,} & ({fn_p:.1f}\%) & {fm_n:,} & ({fm_p:.1f}\%) \\")

    # By clinic size
    latex_lines.append(r"\rule{0pt}{14pt}\textit{By clinic size} & & & & & & & & \\")
    for label, mask in [("Large clinic", df["large_clinic"] == 1), ("Small clinic", df["large_clinic"] == 0)]:
        n, n_pct, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df[mask])
        latex_lines.append(rf"\quad {label} & {n:,} & ({n_pct:.1f}\%) & {fp_n:,} & ({fp_p:.1f}\%) & {fn_n:,} & ({fn_p:.1f}\%) & {fm_n:,} & ({fm_p:.1f}\%) \\")

    # All
    n, n_pct, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df)
    latex_lines.append(rf"\rule{{0pt}}{{14pt}}All & {n:,} & ({n_pct:.1f}\%) & {fp_n:,} & ({fp_p:.1f}\%) & {fn_n:,} & ({fn_p:.1f}\%) & {fm_n:,} & ({fm_p:.1f}\%) \\")

    latex_lines.append(r"\hline\hline")
    latex_lines.append(r"\end{tabular}}}")
    latex_lines.append(r"\begin{minipage}{\wd\ClinicErrBox}")
    latex_lines.append(r"\usebox{\ClinicErrBox}")
    latex_lines.append(r"\par\vspace{4pt}\noindent")
    latex_lines.append(
        rf"\scriptsize{{\textit{{Note:}} Urban/rural classification is based on clinic records. "
        rf"Large clinics are those with at least {int(median_size):,} patients registered in TIBU (median); "
        rf"small clinics are those with fewer.}}"
    )
    latex_lines.append(r"\end{minipage}")

    with open(os.path.join(OUTPUT_DIR, "tblSI_error_by_clinic_char.tex"), "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved error_rates_by_clinic.csv, error_rates_by_clinic_n10.csv, tblSI_error_by_clinic_char.tex")

def generate_clinic_characteristics_table(main_df, dqa_df):
    """
    Simple summary table of N, mismatch rate, Type I and Type II error rates
    broken down by All / Urban / Rural.
    Saves tblSI_clinic_characteristics.tex
    """
    print("\n--- Generating Clinic Characteristics Table ---")

    clinic_summary = pd.read_csv(os.path.join(DEIDENTIFIED_DIR, "clinic_summary_deidentified.csv"))
    clinic_summary["clinic_id"] = pd.to_numeric(clinic_summary["clinic_id"], errors="coerce")

    df = _build_error_df(main_df, dqa_df)
    df["clinic_id_num"] = pd.to_numeric(df["clinic_id"], errors="coerce")
    df = df.merge(
        clinic_summary[["clinic_id", "urban"]],
        left_on="clinic_id_num", right_on="clinic_id", how="left"
    )
    df["mismatch"] = (df["false_neg"] | df["false_pos"] | df["false_miss"]).astype(int)

    def row_stats(sub):
        n = len(sub)
        mismatch = sub["mismatch"].mean() * 100
        type1    = sub["false_neg"].mean() * 100
        type2    = sub["false_pos"].mean() * 100
        return n, mismatch, type1, type2

    all_s   = row_stats(df)
    urban_s = row_stats(df[df["urban"] == 1])
    rural_s = row_stats(df[df["urban"] == 0])

    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{lcccc}")
    latex_lines.append(r"\hline\hline \\[-8pt]")
    latex_lines.append(r"\rowcolor{yellow!15} & All & Urban & Rural \\")
    latex_lines.append(r"\hline \\[-8pt]")
    latex_lines.append(rf"N & {all_s[0]:,} & {urban_s[0]:,} & {rural_s[0]:,} \\")
    latex_lines.append(rf"Mismatch rate & {all_s[1]:.1f}\% & {urban_s[1]:.1f}\% & {rural_s[1]:.1f}\% \\")
    latex_lines.append(rf"Type I error rate & {all_s[2]:.1f}\% & {urban_s[2]:.1f}\% & {rural_s[2]:.1f}\% \\")
    latex_lines.append(rf"Type II error rate & {all_s[3]:.1f}\% & {urban_s[3]:.1f}\% & {rural_s[3]:.1f}\% \\")
    latex_lines.append(r"\hline\hline")
    latex_lines.append(r"\end{tabular}}")

    out_file = os.path.join(OUTPUT_DIR, "tblSI_clinic_characteristics.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved {out_file}")


def generate_error_by_outcome(main_df, dqa_df):
    """
    Computes mismatch rates by TIBU outcome.
    Saves error_rates_by_outcome.csv
    """
    print("\n--- Generating error rates by outcome ---")

    merged = pd.merge(main_df, dqa_df[["scrn", "to_paper"]], on="scrn", how="inner")

    df = merged.copy()
    df["tibu"] = df["treatmentoutcome"].fillna("Blank")
    df["paper"] = df["to_paper"].fillna("Blank")
    df["mismatch"] = (df["tibu"] != df["paper"]).astype(int)

    outcome_stats = (
        df.groupby("tibu")
        .agg(n=("scrn", "count"), mismatch_rate=("mismatch", "mean"))
        .reset_index()
        .sort_values("mismatch_rate", ascending=False)
    )

    out_file = os.path.join(OUTPUT_DIR, "error_rates_by_outcome.csv")
    outcome_stats.to_csv(out_file, index=False)

    print(f"Saved {out_file}")
    print("\nError rates by outcome:")
    print(outcome_stats)

def generate_error_by_patient_characteristics(main_df, dqa_df):
    """
    Fig 4: For whom are errors most severe?
    Error rates by patient and disease characteristics.
    Outputs tblSI_error_by_patient_char.tex
    """
    print("\n--- Generating Error by Patient Characteristics (Fig 4) ---")

    df = _build_error_df(main_df, dqa_df)
    df["mismatch"] = (df["false_neg"] | df["false_pos"] | df["false_miss"]).astype(int)

    df["age_group"] = pd.cut(
        pd.to_numeric(df["age_in_years"], errors="coerce"),
        bins=[0, 15, 35, 55, 200],
        labels=[r"$<$15", "15--34", "35--54", "55+"]
    )

    total_n = len(df)

    def stats(sub):
        if len(sub) == 0:
            return 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0
        n = len(sub)
        n_pct = n / total_n * 100 if total_n else 0
        # `false_neg` and `false_pos` in _build_error_df are semantically swapped.
        # Rebind so fp_* = false positive (Type I) and fn_* = false negative (Type II).
        fp_n = int(sub["false_neg"].sum())
        fp_p = sub["false_neg"].mean() * 100
        fn_n = int(sub["false_pos"].sum())
        fn_p = sub["false_pos"].mean() * 100
        fm_n = int(sub["false_miss"].sum())
        fm_p = sub["false_miss"].mean() * 100
        return n, n_pct, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p

    def fmt(count, pct):
        return rf"{count:,} ({pct:.1f}\%)"

    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{l|rr|rr|rr|rr}")
    latex_lines.append(r"\hline\hline \\[-8pt]")
    latex_lines.append(
        r"& \multicolumn{2}{c|}{}"
        r" & \multicolumn{2}{c|}{False positive}"
        r" & \multicolumn{2}{c|}{False negative}"
        r" & \multicolumn{2}{c}{Missing in TIBU} \\"
    )
    latex_lines.append(r"& $N$ & (\%) & $N$ & (\%) & $N$ & (\%) & $N$ & (\%) \\")
    latex_lines.append(r"\hline \\[-8pt]")

    # Individual characteristics
    latex_lines.append(r"\textit{Individual characteristics} & & & & & & & & \\")
    n, n_pct, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df[df["male"] == 1])
    latex_lines.append(rf"\quad Male (\%) & {n:,} & ({n_pct:.1f}\%) & {fp_n:,} & ({fp_p:.1f}\%) & {fn_n:,} & ({fn_p:.1f}\%) & {fm_n:,} & ({fm_p:.1f}\%) \\")

    # Age
    latex_lines.append(r"\rule{0pt}{14pt}\textit{Age group} & & & & & & & & \\")
    for grp in [r"$<$15", "15--34", "35--54", "55+"]:
        n, n_pct, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df[df["age_group"] == grp])
        latex_lines.append(rf"\quad {grp} & {n:,} & ({n_pct:.1f}\%) & {fp_n:,} & ({fp_p:.1f}\%) & {fn_n:,} & ({fn_p:.1f}\%) & {fm_n:,} & ({fm_p:.1f}\%) \\")

    # Disease characteristics
    latex_lines.append(r"\rule{0pt}{14pt}\textit{Disease characteristics} & & & & & & & & \\")
    for col, label in [
        ("bacteriologically_confirmed", r"\quad Bacteriologically confirmed"),
        ("extrapulmonary",              r"\quad Extrapulmonary"),
        ("retreatment",                 r"\quad Retreatment"),
    ]:
        n, n_pct, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df[df[col] == 1])
        latex_lines.append(rf"{label} & {n:,} & ({n_pct:.1f}\%) & {fp_n:,} & ({fp_p:.1f}\%) & {fn_n:,} & ({fn_p:.1f}\%) & {fm_n:,} & ({fm_p:.1f}\%) \\")
    n, n_pct, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df[df["hiv_positive"] == 1])
    latex_lines.append(rf"\quad HIV positive & {n:,} & ({n_pct:.1f}\%) & {fp_n:,} & ({fp_p:.1f}\%) & {fn_n:,} & ({fn_p:.1f}\%) & {fm_n:,} & ({fm_p:.1f}\%) \\")

    # All
    n, n_pct, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df)
    latex_lines.append(rf"\rule{{0pt}}{{14pt}}All & {n:,} & ({n_pct:.1f}\%) & {fp_n:,} & ({fp_p:.1f}\%) & {fn_n:,} & ({fn_p:.1f}\%) & {fm_n:,} & ({fm_p:.1f}\%) \\")

    latex_lines.append(r"\hline\hline")
    latex_lines.append(r"\end{tabular}}")

    out_file = os.path.join(OUTPUT_DIR, "tblSI_error_by_patient_char.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved {out_file}")

def _build_error_df(main_df, dqa_df):
    """
    Merge TIBU-side study records with the paper-registry audit, classify each
    record into one of four DQA categories, and attach relevant dates.

    Denominator: records for which paper has a classifiable outcome (the ground
    truth). Records where paper is blank are dropped as uncheckable; transfer-out
    (TO, which includes the recoded MT4) is dropped as not auditable.

    Variables (retained names from earlier code for downstream compatibility,
    though they are semantically swapped):
      false_neg   → Type I:  TIBU success, paper failure (digital false positive)
      false_pos   → Type II: TIBU failure, paper success (digital false negative)
      false_miss  → TIBU blank, paper has outcome       (digital missing)
    """
    merged = pd.merge(main_df, dqa_df[["scrn", "to_paper", "date_paper"]], on="scrn", how="inner")

    # Drop transfer-out (not auditable). MT4 is already recoded to TO on load.
    merged = merged[merged["treatmentoutcome"] != "TO"].copy()

    # NC ("not completed") is a placeholder/interim code, not a medical outcome,
    # so we treat it as missing rather than bad — records with TIBU = NC roll
    # into false_miss when paper has a classifiable outcome.
    bad  = ["D", "F", "LTFU"]
    good = ["C", "TC"]
    merged["uo_tibu"] = np.nan
    merged.loc[merged["treatmentoutcome"].isin(bad),  "uo_tibu"] = 1
    merged.loc[merged["treatmentoutcome"].isin(good), "uo_tibu"] = 0
    merged["uo_paper"] = np.nan
    merged.loc[merged["to_paper"].isin(bad),  "uo_paper"] = 1
    merged.loc[merged["to_paper"].isin(good), "uo_paper"] = 0

    # New denominator: paper must be classifiable (ground truth present).
    df = merged.dropna(subset=["uo_paper"]).copy()
    df["false_neg"]  = ((df["uo_tibu"] == 0) & (df["uo_paper"] == 1)).astype(int)  # Type I
    df["false_pos"]  = ((df["uo_tibu"] == 1) & (df["uo_paper"] == 0)).astype(int)  # Type II
    df["false_miss"] = df["uo_tibu"].isna().astype(int)                            # TIBU blank

    # Parse dates; censor sentinels and absurd values (study timeline is 2018-2021)
    df["tibu_date"]  = pd.to_datetime(df["treatmentoutcomedate_formatted"], errors="coerce")
    df["paper_date"] = pd.to_datetime(df["date_paper"], dayfirst=True, errors="coerce")
    df["reg_date"]   = pd.to_datetime(df["dateregistered_formatted"], errors="coerce")
    for col in ["tibu_date", "paper_date", "reg_date"]:
        bad_year = (df[col].dt.year < 2010) | (df[col].dt.year > 2025)
        df.loc[bad_year, col] = pd.NaT

    # outcome_date = TIBU's outcome date when available, else paper-registry
    # outcome date. The fallback matters for false-missing records (TIBU blank),
    # where the treatment outcome actually happened and paper dated it even
    # though TIBU never captured it.
    df["outcome_date"] = df["tibu_date"].fillna(df["paper_date"])

    # Single global TIBU snapshot. The study2_cleaned extract reflects a final
    # consolidated pull; max(tibu_date) = 2021-12-31 with outcomes clustering
    # in late Dec 2021, so we treat the snapshot as 1 Jan 2022.
    df["snapshot_date"] = pd.Timestamp("2022-01-01")

    # Two age axes, both measured against the snapshot:
    #   age_reg: days since registration  (record age at snapshot)
    #   age_out: days since outcome date  (reporting lag)
    df["age_reg"] = (df["snapshot_date"] - df["reg_date"]).dt.days
    df["age_out"] = (df["snapshot_date"] - df["outcome_date"]).dt.days
    for col in ["age_reg", "age_out"]:
        df.loc[df[col] < 0, col] = pd.NA
    return df


def _smoothed_curve_with_ci(age, y, frac=0.35, n_boot=500, n_eval=60, seed=0):
    """Lowess smooth of P(y=1 | age) with percentile bootstrap CI band."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    a = np.asarray(age, dtype=float)
    b = np.asarray(y,   dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 30:
        return None
    x_eval = np.linspace(np.quantile(a, 0.01), np.quantile(a, 0.99), n_eval)
    base = lowess(b, a, frac=frac, it=0, return_sorted=True)
    main_curve = np.interp(x_eval, base[:, 0], base[:, 1])
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, n_eval))
    for i in range(n_boot):
        idx = rng.integers(0, len(a), size=len(a))
        s = lowess(b[idx], a[idx], frac=frac, it=0, return_sorted=True)
        boots[i] = np.interp(x_eval, s[:, 0], s[:, 1])
    lo = np.percentile(boots, 2.5,  axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return x_eval, main_curve, lo, hi


def generate_error_by_age_figure(main_df, dqa_df):
    """
    Smoothed (lowess) curves of false-positive, false-negative, and false-missing
    rates against two "age of record" definitions, both measured from the TIBU
    download snapshot:
      age_reg: days since patient registration  (record age at snapshot)
      age_out: days since TIBU outcome date     (reporting lag)
    Download date is inferred per source_file as max(outcome_date) in that file.
    Shaded bands = percentile bootstrap 95% CI (500 resamples).
    Saves fig_error_by_age_reg.pdf and fig_error_by_age_out.pdf.
    """
    print("\n--- Generating error-by-age figures (smoothed) ---")
    df = _build_error_df(main_df, dqa_df)

    definitions = [
        ("age_reg", "fig_error_by_age_reg.pdf",
         "Record age at TIBU snapshot (days since registration)"),
        ("age_out", "fig_error_by_age_out.pdf",
         "Reporting lag (days from TIBU outcome date to snapshot)"),
    ]

    # NB: _build_error_df's variable names are semantically swapped — `false_neg`
    # actually stores Type I (TIBU false positive) events, and `false_pos` stores
    # Type II. We follow the crosstab (Table 2) convention: red = Type I.
    series = [
        ("false_neg", "red",   "False positive (Type I)"),
        ("false_pos", "green", "False negative (Type II)"),
        ("false_miss","blue",  "Missing in TIBU"),
    ]

    for col, fname, xlabel in definitions:
        sub = df.dropna(subset=[col]).copy()
        print(f"  {col}: N={len(sub):,} with valid age; "
              f"range {sub[col].min():.0f}-{sub[col].max():.0f} days")
        fig, ax = plt.subplots(figsize=(10, 4))
        for ecol, color, label in series:
            res = _smoothed_curve_with_ci(sub[col].values, sub[ecol].values)
            if res is None:
                continue
            x, mu, lo, hi = res
            ax.plot(x, mu * 100, color=color, linewidth=1.8, label=label)
            ax.fill_between(x, lo * 100, hi * 100, color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Error rate (%)")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
        ax.set_ylim(bottom=0)
        # age_out: invert so the x-axis reads calendar-forward (old outcomes on
        # the left, recent outcomes on the right), matching Figure 5.
        if col == "age_out":
            ax.invert_xaxis()
        ax.legend(framealpha=0.9, handlelength=1.0, labelspacing=0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        out_file = os.path.join(OUTPUT_DIR, fname)
        fig.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_file}")


def generate_error_density_figure(main_df, dqa_df):
    """
    ECDF comparison of reporting lag (age_out) for each error category vs. the
    no-error baseline. Asks: where on the age axis are errors over-represented?
    Saves fig_error_density.pdf.
    """
    print("\n--- Generating error density figure ---")
    df = _build_error_df(main_df, dqa_df).dropna(subset=["age_out"]).copy()
    # See note in generate_error_by_age_figure: false_neg stores Type I, false_pos stores Type II.
    df["cat"] = "No error"
    df.loc[df["false_neg"]  == 1, "cat"] = "False positive (Type I)"
    df.loc[df["false_pos"]  == 1, "cat"] = "False negative (Type II)"
    df.loc[df["false_miss"] == 1, "cat"] = "Missing in TIBU"

    cats = [
        ("No error",                 "gray"),
        ("False positive (Type I)",  "red"),
        ("False negative (Type II)", "green"),
        ("Missing in TIBU",          "blue"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    for cat, color in cats:
        vals = np.sort(df.loc[df["cat"] == cat, "age_out"].values)
        if len(vals) < 5:
            continue
        ys = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, ys, where="post", color=color, linewidth=1.5,
                label=f"{cat} (N={len(vals):,})")
    ax.set_xlabel("Reporting lag (days from TIBU outcome date to snapshot)")
    ax.set_ylabel("Cumulative proportion of records")
    ax.set_ylim(0, 1.02)
    ax.legend(framealpha=0.9, handlelength=1.2, labelspacing=0.5, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "fig_error_density.pdf")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")


def generate_error_over_time_figure(main_df, dqa_df):
    """
    Smoothed (lowess) curves of false-positive (Type I), false-negative (Type II),
    and false-missing rates against outcome date. Shaded bands = percentile
    bootstrap 95% CI (500 resamples). Saves fig_error_over_time.pdf.

    Outcome date is TIBU's outcome date when available, otherwise the paper
    registry's outcome date (used for false-missing records, where TIBU is blank).
    """
    print("\n--- Generating error-over-time figure (smoothed) ---")
    df = _build_error_df(main_df, dqa_df).dropna(subset=["outcome_date"]).copy()
    df["t_days"] = df["outcome_date"].map(lambda d: d.toordinal())

    # See note in generate_error_by_age_figure: false_neg stores Type I, false_pos Type II.
    series = [
        ("false_neg", "red",   "False positive (Type I)"),
        ("false_pos", "green", "False negative (Type II)"),
        ("false_miss","blue",  "Missing in TIBU"),
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    for ecol, color, label in series:
        res = _smoothed_curve_with_ci(df["t_days"].values, df[ecol].values, frac=0.35, n_boot=500)
        if res is None:
            continue
        x_ord, mu, lo, hi = res
        x_dates = pd.to_datetime([pd.Timestamp.fromordinal(int(x)) for x in x_ord])
        ax.plot(x_dates, mu * 100, color=color, linewidth=1.8, label=label)
        ax.fill_between(x_dates, lo * 100, hi * 100, color=color, alpha=0.15, linewidth=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.set_xlabel("Outcome date (TIBU if available, else paper)")
    ax.set_ylabel("Error rate (%)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.9, handlelength=1.0, labelspacing=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "fig_error_over_time.pdf")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")


def generate_correction_lag_table(main_df, dqa_df):
    """
    Table 5: Record age and reporting lag (days), stratified by error category.
      Record age    = days from patient registration to TIBU download snapshot
      Reporting lag = days from TIBU outcome date      to TIBU download snapshot
    Saves tblSI_correction_lag.tex
    """
    print("\n--- Generating record-age / reporting-lag table ---")
    df = _build_error_df(main_df, dqa_df)
    total_n = len(df)

    def summarize(mask):
        sub = df[mask]
        n = len(sub)
        n_pct = n / total_n * 100 if total_n else 0
        out = [n, n_pct]
        for col in ["age_reg", "age_out"]:
            v = sub[col].dropna()
            if len(v) == 0:
                out += ["--", "--", "--"]
            else:
                out += [len(v), f"{v.mean():.0f}", f"{v.median():.0f}"]
        return out  # [N, %, n_reg, mean_reg, med_reg, n_out, mean_out, med_out]

    rows = [
        (df["false_neg"]  == 1, "False positive (Type I)",  "red"),
        (df["false_pos"]  == 1, "False negative (Type II)", "green!60!black"),
        (df["false_miss"] == 1, "Missing in TIBU",           "blue"),
        ((df["false_neg"] == 0) & (df["false_pos"] == 0) & (df["false_miss"] == 0),
                                  "No error",                None),
    ]

    latex_lines = []
    latex_lines.append(r"\newsavebox{\CorrLagBox}")
    latex_lines.append(r"\savebox{\CorrLagBox}{\scriptsize{")
    latex_lines.append(r"\begin{tabular}{l|rr|rrr|rrr}")
    latex_lines.append(r"\hline\hline \\[-8pt]")
    latex_lines.append(
        r" & & & \multicolumn{3}{c|}{Record age (days)}"
        r"   & \multicolumn{3}{c}{Reporting lag (days)} \\"
    )
    latex_lines.append(
        r"Error type & $N$ & (\%) & $n$ & Mean & Median & $n$ & Mean & Median \\"
    )
    latex_lines.append(r"\hline \\[-8pt]")

    for mask, label, color in rows:
        n, n_pct, n_reg, m_reg, md_reg, n_out, m_out, md_out = summarize(mask)
        label_tex = rf"\textcolor{{{color}}}{{{label}}}" if color else label
        latex_lines.append(
            rf"{label_tex} & {n:,} & ({n_pct:.1f}\%) & "
            rf"{n_reg if n_reg=='--' else f'{n_reg:,}'} & {m_reg} & {md_reg} & "
            rf"{n_out if n_out=='--' else f'{n_out:,}'} & {m_out} & {md_out} \\"
        )

    latex_lines.append(r"\hline\hline")
    latex_lines.append(r"\end{tabular}}}")
    latex_lines.append(r"\begin{minipage}{\wd\CorrLagBox}")
    latex_lines.append(r"\usebox{\CorrLagBox}")
    latex_lines.append(r"\par\vspace{4pt}\noindent")
    latex_lines.append(
        r"\scriptsize{\textit{Note:} The TIBU snapshot is taken to be 1 January 2022, "
        r"based on the latest treatment-outcome dates in the extract clustering in late "
        r"December 2021. Record age is measured from registration to snapshot; reporting "
        r"lag is measured from TIBU outcome date to snapshot. Column $n$ gives the number "
        r"of records with a valid date; $N$ counts all records in the error category.}"
    )
    latex_lines.append(r"\end{minipage}")

    out_file = os.path.join(OUTPUT_DIR, "tblSI_correction_lag.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved {out_file}")


# =============================================================================
# SECTION 5 — DATE-FIELD AUDIT
# =============================================================================

def _load_raw_paper_dates():
    """Load the raw paper-registry CSV (pre-cleaning) for date-field diagnostics.
    We need the raw strings so that unparseable and out-of-range values remain
    visible rather than being filtered out by clean_dqa_data()."""
    path = os.path.join(DQA_DATA_DIR, "DQA_combined.csv")
    raw = pd.read_csv(path, dtype=str)
    raw = raw.rename(columns={
        "anon_scrn":             "scrn",
        "Treatment Outcome":     "to_paper_raw",
        "Treatment Outcome Date":"date_paper_raw",
    })
    return raw[["scrn", "to_paper_raw", "date_paper_raw"]]


# Plausible-window bounds for every date field audited below (inclusive).
# The TIBU registration window is April 2018 to December 2019; outcomes can
# extend through late 2021 (standard DS-TB ~6m, MDR up to 18-24m).
DATE_WINDOW_LO = pd.Timestamp("2017-01-01")
DATE_WINDOW_HI = pd.Timestamp("2022-06-30")


def _date_quality_row(label, series_raw, parser):
    """Return a dict summarizing blank / unparseable / out-of-range / ok
    counts for a single raw-string date column."""
    n = len(series_raw)
    blank = series_raw.isna() | (series_raw.astype(str).str.strip() == "")
    parsed = parser(series_raw.where(~blank))
    unparseable = (~blank) & parsed.isna()
    oor = parsed.notna() & ((parsed < DATE_WINDOW_LO) | (parsed > DATE_WINDOW_HI))
    ok = parsed.notna() & ~oor
    return {
        "label": label, "n": n,
        "blank":        int(blank.sum()),
        "unparseable":  int(unparseable.sum()),
        "out_of_range": int(oor.sum()),
        "ok":           int(ok.sum()),
    }


def generate_date_field_audit(main_df, dqa_df):
    """
    Section 5: Errors in date fields.

    Produces:
      tblSI_date_quality.tex       — blank / unparseable / out-of-range counts
      tblSI_date_consistency.tex   — internal consistency within TIBU
      tblSI_date_agree.tex         — TIBU vs paper outcome-date disagreement
      fig_tibu_vs_paper_date.pdf   — TIBU outcome date vs paper outcome date
    """
    print("\n--- Generating date-field audit ---")

    # -------------------------------------------------------------------------
    # 5.A  Date-field quality (completeness, parseability, plausible range)
    # -------------------------------------------------------------------------
    # Paper side: raw strings from DQA_combined.csv (pre-cleaning).
    raw_paper = _load_raw_paper_dates()
    # TIBU side: the study-clinic extract. Use the un-formatted raw columns so
    # parseability is audited rather than pre-filtered.
    tibu_rows = []
    paper_rows = []

    paper_rows.append(_date_quality_row(
        "Paper outcome date", raw_paper["date_paper_raw"],
        lambda s: pd.to_datetime(s, errors="coerce", format="%d/%m/%Y")
    ))

    for col, label in [
        ("dateofregistration",      "TIBU clinical registration date"),
        ("dateoftreatmentstarted",  "TIBU treatment-start date"),
        ("treatmentoutcomedate",    "TIBU outcome date"),
    ]:
        tibu_rows.append(_date_quality_row(
            label, main_df[col],
            lambda s: pd.to_datetime(s, errors="coerce")
        ))

    # Emit table
    def _fmt_pct(n, d):
        return f"{n:,} ({100*n/d:.1f}\\%)" if d else "--"

    latex = []
    latex.append(r"\newsavebox{\DateQualityBox}")
    latex.append(r"\savebox{\DateQualityBox}{\scriptsize{")
    latex.append(r"\begin{tabular}{lrrrrr}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(r"Field & $N$ & Blank & Unparseable & Out of range & Valid \\")
    latex.append(r"\hline \\[-8pt]")
    latex.append(r"\textit{Paper registry} (audit records) & & & & & \\")
    for r in paper_rows:
        latex.append(
            rf"\quad {r['label']} & {r['n']:,} & "
            rf"{_fmt_pct(r['blank'], r['n'])} & "
            rf"{_fmt_pct(r['unparseable'], r['n'])} & "
            rf"{_fmt_pct(r['out_of_range'], r['n'])} & "
            rf"{_fmt_pct(r['ok'], r['n'])} \\"
        )
    latex.append(r"\rule{0pt}{10pt}\textit{TIBU} (study-clinic extract) & & & & & \\")
    for r in tibu_rows:
        latex.append(
            rf"\quad {r['label']} & {r['n']:,} & "
            rf"{_fmt_pct(r['blank'], r['n'])} & "
            rf"{_fmt_pct(r['unparseable'], r['n'])} & "
            rf"{_fmt_pct(r['out_of_range'], r['n'])} & "
            rf"{_fmt_pct(r['ok'], r['n'])} \\"
        )
    latex.append(r"\hline\hline")
    latex.append(r"\end{tabular}}}")
    latex.append(r"\begin{minipage}{\wd\DateQualityBox}")
    latex.append(r"\usebox{\DateQualityBox}")
    latex.append(r"\par\vspace{4pt}\noindent")
    latex.append(
        rf"\scriptsize{{\textit{{Note:}} Plausible range = "
        rf"{DATE_WINDOW_LO.strftime('%Y-%m-%d')} through "
        rf"{DATE_WINDOW_HI.strftime('%Y-%m-%d')}. "
        rf"``Unparseable'' values are nonblank strings that could not be parsed as a date; "
        rf"``out of range'' are parseable dates outside the plausible window.}}"
    )
    latex.append(r"\end{minipage}")
    out_path = os.path.join(OUTPUT_DIR, "tblSI_date_quality.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(latex))
    print(f"Saved {out_path}")

    # -------------------------------------------------------------------------
    # 5.B  Internal consistency within TIBU
    # -------------------------------------------------------------------------
    td = main_df.copy()
    td["reg_dt"]   = pd.to_datetime(td["dateofregistration"],    errors="coerce")
    td["tx_dt"]    = pd.to_datetime(td["dateoftreatmentstarted"],errors="coerce")
    td["out_dt"]   = pd.to_datetime(td["treatmentoutcomedate"],  errors="coerce")
    # censor implausible values so we only flag real ordering errors
    for c in ["reg_dt", "tx_dt", "out_dt"]:
        bad = (td[c] < DATE_WINDOW_LO) | (td[c] > DATE_WINDOW_HI)
        td.loc[bad, c] = pd.NaT

    N = len(td)
    have_reg_tx = td["reg_dt"].notna() & td["tx_dt"].notna()
    have_tx_out = td["tx_dt"].notna()  & td["out_dt"].notna()
    have_reg_out= td["reg_dt"].notna() & td["out_dt"].notna()
    dur = (td["out_dt"] - td["tx_dt"]).dt.days

    checks = [
        ("Treatment start after outcome date",
         int(((td["tx_dt"] > td["out_dt"]) & have_tx_out).sum()),
         int(have_tx_out.sum())),
        ("Clinical registration after outcome date",
         int(((td["reg_dt"] > td["out_dt"]) & have_reg_out).sum()),
         int(have_reg_out.sum())),
        ("Treatment duration $<$ 30 days",
         int(((dur < 30) & dur.notna()).sum()),
         int(dur.notna().sum())),
        ("Treatment duration $>$ 900 days",
         int(((dur > 900) & dur.notna()).sum()),
         int(dur.notna().sum())),
    ]

    latex = []
    latex.append(r"\newsavebox{\DateConsistBox}")
    latex.append(r"\savebox{\DateConsistBox}{\scriptsize{")
    latex.append(r"\begin{tabular}{lrr}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(r"Invariant violation & Denominator & Violations \\")
    latex.append(r"\hline \\[-8pt]")
    for lbl, violations, denom in checks:
        pct = f"{100*violations/denom:.1f}\\%" if denom else "--"
        latex.append(
            rf"{lbl} & {denom:,} & {violations:,} ({pct}) \\"
        )
    latex.append(r"\hline\hline")
    latex.append(r"\end{tabular}}}")
    latex.append(r"\begin{minipage}{\wd\DateConsistBox}")
    latex.append(r"\usebox{\DateConsistBox}")
    latex.append(r"\par\vspace{4pt}\noindent")
    latex.append(
        rf"\scriptsize{{\textit{{Note:}} Denominator for each row is the number of "
        rf"study-clinic records (of $N={N:,}$) with a valid (non-blank, in-range) date in "
        rf"both fields being compared. Violations count records where the ordering invariant "
        rf"fails.}}"
    )
    latex.append(r"\end{minipage}")
    out_path = os.path.join(OUTPUT_DIR, "tblSI_date_consistency.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(latex))
    print(f"Saved {out_path}")

    # -------------------------------------------------------------------------
    # 5.B.triangulation  — Resolve TIBU inconsistencies via paper audit dates
    # -------------------------------------------------------------------------
    # For records with a TIBU internal-consistency violation, check whether the
    # paper-registry outcome date (when available in the audit) helps identify
    # which TIBU date field is wrong. The paper date is a second independent
    # reading of the treatment outcome's timing; where it agrees with one of
    # TIBU's date fields, the other is likely the one at fault.
    mm = main_df.merge(
        dqa_df[["scrn", "date_paper"]].drop_duplicates("scrn"),
        on="scrn", how="left"
    )
    mm["reg_dt"]  = pd.to_datetime(mm["dateofregistration"],    errors="coerce")
    mm["tx_dt"]   = pd.to_datetime(mm["dateoftreatmentstarted"],errors="coerce")
    mm["out_dt"]  = pd.to_datetime(mm["treatmentoutcomedate"],  errors="coerce")
    mm["pap_dt"]  = pd.to_datetime(mm["date_paper"], errors="coerce", dayfirst=True)
    for c in ["reg_dt", "tx_dt", "out_dt", "pap_dt"]:
        bad = (mm[c] < DATE_WINDOW_LO) | (mm[c] > DATE_WINDOW_HI)
        mm.loc[bad, c] = pd.NaT

    rows = []
    # (i) outcome before registration
    v = mm[(mm["out_dt"] < mm["reg_dt"]) & mm["out_dt"].notna() & mm["reg_dt"].notna()]
    v_aud = v[v["pap_dt"].notna()]
    paper_confirms_reg = (v_aud["pap_dt"] > v_aud["reg_dt"]).sum()
    paper_confirms_out = ((v_aud["pap_dt"] - v_aud["out_dt"]).dt.days.abs() <= 30).sum()
    rows.append(("Outcome date before registration", len(v), len(v_aud),
                 paper_confirms_reg, paper_confirms_out))
    # (ii) outcome before treatment start
    v = mm[(mm["out_dt"] < mm["tx_dt"]) & mm["out_dt"].notna() & mm["tx_dt"].notna()]
    v_aud = v[v["pap_dt"].notna()]
    paper_confirms_tx  = (v_aud["pap_dt"] > v_aud["tx_dt"]).sum()
    paper_confirms_out = ((v_aud["pap_dt"] - v_aud["out_dt"]).dt.days.abs() <= 30).sum()
    rows.append(("Outcome date before treatment start", len(v), len(v_aud),
                 paper_confirms_tx, paper_confirms_out))
    # (iii) treatment duration < 30 days
    dur = (mm["out_dt"] - mm["tx_dt"]).dt.days
    v = mm[(dur < 30) & dur.notna()]
    v_aud = v[v["pap_dt"].notna()]
    paper_suggests_longer = ((v_aud["pap_dt"] - v_aud["tx_dt"]).dt.days >= 30).sum()
    paper_confirms_short  = ((v_aud["pap_dt"] - v_aud["tx_dt"]).dt.days < 30).sum()
    rows.append(("Treatment duration $<$ 30 days", len(v), len(v_aud),
                 paper_suggests_longer, paper_confirms_short))

    latex = []
    latex.append(r"\newsavebox{\DateTriangleBox}")
    latex.append(r"\savebox{\DateTriangleBox}{\scriptsize{")
    latex.append(r"\begin{tabular}{lrrrr}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(r" & & & \multicolumn{2}{c}{Paper audit date resolves violation} \\")
    latex.append(r"Violation & Total & In audit & TIBU outcome suspect & TIBU outcome plausible \\")
    latex.append(r"\hline \\[-8pt]")
    for label, total, in_audit, a, b in rows:
        a_pct = f"{100*a/in_audit:.0f}\\%" if in_audit else "--"
        b_pct = f"{100*b/in_audit:.0f}\\%" if in_audit else "--"
        latex.append(
            rf"{label} & {total:,} & {in_audit:,} & {a} ({a_pct}) & {b} ({b_pct}) \\"
        )
    latex.append(r"\hline\hline")
    latex.append(r"\end{tabular}}}")
    latex.append(r"\begin{minipage}{\wd\DateTriangleBox}")
    latex.append(r"\usebox{\DateTriangleBox}")
    latex.append(r"\par\vspace{4pt}\noindent")
    latex.append(
        r"\scriptsize{\textit{Note:} For each TIBU internal-consistency violation, "
        r"``In audit'' counts records in the DQA-matched subset (with a paper outcome date). "
        r"The two rightmost columns apply a heuristic: we call the TIBU outcome "
        r"\emph{suspect} when the paper date is consistent with the TIBU registration/treatment-start "
        r"timeline (paper outcome clearly later), and \emph{plausible} when the paper date is "
        r"within 30 days of the TIBU outcome date (confirming the outcome really did happen early). "
        r"Neither column is mutually exclusive; some records are classified by both heuristics, "
        r"or by neither.}"
    )
    latex.append(r"\end{minipage}")
    out_path = os.path.join(OUTPUT_DIR, "tblSI_date_triangulation.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(latex))
    print(f"Saved {out_path}")

    # -------------------------------------------------------------------------
    # 5.C  TIBU vs paper outcome-date disagreement (on the DQA-matched sample)
    # -------------------------------------------------------------------------
    df = _build_error_df(main_df, dqa_df)
    pair = df[df["tibu_date"].notna() & df["paper_date"].notna()].copy()
    pair["delta"] = (pair["tibu_date"] - pair["paper_date"]).dt.days

    # Bucket |delta|
    bins  = [-1, 7, 30, 90, 180, 10**6]
    labels= [r"$\leq$ 7 days", "8--30 days", "31--90 days", "91--180 days", "$>$180 days"]
    pair["bucket"] = pd.cut(pair["delta"].abs(), bins=bins, labels=labels, right=True)

    # tex_color is a LaTeX color spec; mpl_color is the matplotlib equivalent.
    categories = [
        ("No error",                (pair["false_neg"]==0)&(pair["false_pos"]==0)&(pair["false_miss"]==0), "black",           "black"),
        ("Type I (false positive)", (pair["false_neg"]==1), "red",             "red"),
        ("Type II (false negative)",(pair["false_pos"]==1), "green!60!black",  "darkgreen"),
        # false_miss records have no tibu_date, so are absent from this table.
    ]

    latex = []
    latex.append(r"\newsavebox{\DateAgreeBox}")
    latex.append(r"\savebox{\DateAgreeBox}{\scriptsize{")
    latex.append(r"\begin{tabular}{l|r|rrrrr|rr}")
    latex.append(r"\hline\hline \\[-8pt]")
    latex.append(
        r" & & \multicolumn{5}{c|}{$|$TIBU $-$ paper$|$ (days)} & \multicolumn{2}{c}{TIBU date is} \\"
    )
    latex.append(
        rf"Category & $N$ & {' & '.join(labels)} & Earlier & Later \\"
    )
    latex.append(r"\hline \\[-8pt]")
    for name, mask, tex_color, _ in categories:
        sub = pair[mask]
        n = len(sub)
        cells = []
        for lbl in labels:
            c = int((sub["bucket"] == lbl).sum())
            cells.append(f"{c} ({100*c/n:.1f}\\%)" if n else "--")
        earlier = int((sub["delta"] < 0).sum())
        later   = int((sub["delta"] > 0).sum())
        e_cell  = f"{earlier} ({100*earlier/n:.1f}\\%)" if n else "--"
        l_cell  = f"{later} ({100*later/n:.1f}\\%)" if n else "--"
        colored = rf"\textcolor{{{tex_color}}}{{{name}}}"
        latex.append(
            rf"{colored} & {n:,} & {' & '.join(cells)} & {e_cell} & {l_cell} \\"
        )
    latex.append(r"\hline\hline")
    latex.append(r"\end{tabular}}}")
    latex.append(r"\begin{minipage}{\wd\DateAgreeBox}")
    latex.append(r"\usebox{\DateAgreeBox}")
    latex.append(r"\par\vspace{4pt}\noindent")
    latex.append(
        r"\scriptsize{\textit{Note:} Denominator is records where both the TIBU and paper "
        r"outcome dates are non-blank and parseable. Missing-in-TIBU records are excluded "
        r"(TIBU date is blank by construction). ``TIBU earlier'' and ``TIBU later'' together "
        r"sum to $<$ 100\% because some records agree exactly.}"
    )
    latex.append(r"\end{minipage}")
    out_path = os.path.join(OUTPUT_DIR, "tblSI_date_agree.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(latex))
    print(f"Saved {out_path}")

    # -------------------------------------------------------------------------
    # 5.D  Figure: TIBU outcome date vs paper outcome date, by error category
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    # Restrict axes to the study window (clips pre-2018 out-of-range outliers).
    ax_lo = pd.Timestamp("2018-01-01")
    ax_hi = pd.Timestamp("2022-01-01")
    # Plot no-error as faint grey background so the error points pop on top.
    ne_mask = (pair["false_neg"]==0) & (pair["false_pos"]==0) & (pair["false_miss"]==0)
    ne = pair[ne_mask]
    ax.scatter(ne["paper_date"], ne["tibu_date"],
               s=8, alpha=0.18, color="0.55",
               label=f"No error (N={len(ne)})")
    # Plot errors prominently on top with larger, edged markers.
    for name, mask, mpl_color, marker in [
        ("Type I (false positive)",  pair["false_neg"]==1, "#d62728", "o"),
        ("Type II (false negative)", pair["false_pos"]==1, "#2ca02c", "o"),
    ]:
        sub = pair[mask]
        ax.scatter(sub["paper_date"], sub["tibu_date"],
                   s=55, alpha=0.85, color=mpl_color, marker=marker,
                   edgecolor="black", linewidth=0.4,
                   label=f"{name} (N={len(sub)})", zorder=3)
    # y=x and year-offset reference lines (diagonals)
    ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], "--", color="gray", lw=0.8, label="$y = x$")
    for yrs in (1, 2, -1):
        off = pd.Timedelta(days=365*yrs)
        ax.plot([ax_lo, ax_hi], [ax_lo + off, ax_hi + off],
                ":", color="gray", lw=0.5, alpha=0.6,
                label=("$y = x \\pm $1--2 yr" if yrs == 1 else None))
    ax.set_xlim(ax_lo, ax_hi)
    ax.set_ylim(ax_lo, ax_hi)
    ax.set_xlabel("Paper outcome date")
    ax.set_ylabel("TIBU outcome date")
    ax.set_title("TIBU vs paper outcome date (matched records)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_locator(mdates.YearLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_tibu_vs_paper_date.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def main():
    print("Starting Consolidated DQA Analysis...")

    main_df = load_study_data()
    dqa_df_cleaned = clean_dqa_data()
    if main_df is None or dqa_df_cleaned is None:
        return

    # Patient-characteristics tables (Tables 1a/1b)
    generate_patient_characteristics_table(main_df, dqa_df_cleaned)

    # Trial-level sensitivity-analysis tables (not included in main.tex)
    stats = calculate_sensitivity_stats(main_df)
    stat_c = stats["Control Group"]
    generate_sensitivity_table(
        "Keheala vs Control", "tblSI_DQA_SA_Keheala.tex",
        stat_c["N_valid"], stats["Keheala Group"]["N_valid"],
        stat_c["prop"], stats["Keheala Group"]["prop"],
        benchmarks={(0, 0): (0.0261, "<.0001")},
    )
    generate_sensitivity_table(
        "Platform vs Control", "tblSI_DQA_SA_Platform.tex",
        stat_c["N_valid"], stats["SBCC Group"]["N_valid"],
        stat_c["prop"], stats["SBCC Group"]["prop"],
    )
    generate_sensitivity_table(
        "SMS vs Control", "tblSI_DQA_SA_SMS.tex",
        stat_c["N_valid"], stats["SMS Reminder Group"]["N_valid"],
        stat_c["prop"], stats["SMS Reminder Group"]["prop"],
    )

    # DQA crosstab and error-rate tables (Tables 2, 3, 4)
    generate_crosstab_table(main_df, dqa_df_cleaned)
    generate_error_by_clinic(main_df, dqa_df_cleaned)
    generate_error_by_patient_characteristics(main_df, dqa_df_cleaned)

    # Supplementary summaries (CSVs and a non-included table)
    generate_clinic_characteristics_table(main_df, dqa_df_cleaned)
    generate_outcome_corrections(main_df, dqa_df_cleaned)
    generate_error_by_outcome(main_df, dqa_df_cleaned)

    # Figures and the record-age table (Figs 2–5, Table 5)
    generate_error_over_time_figure(main_df, dqa_df_cleaned)
    generate_error_by_age_figure(main_df, dqa_df_cleaned)
    generate_error_density_figure(main_df, dqa_df_cleaned)
    generate_correction_lag_table(main_df, dqa_df_cleaned)

    # Section 5 — date-field audit
    generate_date_field_audit(main_df, dqa_df_cleaned)

    print("\nAll DQA outputs generated successfully.")


if __name__ == "__main__":
    main()
