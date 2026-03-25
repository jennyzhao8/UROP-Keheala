print(">>> SCRIPT STARTED <<<")
"""
Keheala Study 2 - DQA Analysis (Consolidated)
=============================================

Purpose:
    Performs all DQA (Data Quality Assurance) analyses comparing 
    TIBU (Main Dataset) vs Paper Registry (Digitized DQA Data).

    Generates the following tables in `output/`:
    1. tblSI_DQA_SA_Keheala.tex   (Sensitivity Analysis: Keheala vs Control)
    2. tblSI_DQA_SA_Platform.tex  (Sensitivity Analysis: Platform vs Control)
    3. tblSI_DQA_SA_SMS.tex       (Sensitivity Analysis: SMS vs Control)
    4. tblSI_DQAcrosstab.tex      (Crosstab of TIBU vs Paper Outcomes)
    5. tblSI_DQAtype12error.tex   (Type 1 & 2 Error Rates by Group)

    Logic derived from:
    - `_study2_analysis.do` (lines 623-673)
"""

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DEIDENTIFIED_DIR = os.path.join(ROOT_DIR, "newdata")
INPUT_DATA_CLEANED = os.path.join(DEIDENTIFIED_DIR, "study2_cleaned.csv")
DQA_DATA_DIR = DEIDENTIFIED_DIR
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
# new clinic IDs
TIBU_DEIDENTIFIED = os.path.join(DEIDENTIFIED_DIR, "TIBU_firstnm_deidentified.csv")
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

    good_t = {"C", "TC"}
    bad_t  = {"D", "F", "LTFU", "NC", "Blank"}
    good_p = {"C", "TC"}
    bad_p  = {"D", "F", "LTFU"}

    col_order = ["Blank", "C", "D", "F", "LTFU", "TC", "Total"]
    row_order = ["Blank", "C", "D", "F", "LTFU", "MT4", "N/A", "NC", "TC", "TO", "Total"]

    data_cols = [c for c in col_order if c != "Total"]  # 6 cols, each split into N + (%)
    n_data_cols = len(data_cols)  # 6
    total_cols = 1 + n_data_cols * 2 + 1  # row label + 6*2 + Total = 14

    col_spec = "l|" + "|".join(["rr"] * n_data_cols) + "|r"

    latex_lines = []
    latex_lines.append(r"\scriptsize{")
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
    latex_lines.append(r"\noalign{\vspace{4pt}}")
    latex_lines.append(
        rf"\multicolumn{{{total_cols}}}{{l}}{{\parbox{{0.65\textwidth}}{{"
        r"\textit{Note:} Percentages are row shares. \\"
        r"\textcolor{red}{Red} = Type I error (TIBU: success; paper: failure). \\"
        r"\textcolor{green!60!black}{Green} = Type II error (TIBU: failure; paper: success). \\"
        r"\textcolor{blue}{Blue} = Incomplete data (TIBU: no outcome; paper: outcome present)."
        r"}} \\"
    )
    latex_lines.append(r"\end{tabular}}")
    
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
    bad_tibu = ["D", "F", "LTFU", "NC"]
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
        r"& \cellcolor{red!20} False negative (\%)"
        r"& \cellcolor{green!20} False positive (\%)"
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
    latex_lines.append(rf"\cellcolor{{red!20}} & \multicolumn{{4}}{{l}}{{\scriptsize{{False negative (TIBU: success; paper: fail): {fn_all:.1f}\%}}}} \\[-4pt]")
    latex_lines.append(rf"\cellcolor{{green!20}} & \multicolumn{{4}}{{l}}{{\scriptsize{{False positive (TIBU: fail; paper: success): {fp_all:.1f}\%}}}} \\[-4pt]")
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
    lines_1a.append(r"\\[-4pt]")

    lines_1a.append(r"\textit{Disease characteristics} & & & & & & \\")
    lines_1a.append(
        rf"\quad Bacteriologically confirmed & -- & & {npct(df_study['bacteriologically_confirmed'])} & {npct(df_paper['bacteriologically_confirmed'])} \\"
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
        ("MT4",  "MT4"),
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

    merged = pd.merge(main_df, dqa_df[["scrn", "to_paper"]], on="scrn", how="inner")
    merged["clinic_id_num"] = pd.to_numeric(merged["clinic_id"], errors="coerce")
    merged = merged.merge(
        clinic_summary[["clinic_id", "urban", "tibu_patients", "province"]],
        left_on="clinic_id_num", right_on="clinic_id", how="left"
    )

    bad  = ["D", "F", "LTFU", "NC"]
    good = ["C", "TC"]
    merged["uo_tibu"]  = np.nan
    merged.loc[merged["treatmentoutcome"].isin(bad),  "uo_tibu"] = 1
    merged.loc[merged["treatmentoutcome"].isin(good), "uo_tibu"] = 0
    merged["uo_paper"] = np.nan
    merged.loc[merged["to_paper"].isin(bad),  "uo_paper"] = 1
    merged.loc[merged["to_paper"].isin(good), "uo_paper"] = 0

    df = merged.dropna(subset=["uo_tibu"]).copy()
    df["mismatch"]    = (df["uo_tibu"] != df["uo_paper"]).astype(int)
    df["false_neg"]   = ((df["uo_tibu"] == 0) & (df["uo_paper"] == 1)).astype(int)
    df["false_pos"]   = ((df["uo_tibu"] == 1) & (df["uo_paper"] == 0)).astype(int)
    df["false_miss"]  = df["uo_paper"].isna().astype(int)

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

    def stats(sub):
        n = len(sub)
        fn_n, fp_n, fm_n = int(sub["false_neg"].sum()), int(sub["false_pos"].sum()), int(sub["false_miss"].sum())
        fn_p, fp_p, fm_p = sub["false_neg"].mean()*100, sub["false_pos"].mean()*100, sub["false_miss"].mean()*100
        return n, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p

    def fmt(count, pct):
        return rf"{count:,} ({pct:.1f}\%)"

    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{lcccc}")
    latex_lines.append(r"\hline\hline \\[-8pt]")
    latex_lines.append(
        r"& N"
        r"& False positive"
        r"& False negative"
        r"& Incomplete data \\"
    )
    latex_lines.append(r"\hline \\[-8pt]")

    # By province
    latex_lines.append(r"\textit{By province} & & & & \\")
    for prov in sorted(df["province"].dropna().unique()):
        n, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df[df["province"] == prov])
        latex_lines.append(rf"\quad {prov} & {n} & {fmt(fp_n, fp_p)} & {fmt(fn_n, fn_p)} & {fmt(fm_n, fm_p)} \\")
    latex_lines.append(r"\hline \\[-8pt]")

    # By urban/rural
    latex_lines.append(r"\textit{By urban/rural} & & & & \\")
    for label, mask in [("Urban", df["urban"] == 1), ("Rural", df["urban"] == 0)]:
        n, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df[mask])
        latex_lines.append(rf"\quad {label} & {n} & {fmt(fp_n, fp_p)} & {fmt(fn_n, fn_p)} & {fmt(fm_n, fm_p)} \\")
    latex_lines.append(r"\hline \\[-8pt]")

    # By clinic size
    latex_lines.append(r"\textit{By clinic size} & & & & \\")
    for label, mask in [("Large clinic", df["large_clinic"] == 1), ("Small clinic", df["large_clinic"] == 0)]:
        n, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df[mask])
        latex_lines.append(rf"\quad {label} & {n} & {fmt(fp_n, fp_p)} & {fmt(fn_n, fn_p)} & {fmt(fm_n, fm_p)} \\")
    latex_lines.append(r"\hline \\[-8pt]")

    # All
    n, fn_n, fn_p, fp_n, fp_p, fm_n, fm_p = stats(df)
    latex_lines.append(rf"All & {n} & {fmt(fp_n, fp_p)} & {fmt(fn_n, fn_p)} & {fmt(fm_n, fm_p)} \\")

    latex_lines.append(r"\hline\hline")
    latex_lines.append(r"\end{tabular}}")

    with open(os.path.join(OUTPUT_DIR, "tblSI_error_by_clinic_char.tex"), "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved error_rates_by_clinic.csv, error_rates_by_clinic_n10.csv, tblSI_error_by_clinic_char.tex")

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

    merged = pd.merge(main_df, dqa_df[["scrn", "to_paper"]], on="scrn", how="inner")

    merged["tibu"]  = merged["treatmentoutcome"].fillna("Blank")
    merged["paper"] = merged["to_paper"].fillna("Blank")

    bad  = ["D", "F", "LTFU", "NC"]
    good = ["C", "TC"]
    merged["uo_tibu"]  = np.nan
    merged.loc[merged["treatmentoutcome"].isin(bad),  "uo_tibu"] = 1
    merged.loc[merged["treatmentoutcome"].isin(good), "uo_tibu"] = 0
    merged["uo_paper"] = np.nan
    merged.loc[merged["to_paper"].isin(bad),  "uo_paper"] = 1
    merged.loc[merged["to_paper"].isin(good), "uo_paper"] = 0

    df = merged.dropna(subset=["uo_tibu"]).copy()
    df["mismatch"]   = (df["uo_tibu"] != df["uo_paper"]).astype(int)
    df["false_neg"]  = ((df["uo_tibu"] == 0) & (df["uo_paper"] == 1)).astype(int)
    df["false_pos"]  = ((df["uo_tibu"] == 1) & (df["uo_paper"] == 0)).astype(int)
    df["false_miss"] = df["uo_paper"].isna().astype(int)

    df["age_group"] = pd.cut(
        pd.to_numeric(df["age_in_years"], errors="coerce"),
        bins=[0, 15, 35, 55, 200],
        labels=[r"$<$15", "15--34", "35--54", "55+"]
    )

    def stats(sub):
        if len(sub) == 0:
            return "-", "-", "-", "-"
        return (str(len(sub)),
                f"{sub['false_neg'].mean()*100:.1f}",
                f"{sub['false_pos'].mean()*100:.1f}",
                f"{sub['false_miss'].mean()*100:.1f}")

    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{lcccc}")
    latex_lines.append(r"\hline\hline \\[-8pt]")
    latex_lines.append(
        r"& N"
        r"& False positive (\%)"
        r"& False negative (\%)"
        r"& Incomplete data (\%) \\"
    )
    latex_lines.append(r"\hline \\[-8pt]")

    # Individual characteristics
    latex_lines.append(r"\textit{Individual characteristics} & & & & \\")
    n, fn, fp, fm = stats(df[df["male"] == 1])
    latex_lines.append(rf"\quad Male (\%) & {n} & {fn} & {fp} & {fm} \\")

    latex_lines.append(r"\\[-4pt]")

    # Age
    latex_lines.append(r"\textit{Age group} & & & & \\")
    for grp in [r"$<$15", "15--34", "35--54", "55+"]:
        n, fn, fp, fm = stats(df[df["age_group"] == grp])
        latex_lines.append(rf"\quad {grp} & {n} & {fn} & {fp} & {fm} \\")

    latex_lines.append(r"\\[-4pt]")

    # Disease characteristics
    latex_lines.append(r"\textit{Disease characteristics} & & & & \\")
    for col, yes_label, no_label in [
        ("bacteriologically_confirmed", r"\quad Bacteriologically confirmed", r"\quad Not bacteriologically confirmed"),
        ("extrapulmonary",              r"\quad Extrapulmonary",              r"\quad Pulmonary"),
        ("retreatment",                 r"\quad Retreatment",                 r"\quad New case"),
    ]:
        for val, label in [(1, yes_label), (0, no_label)]:
            n, fn, fp, fm = stats(df[df[col] == val])
            latex_lines.append(rf"{label} & {n} & {fn} & {fp} & {fm} \\")
    for val, label in [(1, r"\quad HIV positive"), (0, r"\quad HIV negative")]:
        n, fn, fp, fm = stats(df[df["hiv_positive"] == val])
        latex_lines.append(rf"{label} & {n} & {fn} & {fp} & {fm} \\")
    latex_lines.append(r"\end{tabular}}")

    out_file = os.path.join(OUTPUT_DIR, "tblSI_error_by_patient_char.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved {out_file}")

def _build_error_df(main_df, dqa_df):
    """Shared helper: merge, classify errors, parse dates."""
    merged = pd.merge(main_df, dqa_df[["scrn", "to_paper", "date_paper"]], on="scrn", how="inner")

    bad  = ["D", "F", "LTFU", "NC"]
    good = ["C", "TC"]
    merged["uo_tibu"] = np.nan
    merged.loc[merged["treatmentoutcome"].isin(bad),  "uo_tibu"] = 1
    merged.loc[merged["treatmentoutcome"].isin(good), "uo_tibu"] = 0
    merged["uo_paper"] = np.nan
    merged.loc[merged["to_paper"].isin(bad),  "uo_paper"] = 1
    merged.loc[merged["to_paper"].isin(good), "uo_paper"] = 0

    df = merged.dropna(subset=["uo_tibu"]).copy()
    df["false_neg"]  = ((df["uo_tibu"] == 0) & (df["uo_paper"] == 1)).astype(int)
    df["false_pos"]  = ((df["uo_tibu"] == 1) & (df["uo_paper"] == 0)).astype(int)
    df["false_miss"] = df["uo_paper"].isna().astype(int)

    # Parse dates; drop 1900 sentinel values
    df["tibu_date"]  = pd.to_datetime(df["treatmentoutcomedate_formatted"], errors="coerce")
    df["paper_date"] = pd.to_datetime(df["date_paper"], dayfirst=True, errors="coerce")
    df.loc[df["tibu_date"].dt.year < 2010,  "tibu_date"]  = pd.NaT
    df.loc[df["paper_date"].dt.year < 2010, "paper_date"] = pd.NaT
    df["lag_days"] = (df["paper_date"] - df["tibu_date"]).dt.days

    # Age of record: days from registration to TIBU outcome date (treatment duration)
    df["reg_date"] = pd.to_datetime(df["dateregistered_formatted"], errors="coerce")
    df["age_days"] = (df["tibu_date"] - df["reg_date"]).dt.days
    df.loc[df["age_days"] < 0, "age_days"] = pd.NA
    return df


def generate_error_over_time_figure(main_df, dqa_df):
    """
    Line chart of false-negative, false-positive, and false-missing rates
    by quarter of treatment completion date.
    Saves fig_error_over_time.pdf
    """
    print("\n--- Generating error-over-time figure ---")
    df = _build_error_df(main_df, dqa_df)

    df_dated = df.dropna(subset=["tibu_date"]).copy()
    df_dated["quarter"] = df_dated["tibu_date"].dt.to_period("Q")

    by_q = (
        df_dated.groupby("quarter")
        .agg(fn=("false_neg", "mean"), fp=("false_pos", "mean"),
             fm=("false_miss", "mean"), n=("scrn", "count"))
        .reset_index()
    )
    by_q = by_q[by_q["n"] >= 10].copy()
    by_q["date"] = by_q["quarter"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(by_q["date"], by_q["fn"] * 100, color="red",   marker="o", linewidth=1.5, label="False negative")
    ax.plot(by_q["date"], by_q["fp"] * 100, color="green", marker="s", linewidth=1.5, label="False positive")
    ax.plot(by_q["date"], by_q["fm"] * 100, color="blue",  marker="^", linewidth=1.5, label="False missing")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.set_xlabel("Treatment completion date (quarter)")
    ax.set_ylabel("Error rate (%)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax.legend(framealpha=0.9, handlelength=0.8, labelspacing=0.6, borderpad=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_file = os.path.join(OUTPUT_DIR, "fig_error_over_time.pdf")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")


def generate_correction_lag_table(main_df, dqa_df):
    """
    Table: age of TIBU record (days from TIBU outcome date to DQA verification date),
    broken down by error type.
    Saves tblSI_correction_lag.tex
    """
    print("\n--- Generating correction lag table ---")
    df = _build_error_df(main_df, dqa_df)

    def age_stats(mask):
        sub = df.loc[mask & df["age_days"].notna(), "age_days"]
        if len(sub) == 0:
            return "--", "--", "--", "--", "--"
        return (
            f"{len(sub):,}",
            f"{sub.mean():.0f}",
            f"{sub.median():.0f}",
            f"{sub.quantile(0.25):.0f}",
            f"{sub.quantile(0.75):.0f}",
        )

    rows = [
        (df["false_neg"] == 1, "False positive (Type I)", "red"),
        (df["false_pos"] == 1, "False negative (Type II)", "green!60!black"),
    ]

    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{lccccc}")
    latex_lines.append(r"\hline\hline \\[-8pt]")
    latex_lines.append(
        r"Error type & N"
        r"& Mean (days) & Median (days) & Q1 (days) & Q3 (days) \\"
    )
    latex_lines.append(r"\hline \\[-8pt]")

    for mask, label, color in rows:
        n, mean, med, q1, q3 = age_stats(mask)
        latex_lines.append(
            rf"\textcolor{{{color}}}{{{label}}} & {n} & {mean} & {med} & {q1} & {q3} \\"
        )

    # False missing: include age stats if available
    fm_mask = df["false_miss"] == 1
    n, mean, med, q1, q3 = age_stats(fm_mask)
    latex_lines.append(
        rf"\textcolor{{blue}}{{False missing}} & {n} & {mean} & {med} & {q1} & {q3} \\"
    )

    latex_lines.append(r"\hline\hline")
    latex_lines.append(
        r"\multicolumn{6}{l}{\scriptsize{\textit{Note: age of record = TIBU outcome date $-$ date of registration. Negative values excluded.}}} \\"
    )
    latex_lines.append(r"\end{tabular}}")

    out_file = os.path.join(OUTPUT_DIR, "tblSI_correction_lag.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved {out_file}")


def main():
    print("Starting Consolidated DQA Analysis...")
    
    # Load Data
    main_df = load_study_data()
    dqa_df_cleaned = clean_dqa_data()
    
    if main_df is None: return
    if dqa_df_cleaned is None: return

    generate_patient_characteristics_table(main_df, dqa_df_cleaned)

    # 1. Sensitivity Analysis
    stats = calculate_sensitivity_stats(main_df)
    stat_c = stats["Control Group"]
    
    # Keheala
    generate_sensitivity_table(
        "Keheala vs Control", "tblSI_DQA_SA_Keheala.tex",
        stat_c["N_valid"], stats["Keheala Group"]["N_valid"],
        stat_c["prop"], stats["Keheala Group"]["prop"],
        benchmarks={(0, 0): (0.0261, "<.0001")}
    )
    # Platform
    generate_sensitivity_table(
        "Platform vs Control", "tblSI_DQA_SA_Platform.tex",
        stat_c["N_valid"], stats["SBCC Group"]["N_valid"],
        stat_c["prop"], stats["SBCC Group"]["prop"]
    )
    # SMS
    generate_sensitivity_table(
        "SMS vs Control", "tblSI_DQA_SA_SMS.tex",
        stat_c["N_valid"], stats["SMS Reminder Group"]["N_valid"],
        stat_c["prop"], stats["SMS Reminder Group"]["prop"]
    )
    
    if dqa_df_cleaned is None: return
    generate_patient_characteristics_table(main_df, dqa_df_cleaned)
    generate_crosstab_table(main_df, dqa_df_cleaned)
    generate_error_by_clinic(main_df, dqa_df_cleaned)
    generate_outcome_corrections(main_df, dqa_df_cleaned)
    generate_error_by_outcome(main_df, dqa_df_cleaned)
    generate_error_by_patient_characteristics(main_df, dqa_df_cleaned)
    generate_error_over_time_figure(main_df, dqa_df_cleaned)
    generate_correction_lag_table(main_df, dqa_df_cleaned)

    print("\nAll DQA tables generated successfully.")
    print(main_df["clinic_id"].value_counts().head(10))

if __name__ == "__main__":
    main()
