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
    df = df.rename(columns={"treatmentoutcome": "to_paper"})
    
    return df

def load_study_data():
    """Returns the main study dataframe."""

    df = pd.read_csv(INPUT_DATA_CLEANED, low_memory=False)
    if "anon_scrn" in df.columns:
        df = df.rename(columns={"anon_scrn": "scrn"})

    # Replace old clinic_id with new one from TIBU file
    tibu = pd.read_csv(TIBU_DEIDENTIFIED, dtype=str)[["anon_scrn_tibu", "source_file"]]
    tibu = tibu.rename(columns={"anon_scrn_tibu": "scrn", "source_file": "clinic_id"})
    tibu["scrn"] = tibu["scrn"].astype(str)
    df["scrn"] = df["scrn"].astype(str)

    df = df.drop(columns=["clinic_id"], errors="ignore")
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
    latex_lines.append(r"\rowcolor{yellow!15} Type I & \multicolumn{6}{c}{Type II Error Rate (\%)} \\")
    cols_str = " & ".join([f"{int(b*100)}" for b in betas]) 
    latex_lines.append(fr"\rowcolor{{yellow!15}} Error Rate (\%)   & {cols_str} \\ \hline")
    
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
    
    # LaTeX Generation
    col_order = ["Blank", "C", "D", "F", "LTFU", "TC", "Total"]
    row_order = ["Blank", "C", "D", "F", "LTFU", "MT4", "N/A", "NC", "TC", "TO", "Total"]
    
    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{l|cccccc|c}")
    latex_lines.append(r"\hline \hline \\[-8pt]")
    latex_lines.append(r"\rowcolor{yellow!15} Outcome in TIBU & \multicolumn{6}{c}{Treatment Outcome in Paper Registry} & Total\\")
    latex_lines.append(r"\rowcolor{yellow!15} &         Blank &           C   &       D  &        F  &     LTFU  &       TC &    \\ \hline \\[-8pt]")
    
    for row_idx in row_order:
        if row_idx not in ct.index: continue
        if row_idx == "Total":
            latex_lines.append(r"\hline \\[-8pt]")
        row_data = ct.loc[row_idx]
        cells = [f"{row_data.get(col, 0):,}" for col in col_order]
        latex_lines.append(f"{row_idx} & " + " & ".join(cells) + r" \\")

    latex_lines.append(r"\hline \hline")
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
    
    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{lccc}")
    latex_lines.append(r"\hline \hline")
    latex_lines.append(r"\rowcolor{yellow!15} & \multicolumn{2}{c}{Outcome in TIBU} &  \\")
    latex_lines.append(r"\rowcolor{yellow!15} & Successful & Unsuccessful & Total \\")
    latex_lines.append(r"\hline \\[-8pt] ")
    
    for g in groups:
        sub = df_final[df_final["group_label"] == g]
        if sub.empty: continue
        
        # Successful (TIBU=0)
        sub_succ = sub[sub["uo_tibu"] == 0]
        rate_succ = sub_succ["mismatch"].mean() * 100 if len(sub_succ)>0 else 0
        
        # Unsuccessful (TIBU=1)
        sub_unsucc = sub[sub["uo_tibu"] == 1]
        rate_unsucc = sub_unsucc["mismatch"].mean() * 100 if len(sub_unsucc)>0 else 0
        
        # Total
        rate_tot = sub["mismatch"].mean() * 100
        
        latex_lines.append(f"{g}     & {rate_succ:.1f} & {rate_unsucc:.1f} & {rate_tot:.1f} \\\\")
        
    latex_lines.append(r"\hline \\[-8pt]")
    
    # Total Row
    sub_all = df_final[df_final["treatment_group"].isin(group_map.keys())]
    rate_succ_all = sub_all[sub_all["uo_tibu"]==0]["mismatch"].mean() * 100
    rate_unsucc_all = sub_all[sub_all["uo_tibu"]==1]["mismatch"].mean() * 100
    rate_tot_all = sub_all["mismatch"].mean() * 100
    
    latex_lines.append(f"Total       & {rate_succ_all:.1f} & {rate_unsucc_all:.1f} & {rate_tot_all:.1f} \\\\")
    latex_lines.append(r"\hline \hline") 
    latex_lines.append(r"\end{tabular}}")
    
    out_file = os.path.join(OUTPUT_DIR, "tblSI_DQAtype12error.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved {out_file}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
# NEW
def generate_error_by_clinic(main_df, dqa_df):
    """
    Computes mismatch, type I, and type II error rates by clinic.
    """
    print("\n--- Generating error rates by clinic ---")

    merged = pd.merge(main_df, dqa_df[["scrn", "to_paper"]], on="scrn", how="inner")
    # Define outcomes
    bad = ["D", "F", "LTFU", "NC"]
    good = ["C", "TC"]

    merged["uo_tibu"] = np.nan
    merged.loc[merged["treatmentoutcome"].isin(bad), "uo_tibu"] = 1
    merged.loc[merged["treatmentoutcome"].isin(good), "uo_tibu"] = 0

    merged["uo_paper"] = np.nan
    merged.loc[merged["to_paper"].isin(bad), "uo_paper"] = 1
    merged.loc[merged["to_paper"].isin(good), "uo_paper"] = 0

    df = merged.dropna(subset=["uo_tibu"]).copy()

    # Mismatch
    df["mismatch"] = (df["uo_tibu"] != df["uo_paper"]).astype(int)

    # Type I: false positive (TIBU success but paper unsuccess)
    df["type1"] = ((df["uo_tibu"] == 0) & (df["uo_paper"] == 1)).astype(int)

    # Type II: false negative (TIBU unsuccess but paper success)
    df["type2"] = ((df["uo_tibu"] == 1) & (df["uo_paper"] == 0)).astype(int)

    # remove missing clinic IDs
    df = df[df["clinic_id"] != 0]

    # NEW Group by clinic
    clinic_stats = (
        # calculate error rates separately for each clniic
        df.groupby("clinic_id") 
        .agg(
            n=("scrn", "count"), # num of patients
            mismatch_rate=("mismatch", "mean"), # share of records with disagreement
            type1_rate=("type1", "mean"), # share of false pos
            type2_rate=("type2", "mean"), # share of false neg
        )
        .reset_index()
        # shows clinics with highest error rates first
        .sort_values("mismatch_rate", ascending=False) 
    )

    # filter clinics with enough observations (avoid n=1)
    clinic_stats_filtered = clinic_stats[clinic_stats["n"] >= 30]

    # sort again after filtering
    clinic_stats_filtered = clinic_stats_filtered.sort_values(
        "mismatch_rate", ascending=False
    )

    # save full table
    out_file = os.path.join(OUTPUT_DIR, "error_rates_by_clinic.csv")
    clinic_stats.to_csv(out_file, index=False)

    # save filtered table
    filtered_file = os.path.join(OUTPUT_DIR, "error_rates_by_clinic_n30.csv")
    clinic_stats_filtered.to_csv(filtered_file, index=False)
    
    total_clinics = clinic_stats.shape[0]
    filtered_clinics = clinic_stats_filtered.shape[0]
    total_patients = clinic_stats["n"].sum()
    filtered_patients = clinic_stats_filtered["n"].sum()

    print(f"\nClinics total: {total_clinics}")
    print(f"Clinics with n >= 30: {filtered_clinics}")
    print(f"Matched patients total: {total_patients}")
    print(f"Matched patients in n≥30 clinics: {filtered_patients} ({filtered_patients/total_patients:.1%})")

    print(f"Saved {out_file}")
    print("\nTop 10 clinics with highest error rates (n >= 30):")
    print(clinic_stats_filtered.head(10))

def generate_outcome_corrections(main_df, dqa_df):
    """
    Finds the most common outcome discrepancies between TIBU and Paper registry.
    """
    print("\n--- Generating most common outcome corrections ---")

    merged = pd.merge(main_df, dqa_df[["scrn", "to_paper"]], on="scrn", how="inner")

    df = merged.copy()

    # replace missing values so they appear in tables
    df["tibu"] = df["treatmentoutcome"].fillna("Blank")
    df["paper"] = df["to_paper"].fillna("Blank")

    # keep only rows where outcomes differ
    corrections = df[df["tibu"] != df["paper"]]

    # count corrections
    correction_counts = (
        corrections.groupby(["tibu", "paper"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # save full table
    out_file = os.path.join(OUTPUT_DIR, "outcome_corrections.csv")
    correction_counts.to_csv(out_file, index=False)

    print(f"Saved {out_file}")

    print("\nTop 10 most common corrections:")
    print(correction_counts.head(10))

def generate_error_by_outcome(main_df, dqa_df):
    """
    Computes mismatch rates by TIBU outcome.
    """
    print("\n--- Generating error rates by outcome ---")

    merged = pd.merge(main_df, dqa_df[["scrn","to_paper"]], on="scrn", how="inner")

    df = merged.copy()

    df["tibu"] = df["treatmentoutcome"].fillna("Blank")
    df["paper"] = df["to_paper"].fillna("Blank")

    df["mismatch"] = (df["tibu"] != df["paper"]).astype(int)

    outcome_stats = (
        df.groupby("tibu")
        .agg(
            n=("scrn","count"),
            mismatch_rate=("mismatch","mean")
        )
        .reset_index()
        .sort_values("mismatch_rate", ascending=False)
    )

    out_file = os.path.join(OUTPUT_DIR, "error_rates_by_outcome.csv")
    outcome_stats.to_csv(out_file, index=False)

    print(f"Saved {out_file}")
    print("\nError rates by outcome:")
    print(outcome_stats)

def main():
    print("Starting Consolidated DQA Analysis...")
    
    # Load Data
    main_df = load_study_data()
    dqa_df_cleaned = clean_dqa_data()
    
    if main_df is None: return
    
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

    # 2. Crosstab
    generate_crosstab_table(main_df, dqa_df_cleaned)
    
    # 3. Error Rates
    generate_error_table(main_df, dqa_df_cleaned)

    # NEW 4. Error rates by clinic
    generate_error_by_clinic(main_df, dqa_df_cleaned)
    
    # NEW 5. Most common outcome corrections
    generate_outcome_corrections(main_df, dqa_df_cleaned)

    # NEW 6. Check whether errors are higher for certain outcomes
    generate_error_by_outcome(main_df, dqa_df_cleaned)

    print("\nAll DQA tables generated successfully.")
    print(main_df["clinic_id"].value_counts().head(10))

if __name__ == "__main__":
    main()

"""
patient-level comparison logic 
--> generate_error_table summarizes by treatment group
--> generate_error_by_clinic summarizes by clinic
"""

"""
csv
grouping by clinic --> do some clinics systematically produce more data errors
many clinics with n=1-4 --> extreme error rates (0 or 100%)
"""

"""
conclusions
- error rates vary substantially across clinics,
ranging from roughly 2 to 19% (almost 10x difference in DQ)
among clinics with 30+ matched records
- coverage: only 14 clinics are large enough for reliable estimates,
but they contain 90.9% of patients
- what is clinic_id=0?
- interesting examples from table:
clinic 403 mismatch_rate=19.3, type1_rate=1.75, type2_rate=0.88
so most mismatches are not purely type 1 or 2,
suggests other kinds of discrepancies (missing values?)
- which corrections are the most common?
TC --> C (76 cases):
marked "treatment completed" in TIBU, paper recorded "cured"
probably a classificication differ4ence not true error
TO --> TC (62 cases):
TIBU: TO (transfer out), paper: TC
transfer recorded early in TIBU, final outcome recorded later on paper? 
"""