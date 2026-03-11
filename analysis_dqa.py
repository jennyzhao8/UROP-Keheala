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

def generate_patient_characteristics_table(main_df):
    """
    Fig 1: Who is in our study?
    Patient and disease characteristics broken down by urban vs rural.
    Outputs tblSI_patient_characteristics.tex
    """
    print("\n--- Generating Patient Characteristics Table (Fig 1) ---")

    clinic_summary = pd.read_csv(os.path.join(DEIDENTIFIED_DIR, "clinic_summary_deidentified.csv"))

    df = main_df.copy()
    df["clinic_id_num"] = pd.to_numeric(df["clinic_id"], errors="coerce")
    clinic_summary["clinic_id"] = pd.to_numeric(clinic_summary["clinic_id"], errors="coerce")

    df = df.merge(
        clinic_summary[["clinic_id", "urban"]],
        left_on="clinic_id_num",
        right_on="clinic_id",
        how="left"
    )

    # Filter to MITT only (main analysis sample)
    df = df[df["MITT"] == 1].copy()

    all_   = df
    urban  = df[df["urban"] == 1]
    rural  = df[df["urban"] == 0]

    N_all  = len(all_)
    N_u    = len(urban)
    N_r    = len(rural)

    def pct(series):
        """Mean of a 0/1 series as percentage string."""
        v = pd.to_numeric(series, errors="coerce")
        return f"{v.mean()*100:.1f}\\%"

    def mean_val(series):
        v = pd.to_numeric(series, errors="coerce")
        return f"{v.mean():.1f}"

    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{lccc}")
    latex_lines.append(r"\hline\hline \\[-8pt]")
    latex_lines.append(
        rf"\rowcolor{{yellow!15}} Characteristic & All (N={N_all}) & Urban (N={N_u}) & Rural (N={N_r}) \\ \hline \\[-8pt]"
    )

    # Patient characteristics
    latex_lines.append(r"\textit{Patient characteristics} & & & \\")
    latex_lines.append(
        rf"\quad \% Male & {pct(all_['male'])} & {pct(urban['male'])} & {pct(rural['male'])} \\"
    )
    latex_lines.append(
        rf"\quad Mean age (years) & {mean_val(all_['age_in_years'])} & {mean_val(urban['age_in_years'])} & {mean_val(rural['age_in_years'])} \\"
    )
    latex_lines.append(
        rf"\quad \% HIV positive & {pct(all_['hiv_positive'])} & {pct(urban['hiv_positive'])} & {pct(rural['hiv_positive'])} \\"
    )

    latex_lines.append(r"\\[-4pt]")

    # Disease characteristics
    latex_lines.append(r"\textit{Disease characteristics} & & & \\")
    latex_lines.append(
        rf"\quad \% Extrapulmonary TB & {pct(all_['extrapulmonary'])} & {pct(urban['extrapulmonary'])} & {pct(rural['extrapulmonary'])} \\"
    )
    latex_lines.append(
        rf"\quad \% Bacteriologically confirmed & {pct(all_['bacteriologically_confirmed'])} & {pct(urban['bacteriologically_confirmed'])} & {pct(rural['bacteriologically_confirmed'])} \\"
    )
    latex_lines.append(
        rf"\quad \% Retreatment & {pct(all_['retreatment'])} & {pct(urban['retreatment'])} & {pct(rural['retreatment'])} \\"
    )
    latex_lines.append(
        rf"\quad \% Drug resistant & {pct(all_['drugresistant'])} & {pct(urban['drugresistant'])} & {pct(rural['drugresistant'])} \\"
    )

    latex_lines.append(r"\\[-4pt]")

    # Outcomes
    latex_lines.append(r"\textit{Outcomes} & & & \\")
    latex_lines.append(
        rf"\quad \% Unsuccessful outcome & {pct(all_['unsuccessful_outcome'])} & {pct(urban['unsuccessful_outcome'])} & {pct(rural['unsuccessful_outcome'])} \\"
    )

    latex_lines.append(r"\\[-4pt]")

    # Treatment group breakdown
    latex_lines.append(r"\textit{Treatment group, \%} & & & \\")
    for grp, label in [
        ("Control Group",      "Control"),
        ("Keheala Group",      "Keheala"),
        ("SBCC Group",         "Platform"),
        ("SMS Reminder Group", "SMS"),
    ]:
        a = f"{(all_['treatment_group']==grp).mean()*100:.1f}\\%"
        u = f"{(urban['treatment_group']==grp).mean()*100:.1f}\\%"
        r = f"{(rural['treatment_group']==grp).mean()*100:.1f}\\%"
        latex_lines.append(rf"\quad {label} & {a} & {u} & {r} \\")

    latex_lines.append(r"\hline\hline")
    latex_lines.append(r"\end{tabular}}")

    out_file = os.path.join(OUTPUT_DIR, "tblSI_patient_characteristics.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved {out_file}")

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

    # Save CSV
    correction_counts.to_csv(os.path.join(OUTPUT_DIR, "outcome_corrections.csv"), index=False)

    # Save tex (top 10 only)
    total_corrections = len(corrections)
    top10 = correction_counts.head(10)

    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{llcc}")
    latex_lines.append(r"\hline\hline \\[-8pt]")
    latex_lines.append(r"\rowcolor{yellow!15} TIBU Outcome & Paper Outcome & Count & \% of Corrections \\")
    latex_lines.append(r"\hline \\[-8pt]")
    for _, row in top10.iterrows():
        pct = row["count"] / total_corrections * 100
        latex_lines.append(rf"{row['tibu']} & {row['paper']} & {row['count']} & {pct:.1f}\% \\")
    latex_lines.append(r"\hline \\[-8pt]")
    latex_lines.append(rf"Total corrections & & {total_corrections} & 100.0\% \\")
    latex_lines.append(r"\hline\hline")
    latex_lines.append(r"\end{tabular}}")

    with open(os.path.join(OUTPUT_DIR, "tblSI_outcome_corrections.tex"), "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved outcome_corrections.csv and tblSI_outcome_corrections.tex")


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
        clinic_summary[["clinic_id", "urban", "tibu_patients"]],
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
    df["mismatch"] = (df["uo_tibu"] != df["uo_paper"]).astype(int)
    df["type1"]    = ((df["uo_tibu"] == 0) & (df["uo_paper"] == 1)).astype(int)
    df["type2"]    = ((df["uo_tibu"] == 1) & (df["uo_paper"] == 0)).astype(int)

    # --- CSV: per-clinic stats ---
    clinic_n_tibu = main_df.groupby("clinic_id")["scrn"].count().reset_index(name="n_tibu")
    clinic_stats = (
        df.groupby("clinic_id_num")
        .agg(n_study=("scrn","count"), mismatch_rate=("mismatch","mean"),
             type1_rate=("type1","mean"), type2_rate=("type2","mean"))
        .reset_index()
        .rename(columns={"clinic_id_num": "clinic_id"})
    )
    clinic_stats = clinic_stats.merge(clinic_n_tibu, on="clinic_id", how="left")
    clinic_stats = clinic_stats[["clinic_id","n_tibu","n_study","mismatch_rate","type1_rate","type2_rate"]]

    clinic_stats.to_csv(os.path.join(OUTPUT_DIR, "error_rates_by_clinic.csv"), index=False)
    filtered = clinic_stats[clinic_stats["n_study"] >= 10].sort_values("mismatch_rate", ascending=False)
    filtered.to_csv(os.path.join(OUTPUT_DIR, "error_rates_by_clinic_n10.csv"), index=False)

    total_patients = clinic_stats["n_study"].sum()
    filtered_patients = filtered["n_study"].sum()
    print(f"Clinics total: {clinic_stats.shape[0]}, with n>=10: {filtered.shape[0]}")
    print(f"Matched patients: {total_patients}, in n>=10 clinics: {filtered_patients} ({filtered_patients/total_patients:.1%})")
    print("\nTop 10 clinics by mismatch rate (n>=10):")
    print(filtered.head(10))

    # --- Tex: by urban/rural and clinic size ---
    median_size = df["tibu_patients"].median()
    df["large_clinic"] = (df["tibu_patients"] >= median_size).astype(float)

    def stats(sub):
        return len(sub), sub["mismatch"].mean()*100, sub["type1"].mean()*100, sub["type2"].mean()*100

    groups = [
        ("All",          df),
        ("Urban",        df[df["urban"] == 1]),
        ("Rural",        df[df["urban"] == 0]),
        ("Large clinic", df[df["large_clinic"] == 1]),
        ("Small clinic", df[df["large_clinic"] == 0]),
    ]

    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{lcccc}")
    latex_lines.append(r"\hline\hline \\[-8pt]")
    latex_lines.append(r"\rowcolor{yellow!15} & N & Mismatch rate & Type I rate & Type II rate \\")
    latex_lines.append(r"\hline \\[-8pt]")
    for label, sub in groups:
        n, m, t1, t2 = stats(sub)
        latex_lines.append(rf"{label} & {n} & {m:.1f}\% & {t1:.1f}\% & {t2:.1f}\% \\")
        if label == "All":
            latex_lines.append(r"\hline \\[-8pt]")
            latex_lines.append(r"\textit{By urban/rural} & & & & \\")
        if label == "Rural":
            latex_lines.append(r"\hline \\[-8pt]")
            latex_lines.append(r"\textit{By clinic size} & & & & \\")
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
    df["mismatch"] = (df["uo_tibu"] != df["uo_paper"]).astype(int)
    df["type1"]    = ((df["uo_tibu"] == 0) & (df["uo_paper"] == 1)).astype(int)
    df["type2"]    = ((df["uo_tibu"] == 1) & (df["uo_paper"] == 0)).astype(int)

    df["age_group"] = pd.cut(
        pd.to_numeric(df["age_in_years"], errors="coerce"),
        bins=[0, 15, 35, 55, 200],
        labels=[r"$<$15", "15--34", "35--54", "55+"]
    )

    def stats(sub):
        if len(sub) == 0:
            return "-", "-", "-", "-"
        return (str(len(sub)),
                f"{sub['mismatch'].mean()*100:.1f}\\%",
                f"{sub['type1'].mean()*100:.1f}\\%",
                f"{sub['type2'].mean()*100:.1f}\\%")

    latex_lines = []
    latex_lines.append(r"\scriptsize{")
    latex_lines.append(r"\begin{tabular}{lcccc}")
    latex_lines.append(r"\hline\hline \\[-8pt]")
    latex_lines.append(r"\rowcolor{yellow!15} & N & Mismatch rate & Type I rate & Type II rate \\")
    latex_lines.append(r"\hline \\[-8pt]")

    # Sex
    latex_lines.append(r"\textit{Sex} & & & & \\")
    for val, label in [(1, r"\quad Male"), (0, r"\quad Female")]:
        n, m, t1, t2 = stats(df[df["male"] == val])
        latex_lines.append(rf"{label} & {n} & {m} & {t1} & {t2} \\")

    latex_lines.append(r"\\[-4pt]")

    # Age
    latex_lines.append(r"\textit{Age group} & & & & \\")
    for grp in [r"$<$15", "15--34", "35--54", "55+"]:
        n, m, t1, t2 = stats(df[df["age_group"] == grp])
        latex_lines.append(rf"\quad {grp} & {n} & {m} & {t1} & {t2} \\")

    latex_lines.append(r"\\[-4pt]")

    # HIV
    latex_lines.append(r"\textit{HIV status} & & & & \\")
    for val, label in [(1, r"\quad HIV positive"), (0, r"\quad HIV negative")]:
        n, m, t1, t2 = stats(df[df["hiv_positive"] == val])
        latex_lines.append(rf"{label} & {n} & {m} & {t1} & {t2} \\")

    latex_lines.append(r"\\[-4pt]")

    # Disease characteristics
    latex_lines.append(r"\textit{Disease characteristics} & & & & \\")
    for col, label in [
        ("extrapulmonary",          r"\quad Extrapulmonary TB"),
        ("bacteriologically_confirmed", r"\quad Bacteriologically confirmed"),
        ("retreatment",             r"\quad Retreatment"),
    ]:
        for val, suffix in [(1, "Yes"), (0, "No")]:
            n, m, t1, t2 = stats(df[df[col] == val])
            latex_lines.append(rf"{label} ({suffix}) & {n} & {m} & {t1} & {t2} \\")

    latex_lines.append(r"\hline\hline")
    latex_lines.append(r"\end{tabular}}")

    out_file = os.path.join(OUTPUT_DIR, "tblSI_error_by_patient_char.tex")
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

    generate_patient_characteristics_table(main_df)
    
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
    generate_patient_characteristics_table(main_df)
    generate_crosstab_table(main_df, dqa_df_cleaned)
    generate_error_table(main_df, dqa_df_cleaned)
    generate_error_by_clinic(main_df, dqa_df_cleaned)
    generate_outcome_corrections(main_df, dqa_df_cleaned)
    generate_error_by_outcome(main_df, dqa_df_cleaned)
    generate_error_by_patient_characteristics(main_df, dqa_df_cleaned)

    print("\nAll DQA tables generated successfully.")
    print(main_df["clinic_id"].value_counts().head(10))

if __name__ == "__main__":
    main()