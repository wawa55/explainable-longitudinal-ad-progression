from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "08_results"
BENCH_DIR = RESULT_DIR / "benchmark_test_v1" / "statistical_comparison_v1"
XAI_DIR = RESULT_DIR / "ADNI_XAI_TEST_PRODUCTION_V1"
OUT_DIR = RESULT_DIR / "publication_package_v1"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

BENCH_TABLE = BENCH_DIR / "ADNI_BENCHMARK_PUBLICATION_TABLE_V1.csv"
PAIRED_TABLE = BENCH_DIR / "ADNI_BENCHMARK_PAIRED_BOOTSTRAP_COMPARISONS_V1.csv"
BENCH_AUDIT = BENCH_DIR / "ADNI_BENCHMARK_STATISTICAL_COMPARISON_AUDIT_V1.json"
XAI_STAB = XAI_DIR / "ADNI_XAI_CROSS_SEED_COHORT_SUMMARY_V1.csv"
XAI_FAITH = XAI_DIR / "ADNI_XAI_FAITHFULNESS_COHORT_SUMMARY_V1.csv"
XAI_TEMP = XAI_DIR / "ADNI_XAI_INTERNAL_TEMPORAL_COHORT_SUMMARY_V1.csv"

REQUIRED = [BENCH_TABLE, PAIRED_TABLE, BENCH_AUDIT, XAI_STAB, XAI_FAITH, XAI_TEMP]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def save_table(df, stem):
    df.to_csv(TAB_DIR / f"{stem}.csv", index=False)
    (TAB_DIR / f"{stem}.tex").write_text(
        df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.3f}"),
        encoding="utf-8"
    )

def main():
    print("="*110)
    print("ADNI — PUBLICATION RESULTS PACKAGE")
    print("READ-ONLY: NO TRAINING / NO MODEL INFERENCE / NO IG RERUN")
    print("="*110)

    missing = [str(p) for p in REQUIRED if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    audit = json.loads(BENCH_AUDIT.read_text(encoding="utf-8"))
    if str(audit.get("status","")).upper() != "COMPLETE":
        raise RuntimeError("Benchmark statistical audit is not COMPLETE.")

    bench = pd.read_csv(BENCH_TABLE)
    paired = pd.read_csv(PAIRED_TABLE)
    stab = pd.read_csv(XAI_STAB)
    faith = pd.read_csv(XAI_FAITH)
    temp = pd.read_csv(XAI_TEMP)

    t1 = bench[["metric_display","direction","B0_formatted","B1_formatted","B2_formatted"]].copy()
    t1.columns = ["Metric","Direction","B0","B1","B2"]
    save_table(t1, "Table_1_Frozen_Test_Benchmark_Performance")

    t2 = bench[[
        "metric_display",
        "B1_vs_B0_favorable_difference","B1_vs_B0_ci_2_5","B1_vs_B0_ci_97_5","B1_vs_B0_probability_better","B1_vs_B0_inference",
        "B2_vs_B0_favorable_difference","B2_vs_B0_ci_2_5","B2_vs_B0_ci_97_5","B2_vs_B0_probability_better","B2_vs_B0_inference",
        "B2_vs_B1_favorable_difference","B2_vs_B1_ci_2_5","B2_vs_B1_ci_97_5","B2_vs_B1_probability_better","B2_vs_B1_inference"
    ]].copy()
    save_table(t2, "Table_2_Paired_Bootstrap_Comparisons")

    t3 = stab[[
        "horizon_months","subjects","median_subject_mean_spearman",
        "q1_subject_mean_spearman","q3_subject_mean_spearman",
        "median_subject_mean_dice_1pct","median_subject_mean_dice_5pct",
        "median_subject_mean_dice_10pct","min_subject_mean_spearman"
    ]].copy()
    save_table(t3, "Table_3_XAI_Cross_Seed_Stability")

    t4 = faith[[
        "horizon_months","top_k_percent","subjects",
        "median_subject_targeted_drop","median_subject_random_drop",
        "median_subject_faithfulness_advantage","fraction_subjects_positive_advantage",
        "median_targeted_percentile_vs_random"
    ]].copy()
    save_table(t4, "Table_4_XAI_Faithfulness")

    t5 = temp[[
        "horizon_months","subjects","median_latest_visit_contribution",
        "q1_latest_visit_contribution","q3_latest_visit_contribution",
        "median_latest_visit_absolute_share","q1_latest_visit_absolute_share",
        "q3_latest_visit_absolute_share",
        "fraction_subjects_latest_majority_seed_most_influential",
        "median_subject_recency_abs_corr"
    ]].copy()
    save_table(t5, "Table_5_XAI_Temporal_Visit_Analysis")

    # Figure 1: AUC
    months = [12,24,36]
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    for model in ["B0","B1","B2"]:
        vals=[]; los=[]; his=[]
        for m in months:
            r = bench.loc[bench["metric"].eq(f"auc_{m}m")].iloc[0]
            vals.append(float(r[f"{model}_estimate"]))
            los.append(float(r[f"{model}_ci_2_5"]))
            his.append(float(r[f"{model}_ci_97_5"]))
        vals=np.array(vals); los=np.array(los); his=np.array(his)
        ax.errorbar(months, vals, yerr=np.vstack([vals-los, his-vals]), marker="o", capsize=3, label=model)
    ax.set_xlabel("Prediction horizon (months)")
    ax.set_ylabel("Cumulative/dynamic AUC")
    ax.set_xticks(months); ax.set_ylim(0,1); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"Figure_1_Frozen_Test_AUC.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR/"Figure_1_Frozen_Test_AUC.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Brier
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    for model in ["B0","B1","B2"]:
        vals=[]; los=[]; his=[]
        for m in months:
            r = bench.loc[bench["metric"].eq(f"brier_{m}m")].iloc[0]
            vals.append(float(r[f"{model}_estimate"]))
            los.append(float(r[f"{model}_ci_2_5"]))
            his.append(float(r[f"{model}_ci_97_5"]))
        vals=np.array(vals); los=np.array(los); his=np.array(his)
        ax.errorbar(months, vals, yerr=np.vstack([vals-los, his-vals]), marker="o", capsize=3, label=model)
    ax.set_xlabel("Prediction horizon (months)")
    ax.set_ylabel("IPCW Brier score")
    ax.set_xticks(months); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"Figure_2_Frozen_Test_Brier.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR/"Figure_2_Frozen_Test_Brier.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: stability
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    ax.plot(stab["horizon_months"], stab["median_subject_mean_spearman"], marker="o", label="Spearman")
    ax.plot(stab["horizon_months"], stab["median_subject_mean_dice_1pct"], marker="o", label="Dice top 1%")
    ax.plot(stab["horizon_months"], stab["median_subject_mean_dice_5pct"], marker="o", label="Dice top 5%")
    ax.plot(stab["horizon_months"], stab["median_subject_mean_dice_10pct"], marker="o", label="Dice top 10%")
    ax.set_xlabel("Prediction horizon (months)")
    ax.set_ylabel("Cross-seed agreement")
    ax.set_xticks(sorted(stab["horizon_months"].unique())); ax.set_ylim(0,1)
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(FIG_DIR/"Figure_3_XAI_Cross_Seed_Stability.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR/"Figure_3_XAI_Cross_Seed_Stability.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: faithfulness
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    for h in sorted(faith["horizon_months"].unique()):
        s = faith.loc[faith["horizon_months"].eq(h)].sort_values("top_k_percent")
        ax.plot(s["top_k_percent"], s["median_subject_faithfulness_advantage"], marker="o", label=f"{int(h)} months")
    ax.set_xlabel("Top attributed voxels perturbed (%)")
    ax.set_ylabel("Median faithfulness advantage")
    ax.set_xticks(sorted(faith["top_k_percent"].unique()))
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(FIG_DIR/"Figure_4_XAI_Faithfulness_Advantage.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR/"Figure_4_XAI_Faithfulness_Advantage.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 5: temporal contribution
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    ax.bar(temp["horizon_months"].astype(int).astype(str), temp["median_latest_visit_absolute_share"])
    ax.set_xlabel("Prediction horizon (months)")
    ax.set_ylabel("Median latest-visit absolute attribution share")
    ax.set_ylim(0,1); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(FIG_DIR/"Figure_5_XAI_Latest_Visit_Contribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR/"Figure_5_XAI_Latest_Visit_Contribution.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "status":"COMPLETE",
        "read_only_publication_packaging": True,
        "training_performed": False,
        "model_inference_performed": False,
        "integrated_gradients_rerun": False,
        "rows":{
            "benchmark":len(bench),
            "paired":len(paired),
            "xai_stability":len(stab),
            "xai_faithfulness":len(faith),
            "xai_temporal":len(temp)
        },
        "source_sha256":{str(p):sha256(p) for p in REQUIRED},
        "tables_directory":str(TAB_DIR),
        "figures_directory":str(FIG_DIR)
    }
    (OUT_DIR/"ADNI_PUBLICATION_PACKAGE_AUDIT_V1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "ADNI PUBLICATION RESULTS PACKAGE — SUMMARY",
        "="*80,
        f"Benchmark rows: {len(bench)}",
        f"Paired comparison rows: {len(paired)}",
        f"XAI stability rows: {len(stab)}",
        f"XAI faithfulness rows: {len(faith)}",
        f"XAI temporal rows: {len(temp)}",
        "",
        "No training performed.",
        "No model inference performed.",
        "No Integrated Gradients rerun."
    ]
    (OUT_DIR/"ADNI_PUBLICATION_RESULTS_SUMMARY_V1.txt").write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("="*110)
    print("PUBLICATION RESULTS PACKAGE: COMPLETE")
    print("="*110)
    print("Tables :", TAB_DIR)
    print("Figures:", FIG_DIR)
    print("Audit  :", OUT_DIR/"ADNI_PUBLICATION_PACKAGE_AUDIT_V1.json")
    print("Summary:", OUT_DIR/"ADNI_PUBLICATION_RESULTS_SUMMARY_V1.txt")
    print("No training performed.")
    print("No model inference performed.")
    print("No Integrated Gradients rerun.")
    print("="*110)

if __name__ == "__main__":
    main()
