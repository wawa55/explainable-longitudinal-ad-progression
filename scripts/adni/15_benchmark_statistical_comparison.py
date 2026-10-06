from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "08_results" / "benchmark_test_v1"
OUT_DIR = IN_DIR / "statistical_comparison_v1"
FIG_DIR = OUT_DIR / "figures"

METRICS_FILE = IN_DIR / "ADNI_FROZEN_BENCHMARK_TEST_METRICS_V1.csv"
BOOT_FILE = IN_DIR / "ADNI_FROZEN_BENCHMARK_TEST_BOOTSTRAP_CI_V1.csv"
ENSEMBLE_FILE = IN_DIR / "ADNI_FROZEN_BENCHMARK_TEST_ENSEMBLE_PREDICTIONS_V1.csv"
AUDIT_FILE = IN_DIR / "ADNI_FROZEN_BENCHMARK_TEST_AUDIT_V1.json"
STATUS_FILE = IN_DIR / "ADNI_FROZEN_BENCHMARK_TEST_STATUS_V1.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAIRED_OUT = OUT_DIR / "ADNI_BENCHMARK_PAIRED_BOOTSTRAP_COMPARISONS_V1.csv"
FINAL_TABLE_OUT = OUT_DIR / "ADNI_BENCHMARK_PUBLICATION_TABLE_V1.csv"
SUMMARY_OUT = OUT_DIR / "ADNI_BENCHMARK_STATISTICAL_SUMMARY_V1.txt"
AUDIT_OUT = OUT_DIR / "ADNI_BENCHMARK_STATISTICAL_COMPARISON_AUDIT_V1.json"

MODELS = ["B0", "B1", "B2"]
COMPARISONS = [("B1", "B0"), ("B2", "B0"), ("B2", "B1")]
PRIMARY_METRICS = [
    "auc_12m","auc_24m","auc_36m",
    "brier_12m","brier_24m","brier_36m",
    "uno_c_60m","harrell_c_60m","integrated_brier_0_60m",
]
HIGHER_BETTER = {
    "auc_12m","auc_24m","auc_36m","auc_48m","auc_60m",
    "uno_c_60m","harrell_c_60m",
}
LOWER_BETTER = {
    "brier_12m","brier_24m","brier_36m","brier_48m","brier_60m",
    "integrated_brier_0_60m",
}
DISPLAY_NAMES = {
    "auc_12m":"AUC 12m","auc_24m":"AUC 24m","auc_36m":"AUC 36m",
    "brier_12m":"Brier 12m","brier_24m":"Brier 24m","brier_36m":"Brier 36m",
    "uno_c_60m":"Uno C-index 60m","harrell_c_60m":"Harrell C-index 60m",
    "integrated_brier_0_60m":"Integrated Brier 0-60m",
}

def now_utc():
    return datetime.now(timezone.utc).isoformat()

def percentile_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, 0
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi), int(len(x))

def superiority_probability(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x > 0)) if len(x) else np.nan

def comparison_label(lo, hi):
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "UNDEFINED"
    if lo > 0:
        return "SUPPORTED"
    if hi < 0:
        return "SUPPORTED AGAINST"
    return "NO CLEAR DIFFERENCE"

def format_est_ci(est, lo, hi):
    if not all(np.isfinite([est, lo, hi])):
        return "NA"
    return f"{est:.3f} ({lo:.3f}, {hi:.3f})"

def main():
    print("=" * 110)
    print("ADNI — STEP 15: FROZEN BENCHMARK STATISTICAL COMPARISON")
    print("=" * 110)

    for p in [METRICS_FILE, BOOT_FILE, ENSEMBLE_FILE, AUDIT_FILE, STATUS_FILE]:
        if not p.exists():
            raise FileNotFoundError(f"Required frozen Step 14 output missing:\n{p}")

    status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE":
        raise RuntimeError("Frozen Step 14 TEST evaluation is not marked COMPLETE.")

    metrics = pd.read_csv(METRICS_FILE)
    boot = pd.read_csv(BOOT_FILE)

    req_metric_cols = {"model","metric","estimate","ci_2_5","ci_97_5"}
    req_boot_cols = {"bootstrap_id","model","metric","value"}
    if not req_metric_cols.issubset(metrics.columns):
        raise RuntimeError(f"Metrics file missing columns: {sorted(req_metric_cols-set(metrics.columns))}")
    if not req_boot_cols.issubset(boot.columns):
        raise RuntimeError(f"Bootstrap file missing columns: {sorted(req_boot_cols-set(boot.columns))}")

    ids = sorted(boot["bootstrap_id"].dropna().astype(int).unique())
    if len(ids) != 2000:
        raise RuntimeError(f"Expected 2000 bootstrap replicates, found {len(ids)}.")
    print(f"Paired bootstrap replicates verified: {len(ids)}")

    paired_rows = []
    all_metrics = sorted(set(metrics["metric"].astype(str)).intersection(set(boot["metric"].astype(str))))

    for metric in all_metrics:
        if metric not in HIGHER_BETTER and metric not in LOWER_BETTER:
            continue

        point = {
            row["model"]: float(row["estimate"])
            for _, row in metrics.loc[metrics["metric"].astype(str)==metric].iterrows()
        }

        wide = (
            boot.loc[boot["metric"].astype(str)==metric, ["bootstrap_id","model","value"]]
            .pivot(index="bootstrap_id", columns="model", values="value")
            .sort_index()
        )

        for a, b in COMPARISONS:
            raw = wide[a].to_numpy(dtype=float) - wide[b].to_numpy(dtype=float)
            favorable = raw.copy() if metric in HIGHER_BETTER else -raw
            raw_lo, raw_hi, n_raw = percentile_ci(raw)
            fav_lo, fav_hi, n_fav = percentile_ci(favorable)
            point_raw = point[a] - point[b]
            point_fav = point_raw if metric in HIGHER_BETTER else -point_raw

            paired_rows.append({
                "metric": metric,
                "metric_display": DISPLAY_NAMES.get(metric, metric),
                "direction": "higher is better" if metric in HIGHER_BETTER else "lower is better",
                "model_a": a,
                "model_b": b,
                "comparison": f"{a} vs {b}",
                "estimate_model_a": point[a],
                "estimate_model_b": point[b],
                "raw_difference_a_minus_b": point_raw,
                "raw_ci_2_5": raw_lo,
                "raw_ci_97_5": raw_hi,
                "favorable_difference_for_model_a": point_fav,
                "favorable_ci_2_5": fav_lo,
                "favorable_ci_97_5": fav_hi,
                "bootstrap_valid_replicates": min(n_raw, n_fav),
                "bootstrap_probability_model_a_better": superiority_probability(favorable),
                "inference": comparison_label(fav_lo, fav_hi),
            })

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(PAIRED_OUT, index=False)

    pub_rows = []
    for metric in PRIMARY_METRICS:
        m = metrics.loc[metrics["metric"].astype(str)==metric].copy()
        if len(m)==0:
            continue
        row = {
            "metric": metric,
            "metric_display": DISPLAY_NAMES.get(metric, metric),
            "direction": "higher is better" if metric in HIGHER_BETTER else "lower is better",
        }
        for model in MODELS:
            s = m.loc[m["model"].astype(str)==model]
            if len(s)!=1:
                raise RuntimeError(f"Expected one row for {model}/{metric}")
            r = s.iloc[0]
            row[f"{model}_estimate"] = float(r["estimate"])
            row[f"{model}_ci_2_5"] = float(r["ci_2_5"])
            row[f"{model}_ci_97_5"] = float(r["ci_97_5"])
            row[f"{model}_formatted"] = format_est_ci(float(r["estimate"]), float(r["ci_2_5"]), float(r["ci_97_5"]))

        for a, b in COMPARISONS:
            s = paired.loc[(paired["metric"]==metric)&(paired["model_a"]==a)&(paired["model_b"]==b)]
            if len(s)==1:
                r=s.iloc[0]
                p=f"{a}_vs_{b}"
                row[f"{p}_favorable_difference"] = float(r["favorable_difference_for_model_a"])
                row[f"{p}_ci_2_5"] = float(r["favorable_ci_2_5"])
                row[f"{p}_ci_97_5"] = float(r["favorable_ci_97_5"])
                row[f"{p}_probability_better"] = float(r["bootstrap_probability_model_a_better"])
                row[f"{p}_inference"] = str(r["inference"])
        pub_rows.append(row)

    pd.DataFrame(pub_rows).to_csv(FINAL_TABLE_OUT, index=False)

    lines = [
        "ADNI FROZEN BENCHMARK — STEP 15 STATISTICAL COMPARISON",
        "="*80,
        f"Created UTC: {now_utc()}",
        "Paired uncertainty source: same 2,000 participant-cluster bootstrap replicates from Step 14.",
        "",
        "Formal superiority is claimed only when the paired favorable-difference 95% bootstrap CI excludes zero.",
        "",
        "PRIMARY PAIRED COMPARISONS",
        "-"*80,
    ]

    pprimary = paired.loc[paired["metric"].isin(PRIMARY_METRICS)].copy()
    for metric in PRIMARY_METRICS:
        ss = pprimary.loc[pprimary["metric"]==metric]
        if len(ss)==0:
            continue
        lines.append(DISPLAY_NAMES.get(metric, metric))
        for _, r in ss.iterrows():
            lines.append(
                f"  {r['model_a']} vs {r['model_b']}: favorable difference="
                f"{r['favorable_difference_for_model_a']:.4f} "
                f"[95% CI {r['favorable_ci_2_5']:.4f}, {r['favorable_ci_97_5']:.4f}], "
                f"P(bootstrap superiority)={r['bootstrap_probability_model_a_better']:.3f}, "
                f"{r['inference']}"
            )
        lines.append("")
    SUMMARY_OUT.write_text("\n".join(lines), encoding="utf-8")

    # AUC figure
    months = np.array([12,24,36])
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    for model in MODELS:
        vals=[]; lo=[]; hi=[]
        for month in months:
            s=metrics.loc[(metrics["model"].astype(str)==model)&(metrics["metric"].astype(str)==f"auc_{month}m")].iloc[0]
            vals.append(float(s["estimate"])); lo.append(float(s["ci_2_5"])); hi.append(float(s["ci_97_5"]))
        vals=np.asarray(vals); lo=np.asarray(lo); hi=np.asarray(hi)
        ax.errorbar(months, vals, yerr=np.vstack([vals-lo, hi-vals]), marker="o", capsize=3, label=model)
    ax.set_xlabel("Prediction horizon (months)")
    ax.set_ylabel("Cumulative/dynamic AUC")
    ax.set_xticks(months)
    ax.set_ylim(0,1)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"Figure_AUC_12_24_36m.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR/"Figure_AUC_12_24_36m.pdf", bbox_inches="tight")
    plt.close(fig)

    # Brier figure
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    for model in MODELS:
        vals=[]; lo=[]; hi=[]
        for month in months:
            s=metrics.loc[(metrics["model"].astype(str)==model)&(metrics["metric"].astype(str)==f"brier_{month}m")].iloc[0]
            vals.append(float(s["estimate"])); lo.append(float(s["ci_2_5"])); hi.append(float(s["ci_97_5"]))
        vals=np.asarray(vals); lo=np.asarray(lo); hi=np.asarray(hi)
        ax.errorbar(months, vals, yerr=np.vstack([vals-lo, hi-vals]), marker="o", capsize=3, label=model)
    ax.set_xlabel("Prediction horizon (months)")
    ax.set_ylabel("IPCW Brier score")
    ax.set_xticks(months)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"Figure_Brier_12_24_36m.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR/"Figure_Brier_12_24_36m.pdf", bbox_inches="tight")
    plt.close(fig)

    # Favorable paired differences vs B0
    order=PRIMARY_METRICS
    vs_b0=paired.loc[paired["metric"].isin(order)&paired["model_b"].eq("B0")].copy()
    fig, ax = plt.subplots(figsize=(8.5,6.4))
    offsets={"B1":-0.12,"B2":0.12}
    for model in ["B1","B2"]:
        xs=[]; xlo=[]; xhi=[]; ys=[]
        for i, metric in enumerate(order):
            s=vs_b0.loc[(vs_b0["model_a"]==model)&(vs_b0["metric"]==metric)]
            if len(s)!=1:
                continue
            r=s.iloc[0]
            est=float(r["favorable_difference_for_model_a"])
            lo=float(r["favorable_ci_2_5"]); hi=float(r["favorable_ci_97_5"])
            xs.append(est); xlo.append(est-lo); xhi.append(hi-est); ys.append(i+offsets[model])
        ax.errorbar(xs, ys, xerr=np.vstack([xlo,xhi]), fmt="o", capsize=3, label=f"{model} vs B0")
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in order])
    ax.set_xlabel("Favorable paired difference (positive = benchmark model better than B0)")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"Figure_Paired_Differences_vs_B0.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR/"Figure_Paired_Differences_vs_B0.pdf", bbox_inches="tight")
    plt.close(fig)

    audit = {
        "status":"COMPLETE",
        "created_utc":now_utc(),
        "paired_bootstrap_replicates":2000,
        "comparisons":[f"{a} vs {b}" for a,b in COMPARISONS],
        "rule":"Paired percentile 95% confidence intervals from same-resample bootstrap differences.",
        "outputs":{
            "paired_comparisons":str(PAIRED_OUT),
            "publication_table":str(FINAL_TABLE_OUT),
            "summary_text":str(SUMMARY_OUT),
            "figure_directory":str(FIG_DIR),
        },
    }
    AUDIT_OUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("")
    print("="*110)
    print("STEP 15 COMPLETE")
    print("="*110)
    print(f"Paired comparisons: {PAIRED_OUT}")
    print(f"Publication table:  {FINAL_TABLE_OUT}")
    print(f"Summary:            {SUMMARY_OUT}")
    print(f"Figures:            {FIG_DIR}")
    print(f"Audit:              {AUDIT_OUT}")
    print("")
    show=paired.loc[paired["metric"].isin(PRIMARY_METRICS),[
        "metric_display","comparison","favorable_difference_for_model_a",
        "favorable_ci_2_5","favorable_ci_97_5",
        "bootstrap_probability_model_a_better","inference"
    ]]
    print(show.to_string(index=False))

if __name__ == "__main__":
    main()
