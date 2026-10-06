from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 63
# Combined publication figure for frozen OASIS-3 validation
#
# IMPORTANT:
#   - Reads locked Step-59B and Step-60B summaries only.
#   - Does NOT read MRI voxels.
#   - Does NOT load model checkpoints.
#   - Does NOT rerun inference.
#   - Does NOT modify predictions or endpoints.
#   - Does NOT recalibrate the model.
#   - Visualization only.
# ============================================================


ROOT = (Path(__file__).resolve().parents[2] / "external_data" / "oasis3")

STEP59B = (
    ROOT
    / "external_evaluation"
    / "OASIS3_STEP59B_PARTICIPANT_CLUSTER_BOOTSTRAP_SUMMARY.csv"
)

STEP60B = (
    ROOT
    / "external_evaluation"
    / "OASIS3_STEP60B_CALIBRATION_CLUSTER_BOOTSTRAP_SUMMARY.csv"
)

OUTPUT_DIR = (
    ROOT
    / "external_evaluation"
    / "publication_figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTPUT_DIR / "Figure_External_Validation_Combined.png"
OUT_PDF = OUTPUT_DIR / "Figure_External_Validation_Combined.pdf"
OUT_TABLE = OUTPUT_DIR / "OASIS3_STEP63_COMBINED_FIGURE_DATA.csv"
OUT_SUMMARY = OUTPUT_DIR / "OASIS3_STEP63_COMBINED_FIGURE_SUMMARY.json"
OUT_LOCK = OUTPUT_DIR / "OASIS3_STEP63_COMBINED_FIGURE_LOCK.json"


PRIMARY_HORIZONS = [12, 24, 36]
SECONDARY_HORIZONS = [48, 60]
ALL_HORIZONS = PRIMARY_HORIZONS + SECONDARY_HORIZONS


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label} is missing required columns:\n{missing}"
        )


print("=" * 72)
print("STEP 63 - COMBINED EXTERNAL VALIDATION FIGURE")
print("=" * 72)

print("\nLOCKED INPUTS")
print("Step-59B :", STEP59B)
print("Step-60B :", STEP60B)

if not STEP59B.exists():
    raise FileNotFoundError(STEP59B)

if not STEP60B.exists():
    raise FileNotFoundError(STEP60B)


hash59 = sha256_file(STEP59B)
hash60 = sha256_file(STEP60B)

print("\nINPUT SHA256")
print("Step-59B :", hash59)
print("Step-60B :", hash60)


# ============================================================
# READ LOCKED RESULTS
# ============================================================

perf = pd.read_csv(STEP59B, low_memory=False)
cal = pd.read_csv(STEP60B, low_memory=False)


# Exact Step-59B schema observed in locked file
required_perf = [
    "horizon_months",
    "roc_auc",
    "roc_auc_ci_lower",
    "roc_auc_ci_upper",
    "roc_auc_bootstrap_mean",
    "brier_score",
    "brier_ci_lower",
    "brier_ci_upper",
    "brier_bootstrap_mean",
]

# Exact Step-60B schema observed in locked file
required_cal = [
    "horizon_months",
    "observed_event_rate_estimate",
    "observed_event_rate_ci_lower",
    "observed_event_rate_ci_upper",
    "mean_predicted_risk_estimate",
    "mean_predicted_risk_ci_lower",
    "mean_predicted_risk_ci_upper",
    "calibration_intercept_estimate",
    "calibration_intercept_ci_lower",
    "calibration_intercept_ci_upper",
    "calibration_slope_estimate",
    "calibration_slope_ci_lower",
    "calibration_slope_ci_upper",
    "brier_skill_score_estimate",
    "brier_skill_score_ci_lower",
    "brier_skill_score_ci_upper",
]

require_columns(perf, required_perf, "Step-59B performance summary")
require_columns(cal, required_cal, "Step-60B calibration summary")


# ============================================================
# HORIZON INTEGRITY
# ============================================================

perf["horizon_months"] = pd.to_numeric(
    perf["horizon_months"], errors="raise"
).astype(int)

cal["horizon_months"] = pd.to_numeric(
    cal["horizon_months"], errors="raise"
).astype(int)

if perf["horizon_months"].duplicated().any():
    raise RuntimeError("Duplicate horizons detected in Step-59B.")

if cal["horizon_months"].duplicated().any():
    raise RuntimeError("Duplicate horizons detected in Step-60B.")

perf = perf.set_index("horizon_months")
cal = cal.set_index("horizon_months")

for h in ALL_HORIZONS:
    if h not in perf.index:
        raise RuntimeError(f"Horizon {h} missing from Step-59B.")
    if h not in cal.index:
        raise RuntimeError(f"Horizon {h} missing from Step-60B.")

perf = perf.loc[ALL_HORIZONS].copy()
cal = cal.loc[ALL_HORIZONS].copy()


# ============================================================
# NUMERIC EXTRACTION
# ============================================================

horizons = np.array(ALL_HORIZONS, dtype=float)

auc = perf["roc_auc"].astype(float).to_numpy()
auc_lo = perf["roc_auc_ci_lower"].astype(float).to_numpy()
auc_hi = perf["roc_auc_ci_upper"].astype(float).to_numpy()

brier = perf["brier_score"].astype(float).to_numpy()
brier_lo = perf["brier_ci_lower"].astype(float).to_numpy()
brier_hi = perf["brier_ci_upper"].astype(float).to_numpy()

observed = cal["observed_event_rate_estimate"].astype(float).to_numpy()
observed_lo = cal["observed_event_rate_ci_lower"].astype(float).to_numpy()
observed_hi = cal["observed_event_rate_ci_upper"].astype(float).to_numpy()

predicted = cal["mean_predicted_risk_estimate"].astype(float).to_numpy()
predicted_lo = cal["mean_predicted_risk_ci_lower"].astype(float).to_numpy()
predicted_hi = cal["mean_predicted_risk_ci_upper"].astype(float).to_numpy()

slope = cal["calibration_slope_estimate"].astype(float).to_numpy()
slope_lo = cal["calibration_slope_ci_lower"].astype(float).to_numpy()
slope_hi = cal["calibration_slope_ci_upper"].astype(float).to_numpy()


# ============================================================
# CONSISTENCY CHECKS
# ============================================================

for name, estimate, lower, upper in [
    ("AUC", auc, auc_lo, auc_hi),
    ("Brier", brier, brier_lo, brier_hi),
    ("Observed risk", observed, observed_lo, observed_hi),
    ("Predicted risk", predicted, predicted_lo, predicted_hi),
    ("Calibration slope", slope, slope_lo, slope_hi),
]:
    if not np.all(np.isfinite(estimate)):
        raise RuntimeError(f"Non-finite values in {name} estimates.")

    if not np.all(np.isfinite(lower)):
        raise RuntimeError(f"Non-finite values in {name} lower CI.")

    if not np.all(np.isfinite(upper)):
        raise RuntimeError(f"Non-finite values in {name} upper CI.")

    if np.any(lower > estimate):
        raise RuntimeError(f"Invalid lower CI detected for {name}.")

    if np.any(upper < estimate):
        raise RuntimeError(f"Invalid upper CI detected for {name}.")


# ============================================================
# BUILD PUBLICATION DATA TABLE
# ============================================================

combined = pd.DataFrame(
    {
        "horizon_months": ALL_HORIZONS,
        "analysis_role": [
            "Primary",
            "Primary",
            "Primary",
            "Secondary",
            "Secondary",
        ],
        "auc": auc,
        "auc_lower": auc_lo,
        "auc_upper": auc_hi,
        "brier": brier,
        "brier_lower": brier_lo,
        "brier_upper": brier_hi,
        "observed_risk": observed,
        "observed_risk_lower": observed_lo,
        "observed_risk_upper": observed_hi,
        "predicted_risk": predicted,
        "predicted_risk_lower": predicted_lo,
        "predicted_risk_upper": predicted_hi,
        "calibration_slope": slope,
        "calibration_slope_lower": slope_lo,
        "calibration_slope_upper": slope_hi,
    }
)

combined.to_csv(OUT_TABLE, index=False)


# ============================================================
# ERROR BAR HELPERS
# ============================================================

auc_yerr = np.vstack(
    [
        auc - auc_lo,
        auc_hi - auc,
    ]
)

obs_yerr = np.vstack(
    [
        observed - observed_lo,
        observed_hi - observed,
    ]
)

pred_yerr = np.vstack(
    [
        predicted - predicted_lo,
        predicted_hi - predicted,
    ]
)

slope_yerr = np.vstack(
    [
        slope - slope_lo,
        slope_hi - slope,
    ]
)


# ============================================================
# CREATE COMBINED FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.8),
)

ax1, ax2, ax3 = axes


# ------------------------------------------------------------
# Panel A: discrimination
# ------------------------------------------------------------

ax1.errorbar(
    horizons,
    auc,
    yerr=auc_yerr,
    marker="o",
    linewidth=2,
    capsize=4,
)

ax1.axhline(
    0.5,
    linestyle="--",
    linewidth=1.5,
    label="Chance AUC = 0.5",
)

ax1.set_title("(A) External discrimination")
ax1.set_xlabel("Prediction horizon (months)")
ax1.set_ylabel("Time-dependent ROC AUC")
ax1.set_xticks(ALL_HORIZONS)
ax1.set_ylim(0.35, 0.65)
ax1.grid(axis="y", alpha=0.25)
ax1.legend(frameon=False)


# ------------------------------------------------------------
# Panel B: observed versus predicted risk
# ------------------------------------------------------------

ax2.errorbar(
    horizons,
    observed,
    yerr=obs_yerr,
    marker="o",
    linewidth=2,
    capsize=4,
    label="Observed event rate",
)

ax2.errorbar(
    horizons,
    predicted,
    yerr=pred_yerr,
    marker="s",
    linewidth=2,
    capsize=4,
    label="Mean predicted risk",
)

ax2.set_title("(B) Observed versus predicted risk")
ax2.set_xlabel("Prediction horizon (months)")
ax2.set_ylabel("Risk")
ax2.set_xticks(ALL_HORIZONS)
ax2.set_ylim(bottom=0)
ax2.grid(axis="y", alpha=0.25)
ax2.legend(frameon=False)


# ------------------------------------------------------------
# Panel C: calibration slope
# ------------------------------------------------------------

ax3.errorbar(
    horizons,
    slope,
    yerr=slope_yerr,
    marker="o",
    linewidth=2,
    capsize=4,
)

ax3.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
    label="Ideal slope = 1",
)

ax3.axhline(
    0.0,
    linestyle=":",
    linewidth=1.25,
)

ax3.set_title("(C) External calibration slope")
ax3.set_xlabel("Prediction horizon (months)")
ax3.set_ylabel("Calibration slope")
ax3.set_xticks(ALL_HORIZONS)

lower_limit = min(-0.75, float(np.min(slope_lo)) - 0.05)
ax3.set_ylim(lower_limit, 1.10)

ax3.grid(axis="y", alpha=0.25)
ax3.legend(frameon=False)


# ============================================================
# GLOBAL FIGURE FORMATTING
# ============================================================

fig.suptitle(
    "Frozen ADNI model external validation on OASIS-3",
    fontsize=17,
    y=1.02,
)

fig.tight_layout()

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# SUMMARY / LOCK
# ============================================================

summary = {
    "step": 63,
    "status": "COMPLETE",
    "analysis": "combined_external_validation_publication_figure",
    "source_step59b": str(STEP59B),
    "source_step60b": str(STEP60B),
    "source_step59b_sha256": hash59,
    "source_step60b_sha256": hash60,
    "primary_horizons_months": PRIMARY_HORIZONS,
    "secondary_horizons_months": SECONDARY_HORIZONS,
    "figure_panels": {
        "A": "time-dependent ROC AUC with participant-cluster bootstrap 95% CI",
        "B": "observed event rate versus mean predicted risk with participant-cluster bootstrap 95% CI",
        "C": "calibration slope with participant-cluster bootstrap 95% CI",
    },
    "output_png": str(OUT_PNG),
    "output_pdf": str(OUT_PDF),
    "output_table": str(OUT_TABLE),
    "model_modified": False,
    "predictions_modified": False,
    "endpoints_modified": False,
    "model_recalibrated": False,
    "threshold_optimized": False,
    "oasis_based_model_selection": False,
    "mri_voxels_read": False,
    "checkpoints_loaded": False,
    "inference_rerun": False,
}

with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

lock = {
    "step": 63,
    "status": "LOCKED",
    "source_step59b_sha256": hash59,
    "source_step60b_sha256": hash60,
    "combined_table_sha256": sha256_file(OUT_TABLE),
    "figure_png_sha256": sha256_file(OUT_PNG),
    "figure_pdf_sha256": sha256_file(OUT_PDF),
    "summary_sha256": sha256_file(OUT_SUMMARY),
    "visualization_only": True,
    "primary_external_validation_unchanged": True,
}

with open(OUT_LOCK, "w", encoding="utf-8") as f:
    json.dump(lock, f, indent=2)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 72)
print("COMBINED EXTERNAL VALIDATION DATA")
print("=" * 72)

display_cols = [
    "horizon_months",
    "analysis_role",
    "auc",
    "auc_lower",
    "auc_upper",
    "observed_risk",
    "predicted_risk",
    "calibration_slope",
]

print(
    combined[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\n" + "=" * 72)
print("OUTPUT")
print("=" * 72)

print("Combined PNG :", OUT_PNG)
print("Combined PDF :", OUT_PDF)
print("Data table   :", OUT_TABLE)
print("Summary      :", OUT_SUMMARY)
print("Lock         :", OUT_LOCK)

print("\nSHA256")
print("PNG     :", sha256_file(OUT_PNG))
print("PDF     :", sha256_file(OUT_PDF))
print("Table   :", sha256_file(OUT_TABLE))
print("Summary :", sha256_file(OUT_SUMMARY))
print("Lock    :", sha256_file(OUT_LOCK))

print("\n" + "=" * 72)
print("IMPORTANT")
print("=" * 72)

print("MRI VOXELS READ              : NO")
print("FROZEN CHECKPOINTS LOADED    : NO")
print("MODEL INFERENCE RE-RUN       : NO")
print("EXTERNAL PREDICTIONS MODIFIED: NO")
print("EXTERNAL ENDPOINTS MODIFIED  : NO")
print("MODEL TRAINING               : NO")
print("MODEL FINE-TUNING            : NO")
print("MODEL RECALIBRATION          : NO")
print("THRESHOLD OPTIMIZATION       : NO")
print("OASIS-BASED MODEL SELECTION  : NO")
print("PUBLICATION VISUALIZATION    : YES")

print("\nSTEP 63 COMPLETE.")
print(
    "A combined publication figure has been generated from the "
    "locked Step-59B and Step-60B external-validation results."
)
print(
    "The frozen ADNI model, predictions, endpoints, and primary "
    "OASIS-3 validation results remain unchanged."
)