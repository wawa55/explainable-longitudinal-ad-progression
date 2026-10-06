# ============================================================
# STEP 62A
# PUBLICATION-QUALITY OASIS-3 EXTERNAL VALIDATION FIGURES
#
# IMPORTANT:
#   - Reads ONLY locked Step-61 / Step-60B / Step-59B outputs.
#   - Does NOT load MRI voxels.
#   - Does NOT load model checkpoints.
#   - Does NOT run inference.
#   - Does NOT modify predictions or endpoints.
#   - Does NOT recalibrate the model.
#   - Does NOT optimize thresholds.
#
# Primary horizons:   12, 24, 36 months
# Secondary horizons: 48, 60 months
# ============================================================

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

EVAL_DIR = ROOT / "external_evaluation"

STEP61 = (
    EVAL_DIR
    / "OASIS3_STEP61_FINAL_LOCKED_EXTERNAL_VALIDATION_SYNTHESIS.csv"
)

STEP59B = (
    EVAL_DIR
    / "OASIS3_STEP59B_PARTICIPANT_CLUSTER_BOOTSTRAP_SUMMARY.csv"
)

STEP60B = (
    EVAL_DIR
    / "OASIS3_STEP60B_CALIBRATION_CLUSTER_BOOTSTRAP_SUMMARY.csv"
)

STEP61_LOCK = (
    EVAL_DIR
    / "OASIS3_STEP61_FINAL_EXTERNAL_VALIDATION_LOCK.json"
)

OUT_DIR = EVAL_DIR / "publication_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

FIG1 = OUT_DIR / "Figure_External_Discrimination_AUC.png"
FIG1_PDF = OUT_DIR / "Figure_External_Discrimination_AUC.pdf"

FIG2 = OUT_DIR / "Figure_External_Observed_vs_Predicted_Risk.png"
FIG2_PDF = OUT_DIR / "Figure_External_Observed_vs_Predicted_Risk.pdf"

FIG3 = OUT_DIR / "Figure_External_Calibration_Slope.png"
FIG3_PDF = OUT_DIR / "Figure_External_Calibration_Slope.pdf"

TABLE = OUT_DIR / "OASIS3_EXTERNAL_VALIDATION_PUBLICATION_TABLE.csv"

SUMMARY = OUT_DIR / "OASIS3_STEP62A_PUBLICATION_FIGURES_SUMMARY.json"

LOCK = OUT_DIR / "OASIS3_STEP62A_PUBLICATION_FIGURES_LOCK.json"


# ============================================================
# CONSTANTS
# ============================================================

PRIMARY_HORIZONS = [12, 24, 36]
SECONDARY_HORIZONS = [48, 60]
ALL_HORIZONS = [12, 24, 36, 48, 60]

DPI = 600


# ============================================================
# HELPERS
# ============================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required locked input missing:\n{path}"
        )


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        "Could not find any of these columns:\n"
        + "\n".join(candidates)
        + "\n\nAvailable columns:\n"
        + "\n".join(df.columns)
    )


def numeric(df, column):
    return pd.to_numeric(df[column], errors="coerce")


# ============================================================
# VERIFY INPUT FILES
# ============================================================

print("=" * 72)
print("STEP 62A — PUBLICATION FIGURES")
print("=" * 72)

for p in [STEP61, STEP59B, STEP60B, STEP61_LOCK]:
    require_file(p)

print("\nLOCKED INPUTS")
print("Step-61 synthesis :", STEP61)
print("Step-59B bootstrap:", STEP59B)
print("Step-60B calib.   :", STEP60B)
print("Step-61 lock      :", STEP61_LOCK)


# ============================================================
# READ DATA
# ============================================================

s61 = pd.read_csv(STEP61, low_memory=False)
s59 = pd.read_csv(STEP59B, low_memory=False)
s60 = pd.read_csv(STEP60B, low_memory=False)

print("\nROWS")
print("Step-61 :", len(s61))
print("Step-59B:", len(s59))
print("Step-60B:", len(s60))


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

h59 = find_column(
    s59,
    [
        "horizon_months",
        "horizon",
    ],
)

auc_est = find_column(
    s59,
    [
        "roc_auc_estimate",
        "auc_estimate",
        "roc_auc",
    ],
)

auc_lo = find_column(
    s59,
    [
        "roc_auc_ci_lower",
        "auc_ci_lower",
    ],
)

auc_hi = find_column(
    s59,
    [
        "roc_auc_ci_upper",
        "auc_ci_upper",
    ],
)

brier_est = find_column(
    s59,
    [
        "brier_score_estimate",
        "brier_estimate",
        "brier_score",
    ],
)

brier_lo = find_column(
    s59,
    [
        "brier_score_ci_lower",
        "brier_ci_lower",
    ],
)

brier_hi = find_column(
    s59,
    [
        "brier_score_ci_upper",
        "brier_ci_upper",
    ],
)


h60 = find_column(
    s60,
    [
        "horizon_months",
        "horizon",
    ],
)

obs_est = find_column(
    s60,
    [
        "observed_event_rate_estimate",
        "observed_event_rate",
    ],
)

obs_lo = find_column(
    s60,
    [
        "observed_event_rate_ci_lower",
    ],
)

obs_hi = find_column(
    s60,
    [
        "observed_event_rate_ci_upper",
    ],
)

pred_est = find_column(
    s60,
    [
        "mean_predicted_risk_estimate",
        "mean_predicted_risk",
    ],
)

pred_lo = find_column(
    s60,
    [
        "mean_predicted_risk_ci_lower",
    ],
)

pred_hi = find_column(
    s60,
    [
        "mean_predicted_risk_ci_upper",
    ],
)

cal_int_est = find_column(
    s60,
    [
        "calibration_intercept_estimate",
        "calibration_intercept",
    ],
)

cal_int_lo = find_column(
    s60,
    [
        "calibration_intercept_ci_lower",
    ],
)

cal_int_hi = find_column(
    s60,
    [
        "calibration_intercept_ci_upper",
    ],
)

cal_slope_est = find_column(
    s60,
    [
        "calibration_slope_estimate",
        "calibration_slope",
    ],
)

cal_slope_lo = find_column(
    s60,
    [
        "calibration_slope_ci_lower",
    ],
)

cal_slope_hi = find_column(
    s60,
    [
        "calibration_slope_ci_upper",
    ],
)

skill_est = find_column(
    s60,
    [
        "brier_skill_score_estimate",
        "brier_skill_score",
    ],
)

skill_lo = find_column(
    s60,
    [
        "brier_skill_score_ci_lower",
    ],
)

skill_hi = find_column(
    s60,
    [
        "brier_skill_score_ci_upper",
    ],
)


# ============================================================
# STANDARDIZE TABLES
# ============================================================

perf = pd.DataFrame(
    {
        "horizon_months": numeric(s59, h59),
        "auc": numeric(s59, auc_est),
        "auc_lower": numeric(s59, auc_lo),
        "auc_upper": numeric(s59, auc_hi),
        "brier": numeric(s59, brier_est),
        "brier_lower": numeric(s59, brier_lo),
        "brier_upper": numeric(s59, brier_hi),
    }
)

cal = pd.DataFrame(
    {
        "horizon_months": numeric(s60, h60),
        "observed_risk": numeric(s60, obs_est),
        "observed_lower": numeric(s60, obs_lo),
        "observed_upper": numeric(s60, obs_hi),
        "predicted_risk": numeric(s60, pred_est),
        "predicted_lower": numeric(s60, pred_lo),
        "predicted_upper": numeric(s60, pred_hi),
        "calibration_intercept": numeric(s60, cal_int_est),
        "calibration_intercept_lower": numeric(s60, cal_int_lo),
        "calibration_intercept_upper": numeric(s60, cal_int_hi),
        "calibration_slope": numeric(s60, cal_slope_est),
        "calibration_slope_lower": numeric(s60, cal_slope_lo),
        "calibration_slope_upper": numeric(s60, cal_slope_hi),
        "brier_skill_score": numeric(s60, skill_est),
        "brier_skill_score_lower": numeric(s60, skill_lo),
        "brier_skill_score_upper": numeric(s60, skill_hi),
    }
)

perf = perf[
    perf["horizon_months"].isin(ALL_HORIZONS)
].copy()

cal = cal[
    cal["horizon_months"].isin(ALL_HORIZONS)
].copy()

perf = perf.sort_values("horizon_months").reset_index(drop=True)
cal = cal.sort_values("horizon_months").reset_index(drop=True)

if perf["horizon_months"].tolist() != ALL_HORIZONS:
    raise RuntimeError(
        "Step-59B does not contain exactly the expected horizons."
    )

if cal["horizon_months"].tolist() != ALL_HORIZONS:
    raise RuntimeError(
        "Step-60B does not contain exactly the expected horizons."
    )


# ============================================================
# PUBLICATION TABLE
# ============================================================

publication = perf.merge(
    cal,
    on="horizon_months",
    how="inner",
    validate="one_to_one",
)

publication["analysis_role"] = publication[
    "horizon_months"
].map(
    lambda x: (
        "Primary"
        if int(x) in PRIMARY_HORIZONS
        else "Secondary"
    )
)

publication.to_csv(TABLE, index=False)


# ============================================================
# FIGURE 1
# EXTERNAL DISCRIMINATION
# ============================================================

x = publication["horizon_months"].to_numpy(dtype=float)
y = publication["auc"].to_numpy(dtype=float)

lower = publication["auc_lower"].to_numpy(dtype=float)
upper = publication["auc_upper"].to_numpy(dtype=float)

yerr = np.vstack(
    [
        y - lower,
        upper - y,
    ]
)

fig, ax = plt.subplots(figsize=(7.2, 4.8))

ax.errorbar(
    x,
    y,
    yerr=yerr,
    marker="o",
    markersize=6,
    linewidth=1.5,
    capsize=4,
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.2,
)

ax.set_xlabel("Prediction horizon (months)")
ax.set_ylabel("Time-dependent ROC AUC")

ax.set_xticks(ALL_HORIZONS)

ax.set_ylim(
    min(0.35, float(lower.min()) - 0.03),
    max(0.65, float(upper.max()) + 0.03),
)

ax.set_title(
    "Frozen ADNI model discrimination on OASIS-3"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    FIG1,
    dpi=DPI,
    bbox_inches="tight",
)

fig.savefig(
    FIG1_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2
# OBSERVED VS MEAN PREDICTED RISK
# ============================================================

obs = publication["observed_risk"].to_numpy(dtype=float)
pred = publication["predicted_risk"].to_numpy(dtype=float)

obs_low = publication["observed_lower"].to_numpy(dtype=float)
obs_high = publication["observed_upper"].to_numpy(dtype=float)

pred_low = publication["predicted_lower"].to_numpy(dtype=float)
pred_high = publication["predicted_upper"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(7.2, 4.8))

ax.errorbar(
    x,
    obs,
    yerr=np.vstack(
        [
            obs - obs_low,
            obs_high - obs,
        ]
    ),
    marker="o",
    linewidth=1.5,
    capsize=4,
    label="Observed event rate",
)

ax.errorbar(
    x,
    pred,
    yerr=np.vstack(
        [
            pred - pred_low,
            pred_high - pred,
        ]
    ),
    marker="s",
    linewidth=1.5,
    capsize=4,
    label="Mean predicted risk",
)

ax.set_xlabel("Prediction horizon (months)")
ax.set_ylabel("Risk")

ax.set_xticks(ALL_HORIZONS)

ax.set_ylim(
    0,
    min(
        1.0,
        max(
            float(pred_high.max()),
            float(obs_high.max()),
        )
        + 0.06,
    ),
)

ax.set_title(
    "Observed and predicted progression risk on OASIS-3"
)

ax.legend(frameon=False)

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    FIG2,
    dpi=DPI,
    bbox_inches="tight",
)

fig.savefig(
    FIG2_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 3
# CALIBRATION SLOPE
# ============================================================

slope = publication[
    "calibration_slope"
].to_numpy(dtype=float)

slope_low = publication[
    "calibration_slope_lower"
].to_numpy(dtype=float)

slope_high = publication[
    "calibration_slope_upper"
].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(7.2, 4.8))

ax.errorbar(
    x,
    slope,
    yerr=np.vstack(
        [
            slope - slope_low,
            slope_high - slope,
        ]
    ),
    marker="o",
    linewidth=1.5,
    capsize=4,
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="Ideal slope = 1",
)

ax.axhline(
    0.0,
    linestyle=":",
    linewidth=1.0,
)

ax.set_xlabel("Prediction horizon (months)")
ax.set_ylabel("Calibration slope")

ax.set_xticks(ALL_HORIZONS)

ax.set_title(
    "External calibration slope on OASIS-3"
)

ax.legend(frameon=False)

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    FIG3,
    dpi=DPI,
    bbox_inches="tight",
)

fig.savefig(
    FIG3_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# DISPLAY FINAL NUMERICAL TABLE
# ============================================================

display_cols = [
    "horizon_months",
    "analysis_role",
    "auc",
    "auc_lower",
    "auc_upper",
    "brier",
    "brier_lower",
    "brier_upper",
    "observed_risk",
    "predicted_risk",
    "calibration_intercept",
    "calibration_slope",
    "brier_skill_score",
]

print("\n" + "=" * 72)
print("PUBLICATION TABLE")
print("=" * 72)

print(
    publication[
        display_cols
    ].to_string(index=False)
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "step": "62A",
    "status": "COMPLETE",
    "purpose": (
        "Publication-only visualization of locked "
        "OASIS-3 external-validation results"
    ),
    "primary_horizons_months": PRIMARY_HORIZONS,
    "secondary_horizons_months": SECONDARY_HORIZONS,
    "n_horizons": len(ALL_HORIZONS),
    "inputs": {
        "step61_synthesis": str(STEP61),
        "step59b_bootstrap": str(STEP59B),
        "step60b_calibration_bootstrap": str(STEP60B),
        "step61_lock": str(STEP61_LOCK),
    },
    "outputs": {
        "figure_auc_png": str(FIG1),
        "figure_auc_pdf": str(FIG1_PDF),
        "figure_observed_predicted_png": str(FIG2),
        "figure_observed_predicted_pdf": str(FIG2_PDF),
        "figure_calibration_slope_png": str(FIG3),
        "figure_calibration_slope_pdf": str(FIG3_PDF),
        "publication_table": str(TABLE),
    },
    "analysis_constraints": {
        "mri_voxels_read": False,
        "checkpoints_loaded": False,
        "model_inference_rerun": False,
        "predictions_modified": False,
        "endpoints_modified": False,
        "model_training": False,
        "model_finetuning": False,
        "model_recalibration": False,
        "threshold_optimization": False,
        "oasis_based_model_selection": False,
        "publication_visualization_only": True,
    },
}

with open(
    SUMMARY,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# LOCK
# ============================================================

lock = {
    "step": "62A",
    "status": "LOCKED",
    "step61_sha256": sha256_file(STEP61),
    "step59b_sha256": sha256_file(STEP59B),
    "step60b_sha256": sha256_file(STEP60B),
    "publication_table_sha256": sha256_file(TABLE),
    "figure_auc_png_sha256": sha256_file(FIG1),
    "figure_observed_predicted_png_sha256": sha256_file(FIG2),
    "figure_calibration_slope_png_sha256": sha256_file(FIG3),
    "summary_sha256": sha256_file(SUMMARY),
    "primary_external_validation_results_modified": False,
}

with open(
    LOCK,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        lock,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 72)
print("OUTPUT")
print("=" * 72)

print("Figure 1 PNG :", FIG1)
print("Figure 1 PDF :", FIG1_PDF)

print("Figure 2 PNG :", FIG2)
print("Figure 2 PDF :", FIG2_PDF)

print("Figure 3 PNG :", FIG3)
print("Figure 3 PDF :", FIG3_PDF)

print("Table        :", TABLE)
print("Summary      :", SUMMARY)
print("Lock         :", LOCK)


print("\n" + "=" * 72)
print("SHA256")
print("=" * 72)

print(
    "Publication table :",
    sha256_file(TABLE),
)

print(
    "Figure AUC        :",
    sha256_file(FIG1),
)

print(
    "Figure risk       :",
    sha256_file(FIG2),
)

print(
    "Figure slope      :",
    sha256_file(FIG3),
)

print(
    "Summary           :",
    sha256_file(SUMMARY),
)

print(
    "Lock              :",
    sha256_file(LOCK),
)


print("\n" + "=" * 72)
print("IMPORTANT")
print("=" * 72)

print("MRI VOXELS READ                 : NO")
print("FROZEN CHECKPOINTS LOADED       : NO")
print("MODEL INFERENCE RE-RUN          : NO")
print("EXTERNAL PREDICTIONS MODIFIED   : NO")
print("EXTERNAL ENDPOINTS MODIFIED     : NO")
print("MODEL TRAINING                  : NO")
print("MODEL FINE-TUNING               : NO")
print("MODEL RECALIBRATION             : NO")
print("THRESHOLD OPTIMIZATION          : NO")
print("OASIS-BASED MODEL SELECTION     : NO")
print("PUBLICATION VISUALIZATION ONLY  : YES")


print("\nSTEP 62A COMPLETE.")
print(
    "Publication figures were generated only from "
    "the locked external-validation results."
)
print(
    "The frozen ADNI model and primary OASIS-3 "
    "external-validation results remain unchanged."
)