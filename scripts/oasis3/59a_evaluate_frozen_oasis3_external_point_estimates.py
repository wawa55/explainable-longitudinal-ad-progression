# ============================================================
# STEP 59A
# OASIS-3 EXTERNAL VALIDATION
#
# Frozen M0 five-seed ensemble:
#   point-estimate evaluation + endpoint audit
#
# IMPORTANT
# ------------------------------------------------------------
# This script:
#   - DOES NOT train the model
#   - DOES NOT fine-tune the model
#   - DOES NOT recalibrate the model
#   - DOES NOT select thresholds
#   - DOES NOT modify predictions
#   - DOES NOT modify the locked cohort
#
# It evaluates the already-frozen Step-58 predictions against
# the already-locked external OASIS-3 endpoints.
#
# Horizon-specific binary evaluation:
#   evaluable_H == True
#   outcome = event_by_H
#   prediction = risk_H_ensemble
#
# Metrics:
#   ROC-AUC
#   Brier score
#
# Confidence intervals are deliberately NOT computed here.
# Participant-cluster bootstrap uncertainty is reserved for
# Step 59B.
# ============================================================

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, brier_score_loss


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

INPUT_CSV = (
    ROOT
    / "external_inference"
    / "OASIS3_STEP58_FROZEN_M0_EXTERNAL_PREDICTIONS.csv"
)

STEP58_SUMMARY = (
    ROOT
    / "external_inference"
    / "OASIS3_STEP58_FROZEN_M0_EXTERNAL_INFERENCE_SUMMARY.json"
)

STEP58_LOCK = (
    ROOT
    / "external_inference"
    / "OASIS3_STEP58_FROZEN_M0_EXTERNAL_INFERENCE_LOCK.json"
)

OUTPUT_DIR = ROOT / "external_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "OASIS3_STEP59A_EXTERNAL_POINT_ESTIMATES.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "OASIS3_STEP59A_EXTERNAL_POINT_ESTIMATES_SUMMARY.json"
)

OUTPUT_LOCK = (
    OUTPUT_DIR
    / "OASIS3_STEP59A_EXTERNAL_POINT_ESTIMATES_LOCK.json"
)


# ============================================================
# LOCKED CONTRACT
# ============================================================

HORIZONS = [12, 24, 36, 48, 60]

EXPECTED_ROWS = 1559
EXPECTED_SUBJECTS = 542

SUBJECT_COL = "locked_subject_id"


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path, chunk_size=1024 * 1024):
    path = Path(path)

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def require_file(path, label):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{label} missing:\n{path}"
        )


def require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise RuntimeError(
            "Required columns missing:\n"
            + "\n".join(missing)
        )


def python_scalar(x):
    if isinstance(x, np.generic):
        return x.item()

    return x


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("STEP 59A — FROZEN OASIS-3 EXTERNAL POINT-ESTIMATE EVALUATION")
print("=" * 78)

print()
print("This is the first outcome-facing evaluation of the frozen")
print("five-seed ADNI M0 ensemble on the locked OASIS-3 cohort.")
print()

print("No model fitting, tuning, recalibration, or threshold")
print("optimization is permitted in this step.")
print()


# ============================================================
# VERIFY INPUT FILES
# ============================================================

require_file(
    INPUT_CSV,
    "Step-58 frozen external predictions",
)

require_file(
    STEP58_SUMMARY,
    "Step-58 summary",
)

require_file(
    STEP58_LOCK,
    "Step-58 lock",
)

print("LOCKED INPUTS")
print("Predictions :", INPUT_CSV)
print("Summary     :", STEP58_SUMMARY)
print("Lock        :", STEP58_LOCK)
print()


# ============================================================
# HASH INPUTS BEFORE READING
# ============================================================

input_hash = sha256_file(INPUT_CSV)
summary58_hash = sha256_file(STEP58_SUMMARY)
lock58_hash = sha256_file(STEP58_LOCK)

print("STEP-58 SHA256")
print("Predictions :", input_hash)
print("Summary     :", summary58_hash)
print("Lock        :", lock58_hash)
print()


# ============================================================
# READ FROZEN PREDICTIONS
# ============================================================

df = pd.read_csv(
    INPUT_CSV,
    low_memory=False,
)

print("FROZEN EXTERNAL DATA")
print("Rows     :", len(df))
print("Columns  :", len(df.columns))

if SUBJECT_COL in df.columns:
    print(
        "Subjects :",
        df[SUBJECT_COL].nunique()
    )

print()


# ============================================================
# VERIFY GLOBAL CONTRACT
# ============================================================

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Unexpected row count: {len(df)} "
        f"!= {EXPECTED_ROWS}"
    )

if SUBJECT_COL not in df.columns:
    raise RuntimeError(
        f"Missing subject column: {SUBJECT_COL}"
    )

n_subjects = int(df[SUBJECT_COL].nunique())

if n_subjects != EXPECTED_SUBJECTS:
    raise RuntimeError(
        f"Unexpected subject count: {n_subjects} "
        f"!= {EXPECTED_SUBJECTS}"
    )

print("PASS: Step-58 external population size verified.")
print()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    SUBJECT_COL,
    "event_observed",
    "event_day_from_mri",
    "censored",
    "observed_time_days",
    "observed_time_months",
]

for horizon in HORIZONS:
    required_columns.extend(
        [
            f"event_by_{horizon}m",
            f"evaluable_{horizon}m",
            f"risk_{horizon}m_ensemble",
        ]
    )

require_columns(df, required_columns)

print("PASS: all required endpoint and prediction columns exist.")
print()


# ============================================================
# GLOBAL ENDPOINT AUDIT
# ============================================================

print("=" * 78)
print("GLOBAL ENDPOINT AUDIT")
print("=" * 78)

event_observed = (
    df["event_observed"]
    .astype(bool)
)

censored = (
    df["censored"]
    .astype(bool)
)

print("Landmarks      :", len(df))
print("Subjects       :", n_subjects)
print("Event rows     :", int(event_observed.sum()))
print("Censored rows  :", int(censored.sum()))
print()


# ============================================================
# HORIZON-SPECIFIC EVALUATION
# ============================================================

results = []

print("=" * 78)
print("HORIZON-SPECIFIC EXTERNAL PERFORMANCE")
print("=" * 78)

for horizon in HORIZONS:

    eval_col = f"evaluable_{horizon}m"
    event_col = f"event_by_{horizon}m"
    risk_col = f"risk_{horizon}m_ensemble"

    # --------------------------------------------------------
    # Locked evaluability mask
    # --------------------------------------------------------

    evaluable = df[eval_col]

    if evaluable.dtype != bool:
        raise RuntimeError(
            f"{eval_col} is not boolean. "
            f"Observed dtype = {evaluable.dtype}"
        )

    mask = evaluable == True

    sub = df.loc[
        mask,
        [
            SUBJECT_COL,
            event_col,
            risk_col,
        ],
    ].copy()

    # --------------------------------------------------------
    # Strict missing-value audit
    # --------------------------------------------------------

    if sub[event_col].isna().any():
        raise RuntimeError(
            f"{event_col} contains NaN among "
            f"evaluable landmarks."
        )

    if sub[risk_col].isna().any():
        raise RuntimeError(
            f"{risk_col} contains NaN among "
            f"evaluable landmarks."
        )

    # --------------------------------------------------------
    # Convert outcome and prediction
    # --------------------------------------------------------

    y = sub[event_col].astype(int).to_numpy()

    risk = (
        sub[risk_col]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Outcome validity
    # --------------------------------------------------------

    unique_y = np.unique(y)

    if not set(unique_y).issubset({0, 1}):
        raise RuntimeError(
            f"Invalid outcome values at {horizon}m: "
            f"{unique_y}"
        )

    if len(unique_y) != 2:
        raise RuntimeError(
            f"ROC-AUC undefined at {horizon}m because "
            f"only one outcome class is present."
        )

    # --------------------------------------------------------
    # Prediction validity
    # --------------------------------------------------------

    if not np.isfinite(risk).all():
        raise RuntimeError(
            f"Non-finite predictions detected at "
            f"{horizon}m."
        )

    if np.any(risk < 0.0) or np.any(risk > 1.0):
        raise RuntimeError(
            f"Predicted risks outside [0,1] at "
            f"{horizon}m."
        )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    n_evaluable = int(len(sub))
    n_events = int(y.sum())
    n_nonevents = int(n_evaluable - n_events)
    n_subjects_h = int(
        sub[SUBJECT_COL].nunique()
    )

    event_rate = (
        float(n_events / n_evaluable)
        if n_evaluable > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Frozen external metrics
    # --------------------------------------------------------

    roc_auc = float(
        roc_auc_score(
            y,
            risk,
        )
    )

    brier = float(
        brier_score_loss(
            y,
            risk,
        )
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    risk_mean = float(np.mean(risk))
    risk_sd = float(np.std(risk, ddof=1))
    risk_median = float(np.median(risk))
    risk_min = float(np.min(risk))
    risk_max = float(np.max(risk))

    # --------------------------------------------------------
    # Save record
    # --------------------------------------------------------

    record = {
        "horizon_months": int(horizon),
        "n_evaluable_landmarks": n_evaluable,
        "n_evaluable_subjects": n_subjects_h,
        "n_events": n_events,
        "n_nonevents": n_nonevents,
        "event_rate": event_rate,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "risk_mean": risk_mean,
        "risk_sd": risk_sd,
        "risk_median": risk_median,
        "risk_min": risk_min,
        "risk_max": risk_max,
    }

    results.append(record)

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print(f"{horizon}-MONTH HORIZON")
    print("-" * 40)

    print(
        f"Evaluable landmarks : {n_evaluable}"
    )

    print(
        f"Evaluable subjects  : {n_subjects_h}"
    )

    print(
        f"Events              : {n_events}"
    )

    print(
        f"Non-events          : {n_nonevents}"
    )

    print(
        f"Event rate          : {event_rate:.6f}"
    )

    print(
        f"ROC-AUC             : {roc_auc:.6f}"
    )

    print(
        f"Brier score         : {brier:.6f}"
    )

    print(
        f"Mean predicted risk : {risk_mean:.6f}"
    )

    print(
        f"Median risk         : {risk_median:.6f}"
    )

    print(
        f"Risk range          : "
        f"[{risk_min:.6f}, {risk_max:.6f}]"
    )


# ============================================================
# RESULT TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "horizon_months"
).reset_index(drop=True)

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print()
print("=" * 78)
print("EXTERNAL POINT-ESTIMATE TABLE")
print("=" * 78)

display_columns = [
    "horizon_months",
    "n_evaluable_landmarks",
    "n_events",
    "event_rate",
    "roc_auc",
    "brier_score",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()


# ============================================================
# INTERNAL CONSISTENCY AUDIT
# ============================================================

print("=" * 78)
print("CONSISTENCY AUDIT")
print("=" * 78)

# Expected evaluable counts from locked endpoint audit
expected_evaluable = {
    12: 1453,
    24: 1383,
    36: 1242,
    48: 1152,
    60: 1045,
}

expected_events = {
    12: 48,
    24: 88,
    36: 135,
    48: 163,
    60: 198,
}

for record in results:

    horizon = record["horizon_months"]

    if (
        record["n_evaluable_landmarks"]
        != expected_evaluable[horizon]
    ):
        raise RuntimeError(
            f"{horizon}m evaluable-count mismatch: "
            f"{record['n_evaluable_landmarks']} "
            f"!= {expected_evaluable[horizon]}"
        )

    if (
        record["n_events"]
        != expected_events[horizon]
    ):
        raise RuntimeError(
            f"{horizon}m event-count mismatch: "
            f"{record['n_events']} "
            f"!= {expected_events[horizon]}"
        )

print(
    "PASS: horizon-specific evaluable counts match "
    "the locked endpoint audit."
)

print(
    "PASS: horizon-specific event counts match "
    "the locked endpoint audit."
)

print(
    "PASS: all evaluated predictions are finite "
    "probabilities in [0,1]."
)

print()


# ============================================================
# WRITE SUMMARY JSON
# ============================================================

summary = {
    "step": "59A",
    "analysis": (
        "Frozen OASIS-3 external validation "
        "point-estimate evaluation"
    ),
    "model": (
        "Frozen five-seed ADNI "
        "ExactM0LongitudinalSurvivalModel_V3 ensemble"
    ),
    "input_predictions": str(INPUT_CSV),
    "input_predictions_sha256": input_hash,
    "step58_summary_sha256": summary58_hash,
    "step58_lock_sha256": lock58_hash,
    "n_landmarks_total": int(len(df)),
    "n_subjects_total": int(n_subjects),
    "horizons_months": HORIZONS,
    "metrics": [
        "ROC-AUC",
        "Brier score",
    ],
    "evaluation_rule": (
        "At each horizon H, evaluate only rows where "
        "evaluable_H == True; outcome = event_by_H; "
        "prediction = risk_H_ensemble."
    ),
    "results": [
        {
            k: python_scalar(v)
            for k, v in record.items()
        }
        for record in results
    ],
    "confidence_intervals_computed": False,
    "participant_cluster_bootstrap": False,
    "model_training": False,
    "model_fine_tuning": False,
    "model_recalibration": False,
    "threshold_optimization": False,
    "oasis_based_model_selection": False,
    "predictions_modified": False,
    "cohort_modified": False,
    "preprocessed_mri_modified": False,
}

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# HASH OUTPUTS
# ============================================================

output_csv_hash = sha256_file(OUTPUT_CSV)
output_summary_hash = sha256_file(OUTPUT_SUMMARY)


# ============================================================
# WRITE STEP-59A LOCK
# ============================================================

lock = {
    "step": "59A",
    "status": "LOCKED",
    "input_predictions_sha256": input_hash,
    "output_point_estimates_sha256": output_csv_hash,
    "output_summary_sha256": output_summary_hash,
    "n_landmarks": int(len(df)),
    "n_subjects": int(n_subjects),
    "horizons_months": HORIZONS,
    "primary_external_horizons_months": [
        12,
        24,
        36,
    ],
    "secondary_external_horizons_months": [
        48,
        60,
    ],
    "evaluation": {
        "binary_horizon_rule": (
            "evaluable_H == True"
        ),
        "outcome": "event_by_H",
        "prediction": "risk_H_ensemble",
        "metrics": [
            "ROC-AUC",
            "Brier score",
        ],
    },
    "prohibitions": {
        "model_training": True,
        "model_fine_tuning": True,
        "model_recalibration": True,
        "threshold_optimization": True,
        "oasis_based_model_selection": True,
        "prediction_modification": True,
    },
    "next_step": (
        "Participant-cluster bootstrap confidence "
        "intervals and locked survival-performance "
        "analysis."
    ),
}

with open(
    OUTPUT_LOCK,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        lock,
        f,
        indent=2,
    )

output_lock_hash = sha256_file(OUTPUT_LOCK)


# ============================================================
# FINAL REPORT
# ============================================================

print("=" * 78)
print("OUTPUT")
print("=" * 78)

print(OUTPUT_CSV)

print()
print("SUMMARY")
print(OUTPUT_SUMMARY)

print()
print("LOCK")
print(OUTPUT_LOCK)

print()
print("SHA256")
print("Point estimates :", output_csv_hash)
print("Summary         :", output_summary_hash)
print("Lock            :", output_lock_hash)

print()

print("=" * 78)
print("IMPORTANT")
print("=" * 78)

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
print("POINT-ESTIMATE PERFORMANCE READ : YES")
print("CONFIDENCE INTERVALS COMPUTED   : NO")

print()
print(
    "Step 59A complete. Frozen OASIS-3 external "
    "point estimates have now been calculated."
)

print(
    "Do not modify the model or predictions after "
    "observing these results."
)

print(
    "The next step is participant-cluster bootstrap "
    "uncertainty and locked survival-performance analysis."
)