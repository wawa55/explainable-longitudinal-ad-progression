# ============================================================
# STEP 60A
# FROZEN OASIS-3 EXTERNAL CALIBRATION / DOMAIN-SHIFT AUDIT
# ============================================================
#
# IMPORTANT:
#   - Reads the frozen Step-58 predictions only.
#   - Does NOT read MRI voxels.
#   - Does NOT load model checkpoints.
#   - Does NOT rerun inference.
#   - Does NOT recalibrate predictions.
#   - Does NOT optimize thresholds.
#   - Does NOT select models using OASIS-3 outcomes.
#
# Purpose:
#   Quantify calibration of the untouched frozen ADNI M0
#   ensemble when transported to the locked OASIS-3 cohort.
#
# Outputs:
#   1. Horizon-level calibration summary
#   2. Calibration-bin table
#   3. JSON summary
#   4. Lock JSON with SHA256 hashes
#
# ============================================================

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

PREDICTION_FILE = (
    ROOT
    / "external_inference"
    / "OASIS3_STEP58_FROZEN_M0_EXTERNAL_PREDICTIONS.csv"
)

STEP59A_FILE = (
    ROOT
    / "external_evaluation"
    / "OASIS3_STEP59A_EXTERNAL_POINT_ESTIMATES.csv"
)

STEP59B_FILE = (
    ROOT
    / "external_evaluation"
    / "OASIS3_STEP59B_PARTICIPANT_CLUSTER_BOOTSTRAP_SUMMARY.csv"
)

OUTPUT_DIR = ROOT / "external_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "OASIS3_STEP60A_EXTERNAL_CALIBRATION_SUMMARY.csv"
)

BINS_CSV = (
    OUTPUT_DIR
    / "OASIS3_STEP60A_EXTERNAL_CALIBRATION_BINS.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "OASIS3_STEP60A_EXTERNAL_CALIBRATION_SUMMARY.json"
)

LOCK_JSON = (
    OUTPUT_DIR
    / "OASIS3_STEP60A_EXTERNAL_CALIBRATION_LOCK.json"
)


# ============================================================
# FROZEN CONTRACT
# ============================================================

HORIZONS = [12, 24, 36, 48, 60]

PRIMARY_HORIZONS = [12, 24, 36]

EPS = 1e-6

N_CALIBRATION_BINS = 10


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def safe_logit(p):
    p = np.asarray(p, dtype=float)

    p = np.clip(
        p,
        EPS,
        1.0 - EPS,
    )

    return np.log(
        p / (1.0 - p)
    )


def fit_logistic_calibration(y, p):
    """
    Returns:
        calibration_intercept
        calibration_slope

    Model:
        logit(P(Y=1)) =
            intercept +
            slope * logit(predicted risk)

    This is diagnostic only.

    The fitted model is NOT used to alter predictions.
    """

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    x = safe_logit(p).reshape(-1, 1)

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=10000,
    )

    try:
        model.fit(x, y)

    except Exception:
        # Compatibility fallback for older sklearn versions.
        model = LogisticRegression(
            penalty="none",
            solver="lbfgs",
            max_iter=10000,
        )

        model.fit(x, y)

    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0, 0])

    return intercept, slope


def calibration_bins(y, p, n_bins=10):
    """
    Quantile-based calibration bins.

    This is descriptive only.

    Repeated predicted values may result in fewer than
    n_bins effective groups.
    """

    temp = pd.DataFrame(
        {
            "outcome": np.asarray(y, dtype=int),
            "prediction": np.asarray(p, dtype=float),
        }
    )

    temp = temp.sort_values(
        "prediction"
    ).reset_index(drop=True)

    try:
        temp["bin"] = pd.qcut(
            temp["prediction"],
            q=n_bins,
            labels=False,
            duplicates="drop",
        )

    except Exception:
        temp["bin"] = 0

    records = []

    for bin_id, group in temp.groupby(
        "bin",
        observed=True,
    ):
        n = len(group)

        events = int(
            group["outcome"].sum()
        )

        observed = float(
            group["outcome"].mean()
        )

        predicted = float(
            group["prediction"].mean()
        )

        records.append(
            {
                "calibration_bin":
                    int(bin_id) + 1,

                "n_landmarks":
                    int(n),

                "n_events":
                    events,

                "observed_event_rate":
                    observed,

                "mean_predicted_risk":
                    predicted,

                "min_predicted_risk":
                    float(
                        group["prediction"].min()
                    ),

                "max_predicted_risk":
                    float(
                        group["prediction"].max()
                    ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# INPUT VERIFICATION
# ============================================================

print("=" * 72)
print("STEP 60A — FROZEN OASIS-3 EXTERNAL CALIBRATION AUDIT")
print("=" * 72)

require_file(
    PREDICTION_FILE,
    "Step-58 frozen external predictions",
)

require_file(
    STEP59A_FILE,
    "Step-59A point estimates",
)

require_file(
    STEP59B_FILE,
    "Step-59B bootstrap summary",
)

print()
print("LOCKED INPUTS")
print("Step-58 predictions :", PREDICTION_FILE)
print("Step-59A estimates  :", STEP59A_FILE)
print("Step-59B bootstrap  :", STEP59B_FILE)


# ============================================================
# LOAD FROZEN PREDICTIONS
# ============================================================

df = pd.read_csv(
    PREDICTION_FILE,
    low_memory=False,
)

print()
print("FROZEN PREDICTION TABLE")
print("Rows     :", len(df))

if "locked_subject_id" not in df.columns:
    raise RuntimeError(
        "locked_subject_id is missing."
    )

print(
    "Subjects :",
    df["locked_subject_id"].nunique(),
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "locked_subject_id",
]

for horizon in HORIZONS:

    required_columns.extend(
        [
            f"risk_{horizon}m_ensemble",
            f"event_by_{horizon}m",
            f"evaluable_{horizon}m",
        ]
    )

missing_columns = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

print(
    "PASS: all required frozen prediction "
    "and endpoint columns exist."
)


# ============================================================
# LOAD STEP 59 RESULTS FOR CONSISTENCY CHECK
# ============================================================

point_df = pd.read_csv(
    STEP59A_FILE,
    low_memory=False,
)

bootstrap_df = pd.read_csv(
    STEP59B_FILE,
    low_memory=False,
)

print(
    "PASS: Step-59A and Step-59B result "
    "files loaded."
)


# ============================================================
# HORIZON-BY-HORIZON CALIBRATION AUDIT
# ============================================================

summary_records = []
bin_records = []

for horizon in HORIZONS:

    print()
    print("=" * 72)
    print(f"HORIZON: {horizon} MONTHS")
    print("=" * 72)

    risk_col = f"risk_{horizon}m_ensemble"
    event_col = f"event_by_{horizon}m"
    evaluable_col = f"evaluable_{horizon}m"

    evaluable_mask = (
        df[evaluable_col]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "1.0"])
    )

    hdf = df.loc[
        evaluable_mask,
        [
            "locked_subject_id",
            risk_col,
            event_col,
        ],
    ].copy()

    hdf[risk_col] = pd.to_numeric(
        hdf[risk_col],
        errors="coerce",
    )

    hdf[event_col] = pd.to_numeric(
        hdf[event_col],
        errors="coerce",
    )

    if hdf[risk_col].isna().any():
        raise RuntimeError(
            f"{horizon}m contains missing predictions."
        )

    if hdf[event_col].isna().any():
        raise RuntimeError(
            f"{horizon}m contains missing outcomes "
            "among evaluable landmarks."
        )

    y = hdf[event_col].astype(int).to_numpy()

    p = hdf[risk_col].astype(float).to_numpy()

    if not np.isfinite(p).all():
        raise RuntimeError(
            f"{horizon}m contains non-finite predictions."
        )

    if np.any(p < 0.0) or np.any(p > 1.0):
        raise RuntimeError(
            f"{horizon}m predictions outside [0,1]."
        )

    unique_y = set(
        np.unique(y).tolist()
    )

    if not unique_y.issubset({0, 1}):
        raise RuntimeError(
            f"{horizon}m outcome is not binary."
        )

    n = int(len(hdf))
    n_subjects = int(
        hdf["locked_subject_id"].nunique()
    )
    n_events = int(y.sum())

    event_rate = float(
        np.mean(y)
    )

    mean_prediction = float(
        np.mean(p)
    )

    median_prediction = float(
        np.median(p)
    )

    min_prediction = float(
        np.min(p)
    )

    max_prediction = float(
        np.max(p)
    )

    brier = float(
        brier_score_loss(y, p)
    )

    # --------------------------------------------------------
    # Null/prevalence reference prediction
    # --------------------------------------------------------

    null_prediction = np.full(
        len(y),
        event_rate,
        dtype=float,
    )

    null_brier = float(
        brier_score_loss(
            y,
            null_prediction,
        )
    )

    if null_brier > 0:
        brier_skill = float(
            1.0 - (brier / null_brier)
        )
    else:
        brier_skill = np.nan

    # --------------------------------------------------------
    # Calibration-in-the-large descriptive difference
    # --------------------------------------------------------

    mean_risk_minus_observed = float(
        mean_prediction - event_rate
    )

    observed_minus_mean_risk = float(
        event_rate - mean_prediction
    )

    risk_observed_ratio = (
        float(mean_prediction / event_rate)
        if event_rate > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Logistic calibration intercept and slope
    # --------------------------------------------------------

    calibration_intercept, calibration_slope = (
        fit_logistic_calibration(
            y,
            p,
        )
    )

    # --------------------------------------------------------
    # Quantile calibration bins
    # --------------------------------------------------------

    bins = calibration_bins(
        y,
        p,
        n_bins=N_CALIBRATION_BINS,
    )

    bins.insert(
        0,
        "horizon_months",
        horizon,
    )

    bin_records.append(bins)

    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    summary_records.append(
        {
            "horizon_months":
                horizon,

            "primary_horizon":
                horizon in PRIMARY_HORIZONS,

            "n_evaluable_landmarks":
                n,

            "n_unique_subjects":
                n_subjects,

            "n_events":
                n_events,

            "observed_event_rate":
                event_rate,

            "mean_predicted_risk":
                mean_prediction,

            "median_predicted_risk":
                median_prediction,

            "min_predicted_risk":
                min_prediction,

            "max_predicted_risk":
                max_prediction,

            "mean_risk_minus_observed":
                mean_risk_minus_observed,

            "observed_minus_mean_risk":
                observed_minus_mean_risk,

            "mean_risk_to_observed_ratio":
                risk_observed_ratio,

            "calibration_intercept":
                calibration_intercept,

            "calibration_slope":
                calibration_slope,

            "brier_score":
                brier,

            "null_prevalence_brier":
                null_brier,

            "brier_skill_score":
                brier_skill,
        }
    )

    print(
        "Evaluable landmarks :",
        n,
    )

    print(
        "Unique subjects     :",
        n_subjects,
    )

    print(
        "Events              :",
        n_events,
    )

    print(
        "Observed event rate :",
        f"{event_rate:.6f}",
    )

    print(
        "Mean predicted risk :",
        f"{mean_prediction:.6f}",
    )

    print(
        "Risk - observed     :",
        f"{mean_risk_minus_observed:.6f}",
    )

    print(
        "Calibration int.    :",
        f"{calibration_intercept:.6f}",
    )

    print(
        "Calibration slope   :",
        f"{calibration_slope:.6f}",
    )

    print(
        "Brier score         :",
        f"{brier:.6f}",
    )

    print(
        "Null Brier          :",
        f"{null_brier:.6f}",
    )

    print(
        "Brier skill score   :",
        f"{brier_skill:.6f}",
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

summary_df = pd.DataFrame(
    summary_records
)

bins_df = pd.concat(
    bin_records,
    ignore_index=True,
)


# ============================================================
# CONSISTENCY CHECK AGAINST STEP 59A
# ============================================================

print()
print("=" * 72)
print("STEP-59A CONSISTENCY AUDIT")
print("=" * 72)

if "horizon_months" not in point_df.columns:
    raise RuntimeError(
        "Step-59A file has no horizon_months column."
    )

for _, row in summary_df.iterrows():

    horizon = int(
        row["horizon_months"]
    )

    ref = point_df.loc[
        point_df["horizon_months"] == horizon
    ]

    if len(ref) != 1:
        raise RuntimeError(
            f"Expected exactly one Step-59A row "
            f"for {horizon}m."
        )

    ref = ref.iloc[0]

    if "n_evaluable_landmarks" in ref.index:

        if int(ref["n_evaluable_landmarks"]) != int(
            row["n_evaluable_landmarks"]
        ):
            raise RuntimeError(
                f"{horizon}m evaluable-count mismatch "
                "versus Step-59A."
            )

    if "n_events" in ref.index:

        if int(ref["n_events"]) != int(
            row["n_events"]
        ):
            raise RuntimeError(
                f"{horizon}m event-count mismatch "
                "versus Step-59A."
            )

    if "brier_score" in ref.index:

        if not np.isclose(
            float(ref["brier_score"]),
            float(row["brier_score"]),
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"{horizon}m Brier mismatch "
                "versus Step-59A."
            )

print(
    "PASS: horizon counts, events, and Brier "
    "scores reproduce Step-59A."
)


# ============================================================
# SAVE CSV OUTPUTS
# ============================================================

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)

bins_df.to_csv(
    BINS_CSV,
    index=False,
)


# ============================================================
# SUMMARY JSON
# ============================================================

json_records = []

for record in summary_records:

    cleaned = {}

    for key, value in record.items():

        if isinstance(
            value,
            (np.integer,),
        ):
            value = int(value)

        elif isinstance(
            value,
            (np.floating,),
        ):
            value = float(value)

        if isinstance(value, float):
            if not math.isfinite(value):
                value = None

        cleaned[key] = value

    json_records.append(cleaned)


summary_payload = {
    "step":
        "60A",

    "analysis":
        "Frozen OASIS-3 external calibration and domain-shift audit",

    "prediction_source":
        str(PREDICTION_FILE),

    "step59a_source":
        str(STEP59A_FILE),

    "step59b_source":
        str(STEP59B_FILE),

    "horizons_months":
        HORIZONS,

    "primary_horizons_months":
        PRIMARY_HORIZONS,

    "calibration_bins_requested":
        N_CALIBRATION_BINS,

    "model_predictions_modified":
        False,

    "model_recalibrated":
        False,

    "threshold_optimized":
        False,

    "model_retrained":
        False,

    "oasis_based_model_selection":
        False,

    "results":
        json_records,
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary_payload,
        f,
        indent=2,
    )


# ============================================================
# HASH OUTPUTS
# ============================================================

summary_hash = sha256_file(
    SUMMARY_CSV
)

bins_hash = sha256_file(
    BINS_CSV
)

summary_json_hash = sha256_file(
    SUMMARY_JSON
)

prediction_hash = sha256_file(
    PREDICTION_FILE
)

step59a_hash = sha256_file(
    STEP59A_FILE
)

step59b_hash = sha256_file(
    STEP59B_FILE
)


# ============================================================
# LOCK FILE
# ============================================================

lock_payload = {
    "step":
        "60A",

    "status":
        "LOCKED",

    "analysis_type":
        "external calibration diagnostic only",

    "input_prediction_file":
        str(PREDICTION_FILE),

    "input_prediction_sha256":
        prediction_hash,

    "input_step59a_file":
        str(STEP59A_FILE),

    "input_step59a_sha256":
        step59a_hash,

    "input_step59b_file":
        str(STEP59B_FILE),

    "input_step59b_sha256":
        step59b_hash,

    "output_summary_csv":
        str(SUMMARY_CSV),

    "output_summary_csv_sha256":
        summary_hash,

    "output_bins_csv":
        str(BINS_CSV),

    "output_bins_csv_sha256":
        bins_hash,

    "output_summary_json":
        str(SUMMARY_JSON),

    "output_summary_json_sha256":
        summary_json_hash,

    "model_training":
        False,

    "model_fine_tuning":
        False,

    "model_recalibration":
        False,

    "threshold_optimization":
        False,

    "oasis_based_model_selection":
        False,

    "external_predictions_modified":
        False,

    "mri_voxels_read":
        False,

    "checkpoint_tensors_loaded":
        False,
}

with open(
    LOCK_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        lock_payload,
        f,
        indent=2,
    )


lock_hash = sha256_file(
    LOCK_JSON
)


# ============================================================
# FINAL TABLE
# ============================================================

print()
print("=" * 72)
print("FINAL STEP-60A EXTERNAL CALIBRATION TABLE")
print("=" * 72)

display_columns = [
    "horizon_months",
    "n_evaluable_landmarks",
    "n_events",
    "observed_event_rate",
    "mean_predicted_risk",
    "calibration_intercept",
    "calibration_slope",
    "brier_score",
    "null_prevalence_brier",
    "brier_skill_score",
]

print(
    summary_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# OUTPUT LOCATIONS
# ============================================================

print()
print("=" * 72)
print("OUTPUT")
print("=" * 72)

print(
    "Calibration summary :",
    SUMMARY_CSV,
)

print(
    "Calibration bins    :",
    BINS_CSV,
)

print(
    "Summary JSON        :",
    SUMMARY_JSON,
)

print(
    "Lock                :",
    LOCK_JSON,
)


# ============================================================
# HASHES
# ============================================================

print()
print("SHA256")

print(
    "Summary CSV :",
    summary_hash,
)

print(
    "Bins CSV    :",
    bins_hash,
)

print(
    "Summary JSON:",
    summary_json_hash,
)

print(
    "Lock        :",
    lock_hash,
)


# ============================================================
# FIREWALL
# ============================================================

print()
print("=" * 72)
print("IMPORTANT")
print("=" * 72)

print(
    "MRI VOXELS READ                 : NO"
)

print(
    "FROZEN CHECKPOINTS LOADED       : NO"
)

print(
    "MODEL INFERENCE RE-RUN          : NO"
)

print(
    "EXTERNAL PREDICTIONS MODIFIED   : NO"
)

print(
    "EXTERNAL ENDPOINTS MODIFIED     : NO"
)

print(
    "MODEL TRAINING                  : NO"
)

print(
    "MODEL FINE-TUNING               : NO"
)

print(
    "MODEL RECALIBRATION             : NO"
)

print(
    "THRESHOLD OPTIMIZATION          : NO"
)

print(
    "OASIS-BASED MODEL SELECTION     : NO"
)

print(
    "CALIBRATION DIAGNOSTICS READ    : YES"
)


print()
print(
    "Step 60A complete."
)

print(
    "The untouched frozen ADNI M0 predictions have now "
    "been assessed for calibration on OASIS-3."
)

print(
    "These diagnostics describe transportability; "
    "they do not recalibrate or modify the frozen model."
)

print(
    "Do not modify the frozen predictions after "
    "observing these results."
)