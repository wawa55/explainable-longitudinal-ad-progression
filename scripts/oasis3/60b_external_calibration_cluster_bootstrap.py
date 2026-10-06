# ============================================================
# STEP 60B
# OASIS-3 EXTERNAL CALIBRATION / DOMAIN-SHIFT
# PARTICIPANT-CLUSTER BOOTSTRAP
# ============================================================
#
# PURPOSE
# -------
# Quantify uncertainty around the frozen Step-60A external
# calibration/domain-shift results.
#
# IMPORTANT
# ---------
# - Frozen Step-58 predictions are NOT modified.
# - MRI voxels are NOT read.
# - Checkpoints are NOT loaded.
# - Model inference is NOT rerun.
# - No training/fine-tuning/recalibration is performed.
# - No threshold optimization is performed.
# - No OASIS-based model selection is performed.
#
# BOOTSTRAP UNIT
# --------------
# Participant.
#
# If a participant is sampled, ALL evaluable landmarks belonging
# to that participant at the horizon are retained together.
#
# Replicates:
#     2000
#
# Confidence intervals:
#     percentile 95% CI
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

STEP60A_FILE = (
    ROOT
    / "external_evaluation"
    / "OASIS3_STEP60A_EXTERNAL_CALIBRATION_SUMMARY.csv"
)

STEP60A_LOCK = (
    ROOT
    / "external_evaluation"
    / "OASIS3_STEP60A_EXTERNAL_CALIBRATION_LOCK.json"
)

OUTPUT_DIR = ROOT / "external_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPLICATES_CSV = (
    OUTPUT_DIR
    / "OASIS3_STEP60B_CALIBRATION_CLUSTER_BOOTSTRAP_REPLICATES.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "OASIS3_STEP60B_CALIBRATION_CLUSTER_BOOTSTRAP_SUMMARY.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "OASIS3_STEP60B_CALIBRATION_CLUSTER_BOOTSTRAP_SUMMARY.json"
)

LOCK_JSON = (
    OUTPUT_DIR
    / "OASIS3_STEP60B_CALIBRATION_CLUSTER_BOOTSTRAP_LOCK.json"
)


# ============================================================
# LOCKED ANALYSIS CONTRACT
# ============================================================

HORIZONS = [12, 24, 36, 48, 60]

PRIMARY_HORIZONS = [12, 24, 36]

N_BOOTSTRAP = 2000

BOOTSTRAP_SEED = 2026

CI_LOWER = 2.5
CI_UPPER = 97.5

EPS = 1e-6


# ============================================================
# HELPERS
# ============================================================

def require_file(path, label):

    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def safe_logit(p):

    p = np.asarray(
        p,
        dtype=float,
    )

    p = np.clip(
        p,
        EPS,
        1.0 - EPS,
    )

    return np.log(
        p / (1.0 - p)
    )


def fit_logistic_calibration(y, p):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    x = safe_logit(
        p
    ).reshape(-1, 1)

    try:

        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=10000,
        )

        model.fit(
            x,
            y,
        )

    except Exception:

        try:

            model = LogisticRegression(
                penalty="none",
                solver="lbfgs",
                max_iter=10000,
            )

            model.fit(
                x,
                y,
            )

        except Exception:

            return np.nan, np.nan

    intercept = float(
        model.intercept_[0]
    )

    slope = float(
        model.coef_[0, 0]
    )

    return intercept, slope


def calculate_metrics(y, p):

    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    n = len(y)

    n_events = int(
        y.sum()
    )

    event_rate = float(
        np.mean(y)
    )

    mean_predicted_risk = float(
        np.mean(p)
    )

    risk_minus_observed = float(
        mean_predicted_risk
        - event_rate
    )

    brier = float(
        brier_score_loss(
            y,
            p,
        )
    )

    null_prediction = np.full(
        n,
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
            1.0
            - (
                brier
                / null_brier
            )
        )

    else:

        brier_skill = np.nan

    calibration_intercept, calibration_slope = (
        fit_logistic_calibration(
            y,
            p,
        )
    )

    return {
        "n_landmarks":
            int(n),

        "n_events":
            n_events,

        "observed_event_rate":
            event_rate,

        "mean_predicted_risk":
            mean_predicted_risk,

        "risk_minus_observed":
            risk_minus_observed,

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


def percentile_ci(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            0,
        )

    lower = float(
        np.percentile(
            values,
            CI_LOWER,
        )
    )

    upper = float(
        np.percentile(
            values,
            CI_UPPER,
        )
    )

    return (
        lower,
        upper,
        int(len(values)),
    )


# ============================================================
# INPUT VERIFICATION
# ============================================================

print("=" * 74)
print(
    "STEP 60B — OASIS-3 CALIBRATION PARTICIPANT-CLUSTER BOOTSTRAP"
)
print("=" * 74)

require_file(
    PREDICTION_FILE,
    "Step-58 frozen predictions",
)

require_file(
    STEP60A_FILE,
    "Step-60A calibration summary",
)

require_file(
    STEP60A_LOCK,
    "Step-60A calibration lock",
)

print()
print("LOCKED INPUTS")

print(
    "Step-58 predictions :",
    PREDICTION_FILE,
)

print(
    "Step-60A summary    :",
    STEP60A_FILE,
)

print(
    "Step-60A lock       :",
    STEP60A_LOCK,
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    PREDICTION_FILE,
    low_memory=False,
)

step60a = pd.read_csv(
    STEP60A_FILE,
    low_memory=False,
)

print()
print("FROZEN EXTERNAL DATA")

print(
    "Landmarks :",
    len(df),
)

if "locked_subject_id" not in df.columns:

    raise RuntimeError(
        "locked_subject_id missing from predictions."
    )

print(
    "Subjects  :",
    df["locked_subject_id"].nunique(),
)


# ============================================================
# REQUIRED COLUMN AUDIT
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
    column
    for column in required_columns
    if column not in df.columns
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
# VERIFY STEP-60A STRUCTURE
# ============================================================

required_60a = [
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

missing_60a = [
    column
    for column in required_60a
    if column not in step60a.columns
]

if missing_60a:

    raise RuntimeError(
        "Step-60A is missing required columns:\n"
        + "\n".join(missing_60a)
    )

print(
    "PASS: Step-60A summary structure verified."
)


# ============================================================
# BOOTSTRAP
# ============================================================

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

replicate_records = []

print()
print("=" * 74)
print("BOOTSTRAP")
print("=" * 74)

print(
    "Bootstrap unit       : participant"
)

print(
    "Bootstrap replicates :",
    N_BOOTSTRAP,
)

print(
    "Bootstrap seed       :",
    BOOTSTRAP_SEED,
)


for horizon in HORIZONS:

    print()
    print("-" * 74)
    print(
        f"HORIZON {horizon} MONTHS"
    )
    print("-" * 74)

    risk_col = (
        f"risk_{horizon}m_ensemble"
    )

    event_col = (
        f"event_by_{horizon}m"
    )

    evaluable_col = (
        f"evaluable_{horizon}m"
    )

    evaluable_mask = (
        df[evaluable_col]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "1.0",
            ]
        )
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
            f"{horizon}m contains missing outcomes."
        )

    if not np.isfinite(
        hdf[risk_col].to_numpy(
            dtype=float
        )
    ).all():

        raise RuntimeError(
            f"{horizon}m contains non-finite predictions."
        )

    participants = (
        hdf["locked_subject_id"]
        .drop_duplicates()
        .to_numpy()
    )

    n_participants = len(
        participants
    )

    groups = {
        participant:
            group.copy()
        for participant, group
        in hdf.groupby(
            "locked_subject_id",
            sort=False,
        )
    }

    print(
        "Evaluable landmarks :",
        len(hdf),
    )

    print(
        "Unique participants :",
        n_participants,
    )

    # --------------------------------------------------------
    # Original metrics for exact Step-60A reproduction
    # --------------------------------------------------------

    original_metrics = calculate_metrics(
        hdf[event_col]
        .astype(int)
        .to_numpy(),

        hdf[risk_col]
        .astype(float)
        .to_numpy(),
    )

    ref = step60a.loc[
        step60a["horizon_months"]
        == horizon
    ]

    if len(ref) != 1:

        raise RuntimeError(
            f"Expected exactly one Step-60A row "
            f"for {horizon}m."
        )

    ref = ref.iloc[0]

    if (
        int(
            original_metrics[
                "n_landmarks"
            ]
        )
        !=
        int(
            ref[
                "n_evaluable_landmarks"
            ]
        )
    ):

        raise RuntimeError(
            f"{horizon}m landmark count "
            "does not reproduce Step-60A."
        )

    if (
        int(
            original_metrics[
                "n_events"
            ]
        )
        !=
        int(
            ref[
                "n_events"
            ]
        )
    ):

        raise RuntimeError(
            f"{horizon}m event count "
            "does not reproduce Step-60A."
        )

    numeric_checks = [
        (
            "observed_event_rate",
            "observed_event_rate",
        ),
        (
            "mean_predicted_risk",
            "mean_predicted_risk",
        ),
        (
            "calibration_intercept",
            "calibration_intercept",
        ),
        (
            "calibration_slope",
            "calibration_slope",
        ),
        (
            "brier_score",
            "brier_score",
        ),
        (
            "null_prevalence_brier",
            "null_prevalence_brier",
        ),
        (
            "brier_skill_score",
            "brier_skill_score",
        ),
    ]

    for metric_key, ref_column in numeric_checks:

        calculated = float(
            original_metrics[
                metric_key
            ]
        )

        expected = float(
            ref[
                ref_column
            ]
        )

        if not np.isclose(
            calculated,
            expected,
            rtol=0.0,
            atol=1e-9,
            equal_nan=True,
        ):

            raise RuntimeError(
                f"{horizon}m {metric_key} "
                "does not reproduce Step-60A.\n"
                f"Calculated: {calculated}\n"
                f"Expected:   {expected}"
            )

    print(
        "PASS: original horizon metrics reproduce Step-60A."
    )

    # --------------------------------------------------------
    # Cluster bootstrap
    # --------------------------------------------------------

    for bootstrap_index in range(
        1,
        N_BOOTSTRAP + 1,
    ):

        sampled_participants = rng.choice(
            participants,
            size=n_participants,
            replace=True,
        )

        y_parts = []
        p_parts = []

        for participant in sampled_participants:

            participant_rows = groups[
                participant
            ]

            y_parts.append(
                participant_rows[
                    event_col
                ]
                .astype(int)
                .to_numpy()
            )

            p_parts.append(
                participant_rows[
                    risk_col
                ]
                .astype(float)
                .to_numpy()
            )

        y_boot = np.concatenate(
            y_parts
        )

        p_boot = np.concatenate(
            p_parts
        )

        metrics = calculate_metrics(
            y_boot,
            p_boot,
        )

        replicate_records.append(
            {
                "horizon_months":
                    horizon,

                "bootstrap_replicate":
                    bootstrap_index,

                "n_sampled_participant_draws":
                    n_participants,

                "n_unique_participants_in_replicate":
                    int(
                        len(
                            np.unique(
                                sampled_participants
                            )
                        )
                    ),

                **metrics,
            }
        )

        if (
            bootstrap_index == 1
            or bootstrap_index % 250 == 0
            or bootstrap_index == N_BOOTSTRAP
        ):

            print(
                f"{horizon}m bootstrap: "
                f"{bootstrap_index}/{N_BOOTSTRAP}"
            )


# ============================================================
# BUILD REPLICATE TABLE
# ============================================================

replicates_df = pd.DataFrame(
    replicate_records
)

expected_rows = (
    len(HORIZONS)
    * N_BOOTSTRAP
)

if len(replicates_df) != expected_rows:

    raise RuntimeError(
        "Unexpected number of bootstrap rows.\n"
        f"Expected: {expected_rows}\n"
        f"Observed: {len(replicates_df)}"
    )

print()
print(
    "PASS: expected bootstrap replicate count =",
    len(replicates_df),
)


# ============================================================
# SUMMARIZE BOOTSTRAP DISTRIBUTIONS
# ============================================================

metrics_to_summarize = [
    "observed_event_rate",
    "mean_predicted_risk",
    "risk_minus_observed",
    "calibration_intercept",
    "calibration_slope",
    "brier_score",
    "null_prevalence_brier",
    "brier_skill_score",
]

summary_records = []


for horizon in HORIZONS:

    original = step60a.loc[
        step60a["horizon_months"]
        == horizon
    ].iloc[0]

    hboot = replicates_df.loc[
        replicates_df[
            "horizon_months"
        ]
        == horizon
    ].copy()

    summary = {
        "horizon_months":
            horizon,

        "primary_horizon":
            horizon
            in PRIMARY_HORIZONS,

        "n_bootstrap_replicates":
            int(len(hboot)),

        "n_evaluable_landmarks":
            int(
                original[
                    "n_evaluable_landmarks"
                ]
            ),

        "n_events":
            int(
                original[
                    "n_events"
                ]
            ),
    }

    original_mapping = {
        "observed_event_rate":
            "observed_event_rate",

        "mean_predicted_risk":
            "mean_predicted_risk",

        "risk_minus_observed":
            "mean_risk_minus_observed",

        "calibration_intercept":
            "calibration_intercept",

        "calibration_slope":
            "calibration_slope",

        "brier_score":
            "brier_score",

        "null_prevalence_brier":
            "null_prevalence_brier",

        "brier_skill_score":
            "brier_skill_score",
    }

    for metric in metrics_to_summarize:

        lower, upper, n_valid = percentile_ci(
            hboot[
                metric
            ].to_numpy(
                dtype=float
            )
        )

        original_column = (
            original_mapping[
                metric
            ]
        )

        point_estimate = float(
            original[
                original_column
            ]
        )

        summary[
            f"{metric}_estimate"
        ] = point_estimate

        summary[
            f"{metric}_ci_lower"
        ] = lower

        summary[
            f"{metric}_ci_upper"
        ] = upper

        summary[
            f"{metric}_valid_replicates"
        ] = n_valid

    summary_records.append(
        summary
    )


summary_df = pd.DataFrame(
    summary_records
)


# ============================================================
# VALIDITY AUDIT
# ============================================================

print()
print("=" * 74)
print("BOOTSTRAP VALIDITY AUDIT")
print("=" * 74)

for horizon in HORIZONS:

    hboot = replicates_df.loc[
        replicates_df[
            "horizon_months"
        ]
        == horizon
    ]

    print()
    print(
        f"{horizon}m replicates :",
        len(hboot),
    )

    for metric in [
        "calibration_intercept",
        "calibration_slope",
        "brier_skill_score",
    ]:

        n_invalid = int(
            (~np.isfinite(
                hboot[
                    metric
                ].to_numpy(
                    dtype=float
                )
            )).sum()
        )

        print(
            f"{horizon}m invalid "
            f"{metric} replicates :",
            n_invalid,
        )


# ============================================================
# HARD AUDITS
# ============================================================

for horizon in HORIZONS:

    hboot = replicates_df.loc[
        replicates_df[
            "horizon_months"
        ]
        == horizon
    ]

    if len(hboot) != N_BOOTSTRAP:

        raise RuntimeError(
            f"{horizon}m does not contain "
            f"{N_BOOTSTRAP} replicates."
        )

print()
print(
    "PASS: every horizon contains exactly "
    f"{N_BOOTSTRAP} bootstrap replicates."
)

print(
    "PASS: participant was the bootstrap resampling unit."
)

print(
    "PASS: repeated landmarks remained within "
    "their participant clusters."
)


# ============================================================
# SAVE OUTPUT TABLES
# ============================================================

replicates_df.to_csv(
    REPLICATES_CSV,
    index=False,
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


# ============================================================
# JSON SUMMARY
# ============================================================

json_rows = []

for row in summary_records:

    clean = {}

    for key, value in row.items():

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

        if isinstance(
            value,
            float,
        ):

            if not math.isfinite(
                value
            ):

                value = None

        clean[key] = value

    json_rows.append(
        clean
    )


summary_payload = {
    "step":
        "60B",

    "analysis":
        (
            "Participant-cluster bootstrap uncertainty "
            "for frozen OASIS-3 external calibration "
            "and domain-shift diagnostics"
        ),

    "bootstrap_unit":
        "participant",

    "bootstrap_replicates":
        N_BOOTSTRAP,

    "bootstrap_seed":
        BOOTSTRAP_SEED,

    "confidence_interval":
        "percentile 95%",

    "ci_percentiles":
        [
            CI_LOWER,
            CI_UPPER,
        ],

    "horizons_months":
        HORIZONS,

    "primary_horizons_months":
        PRIMARY_HORIZONS,

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
        json_rows,
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
# HASHES
# ============================================================

prediction_hash = sha256_file(
    PREDICTION_FILE
)

step60a_hash = sha256_file(
    STEP60A_FILE
)

step60a_lock_hash = sha256_file(
    STEP60A_LOCK
)

replicates_hash = sha256_file(
    REPLICATES_CSV
)

summary_hash = sha256_file(
    SUMMARY_CSV
)

summary_json_hash = sha256_file(
    SUMMARY_JSON
)


# ============================================================
# LOCK
# ============================================================

lock_payload = {
    "step":
        "60B",

    "status":
        "LOCKED",

    "analysis_type":
        (
            "participant-cluster bootstrap uncertainty "
            "for external calibration diagnostics"
        ),

    "bootstrap_unit":
        "participant",

    "bootstrap_replicates":
        N_BOOTSTRAP,

    "bootstrap_seed":
        BOOTSTRAP_SEED,

    "confidence_interval":
        "percentile 95%",

    "input_prediction_file":
        str(PREDICTION_FILE),

    "input_prediction_sha256":
        prediction_hash,

    "input_step60a_file":
        str(STEP60A_FILE),

    "input_step60a_sha256":
        step60a_hash,

    "input_step60a_lock":
        str(STEP60A_LOCK),

    "input_step60a_lock_sha256":
        step60a_lock_hash,

    "output_replicates_csv":
        str(REPLICATES_CSV),

    "output_replicates_sha256":
        replicates_hash,

    "output_summary_csv":
        str(SUMMARY_CSV),

    "output_summary_sha256":
        summary_hash,

    "output_summary_json":
        str(SUMMARY_JSON),

    "output_summary_json_sha256":
        summary_json_hash,

    "mri_voxels_read":
        False,

    "checkpoint_tensors_loaded":
        False,

    "model_inference_rerun":
        False,

    "external_predictions_modified":
        False,

    "external_endpoints_modified":
        False,

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
# FINAL PUBLICATION-STYLE TABLE
# ============================================================

print()
print("=" * 74)
print(
    "FINAL STEP-60B EXTERNAL CALIBRATION UNCERTAINTY TABLE"
)
print("=" * 74)

for _, row in summary_df.iterrows():

    h = int(
        row[
            "horizon_months"
        ]
    )

    print()
    print(
        f"{h} MONTHS"
    )

    print(
        "Observed event rate : "
        f"{row['observed_event_rate_estimate']:.3f} "
        f"[{row['observed_event_rate_ci_lower']:.3f}, "
        f"{row['observed_event_rate_ci_upper']:.3f}]"
    )

    print(
        "Mean predicted risk : "
        f"{row['mean_predicted_risk_estimate']:.3f} "
        f"[{row['mean_predicted_risk_ci_lower']:.3f}, "
        f"{row['mean_predicted_risk_ci_upper']:.3f}]"
    )

    print(
        "Risk - observed     : "
        f"{row['risk_minus_observed_estimate']:.3f} "
        f"[{row['risk_minus_observed_ci_lower']:.3f}, "
        f"{row['risk_minus_observed_ci_upper']:.3f}]"
    )

    print(
        "Calibration int.    : "
        f"{row['calibration_intercept_estimate']:.3f} "
        f"[{row['calibration_intercept_ci_lower']:.3f}, "
        f"{row['calibration_intercept_ci_upper']:.3f}]"
    )

    print(
        "Calibration slope   : "
        f"{row['calibration_slope_estimate']:.3f} "
        f"[{row['calibration_slope_ci_lower']:.3f}, "
        f"{row['calibration_slope_ci_upper']:.3f}]"
    )

    print(
        "Brier skill score   : "
        f"{row['brier_skill_score_estimate']:.3f} "
        f"[{row['brier_skill_score_ci_lower']:.3f}, "
        f"{row['brier_skill_score_ci_upper']:.3f}]"
    )


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 74)
print("OUTPUT")
print("=" * 74)

print(
    "Replicates   :",
    REPLICATES_CSV,
)

print(
    "Summary CSV  :",
    SUMMARY_CSV,
)

print(
    "Summary JSON :",
    SUMMARY_JSON,
)

print(
    "Lock         :",
    LOCK_JSON,
)


# ============================================================
# HASHES
# ============================================================

print()
print("SHA256")

print(
    "Replicates   :",
    replicates_hash,
)

print(
    "Summary CSV  :",
    summary_hash,
)

print(
    "Summary JSON :",
    summary_json_hash,
)

print(
    "Lock         :",
    lock_hash,
)


# ============================================================
# FIREWALL
# ============================================================

print()
print("=" * 74)
print("IMPORTANT")
print("=" * 74)

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
    "PARTICIPANT-CLUSTER BOOTSTRAP   : YES"
)

print(
    "BOOTSTRAP REPLICATES            :",
    N_BOOTSTRAP,
)


print()
print(
    "Step 60B complete."
)

print(
    "The frozen external calibration/domain-shift "
    "estimates now have participant-cluster "
    "bootstrap uncertainty."
)

print(
    "No model or prediction changes are permitted "
    "after observing these results."
)