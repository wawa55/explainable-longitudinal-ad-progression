# ============================================================
# STEP 59B
# OASIS-3 EXTERNAL VALIDATION
#
# PARTICIPANT-CLUSTER BOOTSTRAP
# FOR FROZEN M0 EXTERNAL PERFORMANCE
#
# Purpose
# ------------------------------------------------------------
# Quantify uncertainty around the LOCKED Step-59A external
# point estimates using participant-level cluster bootstrap.
#
# Resampling unit:
#   participant (locked_subject_id)
#
# All landmarks belonging to a sampled participant are retained.
#
# Bootstrap replicates:
#   2000
#
# Metrics:
#   ROC-AUC
#   Brier score
#
# Horizons:
#   12, 24, 36, 48, 60 months
#
# IMPORTANT
# ------------------------------------------------------------
# This script:
#   - DOES NOT train
#   - DOES NOT fine-tune
#   - DOES NOT recalibrate
#   - DOES NOT optimize thresholds
#   - DOES NOT select a model using OASIS-3
#   - DOES NOT modify predictions
#   - DOES NOT modify endpoints
#   - DOES NOT modify MRI
#
# It reads the frozen Step-58 predictions and the locked
# Step-59A point estimates only.
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

PREDICTIONS_CSV = (
    ROOT
    / "external_inference"
    / "OASIS3_STEP58_FROZEN_M0_EXTERNAL_PREDICTIONS.csv"
)

STEP59A_CSV = (
    ROOT
    / "external_evaluation"
    / "OASIS3_STEP59A_EXTERNAL_POINT_ESTIMATES.csv"
)

STEP59A_SUMMARY = (
    ROOT
    / "external_evaluation"
    / "OASIS3_STEP59A_EXTERNAL_POINT_ESTIMATES_SUMMARY.json"
)

STEP59A_LOCK = (
    ROOT
    / "external_evaluation"
    / "OASIS3_STEP59A_EXTERNAL_POINT_ESTIMATES_LOCK.json"
)

OUTPUT_DIR = ROOT / "external_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_REPLICATES_CSV = (
    OUTPUT_DIR
    / "OASIS3_STEP59B_PARTICIPANT_CLUSTER_BOOTSTRAP_REPLICATES.csv"
)

BOOTSTRAP_SUMMARY_CSV = (
    OUTPUT_DIR
    / "OASIS3_STEP59B_PARTICIPANT_CLUSTER_BOOTSTRAP_SUMMARY.csv"
)

BOOTSTRAP_SUMMARY_JSON = (
    OUTPUT_DIR
    / "OASIS3_STEP59B_PARTICIPANT_CLUSTER_BOOTSTRAP_SUMMARY.json"
)

BOOTSTRAP_LOCK_JSON = (
    OUTPUT_DIR
    / "OASIS3_STEP59B_PARTICIPANT_CLUSTER_BOOTSTRAP_LOCK.json"
)


# ============================================================
# LOCKED ANALYSIS CONTRACT
# ============================================================

SUBJECT_COL = "locked_subject_id"

HORIZONS = [12, 24, 36, 48, 60]

PRIMARY_HORIZONS = [12, 24, 36]

SECONDARY_HORIZONS = [48, 60]

N_BOOTSTRAP = 2000

BOOTSTRAP_SEED = 2026

CI_LOWER_PERCENTILE = 2.5
CI_UPPER_PERCENTILE = 97.5

EXPECTED_ROWS = 1559
EXPECTED_SUBJECTS = 542

EXPECTED_EVALUABLE = {
    12: 1453,
    24: 1383,
    36: 1242,
    48: 1152,
    60: 1045,
}

EXPECTED_EVENTS = {
    12: 48,
    24: 88,
    36: 135,
    48: 163,
    60: 198,
}


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
    missing = [
        c
        for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Required columns missing:\n"
            + "\n".join(missing)
        )


def percentile_ci(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    lower = float(
        np.percentile(
            values,
            CI_LOWER_PERCENTILE,
        )
    )

    upper = float(
        np.percentile(
            values,
            CI_UPPER_PERCENTILE,
        )
    )

    return lower, upper


def python_scalar(x):
    if isinstance(x, np.generic):
        return x.item()

    return x


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("STEP 59B — PARTICIPANT-CLUSTER BOOTSTRAP")
print("=" * 78)

print()
print(
    "Frozen OASIS-3 external validation uncertainty analysis"
)

print(
    f"Bootstrap replicates : {N_BOOTSTRAP}"
)

print(
    f"Bootstrap seed       : {BOOTSTRAP_SEED}"
)

print(
    "Resampling unit      : participant"
)

print(
    "Confidence interval  : percentile 95%"
)

print()


# ============================================================
# VERIFY INPUTS
# ============================================================

require_file(
    PREDICTIONS_CSV,
    "Step-58 predictions",
)

require_file(
    STEP59A_CSV,
    "Step-59A point estimates",
)

require_file(
    STEP59A_SUMMARY,
    "Step-59A summary",
)

require_file(
    STEP59A_LOCK,
    "Step-59A lock",
)

print("LOCKED INPUTS")

print(
    "Predictions :",
    PREDICTIONS_CSV,
)

print(
    "Step-59A    :",
    STEP59A_CSV,
)

print(
    "59A summary :",
    STEP59A_SUMMARY,
)

print(
    "59A lock    :",
    STEP59A_LOCK,
)

print()


# ============================================================
# INPUT HASHES
# ============================================================

predictions_hash = sha256_file(
    PREDICTIONS_CSV
)

step59a_hash = sha256_file(
    STEP59A_CSV
)

step59a_summary_hash = sha256_file(
    STEP59A_SUMMARY
)

step59a_lock_hash = sha256_file(
    STEP59A_LOCK
)

print("INPUT SHA256")

print(
    "Predictions :",
    predictions_hash,
)

print(
    "Step-59A    :",
    step59a_hash,
)

print(
    "59A summary :",
    step59a_summary_hash,
)

print(
    "59A lock    :",
    step59a_lock_hash,
)

print()


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(
    PREDICTIONS_CSV,
    low_memory=False,
)

point_df = pd.read_csv(
    STEP59A_CSV,
    low_memory=False,
)


# ============================================================
# GLOBAL CONTRACT CHECK
# ============================================================

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Step-58 row count mismatch: "
        f"{len(df)} != {EXPECTED_ROWS}"
    )

if SUBJECT_COL not in df.columns:
    raise RuntimeError(
        f"Missing subject column: "
        f"{SUBJECT_COL}"
    )

if df[SUBJECT_COL].isna().any():
    raise RuntimeError(
        "Missing participant IDs detected."
    )

n_subjects = int(
    df[SUBJECT_COL].nunique()
)

if n_subjects != EXPECTED_SUBJECTS:
    raise RuntimeError(
        f"Participant count mismatch: "
        f"{n_subjects} != {EXPECTED_SUBJECTS}"
    )

print("EXTERNAL POPULATION")

print(
    "Landmarks :",
    len(df),
)

print(
    "Subjects  :",
    n_subjects,
)

print()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    SUBJECT_COL,
]

for horizon in HORIZONS:
    required.extend(
        [
            f"event_by_{horizon}m",
            f"evaluable_{horizon}m",
            f"risk_{horizon}m_ensemble",
        ]
    )

require_columns(
    df,
    required,
)

require_columns(
    point_df,
    [
        "horizon_months",
        "n_evaluable_landmarks",
        "n_events",
        "roc_auc",
        "brier_score",
    ],
)

print(
    "PASS: required endpoint and prediction "
    "columns verified."
)

print()


# ============================================================
# VERIFY STEP-59A COUNTS AND POINT ESTIMATES
# ============================================================

print("=" * 78)
print("STEP-59A REPRODUCTION CHECK")
print("=" * 78)

point_lookup = {}

for horizon in HORIZONS:

    rows = point_df.loc[
        point_df["horizon_months"] == horizon
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one Step-59A row "
            f"for {horizon} months."
        )

    point_row = rows.iloc[0]

    eval_col = f"evaluable_{horizon}m"
    event_col = f"event_by_{horizon}m"
    risk_col = f"risk_{horizon}m_ensemble"

    if df[eval_col].dtype != bool:
        raise RuntimeError(
            f"{eval_col} is not boolean."
        )

    mask = df[eval_col] == True

    sub = df.loc[
        mask,
        [
            SUBJECT_COL,
            event_col,
            risk_col,
        ],
    ].copy()

    if sub[event_col].isna().any():
        raise RuntimeError(
            f"{event_col} contains missing values "
            f"among evaluable rows."
        )

    if sub[risk_col].isna().any():
        raise RuntimeError(
            f"{risk_col} contains missing values "
            f"among evaluable rows."
        )

    y = (
        sub[event_col]
        .astype(int)
        .to_numpy()
    )

    risk = (
        sub[risk_col]
        .astype(float)
        .to_numpy()
    )

    n_eval = int(len(sub))
    n_events = int(y.sum())

    if n_eval != EXPECTED_EVALUABLE[horizon]:
        raise RuntimeError(
            f"{horizon}m evaluable count mismatch: "
            f"{n_eval} != "
            f"{EXPECTED_EVALUABLE[horizon]}"
        )

    if n_events != EXPECTED_EVENTS[horizon]:
        raise RuntimeError(
            f"{horizon}m event count mismatch: "
            f"{n_events} != "
            f"{EXPECTED_EVENTS[horizon]}"
        )

    auc = float(
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

    locked_auc = float(
        point_row["roc_auc"]
    )

    locked_brier = float(
        point_row["brier_score"]
    )

    if not np.isclose(
        auc,
        locked_auc,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{horizon}m AUC does not reproduce "
            f"Step-59A."
        )

    if not np.isclose(
        brier,
        locked_brier,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{horizon}m Brier score does not "
            f"reproduce Step-59A."
        )

    point_lookup[horizon] = {
        "auc": auc,
        "brier": brier,
        "n_evaluable": n_eval,
        "n_events": n_events,
    }

    print(
        f"{horizon:>2}m : "
        f"N={n_eval:>4}  "
        f"events={n_events:>3}  "
        f"AUC={auc:.6f}  "
        f"Brier={brier:.6f}"
    )

print()

print(
    "PASS: all Step-59A point estimates "
    "reproduced exactly."
)

print()


# ============================================================
# PREPARE PARTICIPANT CLUSTERS
# ============================================================

subjects = np.asarray(
    sorted(
        df[SUBJECT_COL]
        .astype(str)
        .unique()
        .tolist()
    ),
    dtype=object,
)

if len(subjects) != EXPECTED_SUBJECTS:
    raise RuntimeError(
        "Unexpected participant vector length."
    )


# Map participant -> dataframe row indices.
#
# The dataframe index is reset so these positions are stable.
df = df.reset_index(drop=True)

subject_to_indices = {
    subject: np.asarray(
        df.index[
            df[SUBJECT_COL].astype(str)
            == subject
        ].tolist(),
        dtype=int,
    )
    for subject in subjects
}

if any(
    len(indices) == 0
    for indices in subject_to_indices.values()
):
    raise RuntimeError(
        "At least one participant has no rows."
    )

print("PARTICIPANT CLUSTER CONTRACT")

print(
    "Unique participants :",
    len(subjects),
)

print(
    "Total landmarks     :",
    len(df),
)

cluster_sizes = np.asarray(
    [
        len(subject_to_indices[s])
        for s in subjects
    ],
    dtype=int,
)

print(
    "Min landmarks/person:",
    int(cluster_sizes.min()),
)

print(
    "Max landmarks/person:",
    int(cluster_sizes.max()),
)

print(
    "Mean landmarks/person:",
    f"{cluster_sizes.mean():.4f}",
)

print()


# ============================================================
# BOOTSTRAP
# ============================================================

print("=" * 78)
print("RUNNING PARTICIPANT-CLUSTER BOOTSTRAP")
print("=" * 78)

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

bootstrap_records = []

invalid_auc_counts = {
    horizon: 0
    for horizon in HORIZONS
}

for bootstrap_id in range(
    1,
    N_BOOTSTRAP + 1,
):

    # --------------------------------------------------------
    # Sample PARTICIPANTS with replacement.
    #
    # Exactly 542 participant clusters are sampled in each
    # bootstrap replicate.
    # --------------------------------------------------------

    sampled_subjects = rng.choice(
        subjects,
        size=len(subjects),
        replace=True,
    )

    # --------------------------------------------------------
    # Important cluster-bootstrap detail:
    #
    # If a participant is sampled twice, that participant's
    # complete landmark cluster must appear twice.
    #
    # Therefore we concatenate the participant's row indices
    # for every occurrence in sampled_subjects.
    # --------------------------------------------------------

    sampled_indices = np.concatenate(
        [
            subject_to_indices[
                str(subject)
            ]
            for subject in sampled_subjects
        ]
    )

    boot = df.iloc[
        sampled_indices
    ]

    n_sampled_rows = int(
        len(boot)
    )

    n_unique_sampled_subjects = int(
        len(
            np.unique(
                sampled_subjects
            )
        )
    )

    for horizon in HORIZONS:

        eval_col = (
            f"evaluable_{horizon}m"
        )

        event_col = (
            f"event_by_{horizon}m"
        )

        risk_col = (
            f"risk_{horizon}m_ensemble"
        )

        evaluable_mask = (
            boot[eval_col] == True
        )

        horizon_boot = boot.loc[
            evaluable_mask,
            [
                event_col,
                risk_col,
            ],
        ]

        y = (
            horizon_boot[event_col]
            .astype(int)
            .to_numpy()
        )

        risk = (
            horizon_boot[risk_col]
            .astype(float)
            .to_numpy()
        )

        n_eval = int(
            len(y)
        )

        n_events = int(
            y.sum()
        )

        n_nonevents = int(
            n_eval - n_events
        )

        # ----------------------------------------------------
        # Brier score
        # ----------------------------------------------------

        if n_eval == 0:
            brier = np.nan

        else:
            brier = float(
                brier_score_loss(
                    y,
                    risk,
                )
            )

        # ----------------------------------------------------
        # ROC-AUC
        #
        # A bootstrap sample containing only one outcome class
        # cannot define ROC-AUC. Record NaN and count it.
        # ----------------------------------------------------

        if (
            n_eval == 0
            or len(np.unique(y)) < 2
        ):
            auc = np.nan

            invalid_auc_counts[
                horizon
            ] += 1

        else:
            auc = float(
                roc_auc_score(
                    y,
                    risk,
                )
            )

        bootstrap_records.append(
            {
                "bootstrap_id": int(
                    bootstrap_id
                ),
                "horizon_months": int(
                    horizon
                ),
                "n_sampled_clusters": int(
                    len(subjects)
                ),
                "n_unique_sampled_subjects": (
                    n_unique_sampled_subjects
                ),
                "n_sampled_landmarks_all": (
                    n_sampled_rows
                ),
                "n_evaluable_landmarks": (
                    n_eval
                ),
                "n_events": (
                    n_events
                ),
                "n_nonevents": (
                    n_nonevents
                ),
                "roc_auc": (
                    auc
                ),
                "brier_score": (
                    brier
                ),
            }
        )

    if (
        bootstrap_id == 1
        or bootstrap_id % 100 == 0
        or bootstrap_id == N_BOOTSTRAP
    ):
        print(
            f"Bootstrap "
            f"{bootstrap_id:>4}"
            f"/{N_BOOTSTRAP} complete"
        )


# ============================================================
# SAVE REPLICATES
# ============================================================

bootstrap_df = pd.DataFrame(
    bootstrap_records
)

expected_bootstrap_rows = (
    N_BOOTSTRAP
    * len(HORIZONS)
)

if (
    len(bootstrap_df)
    != expected_bootstrap_rows
):
    raise RuntimeError(
        "Unexpected number of bootstrap "
        "result rows."
    )

bootstrap_df.to_csv(
    BOOTSTRAP_REPLICATES_CSV,
    index=False,
)

print()

print(
    "PASS: bootstrap replicate table "
    "created."
)

print(
    "Rows:",
    len(bootstrap_df),
)

print()


# ============================================================
# SUMMARIZE BOOTSTRAP DISTRIBUTIONS
# ============================================================

summary_records = []

print("=" * 78)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("=" * 78)

for horizon in HORIZONS:

    hboot = bootstrap_df.loc[
        bootstrap_df[
            "horizon_months"
        ] == horizon
    ].copy()

    auc_values = (
        hboot["roc_auc"]
        .to_numpy(
            dtype=float
        )
    )

    brier_values = (
        hboot["brier_score"]
        .to_numpy(
            dtype=float
        )
    )

    auc_valid = np.isfinite(
        auc_values
    )

    brier_valid = np.isfinite(
        brier_values
    )

    n_auc_valid = int(
        auc_valid.sum()
    )

    n_brier_valid = int(
        brier_valid.sum()
    )

    auc_lower, auc_upper = (
        percentile_ci(
            auc_values
        )
    )

    brier_lower, brier_upper = (
        percentile_ci(
            brier_values
        )
    )

    auc_point = float(
        point_lookup[horizon][
            "auc"
        ]
    )

    brier_point = float(
        point_lookup[horizon][
            "brier"
        ]
    )

    auc_boot_mean = float(
        np.nanmean(
            auc_values
        )
    )

    brier_boot_mean = float(
        np.nanmean(
            brier_values
        )
    )

    record = {
        "horizon_months": int(
            horizon
        ),
        "n_evaluable_landmarks": int(
            point_lookup[horizon][
                "n_evaluable"
            ]
        ),
        "n_events": int(
            point_lookup[horizon][
                "n_events"
            ]
        ),
        "roc_auc": auc_point,
        "roc_auc_ci_lower": (
            auc_lower
        ),
        "roc_auc_ci_upper": (
            auc_upper
        ),
        "roc_auc_bootstrap_mean": (
            auc_boot_mean
        ),
        "roc_auc_valid_replicates": (
            n_auc_valid
        ),
        "roc_auc_invalid_replicates": int(
            N_BOOTSTRAP
            - n_auc_valid
        ),
        "brier_score": (
            brier_point
        ),
        "brier_ci_lower": (
            brier_lower
        ),
        "brier_ci_upper": (
            brier_upper
        ),
        "brier_bootstrap_mean": (
            brier_boot_mean
        ),
        "brier_valid_replicates": (
            n_brier_valid
        ),
    }

    summary_records.append(
        record
    )

    print()

    print(
        f"{horizon}-MONTH HORIZON"
    )

    print("-" * 48)

    print(
        f"ROC-AUC : "
        f"{auc_point:.6f} "
        f"[{auc_lower:.6f}, "
        f"{auc_upper:.6f}]"
    )

    print(
        f"Brier   : "
        f"{brier_point:.6f} "
        f"[{brier_lower:.6f}, "
        f"{brier_upper:.6f}]"
    )

    print(
        f"Valid AUC replicates : "
        f"{n_auc_valid}/{N_BOOTSTRAP}"
    )


summary_df = pd.DataFrame(
    summary_records
)

summary_df.to_csv(
    BOOTSTRAP_SUMMARY_CSV,
    index=False,
)


# ============================================================
# BOOTSTRAP QUALITY AUDIT
# ============================================================

print()
print("=" * 78)
print("BOOTSTRAP QUALITY AUDIT")
print("=" * 78)

for horizon in HORIZONS:

    invalid = int(
        invalid_auc_counts[
            horizon
        ]
    )

    print(
        f"{horizon:>2}m invalid "
        f"AUC replicates : "
        f"{invalid}"
    )

# With the observed event counts, we expect the vast majority
# (normally all) participant-level bootstrap samples to contain
# both classes. We do not silently discard the count: it is
# explicitly recorded.

for horizon in HORIZONS:

    hboot = bootstrap_df.loc[
        bootstrap_df[
            "horizon_months"
        ] == horizon
    ]

    if len(hboot) != N_BOOTSTRAP:
        raise RuntimeError(
            f"{horizon}m does not have "
            f"{N_BOOTSTRAP} bootstrap rows."
        )

print()

print(
    "PASS: every horizon contains exactly "
    f"{N_BOOTSTRAP} bootstrap replicates."
)

print(
    "PASS: participant was the bootstrap "
    "resampling unit."
)

print(
    "PASS: repeated landmarks remained within "
    "their participant clusters."
)

print()


# ============================================================
# WRITE SUMMARY JSON
# ============================================================

summary_json = {
    "step": "59B",
    "analysis": (
        "Participant-cluster bootstrap uncertainty "
        "for frozen OASIS-3 external validation"
    ),
    "resampling_unit": (
        "locked_subject_id"
    ),
    "n_bootstrap_replicates": (
        N_BOOTSTRAP
    ),
    "bootstrap_seed": (
        BOOTSTRAP_SEED
    ),
    "confidence_interval": {
        "method": (
            "percentile bootstrap"
        ),
        "level": 0.95,
        "lower_percentile": (
            CI_LOWER_PERCENTILE
        ),
        "upper_percentile": (
            CI_UPPER_PERCENTILE
        ),
    },
    "n_landmarks": int(
        len(df)
    ),
    "n_subjects": int(
        n_subjects
    ),
    "horizons_months": (
        HORIZONS
    ),
    "primary_horizons_months": (
        PRIMARY_HORIZONS
    ),
    "secondary_horizons_months": (
        SECONDARY_HORIZONS
    ),
    "input_hashes": {
        "step58_predictions": (
            predictions_hash
        ),
        "step59a_point_estimates": (
            step59a_hash
        ),
        "step59a_summary": (
            step59a_summary_hash
        ),
        "step59a_lock": (
            step59a_lock_hash
        ),
    },
    "results": [
        {
            k: python_scalar(v)
            for k, v in record.items()
        }
        for record in summary_records
    ],
    "model_training": False,
    "model_fine_tuning": False,
    "model_recalibration": False,
    "threshold_optimization": False,
    "oasis_based_model_selection": False,
    "predictions_modified": False,
    "endpoints_modified": False,
    "mri_modified": False,
}

with open(
    BOOTSTRAP_SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary_json,
        f,
        indent=2,
    )


# ============================================================
# HASH OUTPUTS
# ============================================================

replicates_hash = sha256_file(
    BOOTSTRAP_REPLICATES_CSV
)

summary_csv_hash = sha256_file(
    BOOTSTRAP_SUMMARY_CSV
)

summary_json_hash = sha256_file(
    BOOTSTRAP_SUMMARY_JSON
)


# ============================================================
# WRITE LOCK
# ============================================================

lock = {
    "step": "59B",
    "status": "LOCKED",
    "analysis": (
        "Participant-cluster bootstrap"
    ),
    "resampling_unit": (
        SUBJECT_COL
    ),
    "n_bootstrap_replicates": (
        N_BOOTSTRAP
    ),
    "bootstrap_seed": (
        BOOTSTRAP_SEED
    ),
    "confidence_interval_method": (
        "2.5th and 97.5th percentile"
    ),
    "input_hashes": {
        "step58_predictions": (
            predictions_hash
        ),
        "step59a_point_estimates": (
            step59a_hash
        ),
        "step59a_summary": (
            step59a_summary_hash
        ),
        "step59a_lock": (
            step59a_lock_hash
        ),
    },
    "output_hashes": {
        "bootstrap_replicates": (
            replicates_hash
        ),
        "bootstrap_summary_csv": (
            summary_csv_hash
        ),
        "bootstrap_summary_json": (
            summary_json_hash
        ),
    },
    "prohibitions": {
        "model_training": True,
        "model_fine_tuning": True,
        "model_recalibration": True,
        "threshold_optimization": True,
        "oasis_based_model_selection": True,
        "prediction_modification": True,
        "endpoint_modification": True,
    },
    "next_analysis": (
        "Locked external survival-performance "
        "analysis and calibration/domain-shift analysis"
    ),
}

with open(
    BOOTSTRAP_LOCK_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        lock,
        f,
        indent=2,
    )

lock_hash = sha256_file(
    BOOTSTRAP_LOCK_JSON
)


# ============================================================
# FINAL PUBLICATION TABLE
# ============================================================

print()
print("=" * 78)
print("FINAL STEP-59B EXTERNAL PERFORMANCE TABLE")
print("=" * 78)

for record in summary_records:

    h = record[
        "horizon_months"
    ]

    print(
        f"{h:>2}m | "
        f"AUC "
        f"{record['roc_auc']:.3f} "
        f"[{record['roc_auc_ci_lower']:.3f}, "
        f"{record['roc_auc_ci_upper']:.3f}] | "
        f"Brier "
        f"{record['brier_score']:.3f} "
        f"[{record['brier_ci_lower']:.3f}, "
        f"{record['brier_ci_upper']:.3f}]"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 78)
print("OUTPUT")
print("=" * 78)

print(
    "Replicates :",
    BOOTSTRAP_REPLICATES_CSV,
)

print(
    "Summary CSV:",
    BOOTSTRAP_SUMMARY_CSV,
)

print(
    "Summary JSON:",
    BOOTSTRAP_SUMMARY_JSON,
)

print(
    "Lock        :",
    BOOTSTRAP_LOCK_JSON,
)

print()

print("SHA256")

print(
    "Replicates :",
    replicates_hash,
)

print(
    "Summary CSV:",
    summary_csv_hash,
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
print("=" * 78)
print("IMPORTANT")
print("=" * 78)

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
    "Step 59B complete."
)

print(
    "The frozen external point estimates now have "
    "participant-cluster bootstrap uncertainty."
)

print(
    "No model or prediction changes are permitted "
    "after observing these external results."
)