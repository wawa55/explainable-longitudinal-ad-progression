from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# STEP 55
# OASIS-3 FINAL PRE-INFERENCE INTEGRITY GATE
#
# PURPOSE
# -------
# Cross-check the already locked:
#
#   Step 49 external survival targets
#   Step 53 final external inference manifest
#   Step 53A endpoint-evaluable manifest
#   Step 54 endpoint contract
#
# BEFORE any frozen ADNI model checkpoint is loaded.
#
# THIS SCRIPT MUST NOT:
#   - read MRI voxel arrays
#   - load a neural-network checkpoint
#   - generate predictions
#   - calculate external performance
#   - modify Step 49 / 53 / 53A / 54 source files
# ============================================================


ROOT = Path(__file__).resolve().parent
COHORT_DIR = ROOT / "cohort_construction"

STEP49 = (
    COHORT_DIR
    / "OASIS3_STEP49_EXTERNAL_SURVIVAL_TARGETS_LOCKED.csv"
)

STEP53 = (
    COHORT_DIR
    / "OASIS3_STEP53_FINAL_EXTERNAL_INFERENCE_MANIFEST_LOCKED.csv"
)

STEP53A = (
    COHORT_DIR
    / "OASIS3_STEP53A_ENDPOINT_EVALUABLE_MANIFEST_LOCKED.csv"
)

STEP54_LOCK = (
    COHORT_DIR
    / "OASIS3_STEP54_EXTERNAL_ENDPOINT_CONTRACT_LOCK.json"
)

STEP54_SUMMARY = (
    COHORT_DIR
    / "OASIS3_STEP54_EXTERNAL_ENDPOINT_CONTRACT_SUMMARY.json"
)

OUT_AUDIT = (
    COHORT_DIR
    / "OASIS3_STEP55_PREINFERENCE_INTEGRITY_AUDIT.csv"
)

OUT_SUMMARY = (
    COHORT_DIR
    / "OASIS3_STEP55_PREINFERENCE_INTEGRITY_SUMMARY.json"
)

OUT_LOCK = (
    COHORT_DIR
    / "OASIS3_STEP55_PREINFERENCE_INTEGRITY_LOCK.json"
)


# ============================================================
# Locked expected values
# ============================================================

EXPECTED_STEP53_ROWS = 1559
EXPECTED_STEP53_SUBJECTS = 542

EXPECTED_STEP53A_ROWS = 1486
EXPECTED_STEP53A_SUBJECTS = 541

EXPECTED_STEP54_EVENTS = 344
EXPECTED_STEP54_CENSORED = 1142

EXPECTED_TOTAL_REQUIRED_MRI = 3729
EXPECTED_TOTAL_READY_MRI = 3729

EXPECTED_HORIZONS = [12, 24, 36, 48, 60]

DAYS_PER_YEAR = 365.25

EXPECTED_INTERVAL_EDGES_DAYS = [
    0.0,
    365.25,
    730.5,
    1095.75,
    1461.0,
    1826.25,
]


# ============================================================
# Helpers
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required locked source does not exist:\n{path}"
        )


def require_columns(df, columns, source_name):
    missing = [
        c
        for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{source_name}: "
            f"missing required columns: {missing}"
        )


def numeric_series(df, column):
    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def bool_series(series):

    true_values = {
        "true",
        "1",
        "1.0",
        "yes",
        "y",
        "t",
    }

    false_values = {
        "false",
        "0",
        "0.0",
        "no",
        "n",
        "f",
    }

    def convert(v):

        if pd.isna(v):
            return np.nan

        if isinstance(
            v,
            (bool, np.bool_),
        ):
            return bool(v)

        s = str(v).strip().lower()

        if s in true_values:
            return True

        if s in false_values:
            return False

        return np.nan

    return series.map(convert)


def normalized_id(series):
    return (
        series
        .astype(str)
        .str.strip()
    )


def safe_int(value):

    if pd.isna(value):
        return None

    return int(value)


def print_check(label, value):
    print(
        f"{label:<48}: {value}"
    )


# ============================================================
# Banner
# ============================================================

print("=" * 82)
print(
    "STEP 55 — OASIS-3 FINAL "
    "PRE-INFERENCE INTEGRITY GATE"
)
print("=" * 82)

print(
    "\nThis step performs "
    "metadata/contract integrity checks only."
)

print(
    "MRI voxel arrays will NOT be read."
)

print(
    "Frozen model checkpoints will NOT be loaded."
)

print(
    "Predictions will NOT be generated."
)

print(
    "External performance will NOT be examined."
)


# ============================================================
# Required files
# ============================================================

for p in [
    STEP49,
    STEP53,
    STEP53A,
    STEP54_LOCK,
    STEP54_SUMMARY,
]:
    require_file(p)


# ============================================================
# Freeze source hashes BEFORE audit
# ============================================================

source_hashes_before = {
    "step49": sha256_file(STEP49),
    "step53": sha256_file(STEP53),
    "step53a": sha256_file(STEP53A),
    "step54_lock": sha256_file(STEP54_LOCK),
    "step54_summary": sha256_file(STEP54_SUMMARY),
}


print("\nSOURCE FILES")
print("-" * 82)

print(
    "Step 49 :",
    STEP49,
)

print(
    "Step 53 :",
    STEP53,
)

print(
    "Step 53A:",
    STEP53A,
)

print(
    "Step 54 lock:",
    STEP54_LOCK,
)

print(
    "Step 54 summary:",
    STEP54_SUMMARY,
)


print("\nSOURCE SHA256")
print("-" * 82)

for k, v in source_hashes_before.items():
    print(
        f"{k:<20}: {v}"
    )


# ============================================================
# Read locked metadata
# ============================================================

step53 = pd.read_csv(
    STEP53,
    low_memory=False,
)

step53a = pd.read_csv(
    STEP53A,
    low_memory=False,
)

with open(
    STEP54_LOCK,
    "r",
    encoding="utf-8",
) as f:
    step54_lock = json.load(f)

with open(
    STEP54_SUMMARY,
    "r",
    encoding="utf-8",
) as f:
    step54_summary = json.load(f)


# ============================================================
# Required columns
# ============================================================

step53_required = [
    "locked_subject_id",
    "landmark_mri_id",
    "landmark_day",
    "n_visits",
    "mri_ids",
    "mri_days",
    "relative_years_from_first_mri",
    "years_from_first",
    "years_before_landmark",
    "n_required_mri",
    "n_existing_mri",
    "n_ready_mri",
    "n_missing_mri",
    "n_not_ready_mri",
    "sequence_model_input_ready",
]


step53a_required = [
    "locked_subject_id",
    "landmark_mri_id",
    "landmark_day",
    "event_observed",
    "censored",
    "observed_time_days",
    "observed_time_months",
    "event_within_60m",
    "time_within_60m_days",
    "time_within_60m_months",
    "event_by_12m",
    "evaluable_12m",
    "event_by_24m",
    "evaluable_24m",
    "event_by_36m",
    "evaluable_36m",
    "event_by_48m",
    "evaluable_48m",
    "event_by_60m",
    "evaluable_60m",
    "at_risk_interval_0",
    "event_interval_0",
    "at_risk_interval_1",
    "event_interval_1",
    "at_risk_interval_2",
    "event_interval_2",
    "at_risk_interval_3",
    "event_interval_3",
    "at_risk_interval_4",
    "event_interval_4",
]


require_columns(
    step53,
    step53_required,
    "Step-53",
)

require_columns(
    step53a,
    step53a_required,
    "Step-53A",
)


# ============================================================
# Audit container
# ============================================================

audit_rows = []


def add_check(
    name,
    observed,
    expected,
    passed,
):

    audit_rows.append(
        {
            "check": name,
            "observed": str(observed),
            "expected": str(expected),
            "passed": bool(passed),
        }
    )


# ============================================================
# Step 53 population checks
# ============================================================

print("\n" + "=" * 82)
print(
    "STEP 53 MODEL-INPUT READINESS AUDIT"
)
print("=" * 82)


n53 = len(step53)

subjects53 = (
    normalized_id(
        step53["locked_subject_id"]
    )
    .nunique()
)

seq_ready = bool_series(
    step53[
        "sequence_model_input_ready"
    ]
)

n_seq_missing = int(
    seq_ready.isna().sum()
)

n_seq_ready = int(
    (seq_ready == True).sum()
)

n_seq_not_ready = int(
    (seq_ready == False).sum()
)


n_required = numeric_series(
    step53,
    "n_required_mri",
)

n_existing = numeric_series(
    step53,
    "n_existing_mri",
)

n_ready = numeric_series(
    step53,
    "n_ready_mri",
)

n_missing = numeric_series(
    step53,
    "n_missing_mri",
)

n_not_ready = numeric_series(
    step53,
    "n_not_ready_mri",
)


total_required = int(
    n_required.sum()
)

total_ready = int(
    n_ready.sum()
)


required_ready_mismatch = int(
    (n_required != n_ready).sum()
)

required_existing_mismatch = int(
    (n_required != n_existing).sum()
)

missing_nonzero = int(
    (n_missing != 0).sum()
)

not_ready_nonzero = int(
    (n_not_ready != 0).sum()
)


print_check(
    "Landmarks",
    n53,
)

print_check(
    "Subjects",
    subjects53,
)

print_check(
    "Sequence ready TRUE",
    n_seq_ready,
)

print_check(
    "Sequence ready FALSE",
    n_seq_not_ready,
)

print_check(
    "Sequence ready missing/unparseable",
    n_seq_missing,
)

print_check(
    "Required != ready",
    required_ready_mismatch,
)

print_check(
    "Required != existing",
    required_existing_mismatch,
)

print_check(
    "Missing MRI nonzero",
    missing_nonzero,
)

print_check(
    "Not-ready MRI nonzero",
    not_ready_nonzero,
)

print_check(
    "Total required MRI",
    total_required,
)

print_check(
    "Total ready MRI",
    total_ready,
)


add_check(
    "step53_rows",
    n53,
    EXPECTED_STEP53_ROWS,
    n53 == EXPECTED_STEP53_ROWS,
)

add_check(
    "step53_subjects",
    subjects53,
    EXPECTED_STEP53_SUBJECTS,
    subjects53
    == EXPECTED_STEP53_SUBJECTS,
)

add_check(
    "step53_sequence_ready",
    n_seq_ready,
    EXPECTED_STEP53_ROWS,
    n_seq_ready
    == EXPECTED_STEP53_ROWS,
)

add_check(
    "step53_sequence_not_ready",
    n_seq_not_ready,
    0,
    n_seq_not_ready == 0,
)

add_check(
    "step53_sequence_ready_missing",
    n_seq_missing,
    0,
    n_seq_missing == 0,
)

add_check(
    "step53_required_ready_mismatch",
    required_ready_mismatch,
    0,
    required_ready_mismatch == 0,
)

add_check(
    "step53_required_existing_mismatch",
    required_existing_mismatch,
    0,
    required_existing_mismatch == 0,
)

add_check(
    "step53_missing_mri_nonzero",
    missing_nonzero,
    0,
    missing_nonzero == 0,
)

add_check(
    "step53_not_ready_mri_nonzero",
    not_ready_nonzero,
    0,
    not_ready_nonzero == 0,
)

add_check(
    "step53_total_required_mri",
    total_required,
    EXPECTED_TOTAL_REQUIRED_MRI,
    total_required
    == EXPECTED_TOTAL_REQUIRED_MRI,
)

add_check(
    "step53_total_ready_mri",
    total_ready,
    EXPECTED_TOTAL_READY_MRI,
    total_ready
    == EXPECTED_TOTAL_READY_MRI,
)


# ============================================================
# Step 53 unique landmark keys
# ============================================================

key_cols = [
    "locked_subject_id",
    "landmark_mri_id",
]

duplicate53 = int(
    step53.duplicated(
        subset=key_cols,
        keep=False,
    ).sum()
)

print_check(
    "Duplicate Step-53 landmark keys",
    duplicate53,
)

add_check(
    "step53_duplicate_landmark_keys",
    duplicate53,
    0,
    duplicate53 == 0,
)


# ============================================================
# Step 53 temporal checks
# ============================================================

landmark_day53 = numeric_series(
    step53,
    "landmark_day",
)

n_visits53 = numeric_series(
    step53,
    "n_visits",
)

bad_landmark_day = int(
    landmark_day53.isna().sum()
    + (landmark_day53 < 0).sum()
)

bad_n_visits = int(
    n_visits53.isna().sum()
    + (n_visits53 < 1).sum()
)

nvisit_required_mismatch = int(
    (
        n_visits53
        != n_required
    ).sum()
)


print_check(
    "Invalid landmark day",
    bad_landmark_day,
)

print_check(
    "Invalid n_visits",
    bad_n_visits,
)

print_check(
    "n_visits != n_required_mri",
    nvisit_required_mismatch,
)


add_check(
    "step53_invalid_landmark_day",
    bad_landmark_day,
    0,
    bad_landmark_day == 0,
)

add_check(
    "step53_invalid_n_visits",
    bad_n_visits,
    0,
    bad_n_visits == 0,
)

add_check(
    "step53_nvisits_required_mismatch",
    nvisit_required_mismatch,
    0,
    nvisit_required_mismatch == 0,
)


# ============================================================
# Step 53A population
# ============================================================

print("\n" + "=" * 82)
print(
    "STEP 53A ENDPOINT-EVALUABLE "
    "POPULATION AUDIT"
)
print("=" * 82)


n53a = len(step53a)

subjects53a = (
    normalized_id(
        step53a["locked_subject_id"]
    )
    .nunique()
)


event53a = bool_series(
    step53a[
        "event_observed"
    ]
)

censored53a = bool_series(
    step53a[
        "censored"
    ]
)


n_event = int(
    (event53a == True).sum()
)

n_censored = int(
    (censored53a == True).sum()
)

event_missing = int(
    event53a.isna().sum()
)

censored_missing = int(
    censored53a.isna().sum()
)

event_censor_fail = int(
    (
        event53a.notna()
        & censored53a.notna()
        & (
            event53a
            == censored53a
        )
    ).sum()
)


observed_days = numeric_series(
    step53a,
    "observed_time_days",
)

nonpositive_observed = int(
    (
        observed_days.isna()
        | (
            observed_days
            <= 0
        )
    ).sum()
)


print_check(
    "Landmarks",
    n53a,
)

print_check(
    "Subjects",
    subjects53a,
)

print_check(
    "Events",
    n_event,
)

print_check(
    "Censored",
    n_censored,
)

print_check(
    "Missing event flag",
    event_missing,
)

print_check(
    "Missing censor flag",
    censored_missing,
)

print_check(
    "Event/censor complement failures",
    event_censor_fail,
)

print_check(
    "Non-positive observed time",
    nonpositive_observed,
)


add_check(
    "step53a_rows",
    n53a,
    EXPECTED_STEP53A_ROWS,
    n53a == EXPECTED_STEP53A_ROWS,
)

add_check(
    "step53a_subjects",
    subjects53a,
    EXPECTED_STEP53A_SUBJECTS,
    subjects53a
    == EXPECTED_STEP53A_SUBJECTS,
)

add_check(
    "step53a_events",
    n_event,
    EXPECTED_STEP54_EVENTS,
    n_event
    == EXPECTED_STEP54_EVENTS,
)

add_check(
    "step53a_censored",
    n_censored,
    EXPECTED_STEP54_CENSORED,
    n_censored
    == EXPECTED_STEP54_CENSORED,
)

add_check(
    "step53a_event_missing",
    event_missing,
    0,
    event_missing == 0,
)

add_check(
    "step53a_censored_missing",
    censored_missing,
    0,
    censored_missing == 0,
)

add_check(
    "step53a_event_censor_complement",
    event_censor_fail,
    0,
    event_censor_fail == 0,
)

add_check(
    "step53a_positive_observed_time",
    nonpositive_observed,
    0,
    nonpositive_observed == 0,
)


# ============================================================
# Step 53A unique keys
# ============================================================

duplicate53a = int(
    step53a.duplicated(
        subset=key_cols,
        keep=False,
    ).sum()
)

print_check(
    "Duplicate Step-53A landmark keys",
    duplicate53a,
)

add_check(
    "step53a_duplicate_landmark_keys",
    duplicate53a,
    0,
    duplicate53a == 0,
)


# ============================================================
# Step 53A must be an exact subset of Step 53
# ============================================================

print("\n" + "=" * 82)
print(
    "STEP 53A → STEP 53 LANDMARK MAPPING"
)
print("=" * 82)


k53 = (
    step53[key_cols]
    .astype(str)
    .apply(
        lambda r:
        r["locked_subject_id"].strip()
        + "||"
        + r["landmark_mri_id"].strip(),
        axis=1,
    )
)

k53a = (
    step53a[key_cols]
    .astype(str)
    .apply(
        lambda r:
        r["locked_subject_id"].strip()
        + "||"
        + r["landmark_mri_id"].strip(),
        axis=1,
    )
)


set53 = set(k53)
set53a = set(k53a)

missing_from_step53 = sorted(
    set53a - set53
)

n_missing_from_step53 = len(
    missing_from_step53
)

excluded_from_endpoint = len(
    set53 - set53a
)


print_check(
    "Step-53A keys absent from Step-53",
    n_missing_from_step53,
)

print_check(
    "Step-53 landmarks excluded from endpoint set",
    excluded_from_endpoint,
)


add_check(
    "step53a_keys_missing_from_step53",
    n_missing_from_step53,
    0,
    n_missing_from_step53 == 0,
)

add_check(
    "step53_endpoint_exclusions",
    excluded_from_endpoint,
    (
        EXPECTED_STEP53_ROWS
        - EXPECTED_STEP53A_ROWS
    ),
    excluded_from_endpoint
    == (
        EXPECTED_STEP53_ROWS
        - EXPECTED_STEP53A_ROWS
    ),
)


# ============================================================
# Cross-check shared landmark day
# ============================================================

left = step53a[
    [
        "locked_subject_id",
        "landmark_mri_id",
        "landmark_day",
    ]
].copy()

right = step53[
    [
        "locked_subject_id",
        "landmark_mri_id",
        "landmark_day",
    ]
].copy()


merged = left.merge(
    right,
    on=[
        "locked_subject_id",
        "landmark_mri_id",
    ],
    how="left",
    suffixes=(
        "_53a",
        "_53",
    ),
    validate="one_to_one",
)


day53a = pd.to_numeric(
    merged[
        "landmark_day_53a"
    ],
    errors="coerce",
)

day53 = pd.to_numeric(
    merged[
        "landmark_day_53"
    ],
    errors="coerce",
)


landmark_day_mismatch = int(
    (
        day53a.isna()
        | day53.isna()
        | (
            ~np.isclose(
                day53a,
                day53,
                rtol=0.0,
                atol=1e-8,
            )
        )
    ).sum()
)


print_check(
    "Landmark-day mismatch",
    landmark_day_mismatch,
)

add_check(
    "step53_step53a_landmark_day_mismatch",
    landmark_day_mismatch,
    0,
    landmark_day_mismatch == 0,
)


# ============================================================
# Verify endpoint landmarks inherit model readiness
# ============================================================

ready_table = step53[
    [
        "locked_subject_id",
        "landmark_mri_id",
        "sequence_model_input_ready",
        "n_required_mri",
        "n_ready_mri",
        "n_missing_mri",
        "n_not_ready_mri",
    ]
].copy()


endpoint_ready = step53a[
    [
        "locked_subject_id",
        "landmark_mri_id",
    ]
].merge(
    ready_table,
    on=[
        "locked_subject_id",
        "landmark_mri_id",
    ],
    how="left",
    validate="one_to_one",
)


endpoint_seq_ready = bool_series(
    endpoint_ready[
        "sequence_model_input_ready"
    ]
)

endpoint_required = numeric_series(
    endpoint_ready,
    "n_required_mri",
)

endpoint_ready_n = numeric_series(
    endpoint_ready,
    "n_ready_mri",
)

endpoint_missing_n = numeric_series(
    endpoint_ready,
    "n_missing_mri",
)

endpoint_not_ready_n = numeric_series(
    endpoint_ready,
    "n_not_ready_mri",
)


endpoint_bad_ready = int(
    (
        endpoint_seq_ready.isna()
        | (
            endpoint_seq_ready
            != True
        )
    ).sum()
)

endpoint_required_ready_fail = int(
    (
        endpoint_required.isna()
        | endpoint_ready_n.isna()
        | (
            endpoint_required
            != endpoint_ready_n
        )
    ).sum()
)

endpoint_missing_fail = int(
    (
        endpoint_missing_n.isna()
        | (
            endpoint_missing_n
            != 0
        )
    ).sum()
)

endpoint_not_ready_fail = int(
    (
        endpoint_not_ready_n.isna()
        | (
            endpoint_not_ready_n
            != 0
        )
    ).sum()
)


print("\n" + "=" * 82)
print(
    "ENDPOINT LANDMARK MODEL-READINESS AUDIT"
)
print("=" * 82)


print_check(
    "Endpoint sequences not ready",
    endpoint_bad_ready,
)

print_check(
    "Endpoint required != ready MRI",
    endpoint_required_ready_fail,
)

print_check(
    "Endpoint missing MRI failure",
    endpoint_missing_fail,
)

print_check(
    "Endpoint not-ready MRI failure",
    endpoint_not_ready_fail,
)


add_check(
    "endpoint_sequence_not_ready",
    endpoint_bad_ready,
    0,
    endpoint_bad_ready == 0,
)

add_check(
    "endpoint_required_ready_mismatch",
    endpoint_required_ready_fail,
    0,
    endpoint_required_ready_fail == 0,
)

add_check(
    "endpoint_missing_mri_failure",
    endpoint_missing_fail,
    0,
    endpoint_missing_fail == 0,
)

add_check(
    "endpoint_not_ready_mri_failure",
    endpoint_not_ready_fail,
    0,
    endpoint_not_ready_fail == 0,
)


# ============================================================
# Horizon endpoint checks
#
# IMPORTANT:
# A horizon-specific event value may legitimately be missing
# when the landmark is NOT evaluable at that horizon.
#
# Therefore:
#   - missing event is a failure ONLY when evaluable == True
#   - evaluable itself must never be missing
#   - an observed horizon event requires evaluable == True
# ============================================================

print("\n" + "=" * 82)
print(
    "FROZEN HORIZON / INTERVAL AUDIT"
)
print("=" * 82)


horizon_failures = 0

for h in EXPECTED_HORIZONS:

    e_col = f"event_by_{h}m"
    q_col = f"evaluable_{h}m"

    event_h = bool_series(
        step53a[e_col]
    )

    eval_h = bool_series(
        step53a[q_col]
    )

    missing_event_when_evaluable = int(
        (
            (eval_h == True)
            & event_h.isna()
        ).sum()
    )

    missing_eval = int(
        eval_h.isna().sum()
    )

    event_not_evaluable = int(
        (
            (event_h == True)
            & (eval_h != True)
        ).sum()
    )

    this_fail = (
        missing_event_when_evaluable
        + missing_eval
        + event_not_evaluable
    )

    horizon_failures += this_fail

    print(
        f"{h:>2}m: "
        f"missing_event_when_evaluable="
        f"{missing_event_when_evaluable}, "
        f"missing_evaluable="
        f"{missing_eval}, "
        f"event_not_evaluable="
        f"{event_not_evaluable}"
    )


add_check(
    "horizon_contract_failures",
    horizon_failures,
    0,
    horizon_failures == 0,
)


# ============================================================
# Discrete interval consistency
# ============================================================

interval_failures = 0

for i in range(5):

    risk_col = (
        f"at_risk_interval_{i}"
    )

    event_col = (
        f"event_interval_{i}"
    )

    risk = bool_series(
        step53a[risk_col]
    )

    event = bool_series(
        step53a[event_col]
    )

    missing_risk = int(
        risk.isna().sum()
    )

    missing_event = int(
        event.isna().sum()
    )

    event_without_risk = int(
        (
            (event == True)
            & (risk != True)
        ).sum()
    )

    this_fail = (
        missing_risk
        + missing_event
        + event_without_risk
    )

    interval_failures += this_fail

    print(
        f"Interval {i}: "
        f"missing_risk={missing_risk}, "
        f"missing_event={missing_event}, "
        f"event_without_risk="
        f"{event_without_risk}"
    )


add_check(
    "interval_contract_failures",
    interval_failures,
    0,
    interval_failures == 0,
)


# ============================================================
# Verify Step 54 lock
# ============================================================

print("\n" + "=" * 82)
print(
    "STEP 54 LOCK VERIFICATION"
)
print("=" * 82)


lock_status = str(
    step54_lock.get(
        "status",
        "",
    )
)

lock_contract = str(
    step54_lock.get(
        "contract",
        "",
    )
)

lock_n_landmarks = safe_int(
    step54_lock.get(
        "n_landmarks"
    )
)

lock_n_subjects = safe_int(
    step54_lock.get(
        "n_subjects"
    )
)

lock_horizons = step54_lock.get(
    "horizons_months"
)

lock_edges = step54_lock.get(
    "interval_edges_days"
)

lock_days_per_year = step54_lock.get(
    "days_per_year"
)

source_modified = step54_lock.get(
    "source_modified"
)

checkpoint_loaded = step54_lock.get(
    "model_checkpoint_loaded"
)

voxels_read = step54_lock.get(
    "mri_voxels_read"
)

predictions_generated = step54_lock.get(
    "predictions_generated"
)

performance_examined = step54_lock.get(
    "external_performance_examined"
)


print_check(
    "Status",
    lock_status,
)

print_check(
    "Contract",
    lock_contract,
)

print_check(
    "Locked landmarks",
    lock_n_landmarks,
)

print_check(
    "Locked subjects",
    lock_n_subjects,
)

print_check(
    "Horizons",
    lock_horizons,
)

print_check(
    "Interval edges",
    lock_edges,
)

print_check(
    "Days/year",
    lock_days_per_year,
)

print_check(
    "Source modified",
    source_modified,
)

print_check(
    "Checkpoint loaded",
    checkpoint_loaded,
)

print_check(
    "MRI voxels read",
    voxels_read,
)

print_check(
    "Predictions generated",
    predictions_generated,
)

print_check(
    "Performance examined",
    performance_examined,
)


add_check(
    "step54_status",
    lock_status,
    "LOCKED_PASS",
    lock_status
    == "LOCKED_PASS",
)

add_check(
    "step54_landmarks",
    lock_n_landmarks,
    EXPECTED_STEP53A_ROWS,
    lock_n_landmarks
    == EXPECTED_STEP53A_ROWS,
)

add_check(
    "step54_subjects",
    lock_n_subjects,
    EXPECTED_STEP53A_SUBJECTS,
    lock_n_subjects
    == EXPECTED_STEP53A_SUBJECTS,
)

add_check(
    "step54_horizons",
    lock_horizons,
    EXPECTED_HORIZONS,
    lock_horizons
    == EXPECTED_HORIZONS,
)


edges_match = (
    isinstance(
        lock_edges,
        list,
    )
    and len(lock_edges)
    == len(
        EXPECTED_INTERVAL_EDGES_DAYS
    )
    and np.allclose(
        np.asarray(
            lock_edges,
            dtype=float,
        ),
        np.asarray(
            EXPECTED_INTERVAL_EDGES_DAYS,
            dtype=float,
        ),
        rtol=0.0,
        atol=1e-8,
    )
)


add_check(
    "step54_interval_edges",
    lock_edges,
    EXPECTED_INTERVAL_EDGES_DAYS,
    edges_match,
)


days_match = False

try:

    days_match = np.isclose(
        float(
            lock_days_per_year
        ),
        DAYS_PER_YEAR,
        rtol=0.0,
        atol=1e-8,
    )

except Exception:

    days_match = False


add_check(
    "step54_days_per_year",
    lock_days_per_year,
    DAYS_PER_YEAR,
    bool(days_match),
)


add_check(
    "step54_source_unmodified",
    source_modified,
    False,
    source_modified is False,
)

add_check(
    "step54_checkpoint_not_loaded",
    checkpoint_loaded,
    False,
    checkpoint_loaded is False,
)

add_check(
    "step54_voxels_not_read",
    voxels_read,
    False,
    voxels_read is False,
)

add_check(
    "step54_predictions_not_generated",
    predictions_generated,
    False,
    predictions_generated is False,
)

add_check(
    "step54_performance_not_examined",
    performance_examined,
    False,
    performance_examined is False,
)


# ============================================================
# Verify Step 54 immutable-source provenance
#
# Step 54 records Step 49 as its immutable source using:
#
#   source_step49
#   source_step49_sha256
#
# Therefore Step 55 must compare that recorded SHA directly
# against the CURRENT immutable Step-49 source.
#
# We deliberately do NOT compare this value against Step 53A.
# ============================================================

step49_current_hash = sha256_file(
    STEP49
)

lock_source_hash = str(
    step54_lock.get(
        "source_step49_sha256",
        "",
    )
)


source_hash_matches_lock = (
    lock_source_hash
    == step49_current_hash
)


print_check(
    "Step-54 recorded Step-49 SHA matches current Step-49",
    source_hash_matches_lock,
)


add_check(
    "step54_source_hash_matches_step49",
    source_hash_matches_lock,
    True,
    source_hash_matches_lock,
)


# ============================================================
# Protect all source files
# ============================================================

source_hashes_after = {
    "step49": sha256_file(STEP49),
    "step53": sha256_file(STEP53),
    "step53a": sha256_file(STEP53A),
    "step54_lock": sha256_file(STEP54_LOCK),
    "step54_summary": sha256_file(STEP54_SUMMARY),
}


sources_unchanged = (
    source_hashes_before
    == source_hashes_after
)


print("\n" + "=" * 82)
print(
    "SOURCE IMMUTABILITY CHECK"
)
print("=" * 82)

print_check(
    "All locked sources unchanged",
    sources_unchanged,
)


add_check(
    "locked_sources_unchanged",
    sources_unchanged,
    True,
    sources_unchanged,
)


# ============================================================
# Save audit
# ============================================================

audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    OUT_AUDIT,
    index=False,
)


n_checks = len(audit)

n_pass = int(
    audit["passed"].sum()
)

n_fail = int(
    (~audit["passed"]).sum()
)


# ============================================================
# Final decision
# ============================================================

print("\n" + "=" * 82)
print(
    "FINAL STEP 55 AUDIT"
)
print("=" * 82)


print_check(
    "Checks performed",
    n_checks,
)

print_check(
    "Checks passed",
    n_pass,
)

print_check(
    "Checks failed",
    n_fail,
)


summary = {

    "step": 55,

    "contract":
        "OASIS3_PREINFERENCE_INTEGRITY_GATE_V1",

    "status":
        (
            "READY_FOR_FROZEN_EXTERNAL_INFERENCE"
            if n_fail == 0
            else "NOT_READY"
        ),

    "step49_file":
        str(STEP49),

    "step49_sha256":
        source_hashes_before[
            "step49"
        ],

    "step53_file":
        str(STEP53),

    "step53_sha256":
        source_hashes_before[
            "step53"
        ],

    "step53a_file":
        str(STEP53A),

    "step53a_sha256":
        source_hashes_before[
            "step53a"
        ],

    "step54_lock_file":
        str(STEP54_LOCK),

    "step54_lock_sha256":
        source_hashes_before[
            "step54_lock"
        ],

    "step54_summary_file":
        str(STEP54_SUMMARY),

    "step54_summary_sha256":
        source_hashes_before[
            "step54_summary"
        ],

    "step54_recorded_source_step49_sha256":
        lock_source_hash,

    "step54_source_step49_hash_verified":
        bool(
            source_hash_matches_lock
        ),

    "n_step53_landmarks":
        int(n53),

    "n_step53_subjects":
        int(subjects53),

    "n_endpoint_landmarks":
        int(n53a),

    "n_endpoint_subjects":
        int(subjects53a),

    "n_events":
        int(n_event),

    "n_censored":
        int(n_censored),

    "total_required_mri":
        int(total_required),

    "total_ready_mri":
        int(total_ready),

    "horizons_months":
        EXPECTED_HORIZONS,

    "interval_edges_days":
        EXPECTED_INTERVAL_EDGES_DAYS,

    "days_per_year":
        DAYS_PER_YEAR,

    "n_checks":
        int(n_checks),

    "n_pass":
        int(n_pass),

    "n_fail":
        int(n_fail),

    "locked_sources_modified":
        not sources_unchanged,

    "mri_voxels_read":
        False,

    "model_checkpoint_loaded":
        False,

    "predictions_generated":
        False,

    "external_performance_examined":
        False,
}


with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# Refuse to lock if ANY check fails
# ============================================================

if n_fail != 0:

    print(
        "\nFAILED CHECKS"
    )

    print(
        "-" * 82
    )

    failed = audit.loc[
        ~audit["passed"]
    ]

    print(
        failed.to_string(
            index=False
        )
    )

    print(
        "\nSTEP 55 DID NOT LOCK."
    )

    print(
        "Do NOT run frozen external inference."
    )

    print(
        "\nAudit saved:"
    )

    print(
        OUT_AUDIT
    )

    print(
        "\nSummary saved:"
    )

    print(
        OUT_SUMMARY
    )

    raise RuntimeError(
        f"Step-55 integrity gate found "
        f"{n_fail} failed check(s)."
    )


# ============================================================
# Write final lock only after every check passes
# ============================================================

audit_hash = sha256_file(
    OUT_AUDIT
)

summary_hash = sha256_file(
    OUT_SUMMARY
)


lock = {

    **summary,

    "status":
        "READY_FOR_FROZEN_EXTERNAL_INFERENCE",

    "audit_file":
        str(OUT_AUDIT),

    "audit_sha256":
        audit_hash,

    "summary_file":
        str(OUT_SUMMARY),

    "summary_sha256":
        summary_hash,

    "authorization":
        (
            "Frozen external inference may proceed "
            "without model tuning or endpoint-driven "
            "adaptation."
        ),
}


with open(
    OUT_LOCK,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        lock,
        f,
        indent=2,
    )


lock_hash = sha256_file(
    OUT_LOCK
)


# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 82)

print(
    "STEP 55 PASSED — "
    "READY FOR FROZEN EXTERNAL INFERENCE"
)

print("=" * 82)


print(
    "\nPOPULATION"
)

print_check(
    "Model-ready Step-53 landmarks",
    n53,
)

print_check(
    "Model-ready Step-53 subjects",
    subjects53,
)

print_check(
    "Endpoint-evaluable landmarks",
    n53a,
)

print_check(
    "Endpoint-evaluable subjects",
    subjects53a,
)

print_check(
    "Events",
    n_event,
)

print_check(
    "Censored",
    n_censored,
)


print(
    "\nMRI READINESS"
)

print_check(
    "Total required MRI",
    total_required,
)

print_check(
    "Total ready MRI",
    total_ready,
)

print_check(
    "Missing MRI",
    missing_nonzero,
)

print_check(
    "Not-ready MRI",
    not_ready_nonzero,
)


print(
    "\nPROVENANCE"
)

print_check(
    "Step-54 Step-49 SHA verified",
    source_hash_matches_lock,
)

print_check(
    "All locked sources unchanged",
    sources_unchanged,
)


print(
    "\nOUTPUT"
)

print(
    OUT_AUDIT
)


print(
    "\nSUMMARY"
)

print(
    OUT_SUMMARY
)


print(
    "\nLOCK"
)

print(
    OUT_LOCK
)


print(
    "\nSHA256"
)

print(
    "Audit  :",
    audit_hash,
)

print(
    "Summary:",
    summary_hash,
)

print(
    "Lock   :",
    lock_hash,
)


print(
    "\nIMPORTANT"
)

print(
    "MRI VOXELS READ              : NO"
)

print(
    "FROZEN MODEL CHECKPOINT LOAD : NO"
)

print(
    "PREDICTIONS GENERATED        : NO"
)

print(
    "EXTERNAL PERFORMANCE EXAMINED: NO"
)

print(
    "MODEL TUNING                 : NO"
)


print(
    "\nThe endpoint and model-input "
    "contracts are frozen."
)

print(
    "The next stage may load the frozen "
    "ADNI M0 checkpoints for external inference."
)