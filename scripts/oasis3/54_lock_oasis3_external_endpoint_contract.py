from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime

# ============================================================
# STEP 54
# OASIS-3 EXTERNAL ENDPOINT CONTRACT AUDIT AND LOCK
#
# PURPOSE
# -------
# Audit and freeze the already-constructed OASIS-3 external
# survival targets before frozen ADNI model inference.
#
# IMPORTANT
# ---------
# - Does NOT modify Step-49.
# - Does NOT read MRI voxels.
# - Does NOT load any ADNI checkpoint.
# - Does NOT generate predictions.
# - Does NOT inspect model performance.
# ============================================================

ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

COHORT_DIR = ROOT / "cohort_construction"

# ============================================================
# Helpers
# ============================================================

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_unique_step49_csv():
    candidates = sorted(COHORT_DIR.glob("*STEP49*.csv"))

    # Ignore any future Step54 products if names overlap unexpectedly.
    candidates = [
        p for p in candidates
        if "STEP54" not in p.name.upper()
    ]

    if len(candidates) == 0:
        raise RuntimeError(
            "Could not find a Step-49 CSV in cohort_construction."
        )

    if len(candidates) > 1:
        print("\nSTEP-49 CSV CANDIDATES:")
        for p in candidates:
            print("  ", p)

        # Prefer a filename containing TARGET.
        target_candidates = [
            p for p in candidates
            if "TARGET" in p.name.upper()
        ]

        if len(target_candidates) == 1:
            return target_candidates[0]

        raise RuntimeError(
            "More than one Step-49 CSV found. "
            "Cannot safely choose the endpoint source."
        )

    return candidates[0]


def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label}: missing required columns: {missing}"
        )


def numeric_series(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def bool_series(s):
    """
    Strictly standardize common bool-like representations.
    Raises if an unexpected non-missing representation occurs.
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }

    out = []
    bad = []

    for idx, value in s.items():
        if pd.isna(value):
            out.append(np.nan)
            continue

        key = str(value).strip().lower()

        if key not in mapping:
            bad.append((idx, value))
            out.append(np.nan)
        else:
            out.append(mapping[key])

    if bad:
        raise RuntimeError(
            f"Unexpected boolean values in column {s.name}: "
            f"{bad[:10]}"
        )

    return pd.Series(out, index=s.index, name=s.name)


# ============================================================
# Locate immutable Step-49 source
# ============================================================

STEP49 = find_unique_step49_csv()
STEP53A = COHORT_DIR / "OASIS3_STEP53A_ENDPOINT_EVALUABLE_MANIFEST_LOCKED.csv"
if not STEP53A.exists():
    raise FileNotFoundError(f"Missing locked Step-53A endpoint-evaluable manifest: {STEP53A}")
STEP49_HASH_BEFORE = sha256_file(STEP49)
STEP53A_HASH_BEFORE = sha256_file(STEP53A)

OUT_AUDIT = (
    COHORT_DIR /
    "OASIS3_STEP54_EXTERNAL_ENDPOINT_AUDIT.csv"
)

OUT_SUMMARY = (
    COHORT_DIR /
    "OASIS3_STEP54_EXTERNAL_ENDPOINT_CONTRACT_SUMMARY.json"
)

OUT_LOCK = (
    COHORT_DIR /
    "OASIS3_STEP54_EXTERNAL_ENDPOINT_CONTRACT_LOCK.json"
)

print("=" * 78)
print("STEP 54 - OASIS-3 EXTERNAL ENDPOINT CONTRACT AUDIT AND LOCK")
print("=" * 78)

print("\nSOURCE:")
print(STEP53A)

print("\nSOURCE SHA256:")
print(STEP53A_HASH_BEFORE)

# ============================================================
# Read Step-49
# ============================================================

df = pd.read_csv(STEP53A, low_memory=False)

# Step-53A compatibility aliases for the legacy Step-54 audit interface.
# These are created in memory only; the locked Step-53A source is not modified.
if "locked_mri_id" not in df.columns and "landmark_mri_id" in df.columns:
    df["locked_mri_id"] = df["landmark_mri_id"]
if "mri_day" not in df.columns and "landmark_day" in df.columns:
    df["mri_day"] = pd.to_numeric(df["landmark_day"], errors="coerce")

print("\nSOURCE DIMENSIONS")
print("Rows    :", len(df))
print("Columns :", len(df.columns))

if len(df) != 1486:
    raise RuntimeError(
        f"Expected 1486 Step-53A endpoint-evaluable landmarks, found {len(df)}."
    )

# ============================================================
# Required schema
# ============================================================

base_required = [
    "locked_subject_id",
    "locked_mri_id",
    "mri_day",
    "event_observed",
    "event_day_from_mri",
    "censored",
    "observed_time_days",
    "observed_time_months",
    "event_within_60m",
    "time_within_60m_days",
    "time_within_60m_months",
]

horizons = [12, 24, 36, 48, 60]

for h in horizons:
    base_required.extend([
        f"event_by_{h}m",
        f"evaluable_{h}m",
    ])

for k in range(5):
    base_required.extend([
        f"at_risk_interval_{k}",
        f"event_interval_{k}",
    ])

require_columns(df, base_required, "Step-49")

print("\nREQUIRED TARGET SCHEMA: PASS")

# ============================================================
# Landmark key audit
# ============================================================

key_cols = [
    "locked_subject_id",
    "locked_mri_id",
]

missing_keys = int(
    df[key_cols]
    .isna()
    .any(axis=1)
    .sum()
)

duplicate_keys = int(
    df.duplicated(key_cols, keep=False).sum()
)

print("\nLANDMARK KEY AUDIT")
print("Missing key rows :", missing_keys)
print("Duplicate rows   :", duplicate_keys)

if missing_keys != 0:
    raise RuntimeError("Missing landmark identifiers detected.")

if duplicate_keys != 0:
    raise RuntimeError("Duplicate landmark identifiers detected.")

# ============================================================
# Standardize values in memory only
# ============================================================

event_observed = bool_series(df["event_observed"])
censored = bool_series(df["censored"])
event_within_60m = bool_series(df["event_within_60m"])

observed_time_days = numeric_series(
    df, "observed_time_days"
)

observed_time_months = numeric_series(
    df, "observed_time_months"
)

event_day_from_mri = numeric_series(
    df, "event_day_from_mri"
)

time_within_60m_days = numeric_series(
    df, "time_within_60m_days"
)

time_within_60m_months = numeric_series(
    df, "time_within_60m_months"
)

# ============================================================
# Basic event/censoring contract
# ============================================================

missing_event_flag = int(event_observed.isna().sum())
missing_censor_flag = int(censored.isna().sum())

event_censor_complement_fail = int(
    (
        event_observed.notna()
        & censored.notna()
        & (event_observed == censored)
    ).sum()
)

event_missing_time = int(
    (
        (event_observed == True)
        & event_day_from_mri.isna()
    ).sum()
)

event_nonpositive_time = int(
    (
        (event_observed == True)
        & event_day_from_mri.notna()
        & (event_day_from_mri <= 0)
    ).sum()
)

censored_with_event_day = int(
    (
        (event_observed == False)
        & event_day_from_mri.notna()
    ).sum()
)

missing_observed_time = int(
    observed_time_days.isna().sum()
)

nonpositive_observed_time = int(
    (
        observed_time_days.notna()
        & (observed_time_days <= 0)
    ).sum()
)

event_time_vs_observed_fail = int(
    (
        (event_observed == True)
        & event_day_from_mri.notna()
        & observed_time_days.notna()
        & (
            np.abs(
                event_day_from_mri
                - observed_time_days
            ) > 1e-6
        )
    ).sum()
)

print("\nEVENT / CENSORING CONTRACT")
print("Missing event flags             :", missing_event_flag)
print("Missing censor flags            :", missing_censor_flag)
print("Event/censor complement failures:", event_censor_complement_fail)
print("Events missing event time       :", event_missing_time)
print("Events with nonpositive time    :", event_nonpositive_time)
print("Censored rows with event day    :", censored_with_event_day)
print("Missing observed time           :", missing_observed_time)
print("Nonpositive observed time       :", nonpositive_observed_time)
print("Event-time consistency failures :", event_time_vs_observed_fail)

# ============================================================
# Day/month conversion audit
# ============================================================

expected_months = (
    observed_time_days / 365.25 * 12.0
)

month_conversion_fail = int(
    (
        observed_time_days.notna()
        & observed_time_months.notna()
        & (
            np.abs(
                expected_months
                - observed_time_months
            ) > 1e-4
        )
    ).sum()
)

expected_60m_months = (
    time_within_60m_days / 365.25 * 12.0
)

month60_conversion_fail = int(
    (
        time_within_60m_days.notna()
        & time_within_60m_months.notna()
        & (
            np.abs(
                expected_60m_months
                - time_within_60m_months
            ) > 1e-4
        )
    ).sum()
)

print("\nTIME-SCALE AUDIT")
print("Observed day/month failures :", month_conversion_fail)
print("60m day/month failures      :", month60_conversion_fail)

# ============================================================
# Frozen horizon definitions
# ============================================================

DAYS_PER_YEAR = 365.25

horizon_days = {
    12: 365.25,
    24: 730.50,
    36: 1095.75,
    48: 1461.00,
    60: 1826.25,
}

print("\nFROZEN HORIZONS")
for h in horizons:
    print(
        f"{h:2d} months = "
        f"{horizon_days[h]:.2f} days"
    )

# ============================================================
# Horizon-specific target audit
# ============================================================

horizon_summary = {}

for h in horizons:

    event_col = f"event_by_{h}m"
    eval_col = f"evaluable_{h}m"

    e = numeric_series(df, event_col)
    q = bool_series(df[eval_col])

    bad_event_values = int(
        (
            e.notna()
            & ~e.isin([0.0, 1.0])
        ).sum()
    )

    missing_eval = int(q.isna().sum())

    event_not_evaluable = int(
        (
            (e == 1.0)
            & (q != True)
        ).sum()
    )

    # If event occurs by horizon:
    # event must be observed and event time <= horizon.
    expected_event = (
        (event_observed == True)
        & event_day_from_mri.notna()
        & (
            event_day_from_mri
            <= horizon_days[h] + 1e-6
        )
    )

    observed_event = (e == 1.0)

    event_definition_fail = int(
        (
            expected_event
            != observed_event
        ).sum()
    )

    # A landmark is evaluable at horizon if:
    # 1) event occurred by horizon, OR
    # 2) observed event-free follow-up reaches horizon.
    expected_evaluable = (
        expected_event
        |
        (
            observed_time_days.notna()
            & (
                observed_time_days
                >= horizon_days[h] - 1e-6
            )
        )
    )

    evaluable_definition_fail = int(
        (
            expected_evaluable
            != (q == True)
        ).sum()
    )

    n_events = int((e == 1.0).sum())
    n_evaluable = int((q == True).sum())

    print(f"\nHORIZON {h} MONTHS")
    print("Events                 :", n_events)
    print("Evaluable landmarks    :", n_evaluable)
    print("Bad event values       :", bad_event_values)
    print("Missing evaluable flag :", missing_eval)
    print("Event not evaluable    :", event_not_evaluable)
    print("Event definition fails :", event_definition_fail)
    print("Evaluable definition fails:",
          evaluable_definition_fail)

    horizon_summary[str(h)] = {
        "horizon_days": horizon_days[h],
        "n_events": n_events,
        "n_evaluable": n_evaluable,
        "bad_event_values": bad_event_values,
        "missing_evaluable": missing_eval,
        "event_not_evaluable": event_not_evaluable,
        "event_definition_failures": event_definition_fail,
        "evaluable_definition_failures":
            evaluable_definition_fail,
    }

# ============================================================
# 60-month truncation contract
# ============================================================

expected_time60 = np.minimum(
    observed_time_days,
    horizon_days[60],
)

time60_fail = int(
    (
        time_within_60m_days.notna()
        & expected_time60.notna()
        & (
            np.abs(
                time_within_60m_days
                - expected_time60
            ) > 1e-6
        )
    ).sum()
)

expected_event60 = (
    (event_observed == True)
    & event_day_from_mri.notna()
    & (
        event_day_from_mri
        <= horizon_days[60] + 1e-6
    )
)

event60_fail = int(
    (
        expected_event60
        != (event_within_60m == True)
    ).sum()
)

print("\n60-MONTH TRUNCATION CONTRACT")
print("Time truncation failures :", time60_fail)
print("Event truncation failures:", event60_fail)

# ============================================================
# Discrete interval contract
#
# intervals:
# 0 = (0,12]
# 1 = (12,24]
# 2 = (24,36]
# 3 = (36,48]
# 4 = (48,60]
# ============================================================

edges = [
    0.0,
    365.25,
    730.50,
    1095.75,
    1461.00,
    1826.25,
]

interval_summary = {}

for k in range(5):

    at_risk = bool_series(
        df[f"at_risk_interval_{k}"]
    )

    interval_event = bool_series(
        df[f"event_interval_{k}"]
    )

    lo = edges[k]
    hi = edges[k + 1]

    expected_at_risk = (
        observed_time_days.notna()
        & (observed_time_days > lo + 1e-6)
    )

    expected_interval_event = (
        (event_observed == True)
        & event_day_from_mri.notna()
        & (event_day_from_mri > lo + 1e-6)
        & (event_day_from_mri <= hi + 1e-6)
    )

    at_risk_fail = int(
        (
            expected_at_risk
            != (at_risk == True)
        ).sum()
    )

    interval_event_fail = int(
        (
            expected_interval_event
            != (interval_event == True)
        ).sum()
    )

    event_without_risk = int(
        (
            (interval_event == True)
            & (at_risk != True)
        ).sum()
    )

    print(f"\nINTERVAL {k}: ({lo:.2f}, {hi:.2f}] DAYS")
    print(
        "At risk                :",
        int((at_risk == True).sum())
    )
    print(
        "Events                 :",
        int((interval_event == True).sum())
    )
    print(
        "At-risk definition fail:",
        at_risk_fail
    )
    print(
        "Event definition fail  :",
        interval_event_fail
    )
    print(
        "Event without at-risk  :",
        event_without_risk
    )

    interval_summary[str(k)] = {
        "lower_day_exclusive": lo,
        "upper_day_inclusive": hi,
        "n_at_risk":
            int((at_risk == True).sum()),
        "n_events":
            int((interval_event == True).sum()),
        "at_risk_definition_failures":
            at_risk_fail,
        "event_definition_failures":
            interval_event_fail,
        "event_without_at_risk":
            event_without_risk,
    }

# ============================================================
# Cross-horizon monotonicity
# ============================================================

event_matrix = np.column_stack([
    numeric_series(
        df,
        f"event_by_{h}m"
    ).fillna(0).to_numpy()
    for h in horizons
])

eval_matrix = np.column_stack([
    bool_series(
        df[f"evaluable_{h}m"]
    ).fillna(False).astype(int).to_numpy()
    for h in horizons
])

event_monotonic_fail = int(
    np.any(
        np.diff(event_matrix, axis=1) < 0,
        axis=1
    ).sum()
)

# Evaluable status does not necessarily need to increase:
# if censored before a later horizon it can be non-evaluable.
# However, once an event has occurred, later horizons must remain
# evaluable because the event is known.
post_event_evaluable_fail = 0

for i in range(len(df)):
    event_seen = False

    for j, h in enumerate(horizons):

        if event_matrix[i, j] == 1:
            event_seen = True

        if event_seen and eval_matrix[i, j] != 1:
            post_event_evaluable_fail += 1

print("\nCROSS-HORIZON AUDIT")
print(
    "Event monotonicity failures       :",
    event_monotonic_fail
)
print(
    "Post-event evaluability failures  :",
    post_event_evaluable_fail
)

# ============================================================
# Event can occur in at most one discrete interval
# ============================================================

interval_event_matrix = np.column_stack([
    bool_series(
        df[f"event_interval_{k}"]
    ).fillna(False).astype(int).to_numpy()
    for k in range(5)
])

multiple_interval_events = int(
    (
        interval_event_matrix.sum(axis=1) > 1
    ).sum()
)

print("\nDISCRETE EVENT UNIQUENESS")
print(
    "Rows with >1 event interval:",
    multiple_interval_events
)

# ============================================================
# Build row-level audit
# ============================================================

audit = pd.DataFrame({
    "locked_subject_id":
        df["locked_subject_id"].astype(str),
    "locked_mri_id":
        df["locked_mri_id"].astype(str),
    "mri_day":
        numeric_series(df, "mri_day"),
    "event_observed":
        event_observed,
    "censored":
        censored,
    "event_day_from_mri":
        event_day_from_mri,
    "observed_time_days":
        observed_time_days,
    "observed_time_months":
        observed_time_months,
    "event_within_60m":
        event_within_60m,
    "time_within_60m_days":
        time_within_60m_days,
})

for h in horizons:
    audit[f"event_by_{h}m"] = numeric_series(
        df, f"event_by_{h}m"
    )
    audit[f"evaluable_{h}m"] = bool_series(
        df[f"evaluable_{h}m"]
    )

for k in range(5):
    audit[f"at_risk_interval_{k}"] = bool_series(
        df[f"at_risk_interval_{k}"]
    )
    audit[f"event_interval_{k}"] = bool_series(
        df[f"event_interval_{k}"]
    )

audit.to_csv(
    OUT_AUDIT,
    index=False
)

# ============================================================
# Final failure count
# ============================================================

failure_counts = {
    "missing_keys":
        missing_keys,
    "duplicate_keys":
        duplicate_keys,
    "missing_event_flag":
        missing_event_flag,
    "missing_censor_flag":
        missing_censor_flag,
    "event_censor_complement_fail":
        event_censor_complement_fail,
    "event_missing_time":
        event_missing_time,
    "event_nonpositive_time":
        event_nonpositive_time,
    "censored_with_event_day":
        censored_with_event_day,
    "missing_observed_time":
        missing_observed_time,
    "nonpositive_observed_time":
        nonpositive_observed_time,
    "event_time_vs_observed_fail":
        event_time_vs_observed_fail,
    "month_conversion_fail":
        month_conversion_fail,
    "month60_conversion_fail":
        month60_conversion_fail,
    "time60_fail":
        time60_fail,
    "event60_fail":
        event60_fail,
    "event_monotonic_fail":
        event_monotonic_fail,
    "post_event_evaluable_fail":
        post_event_evaluable_fail,
    "multiple_interval_events":
        multiple_interval_events,
}

for h in horizons:
    failure_counts[
        f"horizon_{h}_event_definition_fail"
    ] = horizon_summary[str(h)][
        "event_definition_failures"
    ]

    failure_counts[
        f"horizon_{h}_evaluable_definition_fail"
    ] = horizon_summary[str(h)][
        "evaluable_definition_failures"
    ]

    failure_counts[
        f"horizon_{h}_bad_event_values"
    ] = horizon_summary[str(h)][
        "bad_event_values"
    ]

    failure_counts[
        f"horizon_{h}_missing_evaluable"
    ] = horizon_summary[str(h)][
        "missing_evaluable"
    ]

    failure_counts[
        f"horizon_{h}_event_not_evaluable"
    ] = horizon_summary[str(h)][
        "event_not_evaluable"
    ]

for k in range(5):
    failure_counts[
        f"interval_{k}_at_risk_fail"
    ] = interval_summary[str(k)][
        "at_risk_definition_failures"
    ]

    failure_counts[
        f"interval_{k}_event_fail"
    ] = interval_summary[str(k)][
        "event_definition_failures"
    ]

    failure_counts[
        f"interval_{k}_event_without_risk"
    ] = interval_summary[str(k)][
        "event_without_at_risk"
    ]

total_failures = int(
    sum(failure_counts.values())
)

# ============================================================
# Protect Step-49 source
# ============================================================

STEP49_HASH_AFTER = sha256_file(STEP49)

source_modified = (
    STEP49_HASH_BEFORE != STEP49_HASH_AFTER
)

if source_modified:
    raise RuntimeError(
        "CRITICAL: Step-49 source file changed during Step 54."
    )

# ============================================================
# Summary
# ============================================================

n_subjects = int(
    df["locked_subject_id"]
    .astype(str)
    .nunique()
)

n_events_total = int(
    (event_observed == True).sum()
)

n_censored_total = int(
    (censored == True).sum()
)

summary = {
    "step": 54,
    "purpose":
        "OASIS-3 external endpoint contract audit and lock",
    "created":
        datetime.now().isoformat(),
    "source_step49":
        str(STEP49),
    "source_step49_sha256":
        STEP49_HASH_BEFORE,
    "source_step49_modified":
        False,
    "n_landmarks":
        int(len(df)),
    "n_subjects":
        n_subjects,
    "n_events_total":
        n_events_total,
    "n_censored_total":
        n_censored_total,
    "days_per_year":
        DAYS_PER_YEAR,
    "horizon_days":
        {
            str(k): v
            for k, v in horizon_days.items()
        },
    "interval_definition":
        {
            "0": "(0,12] months",
            "1": "(12,24] months",
            "2": "(24,36] months",
            "3": "(36,48] months",
            "4": "(48,60] months",
        },
    "horizon_summary":
        horizon_summary,
    "interval_summary":
        interval_summary,
    "failure_counts":
        failure_counts,
    "total_failures":
        total_failures,
    "model_checkpoint_loaded":
        False,
    "mri_voxels_read":
        False,
    "predictions_generated":
        False,
    "performance_examined":
        False,
}

with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )

# ============================================================
# Strict lock gate
# ============================================================

print("\n" + "=" * 78)
print("FINAL STEP-54 SAFETY AUDIT")
print("=" * 78)

for key, value in failure_counts.items():
    print(f"{key:42s}: {value}")

print("-" * 78)
print(f"{'TOTAL FAILURES':42s}: {total_failures}")

if total_failures != 0:
    print("\nSTEP 54 DID NOT LOCK.")
    print(
        "The source has NOT been modified. "
        "Investigate the reported target inconsistencies."
    )
    raise RuntimeError(
        f"Step-54 endpoint audit found "
        f"{total_failures} failure(s)."
    )

# ============================================================
# Write immutable lock only after complete PASS
# ============================================================

AUDIT_HASH = sha256_file(OUT_AUDIT)
SUMMARY_HASH = sha256_file(OUT_SUMMARY)

lock = {
    "step": 54,
    "status":
        "LOCKED_PASS",
    "contract":
        "OASIS3_EXTERNAL_ENDPOINT_CONTRACT_V1",
    "source_step49":
        str(STEP49),
    "source_step49_sha256":
        STEP49_HASH_BEFORE,
    "audit_file":
        str(OUT_AUDIT),
    "audit_sha256":
        AUDIT_HASH,
    "summary_file":
        str(OUT_SUMMARY),
    "summary_sha256":
        SUMMARY_HASH,
    "n_landmarks":
        int(len(df)),
    "n_subjects":
        n_subjects,
    "horizons_months":
        horizons,
    "interval_edges_days":
        edges,
    "days_per_year":
        DAYS_PER_YEAR,
    "source_modified":
        False,
    "model_checkpoint_loaded":
        False,
    "mri_voxels_read":
        False,
    "predictions_generated":
        False,
    "external_performance_examined":
        False,
}

with open(
    OUT_LOCK,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        lock,
        f,
        indent=2
    )

LOCK_HASH = sha256_file(OUT_LOCK)

# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 78)
print("STEP 54 PASSED - EXTERNAL ENDPOINT CONTRACT LOCKED")
print("=" * 78)

print("\nPOPULATION")
print("Subjects  :", n_subjects)
print("Landmarks :", len(df))
print("Events    :", n_events_total)
print("Censored  :", n_censored_total)

print("\nOUTPUT")
print(OUT_AUDIT)

print("\nSUMMARY")
print(OUT_SUMMARY)

print("\nLOCK")
print(OUT_LOCK)

print("\nSHA256")
print("Step-49 :", STEP49_HASH_BEFORE)
print("Audit   :", AUDIT_HASH)
print("Summary :", SUMMARY_HASH)
print("Lock    :", LOCK_HASH)

print("\nIMPORTANT")
print("Step-49 source modified       : NO")
print("MRI voxels read               : NO")
print("Frozen ADNI checkpoint loaded : NO")
print("Predictions generated         : NO")
print("External performance examined : NO")
print("")
print(
    "The external endpoint definition is now frozen "
    "before model inference."
)