# ================================================================
# STEP 49
# OASIS-3 EXTERNAL SURVIVAL TARGET CONSTRUCTION
#
# Purpose:
#   Construct protocol-locked external survival targets for the
#   frozen ADNI longitudinal survival model.
#
# Primary external event:
#   First observed post-landmark CDR >= 0.5
#
# Landmark eligibility:
#   Defined and frozen in Step 48.
#
# IMPORTANT:
#   - No MRI files are modified.
#   - No prediction files are read.
#   - No frozen model inference is performed.
#   - No threshold is chosen from model performance.
# ================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime


ROOT = (Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
COHORT = ROOT / "cohort_construction"

STEP47 = COHORT / "OASIS3_STEP47_LANDMARK_FUTURE_FOLLOWUP_AUDIT.csv"
STEP48 = COHORT / "OASIS3_STEP48_EXTERNAL_LANDMARKS_LOCKED.csv"

OUT = COHORT / "OASIS3_STEP49_EXTERNAL_SURVIVAL_TARGETS_LOCKED.csv"
SUMMARY = COHORT / "OASIS3_STEP49_EXTERNAL_SURVIVAL_TARGETS_SUMMARY.json"
LOCK = COHORT / "OASIS3_STEP49_EXTERNAL_SURVIVAL_TARGETS_LOCK.json"


# ------------------------------------------------
# Constants
# ------------------------------------------------

DAYS_PER_YEAR = 365.25

HORIZON_MONTHS = [12, 24, 36, 48, 60]

HORIZON_DAYS = {
    12: 365.25,
    24: 730.50,
    36: 1095.75,
    48: 1461.00,
    60: 1826.25,
}

EVENT_DEFINITION = "first observed post-landmark CDR >= 0.5"

print("=" * 72)
print("STEP 49 — OASIS-3 EXTERNAL SURVIVAL TARGET CONSTRUCTION")
print("=" * 72)

print("\nPrimary event:")
print(EVENT_DEFINITION)

print("\nInput Step 47:")
print(STEP47)

print("\nInput Step 48:")
print(STEP48)


# ------------------------------------------------
# Helpers
# ------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


# ------------------------------------------------
# Verify files
# ------------------------------------------------

for p in [STEP47, STEP48]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required input: {p}")


# ------------------------------------------------
# Load
# ------------------------------------------------

future = pd.read_csv(STEP47, low_memory=False)
landmarks = pd.read_csv(STEP48, low_memory=False)

print("\nINPUT COUNTS")
print("-" * 72)

print("Step 47 rows :", len(future))
print("Step 48 rows :", len(landmarks))


# ------------------------------------------------
# Required columns
# ------------------------------------------------

future_required = [
    "locked_subject_id",
    "locked_mri_id",
    "mri_day",
    "prior_cdrtot",
    "clinical_to_mri_gap_days",
    "n_future_cdr_visits",
    "has_future_cdr",
    "first_future_clinical_day",
    "first_future_cdrtot",
    "last_future_clinical_day",
    "last_future_cdrtot",
    "followup_days_from_mri",
    "first_cdr_ge_0p5_day",
    "first_cdr_ge_0p5_value",
    "days_to_first_cdr_ge_0p5",
    "first_cdr_ge_1_day",
    "first_cdr_ge_1_value",
    "days_to_first_cdr_ge_1",
    "followup_reaches_12m",
    "followup_reaches_24m",
    "followup_reaches_36m",
    "followup_reaches_48m",
    "followup_reaches_60m",
]

landmark_required = [
    "locked_subject_id",
    "locked_mri_id",
]

missing_future = [c for c in future_required if c not in future.columns]
missing_landmark = [c for c in landmark_required if c not in landmarks.columns]

if missing_future:
    raise RuntimeError(
        "Step 47 missing required columns: "
        + str(missing_future)
    )

if missing_landmark:
    raise RuntimeError(
        "Step 48 missing required columns: "
        + str(missing_landmark)
    )


# ------------------------------------------------
# Identifier normalization
# ------------------------------------------------

for d in [future, landmarks]:
    d["locked_subject_id"] = (
        d["locked_subject_id"]
        .astype(str)
        .str.strip()
    )

    d["locked_mri_id"] = (
        d["locked_mri_id"]
        .astype(str)
        .str.strip()
    )


# ------------------------------------------------
# Uniqueness checks
# ------------------------------------------------

if future["locked_mri_id"].duplicated().any():
    dup = future.loc[
        future["locked_mri_id"].duplicated(False),
        ["locked_subject_id", "locked_mri_id"]
    ]

    raise RuntimeError(
        "Duplicate MRI IDs in Step 47:\n"
        + dup.head(20).to_string(index=False)
    )


if landmarks["locked_mri_id"].duplicated().any():
    dup = landmarks.loc[
        landmarks["locked_mri_id"].duplicated(False),
        ["locked_subject_id", "locked_mri_id"]
    ]

    raise RuntimeError(
        "Duplicate MRI IDs in Step 48:\n"
        + dup.head(20).to_string(index=False)
    )


# ------------------------------------------------
# Restrict strictly to Step 48 frozen landmark cohort
# ------------------------------------------------

keep_ids = set(landmarks["locked_mri_id"])

x = future[
    future["locked_mri_id"].isin(keep_ids)
].copy()

if len(x) != len(landmarks):
    missing_ids = sorted(
        keep_ids - set(x["locked_mri_id"])
    )

    raise RuntimeError(
        f"Step 48 / Step 47 linkage failure. "
        f"Expected {len(landmarks)} landmarks, obtained {len(x)}. "
        f"Missing examples: {missing_ids[:20]}"
    )


# ------------------------------------------------
# Subject integrity
# ------------------------------------------------

subject_check = landmarks[
    ["locked_mri_id", "locked_subject_id"]
].merge(
    x[["locked_mri_id", "locked_subject_id"]],
    on="locked_mri_id",
    how="left",
    suffixes=("_step48", "_step47"),
)

bad_subject = (
    subject_check["locked_subject_id_step48"]
    != subject_check["locked_subject_id_step47"]
)

if bad_subject.any():
    raise RuntimeError(
        "Subject mismatch between Step 47 and Step 48."
    )


# ------------------------------------------------
# Numeric conversion
# ------------------------------------------------

numeric_cols = [
    "mri_day",
    "prior_cdrtot",
    "clinical_to_mri_gap_days",
    "n_future_cdr_visits",
    "first_future_clinical_day",
    "first_future_cdrtot",
    "last_future_clinical_day",
    "last_future_cdrtot",
    "followup_days_from_mri",
    "first_cdr_ge_0p5_day",
    "first_cdr_ge_0p5_value",
    "days_to_first_cdr_ge_0p5",
    "first_cdr_ge_1_day",
    "first_cdr_ge_1_value",
    "days_to_first_cdr_ge_1",
]

for c in numeric_cols:
    x[c] = pd.to_numeric(
        x[c],
        errors="coerce"
    )


# ------------------------------------------------
# Boolean conversion
# ------------------------------------------------

bool_cols = [
    "has_future_cdr",
    "followup_reaches_12m",
    "followup_reaches_24m",
    "followup_reaches_36m",
    "followup_reaches_48m",
    "followup_reaches_60m",
]

for c in bool_cols:
    x[c] = as_bool(x[c])


# ------------------------------------------------
# Frozen Step 48 eligibility verification
# ------------------------------------------------

if not (x["prior_cdrtot"] == 0).all():
    bad = x.loc[
        x["prior_cdrtot"] != 0,
        [
            "locked_subject_id",
            "locked_mri_id",
            "prior_cdrtot",
        ],
    ]

    raise RuntimeError(
        "Step 48 contains non-CDR=0 landmarks:\n"
        + bad.head(20).to_string(index=False)
    )


if (x["clinical_to_mri_gap_days"] > 365).any():
    bad = x.loc[
        x["clinical_to_mri_gap_days"] > 365,
        [
            "locked_subject_id",
            "locked_mri_id",
            "clinical_to_mri_gap_days",
        ],
    ]

    raise RuntimeError(
        "Step 48 contains clinical-to-MRI gap >365 days:\n"
        + bad.head(20).to_string(index=False)
    )


# ------------------------------------------------
# Event definition
#
# Event = first observed future CDR >= 0.5
# ------------------------------------------------

x["event_observed"] = (
    x["days_to_first_cdr_ge_0p5"].notna()
    & (x["days_to_first_cdr_ge_0p5"] >= 0)
)

x["event_day_from_mri"] = np.where(
    x["event_observed"],
    x["days_to_first_cdr_ge_0p5"],
    np.nan,
)


# ------------------------------------------------
# Censoring definition
#
# For non-events:
#   censor at last observed future CDR visit.
#
# For events:
#   observed time is the first CDR >=0.5.
# ------------------------------------------------

if x["followup_days_from_mri"].isna().any():
    bad = x.loc[
        x["followup_days_from_mri"].isna(),
        ["locked_subject_id", "locked_mri_id"],
    ]

    raise RuntimeError(
        "Missing future follow-up for locked landmark:\n"
        + bad.head(20).to_string(index=False)
    )


if (x["followup_days_from_mri"] < 0).any():
    raise RuntimeError(
        "Negative follow-up time detected."
    )


x["observed_time_days"] = np.where(
    x["event_observed"],
    x["event_day_from_mri"],
    x["followup_days_from_mri"],
)

x["censored"] = ~x["event_observed"]


# ------------------------------------------------
# Sanity: event cannot occur after recorded follow-up
# ------------------------------------------------

bad_event_time = (
    x["event_observed"]
    & (
        x["event_day_from_mri"]
        > x["followup_days_from_mri"]
    )
)

if bad_event_time.any():
    bad = x.loc[
        bad_event_time,
        [
            "locked_subject_id",
            "locked_mri_id",
            "event_day_from_mri",
            "followup_days_from_mri",
        ],
    ]

    raise RuntimeError(
        "Event occurs after last recorded follow-up:\n"
        + bad.head(20).to_string(index=False)
    )


# ------------------------------------------------
# Convert observed time to months
# ------------------------------------------------

x["observed_time_months"] = (
    x["observed_time_days"]
    / DAYS_PER_YEAR
    * 12.0
)


# ------------------------------------------------
# Discrete survival interval
#
# Frozen model intervals:
#   0–12
#   12–24
#   24–36
#   36–48
#   48–60 months
#
# Event interval:
#   0 = event <=12m
#   1 = event >12 and <=24m
#   2 = event >24 and <=36m
#   3 = event >36 and <=48m
#   4 = event >48 and <=60m
#
# Events after 60m are treated as event-free through 60m for
# the frozen 60-month prediction horizon.
# ------------------------------------------------

def event_interval(days):
    if pd.isna(days):
        return np.nan

    if days <= HORIZON_DAYS[12]:
        return 0

    if days <= HORIZON_DAYS[24]:
        return 1

    if days <= HORIZON_DAYS[36]:
        return 2

    if days <= HORIZON_DAYS[48]:
        return 3

    if days <= HORIZON_DAYS[60]:
        return 4

    return np.nan


x["event_interval_0to4"] = x[
    "event_day_from_mri"
].apply(event_interval)


# ------------------------------------------------
# Event by horizon
#
# event_by_H:
#   1 = event occurred by H
#   0 = known event-free through H
#   NaN = not evaluable at H because censored before H
#
# IMPORTANT:
# A later event (>H) proves event-free status through H.
# ------------------------------------------------

for months in HORIZON_MONTHS:

    hdays = HORIZON_DAYS[months]

    event_before_h = (
        x["event_observed"]
        & (x["event_day_from_mri"] <= hdays)
    )

    known_event_free_h = (
        (
            x["event_observed"]
            & (x["event_day_from_mri"] > hdays)
        )
        |
        (
            (~x["event_observed"])
            & (x["followup_days_from_mri"] >= hdays)
        )
    )

    target = np.full(
        len(x),
        np.nan,
        dtype=float,
    )

    target[event_before_h.to_numpy()] = 1.0
    target[known_event_free_h.to_numpy()] = 0.0

    x[f"event_by_{months}m"] = target

    x[f"evaluable_{months}m"] = (
        ~pd.isna(target)
    )


# ------------------------------------------------
# Risk-set / discrete interval observability
# ------------------------------------------------

interval_starts = [
    0.0,
    HORIZON_DAYS[12],
    HORIZON_DAYS[24],
    HORIZON_DAYS[36],
    HORIZON_DAYS[48],
]

interval_ends = [
    HORIZON_DAYS[12],
    HORIZON_DAYS[24],
    HORIZON_DAYS[36],
    HORIZON_DAYS[48],
    HORIZON_DAYS[60],
]

for i, (start, end) in enumerate(
    zip(interval_starts, interval_ends)
):

    # Participant is at risk at interval start if event has
    # not occurred before the start and follow-up reaches start.
    at_risk = (
        x["observed_time_days"] >= start
    )

    x[f"at_risk_interval_{i}"] = at_risk

    # Event occurs in this interval.
    event_in_interval = (
        x["event_observed"]
        & (x["event_day_from_mri"] > start)
        & (x["event_day_from_mri"] <= end)
    )

    # Special handling for interval 0:
    # day-0 event belongs to first interval.
    if i == 0:
        event_in_interval = (
            x["event_observed"]
            & (x["event_day_from_mri"] >= 0)
            & (x["event_day_from_mri"] <= end)
        )

    x[f"event_interval_{i}"] = (
        event_in_interval
    )


# ------------------------------------------------
# Primary 60-month survival representation
#
# Events >60m are not events within the model horizon.
# Non-events followed >60m are administratively truncated at 60m.
# ------------------------------------------------

x["event_within_60m"] = (
    x["event_observed"]
    & (
        x["event_day_from_mri"]
        <= HORIZON_DAYS[60]
    )
)

x["time_within_60m_days"] = np.minimum(
    x["observed_time_days"],
    HORIZON_DAYS[60],
)

x["time_within_60m_months"] = (
    x["time_within_60m_days"]
    / DAYS_PER_YEAR
    * 12.0
)


# ------------------------------------------------
# Final strict checks
# ------------------------------------------------

if len(x) != len(landmarks):
    raise RuntimeError(
        "Final row count changed unexpectedly."
    )


if x["locked_mri_id"].nunique() != len(x):
    raise RuntimeError(
        "Final survival target table does not contain "
        "one row per MRI landmark."
    )


if x["locked_subject_id"].nunique() != \
        landmarks["locked_subject_id"].nunique():

    raise RuntimeError(
        "Participant count changed unexpectedly."
    )


for months in HORIZON_MONTHS:

    col = f"event_by_{months}m"

    vals = set(
        x[col]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not vals.issubset({0, 1}):
        raise RuntimeError(
            f"Invalid target values in {col}: {vals}"
        )


# ------------------------------------------------
# Sort deterministically
# ------------------------------------------------

x = x.sort_values(
    [
        "locked_subject_id",
        "mri_day",
        "locked_mri_id",
    ]
).reset_index(drop=True)


# ------------------------------------------------
# Select final columns
# ------------------------------------------------

final_cols = [
    "locked_subject_id",
    "locked_mri_id",
    "mri_day",
    "prior_clinical_day",
    "prior_cdrtot",
    "clinical_to_mri_gap_days",
    "n_future_cdr_visits",
    "first_future_clinical_day",
    "first_future_cdrtot",
    "last_future_clinical_day",
    "last_future_cdrtot",
    "followup_days_from_mri",
    "first_cdr_ge_0p5_day",
    "first_cdr_ge_0p5_value",
    "days_to_first_cdr_ge_0p5",
    "event_observed",
    "event_day_from_mri",
    "censored",
    "observed_time_days",
    "observed_time_months",
    "event_interval_0to4",
    "event_within_60m",
    "time_within_60m_days",
    "time_within_60m_months",
]

for months in HORIZON_MONTHS:
    final_cols.extend([
        f"event_by_{months}m",
        f"evaluable_{months}m",
    ])

for i in range(5):
    final_cols.extend([
        f"at_risk_interval_{i}",
        f"event_interval_{i}",
    ])


final = x[final_cols].copy()


# ------------------------------------------------
# Save target table
# ------------------------------------------------

final.to_csv(
    OUT,
    index=False,
)


# ------------------------------------------------
# Reporting
# ------------------------------------------------

print("\n" + "=" * 72)
print("STEP 49 RESULTS")
print("=" * 72)

print("\nLandmarks              :", len(final))
print(
    "Participants          :",
    final["locked_subject_id"].nunique()
)

print(
    "Any future CDR event  :",
    int(final["event_observed"].sum())
)

print(
    "Event within 60m      :",
    int(final["event_within_60m"].sum())
)

print(
    "No event within 60m   :",
    int((~final["event_within_60m"]).sum())
)


print("\nEVENT TIME SUMMARY — DAYS")

events = final.loc[
    final["event_observed"],
    "event_day_from_mri"
]

if len(events):
    print(events.describe().to_string())
else:
    print("No events.")


print("\nHORIZON EVALUABILITY")

horizon_summary = {}

for months in HORIZON_MONTHS:

    t = final[f"event_by_{months}m"]

    evaluable = int(t.notna().sum())
    events_h = int((t == 1).sum())
    nonevents_h = int((t == 0).sum())
    censored_before = int(t.isna().sum())

    horizon_summary[str(months)] = {
        "evaluable": evaluable,
        "events": events_h,
        "non_events": nonevents_h,
        "censored_before_horizon": censored_before,
    }

    print(
        f"{months:>2} months : "
        f"evaluable={evaluable:4d} | "
        f"events={events_h:4d} | "
        f"non-events={nonevents_h:4d} | "
        f"censored-before={censored_before:4d}"
    )


print("\nEVENT INTERVAL COUNTS")

print(
    final["event_interval_0to4"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)


# ------------------------------------------------
# Summary JSON
# ------------------------------------------------

summary = {
    "step": 49,
    "created": datetime.now().isoformat(),
    "event_definition": EVENT_DEFINITION,
    "days_per_year": DAYS_PER_YEAR,
    "horizon_months": HORIZON_MONTHS,
    "horizon_days": HORIZON_DAYS,
    "landmarks": int(len(final)),
    "participants": int(
        final["locked_subject_id"].nunique()
    ),
    "events_any_followup": int(
        final["event_observed"].sum()
    ),
    "events_within_60m": int(
        final["event_within_60m"].sum()
    ),
    "horizon_summary": horizon_summary,
    "step47_sha256": sha256_file(STEP47),
    "step48_sha256": sha256_file(STEP48),
}

with open(SUMMARY, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# ------------------------------------------------
# Lock file
# ------------------------------------------------

lock = {
    "step": 49,
    "status": "LOCKED",
    "event_definition": EVENT_DEFINITION,
    "landmark_source": str(STEP48),
    "future_followup_source": str(STEP47),
    "output": str(OUT),
    "step47_sha256": sha256_file(STEP47),
    "step48_sha256": sha256_file(STEP48),
    "step49_output_sha256": sha256_file(OUT),
    "model_inference_performed": False,
    "prediction_files_read": False,
    "performance_based_selection": False,
    "MRI_modified": False,
}

with open(LOCK, "w", encoding="utf-8") as f:
    json.dump(
        lock,
        f,
        indent=2,
    )


print("\nOUTPUT:")
print(OUT)

print("\nSUMMARY:")
print(SUMMARY)

print("\nLOCK:")
print(LOCK)

print("\nSHA256")
print(
    "Step-47 input :",
    sha256_file(STEP47)
)
print(
    "Step-48 input :",
    sha256_file(STEP48)
)
print(
    "Step-49 output:",
    sha256_file(OUT)
)


print("\nIMPORTANT:")
print("No MRI files were modified.")
print("No prediction files were read.")
print("No frozen ADNI model inference was performed.")
print("No external model performance was examined.")
print("The event definition was fixed before external inference.")
print("The frozen prediction horizon remains 60 months.")


print("\n" + "=" * 72)
print("STEP 49 PASSED — EXTERNAL SURVIVAL TARGETS LOCKED")
print("=" * 72)