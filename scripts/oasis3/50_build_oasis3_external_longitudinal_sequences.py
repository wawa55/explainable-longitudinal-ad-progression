from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime

# ============================================================
# STEP 50
# OASIS-3 EXTERNAL LONGITUDINAL MRI SEQUENCE LOCK
#
# PURPOSE:
#   Construct the exact longitudinal MRI history available at
#   every locked external landmark.
#
# IMPORTANT:
#   - No outcome data are read.
#   - No prediction files are read.
#   - No model inference is performed.
#   - No MRI files are modified.
#   - No future MRI is allowed in a landmark sequence.
# ============================================================

ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

COHORT_DIR = ROOT / "cohort_construction"

MRI_LOCK = (
    COHORT_DIR /
    "OASIS3_EXACT_T1_ACQUISITION_LOCKED.csv"
)

LANDMARK_LOCK = (
    COHORT_DIR /
    "OASIS3_STEP48_EXTERNAL_LANDMARKS_LOCKED.csv"
)

OUT = (
    COHORT_DIR /
    "OASIS3_STEP50_EXTERNAL_LONGITUDINAL_SEQUENCES_LOCKED.csv"
)

EXPANDED_OUT = (
    COHORT_DIR /
    "OASIS3_STEP50_EXTERNAL_LONGITUDINAL_SEQUENCE_VISITS.csv"
)

SUMMARY_OUT = (
    COHORT_DIR /
    "OASIS3_STEP50_EXTERNAL_LONGITUDINAL_SEQUENCES_SUMMARY.json"
)

LOCK_OUT = (
    COHORT_DIR /
    "OASIS3_STEP50_EXTERNAL_LONGITUDINAL_SEQUENCES_LOCK.json"
)


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{name} missing required columns: {missing}"
        )


# ============================================================
# HEADER
# ============================================================

print("=" * 72)
print("STEP 50 — OASIS-3 EXTERNAL LONGITUDINAL MRI SEQUENCE LOCK")
print("=" * 72)

print("\nMRI acquisition lock:")
print(MRI_LOCK)

print("\nExternal landmark lock:")
print(LANDMARK_LOCK)


# ============================================================
# CHECK INPUTS
# ============================================================

if not MRI_LOCK.exists():
    raise FileNotFoundError(MRI_LOCK)

if not LANDMARK_LOCK.exists():
    raise FileNotFoundError(LANDMARK_LOCK)


# ============================================================
# LOAD INPUTS
# ============================================================

mri = pd.read_csv(MRI_LOCK, low_memory=False)
landmarks = pd.read_csv(LANDMARK_LOCK, low_memory=False)

print("\nINPUT COUNTS")
print("-" * 72)
print("MRI acquisitions :", len(mri))
print("Landmarks        :", len(landmarks))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

require_columns(
    mri,
    [
        "locked_subject_id",
        "locked_mri_id",
    ],
    "MRI acquisition lock"
)

require_columns(
    landmarks,
    [
        "locked_subject_id",
        "locked_mri_id",
        "mri_day",
        "step48_landmark_locked",
    ],
    "Step-48 landmark lock"
)


# ============================================================
# IDENTIFY MRI DAY
#
# Preferred source:
# external_session_key such as:
# OAS30001__day_129
#
# locked_mri_id provides independent consistency check:
# OAS30001_MR_d0129
# ============================================================

if "external_session_key" in mri.columns:

    mri["mri_day_from_session"] = pd.to_numeric(
        mri["external_session_key"]
        .astype(str)
        .str.extract(r"_day_(-?\d+)$")[0],
        errors="coerce"
    )

else:
    mri["mri_day_from_session"] = np.nan


mri["mri_day_from_id"] = pd.to_numeric(
    mri["locked_mri_id"]
    .astype(str)
    .str.extract(r"_d(-?\d+)$")[0],
    errors="coerce"
)


# Prefer external_session_key when available.
mri["mri_day"] = mri["mri_day_from_session"].where(
    mri["mri_day_from_session"].notna(),
    mri["mri_day_from_id"]
)


if mri["mri_day"].isna().any():
    bad = mri.loc[
        mri["mri_day"].isna(),
        ["locked_subject_id", "locked_mri_id"]
    ]

    print("\nMRI rows with unparseable day:")
    print(bad.head(20).to_string(index=False))

    raise RuntimeError(
        "Could not determine MRI day for all locked acquisitions."
    )


# Independent consistency check where both are available.
both = (
    mri["mri_day_from_session"].notna()
    & mri["mri_day_from_id"].notna()
)

day_mismatch = (
    mri.loc[both, "mri_day_from_session"]
    !=
    mri.loc[both, "mri_day_from_id"]
)

if day_mismatch.any():
    bad = mri.loc[
        both
        & (
            mri["mri_day_from_session"]
            != mri["mri_day_from_id"]
        ),
        [
            "locked_subject_id",
            "locked_mri_id",
            "external_session_key",
            "mri_day_from_session",
            "mri_day_from_id",
        ]
    ]

    print("\nMRI day mismatches:")
    print(bad.head(20).to_string(index=False))

    raise RuntimeError(
        "MRI day mismatch between session key and MRI ID."
    )


# ============================================================
# CLEAN TYPES
# ============================================================

mri["locked_subject_id"] = (
    mri["locked_subject_id"].astype(str)
)

mri["locked_mri_id"] = (
    mri["locked_mri_id"].astype(str)
)

mri["mri_day"] = pd.to_numeric(
    mri["mri_day"],
    errors="raise"
)


landmarks["locked_subject_id"] = (
    landmarks["locked_subject_id"].astype(str)
)

landmarks["locked_mri_id"] = (
    landmarks["locked_mri_id"].astype(str)
)

landmarks["mri_day"] = pd.to_numeric(
    landmarks["mri_day"],
    errors="raise"
)


# ============================================================
# VERIFY STEP-48 LOCK FLAG
# ============================================================

lock_values = (
    landmarks["step48_landmark_locked"]
    .astype(str)
    .str.strip()
    .str.lower()
)

valid_lock = lock_values.isin(
    ["true", "1", "yes"]
)

if not valid_lock.all():
    raise RuntimeError(
        "Not all Step-48 rows are marked as locked."
    )


# ============================================================
# UNIQUENESS CHECKS
# ============================================================

if mri["locked_mri_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate locked MRI IDs found in acquisition lock."
    )

if landmarks["locked_mri_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate landmark MRI IDs found in Step 48."
    )


# ============================================================
# VERIFY EACH LANDMARK MRI EXISTS IN MRI LOCK
# ============================================================

mri_ids = set(mri["locked_mri_id"])

missing_landmark_mri = landmarks.loc[
    ~landmarks["locked_mri_id"].isin(mri_ids),
    "locked_mri_id"
]

if len(missing_landmark_mri) > 0:
    print("\nMissing landmark MRI IDs:")
    print(missing_landmark_mri.head(20).to_string(index=False))

    raise RuntimeError(
        "Some Step-48 landmark MRIs are absent from MRI acquisition lock."
    )


# ============================================================
# VERIFY LANDMARK DAY AGAINST ACQUISITION LOCK
# ============================================================

mri_day_lookup = (
    mri.set_index("locked_mri_id")["mri_day"]
    .to_dict()
)

landmark_day_mismatch = []

for _, row in landmarks.iterrows():

    mid = row["locked_mri_id"]
    day_step48 = float(row["mri_day"])
    day_mri_lock = float(mri_day_lookup[mid])

    if not np.isclose(day_step48, day_mri_lock):
        landmark_day_mismatch.append(
            {
                "locked_mri_id": mid,
                "step48_day": day_step48,
                "mri_lock_day": day_mri_lock,
            }
        )


if landmark_day_mismatch:

    bad = pd.DataFrame(landmark_day_mismatch)

    print("\nLandmark day mismatch:")
    print(bad.head(20).to_string(index=False))

    raise RuntimeError(
        "Step-48 MRI day differs from acquisition lock."
    )


# ============================================================
# SORT MRI HISTORY
# ============================================================

mri = mri.sort_values(
    [
        "locked_subject_id",
        "mri_day",
        "locked_mri_id",
    ]
).reset_index(drop=True)


subject_histories = {
    sid: grp.copy()
    for sid, grp in mri.groupby(
        "locked_subject_id",
        sort=False
    )
}


# ============================================================
# BUILD LANDMARK-SPECIFIC LONGITUDINAL SEQUENCES
# ============================================================

sequence_rows = []
expanded_rows = []

landmarks = landmarks.sort_values(
    [
        "locked_subject_id",
        "mri_day",
        "locked_mri_id",
    ]
).reset_index(drop=True)


for landmark_number, (_, lm) in enumerate(
    landmarks.iterrows(),
    start=1
):

    sid = lm["locked_subject_id"]
    landmark_mri_id = lm["locked_mri_id"]
    landmark_day = float(lm["mri_day"])

    if sid not in subject_histories:
        raise RuntimeError(
            f"No MRI history found for participant {sid}"
        )

    history = subject_histories[sid]

    # CRITICAL ANTI-LEAKAGE RULE:
    # Only MRI acquisitions on or before the landmark.
    seq = history.loc[
        history["mri_day"] <= landmark_day
    ].copy()

    seq = seq.sort_values(
        ["mri_day", "locked_mri_id"]
    ).reset_index(drop=True)


    if len(seq) == 0:
        raise RuntimeError(
            f"Empty MRI sequence for landmark {landmark_mri_id}"
        )


    # The landmark MRI itself must be present.
    if landmark_mri_id not in set(seq["locked_mri_id"]):
        raise RuntimeError(
            f"Landmark MRI absent from its own sequence: "
            f"{landmark_mri_id}"
        )


    # Last sequence day must equal landmark day.
    last_day = float(seq.iloc[-1]["mri_day"])

    if not np.isclose(last_day, landmark_day):
        raise RuntimeError(
            f"Last sequence day != landmark day for "
            f"{landmark_mri_id}"
        )


    # The final MRI should be the actual locked landmark MRI.
    # Because there should be one exact locked T1 per session/day.
    last_mri_id = str(seq.iloc[-1]["locked_mri_id"])

    if last_mri_id != landmark_mri_id:
        raise RuntimeError(
            f"Final sequence MRI != landmark MRI for "
            f"{landmark_mri_id}. Final={last_mri_id}"
        )


    # No future MRI.
    if (seq["mri_day"] > landmark_day).any():
        raise RuntimeError(
            f"Future MRI leakage detected for {landmark_mri_id}"
        )


    first_mri_day = float(seq.iloc[0]["mri_day"])

    # Relative time convention:
    # first available MRI = time zero.
    relative_days = (
        seq["mri_day"].astype(float)
        - first_mri_day
    )

    relative_years = relative_days / 365.25


    mri_id_string = "|".join(
        seq["locked_mri_id"].astype(str).tolist()
    )

    mri_day_string = "|".join(
        [
            str(int(x)) if float(x).is_integer()
            else str(float(x))
            for x in seq["mri_day"].tolist()
        ]
    )

    relative_day_string = "|".join(
        [
            str(int(x)) if float(x).is_integer()
            else str(float(x))
            for x in relative_days.tolist()
        ]
    )

    relative_year_string = "|".join(
        [
            f"{float(x):.8f}"
            for x in relative_years.tolist()
        ]
    )


    sequence_rows.append(
        {
            "external_landmark_number": landmark_number,
            "locked_subject_id": sid,
            "landmark_mri_id": landmark_mri_id,
            "landmark_day": landmark_day,
            "n_visits": len(seq),
            "first_mri_day": first_mri_day,
            "last_mri_day": last_day,
            "longitudinal_span_days":
                landmark_day - first_mri_day,
            "mri_ids": mri_id_string,
            "mri_days": mri_day_string,
            "relative_days_from_first_mri":
                relative_day_string,
            "relative_years_from_first_mri":
                relative_year_string,
            "future_mri_count": int(
                (seq["mri_day"] > landmark_day).sum()
            ),
            "sequence_ends_at_landmark": True,
            "step50_sequence_locked": True,
        }
    )


    for visit_number, (_, visit) in enumerate(
        seq.iterrows(),
        start=1
    ):

        rel_day = (
            float(visit["mri_day"])
            - first_mri_day
        )

        expanded_rows.append(
            {
                "external_landmark_number":
                    landmark_number,
                "locked_subject_id":
                    sid,
                "landmark_mri_id":
                    landmark_mri_id,
                "landmark_day":
                    landmark_day,
                "visit_number":
                    visit_number,
                "visit_mri_id":
                    visit["locked_mri_id"],
                "visit_mri_day":
                    float(visit["mri_day"]),
                "relative_day_from_first_mri":
                    rel_day,
                "relative_year_from_first_mri":
                    rel_day / 365.25,
                "is_landmark_visit":
                    bool(
                        visit["locked_mri_id"]
                        == landmark_mri_id
                    ),
            }
        )


# ============================================================
# CREATE TABLES
# ============================================================

sequences = pd.DataFrame(sequence_rows)
expanded = pd.DataFrame(expanded_rows)


# ============================================================
# FINAL INTEGRITY CHECKS
# ============================================================

if len(sequences) != len(landmarks):
    raise RuntimeError(
        "Sequence count does not equal Step-48 landmark count."
    )

if sequences["landmark_mri_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate landmark MRI IDs in Step-50 output."
    )

if (sequences["n_visits"] < 1).any():
    raise RuntimeError(
        "Sequence with zero visits detected."
    )

if (sequences["future_mri_count"] != 0).any():
    raise RuntimeError(
        "Future MRI leakage detected."
    )

if not sequences["sequence_ends_at_landmark"].all():
    raise RuntimeError(
        "Some sequences do not end at landmark MRI."
    )

if not sequences["step50_sequence_locked"].all():
    raise RuntimeError(
        "Step-50 lock flag failure."
    )


# Every expanded sequence should have exactly one landmark visit.
landmark_visit_counts = (
    expanded.groupby("external_landmark_number")
    ["is_landmark_visit"]
    .sum()
)

if not (landmark_visit_counts == 1).all():
    raise RuntimeError(
        "Every sequence must contain exactly one landmark visit."
    )


# No expanded visit may occur after its landmark.
if (
    expanded["visit_mri_day"]
    >
    expanded["landmark_day"]
).any():
    raise RuntimeError(
        "Future MRI leakage found in expanded sequence table."
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

sequences.to_csv(
    OUT,
    index=False
)

expanded.to_csv(
    EXPANDED_OUT,
    index=False
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

visit_counts = sequences["n_visits"]

summary = {
    "step": 50,
    "description":
        "OASIS-3 external longitudinal MRI sequence lock",

    "participants":
        int(sequences["locked_subject_id"].nunique()),

    "landmarks":
        int(len(sequences)),

    "expanded_sequence_rows":
        int(len(expanded)),

    "single_visit_landmarks":
        int((visit_counts == 1).sum()),

    "multi_visit_landmarks":
        int((visit_counts >= 2).sum()),

    "max_visits":
        int(visit_counts.max()),

    "mean_visits":
        float(visit_counts.mean()),

    "median_visits":
        float(visit_counts.median()),

    "future_mri_leakage_count":
        int(sequences["future_mri_count"].sum()),

    "sequence_rule":
        "all locked MRI acquisitions for participant with MRI day <= landmark day",

    "temporal_origin":
        "first MRI in each landmark-specific sequence",

    "time_scale":
        "days / 365.25",

    "outcome_data_read":
        False,

    "prediction_files_read":
        False,

    "model_inference_performed":
        False,

    "mri_files_modified":
        False,
}


with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# HASHES / LOCK
# ============================================================

hash_mri_lock = sha256_file(MRI_LOCK)
hash_landmark_lock = sha256_file(LANDMARK_LOCK)
hash_sequence_output = sha256_file(OUT)
hash_expanded_output = sha256_file(EXPANDED_OUT)


lock = {
    "step": 50,

    "created_utc":
        datetime.utcnow().isoformat() + "Z",

    "mri_acquisition_lock":
        str(MRI_LOCK),

    "mri_acquisition_lock_sha256":
        hash_mri_lock,

    "step48_landmark_lock":
        str(LANDMARK_LOCK),

    "step48_landmark_lock_sha256":
        hash_landmark_lock,

    "sequence_output":
        str(OUT),

    "sequence_output_sha256":
        hash_sequence_output,

    "expanded_sequence_output":
        str(EXPANDED_OUT),

    "expanded_sequence_output_sha256":
        hash_expanded_output,

    "participants":
        int(sequences["locked_subject_id"].nunique()),

    "landmarks":
        int(len(sequences)),

    "future_mri_leakage_count":
        int(sequences["future_mri_count"].sum()),

    "outcome_data_used":
        False,

    "prediction_files_used":
        False,

    "model_inference_performed":
        False,

    "locked":
        True,
}


with open(
    LOCK_OUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        lock,
        f,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 72)
print("STEP 50 RESULTS")
print("=" * 72)

print(
    "Participants              :",
    sequences["locked_subject_id"].nunique()
)

print(
    "Locked external landmarks :",
    len(sequences)
)

print(
    "Expanded sequence rows     :",
    len(expanded)
)

print(
    "Single-visit landmarks     :",
    int((visit_counts == 1).sum())
)

print(
    "Multi-visit landmarks      :",
    int((visit_counts >= 2).sum())
)

print(
    "Maximum visits/sequence    :",
    int(visit_counts.max())
)

print(
    "Mean visits/sequence       :",
    f"{visit_counts.mean():.3f}"
)

print(
    "Median visits/sequence     :",
    f"{visit_counts.median():.1f}"
)

print(
    "Future MRI leakage         :",
    int(sequences["future_mri_count"].sum())
)


print("\nVISITS PER LANDMARK")
print(
    visit_counts
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nFIRST 5 LOCKED SEQUENCES")

show_cols = [
    "locked_subject_id",
    "landmark_mri_id",
    "landmark_day",
    "n_visits",
    "first_mri_day",
    "mri_ids",
]

print(
    sequences[show_cols]
    .head(5)
    .to_string(index=False)
)


print("\nOUTPUT:")
print(OUT)

print("\nEXPANDED VISITS:")
print(EXPANDED_OUT)

print("\nSUMMARY:")
print(SUMMARY_OUT)

print("\nLOCK:")
print(LOCK_OUT)


print("\nSHA256")
print(
    "MRI acquisition lock :",
    hash_mri_lock
)

print(
    "Step-48 landmarks    :",
    hash_landmark_lock
)

print(
    "Step-50 sequences    :",
    hash_sequence_output
)

print(
    "Expanded visits      :",
    hash_expanded_output
)


print("\nIMPORTANT:")
print("No external outcome data were read.")
print("No prediction files were read.")
print("No frozen ADNI model inference was performed.")
print("No MRI files were modified.")
print("Every sequence contains only MRI scans on/before its landmark.")
print("Future MRI leakage count must equal zero.")


print("\n" + "=" * 72)
print(
    "STEP 50 PASSED — EXTERNAL LONGITUDINAL MRI SEQUENCES LOCKED"
)
print("=" * 72)