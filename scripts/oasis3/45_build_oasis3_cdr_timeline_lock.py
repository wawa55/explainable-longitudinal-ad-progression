from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime

# ============================================================
# STEP 45
# OASIS-3 EXTERNAL VALIDATION
# BUILD + LOCK CDR CLINICAL TIMELINE
#
# IMPORTANT:
# - No MRI files are modified.
# - No frozen ADNI model is run.
# - No predictions are read.
# - No threshold is selected from model performance.
# - CDR >= 0.5 and CDR >= 1 are preserved as candidate
#   clinical transition definitions for protocol mapping.
# ============================================================

ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

COHORT_DIR = ROOT / "cohort_construction"

MRI_LOCK = (
    COHORT_DIR /
    "OASIS3_EXACT_T1_ACQUISITION_LOCKED.csv"
)

METADATA_ROOT = (
    ROOT /
    "metadata" /
    "OASIS3_extracted" /
    "OASIS3_data_files" /
    "scans"
)

OUTPUT_CSV = (
    COHORT_DIR /
    "OASIS3_STEP45_CDR_CLINICAL_TIMELINE_LOCKED.csv"
)

SUBJECT_SUMMARY_CSV = (
    COHORT_DIR /
    "OASIS3_STEP45_CDR_SUBJECT_SUMMARY.csv"
)

SUMMARY_JSON = (
    COHORT_DIR /
    "OASIS3_STEP45_CDR_TIMELINE_SUMMARY.json"
)

LOCK_JSON = (
    COHORT_DIR /
    "OASIS3_STEP45_CDR_TIMELINE_LOCK.json"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def find_exactly_one(root, filename):

    matches = list(root.rglob(filename))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {filename}, "
            f"found {len(matches)}:\n"
            + "\n".join(str(x) for x in matches)
        )

    return matches[0]


# ============================================================
# 1. BASIC INPUT VALIDATION
# ============================================================

print("=" * 78)
print("STEP 45 — OASIS-3 CDR CLINICAL TIMELINE LOCK")
print("=" * 78)

if not MRI_LOCK.exists():
    raise FileNotFoundError(
        f"Missing MRI acquisition lock:\n{MRI_LOCK}"
    )

CDR_FILE = find_exactly_one(
    METADATA_ROOT,
    "OASIS3_UDSb4_cdr.csv"
)

print("\nMRI lock:")
print(MRI_LOCK)

print("\nCDR source:")
print(CDR_FILE)


# ============================================================
# 2. LOAD FINAL LOCKED MRI COHORT
# ============================================================

mri = pd.read_csv(
    MRI_LOCK,
    low_memory=False
)

required_mri = {
    "locked_subject_id",
    "locked_mri_id",
}

missing = required_mri - set(mri.columns)

if missing:
    raise RuntimeError(
        f"MRI lock missing required columns: {sorted(missing)}"
    )

mri["locked_subject_id"] = (
    mri["locked_subject_id"]
    .astype(str)
    .str.strip()
)

locked_subjects = sorted(
    mri["locked_subject_id"]
    .dropna()
    .unique()
)

locked_subject_set = set(locked_subjects)

n_locked_subjects = len(locked_subjects)
n_locked_acquisitions = len(mri)

print("\nFinal MRI cohort:")
print("Participants       :", n_locked_subjects)
print("Locked acquisitions:", n_locked_acquisitions)

if n_locked_subjects != 605:
    raise RuntimeError(
        f"Expected 605 locked participants, "
        f"found {n_locked_subjects}"
    )

if n_locked_acquisitions != 1901:
    raise RuntimeError(
        f"Expected 1901 locked acquisitions, "
        f"found {n_locked_acquisitions}"
    )


# ============================================================
# 3. LOAD CDR TABLE
# ============================================================

cdr = pd.read_csv(
    CDR_FILE,
    low_memory=False
)

required_cdr = {
    "OASISID",
    "OASIS_session_label",
    "days_to_visit",
    "age at visit",
    "CDRSUM",
    "CDRTOT",
}

missing = required_cdr - set(cdr.columns)

if missing:
    raise RuntimeError(
        f"CDR table missing required columns: {sorted(missing)}"
    )

cdr["OASISID"] = (
    cdr["OASISID"]
    .astype(str)
    .str.strip()
)

cdr["days_to_visit"] = pd.to_numeric(
    cdr["days_to_visit"],
    errors="coerce"
)

cdr["age at visit"] = pd.to_numeric(
    cdr["age at visit"],
    errors="coerce"
)

cdr["CDRSUM"] = pd.to_numeric(
    cdr["CDRSUM"],
    errors="coerce"
)

cdr["CDRTOT"] = pd.to_numeric(
    cdr["CDRTOT"],
    errors="coerce"
)


# ============================================================
# 4. RESTRICT TO FINAL LOCKED 605-PARTICIPANT COHORT
# ============================================================

x = cdr[
    cdr["OASISID"].isin(locked_subject_set)
].copy()

print("\nCDR records in locked cohort:")
print("Rows                :", len(x))
print(
    "Participants found  :",
    x["OASISID"].nunique()
)

if x["OASISID"].nunique() != n_locked_subjects:
    missing_subjects = sorted(
        locked_subject_set -
        set(x["OASISID"].unique())
    )

    raise RuntimeError(
        "Not every locked participant has CDR data.\n"
        f"Missing participants: {missing_subjects[:20]}"
    )


# ============================================================
# 5. STRICT TIMELINE VALIDATION
# ============================================================

missing_days = int(
    x["days_to_visit"].isna().sum()
)

missing_cdrtot = int(
    x["CDRTOT"].isna().sum()
)

print("\nMissingness:")
print("Missing days_to_visit:", missing_days)
print("Missing CDRTOT        :", missing_cdrtot)

if missing_days != 0:
    raise RuntimeError(
        "Missing days_to_visit detected in locked cohort."
    )

if missing_cdrtot != 0:
    raise RuntimeError(
        "Missing CDRTOT detected in locked cohort."
    )


# ============================================================
# 6. VALIDATE CDR VALUES
# ============================================================

observed_values = sorted(
    x["CDRTOT"]
    .dropna()
    .unique()
    .tolist()
)

allowed_values = {
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
}

unexpected = sorted(
    set(observed_values) - allowed_values
)

print("\nObserved CDRTOT values:")
print(observed_values)

if unexpected:
    raise RuntimeError(
        f"Unexpected CDRTOT values: {unexpected}"
    )


# ============================================================
# 7. CHECK DUPLICATE SUBJECT-DAY RECORDS
# ============================================================

duplicate_subject_day = (
    x.duplicated(
        subset=[
            "OASISID",
            "days_to_visit"
        ],
        keep=False
    )
)

n_duplicate_rows = int(
    duplicate_subject_day.sum()
)

print("\nDuplicate subject/day rows:")
print(n_duplicate_rows)

if n_duplicate_rows > 0:

    dup = (
        x.loc[
            duplicate_subject_day,
            [
                "OASISID",
                "OASIS_session_label",
                "days_to_visit",
                "CDRTOT",
                "CDRSUM"
            ]
        ]
        .sort_values(
            [
                "OASISID",
                "days_to_visit"
            ]
        )
    )

    duplicate_file = (
        COHORT_DIR /
        "OASIS3_STEP45_DUPLICATE_CDR_SUBJECT_DAY.csv"
    )

    dup.to_csv(
        duplicate_file,
        index=False
    )

    # Allow exact duplicate-day clinical values,
    # but reject conflicting CDR values.

    conflict = (
        dup.groupby(
            [
                "OASISID",
                "days_to_visit"
            ]
        )["CDRTOT"]
        .nunique()
        .gt(1)
    )

    if conflict.any():
        raise RuntimeError(
            "Conflicting CDRTOT values exist for the same "
            "participant and clinical day. Review:\n"
            f"{duplicate_file}"
        )


# ============================================================
# 8. COLLAPSE EXACT SAME-DAY DUPLICATES IF PRESENT
# ============================================================

x = (
    x.sort_values(
        [
            "OASISID",
            "days_to_visit",
            "OASIS_session_label"
        ]
    )
    .drop_duplicates(
        subset=[
            "OASISID",
            "days_to_visit"
        ],
        keep="first"
    )
    .copy()
)


# ============================================================
# 9. CREATE CLINICAL TRANSITION FLAGS
# ============================================================

x["cdr_ge_0p5"] = (
    x["CDRTOT"] >= 0.5
).astype(int)

x["cdr_ge_1"] = (
    x["CDRTOT"] >= 1.0
).astype(int)

x["cdr_zero"] = (
    x["CDRTOT"] == 0.0
).astype(int)

x["clinical_visit_number"] = (
    x.groupby("OASISID")
    .cumcount()
    + 1
)

x["previous_cdrtot"] = (
    x.groupby("OASISID")["CDRTOT"]
    .shift(1)
)

x["transition_0_to_ge_0p5"] = (
    (x["previous_cdrtot"] == 0.0) &
    (x["CDRTOT"] >= 0.5)
).astype(int)

x["transition_lt1_to_ge1"] = (
    (x["previous_cdrtot"] < 1.0) &
    (x["CDRTOT"] >= 1.0)
).astype(int)


# ============================================================
# 10. SUBJECT-LEVEL SUMMARY
# ============================================================

rows = []

for subject_id, g in x.groupby(
    "OASISID",
    sort=True
):

    g = g.sort_values(
        "days_to_visit"
    )

    first = g.iloc[0]
    last = g.iloc[-1]

    ge05 = g[
        g["CDRTOT"] >= 0.5
    ]

    ge1 = g[
        g["CDRTOT"] >= 1.0
    ]

    trans05 = g[
        g["transition_0_to_ge_0p5"] == 1
    ]

    trans1 = g[
        g["transition_lt1_to_ge1"] == 1
    ]

    rows.append(
        {
            "locked_subject_id":
                subject_id,

            "n_cdr_visits":
                len(g),

            "first_cdr_day":
                float(first["days_to_visit"]),

            "last_cdr_day":
                float(last["days_to_visit"]),

            "clinical_followup_days":
                float(
                    last["days_to_visit"] -
                    first["days_to_visit"]
                ),

            "first_cdrtot":
                float(first["CDRTOT"]),

            "last_cdrtot":
                float(last["CDRTOT"]),

            "max_cdrtot":
                float(g["CDRTOT"].max()),

            "ever_cdr_ge_0p5":
                int(len(ge05) > 0),

            "first_day_cdr_ge_0p5":
                (
                    float(
                        ge05.iloc[0]["days_to_visit"]
                    )
                    if len(ge05) > 0
                    else np.nan
                ),

            "ever_cdr_ge_1":
                int(len(ge1) > 0),

            "first_day_cdr_ge_1":
                (
                    float(
                        ge1.iloc[0]["days_to_visit"]
                    )
                    if len(ge1) > 0
                    else np.nan
                ),

            "observed_transition_0_to_ge_0p5":
                int(len(trans05) > 0),

            "first_transition_day_0_to_ge_0p5":
                (
                    float(
                        trans05.iloc[0]["days_to_visit"]
                    )
                    if len(trans05) > 0
                    else np.nan
                ),

            "observed_transition_lt1_to_ge1":
                int(len(trans1) > 0),

            "first_transition_day_lt1_to_ge1":
                (
                    float(
                        trans1.iloc[0]["days_to_visit"]
                    )
                    if len(trans1) > 0
                    else np.nan
                ),
        }
    )

subject_summary = pd.DataFrame(
    rows
)


# ============================================================
# 11. FINAL STRICT CHECKS
# ============================================================

if len(subject_summary) != 605:
    raise RuntimeError(
        f"Expected 605 subject summaries, "
        f"found {len(subject_summary)}"
    )

if subject_summary[
    "locked_subject_id"
].nunique() != 605:
    raise RuntimeError(
        "Subject summary IDs are not unique."
    )

if set(
    subject_summary[
        "locked_subject_id"
    ]
) != locked_subject_set:
    raise RuntimeError(
        "Subject summary does not exactly match "
        "the frozen MRI cohort."
    )


# ============================================================
# 12. SAVE LOCKED CLINICAL TIMELINE
# ============================================================

timeline_columns = [
    "OASISID",
    "OASIS_session_label",
    "days_to_visit",
    "age at visit",
    "CDRTOT",
    "CDRSUM",
    "clinical_visit_number",
    "previous_cdrtot",
    "cdr_zero",
    "cdr_ge_0p5",
    "cdr_ge_1",
    "transition_0_to_ge_0p5",
    "transition_lt1_to_ge1",
]

timeline = x[
    timeline_columns
].copy()

timeline = timeline.rename(
    columns={
        "OASISID":
            "locked_subject_id"
    }
)

timeline.to_csv(
    OUTPUT_CSV,
    index=False
)

subject_summary.to_csv(
    SUBJECT_SUMMARY_CSV,
    index=False
)


# ============================================================
# 13. HASH OUTPUTS
# ============================================================

mri_lock_sha = sha256_file(
    MRI_LOCK
)

cdr_source_sha = sha256_file(
    CDR_FILE
)

timeline_sha = sha256_file(
    OUTPUT_CSV
)

subject_summary_sha = sha256_file(
    SUBJECT_SUMMARY_CSV
)


# ============================================================
# 14. SUMMARY
# ============================================================

cdr_counts = (
    timeline["CDRTOT"]
    .value_counts()
    .sort_index()
)

n_ever_05 = int(
    subject_summary[
        "ever_cdr_ge_0p5"
    ].sum()
)

n_ever_1 = int(
    subject_summary[
        "ever_cdr_ge_1"
    ].sum()
)

n_transition_05 = int(
    subject_summary[
        "observed_transition_0_to_ge_0p5"
    ].sum()
)

n_transition_1 = int(
    subject_summary[
        "observed_transition_lt1_to_ge1"
    ].sum()
)

summary = {
    "step": 45,
    "purpose":
        "Lock OASIS-3 CDR clinical timeline "
        "before external outcome construction",

    "locked_participants":
        605,

    "locked_mri_acquisitions":
        1901,

    "clinical_rows":
        int(len(timeline)),

    "participants_with_cdr":
        int(
            timeline[
                "locked_subject_id"
            ].nunique()
        ),

    "observed_cdrtot_values":
        [
            float(v)
            for v in observed_values
        ],

    "cdrtot_counts":
        {
            str(k): int(v)
            for k, v in cdr_counts.items()
        },

    "participants_ever_cdr_ge_0p5":
        n_ever_05,

    "participants_ever_cdr_ge_1":
        n_ever_1,

    "observed_transition_0_to_ge_0p5":
        n_transition_05,

    "observed_transition_lt1_to_ge1":
        n_transition_1,

    "event_threshold_selected":
        False,

    "model_predictions_used":
        False,

    "external_outcomes_used_for_model_tuning":
        False,
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# 15. WRITE FORMAL LOCK
# ============================================================

lock = {
    "step": 45,

    "created":
        datetime.now().isoformat(),

    "status":
        "LOCKED",

    "locked_cohort_participants":
        605,

    "locked_mri_acquisitions":
        1901,

    "clinical_source":
        str(CDR_FILE),

    "clinical_timeline":
        str(OUTPUT_CSV),

    "subject_summary":
        str(SUBJECT_SUMMARY_CSV),

    "sha256": {
        "mri_acquisition_lock":
            mri_lock_sha,

        "cdr_source":
            cdr_source_sha,

        "clinical_timeline":
            timeline_sha,

        "subject_summary":
            subject_summary_sha,
    },

    "protocol_notes": [
        "No MRI files were modified.",
        "No frozen ADNI model was executed.",
        "No prediction files were read.",
        "No clinical threshold was selected "
        "using predictive performance.",
        "CDR >= 0.5 and CDR >= 1 were preserved "
        "for subsequent protocol mapping.",
    ],
}

with open(
    LOCK_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        lock,
        f,
        indent=2
    )


# ============================================================
# 16. REPORT
# ============================================================

print("\n" + "=" * 78)
print("STEP 45 RESULTS")
print("=" * 78)

print("Participants              :", 605)
print("Locked MRI acquisitions   :", 1901)
print("Locked CDR visits         :", len(timeline))

print("\nCDRTOT counts:")
print(cdr_counts.to_string())

print(
    "\nParticipants ever CDR >= 0.5:",
    n_ever_05
)

print(
    "Participants ever CDR >= 1  :",
    n_ever_1
)

print(
    "Observed 0 -> >=0.5 transitions:",
    n_transition_05
)

print(
    "Observed <1 -> >=1 transitions :",
    n_transition_1
)

print("\nOUTPUT:")
print(OUTPUT_CSV)

print("\nSUBJECT SUMMARY:")
print(SUBJECT_SUMMARY_CSV)

print("\nSUMMARY:")
print(SUMMARY_JSON)

print("\nLOCK:")
print(LOCK_JSON)

print("\nSHA256:")
print("MRI acquisition lock :", mri_lock_sha)
print("CDR source           :", cdr_source_sha)
print("Clinical timeline    :", timeline_sha)
print("Subject summary      :", subject_summary_sha)

print("\nIMPORTANT:")
print("No MRI files were modified.")
print("No frozen ADNI model inference was performed.")
print("No prediction files were read.")
print("No event threshold was selected from model performance.")

print("\n" + "=" * 78)
print("STEP 45 PASSED — CDR CLINICAL TIMELINE LOCKED")
print("=" * 78)