from pathlib import Path
import hashlib
import json
import pandas as pd


# ============================================================
# STEP 47
# OASIS-3 EXTERNAL LANDMARK FUTURE-FOLLOW-UP AUDIT
#
# PURPOSE
# ------------------------------------------------------------
# Starting from the locked Step-46 CDR=0 MRI landmarks:
#
#   1. Preserve all 1,643 candidate landmarks.
#   2. Examine subsequent locked CDR observations.
#   3. Record first post-landmark CDR > 0.
#   4. Record first post-landmark CDR >= 0.5.
#   5. Record first post-landmark CDR >= 1.
#   6. Record last available clinical follow-up.
#   7. Audit support for frozen 12/24/36/48/60-month horizons.
#
# IMPORTANT
# ------------------------------------------------------------
# This is an outcome-availability audit only.
#
# It does NOT:
#   - modify MRI files,
#   - read model predictions,
#   - run frozen M0,
#   - select a threshold from model performance,
#   - tune the external validation cohort.
# ============================================================


ROOT = Path(__file__).resolve().parent
COHORT = ROOT / "cohort_construction"

LANDMARK_FILE = (
    COHORT /
    "OASIS3_STEP46_CDR0_ELIGIBLE_MRI_LANDMARKS.csv"
)

CDR_FILE = (
    COHORT /
    "OASIS3_STEP45_CDR_CLINICAL_TIMELINE_LOCKED.csv"
)

OUTPUT_FILE = (
    COHORT /
    "OASIS3_STEP47_LANDMARK_FUTURE_FOLLOWUP_AUDIT.csv"
)

SUMMARY_FILE = (
    COHORT /
    "OASIS3_STEP47_LANDMARK_FUTURE_FOLLOWUP_SUMMARY.json"
)

LOCK_FILE = (
    COHORT /
    "OASIS3_STEP47_LANDMARK_FUTURE_FOLLOWUP_LOCK.json"
)


# ============================================================
# CONSTANTS
# ============================================================

DAYS_PER_YEAR = 365.25

HORIZONS_MONTHS = [
    12,
    24,
    36,
    48,
    60,
]

HORIZON_DAYS = {
    month: DAYS_PER_YEAR * month / 12.0
    for month in HORIZONS_MONTHS
}


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def first_future_threshold(
    future,
    threshold,
):
    """
    Return the first future CDR visit reaching the specified
    threshold. Returns (day, CDRTOT), or (NA, NA).
    """

    hit = future[
        future["CDRTOT"] >= threshold
    ]

    if len(hit) == 0:
        return pd.NA, pd.NA

    row = hit.iloc[0]

    return (
        float(row["days_to_visit"]),
        float(row["CDRTOT"]),
    )


# ============================================================
# 1. HEADER
# ============================================================

print("=" * 76)
print("STEP 47 — OASIS-3 EXTERNAL LANDMARK FUTURE-FOLLOW-UP AUDIT")
print("=" * 76)


# ============================================================
# 2. INPUT CHECKS
# ============================================================

for path in [
    LANDMARK_FILE,
    CDR_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required locked input missing:\n{path}"
        )

print("\nLANDMARK INPUT:")
print(LANDMARK_FILE)

print("\nCDR TIMELINE INPUT:")
print(CDR_FILE)


# ============================================================
# 3. LOAD
# ============================================================

landmarks = pd.read_csv(
    LANDMARK_FILE,
    low_memory=False,
)

cdr = pd.read_csv(
    CDR_FILE,
    low_memory=False,
)


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

landmark_required = [
    "locked_subject_id",
    "locked_mri_id",
    "mri_day",
    "prior_cdr_available",
    "prior_clinical_day",
    "prior_cdrtot",
    "clinical_to_mri_gap_days",
    "eligible_cdr0_landmark",
]

cdr_required = [
    "locked_subject_id",
    "days_to_visit",
    "CDRTOT",
]

missing_landmark = [
    c
    for c in landmark_required
    if c not in landmarks.columns
]

missing_cdr = [
    c
    for c in cdr_required
    if c not in cdr.columns
]

if missing_landmark:
    raise RuntimeError(
        "Step-46 landmark table missing columns: "
        + str(missing_landmark)
    )

if missing_cdr:
    raise RuntimeError(
        "Step-45 CDR table missing columns: "
        + str(missing_cdr)
    )


# ============================================================
# 5. VERIFY STEP-46 LOCKED CANDIDATE COHORT
# ============================================================

landmarks["locked_subject_id"] = (
    landmarks["locked_subject_id"]
    .astype(str)
    .str.strip()
)

landmarks["locked_mri_id"] = (
    landmarks["locked_mri_id"]
    .astype(str)
    .str.strip()
)

landmarks["mri_day"] = pd.to_numeric(
    landmarks["mri_day"],
    errors="coerce",
)

landmarks["prior_cdrtot"] = pd.to_numeric(
    landmarks["prior_cdrtot"],
    errors="coerce",
)

landmarks["clinical_to_mri_gap_days"] = pd.to_numeric(
    landmarks["clinical_to_mri_gap_days"],
    errors="coerce",
)

if len(landmarks) != 1643:
    raise RuntimeError(
        "Expected 1643 Step-46 eligible landmarks, "
        f"found {len(landmarks)}"
    )

n_subjects = (
    landmarks["locked_subject_id"]
    .nunique()
)

if n_subjects != 542:
    raise RuntimeError(
        "Expected 542 Step-46 eligible participants, "
        f"found {n_subjects}"
    )

if landmarks["locked_mri_id"].nunique() != 1643:
    raise RuntimeError(
        "Step-46 locked_mri_id values are not unique."
    )

if landmarks["mri_day"].isna().any():
    raise RuntimeError(
        "Missing MRI day in Step-46 landmarks."
    )

if landmarks["prior_cdrtot"].isna().any():
    raise RuntimeError(
        "Missing prior CDR in Step-46 eligible landmarks."
    )

if not (
    landmarks["prior_cdrtot"] == 0
).all():
    raise RuntimeError(
        "Step-46 eligible table contains non-zero "
        "prior CDR values."
    )

print("\nSTEP-46 COHORT VERIFICATION: PASS")
print("Candidate landmarks :", len(landmarks))
print("Participants        :", n_subjects)


# ============================================================
# 6. CLEAN CDR TIMELINE
# ============================================================

cdr["locked_subject_id"] = (
    cdr["locked_subject_id"]
    .astype(str)
    .str.strip()
)

cdr["days_to_visit"] = pd.to_numeric(
    cdr["days_to_visit"],
    errors="coerce",
)

cdr["CDRTOT"] = pd.to_numeric(
    cdr["CDRTOT"],
    errors="coerce",
)

cdr = cdr[
    cdr["days_to_visit"].notna()
    & cdr["CDRTOT"].notna()
].copy()

cdr = cdr.sort_values(
    [
        "locked_subject_id",
        "days_to_visit",
    ]
).reset_index(drop=True)

cdr_groups = {
    subject: group.copy()
    for subject, group
    in cdr.groupby("locked_subject_id")
}

print("\nValid locked CDR visits :", len(cdr))
print(
    "Participants in CDR table :",
    cdr["locked_subject_id"].nunique(),
)


# ============================================================
# 7. AUDIT EACH MRI LANDMARK
# ============================================================

records = []

for _, row in landmarks.iterrows():

    subject = row["locked_subject_id"]
    mri_id = row["locked_mri_id"]
    mri_day = float(row["mri_day"])

    subject_cdr = cdr_groups.get(subject)

    if subject_cdr is None:
        raise RuntimeError(
            f"No CDR timeline for eligible subject {subject}"
        )

    # --------------------------------------------------------
    # Future means STRICTLY after MRI landmark.
    # Same-day CDR belongs to baseline/landmark state,
    # not future outcome.
    # --------------------------------------------------------

    future = subject_cdr[
        subject_cdr["days_to_visit"] > mri_day
    ].copy()

    future = future.sort_values(
        "days_to_visit"
    )

    rec = {
        "locked_subject_id":
            subject,

        "locked_mri_id":
            mri_id,

        "mri_day":
            int(mri_day),

        "prior_clinical_day":
            row["prior_clinical_day"],

        "prior_cdrtot":
            row["prior_cdrtot"],

        "clinical_to_mri_gap_days":
            row["clinical_to_mri_gap_days"],

        "n_future_cdr_visits":
            len(future),

        "has_future_cdr":
            len(future) > 0,

        "first_future_clinical_day":
            pd.NA,

        "first_future_cdrtot":
            pd.NA,

        "last_future_clinical_day":
            pd.NA,

        "last_future_cdrtot":
            pd.NA,

        "followup_days_from_mri":
            0.0,
    }

    # --------------------------------------------------------
    # First / last future clinical observation
    # --------------------------------------------------------

    if len(future) > 0:

        first = future.iloc[0]
        last = future.iloc[-1]

        rec[
            "first_future_clinical_day"
        ] = float(first["days_to_visit"])

        rec[
            "first_future_cdrtot"
        ] = float(first["CDRTOT"])

        rec[
            "last_future_clinical_day"
        ] = float(last["days_to_visit"])

        rec[
            "last_future_cdrtot"
        ] = float(last["CDRTOT"])

        rec[
            "followup_days_from_mri"
        ] = float(
            last["days_to_visit"] - mri_day
        )

    # --------------------------------------------------------
    # First CDR > 0
    # --------------------------------------------------------

    positive = future[
        future["CDRTOT"] > 0
    ]

    if len(positive) > 0:

        hit = positive.iloc[0]

        event_day = float(
            hit["days_to_visit"]
        )

        rec[
            "first_cdr_gt0_day"
        ] = event_day

        rec[
            "first_cdr_gt0_value"
        ] = float(hit["CDRTOT"])

        rec[
            "days_to_first_cdr_gt0"
        ] = event_day - mri_day

    else:

        rec["first_cdr_gt0_day"] = pd.NA
        rec["first_cdr_gt0_value"] = pd.NA
        rec["days_to_first_cdr_gt0"] = pd.NA

    # --------------------------------------------------------
    # First CDR >= 0.5
    # --------------------------------------------------------

    event_day, event_value = (
        first_future_threshold(
            future,
            0.5,
        )
    )

    rec[
        "first_cdr_ge_0p5_day"
    ] = event_day

    rec[
        "first_cdr_ge_0p5_value"
    ] = event_value

    if pd.notna(event_day):

        rec[
            "days_to_first_cdr_ge_0p5"
        ] = float(event_day - mri_day)

    else:

        rec[
            "days_to_first_cdr_ge_0p5"
        ] = pd.NA

    # --------------------------------------------------------
    # First CDR >= 1
    # --------------------------------------------------------

    event_day, event_value = (
        first_future_threshold(
            future,
            1.0,
        )
    )

    rec[
        "first_cdr_ge_1_day"
    ] = event_day

    rec[
        "first_cdr_ge_1_value"
    ] = event_value

    if pd.notna(event_day):

        rec[
            "days_to_first_cdr_ge_1"
        ] = float(event_day - mri_day)

    else:

        rec[
            "days_to_first_cdr_ge_1"
        ] = pd.NA

    # --------------------------------------------------------
    # Frozen horizon observation-support audit
    #
    # IMPORTANT:
    # These flags describe whether the observed clinical
    # follow-up extends to each horizon.
    #
    # They do NOT yet define evaluability in participants
    # who experience an event before the horizon.
    # That will be handled explicitly when outcomes are built.
    # --------------------------------------------------------

    for month in HORIZONS_MONTHS:

        hdays = HORIZON_DAYS[month]

        rec[
            f"followup_reaches_{month}m"
        ] = (
            rec["followup_days_from_mri"]
            >= hdays
        )

    records.append(rec)


audit = pd.DataFrame(records)


# ============================================================
# 8. POST-AUDIT INTEGRITY CHECKS
# ============================================================

if len(audit) != 1643:
    raise RuntimeError(
        f"Expected 1643 output rows, found {len(audit)}"
    )

if audit["locked_mri_id"].nunique() != 1643:
    raise RuntimeError(
        "Duplicate MRI landmark IDs in Step 47."
    )

if (
    audit["locked_subject_id"].nunique()
    != 542
):
    raise RuntimeError(
        "Step-47 participant count changed unexpectedly."
    )

if (
    pd.to_numeric(
        audit["followup_days_from_mri"],
        errors="coerce",
    ) < 0
).any():
    raise RuntimeError(
        "Negative future follow-up detected."
    )


# ============================================================
# 9. SAVE AUDIT
# ============================================================

audit.to_csv(
    OUTPUT_FILE,
    index=False,
)

output_sha = sha256_file(
    OUTPUT_FILE
)

step46_sha = sha256_file(
    LANDMARK_FILE
)

step45_sha = sha256_file(
    CDR_FILE
)


# ============================================================
# 10. SUMMARY STATISTICS
# ============================================================

n_future = int(
    audit["has_future_cdr"].sum()
)

n_no_future = int(
    (~audit["has_future_cdr"]).sum()
)

n_event_gt0 = int(
    audit["first_cdr_gt0_day"]
    .notna()
    .sum()
)

n_event_ge05 = int(
    audit["first_cdr_ge_0p5_day"]
    .notna()
    .sum()
)

n_event_ge1 = int(
    audit["first_cdr_ge_1_day"]
    .notna()
    .sum()
)


horizon_counts = {}

for month in HORIZONS_MONTHS:

    col = f"followup_reaches_{month}m"

    horizon_counts[
        str(month)
    ] = int(
        audit[col].sum()
    )


# ============================================================
# 11. WRITE SUMMARY
# ============================================================

summary = {
    "step": 47,

    "purpose":
        "OASIS-3 external MRI landmark "
        "future clinical follow-up audit",

    "candidate_landmarks":
        int(len(audit)),

    "participants":
        int(
            audit[
                "locked_subject_id"
            ].nunique()
        ),

    "landmarks_with_future_cdr":
        n_future,

    "landmarks_without_future_cdr":
        n_no_future,

    "landmarks_ever_future_cdr_gt_0":
        n_event_gt0,

    "landmarks_ever_future_cdr_ge_0p5":
        n_event_ge05,

    "landmarks_ever_future_cdr_ge_1":
        n_event_ge1,

    "followup_reaches_horizon_counts":
        horizon_counts,

    "horizon_months":
        HORIZONS_MONTHS,

    "days_per_year":
        DAYS_PER_YEAR,

    "selection_note":
        (
            "No landmark was excluded based on "
            "model predictions or model performance."
        ),

    "step46_landmark_sha256":
        step46_sha,

    "step45_cdr_sha256":
        step45_sha,

    "step47_output_sha256":
        output_sha,
}

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# 12. WRITE LOCK
# ============================================================

lock = {
    "step": 47,

    "locked_input_landmarks":
        str(LANDMARK_FILE),

    "locked_cdr_timeline":
        str(CDR_FILE),

    "output":
        str(OUTPUT_FILE),

    "candidate_landmarks":
        int(len(audit)),

    "participants":
        int(
            audit[
                "locked_subject_id"
            ].nunique()
        ),

    "future_definition":
        "CDR observation strictly after MRI day",

    "baseline_state":
        "most recent CDR at or before MRI = 0",

    "primary_event_threshold":
        "NOT YET LOCKED IN STEP 47",

    "clinical_mri_gap_cutoff":
        "NONE APPLIED IN STEP 47",

    "performance_based_selection":
        False,

    "model_inference_performed":
        False,

    "step46_landmark_sha256":
        step46_sha,

    "step45_cdr_sha256":
        step45_sha,

    "step47_output_sha256":
        output_sha,
}

with open(
    LOCK_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        lock,
        f,
        indent=2,
    )


# ============================================================
# 13. REPORT
# ============================================================

print("\n" + "=" * 76)
print("STEP 47 RESULTS")
print("=" * 76)

print("\nCANDIDATE COHORT")
print(
    "MRI landmarks                    :",
    len(audit),
)

print(
    "Participants                     :",
    audit["locked_subject_id"].nunique(),
)


print("\nFUTURE CLINICAL FOLLOW-UP")

print(
    "Landmarks with future CDR        :",
    n_future,
)

print(
    "Landmarks without future CDR     :",
    n_no_future,
)


print("\nOBSERVED FUTURE CDR EVENTS")

print(
    "Ever future CDR > 0              :",
    n_event_gt0,
)

print(
    "Ever future CDR >= 0.5           :",
    n_event_ge05,
)

print(
    "Ever future CDR >= 1             :",
    n_event_ge1,
)


print("\nFOLLOW-UP REACHES FROZEN HORIZON")

for month in HORIZONS_MONTHS:

    print(
        f"{month:>2} months                       :",
        horizon_counts[str(month)],
    )


print("\nFOLLOW-UP DAYS FROM MRI")

print(
    audit[
        "followup_days_from_mri"
    ]
    .describe()
    .to_string()
)


print("\nCLINICAL-TO-MRI BASELINE GAP")

print(
    audit[
        "clinical_to_mri_gap_days"
    ]
    .describe()
    .to_string()
)


print("\nOUTPUT:")
print(OUTPUT_FILE)

print("\nSUMMARY:")
print(SUMMARY_FILE)

print("\nLOCK:")
print(LOCK_FILE)


print("\nSHA256")

print(
    "Step-46 landmarks :",
    step46_sha,
)

print(
    "Step-45 CDR       :",
    step45_sha,
)

print(
    "Step-47 output    :",
    output_sha,
)


print("\nIMPORTANT:")
print("No MRI files were modified.")
print("No prediction files were read.")
print("No frozen ADNI model inference was performed.")
print("No external model performance was examined.")
print("No clinical-to-MRI gap cutoff was applied.")
print("No event threshold was selected from model performance.")


print("\n" + "=" * 76)
print(
    "STEP 47 PASSED — EXTERNAL LANDMARK "
    "FUTURE FOLLOW-UP AUDITED"
)
print("=" * 76)