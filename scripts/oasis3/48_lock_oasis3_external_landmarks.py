# ================================================================
# STEP 48 — LOCK OASIS-3 EXTERNAL VALIDATION MRI LANDMARKS
# ================================================================
#
# Purpose
# -------
# Construct and lock the final MRI landmark cohort for external
# validation using ONLY information audited before model evaluation.
#
# Locked eligibility rule:
#   1. MRI belongs to the locked OASIS-3 MRI acquisition set.
#   2. A prior/on-date clinical CDR assessment exists.
#   3. CDRTOT at that assessment = 0.
#   4. Clinical assessment occurred on or before MRI.
#   5. Clinical-to-MRI gap <= 365 days.
#
# IMPORTANT:
#   - No model predictions are read.
#   - No frozen ADNI model is executed.
#   - No external performance is examined.
#   - No MRI file is modified.
#   - The 365-day rule is applied independently of model performance.
#
# ================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime

# ------------------------------------------------
# Paths
# ------------------------------------------------

ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

COHORT = ROOT / "cohort_construction"

STEP46 = (
    COHORT /
    "OASIS3_STEP46_CDR0_ELIGIBLE_MRI_LANDMARKS.csv"
)

STEP47 = (
    COHORT /
    "OASIS3_STEP47_LANDMARK_FUTURE_FOLLOWUP_AUDIT.csv"
)

OUT = (
    COHORT /
    "OASIS3_STEP48_EXTERNAL_LANDMARKS_LOCKED.csv"
)

EXCLUDED = (
    COHORT /
    "OASIS3_STEP48_EXTERNAL_LANDMARKS_EXCLUDED.csv"
)

SUBJECT_SUMMARY = (
    COHORT /
    "OASIS3_STEP48_EXTERNAL_LANDMARK_SUBJECT_SUMMARY.csv"
)

SUMMARY = (
    COHORT /
    "OASIS3_STEP48_EXTERNAL_LANDMARKS_SUMMARY.json"
)

LOCK = (
    COHORT /
    "OASIS3_STEP48_EXTERNAL_LANDMARKS_LOCK.json"
)

MAX_BASELINE_GAP_DAYS = 365

# ------------------------------------------------
# Utility
# ------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def boolify(series):
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


print("=" * 72)
print("STEP 48 — LOCK OASIS-3 EXTERNAL VALIDATION LANDMARKS")
print("=" * 72)

# ------------------------------------------------
# Verify inputs
# ------------------------------------------------

for p in [STEP46, STEP47]:
    if not p.exists():
        raise FileNotFoundError(
            f"Required input not found:\n{p}"
        )

print("\nINPUT:")
print("Step 46:", STEP46)
print("Step 47:", STEP47)

step46_hash = sha256_file(STEP46)
step47_hash = sha256_file(STEP47)

print("\nSHA256 INPUTS")
print("Step 46 :", step46_hash)
print("Step 47 :", step47_hash)

# ------------------------------------------------
# Read data
# ------------------------------------------------

s46 = pd.read_csv(STEP46, low_memory=False)
s47 = pd.read_csv(STEP47, low_memory=False)

print("\nINPUT COUNTS")
print("Step 46 landmarks :", len(s46))
print("Step 47 rows      :", len(s47))

# ------------------------------------------------
# Required Step 46 columns
# ------------------------------------------------

required46 = [
    "locked_subject_id",
    "locked_mri_id",
    "mri_day",
    "prior_cdr_available",
    "prior_clinical_day",
    "prior_cdrtot",
    "clinical_to_mri_gap_days",
    "eligible_cdr0_landmark",
]

missing46 = [
    c for c in required46
    if c not in s46.columns
]

if missing46:
    raise RuntimeError(
        f"Step 46 missing required columns: {missing46}"
    )

# ------------------------------------------------
# Basic normalization
# ------------------------------------------------

for c in [
    "mri_day",
    "prior_clinical_day",
    "prior_cdrtot",
    "clinical_to_mri_gap_days",
]:
    s46[c] = pd.to_numeric(
        s46[c],
        errors="coerce"
    )

s46["prior_cdr_available"] = boolify(
    s46["prior_cdr_available"]
)

s46["eligible_cdr0_landmark"] = boolify(
    s46["eligible_cdr0_landmark"]
)

# ------------------------------------------------
# Integrity checks
# ------------------------------------------------

if s46["locked_mri_id"].duplicated().any():
    dup = s46.loc[
        s46["locked_mri_id"].duplicated(False),
        "locked_mri_id"
    ]

    raise RuntimeError(
        "Duplicate MRI IDs in Step 46:\n"
        + dup.to_string(index=False)
    )

if s46["locked_subject_id"].isna().any():
    raise RuntimeError(
        "Missing locked_subject_id in Step 46."
    )

if s46["locked_mri_id"].isna().any():
    raise RuntimeError(
        "Missing locked_mri_id in Step 46."
    )

# ------------------------------------------------
# Apply LOCKED eligibility rule
# ------------------------------------------------

base_eligible = (
    s46["eligible_cdr0_landmark"]
    & s46["prior_cdr_available"]
    & s46["prior_cdrtot"].notna()
    & np.isclose(
        s46["prior_cdrtot"],
        0.0,
        atol=1e-8
    )
    & s46["clinical_to_mri_gap_days"].notna()
    & (
        s46["clinical_to_mri_gap_days"] >= 0
    )
)

gap_pass = (
    s46["clinical_to_mri_gap_days"]
    <= MAX_BASELINE_GAP_DAYS
)

final_mask = base_eligible & gap_pass

locked = s46.loc[final_mask].copy()

excluded = s46.loc[
    base_eligible & (~gap_pass)
].copy()

# ------------------------------------------------
# Exclusion reason
# ------------------------------------------------

excluded["step48_exclusion_reason"] = (
    "CLINICAL_TO_MRI_GAP_GT_365_DAYS"
)

# ------------------------------------------------
# Final lock fields
# ------------------------------------------------

locked["step48_baseline_definition"] = (
    "latest CDR assessment on/before MRI"
)

locked["step48_required_cdrtot"] = 0.0

locked["step48_max_clinical_to_mri_gap_days"] = (
    MAX_BASELINE_GAP_DAYS
)

locked["step48_landmark_locked"] = True

# Stable ordering
locked = locked.sort_values(
    [
        "locked_subject_id",
        "mri_day",
        "locked_mri_id",
    ]
).reset_index(drop=True)

excluded = excluded.sort_values(
    [
        "locked_subject_id",
        "mri_day",
        "locked_mri_id",
    ]
).reset_index(drop=True)

# ------------------------------------------------
# Sanity assertions
# ------------------------------------------------

if len(locked) == 0:
    raise RuntimeError(
        "Step 48 produced zero landmarks."
    )

if (
    locked["clinical_to_mri_gap_days"]
    > MAX_BASELINE_GAP_DAYS
).any():
    raise RuntimeError(
        "Locked set contains gap >365 days."
    )

if (
    locked["clinical_to_mri_gap_days"] < 0
).any():
    raise RuntimeError(
        "Locked set contains negative clinical-to-MRI gap."
    )

if not np.allclose(
    locked["prior_cdrtot"].astype(float),
    0.0,
):
    raise RuntimeError(
        "Locked set contains non-zero baseline CDRTOT."
    )

if locked["locked_mri_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate MRI landmark in final lock."
    )

# ------------------------------------------------
# Participant summary
# ------------------------------------------------

subject_summary = (
    locked
    .groupby("locked_subject_id")
    .agg(
        n_locked_landmarks=(
            "locked_mri_id",
            "count"
        ),
        first_landmark_day=(
            "mri_day",
            "min"
        ),
        last_landmark_day=(
            "mri_day",
            "max"
        ),
        median_baseline_gap_days=(
            "clinical_to_mri_gap_days",
            "median"
        ),
        max_baseline_gap_days=(
            "clinical_to_mri_gap_days",
            "max"
        ),
    )
    .reset_index()
)

# ------------------------------------------------
# Save outputs
# ------------------------------------------------

locked.to_csv(
    OUT,
    index=False
)

excluded.to_csv(
    EXCLUDED,
    index=False
)

subject_summary.to_csv(
    SUBJECT_SUMMARY,
    index=False
)

# ------------------------------------------------
# Re-read output and verify exact serialization
# ------------------------------------------------

check = pd.read_csv(
    OUT,
    low_memory=False
)

if len(check) != len(locked):
    raise RuntimeError(
        "Saved Step 48 output row count mismatch."
    )

if check["locked_mri_id"].nunique() != len(check):
    raise RuntimeError(
        "Saved Step 48 output contains duplicate MRI IDs."
    )

# ------------------------------------------------
# Descriptive results
# ------------------------------------------------

n_landmarks = len(locked)

n_participants = (
    locked["locked_subject_id"].nunique()
)

n_excluded_gap = len(excluded)

gap_stats = (
    locked["clinical_to_mri_gap_days"]
    .describe()
    .to_dict()
)

landmarks_per_subject = (
    subject_summary["n_locked_landmarks"]
    .describe()
    .to_dict()
)

# ------------------------------------------------
# Summary JSON
# ------------------------------------------------

summary = {
    "step": 48,
    "purpose": (
        "Lock final OASIS-3 external validation MRI "
        "landmarks before model evaluation."
    ),
    "created": datetime.now().isoformat(),
    "baseline_definition": (
        "Latest available CDR assessment on or before MRI."
    ),
    "required_baseline_cdrtot": 0.0,
    "maximum_clinical_to_mri_gap_days": (
        MAX_BASELINE_GAP_DAYS
    ),
    "step46_input_landmarks": int(len(s46)),
    "final_locked_landmarks": int(n_landmarks),
    "final_locked_participants": int(n_participants),
    "excluded_for_gap_gt_365": int(n_excluded_gap),
    "clinical_to_mri_gap_statistics": {
        k: (
            None if pd.isna(v)
            else float(v)
        )
        for k, v in gap_stats.items()
    },
    "landmarks_per_subject_statistics": {
        k: (
            None if pd.isna(v)
            else float(v)
        )
        for k, v in landmarks_per_subject.items()
    },
    "prediction_horizons_months": [
        12,
        24,
        36,
    ],
    "maximum_survival_followup_months": 60,
    "model_predictions_read": False,
    "frozen_adni_model_executed": False,
    "external_performance_examined": False,
    "mri_files_modified": False,
    "selection_based_on_model_performance": False,
}

with open(
    SUMMARY,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )

# ------------------------------------------------
# Final hashes
# ------------------------------------------------

out_hash = sha256_file(OUT)
excluded_hash = sha256_file(EXCLUDED)
subject_hash = sha256_file(SUBJECT_SUMMARY)

lock = {
    "step": 48,
    "locked": True,
    "step46_sha256": step46_hash,
    "step47_sha256": step47_hash,
    "step48_landmarks_sha256": out_hash,
    "step48_excluded_sha256": excluded_hash,
    "step48_subject_summary_sha256": subject_hash,
    "baseline_cdrtot": 0.0,
    "maximum_clinical_to_mri_gap_days": 365,
    "rule_selected_without_model_performance": True,
    "do_not_modify_after_external_model_evaluation": True,
}

with open(
    LOCK,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        lock,
        f,
        indent=2
    )

# ------------------------------------------------
# Report
# ------------------------------------------------

print("\n" + "=" * 72)
print("STEP 48 RESULTS")
print("=" * 72)

print(
    "Step 46 eligible landmarks :",
    len(s46)
)

print(
    "Final locked landmarks     :",
    n_landmarks
)

print(
    "Final locked participants  :",
    n_participants
)

print(
    "Excluded gap >365 days     :",
    n_excluded_gap
)

print("\nBASELINE GAP — LOCKED LANDMARKS")
print(
    locked[
        "clinical_to_mri_gap_days"
    ].describe().to_string()
)

print("\nLANDMARKS PER PARTICIPANT")
print(
    subject_summary[
        "n_locked_landmarks"
    ].describe().to_string()
)

print("\nOUTPUT:")
print(OUT)

print("\nEXCLUDED:")
print(EXCLUDED)

print("\nSUBJECT SUMMARY:")
print(SUBJECT_SUMMARY)

print("\nSUMMARY:")
print(SUMMARY)

print("\nLOCK:")
print(LOCK)

print("\nSHA256")
print("Step-46 input :", step46_hash)
print("Step-47 input :", step47_hash)
print("Step-48 lock  :", out_hash)

print("\nIMPORTANT:")
print("No MRI files were modified.")
print("No prediction files were read.")
print("No frozen ADNI model inference was performed.")
print("No external model performance was examined.")
print(
    "The 365-day clinical-to-MRI rule was applied "
    "before external model evaluation."
)
print(
    "The external landmark cohort is now locked "
    "against performance-driven modification."
)

print("\n" + "=" * 72)
print(
    "STEP 48 PASSED — EXTERNAL MRI LANDMARK COHORT LOCKED"
)
print("=" * 72)