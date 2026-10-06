from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import json
import ast

# ============================================================
# STEP 53
# FINAL OASIS-3 EXTERNAL INFERENCE MANIFEST LOCK
#
# Purpose:
#   Join the already-locked:
#       Step 49 survival targets
#       Step 51 frozen temporal inputs
#       Step 52 MRI readiness
#
# IMPORTANT:
#   - No MRI is modified.
#   - No prediction file is read.
#   - No frozen ADNI checkpoint is loaded.
#   - No model inference is performed.
#   - No performance metric is calculated.
#
# Step-49 schema note:
#   Step 49 identifies each landmark using:
#       locked_subject_id
#       locked_mri_id
#       mri_day
#
#   Steps 51/52 identify the same landmark using:
#       locked_subject_id
#       landmark_mri_id
#       landmark_day
#
#   Therefore Step 49 is standardized IN MEMORY ONLY:
#       locked_mri_id -> landmark_mri_id
#       mri_day       -> landmark_day
#
#   The locked Step-49 source file is NEVER modified.
# ============================================================

ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)
COHORT = ROOT / "cohort_construction"

STEP49 = (
    COHORT
    / "OASIS3_STEP49_EXTERNAL_SURVIVAL_TARGETS_LOCKED.csv"
)

STEP51 = (
    COHORT
    / "OASIS3_STEP51_FROZEN_TEMPORAL_INPUTS_LOCKED.csv"
)

STEP52 = (
    COHORT
    / "OASIS3_STEP52_SEQUENCE_READINESS.csv"
)

OUT = (
    COHORT
    / "OASIS3_STEP53_FINAL_EXTERNAL_INFERENCE_MANIFEST_LOCKED.csv"
)

SUMMARY = (
    COHORT
    / "OASIS3_STEP53_FINAL_EXTERNAL_INFERENCE_MANIFEST_SUMMARY.json"
)

LOCK = (
    COHORT
    / "OASIS3_STEP53_FINAL_EXTERNAL_INFERENCE_MANIFEST_LOCK.json"
)

KEYS = [
    "locked_subject_id",
    "landmark_mri_id",
    "landmark_day",
]

DAYS_PER_YEAR = 365.25


# ============================================================
# HELPERS
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


def parse_list(value):

    if isinstance(value, list):
        return value

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return []

    text = str(value).strip()

    if text == "":
        return []

    try:
        result = ast.literal_eval(text)

        if isinstance(
            result,
            (list, tuple, np.ndarray),
        ):
            return list(result)

    except Exception:
        pass

    # Fallback for pipe-separated MRI identifiers
    if "|" in text:
        return [
            x.strip()
            for x in text.split("|")
            if x.strip()
        ]

    return [text]


def bool_value(value):

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def require_columns(df, columns, name):

    missing = [
        c
        for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{name}: missing required columns: {missing}"
        )


# ============================================================
# LOAD LOCKED INPUTS
# ============================================================

print("=" * 76)
print(
    "STEP 53 — FINAL OASIS-3 "
    "EXTERNAL INFERENCE MANIFEST LOCK"
)
print("=" * 76)

for p in [
    STEP49,
    STEP51,
    STEP52,
]:
    if not p.exists():
        raise FileNotFoundError(p)


# ------------------------------------------------------------
# Record hashes BEFORE any in-memory standardization
# ------------------------------------------------------------

step49_hash = sha256_file(STEP49)
step51_hash = sha256_file(STEP51)
step52_hash = sha256_file(STEP52)


target = pd.read_csv(
    STEP49,
    low_memory=False,
)

temporal = pd.read_csv(
    STEP51,
    low_memory=False,
)

ready = pd.read_csv(
    STEP52,
    low_memory=False,
)


print("\nINPUT ROW COUNTS")
print("Step-49 targets   :", len(target))
print("Step-51 temporal  :", len(temporal))
print("Step-52 readiness :", len(ready))


# ============================================================
# STEP-49 SCHEMA STANDARDIZATION
# ============================================================
#
# Step 49 uses:
#
#   locked_subject_id
#   locked_mri_id
#   mri_day
#
# Steps 51 and 52 use:
#
#   locked_subject_id
#   landmark_mri_id
#   landmark_day
#
# These identify the same landmark.
#
# Standardization occurs only in the dataframe loaded in RAM.
# STEP49 on disk is NOT changed.
# ============================================================

print("\nSTEP-49 SCHEMA STANDARDIZATION")

if "landmark_mri_id" not in target.columns:

    if "locked_mri_id" not in target.columns:
        raise RuntimeError(
            "Step-49 contains neither "
            "'landmark_mri_id' nor 'locked_mri_id'."
        )

    target = target.rename(
        columns={
            "locked_mri_id": "landmark_mri_id"
        }
    )

    print(
        "locked_mri_id -> landmark_mri_id : APPLIED"
    )

else:
    print(
        "landmark_mri_id already present : YES"
    )


if "landmark_day" not in target.columns:

    if "mri_day" not in target.columns:
        raise RuntimeError(
            "Step-49 contains neither "
            "'landmark_day' nor 'mri_day'."
        )

    target = target.rename(
        columns={
            "mri_day": "landmark_day"
        }
    )

    print(
        "mri_day       -> landmark_day    : APPLIED"
    )

else:
    print(
        "landmark_day already present    : YES"
    )


print(
    "Original Step-49 file modified    : NO"
)


# ============================================================
# REQUIRED IDENTIFIERS
# ============================================================

require_columns(
    target,
    KEYS,
    "Step-49",
)

require_columns(
    temporal,
    KEYS,
    "Step-51",
)

require_columns(
    ready,
    KEYS,
    "Step-52",
)


# Standardize identity fields
for df in [
    target,
    temporal,
    ready,
]:

    df["locked_subject_id"] = (
        df["locked_subject_id"]
        .astype(str)
        .str.strip()
    )

    df["landmark_mri_id"] = (
        df["landmark_mri_id"]
        .astype(str)
        .str.strip()
    )

    df["landmark_day"] = pd.to_numeric(
        df["landmark_day"],
        errors="raise",
    )


# ============================================================
# UNIQUENESS CHECKS
# ============================================================

dup49 = int(
    target.duplicated(KEYS).sum()
)

dup51 = int(
    temporal.duplicated(KEYS).sum()
)

dup52 = int(
    ready.duplicated(KEYS).sum()
)


print("\nDUPLICATE LANDMARK KEYS")
print("Step-49 :", dup49)
print("Step-51 :", dup51)
print("Step-52 :", dup52)


if dup49 != 0:
    raise RuntimeError(
        "Step-49 contains duplicate "
        "inference landmark keys."
    )

if dup51 != 0:
    raise RuntimeError(
        "Step-51 contains duplicate "
        "inference landmark keys."
    )

if dup52 != 0:
    raise RuntimeError(
        "Step-52 contains duplicate "
        "inference landmark keys."
    )


# ============================================================
# EXACT KEY-SET AGREEMENT
# ============================================================

k49 = set(
    map(
        tuple,
        target[KEYS].to_numpy(),
    )
)

k51 = set(
    map(
        tuple,
        temporal[KEYS].to_numpy(),
    )
)

k52 = set(
    map(
        tuple,
        ready[KEYS].to_numpy(),
    )
)


missing_51_from_49 = (
    k49 - k51
)

extra_51_vs_49 = (
    k51 - k49
)

missing_52_from_51 = (
    k51 - k52
)

extra_52_vs_51 = (
    k52 - k51
)


print("\nKEY-SET AGREEMENT")
print(
    "Step49 missing from Step51 :",
    len(missing_51_from_49),
)
print(
    "Step51 extra vs Step49     :",
    len(extra_51_vs_49),
)
print(
    "Step51 missing from Step52 :",
    len(missing_52_from_51),
)
print(
    "Step52 extra vs Step51     :",
    len(extra_52_vs_51),
)


if (
    missing_51_from_49
    or extra_51_vs_49
):
    raise RuntimeError(
        "Step-49 and Step-51 landmark "
        "key sets are not identical."
    )


if (
    missing_52_from_51
    or extra_52_vs_51
):
    raise RuntimeError(
        "Step-51 and Step-52 landmark "
        "key sets are not identical."
    )


# ============================================================
# STEP 52 READINESS CONTRACT
# ============================================================

require_columns(
    ready,
    [
        "n_visits",
        "n_required_mri",
        "n_existing_mri",
        "n_ready_mri",
        "n_missing_mri",
        "n_not_ready_mri",
        "sequence_model_input_ready",
        "mri_ids",
    ],
    "Step-52",
)


ready["sequence_model_input_ready_bool"] = (
    ready["sequence_model_input_ready"]
    .map(bool_value)
)


not_ready_sequences = int(
    (
        ~ready[
            "sequence_model_input_ready_bool"
        ]
    ).sum()
)


missing_mri_total = int(
    pd.to_numeric(
        ready["n_missing_mri"],
        errors="raise",
    ).sum()
)


not_ready_mri_total = int(
    pd.to_numeric(
        ready["n_not_ready_mri"],
        errors="raise",
    ).sum()
)


print("\nMRI READINESS CONTRACT")
print(
    "Sequences not ready :",
    not_ready_sequences,
)
print(
    "Missing MRI total   :",
    missing_mri_total,
)
print(
    "Not-ready MRI total :",
    not_ready_mri_total,
)


if not_ready_sequences != 0:
    raise RuntimeError(
        "At least one sequence is "
        "not model-input ready."
    )

if missing_mri_total != 0:
    raise RuntimeError(
        "At least one required MRI is missing."
    )

if not_ready_mri_total != 0:
    raise RuntimeError(
        "At least one MRI failed readiness."
    )


# ============================================================
# STEP 51 TEMPORAL CONTRACT
# ============================================================

require_columns(
    temporal,
    [
        "n_visits",
        "mri_ids",
        "mri_days",
        "days_from_first_mri",
        "days_before_landmark",
        "years_from_first",
        "years_before_landmark",
        "future_mri_count",
        "sequence_ends_at_landmark",
        "step51_temporal_input_locked",
    ],
    "Step-51",
)


temporal_length_mismatch = 0
future_mri_leakage = 0
landmark_end_failure = 0
temporal_lock_failure = 0
temporal_formula_failure = 0
mri_id_mismatch_51_52 = 0


ready_lookup = ready.set_index(KEYS)


for _, row in temporal.iterrows():

    mri_ids = parse_list(
        row["mri_ids"]
    )

    mri_days = [
        float(x)
        for x in parse_list(
            row["mri_days"]
        )
    ]

    dff = [
        float(x)
        for x in parse_list(
            row["days_from_first_mri"]
        )
    ]

    dbl = [
        float(x)
        for x in parse_list(
            row["days_before_landmark"]
        )
    ]

    yff = [
        float(x)
        for x in parse_list(
            row["years_from_first"]
        )
    ]

    ybl = [
        float(x)
        for x in parse_list(
            row["years_before_landmark"]
        )
    ]


    n = int(
        row["n_visits"]
    )


    if not (
        len(mri_ids)
        == len(mri_days)
        == len(dff)
        == len(dbl)
        == len(yff)
        == len(ybl)
        == n
    ):
        temporal_length_mismatch += 1
        continue


    landmark_day = float(
        row["landmark_day"]
    )


    # --------------------------------------------------------
    # Future MRI leakage audit
    # --------------------------------------------------------

    if any(
        day > landmark_day + 1e-8
        for day in mri_days
    ):
        future_mri_leakage += 1


    if int(
        row["future_mri_count"]
    ) != 0:
        future_mri_leakage += 1


    # --------------------------------------------------------
    # Sequence must terminate at the landmark
    # --------------------------------------------------------

    if not bool_value(
        row["sequence_ends_at_landmark"]
    ):
        landmark_end_failure += 1


    # --------------------------------------------------------
    # Step-51 lock must be active
    # --------------------------------------------------------

    if not bool_value(
        row["step51_temporal_input_locked"]
    ):
        temporal_lock_failure += 1


    # --------------------------------------------------------
    # Exact frozen ADNI temporal convention
    #
    # days_from_first_mri
    #     = mri_day - first_mri_day
    #
    # days_before_landmark
    #     = landmark_day - mri_day
    #
    # years = days / 365.25
    # --------------------------------------------------------

    expected_dff = (
        np.asarray(
            mri_days,
            dtype=float,
        )
        - float(mri_days[0])
    )

    expected_dbl = (
        landmark_day
        - np.asarray(
            mri_days,
            dtype=float,
        )
    )


    if not np.allclose(
        np.asarray(
            dff,
            dtype=float,
        ),
        expected_dff,
        atol=1e-6,
        rtol=0,
    ):
        temporal_formula_failure += 1


    if not np.allclose(
        np.asarray(
            dbl,
            dtype=float,
        ),
        expected_dbl,
        atol=1e-6,
        rtol=0,
    ):
        temporal_formula_failure += 1


    if not np.allclose(
        np.asarray(
            yff,
            dtype=float,
        ),
        expected_dff / DAYS_PER_YEAR,
        atol=1e-6,
        rtol=0,
    ):
        temporal_formula_failure += 1


    if not np.allclose(
        np.asarray(
            ybl,
            dtype=float,
        ),
        expected_dbl / DAYS_PER_YEAR,
        atol=1e-6,
        rtol=0,
    ):
        temporal_formula_failure += 1


    # --------------------------------------------------------
    # Step 51 / Step 52 MRI identity agreement
    # --------------------------------------------------------

    key = (
        str(
            row["locked_subject_id"]
        ),
        str(
            row["landmark_mri_id"]
        ),
        float(
            row["landmark_day"]
        ),
    )


    rr = ready_lookup.loc[key]

    ready_ids = parse_list(
        rr["mri_ids"]
    )


    if (
        [str(x) for x in mri_ids]
        !=
        [str(x) for x in ready_ids]
    ):
        mri_id_mismatch_51_52 += 1


print("\nTEMPORAL CONTRACT")
print(
    "Length mismatches          :",
    temporal_length_mismatch,
)
print(
    "Future MRI leakage         :",
    future_mri_leakage,
)
print(
    "Sequence-end failures      :",
    landmark_end_failure,
)
print(
    "Temporal-lock failures     :",
    temporal_lock_failure,
)
print(
    "Temporal formula failures  :",
    temporal_formula_failure,
)
print(
    "Step51/52 MRI-ID mismatch  :",
    mri_id_mismatch_51_52,
)


if temporal_length_mismatch != 0:
    raise RuntimeError(
        "Temporal sequence length "
        "mismatch detected."
    )


if future_mri_leakage != 0:
    raise RuntimeError(
        "Future MRI leakage detected."
    )


if landmark_end_failure != 0:
    raise RuntimeError(
        "A sequence does not end "
        "at its landmark."
    )


if temporal_lock_failure != 0:
    raise RuntimeError(
        "An unlocked temporal "
        "sequence was detected."
    )


if temporal_formula_failure != 0:
    raise RuntimeError(
        "Frozen temporal formula "
        "contract failed."
    )


if mri_id_mismatch_51_52 != 0:
    raise RuntimeError(
        "Step-51 and Step-52 MRI "
        "sequence identities disagree."
    )


# ============================================================
# BUILD FINAL MANIFEST
# ============================================================
#
# Step 49 = outcome/target authority.
# Step 51 = temporal sequence authority.
# Step 52 = MRI-readiness authority.
# ============================================================

temporal_keep = [
    c
    for c in temporal.columns
    if c not in KEYS
]


ready_keep = [
    c
    for c in [
        "n_required_mri",
        "n_existing_mri",
        "n_ready_mri",
        "n_missing_mri",
        "n_not_ready_mri",
        "sequence_model_input_ready",
    ]
    if c in ready.columns
]


manifest = target.merge(
    temporal[
        KEYS + temporal_keep
    ],
    on=KEYS,
    how="inner",
    validate="one_to_one",
    suffixes=(
        "",
        "_step51",
    ),
)


manifest = manifest.merge(
    ready[
        KEYS + ready_keep
    ],
    on=KEYS,
    how="inner",
    validate="one_to_one",
)


# ============================================================
# FINAL STRICT AUDIT
# ============================================================

if len(manifest) != len(target):
    raise RuntimeError(
        f"Final manifest row count "
        f"{len(manifest)} does not equal "
        f"Step-49 row count {len(target)}."
    )


if manifest.duplicated(KEYS).any():
    raise RuntimeError(
        "Duplicate keys appeared "
        "after final merge."
    )


n_subjects = int(
    manifest[
        "locked_subject_id"
    ].nunique()
)


if "n_visits_step51" in manifest.columns:

    visit_col = "n_visits_step51"

elif "n_visits" in manifest.columns:

    visit_col = "n_visits"

else:

    visit_col = None


if visit_col is not None:

    visit_distribution = (
        pd.to_numeric(
            manifest[visit_col],
            errors="raise",
        )
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )

else:

    visit_distribution = {}


# ============================================================
# FINAL LANDMARK IDENTITY AUDIT
# ============================================================

subject_missing = int(
    manifest[
        "locked_subject_id"
    ].isna().sum()
)

mri_id_missing = int(
    manifest[
        "landmark_mri_id"
    ].isna().sum()
)

landmark_day_missing = int(
    manifest[
        "landmark_day"
    ].isna().sum()
)


print("\nFINAL LANDMARK IDENTITY AUDIT")
print(
    "Missing subject IDs :",
    subject_missing,
)
print(
    "Missing MRI IDs     :",
    mri_id_missing,
)
print(
    "Missing landmark day:",
    landmark_day_missing,
)


if (
    subject_missing != 0
    or mri_id_missing != 0
    or landmark_day_missing != 0
):
    raise RuntimeError(
        "Missing final landmark identity detected."
    )


# ============================================================
# SAVE FINAL LOCKED MANIFEST
# ============================================================

manifest = (
    manifest
    .sort_values(
        [
            "locked_subject_id",
            "landmark_day",
            "landmark_mri_id",
        ]
    )
    .reset_index(drop=True)
)


manifest.to_csv(
    OUT,
    index=False,
)


out_hash = sha256_file(
    OUT
)


summary = {

    "step": 53,

    "purpose":
        "Final locked OASIS-3 "
        "external inference manifest",

    "n_landmarks":
        int(len(manifest)),

    "n_participants":
        int(n_subjects),

    "duplicate_landmark_keys":
        0,

    "step49_step51_key_mismatch":
        0,

    "step51_step52_key_mismatch":
        0,

    "not_ready_sequences":
        int(not_ready_sequences),

    "missing_mri_total":
        int(missing_mri_total),

    "not_ready_mri_total":
        int(not_ready_mri_total),

    "temporal_length_mismatch":
        int(temporal_length_mismatch),

    "future_mri_leakage":
        int(future_mri_leakage),

    "landmark_end_failure":
        int(landmark_end_failure),

    "temporal_lock_failure":
        int(temporal_lock_failure),

    "temporal_formula_failure":
        int(temporal_formula_failure),

    "step51_step52_mri_id_mismatch":
        int(mri_id_mismatch_51_52),

    "missing_subject_ids":
        int(subject_missing),

    "missing_landmark_mri_ids":
        int(mri_id_missing),

    "missing_landmark_days":
        int(landmark_day_missing),

    "visit_distribution": {
        str(k): int(v)
        for k, v
        in visit_distribution.items()
    },

    "days_per_year":
        DAYS_PER_YEAR,

    "step49_schema_standardization": {
        "locked_mri_id":
            "landmark_mri_id",
        "mri_day":
            "landmark_day",
        "source_file_modified":
            False,
    },

    "model_inference_performed":
        False,

    "prediction_files_read":
        False,

    "mri_files_modified":
        False,
}


with open(
    SUMMARY,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


lock = {

    "step":
        53,

    "status":
        "LOCKED",

    "step49_sha256":
        step49_hash,

    "step51_sha256":
        step51_hash,

    "step52_sha256":
        step52_hash,

    "step53_manifest_sha256":
        out_hash,

    "join_keys":
        KEYS,

    "step49_original_key_names": [
        "locked_subject_id",
        "locked_mri_id",
        "mri_day",
    ],

    "step49_in_memory_standardization": {
        "locked_mri_id":
            "landmark_mri_id",
        "mri_day":
            "landmark_day",
    },

    "step49_source_modified":
        False,

    "n_landmarks":
        int(len(manifest)),

    "n_participants":
        int(n_subjects),

    "frozen_temporal_days_per_year":
        DAYS_PER_YEAR,

    "external_model_inference_allowed_after_this_lock":
        True,

    "external_model_inference_performed_in_step53":
        False,
}


with open(
    LOCK,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        lock,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

print(
    "\nFINAL EXTERNAL INFERENCE POPULATION"
)
print(
    "Landmarks    :",
    len(manifest),
)
print(
    "Participants :",
    n_subjects,
)


if visit_distribution:

    print(
        "\nVISITS PER LANDMARK"
    )

    for k, v in visit_distribution.items():

        print(
            f"{k:>2} visits : {v}"
        )


print(
    "\nFINAL SAFETY AUDIT"
)

print(
    "Duplicate keys             : 0"
)
print(
    "Missing targets            : 0"
)
print(
    "Missing temporal sequences : 0"
)
print(
    "Sequences not MRI-ready    : 0"
)
print(
    "Future MRI leakage         : 0"
)
print(
    "Temporal length mismatch   : 0"
)
print(
    "Temporal formula failures  : 0"
)
print(
    "MRI identity mismatch      : 0"
)
print(
    "Missing landmark IDs       : 0"
)


print(
    "\nSTEP-49 SOURCE PROTECTION"
)
print(
    "Original Step-49 SHA256 :",
    step49_hash,
)
print(
    "Original file modified  : NO"
)


print(
    "\nOUTPUT:"
)
print(
    OUT
)


print(
    "\nSUMMARY:"
)
print(
    SUMMARY
)


print(
    "\nLOCK:"
)
print(
    LOCK
)


print(
    "\nSHA256"
)
print(
    "Step-49 :",
    step49_hash,
)
print(
    "Step-51 :",
    step51_hash,
)
print(
    "Step-52 :",
    step52_hash,
)
print(
    "Step-53 :",
    out_hash,
)


print(
    "\nIMPORTANT:"
)
print(
    "No MRI files were modified."
)
print(
    "No prediction files were read."
)
print(
    "No frozen ADNI checkpoint was loaded."
)
print(
    "No frozen ADNI model inference was performed."
)
print(
    "No external performance metric was examined."
)
print(
    "Step-49 was standardized only in memory."
)
print(
    "The original Step-49 locked file was not modified."
)
print(
    "The exact external inference population is now locked."
)


print(
    "\n" + "=" * 76
)

print(
    "STEP 53 PASSED — FINAL EXTERNAL "
    "INFERENCE MANIFEST LOCKED"
)

print(
    "=" * 76
)