# ============================================================
# STEP 52
# OASIS-3 EXTERNAL MODEL-INPUT MRI READINESS AUDIT
#
# PURPOSE
# -------
# Read-only audit linking the locked Step-51 longitudinal
# sequences to the preprocessed OASIS-3 MNI-normalized MRI files.
#
# THIS SCRIPT DOES NOT:
#   - read external outcome data
#   - read prediction files
#   - load any frozen ADNI model checkpoint
#   - perform model inference
#   - modify any MRI file
#   - select subjects based on model performance
#
# EXPECTED PREPROCESSED FILE:
#
# E:\OASIS3_EXTERNAL_VALIDATION\03_preprocessed\
# external_available\<MRI_ID>\05_normalized\
# <MRI_ID>_MNI_NORMALIZED.nii.gz
#
# Frozen ADNI model input crop:
#   source MNI grid: 197 x 233 x 189
#   crop start:     (18, 20, 14)
#   crop end:       (178, 212, 174)
#   crop shape:     160 x 192 x 160
# ============================================================

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    raise RuntimeError(
        "\nERROR: nibabel is required for Step 52.\n"
        "Install with:\n"
        "    pip install nibabel\n"
    )


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

COHORT_DIR = PROJECT_ROOT / "cohort_construction"

STEP51_FILE = (
    COHORT_DIR
    / "OASIS3_STEP51_FROZEN_TEMPORAL_INPUTS_LOCKED.csv"
)

PREPROC_ROOT = Path(
    r"E:\OASIS3_EXTERNAL_VALIDATION"
    r"\03_preprocessed\external_available"
)

OUTPUT_CSV = (
    COHORT_DIR
    / "OASIS3_STEP52_MODEL_INPUT_READINESS_AUDIT.csv"
)

READY_CSV = (
    COHORT_DIR
    / "OASIS3_STEP52_MODEL_INPUT_READY_MRI.csv"
)

MISSING_CSV = (
    COHORT_DIR
    / "OASIS3_STEP52_MODEL_INPUT_MISSING_MRI.csv"
)

SEQUENCE_CSV = (
    COHORT_DIR
    / "OASIS3_STEP52_SEQUENCE_READINESS.csv"
)

SUMMARY_JSON = (
    COHORT_DIR
    / "OASIS3_STEP52_MODEL_INPUT_READINESS_SUMMARY.json"
)

LOCK_JSON = (
    COHORT_DIR
    / "OASIS3_STEP52_MODEL_INPUT_READINESS_LOCK.json"
)


# ============================================================
# 2. FROZEN MRI GEOMETRY CONTRACT
# ============================================================

EXPECTED_MNI_SHAPE = (197, 233, 189)

CROP_START = (18, 20, 14)
CROP_END = (178, 212, 174)

EXPECTED_CROP_SHAPE = (
    CROP_END[0] - CROP_START[0],
    CROP_END[1] - CROP_START[1],
    CROP_END[2] - CROP_START[2],
)

assert EXPECTED_CROP_SHAPE == (160, 192, 160)

EXPECTED_SPACING = (1.0, 1.0, 1.0)

SPACING_TOLERANCE = 1e-3


# ============================================================
# 3. HELPERS
# ============================================================

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def safe_int(value):
    try:
        return int(float(value))
    except Exception:
        return None


def parse_mri_ids(value):
    """
    Step-51 mri_ids are pipe-separated, e.g.

    OAS30001_MR_d0129|OAS30001_MR_d0757
    """

    if pd.isna(value):
        return []

    text = str(value).strip()

    if text == "":
        return []

    return [
        x.strip()
        for x in text.split("|")
        if x.strip() != ""
    ]


def expected_mri_path(mri_id):
    return (
        PREPROC_ROOT
        / mri_id
        / "05_normalized"
        / f"{mri_id}_MNI_NORMALIZED.nii.gz"
    )


def inspect_mri(mri_id):
    """
    Read-only inspection of one normalized MRI.
    """

    path = expected_mri_path(mri_id)

    rec = {
        "locked_mri_id": mri_id,
        "expected_path": str(path),
        "file_exists": path.is_file(),
        "load_success": False,

        "dim_x": np.nan,
        "dim_y": np.nan,
        "dim_z": np.nan,

        "spacing_x": np.nan,
        "spacing_y": np.nan,
        "spacing_z": np.nan,

        "expected_mni_shape": False,
        "expected_spacing": False,
        "crop_bounds_valid": False,
        "crop_shape_valid": False,

        "finite_data": False,
        "nonzero_voxels": np.nan,

        "data_min": np.nan,
        "data_max": np.nan,
        "data_mean": np.nan,
        "data_std": np.nan,

        "ready_for_frozen_input": False,
        "failure_reason": "",
    }

    if not path.is_file():
        rec["failure_reason"] = "MISSING_NORMALIZED_MRI"
        return rec

    try:
        img = nib.load(str(path))

        shape = tuple(int(x) for x in img.shape[:3])

        if len(shape) != 3:
            rec["failure_reason"] = (
                f"NON_3D_IMAGE_SHAPE_{shape}"
            )
            return rec

        rec["dim_x"] = shape[0]
        rec["dim_y"] = shape[1]
        rec["dim_z"] = shape[2]

        zooms = tuple(
            float(x)
            for x in img.header.get_zooms()[:3]
        )

        rec["spacing_x"] = zooms[0]
        rec["spacing_y"] = zooms[1]
        rec["spacing_z"] = zooms[2]

        rec["expected_mni_shape"] = (
            shape == EXPECTED_MNI_SHAPE
        )

        rec["expected_spacing"] = bool(
            np.allclose(
                np.asarray(zooms, dtype=float),
                np.asarray(EXPECTED_SPACING, dtype=float),
                atol=SPACING_TOLERANCE,
                rtol=0.0,
            )
        )

        crop_bounds_valid = (
            CROP_START[0] >= 0
            and CROP_START[1] >= 0
            and CROP_START[2] >= 0
            and CROP_END[0] <= shape[0]
            and CROP_END[1] <= shape[1]
            and CROP_END[2] <= shape[2]
        )

        rec["crop_bounds_valid"] = bool(
            crop_bounds_valid
        )

        # Load MRI only for numerical QC.
        # Nothing is written back to disk.
        data = np.asarray(
            img.dataobj,
            dtype=np.float32
        )

        rec["load_success"] = True

        finite = bool(
            np.isfinite(data).all()
        )

        rec["finite_data"] = finite

        if data.size > 0:

            finite_values = data[
                np.isfinite(data)
            ]

            if finite_values.size > 0:
                rec["data_min"] = float(
                    np.min(finite_values)
                )

                rec["data_max"] = float(
                    np.max(finite_values)
                )

                rec["data_mean"] = float(
                    np.mean(finite_values)
                )

                rec["data_std"] = float(
                    np.std(finite_values)
                )

                rec["nonzero_voxels"] = int(
                    np.count_nonzero(
                        finite_values
                    )
                )

        crop_shape_valid = False

        if crop_bounds_valid:

            crop = data[
                CROP_START[0]:CROP_END[0],
                CROP_START[1]:CROP_END[1],
                CROP_START[2]:CROP_END[2],
            ]

            crop_shape_valid = (
                tuple(crop.shape)
                == EXPECTED_CROP_SHAPE
            )

        rec["crop_shape_valid"] = bool(
            crop_shape_valid
        )

        failures = []

        if not rec["expected_mni_shape"]:
            failures.append(
                "WRONG_MNI_SHAPE"
            )

        if not rec["expected_spacing"]:
            failures.append(
                "WRONG_VOXEL_SPACING"
            )

        if not rec["crop_bounds_valid"]:
            failures.append(
                "INVALID_CROP_BOUNDS"
            )

        if not rec["crop_shape_valid"]:
            failures.append(
                "WRONG_CROP_SHAPE"
            )

        if not rec["finite_data"]:
            failures.append(
                "NONFINITE_DATA"
            )

        if (
            not pd.isna(rec["nonzero_voxels"])
            and rec["nonzero_voxels"] == 0
        ):
            failures.append(
                "ALL_ZERO_IMAGE"
            )

        rec["ready_for_frozen_input"] = (
            len(failures) == 0
        )

        rec["failure_reason"] = (
            "PASS"
            if len(failures) == 0
            else "|".join(failures)
        )

        return rec

    except Exception as e:

        rec["failure_reason"] = (
            "LOAD_ERROR: "
            + str(e).replace("\n", " ")[:300]
        )

        return rec


# ============================================================
# 4. START
# ============================================================

print("=" * 72)
print("STEP 52 - OASIS-3 MODEL-INPUT MRI READINESS AUDIT")
print("=" * 72)

print("\nPROJECT ROOT:")
print(PROJECT_ROOT)

print("\nSTEP-51 INPUT:")
print(STEP51_FILE)

print("\nPREPROCESSED ROOT:")
print(PREPROC_ROOT)


# ============================================================
# 5. VERIFY INPUTS
# ============================================================

if not STEP51_FILE.is_file():
    raise FileNotFoundError(
        f"\nMissing Step-51 input:\n{STEP51_FILE}"
    )

if not PREPROC_ROOT.is_dir():
    raise FileNotFoundError(
        f"\nMissing preprocessing root:\n{PREPROC_ROOT}"
    )

step51_sha = sha256_file(STEP51_FILE)

print("\nSTEP-51 SHA256:")
print(step51_sha)


# ============================================================
# 6. READ STEP 51
# ============================================================

df = pd.read_csv(
    STEP51_FILE,
    low_memory=False
)

print("\nSTEP-51 ROWS:", len(df))
print("STEP-51 COLUMNS:", len(df.columns))


required = [
    "locked_subject_id",
    "landmark_mri_id",
    "landmark_day",
    "n_visits",
    "mri_ids",
]

missing_columns = [
    c for c in required
    if c not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "\nSTEP-51 required columns missing:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 7. BASIC STEP-51 INTEGRITY
# ============================================================

if df["landmark_mri_id"].duplicated().any():
    dup = df.loc[
        df["landmark_mri_id"].duplicated(
            keep=False
        ),
        [
            "locked_subject_id",
            "landmark_mri_id",
            "landmark_day",
        ],
    ]

    raise RuntimeError(
        "\nDuplicate landmark MRI IDs found:\n"
        + dup.head(20).to_string(index=False)
    )


# ============================================================
# 8. EXTRACT ALL REQUIRED MRI IDS
# ============================================================

sequence_mri_ids = []

sequence_length_errors = []

for idx, row in df.iterrows():

    ids = parse_mri_ids(
        row["mri_ids"]
    )

    declared_n = safe_int(
        row["n_visits"]
    )

    if declared_n is None:
        sequence_length_errors.append(
            {
                "row": int(idx),
                "locked_subject_id":
                    row["locked_subject_id"],
                "landmark_mri_id":
                    row["landmark_mri_id"],
                "declared_n_visits":
                    row["n_visits"],
                "parsed_n_visits":
                    len(ids),
                "reason":
                    "INVALID_DECLARED_N_VISITS",
            }
        )

    elif declared_n != len(ids):
        sequence_length_errors.append(
            {
                "row": int(idx),
                "locked_subject_id":
                    row["locked_subject_id"],
                "landmark_mri_id":
                    row["landmark_mri_id"],
                "declared_n_visits":
                    declared_n,
                "parsed_n_visits":
                    len(ids),
                "reason":
                    "SEQUENCE_LENGTH_MISMATCH",
            }
        )

    for position, mri_id in enumerate(
        ids,
        start=1
    ):
        sequence_mri_ids.append(
            {
                "step51_row": int(idx),
                "locked_subject_id":
                    str(row["locked_subject_id"]),
                "landmark_mri_id":
                    str(row["landmark_mri_id"]),
                "landmark_day":
                    row["landmark_day"],
                "visit_position":
                    int(position),
                "locked_mri_id":
                    str(mri_id),
            }
        )


if sequence_length_errors:
    err_df = pd.DataFrame(
        sequence_length_errors
    )

    raise RuntimeError(
        "\nSTEP-51 sequence length integrity failure:\n"
        + err_df.head(30).to_string(
            index=False
        )
    )


expanded = pd.DataFrame(
    sequence_mri_ids
)

if len(expanded) == 0:
    raise RuntimeError(
        "\nNo MRI IDs parsed from Step 51."
    )


unique_ids = sorted(
    expanded[
        "locked_mri_id"
    ].dropna().astype(str).unique()
)


print("\nSEQUENCE CONTRACT")
print(
    "Locked sequences       :",
    len(df)
)

print(
    "Expanded MRI visits    :",
    len(expanded)
)

print(
    "Unique MRI acquisitions:",
    len(unique_ids)
)


# ============================================================
# 9. AUDIT EACH UNIQUE MRI
# ============================================================

print("\nAUDITING NORMALIZED MRI FILES...")
print(
    "Expected source geometry:",
    EXPECTED_MNI_SHAPE
)

print(
    "Frozen crop shape       :",
    EXPECTED_CROP_SHAPE
)

records = []

total = len(unique_ids)

for i, mri_id in enumerate(
    unique_ids,
    start=1
):

    if (
        i == 1
        or i % 50 == 0
        or i == total
    ):
        print(
            f"  [{i:4d}/{total:4d}] "
            f"{mri_id}"
        )

    records.append(
        inspect_mri(mri_id)
    )


audit = pd.DataFrame(records)


# ============================================================
# 10. CHECK FILE-PATH UNIQUENESS
# ============================================================

existing = audit[
    audit["file_exists"]
].copy()

duplicate_paths = (
    existing["expected_path"]
    .duplicated(keep=False)
)

n_duplicate_paths = int(
    duplicate_paths.sum()
)

if n_duplicate_paths > 0:

    print(
        "\nWARNING: duplicate normalized paths detected."
    )

    print(
        existing.loc[
            duplicate_paths,
            [
                "locked_mri_id",
                "expected_path",
            ],
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 11. BUILD SEQUENCE-LEVEL READINESS
# ============================================================

ready_map = dict(
    zip(
        audit["locked_mri_id"],
        audit["ready_for_frozen_input"],
    )
)

exists_map = dict(
    zip(
        audit["locked_mri_id"],
        audit["file_exists"],
    )
)

sequence_records = []

for idx, row in df.iterrows():

    ids = parse_mri_ids(
        row["mri_ids"]
    )

    statuses = [
        bool(
            ready_map.get(
                mri_id,
                False
            )
        )
        for mri_id in ids
    ]

    exists = [
        bool(
            exists_map.get(
                mri_id,
                False
            )
        )
        for mri_id in ids
    ]

    n_required = len(ids)

    n_existing = int(
        sum(exists)
    )

    n_ready = int(
        sum(statuses)
    )

    sequence_records.append(
        {
            "locked_subject_id":
                row["locked_subject_id"],

            "landmark_mri_id":
                row["landmark_mri_id"],

            "landmark_day":
                row["landmark_day"],

            "n_visits":
                row["n_visits"],

            "n_required_mri":
                n_required,

            "n_existing_mri":
                n_existing,

            "n_ready_mri":
                n_ready,

            "n_missing_mri":
                n_required - n_existing,

            "n_not_ready_mri":
                n_required - n_ready,

            "sequence_model_input_ready":
                bool(
                    n_required > 0
                    and n_ready == n_required
                ),

            "mri_ids":
                row["mri_ids"],
        }
    )


sequence_ready = pd.DataFrame(
    sequence_records
)


# ============================================================
# 12. SPLIT READY / MISSING
# ============================================================

ready = audit[
    audit["ready_for_frozen_input"]
].copy()

not_ready = audit[
    ~audit["ready_for_frozen_input"]
].copy()


# ============================================================
# 13. SUMMARY COUNTS
# ============================================================

n_unique = len(audit)

n_exists = int(
    audit["file_exists"].sum()
)

n_missing = int(
    (~audit["file_exists"]).sum()
)

n_loaded = int(
    audit["load_success"].sum()
)

n_correct_shape = int(
    audit["expected_mni_shape"].sum()
)

n_correct_spacing = int(
    audit["expected_spacing"].sum()
)

n_crop_valid = int(
    audit["crop_shape_valid"].sum()
)

n_finite = int(
    audit["finite_data"].sum()
)

n_ready = int(
    audit[
        "ready_for_frozen_input"
    ].sum()
)

n_not_ready = int(
    n_unique - n_ready
)

n_sequences = len(sequence_ready)

n_sequences_ready = int(
    sequence_ready[
        "sequence_model_input_ready"
    ].sum()
)

n_sequences_not_ready = int(
    n_sequences - n_sequences_ready
)


# ============================================================
# 14. PRINT AUDIT
# ============================================================

print("\n" + "=" * 72)
print("MODEL-INPUT READINESS RESULTS")
print("=" * 72)

print(
    "Unique required MRI     :",
    n_unique
)

print(
    "Normalized MRI exists   :",
    n_exists
)

print(
    "Missing normalized MRI  :",
    n_missing
)

print(
    "Successfully loaded     :",
    n_loaded
)

print(
    "Correct MNI shape       :",
    n_correct_shape
)

print(
    "Correct 1-mm spacing    :",
    n_correct_spacing
)

print(
    "Valid frozen crop       :",
    n_crop_valid
)

print(
    "Finite MRI data         :",
    n_finite
)

print(
    "MRI ready for M0 input  :",
    n_ready
)

print(
    "MRI not ready           :",
    n_not_ready
)

print("\nSEQUENCE READINESS")

print(
    "Locked sequences        :",
    n_sequences
)

print(
    "Sequences fully ready   :",
    n_sequences_ready
)

print(
    "Sequences not ready     :",
    n_sequences_not_ready
)


# ============================================================
# 15. FAILURE REASONS
# ============================================================

if n_not_ready > 0:

    print("\nMRI FAILURE REASONS")

    counts = (
        not_ready[
            "failure_reason"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        counts.to_string()
    )

    print("\nFIRST 20 NOT-READY MRI")

    print(
        not_ready[
            [
                "locked_mri_id",
                "file_exists",
                "dim_x",
                "dim_y",
                "dim_z",
                "spacing_x",
                "spacing_y",
                "spacing_z",
                "failure_reason",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 16. SAVE OUTPUTS
# ============================================================

COHORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

audit.to_csv(
    OUTPUT_CSV,
    index=False
)

ready.to_csv(
    READY_CSV,
    index=False
)

not_ready.to_csv(
    MISSING_CSV,
    index=False
)

sequence_ready.to_csv(
    SEQUENCE_CSV,
    index=False
)


# ============================================================
# 17. HASH OUTPUTS
# ============================================================

audit_sha = sha256_file(
    OUTPUT_CSV
)

ready_sha = sha256_file(
    READY_CSV
)

not_ready_sha = sha256_file(
    MISSING_CSV
)

sequence_sha = sha256_file(
    SEQUENCE_CSV
)


# ============================================================
# 18. SUMMARY
# ============================================================

summary = {
    "step": 52,

    "description":
        "OASIS-3 external model-input MRI readiness audit",

    "step51_input":
        str(STEP51_FILE),

    "step51_sha256":
        step51_sha,

    "preprocessed_root":
        str(PREPROC_ROOT),

    "normalized_file_contract":
        (
            "<MRI_ID>/05_normalized/"
            "<MRI_ID>_MNI_NORMALIZED.nii.gz"
        ),

    "expected_mni_shape":
        list(EXPECTED_MNI_SHAPE),

    "expected_spacing_mm":
        list(EXPECTED_SPACING),

    "crop_start":
        list(CROP_START),

    "crop_end":
        list(CROP_END),

    "expected_crop_shape":
        list(EXPECTED_CROP_SHAPE),

    "locked_sequences":
        int(n_sequences),

    "expanded_sequence_visits":
        int(len(expanded)),

    "unique_required_mri":
        int(n_unique),

    "normalized_mri_exists":
        int(n_exists),

    "missing_normalized_mri":
        int(n_missing),

    "successfully_loaded":
        int(n_loaded),

    "correct_mni_shape":
        int(n_correct_shape),

    "correct_spacing":
        int(n_correct_spacing),

    "valid_frozen_crop":
        int(n_crop_valid),

    "finite_data":
        int(n_finite),

    "mri_ready_for_frozen_input":
        int(n_ready),

    "mri_not_ready":
        int(n_not_ready),

    "sequences_fully_ready":
        int(n_sequences_ready),

    "sequences_not_ready":
        int(n_sequences_not_ready),

    "duplicate_expected_paths":
        int(n_duplicate_paths),

    "output_sha256":
        audit_sha,

    "ready_sha256":
        ready_sha,

    "not_ready_sha256":
        not_ready_sha,

    "sequence_readiness_sha256":
        sequence_sha,

    "model_checkpoint_loaded":
        False,

    "model_inference_performed":
        False,

    "prediction_files_read":
        False,

    "external_outcome_data_read":
        False,

    "mri_files_modified":
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
# 19. LOCK
# ============================================================

lock = {
    "step": 52,

    "status":
        "LOCKED_READ_ONLY_MODEL_INPUT_READINESS_AUDIT",

    "step51_sha256":
        step51_sha,

    "audit_sha256":
        audit_sha,

    "ready_sha256":
        ready_sha,

    "not_ready_sha256":
        not_ready_sha,

    "sequence_readiness_sha256":
        sequence_sha,

    "expected_mni_shape":
        list(EXPECTED_MNI_SHAPE),

    "expected_spacing_mm":
        list(EXPECTED_SPACING),

    "crop_start":
        list(CROP_START),

    "crop_end":
        list(CROP_END),

    "expected_crop_shape":
        list(EXPECTED_CROP_SHAPE),

    "normalized_filename_suffix":
        "_MNI_NORMALIZED.nii.gz",

    "external_outcome_data_used":
        False,

    "external_prediction_data_used":
        False,

    "frozen_model_loaded":
        False,

    "frozen_model_inference":
        False,

    "mri_modified":
        False,
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
# 20. OUTPUT PATHS
# ============================================================

print("\nOUTPUT:")
print(OUTPUT_CSV)

print("\nREADY MRI:")
print(READY_CSV)

print("\nNOT READY / MISSING MRI:")
print(MISSING_CSV)

print("\nSEQUENCE READINESS:")
print(SEQUENCE_CSV)

print("\nSUMMARY:")
print(SUMMARY_JSON)

print("\nLOCK:")
print(LOCK_JSON)

print("\nSHA256")
print(
    "Step-51 input      :",
    step51_sha
)

print(
    "Step-52 audit      :",
    audit_sha
)

print(
    "Ready MRI          :",
    ready_sha
)

print(
    "Not-ready MRI      :",
    not_ready_sha
)

print(
    "Sequence readiness :",
    sequence_sha
)


# ============================================================
# 21. FINAL SAFETY MESSAGE
# ============================================================

print("\nIMPORTANT:")
print(
    "No external outcome data were read."
)
print(
    "No prediction files were read."
)
print(
    "No frozen ADNI model checkpoint was loaded."
)
print(
    "No frozen ADNI model inference was performed."
)
print(
    "No MRI files were modified."
)
print(
    "Only *_MNI_NORMALIZED.nii.gz files were audited."
)
print(
    "The frozen ADNI central crop contract was checked."
)


# ============================================================
# 22. FINAL STATUS
# ============================================================

print("\n" + "=" * 72)

if n_not_ready == 0 and n_sequences_not_ready == 0:

    print(
        "STEP 52 PASSED - ALL LOCKED EXTERNAL MRI INPUTS READY"
    )

else:

    print(
        "STEP 52 AUDIT COMPLETE - "
        "MODEL INPUTS NOT YET FULLY READY"
    )

    print(
        f"Missing/not-ready MRI: {n_not_ready}"
    )

    print(
        f"Not-ready sequences: {n_sequences_not_ready}"
    )

print("=" * 72)