from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import hashlib
import json
from datetime import datetime

# ============================================================
# STEP 44
# OASIS-3 FROZEN EXTERNAL PREPROCESSING INTEGRITY LOCK
#
# READ-ONLY AUDIT:
#   - does NOT modify MRI files
#   - does NOT use outcomes
#   - does NOT run the frozen ADNI model
# ============================================================

ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

STATUS_FILE = (
    ROOT
    / "cohort_construction"
    / "OASIS3_STEP43_PREPROCESSING_STATUS.csv"
)

OUT_DIR = ROOT / "cohort_construction"

AUDIT_CSV = (
    OUT_DIR
    / "OASIS3_STEP44_PREPROCESSING_INTEGRITY_AUDIT.csv"
)

SUMMARY_JSON = (
    OUT_DIR
    / "OASIS3_STEP44_PREPROCESSING_INTEGRITY_SUMMARY.json"
)

LOCK_JSON = (
    OUT_DIR
    / "OASIS3_STEP44_PREPROCESSING_INTEGRITY_LOCK.json"
)

# Exact frozen ADNI reference
MNI_FILE = Path(
    str(Path(__file__).resolve().parents[2])
) / "03_preprocessed" / "templates" / "MNI152_T1_1mm.nii.gz"


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path, block_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def close_tuple(a, b, atol=1e-4):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return (
        a.shape == b.shape
        and np.allclose(a, b, atol=atol, rtol=0)
    )


# ============================================================
# START
# ============================================================

print("=" * 78)
print("STEP 44 — FROZEN EXTERNAL PREPROCESSING INTEGRITY AUDIT")
print("=" * 78)

assert STATUS_FILE.exists(), (
    f"Step-43 status file missing:\n{STATUS_FILE}"
)

assert MNI_FILE.exists(), (
    f"Frozen ADNI MNI reference missing:\n{MNI_FILE}"
)

print("\nStep-43 status:")
print(STATUS_FILE)

print("\nFrozen MNI reference:")
print(MNI_FILE)


# ============================================================
# LOAD STEP 43
# ============================================================

df = pd.read_csv(
    STATUS_FILE,
    low_memory=False
)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

assert len(df) == 1901, (
    f"Expected 1901 Step-43 rows, found {len(df)}"
)

required_columns = [
    "locked_subject_id",
    "locked_mri_id",
    "step43_status",
    "step43_final_nifti",
    "step43_qc_csv",
    "step43_summary_json",
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

assert not missing_columns, (
    "Missing required Step-43 columns:\n"
    + "\n".join(missing_columns)
)


# ============================================================
# VERIFY STEP 43 STATUS
# ============================================================

allowed = {
    "SUCCESS",
    "ALREADY_COMPLETE",
}

bad_status = df[
    ~df["step43_status"].astype(str).isin(allowed)
]

assert len(bad_status) == 0, (
    f"Found {len(bad_status)} non-complete Step-43 rows."
)

assert df["locked_mri_id"].astype(str).nunique() == 1901, (
    "locked_mri_id is not unique."
)

assert df["locked_subject_id"].astype(str).nunique() == 605, (
    "Expected 605 participants."
)

print("\nStep-43 completion status: PASS")
print("Participants:", df["locked_subject_id"].nunique())
print("MRI acquisitions:", df["locked_mri_id"].nunique())


# ============================================================
# LOAD FROZEN MNI GEOMETRY
# ============================================================

mni = nib.load(str(MNI_FILE))

mni_shape = tuple(int(x) for x in mni.shape[:3])

mni_zooms = tuple(
    float(x)
    for x in mni.header.get_zooms()[:3]
)

mni_affine = np.asarray(
    mni.affine,
    dtype=float
)

print("\nFrozen MNI geometry")
print("Shape :", mni_shape)
print("Zooms :", mni_zooms)


# ============================================================
# AUDIT ALL 1901 FINAL MRIs
# ============================================================

records = []

print("\nAuditing 1901 frozen preprocessing outputs...\n")

for i, row in df.iterrows():

    subject_id = str(row["locked_subject_id"])
    image_id = str(row["locked_mri_id"])

    final_file = Path(
        str(row["step43_final_nifti"])
    )

    qc_csv = Path(
        str(row["step43_qc_csv"])
    )

    summary_json = Path(
        str(row["step43_summary_json"])
    )

    final_exists = final_file.exists()
    qc_exists = qc_csv.exists()
    summary_exists = summary_json.exists()

    readable = False
    shape_match = False
    zoom_match = False
    affine_match = False
    finite_pass = False
    nonempty_pass = False

    shape = None
    zooms = None
    finite_fraction = np.nan
    nonzero_voxels = np.nan
    data_min = np.nan
    data_max = np.nan
    data_mean = np.nan
    data_std = np.nan

    error = ""

    if final_exists:

        try:
            img = nib.load(str(final_file))

            shape = tuple(
                int(x)
                for x in img.shape[:3]
            )

            zooms = tuple(
                float(x)
                for x in img.header.get_zooms()[:3]
            )

            affine = np.asarray(
                img.affine,
                dtype=float
            )

            readable = True

            shape_match = (
                shape == mni_shape
            )

            zoom_match = close_tuple(
                zooms,
                mni_zooms,
                atol=1e-5,
            )

            affine_match = np.allclose(
                affine,
                mni_affine,
                atol=1e-4,
                rtol=0,
            )

            data = np.asarray(
                img.dataobj,
                dtype=np.float32
            )

            finite = np.isfinite(data)

            finite_fraction = float(
                finite.mean()
            )

            finite_pass = bool(
                finite.all()
            )

            if finite.any():

                finite_data = data[finite]

                data_min = float(
                    finite_data.min()
                )

                data_max = float(
                    finite_data.max()
                )

                data_mean = float(
                    finite_data.mean()
                )

                data_std = float(
                    finite_data.std()
                )

                nonzero_voxels = int(
                    np.count_nonzero(finite_data)
                )

                nonempty_pass = (
                    nonzero_voxels > 0
                    and data_std > 0
                )

        except Exception as exc:
            error = repr(exc)

    integrity_pass = all([
        final_exists,
        qc_exists,
        summary_exists,
        readable,
        shape_match,
        zoom_match,
        affine_match,
        finite_pass,
        nonempty_pass,
    ])

    records.append({
        "audit_row": i + 1,
        "locked_subject_id": subject_id,
        "locked_mri_id": image_id,
        "step43_status": str(row["step43_status"]),

        "final_nifti": str(final_file),
        "final_exists": final_exists,

        "qc_csv": str(qc_csv),
        "qc_csv_exists": qc_exists,

        "summary_json": str(summary_json),
        "summary_json_exists": summary_exists,

        "nifti_readable": readable,

        "shape": str(shape),
        "expected_shape": str(mni_shape),
        "shape_match": shape_match,

        "zooms": str(zooms),
        "expected_zooms": str(mni_zooms),
        "zoom_match": zoom_match,

        "affine_match": affine_match,

        "finite_fraction": finite_fraction,
        "finite_pass": finite_pass,

        "nonzero_voxels": nonzero_voxels,
        "nonempty_pass": nonempty_pass,

        "data_min": data_min,
        "data_max": data_max,
        "data_mean": data_mean,
        "data_std": data_std,

        "integrity_pass": integrity_pass,
        "error": error,
    })

    if (
        (i + 1) % 100 == 0
        or (i + 1) == len(df)
    ):
        print(
            f"Checked {i + 1:4d} / {len(df)}"
        )


# ============================================================
# BUILD AUDIT TABLE
# ============================================================

audit = pd.DataFrame(records)

audit.to_csv(
    AUDIT_CSV,
    index=False
)

passed = int(
    audit["integrity_pass"].sum()
)

failed = int(
    (~audit["integrity_pass"]).sum()
)

print("\n" + "=" * 78)
print("STEP 44 AUDIT RESULTS")
print("=" * 78)

print("Total acquisitions :", len(audit))
print("Integrity PASS     :", passed)
print("Integrity FAIL     :", failed)

print("\nComponent checks:")

checks = [
    "final_exists",
    "qc_csv_exists",
    "summary_json_exists",
    "nifti_readable",
    "shape_match",
    "zoom_match",
    "affine_match",
    "finite_pass",
    "nonempty_pass",
]

for col in checks:
    n_pass = int(
        audit[col].sum()
    )

    print(
        f"{col:22s}: "
        f"{n_pass} / {len(audit)}"
    )


# ============================================================
# STRICT FINAL ASSERTIONS
# ============================================================

assert len(audit) == 1901

assert passed == 1901, (
    f"Step 44 failed: {failed} MRI(s) "
    "did not pass the frozen preprocessing "
    "integrity audit."
)


# ============================================================
# HASH INPUT + OUTPUT
# ============================================================

step43_sha = sha256_file(
    STATUS_FILE
)

audit_sha = sha256_file(
    AUDIT_CSV
)

mni_sha = sha256_file(
    MNI_FILE
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "step": 44,
    "purpose":
        "Frozen OASIS-3 external preprocessing "
        "integrity audit",

    "timestamp":
        datetime.now().isoformat(),

    "participants": 605,
    "locked_acquisitions": 1901,

    "integrity_pass": passed,
    "integrity_fail": failed,

    "mni_reference":
        str(MNI_FILE),

    "mni_shape":
        list(mni_shape),

    "mni_zooms":
        list(mni_zooms),

    "step43_status_file":
        str(STATUS_FILE),

    "step43_status_sha256":
        step43_sha,

    "step44_audit_csv":
        str(AUDIT_CSV),

    "step44_audit_sha256":
        audit_sha,

    "mni_reference_sha256":
        mni_sha,

    "external_outcomes_used": False,
    "frozen_model_inference_performed": False,
    "mri_files_modified": False,
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# SAVE FORMAL LOCK
# ============================================================

lock = {
    "step": 44,

    "status":
        "LOCKED_PASS",

    "participants":
        605,

    "locked_acquisitions":
        1901,

    "integrity_pass":
        1901,

    "integrity_fail":
        0,

    "step43_status_sha256":
        step43_sha,

    "step44_audit_sha256":
        audit_sha,

    "mni_reference_sha256":
        mni_sha,

    "protocol":
        "Frozen OASIS-3 external preprocessing "
        "outputs verified against the frozen "
        "ADNI MNI152 1-mm reference geometry.",

    "external_outcomes_used":
        False,

    "frozen_model_inference_performed":
        False,

    "mri_files_modified":
        False,
}

with open(
    LOCK_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        lock,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 78)
print("STEP 44 FINAL REPORT")
print("=" * 78)

print("\nParticipants        :", 605)
print("Locked acquisitions :", 1901)
print("Integrity PASS      :", passed)
print("Integrity FAIL      :", failed)

print("\nAUDIT:")
print(AUDIT_CSV)

print("\nSUMMARY:")
print(SUMMARY_JSON)

print("\nLOCK:")
print(LOCK_JSON)

print("\nSHA256")
print("Step-43 status :", step43_sha)
print("Step-44 audit  :", audit_sha)
print("MNI reference  :", mni_sha)

print("\nIMPORTANT:")
print("No MRI files were modified.")
print("No external outcome data were used.")
print("No frozen ADNI model inference was performed.")

print("\n" + "=" * 78)
print(
    "STEP 44 PASSED — "
    "EXTERNAL PREPROCESSING INTEGRITY LOCKED"
)
print("=" * 78)