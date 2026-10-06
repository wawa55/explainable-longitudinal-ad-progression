from pathlib import Path
import pandas as pd

# ============================================================
# STEP 58 PRE-CHECK
# Locate the existing MRI-ID -> preprocessed NIfTI mapping.
#
# IMPORTANT:
#   - READ ONLY
#   - Does NOT load MRI voxels
#   - Does NOT load the frozen model
#   - Does NOT generate predictions
#   - Does NOT modify any locked files
# ============================================================

PROJECT_ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

COHORT_DIR = PROJECT_ROOT / "cohort_construction"

EXTERNAL_ROOT = Path(
    r"E:\OASIS3_EXTERNAL_VALIDATION"
)

TARGET_MRI_IDS = [
    "OAS30001_MR_d0129",
    "OAS31172_MR_d0407",
    "OAS31172_MR_d1717",
]

SEARCH_ROOTS = [
    COHORT_DIR,
    EXTERNAL_ROOT,
]


def is_useful_column(name):
    name = str(name).lower()

    terms = [
        "mri",
        "path",
        "file",
        "nii",
        "preprocess",
        "ready",
        "output",
        "registered",
        "normalized",
        "image",
    ]

    return any(term in name for term in terms)


print("=" * 78)
print("STEP 58 PRE-CHECK — LOCATE PREPROCESSED MRI PATH MAPPING")
print("=" * 78)

print("\nSEARCH ROOTS")
for root in SEARCH_ROOTS:
    print(f"  {root}")
    print(f"    exists: {root.exists()}")

# ------------------------------------------------------------
# Collect CSV files
# ------------------------------------------------------------

csv_files = []

for root in SEARCH_ROOTS:
    if root.exists():
        csv_files.extend(root.rglob("*.csv"))

# Remove duplicate paths
csv_files = sorted(set(csv_files))

print("\nCSV FILES FOUND:", len(csv_files))

if len(csv_files) == 0:
    raise RuntimeError("No CSV files were found in the search roots.")

# ------------------------------------------------------------
# Search CSV contents
# ------------------------------------------------------------

matches = []

for i, path in enumerate(csv_files, start=1):

    if i % 100 == 0:
        print(f"Scanned {i}/{len(csv_files)} CSV files...")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        continue

    if df.empty:
        continue

    useful_cols = [
        c for c in df.columns
        if is_useful_column(c)
    ]

    matched_ids = []
    matched_columns = []

    for col in df.columns:

        try:
            values = df[col].astype(str)
        except Exception:
            continue

        for mri_id in TARGET_MRI_IDS:

            try:
                hit = values.str.contains(
                    mri_id,
                    regex=False,
                    na=False
                )
            except Exception:
                continue

            if hit.any():
                if mri_id not in matched_ids:
                    matched_ids.append(mri_id)

                if col not in matched_columns:
                    matched_columns.append(col)

    if matched_ids:

        matches.append(
            {
                "path": path,
                "rows": len(df),
                "columns": len(df.columns),
                "matched_ids": matched_ids,
                "matched_columns": matched_columns,
                "useful_columns": useful_cols,
            }
        )

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("SEARCH COMPLETE")
print("=" * 78)

print("\nMATCHING CSV FILES:", len(matches))

for number, item in enumerate(matches, start=1):

    print("\n" + "-" * 78)
    print(f"MATCH {number}")
    print("-" * 78)

    print("FILE:")
    print(item["path"])

    print("\nROWS:")
    print(item["rows"])

    print("\nCOLUMNS:")
    print(item["columns"])

    print("\nMATCHED MRI IDS:")
    for value in item["matched_ids"]:
        print("  ", value)

    print("\nMATCHED COLUMNS:")
    for value in item["matched_columns"]:
        print("  ", value)

    print("\nUSEFUL COLUMNS:")
    for value in item["useful_columns"]:
        print("  ", value)

    # --------------------------------------------------------
    # Show matching rows, but only useful columns
    # --------------------------------------------------------

    try:
        df = pd.read_csv(item["path"], low_memory=False)

        mask = pd.Series(False, index=df.index)

        for col in df.columns:

            try:
                values = df[col].astype(str)
            except Exception:
                continue

            for mri_id in TARGET_MRI_IDS:

                try:
                    mask = mask | values.str.contains(
                        mri_id,
                        regex=False,
                        na=False
                    )
                except Exception:
                    pass

        subset = df.loc[mask].copy()

        display_cols = []

        for col in df.columns:
            if (
                col in item["matched_columns"]
                or is_useful_column(col)
            ):
                display_cols.append(col)

        # Remove duplicate column names while preserving order
        display_cols = list(dict.fromkeys(display_cols))

        if display_cols:
            subset = subset[display_cols]

        print("\nMATCHING ROWS:")
        print(subset.head(20).to_string(index=False))

    except Exception as exc:
        print("\nCould not display matching rows:")
        print(exc)

# ------------------------------------------------------------
# Search actual NIfTI filenames too
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("DIRECT NIFTI FILE SEARCH")
print("=" * 78)

nifti_matches = []

if EXTERNAL_ROOT.exists():

    for path in EXTERNAL_ROOT.rglob("*.nii.gz"):

        text = str(path)

        for mri_id in TARGET_MRI_IDS:

            if mri_id in text:

                nifti_matches.append(
                    (mri_id, path)
                )

print("\nDIRECT NIFTI MATCHES:", len(nifti_matches))

for mri_id, path in nifti_matches:
    print("\nMRI ID:")
    print(mri_id)
    print("FILE:")
    print(path)

# ------------------------------------------------------------
# Final interpretation
# ------------------------------------------------------------

print("\n" + "=" * 78)

if matches or nifti_matches:

    print("PRE-CHECK COMPLETE — CANDIDATE MRI PATH MAPPING FOUND")

else:

    print("PRE-CHECK COMPLETE — NO MRI PATH MAPPING FOUND")

print("=" * 78)

print("\nIMPORTANT")
print("MRI voxels read                 : NO")
print("Frozen checkpoint loaded        : NO")
print("Model instantiated              : NO")
print("Predictions generated           : NO")
print("External performance examined   : NO")
print("Locked cohort modified          : NO")
print("Preprocessed MRI modified       : NO")