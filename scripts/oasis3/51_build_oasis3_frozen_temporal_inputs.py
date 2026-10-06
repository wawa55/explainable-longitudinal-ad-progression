from pathlib import Path
import ast
import hashlib
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# STEP 51
# OASIS-3 frozen temporal input construction
#
# PURPOSE
# -------
# Reproduce the exact temporal-coordinate convention used by
# the frozen ADNI longitudinal model:
#
#   days_from_first_mri[j] =
#       mri_day[j] - first_mri_day
#
#   days_before_landmark[j] =
#       landmark_day - mri_day[j]
#
#   years_from_first[j] =
#       days_from_first_mri[j] / 365.25
#
#   years_before_landmark[j] =
#       days_before_landmark[j] / 365.25
#
# This step DOES NOT:
#   - read external outcomes
#   - read prediction files
#   - load model checkpoints
#   - run frozen ADNI inference
#   - modify MRI files
#
# It only constructs and locks temporal model inputs.
# ============================================================


ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

COHORT_DIR = ROOT / "cohort_construction"

STEP50 = (
    COHORT_DIR
    / "OASIS3_STEP50_EXTERNAL_LONGITUDINAL_SEQUENCES_LOCKED.csv"
)

STEP50_VISITS = (
    COHORT_DIR
    / "OASIS3_STEP50_EXTERNAL_LONGITUDINAL_SEQUENCE_VISITS.csv"
)

OUT = (
    COHORT_DIR
    / "OASIS3_STEP51_FROZEN_TEMPORAL_INPUTS_LOCKED.csv"
)

OUT_VISITS = (
    COHORT_DIR
    / "OASIS3_STEP51_FROZEN_TEMPORAL_INPUT_VISITS.csv"
)

SUMMARY = (
    COHORT_DIR
    / "OASIS3_STEP51_FROZEN_TEMPORAL_INPUTS_SUMMARY.json"
)

LOCK = (
    COHORT_DIR
    / "OASIS3_STEP51_FROZEN_TEMPORAL_INPUTS_LOCK.json"
)

DAYS_PER_YEAR = 365.25


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def parse_list_field(value):
    """
    Parse list-like fields robustly.

    Accepts:
        Python lists
        tuples
        numpy arrays
        string representations of lists
        pipe-separated strings
        comma-separated strings
        scalar values
    """

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return []

    s = str(value).strip()

    if not s:
        return []

    # Python/JSON-style list
    if s.startswith("[") and s.endswith("]"):
        try:
            x = ast.literal_eval(s)

            if isinstance(x, (list, tuple, np.ndarray)):
                return list(x)

            return [x]

        except Exception:
            pass

    # Step-50 MRI IDs are commonly pipe-separated.
    if "|" in s:
        return [
            x.strip()
            for x in s.split("|")
            if x.strip()
        ]

    if "," in s:
        return [
            x.strip()
            for x in s.split(",")
            if x.strip()
        ]

    return [s]


def serialize_numeric_list(values):
    """
    Stable integer-list representation.
    """

    return json.dumps(
        [int(round(float(x))) for x in values],
        separators=(",", ":"),
    )


def serialize_float_list(values):
    """
    Stable floating-point list representation.
    """

    return json.dumps(
        [round(float(x), 10) for x in values],
        separators=(",", ":"),
    )


def fail(message):
    raise RuntimeError(message)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 51 — OASIS-3 FROZEN TEMPORAL INPUT CONSTRUCTION")
print("=" * 70)


# ============================================================
# 1. VERIFY INPUT FILES
# ============================================================

for p in [STEP50, STEP50_VISITS]:
    if not p.exists():
        fail(f"Missing required Step-50 input:\n{p}")


step50_hash = sha256_file(STEP50)
step50_visits_hash = sha256_file(STEP50_VISITS)

print("\nINPUT:")
print(STEP50)
print("\nEXPANDED VISITS:")
print(STEP50_VISITS)

print("\nSHA256")
print("Step-50 sequences :", step50_hash)
print("Step-50 visits    :", step50_visits_hash)


# ============================================================
# 2. READ LOCKED STEP-50 DATA
# ============================================================

seq = pd.read_csv(
    STEP50,
    low_memory=False,
)

vis = pd.read_csv(
    STEP50_VISITS,
    low_memory=False,
)

print("\nINPUT COUNTS")
print("-" * 70)
print("Locked sequences :", len(seq))
print("Expanded visits  :", len(vis))


# ============================================================
# 3. REQUIRED STEP-50 SEQUENCE COLUMNS
# ============================================================

required_seq = [
    "locked_subject_id",
    "landmark_mri_id",
    "landmark_day",
    "n_visits",
    "first_mri_day",
    "mri_ids",
]

missing_seq = [
    c for c in required_seq
    if c not in seq.columns
]

if missing_seq:
    fail(
        "Step-50 sequence table missing required columns: "
        + repr(missing_seq)
    )


# ============================================================
# 4. REQUIRED EXPANDED-VISIT COLUMNS
# ============================================================

required_vis = [
    "locked_subject_id",
    "landmark_mri_id",
]

missing_vis = [
    c for c in required_vis
    if c not in vis.columns
]

if missing_vis:
    fail(
        "Step-50 expanded visit table missing required columns: "
        + repr(missing_vis)
    )


# Detect MRI ID column in expanded table.

visit_mri_candidates = [
    "visit_mri_id",
    "locked_mri_id",
    "mri_id",
]

visit_mri_col = next(
    (c for c in visit_mri_candidates if c in vis.columns),
    None,
)

if visit_mri_col is None:
    fail(
        "Could not identify MRI ID column in Step-50 expanded visits."
    )


# Detect MRI day column.

visit_day_candidates = [
    "visit_mri_day",
    "mri_day",
    "visit_day",
]

visit_day_col = next(
    (c for c in visit_day_candidates if c in vis.columns),
    None,
)

if visit_day_col is None:
    fail(
        "Could not identify MRI day column in Step-50 expanded visits."
    )


print("\nEXPANDED VISIT SCHEMA")
print("MRI ID column  :", visit_mri_col)
print("MRI day column :", visit_day_col)


# ============================================================
# 5. NUMERIC NORMALIZATION
# ============================================================

seq["landmark_day"] = pd.to_numeric(
    seq["landmark_day"],
    errors="coerce",
)

seq["first_mri_day"] = pd.to_numeric(
    seq["first_mri_day"],
    errors="coerce",
)

seq["n_visits"] = pd.to_numeric(
    seq["n_visits"],
    errors="coerce",
)

vis[visit_day_col] = pd.to_numeric(
    vis[visit_day_col],
    errors="coerce",
)


if seq[
    ["landmark_day", "first_mri_day", "n_visits"]
].isna().any().any():
    fail(
        "Missing/non-numeric landmark_day, first_mri_day, "
        "or n_visits in Step-50 sequence table."
    )


if vis[visit_day_col].isna().any():
    fail(
        "Missing/non-numeric MRI day in Step-50 expanded visits."
    )


# ============================================================
# 6. UNIQUENESS AUDITS
# ============================================================

sequence_key = [
    "locked_subject_id",
    "landmark_mri_id",
]

if seq.duplicated(sequence_key).any():
    bad = seq.loc[
        seq.duplicated(sequence_key, keep=False),
        sequence_key,
    ]

    print("\nDUPLICATE SEQUENCE KEYS:")
    print(bad.head(20).to_string(index=False))

    fail(
        "Duplicate Step-50 longitudinal sequence keys detected."
    )


# ============================================================
# 7. CONSTRUCT TEMPORAL INPUTS
# ============================================================

output_rows = []
expanded_rows = []

for row_number, row in seq.iterrows():

    subject = str(row["locked_subject_id"])
    landmark_mri_id = str(row["landmark_mri_id"])

    landmark_day = int(round(float(row["landmark_day"])))
    first_mri_day = int(round(float(row["first_mri_day"])))
    expected_n = int(round(float(row["n_visits"])))

    # --------------------------------------------------------
    # Select expanded visits for this exact landmark sequence
    # --------------------------------------------------------

    tmp = vis.loc[
        (
            vis["locked_subject_id"].astype(str)
            == subject
        )
        &
        (
            vis["landmark_mri_id"].astype(str)
            == landmark_mri_id
        )
    ].copy()

    if len(tmp) == 0:
        fail(
            f"No expanded visits found for "
            f"{subject} / {landmark_mri_id}"
        )

    tmp = tmp.sort_values(
        visit_day_col,
        kind="mergesort",
    ).reset_index(drop=True)

    mri_days = (
        tmp[visit_day_col]
        .round()
        .astype(int)
        .tolist()
    )

    mri_ids = (
        tmp[visit_mri_col]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Length checks
    # --------------------------------------------------------

    if len(mri_days) != expected_n:
        fail(
            f"{subject} / {landmark_mri_id}: "
            f"Step-50 n_visits={expected_n}, "
            f"expanded visits={len(mri_days)}"
        )

    if len(mri_ids) != expected_n:
        fail(
            f"{subject} / {landmark_mri_id}: "
            "MRI ID count mismatch."
        )

    # --------------------------------------------------------
    # Compare against Step-50 sequence MRI IDs
    # --------------------------------------------------------

    seq_mri_ids = [
        str(x)
        for x in parse_list_field(row["mri_ids"])
    ]

    if len(seq_mri_ids) != expected_n:
        fail(
            f"{subject} / {landmark_mri_id}: "
            "Step-50 mri_ids length mismatch."
        )

    if seq_mri_ids != mri_ids:
        fail(
            f"{subject} / {landmark_mri_id}: "
            "Expanded MRI order does not match Step-50 mri_ids."
        )

    # --------------------------------------------------------
    # Chronology
    # --------------------------------------------------------

    if expected_n < 1:
        fail(
            f"{subject} / {landmark_mri_id}: "
            "sequence contains zero MRI visits."
        )

    if mri_days[0] != first_mri_day:
        fail(
            f"{subject} / {landmark_mri_id}: "
            f"first MRI mismatch "
            f"({mri_days[0]} != {first_mri_day})."
        )

    if any(
        mri_days[i] >= mri_days[i + 1]
        for i in range(len(mri_days) - 1)
    ):
        fail(
            f"{subject} / {landmark_mri_id}: "
            "MRI days are not strictly increasing."
        )

    if any(day > landmark_day for day in mri_days):
        fail(
            f"{subject} / {landmark_mri_id}: "
            "future MRI leakage detected."
        )

    # --------------------------------------------------------
    # EXACT FROZEN ADNI TEMPORAL CONVENTION
    # --------------------------------------------------------

    days_from_first_mri = [
        int(day - first_mri_day)
        for day in mri_days
    ]

    days_before_landmark = [
        int(landmark_day - day)
        for day in mri_days
    ]

    years_from_first = [
        float(x) / DAYS_PER_YEAR
        for x in days_from_first_mri
    ]

    years_before_landmark = [
        float(x) / DAYS_PER_YEAR
        for x in days_before_landmark
    ]

    # --------------------------------------------------------
    # Frozen-contract invariants
    # --------------------------------------------------------

    if days_from_first_mri[0] != 0:
        fail(
            f"{subject} / {landmark_mri_id}: "
            "first days_from_first_mri is not zero."
        )

    if any(x < 0 for x in days_from_first_mri):
        fail(
            f"{subject} / {landmark_mri_id}: "
            "negative days_from_first_mri detected."
        )

    if any(x < 0 for x in days_before_landmark):
        fail(
            f"{subject} / {landmark_mri_id}: "
            "negative days_before_landmark detected."
        )

    expected_span = landmark_day - first_mri_day

    invariant_values = [
        a + b
        for a, b in zip(
            days_from_first_mri,
            days_before_landmark,
        )
    ]

    if any(x != expected_span for x in invariant_values):
        fail(
            f"{subject} / {landmark_mri_id}: "
            "temporal-coordinate invariant failed."
        )

    # The landmark MRI should be the final MRI in the sequence.
    if mri_ids[-1] != landmark_mri_id:
        fail(
            f"{subject} / {landmark_mri_id}: "
            "landmark MRI is not final MRI in sequence."
        )

    if mri_days[-1] != landmark_day:
        fail(
            f"{subject} / {landmark_mri_id}: "
            "landmark MRI day does not equal landmark_day."
        )

    # --------------------------------------------------------
    # Sequence-level output
    # --------------------------------------------------------

    rec = row.to_dict()

    rec["mri_days"] = serialize_numeric_list(
        mri_days
    )

    rec["days_from_first_mri"] = serialize_numeric_list(
        days_from_first_mri
    )

    rec["days_before_landmark"] = serialize_numeric_list(
        days_before_landmark
    )

    rec["years_from_first"] = serialize_float_list(
        years_from_first
    )

    rec["years_before_landmark"] = serialize_float_list(
        years_before_landmark
    )

    rec["temporal_history_span_days"] = int(
        expected_span
    )

    rec["temporal_history_span_years"] = float(
        expected_span / DAYS_PER_YEAR
    )

    rec["days_since_latest_mri"] = int(
        days_before_landmark[-1]
    )

    rec["step51_days_per_year"] = DAYS_PER_YEAR

    rec["step51_temporal_definition"] = (
        "ADNI frozen contract: "
        "days_from_first_mri=mri_day-first_mri_day; "
        "days_before_landmark=landmark_day-mri_day"
    )

    rec["step51_temporal_input_locked"] = True

    output_rows.append(rec)

    # --------------------------------------------------------
    # Visit-level output
    # --------------------------------------------------------

    for j in range(expected_n):

        expanded_rows.append(
            {
                "locked_subject_id": subject,
                "landmark_mri_id": landmark_mri_id,
                "landmark_day": landmark_day,
                "visit_index": j,
                "visit_number": j + 1,
                "visit_mri_id": mri_ids[j],
                "mri_day": mri_days[j],
                "first_mri_day": first_mri_day,
                "days_from_first_mri":
                    days_from_first_mri[j],
                "days_before_landmark":
                    days_before_landmark[j],
                "years_from_first":
                    years_from_first[j],
                "years_before_landmark":
                    years_before_landmark[j],
                "is_landmark_mri":
                    bool(mri_ids[j] == landmark_mri_id),
            }
        )


# ============================================================
# 8. BUILD OUTPUT TABLES
# ============================================================

out = pd.DataFrame(output_rows)

expanded = pd.DataFrame(expanded_rows)


# ============================================================
# 9. GLOBAL AUDITS
# ============================================================

if len(out) != len(seq):
    fail(
        "Output sequence count differs from Step-50 input."
    )


if out.duplicated(sequence_key).any():
    fail(
        "Duplicate sequence keys in Step-51 output."
    )


if len(expanded) != int(out["n_visits"].sum()):
    fail(
        "Expanded Step-51 visit count does not equal "
        "sum(n_visits)."
    )


# Each sequence must contain exactly one landmark MRI.

landmark_counts = (
    expanded
    .groupby(sequence_key)["is_landmark_mri"]
    .sum()
)

if not (landmark_counts == 1).all():
    fail(
        "Not every sequence contains exactly one landmark MRI."
    )


# No future MRI leakage.

future_leakage = int(
    (
        expanded["days_before_landmark"] < 0
    ).sum()
)

if future_leakage != 0:
    fail(
        f"Future MRI leakage count is {future_leakage}, "
        "expected zero."
    )


# All first temporal coordinates must be zero.

first_visit = (
    expanded
    .sort_values(
        sequence_key + ["visit_index"]
    )
    .groupby(sequence_key)
    .first()
)

bad_first = int(
    (
        first_visit["days_from_first_mri"] != 0
    ).sum()
)

if bad_first != 0:
    fail(
        "At least one sequence does not begin at temporal "
        "coordinate zero."
    )


# Landmark visit should have zero days before landmark.

landmark_rows = expanded.loc[
    expanded["is_landmark_mri"]
].copy()

bad_landmark_zero = int(
    (
        landmark_rows["days_before_landmark"] != 0
    ).sum()
)

if bad_landmark_zero != 0:
    fail(
        "At least one landmark MRI does not have "
        "days_before_landmark = 0."
    )


# ============================================================
# 10. SAVE OUTPUTS
# ============================================================

out.to_csv(
    OUT,
    index=False,
)

expanded.to_csv(
    OUT_VISITS,
    index=False,
)


out_hash = sha256_file(OUT)
expanded_hash = sha256_file(OUT_VISITS)


# ============================================================
# 11. SUMMARY
# ============================================================

n_participants = int(
    out["locked_subject_id"].nunique()
)

n_sequences = int(len(out))

n_expanded_visits = int(len(expanded))

n_single_visit = int(
    (out["n_visits"] == 1).sum()
)

n_multi_visit = int(
    (out["n_visits"] >= 2).sum()
)

max_visits = int(
    out["n_visits"].max()
)

summary = {
    "step": 51,
    "purpose": (
        "Construct frozen ADNI-compatible temporal inputs "
        "for locked OASIS-3 longitudinal MRI sequences."
    ),
    "days_per_year": DAYS_PER_YEAR,
    "temporal_contract": {
        "days_from_first_mri":
            "mri_day - first_mri_day",
        "days_before_landmark":
            "landmark_day - mri_day",
        "years_from_first":
            "days_from_first_mri / 365.25",
        "years_before_landmark":
            "days_before_landmark / 365.25",
    },
    "participants": n_participants,
    "sequences": n_sequences,
    "expanded_visits": n_expanded_visits,
    "single_visit_sequences": n_single_visit,
    "multi_visit_sequences": n_multi_visit,
    "maximum_visits": max_visits,
    "future_mri_leakage_count": future_leakage,
    "step50_sequences_sha256": step50_hash,
    "step50_expanded_visits_sha256":
        step50_visits_hash,
    "step51_sequences_sha256": out_hash,
    "step51_expanded_visits_sha256":
        expanded_hash,
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


# ============================================================
# 12. LOCK
# ============================================================

lock = {
    "step": 51,
    "status": "LOCKED",
    "input_step": 50,
    "input_sequence_file": str(STEP50),
    "input_expanded_visit_file": str(STEP50_VISITS),
    "output_sequence_file": str(OUT),
    "output_expanded_visit_file": str(OUT_VISITS),
    "step50_sequences_sha256": step50_hash,
    "step50_expanded_visits_sha256":
        step50_visits_hash,
    "step51_sequences_sha256": out_hash,
    "step51_expanded_visits_sha256":
        expanded_hash,
    "days_per_year": DAYS_PER_YEAR,
    "temporal_contract": {
        "days_from_first_mri":
            "mri_day - first_mri_day",
        "days_before_landmark":
            "landmark_day - mri_day",
        "years_from_first":
            "days_from_first_mri / 365.25",
        "years_before_landmark":
            "days_before_landmark / 365.25",
    },
    "invariants": {
        "chronological_mri_order": True,
        "first_days_from_first_mri_zero": True,
        "nonnegative_days_from_first_mri": True,
        "nonnegative_days_before_landmark": True,
        "landmark_is_final_sequence_mri": True,
        "landmark_days_before_landmark_zero": True,
        "future_mri_leakage_count": 0,
        "temporal_sum_invariant": True,
    },
    "external_outcomes_read": False,
    "prediction_files_read": False,
    "frozen_adni_model_loaded": False,
    "frozen_adni_model_inference_performed": False,
    "mri_files_modified": False,
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
# 13. REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 51 RESULTS")
print("=" * 70)

print("Participants              :", n_participants)
print("Locked sequences          :", n_sequences)
print("Expanded MRI visits       :", n_expanded_visits)
print("Single-visit sequences    :", n_single_visit)
print("Multi-visit sequences     :", n_multi_visit)
print("Maximum visits            :", max_visits)
print("Future MRI leakage        :", future_leakage)

print("\nTEMPORAL CONTRACT")
print(
    "days_from_first_mri = "
    "mri_day - first_mri_day"
)
print(
    "days_before_landmark = "
    "landmark_day - mri_day"
)
print(
    "years_from_first = "
    "days_from_first_mri / 365.25"
)
print(
    "years_before_landmark = "
    "days_before_landmark / 365.25"
)

print("\nFIRST 5 LOCKED TEMPORAL SEQUENCES")

display_cols = [
    "locked_subject_id",
    "landmark_mri_id",
    "landmark_day",
    "n_visits",
    "mri_days",
    "days_from_first_mri",
    "days_before_landmark",
]

print(
    out[display_cols]
    .head(5)
    .to_string(index=False)
)

print("\nOUTPUT:")
print(OUT)

print("\nEXPANDED VISITS:")
print(OUT_VISITS)

print("\nSUMMARY:")
print(SUMMARY)

print("\nLOCK:")
print(LOCK)

print("\nSHA256")
print("Step-50 sequences :", step50_hash)
print("Step-50 visits    :", step50_visits_hash)
print("Step-51 sequences :", out_hash)
print("Step-51 visits    :", expanded_hash)

print("\nIMPORTANT:")
print("No external outcome data were read.")
print("No prediction files were read.")
print("No frozen ADNI model checkpoint was loaded.")
print("No frozen ADNI model inference was performed.")
print("No MRI files were modified.")
print(
    "Temporal inputs reproduce the frozen ADNI "
    "continuous-time convention."
)

print("\n" + "=" * 70)
print(
    "STEP 51 PASSED — FROZEN TEMPORAL INPUT CONTRACT LOCKED"
)
print("=" * 70)