from pathlib import Path
import pandas as pd


# ============================================================
# STEP 46
# OASIS-3 MRI-to-CDR landmark eligibility audit
#
# PURPOSE:
#   1. Read the locked 1,901 MRI acquisition table.
#   2. Read the locked Step-45 CDR clinical timeline.
#   3. Extract MRI day from locked_mri_id.
#   4. Find the most recent observed CDR at or before each MRI.
#   5. Determine which MRI landmarks are still CDR = 0.
#
# IMPORTANT:
#   - No MRI files are modified.
#   - No prediction files are read.
#   - No frozen ADNI model inference is performed.
#   - No outcome threshold is tuned from model performance.
# ============================================================


ROOT = Path(__file__).resolve().parent
COHORT = ROOT / "cohort_construction"

MRI_FILE = (
    COHORT /
    "OASIS3_EXACT_T1_ACQUISITION_LOCKED.csv"
)

CDR_FILE = (
    COHORT /
    "OASIS3_STEP45_CDR_CLINICAL_TIMELINE_LOCKED.csv"
)

OUTPUT_FILE = (
    COHORT /
    "OASIS3_STEP46_MRI_CDR_LANDMARK_AUDIT.csv"
)

ELIGIBLE_FILE = (
    COHORT /
    "OASIS3_STEP46_CDR0_ELIGIBLE_MRI_LANDMARKS.csv"
)


# ============================================================
# 1. INPUT CHECKS
# ============================================================

print("=" * 72)
print("STEP 46 — OASIS-3 MRI/CDR LANDMARK ELIGIBILITY AUDIT")
print("=" * 72)

for p in [MRI_FILE, CDR_FILE]:
    if not p.exists():
        raise FileNotFoundError(
            f"Required locked input not found:\n{p}"
        )

print("\nMRI lock:")
print(MRI_FILE)

print("\nCDR timeline lock:")
print(CDR_FILE)


# ============================================================
# 2. LOAD LOCKED INPUTS
# ============================================================

mri = pd.read_csv(
    MRI_FILE,
    low_memory=False
)

cdr = pd.read_csv(
    CDR_FILE,
    low_memory=False
)

print("\nINPUT COUNTS")
print("-" * 72)
print("MRI acquisitions :", len(mri))
print("CDR visits       :", len(cdr))


# ============================================================
# 3. REQUIRED COLUMN CHECKS
# ============================================================

mri_required = [
    "locked_subject_id",
    "locked_mri_id",
]

cdr_required = [
    "locked_subject_id",
    "days_to_visit",
    "CDRTOT",
]

missing_mri = [
    c for c in mri_required
    if c not in mri.columns
]

missing_cdr = [
    c for c in cdr_required
    if c not in cdr.columns
]

if missing_mri:
    raise RuntimeError(
        "MRI table missing required columns: "
        + str(missing_mri)
    )

if missing_cdr:
    raise RuntimeError(
        "CDR table missing required columns: "
        + str(missing_cdr)
    )


# ============================================================
# 4. VERIFY LOCKED MRI COHORT
# ============================================================

if len(mri) != 1901:
    raise RuntimeError(
        f"Expected 1901 locked MRI acquisitions, found {len(mri)}"
    )

n_subjects = (
    mri["locked_subject_id"]
    .astype(str)
    .str.strip()
    .nunique()
)

if n_subjects != 605:
    raise RuntimeError(
        f"Expected 605 participants, found {n_subjects}"
    )

print("\nLocked MRI cohort verification: PASS")
print("Participants      :", n_subjects)
print("MRI acquisitions  :", len(mri))


# ============================================================
# 5. EXTRACT MRI DAY
# ============================================================

mri["locked_subject_id"] = (
    mri["locked_subject_id"]
    .astype(str)
    .str.strip()
)

mri["locked_mri_id"] = (
    mri["locked_mri_id"]
    .astype(str)
    .str.strip()
)

mri["mri_day"] = pd.to_numeric(
    mri["locked_mri_id"]
    .str.extract(r"_d(\d+)$", expand=False),
    errors="coerce",
)

bad_mri_day = mri["mri_day"].isna()

if bad_mri_day.any():
    print("\nERROR: MRI day could not be parsed for:")
    print(
        mri.loc[
            bad_mri_day,
            ["locked_subject_id", "locked_mri_id"]
        ].to_string(index=False)
    )
    raise RuntimeError(
        "MRI day parsing failed."
    )

mri["mri_day"] = mri["mri_day"].astype(int)

print("\nMRI day extraction: PASS")


# ============================================================
# 6. CLEAN LOCKED CDR TIMELINE
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

cdr_valid = cdr[
    cdr["days_to_visit"].notna()
    & cdr["CDRTOT"].notna()
].copy()

cdr_valid = cdr_valid.sort_values(
    ["locked_subject_id", "days_to_visit"]
).reset_index(drop=True)

print("\nValid CDR observations :", len(cdr_valid))

cdr_subjects = cdr_valid["locked_subject_id"].nunique()

print("Participants with valid CDR :", cdr_subjects)


# ============================================================
# 7. BUILD PARTICIPANT LOOKUP
# ============================================================

cdr_groups = {
    sid: group.sort_values("days_to_visit")
    for sid, group in cdr_valid.groupby("locked_subject_id")
}


# ============================================================
# 8. LINK EACH MRI TO MOST RECENT CDR AT OR BEFORE MRI
# ============================================================

records = []

for _, row in mri.iterrows():

    subject = str(row["locked_subject_id"])
    mri_id = str(row["locked_mri_id"])
    mri_day = int(row["mri_day"])

    rec = {
        "locked_subject_id": subject,
        "locked_mri_id": mri_id,
        "mri_day": mri_day,
        "prior_cdr_available": False,
        "prior_clinical_day": pd.NA,
        "prior_cdrtot": pd.NA,
        "clinical_to_mri_gap_days": pd.NA,
        "eligible_cdr0_landmark": False,
    }

    subject_cdr = cdr_groups.get(subject)

    if subject_cdr is not None:

        prior = subject_cdr[
            subject_cdr["days_to_visit"] <= mri_day
        ]

        if len(prior) > 0:

            latest = prior.iloc[-1]

            clinical_day = float(
                latest["days_to_visit"]
            )

            prior_cdr = float(
                latest["CDRTOT"]
            )

            gap = float(
                mri_day - clinical_day
            )

            rec["prior_cdr_available"] = True
            rec["prior_clinical_day"] = clinical_day
            rec["prior_cdrtot"] = prior_cdr
            rec["clinical_to_mri_gap_days"] = gap

            if prior_cdr == 0.0:
                rec["eligible_cdr0_landmark"] = True

    records.append(rec)


audit = pd.DataFrame(records)


# ============================================================
# 9. BASIC CONSISTENCY CHECKS
# ============================================================

if len(audit) != 1901:
    raise RuntimeError(
        f"Expected 1901 audit rows, found {len(audit)}"
    )

if audit["locked_mri_id"].nunique() != 1901:
    raise RuntimeError(
        "locked_mri_id is not unique in Step 46 audit."
    )

# Gap must never be negative because only CDR visits at or
# before the MRI landmark are allowed.
available_tmp = audit[
    audit["prior_cdr_available"] == True
].copy()

if len(available_tmp) > 0:
    gap_check = pd.to_numeric(
        available_tmp["clinical_to_mri_gap_days"],
        errors="coerce",
    )

    if gap_check.isna().any():
        raise RuntimeError(
            "Missing clinical-to-MRI gap among linked landmarks."
        )

    if (gap_check < 0).any():
        raise RuntimeError(
            "Negative clinical-to-MRI gap detected."
        )

available = audit[
    audit["prior_cdr_available"] == True
].copy()

eligible = audit[
    audit["eligible_cdr0_landmark"] == True
].copy()

no_prior = audit[
    audit["prior_cdr_available"] == False
].copy()


# ============================================================
# 10. SAVE AUDIT TABLES
# ============================================================

audit.to_csv(
    OUTPUT_FILE,
    index=False
)

eligible.to_csv(
    ELIGIBLE_FILE,
    index=False
)


# ============================================================
# 11. REPORT
# ============================================================

print("\n" + "=" * 72)
print("STEP 46 AUDIT RESULTS")
print("=" * 72)

print("\nTOTAL MRI LANDMARKS")
print("MRI acquisitions              :", len(audit))
print(
    "Participants                  :",
    audit["locked_subject_id"].nunique()
)

print("\nCLINICAL HISTORY AVAILABILITY")
print(
    "MRI with prior/same-day CDR   :",
    len(available)
)
print(
    "MRI without prior CDR         :",
    len(no_prior)
)

print("\nPRIOR CDR AT MRI LANDMARK")

if len(available) > 0:
    print(
        available["prior_cdrtot"]
        .value_counts()
        .sort_index()
        .to_string()
    )
else:
    print("No prior CDR observations found.")


print("\nCDR=0 ELIGIBILITY")

print(
    "Eligible CDR=0 MRI landmarks  :",
    len(eligible)
)

print(
    "Eligible participants         :",
    eligible["locked_subject_id"].nunique()
)


print("\nCLINICAL-TO-MRI GAP (DAYS)")

if len(available) > 0:
    print(
        available[
            "clinical_to_mri_gap_days"
        ]
        .describe()
        .to_string()
    )
else:
    print("No linked CDR observations.")


# ============================================================
# 12. ADDITIONAL GAP WINDOWS
# ============================================================

if len(available) > 0:

    gap = pd.to_numeric(
        available["clinical_to_mri_gap_days"],
        errors="coerce",
    )

    print("\nGAP WINDOW COUNTS")

    print(
        "<= 30 days  :",
        int((gap <= 30).sum())
    )

    print(
        "<= 90 days  :",
        int((gap <= 90).sum())
    )

    print(
        "<= 180 days :",
        int((gap <= 180).sum())
    )

    print(
        "<= 365 days :",
        int((gap <= 365).sum())
    )

    print(
        "> 365 days  :",
        int((gap > 365).sum())
    )


# ============================================================
# 13. ELIGIBLE LANDMARK GAP WINDOWS
# ============================================================

if len(eligible) > 0:

    egap = pd.to_numeric(
        eligible["clinical_to_mri_gap_days"],
        errors="coerce",
    )

    print("\nELIGIBLE CDR=0 LANDMARK GAP WINDOWS")

    print(
        "<= 30 days  :",
        int((egap <= 30).sum())
    )

    print(
        "<= 90 days  :",
        int((egap <= 90).sum())
    )

    print(
        "<= 180 days :",
        int((egap <= 180).sum())
    )

    print(
        "<= 365 days :",
        int((egap <= 365).sum())
    )

    print(
        "> 365 days  :",
        int((egap > 365).sum())
    )


# ============================================================
# 14. OUTPUT LOCATIONS
# ============================================================

print("\nOUTPUT:")
print(OUTPUT_FILE)

print("\nELIGIBLE LANDMARKS:")
print(ELIGIBLE_FILE)


# ============================================================
# 15. SAFETY STATEMENT
# ============================================================

print("\nIMPORTANT:")
print("No MRI files were modified.")
print("No prediction files were read.")
print("No frozen ADNI model inference was performed.")
print("No external model performance was examined.")
print(
    "This step only audits temporal alignment between "
    "locked MRI acquisitions and the locked CDR timeline."
)

print("\n" + "=" * 72)
print("STEP 46 PASSED — MRI/CDR LANDMARK ELIGIBILITY AUDITED")
print("=" * 72)