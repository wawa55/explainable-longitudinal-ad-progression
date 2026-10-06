from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ==============================================================================================================
# ADNI XAI TEST COHORT LOCK
#
# PURPOSE
# -------
# Freeze a representative TEST XAI cohort BEFORE generating any TEST attribution maps.
#
# This script:
#   1. Verifies the frozen XAI production configuration.
#   2. Loads TEST metadata only.
#   3. Selects one deterministic landmark per TEST participant (latest landmark).
#   4. Selects a fixed stratified cohort by event status and MRI-history size.
#   5. Saves the cohort + hashes.
#
# IMPORTANT
# ---------
# This script does NOT load MRI volumes.
# This script does NOT load model checkpoints.
# This script does NOT run inference.
# This script does NOT generate XAI.
# ==============================================================================================================


# ==============================================================================================================
# 1. PATHS
# ==============================================================================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "06_splits"
RESULT_DIR = ROOT / "08_results"

LONGITUDINAL_DATASET_FILE = (
    SPLIT_DIR / "ADNI_LONGITUDINAL_LANDMARK_DATASET_V1.csv"
)

XAI_CONFIG_LOCK = (
    RESULT_DIR / "ADNI_XAI_PRODUCTION_CONFIG_LOCK_V1.json"
)

XAI_CONFIG_LOCK_HASHES = (
    RESULT_DIR / "ADNI_XAI_PRODUCTION_CONFIG_LOCK_HASHES_V1.csv"
)

OUTPUT_CSV = (
    RESULT_DIR / "ADNI_XAI_FROZEN_TEST_COHORT_V1.csv"
)

OUTPUT_JSON = (
    RESULT_DIR / "ADNI_XAI_FROZEN_TEST_COHORT_V1.json"
)

OUTPUT_TXT = (
    RESULT_DIR / "ADNI_XAI_FROZEN_TEST_COHORT_V1.txt"
)

OUTPUT_HASHES = (
    RESULT_DIR / "ADNI_XAI_FROZEN_TEST_COHORT_HASHES_V1.csv"
)


# ==============================================================================================================
# 2. EXPECTED TEST / XAI VALUES
# ==============================================================================================================

EXPECTED_TEST_LANDMARKS = 583
EXPECTED_TEST_SUBJECTS = 146
EXPECTED_TEST_EVENT_SUBJECTS = 49
EXPECTED_TEST_CENSORED_SUBJECTS = 97

EXPECTED_XAI_BASELINE = "blurred latest MRI"
EXPECTED_XAI_STEPS = 16
EXPECTED_XAI_HORIZONS = [24, 36]
EXPECTED_XAI_SEEDS = [2026, 3407, 5813, 7291, 9109]

# 24 participants total:
#   12 event + 12 censored
#   target 4 per MRI-history bin within each outcome class
TARGET_PER_CLASS = 12
TARGET_PER_BIN = 4

HISTORY_BIN_ORDER = [
    "2",
    "3-4",
    "5+",
]


# ==============================================================================================================
# 3. HELPERS
# ==============================================================================================================

def sha256_file(path, chunk_size=1024 * 1024):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def require_file(path, label):

    if not path.exists():

        raise FileNotFoundError(
            f"{label} missing:\n{path}"
        )

    print(
        f"{label:45s}: PASS | {path}"
    )


def history_bin(n_mri):

    n_mri = int(n_mri)

    if n_mri == 2:
        return "2"

    if 3 <= n_mri <= 4:
        return "3-4"

    return "5+"


def recency_bin(days_since_latest_mri):

    value = float(days_since_latest_mri)

    if value <= 183:
        return "<=6m"

    if value <= 365:
        return "6-12m"

    return ">12m"


def deterministic_stratified_select(subject_df):

    selected_indices = []

    for event_value in [0, 1]:

        class_df = (
            subject_df.loc[
                subject_df["event"].astype(int)
                ==
                event_value
            ]
            .copy()
            .sort_values(
                [
                    "history_bin",
                    "subject_id",
                    "landmark_id",
                ]
            )
        )

        class_selected = []

        # First enforce MRI-history coverage.
        for bin_name in HISTORY_BIN_ORDER:

            bin_df = (
                class_df.loc[
                    class_df["history_bin"]
                    ==
                    bin_name
                ]
                .copy()
                .sort_values(
                    [
                        "subject_id",
                        "landmark_id",
                    ]
                )
            )

            take = min(
                TARGET_PER_BIN,
                len(bin_df),
            )

            class_selected.extend(
                bin_df.head(take).index.tolist()
            )

        # If any bin is undersupplied, fill deterministically from remaining participants.
        if len(class_selected) < TARGET_PER_CLASS:

            remaining = (
                class_df.loc[
                    ~class_df.index.isin(
                        class_selected
                    )
                ]
                .copy()
                .sort_values(
                    [
                        "history_bin",
                        "subject_id",
                        "landmark_id",
                    ]
                )
            )

            need = (
                TARGET_PER_CLASS
                -
                len(class_selected)
            )

            class_selected.extend(
                remaining.head(need).index.tolist()
            )

        class_selected = class_selected[:TARGET_PER_CLASS]

        if len(class_selected) != TARGET_PER_CLASS:

            raise RuntimeError(
                f"Unable to select {TARGET_PER_CLASS} "
                f"participants for event={event_value}."
            )

        selected_indices.extend(
            class_selected
        )

    selected = (
        subject_df.loc[
            selected_indices
        ]
        .copy()
        .sort_values(
            [
                "event",
                "history_bin",
                "subject_id",
                "landmark_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return selected


# ==============================================================================================================
# 4. MAIN
# ==============================================================================================================

def main():

    print("=" * 120)
    print("ADNI XAI TEST COHORT LOCK")
    print("=" * 120)

    print(
        "Purpose: freeze TEST XAI cohort before generating TEST attribution maps"
    )

    print(
        "MRI volumes loaded:",
        "NO"
    )

    print(
        "Model inference performed:",
        "NO"
    )

    print(
        "TEST attribution generated:",
        "NO"
    )

    print()

    # ----------------------------------------------------------------------------------------------------------
    # 4A. FILE AUDIT
    # ----------------------------------------------------------------------------------------------------------

    print("FILE AUDIT")
    print("-" * 120)

    require_file(
        LONGITUDINAL_DATASET_FILE,
        "Longitudinal dataset",
    )

    require_file(
        XAI_CONFIG_LOCK,
        "Frozen XAI production lock",
    )

    require_file(
        XAI_CONFIG_LOCK_HASHES,
        "Frozen XAI lock hashes",
    )

    # ----------------------------------------------------------------------------------------------------------
    # 4B. XAI LOCK AUDIT
    # ----------------------------------------------------------------------------------------------------------

    print()
    print("XAI CONFIGURATION LOCK AUDIT")
    print("-" * 120)

    with open(
        XAI_CONFIG_LOCK,
        "r",
        encoding="utf-8",
    ) as f:

        xai_lock = json.load(f)

    observed_baseline = xai_lock.get(
        "baseline"
    )

    observed_steps = int(
        xai_lock.get(
            "n_steps"
        )
    )

    observed_horizons = [
        int(x)
        for x in xai_lock.get(
            "horizons_months",
            [],
        )
    ]

    observed_seeds = [
        int(x)
        for x in xai_lock.get(
            "cross_seed_models",
            [],
        )
    ]

    print(
        "Baseline:",
        observed_baseline
    )

    print(
        "IG steps:",
        observed_steps
    )

    print(
        "Horizons:",
        observed_horizons
    )

    print(
        "Seeds:",
        observed_seeds
    )

    if observed_baseline != EXPECTED_XAI_BASELINE:

        raise RuntimeError(
            "Frozen XAI baseline mismatch."
        )

    if observed_steps != EXPECTED_XAI_STEPS:

        raise RuntimeError(
            "Frozen XAI step count mismatch."
        )

    if observed_horizons != EXPECTED_XAI_HORIZONS:

        raise RuntimeError(
            "Frozen XAI horizons mismatch."
        )

    if observed_seeds != EXPECTED_XAI_SEEDS:

        raise RuntimeError(
            "Frozen XAI seed set mismatch."
        )

    print(
        "XAI lock audit:",
        "PASS"
    )

    # ----------------------------------------------------------------------------------------------------------
    # 4C. LOAD TEST METADATA ONLY
    # ----------------------------------------------------------------------------------------------------------

    print()
    print("TEST METADATA AUDIT")
    print("-" * 120)

    df = pd.read_csv(
        LONGITUDINAL_DATASET_FILE
    )

    test_df = (
        df.loc[
            df["split"]
            .astype(str)
            .str.upper()
            ==
            "TEST"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test_landmarks = len(
        test_df
    )

    test_subjects = int(
        test_df[
            "subject_id"
        ].nunique()
    )

    subject_event = (
        test_df.groupby(
            "subject_id",
            as_index=False,
        )["event"]
        .max()
    )

    event_subjects = int(
        subject_event[
            "event"
        ]
        .astype(int)
        .sum()
    )

    censored_subjects = (
        test_subjects
        -
        event_subjects
    )

    print(
        "TEST landmarks:",
        test_landmarks
    )

    print(
        "TEST participants:",
        test_subjects
    )

    print(
        "TEST event participants:",
        event_subjects
    )

    print(
        "TEST censored participants:",
        censored_subjects
    )

    if test_landmarks != EXPECTED_TEST_LANDMARKS:

        raise RuntimeError(
            "TEST landmark count mismatch."
        )

    if test_subjects != EXPECTED_TEST_SUBJECTS:

        raise RuntimeError(
            "TEST participant count mismatch."
        )

    if event_subjects != EXPECTED_TEST_EVENT_SUBJECTS:

        raise RuntimeError(
            "TEST event participant count mismatch."
        )

    if censored_subjects != EXPECTED_TEST_CENSORED_SUBJECTS:

        raise RuntimeError(
            "TEST censored participant count mismatch."
        )

    print(
        "TEST metadata audit:",
        "PASS"
    )

    # ----------------------------------------------------------------------------------------------------------
    # 4D. ONE LANDMARK PER SUBJECT
    # ----------------------------------------------------------------------------------------------------------

    print()
    print("SUBJECT-LEVEL LANDMARK REDUCTION")
    print("-" * 120)

    test_df[
        "landmark_date_sort"
    ] = pd.to_datetime(
        test_df[
            "landmark_date"
        ],
        errors="coerce",
    )

    # Deterministic: latest TEST landmark per participant.
    subject_df = (
        test_df.sort_values(
            [
                "subject_id",
                "landmark_date_sort",
                "landmark_id",
            ]
        )
        .groupby(
            "subject_id",
            as_index=False,
        )
        .tail(
            1
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(subject_df) != EXPECTED_TEST_SUBJECTS:

        raise RuntimeError(
            "Latest-landmark reduction did not produce 146 participants."
        )

    subject_df[
        "history_bin"
    ] = subject_df[
        "n_mri"
    ].map(
        history_bin
    )

    if "days_since_latest_mri" in subject_df.columns:

        subject_df[
            "recency_bin"
        ] = subject_df[
            "days_since_latest_mri"
        ].map(
            recency_bin
        )

    else:

        subject_df[
            "recency_bin"
        ] = "not_available"

    print(
        "One latest landmark per participant:",
        len(subject_df)
    )

    print()

    print(
        "Available participants by event/history:"
    )

    print(
        subject_df.groupby(
            [
                "event",
                "history_bin",
            ]
        )
        .size()
        .rename(
            "n"
        )
        .to_string()
    )

    # ----------------------------------------------------------------------------------------------------------
    # 4E. FREEZE STRATIFIED TEST XAI COHORT
    # ----------------------------------------------------------------------------------------------------------

    print()
    print("FROZEN TEST XAI COHORT SELECTION")
    print("-" * 120)

    selected = deterministic_stratified_select(
        subject_df
    )

    expected_total = (
        2
        *
        TARGET_PER_CLASS
    )

    if len(selected) != expected_total:

        raise RuntimeError(
            f"Expected {expected_total} selected participants, "
            f"found {len(selected)}."
        )

    if (
        selected[
            "subject_id"
        ].nunique()
        !=
        expected_total
    ):

        raise RuntimeError(
            "Repeated participant detected in frozen TEST XAI cohort."
        )

    selected_events = int(
        selected[
            "event"
        ]
        .astype(int)
        .sum()
    )

    selected_censored = (
        expected_total
        -
        selected_events
    )

    if selected_events != TARGET_PER_CLASS:

        raise RuntimeError(
            "Selected event participant count mismatch."
        )

    if selected_censored != TARGET_PER_CLASS:

        raise RuntimeError(
            "Selected censored participant count mismatch."
        )

    selected[
        "xai_cohort_version"
    ] = "ADNI_XAI_TEST_COHORT_V1_2026-09-11"

    selected[
        "selection_rule"
    ] = (
        "latest TEST landmark per participant; "
        "deterministic stratification by event status and MRI-history size"
    )

    selected[
        "xai_attribution_generated"
    ] = False

    selection_columns = [
        "subject_id",
        "landmark_id",
        "landmark_date",
        "event",
        "n_mri",
        "history_bin",
        "recency_bin",
        "time_to_event_or_censor_days",
        "time_to_event_or_censor_years",
        "image_ids",
        "image_dates",
        "days_from_first_mri",
        "days_before_landmark",
        "image_paths",
        "phases_in_history",
        "scanner_change_in_history",
        "xai_cohort_version",
        "selection_rule",
        "xai_attribution_generated",
    ]

    existing_columns = [
        c
        for c in selection_columns
        if c in selected.columns
    ]

    selected[
        existing_columns
    ].to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        selected[
            [
                "subject_id",
                "landmark_id",
                "event",
                "n_mri",
                "history_bin",
                "recency_bin",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Selected cohort by event/history:"
    )

    print(
        selected.groupby(
            [
                "event",
                "history_bin",
            ]
        )
        .size()
        .rename(
            "n"
        )
        .to_string()
    )

    # ----------------------------------------------------------------------------------------------------------
    # 4F. SAVE JSON/TXT LOCK
    # ----------------------------------------------------------------------------------------------------------

    cohort_lock = {
        "lock_version":
            "ADNI_XAI_TEST_COHORT_V1_2026-09-11",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "FROZEN BEFORE TEST XAI ATTRIBUTION",

        "source_dataset":
            str(
                LONGITUDINAL_DATASET_FILE
            ),

        "xai_config_lock":
            str(
                XAI_CONFIG_LOCK
            ),

        "full_test_support":
            {
                "landmarks":
                    test_landmarks,

                "participants":
                    test_subjects,

                "event_participants":
                    event_subjects,

                "censored_participants":
                    censored_subjects,
            },

        "selection":
            {
                "landmark_rule":
                    "latest TEST landmark per participant",

                "stratification":
                    [
                        "event status",
                        "MRI history size: 2, 3-4, 5+ scans",
                    ],

                "target_per_event_class":
                    TARGET_PER_CLASS,

                "target_per_history_bin_within_class":
                    TARGET_PER_BIN,

                "selected_participants":
                    expected_total,

                "selected_event_participants":
                    selected_events,

                "selected_censored_participants":
                    selected_censored,

                "deterministic":
                    True,

                "attribution_results_used_for_selection":
                    False,

                "model_predictions_used_for_selection":
                    False,
            },

        "frozen_xai_spec":
            {
                "baseline":
                    observed_baseline,

                "n_steps":
                    observed_steps,

                "horizons_months":
                    observed_horizons,

                "seeds":
                    observed_seeds,
            },

        "test_mri_loaded":
            False,

        "test_model_inference_performed":
            False,

        "test_attribution_generated":
            False,

        "participants":
            selected[
                existing_columns
            ].to_dict(
                orient="records"
            ),
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            cohort_lock,
            f,
            indent=2,
        )

    text_lines = [
        "=" * 100,
        "ADNI XAI FROZEN TEST COHORT — V1",
        "=" * 100,
        "",
        "STATUS",
        "------",
        "Frozen before TEST attribution generation.",
        "MRI loaded: NO",
        "Model inference: NO",
        "Attribution generated: NO",
        "",
        "FULL TEST SUPPORT",
        "-----------------",
        f"Landmarks: {test_landmarks}",
        f"Participants: {test_subjects}",
        f"Event participants: {event_subjects}",
        f"Censored participants: {censored_subjects}",
        "",
        "FROZEN XAI COHORT",
        "-----------------",
        f"Selected participants: {expected_total}",
        f"Events: {selected_events}",
        f"Censored: {selected_censored}",
        (
            "Selection rule: latest TEST landmark per participant, "
            "then deterministic stratification by event status and "
            "MRI-history size (2, 3-4, 5+ scans)."
        ),
        "No attribution maps or model predictions were used for selection.",
        "",
        "FROZEN XAI CONFIG",
        "-----------------",
        f"Baseline: {observed_baseline}",
        f"IG steps: {observed_steps}",
        f"Horizons: {observed_horizons}",
        f"Seeds: {observed_seeds}",
        "",
        "NEXT PHASE",
        "----------",
        "Five-seed TEST XAI attribution generation on this frozen cohort.",
        "",
        "=" * 100,
    ]

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(
                text_lines
            )
        )

    # ----------------------------------------------------------------------------------------------------------
    # 4G. HASH LOCK ARTIFACTS
    # ----------------------------------------------------------------------------------------------------------

    hash_paths = [
        LONGITUDINAL_DATASET_FILE,
        XAI_CONFIG_LOCK,
        XAI_CONFIG_LOCK_HASHES,
        OUTPUT_CSV,
        OUTPUT_JSON,
        OUTPUT_TXT,
    ]

    hash_rows = []

    for artifact_path in hash_paths:

        hash_rows.append(
            {
                "artifact":
                    artifact_path.name,

                "path":
                    str(
                        artifact_path
                    ),

                "sha256":
                    sha256_file(
                        artifact_path
                    ),
            }
        )

    with open(
        OUTPUT_HASHES,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "artifact",
                "path",
                "sha256",
            ],
        )

        writer.writeheader()

        writer.writerows(
            hash_rows
        )

    # ----------------------------------------------------------------------------------------------------------
    # 4H. FINAL STATUS
    # ----------------------------------------------------------------------------------------------------------

    print()
    print(
        "Saved cohort CSV :",
        OUTPUT_CSV
    )

    print(
        "Saved cohort JSON:",
        OUTPUT_JSON
    )

    print(
        "Saved cohort TXT :",
        OUTPUT_TXT
    )

    print(
        "Saved hashes     :",
        OUTPUT_HASHES
    )

    print()
    print("=" * 120)
    print("XAI TEST COHORT LOCK: COMPLETE")
    print("=" * 120)

    print(
        "Selected participants:",
        expected_total
    )

    print(
        "Events / censored:",
        f"{selected_events} / {selected_censored}"
    )

    print(
        "Selection used model predictions:",
        "NO"
    )

    print(
        "Selection used attribution results:",
        "NO"
    )

    print(
        "TEST MRI loaded:",
        "NO"
    )

    print(
        "TEST inference:",
        "NO"
    )

    print(
        "TEST XAI:",
        "NOT YET RUN"
    )

    print(
        "Next step:",
        "FIVE-SEED TEST XAI ON FROZEN COHORT"
    )

    print("=" * 120)


if __name__ == "__main__":

    main()
