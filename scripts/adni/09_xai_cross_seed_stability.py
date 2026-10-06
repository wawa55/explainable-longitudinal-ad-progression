from pathlib import Path
import ast
import gc
import hashlib
import json
import time
import traceback

import numpy as np
import pandas as pd
import nibabel as nib


# ==============================================================================================================
# ADNI — CROSS-SEED XAI STABILITY ANALYSIS
#
# PURPOSE
# -------
# Quantify reproducibility of the already-generated TEST attribution maps across the five frozen seeds.
#
# This script:
#   - DOES NOT load model checkpoints.
#   - DOES NOT run model inference.
#   - DOES NOT run Integrated Gradients.
#   - DOES NOT change the frozen XAI configuration.
#
# Analyses:
#   1. Integrity/hash audit of all 240 attribution files.
#   2. Exact voxelwise Spearman correlation of |IG| across seed pairs within the valid brain mask.
#   3. Pairwise top-k Dice overlap for k = 1%, 5%, 10%.
#   4. Participant-level summaries.
#   5. Cohort-level 24m / 36m summaries.
#   6. Event vs censored descriptive summaries.
# ==============================================================================================================


# ==============================================================================================================
# 1. PATHS
# ==============================================================================================================

ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = ROOT / "08_results"

PRODUCTION_DIR = (
    RESULT_DIR / "ADNI_XAI_TEST_PRODUCTION_V1"
)

MASTER_CSV = (
    PRODUCTION_DIR / "ADNI_XAI_TEST_PRODUCTION_MASTER_V1.csv"
)

PRODUCTION_SUMMARY_JSON = (
    PRODUCTION_DIR / "ADNI_XAI_TEST_PRODUCTION_SUMMARY_V1.json"
)

COHORT_CSV = (
    RESULT_DIR / "ADNI_XAI_FROZEN_TEST_COHORT_V1.csv"
)

XAI_CONFIG_LOCK = (
    RESULT_DIR / "ADNI_XAI_PRODUCTION_CONFIG_LOCK_V1.json"
)

OUTPUT_PAIRWISE_CSV = (
    PRODUCTION_DIR / "ADNI_XAI_CROSS_SEED_PAIRWISE_STABILITY_V1.csv"
)

OUTPUT_SUBJECT_CSV = (
    PRODUCTION_DIR / "ADNI_XAI_CROSS_SEED_SUBJECT_SUMMARY_V1.csv"
)

OUTPUT_COHORT_CSV = (
    PRODUCTION_DIR / "ADNI_XAI_CROSS_SEED_COHORT_SUMMARY_V1.csv"
)

OUTPUT_EVENT_CSV = (
    PRODUCTION_DIR / "ADNI_XAI_CROSS_SEED_EVENT_STATUS_SUMMARY_V1.csv"
)

OUTPUT_JSON = (
    PRODUCTION_DIR / "ADNI_XAI_CROSS_SEED_STABILITY_SUMMARY_V1.json"
)

OUTPUT_LOG = (
    PRODUCTION_DIR / "ADNI_XAI_CROSS_SEED_STABILITY_V1.log"
)


# ==============================================================================================================
# 2. FROZEN SPEC
# ==============================================================================================================

SEEDS = [
    2026,
    3407,
    5813,
    7291,
    9109,
]

HORIZONS = [
    24,
    36,
]

TOP_K_PERCENTS = [
    1,
    5,
    10,
]

EXPECTED_SUBJECTS = 24
EXPECTED_RUNS = 240
EXPECTED_PAIRWISE_PER_SUBJECT_HORIZON = 10

SOURCE_SPATIAL_SHAPE = (
    197,
    233,
    189,
)

MODEL_SPATIAL_SHAPE = (
    160,
    192,
    160,
)

BRAIN_EPS = 1e-6


# ==============================================================================================================
# 3. LOGGING
# ==============================================================================================================

def log(message=""):

    text = str(message)

    print(
        text,
        flush=True,
    )

    with open(
        OUTPUT_LOG,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            text + "\n"
        )

        f.flush()


OUTPUT_LOG.write_text(
    "",
    encoding="utf-8",
)


# ==============================================================================================================
# 4. HELPERS
# ==============================================================================================================

def sha256_file(
    path,
    chunk_size=1024 * 1024,
):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def require_file(
    path,
    label,
):

    if not path.exists():

        raise FileNotFoundError(
            f"{label} missing:\n{path}"
        )

    log(
        f"{label:45s}: PASS | {path}"
    )


def parse_list_field(
    value,
):

    if isinstance(
        value,
        list,
    ):

        return value

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()

    if pd.isna(
        value
    ):

        return []

    text = str(
        value
    ).strip()

    if not text:

        return []

    try:

        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            list,
        ):

            return parsed

    except Exception:
        pass

    try:

        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            (
                list,
                tuple,
            ),
        ):

            return list(
                parsed
            )

    except Exception:
        pass

    raise ValueError(
        f"Unable to parse list field: {text[:300]}"
    )


def center_crop_3d(
    volume,
    target_shape=MODEL_SPATIAL_SHAPE,
):

    source_shape = np.asarray(
        volume.shape,
        dtype=int,
    )

    target_shape = np.asarray(
        target_shape,
        dtype=int,
    )

    if np.any(
        source_shape
        <
        target_shape
    ):

        raise RuntimeError(
            f"Source shape {tuple(source_shape)} "
            f"is smaller than target {tuple(target_shape)}."
        )

    start = (
        source_shape
        -
        target_shape
    ) // 2

    end = (
        start
        +
        target_shape
    )

    cropped = volume[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2],
    ]

    if (
        cropped.shape
        !=
        tuple(
            target_shape
        )
    ):

        raise RuntimeError(
            f"Crop shape mismatch: "
            f"{cropped.shape} != "
            f"{tuple(target_shape)}"
        )

    return cropped


def latest_mri_brain_mask(
    latest_path,
):

    latest_path = Path(
        latest_path
    )

    if not latest_path.exists():

        raise FileNotFoundError(
            f"Latest MRI missing:\n{latest_path}"
        )

    nii = nib.load(
        str(
            latest_path
        )
    )

    volume = np.asarray(
        nii.dataobj,
        dtype=np.float32,
    )

    if (
        volume.shape
        !=
        SOURCE_SPATIAL_SHAPE
    ):

        raise RuntimeError(
            f"Unexpected latest MRI shape "
            f"{volume.shape}; expected "
            f"{SOURCE_SPATIAL_SHAPE}"
        )

    cropped = center_crop_3d(
        volume
    )

    finite = np.isfinite(
        cropped
    )

    non_background = np.abs(
        cropped
    ) > BRAIN_EPS

    mask = (
        finite
        &
        non_background
    )

    if (
        mask.sum()
        <
        1000
    ):

        raise RuntimeError(
            f"Brain mask unexpectedly small: "
            f"{int(mask.sum())} voxels"
        )

    return mask


def average_rank(
    values,
):

    # Exact average ranks with tie handling.
    # Returns ranks 1..N, averaged across ties.
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = values.size

    order = np.argsort(
        values,
        kind="mergesort",
    )

    sorted_values = values[
        order
    ]

    ranks_sorted = np.empty(
        n,
        dtype=np.float64,
    )

    start = 0

    while start < n:

        end = start + 1

        while (
            end < n
            and
            sorted_values[end]
            ==
            sorted_values[start]
        ):

            end += 1

        # Average 1-based rank for this tie group.
        average = (
            (
                start + 1
            )
            +
            end
        ) / 2.0

        ranks_sorted[
            start:end
        ] = average

        start = end

    ranks = np.empty(
        n,
        dtype=np.float64,
    )

    ranks[
        order
    ] = ranks_sorted

    return ranks


def spearman_exact(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if (
        x.size
        !=
        y.size
    ):

        raise RuntimeError(
            "Spearman vectors have unequal length."
        )

    if (
        x.size
        <
        2
    ):

        return np.nan

    rx = average_rank(
        x
    )

    ry = average_rank(
        y
    )

    rx_mean = rx.mean()
    ry_mean = ry.mean()

    dx = (
        rx
        -
        rx_mean
    )

    dy = (
        ry
        -
        ry_mean
    )

    denominator = np.sqrt(
        np.sum(
            dx * dx
        )
        *
        np.sum(
            dy * dy
        )
    )

    if (
        denominator
        <=
        0
    ):

        return np.nan

    correlation = float(
        np.sum(
            dx * dy
        )
        /
        denominator
    )

    return correlation


def top_k_mask(
    values,
    valid_mask,
    percent,
):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    valid_indices = np.flatnonzero(
        valid_mask.ravel()
    )

    valid_values = values.ravel()[
        valid_indices
    ]

    n_valid = valid_values.size

    n_select = max(
        1,
        int(
            np.ceil(
                n_valid
                *
                (
                    float(
                        percent
                    )
                    /
                    100.0
                )
            )
        ),
    )

    if (
        n_select
        >=
        n_valid
    ):

        chosen = valid_indices

    else:

        # argpartition is exact for top-k membership and substantially faster than a full sort.
        local_idx = np.argpartition(
            valid_values,
            -n_select,
        )[
            -n_select:
        ]

        chosen = valid_indices[
            local_idx
        ]

    mask = np.zeros(
        values.size,
        dtype=bool,
    )

    mask[
        chosen
    ] = True

    return mask.reshape(
        values.shape
    )


def dice_coefficient(
    mask_a,
    mask_b,
):

    mask_a = np.asarray(
        mask_a,
        dtype=bool,
    )

    mask_b = np.asarray(
        mask_b,
        dtype=bool,
    )

    a = int(
        mask_a.sum()
    )

    b = int(
        mask_b.sum()
    )

    denominator = (
        a
        +
        b
    )

    if (
        denominator
        ==
        0
    ):

        return np.nan

    intersection = int(
        np.logical_and(
            mask_a,
            mask_b,
        ).sum()
    )

    return float(
        2.0
        *
        intersection
        /
        denominator
    )


# ==============================================================================================================
# 5. MAIN
# ==============================================================================================================

def main():

    log("=" * 120)
    log("ADNI XAI — CROSS-SEED STABILITY ANALYSIS")
    log("=" * 120)

    log(
        "Model inference performed: NO"
    )

    log(
        "Integrated Gradients performed: NO"
    )

    log(
        "Analysis input: existing 240 frozen TEST attribution files"
    )

    log("")

    # ----------------------------------------------------------------------------------------------------------
    # 5A. FILE AUDIT
    # ----------------------------------------------------------------------------------------------------------

    log("FILE AUDIT")
    log("-" * 120)

    require_file(
        MASTER_CSV,
        "Production master CSV",
    )

    require_file(
        PRODUCTION_SUMMARY_JSON,
        "Production summary JSON",
    )

    require_file(
        COHORT_CSV,
        "Frozen TEST cohort CSV",
    )

    require_file(
        XAI_CONFIG_LOCK,
        "Frozen XAI config lock",
    )

    # ----------------------------------------------------------------------------------------------------------
    # 5B. PRODUCTION MASTER AUDIT
    # ----------------------------------------------------------------------------------------------------------

    log("")
    log("PRODUCTION MASTER AUDIT")
    log("-" * 120)

    master = pd.read_csv(
        MASTER_CSV
    )

    complete = master.loc[
        master[
            "status"
        ]
        ==
        "COMPLETE"
    ].copy()

    key_df = (
        complete[
            [
                "subject_id",
                "seed",
                "horizon_months",
            ]
        ]
        .drop_duplicates()
    )

    if (
        len(
            key_df
        )
        !=
        EXPECTED_RUNS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} complete unique runs; "
            f"found {len(key_df)}."
        )

    observed_subjects = int(
        complete[
            "subject_id"
        ].nunique()
    )

    observed_seeds = sorted(
        complete[
            "seed"
        ]
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    observed_horizons = sorted(
        complete[
            "horizon_months"
        ]
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    log(
        f"Complete runs: {len(key_df)}"
    )

    log(
        f"Subjects: {observed_subjects}"
    )

    log(
        f"Seeds: {observed_seeds}"
    )

    log(
        f"Horizons: {observed_horizons}"
    )

    if (
        observed_subjects
        !=
        EXPECTED_SUBJECTS
    ):

        raise RuntimeError(
            "Subject count mismatch."
        )

    if (
        observed_seeds
        !=
        SEEDS
    ):

        raise RuntimeError(
            "Seed set mismatch."
        )

    if (
        observed_horizons
        !=
        HORIZONS
    ):

        raise RuntimeError(
            "Horizon set mismatch."
        )

    # ----------------------------------------------------------------------------------------------------------
    # 5C. HASH / FILE INTEGRITY AUDIT
    # ----------------------------------------------------------------------------------------------------------

    log("")
    log("ATTRIBUTION FILE INTEGRITY AUDIT")
    log("-" * 120)

    hash_failures = []
    missing_files = []
    shape_failures = []

    for _, row in complete.iterrows():

        attr_path = Path(
            str(
                row[
                    "attribution_file"
                ]
            )
        )

        if not attr_path.exists():

            missing_files.append(
                str(
                    attr_path
                )
            )

            continue

        observed_hash = sha256_file(
            attr_path
        )

        expected_hash = str(
            row[
                "attribution_sha256"
            ]
        )

        if (
            observed_hash
            !=
            expected_hash
        ):

            hash_failures.append(
                str(
                    attr_path
                )
            )

            continue

        with np.load(
            attr_path,
            allow_pickle=False,
        ) as data:

            attribution = data[
                "attribution"
            ]

            if (
                attribution.shape
                !=
                MODEL_SPATIAL_SHAPE
            ):

                shape_failures.append(
                    str(
                        attr_path
                    )
                )

    log(
        f"Missing files: {len(missing_files)}"
    )

    log(
        f"Hash failures: {len(hash_failures)}"
    )

    log(
        f"Shape failures: {len(shape_failures)}"
    )

    if (
        missing_files
        or
        hash_failures
        or
        shape_failures
    ):

        raise RuntimeError(
            "Attribution integrity audit failed."
        )

    log(
        "Attribution integrity audit: PASS"
    )

    # ----------------------------------------------------------------------------------------------------------
    # 5D. COHORT LOOKUP
    # ----------------------------------------------------------------------------------------------------------

    cohort = pd.read_csv(
        COHORT_CSV
    )

    if (
        cohort[
            "subject_id"
        ].nunique()
        !=
        EXPECTED_SUBJECTS
    ):

        raise RuntimeError(
            "Frozen cohort subject count mismatch."
        )

    cohort = (
        cohort.set_index(
            "subject_id",
            drop=False,
        )
    )

    # ----------------------------------------------------------------------------------------------------------
    # 5E. CROSS-SEED STABILITY
    # ----------------------------------------------------------------------------------------------------------

    log("")
    log("CROSS-SEED STABILITY")
    log("-" * 120)

    pairwise_records = []

    subjects = sorted(
        complete[
            "subject_id"
        ]
        .astype(
            str
        )
        .unique()
        .tolist()
    )

    total_subject_horizons = (
        len(
            subjects
        )
        *
        len(
            HORIZONS
        )
    )

    current = 0

    for subject_id in subjects:

        if (
            subject_id
            not in
            cohort.index
        ):

            raise RuntimeError(
                f"{subject_id}: missing from frozen cohort CSV."
            )

        cohort_row = cohort.loc[
            subject_id
        ]

        image_paths = parse_list_field(
            cohort_row[
                "image_paths"
            ]
        )

        if (
            len(
                image_paths
            )
            <
            1
        ):

            raise RuntimeError(
                f"{subject_id}: no MRI paths available."
            )

        latest_path = image_paths[
            -1
        ]

        brain_mask = latest_mri_brain_mask(
            latest_path
        )

        n_brain_voxels = int(
            brain_mask.sum()
        )

        event_value = int(
            cohort_row[
                "event"
            ]
        )

        n_mri = int(
            cohort_row[
                "n_mri"
            ]
        )

        history_bin = str(
            cohort_row[
                "history_bin"
            ]
        )

        for horizon in HORIZONS:

            current += 1

            log(
                f"[{current:02d}/{total_subject_horizons}] "
                f"{subject_id} | "
                f"{horizon}m | "
                f"brain_voxels={n_brain_voxels}"
            )

            maps = {}

            for seed in SEEDS:

                row = complete.loc[
                    (
                        complete[
                            "subject_id"
                        ].astype(
                            str
                        )
                        ==
                        subject_id
                    )
                    &
                    (
                        complete[
                            "seed"
                        ].astype(
                            int
                        )
                        ==
                        seed
                    )
                    &
                    (
                        complete[
                            "horizon_months"
                        ].astype(
                            int
                        )
                        ==
                        horizon
                    )
                ]

                if (
                    len(
                        row
                    )
                    !=
                    1
                ):

                    raise RuntimeError(
                        f"{subject_id}, seed={seed}, horizon={horizon}: "
                        f"expected one master row, found {len(row)}."
                    )

                attr_path = Path(
                    str(
                        row.iloc[
                            0
                        ][
                            "attribution_file"
                        ]
                    )
                )

                with np.load(
                    attr_path,
                    allow_pickle=False,
                ) as data:

                    attribution = data[
                        "attribution"
                    ].astype(
                        np.float32,
                        copy=False,
                    )

                maps[
                    seed
                ] = np.abs(
                    attribution
                )

            # Precompute valid voxel vectors and top-k masks.
            valid_vectors = {
                seed:
                    maps[
                        seed
                    ][
                        brain_mask
                    ]
                    .astype(
                        np.float32,
                        copy=False,
                    )
                for seed in SEEDS
            }

            top_masks = {
                (
                    seed,
                    percent,
                ):
                    top_k_mask(
                        maps[
                            seed
                        ],
                        brain_mask,
                        percent,
                    )
                for seed in SEEDS
                for percent in TOP_K_PERCENTS
            }

            pair_count = 0

            for i in range(
                len(
                    SEEDS
                )
            ):

                for j in range(
                    i + 1,
                    len(
                        SEEDS
                    ),
                ):

                    seed_a = SEEDS[
                        i
                    ]

                    seed_b = SEEDS[
                        j
                    ]

                    pair_count += 1

                    spearman = spearman_exact(
                        valid_vectors[
                            seed_a
                        ],
                        valid_vectors[
                            seed_b
                        ],
                    )

                    dice_values = {}

                    for percent in TOP_K_PERCENTS:

                        dice_values[
                            percent
                        ] = dice_coefficient(
                            top_masks[
                                (
                                    seed_a,
                                    percent,
                                )
                            ],
                            top_masks[
                                (
                                    seed_b,
                                    percent,
                                )
                            ],
                        )

                    pairwise_records.append(
                        {
                            "subject_id":
                                subject_id,

                            "event":
                                event_value,

                            "n_mri":
                                n_mri,

                            "history_bin":
                                history_bin,

                            "horizon_months":
                                horizon,

                            "seed_a":
                                seed_a,

                            "seed_b":
                                seed_b,

                            "brain_voxels":
                                n_brain_voxels,

                            "spearman_abs_attribution":
                                spearman,

                            "dice_top_1pct":
                                dice_values[
                                    1
                                ],

                            "dice_top_5pct":
                                dice_values[
                                    5
                                ],

                            "dice_top_10pct":
                                dice_values[
                                    10
                                ],
                        }
                    )

            if (
                pair_count
                !=
                EXPECTED_PAIRWISE_PER_SUBJECT_HORIZON
            ):

                raise RuntimeError(
                    f"{subject_id}, {horizon}m: "
                    f"expected 10 seed pairs, found {pair_count}."
                )

            # Save incrementally after each subject/horizon.
            pd.DataFrame(
                pairwise_records
            ).to_csv(
                OUTPUT_PAIRWISE_CSV,
                index=False,
            )

            del (
                maps,
                valid_vectors,
                top_masks,
            )

            gc.collect()

        del brain_mask
        gc.collect()

    # ----------------------------------------------------------------------------------------------------------
    # 5F. SUMMARIES
    # ----------------------------------------------------------------------------------------------------------

    pairwise_df = pd.DataFrame(
        pairwise_records
    )

    expected_pairwise_rows = (
        EXPECTED_SUBJECTS
        *
        len(
            HORIZONS
        )
        *
        EXPECTED_PAIRWISE_PER_SUBJECT_HORIZON
    )

    if (
        len(
            pairwise_df
        )
        !=
        expected_pairwise_rows
    ):

        raise RuntimeError(
            f"Expected {expected_pairwise_rows} pairwise rows; "
            f"found {len(pairwise_df)}."
        )

    subject_summary = (
        pairwise_df.groupby(
            [
                "subject_id",
                "event",
                "n_mri",
                "history_bin",
                "horizon_months",
            ],
            as_index=False,
        )
        .agg(
            mean_spearman=(
                "spearman_abs_attribution",
                "mean",
            ),

            median_spearman=(
                "spearman_abs_attribution",
                "median",
            ),

            min_spearman=(
                "spearman_abs_attribution",
                "min",
            ),

            mean_dice_top_1pct=(
                "dice_top_1pct",
                "mean",
            ),

            mean_dice_top_5pct=(
                "dice_top_5pct",
                "mean",
            ),

            mean_dice_top_10pct=(
                "dice_top_10pct",
                "mean",
            ),

            min_dice_top_1pct=(
                "dice_top_1pct",
                "min",
            ),

            min_dice_top_5pct=(
                "dice_top_5pct",
                "min",
            ),

            min_dice_top_10pct=(
                "dice_top_10pct",
                "min",
            ),

            brain_voxels=(
                "brain_voxels",
                "first",
            ),
        )
    )

    subject_summary.to_csv(
        OUTPUT_SUBJECT_CSV,
        index=False,
    )

    cohort_summary = (
        subject_summary.groupby(
            "horizon_months",
            as_index=False,
        )
        .agg(
            subjects=(
                "subject_id",
                "nunique",
            ),

            median_subject_mean_spearman=(
                "mean_spearman",
                "median",
            ),

            q1_subject_mean_spearman=(
                "mean_spearman",
                lambda x: np.quantile(
                    x,
                    0.25,
                ),
            ),

            q3_subject_mean_spearman=(
                "mean_spearman",
                lambda x: np.quantile(
                    x,
                    0.75,
                ),
            ),

            median_subject_mean_dice_1pct=(
                "mean_dice_top_1pct",
                "median",
            ),

            median_subject_mean_dice_5pct=(
                "mean_dice_top_5pct",
                "median",
            ),

            median_subject_mean_dice_10pct=(
                "mean_dice_top_10pct",
                "median",
            ),

            min_subject_mean_spearman=(
                "mean_spearman",
                "min",
            ),

            min_subject_mean_dice_1pct=(
                "mean_dice_top_1pct",
                "min",
            ),

            min_subject_mean_dice_5pct=(
                "mean_dice_top_5pct",
                "min",
            ),

            min_subject_mean_dice_10pct=(
                "mean_dice_top_10pct",
                "min",
            ),
        )
    )

    cohort_summary.to_csv(
        OUTPUT_COHORT_CSV,
        index=False,
    )

    event_summary = (
        subject_summary.groupby(
            [
                "horizon_months",
                "event",
            ],
            as_index=False,
        )
        .agg(
            subjects=(
                "subject_id",
                "nunique",
            ),

            median_mean_spearman=(
                "mean_spearman",
                "median",
            ),

            median_mean_dice_1pct=(
                "mean_dice_top_1pct",
                "median",
            ),

            median_mean_dice_5pct=(
                "mean_dice_top_5pct",
                "median",
            ),

            median_mean_dice_10pct=(
                "mean_dice_top_10pct",
                "median",
            ),
        )
    )

    event_summary.to_csv(
        OUTPUT_EVENT_CSV,
        index=False,
    )

    # ----------------------------------------------------------------------------------------------------------
    # 5G. FINAL JSON
    # ----------------------------------------------------------------------------------------------------------

    output = {
        "analysis":
            "Cross-seed TEST XAI stability",

        "status":
            "COMPLETE",

        "model_inference_performed":
            False,

        "integrated_gradients_performed":
            False,

        "subjects":
            EXPECTED_SUBJECTS,

        "seeds":
            SEEDS,

        "horizons_months":
            HORIZONS,

        "seed_pairs_per_subject_horizon":
            EXPECTED_PAIRWISE_PER_SUBJECT_HORIZON,

        "pairwise_rows":
            int(
                len(
                    pairwise_df
                )
            ),

        "brain_mask_definition":
            "finite non-background voxels (|intensity| > 1e-6) from the subject's cropped latest preprocessed MRI",

        "stability_map":
            "absolute Integrated Gradients attribution",

        "spearman":
            "exact voxelwise Spearman correlation within brain mask",

        "top_k_percentages":
            TOP_K_PERCENTS,

        "cohort_summary":
            cohort_summary.to_dict(
                orient="records"
            ),

        "event_status_summary":
            event_summary.to_dict(
                orient="records"
            ),

        "integrity":
            {
                "missing_files":
                    len(
                        missing_files
                    ),

                "hash_failures":
                    len(
                        hash_failures
                    ),

                "shape_failures":
                    len(
                        shape_failures
                    ),
            },

        "next_phase":
            "attribution-guided faithfulness perturbation",
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # ----------------------------------------------------------------------------------------------------------
    # 5H. REPORT
    # ----------------------------------------------------------------------------------------------------------

    log("")
    log("=" * 120)
    log("CROSS-SEED STABILITY — COHORT SUMMARY")
    log("=" * 120)

    log(
        cohort_summary.to_string(
            index=False
        )
    )

    log("")
    log("=" * 120)
    log("EVENT / CENSORED DESCRIPTIVE SUMMARY")
    log("=" * 120)

    log(
        event_summary.to_string(
            index=False
        )
    )

    log("")
    log(
        f"Saved pairwise CSV : {OUTPUT_PAIRWISE_CSV}"
    )

    log(
        f"Saved subject CSV  : {OUTPUT_SUBJECT_CSV}"
    )

    log(
        f"Saved cohort CSV   : {OUTPUT_COHORT_CSV}"
    )

    log(
        f"Saved event CSV    : {OUTPUT_EVENT_CSV}"
    )

    log(
        f"Saved JSON         : {OUTPUT_JSON}"
    )

    log("")
    log("=" * 120)
    log("CROSS-SEED XAI STABILITY: COMPLETE")
    log("=" * 120)

    log(
        "Model inference rerun: NO"
    )

    log(
        "Integrated Gradients rerun: NO"
    )

    log(
        "Next step: attribution-guided faithfulness perturbation"
    )

    log("=" * 120)


if __name__ == "__main__":

    try:

        main()

    except Exception:

        log("")
        log("=" * 120)
        log("CROSS-SEED XAI STABILITY: FAILED")
        log("=" * 120)

        log(
            traceback.format_exc()
        )

        raise
