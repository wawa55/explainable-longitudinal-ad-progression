from pathlib import Path
import ast
import gc
import hashlib
import json
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn

from exact_m0_model_v3 import ExactM0LongitudinalSurvivalModel_V3


# ==============================================================================================================
# ADNI — ATTRIBUTION-GUIDED FAITHFULNESS PERTURBATION
#
# PURPOSE
# -------
# Test whether voxels identified by the frozen Integrated-Gradients analysis
# actually influence the frozen model's predicted progression risk more than
# matched random within-brain perturbations.
#
# IMPORTANT
# ---------
# - NO Integrated Gradients are rerun here.
# - NO model training or recalibration occurs.
# - Frozen TEST cohort, seeds, horizons, attribution maps, and top-k thresholds are unchanged.
# - Model inference IS required for perturbation-based faithfulness.
#
# Frozen faithfulness rule:
#   * Attribution target: positive-risk IG attribution voxels.
#   * k: top 1%, 5%, 10% of valid brain voxels, capped by number of positive voxels.
#   * Perturbation value: zero (normalized preprocessing background reference).
#   * Random control: equal-sized within-brain voxel sets excluding targeted voxels.
#   * Random repeats: 20 deterministic repeats per subject/seed/horizon/k.
#   * Primary effect: original risk - perturbed risk.
#     Positive values mean perturbation reduced predicted progression risk.
#   * Faithfulness advantage:
#         targeted risk drop - mean random risk drop
#
# This script is resume-safe at the subject/seed/horizon/top-k level.
# ==============================================================================================================


# ==============================================================================================================
# 1. PATHS
# ==============================================================================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "07_models"
RESULT_DIR = ROOT / "08_results"

PRODUCTION_DIR = (
    RESULT_DIR / "ADNI_XAI_TEST_PRODUCTION_V1"
)

MASTER_CSV = (
    PRODUCTION_DIR / "ADNI_XAI_TEST_PRODUCTION_MASTER_V1.csv"
)

STABILITY_SUMMARY_JSON = (
    PRODUCTION_DIR / "ADNI_XAI_CROSS_SEED_STABILITY_SUMMARY_V1.json"
)

COHORT_CSV = (
    RESULT_DIR / "ADNI_XAI_FROZEN_TEST_COHORT_V1.csv"
)

XAI_CONFIG_LOCK = (
    RESULT_DIR / "ADNI_XAI_PRODUCTION_CONFIG_LOCK_V1.json"
)

FAITHFULNESS_RUNS_CSV = (
    PRODUCTION_DIR / "ADNI_XAI_FAITHFULNESS_RUNS_V1.csv"
)

FAITHFULNESS_RANDOM_CSV = (
    PRODUCTION_DIR / "ADNI_XAI_FAITHFULNESS_RANDOM_CONTROLS_V1.csv"
)

FAITHFULNESS_SUBJECT_CSV = (
    PRODUCTION_DIR / "ADNI_XAI_FAITHFULNESS_SUBJECT_SUMMARY_V1.csv"
)

FAITHFULNESS_COHORT_CSV = (
    PRODUCTION_DIR / "ADNI_XAI_FAITHFULNESS_COHORT_SUMMARY_V1.csv"
)

FAITHFULNESS_EVENT_CSV = (
    PRODUCTION_DIR / "ADNI_XAI_FAITHFULNESS_EVENT_STATUS_SUMMARY_V1.csv"
)

FAITHFULNESS_JSON = (
    PRODUCTION_DIR / "ADNI_XAI_FAITHFULNESS_SUMMARY_V1.json"
)

FAITHFULNESS_LOG = (
    PRODUCTION_DIR / "ADNI_XAI_FAITHFULNESS_V1.log"
)


CHECKPOINTS = {
    2026: MODEL_DIR / "ADNI_LONGITUDINAL_SURVIVAL_BEST_V1.pt",
    3407: MODEL_DIR / "ADNI_M0_SEED_3407_BEST_V1.pt",
    5813: MODEL_DIR / "ADNI_M0_SEED_5813_BEST_V1.pt",
    7291: MODEL_DIR / "ADNI_M0_SEED_7291_BEST_V1.pt",
    9109: MODEL_DIR / "ADNI_M0_SEED_9109_BEST_V1.pt",
}


# ==============================================================================================================
# 2. FROZEN EXPECTATIONS
# ==============================================================================================================

EXPECTED_MODEL_PARAMETERS = 4_269_109

SEEDS = [
    2026,
    3407,
    5813,
    7291,
    9109,
]

EXPECTED_EPOCHS = {
    2026: 12,
    3407: 7,
    5813: 11,
    7291: 5,
    9109: 9,
}

EXPECTED_VAL_NLL = {
    2026: 0.9486606425046921,
    3407: 0.9454395517706871,
    5813: 0.9594686004519463,
    7291: 0.9487041115760804,
    9109: 0.9548525424301624,
}

HORIZONS = {
    24: 1,
    36: 2,
}

TOP_K_PERCENTS = [
    1,
    5,
    10,
]

RANDOM_REPEATS = 20
RANDOM_BASE_SEED = 20260911

EXPECTED_SUBJECTS = 24
EXPECTED_PRODUCTION_RUNS = 240

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

DAYS_PER_YEAR = 365.25
BRAIN_EPS = 1e-6

DEVICE = torch.device("cpu")

torch.set_num_threads(2)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True",
)


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
        FAITHFULNESS_LOG,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            text + "\n"
        )

        f.flush()


if not FAITHFULNESS_LOG.exists():

    FAITHFULNESS_LOG.write_text(
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
            f"Crop mismatch: "
            f"{cropped.shape} != "
            f"{tuple(target_shape)}"
        )

    return cropped


def load_mri_tensor_and_mask(
    path,
):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"MRI file missing:\n{path}"
        )

    nii = nib.load(
        str(
            path
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
            f"Unexpected MRI shape "
            f"{volume.shape}; expected "
            f"{SOURCE_SPATIAL_SHAPE}"
        )

    cropped = center_crop_3d(
        volume
    )

    if not np.isfinite(
        cropped
    ).all():

        raise RuntimeError(
            f"Non-finite MRI values:\n{path}"
        )

    brain_mask = (
        np.abs(
            cropped
        )
        >
        BRAIN_EPS
    )

    if (
        brain_mask.sum()
        <
        1000
    ):

        raise RuntimeError(
            f"Brain mask unexpectedly small: "
            f"{int(brain_mask.sum())}"
        )

    tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                cropped
            )
        )
        .unsqueeze(
            0
        )
        .unsqueeze(
            0
        )
        .to(
            device=DEVICE,
            dtype=torch.float32,
        )
    )

    return (
        tensor,
        brain_mask,
    )


def deterministic_rng_seed(
    subject_id,
    seed,
    horizon_months,
    top_k_percent,
    repeat_index,
):

    key = (
        f"{RANDOM_BASE_SEED}|"
        f"{subject_id}|"
        f"{seed}|"
        f"{horizon_months}|"
        f"{top_k_percent}|"
        f"{repeat_index}"
    )

    digest = hashlib.sha256(
        key.encode(
            "utf-8"
        )
    ).digest()

    return int.from_bytes(
        digest[
            :8
        ],
        byteorder="little",
        signed=False,
    ) % (
        2**32
    )


def select_top_positive_indices(
    attribution,
    brain_mask,
    percent,
):

    attribution = np.asarray(
        attribution,
        dtype=np.float32,
    )

    brain_mask = np.asarray(
        brain_mask,
        dtype=bool,
    )

    flat_attr = attribution.ravel()
    flat_mask = brain_mask.ravel()

    brain_indices = np.flatnonzero(
        flat_mask
    )

    n_brain = int(
        brain_indices.size
    )

    target_n = max(
        1,
        int(
            np.ceil(
                n_brain
                *
                float(
                    percent
                )
                /
                100.0
            )
        ),
    )

    brain_attr = flat_attr[
        brain_indices
    ]

    positive_local = np.flatnonzero(
        brain_attr
        >
        0.0
    )

    n_positive = int(
        positive_local.size
    )

    n_select = min(
        target_n,
        n_positive,
    )

    if (
        n_select
        <=
        0
    ):

        return (
            np.array(
                [],
                dtype=np.int64,
            ),
            n_brain,
            n_positive,
            target_n,
        )

    positive_values = brain_attr[
        positive_local
    ]

    if (
        n_select
        >=
        n_positive
    ):

        selected_positive_local = positive_local

    else:

        top_local_within_positive = np.argpartition(
            positive_values,
            -n_select,
        )[
            -n_select:
        ]

        selected_positive_local = positive_local[
            top_local_within_positive
        ]

    selected_flat_indices = brain_indices[
        selected_positive_local
    ]

    return (
        selected_flat_indices.astype(
            np.int64,
            copy=False,
        ),
        n_brain,
        n_positive,
        target_n,
    )


def perturb_flat_indices_to_zero(
    latest_tensor,
    flat_indices,
):

    perturbed = (
        latest_tensor
        .detach()
        .clone()
    )

    flat = perturbed.view(
        -1
    )

    if (
        len(
            flat_indices
        )
        >
        0
    ):

        index_tensor = torch.as_tensor(
            flat_indices,
            dtype=torch.long,
            device=perturbed.device,
        )

        flat[
            index_tensor
        ] = 0.0

    return perturbed


def random_matched_indices(
    brain_mask,
    targeted_indices,
    n_select,
    rng,
):

    flat_mask = np.asarray(
        brain_mask,
        dtype=bool,
    ).ravel()

    brain_indices = np.flatnonzero(
        flat_mask
    )

    targeted_indices = np.asarray(
        targeted_indices,
        dtype=np.int64,
    )

    candidate_mask = np.ones(
        brain_indices.shape[
            0
        ],
        dtype=bool,
    )

    if (
        targeted_indices.size
        >
        0
    ):

        # Remove targeted indices from random pool.
        targeted_set = set(
            targeted_indices.tolist()
        )

        candidate_mask = np.array(
            [
                idx
                not in
                targeted_set
                for idx in brain_indices
            ],
            dtype=bool,
        )

    candidates = brain_indices[
        candidate_mask
    ]

    if (
        n_select
        >
        candidates.size
    ):

        raise RuntimeError(
            f"Random control pool too small: "
            f"need {n_select}, have {candidates.size}"
        )

    chosen = rng.choice(
        candidates,
        size=n_select,
        replace=False,
    )

    return chosen.astype(
        np.int64,
        copy=False,
    )


def save_csv_records(
    records,
    path,
    sort_columns,
):

    df = pd.DataFrame(
        records
    )

    if not df.empty:

        df = df.sort_values(
            sort_columns
        ).reset_index(
            drop=True
        )

    df.to_csv(
        path,
        index=False,
    )


# ==============================================================================================================
# 5. MODEL WRAPPER
# ==============================================================================================================

class GeneralLatestVisitConditionalRiskWrapper(
    nn.Module
):

    def __init__(
        self,
        frozen_model,
        fixed_prior_embeddings,
        fixed_years_from_first,
        fixed_years_before_landmark,
        horizon_index,
    ):

        super().__init__()

        self.frozen_model = frozen_model

        self.register_buffer(
            "fixed_prior_embeddings",
            fixed_prior_embeddings.detach().clone(),
        )

        self.register_buffer(
            "fixed_years_from_first",
            fixed_years_from_first.detach().clone(),
        )

        self.register_buffer(
            "fixed_years_before_landmark",
            fixed_years_before_landmark.detach().clone(),
        )

        self.horizon_index = int(
            horizon_index
        )


    def forward(
        self,
        latest_mri_input,
    ):

        latest_embedding = self.frozen_model.mri_encoder(
            latest_mri_input
        )

        batch_size = latest_mri_input.shape[
            0
        ]

        prior_embeddings = (
            self.fixed_prior_embeddings
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        latest_embedding = latest_embedding.unsqueeze(
            1
        )

        mri_embeddings = torch.cat(
            [
                prior_embeddings,
                latest_embedding,
            ],
            dim=1,
        )

        years_from_first = (
            self.fixed_years_from_first
            .expand(
                batch_size,
                -1,
            )
        )

        years_before_landmark = (
            self.fixed_years_before_landmark
            .expand(
                batch_size,
                -1,
            )
        )

        visit_mask = torch.ones(
            (
                batch_size,
                mri_embeddings.shape[
                    1
                ],
            ),
            dtype=torch.bool,
            device=mri_embeddings.device,
        )

        time_embeddings = self.frozen_model.time_encoder(
            years_from_first,
            years_before_landmark,
        )

        x = (
            mri_embeddings
            +
            time_embeddings
        )

        mask_float = (
            visit_mask
            .unsqueeze(
                -1
            )
            .to(
                x.dtype
            )
        )

        x = x * mask_float

        x = self.frozen_model.transformer(
            x,
            src_key_padding_mask=(
                ~visit_mask
            ),
        )

        x = self.frozen_model.final_norm(
            x
        )

        pooled = (
            (
                x
                *
                mask_float
            )
            .sum(
                dim=1
            )
            /
            mask_float
            .sum(
                dim=1
            )
            .clamp_min(
                1.0
            )
        )

        logits = self.frozen_model.survival_head(
            pooled
        )

        hazards = torch.sigmoid(
            logits
        )

        survival = torch.cumprod(
            1.0
            -
            hazards,
            dim=1,
        )

        risk = (
            1.0
            -
            survival
        )

        return risk[
            :,
            self.horizon_index
        ]


# ==============================================================================================================
# 6. MAIN
# ==============================================================================================================

def main():

    log("=" * 120)
    log("ADNI XAI — ATTRIBUTION-GUIDED FAITHFULNESS PERTURBATION")
    log("=" * 120)

    log(
        "Integrated Gradients rerun: NO"
    )

    log(
        "Model training: NO"
    )

    log(
        "Model inference for perturbation testing: YES"
    )

    log(
        f"Random controls per targeted perturbation: {RANDOM_REPEATS}"
    )

    log(
        f"Top-k thresholds: {TOP_K_PERCENTS}"
    )

    log("")

    # ----------------------------------------------------------------------------------------------------------
    # 6A. INPUT AUDIT
    # ----------------------------------------------------------------------------------------------------------

    log("FILE / LOCK AUDIT")
    log("-" * 120)

    require_file(
        MASTER_CSV,
        "Production master CSV",
    )

    require_file(
        STABILITY_SUMMARY_JSON,
        "Cross-seed stability summary",
    )

    require_file(
        COHORT_CSV,
        "Frozen TEST cohort CSV",
    )

    require_file(
        XAI_CONFIG_LOCK,
        "Frozen XAI config lock",
    )

    with open(
        XAI_CONFIG_LOCK,
        "r",
        encoding="utf-8",
    ) as f:

        xai_lock = json.load(
            f
        )

    locked_top_k = [
        int(x)
        for x in xai_lock.get(
            "top_k_percentages",
            []
        )
    ]

    if (
        locked_top_k
        !=
        TOP_K_PERCENTS
    ):

        raise RuntimeError(
            f"Top-k mismatch with XAI lock: "
            f"{locked_top_k} != {TOP_K_PERCENTS}"
        )

    if (
        xai_lock.get(
            "faithfulness",
            {}
        ).get(
            "replacement_value"
        )
        !=
        "normalized background reference consistent with frozen preprocessing"
    ):

        raise RuntimeError(
            "Faithfulness replacement-value lock mismatch."
        )

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

    unique_keys = (
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
            unique_keys
        )
        !=
        EXPECTED_PRODUCTION_RUNS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_PRODUCTION_RUNS} production runs; "
            f"found {len(unique_keys)}."
        )

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

    cohort = cohort.set_index(
        "subject_id",
        drop=False,
    )

    # ----------------------------------------------------------------------------------------------------------
    # 6B. RESUME STATE
    # ----------------------------------------------------------------------------------------------------------

    run_records = (
        pd.read_csv(
            FAITHFULNESS_RUNS_CSV
        ).to_dict(
            orient="records"
        )
        if FAITHFULNESS_RUNS_CSV.exists()
        else []
    )

    random_records = (
        pd.read_csv(
            FAITHFULNESS_RANDOM_CSV
        ).to_dict(
            orient="records"
        )
        if FAITHFULNESS_RANDOM_CSV.exists()
        else []
    )

    completed_keys = {
        (
            str(
                r[
                    "subject_id"
                ]
            ),
            int(
                r[
                    "seed"
                ]
            ),
            int(
                r[
                    "horizon_months"
                ]
            ),
            int(
                r[
                    "top_k_percent"
                ]
            ),
        )
        for r in run_records
        if str(
            r.get(
                "status",
                ""
            )
        )
        ==
        "COMPLETE"
    }

    expected_total_targeted = (
        EXPECTED_SUBJECTS
        *
        len(
            SEEDS
        )
        *
        len(
            HORIZONS
        )
        *
        len(
            TOP_K_PERCENTS
        )
    )

    log("")
    log(
        f"Expected targeted perturbation runs: {expected_total_targeted}"
    )

    log(
        f"Already complete: {len(completed_keys)}"
    )

    # ----------------------------------------------------------------------------------------------------------
    # 6C. SEED LOOP
    # ----------------------------------------------------------------------------------------------------------

    for seed_index, seed in enumerate(
        SEEDS,
        start=1,
    ):

        checkpoint_path = CHECKPOINTS[
            seed
        ]

        require_file(
            checkpoint_path,
            f"Seed {seed} checkpoint",
        )

        log("")
        log("=" * 120)
        log(
            f"SEED {seed} ({seed_index}/{len(SEEDS)})"
        )
        log("=" * 120)

        model = ExactM0LongitudinalSurvivalModel_V3()

        parameter_count = sum(
            p.numel()
            for p in model.parameters()
        )

        if (
            parameter_count
            !=
            EXPECTED_MODEL_PARAMETERS
        ):

            raise RuntimeError(
                f"Seed {seed}: parameter mismatch."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        epoch = checkpoint.get(
            "epoch"
        )

        val_nll = float(
            checkpoint.get(
                "validation_loss"
            )
        )

        if (
            epoch
            !=
            EXPECTED_EPOCHS[
                seed
            ]
        ):

            raise RuntimeError(
                f"Seed {seed}: epoch mismatch."
            )

        if (
            abs(
                val_nll
                -
                EXPECTED_VAL_NLL[
                    seed
                ]
            )
            >
            1e-10
        ):

            raise RuntimeError(
                f"Seed {seed}: validation NLL mismatch."
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        model = model.to(
            DEVICE
        )

        model.eval()

        for p in model.parameters():

            p.requires_grad_(
                False
            )

        # ------------------------------------------------------------------------------------------------------
        # SUBJECT LOOP
        # ------------------------------------------------------------------------------------------------------

        for subject_index, subject_id in enumerate(
            sorted(
                cohort.index
                .astype(
                    str
                )
                .tolist()
            ),
            start=1,
        ):

            row = cohort.loc[
                subject_id
            ]

            event_value = int(
                row[
                    "event"
                ]
            )

            n_mri = int(
                row[
                    "n_mri"
                ]
            )

            history_bin = str(
                row[
                    "history_bin"
                ]
            )

            image_paths = parse_list_field(
                row[
                    "image_paths"
                ]
            )

            days_from_first = np.asarray(
                parse_list_field(
                    row[
                        "days_from_first_mri"
                    ]
                ),
                dtype=np.float32,
            )

            days_before_landmark = np.asarray(
                parse_list_field(
                    row[
                        "days_before_landmark"
                    ]
                ),
                dtype=np.float32,
            )

            if (
                len(
                    image_paths
                )
                !=
                n_mri
            ):

                raise RuntimeError(
                    f"{subject_id}: MRI path count mismatch."
                )

            log("")
            log(
                f"[{subject_index:02d}/{EXPECTED_SUBJECTS}] "
                f"{subject_id} | seed={seed} | "
                f"event={event_value} | n_mri={n_mri}"
            )

            # Encode all previous visits once per seed/subject.
            prior_embeddings_list = []

            for prior_path in image_paths[
                :-1
            ]:

                prior_tensor, _ = load_mri_tensor_and_mask(
                    prior_path
                )

                with torch.inference_mode():

                    prior_embedding = model.mri_encoder(
                        prior_tensor
                    ).detach().clone()

                prior_embeddings_list.append(
                    prior_embedding
                )

                del (
                    prior_tensor,
                    prior_embedding,
                )

                gc.collect()

            fixed_prior_embeddings = torch.stack(
                prior_embeddings_list,
                dim=1,
            )

            latest_tensor, brain_mask = load_mri_tensor_and_mask(
                image_paths[
                    -1
                ]
            )

            years_from_first = (
                torch.tensor(
                    days_from_first
                    /
                    DAYS_PER_YEAR,
                    dtype=torch.float32,
                    device=DEVICE,
                )
                .unsqueeze(
                    0
                )
            )

            years_before_landmark = (
                torch.tensor(
                    days_before_landmark
                    /
                    DAYS_PER_YEAR,
                    dtype=torch.float32,
                    device=DEVICE,
                )
                .unsqueeze(
                    0
                )
            )

            for horizon_months, horizon_index in HORIZONS.items():

                master_row = complete.loc[
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
                        horizon_months
                    )
                ]

                if (
                    len(
                        master_row
                    )
                    !=
                    1
                ):

                    raise RuntimeError(
                        f"{subject_id}, seed={seed}, "
                        f"horizon={horizon_months}: "
                        f"expected one production row, "
                        f"found {len(master_row)}."
                    )

                attribution_file = Path(
                    str(
                        master_row.iloc[
                            0
                        ][
                            "attribution_file"
                        ]
                    )
                )

                if not attribution_file.exists():

                    raise FileNotFoundError(
                        attribution_file
                    )

                expected_attr_hash = str(
                    master_row.iloc[
                        0
                    ][
                        "attribution_sha256"
                    ]
                )

                if (
                    sha256_file(
                        attribution_file
                    )
                    !=
                    expected_attr_hash
                ):

                    raise RuntimeError(
                        f"Attribution hash mismatch: "
                        f"{attribution_file}"
                    )

                with np.load(
                    attribution_file,
                    allow_pickle=False,
                ) as data:

                    attribution = data[
                        "attribution"
                    ].astype(
                        np.float32,
                        copy=False,
                    )

                if (
                    attribution.shape
                    !=
                    MODEL_SPATIAL_SHAPE
                ):

                    raise RuntimeError(
                        "Attribution spatial shape mismatch."
                    )

                risk_model = GeneralLatestVisitConditionalRiskWrapper(
                    frozen_model=model,
                    fixed_prior_embeddings=fixed_prior_embeddings,
                    fixed_years_from_first=years_from_first,
                    fixed_years_before_landmark=years_before_landmark,
                    horizon_index=horizon_index,
                ).to(
                    DEVICE
                )

                risk_model.eval()

                with torch.inference_mode():

                    original_risk = float(
                        risk_model(
                            latest_tensor
                        ).item()
                    )

                for top_k_percent in TOP_K_PERCENTS:

                    run_key = (
                        subject_id,
                        seed,
                        horizon_months,
                        top_k_percent,
                    )

                    if run_key in completed_keys:

                        log(
                            f"  {horizon_months}m | "
                            f"top{top_k_percent}% | complete -> SKIP"
                        )

                        continue

                    targeted_indices, n_brain, n_positive, target_n = (
                        select_top_positive_indices(
                            attribution=attribution,
                            brain_mask=brain_mask,
                            percent=top_k_percent,
                        )
                    )

                    n_selected = int(
                        targeted_indices.size
                    )

                    effective_percent = (
                        100.0
                        *
                        n_selected
                        /
                        n_brain
                        if n_brain
                        >
                        0
                        else np.nan
                    )

                    if (
                        n_selected
                        ==
                        0
                    ):

                        raise RuntimeError(
                            f"{subject_id}, seed={seed}, "
                            f"{horizon_months}m: "
                            f"no positive IG voxels available "
                            f"for top {top_k_percent}% perturbation."
                        )

                    start = time.time()

                    targeted_tensor = perturb_flat_indices_to_zero(
                        latest_tensor,
                        targeted_indices,
                    )

                    with torch.inference_mode():

                        targeted_risk = float(
                            risk_model(
                                targeted_tensor
                            ).item()
                        )

                    targeted_risk_drop = (
                        original_risk
                        -
                        targeted_risk
                    )

                    del targeted_tensor
                    gc.collect()

                    random_drops = []

                    for repeat_index in range(
                        RANDOM_REPEATS
                    ):

                        rng_seed = deterministic_rng_seed(
                            subject_id=subject_id,
                            seed=seed,
                            horizon_months=horizon_months,
                            top_k_percent=top_k_percent,
                            repeat_index=repeat_index,
                        )

                        rng = np.random.default_rng(
                            rng_seed
                        )

                        random_indices = random_matched_indices(
                            brain_mask=brain_mask,
                            targeted_indices=targeted_indices,
                            n_select=n_selected,
                            rng=rng,
                        )

                        random_tensor = perturb_flat_indices_to_zero(
                            latest_tensor,
                            random_indices,
                        )

                        with torch.inference_mode():

                            random_risk = float(
                                risk_model(
                                    random_tensor
                                ).item()
                            )

                        random_drop = (
                            original_risk
                            -
                            random_risk
                        )

                        random_drops.append(
                            random_drop
                        )

                        random_records.append(
                            {
                                "subject_id":
                                    subject_id,

                                "event":
                                    event_value,

                                "n_mri":
                                    n_mri,

                                "history_bin":
                                    history_bin,

                                "seed":
                                    seed,

                                "horizon_months":
                                    horizon_months,

                                "top_k_percent":
                                    top_k_percent,

                                "repeat_index":
                                    repeat_index,

                                "rng_seed":
                                    int(
                                        rng_seed
                                    ),

                                "n_selected":
                                    n_selected,

                                "random_risk":
                                    random_risk,

                                "random_risk_drop":
                                    random_drop,
                            }
                        )

                        del random_tensor
                        gc.collect()

                    random_drops_np = np.asarray(
                        random_drops,
                        dtype=np.float64,
                    )

                    random_mean_drop = float(
                        random_drops_np.mean()
                    )

                    random_median_drop = float(
                        np.median(
                            random_drops_np
                        )
                    )

                    random_sd_drop = float(
                        random_drops_np.std(
                            ddof=1
                        )
                    )

                    faithfulness_advantage = (
                        targeted_risk_drop
                        -
                        random_mean_drop
                    )

                    percentile_vs_random = float(
                        np.mean(
                            random_drops_np
                            <=
                            targeted_risk_drop
                        )
                    )

                    elapsed_seconds = (
                        time.time()
                        -
                        start
                    )

                    run_records.append(
                        {
                            "status":
                                "COMPLETE",

                            "subject_id":
                                subject_id,

                            "event":
                                event_value,

                            "n_mri":
                                n_mri,

                            "history_bin":
                                history_bin,

                            "seed":
                                seed,

                            "horizon_months":
                                horizon_months,

                            "top_k_percent":
                                top_k_percent,

                            "brain_voxels":
                                n_brain,

                            "positive_attribution_voxels":
                                n_positive,

                            "target_requested_voxels":
                                target_n,

                            "n_selected":
                                n_selected,

                            "effective_selected_percent":
                                effective_percent,

                            "original_risk":
                                original_risk,

                            "targeted_perturbed_risk":
                                targeted_risk,

                            "targeted_risk_drop":
                                targeted_risk_drop,

                            "random_repeats":
                                RANDOM_REPEATS,

                            "random_mean_risk_drop":
                                random_mean_drop,

                            "random_median_risk_drop":
                                random_median_drop,

                            "random_sd_risk_drop":
                                random_sd_drop,

                            "faithfulness_advantage":
                                faithfulness_advantage,

                            "targeted_percentile_vs_random":
                                percentile_vs_random,

                            "elapsed_seconds":
                                elapsed_seconds,

                            "attribution_file":
                                str(
                                    attribution_file
                                ),

                            "attribution_sha256":
                                expected_attr_hash,
                        }
                    )

                    completed_keys.add(
                        run_key
                    )

                    save_csv_records(
                        run_records,
                        FAITHFULNESS_RUNS_CSV,
                        [
                            "subject_id",
                            "seed",
                            "horizon_months",
                            "top_k_percent",
                        ],
                    )

                    save_csv_records(
                        random_records,
                        FAITHFULNESS_RANDOM_CSV,
                        [
                            "subject_id",
                            "seed",
                            "horizon_months",
                            "top_k_percent",
                            "repeat_index",
                        ],
                    )

                    log(
                        f"  {horizon_months}m | "
                        f"top{top_k_percent}% | "
                        f"n={n_selected} | "
                        f"target_drop={targeted_risk_drop:+.6f} | "
                        f"random_mean={random_mean_drop:+.6f} | "
                        f"advantage={faithfulness_advantage:+.6f} | "
                        f"percentile={percentile_vs_random:.2f} | "
                        f"{elapsed_seconds:.2f}s"
                    )

                del (
                    risk_model,
                    attribution,
                )

                gc.collect()

            del (
                fixed_prior_embeddings,
                latest_tensor,
                brain_mask,
                years_from_first,
                years_before_landmark,
            )

            gc.collect()

        del (
            model,
            checkpoint,
        )

        gc.collect()

    # ----------------------------------------------------------------------------------------------------------
    # 6D. FINAL AUDIT
    # ----------------------------------------------------------------------------------------------------------

    runs_df = pd.read_csv(
        FAITHFULNESS_RUNS_CSV
    )

    complete_runs = runs_df.loc[
        runs_df[
            "status"
        ]
        ==
        "COMPLETE"
    ].copy()

    unique_run_keys = (
        complete_runs[
            [
                "subject_id",
                "seed",
                "horizon_months",
                "top_k_percent",
            ]
        ]
        .drop_duplicates()
    )

    if (
        len(
            unique_run_keys
        )
        !=
        expected_total_targeted
    ):

        log("")
        log("=" * 120)
        log("FAITHFULNESS ANALYSIS: INCOMPLETE")
        log("=" * 120)

        log(
            f"Completed targeted runs: "
            f"{len(unique_run_keys)}/{expected_total_targeted}"
        )

        log(
            "Re-run this same script to resume."
        )

        return

    random_df = pd.read_csv(
        FAITHFULNESS_RANDOM_CSV
    )

    expected_random_rows = (
        expected_total_targeted
        *
        RANDOM_REPEATS
    )

    if (
        len(
            random_df
        )
        !=
        expected_random_rows
    ):

        raise RuntimeError(
            f"Expected {expected_random_rows} random-control rows; "
            f"found {len(random_df)}."
        )

    # ----------------------------------------------------------------------------------------------------------
    # 6E. SUBJECT-LEVEL AGGREGATION ACROSS FIVE SEEDS
    # ----------------------------------------------------------------------------------------------------------

    subject_summary = (
        complete_runs.groupby(
            [
                "subject_id",
                "event",
                "n_mri",
                "history_bin",
                "horizon_months",
                "top_k_percent",
            ],
            as_index=False,
        )
        .agg(
            mean_targeted_risk_drop=(
                "targeted_risk_drop",
                "mean",
            ),

            median_targeted_risk_drop=(
                "targeted_risk_drop",
                "median",
            ),

            mean_random_risk_drop=(
                "random_mean_risk_drop",
                "mean",
            ),

            median_random_risk_drop=(
                "random_mean_risk_drop",
                "median",
            ),

            mean_faithfulness_advantage=(
                "faithfulness_advantage",
                "mean",
            ),

            median_faithfulness_advantage=(
                "faithfulness_advantage",
                "median",
            ),

            mean_targeted_percentile_vs_random=(
                "targeted_percentile_vs_random",
                "mean",
            ),

            seeds=(
                "seed",
                "nunique",
            ),
        )
    )

    if not np.all(
        subject_summary[
            "seeds"
        ]
        ==
        len(
            SEEDS
        )
    ):

        raise RuntimeError(
            "Some subject-level faithfulness summaries do not contain all five seeds."
        )

    subject_summary.to_csv(
        FAITHFULNESS_SUBJECT_CSV,
        index=False,
    )

    # ----------------------------------------------------------------------------------------------------------
    # 6F. COHORT SUMMARY — PARTICIPANT IS THE UNIT OF SUMMARY
    # ----------------------------------------------------------------------------------------------------------

    cohort_summary = (
        subject_summary.groupby(
            [
                "horizon_months",
                "top_k_percent",
            ],
            as_index=False,
        )
        .agg(
            subjects=(
                "subject_id",
                "nunique",
            ),

            median_subject_targeted_drop=(
                "mean_targeted_risk_drop",
                "median",
            ),

            q1_subject_targeted_drop=(
                "mean_targeted_risk_drop",
                lambda x: np.quantile(
                    x,
                    0.25,
                ),
            ),

            q3_subject_targeted_drop=(
                "mean_targeted_risk_drop",
                lambda x: np.quantile(
                    x,
                    0.75,
                ),
            ),

            median_subject_random_drop=(
                "mean_random_risk_drop",
                "median",
            ),

            median_subject_faithfulness_advantage=(
                "mean_faithfulness_advantage",
                "median",
            ),

            q1_subject_faithfulness_advantage=(
                "mean_faithfulness_advantage",
                lambda x: np.quantile(
                    x,
                    0.25,
                ),
            ),

            q3_subject_faithfulness_advantage=(
                "mean_faithfulness_advantage",
                lambda x: np.quantile(
                    x,
                    0.75,
                ),
            ),

            fraction_subjects_positive_advantage=(
                "mean_faithfulness_advantage",
                lambda x: float(
                    np.mean(
                        np.asarray(
                            x
                        )
                        >
                        0.0
                    )
                ),
            ),

            median_targeted_percentile_vs_random=(
                "mean_targeted_percentile_vs_random",
                "median",
            ),
        )
    )

    cohort_summary.to_csv(
        FAITHFULNESS_COHORT_CSV,
        index=False,
    )

    # ----------------------------------------------------------------------------------------------------------
    # 6G. EVENT / CENSORED DESCRIPTIVE SUMMARY
    # ----------------------------------------------------------------------------------------------------------

    event_summary = (
        subject_summary.groupby(
            [
                "horizon_months",
                "top_k_percent",
                "event",
            ],
            as_index=False,
        )
        .agg(
            subjects=(
                "subject_id",
                "nunique",
            ),

            median_targeted_drop=(
                "mean_targeted_risk_drop",
                "median",
            ),

            median_random_drop=(
                "mean_random_risk_drop",
                "median",
            ),

            median_faithfulness_advantage=(
                "mean_faithfulness_advantage",
                "median",
            ),

            fraction_positive_advantage=(
                "mean_faithfulness_advantage",
                lambda x: float(
                    np.mean(
                        np.asarray(
                            x
                        )
                        >
                        0.0
                    )
                ),
            ),
        )
    )

    event_summary.to_csv(
        FAITHFULNESS_EVENT_CSV,
        index=False,
    )

    # ----------------------------------------------------------------------------------------------------------
    # 6H. JSON + REPORT
    # ----------------------------------------------------------------------------------------------------------

    output = {
        "analysis":
            "Attribution-guided TEST XAI faithfulness perturbation",

        "status":
            "COMPLETE",

        "subjects":
            EXPECTED_SUBJECTS,

        "seeds":
            SEEDS,

        "horizons_months":
            list(
                HORIZONS.keys()
            ),

        "top_k_percentages":
            TOP_K_PERCENTS,

        "random_repeats_per_targeted_run":
            RANDOM_REPEATS,

        "randomization":
            "deterministic SHA256-derived NumPy RNG seeds",

        "attribution_selection":
            "highest positive-risk IG voxels within valid brain mask",

        "target_requested_definition":
            "ceil(k% of valid brain voxels), capped by number of positive attribution voxels",

        "perturbation":
            "set selected latest-MRI voxels to normalized background value zero",

        "random_control":
            "equal-size within-brain voxel sets excluding targeted voxels",

        "primary_effect":
            "original risk minus perturbed risk",

        "primary_faithfulness_measure":
            "targeted risk drop minus mean matched-random risk drop",

        "aggregation_unit":
            "participant; five seeds averaged within participant before cohort summaries",

        "targeted_runs":
            int(
                len(
                    unique_run_keys
                )
            ),

        "random_control_runs":
            int(
                len(
                    random_df
                )
            ),

        "cohort_summary":
            cohort_summary.to_dict(
                orient="records"
            ),

        "event_status_summary":
            event_summary.to_dict(
                orient="records"
            ),

        "model_training":
            False,

        "integrated_gradients_rerun":
            False,

        "model_inference_for_perturbation":
            True,
    }

    with open(
        FAITHFULNESS_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    log("")
    log("=" * 120)
    log("FAITHFULNESS — COHORT SUMMARY")
    log("=" * 120)

    log(
        cohort_summary.to_string(
            index=False
        )
    )

    log("")
    log("=" * 120)
    log("FAITHFULNESS — EVENT / CENSORED DESCRIPTIVE SUMMARY")
    log("=" * 120)

    log(
        event_summary.to_string(
            index=False
        )
    )

    log("")
    log(
        f"Saved targeted runs CSV : {FAITHFULNESS_RUNS_CSV}"
    )

    log(
        f"Saved random controls CSV: {FAITHFULNESS_RANDOM_CSV}"
    )

    log(
        f"Saved subject summary    : {FAITHFULNESS_SUBJECT_CSV}"
    )

    log(
        f"Saved cohort summary     : {FAITHFULNESS_COHORT_CSV}"
    )

    log(
        f"Saved event summary      : {FAITHFULNESS_EVENT_CSV}"
    )

    log(
        f"Saved JSON               : {FAITHFULNESS_JSON}"
    )

    log("")
    log("=" * 120)
    log("XAI FAITHFULNESS PERTURBATION: COMPLETE")
    log("=" * 120)

    log(
        "Integrated Gradients rerun: NO"
    )

    log(
        "Model retraining: NO"
    )

    log(
        "Next step: interpret stability + faithfulness jointly and prepare publication figures/tables"
    )

    log("=" * 120)


if __name__ == "__main__":

    try:

        main()

    except Exception:

        log("")
        log("=" * 120)
        log("XAI FAITHFULNESS PERTURBATION: FAILED")
        log("=" * 120)

        log(
            traceback.format_exc()
        )

        log(
            "This analysis is resume-safe. Fix the error and rerun this same script."
        )

        raise
