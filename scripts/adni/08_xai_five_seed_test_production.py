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
import torch.nn.functional as F

from captum.attr import IntegratedGradients

from exact_m0_model_v3 import ExactM0LongitudinalSurvivalModel_V3


# ==============================================================================================================
# ADNI — FIVE-SEED TEST XAI PRODUCTION
#
# PURPOSE
# -------
# Generate the frozen TEST Integrated-Gradients explanations on the already locked
# 24-participant TEST XAI cohort.
#
# Frozen configuration:
#   - Model: exact frozen M0
#   - Seeds: 2026, 3407, 5813, 7291, 9109
#   - Horizons: 24m, 36m
#   - Explained input: latest MRI
#   - Prior MRI embeddings + timing: fixed context
#   - Baseline: blurred latest MRI
#   - Blur: 9x9x9 local mean, replicate padding
#   - IG: Gauss-Legendre, 16 steps, internal_batch_size=1
#
# Design:
#   - Resume-safe
#   - Writes one attribution file per subject/seed/horizon
#   - Appends/rewrites master audit CSV after every completed run
#   - Does NOT change any model/XAI setting based on TEST results
#
# IMPORTANT
# ---------
# This script begins TEST XAI generation.
# After this starts, the frozen XAI configuration and cohort must NOT be changed.
# ==============================================================================================================


# ==============================================================================================================
# 1. PATHS
# ==============================================================================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "07_models"
RESULT_DIR = ROOT / "08_results"

XAI_CONFIG_LOCK = (
    RESULT_DIR / "ADNI_XAI_PRODUCTION_CONFIG_LOCK_V1.json"
)

XAI_COHORT_LOCK = (
    RESULT_DIR / "ADNI_XAI_FROZEN_TEST_COHORT_V1.json"
)

XAI_COHORT_CSV = (
    RESULT_DIR / "ADNI_XAI_FROZEN_TEST_COHORT_V1.csv"
)

XAI_COHORT_HASHES = (
    RESULT_DIR / "ADNI_XAI_FROZEN_TEST_COHORT_HASHES_V1.csv"
)

PRODUCTION_DIR = (
    RESULT_DIR / "ADNI_XAI_TEST_PRODUCTION_V1"
)

ATTRIBUTION_DIR = (
    PRODUCTION_DIR / "attributions"
)

MASTER_CSV = (
    PRODUCTION_DIR / "ADNI_XAI_TEST_PRODUCTION_MASTER_V1.csv"
)

SUMMARY_JSON = (
    PRODUCTION_DIR / "ADNI_XAI_TEST_PRODUCTION_SUMMARY_V1.json"
)

RUN_LOG = (
    PRODUCTION_DIR / "ADNI_XAI_TEST_PRODUCTION_V1.log"
)

RUN_LOCK_SNAPSHOT = (
    PRODUCTION_DIR / "ADNI_XAI_TEST_PRODUCTION_LOCK_SNAPSHOT_V1.json"
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

EXPECTED_COHORT_PARTICIPANTS = 24
EXPECTED_EVENT_PARTICIPANTS = 12
EXPECTED_CENSORED_PARTICIPANTS = 12

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

BLUR_KERNEL_SIZE = 9

IG_STEPS = 16
IG_METHOD = "gausslegendre"
INTERNAL_BATCH_SIZE = 1

DEVICE = torch.device("cpu")

# Keep CPU usage conservative / reproducible.
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
# 3. CREATE OUTPUT DIRECTORIES
# ==============================================================================================================

PRODUCTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ATTRIBUTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==============================================================================================================
# 4. LOGGING
# ==============================================================================================================

def log(message=""):

    text = str(message)

    print(
        text,
        flush=True,
    )

    with open(
        RUN_LOG,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            text + "\n"
        )

        f.flush()


# Do not erase an existing log when resuming.
if not RUN_LOG.exists():

    RUN_LOG.write_text(
        "",
        encoding="utf-8",
    )


# ==============================================================================================================
# 5. HELPERS
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

        raise ValueError(
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
            f"Crop failed: "
            f"{cropped.shape} != "
            f"{tuple(target_shape)}"
        )

    return cropped


def load_mri_tensor(
    path,
):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"MRI file not found:\n{path}"
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
            f"{volume.shape}; "
            f"expected "
            f"{SOURCE_SPATIAL_SHAPE}"
        )

    if not np.isfinite(
        volume
    ).all():

        raise RuntimeError(
            f"Non-finite MRI values:\n{path}"
        )

    volume = center_crop_3d(
        volume
    )

    return (
        torch.from_numpy(
            np.ascontiguousarray(
                volume
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


def make_local_mean_blur(
    x,
    kernel_size=BLUR_KERNEL_SIZE,
):

    if (
        kernel_size
        %
        2
        !=
        1
    ):

        raise ValueError(
            "BLUR_KERNEL_SIZE must be odd."
        )

    pad = (
        kernel_size
        //
        2
    )

    padded = F.pad(
        x,
        (
            pad,
            pad,
            pad,
            pad,
            pad,
            pad,
        ),
        mode="replicate",
    )

    blurred = F.avg_pool3d(
        padded,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
    )

    if (
        blurred.shape
        !=
        x.shape
    ):

        raise RuntimeError(
            f"Blurred shape mismatch: "
            f"{tuple(blurred.shape)} != "
            f"{tuple(x.shape)}"
        )

    return blurred


def load_existing_master():

    if not MASTER_CSV.exists():

        return pd.DataFrame()

    df = pd.read_csv(
        MASTER_CSV
    )

    return df


def completed_key_set(
    master_df,
):

    if master_df.empty:
        return set()

    required = {
        "subject_id",
        "seed",
        "horizon_months",
        "status",
        "attribution_file",
    }

    if not required.issubset(
        set(
            master_df.columns
        )
    ):

        return set()

    done = master_df.loc[
        master_df[
            "status"
        ]
        ==
        "COMPLETE"
    ]

    keys = set()

    for _, row in done.iterrows():

        attr_path = Path(
            str(
                row[
                    "attribution_file"
                ]
            )
        )

        if attr_path.exists():

            keys.add(
                (
                    str(
                        row[
                            "subject_id"
                        ]
                    ),
                    int(
                        row[
                            "seed"
                        ]
                    ),
                    int(
                        row[
                            "horizon_months"
                        ]
                    ),
                )
            )

    return keys


def save_master(
    records,
):

    df = pd.DataFrame(
        records
    )

    if not df.empty:

        df = df.sort_values(
            [
                "subject_id",
                "seed",
                "horizon_months",
            ]
        ).reset_index(
            drop=True
        )

    df.to_csv(
        MASTER_CSV,
        index=False,
    )


# ==============================================================================================================
# 6. CONDITIONAL LATEST-VISIT RISK WRAPPER
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

        x = (
            x
            *
            mask_float
        )

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

        hazard_logits = self.frozen_model.survival_head(
            pooled
        )

        hazards = torch.sigmoid(
            hazard_logits
        )

        survival = torch.cumprod(
            1.0
            -
            hazards,
            dim=1,
        )

        cumulative_risk = (
            1.0
            -
            survival
        )

        return cumulative_risk[
            :,
            self.horizon_index
        ]


# ==============================================================================================================
# 7. VERIFY LOCKS BEFORE ANY TEST MRI LOAD
# ==============================================================================================================

def verify_frozen_locks():

    log("=" * 120)
    log("FROZEN TEST XAI PRE-FLIGHT AUDIT")
    log("=" * 120)

    require_file(
        XAI_CONFIG_LOCK,
        "XAI production config lock",
    )

    require_file(
        XAI_COHORT_LOCK,
        "Frozen TEST cohort lock",
    )

    require_file(
        XAI_COHORT_CSV,
        "Frozen TEST cohort CSV",
    )

    require_file(
        XAI_COHORT_HASHES,
        "Frozen TEST cohort hash file",
    )

    with open(
        XAI_CONFIG_LOCK,
        "r",
        encoding="utf-8",
    ) as f:

        xai_config = json.load(
            f
        )

    with open(
        XAI_COHORT_LOCK,
        "r",
        encoding="utf-8",
    ) as f:

        cohort_lock = json.load(
            f
        )

    if (
        xai_config.get(
            "baseline"
        )
        !=
        "blurred latest MRI"
    ):

        raise RuntimeError(
            "Frozen XAI baseline mismatch."
        )

    if (
        int(
            xai_config.get(
                "n_steps"
            )
        )
        !=
        IG_STEPS
    ):

        raise RuntimeError(
            "Frozen IG step count mismatch."
        )

    if (
        [
            int(x)
            for x in xai_config.get(
                "horizons_months",
                [],
            )
        ]
        !=
        [
            24,
            36,
        ]
    ):

        raise RuntimeError(
            "Frozen horizon mismatch."
        )

    if (
        [
            int(x)
            for x in xai_config.get(
                "cross_seed_models",
                [],
            )
        ]
        !=
        SEEDS
    ):

        raise RuntimeError(
            "Frozen seed set mismatch."
        )

    selection = cohort_lock.get(
        "selection",
        {},
    )

    if (
        int(
            selection.get(
                "selected_participants"
            )
        )
        !=
        EXPECTED_COHORT_PARTICIPANTS
    ):

        raise RuntimeError(
            "Frozen TEST XAI cohort size mismatch."
        )

    if (
        int(
            selection.get(
                "selected_event_participants"
            )
        )
        !=
        EXPECTED_EVENT_PARTICIPANTS
    ):

        raise RuntimeError(
            "Frozen TEST XAI event count mismatch."
        )

    if (
        int(
            selection.get(
                "selected_censored_participants"
            )
        )
        !=
        EXPECTED_CENSORED_PARTICIPANTS
    ):

        raise RuntimeError(
            "Frozen TEST XAI censored count mismatch."
        )

    # Verify current cohort CSV hash against the previously written hash table.
    hashes_df = pd.read_csv(
        XAI_COHORT_HASHES
    )

    row = hashes_df.loc[
        hashes_df[
            "artifact"
        ]
        ==
        XAI_COHORT_CSV.name
    ]

    if row.empty:

        raise RuntimeError(
            "Frozen TEST cohort CSV hash not found in hash file."
        )

    expected_hash = str(
        row.iloc[
            0
        ][
            "sha256"
        ]
    )

    observed_hash = sha256_file(
        XAI_COHORT_CSV
    )

    if (
        observed_hash
        !=
        expected_hash
    ):

        raise RuntimeError(
            "Frozen TEST cohort CSV hash mismatch."
        )

    log(
        "Frozen XAI config:",
        # separate message below because log() takes one arg
    )


# ==============================================================================================================
# 8. MAIN
# ==============================================================================================================

def main():

    verify_frozen_locks()

    log(
        "Frozen XAI configuration: PASS"
    )

    log(
        "Frozen TEST cohort hash: PASS"
    )

    log(
        f"Seeds: {SEEDS}"
    )

    log(
        f"Horizons: {list(HORIZONS.keys())}"
    )

    log(
        f"IG: {IG_METHOD}, steps={IG_STEPS}, internal_batch_size={INTERNAL_BATCH_SIZE}"
    )

    log(
        f"Baseline: blurred latest MRI, {BLUR_KERNEL_SIZE}x{BLUR_KERNEL_SIZE}x{BLUR_KERNEL_SIZE} local mean"
    )

    # Snapshot frozen production rule at first run.
    if not RUN_LOCK_SNAPSHOT.exists():

        snapshot = {
            "status":
                "TEST XAI PRODUCTION STARTED",

            "xai_config_lock_sha256":
                sha256_file(
                    XAI_CONFIG_LOCK
                ),

            "test_cohort_lock_sha256":
                sha256_file(
                    XAI_COHORT_LOCK
                ),

            "test_cohort_csv_sha256":
                sha256_file(
                    XAI_COHORT_CSV
                ),

            "seeds":
                SEEDS,

            "horizons":
                list(
                    HORIZONS.keys()
                ),

            "baseline":
                "blurred latest MRI",

            "blur_kernel":
                BLUR_KERNEL_SIZE,

            "ig_method":
                IG_METHOD,

            "ig_steps":
                IG_STEPS,

            "internal_batch_size":
                INTERNAL_BATCH_SIZE,

            "post_test_method_changes":
                "PROHIBITED",
        }

        with open(
            RUN_LOCK_SNAPSHOT,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                snapshot,
                f,
                indent=2,
            )

    # ----------------------------------------------------------------------------------------------------------
    # Load frozen cohort metadata.
    # ----------------------------------------------------------------------------------------------------------

    cohort = pd.read_csv(
        XAI_COHORT_CSV
    )

    if (
        cohort[
            "subject_id"
        ].nunique()
        !=
        EXPECTED_COHORT_PARTICIPANTS
    ):

        raise RuntimeError(
            "Frozen cohort CSV does not contain 24 unique participants."
        )

    if (
        int(
            cohort[
                "event"
            ]
            .astype(
                int
            )
            .sum()
        )
        !=
        EXPECTED_EVENT_PARTICIPANTS
    ):

        raise RuntimeError(
            "Frozen cohort CSV event count mismatch."
        )

    # ----------------------------------------------------------------------------------------------------------
    # Resume state.
    # ----------------------------------------------------------------------------------------------------------

    existing_master = load_existing_master()

    existing_records = (
        existing_master.to_dict(
            orient="records"
        )
        if not existing_master.empty
        else []
    )

    completed = completed_key_set(
        existing_master
    )

    expected_total_runs = (
        EXPECTED_COHORT_PARTICIPANTS
        *
        len(
            SEEDS
        )
        *
        len(
            HORIZONS
        )
    )

    log("")
    log("=" * 120)
    log("TEST XAI PRODUCTION")
    log("=" * 120)

    log(
        f"Expected total IG runs: {expected_total_runs}"
    )

    log(
        f"Already completed and resumable: {len(completed)}"
    )

    log(
        f"Remaining: {expected_total_runs - len(completed)}"
    )

    # ----------------------------------------------------------------------------------------------------------
    # Seed loop: load each frozen model once.
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
                f"Seed {seed}: model parameter count mismatch."
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

        state_dict = checkpoint[
            "model_state_dict"
        ]

        if (
            set(
                model.state_dict().keys()
            )
            !=
            set(
                state_dict.keys()
            )
        ):

            raise RuntimeError(
                f"Seed {seed}: state-dict key mismatch."
            )

        for key, tensor in model.state_dict().items():

            if (
                tensor.shape
                !=
                state_dict[
                    key
                ].shape
            ):

                raise RuntimeError(
                    f"Seed {seed}: shape mismatch for {key}."
                )

        model.load_state_dict(
            state_dict,
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

        log(
            f"Checkpoint epoch={epoch} | "
            f"validation NLL={val_nll:.12f} | "
            f"parameters={parameter_count:,} | PASS"
        )

        # ------------------------------------------------------------------------------------------------------
        # Participant loop.
        # ------------------------------------------------------------------------------------------------------

        for participant_index, (_, row) in enumerate(
            cohort.iterrows(),
            start=1,
        ):

            subject_id = str(
                row[
                    "subject_id"
                ]
            )

            landmark_id = str(
                row[
                    "landmark_id"
                ]
            )

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

            # Skip participant entirely for this seed if both horizons are already complete.
            participant_keys = {
                (
                    subject_id,
                    seed,
                    horizon,
                )
                for horizon in HORIZONS
            }

            if participant_keys.issubset(
                completed
            ):

                log(
                    f"[{participant_index:02d}/24] "
                    f"{subject_id} | seed={seed} | already complete -> SKIP"
                )

                continue

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
                    f"{subject_id}: image path count mismatch."
                )

            if (
                len(
                    days_from_first
                )
                !=
                n_mri
            ):

                raise RuntimeError(
                    f"{subject_id}: days_from_first count mismatch."
                )

            if (
                len(
                    days_before_landmark
                )
                !=
                n_mri
            ):

                raise RuntimeError(
                    f"{subject_id}: days_before_landmark count mismatch."
                )

            log("")
            log(
                f"[{participant_index:02d}/24] "
                f"{subject_id} | "
                f"seed={seed} | "
                f"event={event_value} | "
                f"n_mri={n_mri} | "
                f"history={history_bin}"
            )

            # --------------------------------------------------------------------------------------------------
            # Encode all prior scans under no-grad.
            # --------------------------------------------------------------------------------------------------

            prior_embeddings_list = []

            for prior_path in image_paths[
                :-1
            ]:

                prior_tensor = load_mri_tensor(
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

            latest_tensor = load_mri_tensor(
                image_paths[
                    -1
                ]
            )

            blurred_latest = make_local_mean_blur(
                latest_tensor
            ).detach()

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

            # --------------------------------------------------------------------------------------------------
            # Horizon loop.
            # --------------------------------------------------------------------------------------------------

            for horizon_months, horizon_index in HORIZONS.items():

                run_key = (
                    subject_id,
                    seed,
                    horizon_months,
                )

                if run_key in completed:

                    log(
                        f"  {horizon_months}m | already complete -> SKIP"
                    )

                    continue

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

                    actual_risk = float(
                        risk_model(
                            latest_tensor
                        ).item()
                    )

                    baseline_risk = float(
                        risk_model(
                            blurred_latest
                        ).item()
                    )

                risk_difference = (
                    actual_risk
                    -
                    baseline_risk
                )

                ig_input = (
                    latest_tensor
                    .detach()
                    .clone()
                    .requires_grad_(
                        True
                    )
                )

                ig_baseline = (
                    blurred_latest
                    .detach()
                    .clone()
                )

                ig = IntegratedGradients(
                    risk_model
                )

                gc.collect()

                start = time.time()

                attributions, convergence_delta = ig.attribute(
                    inputs=ig_input,
                    baselines=ig_baseline,
                    n_steps=IG_STEPS,
                    method=IG_METHOD,
                    internal_batch_size=INTERNAL_BATCH_SIZE,
                    return_convergence_delta=True,
                )

                elapsed_seconds = (
                    time.time()
                    -
                    start
                )

                finite_fraction = float(
                    torch.isfinite(
                        attributions
                    )
                    .float()
                    .mean()
                    .item()
                )

                signed_sum = float(
                    attributions
                    .sum()
                    .item()
                )

                sum_abs = float(
                    attributions
                    .abs()
                    .sum()
                    .item()
                )

                mean_abs = float(
                    attributions
                    .abs()
                    .mean()
                    .item()
                )

                max_abs = float(
                    attributions
                    .abs()
                    .max()
                    .item()
                )

                delta_value = float(
                    convergence_delta
                    .detach()
                    .cpu()
                    .reshape(
                        -1
                    )[0]
                    .item()
                )

                absolute_completeness_error = abs(
                    signed_sum
                    -
                    risk_difference
                )

                relative_completeness_error = (
                    absolute_completeness_error
                    /
                    abs(
                        risk_difference
                    )
                    if abs(
                        risk_difference
                    )
                    >
                    1e-12
                    else np.nan
                )

                # ----------------------------------------------------------------------------------------------
                # Save attribution map.
                # ----------------------------------------------------------------------------------------------

                seed_dir = (
                    ATTRIBUTION_DIR
                    /
                    f"seed_{seed}"
                    /
                    f"{horizon_months}m"
                )

                seed_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                safe_landmark = landmark_id.replace(
                    "\\",
                    "_",
                ).replace(
                    "/",
                    "_",
                )

                attr_file = (
                    seed_dir
                    /
                    f"{subject_id}__{safe_landmark}__seed{seed}__{horizon_months}m.npz"
                )

                attr_np = (
                    attributions[
                        0,
                        0,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                # Save compressed attribution + minimal immutable metadata.
                np.savez_compressed(
                    attr_file,
                    attribution=attr_np,
                    subject_id=np.array(
                        subject_id
                    ),
                    landmark_id=np.array(
                        landmark_id
                    ),
                    seed=np.array(
                        seed,
                        dtype=np.int32,
                    ),
                    horizon_months=np.array(
                        horizon_months,
                        dtype=np.int32,
                    ),
                    actual_risk=np.array(
                        actual_risk,
                        dtype=np.float64,
                    ),
                    baseline_risk=np.array(
                        baseline_risk,
                        dtype=np.float64,
                    ),
                    risk_difference=np.array(
                        risk_difference,
                        dtype=np.float64,
                    ),
                    convergence_delta=np.array(
                        delta_value,
                        dtype=np.float64,
                    ),
                    spatial_shape=np.array(
                        MODEL_SPATIAL_SHAPE,
                        dtype=np.int32,
                    ),
                )

                attr_sha256 = sha256_file(
                    attr_file
                )

                record = {
                    "status":
                        "COMPLETE",

                    "subject_id":
                        subject_id,

                    "landmark_id":
                        landmark_id,

                    "event":
                        event_value,

                    "n_mri":
                        n_mri,

                    "history_bin":
                        history_bin,

                    "seed":
                        seed,

                    "checkpoint_epoch":
                        epoch,

                    "checkpoint_validation_nll":
                        val_nll,

                    "horizon_months":
                        horizon_months,

                    "baseline":
                        "blurred_latest_k9",

                    "ig_method":
                        IG_METHOD,

                    "ig_steps":
                        IG_STEPS,

                    "internal_batch_size":
                        INTERNAL_BATCH_SIZE,

                    "actual_risk":
                        actual_risk,

                    "baseline_risk":
                        baseline_risk,

                    "risk_difference":
                        risk_difference,

                    "signed_attribution_sum":
                        signed_sum,

                    "absolute_completeness_error":
                        absolute_completeness_error,

                    "relative_completeness_error":
                        (
                            float(
                                relative_completeness_error
                            )
                            if np.isfinite(
                                relative_completeness_error
                            )
                            else None
                        ),

                    "captum_convergence_delta":
                        delta_value,

                    "finite_fraction":
                        finite_fraction,

                    "sum_absolute_attribution":
                        sum_abs,

                    "mean_absolute_attribution":
                        mean_abs,

                    "max_absolute_attribution":
                        max_abs,

                    "elapsed_seconds":
                        elapsed_seconds,

                    "attribution_file":
                        str(
                            attr_file
                        ),

                    "attribution_sha256":
                        attr_sha256,
                }

                # Remove stale duplicate row for the same key, if any.
                existing_records = [
                    r
                    for r in existing_records
                    if not (
                        str(
                            r.get(
                                "subject_id"
                            )
                        )
                        ==
                        subject_id
                        and
                        int(
                            r.get(
                                "seed"
                            )
                        )
                        ==
                        seed
                        and
                        int(
                            r.get(
                                "horizon_months"
                            )
                        )
                        ==
                        horizon_months
                    )
                ]

                existing_records.append(
                    record
                )

                save_master(
                    existing_records
                )

                completed.add(
                    run_key
                )

                log(
                    f"  {horizon_months}m | "
                    f"risk={actual_risk:.6f} | "
                    f"baseline={baseline_risk:.6f} | "
                    f"diff={risk_difference:+.6f} | "
                    f"IGsum={signed_sum:+.6f} | "
                    f"abs_err={absolute_completeness_error:.3e} | "
                    f"delta={delta_value:+.3e} | "
                    f"time={elapsed_seconds:.2f}s | "
                    f"SAVED"
                )

                del (
                    risk_model,
                    ig_input,
                    ig_baseline,
                    ig,
                    attributions,
                    convergence_delta,
                    attr_np,
                )

                gc.collect()

            del (
                fixed_prior_embeddings,
                latest_tensor,
                blurred_latest,
                years_from_first,
                years_before_landmark,
            )

            gc.collect()

        del (
            model,
            checkpoint,
            state_dict,
        )

        gc.collect()

    # ----------------------------------------------------------------------------------------------------------
    # Final production summary.
    # ----------------------------------------------------------------------------------------------------------

    final_master = pd.read_csv(
        MASTER_CSV
    )

    complete_df = final_master.loc[
        final_master[
            "status"
        ]
        ==
        "COMPLETE"
    ].copy()

    unique_keys = complete_df[
        [
            "subject_id",
            "seed",
            "horizon_months",
        ]
    ].drop_duplicates()

    completed_runs = len(
        unique_keys
    )

    if (
        completed_runs
        !=
        expected_total_runs
    ):

        log("")
        log("=" * 120)
        log("TEST XAI PRODUCTION: INCOMPLETE")
        log("=" * 120)

        log(
            f"Completed: {completed_runs}/{expected_total_runs}"
        )

        log(
            "Re-run this same script to resume."
        )

        return

    all_files_exist = bool(
        complete_df[
            "attribution_file"
        ]
        .map(
            lambda p: Path(
                str(
                    p
                )
            ).exists()
        )
        .all()
    )

    all_finite = bool(
        np.all(
            np.isclose(
                complete_df[
                    "finite_fraction"
                ].to_numpy(
                    dtype=float
                ),
                1.0,
            )
        )
    )

    max_abs_completeness_error = float(
        complete_df[
            "absolute_completeness_error"
        ].max()
    )

    median_abs_completeness_error = float(
        complete_df[
            "absolute_completeness_error"
        ].median()
    )

    median_runtime = float(
        complete_df[
            "elapsed_seconds"
        ].median()
    )

    total_ig_runtime = float(
        complete_df[
            "elapsed_seconds"
        ].sum()
    )

    # Verify attribution hashes at end.
    hash_failures = []

    for _, row in complete_df.iterrows():

        attr_path = Path(
            str(
                row[
                    "attribution_file"
                ]
            )
        )

        observed = sha256_file(
            attr_path
        )

        expected = str(
            row[
                "attribution_sha256"
            ]
        )

        if (
            observed
            !=
            expected
        ):

            hash_failures.append(
                str(
                    attr_path
                )
            )

    summary = {
        "status":
            "COMPLETE",

        "model":
            "Frozen M0 five-seed ensemble constituents",

        "subjects":
            EXPECTED_COHORT_PARTICIPANTS,

        "seeds":
            SEEDS,

        "horizons_months":
            list(
                HORIZONS.keys()
            ),

        "expected_runs":
            expected_total_runs,

        "completed_runs":
            completed_runs,

        "attribution_files_all_exist":
            all_files_exist,

        "attribution_hash_failures":
            hash_failures,

        "all_attributions_finite":
            all_finite,

        "median_absolute_completeness_error":
            median_abs_completeness_error,

        "maximum_absolute_completeness_error":
            max_abs_completeness_error,

        "median_ig_runtime_seconds":
            median_runtime,

        "total_ig_runtime_seconds":
            total_ig_runtime,

        "xai_config_lock_sha256":
            sha256_file(
                XAI_CONFIG_LOCK
            ),

        "xai_cohort_lock_sha256":
            sha256_file(
                XAI_COHORT_LOCK
            ),

        "xai_cohort_csv_sha256":
            sha256_file(
                XAI_COHORT_CSV
            ),

        "master_csv_sha256":
            sha256_file(
                MASTER_CSV
            ),

        "next_phase":
            "cross-seed stability + top-k Dice + faithfulness",
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

    log("")
    log("=" * 120)
    log("FIVE-SEED TEST XAI PRODUCTION: COMPLETE")
    log("=" * 120)

    log(
        f"Completed runs: {completed_runs}/{expected_total_runs}"
    )

    log(
        f"Attribution files all exist: {all_files_exist}"
    )

    log(
        f"Hash failures: {len(hash_failures)}"
    )

    log(
        f"All attributions finite: {all_finite}"
    )

    log(
        f"Median absolute completeness error: "
        f"{median_abs_completeness_error:.6e}"
    )

    log(
        f"Maximum absolute completeness error: "
        f"{max_abs_completeness_error:.6e}"
    )

    log(
        f"Median IG runtime/run: "
        f"{median_runtime:.2f}s"
    )

    log(
        f"Total IG runtime: "
        f"{total_ig_runtime / 60.0:.2f} minutes"
    )

    log(
        f"Master CSV: {MASTER_CSV}"
    )

    log(
        f"Summary JSON: {SUMMARY_JSON}"
    )

    log(
        "Next step: cross-seed stability + top-k Dice + faithfulness"
    )

    log("=" * 120)


if __name__ == "__main__":

    try:

        main()

    except Exception:

        log("")
        log("=" * 120)
        log("FIVE-SEED TEST XAI PRODUCTION: FAILED")
        log("=" * 120)

        log(
            traceback.format_exc()
        )

        log(
            "The run is resume-safe. Fix the error and rerun this same script."
        )

        raise
