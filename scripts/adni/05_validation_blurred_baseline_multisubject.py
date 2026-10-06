from pathlib import Path
import ast
import gc
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F

from captum.attr import IntegratedGradients

from exact_m0_model_v3 import ExactM0LongitudinalSurvivalModel_V3


# ==============================================================================================================
# CONFIGURATION
# ==============================================================================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "06_splits"
MODEL_DIR = ROOT / "07_models"
RESULT_DIR = ROOT / "08_results"

LONGITUDINAL_DATASET_FILE = (
    SPLIT_DIR / "ADNI_LONGITUDINAL_LANDMARK_DATASET_V1.csv"
)

CHECKPOINT_FILE = (
    MODEL_DIR / "ADNI_LONGITUDINAL_SURVIVAL_BEST_V1.pt"
)

OUTPUT_SELECTION_CSV = (
    RESULT_DIR / "ADNI_XAI_VALIDATION_MULTISUBJECT_SELECTION_V1.csv"
)

OUTPUT_RESULTS_CSV = (
    RESULT_DIR / "ADNI_XAI_VALIDATION_BLURRED_BASELINE_MULTISUBJECT_V1.csv"
)

OUTPUT_JSON = (
    RESULT_DIR / "ADNI_XAI_VALIDATION_BLURRED_BASELINE_MULTISUBJECT_V1.json"
)

OUTPUT_LOG = (
    RESULT_DIR / "ADNI_XAI_VALIDATION_BLURRED_BASELINE_MULTISUBJECT_V1.log"
)

EXPECTED_MODEL_PARAMETERS = 4_269_109
EXPECTED_VALIDATION_LANDMARKS = 521
EXPECTED_VALIDATION_SUBJECTS = 146
EXPECTED_EPOCH = 12
EXPECTED_VAL_NLL = 0.9486606425046921

SOURCE_SPATIAL_SHAPE = (197, 233, 189)
MODEL_SPATIAL_SHAPE = (160, 192, 160)

DAYS_PER_YEAR = 365.25

HORIZONS = {
    24: 1,
    36: 2,
}

# Proposed frozen setting from single-subject validation comparison.
IG_STEPS_LIST = [16, 32]
IG_METHOD = "gausslegendre"
INTERNAL_BATCH_SIZE = 1

BLUR_KERNEL_SIZE = 9

# 12 unique validation participants:
# 6 event + 6 censored, with deterministic coverage of MRI-history bins.
TARGET_PER_CLASS = 6
PER_BIN_TARGET = 2

DEVICE = torch.device("cpu")

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================================================
# LOGGING
# ==============================================================================================================

def log(message=""):
    text = str(message)
    print(text, flush=True)
    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()


with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
    f.write("")


# ==============================================================================================================
# HELPERS
# ==============================================================================================================

def parse_list_field(value):

    if isinstance(value, list):
        return value

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except Exception:
        pass

    raise ValueError(
        f"Unable to parse list field: {text[:300]}"
    )


def center_crop_3d(volume, target_shape=MODEL_SPATIAL_SHAPE):

    source_shape = np.asarray(
        volume.shape,
        dtype=int,
    )

    target_shape = np.asarray(
        target_shape,
        dtype=int,
    )

    if np.any(source_shape < target_shape):
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

    if cropped.shape != tuple(target_shape):
        raise RuntimeError(
            f"Crop failed: {cropped.shape} != {tuple(target_shape)}"
        )

    return cropped


def load_mri_tensor(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"MRI file not found: {path}"
        )

    nii = nib.load(
        str(path)
    )

    volume = np.asarray(
        nii.dataobj,
        dtype=np.float32,
    )

    if volume.shape != SOURCE_SPATIAL_SHAPE:
        raise RuntimeError(
            f"Unexpected MRI shape {volume.shape}; "
            f"expected {SOURCE_SPATIAL_SHAPE}"
        )

    if not np.isfinite(volume).all():
        raise RuntimeError(
            f"Non-finite MRI values: {path}"
        )

    cropped = center_crop_3d(
        volume
    )

    return (
        torch.from_numpy(
            np.ascontiguousarray(cropped)
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(
            device=DEVICE,
            dtype=torch.float32,
        )
    )


def make_local_mean_blur(
    x,
    kernel_size=BLUR_KERNEL_SIZE,
):

    if kernel_size % 2 != 1:
        raise ValueError(
            "BLUR_KERNEL_SIZE must be odd."
        )

    pad = kernel_size // 2

    padded = F.pad(
        x,
        (
            pad, pad,
            pad, pad,
            pad, pad,
        ),
        mode="replicate",
    )

    blurred = F.avg_pool3d(
        padded,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
    )

    if blurred.shape != x.shape:
        raise RuntimeError(
            f"Blurred shape mismatch: "
            f"{tuple(blurred.shape)} != {tuple(x.shape)}"
        )

    return blurred


def n_mri_bin(n):

    n = int(n)

    if n == 2:
        return "2"

    if 3 <= n <= 4:
        return "3-4"

    return "5+"


class GeneralLatestVisitConditionalRiskWrapper(nn.Module):

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

        batch_size = latest_mri_input.shape[0]

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
                mri_embeddings.shape[1],
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
            .unsqueeze(-1)
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


def select_validation_landmarks(
    val_df,
):

    # One deterministic landmark per participant:
    # choose latest landmark, which generally gives the richest MRI history.
    work = val_df.copy()

    work["landmark_date_sort"] = pd.to_datetime(
        work["landmark_date"],
        errors="coerce",
    )

    work = (
        work.sort_values(
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
    )

    work = work.loc[
        work["n_mri"] >= 2
    ].copy()

    work["history_bin"] = work[
        "n_mri"
    ].map(
        n_mri_bin
    )

    work = work.sort_values(
        [
            "event",
            "history_bin",
            "subject_id",
            "landmark_id",
        ]
    ).reset_index(
        drop=True
    )

    selected_indices = []

    # First, deterministic coverage across history bins.
    for event_value in [0, 1]:

        class_df = work.loc[
            work["event"].astype(int)
            ==
            event_value
        ].copy()

        class_selected = []

        for history_bin in [
            "2",
            "3-4",
            "5+",
        ]:

            bin_df = class_df.loc[
                class_df["history_bin"]
                ==
                history_bin
            ].sort_values(
                [
                    "subject_id",
                    "landmark_id",
                ]
            )

            take = min(
                PER_BIN_TARGET,
                len(
                    bin_df
                ),
            )

            class_selected.extend(
                bin_df.head(
                    take
                ).index.tolist()
            )

        # Fill deterministically if a bin had too few candidates.
        if (
            len(
                class_selected
            )
            <
            TARGET_PER_CLASS
        ):

            remaining = class_df.loc[
                ~class_df.index.isin(
                    class_selected
                )
            ].sort_values(
                [
                    "subject_id",
                    "landmark_id",
                ]
            )

            need = (
                TARGET_PER_CLASS
                -
                len(
                    class_selected
                )
            )

            class_selected.extend(
                remaining.head(
                    need
                ).index.tolist()
            )

        selected_indices.extend(
            class_selected[
                :TARGET_PER_CLASS
            ]
        )

    selected = (
        work.loc[
            selected_indices
        ]
        .copy()
        .sort_values(
            [
                "event",
                "history_bin",
                "subject_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        selected[
            "subject_id"
        ].nunique()
        !=
        len(
            selected
        )
    ):
        raise RuntimeError(
            "Selection contains repeated participants."
        )

    if (
        len(
            selected
        )
        !=
        2
        *
        TARGET_PER_CLASS
    ):
        raise RuntimeError(
            f"Expected "
            f"{2 * TARGET_PER_CLASS} selected participants; "
            f"found {len(selected)}."
        )

    return selected


# ==============================================================================================================
# MAIN
# ==============================================================================================================

def main():

    log("=" * 120)
    log("VALIDATION-ONLY MULTI-SUBJECT CONFIRMATION — BLURRED-LATEST IG BASELINE")
    log("=" * 120)

    log(
        f"PID: {os.getpid()}"
    )

    log(
        f"Python: {sys.version.splitlines()[0]}"
    )

    log(
        f"PyTorch: {torch.__version__}"
    )

    log(
        f"Device: {DEVICE}"
    )

    log(
        f"Horizons: {list(HORIZONS.keys())}"
    )

    log(
        f"IG steps: {IG_STEPS_LIST}"
    )

    log(
        f"Blur kernel: {BLUR_KERNEL_SIZE}x{BLUR_KERNEL_SIZE}x{BLUR_KERNEL_SIZE}"
    )

    log(
        f"Target participants: {2 * TARGET_PER_CLASS}"
    )

    log(
        "TEST data accessed: NO"
    )

    # ----------------------------------------------------------------------------------------------------------
    # 1. EXACT FROZEN MODEL
    # ----------------------------------------------------------------------------------------------------------

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
            f"Parameter mismatch: "
            f"{parameter_count:,} != "
            f"{EXPECTED_MODEL_PARAMETERS:,}"
        )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=False,
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
            "Model/checkpoint state-dict key mismatch."
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
                f"Shape mismatch for {key}: "
                f"{tuple(tensor.shape)} != "
                f"{tuple(state_dict[key].shape)}"
            )

    model.load_state_dict(
        state_dict,
        strict=True,
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
        EXPECTED_EPOCH
    ):
        raise RuntimeError(
            f"Expected epoch {EXPECTED_EPOCH}, "
            f"found {epoch}"
        )

    if (
        abs(
            val_nll
            -
            EXPECTED_VAL_NLL
        )
        >=
        1e-10
    ):
        raise RuntimeError(
            f"Validation NLL mismatch: "
            f"{val_nll} != "
            f"{EXPECTED_VAL_NLL}"
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
        f"Parameters: {parameter_count:,}"
    )

    log(
        f"Epoch: {epoch}"
    )

    log(
        f"Validation NLL: {val_nll}"
    )

    log(
        "Strict checkpoint load: PASS"
    )

    # ----------------------------------------------------------------------------------------------------------
    # 2. VALIDATION SELECTION
    # ----------------------------------------------------------------------------------------------------------

    df = pd.read_csv(
        LONGITUDINAL_DATASET_FILE
    )

    val_df = (
        df.loc[
            df[
                "split"
            ]
            .astype(
                str
            )
            .str.upper()
            ==
            "VALIDATION"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            val_df
        )
        !=
        EXPECTED_VALIDATION_LANDMARKS
    ):
        raise RuntimeError(
            "Validation landmark count mismatch."
        )

    if (
        val_df[
            "subject_id"
        ].nunique()
        !=
        EXPECTED_VALIDATION_SUBJECTS
    ):
        raise RuntimeError(
            "Validation participant count mismatch."
        )

    selected = select_validation_landmarks(
        val_df
    )

    selection_columns = [
        "subject_id",
        "landmark_id",
        "landmark_date",
        "event",
        "n_mri",
        "history_bin",
        "time_to_event_or_censor_days",
    ]

    selected[
        selection_columns
    ].to_csv(
        OUTPUT_SELECTION_CSV,
        index=False,
    )

    log("")
    log("VALIDATION SELECTION")
    log("-" * 120)

    log(
        selected[
            [
                "subject_id",
                "landmark_id",
                "event",
                "n_mri",
                "history_bin",
            ]
        ].to_string(
            index=False
        )
    )

    log("")

    log(
        "Selection counts by event/history:"
    )

    log(
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
    # 3. RUN MULTI-SUBJECT CONFIRMATION
    # ----------------------------------------------------------------------------------------------------------

    results = []

    total_subjects = len(
        selected
    )

    for subject_index, (_, row) in enumerate(
        selected.iterrows(),
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

        log("")
        log("=" * 120)
        log(
            f"SUBJECT {subject_index}/{total_subjects}: "
            f"{subject_id} | "
            f"event={event_value} | "
            f"n_mri={n_mri} | "
            f"bin={history_bin}"
        )
        log("=" * 120)

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
                f"{subject_id}: image_paths length "
                f"{len(image_paths)} != n_mri {n_mri}"
            )

        if (
            len(
                days_from_first
            )
            !=
            n_mri
        ):
            raise RuntimeError(
                f"{subject_id}: days_from_first length mismatch."
            )

        if (
            len(
                days_before_landmark
            )
            !=
            n_mri
        ):
            raise RuntimeError(
                f"{subject_id}: days_before_landmark length mismatch."
            )

        # Encode every prior MRI independently, then release raw prior volumes.
        prior_embeddings_list = []

        for prior_path in image_paths[:-1]:

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

        for horizon_months, horizon_index in HORIZONS.items():

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

            previous_signed_sum = None

            for n_steps in IG_STEPS_LIST:

                gc.collect()

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

                start = time.time()

                attributions, convergence_delta = ig.attribute(
                    inputs=ig_input,
                    baselines=ig_baseline,
                    n_steps=n_steps,
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

                delta_value = float(
                    convergence_delta
                    .detach()
                    .cpu()
                    .reshape(
                        -1
                    )[0]
                    .item()
                )

                absolute_error = abs(
                    signed_sum
                    -
                    risk_difference
                )

                if (
                    abs(
                        risk_difference
                    )
                    >
                    1e-12
                ):

                    relative_error = (
                        absolute_error
                        /
                        abs(
                            risk_difference
                        )
                    )

                else:

                    relative_error = np.nan

                signed_sum_change = (
                    np.nan
                    if previous_signed_sum is None
                    else abs(
                        signed_sum
                        -
                        previous_signed_sum
                    )
                )

                results.append(
                    {
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

                        "horizon_months":
                            horizon_months,

                        "n_steps":
                            n_steps,

                        "baseline":
                            "blurred_latest_k9",

                        "baseline_risk":
                            baseline_risk,

                        "actual_risk":
                            actual_risk,

                        "risk_difference":
                            risk_difference,

                        "signed_attribution_sum":
                            signed_sum,

                        "absolute_completeness_error":
                            absolute_error,

                        "relative_completeness_error":
                            (
                                float(
                                    relative_error
                                )
                                if np.isfinite(
                                    relative_error
                                )
                                else None
                            ),

                        "captum_convergence_delta":
                            delta_value,

                        "finite_fraction":
                            finite_fraction,

                        "signed_sum_change_from_previous":
                            (
                                float(
                                    signed_sum_change
                                )
                                if np.isfinite(
                                    signed_sum_change
                                )
                                else None
                            ),

                        "elapsed_seconds":
                            elapsed_seconds,
                    }
                )

                log(
                    f"{horizon_months:2d}m | "
                    f"steps={n_steps:2d} | "
                    f"risk_diff={risk_difference:+.8f} | "
                    f"signed_sum={signed_sum:+.8f} | "
                    f"rel_err={relative_error:.6f} | "
                    f"delta={delta_value:+.8e} | "
                    f"time={elapsed_seconds:.2f}s"
                )

                previous_signed_sum = (
                    signed_sum
                )

                del (
                    ig_input,
                    ig_baseline,
                    ig,
                    attributions,
                    convergence_delta,
                )

                gc.collect()

                # Save after every completed run.
                pd.DataFrame(
                    results
                ).to_csv(
                    OUTPUT_RESULTS_CSV,
                    index=False,
                )

            del risk_model
            gc.collect()

        del (
            fixed_prior_embeddings,
            latest_tensor,
            blurred_latest,
            years_from_first,
            years_before_landmark,
        )

        gc.collect()

    # ----------------------------------------------------------------------------------------------------------
    # 4. SUMMARY
    # ----------------------------------------------------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    summary = (
        results_df.groupby(
            [
                "horizon_months",
                "n_steps",
            ],
            as_index=False,
        )
        .agg(
            n_runs=(
                "subject_id",
                "size",
            ),

            median_abs_error=(
                "absolute_completeness_error",
                "median",
            ),

            max_abs_error=(
                "absolute_completeness_error",
                "max",
            ),

            median_rel_error=(
                "relative_completeness_error",
                "median",
            ),

            max_rel_error=(
                "relative_completeness_error",
                "max",
            ),

            median_abs_delta=(
                "captum_convergence_delta",
                lambda x: np.median(
                    np.abs(
                        x
                    )
                ),
            ),

            max_abs_delta=(
                "captum_convergence_delta",
                lambda x: np.max(
                    np.abs(
                        x
                    )
                ),
            ),

            all_finite=(
                "finite_fraction",
                lambda x: bool(
                    np.all(
                        np.isclose(
                            x,
                            1.0,
                        )
                    )
                ),
            ),

            median_elapsed_seconds=(
                "elapsed_seconds",
                "median",
            ),
        )
    )

    # Per-run 16 vs 32 stability.
    pivot = results_df.pivot_table(
        index=[
            "subject_id",
            "horizon_months",
        ],
        columns="n_steps",
        values="signed_attribution_sum",
        aggfunc="first",
    )

    if (
        16 in pivot.columns
        and
        32 in pivot.columns
    ):

        pivot[
            "abs_signed_sum_change_16_to_32"
        ] = (
            pivot[
                32
            ]
            -
            pivot[
                16
            ]
        ).abs()

        stability_median = float(
            pivot[
                "abs_signed_sum_change_16_to_32"
            ].median()
        )

        stability_max = float(
            pivot[
                "abs_signed_sum_change_16_to_32"
            ].max()
        )

    else:

        stability_median = np.nan
        stability_max = np.nan

    log("")
    log("=" * 120)
    log("MULTI-SUBJECT VALIDATION SUMMARY")
    log("=" * 120)

    log(
        summary.to_string(
            index=False
        )
    )

    log("")

    log(
        f"Median |signed IG sum change| 16->32: "
        f"{stability_median:.8e}"
    )

    log(
        f"Maximum |signed IG sum change| 16->32: "
        f"{stability_max:.8e}"
    )

    # Proposed acceptance evidence, not automatic methodological lock.
    step16 = results_df.loc[
        results_df[
            "n_steps"
        ]
        ==
        16
    ].copy()

    all_step16_finite = bool(
        np.all(
            np.isclose(
                step16[
                    "finite_fraction"
                ],
                1.0,
            )
        )
    )

    # Relative error can become unstable for tiny risk differences,
    # so absolute error is the primary numerical criterion.
    step16_max_abs_error = float(
        step16[
            "absolute_completeness_error"
        ].max()
    )

    step16_median_abs_error = float(
        step16[
            "absolute_completeness_error"
        ].median()
    )

    log("")
    log("PROPOSED 16-STEP NUMERICAL CHECK")
    log("-" * 120)

    log(
        f"All finite: {all_step16_finite}"
    )

    log(
        f"Median absolute completeness error: "
        f"{step16_median_abs_error:.8e}"
    )

    log(
        f"Maximum absolute completeness error: "
        f"{step16_max_abs_error:.8e}"
    )

    output = {
        "analysis":
            "Validation-only multi-subject blurred-latest IG confirmation",

        "model":
            "Exact frozen M0 V3",

        "model_parameters":
            EXPECTED_MODEL_PARAMETERS,

        "checkpoint_epoch":
            epoch,

        "checkpoint_validation_nll":
            val_nll,

        "split":
            "VALIDATION",

        "selected_participants":
            int(
                selected[
                    "subject_id"
                ].nunique()
            ),

        "selected_landmarks":
            int(
                len(
                    selected
                )
            ),

        "event_participants":
            int(
                selected[
                    "event"
                ].astype(
                    int
                ).sum()
            ),

        "censored_participants":
            int(
                (
                    selected[
                        "event"
                    ].astype(
                        int
                    )
                    ==
                    0
                ).sum()
            ),

        "baseline":
            "blurred latest MRI",

        "blur_kernel_size":
            BLUR_KERNEL_SIZE,

        "blur_padding":
            "replicate",

        "ig_method":
            IG_METHOD,

        "ig_steps_tested":
            IG_STEPS_LIST,

        "horizons_months":
            list(
                HORIZONS.keys()
            ),

        "multi_visit_wrapper":
            "all prior MRI embeddings fixed; latest MRI differentiable",

        "step16_all_finite":
            all_step16_finite,

        "step16_median_absolute_completeness_error":
            step16_median_abs_error,

        "step16_max_absolute_completeness_error":
            step16_max_abs_error,

        "median_abs_signed_sum_change_16_to_32":
            (
                float(
                    stability_median
                )
                if np.isfinite(
                    stability_median
                )
                else None
            ),

        "max_abs_signed_sum_change_16_to_32":
            (
                float(
                    stability_max
                )
                if np.isfinite(
                    stability_max
                )
                else None
            ),

        "test_data_accessed":
            False,

        "model_modified":
            False,

        "retraining":
            False,

        "recalibration":
            False,

        "selection":
            selected[
                selection_columns
            ].to_dict(
                orient="records"
            ),

        "summary":
            summary.to_dict(
                orient="records"
            ),
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

    log("")
    log(
        f"Saved selection CSV: {OUTPUT_SELECTION_CSV}"
    )

    log(
        f"Saved results CSV  : {OUTPUT_RESULTS_CSV}"
    )

    log(
        f"Saved JSON         : {OUTPUT_JSON}"
    )

    log(
        f"Saved log          : {OUTPUT_LOG}"
    )

    log("")
    log(
        "TEST XAI remains untouched."
    )

    log(
        "Do not freeze the production XAI setting until this summary is reviewed."
    )

    log("=" * 120)


if __name__ == "__main__":

    try:

        main()

    except Exception:

        log("")
        log("=" * 120)
        log(
            "MULTI-SUBJECT BLURRED-BASELINE VALIDATION: FAILED"
        )
        log("=" * 120)

        log(
            traceback.format_exc()
        )

        raise
