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

OUTPUT_CSV = (
    RESULT_DIR / "ADNI_XAI_VALIDATION_BASELINE_COMPARISON_V1.csv"
)

OUTPUT_PATH_CSV = (
    RESULT_DIR / "ADNI_XAI_VALIDATION_BASELINE_PATHS_V1.csv"
)

OUTPUT_JSON = (
    RESULT_DIR / "ADNI_XAI_VALIDATION_BASELINE_COMPARISON_V1.json"
)

OUTPUT_LOG = (
    RESULT_DIR / "ADNI_XAI_VALIDATION_BASELINE_COMPARISON_V1.log"
)

EXPECTED_MODEL_PARAMETERS = 4_269_109
EXPECTED_VALIDATION_LANDMARKS = 521
EXPECTED_VALIDATION_SUBJECTS = 146
EXPECTED_EPOCH = 12
EXPECTED_VAL_NLL = 0.9486606425046921

SOURCE_SPATIAL_SHAPE = (197, 233, 189)
MODEL_SPATIAL_SHAPE = (160, 192, 160)

DAYS_PER_YEAR = 365.25

# We compare both primary XAI horizons before freezing the baseline.
HORIZONS = {
    24: 1,
    36: 2,
}

# Validation-only convergence ladder.
IG_STEPS_LIST = [8, 16, 32, 64]
IG_METHOD = "gausslegendre"
INTERNAL_BATCH_SIZE = 1

# Path diagnostic grid from baseline (alpha=0) to latest MRI (alpha=1).
PATH_ALPHAS = np.array(
    [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
     0.60, 0.70, 0.80, 0.90, 1.0],
    dtype=np.float64,
)

# Blurred-latest sensitivity baseline.
# A 9x9x9 local-mean filter preserves coarse anatomy while suppressing local detail.
BLUR_KERNEL_SIZE = 9

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

    with open(
        OUTPUT_LOG,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(text + "\n")
        f.flush()


with open(
    OUTPUT_LOG,
    "w",
    encoding="utf-8",
) as f:
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

    volume = center_crop_3d(
        volume
    )

    return (
        torch.from_numpy(
            np.ascontiguousarray(volume)
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

    # Replicate padding avoids artificial zero borders.
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


class LatestVisitConditionalRiskWrapper(nn.Module):

    def __init__(
        self,
        frozen_model,
        fixed_earlier_embedding,
        fixed_years_from_first,
        fixed_years_before_landmark,
        fixed_visit_mask,
        horizon_index,
    ):

        super().__init__()

        self.frozen_model = frozen_model

        self.register_buffer(
            "fixed_earlier_embedding",
            fixed_earlier_embedding.detach().clone(),
        )

        self.register_buffer(
            "fixed_years_from_first",
            fixed_years_from_first.detach().clone(),
        )

        self.register_buffer(
            "fixed_years_before_landmark",
            fixed_years_before_landmark.detach().clone(),
        )

        self.register_buffer(
            "fixed_visit_mask",
            fixed_visit_mask.detach().clone(),
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

        earlier_embedding = (
            self.fixed_earlier_embedding
            .expand(
                batch_size,
                -1,
            )
        )

        mri_embeddings = torch.stack(
            [
                earlier_embedding,
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

        visit_mask = (
            self.fixed_visit_mask
            .expand(
                batch_size,
                -1,
            )
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
            1.0 - hazards,
            dim=1,
        )

        cumulative_risk = (
            1.0 - survival
        )

        return cumulative_risk[
            :,
            self.horizon_index
        ]


def run_path_diagnostic(
    risk_model,
    baseline,
    actual,
    baseline_name,
    horizon_months,
):

    rows = []

    with torch.inference_mode():
        baseline_risk = float(
            risk_model(
                baseline
            ).item()
        )

        actual_risk = float(
            risk_model(
                actual
            ).item()
        )

    for alpha_value in PATH_ALPHAS:

        gc.collect()

        alpha = torch.tensor(
            float(alpha_value),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )

        interpolated = (
            baseline
            +
            alpha
            *
            (
                actual
                -
                baseline
            )
        )

        risk = risk_model(
            interpolated
        )[0]

        derivative = torch.autograd.grad(
            outputs=risk,
            inputs=alpha,
            retain_graph=False,
            create_graph=False,
        )[0]

        rows.append(
            {
                "baseline":
                    baseline_name,

                "horizon_months":
                    horizon_months,

                "alpha":
                    float(
                        alpha_value
                    ),

                "risk":
                    float(
                        risk.detach().cpu().item()
                    ),

                "d_risk_d_alpha":
                    float(
                        derivative.detach().cpu().item()
                    ),
            }
        )

        del (
            alpha,
            interpolated,
            risk,
            derivative,
        )

    path_df = pd.DataFrame(
        rows
    )

    alpha_np = path_df[
        "alpha"
    ].to_numpy(
        dtype=np.float64
    )

    derivative_np = path_df[
        "d_risk_d_alpha"
    ].to_numpy(
        dtype=np.float64
    )

    risk_np = path_df[
        "risk"
    ].to_numpy(
        dtype=np.float64
    )

    integral = float(
        np.trapz(
            derivative_np,
            alpha_np,
        )
    )

    risk_difference = (
        actual_risk
        -
        baseline_risk
    )

    abs_error = abs(
        integral
        -
        risk_difference
    )

    rel_error = (
        abs_error
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

    max_jump = float(
        np.max(
            np.abs(
                np.diff(
                    risk_np
                )
            )
        )
    )

    max_abs_derivative = float(
        np.max(
            np.abs(
                derivative_np
            )
        )
    )

    summary = {
        "baseline_risk":
            baseline_risk,

        "actual_risk":
            actual_risk,

        "risk_difference":
            risk_difference,

        "path_integral":
            integral,

        "path_absolute_error":
            abs_error,

        "path_relative_error":
            (
                float(
                    rel_error
                )
                if np.isfinite(
                    rel_error
                )
                else None
            ),

        "max_adjacent_risk_jump":
            max_jump,

        "max_abs_d_risk_d_alpha":
            max_abs_derivative,
    }

    return (
        rows,
        summary,
    )


def run_ig_convergence(
    risk_model,
    baseline,
    actual,
    baseline_name,
    horizon_months,
):

    with torch.inference_mode():

        baseline_risk = float(
            risk_model(
                baseline
            ).item()
        )

        actual_risk = float(
            risk_model(
                actual
            ).item()
        )

    risk_difference = (
        actual_risk
        -
        baseline_risk
    )

    results = []

    previous_signed_sum = None

    for n_steps in IG_STEPS_LIST:

        gc.collect()

        ig_input = (
            actual
            .detach()
            .clone()
            .requires_grad_(
                True
            )
        )

        ig_baseline = (
            baseline
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

        sum_abs = float(
            attributions
            .abs()
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

        relative_error = (
            absolute_error
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

        previous_change = (
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
                "baseline":
                    baseline_name,

                "horizon_months":
                    horizon_months,

                "n_steps":
                    n_steps,

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

                "sum_absolute_attribution":
                    sum_abs,

                "finite_fraction":
                    finite_fraction,

                "signed_sum_change_from_previous":
                    (
                        float(
                            previous_change
                        )
                        if np.isfinite(
                            previous_change
                        )
                        else None
                    ),

                "elapsed_seconds":
                    elapsed_seconds,
            }
        )

        log(
            f"{baseline_name:18s} | "
            f"{horizon_months:2d}m | "
            f"steps={n_steps:3d} | "
            f"risk_diff={risk_difference:+.8f} | "
            f"signed_sum={signed_sum:+.8f} | "
            f"rel_err={relative_error:.6f} | "
            f"delta={delta_value:+.8f} | "
            f"time={elapsed_seconds:.3f}s"
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

    return results


# ==============================================================================================================
# MAIN
# ==============================================================================================================

def main():

    log("=" * 120)
    log("VALIDATION-ONLY IG BASELINE COMPARISON")
    log("=" * 120)

    log(f"PID: {os.getpid()}")
    log(f"Python: {sys.version.splitlines()[0]}")
    log(f"PyTorch: {torch.__version__}")
    log(f"Device: {DEVICE}")
    log(f"Horizons: {list(HORIZONS.keys())}")
    log(f"IG steps: {IG_STEPS_LIST}")
    log("Candidate baseline 1: previous longitudinal MRI")
    log(
        f"Candidate baseline 2: latest MRI local-mean blur "
        f"(kernel={BLUR_KERNEL_SIZE})"
    )
    log("Zero baseline is NOT retested as a candidate; it is the diagnosed negative control.")
    log("TEST data accessed: NO")
    log("")

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
    # 2. VALIDATION SAMPLE
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

    candidate_df = (
        val_df.loc[
            val_df[
                "n_mri"
            ]
            ==
            2
        ]
        .copy()
        .sort_values(
            "landmark_id"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            candidate_df
        )
        ==
        0
    ):
        raise RuntimeError(
            "No two-MRI validation sample found."
        )

    sample = candidate_df.iloc[
        0
    ]

    subject_id = str(
        sample[
            "subject_id"
        ]
    )

    landmark_id = str(
        sample[
            "landmark_id"
        ]
    )

    image_paths = parse_list_field(
        sample[
            "image_paths"
        ]
    )

    days_from_first = np.asarray(
        parse_list_field(
            sample[
                "days_from_first_mri"
            ]
        ),
        dtype=np.float32,
    )

    days_before_landmark = np.asarray(
        parse_list_field(
            sample[
                "days_before_landmark"
            ]
        ),
        dtype=np.float32,
    )

    log("")
    log("VALIDATION SAMPLE")
    log("-" * 120)
    log(
        f"Subject: {subject_id}"
    )
    log(
        f"Landmark: {landmark_id}"
    )
    log(
        f"Event: {int(sample['event'])}"
    )
    log(
        "MRI visits: 2"
    )

    # ----------------------------------------------------------------------------------------------------------
    # 3. MRI + FIXED LONGITUDINAL CONTEXT
    # ----------------------------------------------------------------------------------------------------------

    earlier_tensor = load_mri_tensor(
        image_paths[
            0
        ]
    )

    latest_tensor = load_mri_tensor(
        image_paths[
            1
        ]
    )

    visit_mask = torch.ones(
        (
            1,
            2,
        ),
        dtype=torch.bool,
        device=DEVICE,
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

    with torch.inference_mode():

        earlier_embedding = model.mri_encoder(
            earlier_tensor
        ).detach().clone()

    blurred_latest = make_local_mean_blur(
        latest_tensor
    ).detach()

    baseline_candidates = {
        "previous_mri":
            earlier_tensor.detach().clone(),

        "blurred_latest":
            blurred_latest,
    }

    # ----------------------------------------------------------------------------------------------------------
    # 4. RUN PATH + IG COMPARISON
    # ----------------------------------------------------------------------------------------------------------

    all_ig_rows = []
    all_path_rows = []
    path_summaries = []

    for horizon_months, horizon_index in HORIZONS.items():

        log("")
        log("=" * 120)
        log(
            f"HORIZON: {horizon_months} MONTHS"
        )
        log("=" * 120)

        risk_model = LatestVisitConditionalRiskWrapper(
            frozen_model=model,
            fixed_earlier_embedding=earlier_embedding,
            fixed_years_from_first=years_from_first,
            fixed_years_before_landmark=years_before_landmark,
            fixed_visit_mask=visit_mask,
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

        log(
            f"Actual latest-MRI risk: "
            f"{actual_risk:.8f}"
        )

        for baseline_name, baseline_tensor in baseline_candidates.items():

            log("")
            log(
                f"BASELINE: {baseline_name}"
            )
            log("-" * 120)

            path_rows, path_summary = run_path_diagnostic(
                risk_model=risk_model,
                baseline=baseline_tensor,
                actual=latest_tensor,
                baseline_name=baseline_name,
                horizon_months=horizon_months,
            )

            all_path_rows.extend(
                path_rows
            )

            path_summary_record = {
                "baseline":
                    baseline_name,

                "horizon_months":
                    horizon_months,

                **path_summary,
            }

            path_summaries.append(
                path_summary_record
            )

            log(
                f"Baseline risk: "
                f"{path_summary['baseline_risk']:.8f}"
            )

            log(
                f"Risk difference: "
                f"{path_summary['risk_difference']:+.8f}"
            )

            log(
                f"Path relative error: "
                f"{path_summary['path_relative_error']}"
            )

            log(
                f"Max adjacent path risk jump: "
                f"{path_summary['max_adjacent_risk_jump']:.8f}"
            )

            log(
                f"Max |dF/dalpha|: "
                f"{path_summary['max_abs_d_risk_d_alpha']:.8e}"
            )

            ig_rows = run_ig_convergence(
                risk_model=risk_model,
                baseline=baseline_tensor,
                actual=latest_tensor,
                baseline_name=baseline_name,
                horizon_months=horizon_months,
            )

            all_ig_rows.extend(
                ig_rows
            )

            # Save incrementally.
            pd.DataFrame(
                all_ig_rows
            ).to_csv(
                OUTPUT_CSV,
                index=False,
            )

            pd.DataFrame(
                all_path_rows
            ).to_csv(
                OUTPUT_PATH_CSV,
                index=False,
            )

            with open(
                OUTPUT_JSON,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    {
                        "analysis":
                            "Validation-only IG baseline comparison",

                        "model":
                            "Exact frozen M0 V3",

                        "model_parameters":
                            EXPECTED_MODEL_PARAMETERS,

                        "checkpoint_epoch":
                            epoch,

                        "checkpoint_validation_nll":
                            val_nll,

                        "subject_id":
                            subject_id,

                        "landmark_id":
                            landmark_id,

                        "split":
                            "VALIDATION",

                        "candidate_baselines":
                            [
                                "previous_mri",
                                "blurred_latest",
                            ],

                        "blur_kernel_size":
                            BLUR_KERNEL_SIZE,

                        "horizons_months":
                            list(
                                HORIZONS.keys()
                            ),

                        "ig_steps":
                            IG_STEPS_LIST,

                        "zero_baseline_status":
                            "rejected candidate due validation-only path pathology",

                        "test_data_accessed":
                            False,

                        "model_modified":
                            False,

                        "retraining":
                            False,

                        "recalibration":
                            False,

                        "path_summaries":
                            path_summaries,

                        "ig_results":
                            all_ig_rows,
                    },
                    f,
                    indent=2,
                )

        del risk_model
        gc.collect()

    # ----------------------------------------------------------------------------------------------------------
    # 5. FINAL SUMMARY
    # ----------------------------------------------------------------------------------------------------------

    ig_df = pd.DataFrame(
        all_ig_rows
    )

    path_summary_df = pd.DataFrame(
        path_summaries
    )

    log("")
    log("=" * 120)
    log("BASELINE PATH SUMMARY")
    log("=" * 120)

    log(
        path_summary_df[
            [
                "baseline",
                "horizon_months",
                "baseline_risk",
                "actual_risk",
                "risk_difference",
                "path_relative_error",
                "max_adjacent_risk_jump",
                "max_abs_d_risk_d_alpha",
            ]
        ].to_string(
            index=False
        )
    )

    log("")
    log("=" * 120)
    log("IG CONVERGENCE SUMMARY")
    log("=" * 120)

    log(
        ig_df[
            [
                "baseline",
                "horizon_months",
                "n_steps",
                "risk_difference",
                "signed_attribution_sum",
                "relative_completeness_error",
                "captum_convergence_delta",
                "signed_sum_change_from_previous",
                "elapsed_seconds",
            ]
        ].to_string(
            index=False
        )
    )

    log("")
    log(
        f"Saved IG CSV   : {OUTPUT_CSV}"
    )

    log(
        f"Saved path CSV : {OUTPUT_PATH_CSV}"
    )

    log(
        f"Saved JSON     : {OUTPUT_JSON}"
    )

    log(
        f"Saved log      : {OUTPUT_LOG}"
    )

    log("")
    log(
        "No baseline is automatically selected by this script."
    )

    log(
        "Review numerical completeness and path stability before freezing the final baseline."
    )

    log(
        "TEST XAI remains untouched."
    )

    log("=" * 120)


if __name__ == "__main__":

    try:
        main()

    except Exception:

        log("")
        log("=" * 120)
        log(
            "VALIDATION IG BASELINE COMPARISON: FAILED"
        )
        log("=" * 120)

        log(
            traceback.format_exc()
        )

        raise
