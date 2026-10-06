from pathlib import Path
import ast
import gc
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn

from exact_m0_model_v3 import ExactM0LongitudinalSurvivalModel_V3


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "06_splits"
MODEL_DIR = ROOT / "07_models"
RESULT_DIR = ROOT / "08_results"

LONGITUDINAL_DATASET_FILE = SPLIT_DIR / "ADNI_LONGITUDINAL_LANDMARK_DATASET_V1.csv"
CHECKPOINT_FILE = MODEL_DIR / "ADNI_LONGITUDINAL_SURVIVAL_BEST_V1.pt"

OUTPUT_CSV = RESULT_DIR / "ADNI_XAI_ZERO_BASELINE_PATH_DIAGNOSTIC_V1.csv"
OUTPUT_JSON = RESULT_DIR / "ADNI_XAI_ZERO_BASELINE_PATH_DIAGNOSTIC_V1.json"
OUTPUT_LOG = RESULT_DIR / "ADNI_XAI_ZERO_BASELINE_PATH_DIAGNOSTIC_V1.log"

EXPECTED_MODEL_PARAMETERS = 4_269_109
EXPECTED_VALIDATION_LANDMARKS = 521
EXPECTED_VALIDATION_SUBJECTS = 146
EXPECTED_EPOCH = 12
EXPECTED_VAL_NLL = 0.9486606425046921

SOURCE_SPATIAL_SHAPE = (197, 233, 189)
MODEL_SPATIAL_SHAPE = (160, 192, 160)
DAYS_PER_YEAR = 365.25
HORIZON_MONTHS = 24
HORIZON_INDEX = 1
DEVICE = torch.device("cpu")

ALPHAS = np.array([
    0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4,
    3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1,
    5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1.0
], dtype=np.float64)

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

RESULT_DIR.mkdir(parents=True, exist_ok=True)


def log(message=""):
    text = str(message)
    print(text, flush=True)
    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()


with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
    f.write("")


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

    raise ValueError(f"Unable to parse list field: {text[:300]}")


def center_crop_3d(volume, target_shape=MODEL_SPATIAL_SHAPE):
    source_shape = np.asarray(volume.shape, dtype=int)
    target_shape = np.asarray(target_shape, dtype=int)

    if np.any(source_shape < target_shape):
        raise ValueError(
            f"Source shape {tuple(source_shape)} is smaller than target {tuple(target_shape)}"
        )

    start = (source_shape - target_shape) // 2
    end = start + target_shape

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
        raise FileNotFoundError(f"MRI file not found: {path}")

    nii = nib.load(str(path))
    volume = np.asarray(nii.dataobj, dtype=np.float32)

    if volume.shape != SOURCE_SPATIAL_SHAPE:
        raise RuntimeError(
            f"Unexpected MRI shape {volume.shape}; expected {SOURCE_SPATIAL_SHAPE}"
        )

    if not np.isfinite(volume).all():
        raise RuntimeError(f"Non-finite MRI values: {path}")

    volume = center_crop_3d(volume)

    return (
        torch.from_numpy(np.ascontiguousarray(volume))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=DEVICE, dtype=torch.float32)
    )


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

        self.horizon_index = int(horizon_index)

    def forward(self, latest_mri_input):
        latest_embedding = self.frozen_model.mri_encoder(latest_mri_input)
        batch_size = latest_mri_input.shape[0]

        earlier_embedding = self.fixed_earlier_embedding.expand(batch_size, -1)
        mri_embeddings = torch.stack(
            [earlier_embedding, latest_embedding],
            dim=1,
        )

        years_from_first = self.fixed_years_from_first.expand(batch_size, -1)
        years_before_landmark = self.fixed_years_before_landmark.expand(batch_size, -1)
        visit_mask = self.fixed_visit_mask.expand(batch_size, -1)

        time_embeddings = self.frozen_model.time_encoder(
            years_from_first,
            years_before_landmark,
        )

        x = mri_embeddings + time_embeddings
        mask_float = visit_mask.unsqueeze(-1).to(x.dtype)
        x = x * mask_float

        x = self.frozen_model.transformer(
            x,
            src_key_padding_mask=(~visit_mask),
        )

        x = self.frozen_model.final_norm(x)

        pooled = (
            (x * mask_float).sum(dim=1)
            / mask_float.sum(dim=1).clamp_min(1.0)
        )

        hazard_logits = self.frozen_model.survival_head(pooled)
        hazards = torch.sigmoid(hazard_logits)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        cumulative_risk = 1.0 - survival

        return cumulative_risk[:, self.horizon_index]


def main():
    log("=" * 120)
    log("VALIDATION-ONLY ZERO-BASELINE IG PATH DIAGNOSTIC")
    log("=" * 120)
    log(f"PID: {os.getpid()}")
    log(f"Python: {sys.version.splitlines()[0]}")
    log(f"PyTorch: {torch.__version__}")
    log(f"Device: {DEVICE}")
    log(f"Target horizon: {HORIZON_MONTHS} months")
    log("TEST data accessed: NO")
    log("")

    model = ExactM0LongitudinalSurvivalModel_V3()

    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != EXPECTED_MODEL_PARAMETERS:
        raise RuntimeError(
            f"Parameter mismatch: {parameter_count:,} != {EXPECTED_MODEL_PARAMETERS:,}"
        )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint["model_state_dict"]

    if set(model.state_dict().keys()) != set(state_dict.keys()):
        raise RuntimeError("Model/checkpoint state-dict key mismatch.")

    for key, tensor in model.state_dict().items():
        if tensor.shape != state_dict[key].shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: "
                f"{tuple(tensor.shape)} != {tuple(state_dict[key].shape)}"
            )

    model.load_state_dict(state_dict, strict=True)

    epoch = checkpoint.get("epoch")
    val_nll = float(checkpoint.get("validation_loss"))

    if epoch != EXPECTED_EPOCH:
        raise RuntimeError(
            f"Expected epoch {EXPECTED_EPOCH}, found {epoch}"
        )

    if abs(val_nll - EXPECTED_VAL_NLL) >= 1e-10:
        raise RuntimeError(
            f"Validation NLL mismatch: {val_nll} != {EXPECTED_VAL_NLL}"
        )

    model = model.to(DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    log(f"Parameters: {parameter_count:,}")
    log(f"Epoch: {epoch}")
    log(f"Validation NLL: {val_nll}")
    log("Strict checkpoint load: PASS")

    df = pd.read_csv(LONGITUDINAL_DATASET_FILE)

    val_df = (
        df.loc[
            df["split"].astype(str).str.upper() == "VALIDATION"
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(val_df) != EXPECTED_VALIDATION_LANDMARKS:
        raise RuntimeError("Validation landmark count mismatch.")

    if val_df["subject_id"].nunique() != EXPECTED_VALIDATION_SUBJECTS:
        raise RuntimeError("Validation participant count mismatch.")

    candidate_df = (
        val_df.loc[val_df["n_mri"] == 2]
        .copy()
        .sort_values("landmark_id")
        .reset_index(drop=True)
    )

    sample = candidate_df.iloc[0]

    subject_id = str(sample["subject_id"])
    landmark_id = str(sample["landmark_id"])
    image_paths = parse_list_field(sample["image_paths"])
    days_from_first = np.asarray(
        parse_list_field(sample["days_from_first_mri"]),
        dtype=np.float32,
    )
    days_before_landmark = np.asarray(
        parse_list_field(sample["days_before_landmark"]),
        dtype=np.float32,
    )

    log("")
    log("VALIDATION SAMPLE")
    log("-" * 120)
    log(f"Subject: {subject_id}")
    log(f"Landmark: {landmark_id}")
    log(f"Event: {int(sample['event'])}")
    log("MRI visits: 2")

    earlier_tensor = load_mri_tensor(image_paths[0])
    latest_tensor = load_mri_tensor(image_paths[1])

    visit_mask = torch.ones(
        (1, 2),
        dtype=torch.bool,
        device=DEVICE,
    )

    years_from_first = (
        torch.tensor(
            days_from_first / DAYS_PER_YEAR,
            dtype=torch.float32,
            device=DEVICE,
        )
        .unsqueeze(0)
    )

    years_before_landmark = (
        torch.tensor(
            days_before_landmark / DAYS_PER_YEAR,
            dtype=torch.float32,
            device=DEVICE,
        )
        .unsqueeze(0)
    )

    with torch.inference_mode():
        earlier_embedding = model.mri_encoder(
            earlier_tensor
        ).detach().clone()

    risk_model = LatestVisitConditionalRiskWrapper(
        frozen_model=model,
        fixed_earlier_embedding=earlier_embedding,
        fixed_years_from_first=years_from_first,
        fixed_years_before_landmark=years_before_landmark,
        fixed_visit_mask=visit_mask,
        horizon_index=HORIZON_INDEX,
    ).to(DEVICE)

    risk_model.eval()

    zero_latest = torch.zeros_like(latest_tensor)

    with torch.inference_mode():
        baseline_risk = float(risk_model(zero_latest).item())
        original_risk = float(risk_model(latest_tensor).item())

    risk_difference = original_risk - baseline_risk

    log("")
    log("REFERENCE RISKS")
    log("-" * 120)
    log(f"Baseline risk F(0): {baseline_risk:.8f}")
    log(f"Original risk F(1): {original_risk:.8f}")
    log(f"F(1)-F(0): {risk_difference:.8f}")

    log("")
    log("ZERO -> ACTUAL MRI PATH")
    log("-" * 120)

    rows = []

    for alpha_value in ALPHAS:
        gc.collect()

        alpha = torch.tensor(
            float(alpha_value),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )

        scaled_input = latest_tensor * alpha
        risk = risk_model(scaled_input)[0]

        derivative = torch.autograd.grad(
            outputs=risk,
            inputs=alpha,
            retain_graph=False,
            create_graph=False,
        )[0]

        risk_value = float(risk.detach().cpu().item())
        derivative_value = float(derivative.detach().cpu().item())

        rows.append(
            {
                "alpha": float(alpha_value),
                "risk": risk_value,
                "d_risk_d_alpha": derivative_value,
                "risk_minus_baseline": risk_value - baseline_risk,
            }
        )

        log(
            f"alpha={alpha_value:10.8g} | "
            f"risk={risk_value:.8f} | "
            f"dF/dalpha={derivative_value:.8e} | "
            f"F(a)-F(0)={risk_value-baseline_risk:.8f}"
        )

        del alpha, scaled_input, risk, derivative

    path_df = pd.DataFrame(rows)

    alpha_np = path_df["alpha"].to_numpy(dtype=np.float64)
    deriv_np = path_df["d_risk_d_alpha"].to_numpy(dtype=np.float64)
    risk_np = path_df["risk"].to_numpy(dtype=np.float64)

    trapezoid_integral = float(
        np.trapz(
            deriv_np,
            alpha_np,
        )
    )

    integral_absolute_error = abs(
        trapezoid_integral - risk_difference
    )

    integral_relative_error = (
        integral_absolute_error / abs(risk_difference)
        if abs(risk_difference) > 1e-12
        else np.nan
    )

    risk_jumps = np.abs(np.diff(risk_np))
    max_jump_index = int(np.argmax(risk_jumps))
    max_adjacent_risk_jump = float(risk_jumps[max_jump_index])

    jump_alpha_left = float(alpha_np[max_jump_index])
    jump_alpha_right = float(alpha_np[max_jump_index + 1])

    max_abs_derivative_index = int(
        np.argmax(np.abs(deriv_np))
    )
    max_abs_derivative = float(
        np.abs(deriv_np[max_abs_derivative_index])
    )
    max_abs_derivative_alpha = float(
        alpha_np[max_abs_derivative_index]
    )

    near_zero_mask = path_df["alpha"] <= 1e-3
    near_zero_risk_range = float(
        path_df.loc[near_zero_mask, "risk"].max()
        -
        path_df.loc[near_zero_mask, "risk"].min()
    )

    likely_zero_baseline_pathology = bool(
        (near_zero_risk_range > 0.05)
        or
        (integral_relative_error > 0.25)
    )

    log("")
    log("=" * 120)
    log("PATH DIAGNOSTIC SUMMARY")
    log("=" * 120)
    log(f"F(1)-F(0): {risk_difference:.8f}")
    log(
        f"Trapezoid integral of dF/dalpha: "
        f"{trapezoid_integral:.8f}"
    )
    log(
        f"Integral absolute error: "
        f"{integral_absolute_error:.8f}"
    )
    log(
        f"Integral relative error: "
        f"{integral_relative_error:.6f}"
    )
    log(
        f"Maximum adjacent risk jump: "
        f"{max_adjacent_risk_jump:.8f}"
    )
    log(
        f"Maximum jump interval: "
        f"[{jump_alpha_left:g}, {jump_alpha_right:g}]"
    )
    log(
        f"Maximum |dF/dalpha|: "
        f"{max_abs_derivative:.8e}"
    )
    log(
        f"Alpha at maximum |dF/dalpha|: "
        f"{max_abs_derivative_alpha:g}"
    )
    log(
        f"Risk range over alpha <= 1e-3: "
        f"{near_zero_risk_range:.8f}"
    )
    log(
        "Zero-baseline path pathology flag: "
        f"{'YES' if likely_zero_baseline_pathology else 'NO'}"
    )

    path_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    output = {
        "analysis":
            "Validation-only zero-baseline IG path diagnostic",
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
        "target_horizon_months":
            HORIZON_MONTHS,
        "baseline_definition":
            "zero-valued normalized latest MRI",
        "path_definition":
            "x(alpha) = alpha * latest MRI",
        "baseline_risk":
            baseline_risk,
        "original_risk":
            original_risk,
        "risk_difference":
            risk_difference,
        "trapezoid_integral_dF_dalpha":
            trapezoid_integral,
        "integral_absolute_error":
            integral_absolute_error,
        "integral_relative_error":
            (
                float(integral_relative_error)
                if np.isfinite(integral_relative_error)
                else None
            ),
        "max_adjacent_risk_jump":
            max_adjacent_risk_jump,
        "max_jump_interval":
            [jump_alpha_left, jump_alpha_right],
        "max_abs_derivative":
            max_abs_derivative,
        "max_abs_derivative_alpha":
            max_abs_derivative_alpha,
        "near_zero_risk_range_alpha_le_1e_3":
            near_zero_risk_range,
        "zero_baseline_pathology_flag":
            likely_zero_baseline_pathology,
        "test_data_accessed":
            False,
        "model_modified":
            False,
        "retraining":
            False,
        "recalibration":
            False,
        "path_rows":
            rows,
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
    log(f"Saved CSV : {OUTPUT_CSV}")
    log(f"Saved JSON: {OUTPUT_JSON}")
    log(f"Saved log : {OUTPUT_LOG}")
    log("")
    log("TEST XAI remains untouched.")
    log("Do not change the baseline until this diagnostic is reviewed.")
    log("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("")
        log("=" * 120)
        log("ZERO-BASELINE PATH DIAGNOSTIC: FAILED")
        log("=" * 120)
        log(traceback.format_exc())
        raise
