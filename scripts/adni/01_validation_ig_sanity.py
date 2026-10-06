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
from captum.attr import IntegratedGradients

from exact_m0_model_v3 import ExactM0LongitudinalSurvivalModel_V3


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "06_splits"
MODEL_DIR = ROOT / "07_models"
RESULT_DIR = ROOT / "08_results"

LONGITUDINAL_DATASET_FILE = SPLIT_DIR / "ADNI_LONGITUDINAL_LANDMARK_DATASET_V1.csv"
CHECKPOINT_FILE = MODEL_DIR / "ADNI_LONGITUDINAL_SURVIVAL_BEST_V1.pt"

OUTPUT_JSON = RESULT_DIR / "ADNI_XAI_IG_VALIDATION_LATEST_VISIT_STANDALONE_SANITY_V1.json"
OUTPUT_LOG = RESULT_DIR / "ADNI_XAI_IG_VALIDATION_LATEST_VISIT_STANDALONE_SANITY_V1.log"

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

IG_STEPS = 4
IG_METHOD = "gausslegendre"
DEVICE = torch.device("cpu")

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
            f"Source shape {tuple(source_shape)} smaller than target {tuple(target_shape)}"
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


def cumulative_risk_from_logits(logits):
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    risks = 1.0 - survival
    return hazards, risks


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
    log("STANDALONE VALIDATION IG SANITY — EXACT FROZEN M0 V3")
    log("=" * 120)
    log(f"PID: {os.getpid()}")
    log(f"Python: {sys.version.splitlines()[0]}")
    log(f"PyTorch: {torch.__version__}")
    log(f"Device: {DEVICE}")
    log(f"CPU threads: {torch.get_num_threads()}")
    log("")

    for label, path in {
        "Longitudinal dataset": LONGITUDINAL_DATASET_FILE,
        "Frozen checkpoint": CHECKPOINT_FILE,
    }.items():
        log(f"{label}: {'PASS' if path.exists() else 'MISSING'} | {path}")
        if not path.exists():
            raise FileNotFoundError(path)

    log("")
    log("MODEL / CHECKPOINT AUDIT")
    log("-" * 120)

    model = ExactM0LongitudinalSurvivalModel_V3()
    parameter_count = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {parameter_count:,}")

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
                f"Shape mismatch for {key}: {tuple(tensor.shape)} != {tuple(state_dict[key].shape)}"
            )

    model.load_state_dict(state_dict, strict=True)

    epoch = checkpoint.get("epoch")
    val_nll = float(checkpoint.get("validation_loss"))

    log(f"Epoch: {epoch}")
    log(f"Validation NLL: {val_nll}")
    log("Strict checkpoint load: PASS")

    if epoch != EXPECTED_EPOCH:
        raise RuntimeError(f"Expected epoch {EXPECTED_EPOCH}, found {epoch}")

    if abs(val_nll - EXPECTED_VAL_NLL) >= 1e-10:
        raise RuntimeError(
            f"Validation NLL mismatch: {val_nll} != {EXPECTED_VAL_NLL}"
        )

    model = model.to(DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    log("")
    log("VALIDATION DATA")
    log("-" * 120)

    df = pd.read_csv(LONGITUDINAL_DATASET_FILE)

    val_df = (
        df.loc[
            df["split"].astype(str).str.upper() == "VALIDATION"
        ]
        .copy()
        .reset_index(drop=True)
    )

    log(f"Landmarks: {len(val_df)}")
    log(f"Participants: {val_df['subject_id'].nunique()}")

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

    if len(candidate_df) == 0:
        raise RuntimeError("No two-MRI validation sample found.")

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

    if len(image_paths) != 2:
        raise RuntimeError(
            f"Expected exactly 2 MRI visits, found {len(image_paths)}"
        )

    log(f"Subject: {subject_id}")
    log(f"Landmark: {landmark_id}")
    log(f"Event: {int(sample['event'])}")
    log("MRI visits: 2")

    log("")
    log("MRI LOAD")
    log("-" * 120)

    earlier_tensor = load_mri_tensor(image_paths[0])
    latest_tensor = load_mri_tensor(image_paths[1])

    log(f"Earlier tensor: {tuple(earlier_tensor.shape)}")
    log(f"Latest tensor: {tuple(latest_tensor.shape)}")

    images = torch.stack(
        [earlier_tensor, latest_tensor],
        dim=1,
    )

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

    log("")
    log("FULL-HISTORY FORWARD PASS")
    log("-" * 120)

    start = time.time()

    with torch.inference_mode():
        logits = model(
            images=images,
            visit_mask=visit_mask,
            years_from_first=years_from_first,
            years_before_landmark=years_before_landmark,
        )
        _, risks = cumulative_risk_from_logits(logits)

    forward_seconds = time.time() - start
    risk_24m = float(risks[0, HORIZON_INDEX].item())

    log(f"Forward seconds: {forward_seconds:.3f}")
    log(f"24m risk: {risk_24m:.8f}")

    zero_latest = torch.zeros_like(latest_tensor)

    baseline_images = torch.stack(
        [earlier_tensor, zero_latest],
        dim=1,
    )

    with torch.inference_mode():
        baseline_logits = model(
            images=baseline_images,
            visit_mask=visit_mask,
            years_from_first=years_from_first,
            years_before_landmark=years_before_landmark,
        )
        _, baseline_risks = cumulative_risk_from_logits(baseline_logits)

    baseline_risk_24m = float(
        baseline_risks[0, HORIZON_INDEX].item()
    )
    risk_difference = risk_24m - baseline_risk_24m

    log(f"Zero-latest baseline risk: {baseline_risk_24m:.8f}")
    log(f"Risk difference: {risk_difference:.8f}")

    del images, baseline_images, logits, risks, baseline_logits, baseline_risks
    gc.collect()

    log("")
    log("FIXED EARLIER CONTEXT")
    log("-" * 120)

    with torch.inference_mode():
        earlier_embedding = model.mri_encoder(
            earlier_tensor
        ).detach().clone()

    log(f"Earlier embedding: {tuple(earlier_embedding.shape)}")

    del earlier_tensor
    gc.collect()

    log("")
    log("WRAPPER CONSISTENCY")
    log("-" * 120)

    risk_model = LatestVisitConditionalRiskWrapper(
        frozen_model=model,
        fixed_earlier_embedding=earlier_embedding,
        fixed_years_from_first=years_from_first,
        fixed_years_before_landmark=years_before_landmark,
        fixed_visit_mask=visit_mask,
        horizon_index=HORIZON_INDEX,
    ).to(DEVICE)

    risk_model.eval()

    with torch.inference_mode():
        wrapper_original = float(risk_model(latest_tensor).item())
        wrapper_baseline = float(risk_model(zero_latest).item())

    log(f"Wrapper original: {wrapper_original:.8f}")
    log(f"Wrapper baseline: {wrapper_baseline:.8f}")

    if not np.isclose(wrapper_original, risk_24m, atol=1e-6, rtol=0.0):
        raise RuntimeError("Wrapper original risk mismatch.")

    if not np.isclose(wrapper_baseline, baseline_risk_24m, atol=1e-6, rtol=0.0):
        raise RuntimeError("Wrapper baseline risk mismatch.")

    log("Wrapper consistency: PASS")

    log("")
    log("INTEGRATED GRADIENTS")
    log("-" * 120)
    log(f"IG steps: {IG_STEPS}")
    log(f"IG method: {IG_METHOD}")
    log("internal_batch_size: 1")

    ig_input = (
        latest_tensor
        .detach()
        .clone()
        .requires_grad_(True)
    )
    ig_baseline = torch.zeros_like(ig_input)

    del latest_tensor, zero_latest
    gc.collect()

    ig = IntegratedGradients(risk_model)

    start = time.time()

    attributions, convergence_delta = ig.attribute(
        inputs=ig_input,
        baselines=ig_baseline,
        n_steps=IG_STEPS,
        method=IG_METHOD,
        internal_batch_size=1,
        return_convergence_delta=True,
    )

    ig_seconds = time.time() - start
    log(f"IG complete in {ig_seconds:.3f} seconds")

    abs_attr = attributions.abs()

    finite_fraction = float(
        torch.isfinite(attributions)
        .float()
        .mean()
        .item()
    )
    mean_abs_attr = float(abs_attr.mean().item())
    max_abs_attr = float(abs_attr.max().item())
    sum_abs_attr = float(abs_attr.sum().item())
    signed_attr_sum = float(attributions.sum().item())

    delta_value = float(
        convergence_delta.detach().cpu().reshape(-1)[0].item()
    )

    absolute_completeness_error = abs(
        signed_attr_sum - risk_difference
    )

    relative_completeness_error = (
        absolute_completeness_error / abs(risk_difference)
        if abs(risk_difference) > 1e-12
        else None
    )

    log("")
    log("IG NUMERICAL AUDIT")
    log("-" * 120)
    log(f"Finite fraction: {finite_fraction:.8f}")
    log(f"Mean |IG|: {mean_abs_attr:.12e}")
    log(f"Max |IG|: {max_abs_attr:.12e}")
    log(f"Sum |IG|: {sum_abs_attr:.8f}")
    log(f"Signed IG sum: {signed_attr_sum:.8f}")
    log(f"Risk difference: {risk_difference:.8f}")
    log(f"Absolute completeness error: {absolute_completeness_error:.8f}")
    log(f"Relative completeness error: {relative_completeness_error}")
    log(f"Captum convergence delta: {delta_value:.8f}")

    if finite_fraction != 1.0:
        raise RuntimeError("Non-finite IG values.")

    if sum_abs_attr <= 0:
        raise RuntimeError("IG attribution is all zero.")

    audit = {
        "analysis": "Standalone validation-only latest-visit IG sanity",
        "subject_id": subject_id,
        "landmark_id": landmark_id,
        "split": "VALIDATION",
        "event": int(sample["event"]),
        "n_mri": 2,
        "model": "Exact frozen M0 V3",
        "model_parameters": EXPECTED_MODEL_PARAMETERS,
        "checkpoint_epoch": epoch,
        "checkpoint_validation_nll": val_nll,
        "device": str(DEVICE),
        "target_horizon_months": HORIZON_MONTHS,
        "original_risk": wrapper_original,
        "baseline_risk": wrapper_baseline,
        "risk_difference": risk_difference,
        "ig_steps": IG_STEPS,
        "ig_method": IG_METHOD,
        "finite_fraction": finite_fraction,
        "mean_absolute_attribution": mean_abs_attr,
        "max_absolute_attribution": max_abs_attr,
        "sum_absolute_attribution": sum_abs_attr,
        "signed_attribution_sum": signed_attr_sum,
        "absolute_completeness_error": absolute_completeness_error,
        "relative_completeness_error": relative_completeness_error,
        "captum_convergence_delta": delta_value,
        "forward_seconds": forward_seconds,
        "ig_seconds": ig_seconds,
        "test_data_accessed": False,
        "test_xai_performed": False,
        "model_modified": False,
        "retraining": False,
        "recalibration": False,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    log("")
    log("=" * 120)
    log("STANDALONE VALIDATION IG SANITY: PASS")
    log("=" * 120)
    log(f"Saved JSON: {OUTPUT_JSON}")
    log(f"Saved log : {OUTPUT_LOG}")
    log("Next step: validation-only IG convergence study.")
    log("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("")
        log("=" * 120)
        log("STANDALONE VALIDATION IG SANITY: FAILED")
        log("=" * 120)
        log(traceback.format_exc())
        raise
