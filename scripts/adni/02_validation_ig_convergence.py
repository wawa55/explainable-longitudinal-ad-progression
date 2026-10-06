from pathlib import Path
import ast, gc, json, time, traceback
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
DATA_FILE = SPLIT_DIR / "ADNI_LONGITUDINAL_LANDMARK_DATASET_V1.csv"
CKPT_FILE = MODEL_DIR / "ADNI_LONGITUDINAL_SURVIVAL_BEST_V1.pt"
OUT_CSV = RESULT_DIR / "ADNI_XAI_IG_VALIDATION_CONVERGENCE_V1.csv"
OUT_JSON = RESULT_DIR / "ADNI_XAI_IG_VALIDATION_CONVERGENCE_V1.json"
OUT_LOG = RESULT_DIR / "ADNI_XAI_IG_VALIDATION_CONVERGENCE_V1.log"

EXPECTED_PARAMS = 4_269_109
EXPECTED_EPOCH = 12
EXPECTED_VAL_NLL = 0.9486606425046921
SOURCE_SHAPE = (197, 233, 189)
MODEL_SHAPE = (160, 192, 160)
DAYS_PER_YEAR = 365.25
HORIZON_INDEX = 1
HORIZON_MONTHS = 24
IG_STEPS_LIST = [4, 8, 16, 32, 64]
IG_METHOD = "gausslegendre"
DEVICE = torch.device("cpu")

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

RESULT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_LOG, "w", encoding="utf-8") as f:
    f.write("")

def log(msg=""):
    print(msg, flush=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
        f.flush()

def parse_list_field(value):
    if isinstance(value, list): return value
    if isinstance(value, np.ndarray): return value.tolist()
    if pd.isna(value): return []
    text = str(value).strip()
    if not text: return []
    try:
        x = json.loads(text)
        if isinstance(x, list): return x
    except Exception:
        pass
    try:
        x = ast.literal_eval(text)
        if isinstance(x, (list, tuple)): return list(x)
    except Exception:
        pass
    raise ValueError(f"Cannot parse list field: {text[:300]}")

def center_crop_3d(vol, target_shape=MODEL_SHAPE):
    src = np.asarray(vol.shape)
    tgt = np.asarray(target_shape)
    if np.any(src < tgt):
        raise ValueError(f"Source shape {tuple(src)} smaller than target {tuple(tgt)}")
    start = (src - tgt) // 2
    end = start + tgt
    out = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    if out.shape != tuple(target_shape):
        raise RuntimeError(f"Crop failed: {out.shape}")
    return out

def load_mri_tensor(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    vol = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    if vol.shape != SOURCE_SHAPE:
        raise RuntimeError(f"Unexpected MRI shape {vol.shape}")
    if not np.isfinite(vol).all():
        raise RuntimeError(f"Non-finite MRI values: {path}")
    vol = center_crop_3d(vol)
    return torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0).unsqueeze(0).to(DEVICE)

def cumulative_risk_from_logits(logits):
    hazards = torch.sigmoid(logits)
    surv = torch.cumprod(1.0 - hazards, dim=1)
    return hazards, 1.0 - surv

class LatestVisitConditionalRiskWrapper(nn.Module):
    def __init__(self, model, earlier_emb, years_from_first, years_before_landmark, visit_mask, horizon_index):
        super().__init__()
        self.model = model
        self.register_buffer("earlier_emb", earlier_emb.detach().clone())
        self.register_buffer("years_from_first", years_from_first.detach().clone())
        self.register_buffer("years_before_landmark", years_before_landmark.detach().clone())
        self.register_buffer("visit_mask", visit_mask.detach().clone())
        self.horizon_index = int(horizon_index)

    def forward(self, latest_mri):
        b = latest_mri.shape[0]
        latest_emb = self.model.mri_encoder(latest_mri)
        earlier = self.earlier_emb.expand(b, -1)
        mri_emb = torch.stack([earlier, latest_emb], dim=1)
        yff = self.years_from_first.expand(b, -1)
        ybl = self.years_before_landmark.expand(b, -1)
        vm = self.visit_mask.expand(b, -1)
        x = mri_emb + self.model.time_encoder(yff, ybl)
        mf = vm.unsqueeze(-1).to(x.dtype)
        x = self.model.transformer(x * mf, src_key_padding_mask=(~vm))
        x = self.model.final_norm(x)
        pooled = (x * mf).sum(dim=1) / mf.sum(dim=1).clamp_min(1.0)
        logits = self.model.survival_head(pooled)
        hazards = torch.sigmoid(logits)
        risks = 1.0 - torch.cumprod(1.0 - hazards, dim=1)
        return risks[:, self.horizon_index]

def save_progress(results, meta):
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({**meta, "results": results}, f, indent=2)

def main():
    log("=" * 120)
    log("VALIDATION-ONLY IG CONVERGENCE STUDY")
    log("=" * 120)
    log(f"PyTorch: {torch.__version__}")
    log(f"Device: {DEVICE}")
    log(f"IG steps: {IG_STEPS_LIST}")
    log("TEST data accessed: NO")

    model = ExactM0LongitudinalSurvivalModel_V3()
    nparams = sum(p.numel() for p in model.parameters())
    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(f"Parameter mismatch: {nparams}")

    ckpt = torch.load(CKPT_FILE, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if ckpt.get("epoch") != EXPECTED_EPOCH:
        raise RuntimeError("Checkpoint epoch mismatch")
    if abs(float(ckpt.get("validation_loss")) - EXPECTED_VAL_NLL) >= 1e-10:
        raise RuntimeError("Checkpoint validation NLL mismatch")
    model.to(DEVICE).eval()
    for p in model.parameters(): p.requires_grad_(False)
    log(f"Model parameters: {nparams:,}")
    log("Strict checkpoint load: PASS")

    df = pd.read_csv(DATA_FILE)
    val = df.loc[df["split"].astype(str).str.upper().eq("VALIDATION")].copy()
    if len(val) != 521 or val["subject_id"].nunique() != 146:
        raise RuntimeError("Validation cohort count mismatch")
    sample = val.loc[val["n_mri"].eq(2)].sort_values("landmark_id").iloc[0]
    subject_id = str(sample["subject_id"])
    landmark_id = str(sample["landmark_id"])
    image_paths = parse_list_field(sample["image_paths"])
    days_from_first = np.asarray(parse_list_field(sample["days_from_first_mri"]), dtype=np.float32)
    days_before_landmark = np.asarray(parse_list_field(sample["days_before_landmark"]), dtype=np.float32)
    log(f"Subject: {subject_id}")
    log(f"Landmark: {landmark_id}")

    earlier = load_mri_tensor(image_paths[0])
    latest = load_mri_tensor(image_paths[1])
    vm = torch.ones((1, 2), dtype=torch.bool, device=DEVICE)
    yff = torch.tensor(days_from_first / DAYS_PER_YEAR, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    ybl = torch.tensor(days_before_landmark / DAYS_PER_YEAR, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    full = torch.stack([earlier, latest], dim=1)
    with torch.inference_mode():
        _, risks = cumulative_risk_from_logits(model(full, vm, yff, ybl))
    original_risk = float(risks[0, HORIZON_INDEX].item())

    zero_latest = torch.zeros_like(latest)
    base = torch.stack([earlier, zero_latest], dim=1)
    with torch.inference_mode():
        _, base_risks = cumulative_risk_from_logits(model(base, vm, yff, ybl))
    baseline_risk = float(base_risks[0, HORIZON_INDEX].item())
    risk_difference = original_risk - baseline_risk
    log(f"Original {HORIZON_MONTHS}m risk: {original_risk:.8f}")
    log(f"Baseline risk: {baseline_risk:.8f}")
    log(f"Risk difference: {risk_difference:.8f}")
    del full, base, risks, base_risks
    gc.collect()

    with torch.inference_mode():
        earlier_emb = model.mri_encoder(earlier).detach().clone()
    del earlier
    gc.collect()

    wrapper = LatestVisitConditionalRiskWrapper(model, earlier_emb, yff, ybl, vm, HORIZON_INDEX).to(DEVICE).eval()
    with torch.inference_mode():
        wo = float(wrapper(latest).item())
        wb = float(wrapper(zero_latest).item())
    if not np.isclose(wo, original_risk, atol=1e-6, rtol=0.0):
        raise RuntimeError("Wrapper original mismatch")
    if not np.isclose(wb, baseline_risk, atol=1e-6, rtol=0.0):
        raise RuntimeError("Wrapper baseline mismatch")
    log("Wrapper consistency: PASS")

    results = []
    meta = {
        "analysis": "Validation-only IG convergence study",
        "model": "Exact frozen M0 V3",
        "model_parameters": EXPECTED_PARAMS,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_validation_nll": float(ckpt["validation_loss"]),
        "subject_id": subject_id,
        "landmark_id": landmark_id,
        "split": "VALIDATION",
        "target_horizon_months": HORIZON_MONTHS,
        "baseline": "zero-valued normalized latest MRI",
        "fixed_context": "earlier MRI embedding and longitudinal timing",
        "ig_method": IG_METHOD,
        "ig_steps_tested": IG_STEPS_LIST,
        "test_data_accessed": False,
        "model_modified": False,
        "retraining": False,
        "recalibration": False,
    }

    previous_signed = None
    for n_steps in IG_STEPS_LIST:
        gc.collect()
        ig_input = latest.detach().clone().requires_grad_(True)
        ig_baseline = torch.zeros_like(ig_input)
        ig = IntegratedGradients(wrapper)
        t0 = time.time()
        attrs, delta = ig.attribute(
            inputs=ig_input,
            baselines=ig_baseline,
            n_steps=n_steps,
            method=IG_METHOD,
            internal_batch_size=1,
            return_convergence_delta=True,
        )
        elapsed = time.time() - t0
        finite_fraction = float(torch.isfinite(attrs).float().mean().item())
        abs_attr = attrs.abs()
        signed_sum = float(attrs.sum().item())
        sum_abs = float(abs_attr.sum().item())
        mean_abs = float(abs_attr.mean().item())
        max_abs = float(abs_attr.max().item())
        delta_value = float(delta.detach().cpu().reshape(-1)[0].item())
        abs_err = abs(signed_sum - risk_difference)
        rel_err = abs_err / abs(risk_difference) if abs(risk_difference) > 1e-12 else np.nan
        signed_change = abs(signed_sum - previous_signed) if previous_signed is not None else np.nan

        rec = {
            "subject_id": subject_id,
            "landmark_id": landmark_id,
            "split": "VALIDATION",
            "horizon_months": HORIZON_MONTHS,
            "n_steps": n_steps,
            "ig_method": IG_METHOD,
            "original_risk": original_risk,
            "baseline_risk": baseline_risk,
            "risk_difference": risk_difference,
            "signed_attribution_sum": signed_sum,
            "absolute_completeness_error": abs_err,
            "relative_completeness_error": float(rel_err),
            "captum_convergence_delta": delta_value,
            "sum_absolute_attribution": sum_abs,
            "mean_absolute_attribution": mean_abs,
            "max_absolute_attribution": max_abs,
            "finite_fraction": finite_fraction,
            "signed_sum_change_from_previous": None if np.isnan(signed_change) else float(signed_change),
            "elapsed_seconds": elapsed,
        }
        results.append(rec)
        save_progress(results, meta)
        log(
            f"n_steps={n_steps:3d} | signed_sum={signed_sum:.8f} | "
            f"rel_error={rel_err:.6f} | delta={delta_value:.8f} | time={elapsed:.3f}s"
        )
        previous_signed = signed_sum
        del attrs, delta, abs_attr, ig_input, ig_baseline, ig
        gc.collect()

    out = pd.DataFrame(results)
    log("")
    log("=" * 120)
    log("IG CONVERGENCE SUMMARY")
    log("=" * 120)
    log(out[[
        "n_steps",
        "signed_attribution_sum",
        "absolute_completeness_error",
        "relative_completeness_error",
        "captum_convergence_delta",
        "signed_sum_change_from_previous",
        "elapsed_seconds",
    ]].to_string(index=False))
    log("")
    log(f"Saved CSV : {OUT_CSV}")
    log(f"Saved JSON: {OUT_JSON}")
    log(f"Saved log : {OUT_LOG}")
    log("TEST XAI remains untouched.")
    log("Do not select the final IG step count until this table is reviewed.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("")
        log("VALIDATION IG CONVERGENCE STUDY: FAILED")
        log(traceback.format_exc())
        raise
