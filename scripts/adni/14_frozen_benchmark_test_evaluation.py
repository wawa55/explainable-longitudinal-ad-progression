from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ==============================================================================================================
# ADNI — STEP 14: ONE FROZEN BENCHMARK TEST EVALUATION
# ==============================================================================================================
# Purpose
# -------
# Evaluate the already-trained frozen benchmark suite B0/B1/B2 on the untouched TEST split.
#
# Frozen rules enforced here
# --------------------------
# 1. Use only BEST checkpoints from the five frozen seeds:
#       [2026, 3407, 5813, 7291, 9109]
# 2. Evaluate ALL benchmark architectures B0/B1/B2 on the identical frozen TEST cohort.
# 3. For each architecture, ensemble by arithmetic mean of the five seed cumulative risks.
# 4. No TEST tuning, checkpoint selection, model removal, threshold optimization, or hyperparameter changes.
# 5. Censoring distribution is estimated from TRAIN only.
# 6. Uncertainty uses 2,000 participant-cluster bootstrap percentile confidence intervals.
# 7. This script is resume-safe for per-seed inference. Re-running after interruption does not alter the protocol.
#
# Important
# ---------
# This script dynamically imports Step 13 so that dataset loading, MRI preprocessing, target construction,
# model definitions, and frozen constants are IDENTICAL to benchmark training.
# ==============================================================================================================


ROOT = Path(__file__).resolve().parents[2]
SCRIPT13 = ROOT / "scripts" / "13_train_frozen_benchmark_models.py"

RESULT_DIR = ROOT / "08_results"
MODEL_DIR = ROOT / "07_models"

LOCK_FILE = RESULT_DIR / "ADNI_BENCHMARK_SUITE_LOCK_V1.json"
TRAINING_SUMMARY = RESULT_DIR / "benchmark_v1" / "ADNI_BENCHMARK_TRAINING_SUMMARY_V1.csv"
DATASET_FILE = ROOT / "06_splits" / "ADNI_LONGITUDINAL_LANDMARK_DATASET_V1.csv"

BENCHMARK_MODEL_DIR = MODEL_DIR / "benchmark_v1"
OUT_DIR = RESULT_DIR / "benchmark_test_v1"
PRED_DIR = OUT_DIR / "seed_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

RUN_LOG = OUT_DIR / "ADNI_FROZEN_BENCHMARK_TEST_EVALUATION_V1.log"
STATUS_JSON = OUT_DIR / "ADNI_FROZEN_BENCHMARK_TEST_STATUS_V1.json"
AUDIT_JSON = OUT_DIR / "ADNI_FROZEN_BENCHMARK_TEST_AUDIT_V1.json"
ENSEMBLE_CSV = OUT_DIR / "ADNI_FROZEN_BENCHMARK_TEST_ENSEMBLE_PREDICTIONS_V1.csv"
METRICS_CSV = OUT_DIR / "ADNI_FROZEN_BENCHMARK_TEST_METRICS_V1.csv"
BOOTSTRAP_CSV = OUT_DIR / "ADNI_FROZEN_BENCHMARK_TEST_BOOTSTRAP_CI_V1.csv"
VALIDATION_AUDIT_CSV = OUT_DIR / "ADNI_BENCHMARK_VALIDATION_AUDIT_V1.csv"

SEEDS = [2026, 3407, 5813, 7291, 9109]
MODELS = ["B0", "B1", "B2"]
HORIZON_MONTHS = np.asarray([12, 24, 36, 48, 60], dtype=int)
DAYS_PER_YEAR = 365.25
HORIZON_DAYS = HORIZON_MONTHS.astype(float) / 12.0 * DAYS_PER_YEAR
TAU_DAYS = 5.0 * DAYS_PER_YEAR

EXPECTED = {
    "subjects": 972,
    "landmarks": 3634,
    "train_subjects": 680,
    "validation_subjects": 146,
    "test_subjects": 146,
    "train_landmarks": 2530,
    "validation_landmarks": 521,
    "test_landmarks": 583,
}

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260912
EPS = 1e-8


def log(message=""):
    text = str(message)
    print(text, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_step13_module():
    if not SCRIPT13.exists():
        raise FileNotFoundError(f"Missing Step 13 script:\n{SCRIPT13}")

    # Step 13 imports exact_m0_model_v3.py from its script directory.
    script_dir = str(SCRIPT13.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    spec = importlib.util.spec_from_file_location("adni_step13_frozen_benchmark", SCRIPT13)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not construct import spec for Step 13.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_lock(lock):
    status = str(lock.get("status", ""))
    if status != "FROZEN BEFORE BENCHMARK TRAINING":
        raise RuntimeError(f"Benchmark lock status mismatch: {status!r}")

    # Require the frozen protocol statements visible in the lock.
    evaluation = lock.get("evaluation", {})
    ensemble_text = str(evaluation.get("ensemble", "")).lower()
    comparison_text = str(evaluation.get("comparison_rule", "")).lower()
    censoring_text = str(evaluation.get("censoring_distribution", "")).lower()
    uncertainty_text = str(evaluation.get("uncertainty_intervals", "")).lower()

    required_ensemble_tokens = ["arithmetic mean", "five", "seed", "cumulative risk"]
    if not all(tok in ensemble_text for tok in required_ensemble_tokens):
        raise RuntimeError(
            "Frozen ensemble rule is missing or unexpected.\n"
            f"Found: {evaluation.get('ensemble')!r}"
        )

    if "identical frozen test" not in comparison_text:
        raise RuntimeError(
            "Frozen TEST comparison rule is missing or unexpected.\n"
            f"Found: {evaluation.get('comparison_rule')!r}"
        )

    if "train" not in censoring_text or "censor" not in censoring_text:
        raise RuntimeError(
            "TRAIN-only censoring rule is missing or unexpected.\n"
            f"Found: {evaluation.get('censoring_distribution')!r}"
        )

    if "2000" not in uncertainty_text or "participant" not in uncertainty_text:
        raise RuntimeError(
            "Frozen 2000 participant-cluster bootstrap rule is missing or unexpected.\n"
            f"Found: {evaluation.get('uncertainty_intervals')!r}"
        )


def validate_training_summary(summary):
    required_cols = {
        "status", "model", "seed", "best_epoch", "best_validation_nll",
        "best_checkpoint", "test_inference_performed"
    }
    missing = required_cols.difference(summary.columns)
    if missing:
        raise RuntimeError(f"Training summary missing columns: {sorted(missing)}")

    complete = summary.loc[summary["status"].astype(str) == "COMPLETE"].copy()
    if len(complete) != 15:
        raise RuntimeError(f"Expected 15 COMPLETE benchmark runs, found {len(complete)}.")

    for model in MODELS:
        sub = complete.loc[complete["model"].astype(str) == model]
        got = sorted(sub["seed"].astype(int).tolist())
        if got != sorted(SEEDS):
            raise RuntimeError(f"{model}: frozen seeds mismatch: {got}")

    # Training summary must still state that benchmark TEST inference had not been performed.
    vals = complete["test_inference_performed"].astype(str).str.lower().unique().tolist()
    if any(v not in {"false", "0", "0.0"} for v in vals):
        raise RuntimeError(
            "Training summary indicates TEST inference was already performed for benchmark training runs."
        )

    return complete


def save_validation_audit(complete):
    rows = []
    for model in MODELS:
        x = complete.loc[
            complete["model"].astype(str) == model,
            "best_validation_nll"
        ].astype(float).to_numpy()
        rows.append({
            "model": model,
            "n_seeds": int(len(x)),
            "mean_best_validation_nll": float(np.mean(x)),
            "sd_best_validation_nll": float(np.std(x, ddof=1)),
            "median_best_validation_nll": float(np.median(x)),
            "min_best_validation_nll": float(np.min(x)),
            "max_best_validation_nll": float(np.max(x)),
        })
    pd.DataFrame(rows).to_csv(VALIDATION_AUDIT_CSV, index=False)


def checkpoint_path(model, seed):
    return BENCHMARK_MODEL_DIR / f"ADNI_{model}_SEED_{seed}_BEST_V1.pt"


def prediction_path(model, seed):
    return PRED_DIR / f"ADNI_{model}_SEED_{seed}_TEST_PREDICTIONS_V1.csv"


@torch.inference_mode()
def infer_one_seed(step13, model_code, seed, test_df):
    out_file = prediction_path(model_code, seed)

    # Resume-safe: reuse only if structurally complete.
    if out_file.exists():
        cached = pd.read_csv(out_file)
        expected_cols = {
            "model", "seed", "row_index", "subject_id", "landmark_id",
            "event", "time_days",
            "risk_12m", "risk_24m", "risk_36m", "risk_48m", "risk_60m"
        }
        if expected_cols.issubset(cached.columns) and len(cached) == EXPECTED["test_landmarks"]:
            log(f"{model_code} seed {seed}: complete cached TEST predictions -> REUSE")
            return cached
        raise RuntimeError(
            f"Incomplete/invalid cached prediction file exists:\n{out_file}\n"
            "Move/delete this file only if you have confirmed it is from an interrupted run."
        )

    ckpt_path = checkpoint_path(model_code, seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing BEST checkpoint:\n{ckpt_path}")

    dataset = step13.LongitudinalBenchmarkDataset(
        test_df,
        model_code=model_code,
    )
    loader = DataLoader(
        dataset,
        batch_size=step13.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=step13.collate_longitudinal,
    )

    device = step13.DEVICE
    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    if str(checkpoint.get("model_code")) != model_code:
        raise RuntimeError(f"Checkpoint model mismatch: {ckpt_path}")
    if int(checkpoint.get("seed")) != int(seed):
        raise RuntimeError(f"Checkpoint seed mismatch: {ckpt_path}")

    model = step13.build_model(model_code).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    rows = []
    row_index = 0

    for batch in loader:
        images = batch["images"].to(device)
        years_from_first = batch["years_from_first"].to(device)
        years_before_landmark = batch["years_before_landmark"].to(device)
        visit_mask = batch["visit_mask"].to(device)

        logits = model(
            images=images,
            years_from_first=years_from_first,
            years_before_landmark=years_before_landmark,
            visit_mask=visit_mask,
        )

        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        risks = 1.0 - survival
        risks_np = risks.detach().cpu().numpy()

        events = batch["event"].cpu().numpy()
        times = batch["time_days"].cpu().numpy()

        for i in range(len(batch["subject_id"])):
            rows.append({
                "model": model_code,
                "seed": int(seed),
                "row_index": int(row_index),
                "subject_id": str(batch["subject_id"][i]),
                "landmark_id": str(batch["landmark_id"][i]),
                "event": int(events[i]),
                "time_days": float(times[i]),
                "risk_12m": float(risks_np[i, 0]),
                "risk_24m": float(risks_np[i, 1]),
                "risk_36m": float(risks_np[i, 2]),
                "risk_48m": float(risks_np[i, 3]),
                "risk_60m": float(risks_np[i, 4]),
            })
            row_index += 1

        del images, years_from_first, years_before_landmark, visit_mask, logits, hazards, survival, risks

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred = pd.DataFrame(rows)

    if len(pred) != EXPECTED["test_landmarks"]:
        raise RuntimeError(
            f"{model_code} seed {seed}: expected {EXPECTED['test_landmarks']} TEST rows, got {len(pred)}."
        )
    if pred["subject_id"].nunique() != EXPECTED["test_subjects"]:
        raise RuntimeError(
            f"{model_code} seed {seed}: expected {EXPECTED['test_subjects']} TEST subjects, "
            f"got {pred['subject_id'].nunique()}."
        )

    # Atomic-ish save.
    tmp = out_file.with_suffix(".csv.tmp")
    pred.to_csv(tmp, index=False)
    os.replace(tmp, out_file)

    log(
        f"{model_code} seed {seed}: TEST inference complete | "
        f"checkpoint epoch={checkpoint.get('epoch')} | val NLL={checkpoint.get('validation_loss')}"
    )

    del model, checkpoint, loader, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred


def build_ensemble(seed_predictions):
    ensemble_frames = []

    key_cols = ["row_index", "subject_id", "landmark_id", "event", "time_days"]
    risk_cols = [f"risk_{m}m" for m in HORIZON_MONTHS]

    for model in MODELS:
        frames = [seed_predictions[(model, seed)].copy() for seed in SEEDS]

        base = frames[0][key_cols].copy()
        for other in frames[1:]:
            if not base.astype(str).equals(other[key_cols].astype(str)):
                raise RuntimeError(f"{model}: seed prediction row alignment mismatch.")

        risks = np.stack(
            [f[risk_cols].to_numpy(dtype=float) for f in frames],
            axis=0,
        )

        mean_risk = risks.mean(axis=0)

        ens = base.copy()
        ens.insert(0, "model", model)

        for j, month in enumerate(HORIZON_MONTHS):
            ens[f"risk_{month}m"] = mean_risk[:, j]
            ens[f"seed_sd_risk_{month}m"] = risks[:, :, j].std(axis=0, ddof=1)

        ensemble_frames.append(ens)

    out = pd.concat(ensemble_frames, ignore_index=True)

    # Common TEST outcomes/identity must be identical across architectures.
    ref = out.loc[out["model"] == MODELS[0], key_cols].reset_index(drop=True)
    for model in MODELS[1:]:
        cur = out.loc[out["model"] == model, key_cols].reset_index(drop=True)
        if not ref.astype(str).equals(cur.astype(str)):
            raise RuntimeError(f"{model}: TEST cohort mismatch versus {MODELS[0]}.")

    tmp = ENSEMBLE_CSV.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, ENSEMBLE_CSV)
    return out


# ==============================================================================================================
# TRAIN-only censoring Kaplan-Meier
# ==============================================================================================================

class CensoringKM:
    def __init__(self, times, event):
        # event=1 means clinical event occurred; censoring-event indicator is 1-event.
        times = np.asarray(times, dtype=float)
        event = np.asarray(event, dtype=int)
        censor_event = 1 - event

        order = np.argsort(times, kind="mergesort")
        t = times[order]
        c = censor_event[order]

        unique_times = np.unique(t)
        surv_after = []
        surv_before = []
        s = 1.0

        for u in unique_times:
            surv_before.append(s)
            at_risk = np.sum(t >= u)
            d_censor = np.sum((t == u) & (c == 1))
            if at_risk > 0:
                s *= (1.0 - d_censor / at_risk)
            surv_after.append(s)

        self.unique_times = unique_times
        self.surv_before = np.asarray(surv_before, dtype=float)
        self.surv_after = np.asarray(surv_after, dtype=float)

    def g_left(self, x):
        """G(x-) left-limit censoring survival."""
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.unique_times, x, side="left") - 1
        out = np.ones_like(x, dtype=float)
        valid = idx >= 0
        out[valid] = self.surv_after[idx[valid]]
        return np.clip(out, EPS, 1.0)

    def g_at(self, x):
        """G(x) right-continuous censoring survival after any censoring at x."""
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.unique_times, x, side="right") - 1
        out = np.ones_like(x, dtype=float)
        valid = idx >= 0
        out[valid] = self.surv_after[idx[valid]]
        return np.clip(out, EPS, 1.0)


# ==============================================================================================================
# METRICS
# ==============================================================================================================

def weighted_dynamic_auc(time_days, event, risk, horizon, censor_km, row_weight=None):
    time_days = np.asarray(time_days, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    if row_weight is None:
        row_weight = np.ones(len(time_days), dtype=float)
    else:
        row_weight = np.asarray(row_weight, dtype=float)

    case_mask = (event == 1) & (time_days <= horizon)
    control_mask = time_days > horizon

    ci = np.where(case_mask)[0]
    cj = np.where(control_mask)[0]
    if len(ci) == 0 or len(cj) == 0:
        return np.nan

    w_case = row_weight[ci] / censor_km.g_left(time_days[ci])
    w_ctrl = row_weight[cj] / censor_km.g_at(np.full(len(cj), horizon))

    pair_w = w_case[:, None] * w_ctrl[None, :]
    cmp = (risk[ci, None] > risk[cj][None, :]).astype(float)
    cmp += 0.5 * (risk[ci, None] == risk[cj][None, :])

    den = pair_w.sum()
    if den <= 0:
        return np.nan
    return float((pair_w * cmp).sum() / den)


def ipcw_brier(time_days, event, risk, horizon, censor_km, row_weight=None):
    time_days = np.asarray(time_days, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)

    if row_weight is None:
        row_weight = np.ones(len(time_days), dtype=float)
    else:
        row_weight = np.asarray(row_weight, dtype=float)

    surv_pred = 1.0 - risk
    total_cluster_weight = row_weight.sum()
    if total_cluster_weight <= 0:
        return np.nan

    event_by_t = (event == 1) & (time_days <= horizon)
    beyond_t = time_days > horizon

    contribution = np.zeros(len(time_days), dtype=float)

    if np.any(event_by_t):
        idx = np.where(event_by_t)[0]
        contribution[idx] = (
            row_weight[idx]
            * (surv_pred[idx] ** 2)
            / censor_km.g_left(time_days[idx])
        )

    if np.any(beyond_t):
        idx = np.where(beyond_t)[0]
        g_t = float(censor_km.g_at(np.asarray([horizon]))[0])
        contribution[idx] = (
            row_weight[idx]
            * ((1.0 - surv_pred[idx]) ** 2)
            / g_t
        )

    # Subjects censored at/before t contribute zero, as in the IPCW estimator.
    return float(contribution.sum() / total_cluster_weight)


def integrated_brier_0_60(time_days, event, risks_5, censor_km, row_weight=None):
    # Discrete annual survival model: risk(0)=0, hence Brier(0)=0.
    briers = [
        ipcw_brier(
            time_days, event, risks_5[:, j], HORIZON_DAYS[j], censor_km, row_weight
        )
        for j in range(5)
    ]
    times = np.concatenate([[0.0], HORIZON_DAYS])
    values = np.concatenate([[0.0], np.asarray(briers, dtype=float)])
    if not np.all(np.isfinite(values)):
        return np.nan
    return float(np.trapz(values, times) / TAU_DAYS)


def harrell_c_index(time_days, event, risk60, tau=TAU_DAYS, row_weight=None):
    time_days = np.asarray(time_days, dtype=float)
    event = np.asarray(event, dtype=int)
    risk60 = np.asarray(risk60, dtype=float)
    if row_weight is None:
        row_weight = np.ones(len(time_days), dtype=float)
    else:
        row_weight = np.asarray(row_weight, dtype=float)

    event_idx = np.where((event == 1) & (time_days <= tau))[0]
    numerator = 0.0
    denominator = 0.0

    for i in event_idx:
        j = np.where(time_days > time_days[i])[0]
        if len(j) == 0:
            continue
        w = row_weight[i] * row_weight[j]
        cmp = (risk60[i] > risk60[j]).astype(float)
        cmp += 0.5 * (risk60[i] == risk60[j])
        numerator += float(np.sum(w * cmp))
        denominator += float(np.sum(w))

    return float(numerator / denominator) if denominator > 0 else np.nan


def uno_c_index(time_days, event, risk60, censor_km, tau=TAU_DAYS, row_weight=None):
    time_days = np.asarray(time_days, dtype=float)
    event = np.asarray(event, dtype=int)
    risk60 = np.asarray(risk60, dtype=float)
    if row_weight is None:
        row_weight = np.ones(len(time_days), dtype=float)
    else:
        row_weight = np.asarray(row_weight, dtype=float)

    event_idx = np.where((event == 1) & (time_days <= tau))[0]
    numerator = 0.0
    denominator = 0.0

    for i in event_idx:
        j = np.where(time_days > time_days[i])[0]
        if len(j) == 0:
            continue

        gi = float(censor_km.g_left(np.asarray([time_days[i]]))[0])
        base = row_weight[i] / (gi ** 2)
        w = base * row_weight[j]

        cmp = (risk60[i] > risk60[j]).astype(float)
        cmp += 0.5 * (risk60[i] == risk60[j])

        numerator += float(np.sum(w * cmp))
        denominator += float(np.sum(w))

    return float(numerator / denominator) if denominator > 0 else np.nan


def _weighted_logistic_fit(x, y, w, max_iter=100, tol=1e-10):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    X = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(2, dtype=float)

    for _ in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        v = w * p * (1.0 - p)
        H = X.T @ (v[:, None] * X)
        g = X.T @ (w * (y - p))
        try:
            step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return float(beta[0]), float(beta[1])


def calibration_stats(time_days, event, risk, horizon, censor_km, row_weight=None, n_bins=10):
    time_days = np.asarray(time_days, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)

    if row_weight is None:
        row_weight = np.ones(len(time_days), dtype=float)
    else:
        row_weight = np.asarray(row_weight, dtype=float)

    case = (event == 1) & (time_days <= horizon)
    control = time_days > horizon
    valid = case | control

    if valid.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan

    y = case[valid].astype(float)
    r = np.clip(risk[valid], 1e-6, 1 - 1e-6)
    rw = row_weight[valid]
    tv = time_days[valid]

    w = np.zeros(len(r), dtype=float)
    case_v = y == 1
    ctrl_v = y == 0

    if np.any(case_v):
        w[case_v] = rw[case_v] / censor_km.g_left(tv[case_v])
    if np.any(ctrl_v):
        g_t = float(censor_km.g_at(np.asarray([horizon]))[0])
        w[ctrl_v] = rw[ctrl_v] / g_t

    logit_risk = np.log(r / (1.0 - r))
    intercept, slope = _weighted_logistic_fit(logit_risk, y, w)

    # Weighted equal-frequency-ish bins based on predicted risk quantiles.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(r, quantiles))
    if len(edges) < 3:
        return intercept, slope, np.nan, np.nan

    bins = np.digitize(r, edges[1:-1], right=True)
    total_w = np.sum(w)
    ece = 0.0
    mce = 0.0

    for b in np.unique(bins):
        idx = bins == b
        wb = w[idx]
        sw = np.sum(wb)
        if sw <= 0:
            continue
        pred_mean = np.sum(wb * r[idx]) / sw
        obs_mean = np.sum(wb * y[idx]) / sw
        gap = abs(pred_mean - obs_mean)
        ece += (sw / total_w) * gap
        mce = max(mce, gap)

    return intercept, slope, float(ece), float(mce)


def calculate_all_metrics(df_model, censor_km, row_weight=None):
    time_days = df_model["time_days"].to_numpy(dtype=float)
    event = df_model["event"].to_numpy(dtype=int)
    risks = df_model[[f"risk_{m}m" for m in HORIZON_MONTHS]].to_numpy(dtype=float)

    out = {}

    for j, month in enumerate(HORIZON_MONTHS):
        horizon = HORIZON_DAYS[j]
        risk = risks[:, j]

        out[f"auc_{month}m"] = weighted_dynamic_auc(
            time_days, event, risk, horizon, censor_km, row_weight
        )
        out[f"brier_{month}m"] = ipcw_brier(
            time_days, event, risk, horizon, censor_km, row_weight
        )

        intercept, slope, ece, mce = calibration_stats(
            time_days, event, risk, horizon, censor_km, row_weight
        )
        out[f"calibration_intercept_{month}m"] = intercept
        out[f"calibration_slope_{month}m"] = slope
        out[f"ece_{month}m"] = ece
        out[f"mce_{month}m"] = mce

    out["harrell_c_60m"] = harrell_c_index(
        time_days, event, risks[:, 4], TAU_DAYS, row_weight
    )
    out["uno_c_60m"] = uno_c_index(
        time_days, event, risks[:, 4], censor_km, TAU_DAYS, row_weight
    )
    out["integrated_brier_0_60m"] = integrated_brier_0_60(
        time_days, event, risks, censor_km, row_weight
    )

    return out


def participant_row_weights(subject_ids, sampled_subject_counts):
    subject_ids = np.asarray(subject_ids, dtype=str)
    return np.asarray(
        [sampled_subject_counts.get(s, 0) for s in subject_ids],
        dtype=float,
    )


def bootstrap_metrics(ensemble_df, censor_km, n_bootstrap, seed):
    rng = np.random.default_rng(seed)

    # All architectures have identical outcome rows; obtain subject list from B0.
    ref = ensemble_df.loc[ensemble_df["model"] == MODELS[0]].reset_index(drop=True)
    subjects = np.asarray(sorted(ref["subject_id"].astype(str).unique()))
    n_subjects = len(subjects)

    model_frames = {
        model: ensemble_df.loc[ensemble_df["model"] == model].reset_index(drop=True)
        for model in MODELS
    }

    boot_rows = []

    log("")
    log(f"Participant-cluster bootstrap: {n_bootstrap} replicates | subjects={n_subjects}")

    for b in range(1, n_bootstrap + 1):
        sampled = rng.choice(subjects, size=n_subjects, replace=True)
        uniq, counts = np.unique(sampled, return_counts=True)
        count_map = {str(s): int(c) for s, c in zip(uniq, counts)}
        row_weight = participant_row_weights(
            ref["subject_id"].astype(str).to_numpy(),
            count_map,
        )

        for model in MODELS:
            metrics = calculate_all_metrics(
                model_frames[model],
                censor_km,
                row_weight=row_weight,
            )
            for metric, value in metrics.items():
                boot_rows.append({
                    "bootstrap_id": b,
                    "model": model,
                    "metric": metric,
                    "value": value,
                })

        if b == 1 or b % 100 == 0 or b == n_bootstrap:
            log(f"Bootstrap {b}/{n_bootstrap} complete")

    boot = pd.DataFrame(boot_rows)

    ci_rows = []
    for (model, metric), sub in boot.groupby(["model", "metric"], sort=True):
        x = sub["value"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            low = high = np.nan
        else:
            low, high = np.percentile(x, [2.5, 97.5])
        ci_rows.append({
            "model": model,
            "metric": metric,
            "bootstrap_valid_replicates": int(len(x)),
            "ci_2_5": float(low) if np.isfinite(low) else np.nan,
            "ci_97_5": float(high) if np.isfinite(high) else np.nan,
        })

    ci = pd.DataFrame(ci_rows)
    return boot, ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Must remain 2000 for the frozen final analysis. Smaller values are not accepted without --development-only.",
    )
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Allows a smaller bootstrap count ONLY for code testing. Does not write final COMPLETE status.",
    )
    args = parser.parse_args()

    if args.bootstrap != N_BOOTSTRAP and not args.development_only:
        raise RuntimeError(
            f"Frozen protocol requires exactly {N_BOOTSTRAP} bootstrap replicates. "
            "Use --development-only only for a dry code check before final evaluation."
        )

    if STATUS_JSON.exists():
        try:
            status = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
        except Exception:
            status = {}
        if status.get("status") == "COMPLETE" and not args.development_only:
            log("Frozen benchmark TEST evaluation already marked COMPLETE.")
            log(f"Status: {STATUS_JSON}")
            log("No new TEST evaluation performed.")
            return

    if not RUN_LOG.exists():
        RUN_LOG.write_text("", encoding="utf-8")

    log("=" * 120)
    log("ADNI — STEP 14: ONE FROZEN BENCHMARK TEST EVALUATION")
    log("=" * 120)
    log(f"Start UTC: {utc_now()}")
    log(f"Development-only: {args.development_only}")
    log(f"Bootstrap replicates: {args.bootstrap}")

    required = [LOCK_FILE, TRAINING_SUMMARY, DATASET_FILE, SCRIPT13]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    with open(LOCK_FILE, "r", encoding="utf-8") as f:
        lock = json.load(f)
    verify_lock(lock)

    summary = pd.read_csv(TRAINING_SUMMARY)
    complete = validate_training_summary(summary)
    save_validation_audit(complete)

    # Verify all frozen BEST checkpoints before touching TEST.
    checkpoint_hashes = {}
    for model in MODELS:
        for seed in SEEDS:
            p = checkpoint_path(model, seed)
            if not p.exists():
                raise FileNotFoundError(p)
            checkpoint_hashes[f"{model}_{seed}"] = sha256_file(p)

    step13 = load_step13_module()

    df = pd.read_csv(DATASET_FILE)
    if len(df) != EXPECTED["landmarks"]:
        raise RuntimeError(f"Expected {EXPECTED['landmarks']} landmarks, found {len(df)}.")

    train_df = df.loc[df["split"].astype(str).str.upper() == "TRAIN"].copy().reset_index(drop=True)
    val_df = df.loc[df["split"].astype(str).str.upper() == "VALIDATION"].copy().reset_index(drop=True)
    test_df = df.loc[df["split"].astype(str).str.upper() == "TEST"].copy().reset_index(drop=True)

    checks = [
        ("TRAIN landmarks", len(train_df), EXPECTED["train_landmarks"]),
        ("VALIDATION landmarks", len(val_df), EXPECTED["validation_landmarks"]),
        ("TEST landmarks", len(test_df), EXPECTED["test_landmarks"]),
        ("TRAIN subjects", train_df["subject_id"].nunique(), EXPECTED["train_subjects"]),
        ("VALIDATION subjects", val_df["subject_id"].nunique(), EXPECTED["validation_subjects"]),
        ("TEST subjects", test_df["subject_id"].nunique(), EXPECTED["test_subjects"]),
    ]
    for name, got, expected in checks:
        log(f"{name}: {got}")
        if int(got) != int(expected):
            raise RuntimeError(f"{name} mismatch: got {got}, expected {expected}")

    # Ensure subject-disjoint split.
    tr_sub = set(train_df["subject_id"].astype(str))
    va_sub = set(val_df["subject_id"].astype(str))
    te_sub = set(test_df["subject_id"].astype(str))
    if tr_sub & va_sub or tr_sub & te_sub or va_sub & te_sub:
        raise RuntimeError("Subject leakage detected between frozen splits.")

    # TRAIN-only censoring KM.
    censor_km = CensoringKM(
        train_df["time_to_event_or_censor_days"].to_numpy(dtype=float),
        train_df["event"].to_numpy(dtype=int),
    )

    # TEST inference, resume-safe per seed.
    seed_predictions = {}
    for model in MODELS:
        log("")
        log("=" * 120)
        log(f"TEST INFERENCE — {model}")
        log("=" * 120)
        for seed in SEEDS:
            seed_predictions[(model, seed)] = infer_one_seed(
                step13=step13,
                model_code=model,
                seed=seed,
                test_df=test_df,
            )

    # Five-seed arithmetic-mean cumulative-risk ensemble for each architecture.
    ensemble = build_ensemble(seed_predictions)
    log("")
    log("Five-seed cumulative-risk ensembles created for B0/B1/B2.")

    # Point estimates.
    metric_rows = []
    point_metrics = {}

    for model in MODELS:
        frame = ensemble.loc[ensemble["model"] == model].reset_index(drop=True)
        metrics = calculate_all_metrics(frame, censor_km, row_weight=None)
        point_metrics[model] = metrics
        for metric, value in metrics.items():
            metric_rows.append({
                "model": model,
                "metric": metric,
                "estimate": value,
            })

    point_df = pd.DataFrame(metric_rows)

    # Bootstrap CI.
    boot, ci = bootstrap_metrics(
        ensemble_df=ensemble,
        censor_km=censor_km,
        n_bootstrap=args.bootstrap,
        seed=BOOTSTRAP_SEED,
    )

    final_metrics = point_df.merge(
        ci,
        on=["model", "metric"],
        how="left",
    )

    # Final files. In development mode, use separate names and DO NOT mark protocol complete.
    if args.development_only:
        dev_metrics = OUT_DIR / "DEVELOPMENT_ONLY_METRICS.csv"
        dev_boot = OUT_DIR / "DEVELOPMENT_ONLY_BOOTSTRAP_DRAWS.csv"
        final_metrics.to_csv(dev_metrics, index=False)
        boot.to_csv(dev_boot, index=False)
        log("")
        log("DEVELOPMENT-ONLY run finished. Final frozen status NOT written.")
        log(f"Metrics: {dev_metrics}")
        return

    final_metrics.to_csv(METRICS_CSV, index=False)
    boot.to_csv(BOOTSTRAP_CSV, index=False)

    audit = {
        "status": "COMPLETE",
        "completed_utc": utc_now(),
        "protocol": {
            "models": MODELS,
            "seeds": SEEDS,
            "ensemble": "arithmetic mean of five frozen seed cumulative risks for each benchmark",
            "test_comparison": "B0/B1/B2 evaluated on identical frozen TEST landmarks and subjects",
            "censoring_distribution": "TRAIN-only Kaplan-Meier of censoring",
            "bootstrap": {
                "type": "participant-cluster percentile bootstrap",
                "replicates": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "ci": "2.5th and 97.5th percentiles",
            },
            "test_tuning": "NONE",
            "test_threshold_optimization": "NONE",
            "test_model_selection": "NONE",
        },
        "cohort": {
            "train_landmarks": int(len(train_df)),
            "validation_landmarks": int(len(val_df)),
            "test_landmarks": int(len(test_df)),
            "train_subjects": int(train_df["subject_id"].nunique()),
            "validation_subjects": int(val_df["subject_id"].nunique()),
            "test_subjects": int(test_df["subject_id"].nunique()),
        },
        "files": {
            "benchmark_lock": str(LOCK_FILE),
            "benchmark_lock_sha256": sha256_file(LOCK_FILE),
            "dataset": str(DATASET_FILE),
            "dataset_sha256": sha256_file(DATASET_FILE),
            "training_summary": str(TRAINING_SUMMARY),
            "training_summary_sha256": sha256_file(TRAINING_SUMMARY),
            "step13_script": str(SCRIPT13),
            "step13_script_sha256": sha256_file(SCRIPT13),
            "ensemble_predictions": str(ENSEMBLE_CSV),
            "metrics": str(METRICS_CSV),
            "bootstrap_draws": str(BOOTSTRAP_CSV),
            "validation_audit": str(VALIDATION_AUDIT_CSV),
        },
        "checkpoint_sha256": checkpoint_hashes,
        "metric_definitions": {
            "risk": "cumulative risk = 1 - cumulative product of (1 - sigmoid(interval logit))",
            "ipcw_auc": "cumulative/dynamic AUC using TRAIN-only censoring KM; cases event<=t, controls T>t",
            "ipcw_brier": "IPCW Brier using TRAIN-only censoring KM",
            "integrated_brier_0_60m": "trapezoidal integration at 0,12,24,36,48,60 months divided by 60 months; Brier(0)=0",
            "harrell_c_60m": "Harrell concordance using 60-month cumulative risk and events through 60 months",
            "uno_c_60m": "Uno IPCW concordance using TRAIN-only censoring KM and 60-month cumulative risk",
            "calibration": "IPCW weighted logistic calibration intercept/slope plus weighted ECE/MCE at each annual horizon",
        },
    }

    write_json(AUDIT_JSON, audit)
    write_json(STATUS_JSON, {
        "status": "COMPLETE",
        "completed_utc": audit["completed_utc"],
        "audit_file": str(AUDIT_JSON),
    })

    log("")
    log("=" * 120)
    log("FROZEN BENCHMARK TEST EVALUATION: COMPLETE")
    log("=" * 120)
    log(f"Ensemble predictions: {ENSEMBLE_CSV}")
    log(f"Metrics + 95% CI:      {METRICS_CSV}")
    log(f"Bootstrap draws:       {BOOTSTRAP_CSV}")
    log(f"Audit record:          {AUDIT_JSON}")
    log(f"Status lock:           {STATUS_JSON}")
    log("=" * 120)

    # Compact primary-results display.
    primary_order = [
        "auc_12m", "auc_24m", "auc_36m",
        "brier_12m", "brier_24m", "brier_36m",
        "uno_c_60m", "harrell_c_60m", "integrated_brier_0_60m",
    ]
    show = final_metrics.loc[final_metrics["metric"].isin(primary_order)].copy()
    show["metric_order"] = show["metric"].map({m: i for i, m in enumerate(primary_order)})
    show = show.sort_values(["metric_order", "model"])
    log("")
    log("PRIMARY FROZEN TEST RESULTS")
    log(show[["model", "metric", "estimate", "ci_2_5", "ci_97_5"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            log("")
            log("=" * 120)
            log("FROZEN BENCHMARK TEST EVALUATION: FAILED / INTERRUPTED")
            log("=" * 120)
            log(traceback.format_exc())
            log("Per-seed prediction files already completed can be safely reused on rerun.")
        finally:
            raise
