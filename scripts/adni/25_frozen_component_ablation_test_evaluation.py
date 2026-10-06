from pathlib import Path
import json
import math
import importlib.util
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ==============================================================================================================
# ADNI — STEP 25: FROZEN COMPONENT-ABLATION TEST EVALUATION
# ==============================================================================================================
#
# This is the one-time frozen TEST evaluation for the locked component-ablation suite.
#
# Models:
#   A0_FULL_M0            existing frozen five-seed M0 reference
#   A1_NO_TIME_ENCODING   trained in Step 24
#   A2_NO_TRANSFORMER     trained in Step 24
#   A3_LATEST_VISIT_ONLY  trained in Step 24
#
# Evaluation:
#   - five-seed arithmetic-mean ensemble
#   - TRAIN-only Kaplan-Meier censoring distribution
#   - IPCW cumulative/dynamic AUC at 12/24/36/48/60 months
#   - IPCW Brier at 12/24/36/48/60 months
#   - Uno C-index at 60 months
#   - Harrell C-index at 60 months
#   - integrated Brier score over 0–60 months
#   - 2,000 participant-cluster bootstrap replicates, seed 20260912
#   - same-resample paired comparisons A1/A2/A3 versus A0
#
# Resume-safe:
#   - per-seed TEST predictions are written separately
#   - an existing complete prediction file is reused
#
# No training, tuning, threshold optimisation, or architecture modification is performed here.
# ==============================================================================================================

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "06_splits"
MODEL_DIR = ROOT / "07_models"
RESULT_DIR = ROOT / "08_results"
SCRIPT_DIR = ROOT / "scripts"

DATA_FILE = SPLIT_DIR / "ADNI_LONGITUDINAL_LANDMARK_DATASET_V1.csv"
LOCK_FILE = RESULT_DIR / "ablation_v1" / "ADNI_COMPONENT_ABLATION_LOCK_V1.json"
TRAIN_REPORT = RESULT_DIR / "ablation_v1" / "ADNI_COMPONENT_ABLATION_TRAINING_REPORT_V1.txt"
STEP24_FILE = SCRIPT_DIR / "24_train_component_ablation_suite_FIXED.py"
M0_SOURCE = SCRIPT_DIR / "exact_m0_model_v3.py"

OUT_DIR = RESULT_DIR / "ablation_test_v1"
PRED_DIR = OUT_DIR / "per_seed_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [2026, 3407, 5813, 7291, 9109]
MODELS = ["A0_FULL_M0", "A1_NO_TIME_ENCODING", "A2_NO_TRANSFORMER", "A3_LATEST_VISIT_ONLY"]
ABLATIONS = ["A1_NO_TIME_ENCODING", "A2_NO_TRANSFORMER", "A3_LATEST_VISIT_ONLY"]

HORIZONS = [12, 24, 36, 48, 60]
DAYS_PER_YEAR = 365.25
HORIZON_DAYS = {m: (m / 12.0) * DAYS_PER_YEAR for m in HORIZONS}
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260912
BATCH_SIZE = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REFERENCE_CKPTS = {
    2026: MODEL_DIR / "ADNI_LONGITUDINAL_SURVIVAL_BEST_V1.pt",
    3407: MODEL_DIR / "ADNI_M0_SEED_3407_BEST_V1.pt",
    5813: MODEL_DIR / "ADNI_M0_SEED_5813_BEST_V1.pt",
    7291: MODEL_DIR / "ADNI_M0_SEED_7291_BEST_V1.pt",
    9109: MODEL_DIR / "ADNI_M0_SEED_9109_BEST_V1.pt",
}

PRIMARY_METRICS = [
    "auc_12m", "auc_24m", "auc_36m",
    "brier_12m", "brier_24m", "brier_36m",
    "uno_c_60m", "harrell_c_60m", "integrated_brier_0_60m",
]
HIGHER_BETTER = {
    "auc_12m","auc_24m","auc_36m","auc_48m","auc_60m",
    "uno_c_60m","harrell_c_60m",
}
LOWER_BETTER = {
    "brier_12m","brier_24m","brier_36m","brier_48m","brier_60m",
    "integrated_brier_0_60m",
}

def now_utc():
    return datetime.now(timezone.utc).isoformat()

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def cumulative_risk_from_logits(logits):
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return 1.0 - survival

# --------------------------------------------------------------------------------------------------------------
# TRAIN-only censoring Kaplan-Meier
# --------------------------------------------------------------------------------------------------------------

class CensoringKM:
    def __init__(self, times, events):
        # censoring event = 1 - progression event
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=int)
        censor_event = 1 - events

        order = np.argsort(times, kind="mergesort")
        t = times[order]
        c = censor_event[order]

        unique = np.unique(t)
        surv = 1.0
        self.times = []
        self.surv_after = []
        self.surv_before = []

        for ut in unique:
            at_risk = np.sum(t >= ut)
            d = np.sum((t == ut) & (c == 1))
            self.times.append(float(ut))
            self.surv_before.append(float(surv))
            if at_risk > 0:
                surv *= (1.0 - d / at_risk)
            self.surv_after.append(float(surv))

        self.times = np.asarray(self.times)
        self.surv_before = np.asarray(self.surv_before)
        self.surv_after = np.asarray(self.surv_after)

    def G(self, x):
        """G(x): censoring survival after time x."""
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.times, x, side="right") - 1
        out = np.ones_like(x, dtype=float)
        valid = idx >= 0
        out[valid] = self.surv_after[idx[valid]]
        return np.clip(out, 1e-8, None)

    def G_left(self, x):
        """G(x-): censoring survival immediately before x."""
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.times, x, side="left")
        out = np.ones_like(x, dtype=float)
        valid = idx > 0
        out[valid] = self.surv_after[idx[valid] - 1]
        return np.clip(out, 1e-8, None)

# --------------------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------------------

def ipcw_auc(times, events, risk, tau, km):
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    risk = np.asarray(risk, float)

    case = (events == 1) & (times <= tau)
    control = times > tau
    if case.sum() == 0 or control.sum() == 0:
        return np.nan

    wc = 1.0 / km.G_left(times[case])
    w0 = np.full(control.sum(), 1.0 / float(km.G(np.array([tau]))[0]))

    rc = risk[case]
    r0 = risk[control]

    # Weighted pairwise concordance with 0.5 credit for ties.
    diff = rc[:, None] - r0[None, :]
    score = (diff > 0).astype(float) + 0.5 * (diff == 0)
    weights = wc[:, None] * w0[None, :]
    den = weights.sum()
    return float((score * weights).sum() / den) if den > 0 else np.nan

def ipcw_brier(times, events, risk, tau, km):
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    risk = np.asarray(risk, float)

    w = np.zeros(len(times), dtype=float)
    y = np.zeros(len(times), dtype=float)

    ev = (events == 1) & (times <= tau)
    at_risk = times > tau

    y[ev] = 1.0
    w[ev] = 1.0 / km.G_left(times[ev])
    w[at_risk] = 1.0 / float(km.G(np.array([tau]))[0])

    return float(np.mean(w * (y - risk) ** 2))

def harrell_c(times, events, risk, tau):
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    risk = np.asarray(risk, float)

    num = 0.0
    den = 0.0
    n = len(times)
    for i in range(n):
        if events[i] != 1 or times[i] > tau:
            continue
        for j in range(n):
            if times[j] > times[i]:
                den += 1.0
                if risk[i] > risk[j]:
                    num += 1.0
                elif risk[i] == risk[j]:
                    num += 0.5
    return float(num / den) if den > 0 else np.nan

def uno_c(times, events, risk, tau, km):
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    risk = np.asarray(risk, float)

    num = 0.0
    den = 0.0
    n = len(times)
    for i in range(n):
        if events[i] != 1 or times[i] > tau:
            continue
        gi = float(km.G_left(np.array([times[i]]))[0])
        wi = 1.0 / (gi * gi)
        for j in range(n):
            if times[j] > times[i]:
                den += wi
                if risk[i] > risk[j]:
                    num += wi
                elif risk[i] == risk[j]:
                    num += 0.5 * wi
    return float(num / den) if den > 0 else np.nan

def all_metrics(df, model, km):
    times = df["time_to_event_or_censor_days"].to_numpy(float)
    events = df["event"].to_numpy(int)

    result = {}
    briers = [0.0]

    for m in HORIZONS:
        r = df[f"{model}_risk_{m}m"].to_numpy(float)
        tau = HORIZON_DAYS[m]
        result[f"auc_{m}m"] = ipcw_auc(times, events, r, tau, km)
        b = ipcw_brier(times, events, r, tau, km)
        result[f"brier_{m}m"] = b
        briers.append(b)

    r60 = df[f"{model}_risk_60m"].to_numpy(float)
    result["uno_c_60m"] = uno_c(times, events, r60, HORIZON_DAYS[60], km)
    result["harrell_c_60m"] = harrell_c(times, events, r60, HORIZON_DAYS[60])

    grid = np.asarray([0, 12, 24, 36, 48, 60], dtype=float)
    result["integrated_brier_0_60m"] = float(np.trapz(np.asarray(briers), grid) / 60.0)
    return result

# --------------------------------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------------------------------

def checkpoint_path(model_name, seed):
    if model_name == "A0_FULL_M0":
        return REFERENCE_CKPTS[seed]
    return MODEL_DIR / "ablation_v1" / f"{model_name}_SEED_{seed}_BEST_V1.pt"

def make_model(model_name, step24, m0mod):
    if model_name == "A0_FULL_M0":
        return m0mod.ExactM0LongitudinalSurvivalModel_V3()
    return step24.make_model(model_name, m0mod)

def prediction_file(model_name, seed):
    return PRED_DIR / f"{model_name}_SEED_{seed}_TEST_PREDICTIONS_V1.csv"

@torch.inference_mode()
def infer_one(model_name, seed, test_df, step24, m0mod):
    out = prediction_file(model_name, seed)

    if out.exists():
        old = pd.read_csv(out)
        required = {"subject_id","landmark_id","event","time_to_event_or_censor_days"} | {
            f"risk_{m}m" for m in HORIZONS
        }
        if len(old) == 583 and required.issubset(old.columns):
            print(f"REUSE TEST predictions: {out.name}")
            return old

    ckpt_file = checkpoint_path(model_name, seed)
    if not ckpt_file.exists():
        raise FileNotFoundError(ckpt_file)

    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if model_name != "A0_FULL_M0" and not bool(ckpt.get("completed", False)):
        raise RuntimeError(f"Incomplete ablation checkpoint: {ckpt_file}")

    model = make_model(model_name, step24, m0mod)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE).eval()

    latest_only = (model_name == "A3_LATEST_VISIT_ONLY")
    ds = step24.LandmarkDataset(test_df, latest_only=latest_only)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(), collate_fn=step24.collate_longitudinal
    )

    print("\n" + "=" * 110)
    print(f"FROZEN TEST INFERENCE — {model_name} — SEED {seed}")
    print("=" * 110)
    print(f"Checkpoint: {ckpt_file.name}")
    print(f"Best epoch: {ckpt.get('epoch')}")
    print(f"Validation NLL: {ckpt.get('validation_loss')}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"TEST landmarks: {len(ds)}")

    risks = []
    n_batches = len(loader)
    for bi, batch in enumerate(loader, start=1):
        logits = model(
            images=batch["images"].to(DEVICE, non_blocking=True),
            visit_mask=batch["visit_mask"].to(DEVICE, non_blocking=True),
            years_from_first=batch["years_from_first"].to(DEVICE, non_blocking=True),
            years_before_landmark=batch["years_before_landmark"].to(DEVICE, non_blocking=True),
        )
        cr = cumulative_risk_from_logits(logits).cpu().numpy()
        risks.append(cr)
        if bi % 50 == 0 or bi == n_batches:
            print(f"Processed {bi}/{n_batches} batches")

    risks = np.concatenate(risks, axis=0)
    if risks.shape != (583, 5):
        raise RuntimeError(f"Prediction shape mismatch: {risks.shape}")
    if not np.isfinite(risks).all():
        raise RuntimeError("Non-finite TEST risks.")
    if np.any(np.diff(risks, axis=1) < -1e-6):
        raise RuntimeError("Cumulative risk monotonicity failed.")

    pred = test_df[["subject_id","landmark_id","event","time_to_event_or_censor_days"]].reset_index(drop=True).copy()
    for j, m in enumerate(HORIZONS):
        pred[f"risk_{m}m"] = risks[:, j]

    pred.to_csv(out, index=False)
    print(f"Saved: {out}")
    return pred

def cluster_bootstrap_indices(df, rng):
    subjects = df["subject_id"].astype(str).unique()
    sampled = rng.choice(subjects, size=len(subjects), replace=True)

    pieces = []
    for draw_id, sid in enumerate(sampled):
        x = df.loc[df["subject_id"].astype(str) == sid].copy()
        # assign unique bootstrap cluster id to duplicated participant draws
        x["_bootstrap_subject"] = f"{sid}__draw{draw_id}"
        pieces.append(x)
    return pd.concat(pieces, ignore_index=True)

def main():
    print("=" * 110)
    print("STEP 25 — FROZEN COMPONENT-ABLATION TEST EVALUATION")
    print("=" * 110)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    required = [DATA_FILE, LOCK_FILE, TRAIN_REPORT, STEP24_FILE, M0_SOURCE]
    required += list(REFERENCE_CKPTS.values())
    for a in ABLATIONS:
        for s in SEEDS:
            required.append(checkpoint_path(a, s))

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required files:\n" + "\n".join(missing))

    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    if lock.get("experiment") != "ADNI_COMPONENT_ABLATION_V1":
        raise RuntimeError("Ablation lock mismatch.")

    # Load frozen data and split.
    df = pd.read_csv(DATA_FILE)
    train_df = df.loc[df["split"].astype(str).str.upper().eq("TRAIN")].copy().reset_index(drop=True)
    test_df = df.loc[df["split"].astype(str).str.upper().eq("TEST")].copy().reset_index(drop=True)

    if len(train_df) != 2530 or train_df["subject_id"].nunique() != 680:
        raise RuntimeError("TRAIN cohort mismatch.")
    if len(test_df) != 583 or test_df["subject_id"].nunique() != 146:
        raise RuntimeError("TEST cohort mismatch.")

    print(f"TRAIN censoring support: {len(train_df)} landmarks / {train_df['subject_id'].nunique()} participants")
    print(f"FROZEN TEST: {len(test_df)} landmarks / {test_df['subject_id'].nunique()} participants")
    print("Test model selection: NONE")
    print("Test tuning: NONE")
    print("Threshold optimisation: NONE")

    # TRAIN-only censoring KM.
    km = CensoringKM(
        train_df["time_to_event_or_censor_days"].to_numpy(float),
        train_df["event"].to_numpy(int),
    )
    print("\nTRAIN-ONLY CENSORING KM")
    for m in HORIZONS:
        print(f"G({m}m) = {float(km.G(np.array([HORIZON_DAYS[m]]))[0]):.6f}")

    step24 = load_module(STEP24_FILE, "step24_ablation")
    m0mod = load_module(M0_SOURCE, "exact_m0_model_v3_step25")

    # Per-seed TEST inference.
    per_seed = {}
    for model_name in MODELS:
        model_preds = []
        for seed in SEEDS:
            p = infer_one(model_name, seed, test_df, step24, m0mod)
            p = p.rename(columns={f"risk_{m}m": f"seed_{seed}_risk_{m}m" for m in HORIZONS})
            model_preds.append(p)

        base = model_preds[0][["subject_id","landmark_id","event","time_to_event_or_censor_days"]].copy()
        for p in model_preds:
            if not np.array_equal(base["landmark_id"].astype(str).to_numpy(), p["landmark_id"].astype(str).to_numpy()):
                raise RuntimeError(f"Seed alignment mismatch for {model_name}")
        merged = base.copy()
        for p in model_preds:
            riskcols = [c for c in p.columns if c.startswith("seed_")]
            merged = pd.concat([merged, p[riskcols].reset_index(drop=True)], axis=1)
        per_seed[model_name] = merged

    # Build aligned five-seed ensembles.
    ensemble = test_df[["subject_id","landmark_id","event","time_to_event_or_censor_days"]].reset_index(drop=True).copy()
    for model_name in MODELS:
        x = per_seed[model_name]
        for m in HORIZONS:
            cols = [f"seed_{s}_risk_{m}m" for s in SEEDS]
            ensemble[f"{model_name}_risk_{m}m"] = x[cols].mean(axis=1)

    ensemble_file = OUT_DIR / "ADNI_COMPONENT_ABLATION_TEST_ENSEMBLE_PREDICTIONS_V1.csv"
    ensemble.to_csv(ensemble_file, index=False)

    # Point estimates.
    point = {model: all_metrics(ensemble, model, km) for model in MODELS}

    print("\n" + "=" * 110)
    print("FROZEN TEST ENSEMBLE POINT ESTIMATES")
    print("=" * 110)
    for model in MODELS:
        print(f"\n{model}")
        for metric in PRIMARY_METRICS:
            print(f"  {metric:28s} = {point[model][metric]:.6f}")

    # Participant-cluster bootstrap.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_rows = []
    print("\n" + "=" * 110)
    print(f"PARTICIPANT-CLUSTER BOOTSTRAP — {BOOTSTRAP_REPS} REPLICATES")
    print("=" * 110)

    for b in range(BOOTSTRAP_REPS):
        bd = cluster_bootstrap_indices(ensemble, rng)
        for model in MODELS:
            vals = all_metrics(bd, model, km)
            for metric, value in vals.items():
                boot_rows.append({
                    "replicate": b,
                    "model": model,
                    "metric": metric,
                    "value": value,
                })
        if (b + 1) % 100 == 0:
            print(f"Completed {b+1}/{BOOTSTRAP_REPS}")

    boot = pd.DataFrame(boot_rows)
    boot_file = OUT_DIR / "ADNI_COMPONENT_ABLATION_TEST_BOOTSTRAP_V1.csv"
    boot.to_csv(boot_file, index=False)

    # CIs.
    metric_rows = []
    all_metric_names = list(point[MODELS[0]].keys())
    for model in MODELS:
        for metric in all_metric_names:
            arr = boot.loc[(boot["model"] == model) & (boot["metric"] == metric), "value"].to_numpy(float)
            arr = arr[np.isfinite(arr)]
            lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)
            metric_rows.append({
                "model": model,
                "metric": metric,
                "estimate": point[model][metric],
                "ci_2_5": lo,
                "ci_97_5": hi,
                "bootstrap_valid": len(arr),
            })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_file = OUT_DIR / "ADNI_COMPONENT_ABLATION_TEST_METRICS_V1.csv"
    metrics_df.to_csv(metrics_file, index=False)

    # Same-resample paired comparisons against A0.
    paired_rows = []
    for ablation in ABLATIONS:
        for metric in PRIMARY_METRICS:
            a0 = point["A0_FULL_M0"][metric]
            aa = point[ablation][metric]

            if metric in HIGHER_BETTER:
                favorable_point = aa - a0
            elif metric in LOWER_BETTER:
                favorable_point = a0 - aa
            else:
                raise RuntimeError(metric)

            a = boot.loc[(boot["model"] == ablation) & (boot["metric"] == metric), ["replicate","value"]]
            z = boot.loc[(boot["model"] == "A0_FULL_M0") & (boot["metric"] == metric), ["replicate","value"]]
            mrg = a.merge(z, on="replicate", suffixes=("_a","_a0"))

            if metric in HIGHER_BETTER:
                diff = mrg["value_a"].to_numpy(float) - mrg["value_a0"].to_numpy(float)
            else:
                diff = mrg["value_a0"].to_numpy(float) - mrg["value_a"].to_numpy(float)

            diff = diff[np.isfinite(diff)]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            pbetter = float(np.mean(diff > 0))

            if lo > 0:
                inference = "CLEAR BENEFIT OF ABLATION"
            elif hi < 0:
                inference = "CLEAR DETRIMENT FROM ABLATION"
            else:
                inference = "NO CLEAR ABLATION EFFECT"

            paired_rows.append({
                "ablation": ablation,
                "reference": "A0_FULL_M0",
                "metric": metric,
                "favorable_difference_for_ablation": favorable_point,
                "favorable_ci_2_5": lo,
                "favorable_ci_97_5": hi,
                "bootstrap_probability_ablation_better": pbetter,
                "inference": inference,
            })

    paired = pd.DataFrame(paired_rows)
    paired_file = OUT_DIR / "ADNI_COMPONENT_ABLATION_PAIRED_COMPARISONS_V1.csv"
    paired.to_csv(paired_file, index=False)

    print("\n" + "=" * 110)
    print("PRIMARY PAIRED ABLATION COMPARISONS")
    print("=" * 110)
    for ablation in ABLATIONS:
        print(f"\n{ablation} vs A0_FULL_M0")
        ss = paired.loc[paired["ablation"] == ablation]
        for _, r in ss.iterrows():
            print(
                f"{r['metric']:28s} | favorable diff={r['favorable_difference_for_ablation']:+.6f} "
                f"[{r['favorable_ci_2_5']:+.6f}, {r['favorable_ci_97_5']:+.6f}] "
                f"| Pbetter={r['bootstrap_probability_ablation_better']:.3f} "
                f"| {r['inference']}"
            )

    # Final audit/status.
    status = {
        "experiment": "ADNI_COMPONENT_ABLATION_TEST_V1",
        "completed_utc": now_utc(),
        "status": "COMPLETE",
        "models": MODELS,
        "seeds": SEEDS,
        "cohort": {
            "train_landmarks": len(train_df),
            "train_subjects": int(train_df["subject_id"].nunique()),
            "test_landmarks": len(test_df),
            "test_subjects": int(test_df["subject_id"].nunique()),
        },
        "protocol": {
            "ensemble": "arithmetic mean of five seed cumulative risks",
            "censoring_distribution": "TRAIN-only Kaplan-Meier of censoring",
            "bootstrap": {
                "type": "participant-cluster percentile bootstrap",
                "replicates": BOOTSTRAP_REPS,
                "seed": BOOTSTRAP_SEED,
                "ci": "2.5th and 97.5th percentiles",
            },
            "test_model_selection": "NONE",
            "test_tuning": "NONE",
            "test_threshold_optimization": "NONE",
            "paired_comparisons": [f"{a} vs A0_FULL_M0" for a in ABLATIONS],
        },
        "metric_definitions": {
            "risk": "cumulative risk = 1 - cumulative product of (1 - sigmoid(interval logit))",
            "ipcw_auc": "cumulative/dynamic AUC using TRAIN-only censoring KM",
            "ipcw_brier": "IPCW Brier using TRAIN-only censoring KM",
            "uno_c_60m": "Uno IPCW concordance at 60 months",
            "harrell_c_60m": "Harrell concordance at 60 months",
            "integrated_brier_0_60m": "trapezoidal integration at 0,12,24,36,48,60 months divided by 60",
        },
        "files": {
            "ensemble_predictions": str(ensemble_file),
            "bootstrap": str(boot_file),
            "metrics": str(metrics_file),
            "paired_comparisons": str(paired_file),
        },
    }
    status_file = OUT_DIR / "ADNI_COMPONENT_ABLATION_TEST_STATUS_V1.json"
    status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")

    report_lines = [
        "ADNI COMPONENT ABLATION — FROZEN TEST EVALUATION V1",
        "=" * 90,
        f"Completed UTC: {status['completed_utc']}",
        f"Models: {MODELS}",
        f"Seeds: {SEEDS}",
        f"Frozen TEST: {len(test_df)} landmarks / {test_df['subject_id'].nunique()} participants",
        "Test selection/tuning/threshold optimisation: NONE",
        "",
        "PRIMARY POINT ESTIMATES",
        "-" * 90,
    ]
    for metric in PRIMARY_METRICS:
        report_lines.append(metric)
        for model in MODELS:
            rr = metrics_df.loc[(metrics_df.model == model) & (metrics_df.metric == metric)].iloc[0]
            report_lines.append(
                f"  {model}: {rr.estimate:.6f} "
                f"[95% CI {rr.ci_2_5:.6f}, {rr.ci_97_5:.6f}]"
            )
        report_lines.append("")

    report_lines += [
        "PAIRED ABLATION EFFECTS",
        "-" * 90,
    ]
    for _, r in paired.iterrows():
        report_lines.append(
            f"{r.ablation} vs A0 | {r.metric}: favorable difference="
            f"{r.favorable_difference_for_ablation:+.6f} "
            f"[95% CI {r.favorable_ci_2_5:+.6f}, {r.favorable_ci_97_5:+.6f}] "
            f"| Pbetter={r.bootstrap_probability_ablation_better:.3f} | {r.inference}"
        )

    report_lines += [
        "",
        "INTERPRETATION RULE:",
        "Positive favorable differences favour the ablated model.",
        "A CI entirely above zero = clear benefit from ablation.",
        "A CI entirely below zero = clear detriment from ablation.",
        "A CI including zero = no clear ablation effect; this is NOT evidence of equivalence.",
        "",
        "STATUS: COMPLETE — FROZEN COMPONENT-ABLATION TEST EVALUATION FINISHED.",
    ]
    report_file = OUT_DIR / "ADNI_COMPONENT_ABLATION_TEST_REPORT_V1.txt"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nSaved:")
    for p in [ensemble_file, metrics_file, paired_file, boot_file, status_file, report_file]:
        print(" ", p)

    print("\nSTATUS: COMPLETE — FROZEN COMPONENT-ABLATION TEST EVALUATION FINISHED.")

if __name__ == "__main__":
    main()
